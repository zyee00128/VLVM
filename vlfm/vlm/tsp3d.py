import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from .server_wrapper import (
    ServerMixin,
    host_model,
    ndarray_to_str,
    send_request,
    str_to_ndarray,
)

from vlfm.tsp3d_models.bdetr import BeaUTyDETR

PROMPT_SEPARATOR = "|"
# 权重/配置目录默认 <repo>/data/tsp3d_models。
TSP3D_DATA_DIR = str(Path(__file__).resolve().parents[2] / "data" / "tsp3d_models")
VLVM_V1_CHECKPOINT = str(
    Path(__file__).resolve().parents[2] / "data" / "vlvm_v1" / "vlvm_v1.pth"
)


def _default_checkpoint(data_dir: str) -> str:
    """
    默认权重路径：按候选名探测，都无则回退旧默认名。
    """
    for name in ("tsp3d_scanrefer.pth", "ckpt_sr3d.pth", "ckpt_nr3d.pth"):
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(data_dir, "tsp3d_scanrefer.pth")
TSP3D_DEFAULT_CHECKPOINT = (
    os.environ.get("TSP3D_CHECKPOINT")
    or (VLVM_V1_CHECKPOINT if os.path.exists(VLVM_V1_CHECKPOINT) else None)
    or _default_checkpoint(os.environ.get("TSP3D_DATA_PATH", TSP3D_DATA_DIR))
)


def _load_tsp3d_checkpoint(path: str, map_location: Any = "cpu") -> Any:
    """
    Load a checkpoint on CPU, with torch 2.6 compatibility.

    Training checkpoints also contain optimizer state.  Loading the complete
    object directly on CUDA needlessly places both model and optimizer tensors
    on the GPU before the inference model itself is moved there, causing a
    large transient memory spike.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _checkpoint_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("TSP3D checkpoint does not contain a state dictionary")
    return {
        (key[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state_dict.items()
    }


def _checkpoint_ft_config(checkpoint: Any, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Read the saved architecture config, with conservative old-ckpt fallback."""
    candidates = []
    if isinstance(checkpoint, dict):
        candidates.extend([checkpoint.get("ft_config"), checkpoint.get("model_config")])
        config = checkpoint.get("config")
        candidates.append(config)
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "__dict__"):
            candidate = vars(candidate)
        if not isinstance(candidate, dict):
            continue
        nested = candidate.get("ft_config", candidate)
        if hasattr(nested, "__dict__"):
            nested = vars(nested)
        if isinstance(nested, dict) and any(
            key in nested for key in ("mode", "enable_gate")
        ):
            return dict(nested)

    # Only the gate head is conditionally instantiated, so its parameter names
    # are the single key-name evidence of the stored configuration.
    has_gate = any(key.startswith("presence_gate.") for key in state_dict)
    return {"enable_gate": has_gate}


class TSP3D:
    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def __init__(
        self,
        d_model=128,
        voxel_size: float = 0.01,
        data_path: str = os.environ.get("TSP3D_DATA_PATH", TSP3D_DATA_DIR),
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ft_config: Optional[Dict[str, Any]] = None,
        top1: Optional[bool] = None,
    ):
        # 权重从 weights_path 显式传入 > TSP3D_CHECKPOINT > _default_checkpoint 探测 data_path 目录。
        if weights_path is None:
            weights_path = os.environ.get("TSP3D_CHECKPOINT") or _default_checkpoint(data_path)
        
        ft_config = dict(ft_config or {})
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                file_config = json.load(handle)
            if isinstance(file_config, dict):
                file_config = file_config.get("ft_config", file_config)
            if isinstance(file_config, dict):
                for key, value in file_config.items():
                    ft_config.setdefault(key, value)
        env_mode = os.environ.get("TSP3D_MODE")
        if env_mode and "mode" not in ft_config:
            ft_config["mode"] = env_mode
        env_gate_threshold = os.environ.get("TSP3D_GATE_THRESHOLD")
        if env_gate_threshold is not None and "gate_threshold" not in ft_config:
            ft_config["gate_threshold"] = float(env_gate_threshold)
        checkpoint = None
        new_sd = None
        if os.path.exists(weights_path):
            checkpoint = _load_tsp3d_checkpoint(weights_path)
            new_sd = _checkpoint_state_dict(checkpoint)
            inferred_config = _checkpoint_ft_config(checkpoint, new_sd)
            for key, value in inferred_config.items():
                ft_config.setdefault(key, value)
        if top1 is None:
            top1 = ft_config.get("top1", os.environ.get("TSP3D_TOP1", "0"))
        
        self.ft_config = ft_config
        self.top1 = self._coerce_bool(top1)
        self.device = device
        self.voxel_size = voxel_size
        self._infer_seed_base = int(os.environ.get("TSP3D_SEED", "0"))
        self._n_infer = 0

        # Initialize the core 3D language grounding model BeaUTyDETR
        self.model = BeaUTyDETR(d_model=d_model, voxel_size=voxel_size,
                    data_path=data_path, ft_config=ft_config)
        if new_sd is not None:
            # Missing/unexpected keys are reported below instead of raising.
            incompatible = self.model.load_state_dict(new_sd, strict=False)
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
            if missing:
                print(f"[TSP3D] Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"[TSP3D] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            matched = len(new_sd) - len(unexpected)
            print(f"[TSP3D] Weights loaded from {weights_path} (matched {matched} keys)")
            # Drop the full training checkpoint (especially AdamW state)
            # before allocating the inference model on CUDA.
            del new_sd
            del checkpoint
        else:
            print(f"[Warning] TSP3D weights not found at {weights_path}. Running with uninitialized weights.")
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert diagnostics to values accepted by Flask/JSON clients."""
        if torch.is_tensor(value):
            value = value.detach().cpu()
            return value.item() if value.ndim == 0 else value.tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): TSP3D._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TSP3D._json_safe(v) for v in value]
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    def predict(
            self, 
            pcd: np.ndarray, 
            text: str, 
            sigma_sce: float = 0.3, 
            sigma_tar: float = 0.05, 
            tau: float = 0.15,
            use_vlfm_nlp: bool = True,
            top1: Optional[bool] = None,
            gate_threshold: Optional[float] = None,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Use the TSP3D model to perform 3D object detection and visual grounding.

        Args:
            pcd (np.ndarray): Input 3D point cloud, shape (N, 6).
            text (str): Query prompt, classes can be separated by '|'.
            sigma_sce (float): Scene voxel pruning threshold (TGP).
            sigma_tar (float): Target confidence threshold.
            tau (float): Soft-pruning temperature coefficient.
            use_vlfm_nlp (bool): Whether to use raw natural language prompt formatting.
        Returns:
            Tuple containing detections and JSON-safe diagnostics.
        """
        if len(pcd) == 0:
            return [], {"presence_prob": None, 
                        "num_returned": 0,
                        "top1": self.top1, 
                        "gate": bool(self.model.ft_config.get("enable_gate", False))}
        
        torch.manual_seed((self._infer_seed_base + self._n_infer) & 0x7FFFFFFF)
        self._n_infer += 1
        # Baseline: only boxes at/above sigma_tar are returned.
        # min_conf = float(sigma_tar)
        return_top1 = self.top1 if top1 is None else self._coerce_bool(top1)

        if use_vlfm_nlp:
            # Multi-class synonym merging caption
            classes = [c.strip() for c in text.split(PROMPT_SEPARATOR) if c.strip()]
            if len(classes) > 1:
                processed_text = " . ".join(classes) + " ."
            else:
                processed_text = text if text.endswith(".") else text + " ."
        else:
            # Extract only the primary target word (handling "chair|armchair" or pure "chair")
            primary_classes = [c.strip() for c in text.split(PROMPT_SEPARATOR) if c.strip()]
            target_query = primary_classes[0] if primary_classes else text
            processed_text = target_query

        points_tensor = torch.tensor(pcd, dtype=torch.float32, device=self.device)
        inputs = {
            'point_clouds': [points_tensor],
            'text': [processed_text],
            'sigma_sce': sigma_sce,
            'tau': tau
        }
        if gate_threshold is not None:
            inputs["gate_threshold"] = float(gate_threshold)
        
        with torch.inference_mode():
            bbox_results, _, diagnostics, times = self.model(inputs)
        diagnostics = self._json_safe(diagnostics)
        gate_prob = diagnostics.get("gate_prob")
        if isinstance(gate_prob, list) and gate_prob:
            presence_prob = float(gate_prob[0])
        elif isinstance(gate_prob, (int, float)):
            presence_prob = float(gate_prob)
        else:
            presence_prob = None
        diagnostics["presence_prob"] = presence_prob

        # Parse inference results and convert format to adapt to the policy layer
        # bbox_results corresponds to the mmdet3d output structure (bboxes, scores, labels)
        formatted_detections = []
        if len(bbox_results) > 0:
            res = bbox_results[0]  # batch size = 1
            boxes = res.get('bboxes_3d', None)  # mmdet3d DepthInstance3DBoxes object
            scores = res.get('scores_3d', None)
            if boxes is not None and scores is not None:
                if hasattr(boxes, 'corners'):
                    boxes_np = boxes.corners.cpu().numpy()  # (N, 8, 3)
                else:
                    # If it is not an mmdet3d Box object, attempt to convert to a tensor
                    boxes_tensor = boxes.tensor if hasattr(boxes, 'tensor') else torch.tensor(boxes)
                    boxes_np = boxes_tensor.cpu().numpy()
                    # If it returns (N, 7+) format (center, dimensions, yaw), convert to 8-vertex corner format
                    if len(boxes_np.shape) == 2 and boxes_np.shape[1] >= 7:
                        corners_list = []
                        for b in boxes_np:
                            x, y, z, dx, dy, dz, r = b[:7]
                            cx = np.array([-dx, dx, dx, -dx, -dx, dx, dx, -dx]) / 2.0
                            cy = np.array([-dy, -dy, dy, dy, -dy, -dy, dy, dy]) / 2.0
                            cz = np.array([-dz, -dz, -dz, -dz, dz, dz, dz, dz]) / 2.0
                            local_corners = np.stack([cx, cy, cz], axis=1)
                            cos_r, sin_r = np.cos(r), np.sin(r)
                            R = np.array([
                                [cos_r, -sin_r, 0],
                                [sin_r, cos_r, 0],
                                [0, 0, 1]
                            ])
                            corners_list.append(local_corners @ R.T + np.array([x, y, z]))
                        boxes_np = np.stack(corners_list, axis=0) if len(corners_list) > 0 else np.empty((0, 8, 3))
                scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else np.array(scores)

                # scores_3d is (N, n_classes) on the single-candidate path 
                # and (N,) after _nms ; handle both shapes.
                candidates = []
                for i, box in enumerate(boxes_np):
                    if scores_np.ndim == 2:
                        conf = float(scores_np[i, 0])
                    else:
                        conf = float(scores_np[i])
                    if conf >= float(sigma_tar):
                        # Gate is veto-only: a surviving ``scores_3d`` remains the core raw score.
                        # Keep the presence probability alongside each candidate for calibration.
                        candidates.append({
                            "box_3d": box.tolist(),
                            "confidence": conf,
                            "score": conf,
                            "presence_prob": presence_prob,
                            "candidate_rank": i,
                        })
                candidates.sort(key=lambda item: item["confidence"], reverse=True)
                if return_top1:
                    candidates = candidates[:1]
                formatted_detections.extend(candidates)
            else:
                print("[TSP3D Server] NMS returned empty (no box passed score_thr)")
        else:
            print(f"[TSP3D Server] bbox_results empty (all voxels likely pruned)! sigma_sce={sigma_sce}, text='{processed_text}'")

        diagnostics["num_returned"] = len(formatted_detections)
        diagnostics["top1"] = return_top1
        return formatted_detections, diagnostics

    def segment_bbox(self, pcd: np.ndarray, box_3d: np.ndarray) -> np.ndarray:
        if len(pcd) == 0:
            return np.empty(0, dtype=bool)

        points = pcd[:, :3]
        min_bound = np.min(box_3d, axis=0) if len(box_3d.shape) > 1 else box_3d[:3] - box_3d[3:]/2
        max_bound = np.max(box_3d, axis=0) if len(box_3d.shape) > 1 else box_3d[:3] + box_3d[3:]/2

        mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        return mask

class TSP3DClient:
    def __init__(self, port: int = 12186):
        self.url = f"http://localhost:{port}/tsp3d"

    def predict(
        self, 
        pcd: np.ndarray, 
        text: str, 
        sigma_sce: float = 0.3, 
        sigma_tar: float = 0.05, 
        tau: float = 0.15,
        use_vlfm_nlp: bool = True,
        top1: Optional[bool] = None,
        gate_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Send point cloud as compact binary (float16) + base64, 
        # replacing the slow tolist()+JSON-text serialization of the raw float32 array.
        payload = {
            "pcd_b64": ndarray_to_str(pcd, dtype="float16"),
            "pcd_shape": list(pcd.shape),
            "pcd_dtype": "float16",
            "text": text,
            "sigma_sce": sigma_sce,
            "sigma_tar": sigma_tar,
            "tau": tau,
            "use_vlfm_nlp": use_vlfm_nlp,
        }
        if top1 is not None:
            payload["top1"] = bool(top1)
        if gate_threshold is not None:
            payload["gate_threshold"] = float(gate_threshold)
        response = send_request(self.url, **payload)
        return response.get("detections", []), response.get("diagnostics", {})

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12186)
    args = parser.parse_args()

    print("Loading TSP3D model...")

    class TSP3DServer(ServerMixin, TSP3D):
        def process_payload(self, payload: dict) -> dict:
            """Parse point clouds and text from the client, and call the model to perform 3D visual grounding inference."""
            # Fast route: binary float16 + base64 point cloud (used by TSP3DClient)
            if "pcd_b64" in payload and "text" in payload:
                pcd = str_to_ndarray(
                    payload["pcd_b64"],
                    tuple(payload["pcd_shape"]),
                    payload.get("pcd_dtype", "float16"),
                ).astype(np.float32)
                text = payload["text"]
                sigma_sce = payload.get("sigma_sce", 0.7)
                sigma_tar = payload.get("sigma_tar", 0.3)
                tau = payload.get("tau", 0.15)
                use_vlfm_nlp = payload.get("use_vlfm_nlp", True)
                top1 = payload.get("top1", None)
                gate_threshold = payload.get("gate_threshold", None)

                detections, diagnostics = self.predict(
                    pcd, text, sigma_sce, sigma_tar, tau, use_vlfm_nlp, top1,
                    gate_threshold
                )
                return {"detections": detections, "diagnostics": diagnostics}
            return {}

    tsp3d_server = TSP3DServer()
    print("TSP3D Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(tsp3d_server, name="tsp3d", port=args.port)
