import os
from pathlib import Path
from typing import Any, Dict, List, Optional
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

# 仓库根由本文件位置推导：vlfm/vlm/tsp3d.py -> parents[2] = 仓库根。
# 权重/配置目录默认 <repo>/data/tsp3d_models（可被 TSP3D_DATA_PATH / TSP3D_CHECKPOINT 覆盖）。
TSP3D_DATA_DIR = str(Path(__file__).resolve().parents[2] / "data" / "tsp3d_models")


def _default_checkpoint(data_dir: str) -> str:
    """默认权重路径：按候选名探测（各机在位的档名不同），都无则回退旧默认名。"""
    for name in ("tsp3d_scanrefer.pth", "ckpt_sr3d.pth", "ckpt_nr3d.pth"):
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(data_dir, "tsp3d_scanrefer.pth")


TSP3D_DEFAULT_CHECKPOINT = os.environ.get("TSP3D_CHECKPOINT") or _default_checkpoint(
    os.environ.get("TSP3D_DATA_PATH", TSP3D_DATA_DIR)
)


class TSP3D:
    def __init__(
        self,
        d_model=128,
        voxel_size: float = 0.01,
        data_path: str = os.environ.get("TSP3D_DATA_PATH", TSP3D_DATA_DIR),
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        # 权重：显式传入 > TSP3D_CHECKPOINT > 按候选名探测 data_path 目录。
        if weights_path is None:
            weights_path = os.environ.get("TSP3D_CHECKPOINT") or _default_checkpoint(data_path)
        self.device = device
        self.voxel_size = voxel_size
        self._infer_seed_base = int(os.environ.get("TSP3D_SEED", "0"))
        self._n_infer = 0

        # Initialize the core 3D language grounding model BeaUTyDETR
        self.model = BeaUTyDETR(d_model=d_model, voxel_size=voxel_size,
                    data_path=data_path)
        if os.path.exists(weights_path):
            try:
                # torch >= 2.6 的 torch.load 默认 weights_only=True；本 checkpoint 含 config
                # 元数据（argparse.Namespace），必须显式关闭。torch < 1.13 无该参数 ⇒ 回退。
                checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(weights_path, map_location=device)
            # Checkpoint format: {'config': ..., 'model': OrderedDict, ...}
            # Use 'model' key first, fall back to 'state_dict', then raw dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # Strip 'module.' prefix (from DataParallel/DDP wrapper)
            new_sd = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '') if k.startswith('module.') else k
                new_sd[new_key] = v
            missing, unexpected = self.model.load_state_dict(new_sd, strict=False)
            if missing:
                print(f"[TSP3D] Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"[TSP3D] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            print(f"[TSP3D] Weights loaded from {weights_path} (matched {len(new_sd) - len(unexpected) - len(missing) if isinstance(missing, list) else '?'} keys)")
        else:
            print(f"[Warning] TSP3D weights not found at {weights_path}. Running with uninitialized weights.")
        self.model.to(device)
        self.model.eval()

    def predict(
            self, 
            pcd: np.ndarray, 
            text: str, 
            sigma_sce: float = 0.3, 
            sigma_tar: float = 0.05, 
            tau: float = 0.15,
            use_vlfm_nlp: bool = True,
        ) -> List[Dict[str, Any]]:
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
            List[Dict[str, Any]]: Detected 3D bounding boxes and scores.
        """
        if len(pcd) == 0:
            return [], {}
        torch.manual_seed((self._infer_seed_base + self._n_infer) & 0x7FFFFFFF)
        self._n_infer += 1
        # Baseline: only boxes at/above sigma_tar are returned.
        min_conf = float(sigma_tar)

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
        
        with torch.inference_mode():
            bbox_results, _, diagnostics, times = self.model(inputs)
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

                # scores_3d is (N, n_classes) on the single-candidate path and (N,)
                # after _nms (4.6.1 multi-candidate); handle both shapes.
                for i, box in enumerate(boxes_np):
                    if scores_np.ndim == 2:
                        conf = float(scores_np[i, 0])
                    else:
                        conf = float(scores_np[i])
                    if conf >= min_conf:
                        # box: (8, 3)
                        formatted_detections.append({
                            "box_3d": box.tolist(),
                            "confidence": conf
                        })
            else:
                print("[TSP3D Server] NMS returned empty (no box passed score_thr)")
        else:
            print(f"[TSP3D Server] bbox_results empty (all voxels likely pruned)! sigma_sce={sigma_sce}, text='{processed_text}'")
                        
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
    ) -> List[Dict[str, Any]]:
        # Send point cloud as compact binary (float16) + base64, replacing the
        # slow tolist()+JSON-text serialization of the raw float32 array.
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

                detections, diagnostics = self.predict(
                    pcd, text, sigma_sce, sigma_tar, tau, use_vlfm_nlp
                )
                return {"detections": detections, "diagnostics": diagnostics}
            return {}

    tsp3d_server = TSP3DServer()
    print("TSP3D Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(tsp3d_server, name="tsp3d", port=args.port)