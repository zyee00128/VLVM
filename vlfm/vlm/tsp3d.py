import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from vlfm.vlm.detections import ObjectDetections
from .server_wrapper import (
    ServerMixin,
    bool_arr_to_str,
    host_model,
    ndarray_to_str,
    send_request,
    str_to_bool_arr,
    str_to_image,
    str_to_ndarray,
)

from vlfm.tsp3d_models.bdetr import BeaUTyDETR

PROMPT_SEPARATOR = "|"

class TSP3D:
    def __init__(
        self,
        d_model=128,
        voxel_size: float = 0.01,
        data_path: str = os.environ.get("TSP3D_DATA_PATH", "/root/autodl-tmp/vlvm/data/tsp3d_models/"),
        config_path: Optional[str] = None,
        weights_path: str = os.environ.get("TSP3D_CHECKPOINT", "/root/autodl-tmp/vlvm/data/tsp3d_models/tsp3d_scanrefer.pth"),
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device
        self.voxel_size = voxel_size

        # Initialize the core 3D language grounding model BeaUTyDETR
        self.model = BeaUTyDETR(d_model=d_model, voxel_size=voxel_size,
                    data_path=data_path)
        if os.path.exists(weights_path):
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
            use_raw_nlp: bool = True
        ) -> List[Dict[str, Any]]:
        """
        Use the TSP3D model to perform 3D object detection and visual grounding.

        Args:
            pcd (np.ndarray): Input 3D point cloud, shape (N, 6).
            text (str): Query prompt, classes can be separated by '|'.
            sigma_sce (float): Scene voxel pruning threshold (TGP).
            sigma_tar (float): Target confidence threshold.
            tau (float): Soft-pruning temperature coefficient.
            use_raw_nlp (bool): Whether to use raw natural language prompt formatting.
        Returns:
            List[Dict[str, Any]]: Detected 3D bounding boxes and scores.
        """
        if len(pcd) == 0:
            return []

        if use_raw_nlp:
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
            bbox_results, _, _, times = self.model(inputs)
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
                print(f"[TSP3D Server] Post-NMS detections: {len(boxes_np)} boxes, scores={np.round(scores_np.flatten(), 3).tolist()} (sigma_tar={sigma_tar})")
                
                for box, score in zip(boxes_np, scores_np):
                    conf = float(score[0]) if hasattr(score, "__getitem__") else float(score)
                    if conf >= sigma_tar:
                        # box: (8, 3)
                        formatted_detections.append({
                            "box_3d": box.tolist(),
                            "confidence": conf
                        })
            else:
                print(f"[TSP3D Server] NMS returned empty (no box passed score_thr)")
        else:
            print(f"[TSP3D Server] bbox_results empty (all voxels likely pruned)! sigma_sce={sigma_sce}, text='{processed_text}'")
                        
        return formatted_detections

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
        use_raw_nlp: bool = True
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
            "use_raw_nlp": use_raw_nlp
        }
        response = send_request(self.url, **payload)
        return response.get("detections", [])

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
                use_raw_nlp = payload.get("use_raw_nlp", True)

                detections = self.predict(pcd, text, sigma_sce, sigma_tar, tau, use_raw_nlp)
                return {"detections": detections}
            return {}

    tsp3d_server = TSP3DServer()
    print("TSP3D Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(tsp3d_server, name="tsp3d", port=args.port)