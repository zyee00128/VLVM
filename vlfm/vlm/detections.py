import cv2
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any

try:
    from torchvision.ops import box_convert
except ImportError:  # e.g. client-only usage without torchvision.ops
    box_convert = None

class ObjectDetections:
    """
    Provides a consistent format for object detections generated 
    by both object detection and grounding models.

    Supports two flavours:
      - 3D boxes: shape (N, 8, 3) corners (TSP3D path; original behaviour).
      - 2D boxes: shape (N, 4) in `fmt` (default "cxcywh"); converted to xyxy
        on construction (Grounding-DINO path: normalized cxcywh -> xyxy).
    """

    def __init__(
        self,
        boxes: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        logits: Union[torch.Tensor, np.ndarray, List[float]],
        phrases: List[str],
        pcd_source: Optional[np.ndarray] = None,
        image_source: Optional[np.ndarray] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        tf_camera_to_episodic: Optional[np.ndarray] = None,
        fmt: str = "cxcywh",
    ) -> None:
        """
        Args:
            boxes: Bounding boxes.
                      - 3D: list/ndarray/Tensor of shape (N, 8, 3) corners.
                      - 2D: Tensor/ndarray of shape (N, 4) in `fmt`; converted
                        to xyxy (Grounding-DINO returns normalized cxcywh).
            logits: Confidence score of each detection, shape (N,).
            phrases: Text label or query associated with each box.
            pcd_source: Optional 3D point cloud source, shape (M, 6).
            image_source: Optional 2D source image.
            fx: Camera intrinsic focal length along the x-axis.
            fy: Camera intrinsic focal length along the y-axis.
            tf_camera_to_episodic: Camera to episodic coordinates transformation matrix.
            fmt: 2D box input format ("cxcywh" / "xywh" / "xyxy"); ignored for 3D.
        """
        self.phrases = list(phrases)
        self.pcd_source = pcd_source
        # Keep the data context used for 2D projection rendering
        self.image_source = image_source
        self.fx = fx
        self.fy = fy
        self.tf_camera_to_episodic = tf_camera_to_episodic
        self._annotated_frame: Optional[np.ndarray] = None

        # Normalize 3D bounding boxes to (N, 8, 3) torch.Tensor
        if isinstance(boxes, list):
            if len(boxes) > 0:
                boxes_np = np.stack([np.array(b) for b in boxes], axis=0)
                self.boxes = torch.from_numpy(boxes_np).float()
            else:
                self.boxes = torch.empty((0, 8, 3), dtype=torch.float32)
        elif isinstance(boxes, np.ndarray):
            self.boxes = torch.from_numpy(boxes).float()
        elif isinstance(boxes, torch.Tensor):
            self.boxes = boxes.float()
        else:
            raise TypeError("Unsupported type for boxes. Expected list, ndarray or Tensor.")

        # 2D (image) support: shape (N, 4) in `fmt` -> xyxy.
        # Lets Grounding-DINO detections (normalized cxcywh) share this container.
        self._is_2d = self.boxes.dim() == 2 and self.boxes.shape[-1] == 4
        if self._is_2d and self.boxes.shape[0] > 0 and fmt != "xyxy":
            if box_convert is None:
                raise ImportError(
                    "torchvision.ops.box_convert is required for non-xyxy 2D boxes."
                )
            self.boxes = box_convert(boxes=self.boxes, in_fmt=fmt, out_fmt="xyxy")

        # Normalize confidence scores to (N,) torch.Tensor
        if isinstance(logits, list):
            self.logits = torch.tensor(logits, dtype=torch.float32)
        elif isinstance(logits, np.ndarray):
            self.logits = torch.from_numpy(logits).float()
        elif isinstance(logits, torch.Tensor):
            self.logits = logits.float()
        else:
            raise TypeError("Unsupported type for logits. Expected list, ndarray or Tensor.")

        # Validate dimension consistency
        assert len(self.boxes) == len(self.logits) == len(self.phrases), (
            f"Dimension mismatch: boxes({len(self.boxes)}), "
            f"logits({len(self.logits)}), phrases({len(self.phrases)})"
        )

    @property
    def annotated_frame(self) -> Optional[np.ndarray]:
        """
        2D detections: draw rectangles on the source image.
        3D detections: perspective-project the 12 edges of each 3D box.
        """
        if self._annotated_frame is not None:
            return self._annotated_frame

        if self._is_2d:
            if self.image_source is None or len(self.boxes) == 0:
                return self.image_source
            self._annotated_frame = self._render_2d_boxes()
            return self._annotated_frame

        if (
            self.image_source is None 
            or self.fx is None 
            or self.fy is None 
            or self.tf_camera_to_episodic is None
            or len(self.boxes) == 0
        ):
            return self.image_source
        self._annotated_frame = self._render_3d_wireframes()
        return self._annotated_frame

    def _render_2d_boxes(self) -> np.ndarray:
        """
        Draw 2D xyxy boxes (normalized coords auto de-normalized) on the image.
        """
        img = self.image_source
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        img = img.copy()
        H, W = img.shape[:2]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for box, score, phrase in zip(self.boxes, self.logits, self.phrases):
            b = box.detach().cpu().numpy().astype(float)
            if b.max() <= 1.0:  # normalized -> pixels
                b = b * np.array([W, H, W, H])
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            label = f"{phrase}: {int(float(score) * 100)}%"
            cv2.putText(
                img_bgr, label, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA
            )
            cv2.putText(
                img_bgr, label, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA
            )
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _render_3d_wireframes(self) -> np.ndarray:
        """
        Perspective-project the 12 edges of each 3D box and draw them onto the 2D pixel plane.
        """
        img = self.image_source
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        img = img.copy()
        H, W = img.shape[:2]
        # Convert RGB to OpenCV's BGR format
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Compute the inverse transform from the episodic world back to the camera frame
        tf_matrix = self.tf_camera_to_episodic
        if torch.is_tensor(tf_matrix):
            tf_matrix = tf_matrix.detach().cpu().numpy()
        tf_episodic_to_camera = np.linalg.inv(tf_matrix)

        # Edge connectivity order of the 12 cube edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # four vertical side edges
        ]

        # Iterate over all detected 3D bounding boxes
        # boxes: (N, 8, 3)
        for box, score, phrase in zip(self.boxes, self.logits, self.phrases):
            box_np = box.cpu().numpy()  # (8, 3)
            # Project the 3D vertices into the camera frame
            pts_world_homo = np.hstack([box_np, np.ones((8, 1))])
            pts_cam = (tf_episodic_to_camera @ pts_world_homo.T).T[:, :3]  # (8, 3)
            # Project onto the 2D pixel plane
            pts_2d = []
            for x, y, z in pts_cam:
                if z <= 0.1:  # Filter out points behind the camera
                    pts_2d.append(None)
                    continue
                u = int((x * self.fx) / z + W / 2.0)
                v = int((y * self.fy) / z + H / 2.0)
                pts_2d.append((u, v))

            if any(p is None for p in pts_2d):
                continue

            # Draw the 12 edges
            color = (0, 255, 0)  # green wireframe
            for start_idx, end_idx in edges:
                pt1 = pts_2d[start_idx]
                pt2 = pts_2d[end_idx]
                cv2.line(img_bgr, pt1, pt2, color, thickness=2)

            # Draw a text label above the 3D box
            # Use vertex 4 of the top face as the label anchor point
            label_pos = pts_2d[4]
            text_label = f"{phrase}: {int(score * 100)}%"
            cv2.putText(
                img_bgr, text_label, (label_pos[0], label_pos[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA
            )
            cv2.putText(
                img_bgr, text_label, (label_pos[0], label_pos[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA
            )

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


    @property
    def centroids(self) -> torch.Tensor:
        """
        Automatically compute and return the spatial centroids of all 3D bounding boxes.

        Returns:
            torch.Tensor: Spatial physical centroids of shape (N, 3). 
            Empty for 2D detections (image-space boxes have no 3D centroid).
        """
        if self._is_2d:
            return torch.empty((0, 3), dtype=torch.float32)
        
        if len(self.boxes) == 0:
            return torch.empty((0, 3), dtype=torch.float32)
        # Average the 8 vertices of each (N, 8, 3) box to get the (N, 3) centroid
        return torch.mean(self.boxes, dim=1)

    @property
    def num_detections(self) -> int:
        """Returns the number of detections."""
        return len(self.phrases)

    def __repr__(self) -> str:
        """Print each detection's class, score, and box"""
        if self._is_2d:
            dets = [
                f"[{phrase}] Conf: {logit:.2f} | Box: [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]"
                for phrase, logit, b in zip(self.phrases, self.logits, self.boxes)
            ]
            return "No detections" if len(dets) == 0 else "\n".join(dets)
        
        centroids = self.centroids
        dets = [
            f"[{phrase}] Conf: {logit:.2f} | Centroid: [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]"
            for phrase, logit, c in zip(self.phrases, self.logits, centroids)
        ]
        if len(dets) == 0:
            return "No detections"
        
        return "\n".join(dets)


    def filter_by_conf(self, conf_thresh: float, use_vlfm_nlp: bool = True) -> "ObjectDetections":
        """
        Filter detections in-place according to the confidence threshold.

        Args:
            conf_thresh (float): Confidence threshold. A higher value keeps
                fewer but more reliable detections, while a lower value keeps
                more detections at the cost of more false positives.
            use_vlfm_nlp (bool): Unused; kept for API compatibility.
        """
        if len(self.logits) == 0:
            return self
        keep = torch.ge(self.logits, conf_thresh)
        self._filter(keep)
        return self

    def filter_by_class(self, classes: List[str], use_vlfm_nlp: bool = True) -> "ObjectDetections":
        """
        Filters detections in-place to keep only specified classes.
        """
        if len(self.phrases) == 0:
            return self

        # Normalize classes to set for faster lookup
        target_classes = {c.strip().lower() for c in classes if c.strip()}

        keep_indices = []
        for p in self.phrases:
            p_lower = p.lower()
            
            # Support parsing raw combined text labels
            if "|" in p_lower:
                raw_split = [c.strip() for c in p_lower.split("|") if c.strip()]
            elif " . " in p_lower:
                # Parse strings joined via " . " from the original framework formatting
                raw_split = [c.replace(".", "").strip() for c in p_lower.split(" . ") if c.strip()]
            else:
                raw_split = [p_lower.strip().replace(".", "")]

            # Determine active queries based on VLM NLP parsing settings
            if use_vlfm_nlp:
                # Keep all split terms active
                active_queries = raw_split
            else:
                # Only the primary class is active, others are ignored
                active_queries = [raw_split[0]] if len(raw_split) > 0 else []

            # Check for matching query terms
            has_match = any(
                q in target_classes or any(tc in q or q in tc for tc in target_classes) 
                for q in active_queries
            )
            keep_indices.append(has_match)

        keep = torch.tensor(keep_indices, dtype=torch.bool)
        self._filter(keep)
        return self

    def filter_by_mask(self, keep) -> "ObjectDetections":
        """
        In-place filter with an explicit boolean mask (e.g. geometric admission gate).

        Args:
            keep: bool mask (numpy / torch) of length == num_detections.
        """
        if len(self.logits) == 0:
            return self
        if not torch.is_tensor(keep):
            keep = torch.as_tensor(np.asarray(keep, dtype=bool))
        keep = keep.to(device=self.logits.device)
        self._filter(keep)
        return self

    def _filter(self, keep: torch.Tensor) -> None:
        """Filters detections in-place."""
        # Return early if no detections to filter
        if keep.all():
            return

        self.boxes = self.boxes[keep]
        self.logits = self.logits[keep]
        self.phrases = [p for i, p in enumerate(self.phrases) if keep[i].item()]
        self._annotated_frame = None


    def to_json(self) -> dict:
        """
        Converts the object detections to a JSON serializable format.

        Returns:
            dict: A dictionary containing the object detections.
        """
        return {
            "boxes": self.boxes.tolist(),
            "logits": self.logits.tolist(),
            "phrases": self.phrases,
        }

    @classmethod
    def from_json(
        cls,
        json_dict: Dict[str, Any],
        pcd_source: Optional[np.ndarray] = None,
        image_source: Optional[np.ndarray] = None,
    ) -> "ObjectDetections":
        """
        Converts the object detections from a JSON serializable format.

        Args:
            json_dict (dict): A dictionary containing the object detections.
            image_source (Optional[np.ndarray], optional): Optionally provide the
                original image source. Defaults to None.
        """
        boxes = torch.tensor(json_dict["boxes"], dtype=torch.float32)
        # 2D boxes were already converted to xyxy before serialization; 
        # pass fmt="xyxy" to avoid a second conversion. 3D boxes ignore fmt.
        is_2d = boxes.dim() == 2 and boxes.shape[-1] == 4
        return cls(
            boxes=boxes,
            logits=torch.tensor(json_dict["logits"], dtype=torch.float32),
            phrases=json_dict["phrases"],
            pcd_source=pcd_source,
            image_source=image_source,
            fmt="xyxy" if is_2d else "cxcywh",
        )
