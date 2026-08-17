import cv2
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any

class ObjectDetections:
    """
    Provides a consistent format for object detections generated 
    by both object detection and grounding models.
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
    ) -> None:
        """
        Args:
            boxes: 3D bounding boxes.
                      - If list or ndarray, shape should be (N, 8, 3) corners.
                      - Can also be a torch.Tensor.
            logits: Confidence score of each 3D box, shape (N,).
            phrases: Text label or query associated with each box.
            pcd_source: Optional 3D point cloud source, shape (M, 6).
            image_source: Optional 2D source image.
            fx: Camera intrinsic focal length along the x-axis.
            fy: Camera intrinsic focal length along the y-axis.
            tf_camera_to_episodic: Camera to episodic coordinates transformation matrix.
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
        If camera parameters and the original image are provided, dynamically
        compute the 3D projection and return the image annotated with 3D wireframe boxes.
        """
        if self._annotated_frame is not None:
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
        """
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
        centroids = self.centroids
        dets = [
            f"[{phrase}] Conf: {logit:.2f} | Centroid: [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]"
            for phrase, logit, c in zip(self.phrases, self.logits, centroids)
        ]
        if len(dets) == 0:
            return "No detections"
        return "\n".join(dets)

    def filter_by_conf(self, conf_thresh: float, use_raw_nlp: bool = True) -> "ObjectDetections":
        """
        Filter detections in-place according to the confidence threshold.

        Args:
            conf_thresh (float): Confidence threshold. A higher value keeps
                fewer but more reliable detections, while a lower value keeps
                more detections at the cost of more false positives.
            use_raw_nlp (bool): Unused; kept for API compatibility.
        """
        if len(self.logits) == 0:
            return self
        keep = torch.ge(self.logits, conf_thresh)
        self._filter(keep)
        return self

    def filter_by_class(self, classes: List[str], use_raw_nlp: bool = True) -> "ObjectDetections":
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
            if use_raw_nlp:
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
        return cls(
            boxes=torch.tensor(json_dict["boxes"], dtype=torch.float32),
            logits=torch.tensor(json_dict["logits"], dtype=torch.float32),
            phrases=json_dict["phrases"],
            pcd_source=pcd_source,
            image_source=image_source,
        )
