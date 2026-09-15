"""2D box -> 3D point cluster: VLFM `ObjectPointCloudMap._extract_object_cloud`.

The chain is copied from VLFM's write path, with the SAM mask replaced by the
in-box rectangle (SAM's role is "2D box -> 2D region", and `too_offset` only ever
looks at the bounding rectangle anyway):

    rectangle -> erode(object_map_erosion_size) -> holes to far plane
              -> backproject (fx, fy) -> random 5000 -> DBSCAN largest cluster

Two VLFM mechanisms are deliberately **not** ported here, because the VLVM memory
lifecycle already covers their role:
  * `too_offset` + random `range_id`  -> `too_offset_rect` only *flags* the box
    (`suspect`), and the caller writes it as a suspicious entry;
  * `update_explored` (near re-look deletion of flagged ids) -> the existing
    suspicious near-field hysteresis / soft-only free-space erasure.
"""

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from vlfm.utils.geometry_utils import get_point_cloud, transform_points

from .types import CameraView


def too_offset_rect(box_px: np.ndarray, width: int, height: int) -> bool:
    """VLFM `too_offset`: the whole box sits in the left/right third and at the edge."""
    x1, y1, x2, y2 = [float(t) for t in box_px]
    third = int(width // 3)
    if x2 <= third:
        return x1 <= int(0.05 * width)
    if x1 >= 2 * third:
        return x2 >= int(0.95 * width)
    return False


def rect_mask(box_px: np.ndarray, height: int, width: int) -> np.ndarray:
    """Binary rectangle (uint8 0/1) of a pixel box, clipped to the image."""
    x1 = int(max(0, np.floor(min(box_px[0], box_px[2]))))
    x2 = int(min(width - 1, np.ceil(max(box_px[0], box_px[2]))))
    y1 = int(max(0, np.floor(min(box_px[1], box_px[3]))))
    y2 = int(min(height - 1, np.ceil(max(box_px[1], box_px[3]))))
    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return mask
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return mask


def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """VLFM `cv2.erode` on the pseudo-mask (shrink away the box edge)."""
    if iterations <= 0:
        return mask.astype(np.uint8)
    eroded = cv2.erode((mask > 0).astype(np.uint8) * 255, None, iterations=int(iterations))
    return (eroded > 0).astype(np.uint8)


def depth_holes_to_far(depth_m: np.ndarray, max_depth: float) -> np.ndarray:
    """VLFM: pixels with no depth are pushed to the far plane (never dropped)."""
    depth = np.asarray(depth_m, dtype=np.float64).copy()
    depth[(~np.isfinite(depth)) | (depth <= 0)] = float(max_depth)
    return depth


def backproject_mask(depth_m: np.ndarray, mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Masked depth -> camera-base points `(forward, left, up)` via `get_point_cloud`."""
    return get_point_cloud(depth_m, mask, fx, fy)


def random_subarray(points: np.ndarray, size: int, rng: Optional[Any] = None) -> np.ndarray:
    """VLFM `get_random_subarray`, driven by a caller-owned RNG (reproducibility)."""
    pts = np.asarray(points, dtype=np.float64)
    if size <= 0 or len(pts) <= size:
        return pts
    if rng is None:
        rng = np.random
    idx = rng.choice(len(pts), size=int(size), replace=False)
    return pts[np.sort(idx)]


def largest_cluster(points: np.ndarray, eps: float = 0.2, min_points: int = 100) -> np.ndarray:
    """VLFM `open3d_dbscan_filtering`: keep only the largest non-noise cluster."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < int(min_points):
        return np.zeros((0, 3))
    try:
        import open3d as o3d
    except Exception:  # noqa: BLE001 - dependency missing -> "no cluster"
        return np.zeros((0, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    labels = np.asarray(pcd.cluster_dbscan(float(eps), int(min_points)))
    non_noise = labels[labels >= 0]
    if non_noise.size == 0:
        return np.zeros((0, 3))
    uniq, counts = np.unique(non_noise, return_counts=True)
    return pts[labels == uniq[int(np.argmax(counts))]]


def region_points_world(
    box_px: np.ndarray,
    depth_m: np.ndarray,
    cam: CameraView,
    erode_iters: int = 5,
    rand_subarray: int = 5000,
    dbscan_eps: float = 0.2,
    dbscan_min_points: int = 100,
    min_pixels: int = 256,
    rng: Optional[Any] = None,
) -> Tuple[np.ndarray, dict]:
    """Full 2D box -> world-frame cluster chain.

    Returns `(points_world, diag)`; `points_world` is `(0, 3)` when the box does not
    produce a usable cluster, and `diag["reject"]` then says which stage dropped it.
    """
    diag: dict = {"reject": None, "eroded_px": 0, "cluster_px": 0}
    mask = rect_mask(box_px, int(cam.height), int(cam.width))
    if int(mask.sum()) < 4:
        diag["reject"] = "tiny_box"
        return np.zeros((0, 3)), diag

    mask = erode_mask(mask, erode_iters)
    diag["eroded_px"] = int(mask.sum())
    if diag["eroded_px"] < int(min_pixels):
        diag["reject"] = "eroded_empty"
        return np.zeros((0, 3)), diag

    depth = depth_holes_to_far(depth_m, cam.max_depth)
    base = backproject_mask(depth, mask, cam.fx, cam.fy)
    base = random_subarray(base, rand_subarray, rng)

    cluster = largest_cluster(base, dbscan_eps, dbscan_min_points)
    diag["cluster_px"] = int(len(cluster))
    if len(cluster) == 0:
        diag["reject"] = "no_cluster"
        return np.zeros((0, 3)), diag

    world = transform_points(np.asarray(cam.tf_camera_to_episodic, dtype=np.float64), cluster)
    return world, diag
