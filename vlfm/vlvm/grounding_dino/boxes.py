"""3D AABB geometry: corners, centre, overlap (IaU / containment), nearest point.

Used for (a) turning the GD point cluster into a 3D candidate and (b) the
frame-level merge of a GD proposal with the TSP3D candidates.
"""

from typing import Optional, Tuple

import numpy as np

# (lo, hi) corners of an axis-aligned 3D box
Bounds3D = Tuple[np.ndarray, np.ndarray]


def corners_to_bounds(box8: np.ndarray) -> Bounds3D:
    c = np.asarray(box8, dtype=np.float64).reshape(-1, 3)
    return c.min(axis=0), c.max(axis=0)


def bounds_to_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The 8 corners of an AABB, in the project's canonical order."""
    x0, y0, z0 = float(lo[0]), float(lo[1]), float(lo[2])
    x1, y1, z1 = float(hi[0]), float(hi[1]), float(hi[2])
    return np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float64,
    )


def aabb_corners(points: np.ndarray) -> Tuple[np.ndarray, Bounds3D]:
    """AABB of a point set -> (8 corners, (lo, hi))."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return bounds_to_corners(lo, hi), (lo, hi)


def aabb_center(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return 0.5 * (pts.min(axis=0) + pts.max(axis=0))


def nearest_point(points: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Cluster point closest to a planar reference point (VLFM's "surface point")."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(xy, dtype=np.float64).reshape(-1)
    if pts.shape[0] == 0:
        return np.zeros(3)
    if ref.size < 2:
        return pts[0]
    d = np.linalg.norm(pts[:, :2] - ref[:2], axis=1)
    return pts[int(np.argmin(d))]


def intra_camera_distance(points: np.ndarray, camera_pos: np.ndarray) -> float:
    """Closest 3D distance from the camera position to a point set (VLFM near gate)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return float("inf")
    cam = np.asarray(camera_pos, dtype=np.float64).reshape(-1)[:3]
    return float(np.min(np.linalg.norm(pts - cam, axis=1)))


def _volume(lo: np.ndarray, hi: np.ndarray) -> float:
    d = np.maximum(0.0, np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64))
    return float(d[0] * d[1] * d[2])


def iau_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-union of two AABBs (0 = disjoint, 1 = identical)."""
    lo_a, hi_a = corners_to_bounds(box_a)
    lo_b, hi_b = corners_to_bounds(box_b)
    inter_lo, inter_hi = np.maximum(lo_a, lo_b), np.minimum(hi_a, hi_b)
    inter = _volume(inter_lo, inter_hi)
    union = _volume(lo_a, hi_a) + _volume(lo_b, hi_b) - inter
    return float(inter / union) if union > 0 else 0.0


def containment(inner_box: np.ndarray, outer_box: np.ndarray) -> float:
    """Fraction of `inner_box`'s volume that lies inside `outer_box`."""
    lo_i, hi_i = corners_to_bounds(inner_box)
    lo_o, hi_o = corners_to_bounds(outer_box)
    inter_lo, inter_hi = np.maximum(lo_i, lo_o), np.minimum(hi_i, hi_o)
    vol_i = _volume(lo_i, hi_i)
    return float(_volume(inter_lo, inter_hi) / vol_i) if vol_i > 0 else 0.0


def planar_distance(a: np.ndarray, b: np.ndarray) -> float:
    """XY distance between two 3D points (the memory-merge key space)."""
    pa = np.asarray(a, dtype=np.float64).reshape(-1)
    pb = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(pa[:2] - pb[:2]))


def same_object(box_a: np.ndarray, centroid_a: np.ndarray, box_b: Optional[np.ndarray], centroid_b: np.ndarray,
                max_dist: float = 0.5, min_iau: float = 0.3) -> bool:
    """Candidate-level "same object" test: XY distance OR 3D box overlap."""
    if planar_distance(centroid_a, centroid_b) <= float(max_dist):
        return True
    if box_b is None:
        return False
    if iau_3d(box_a, box_b) >= float(min_iau):
        return True
    return containment(box_a, box_b) >= 0.8 or containment(box_b, box_a) >= 0.8
