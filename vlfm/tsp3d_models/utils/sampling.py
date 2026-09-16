"""Sampling helpers: world voxelization, distance-adaptive sampling, unified point cap."""
from typing import Tuple
import numpy as np


def voxelize_world(pcd_world: np.ndarray, voxel_size: float) -> np.ndarray:
    """Floor-voxelize in world frame, one representative point per voxel.

    Args:
        pcd_world: world-frame point cloud (N, 6) [x, y, z, r, g, b].
        voxel_size: voxel size (m).

    Returns:
        (M, 6) deduplicated cloud (M <= N).
    """
    if len(pcd_world) == 0:
        return np.empty((0, 6), dtype=pcd_world.dtype)
    vox = np.floor(pcd_world[:, :3] / float(voxel_size)).astype(np.int64)
    _, first_idx = np.unique(vox, axis=0, return_index=True)
    return pcd_world[np.sort(first_idx)]


def _voxelize_at(pcd: np.ndarray, voxel_size: float) -> np.ndarray:
    """Floor-voxelize with the given size (local helper)."""
    vox = np.floor(pcd[:, :3] / float(voxel_size)).astype(np.int64)
    _, first_idx = np.unique(vox, axis=0, return_index=True)
    return pcd[np.sort(first_idx)]


def distance_adaptive_sample(
    pcd: np.ndarray,
    camera_pos_local: np.ndarray,
    near_dist: float = 1.5,
    mid_dist: float = 3.0,
    near_voxel: float = 0.01,
    mid_voxel: float = 0.02,
    far_voxel: float = 0.05,
) -> np.ndarray:
    """Distance-adaptive sampling (dense near, sparse far) via multi-band voxelization.

    Points are split by 3D distance to the camera into near / mid / far bands,
    each voxelized at its own resolution, then concatenated:
    - near (<= near_dist): near_voxel (default 0.01, matches native model granularity);
    - mid ((near_dist, mid_dist]): mid_voxel (default 0.02);
    - far (> mid_dist): far_voxel (default 0.05).

    Args:
        pcd: camera-canonical (or any shared frame) cloud (N, 6).
        camera_pos_local: camera position in that frame (e.g. [0, 0, camera_height]).
        near_dist / mid_dist: near / mid / far band boundaries (m).
        near_voxel / mid_voxel / far_voxel: per-band voxel sizes (m).

    Returns:
        (M, 6) distance-adaptive sampled cloud.
    """
    if len(pcd) == 0:
        return np.empty((0, 6), dtype=pcd.dtype)

    dist = np.linalg.norm(pcd[:, :3] - np.asarray(camera_pos_local)[:3], axis=1)
    bands: Tuple[Tuple[np.ndarray, float], ...] = (
        (dist <= near_dist, near_voxel),
        ((dist > near_dist) & (dist <= mid_dist), mid_voxel),
        (dist > mid_dist, far_voxel),
    )
    out = []
    for mask, vox in bands:
        if not mask.any():
            continue
        out.append(_voxelize_at(pcd[mask], vox))
    if len(out) == 0:
        return np.empty((0, 6), dtype=pcd.dtype)
    return np.concatenate(out, axis=0)


def _bound_point_count(
    pcd: np.ndarray,
    camera_pos_local: np.ndarray,
    max_points: int,
) -> np.ndarray:
    """Cap to max_points keeping the nearest points (near-field priority, deterministic)."""
    if len(pcd) <= max_points:
        return pcd
    dist = np.linalg.norm(pcd[:, :3] - np.asarray(camera_pos_local)[:3], axis=1)
    order = np.argsort(dist, kind="stable")
    return pcd[order[:max_points]]


def cap_point_count(
    pcd: np.ndarray,
    camera_pos_local: np.ndarray,
    max_points: int,
    style: str = "random",
) -> np.ndarray:
    """Unified point cap shared by both fusion routes (D1).

    Args:
        pcd: camera-canonical cloud (N, 6).
        camera_pos_local: camera position.
        max_points: point cap.
        style: "random" (uniform random, baseline behavior) / "near_first"
            (near-field priority, deterministic; near field is the key detection zone).

    Returns:
        (max_points, 6) or unchanged if within cap.
    """
    if len(pcd) <= max_points:
        return pcd
    if style == "near_first":
        return _bound_point_count(pcd, camera_pos_local, max_points)
    idx = np.random.choice(len(pcd), max_points, replace=False)
    return pcd[idx]
