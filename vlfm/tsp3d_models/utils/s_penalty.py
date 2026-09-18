from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

@dataclass
class SPenaltyConfig:
    """S-penalty hyper-parameters."""

    enable: bool = True
    thresh: float = 0.15   # S below this -> penalty (ITM raw cosine scale ~0.10-0.15)
    floor: float = 0.3     # w_S lower bound (keeps recall for borderline true targets)
    radius_m: float = 0.5  # S query radius (m) around the detection point
    uncovered_w: float = 1.0  # weight when there is NO semantic coverage (S=None/<=0);
                              # 1.0 = free pass (A1 default), <1.0 = mild penalty for
                              # detections the value map cannot corroborate at all.


def compute_w_s(s: Optional[float], cfg: SPenaltyConfig) -> Tuple[float, Optional[float]]:
    """Map a semantic-field value S to a penalty weight w_S; returns (w_S, S).

    S=None/<=0 (no coverage) -> (cfg.uncovered_w, None): the historical A1 default is
    1.0 = free pass; a value <1.0 additionally discounts detections in never-scored
    areas (the value map cannot corroborate them, which is not positive evidence)."""
    if not cfg.enable:
        return 1.0, None
    if s is None or s <= 0.0:
        return float(cfg.uncovered_w), None
    if s >= cfg.thresh:
        return 1.0, s
    return max(cfg.floor, s / max(cfg.thresh, 1e-6)), s


def amplified_confidence(conf: float, s: Optional[float], cfg: SPenaltyConfig) -> float:
    """c' = conf * w_S(S): amplified confidence after the S-penalty."""
    w_s, _ = compute_w_s(s, cfg)
    return conf * w_s


def query_semantic_at(x: float, y: float, radius_m: Optional[float] = None) -> Optional[float]:
    """Default semantic-field query hook; returns None = no coverage.

    Policy subclasses override the instance-level hook to query the 2.5D value map."""
    return None


def near_surface_point(box_corners: np.ndarray, camera_pos: np.ndarray) -> Optional[np.ndarray]:
    """Mean of the 4 bbox corners nearest to the camera (near-face center).

    Returns None if the box is malformed."""
    corners = np.asarray(box_corners, dtype=np.float64)
    if corners.ndim != 2 or corners.shape[0] < 4 or corners.shape[1] < 2:
        return None
    camera = np.asarray(camera_pos, dtype=np.float64)
    if camera.shape[0] >= 3:
        dists = np.linalg.norm(corners[:, :3] - camera[:3], axis=1)
    else:
        dists = np.linalg.norm(corners[:, :2] - camera[:2], axis=1)
    idx = np.argsort(dists)[:4]
    return corners[idx].mean(axis=0)


def apply_s_penalty(
    detections,
    target_classes: List[str],
    robot_xyz: np.ndarray,
    query_semantic: Callable[[float, float, Optional[float]], Optional[float]],
    cfg: SPenaltyConfig,
    sigma_tar: float,
    use_surface: bool,
    goal_use_surface: bool,
    nlp_mode: bool,
    out_log: Optional[List[Dict[str, Any]]] = None,
    per_det_meta: Optional[List[Any]] = None,
    meta_out: Optional[List[Any]] = None,
    conflict_gate: bool = True,
    conflict_conf_hi: float = 0.80,
) -> List[Tuple[float, np.ndarray, Optional[np.ndarray], List[str]]]:
    """S-penalty + admission gate for filtered detections.

    Returns [(conf_amp, centroid, near_surface, active_classes)].

    If ``out_log`` is given, one dict per detection is appended for
    detection-level diagnostics: {conf, conf_amp, centroid, s, admitted}.

    ``per_det_meta`` is copied item-by-item into ``meta_out`` 
    for each ADMITTED detection.

    冲突门：``conflict_gate=True`` 时，对“conf ≥ conflict_conf_hi ∧ 0 < S < cfg.thresh”
    的组合在写入口定向拒绝（即便 conf·w_S 已达准入线）。
    """
    import torch  # local import keeps the module torch-free at import time

    pending = []
    if per_det_meta is not None:
        assert len(per_det_meta) == len(detections.boxes), "per_det_meta length mismatch"
    for k, (centroid, conf, box) in enumerate(
        zip(detections.centroids, detections.logits, detections.boxes)
    ):
        centroid_np = centroid.cpu().numpy()
        box_np = np.asarray(box, dtype=np.float64) if box is not None else None
        conf_f = float(conf.cpu().numpy()) if torch.is_tensor(conf) else float(conf)
        active_classes = target_classes if nlp_mode else ([target_classes[0]] if target_classes else [])
        if not active_classes:
            continue

        # query S at the bbox near-surface point (facing the camera).
        s_pt = centroid_np
        if use_surface and box_np is not None:
            s_pt = near_surface_point(box_np, robot_xyz)
            if s_pt is None:
                s_pt = centroid_np
        s = query_semantic(s_pt[0], s_pt[1], cfg.radius_m)
        w_s, _ = compute_w_s(s, cfg)
        conf_amp = conf_f * w_s
        admitted = conf_amp >= sigma_tar
        # 冲突门：对“conf ≥ conf_hi ∧ 0 < S < thresh”的组合定向拒绝 —— “高 conf
        # 补偿低 S”的放行通道不再进入写入/锁定链。
        if admitted and conflict_gate and cfg.enable and s is not None and float(s) > 0.0:
            if float(conf_f) >= float(conflict_conf_hi) and float(s) < float(cfg.thresh):
                admitted = False
        if out_log is not None:
            out_log.append(
                {
                    "conf": conf_f,
                    "conf_amp": conf_amp,
                    "centroid": centroid_np,
                    "s": s,
                    "admitted": bool(admitted),
                }
            )
        if not admitted:
            continue  # gate: reject
        # near-surface point stored for the nav goal (first-write fixed).
        near_surface = None
        if goal_use_surface and box_np is not None:
            near_surface = near_surface_point(box_np, robot_xyz)
        pending.append((conf_amp, centroid_np, near_surface, active_classes))
        if meta_out is not None:
            meta_out.append(per_det_meta[k] if per_det_meta is not None else None)
    return pending
