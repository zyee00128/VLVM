from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

# Fixed S-penalty constants (Phase G): S lower clamp and query radius.
S_MIN = 0.1
S_QUERY_RADIUS_M = 0.5


@dataclass
class SPenaltyConfig:
    """Runtime S-penalty settings; only ``thresh`` is searched in P1."""

    enable: bool = True
    thresh: float = 0.15
    uncovered_w: float = 1.0  # weight when there is NO semantic coverage (S=None/<=0);
                              # 1.0 = free pass (A1 default), <1.0 = mild penalty for
                              # detections the value map cannot corroborate at all.


def query_semantic_at(x: float, y: float, radius_m: Optional[float] = None) -> Optional[float]:
    """
    Default semantic-field query hook; returns None = no coverage.

    Policy subclasses override the instance-level hook to query the 2.5D value map.
    """
    return None


def near_surface_point(box_corners: np.ndarray, camera_pos: np.ndarray) -> Optional[np.ndarray]:
    """
    Mean of the 4 bbox corners nearest to the camera (near-face center).

    Returns None if the box is malformed.
    """
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


def query_point_distance(s_pt: Any, robot_xyz: np.ndarray) -> Optional[float]:
    """Distance from the robot to the S query point for SP1 stratification."""
    try:
        point = np.asarray(s_pt, dtype=np.float64).reshape(-1)
        robot = np.asarray(robot_xyz, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    dims = int(min(point.size, robot.size, 3))
    if dims < 2:
        return None
    return float(np.linalg.norm(point[:dims] - robot[:dims]))


def compute_w_s(s: Optional[float], cfg: SPenaltyConfig) -> Tuple[float, Optional[float]]:
    """
    Map a semantic-field value S to a penalty weight w_S

    returns (w_S, S)
    """
    if not cfg.enable:
        return 1.0, None
    
    if s is None or s <= 0.0:
        return float(cfg.uncovered_w), None
    s_f = float(s)
    w_s = min(1.0, max(s_f, S_MIN) / max(float(cfg.thresh), 1e-6))
    return float(w_s), s_f


def apply_s_penalty(
    detections,
    target_classes: List[str],
    robot_xyz: np.ndarray,
    query_semantic: Callable[[float, float, Optional[float]], Optional[float]],
    cfg: SPenaltyConfig,
    sigma_tar: float,
    sigma_prop: Optional[float],
    use_surface: bool,
    goal_use_surface: bool,
    nlp_mode: bool,
    out_log: Optional[List[Dict[str, Any]]] = None,
    per_det_meta: Optional[List[Any]] = None,
    meta_out: Optional[List[Any]] = None,
    step: Optional[int] = None,
    candidate_records: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[float, np.ndarray, Optional[np.ndarray], List[str]]]:
    """
    S-penalty + admission gate for filtered detections.

    Returns [(conf_amp, centroid, near_surface, active_classes)].

    If ``out_log`` is given, one SP1 record per detection is appended for both
    admitted and rejected detections. The record preserves raw c/S inputs, the
    exact S branch, query point and distance, and the admission decision.

    ``per_det_meta`` is copied item-by-item into ``meta_out`` for each ADMITTED detection.
    """
    import torch  # local import keeps the module torch-free at import time

    pending = []
    if per_det_meta is not None:
        assert len(per_det_meta) == len(detections.boxes), "per_det_meta length mismatch"
    if candidate_records is not None:
        assert len(candidate_records) == len(detections.boxes), "candidate_records length mismatch"
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
        s_pt_source = "centroid"
        if use_surface and box_np is not None:
            near_pt = near_surface_point(box_np, robot_xyz)
            if near_pt is not None:
                s_pt = near_pt
                s_pt_source = "surface"
        s = query_semantic(s_pt[0], s_pt[1], S_QUERY_RADIUS_M)
        w_s, _ = compute_w_s(s, cfg)
        conf_amp = conf_f * w_s
        admitted = conf_amp >= sigma_tar
        reject_reason = None if admitted else (
            "uncovered" if (s is None or float(s) <= 0.0) else "low_conf"
        )
        if out_log is not None:
            if s is None or float(s) <= 0.0:
                branch = "uncovered"
            elif float(s) >= float(cfg.thresh):
                branch = "covered-high"
            else:
                branch = "covered-low"
            candidate_id = None
            if per_det_meta is not None and isinstance(per_det_meta[k], dict):
                candidate_id = per_det_meta[k].get("candidate_id")
            log_record = {
                    "candidate_id": candidate_id,
                    "step": None if step is None else int(step),
                    "conf": conf_f,
                    "pass_prop_045": bool(conf_f >= 0.45),
                    "pass_prop_050": bool(conf_f >= 0.50),
                    "pass_prop_055": bool(conf_f >= 0.55),
                    "pass_prop_060": bool(conf_f >= 0.60),
                    # Keep the service return floor and the policy admission
                    # line explicit.  Phase-G can request a wider proposal
                    # pool without allowing sub-admission candidates into
                    # memory/navigation.
                    "sigma_prop_floor": float(
                        sigma_tar if sigma_prop is None else sigma_prop
                    ),
                    "sigma_tar_admit": float(sigma_tar),
                    "conf_amp": conf_amp,
                    "sigma_c": conf_amp,
                    "centroid": centroid_np,
                    "s": s,
                    "s_pt": np.asarray(s_pt, dtype=np.float64).reshape(-1)[:3].tolist(),
                    "s_pt_source": s_pt_source,
                    "w_s": float(w_s),
                    "branch": branch,
                    "dist": query_point_distance(s_pt, robot_xyz),
                    "admitted": bool(admitted),
                    "s_penalty_admitted": bool(admitted),
                    "s_stage_seen": True,
                    "reject": reject_reason,
                    "use_surface": bool(use_surface),
                    "s_enabled": bool(cfg.enable),
                    "radius_m": S_QUERY_RADIUS_M,
                    "thresh": float(cfg.thresh),
                    "s_min": S_MIN,
                    "derived_min_w": min(1.0, S_MIN / max(float(cfg.thresh), 1e-6)),
                }
            out_log.append(log_record)
            if candidate_records is not None:
                candidate_records[k].update(log_record)
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


# ---------------------------------------------------------------------------
# S-penalty v2: joint score q = c^(1-beta) * S^beta (unnormalized S)
# ---------------------------------------------------------------------------


@dataclass
class SPenaltyV2Config:
    """Runtime settings for S-penalty v2.

    Score (covered):   q = c^(1-beta) * S^beta, S on the raw [0,1] value-map scale.
    Score (uncovered): q = c  (neutral: missing coverage is not evidence against).

    ``beta`` is the only dominance parameter (search range 0.1-0.4, must stay < 0.5).
    ``q_admit`` is the independent admission line on q.
    ``thresh`` is used ONLY to label covered-high/covered-low in logs; it does not
    enter the v2 score.
    """

    enable: bool = True
    beta: float = 0.30
    q_admit: float = 0.60
    thresh: float = 0.15


def compute_q_v2(
    conf: float, s: Optional[float], cfg: SPenaltyV2Config
) -> Tuple[float, str]:
    """Return (q, branch) for S-penalty v2.

    branch is one of ``uncovered`` / ``covered-high`` / ``covered-low`` (the
    high/low split uses ``cfg.thresh`` for stratification only).
    """
    c = float(conf)
    if not cfg.enable:
        return c, "disabled"
    beta = float(cfg.beta)
    if not 0.0 < beta < 1.0:
        raise ValueError(f"SPenaltyV2Config.beta must be in (0,1), got {beta}")
    if s is None or float(s) <= 0.0:
        return c, "uncovered"
    s_f = float(s)
    q = (max(c, 0.0) ** (1.0 - beta)) * (s_f ** beta)
    branch = "covered-high" if s_f >= float(cfg.thresh) else "covered-low"
    return float(q), branch


def apply_s_penalty_v2(
    detections,
    target_classes: List[str],
    robot_xyz: np.ndarray,
    query_semantic: Callable[[float, float, Optional[float]], Optional[float]],
    cfg: SPenaltyV2Config,
    sigma_prop: Optional[float],
    use_surface: bool,
    goal_use_surface: bool,
    nlp_mode: bool,
    out_log: Optional[List[Dict[str, Any]]] = None,
    per_det_meta: Optional[List[Any]] = None,
    meta_out: Optional[List[Any]] = None,
    step: Optional[int] = None,
    candidate_records: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[float, np.ndarray, Optional[np.ndarray], List[str]]]:
    """S-penalty v2 admission gate (v1 stays untouched).

    Covered (S>0):  q = c^(1-beta) * S^beta
    Uncovered:      q = c
    Admission:      q >= cfg.q_admit

    Returns [(q, centroid, near_surface, active_classes)] for admitted detections.
    ``out_log`` receives one record per detection (admitted and rejected) with raw
    c/S, branch, beta, q and the admission decision, so v1 and v2 can be compared
    offline on the same candidates.
    """
    import torch  # local import keeps the module torch-free at import time

    pending = []
    if per_det_meta is not None:
        assert len(per_det_meta) == len(detections.boxes), "per_det_meta length mismatch"
    if candidate_records is not None:
        assert len(candidate_records) == len(detections.boxes), "candidate_records length mismatch"
    for k, (centroid, conf, box) in enumerate(
        zip(detections.centroids, detections.logits, detections.boxes)
    ):
        centroid_np = centroid.cpu().numpy()
        box_np = np.asarray(box, dtype=np.float64) if box is not None else None
        conf_f = float(conf.cpu().numpy()) if torch.is_tensor(conf) else float(conf)
        active_classes = target_classes if nlp_mode else ([target_classes[0]] if target_classes else [])
        if not active_classes:
            continue

        # same S query point policy as v1: near-surface point facing the camera.
        s_pt = centroid_np
        s_pt_source = "centroid"
        if use_surface and box_np is not None:
            near_pt = near_surface_point(box_np, robot_xyz)
            if near_pt is not None:
                s_pt = near_pt
                s_pt_source = "surface"
        s = query_semantic(s_pt[0], s_pt[1], S_QUERY_RADIUS_M)
        q, branch = compute_q_v2(conf_f, s, cfg)
        admitted = q >= float(cfg.q_admit)
        if admitted:
            reject_reason = None
        elif branch == "uncovered" or not cfg.enable:
            reject_reason = "low_conf"
        else:
            reject_reason = "low_q"
        if out_log is not None:
            candidate_id = None
            if per_det_meta is not None and isinstance(per_det_meta[k], dict):
                candidate_id = per_det_meta[k].get("candidate_id")
            log_record = {
                "candidate_id": candidate_id,
                "step": None if step is None else int(step),
                "s_penalty_version": "v2",
                "conf": conf_f,
                "pass_prop_045": bool(conf_f >= 0.45),
                "pass_prop_050": bool(conf_f >= 0.50),
                "pass_prop_055": bool(conf_f >= 0.55),
                "pass_prop_060": bool(conf_f >= 0.60),
                "sigma_prop_floor": float(
                    cfg.q_admit if sigma_prop is None else sigma_prop
                ),
                "q": float(q),
                "conf_amp": float(q),
                "beta": float(cfg.beta),
                "q_admit": float(cfg.q_admit),
                "s": s,
                "s_pt": np.asarray(s_pt, dtype=np.float64).reshape(-1)[:3].tolist(),
                "s_pt_source": s_pt_source,
                "branch": branch,
                "dist": query_point_distance(s_pt, robot_xyz),
                "admitted": bool(admitted),
                "reject": reject_reason,
                "use_surface": bool(use_surface),
                "s_enabled": bool(cfg.enable),
                "radius_m": S_QUERY_RADIUS_M,
                "thresh": float(cfg.thresh),
            }
            out_log.append(log_record)
            if candidate_records is not None:
                candidate_records[k].update(log_record)
        if not admitted:
            continue  # gate: reject
        near_surface = None
        if goal_use_surface and box_np is not None:
            near_surface = near_surface_point(box_np, robot_xyz)
        pending.append((float(q), centroid_np, near_surface, active_classes))
        if meta_out is not None:
            meta_out.append(per_det_meta[k] if per_det_meta is not None else None)
    return pending
