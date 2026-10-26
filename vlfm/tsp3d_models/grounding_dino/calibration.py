"""GD calibration logger: log-only, zero behaviour change.

The logger is intentionally *outside* the navigation decision path:

  * it never writes memory, never changes a state, never filters a candidate;
  * it only runs `GdProposer` on the current RGB-D frame and records what GD
    would have proposed, how it maps to TSP3D candidates / memory tracks, and
    which geometric evidence the 2D -> 3D chain produced;
  * every record is JSON-safe so the episode logger can append it to
    `gd_frames.jsonl` / `gd_proposals.jsonl` and label `tp` with the same
    target-bbox rule used by the other offline-calibration logs.

What is recorded (all optional fields are populated when available):

  frame-level
    step, target classes, robot pose/yaw, camera intrinsics, depth summary,
    GD execution trace (raw boxes/logits/phrases + per-box decisions),
    TSP3D candidates of this frame (conf / sigma_c / S / occ / admission /
    memory id / lifecycle state), memory-track snapshot, proposal buffer,
    counters, latency.

  proposal-level
    2D box, GD conf, phrase / canonical class, 3D AABB, centroid, surface,
    nearest-camera / nearest-robot distance, FOV and depth eligibility,
    2D->3D chain diagnostics, same-object match against TSP3D candidates and
    memory tracks (distance / 3D IaU / containment), new-information flag,
    track-level `gd_pos` and cumulative `gd_hits`.

Enable with environment variables (no Hydra/YAML change):

    GD_CALIB_ENABLE=1
    GD_CALIB_EVERY_N=5
    GD_CALIB_RECORD_EVERY_N=1
    GD_CALIB_BOX_THR=0.4
    GD_CALIB_MAX_BOXES=1
    GD_CALIB_PORT=12181
    GD_CALIB_MAX_FAILS=2          # disable GD logging after N consecutive no_gd
    GD_CALIB_SAVE_POINTS=0        # 1 -> also store the world point cluster (capped)
    GD_CALIB_SAVE_IMAGES=0        # 1 -> write RGB JPG + depth .npy per GD frame
    GD_CALIB_IMAGE_DIR=<detections>/gd_images
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .boxes import containment, iau_3d, planar_distance, same_object
from .config import GdProposalConfig
from .proposer import GdProposer, should_run
from .types import CameraView, GdProposal


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    data = sorted(float(v) for v in values if np.isfinite(float(v)))
    if not data:
        return {}

    def pick(frac: float) -> float:
        return data[min(len(data) - 1, int(frac * len(data)))]

    return {
        "n": len(data),
        "p10": pick(0.10),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "max": data[-1],
    }


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _depth_summary(
    depth_norm: Optional[np.ndarray], min_depth: float, max_depth: float
) -> Dict[str, Any]:
    if depth_norm is None:
        return {}
    depth = np.asarray(depth_norm, dtype=np.float32)
    valid = (depth > 0.0) & (depth < 1.0)
    ratio = float(valid.mean()) if depth.size else 0.0
    if not valid.any():
        return {"valid_ratio": ratio}
    meters = depth[valid] * float(max_depth - min_depth) + float(min_depth)
    return {
        "valid_ratio": ratio,
        "min": float(meters.min()),
        "p10": float(np.percentile(meters, 10)),
        "p50": float(np.percentile(meters, 50)),
        "mean": float(meters.mean()),
        "p90": float(np.percentile(meters, 90)),
        "max": float(meters.max()),
    }


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class GdCalibConfig:
    enabled: bool = False
    every_n: int = 5
    record_every_n: int = 1
    box_thr: float = 0.4
    max_boxes: int = 1
    port: int = 12181
    caption_style: str = "vocab"
    min_dist: float = 1.0
    erode_iters: int = 5
    min_pixels: int = 256
    rand_subarray: int = 5000
    dbscan_eps: float = 0.2
    dbscan_min_points: int = 100
    merge_dist: float = 0.5
    merge_iau: float = 0.3
    merge_containment: float = 0.8
    new_info_dist: float = 0.2
    new_info_yaw_deg: float = 10.0
    new_info_steps: int = 3
    ttl_steps: int = 20
    max_fails: int = 2
    save_points: bool = False
    max_points_saved: int = 64
    save_images: bool = False
    image_dir: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GdCalibConfig":
        return cls(
            enabled=_env_bool("GD_CALIB_ENABLE", False),
            every_n=max(1, _env_int("GD_CALIB_EVERY_N", 5)),
            record_every_n=max(1, _env_int("GD_CALIB_RECORD_EVERY_N", 1)),
            box_thr=_env_float("GD_CALIB_BOX_THR", 0.4),
            max_boxes=max(1, _env_int("GD_CALIB_MAX_BOXES", 1)),
            port=_env_int("GD_CALIB_PORT", 12181),
            caption_style=os.environ.get("GD_CALIB_CAPTION_STYLE", "vocab"),
            min_dist=_env_float("GD_CALIB_MIN_DIST", 1.0),
            erode_iters=_env_int("GD_CALIB_ERODE_ITERS", 5),
            min_pixels=_env_int("GD_CALIB_MIN_PIXELS", 256),
            rand_subarray=_env_int("GD_CALIB_RAND_SUBARRAY", 5000),
            dbscan_eps=_env_float("GD_CALIB_DBSCAN_EPS", 0.2),
            dbscan_min_points=_env_int("GD_CALIB_DBSCAN_MIN_POINTS", 100),
            merge_dist=_env_float("GD_CALIB_MERGE_DIST", 0.5),
            merge_iau=_env_float("GD_CALIB_MERGE_IAU", 0.3),
            merge_containment=_env_float("GD_CALIB_MERGE_CONTAINMENT", 0.8),
            new_info_dist=_env_float("GD_CALIB_NEW_INFO_DIST", 0.2),
            new_info_yaw_deg=_env_float("GD_CALIB_NEW_INFO_YAW_DEG", 10.0),
            new_info_steps=_env_int("GD_CALIB_NEW_INFO_STEPS", 3),
            ttl_steps=_env_int("GD_CALIB_TTL_STEPS", 20),
            max_fails=max(1, _env_int("GD_CALIB_MAX_FAILS", 2)),
            save_points=_env_bool("GD_CALIB_SAVE_POINTS", False),
            max_points_saved=_env_int("GD_CALIB_MAX_POINTS_SAVED", 64),
            save_images=_env_bool("GD_CALIB_SAVE_IMAGES", False),
            image_dir=os.environ.get("GD_CALIB_IMAGE_DIR") or None,
        )


# ---------------------------------------------------------------------------
# logger
# ---------------------------------------------------------------------------
class GdCalibrationLogger:
    """Per-step, no-op-on-the-policy GD logger."""

    def __init__(self, config: Optional[GdCalibConfig] = None) -> None:
        self.cfg = config or GdCalibConfig.from_env()
        self.proposer = GdProposer(
            GdProposalConfig(
                enabled=True,
                every_n=int(self.cfg.every_n),
                port=int(self.cfg.port),
                box_thr=float(self.cfg.box_thr),
                caption_style=str(self.cfg.caption_style),
                max_boxes=int(self.cfg.max_boxes),
                erode_iters=int(self.cfg.erode_iters),
                min_pixels=int(self.cfg.min_pixels),
                rand_subarray=int(self.cfg.rand_subarray),
                dbscan_eps=float(self.cfg.dbscan_eps),
                dbscan_min_points=int(self.cfg.dbscan_min_points),
                min_dist=float(self.cfg.min_dist),
                merge_dist=float(self.cfg.merge_dist),
                merge_iau=float(self.cfg.merge_iau),
            )
        )
        self._rng = np.random.RandomState(20261026)
        self._gd_hits: Dict[str, Dict[str, Any]] = {}
        self._buffer: Dict[str, Dict[str, Any]] = {}
        self._frames = 0
        self._gd_frames = 0
        self._proposals = 0
        self._errors = 0
        self._last_latency_ms = 0.0
        self._gd_fail_streak = 0
        self._gd_disabled = False

    # ---- episode lifecycle ----
    def reset(self) -> None:
        self.proposer.reset()
        self._rng = np.random.RandomState(20261026)
        self._gd_hits.clear()
        self._buffer.clear()
        self._frames = 0
        self._gd_frames = 0
        self._proposals = 0
        self._errors = 0
        self._last_latency_ms = 0.0
        self._gd_fail_streak = 0
        self._gd_disabled = False

    # ---- per-step entry ----
    def observe(
        self,
        *,
        step: int,
        rgb: Optional[np.ndarray],
        depth_norm: Optional[np.ndarray],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        target_classes: Sequence[str],
        robot_xyz: np.ndarray,
        robot_yaw: float,
        fov_deg: float,
        candidates: Sequence[Dict[str, Any]],
        memory_events: Sequence[Dict[str, Any]],
        tracks: Sequence[Dict[str, Any]],
        done_initializing: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not self.cfg.enabled or not done_initializing:
            return None
        if int(step) % int(self.cfg.record_every_n) != 0:
            return None

        robot = np.asarray(robot_xyz, dtype=np.float64).reshape(-1)
        robot_xy = robot[:2] if robot.size >= 2 else np.zeros(2)
        depth = None if depth_norm is None else np.asarray(depth_norm, dtype=np.float32)
        height, width = (0, 0)
        if depth is not None:
            height, width = int(depth.shape[0]), int(depth.shape[1])
        elif rgb is not None:
            height, width = int(np.asarray(rgb).shape[0]), int(np.asarray(rgb).shape[1])

        candidate_rows, cand_to_mem = self._candidate_rows(candidates, memory_events)
        track_rows = [self._track_row(t, step) for t in tracks]

        record: Dict[str, Any] = {
            "step": int(step),
            "target_classes": [str(c) for c in target_classes],
            "robot_xy": np.asarray(robot_xy, dtype=np.float64).tolist(),
            "robot_xyz": np.asarray(robot[:3], dtype=np.float64).tolist(),
            "robot_yaw": float(robot_yaw),
            "camera": {
                "fx": float(fx),
                "fy": float(fy),
                "width": int(width),
                "height": int(height),
                "min_depth": float(min_depth),
                "max_depth": float(max_depth),
                "fov_deg": float(fov_deg),
            },
            "depth": _depth_summary(depth, min_depth, max_depth),
            "gd": {
                "enabled": True,
                "ran": False,
                "every_n": int(self.cfg.every_n),
                "box_thr": float(self.cfg.box_thr),
                "max_boxes": int(self.cfg.max_boxes),
                "caption_style": str(self.cfg.caption_style),
            },
            "tsp3d": {
                "n_candidates": len(candidate_rows),
                "candidates": candidate_rows,
            },
            "tracks": track_rows,
            "proposals": [],
            "buffer": [],
            "stats": {},
        }

        run_gd = should_run(int(step), int(self.cfg.every_n))
        if self._gd_disabled:
            record["gd"]["reason"] = "disabled_after_failures"
        elif not run_gd:
            record["gd"]["reason"] = "not_scheduled"
        elif rgb is None or depth_norm is None:
            record["gd"]["reason"] = "no_frame"
        else:
            self._run_gd(
                record=record,
                step=int(step),
                rgb=rgb,
                depth=depth,
                tf_camera_to_episodic=tf_camera_to_episodic,
                min_depth=float(min_depth),
                max_depth=float(max_depth),
                fx=float(fx),
                fy=float(fy),
                target_classes=target_classes,
                robot_xy=robot_xy,
                robot_yaw=float(robot_yaw),
                fov_deg=float(fov_deg),
                candidate_rows=candidate_rows,
                track_rows=track_rows,
                cand_to_mem=cand_to_mem,
            )

        self._expire_buffer(int(step))
        record["buffer"] = self._buffer_rows()
        record["stats"] = self._stats()
        self._frames += 1
        return _jsonable(record)

    # ---- GD execution + association ----
    def _run_gd(
        self,
        *,
        record: Dict[str, Any],
        step: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        target_classes: Sequence[str],
        robot_xy: np.ndarray,
        robot_yaw: float,
        fov_deg: float,
        candidate_rows: List[Dict[str, Any]],
        track_rows: List[Dict[str, Any]],
        cand_to_mem: Dict[str, str],
    ) -> None:
        height, width = int(depth.shape[0]), int(depth.shape[1])
        depth_m = np.zeros_like(depth, dtype=np.float32)
        valid = (depth > 0.0) & (depth < 1.0)
        depth_m[valid] = depth[valid] * float(max_depth - min_depth) + float(min_depth)

        cam = CameraView(
            fx=float(fx),
            fy=float(fy),
            height=height,
            width=width,
            tf_camera_to_episodic=np.asarray(tf_camera_to_episodic, dtype=np.float64),
            min_depth=float(min_depth),
            max_depth=float(max_depth),
        )
        image = np.asarray(rgb)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        image_paths: Dict[str, str] = {}
        if self.cfg.save_images and self.cfg.image_dir:
            try:
                import cv2

                os.makedirs(str(self.cfg.image_dir), exist_ok=True)
                stem = os.path.join(str(self.cfg.image_dir), f"step_{int(step):06d}")
                rgb_path = stem + "_rgb.jpg"
                depth_path = stem + "_depth_m.npy"
                cv2.imwrite(rgb_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                np.save(depth_path, depth_m)
                image_paths = {"rgb": rgb_path, "depth_m": depth_path}
            except BaseException as exc:  # noqa: BLE001 - image dump is optional
                image_paths = {"error": f"{type(exc).__name__}: {exc}"}

        start = time.perf_counter()
        trace_override: Optional[Dict[str, Any]] = None
        try:
            proposals = self.proposer.propose(
                image,
                depth_m,
                cam,
                target_classes,
                robot_xy=robot_xy,
                rng=self._rng,
            )
        except BaseException as exc:  # noqa: BLE001 - calibration must never kill nav
            self._errors += 1
            proposals = []
            trace_override = {
                "ran": False,
                "reason": "exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._last_latency_ms = float(latency_ms)
        trace = trace_override or dict(getattr(self.proposer, "last_trace", {}) or {})
        self._gd_frames += 1
        if trace.get("reason") in ("no_gd", "exception"):
            self._gd_fail_streak += 1
            if self._gd_fail_streak >= int(self.cfg.max_fails):
                self._gd_disabled = True
        elif trace.get("ran"):
            self._gd_fail_streak = 0
        record["gd"].update(
            {
                "ran": bool(trace.get("ran", True)),
                "reason": trace.get("reason"),
                "latency_ms": float(latency_ms),
                "trace": trace,
                "n_proposals": len(proposals),
                "images": image_paths,
            }
        )

        half_fov = np.deg2rad(float(fov_deg)) * 0.5
        camera_pos = np.asarray(tf_camera_to_episodic, dtype=np.float64).reshape(4, 4)[:3, 3]
        proposal_rows: List[Dict[str, Any]] = []
        for idx, proposal in enumerate(proposals):
            row = self._proposal_row(
                step=int(step),
                index=int(idx),
                proposal=proposal,
                robot_xy=robot_xy,
                robot_yaw=float(robot_yaw),
                camera_pos=camera_pos,
                half_fov=float(half_fov),
                max_depth=float(max_depth),
                candidate_rows=candidate_rows,
                track_rows=track_rows,
                cand_to_mem=cand_to_mem,
            )
            proposal_rows.append(row)
            self._update_buffer(row, int(step))

        self._proposals += len(proposal_rows)
        record["proposals"] = proposal_rows
        # Remove buffer entries that this frame already lifted to a track.
        for row in proposal_rows:
            key = row.get("buffer_key")
            if key and row.get("match_track", {}).get("same_object"):
                self._buffer.pop(str(key), None)

    def _proposal_row(
        self,
        *,
        step: int,
        index: int,
        proposal: GdProposal,
        robot_xy: np.ndarray,
        robot_yaw: float,
        camera_pos: np.ndarray,
        half_fov: float,
        max_depth: float,
        candidate_rows: List[Dict[str, Any]],
        track_rows: List[Dict[str, Any]],
        cand_to_mem: Dict[str, str],
    ) -> Dict[str, Any]:
        centroid = np.asarray(proposal.centroid, dtype=np.float64).reshape(-1)
        surface = np.asarray(proposal.surface, dtype=np.float64).reshape(-1)
        box8 = np.asarray(proposal.box8, dtype=np.float64).reshape(-1, 3)
        dist_robot = float(np.linalg.norm(centroid[:2] - robot_xy[:2]))
        dist_cam = float(np.linalg.norm(centroid[:3] - np.asarray(camera_pos, dtype=np.float64).reshape(-1)[:3]))
        azimuth = float(
            np.arctan2(centroid[1] - robot_xy[1], centroid[0] - robot_xy[0]) - robot_yaw
        )
        azimuth = _wrap_angle(azimuth)

        match_candidate = self._match_candidate(proposal, candidate_rows)
        match_track = self._match_track(proposal, track_rows)
        memory_id = match_track.get("memory_id")
        candidate_id = match_candidate.get("candidate_id")
        # Track-level positive vote.  A candidate-only match is useful for
        # coverage/merge analysis but never becomes a track flag.
        gd_pos = bool(match_track.get("same_object"))
        gd_candidate_pos = bool(match_candidate.get("same_object"))

        new_info = True
        hits_before = 0
        hits_after = 0
        key: Optional[str] = None
        if memory_id:
            key = str(memory_id)
            prev = self._gd_hits.get(key, {})
            hits_before = int(prev.get("count", 0))
            new_info = self._is_new_info(prev, robot_xy, robot_yaw, step)
            if gd_pos and new_info:
                hits_after = hits_before + 1
                self._gd_hits[key] = {
                    "count": hits_after,
                    "first_step": int(prev.get("first_step", step)),
                    "last_step": int(step),
                    "last_robot_xy": np.asarray(robot_xy, dtype=np.float64).tolist(),
                    "last_robot_yaw": float(robot_yaw),
                    "conf_max": max(float(prev.get("conf_max", 0.0)), float(proposal.conf)),
                    "conf_last": float(proposal.conf),
                    "target_class": str(proposal.canon or ""),
                }
            elif gd_pos:
                hits_after = hits_before
                current = dict(prev)
                current.update(
                    {
                        "last_step": int(step),
                        "conf_last": float(proposal.conf),
                        "conf_max": max(float(current.get("conf_max", 0.0)), float(proposal.conf)),
                    }
                )
                self._gd_hits[key] = current
            else:
                hits_after = hits_before

        buffer_key = None
        if not gd_pos:
            buffer_key = self._buffer_key(proposal, memory_id)

        return {
            "proposal_id": f"g{step:06d}_{index:02d}",
            "index": int(index),
            "box_px": np.asarray(proposal.box_px, dtype=np.float64).reshape(4).tolist(),
            "conf": float(proposal.conf),
            "phrase": str(proposal.phrase),
            "canon": proposal.canon,
            "centroid": centroid[:3].tolist(),
            "surface": surface[:3].tolist(),
            "box8": box8.tolist(),
            "n_points": int(proposal.n_points),
            "suspect": bool(proposal.suspect),
            "diag": _jsonable(proposal.diag),
            "points_summary": self._points_summary(proposal.points_world),
            "points_world": (
                np.asarray(proposal.points_world, dtype=np.float64)[
                    : int(self.cfg.max_points_saved)
                ].tolist()
                if self.cfg.save_points
                else None
            ),
            "dist_robot": dist_robot,
            "dist_camera": dist_cam,
            "azimuth_deg": float(np.rad2deg(azimuth)),
            "in_fov": bool(abs(azimuth) <= half_fov),
            "depth_ok": bool(dist_cam <= float(max_depth) * 0.95),
            "match_candidate": match_candidate,
            "match_track": match_track,
            "gd_pos": gd_pos,
            "gd_candidate_pos": gd_candidate_pos,
            "new_info": bool(new_info),
            "gd_hits_before": int(hits_before),
            "gd_hits_after": int(hits_after),
            "gd_hits_key": key,
            "buffer_key": buffer_key,
            "matched_memory_id": memory_id,
            "matched_candidate_id": candidate_id,
            "matched_memory_uid_src": cand_to_mem.get(str(candidate_id)) if candidate_id else None,
        }

    # ---- matching ----
    def _match_candidate(
        self, proposal: GdProposal, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        best: Optional[Dict[str, Any]] = None
        best_score: Optional[Tuple[bool, float]] = None
        prop_centroid = np.asarray(proposal.centroid, dtype=np.float64)
        prop_box = np.asarray(proposal.box8, dtype=np.float64)
        for row in candidates:
            centroid = row.get("centroid")
            if centroid is None:
                continue
            centroid = np.asarray(centroid, dtype=np.float64).reshape(-1)
            box = row.get("box8")
            box = None if box is None else np.asarray(box, dtype=np.float64).reshape(-1, 3)
            dist_xy = planar_distance(prop_centroid, centroid)
            dist_3d = float(np.linalg.norm(prop_centroid[:3] - centroid[:3]))
            iau = float(iau_3d(prop_box, box)) if box is not None else None
            cont = float(containment(prop_box, box)) if box is not None else None
            cont_rev = float(containment(box, prop_box)) if box is not None else None
            same = bool(
                dist_xy <= float(self.cfg.merge_dist)
                or (iau is not None and iau >= float(self.cfg.merge_iau))
                or (cont is not None and cont >= float(self.cfg.merge_containment))
                or (cont_rev is not None and cont_rev >= float(self.cfg.merge_containment))
            )
            score = (same, -dist_xy)
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "candidate_id": row.get("candidate_id"),
                    "memory_id": row.get("memory_id"),
                    "dist_xy": dist_xy,
                    "dist_3d": dist_3d,
                    "iau": iau,
                    "containment": cont,
                    "containment_reverse": cont_rev,
                    "same_object": same,
                    "conf": row.get("conf"),
                    "sigma_c": row.get("sigma_c"),
                    "admitted": row.get("admitted"),
                    "reject": row.get("reject"),
                    "occ_passed": row.get("occ_passed"),
                    "occ_reason": row.get("occ_reason"),
                    "geom_status": row.get("geom_status"),
                    "lifecycle_state": row.get("lifecycle_state"),
                    "num_obs": row.get("num_obs"),
                }
        return best or {
            "candidate_id": None,
            "memory_id": None,
            "dist_xy": None,
            "dist_3d": None,
            "iau": None,
            "containment": None,
            "containment_reverse": None,
            "same_object": False,
        }

    def _match_track(
        self, proposal: GdProposal, tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        best: Optional[Dict[str, Any]] = None
        best_score: Optional[Tuple[bool, float]] = None
        prop_centroid = np.asarray(proposal.centroid, dtype=np.float64)
        prop_box = np.asarray(proposal.box8, dtype=np.float64)
        for row in tracks:
            centroid = row.get("centroid")
            if centroid is None:
                continue
            centroid = np.asarray(centroid, dtype=np.float64).reshape(-1)
            box = row.get("box8")
            box = None if box is None else np.asarray(box, dtype=np.float64).reshape(-1, 3)
            dist_xy = planar_distance(prop_centroid, centroid)
            dist_3d = float(np.linalg.norm(prop_centroid[:3] - centroid[:3]))
            iau = float(iau_3d(prop_box, box)) if box is not None else None
            cont = float(containment(prop_box, box)) if box is not None else None
            cont_rev = float(containment(box, prop_box)) if box is not None else None
            same = bool(
                dist_xy <= float(self.cfg.merge_dist)
                or (iau is not None and iau >= float(self.cfg.merge_iau))
                or (cont is not None and cont >= float(self.cfg.merge_containment))
                or (cont_rev is not None and cont_rev >= float(self.cfg.merge_containment))
            )
            score = (same, -dist_xy)
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "memory_id": row.get("memory_id"),
                    "target_class": row.get("target_class"),
                    "dist_xy": dist_xy,
                    "dist_3d": dist_3d,
                    "iau": iau,
                    "containment": cont,
                    "containment_reverse": cont_rev,
                    "same_object": same,
                    "lifecycle_state": row.get("lifecycle_state"),
                    "num_obs": row.get("num_obs"),
                    "sigma_c_max": row.get("sigma_c_max"),
                    "sigma_c_latest": row.get("sigma_c_latest"),
                    "sigma_c_ema_03": row.get("sigma_c_ema_03"),
                    "sigma_c_ema_05": row.get("sigma_c_ema_05"),
                    "sigma_c_ema_08": row.get("sigma_c_ema_08"),
                    "gd_hits": row.get("gd_hits"),
                    "gd_pos": row.get("gd_pos"),
                }
        return best or {
            "memory_id": None,
            "target_class": None,
            "dist_xy": None,
            "dist_3d": None,
            "iau": None,
            "containment": None,
            "containment_reverse": None,
            "same_object": False,
        }

    # ---- new information / buffer ----
    def _is_new_info(
        self,
        prev: Dict[str, Any],
        robot_xy: np.ndarray,
        robot_yaw: float,
        step: int,
    ) -> bool:
        if not prev:
            return True
        last_xy = prev.get("last_robot_xy")
        last_yaw = prev.get("last_robot_yaw")
        last_step = prev.get("last_step")
        if last_xy is None or last_yaw is None or last_step is None:
            return True
        dist = float(np.linalg.norm(np.asarray(robot_xy, dtype=np.float64) - np.asarray(last_xy, dtype=np.float64)))
        dyaw = abs(float(np.rad2deg(_wrap_angle(float(robot_yaw) - float(last_yaw)))))
        dstep = abs(int(step) - int(last_step))
        return bool(
            dist >= float(self.cfg.new_info_dist)
            or dyaw >= float(self.cfg.new_info_yaw_deg)
            or dstep >= int(self.cfg.new_info_steps)
        )

    def _buffer_key(self, proposal: GdProposal, memory_id: Optional[str]) -> str:
        if memory_id:
            return str(memory_id)
        centroid = np.asarray(proposal.centroid, dtype=np.float64)
        return "unmatched:{}:{:.2f}:{:.2f}:{:.2f}".format(
            str(proposal.canon or "na"), float(centroid[0]), float(centroid[1]), float(centroid[2])
        )

    def _update_buffer(self, row: Dict[str, Any], step: int) -> None:
        key = row.get("buffer_key")
        if not key:
            return
        current = self._buffer.get(str(key), {})
        self._buffer[str(key)] = {
            "key": str(key),
            "first_step": int(current.get("first_step", step)),
            "last_step": int(step),
            "hits": int(current.get("hits", 0)) + 1,
            "conf_max": max(float(current.get("conf_max", 0.0)), float(row.get("conf", 0.0))),
            "conf_last": float(row.get("conf", 0.0)),
            "centroid": row.get("centroid"),
            "canon": row.get("canon"),
            "box_px": row.get("box_px"),
        }

    def _expire_buffer(self, step: int) -> None:
        ttl = int(self.cfg.ttl_steps)
        for key in list(self._buffer.keys()):
            if int(step) - int(self._buffer[key]["last_step"]) > ttl:
                del self._buffer[key]

    def _buffer_rows(self) -> List[Dict[str, Any]]:
        return [dict(v) for v in self._buffer.values()]

    # ---- snapshots ----
    def _candidate_rows(
        self,
        candidates: Sequence[Dict[str, Any]],
        memory_events: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        cand_to_mem: Dict[str, str] = {}
        cand_to_state: Dict[str, str] = {}
        cand_to_obs: Dict[str, int] = {}
        for event in memory_events:
            cid = event.get("candidate_id")
            if not cid:
                continue
            if event.get("memory_id") is not None:
                cand_to_mem[str(cid)] = str(event.get("memory_id"))
            if event.get("lifecycle_state") is not None:
                cand_to_state[str(cid)] = str(event.get("lifecycle_state"))
            if event.get("num_obs") is not None:
                cand_to_obs[str(cid)] = int(event.get("num_obs"))
        rows: List[Dict[str, Any]] = []
        for row in candidates:
            cid = row.get("candidate_id")
            box = row.get("box")
            rows.append(
                {
                    "candidate_id": cid,
                    "conf": row.get("conf"),
                    "sigma_c": row.get("sigma_c"),
                    "s": row.get("s"),
                    "w_s": row.get("w_s"),
                    "centroid": row.get("centroid"),
                    "surface": row.get("surface"),
                    "box8": box,
                    "distance": row.get("distance"),
                    "pass_prop_045": row.get("pass_prop_045"),
                    "pass_prop_050": row.get("pass_prop_050"),
                    "pass_prop_055": row.get("pass_prop_055"),
                    "pass_prop_060": row.get("pass_prop_060"),
                    "admitted": row.get("admitted"),
                    "reject": row.get("reject"),
                    "occ_passed": row.get("occ_passed"),
                    "occ_reason": row.get("occ_reason"),
                    "geom_status": row.get("geom_status"),
                    "provisional_gate_state": row.get("provisional_gate_state"),
                    "provisional_gate_outcome": row.get("provisional_gate_outcome"),
                    "memory_written": row.get("memory_written"),
                    "memory_merged": row.get("memory_merged"),
                    "memory_lifecycle_states": row.get("memory_lifecycle_states"),
                    "memory_id": cand_to_mem.get(str(cid)) if cid else None,
                    "lifecycle_state": cand_to_state.get(str(cid)) if cid else None,
                    "num_obs": cand_to_obs.get(str(cid)) if cid else row.get("num_obs"),
                }
            )
        return rows, cand_to_mem

    def _track_row(self, track: Dict[str, Any], step: int) -> Dict[str, Any]:
        memory_id = track.get("memory_id")
        hits = self._gd_hits.get(str(memory_id), {}) if memory_id is not None else {}
        last_step = hits.get("last_step")
        row = {
            "memory_id": memory_id,
            "target_class": track.get("target_class"),
            "centroid": track.get("centroid"),
            "box8": track.get("box8"),
            "lifecycle_state": track.get("lifecycle_state"),
            "num_obs": track.get("num_obs"),
            "sigma_c_max": track.get("sigma_c_max"),
            "sigma_c_latest": track.get("sigma_c_latest"),
            "sigma_c_ema_03": track.get("sigma_c_ema_03"),
            "sigma_c_ema_05": track.get("sigma_c_ema_05"),
            "sigma_c_ema_08": track.get("sigma_c_ema_08"),
            "last_obs_step": track.get("last_obs_step"),
            "near_miss": track.get("near_miss"),
            "total_miss": track.get("total_miss"),
            "revive_count": track.get("revive_count"),
            "src": track.get("src"),
            "stale_steps": track.get("stale_steps"),
            "gd_hits": int(hits.get("count", 0)),
            "gd_pos": bool(hits.get("count", 0) > 0),
            "gd_last_step": last_step,
            "gd_age": None if last_step is None else int(step) - int(last_step),
            "gd_conf_max": hits.get("conf_max"),
            "gd_conf_last": hits.get("conf_last"),
        }
        return row

    def _points_summary(self, points: np.ndarray) -> Dict[str, Any]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return {"n": 0}
        return {
            "n": int(len(pts)),
            "min": pts.min(axis=0).tolist(),
            "max": pts.max(axis=0).tolist(),
            "mean": pts.mean(axis=0).tolist(),
            "std": pts.std(axis=0).tolist(),
        }

    def _stats(self) -> Dict[str, Any]:
        summary = {}
        try:
            summary = dict(self.proposer.summary())
        except Exception:  # noqa: BLE001
            summary = {}
        return {
            "frames_logged": int(self._frames + 1),
            "gd_frames": int(self._gd_frames),
            "proposals": int(self._proposals),
            "errors": int(self._errors),
            "gd_fail_streak": int(self._gd_fail_streak),
            "gd_disabled": bool(self._gd_disabled),
            "last_latency_ms": float(self._last_latency_ms),
            "proposer": summary,
            "buffer_size": len(self._buffer),
            "tracked_keys": len(self._gd_hits),
        }


__all__ = ["GdCalibConfig", "GdCalibrationLogger"]
