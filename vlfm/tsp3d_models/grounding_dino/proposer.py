from typing import Any, List, Optional, Sequence
import numpy as np

from .boxes import aabb_corners, aabb_center, intra_camera_distance, nearest_point
from .client import GdDetector
from .config import GdProposalConfig
from .region import region_points_world, too_offset_rect
from .types import CameraView, GdProposal, ProposalStats
from .vocab import DEFAULT_VOCAB, build_alias_map, build_caption, match_phrase


def should_run(step: int, every_n: int) -> bool:
    """Frame-rate guard: GD runs on every `every_n`-th step (1 = every frame)."""
    return int(step) % max(1, int(every_n)) == 0


class GdProposer:
    """Stateless w.r.t. the episode except for `stats` (counters only)."""

    def __init__(
        self,
        config: Optional[GdProposalConfig] = None,
        detector: Optional[GdDetector] = None,
        vocab: Optional[Sequence[str]] = None,
    ) -> None:
        self.cfg = config or GdProposalConfig()
        self.detector = detector if detector is not None else GdDetector(self.cfg.port, self.cfg.enabled)
        self.vocab = list(vocab) if vocab is not None else list(DEFAULT_VOCAB)
        self.alias_map = build_alias_map(self.vocab)
        self.stats = ProposalStats()

    def reset(self) -> None:
        self.stats.reset()
        self.detector.reset()

    # ---- per-frame entry point ----
    def propose(
        self,
        rgb: Optional[np.ndarray],
        depth_m: Optional[np.ndarray],
        cam: CameraView,
        target_classes: Sequence[str],
        robot_xy: Optional[np.ndarray] = None,
        rng: Optional[Any] = None,
    ) -> List[GdProposal]:
        """
        GD-only candidates of this frame (`[]` = GD ran and proposed nothing).

        `rng` should be a caller-owned `np.random.RandomState` (the chain subsamples 5000 points); 
        passing it keeps the run reproducible without touching the process-global numpy stream.
        """
        if not self.cfg.enabled or rgb is None or depth_m is None:
            self.stats.bump("no_frame")
            return []

        caption = build_caption(target_classes, self.cfg.caption_style, vocab=self.vocab)
        det = self.detector.detect(rgb, caption)
        if det is None:                      # server down / disabled -> "not checked"
            self.stats.bump("no_gd")
            return []
        self.stats.frames += 1

        boxes_norm, scores, phrases = det
        if len(boxes_norm) == 0:
            return []
        
        boxes_px = boxes_norm * np.array([cam.width, cam.height, cam.width, cam.height], dtype=np.float64)
        picked: List[tuple] = []
        for i in range(len(boxes_px)):
            conf = float(scores[i]) if i < len(scores) else 0.0
            if conf < float(self.cfg.box_thr):
                self.stats.bump("below_thr")
                continue
            match = match_phrase(phrases[i] if i < len(phrases) else "", target_classes, self.alias_map)
            # A4 measurement #6: count EVERY above-threshold box's phrase, including the ones
            # the strict single-class rule rejects (that is exactly the multi-class share).
            _phr = str(phrases[i]) if i < len(phrases) else ""
            self.stats.bump_phrase(_phr, int(match.n_named), bool(match.is_target), not _phr.strip())
            if not match.is_target:          # strict single-class rule (T1/T2)
                self.stats.bump("other_class")
                continue
            picked.append((i, conf, match))
        self.stats.boxes_above_thr += len(picked)

        picked.sort(key=lambda t: -t[1])
        proposals: List[GdProposal] = []
        for i, conf, match in picked[: max(1, int(self.cfg.max_boxes))]:
            proposal = self._build(boxes_px[i], conf, phrases[i], match.canon, depth_m, cam, robot_xy, rng)
            if proposal is not None:
                proposals.append(proposal)
        self.stats.proposals += len(proposals)
        return proposals

    # ---- one box -> one candidate ----
    def _build(
        self,
        box_px: np.ndarray,
        conf: float,
        phrase: str,
        canon: Optional[str],
        depth_m: np.ndarray,
        cam: CameraView,
        robot_xy: Optional[np.ndarray],
        rng: Optional[Any],
    ) -> Optional[GdProposal]:
        world, diag = region_points_world(
            box_px,
            depth_m,
            cam,
            erode_iters=self.cfg.erode_iters,
            rand_subarray=self.cfg.rand_subarray,
            dbscan_eps=self.cfg.dbscan_eps,
            dbscan_min_points=self.cfg.dbscan_min_points,
            min_pixels=self.cfg.min_pixels,
            rng=rng,
        )
        if len(world) == 0:
            self.stats.bump(str(diag.get("reject") or "no_cluster"))
            return None

        cam_pos = np.asarray(cam.tf_camera_to_episodic, dtype=np.float64)[:3, 3]
        d_min = intra_camera_distance(world, cam_pos)
        if d_min < float(self.cfg.min_dist):        # VLFM near rejection
            self.stats.bump("too_close")
            return None

        box8, _bounds = aabb_corners(world)
        return GdProposal(
            box_px=np.asarray(box_px, dtype=np.float64).reshape(4),
            conf=float(conf),
            phrase=str(phrase),
            canon=canon,
            points_world=world,
            centroid=aabb_center(world),
            surface=nearest_point(world, np.zeros(2) if robot_xy is None else robot_xy),
            box8=box8,
            n_points=int(len(world)),
            suspect=too_offset_rect(box_px, int(cam.width), int(cam.height)),
            diag={"d_min": d_min, **{k: v for k, v in diag.items() if k != "reject"}},
        )

    # ---- diagnostics ----
    def summary(self) -> dict:
        s = self.stats
        return {
            "frames": s.frames,
            "boxes_above_thr": s.boxes_above_thr,
            "proposals": s.proposals,
            "rejects": dict(s.rejects),
            # A4 measurement #6 (§3.7.10): phrase shape of the above-threshold boxes.
            "phrases": s.top_phrases(20),
            "multi_named": s.multi_named,
            "empty_phrase": s.empty_phrase,
            "target_hit": s.target_hit,
        }
