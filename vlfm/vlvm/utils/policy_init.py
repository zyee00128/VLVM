"""Policy mechanism initialisers (spatial map / memory / fusion / gates).

Moved out of ``TSP3DObjectNavPolicy`` so the policy file only keeps the
per-step control flow.  These are plain mixin methods: they operate on
``self`` exactly like before the split.
"""

import os
from typing import Any, Optional

import numpy as np

from vlfm.mapping.obstacle_map import ObstacleMap3D, ProbabilisticGrid
from vlfm.tsp3d_models.fussion_utils.pipeline import TSP3DInputPreprocessor
from vlfm.vlvm.memory_life.s_penalty import SPenaltyConfig
from vlfm.vlvm.memory_life.target_geometric_gating import (
    BoxOccupancyConfig,
    TargetGeometricGatingEngine,
)
from vlfm.vlvm.memory_life.target_memory_manager import (
    MemoryManagerConfig,
    ProvisionalGate,
    ProvisionalGateConfig,
    TargetMemoryManager,
)


class PolicyInitMixin:
    def _init_spatial_representations(
        self,
        *,
        om_style: str,
        voxel_size: float,
        min_obstacle_height: float,
        max_obstacle_height: float,
        agent_radius: float,
        nav_slice_height: float,
        agent_height: float,
        hole_area_thresh: int,
        obstacle_map_area_threshold: float,
        log_odds_occ: float,
        log_odds_free: float,
        occ_threshold: float,
        free_threshold: float,
    ) -> None:
        """Core 3D spatial representations: the occupancy / obstacle map backend."""
        height_range = max_obstacle_height - min_obstacle_height
        height_size = int(height_range / voxel_size) + 1
        pixels_per_meter = int(1.0 / voxel_size)
        size = 400
        if self._om_style == "obstacle":
            self._obstacle_map3d = ObstacleMap3D(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                agent_radius=agent_radius,
                area_thresh=obstacle_map_area_threshold,
                hole_area_thresh=hole_area_thresh,
                size=size,
                pixels_per_meter=pixels_per_meter,
                voxel_size=voxel_size,
                height_size=height_size,
                nav_slice_height=nav_slice_height,
                agent_height=agent_height,
                compute_navigable=self._visualize,
            )
        elif self._om_style == "probabilistic":
            self._obstacle_map3d = ProbabilisticGrid(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                agent_radius=agent_radius,
                area_thresh=obstacle_map_area_threshold,
                hole_area_thresh=hole_area_thresh,
                size=size,
                pixels_per_meter=pixels_per_meter,
                voxel_size=voxel_size,
                height_size=height_size,
                nav_slice_height=nav_slice_height,
                agent_height=agent_height,
                compute_navigable=self._visualize,
                log_odds_occ=self._log_odds_occ,
                log_odds_free=self._log_odds_free,
                occ_threshold=self._occ_threshold,
                free_threshold=self._free_threshold,
            )

    def _init_memory_manager(
        self,
        *,
        merge_dist_thresh: float,
        ema_weight_old: float,
        upgrade_min_obs: int = 4,
        enable_stable_pool: bool = True,
        enable_gd_pool: bool = False,
        enable_provisional_gate: bool = True,
        provisional_sigma: float = 0.55,
        provisional_min_dist: float = 1.0,
        provisional_max_dist: float = 4.0,
        provisional_s_min: float = 0.07,
        merge_new_info_dist: float = 0.2,
        merge_new_info_yaw: float = 10.0,
        merge_new_info_steps: int = 3,
        stable_stale_purge: int = 60,
        stable_miss_delete: int = 5,
        stable_total_miss: int = 8,
        stable_revive_radius: float = 0.8,
        stable_revive_window: int = 20,
        stable_revive_min_sigma: float = 0.50,
        confirmed_to_stable_misses: int = 3,
        provisional_to_stable_misses: int = 3,
        lock_exempt: bool = True,
        lock_exempt_identity: bool = True,
    ) -> None:
        """
        Target-memory lifecycle manager (EMA merge + stable pool + erasure)
        bound to the policy's memory dicts.
        """
        self._memory_manager = TargetMemoryManager(MemoryManagerConfig(
            near_field_radius=float(self._max_depth) * 0.5,
            merge_dist_thresh=merge_dist_thresh,
            ema_weight_old=ema_weight_old,
            upgrade_min_obs=upgrade_min_obs,
            enable_stable_pool=enable_stable_pool,
            enable_gd_pool=enable_gd_pool,
            merge_new_info_dist=merge_new_info_dist,
            merge_new_info_yaw=merge_new_info_yaw,
            merge_new_info_steps=merge_new_info_steps,
            stable_stale_purge=stable_stale_purge,
            stable_miss_delete=stable_miss_delete,
            stable_total_miss=stable_total_miss,
            stable_revive_radius=stable_revive_radius,
            stable_revive_window=stable_revive_window,
            stable_revive_min_sigma=stable_revive_min_sigma,
            confirmed_to_stable_misses=confirmed_to_stable_misses,
            provisional_to_stable_misses=provisional_to_stable_misses,
        ))
        self._memory_manager.bind(
            self._target_3d_memory, self._target_surface_memory,
            self._target_verify_state, self._target_lifecycle_state,
        )
        self._lock_exempt = bool(lock_exempt)
        self._lock_exempt_identity = bool(lock_exempt_identity)
        self._lock_step: int = -1
        self._provisional_gate = ProvisionalGate(ProvisionalGateConfig(
            enabled=enable_provisional_gate,
            confirmed_sigma=self._sigma_tar,
            provisional_sigma=provisional_sigma,
            provisional_min_dist=provisional_min_dist,
            provisional_max_dist=provisional_max_dist,
            s_min=provisional_s_min,
        ))

    def _init_gd_mechanism(
        self,
        *,
        enabled: bool,
        conf_min: float,
        max_dist: float,
        gd_only_classes: Any,
    ) -> None:
        """Initialise GD calibration and the GD mechanism.

        ``enabled`` is the single binary switch for the GD mechanism.  The
        calibration logger remains available log-only via ``GD_CALIB_ENABLE``
        so offline GD calibration does not require the mechanism.
        """
        self._gd_calib = None
        self._gd_calib_logs: List[Dict[str, Any]] = []
        self._gd_calib_error: Optional[str] = None
        self._gd_mech = None
        self._gd_mech_error: Optional[str] = None
        self._gd_mech_last_step: int = -1
        calib_requested = (
            os.environ.get("GD_CALIB_ENABLE", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if not bool(enabled) and not calib_requested:
            return
        try:
            from vlfm.vlvm.grounding_dino.calibration import (
                GdCalibConfig,
                GdCalibrationLogger,
            )

            gd_cfg = GdCalibConfig.from_env()
            if bool(enabled):
                gd_cfg.enabled = True
            if gd_cfg.enabled:
                self._gd_calib = GdCalibrationLogger(gd_cfg)
                print(
                    "[GD-CALIB] enabled "
                    f"every_n={gd_cfg.every_n} record_every_n={gd_cfg.record_every_n} "
                    f"box_thr={gd_cfg.box_thr} max_boxes={gd_cfg.max_boxes} "
                    f"mechanism={bool(enabled)}",
                    flush=True,
                )

            if bool(enabled):
                from vlfm.vlvm.grounding_dino.mechanism import (
                    GdMechanism,
                    GdMechanismConfig,
                )

                self._gd_mech = GdMechanism(
                    memory_manager=self._memory_manager,
                    geom_gate=self._geom_gate_engine,
                    obstacle_map3d=self._obstacle_map3d,
                    config=GdMechanismConfig(
                        enabled=True,
                        conf_min=float(conf_min),
                        max_dist_m=float(max_dist),
                        gd_only_classes=tuple(gd_only_classes),
                    ),
                )
                print(
                    "[GD-MECH] enabled "
                    f"conf_min={conf_min} max_dist={max_dist} "
                    f"classes={tuple(gd_only_classes) or 'ALL'} "
                    "depth=gd-only-required/gd-tsp-track-merge "
                    "blacklist-promotion-gate",
                    flush=True,
                )
        except BaseException as exc:  # noqa: BLE001 - GD must not kill nav
            self._gd_calib_error = f"{type(exc).__name__}: {exc}"
            self._gd_mech_error = self._gd_calib_error
            print(f"[GD-MECH] init failed: {self._gd_calib_error}", flush=True)

    def _init_fusion(
        self,
        *,
        fusion_style: str,
        fuse_max_points: int,
        wm_voxel_size: float,
        wm_radius: Optional[float],
        wm_min_view_disp: float,
        wm_min_view_yaw: float,
        wm_max_voxels: int,
        wm_max_frames: Optional[int],
        wm_near_refresh_radius: Optional[float],
        wm_near_refresh_value: bool,
        cap_style: str,
        distance_sample: bool,
        near_dist: float,
        mid_dist: float,
        near_voxel: float,
        mid_voxel: float,
        far_voxel: float,
    ) -> None:
        """Fusion route: world-frame local map or no fusion (raw current frame)."""
        if fusion_style not in ("world", "none"):
            raise ValueError("fusion_style MUST BE 'world' or 'none'")
        self._fusion_style = fusion_style
        self._last_lock_step: int = 0
        self._preprocessor = TSP3DInputPreprocessor(
            fusion_style=fusion_style,
            map_voxel_size=wm_voxel_size,
            map_radius=wm_radius,
            min_view_disp=wm_min_view_disp,
            min_view_yaw=np.deg2rad(wm_min_view_yaw),
            max_map_voxels=wm_max_voxels,
            map_max_frames=wm_max_frames,
            map_near_refresh_radius=wm_near_refresh_radius,
            map_near_refresh_value=wm_near_refresh_value,
            max_points=fuse_max_points,
            cap_style=cap_style,
            use_distance_sampling=distance_sample,
            near_dist=near_dist,
            mid_dist=mid_dist,
            near_voxel=near_voxel,
            mid_voxel=mid_voxel,
            far_voxel=far_voxel,
        )

    def _init_s_penalty(
        self,
        *,
        enable_s_penalty: bool,
        s_penalty_thresh: float,
        s_penalty_uncovered_w: float,
        s_penalty_use_surface: bool,
        goal_use_surface: bool,
        s_min: float = 0.07,
        s_query_radius_m: float = 0.5,
        det_log_dir: Optional[str] = None,
    ) -> None:
        """
        S-penalty: c' = c * w_S (semantic field), plus the surface-point goal option.

        ``det_log_dir`` (when set) only turns on the SP1 per-detection jsonl dump; it
        changes no admission behaviour.
        """
        self._s_penalty_cfg = SPenaltyConfig(
            enable=enable_s_penalty,
            thresh=s_penalty_thresh,
            uncovered_w=s_penalty_uncovered_w,
            s_min=float(s_min),
            query_radius_m=float(s_query_radius_m),
        )
        self._s_penalty_use_surface = s_penalty_use_surface
        self._goal_use_surface = goal_use_surface
        self._det_log_dir = None if det_log_dir in (None, "", "null") else str(det_log_dir)

    def _init_geom_gate(
        self,
        *,
        enable_occ_consistency: bool,
        occ_min_voxels: int,
        occ_min_ratio: float,
    ) -> None:
        """
        Geometric admission gate (occupancy-consistency). Runs before S-penalty.
        """
        self._geom_gate_engine = TargetGeometricGatingEngine(
            occ_cfg=BoxOccupancyConfig(
                enable=enable_occ_consistency,
                min_occupied_voxels=occ_min_voxels,
                min_occupancy_ratio=occ_min_ratio,
            ),
        )

    # ==========================================================================
    # === Input & Mapping Module ===
    # ==========================================================================

