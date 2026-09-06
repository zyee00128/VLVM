from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# 基础几何与 3D Box 工具函数
def decompose_box_8corners(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    通过 SVD 主成分分解从 8 个角点中恢复 3D OBB（有向包围盒）的几何中心、3 个正交轴与半轴长。
    不依赖角点的特定排列顺序，对任何 3DVG 模型输出均鲁棒。

    Args:
        corners: shape (8, 3) 的 3D 角点坐标

    Returns:
        center: shape (3,) 几何中心
        axes: shape (3, 3) 3个正交单位主轴 [u0, u1, u2]
        half_extents: shape (3,) 3个轴方向的半长 [e0, e1, e2]
    """
    center = np.mean(corners, axis=0)
    centered = corners - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt  # shape (3, 3), 每一行是一个单位向量
    projections = np.abs(centered @ axes.T)  # (8, 3)
    half_extents = np.max(projections, axis=0)
    return center, axes, half_extents

def points_in_box_3d(points: np.ndarray, box_corners: np.ndarray, margin: float = 0.02) -> np.ndarray:
    """
    向量化检测点集是否落在 3D OBB 包围盒内。

    Args:
        points: shape (N, 3) 或 (N, >=3) 的点云
        box_corners: shape (8, 3) 的 3D BBox 角点
        margin: 边界容差 (米)

    Returns:
        in_box_mask: shape (N,) bool 数组
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    pts_xyz = points[:, :3]
    center, axes, half_extents = decompose_box_8corners(box_corners)

    rel_pts = pts_xyz - center  # (N, 3)
    proj = np.abs(rel_pts @ axes.T)  # (N, 3)

    in_box_mask = np.all(proj <= (half_extents + margin), axis=1)
    return in_box_mask

def voxel_downsample(points: np.ndarray, voxel_size: float = 0.02) -> np.ndarray:
    """快速 3D 体素网格点云降采样。"""
    if len(points) == 0:
        return points
    coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return points[unique_indices]

def project_points_to_pixels(
    points_world: np.ndarray,
    tf_camera_to_episodic: np.ndarray,
    fx: float,
    fy: float,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将世界坐标系下的 3D 点投影到相机的 2D 像素坐标 (u, v)。
    适配 VLVM/VLFM 的坐标转换约定：
    相机系 base frame 为 [forward, left, up] = [zc, -xc, -yc]。

    Args:
        points_world: shape (N, 3) 世界系坐标
        tf_camera_to_episodic: shape (4, 4) 相机到世界的外参矩阵
        fx, fy: 相机焦距
        img_w, img_h: 图像分辨率宽高

    Returns:
        pixels: shape (N, 2) 像素坐标 [u, v]
        valid_front: shape (N,) bool 是否在相机前方 (zc > 0.1m)
    """
    R = tf_camera_to_episodic[:3, :3]
    t = tf_camera_to_episodic[:3, 3]

    # 世界坐标变换回相机基座系 (Camera Base Frame)
    pts_base = (points_world - t) @ R

    # pts_base: [forward, left, up] = [zc, -xc, -yc]
    zc = pts_base[:, 0]
    xc = -pts_base[:, 1]
    yc = -pts_base[:, 2]

    valid_front = zc > 0.1  # 位于相机前方至少 10cm
    u = np.zeros(len(points_world), dtype=np.float32)
    v = np.zeros(len(points_world), dtype=np.float32)

    u[valid_front] = (xc[valid_front] * fx / zc[valid_front]) + (img_w / 2.0)
    v[valid_front] = (yc[valid_front] * fy / zc[valid_front]) + (img_h / 2.0)

    return np.stack([u, v], axis=-1), valid_front


# M4 & M2（前置）：视锥边缘、远端不可信与近距硬拒绝门控
@dataclass
class FrustumAndDistanceConfig:
    enable_near_reject: bool = False     # M2：近距硬拒绝 (< min_dist_reject 直接丢弃)
    enable_suspicious: bool = False      # M4：远端/视锥边缘检测标记为 suspicious

    # M2：近距硬拒绝阈值
    min_dist_reject: float = 1.0         # 距离相机 < 1.0m 时直接丢弃
    
    # M4：视锥边缘与远端不可信门控
    max_dist_trusted: float = 4.0        # 距离 > 4.0m 标记为 suspicious (远端不可信)
    edge_margin_px: int = 8              # 2D 投影与图像边界的贴合边距容差 (px)
    img_w: int = 224                     # 相机图像宽度
    img_h: int = 224                     # 相机图像高度
class FrustumAndDistanceGate:
    """
    M2 (近距硬拒绝) 与 M4 (视锥边缘/远端不可信) 联合评估器。
    """

    def __init__(self, cfg: Optional[FrustumAndDistanceConfig] = None):
        self.cfg = cfg or FrustumAndDistanceConfig()

    def evaluate(
        self,
        box_corners: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        fx: float,
        fy: float,
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        评估单个 3D Box 的视场有效性与距离定级。

        Args:
            box_corners: shape (8, 3) 3D Box 世界系角点
            tf_camera_to_episodic: shape (4, 4) 相机外参
            fx, fy: 相机焦距

        Returns:
            should_reject: 是否应当被硬丢弃 (M2: 近距拒绝 / 严重越界)
            is_suspicious: 是否应当被标记为可疑/软目标 (M4: 边缘截断或远端)
            diagnostics: 诊断信息
        """
        camera_pos = tf_camera_to_episodic[:3, 3]
        box_center, _, _ = decompose_box_8corners(box_corners)
        dist_to_cam = float(np.linalg.norm(box_center - camera_pos))
        cfg = self.cfg

        # M2/M4 全关 -> 直接放行，保持基线行为
        if not (cfg.enable_near_reject or cfg.enable_suspicious):
            return False, False, {
                "rejected": False, "is_suspicious": False, "reason": "geom_off",
                "dist_to_cam": dist_to_cam, "is_far": False,
                "is_edge_truncated": False, "bbox_2d": None,
            }

        # M2：近距硬拒绝 (< min_dist_reject)
        if cfg.enable_near_reject and dist_to_cam < cfg.min_dist_reject:
            return True, True, {
                "rejected": True,
                "reason": f"near_field_reject (dist={dist_to_cam:.2f}m < {cfg.min_dist_reject}m)",
                "dist_to_cam": dist_to_cam,
                "is_suspicious": True,
            }

        # M4：视锥边缘截断与远端不可信分析
        # 远端距离判定 (> max_dist_trusted)
        is_far = cfg.enable_suspicious and dist_to_cam > cfg.max_dist_trusted

        # 2D 视锥投影与边缘贴合 (too_offset) 检测
        if cfg.enable_suspicious:
            pixels, valid_front = project_points_to_pixels(
                box_corners, tf_camera_to_episodic, fx, fy, cfg.img_w, cfg.img_h
            )

            # 若角点跨越了相机后方 (valid_front 不全为 True)，说明包围盒被视锥极近端严重截断
            if not np.all(valid_front):
                is_edge_truncated = True
                bbox_2d = None
            else:
                u_min, v_min = np.min(pixels, axis=0)
                u_max, v_max = np.max(pixels, axis=0)
                bbox_2d = [float(u_min), float(v_min), float(u_max), float(v_max)]

                # 判定 2D 投影包围盒是否贴紧/越过图像边缘
                is_edge_truncated = (
                    u_min <= cfg.edge_margin_px
                    or v_min <= cfg.edge_margin_px
                    or u_max >= (cfg.img_w - cfg.edge_margin_px)
                    or v_max >= (cfg.img_h - cfg.edge_margin_px)
                )
        else:
            is_edge_truncated = False
            bbox_2d = None

        is_suspicious = is_far or is_edge_truncated

        reason = "trusted_view"
        if is_far and is_edge_truncated:
            reason = "far_and_edge_truncated"
        elif is_far:
            reason = "far_field_weak"
        elif is_edge_truncated:
            reason = "frustum_edge_truncated"

        return False, is_suspicious, {
            "rejected": False,
            "is_suspicious": is_suspicious,
            "reason": reason,
            "dist_to_cam": dist_to_cam,
            "is_far": is_far,
            "is_edge_truncated": is_edge_truncated,
            "bbox_2d": bbox_2d,
        }


# M8：Box–占据一致性门控
@dataclass
class BoxOccupancyConfig:
    enable: bool = False                # M8 独立开关
    min_occupied_voxels: int = 8        # Box 包围区域内至少包含的已占据体素数
    min_occupancy_ratio: float = 0.001  # 占据体素占 Box 总体素体积的最小比例
    check_explored_free: bool = True    # 是否检查穿透状态：若 Box 内全被标记为 Free，立即判假
    free_ratio_rejection_thresh: float = 0.90   # 若 Box 内已知区域有 >90% 是 Free 空气，直接剔除
class BoxOccupancyGate:
    """
    M8：基于已建 3D 占据栅格地图的先验几何物理一致性门控。
    """

    def __init__(self, cfg: Optional[BoxOccupancyConfig] = None):
        self.cfg = cfg or BoxOccupancyConfig()

    def evaluate(
        self,
        box_corners: np.ndarray,
        obstacle_map_3d: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        if not self.cfg.enable:
            return True, {"passed": True, "reason": "geom_off_m8",
                          "num_occ": 0, "num_explored": 0, "occ_ratio": 0.0}
        center, _, _ = decompose_box_8corners(box_corners)

        min_xyz = np.min(box_corners, axis=0)
        max_xyz = np.max(box_corners, axis=0)

        bbox_corners_aabb = np.array([
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
        ])
        grid_idx = obstacle_map_3d._xyz_to_grid_index(bbox_corners_aabb)

        px_min = max(0, min(grid_idx[0, 0], grid_idx[1, 0]))
        px_max = min(obstacle_map_3d.size - 1, max(grid_idx[0, 0], grid_idx[1, 0]))
        py_min = max(0, min(grid_idx[0, 1], grid_idx[1, 1]))
        py_max = min(obstacle_map_3d.size - 1, max(grid_idx[0, 1], grid_idx[1, 1]))
        cz_min = max(0, min(grid_idx[0, 2], grid_idx[1, 2]))
        cz_max = min(obstacle_map_3d._height_size - 1, max(grid_idx[0, 2], grid_idx[1, 2]))

        if px_min > px_max or py_min > py_max or cz_min > cz_max:
            return False, {"reason": "out_of_bounds", "num_occ": 0}

        occ_subgrid = obstacle_map_3d._map[py_min:py_max + 1, px_min:px_max + 1, cz_min:cz_max + 1]
        explored_subgrid = obstacle_map_3d.explored_area[py_min:py_max + 1, px_min:px_max + 1, cz_min:cz_max + 1]

        num_occ = int(np.sum(occ_subgrid))
        num_explored = int(np.sum(explored_subgrid))
        total_sub_voxels = occ_subgrid.size

        # 规则 A: 占据体素硬数量门槛
        if num_occ < self.cfg.min_occupied_voxels:
            return False, {
                "passed": False,
                "reason": f"insufficient_occ_voxels ({num_occ} < {self.cfg.min_occupied_voxels})",
                "num_occ": num_occ,
                "num_explored": num_explored,
            }

        # 规则 B: 穿透反幻觉校验
        if self.cfg.check_explored_free and num_explored > (self.cfg.min_occupied_voxels * 2):
            free_in_box = num_explored - num_occ
            free_ratio = free_in_box / float(num_explored)
            if free_ratio > self.cfg.free_ratio_rejection_thresh and num_occ < (self.cfg.min_occupied_voxels * 2):
                return False, {
                    "passed": False,
                    "reason": f"air_penetration_detected (free_ratio={free_ratio:.2f})",
                    "num_occ": num_occ,
                    "free_ratio": free_ratio,
                }

        # 规则 C: 占据体积比率
        occ_ratio = num_occ / float(total_sub_voxels)
        if occ_ratio < self.cfg.min_occupancy_ratio:
            return False, {
                "passed": False,
                "reason": f"low_occupancy_ratio ({occ_ratio:.4f} < {self.cfg.min_occupancy_ratio})",
                "num_occ": num_occ,
                "occ_ratio": occ_ratio,
            }

        return True, {
            "passed": True,
            "reason": "pass",
            "num_occ": num_occ,
            "num_explored": num_explored,
            "occ_ratio": occ_ratio,
        }


# M3：Box 内深度点累积与空间密度门控
@dataclass
class TargetPointCluster:
    """维护单一目标候选的空间点云池与跨帧时空统计"""
    cluster_id: int
    target_class: str
    centroid: np.ndarray                     # (3,) EMA 几何中心
    points: np.ndarray                       # (K, 3) 累积的高置信表面深度点
    obs_count: int = 1                       # 累积观测帧数
    first_seen_step: int = 0
    last_seen_step: int = 0
    is_confirmed: bool = False               # 是否已跨越硬门槛 (通过密度门控)
    is_suspicious: bool = False              # 最近一次观测是否处于 suspicious 态
    max_confidence: float = 0.0              # 历史最高检测置信度
@dataclass
class BoxPointDensityConfig:
    enable: bool = False                    # M3 独立开关 (默认关 = 跳过密度硬门槛，直接放行)
    min_points_confirm: int = 150           # 硬确认所需的累积物理点云数 (VLFM min_points=100~150)
    min_frames_confirm: int = 2             # 硬确认所需的最少独立观测帧数 (防止单帧暴走)
    cluster_merge_dist: float = 0.50        # 聚类关联半径 (米)，用于跨帧合并
    point_downsample_voxel: float = 0.02    # 点云池体素化分辨率 (2cm)
    max_points_per_cluster: int = 2000      # 单个点云池容量上限
    cluster_decay_steps: int = 50           # 超过此步数未更新的未确认簇自动废弃
class BoxPointDensityAccumulator:
    """
    M3：时空点云累加器与空间密度聚类门控。
    """

    def __init__(self, cfg: Optional[BoxPointDensityConfig] = None):
        self.cfg = cfg or BoxPointDensityConfig()
        self._clusters: List[TargetPointCluster] = []
        self._next_cluster_id = 0

    def reset(self) -> None:
        self._clusters.clear()
        self._next_cluster_id = 0

    def update_and_evaluate(
        self,
        target_class: str,
        box_corners: np.ndarray,
        current_frame_pcd: np.ndarray,
        confidence: float,
        step: int,
        is_suspicious: bool = False,
    ) -> Tuple[bool, TargetPointCluster, Dict[str, Any]]:
        """
        将当前帧 Box 范围内的点云并入聚类池，并评估是否达到硬准入门槛。

        Args:
            target_class: 类别名称
            box_corners: shape (8, 3) 3D Box 角点
            current_frame_pcd: shape (N, 3) 当前帧世界点云
            confidence: 检测置信度
            step: 当前环境步数
            is_suspicious: 来自 M4 的标记（若为 True，本帧不可直接激活 is_confirmed）
        """
        in_mask = points_in_box_3d(current_frame_pcd, box_corners)
        pts_in_box = current_frame_pcd[in_mask, :3]
        pts_in_box = voxel_downsample(pts_in_box, self.cfg.point_downsample_voxel)

        box_center, _, _ = decompose_box_8corners(box_corners)

        matched_cluster: Optional[TargetPointCluster] = None
        min_dist = float("inf")

        for cl in self._clusters:
            if cl.target_class != target_class:
                continue
            dist = float(np.linalg.norm(cl.centroid - box_center))
            if dist < self.cfg.cluster_merge_dist and dist < min_dist:
                min_dist = dist
                matched_cluster = cl

        if matched_cluster is not None:
            if len(pts_in_box) > 0:
                merged_pts = np.vstack([matched_cluster.points, pts_in_box])
                matched_cluster.points = voxel_downsample(merged_pts, self.cfg.point_downsample_voxel)
                if len(matched_cluster.points) > self.cfg.max_points_per_cluster:
                    idx = np.random.choice(len(matched_cluster.points), self.cfg.max_points_per_cluster, replace=False)
                    matched_cluster.points = matched_cluster.points[idx]

            matched_cluster.centroid = 0.8 * matched_cluster.centroid + 0.2 * box_center
            matched_cluster.obs_count += 1
            matched_cluster.last_seen_step = step
            matched_cluster.max_confidence = max(matched_cluster.max_confidence, confidence)
            matched_cluster.is_suspicious = is_suspicious
            cluster = matched_cluster
        else:
            cluster = TargetPointCluster(
                cluster_id=self._next_cluster_id,
                target_class=target_class,
                centroid=box_center.copy(),
                points=pts_in_box.copy(),
                obs_count=1,
                first_seen_step=step,
                last_seen_step=step,
                is_confirmed=False,
                is_suspicious=is_suspicious,
                max_confidence=confidence,
            )
            self._next_cluster_id += 1
            self._clusters.append(cluster)

        # 空间密度门控准入判定：
        # 必须满足点数与帧数条件，且当前不能处于 suspicious 态（M4 约束）
        total_points = len(cluster.points)
        meets_points = total_points >= self.cfg.min_points_confirm
        meets_frames = cluster.obs_count >= self.cfg.min_frames_confirm

        if meets_points and meets_frames and (not is_suspicious):
            cluster.is_confirmed = True

        diagnostics = {
            "cluster_id": cluster.cluster_id,
            "total_points": total_points,
            "obs_count": cluster.obs_count,
            "meets_points": meets_points,
            "meets_frames": meets_frames,
            "is_suspicious": is_suspicious,
            "is_confirmed": cluster.is_confirmed,
        }

        self._prune_stale_clusters(step)
        return cluster.is_confirmed, cluster, diagnostics

    def _prune_stale_clusters(self, current_step: int) -> None:
        self._clusters = [
            cl for cl in self._clusters
            if cl.is_confirmed or (current_step - cl.last_seen_step <= self.cfg.cluster_decay_steps)
        ]


# 统一集成门控引擎
class TargetGeometricGatingEngine:
    """
    前置四重几何空间护栏统一引擎：
    流水线执行：机制 2(前) -> 机制 8 -> 机制 4 -> 机制 3
    """

    def __init__(
        self,
        frustum_cfg: Optional[FrustumAndDistanceConfig] = None,
        occ_cfg: Optional[BoxOccupancyConfig] = None,
        density_cfg: Optional[BoxPointDensityConfig] = None,
    ):
        self.frustum_gate = FrustumAndDistanceGate(frustum_cfg)
        self.occ_gate = BoxOccupancyGate(occ_cfg)
        self.density_acc = BoxPointDensityAccumulator(density_cfg)

    def reset(self) -> None:
        self.density_acc.reset()

    @property
    def enabled(self) -> bool:
        """True if any geometric gate 
        (M2 near-reject / M4 suspicious / M8 occupancy / M3 density) is on.
        """
        return bool(
            self.frustum_gate.cfg.enable_near_reject
            or self.frustum_gate.cfg.enable_suspicious
            or self.occ_gate.cfg.enable
            or self.density_acc.cfg.enable
        )

    def process_detection(
        self,
        target_class: str,
        box_corners: np.ndarray,
        current_frame_pcd: np.ndarray,
        obstacle_map_3d: Any,
        tf_camera_to_episodic: np.ndarray,
        fx: float,
        fy: float,
        confidence: float,
        step: int,
    ) -> Tuple[str, Optional[TargetPointCluster], Dict[str, Any]]:
        """
        四重流水线处理单个检测框：

        Returns:
            status: 'REJECTED' | 'SOFT_ACCUMULATING' | 'HARD_CONFIRMED'
            cluster: 关联的目标聚类池对象（若被 Reject 则为 None）
            diagnostics: 全流水线诊断日志
        """
        diag_summary: Dict[str, Any] = {}

        # M2 + M4 视场有效性与定级
        should_reject, is_suspicious, frustum_diag = self.frustum_gate.evaluate(
            box_corners=box_corners,
            tf_camera_to_episodic=tf_camera_to_episodic,
            fx=fx,
            fy=fy,
        )
        diag_summary["frustum_diag"] = frustum_diag

        if should_reject:
            return "REJECTED", None, {
                "status": "REJECTED",
                "rejected_at": "mechanism_2_near_reject",
                "diagnostics": diag_summary,
            }

        # M8 3D 占据网格物理支撑检验
        occ_passed, occ_diag = self.occ_gate.evaluate(box_corners, obstacle_map_3d)
        diag_summary["occ_diag"] = occ_diag

        if not occ_passed:
            return "REJECTED", None, {
                "status": "REJECTED",
                "rejected_at": "mechanism_8_occupancy_gate",
                "diagnostics": diag_summary,
            }

        # M3 点云空间密度累积与 Hard/Soft 分流
        if self.density_acc.cfg.enable:
            density_passed, cluster, density_diag = self.density_acc.update_and_evaluate(
                target_class=target_class,
                box_corners=box_corners,
                current_frame_pcd=current_frame_pcd,
                confidence=confidence,
                step=step,
                is_suspicious=is_suspicious,
            )
            diag_summary["density_diag"] = density_diag
            status = "HARD_CONFIRMED" if density_passed else "SOFT_ACCUMULATING"
            cluster_id = cluster.cluster_id
            cluster_points = len(cluster.points)
            obs_count = cluster.obs_count
        else:
            # M3 off -> 不做密度硬门槛，直接放行（cluster 无关）
            status = "HARD_CONFIRMED"
            cluster = None
            cluster_id = None
            cluster_points = 0
            obs_count = 0

        return status, cluster, {
            "status": status,
            "is_suspicious": is_suspicious,
            "cluster_id": cluster_id,
            "cluster_points": cluster_points,
            "obs_count": obs_count,
            "diagnostics": diag_summary,
        }
