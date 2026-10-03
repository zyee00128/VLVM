#!/usr/bin/env python3
"""Stage3 微调管道 · 共享几何工具（纯 numpy；无 torch / habitat 依赖）。

坐标系约定（与 `data_preprocess.py`、在线 camera-canonical 完全同源）：
  * canonical 系（帧/查询局部系）：x/y 相对机器人（yaw 对齐，旋转 R(-yaw)），
    z 相对 local floor（0 = 地板，传感器在 0.88）。
  * 位姿记录 `records.poses` = {"x", "y", "floor", "yaw_deg"}：episodic 平面 x/y、
    floor = 绝对高度（habitat y）、yaw_deg = 朝向角（度）。
  * 视锥判定与 `data_preprocess.in_frustum` 完全同口径：
    前向 +x、深度（平面 z）∈ [min_depth, max_depth]、hfov 79°、640×480、相机 (0, 0, 0.88)。

用于：
  * GD 教师质检层（§4（2））的 GT 投影验证 —— `box_visibility`；
  * L_s1 标签面（§6/§8·3）的「视锥内可见实体」判定 —— `box_visibility` 同款；
  * 与 `data_preprocess.SceneRenderer.instance_aabb_surface` 同生成法的表面采样。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class FrameGeometry:
    """相机/视锥参数（默认 = 在线与数据生成的同源 sensor 配置）。"""

    image_h: int = 480
    image_w: int = 640
    hfov_deg: float = 79.0
    camera_height: float = 0.88
    min_depth: float = 0.5
    max_depth: float = 5.0

    @property
    def tan_h(self) -> float:
        return math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def tan_v(self) -> float:
        return (self.image_h / 2.0) / ((self.image_w / 2.0) / self.tan_h)


def pose_canonical_to_frame(
    points_q: np.ndarray,
    q_pose: Dict[str, float],
    f_pose: Dict[str, float],
) -> np.ndarray:
    """查询 canonical 系点 -> 帧 f 的 canonical 系点。

    由 q = R(-yawq)·(p - tq)、f = R(-yawf)·(p - tf) 推得：
        f_xy = R(yawq - yawf)·q_xy + R(-yawf)·(tq - tf)，f_z = q_z + floorq - floorf
    （z 为相对 local floor 的高度；floor 为各自位姿的绝对地面高度）。
    """
    pts = np.asarray(points_q, dtype=np.float64)
    out = pts.copy()
    d_yaw = math.radians(float(q_pose["yaw_deg"])) - math.radians(float(f_pose["yaw_deg"]))
    c, s = math.cos(d_yaw), math.sin(d_yaw)
    xy = pts[:, :2]
    out[:, 0] = c * xy[:, 0] - s * xy[:, 1]
    out[:, 1] = s * xy[:, 0] + c * xy[:, 1]
    dyaw_f = math.radians(float(f_pose["yaw_deg"]))
    cf, sf = math.cos(dyaw_f), math.sin(dyaw_f)
    tx = float(q_pose["x"]) - float(f_pose["x"])
    ty = float(q_pose["y"]) - float(f_pose["y"])
    out[:, 0] += cf * tx + sf * ty
    out[:, 1] += -sf * tx + cf * ty
    out[:, 2] += float(q_pose["floor"]) - float(f_pose["floor"])
    return out


def frustum_mask_frame(points_f: np.ndarray, geo: Optional[FrameGeometry] = None) -> np.ndarray:
    """帧 canonical 系点是否落在相机视锥内（深度 ∈ [min, max]；平面 z 深度）。

    与 `data_preprocess.in_frustum(yaw=0)` 同公式：
        z = rel_x（前向）、x = -rel_y（右正）、y = -rel_z（下正）、|x| <= tan_h·z、|y| <= tan_v·z。
    """
    g = geo or FrameGeometry()
    pts = np.asarray(points_f, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    rel = pts - np.array([0.0, 0.0, g.camera_height])
    z = rel[:, 0]
    x = -rel[:, 1]
    y = -rel[:, 2]
    ok = (z >= g.min_depth) & (z <= g.max_depth)
    zc = np.maximum(z, 1e-6)
    ok &= np.abs(x) <= g.tan_h * zc
    ok &= np.abs(y) <= g.tan_v * zc
    return ok


def aabb_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """轴对齐 AABB 8 角点（center/size 各 3 元）。"""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    s = np.asarray(size, dtype=np.float64).reshape(3)
    lo, hi = c - s / 2.0, c + s / 2.0
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def aabb_surface_points(
    center: np.ndarray,
    size: np.ndarray,
    grid: float = 0.10,
    max_pts: int = 1500,
) -> np.ndarray:
    """AABB 表面网格采样（与 `SceneRenderer.instance_aabb_surface` 同生成法）。"""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    s = np.asarray(size, dtype=np.float64).reshape(3)
    lo, hi = c - s / 2.0, c + s / 2.0
    n = np.maximum(2, np.ceil(s / float(grid)).astype(int))
    xs = np.linspace(lo[0], hi[0], n[0])
    ys = np.linspace(lo[1], hi[1], n[1])
    zs = np.linspace(lo[2], hi[2], n[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    on_surface = (
        np.isclose(pts[:, 0], lo[0]) | np.isclose(pts[:, 0], hi[0])
        | np.isclose(pts[:, 1], lo[1]) | np.isclose(pts[:, 1], hi[1])
        | np.isclose(pts[:, 2], lo[2]) | np.isclose(pts[:, 2], hi[2])
    )
    pts = pts[on_surface]
    if len(pts) > max_pts:
        idx = np.linspace(0, len(pts) - 1, max_pts).astype(int)
        pts = pts[idx]
    return pts


def box_visibility(
    box_q: np.ndarray,
    q_pose: Dict[str, float],
    frame_poses: Sequence[Dict[str, float]],
    geo: Optional[FrameGeometry] = None,
    surface_grid: float = 0.10,
    max_pts: int = 1500,
    min_frac: float = 0.05,
) -> Dict[str, float]:
    """GT AABB（查询 canonical 系 (x, y, z, l, w, h)）在 K 帧视锥中的可见性。

    返回 {"any_visible": bool, "best_frac": float, "frames_visible": int}。
    「可见」= 该帧内 AABB 表面采样点落入视锥的比例 >= `min_frac`（任一帧可见即 any）。
    """
    g = geo or FrameGeometry()
    box = np.asarray(box_q, dtype=np.float64).reshape(6)
    surf_q = aabb_surface_points(box[:3], box[3:6], grid=surface_grid, max_pts=max_pts)
    if len(surf_q) == 0 or len(frame_poses) == 0:
        return {"any_visible": False, "best_frac": 0.0, "frames_visible": 0}
    best = 0.0
    n_vis = 0
    for f_pose in frame_poses:
        frac = float(frustum_mask_frame(pose_canonical_to_frame(surf_q, q_pose, f_pose), g).mean())
        best = max(best, frac)
        if frac >= min_frac:
            n_vis += 1
    return {"any_visible": n_vis > 0, "best_frac": best, "frames_visible": n_vis}


def visible_entities_query_canonical(
    instances: Sequence[Dict],
    poses: Sequence[Dict[str, float]],
    q_pose: Dict[str, float],
    geo: Optional[FrameGeometry] = None,
    near_radius: float = 8.0,
    z_band: Optional[tuple] = (-0.5, 3.0),
    surface_grid: float = 0.10,
    max_pts: int = 600,
    min_frac: float = 0.05,
) -> np.ndarray:
    """实例表 -> 查询 canonical 系中「K 帧任一帧可见」的实体 AABB 集合。

    输入 `instances`：逐场景实例表（episodic 绝对坐标），条目
        {"id": int, "cat": str|None, "center": [x, y, z_abs], "size": [sx, sy, sz]}
    其中 center 的 x/y 为 episodic 平面坐标、z 为绝对高度（habitat y 经 `to_episodic`）。

    过滤：`near_radius`（查询点平面距离预筛，视锥深度上限 5 m 留裕量）；
    `z_band`（实体中心相对查询 floor 的高度带；剔除楼上/楼下实例——
    几何视锥判定不做遮挡剔除，多层场景需此带过滤）。

    返回 (M, 7)：[x, y, z, l, w, h, instance_id]（查询 canonical 系；z 相对查询 floor）。
    """
    g = geo or FrameGeometry()
    out: List[List[float]] = []
    tx, ty = float(q_pose["x"]), float(q_pose["y"])
    for inst in instances:
        c_epi = np.asarray(inst["center"], dtype=np.float64).reshape(3)
        size = np.asarray(inst["size"], dtype=np.float64).reshape(3)
        if float(np.hypot(c_epi[0] - tx, c_epi[1] - ty)) > float(near_radius):
            continue  # 远距预过滤（视锥深度上限 5 m，留裕量）
        z_q = c_epi[2] - float(q_pose["floor"])
        if z_band is not None and (z_q < float(z_band[0]) or z_q > float(z_band[1])):
            continue
        # episodic 绝对 -> 查询 canonical：xy 平移 + R(-yawq)；z 减查询 floor
        d_yaw = math.radians(float(q_pose["yaw_deg"]))
        c, s = math.cos(-d_yaw), math.sin(-d_yaw)
        dx, dy = c_epi[0] - tx, c_epi[1] - ty
        cq = np.array([c * dx - s * dy, s * dx + c * dy, c_epi[2] - float(q_pose["floor"])])
        # size 轴对齐不变（canonical 只做绕 z 旋转，轴对齐 AABB 在旋转下重包裹为 size 变化——
        # 这里保守取各轴最大值以不丢失可见性判定所需覆盖）
        if abs(math.sin(d_yaw)) > 1e-6:
            l_new = abs(math.cos(d_yaw)) * size[0] + abs(math.sin(d_yaw)) * size[1]
            w_new = abs(math.sin(d_yaw)) * size[0] + abs(math.cos(d_yaw)) * size[1]
        else:
            l_new, w_new = size[0], size[1]
        sq = np.array([cq[0], cq[1], cq[2], l_new, w_new, size[2]])
        info = box_visibility(sq, q_pose, poses, g, surface_grid=surface_grid, max_pts=max_pts,
                              min_frac=min_frac)
        if info["any_visible"]:
            out.append([*sq.tolist(), float(inst.get("id", -1))])
    if not out:
        return np.zeros((0, 7), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)
