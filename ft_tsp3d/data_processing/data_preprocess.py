#!/usr/bin/env python3
"""数据供给层 · 离线模拟局部视角采样（Synthetic Local-View Slicing）

1) 仿真与渲染（与 VLVM 在线传感器同源）
   - 独立 habitat_sim 实例；传感器 = COLOR + DEPTH + SEMANTIC（RGB 必需，否则语义图恒全零）。
   - 640x480、hfov 79°、传感器高度 0.88 m；仿真 pitch = 0（在线 tf 只含 yaw）。
   - 深度 = habitat 度量平面 z 深度；先按在线公式 `norm = clip((d - 0.5) / 4.5, 0, 1)`，再走在线同款 `filter_depth`，然后反投影：
     `xc = (u-W/2)d/fx`、`yc = (v-H/2)d/fy`；
     `pts_base = [d, -xc, -yc]`（前/左/上）；`fx = fy = (W/2)/tan(39.5°) = 388.2`。
   - 几何约定已由标定探针验证（小/中件实例 AABB 包含率 100%@+0.10 m、跨视角点云 NN 中位 0.013 m）：
     episodic 系 `p_e = (x, -z, y)`；相机局部轴向 (x 右, y 上, z 后)。
   - 机器人位姿 = 导航网格随机可达点 + 均匀 yaw；云坐标与在线一致：
     `robot_xyz = [x_e, y_e, 0.88]`（z 恒 0.88，即“离地高度 + 0.88”约定）。

2) 样本形态（复用在线 world 融合与发送侧）
   - `TSP3DInputPreprocessor(fusion_style="world", ...)`：
     体素 0.02、半径 6.0、去重门 0.15 m / 15°、8 帧窗、near_refresh 3.0=True；
     发送侧 distance_sample(1.5/3.0 → 0.01/0.02/0.05) + cap 200k(random)。
   - 每个“视图样本”= K~U(1,8) 帧短轨迹（连续帧须跨过 0.15 m / 15° 去重门），查询位姿 = 最后一帧。
   - 点云以 float16 落盘（与在线 wire dtype 一致）。

3) 文本与标签
   - 目标词表 = 框架 6 类 `["chair","bed","potted plant","toilet","tv","couch"]`；
     原始语义名经 `grounding_dino/vocab.py` 的 alias/单复数/小写归一映射到规范类（多词变体不映射，按用户裁定）。
   - 正样本：查询类在视图内有可见实例（像素 >= `--min-vis-px`），目标 = 该类可见像素最多的实例；
     GT = **完整实例 AABB**（canonical 系轴对齐重包裹）；云内目标点数须 >= `--min-target-pts`。
   - 负样本：从 6 类词表随机取一个“视图内所有实例像素均 < `--min-vis-px`(40)（弱可见 1~39 px 视为不存在）”的规范类，GT 为空。
   - 正负配比全局控制 = 1:`--neg-per-pos`（默认 1:3）。
   - 截断比例（目标 AABB 表面采样点中落在 K 帧任一视锥且深度 ∈[0.5,5.0] 之外的比例）：
     `> --trunc-drop`(0.85) 直接剔除（只写 `dropped.jsonl`）；
     `(--trunc-low, --trunc-drop]` 记 weight=0.5；其余 weight=1.0。负样本不受影响。

4) 场景池与划分
   - 场景目录 = `<hm3d_root>/<split_dir>/<scene_id>/`（默认 train，137 个 annotated train 场景）。
   - 9:1 场景级留出（`--val-ratio 0.1`，固定 seed，写入 `splits.json` 冻结）；nav 20 场景不生成。

5) 产物（默认 `vlvm/data/ft_datasets/local_view`）
   - `splits.json` / `stats.json` / `dropped.jsonl`
   - `{train,val}/<scene_id>.npz`：`points`(float16 (M,6)) + `offsets`(int64 (V+1,)) + `records_json`(str)
     `records_json` 每条记录 = 一个训练样本：`{"view", "k", "poses", "query", "label", "weight", "target": {...}}`；
     同一视图的正/负样本共享一份点云（`points[offsets[v]:offsets[v+1]]`）。

本文件只负责主 npz（`splits.json` / `stats.json` / `dropped.jsonl` / `{train,val}/*.npz`）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import habitat_sim  # noqa: E402
import quaternion as qt  # noqa: E402
from depth_camera_filtering import filter_depth  # noqa: E402
from habitat_sim.utils.common import quat_from_angle_axis  # noqa: E402

from vlfm.tsp3d_models.grounding_dino.vocab import (  # noqa: E402
    DEFAULT_VOCAB,
    build_alias_map,
    normalize_phrase,
    singularize,
)
from vlfm.tsp3d_models.fussion_utils.pipeline import TSP3DInputPreprocessor  # noqa: E402

# episodic 变换：p_e = (x, -z, y)（habitat y-up → VLVM z-up）
_M_EP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def to_episodic(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, dtype=np.float64) @ _M_EP.T


def xyz_yaw_to_tf_matrix(xyz: Sequence[float], yaw: float) -> np.ndarray:
    """VLVM `xyz_yaw_to_tf_matrix`。"""
    x, y, z = xyz
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0, x],
            [np.sin(yaw), np.cos(yaw), 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@dataclass
class Config:
    # ---- paths ----
    hm3d_root: str = "/root/autodl-tmp/vlfm/data/versioned_data/hm3d-0.2/hm3d"
    split_dir: str = "train"
    out_dir: str = str(_REPO / "data" / "ft_datasets" / "local_view")
    scenes: Optional[str] = None  # 逗号分隔的 scene_id 覆盖
    limit: int = 0
    skip_existing: bool = False
    gpu_id: int = 0
    min_asset_bytes: int = 1024  # 小于该值的资产文件视为下载损坏（0 字节 / 截断）
    # ---- split ----
    seed: int = 0
    val_ratio: float = 0.1  # 9:1
    # ---- camera (online) ----
    image_w: int = 640
    image_h: int = 480
    hfov_deg: float = 79.0
    camera_height: float = 0.88
    min_depth: float = 0.5
    max_depth: float = 5.0
    # ---- fusion / send side (online YAML) ----
    map_voxel_size: float = 0.02
    map_radius: float = 6.0
    view_disp: float = 0.15
    view_yaw_deg: float = 15.0
    map_max_voxels: int = 400000
    map_max_frames: int = 8
    near_refresh_radius: float = 3.0
    near_refresh_value: bool = True
    max_points: int = 200000
    cap_style: str = "random"
    distance_sample: bool = True
    near_dist: float = 1.5
    mid_dist: float = 3.0
    near_voxel: float = 0.01
    mid_voxel: float = 0.02
    far_voxel: float = 0.05
    # ---- view / trajectory sampling ----
    views_per_scene: int = 200
    k_max: int = 8
    traj_step: float = 1.5
    gate_disp: float = 0.15
    gate_yaw_deg: float = 15.0
    gate_tries: int = 12
    # ---- labels ----
    min_vis_px: int = 40
    min_target_pts: int = 40
    box_margin: float = 0.05
    neg_per_pos: int = 3
    trunc_drop: float = 0.85
    trunc_low: float = 0.70
    weight_mid: float = 0.5
    trunc_grid: float = 0.10
    trunc_max_pts: int = 1500
    # ---- misc ----
    verify_calib: bool = False

    @property
    def dataset_config(self) -> str:
        return str(Path(self.hm3d_root) / "hm3d_annotated_basis.scene_dataset_config.json")


def discover_scenes(cfg: Config) -> List[str]:
    root = Path(cfg.hm3d_root) / cfg.split_dir
    if not root.is_dir():
        raise SystemExit(f"scene root not found: {root}")
    scenes = []
    broken: List[str] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        short = d.name.split("-", 1)[1] if "-" in d.name else None
        if short is None:
            continue
        need = [f"{short}.basis.glb", f"{short}.semantic.glb", f"{short}.semantic.txt"]
        sizes = [(d / n).stat().st_size if (d / n).is_file() else -1 for n in need]
        if all(s >= cfg.min_asset_bytes for s in sizes):
            scenes.append(d.name)
        else:
            broken.append(f"{d.name} " + ",".join(f"{n}={s}" for n, s in zip(need, sizes)))
    if broken:
        print(f"[warn] {len(broken)} scene dir(s) incomplete or truncated -> skipped:")
        for b in broken:
            print("   ", b)
    if cfg.scenes:
        want = {s.strip() for s in cfg.scenes.split(",") if s.strip()}
        scenes = [s for s in scenes if s in want]
    if cfg.limit > 0:
        scenes = scenes[: cfg.limit]
    if not scenes:
        raise SystemExit("no scenes with complete assets (basis.glb + semantic.glb + semantic.txt)")
    return scenes


def make_sim(scene_dir: Path, short: str, cfg: Config) -> habitat_sim.Simulator:
    def mk(uuid, stype):
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = stype
        s.resolution = [cfg.image_h, cfg.image_w]
        s.hfov = cfg.hfov_deg
        s.position = [0.0, cfg.camera_height, 0.0]
        return s

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_dir / f"{short}.basis.glb")
    if Path(cfg.dataset_config).is_file():
        sim_cfg.scene_dataset_config_file = cfg.dataset_config
    sim_cfg.gpu_device_id = cfg.gpu_id
    sim_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        mk("rgb", habitat_sim.SensorType.COLOR),
        mk("depth", habitat_sim.SensorType.DEPTH),
        mk("semantic", habitat_sim.SensorType.SEMANTIC),
    ]
    agent_cfg.height = cfg.camera_height
    agent_cfg.radius = 0.18
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


class SceneRenderer:
    """单场景渲染器：位姿采样、帧渲染、实例表（类别 + AABB）。"""

    def __init__(self, scene_dir: Path, cfg: Config) -> None:
        self.cfg = cfg
        self.short = scene_dir.name.split("-", 1)[1]
        self.sim = make_sim(scene_dir, self.short, cfg)
        self.agent = self.sim.initialize_agent(0)
        self.fx = self.fy = (cfg.image_w / 2.0) / math.tan(math.radians(cfg.hfov_deg) / 2.0)
        self.u = (np.meshgrid(np.arange(cfg.image_w) + 0.5, np.arange(cfg.image_h) + 0.5)[0]).reshape(-1)
        self.v = (np.meshgrid(np.arange(cfg.image_w) + 0.5, np.arange(cfg.image_h) + 0.5)[1]).reshape(-1)
        self.u = self.u - cfg.image_w / 2.0
        self.v = self.v - cfg.image_h / 2.0
        # 实例表：semantic_id -> (raw_category, aabb_center_e, aabb_sizes_e)
        self.inst_raw: Dict[int, str] = {}
        self.inst_center: Dict[int, np.ndarray] = {}
        self.inst_sizes: Dict[int, np.ndarray] = {}
        for o in self.sim.semantic_scene.objects:
            sid = int(o.semantic_id)
            self.inst_raw[sid] = o.category.name() if o.category else "unknown"
            self.inst_center[sid] = to_episodic(np.asarray(o.aabb.center, dtype=np.float64))
            s = np.asarray(o.aabb.sizes, dtype=np.float64)
            self.inst_sizes[sid] = np.array([s[0], s[2], s[1]], dtype=np.float64)  # (sx, sz, sy)
        # 规范类映射
        self.alias = build_alias_map(DEFAULT_VOCAB)
        self.inst_class = {sid: self._canonical(raw) for sid, raw in self.inst_raw.items()}
        self.pf = self.sim.pathfinder

    def _canonical(self, raw: str) -> Optional[str]:
        s = normalize_phrase(raw)
        if s in self.alias:
            return self.alias[s]
        return self.alias.get(singularize(s))

    def close(self) -> None:
        self.sim.close()

    def sample_position(self) -> np.ndarray:
        """返回 habitat 位置 (x, floor_y, z)。"""
        return np.asarray(self.pf.get_random_navigable_point(), dtype=np.float64)

    def render(self, pos_h: np.ndarray, yaw_e: float) -> Tuple[dict, np.ndarray, np.ndarray]:
        """在给定 habitat 位置 + episodic yaw 渲染一帧。

        返回 (obs, sensor_pos_h, R_sensor)；相机轴约定 (x 右, y 上, z 后)。
        """
        phi_h = yaw_e - math.pi / 2.0  # 标定结论：yaw_e = phi_h + pi/2
        st = habitat_sim.AgentState()
        st.position = np.asarray(pos_h, dtype=np.float64)
        st.rotation = quat_from_angle_axis(phi_h, np.array([0.0, 1.0, 0.0]))
        self.agent.set_state(st)
        obs = self.sim.get_sensor_observations()
        ss = self.agent.get_state().sensor_states["depth"]
        return obs, np.asarray(ss.position, dtype=np.float64), np.asarray(qt.as_rotation_matrix(ss.rotation), dtype=np.float64)

    def yaw_of(self, R: np.ndarray) -> float:
        f_e = to_episodic(R @ np.array([0.0, 0.0, -1.0]))
        return math.atan2(f_e[1], f_e[0])

    def frame_cloud(self, obs: dict, sensor_pos_h: np.ndarray, yaw_e: float) -> Tuple[np.ndarray, np.ndarray]:
        """按在线政策公式生成 world 帧点云 (N,6) 与逐点语义 id。"""
        cfg = self.cfg
        rgb = np.asarray(obs["rgb"])[:, :, :3].reshape(-1, 3)
        sem = np.asarray(obs["semantic"]).reshape(-1).astype(np.int64)
        depth_m2 = np.asarray(obs["depth"], dtype=np.float32)  # (H, W) 度量平面 z 深度

        norm2 = np.zeros_like(depth_m2)
        hit = depth_m2 > 0.0
        norm2[hit] = (depth_m2[hit] - cfg.min_depth) / (cfg.max_depth - cfg.min_depth)
        np.clip(norm2, 0.0, 1.0, out=norm2)
        norm2 = filter_depth(norm2, blur_type=None)  # 2D 输入
        norm = norm2.reshape(-1)
        valid = (norm > 0.0) & (norm < 1.0)
        if not np.any(valid):
            return np.empty((0, 6), dtype=np.float32), np.empty(0, dtype=np.int64)

        d = norm[valid] * (cfg.max_depth - cfg.min_depth) + cfg.min_depth
        xc = self.u[valid] * d / self.fx
        yc = self.v[valid] * d / self.fy
        pts_base = np.stack([d, -xc, -yc], axis=1)  # [forward, left, up]
        cam_pos_e = np.array([to_episodic(sensor_pos_h)[0], to_episodic(sensor_pos_h)[1], cfg.camera_height])
        tf = xyz_yaw_to_tf_matrix(cam_pos_e, yaw_e)
        pts_w = pts_base @ tf[:3, :3].T + tf[:3, 3]
        colors = rgb[valid].astype(np.float32) / 255.0
        return np.hstack([pts_w, colors]).astype(np.float32), sem[valid]

    def instance_aabb_surface(self, sid: int, grid: float, max_pts: int) -> np.ndarray:
        c, s = self.inst_center[sid], self.inst_sizes[sid]
        lo, hi = c - s / 2.0, c + s / 2.0
        n = np.maximum(2, np.ceil(s / grid).astype(int))
        xs = np.linspace(lo[0], hi[0], n[0])
        ys = np.linspace(lo[1], hi[1], n[1])
        zs = np.linspace(lo[2], hi[2], n[2])
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        on_surface = (
            np.isclose(pts[:, 0], lo[0]) | np.isclose(pts[:, 0], hi[0]) |
            np.isclose(pts[:, 1], lo[1]) | np.isclose(pts[:, 1], hi[1]) |
            np.isclose(pts[:, 2], lo[2]) | np.isclose(pts[:, 2], hi[2])
        )
        pts = pts[on_surface]
        if len(pts) > max_pts:
            idx = np.linspace(0, len(pts) - 1, max_pts).astype(int)
            pts = pts[idx]
        return pts


def in_frustum(pts: np.ndarray, cam_true_e: np.ndarray, yaw: float, cfg: Config) -> np.ndarray:
    """点是否落在相机视锥内且深度 ∈ [min_depth, max_depth]（平面 z 深度）。"""
    rel = pts - cam_true_e[None, :]
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    z = rel @ fwd
    x = -(rel @ left)  # 右为正
    y = -rel[:, 2]  # 下为正
    tan_h = math.tan(math.radians(cfg.hfov_deg) / 2.0)
    tan_v = (cfg.image_h / 2.0) / ((cfg.image_w / 2.0) / tan_h)
    ok = (z >= cfg.min_depth) & (z <= cfg.max_depth)
    ok &= np.abs(x) <= tan_h * np.maximum(z, 1e-6)
    ok &= np.abs(y) <= tan_v * np.maximum(z, 1e-6)
    return ok


def box_corners(center: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    lo, hi = center - sizes / 2.0, center + sizes / 2.0
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

def canonicalize_box(center_e: np.ndarray, sizes_e: np.ndarray, floor_y: float, robot_e: np.ndarray, yaw: float) -> Dict[str, List[float]]:
    """完整实例 AABB → canonical 系轴对齐重包裹。"""
    corners = box_corners(center_e, sizes_e).copy()
    corners[:, 2] -= floor_y
    corners[:, :2] -= robot_e[:2]
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[cy, -sy], [sy, cy]])
    corners[:, :2] = corners[:, :2] @ rot.T
    lo, hi = corners.min(0), corners.max(0)
    return {
        "min": lo.round(4).tolist(),
        "max": hi.round(4).tolist(),
        "center": ((lo + hi) / 2.0).round(4).tolist(),
        "size": (hi - lo).round(4).tolist(),
    }


def fast_voxel_first(pcd: np.ndarray, voxel: float) -> np.ndarray:
    """
    与 `np.unique(vox, axis=0, return_index=True)` 同义（每体素保留首次出现的点），
    但改用 1D int64 打包键，307k 点 0.25 s → ~0.03 s；
    结果与 `WorldLocalMap.update`内部体素化完全一致。
    """
    if len(pcd) == 0:
        return pcd
    vox = np.floor(pcd[:, :3] / float(voxel)).astype(np.int64)
    off = 1 << 20
    key = ((vox[:, 0] + off) << 42) | ((vox[:, 1] + off) << 21) | (vox[:, 2] + off)
    _, first = np.unique(key, return_index=True)
    return pcd[np.sort(first)]


def build_preprocessor(cfg: Config) -> TSP3DInputPreprocessor:
    return TSP3DInputPreprocessor(
        fusion_style="world",
        map_voxel_size=cfg.map_voxel_size,
        map_radius=cfg.map_radius,
        min_view_disp=cfg.view_disp,
        min_view_yaw=math.radians(cfg.view_yaw_deg),
        max_map_voxels=cfg.map_max_voxels,
        map_max_frames=cfg.map_max_frames,
        map_near_refresh_radius=cfg.near_refresh_radius,
        map_near_refresh_value=cfg.near_refresh_value,
        max_points=cfg.max_points,
        cap_style=cfg.cap_style,
        use_distance_sampling=cfg.distance_sample,
        near_dist=cfg.near_dist,
        mid_dist=cfg.mid_dist,
        near_voxel=cfg.near_voxel,
        mid_voxel=cfg.mid_voxel,
        far_voxel=cfg.far_voxel,
    )


@dataclass
class Stats:
    views: int = 0
    pos: int = 0
    neg: int = 0
    dropped_trunc: int = 0
    dropped_no_pts: int = 0
    empty_views: int = 0
    skipped_views: int = 0
    class_pos: Dict[str, int] = field(default_factory=dict)
    class_neg: Dict[str, int] = field(default_factory=dict)
    trunc_hist: Dict[str, int] = field(default_factory=lambda: {"<=0.70": 0, "0.70-0.85": 0, ">0.85": 0})
    pts: List[int] = field(default_factory=list)
    k_used: List[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {
            "views": self.views, "pos": self.pos, "neg": self.neg,
            "dropped_trunc": self.dropped_trunc, "dropped_no_pts": self.dropped_no_pts,
            "empty_views": self.empty_views, "skipped_views": self.skipped_views, "class_pos": self.class_pos, "class_neg": self.class_neg,
            "trunc_hist": self.trunc_hist,
        }
        if self.pts:
            a = np.asarray(self.pts, dtype=np.float64)
            d["cloud_points"] = {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
                                 "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}
        if self.k_used:
            d["k_frames"] = {"mean": float(np.mean(self.k_used)), "min": int(np.min(self.k_used)),
                             "max": int(np.max(self.k_used))}
        return d


def process_scene(scene_id: str, cfg: Config, rr: SceneRenderer, stats: Stats,
                  dropped_fh) -> Optional[dict]:
    """生成单场景主 npz 内容（侧车 / 实例表面已随归档移出，此处只产主数据）。"""
    pre = build_preprocessor(cfg)
    rng = np.random.default_rng(cfg.seed * 1000003 + sum(map(ord, scene_id)))  # 逐场景确定性 seed
    points_chunks: List[np.ndarray] = []
    offsets: List[int] = [0]
    records: List[dict] = []
    n_pts_total = 0
    saved_views = 0
    attempts = 0

    for view_idx in range(cfg.views_per_scene * 3):
        if saved_views >= cfg.views_per_scene:
            break
        attempts += 1
        pre.reset()
        frames: List[dict] = []
        last_pos_e: Optional[np.ndarray] = None
        last_pos_h: Optional[np.ndarray] = None
        last_yaw: Optional[float] = None
        k_target = int(rng.integers(1, cfg.k_max + 1))
        for k in range(k_target):
            placed = False
            for _ in range(cfg.gate_tries):
                if last_pos_e is None:
                    pos_h = rr.sample_position()
                    yaw_e = float(rng.uniform(-math.pi, math.pi))
                elif rng.random() < 0.5:
                    # 原地观测：同位置、yaw 变化 >= 去重门（必然过门）
                    pos_h = last_pos_h
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    yaw_e = last_yaw + sign * float(rng.uniform(math.radians(cfg.gate_yaw_deg) + 0.05, 1.2))
                else:
                    base = rr.sample_position()
                    if np.linalg.norm(base[[0, 2]] - last_pos_e[[0, 2]]) > cfg.traj_step:
                        continue
                    pos_h = base
                    yaw_e = float(rng.uniform(-math.pi, math.pi))
                    if (np.linalg.norm(pos_h[[0, 2]] - last_pos_e[[0, 2]]) < cfg.gate_disp
                            and abs((yaw_e - last_yaw + math.pi) % (2 * math.pi) - math.pi) < math.radians(cfg.gate_yaw_deg)):
                        continue  # 未跨去重门
                obs, sensor_pos_h, R = rr.render(pos_h, yaw_e)
                yaw_meas = rr.yaw_of(R)
                pos_e = np.array([pos_h[0], -pos_h[2], 0.0])
                floor_y = float(pos_h[1])
                cloud_w, sem_ids = rr.frame_cloud(obs, sensor_pos_h, yaw_meas)
                if len(cloud_w) == 0:
                    continue
                cam_pos_e = np.array([pos_e[0], pos_e[1], cfg.camera_height])
                pre.update(fast_voxel_first(cloud_w, cfg.map_voxel_size), cam_pos_e, yaw_meas)
                frame = {"pos_e": pos_e, "floor": floor_y, "yaw": yaw_meas, "sem": sem_ids}
                frames.append(frame)
                last_pos_h, last_pos_e, last_yaw = pos_h, pos_e, yaw_meas
                placed = True
                break
            if not placed:
                break

        if not frames:
            stats.empty_views += 1
            continue

        query_pose = frames[-1]
        cam_pos_e = np.array([query_pose["pos_e"][0], query_pose["pos_e"][1], cfg.camera_height])
        fused = pre.prepare(cam_pos_e, query_pose["yaw"], camera_height=cfg.camera_height)
        if len(fused) == 0:
            stats.empty_views += 1
            continue

        # 视图级类别像素统计（跨帧求和；负样本判据 = 该类像素恒 0）
        cls_px: Dict[str, int] = {}
        inst_px: Dict[int, int] = {}
        for f in frames:
            ids, cnts = np.unique(f["sem"], return_counts=True)
            for sid, c in zip(ids.tolist(), cnts.tolist()):
                if sid == 0:
                    continue
                inst_px[sid] = inst_px.get(sid, 0) + int(c)
                cls = rr.inst_class.get(sid)
                if cls is not None:
                    cls_px[cls] = cls_px.get(cls, 0) + int(c)

        visible_targets = {sid: c for sid, c in inst_px.items()
                           if rr.inst_class.get(sid) in DEFAULT_VOCAB and c >= cfg.min_vis_px}
        cls_visible = {rr.inst_class[sid] for sid in visible_targets}

        view_records: List[dict] = []
        cloud_f16 = fused[:, :6].astype(np.float16)

        def target_pts(box: dict) -> int:
            lo, hi = np.asarray(box["min"]), np.asarray(box["max"])
            lo = lo - cfg.box_margin
            hi = hi + cfg.box_margin
            return int(np.all((fused[:, :3] >= lo) & (fused[:, :3] <= hi), axis=1).sum())

        # ---- 正样本：该类可见像素最多、云内目标点 >= 门槛、截断规则通过 ----
        if cls_visible:
            cls = sorted(cls_visible)[int(rng.integers(0, len(cls_visible)))]
            sid = max((s for s in visible_targets if rr.inst_class[s] == cls), key=lambda s: inst_px[s])
            box = canonicalize_box(rr.inst_center[sid], rr.inst_sizes[sid], query_pose["floor"], query_pose["pos_e"],
                                   query_pose["yaw"])
            n_tp = target_pts(box)
            surf = rr.instance_aabb_surface(sid, cfg.trunc_grid, cfg.trunc_max_pts)
            seen = np.zeros(len(surf), dtype=bool)
            for f in frames:
                cam_true = np.array([f["pos_e"][0], f["pos_e"][1], f["floor"] + cfg.camera_height])
                seen |= in_frustum(surf, cam_true, f["yaw"], cfg)
            trunc = 1.0 - float(seen.mean()) if len(surf) else 1.0
            if n_tp < cfg.min_target_pts:
                stats.dropped_no_pts += 1
                dropped_fh.write(json.dumps({"scene": scene_id, "view": view_idx, "query": cls, "reason": "few_target_pts",
                                             "target_pts": n_tp, "trunc": round(trunc, 3)}) + "\n")
            elif trunc > cfg.trunc_drop:
                stats.dropped_trunc += 1
                dropped_fh.write(json.dumps({"scene": scene_id, "view": view_idx, "query": cls, "reason": "truncated",
                                             "trunc": round(trunc, 3), "target_pts": n_tp}) + "\n")
            else:
                weight = cfg.weight_mid if trunc > cfg.trunc_low else 1.0
                stats.trunc_hist["<=0.70" if trunc <= cfg.trunc_low else "0.70-0.85"] += 1
                view_records.append({
                    "view": view_idx, "k": len(frames), "query": cls, "label": 1, "weight": weight,
                    "target": {"id": int(sid), "raw_cat": rr.inst_raw[sid], "cat": cls, "box": box,
                               "pix": int(inst_px[sid]), "target_pts": n_tp, "trunc": round(trunc, 4)},
                })
                stats.pos += 1
                stats.class_pos[cls] = stats.class_pos.get(cls, 0) + 1

        # ---- 负样本：从 6 类词表随机取“像素恒 0”的类；全局比例控制 ----
        absent = [c for c in DEFAULT_VOCAB if c not in cls_visible]
        rng.shuffle(absent)
        for c in absent[: cfg.neg_per_pos]:
            if stats.neg >= stats.pos * cfg.neg_per_pos and stats.pos > 0:
                break
            view_records.append({"view": view_idx, "k": len(frames), "query": c, "label": 0, "weight": 1.0,
                                 "target": None})
            stats.neg += 1
            stats.class_neg[c] = stats.class_neg.get(c, 0) + 1

        if not view_records:
            stats.skipped_views += 1
            continue
        saved_views += 1
        for r in view_records:
            r["view_pts"] = int(len(cloud_f16))
            r["poses"] = [{"x": round(f["pos_e"][0], 3), "y": round(f["pos_e"][1], 3), "floor": round(f["floor"], 3),
                           "yaw_deg": round(math.degrees(f["yaw"]), 1)} for f in frames]
            records.append(r)
        points_chunks.append(cloud_f16)
        n_pts_total += len(cloud_f16)
        offsets.append(n_pts_total)
        stats.views += 1
        stats.pts.append(int(len(cloud_f16)))
        stats.k_used.append(len(frames))

    if not points_chunks:
        return None
    return {
        "points": np.concatenate(points_chunks, axis=0).astype(np.float16),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "records_json": json.dumps(records),
        "attempts": attempts, "saved_views": saved_views,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hm3d-root", default=Config.hm3d_root)
    ap.add_argument("--split-dir", default=Config.split_dir)
    ap.add_argument("--out-dir", default=Config.out_dir)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--views", type=int, default=Config.views_per_scene)
    ap.add_argument("--k-max", type=int, default=Config.k_max)
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--val-ratio", type=float, default=Config.val_ratio)
    ap.add_argument("--neg-per-pos", type=int, default=Config.neg_per_pos)
    ap.add_argument("--min-vis-px", type=int, default=Config.min_vis_px)
    ap.add_argument("--min-target-pts", type=int, default=Config.min_target_pts)
    ap.add_argument("--trunc-drop", type=float, default=Config.trunc_drop)
    ap.add_argument("--trunc-low", type=float, default=Config.trunc_low)
    ap.add_argument("--max-points", type=int, default=Config.max_points)
    ap.add_argument("--gpu-id", type=int, default=Config.gpu_id)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--shard", type=int, default=0, help="分片编号（多进程并行时用，0-based）")
    ap.add_argument("--shards", type=int, default=1, help="分片总数（>1 时不重复写 splits.json）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = Config(
        hm3d_root=args.hm3d_root, split_dir=args.split_dir, out_dir=args.out_dir, scenes=args.scenes,
        limit=args.limit, seed=args.seed, val_ratio=args.val_ratio, views_per_scene=args.views,
        k_max=args.k_max, neg_per_pos=args.neg_per_pos, min_vis_px=args.min_vis_px,
        min_target_pts=args.min_target_pts, trunc_drop=args.trunc_drop, trunc_low=args.trunc_low,
        max_points=args.max_points, gpu_id=args.gpu_id, skip_existing=args.skip_existing,
    )

    scenes = discover_scenes(cfg)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(scenes))
    n_val = max(1, int(round(len(scenes) * cfg.val_ratio))) if len(scenes) > 1 else 0
    val_scenes = sorted(scenes[i] for i in order[:n_val])
    train_scenes = sorted(scenes[i] for i in order[n_val:])
    plan = {"all": scenes, "train": train_scenes, "val": val_scenes}
    if args.shards > 1:
        train_scenes = train_scenes[args.shard::args.shards]
        val_scenes = val_scenes[args.shard::args.shards]
    if args.dry_run:
        print(json.dumps({k: len(v) for k, v in plan.items()}, ensure_ascii=False))
        print("train:", plan["train"][:5], "...", "val:", plan["val"][:5])
        return

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.shards == 1 or args.shard == 0:
        (out / "splits.json").write_text(json.dumps({
            "seed": cfg.seed, "val_ratio": cfg.val_ratio, "pool": plan["all"],
            "train": plan["train"], "val": plan["val"],
            "note": "nav 20 scenes (val split) are excluded from generation by design",
        }, indent=1, ensure_ascii=False))

    stats = Stats()
    dropped_path = out / "dropped.jsonl"
    failures_path = out / "failures.jsonl"
    t_start = time.time()
    fail_streak = 0
    with open(dropped_path, "a") as dropped_fh, open(failures_path, "a") as fail_fh:
        for split, split_scenes in (("train", train_scenes), ("val", val_scenes)):
            (out / split).mkdir(parents=True, exist_ok=True)
            for i, scene_id in enumerate(split_scenes, 1):
                dst = out / split / f"{scene_id}.npz"
                if cfg.skip_existing and dst.is_file():
                    print(f"[{split}] {i}/{len(split_scenes)} {scene_id}: skip (exists)")
                    continue
                scene_dir = Path(cfg.hm3d_root) / cfg.split_dir / scene_id
                t0 = time.time()
                rr = None
                res = None
                failed = False
                try:
                    rr = SceneRenderer(scene_dir, cfg)
                    res = process_scene(scene_id, cfg, rr, stats, dropped_fh)
                except Exception as exc:  # 单场景失败（资产损坏 / GL 显存不足）不拖垮整个分片
                    failed = True
                    msg = f"{type(exc).__name__}: {exc}"
                    fail_fh.write(json.dumps({
                        "split": split, "scene_id": scene_id, "error": msg,
                        "elapsed_s": round(time.time() - t0, 1),
                    }, ensure_ascii=False) + "\n")
                    fail_fh.flush()
                    print(f"[{split}] {i}/{len(split_scenes)} {scene_id}: FAILED "
                          f"{msg.splitlines()[0][:200]}  -> see {failures_path.name}")
                finally:
                    if rr is not None:
                        rr.close()
                if failed:
                    fail_streak += 1
                    if fail_streak >= 3:
                        raise SystemExit(f"[abort] {fail_streak} consecutive scene failures — "
                                         f"check GPU/driver/data; see {failures_path}")
                    continue
                fail_streak = 0
                if res is None:
                    print(f"[{split}] {i}/{len(split_scenes)} {scene_id}: NO SAMPLES ({time.time() - t0:.1f}s)")
                    continue
                np.savez_compressed(dst, **res)
                n_rec = len(json.loads(res["records_json"]))
                print(f"[{split}] {i}/{len(split_scenes)} {scene_id}: views={len(res['offsets']) - 1} "
                      f"records={n_rec} pts={len(res['points'])} ({time.time() - t0:.1f}s, "
                      f"elapsed {time.time() - t_start:.0f}s)")

    (out / "stats.json").write_text(json.dumps(stats.as_dict(), indent=1, ensure_ascii=False))
    print("stats:", json.dumps(stats.as_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
