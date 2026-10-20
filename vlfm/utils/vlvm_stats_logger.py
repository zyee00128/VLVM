from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class VlvmStatsLogger:
    """通用前置量测器。

    `enabled=False` 时计数 / 打印 / 快照全部空转（`last_meta` 仍照常维护，见模块 docstring）。
    实例持有 policy 引用，只读其状态（`_num_steps` / `_lock_step` / 记忆表 / 观测缓存）。
    """

    def __init__(self, policy: Any, enabled: bool = True):
        self.p = policy
        self.enabled = bool(enabled)
        self.reset()

    # ------------------------------------------------------------------ 每集状态
    def reset(self) -> None:
        """清零全部每集状态（policy `_reset` 调用）。"""
        self.nav: Dict[str, int] = {}          # 停点 / 接近确认
        self.pick: Dict[str, int] = {}         # 选点票型分布
        self._pick_key: Optional[tuple] = None
        self._approach_key: Optional[tuple] = None
        self.frame_admitted: List[Tuple[str, np.ndarray]] = []   # 本帧 admitted 的 TSP3D 候选
        self.last_meta: Dict[str, Any] = {}                      # src / n_obs / gd_conf / xy

    # ------------------------------------------------------------ 帧级记账 / 打印
    def note_frame_admitted(self, pending: List[Any]) -> None:
        """记录本帧 admitted 的 TSP3D 候选（cls, XY），供停点 / 接近段共位判定。"""
        if not self.enabled:
            return
        self.frame_admitted = [
            (str(cls), np.asarray(centroid_np, dtype=np.float64).reshape(-1)[:2])
            for _conf, centroid_np, _surf, classes in pending
            for cls in classes
        ]

    def note_pick(
        self,
        valid_centroids: List[np.ndarray],
        valid_obs: List[int],
        valid_src: List[str],
        chosen_idx: int,
        position: np.ndarray,
        valid_gconf: Optional[List[float]] = None,
    ) -> None:
        """记录本帧锁定的导航条目 + 打印 `[NAV] pick`。

        `last_meta` 恒记（`lock_exempt_identity` 的要决策输入）；打印按
        (src, n_obs, 0.5 m 距离档) 去重，并把候选集按距离排序打印前 4 个
        （`cands=[src:o<n_obs>:g<gd_conf>@<dist>*]`，`*` = 实际锁定的那个）。
        """
        src = str(valid_src[chosen_idx])
        obs = int(valid_obs[chosen_idx])
        robot_xy = np.asarray(position, dtype=np.float64).reshape(-1)[:2]

        def _gconf(j: int) -> float:
            if valid_gconf is None or j >= len(valid_gconf):
                return 0.0
            return float(valid_gconf[j] or 0.0)

        def _dist(j: int) -> float:
            cen = np.asarray(valid_centroids[j], dtype=np.float64).reshape(-1)
            return float(np.linalg.norm(cen[:2] - robot_xy))

        gconf = _gconf(chosen_idx)
        dist = _dist(chosen_idx)
        self.last_meta = {
            "src": src,
            "n_obs": obs,
            "gd_conf": gconf,
            "xy": np.asarray(valid_centroids[chosen_idx], dtype=np.float64).reshape(-1)[:2].copy(),
        }
        if not self.enabled:
            return
        # 票型分布（不受打印去重影响）
        self.pick["n"] = int(self.pick.get("n", 0)) + 1
        self.pick[src] = int(self.pick.get(src, 0)) + 1
        key = (src, obs, round(dist / 0.5))
        if key == self._pick_key:
            return
        self._pick_key = key
        cands = sorted(
            (
                (_dist(j), int(j), str(valid_src[j]), int(valid_obs[j]), _gconf(j))
                for j in range(len(valid_centroids))
            ),
            key=lambda t: (t[0], t[1]),
        )
        brief = ", ".join(
            f"{s}:o{o}:g{g:.2f}@{d:.1f}{'*' if j == chosen_idx else ''}"
            for d, j, s, o, g in cands[:4]
        )
        yaw = float(self.p._observations_cache.get("robot_heading", 0.0))
        print(
            f"[NAV] pick src={src} n_obs={obs} gd_conf={gconf:.2f} "
            f"dist={dist:.2f} "
            f"robot={np.round(robot_xy, 2)} yaw={yaw:.2f} "
            f"n_cand={len(valid_centroids)} cands=[{brief}]"
        )

    def note_approach(self, goal: np.ndarray, robot_xyz: np.ndarray) -> None:
        """接近段读数（0.5 m 距离档变化时打印一次）：距离 + 本帧 TSP3D 共位确认。"""
        if not self.enabled:
            return
        meta = self.last_meta or {}
        g = np.asarray(goal, dtype=np.float64).reshape(-1)[:2]
        robot = np.asarray(robot_xyz, dtype=np.float64).reshape(-1)[:2]
        dist = float(np.linalg.norm(g - robot))
        dedup = (str(meta.get("src", "?")), int(round(dist / 0.5)))
        if dedup == self._approach_key:
            return
        self._approach_key = dedup
        confirm = self._tsp3d_confirms(meta.get("xy"))
        key = "approach_confirm" if confirm else "approach_unconfirm"
        self.nav[key] = int(self.nav.get(key, 0)) + 1
        print(f"[NAV] approach step={self.p._num_steps} src={meta.get('src', '?')} "
              f"n_obs={meta.get('n_obs', 1)} dist={dist:.2f} tsp3d_confirm={confirm}")

    def note_stop(self, allowed: bool = True) -> None:
        """停点读数：票型 / 单帧共位 / 距锁步数（`[STOP]` 行）。"""
        if not self.enabled:
            return
        meta = self.last_meta or {}
        confirm = self._tsp3d_confirms(meta.get("xy"))
        lock_step = int(getattr(self.p, "_lock_step", -1))
        steps_since_lock = int(self.p._num_steps) - lock_step if lock_step >= 0 else -1
        if allowed:
            self.nav["stops"] = int(self.nav.get("stops", 0)) + 1
            key = "stop_confirmed" if confirm else "stop_unconfirmed"
            self.nav[key] = int(self.nav.get(key, 0)) + 1
            if 0 <= steps_since_lock <= 5:
                self.nav["stops_within5"] = int(self.nav.get("stops_within5", 0)) + 1
        print(f"[STOP] step={self.p._num_steps} src={meta.get('src', '?')} "
              f"n_obs={meta.get('n_obs', 1)} "
              f"gd_conf={float(meta.get('gd_conf', 0.0) or 0.0):.2f} "
              f"tsp3d_confirm={confirm} steps_since_lock={steps_since_lock} "
              f"stop_allowed={bool(allowed)}")

    # ---------------------------------------------------------------- 每集汇总
    def episode_summary(self) -> None:
        """打印上一集的 `[NAV]` summary（policy `_reset` 调用）。"""
        if not self.enabled:
            return
        nav, pick = self.nav, self.pick
        if nav or pick:
            print(f"[NAV] episode summary: stops={nav.get('stops', 0)} "
                  f"stops_within5={nav.get('stops_within5', 0)} "
                  f"stop_confirmed={nav.get('stop_confirmed', 0)} "
                  f"stop_unconfirmed={nav.get('stop_unconfirmed', 0)} "
                  f"approach_confirmed={nav.get('approach_confirm', 0)} "
                  f"approach_unconfirmed={nav.get('approach_unconfirm', 0)} "
                  f"picks={pick.get('n', 0)} pick_gd={pick.get('gd', 0)} "
                  f"pick_both={pick.get('both', 0)} pick_tsp3d={pick.get('tsp3d', 0)}")

    # ---------------------------------------------------------------- 数据出口
    def memory_entries_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """当前记忆条目的紧凑快照（trainer 逐集读取，做条目级 tp / fp 标注）。

        在集结束、下一次 `act()` 之前读取（此时 policy 尚未清记忆）。只放标量，键为
        `"<class>[i]"`（不注入 `policy_info`：habitat 的 `extract_scalars_from_info`
        会对非标量执行 `float()`）。`conf` = 条目 `c'`，`gd_conf` = 该条目历史上最强的
        原始 GD sigmoid（量测），两者并列便于比较尺度。
        """
        out: Dict[str, Dict[str, Any]] = {}
        if not self.enabled:
            return out
        mem = getattr(self.p, "_target_3d_memory", {}) or {}
        ver_all = getattr(self.p, "_target_verify_state", {}) or {}
        states_all = getattr(self.p, "_target_fallback_state", {}) or {}
        for cls, centroids in mem.items():
            ver = ver_all.get(cls, [])
            states = states_all.get(cls, [])
            for i, centroid in enumerate(centroids):
                st = states[i] if i < len(states) else {}
                v = ver[i] if i < len(ver) else (1, np.zeros(2), 0.0, 0.0)
                c = np.asarray(centroid, dtype=np.float64).reshape(-1)
                out[f"{cls}[{i}]"] = {
                    "class": str(cls),
                    "x": float(c[0]),
                    "y": float(c[1]) if c.size > 1 else 0.0,
                    "src": str(st.get("src") or "tsp3d"),
                    "num_obs": int(v[0]) if len(v) >= 1 else 1,
                    "conf": float(v[3]) if len(v) > 3 else 0.0,
                    "gd_conf": float(st.get("gd_conf", 0.0) or 0.0),
                    "suspicious": bool(st.get("suspicious", False)),
                }
        return out

    # ------------------------------------------------------------------ 内部
    def _tsp3d_confirms(self, xy: Any, tol: float = 0.5) -> bool:
        """本帧 admitted 的 TSP3D 候选是否与 `xy` 共位（0.5 m）。"""
        if xy is None:
            return False
        p = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
        for _cls, cxy in self.frame_admitted:
            c = np.asarray(cxy, dtype=np.float64).reshape(-1)[:2]
            if float(np.linalg.norm(c - p)) <= float(tol):
                return True
        return False
