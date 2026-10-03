# Project VLVM: 3D Native 主动视觉语言导航系统 · Stage2

> **阶段状态（2026-09-18）**：**Stage2 机制收官**——单场景最终档 = occ 门 + GD 辅助源 + 锁保持（`lock_exempt`）+ 身份豁免 + 冲突门（`sx_conflict_gate`），SR **0.5657（56/99）** · Oracle **55.6%** · spl 0.2118 · soft_spl 0.2540（已超 VLFM 同场景 0.5253）。
> **承接**：`VLVM-S1.md`（Stage1：单场景 0.4444 定稿）。


## 1. Stage2 目标与路径

Stage1 定稿 `wm_near_refresh_val`（SR 0.4444）之后，剩余差距 = **"锁对与到达"**（Oracle 与 VLFM 持平，而 fp 约为 VLFM 的 1.7 倍）。Stage2 沿三条线推进：

1. **检测准入治理**：occ 几何门、S-penalty 参数收口、冲突门（高 conf × 低 S 定向拒绝）；
2. **记忆生命周期治理**：锁保持（锁定条目豁免删除预算）、身份豁免（豁免匹配基准修正）；
3. **第二提议源**：GD 辅助机制（帧级 both/gd 票型）。

**成果**：单场景 SR 0.4444 → **0.5657**（+12.1pp），其中 S-penalty 参数/通道治理与三件新机制各自可归因；检测级假阳性治理首次形成闭环（写入口、生命周期、锁定期三层护栏）。


## 2. 核心架构增量（相对 Stage1）

### 2.1 GD 辅助提议源（`gdp_*`，`_init_gd_proposer`）

- **形态**：同一帧的第二个独立提议源（GroundingDINO 服务，port 12181），与 TSP3D 候选做帧级共位匹配 → 票型 `both`（共位）/ `gd`（单票）。
- **写记忆规则**：`both` ⇒ 正常流程并**清除可疑标志**（跨帧票型合并同理）；`gd` 单票 ⇒ 过同一几何门后写 `suspicious`；TSP3D 新建条目在 GD 开启时恒为可疑（2 帧预算）。
- **弱证不锁**（`gdp_weak_guard` × `lock_min_obs`）：`gd` 单票需 ≥2 次观测才可驱动 explore → navigate。
- **读数**：`every_n=5` 最优（n1 40/99、n3 46/99、**n5 50/99**）；`gdp_max_boxes` 2→1 再 +1（并入默认）；`caption_style=target` 判负 → `caption_style=vocab`。

### 2.2 记忆生命周期治理（`lock_exempt` / `lock_exempt_identity`，`_init_memory_manager`）

- **锁保持**：navigate 锁定期间，与 nav goal ≤0.5 m 的条目豁免近场 `near_miss` 删除预算（防"延迟停靠"变"丢条目"）。读数：**52/99（+1）· 步数 −26.5**。
- **身份豁免**：豁免匹配基准由 nav goal 近表面点改为**被追踪条目质心**（修正大件目标近表面点偏离质心 >0.5 m 的漏豁免，52.5%）。读数：**54/99（+2）· 两次复跑逐集一致**。
- 配套锁状态 `_lock_step`（当前锁开始步数）；回退机制（`near_miss` 滞回删除 → fallback explore）为 Stage1 延续。

### 2.3 检测准入链（occ 门 / S-penalty 收口 / 冲突门）

- **occ 几何门**：3D 体素占据一致性（8 voxels / ratio 0.001），REJECTED（空腔）检测在 S-penalty 前丢弃。
- **S-penalty 参数收口**（四项维持）：`sigma_tar=0.70` · `s_penalty_thresh=0.15` · `s_penalty_uncovered_w=1.0`（未覆盖免罚）· `s_penalty_floor=0.3`；`use_surface=True` + `goal_use_surface=True`。
- **冲突门**（`sx_conflict_gate`）：写入口对 "conf ≥ 0.80 ∧ 0 < S < thresh"（全场 tp 率最低格）定向拒绝。读数：**56/99（+2 赢 0 输）· Oracle +2 · 冲突检测 141 起全拒零逃逸**。


## 3. 机制定稿与判定总表

按模块分六部分：memory-life / S-penalty / occ geometry / world fusion / GD / 通用前置量测器。

### 3.1 memory-life（记忆生命周期）

覆盖条目「写入合并 → 存活回退 → 锁定保护」三段。

| 机制名 | 操作 | 落点 | 判定（读数） |
| :--- | :--- | :--- | :--- |
| EMA 跨帧合并 | 同类别质心距离 ≤ `merge_dist_thresh`（0.5 m）的观测合并进原条目：质心 EMA（`ema_weight_old`=0.8）、`num_obs+1`、保留 max c'、`near_miss` 清零；超出则注册新条目（幻觉框漂移无法合并） | `target_memory_manager.accumulate()` | 定稿 |
| 可疑标志链 | 置信度 < `fb_suspicious_conf`（0.75）或超距（> `max_depth`×0.95）→ 标可疑；GD 开启时新条目恒为可疑（2 帧预算），`both` 票型写入即清除 | 同上 | 定稿 |
| 近场滞回回退（fallback） | 条目落入近场 FOV 锥（`fb_near_radius`=2.5 m）且无新写 → 逐帧累计 `near_miss`；可疑条目超 `fb_suspicious_hysteresis`（2）帧、可信条目超 `fb_hysteresis`（5）帧 → 删除条目、回退 explore | `target_memory_manager.step()` | 定稿（对齐 VLFM；fb 三参未动） |
| 锁保持（`lock_exempt`） | navigate 锁定期间，与 nav goal 距离 ≤0.5 m 的条目豁免 `near_miss` 删除预算 | `_query_and_process` 传 `nav_goal_xy` + `step(exempt_nav_goal=True)` | +1 · 步数 −26.5 |
| 身份豁免（`lock_exempt_identity`） | 豁免匹配基准由 nav goal 近表面点改为被追踪条目质心（修正大件目标近表面点偏离质心 >0.5 m 的漏豁免） | `_query_and_process`（默认开） | +2 · 复跑逐集一致 |
| 锁状态 `_lock_step` | 记录当前锁定开始步数（锁保持与停止确认的配套状态） | `tsp3d_objectnav_policy.py` | 定稿 |

### 3.2 S-penalty

| 机制名 | 操作 | 落点 | 判定（读数） |
| :--- | :--- | :--- | :--- |
| S-penalty 软 gate | $c' = c \cdot w_S(S)$：S < `thresh` 时 $w_S = \max(floor,\ S/thresh)$，未覆盖（S≤0/None）取 `uncovered_w` | `s_penalty.py`（`compute_w_s` / `apply_s_penalty`） | +12.12pp（0.3030→0.4242，gate015） |
| 参数收口 | `sigma_tar`=0.70 · `s_penalty_thresh`=0.15 · `s_penalty_floor`=0.3 · `s_penalty_uncovered_w`=1.0（未覆盖免罚） | 同上 | 定稿 |
| 近表面点（`s_penalty_use_surface` / `goal_use_surface`） | S 查询点与导航终点改用 bbox 近表面点（面向相机）而非质心 | `s_penalty.near_surface_point` / 策略导航口 | 定稿 |
| 冲突门（`sx_conflict_gate`） | 写入口定向拒绝 conf ≥0.80 ∧ 0 < S < `thresh` 的检测（不写记忆） | `apply_s_penalty(conflict_gate=True)`；开关 `_init_s_penalty` | +2 赢 0 输 · 零逃逸 |

### 3.3 occ geometry（几何准入门）

| 机制名 | 操作 | 落点 | 判定（读数） |
| :--- | :--- | :--- | :--- |
| occ 几何门（`enable_occ_consistency`） | 检测框须在 3D 占据栅格中有占据体素支撑（≥`occ_min_voxels`=8 voxels / ratio ≥0.001）；空腔框 REJECTED，在 S-penalty 前丢弃 | `target_geometric_gating.py`（`BoxOccupancyGate.evaluate` / `TargetGeometricGatingEngine.process_detection`） | 合并档 +1（45/99） |

### 3.4 world fusion（融合器 / 输入适配）

| 机制名 | 操作 | 落点 | 判定（读数） |
| :--- | :--- | :--- | :--- |
| world 增量地图 | 体素 0.02 m 定栅格、上限 400k 体素、半径 6.0 m 滑出；帧窗口 8 帧 | `world_map.WorldLocalMap.update` | Stage1 定稿（0.4444） |
| 视角去重 | 位移 <0.15 m 且 yaw <15° 的帧不进入融合（进入端门） | `WorldLocalMap._view_gate` | 同上 |
| 近场刷新 | ≤3.0 m 的重复体素由当前帧重新持有（键层） | `WorldLocalMap.update`（`near_refresh_radius`） | 同上 |
| 值层滑动（`wm_near_refresh_value`） | 近场刷新时值层（points）同步由最近帧覆盖：进场由最近帧、远场由首见帧决定 | 同上 | 同上（wm_near_refresh_val） |
| 距离自适应采样 | 近/中/远分带体素化 0.01 / 0.02 / 0.05 m（分界 1.5 / 3.0 m），近密远疏 | `sampling.distance_adaptive_sample` | 定稿 |
| 统一点云 cap | 发送 TSP3D 前按 `cap_style`=random 限制点数上限 | `sampling.cap_point_count` / `pipeline.TSP3DInputPreprocessor.prepare` | 定稿 |

### 3.5 GD 辅助提议源

| 机制名 | 操作 | 落点 | 判定（读数） |
| :--- | :--- | :--- | :--- |
| GD 提议器（`gdp_*`） | 同帧第二提议源（GroundingDINO 服务，端口 12181）：每 `gdp_every_n`=5 帧跑一次、每帧至多 `gdp_max_boxes`=1 框、阈值 `gdp_box_thr`=0.4、`gdp_caption_style`=vocab | `grounding_dino/proposer.py`；策略 `_init_gd_proposer` / `_gd_apply` | n5 50/99；mb1 +1 |
| 票型模型（`both` / `gd`） | GD 提案与准入 TSP3D 候选帧级共位（≤`gdp_merge_dist`=0.5 m / IaU ≥`gdp_merge_iau`=0.3 / 包含 ≥0.8）→ `both`；否则 `gd` 单票；`both` 清除可疑标志 | `grounding_dino/provenance.py` | 定稿 |
| 弱证不锁（`gdp_weak_guard`） | `gd` 单票需 ≥`lock_min_obs`（2）次观测才可驱动 explore → navigate | 同上 + 策略锁定口 | 定稿 |

### 3.6 通用前置量测器

| 机制名 | 操作 | 落点 | 判定（读数） |
| :--- | :--- | :--- | :--- |
| `enable_stats` 量测器 | 零行为（不参与决策）：输出 [NAV] / [STOP] / 摘要（检测级与条目级判决读点） | `vlvm_stats_logger.py` | 默认开（零行为，可按需关） |


## 5. 参数总览

### 5.1 memory-life（记忆生命周期）

| 参数 | 机制（归属） | 值 |
| :--- | :--- | :--- |
| `enable_fb` / `fb_hysteresis` / `fb_suspicious_hysteresis` | 回退（可信 / 可疑滞回） | True / 5 / 2 |
| `fb_suspicious_conf` | 可信-可疑分界（GD 开启时不作用，仅留档） | 0.75 |
| `merge_dist_thresh` / `ema_weight_old` | 跨帧合并 / EMA | 0.5 / 0.8 |
| `lock_exempt` / `lock_exempt_identity` | 锁保持 / 身份豁免 | True / True（默认开） |
| `lock_min_obs` | GD 单票解锁门槛 | 2 |

### 5.2 S-penalty

| 参数 | 机制（归属） | 值 |
| :--- | :--- | :--- |
| `enable_s_penalty` / `s_penalty_thresh` | 开关 / S 阈值 | True / 0.15 |
| `s_penalty_floor` / `s_penalty_radius_m` | 惩罚下限 / 查询半径 | 0.3 / 0.5 |
| `s_penalty_uncovered_w` | 未覆盖（S≤0/None）权重 | 1.0（免罚） |
| `s_penalty_use_surface` / `goal_use_surface` | S 查询点 / 导航终点用近表面点 | True / True |
| `sx_conflict_gate` | 冲突门（conf≥0.80 ∧ 0<S<0.15 拒绝） | True |
| `sigma_tar` | 写入口准入阈值（c' = c·w_S ≥ 本值） | 0.70 |

### 5.3 occ geometry（几何准入门）

| 参数 | 机制（归属） | 值 |
| :--- | :--- | :--- |
| `enable_occ_consistency` / `occ_min_voxels` / `occ_min_ratio` | occ 门 | True / 8 / 0.001 |

### 5.4 world fusion（融合器 / 输入适配）

| 参数 | 机制（归属） | 值 |
| :--- | :--- | :--- |
| `fusion_style` / `enable_scan` | 融合路线（world / world+scan） | world / False |
| `pcd_window_size` / `wm_max_frames` | 帧窗口上限（camera / world） | 8 / 8 |
| `wm_voxel_size` / `wm_max_voxels` / `wm_radius` | world 地图体素 / 上限 / 半径 | 0.02 / 400000 / 6.0 |
| `wm_min_view_disp` / `wm_min_view_yaw` | 视角去重（world 进入端） | 0.15 / 15° |
| `wm_near_refresh_radius` / `wm_near_refresh_value` | 近场刷新（半径 / 值层滑动） | 3.0 / True |
| `cap_style` / `distance_sample` | 点云 cap / 距离自适应采样 | random / True |
| `near_dist` / `mid_dist`（+ 各档 voxel） | D3 分带 | 1.5 / 3.0（0.01/0.02/0.05） |

### 5.5 GD 辅助提议源

| 参数 | 机制（归属） | 值 |
| :--- | :--- | :--- |
| `gdp_enable` / `gdp_every_n` / `gdp_max_boxes` | 开关 / 频率 / 每帧框数 | True / 5 / 1 |
| `gdp_box_thr` / `gdp_caption_style` | 框阈值 / 文本风格 | 0.4 / vocab |
| `gdp_weak_guard` / `gdp_merge_dist` / `gdp_merge_iau` | 弱证不锁 / 共位判据 | True / 0.5 / 0.3 |

### 5.6 通用前置量测器

| 参数 | 机制（归属） | 值 |
| :--- | :--- | :--- |
| `enable_stats` | 通用前置量测（[NAV]/[STOP]/摘要；零行为） | True（可按需关） |

**--- 基础（感知 / 导航；非 Stage2 机制） ---**
| `sigma_sce` / `tau` | TGP 体素保留 / CBA 补全（TSP3D 推理内参） | 0.15 / 0.15 |
| `min_depth` / `max_depth` / `fov_angle` | 深度带 / FOV | 0.5 / 5.0 / 79.0 |
| `det_seed` | 每步 numpy 重播种基数 | 0 |
| `pointnav_stop_radius` / `nav_slice_height` | 停靠半径 / 导航切片高度 | 0.90 / 0.30 |
