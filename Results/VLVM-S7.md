# Project VLVM Stage 7：四机制导航优化

> 定位：S7 承接 S6 的 GD 机制与导航链路，按提高 ObjectNav SR 的杠杆组织机制。**目标确认与探索终止 → 导航点有效性与时效性** 仍在线；**field 目标覆盖 → middle-tier 提升概率** 已移出 policy，仅保留离线模块。四者不混在一次 A/B 中。

## 0. 机制顺序与实验总览

> 目标：提高 ObjectNav SR。按 S7 baseline 的失败杠杆排序推进；field/middle-tier 当前不进入在线主链路。

| 顺序 | 机制 | 直接解决的问题 | 主要证据 | 当前状态 | 实验形式 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 目标确认与探索终止 | 目标未确认时 explore STOP、跨楼层找不到目标、错误 STOP、正确目标迟迟不确认 | 33 个 FP + 48 个 never_saw 以 `explore/0` 结束；45 个错误 STOP 中 42 个已在 0.9 m 内；13 个超时 oracle=1 | 1A 已运行；1B floor shadow 已运行；1C 只保留 blacklist 状态门；几何/时效/独立证据门均弃用 | 分 1A/1B/1C，逐步 A/B |
| 2 | 导航点有效性与时效性 | 大目标表面点落在 bbox 外、first-write surface 过期 | 9 个“质心 TP、目标点非 TP”错误 STOP；S1/S2 只有间接 surface 证据 | 设计记录，未实现 | 先离线回放，再在线 A/B |
| 3 | field 目标覆盖 | no-TP / 未看到目标时的区域搜索 | Top-5 条件覆盖 0.8726，no-TP 0.7166 | **已移出 policy**；仅保留离线模块 | 暂不进入在线 A/B |
| 4 | middle-tier 提升概率 | 中间档 `gd-only / tsp-confirmed` 的目标排序 | 652 样本；`tsp-confirmed` held-out AUC 约 0.62–0.65 | **已移出 policy**；仅保留离线模块 | 暂不进入在线 A/B |

### 0.1 实验批次

| 批次 | 改动 | 对照 | 主指标 | 进入下一批条件 |
| :--- | :--- | :--- | :--- | :--- |
| R0 | S7 baseline | — | 141/297，SR 47.47% | 已完成，作为所有 A/B 的基线 |
| E1 | 1A：目标未确认时禁止 explore STOP（已运行，结果见 [VLVM-S7M1-exp.md](VLVM-S7M1-exp.md)） | R0 | `explore/0` 数、success、steps、FP | 5cd/QaL 不退化，mL8 探索早停下降 |
| E2 | 1B：楼层/楼梯感知 frontier（shadow 已运行；3D frontier 为空，需修数据源） | E1 | mL8 oracle 覆盖、floor transition、SR | 跨楼层 episode 出现真实楼梯尝试，且不增加 timeout |
| E3 | 1C：gd-only 升级只保留 blacklist 状态门（几何/时效门已弃用） | E2 | gd-only 提升 TP/FP、wrong stop、timeout、success | FP 下降且 TP 提升不显著下降 |
| E4 | 机制二：目标点有效性 + surface 时效性 | E3（先离线回放） | 9 个大目标 near-miss、goal switch、SR/SPL | 目标点不越出 bbox，goal 抖动可控 |
| E5 | field 目标覆盖动作臂 | E4 | no-TP 覆盖、首次 TP 提前量、FP/steps | 已移出 policy，暂缓 |
| E6 | middle-tier 在线排序 | E5（先离线 gate） | success、SPL、wrong stop、timeout | 已移出 policy，暂缓 |

E1–E6 不并行改代码；每一批都要能单独回退。field 和 middle-tier 已移出 policy，不再影响机制一/二的 STOP 和身份确认逻辑。

> 1A 已实现；1B floor shadow 已运行；1C 现在只保留 STOP blacklist 状态门。几何门、时效门、独立命中计数、目标关联门、语义门、视点独立性门、主动复核及所有独立证据门均已弃用。后续实验只考虑 floor 机制和 blacklist。

- 所有机制用同一组场景和 seed，分开 A/B；不同时改两个机制，避免无法归因。
- 评测按 `5cd / QaL / mL8` 分场景报告，不能只看总体 SR。
- 机制 1 的三部分按 1A → 1B → 1C 顺序做消融；1A 先验证“禁止 explore STOP”是否单独有效。
- 机制 2 不改变机制 1 的目标确认逻辑；field 和 middle-tier 已移出 policy，不进入在线主链路。
- `likely_infeasible` 是最终地图标签，不是 GT 不可达标签；跨楼层样本必须单独报告。

## 1. 机制一：目标确认与探索终止

> 这个机制把三件事合成一个动作/证据闭环：**目标未确认时不能结束探索、跨楼层必须继续找、找到 gd-only 候选后必须通过几何/状态和时效门再提升**。它直接对应 baseline 中最大的三组失败。

### 1.1 问题与证据

- 33 个 `false_positive` 和 48 个 `never_saw...likely_infeasible` 的最后动作都是 `Mode: explore | Action: 0`，合计 81/156 个失败。
- 48 个 never_saw 全部在 mL8；中位 93 步、范围 13–159，不是 500 步超时；38/48 起始点与目标高度差超过 1.5 m。
- 45 个错误 STOP 全部来自 `gd-tsp confirmed`；42/45 已在选中目标点 0.9 m 内；33/45 没有任何 TP 候选或 TP 记忆。
- 22 个超时中 13 个 oracle=1，17 个曾有 TP 记忆；正确目标看到了，但没有积累到可 STOP 的确认条件。

结论：当前主要不是“排序不够聪明”，而是**探索终止、楼层覆盖和目标身份确认三个动作语义问题**。

### 1.2 1A：目标未确认时禁止 explore STOP

> 状态：**已运行；结果见 [VLVM-S7M1-exp.md](VLVM-S7M1-exp.md)**。实现位置：`vlfm/policy/tsp3d_objectnav_policy.py` 的 `_explore_safe()`；`_explore()` 的 STOP 被拦截，frontier 耗尽进入回退动作。

- **`explore_exhausted` 是策略内部状态，不是环境终止状态。** 它对应 `vlfm/policy/itm_policy.py::_explore()` 在 `frontier_sensor` 为空或全零时返回 `_stop_action` 的分支。
- **STOP 权限要分两层看**：baseline `_pointnav(stop=True)` 的 STOP gate 只允许 `state=confirmed` 且 `is_gd_tsp` 的目标；但 `_explore()` 的空 frontier 分支直接返回动作 0，绕过了这个 gate。因此“只有 gd-tsp 可以产生 STOP”目前只是 pointnav 目标 STOP 的资格规则，不是完整的动作流不变量；1A 要把它变成完整不变量。

**已实现行为**

- `_explore_safe()` 包装 `_explore()`；检测到动作 0 时不发给环境，设置 `_explore_exhausted=True` 并累计 `_explore_stop_suppressed`。
- 前 3 次 fallback 原地左转扫描；第 4 次尝试回到 `_last_frontier`（`_pointnav(stop=False)`）；若旧 frontier 无效或就在附近，继续转向。
- `explore_exhausted` / `explore_stop_suppressed` 写入 `policy_info`，但不改变 STOP gate；只有 `gd-tsp confirmed` 的 `_pointnav(stop=True)` 能产生目标 STOP。

**做法**

1. 在 policy 执行 `_explore()` 后检查动作：若动作为 0（STOP），且没有已确认的 `gd-tsp` 目标，则不把 STOP 发给环境。
2. 将动作替换为安全的回退动作，并设置 `explore_exhausted` 状态；只有 `gd-tsp confirmed` 的 `_pointnav(stop=True)` 可以产生 STOP。
3. frontier 耗尽时按固定回退阶梯处理：
   - 回到最近一个未完成 frontier；
   - 原地转向/短距离前进，重新激活 frontier；
   - 搜索楼梯入口或下一层 frontier；
   - 仍无 frontier 时继续探索预算，不结束 episode。
4. 记录 `explore_stop_suppressed`、原因、剩余 frontier 数、当前楼层和动作替换结果。

**实验**

- A/B：`static explore` vs `suppress explore STOP`。
- 主指标：success、`explore/0` episode 数、steps、FP、timeout。
- 分场景看 mL8；验收要求 5cd/QaL 不退化，mL8 的探索早停显著减少。
- 这是机制 1 的最小改动，先单独验证，不混入楼层和身份改动。

### 1.3 1B：楼层/楼梯感知 frontier

> 状态：**per-floor frontier 与 floor entry 已接入 explore fallback；结果见 [VLVM-S7M1-4exp.md](VLVM-S7M1-4exp.md)**。实现文件：`vlfm/vlvm/floor_manager/floor_shadow.py` 写 shadow，`FloorManager.nearest_entry()` 在当前层 frontier 耗尽时提供最近的另一层 entry；启用开关 `FLOOR_SHADOW_ENABLE=1`，输出 `detections/floor_shadow.jsonl`。floor 动作只在 explore 分支生效，不改变目标选择或 STOP。floor manager 设计见 [VLVM-S7-floor.md](VLVM-S7-floor.md)。
>
> 输入接线（2026-11-01）：新增 `floor_camera_height` sensor，仅向 floor manager/floor shadow 提供 episode-relative 的动态 `(agent_base_z, camera_z)`；现有导航栈的 `camera_height=0.88` 仍固定不变。旧的 `frontier_sensor_3d` 已移除：它把 2D frontier 按当前 agent 高度 snap，所有 z 相同，不能代表真实楼层。现在由 `per_floor_frontier.py` 用动态相机高度的真实 3D 点云/occupancy 生成每层 2D frontier，frontier z 取所属楼层的 `z_ref`。
>
> mL8 不是单层场景：99 个 episode 的起点高度覆盖约 -2.58 到 2.82 m；但 M3 总体平均 `traveled_stairs=0.0034`、`cumulative_elevation_change=0.18`，说明该 run 基本没有实际跨层。floor shadow 要验证的是 floor entry 是否出现，以及 agent 是否进入其他楼层。
>
> 高度语义：`floor_camera_height` 每帧记录动态 agent/camera 高度；`FloorManager` 只在创建新楼层时写入 `FloorRecord.z_ref`，同层内的动态高度不会改写该层参考高度。跨层后创建新 `FloorRecord` 并切换 current floor。

> 日志：`frontier_count_2d/3d`、`floor_proxy_id/z`、`floor_proxy_changed`、`visible_floor_ids`、`cross_floor_frontier_count`、`frontier_z_min/max/median/span`、`explore_exhausted`、`explore_stop_suppressed`、`entry_candidate_*`。`floor_proxy` 是 frontier 高度聚类代理，不是 Habitat GT 楼层。floor manager 另行记录 `floor_count/level_count`、`floor_records`、`entry_count/entry_status_counts`、`entries` 和独立 `entry_blacklist`。entry 只由 TSP3D `stairs` 检测创建，记录 `entry_xy`、`entry_dz`（到达后测量）和 `floor_id`，不再记录 exit、STAIR 或其他节点类型；其他层 frontier 不创建 entry。

**维护楼层管理器** & **frontier 耗尽后优先跨楼层探索**
不改变目标选择、身份确认、STOP 资格，也不取消 1A 的 explore STOP 抑制不变量。

**(1) 维护楼层管理器**

- 按 frontier 点的高度聚类得到楼层代理，维护每层的 frontier 集合、已覆盖区域、当前层代理高度和可见楼层。
- 把楼梯入口/出口作为 **floor entry**，不是普通 2D frontier；维护跨层可达边、最近一次尝试、失败/阻塞状态和每层探索进度。
- 只有机器人实际发生高度/楼层代理切换时，才更新当前楼层；不能因为“计划去楼梯”就提前把 frontier 记到另一层。
- 低层 frontier 不因切换到高层而删除；每层保留独立的耗尽状态和未完成 frontier，便于回退和再次探索。

**(2) frontier 耗尽后的行为**

触发条件：当前层 frontier 数为 0 或低于阈值，目标未确认/未看到，且当前不在楼梯通行过程中。

执行顺序：

1. **不发送 STOP**，继续遵守 1A 的安全回退；`explore_exhausted` 只表示当前层探索耗尽，不表示 episode 应结束。
2. 查询楼层管理器的已验证 entry；entry 必须垂直连通、未近期失败、未被 entry blacklist，并按“距离、垂直高差、目标层未探索面积、历史失败次数”排序。
3. 若存在候选：用 `stop=False` 导航到最近的楼梯入口，记录 `cross_floor_intent`；到达入口后尝试跨越，不能仅凭意图切换楼层。
4. 实际跨层后：更新当前楼层，刷新新层 frontier，清除新层的耗尽状态，继续正常探索；原楼层状态保留。
5. 若无 entry 或跨越失败：将该 entry 加入独立 entry blacklist（记录 `entry_xy` / `entry_dz`），尝试下一个最近 entry；没有其他 entry 时，回退到最近未完成 frontier。当前层 entry 和 frontier 都耗尽时直接结束 episode，仍受 500 步上限约束。
6. 只有所有楼层都耗尽且回退预算用完，才标记为全局探索耗尽；`likely_infeasible` 只作为离线地图标签，不能直接当成 GT 不可达。

**实验**

- 先离线检查 mL8 的 48 个 never_saw 中目标高度差、楼梯入口可见性、是否存在可行跨层路径。
- A/B：1A + `floor-aware frontier`。
- 指标：mL8 oracle 覆盖、`traveled_stairs`、floor transitions、never_saw 数、SR、timeout。
- 若目标确实不可达，单独标记为 infeasible，不纳入可恢复失败。

### 1.4 1C：gd-only 升级只保留 blacklist 状态门（2026-11-02 口径）

> 决策：机制一保留 **1A（目标未确认时禁止 explore STOP）+ 1B（floor 机制）**，下一阶段重点做 floor，详见 [VLVM-S7-floor.md](VLVM-S7-floor.md)。
>
> **完全放弃**以下机制，不再保留相关代码、shadow 或启动开关：
>
> - 独立命中计数
> - 目标关联门
> - 语义门
> - Identity / 视点独立性门
> - 主动复核
> - 独立证据门
> - gd-only 几何门
> - gd-only 时效门
>
> 当前 gd-only 升级只保留一个状态门：**STOP blacklist**。后续实验只考虑 floor 机制和 blacklist。

**适用范围**

- 只作用于 `lifecycle=gd` 的 gd-only 目标，在 admitted TSP3D 候选与其 merge、准备提升为 `gd-tsp confirmed` 时检查。
- 不改变 gd-only 的新建/merge，不改变 confirmed/provisional 的普通升级，不改变目标选择或 STOP 资格。
- 门不通过时保留 gd-only，不删除记忆；候选照常记录，但不产生 `gd_only_tsp_merge` 升级。

**当前唯一的升级状态门**

- 目标 `memory_id` 不在 STOP blacklist 中才允许提升；
- 不再检查 `depth_ok`、`occ_passed`、`geom_status`；
- 不再检查 `gd_last_evidence_step` 或 age 阈值；
- blacklist 仍来自 STOP 失败路径，后续应进一步解耦“STOP 黑名单”和“升级黑名单”。

**实现位置**

- `vlfm/policy/tsp3d_objectnav_policy.py`：把 STOP blacklist 传给 memory manager；不构造 gd-only 几何门。
- `vlfm/vlvm/memory_life/target_memory_manager.py`：在 `promote_gd_only` 分支只检查 STOP blacklist；不通过时保留 gd-only，并返回 `promotion_gate_reason="blacklist"`。
- `scripts/_nav_s7.sh`：不再导出 `GD_ONLY_PROMOTION_MAX_AGE_STEPS`；启动摘要只声明 blacklist 状态门。
- floor 专用输入：`vlfm/vlvm/floor_manager/floor_camera_height.py` 提供 episode-relative 动态 `(agent_base_z, camera_z)`，只供 floor manager/floor shadow 使用；`vlfm/vlvm/floor_manager/per_floor_frontier.py` 用实际高度点云维护独立 3D occupancy，并投影出每层 2D occupancy/frontier，frontier z 取该层 `z_ref`。现有导航栈的固定 `camera_height` 不变。

**实验**

- A/B：1A + 1B + blacklist 状态门 vs 1A + 1B。
- 主指标：成功、FP、`gd_only_tsp_merge` 提升后的 TP/FP、错误 STOP、timeout。
- 分层指标：按 scene 和 `bed/chair/couch/tv/toilet` 类别报告；重点看 gd-only 的 FP 是否下降，同时 TP 提升是否被过度阻塞。
- 该门只处理升级路径，不与 floor 动作臂混合；floor 的当前层耗尽 fallback 已单独接入，需要新 run 验证。

### 1.5 机制一的验收门

- 1A：`explore/0` 失败数下降，5cd/QaL 不退化；
- 1B：mL8 的 never_saw 有实际 floor transition 尝试，oracle 覆盖提高；
- 1C：gd-only 升级的 TP/FP 比例改善，错误 STOP 不增加，TP 升级不被过度阻塞；
- 任何一步都按 scene 分层通过后再进入下一步。

## 2. 机制二：导航点有效性与时效性

> **目标点既要落在有效目标区域，也要随观测更新**。它不改变目标身份、不改变 STOP 资格。

### 2.1 问题与证据

- 9/45 个错误 STOP 是“质心 TP、目标点非 TP”：QaL 7 个、mL8 2 个，集中在 bed/couch；目标点到质心距离约 1.1–1.4 m。
- 当前 `goal_use_surface=True`，导航终点取首次写入的近表面点；merge 只更新质心，`surface` 保留 first-write 值。
- 首次表面点可能来自远距、斜视或部分遮挡；机器人接近后实际可停靠面变化，导致目标点过期或落在 bbox 外。

### 2.2 大目标目标点有效性

**做法**

1. 对 bed/couch 等大目标，目标点必须约束在目标 3D box 的投影/邻域内，不能只取离机器人最近的任意表面点。
2. 优先使用质心附近的可达点；若质心不可达，再取“在目标 bbox 内、且可到达”的表面点。
3. 目标点与目标质心距离、目标 bbox 一致性作为有效性门；无效时回退质心或重新生成观察视点。
4. 目标类别规则只作为几何约束，不改变候选身份或排序。

**实验**

- 先离线回放 9 个大目标错误 STOP，比较 first-write surface、centroid fallback、bbox-constrained surface 的目标点位置与 STOP 距离。
- 在线 A/B 只在 mechanism 1 稳定后进行；指标为 wrong stop、success、SPL、goal switch。

### 2.3 导航点时效性

保留当前记录的三类更新策略，作为同一个目标点机制的子策略：

| 策略 | 更新规则 | 预期收益 | 风险 |
| :--- | :--- | :--- | :--- |
| latest | merge 时 `surface ← 当前检测 surface` | 贴合当前视角 | 目标点抖动、切换增加 |
| EMA surface | `surface ← (1−α)·old + α·new`，小 α | 平滑刷新 | 可能平均到物体内部/跨面 |
| conditional refresh | 仅当新 surface 更近、可见或位移超阈值时更新 | 抑制无效刷新 | 规则和阈值需标定 |
| centroid fallback | surface 缺失/失效时用质心 | 稳定、已有回退 | 质心可能不可达 |

当前证据：[VLVM.md](VLVM.md) §2.7 记录 surface 作为导航终点、质心用于排序/豁免；S1 `goal_use_surface=True` 步数约 −17.3%，S2 `lock_exempt_identity` 质心基准 +2（54/99）。没有 latest / EMA surface 的直接 A/B。

**边界**

- 只改 `goal_use_surface=True` 时的记忆 surface 更新与目标点选择；不改 memory identity、merge/new/delete/revive、`sigma_c`、TSP3D 输出、目标选择和 STOP 资格。
- 不与机制 1 的身份确认混合；机制 2 只解决“点是否有效、是否过期”。

### 2.4 离线回放与在线门

现有日志字段：

| 文件 | 字段 | 用途 |
| :--- | :--- | :--- |
| `detections/candidates.jsonl` | `candidate_uid`、`step`、`surface`、`centroid` | 每次检测的近表面点 |
| `detections/memory_events.jsonl` | `candidate_uid`、`memory_id`、`event`、`step` | 把 surface 归到 memory |
| `detections/nav_goals.jsonl` | `goal`、`switch_reason`、`distance`、`step` | 实际导航点与切换 |

离线先算：

- first-write 与 latest/EMA/conditional 的 surface 位移分布；
- surface 到质心/目标 bbox 的距离、跨面/回退次数；
- 重放后的 nav goal 切换数与 goal 位移；
- 对 `pointnav_stop_radius=0.90 m` 内 STOP 的影响。

只有离线确认“first-write 过期比例显著”且“新策略 goal 抖动可控”，才进入在线 A/B。在线指标：wrong stop、success、SPL、timeout；大目标子集单独报告。

### 2.5 机制二的验收门

- 9 个大目标 near-miss 的目标点不再越出 bbox；
- goal switch/抖动不增加，success/SPL 不退化；
- 小目标和普通物体不因统一 surface 更新而退化。

## 3. 机制三：field 目标覆盖

> 进入条件：机制一、二稳定；field 只做目标覆盖，不能做排序选择。以下保留 field 的设计、离线证据和只读 shadow 边界。

> 目标：增加一个独立于 TSP3D `sigma_c`、现有 ValueMap `S` 与几何门的判据，估计“对目标位置进行下一步观察，预期能减少多少不确定性”。本节只定义目标覆盖机制与离线旁路验证，不改导航动作、目标选择、记忆生命周期或停止规则。

### 3.1 设计边界

- **能力边界**：field 只用于目标覆盖，不参与目标选择与准入；目标选择仍由 `sigma_c`/TSP3D 与 GD 证据负责。
- 场是 TSP3D 之外的独立证据通道；不改 TSP3D 权重或输出，不接 GD，不把场分数写回 `sigma_c`，也不与现有 `S` 做融合。
- 空间表示以现有 3D 几何地图为准。可复用 ValueMap 的坐标索引和语义推理接口，但不能把 BEV `H1` 或目标词 cosine 直接改名为新场。
- 场的语义依据是目标相关的区域上下文及物体－区域空间关系；几何依据是 3D 可达连通性、视线遮挡、观察视锥与已观测覆盖。初期用现有视觉语言接口对少量上下文提示作零样本打分，不训练 TSP3D，也不引入新 proposer。
- “信息增益”针对目标位置后验，而非一般地图未知度。高不确定但与目标无关的空间不应自动得到高分；未充分可见的区域也不能因没有检测到目标而被记作负证据。
- 场不直接判定 STOP。它只评估搜索／复核动作的信息价值；停止复核仍延后，不在本阶段接入。

### 3.2 输入与符号

以一次 episode 的空间区域或拓扑节点 $r\in\mathcal{R}_t$ 为基本索引。区域由 3D free-space 连通结构锚定；已观测目标线索的 3D 位置映射到相应区域。若现有地图暂时不能稳定分区，可先按固定尺度的 3D 空间块作索引，区域结构后续再细化。

| 输入 | 符号 | 来源与用途 |
| :--- | :--- | :--- |
| 目标类别／文本 | $g$ | episode ObjectNav 指令；生成目标相关区域上下文提示 |
| 当前 RGB-D 与相机位姿 | $I_t,D_t,T_t$ | 产生场语义观测，并把观察投影到 3D 空间 |
| 3D 几何及探索状态 | $M_t^{geo}$ | 现有 obstacle map；区分 free / occupied / unknown 与已观测区域 |
| 可达拓扑 | $G_t=(V_t,E_t)$ | 从 3D free-space 连通性得到区域间连通和路径代价 |
| 目标上下文语义 | $z_{r,k}$ | 视觉语言模型对区域 $r$ 的第 $k$ 种上下文描述打分，例如目标相关的区域功能或邻近物体关系；不使用 `sigma_c` |
| 目标线索坐标 | $x_j$ | 现有目标线索／memory 的坐标和 ID，仅用于统计目标覆盖密度与已覆盖区域；不读取置信分数 |
| 可执行观察动作 | $a=(v,\psi)$ | 位置 $v$ 与朝向 $\psi$；包括未覆盖区域的可达视点及 frontier 视点 |
| 观察模型参数 | $\rho_{a,r}, f_{a,r}$ | 在动作 $a$ 下观测到真目标／假目标信号的概率；由旁路数据标定，并区分可见、部分可见和不可见 |

初期提示集只覆盖项目目标类别相关的区域功能和常见上下文关系。提示打分属于场的独立语义通道；不得直接复用 TSP3D 目标分数，也不得把现有 $S$ 的单一目标 cosine 当成完整区域语义。

### 3.3 中间量计算

**(1) 语义－空间区域证据**

把区域内各次观测的提示打分按语义置信度和空间投影可靠性聚合：

$$
\bar z_{r,k}=\frac{\sum_{t\in\mathcal{O}(r)} q_{t,r}z_{t,r,k}}{\epsilon+\sum_{t\in\mathcal{O}(r)}q_{t,r}},\qquad
q_{t,r}=q^{pose}_{t,r}q^{depth}_{t,r}q^{vis}_{t,r}.
$$

其中 $\mathcal{O}(r)$ 是空间上覆盖区域 $r$ 的观测集合；$q^{pose}$ 表示位姿投影可靠性，$q^{depth}$ 表示有效深度比例，$q^{vis}$ 表示视角和遮挡可靠性。重复且高度重叠的视图按覆盖重叠折减权重，不能仅凭经过了若干帧就累计成多份独立证据。

目标相关的语义 logit 定义为：

$$
\ell^{sem}_r=b_g+\sum_k w_{g,k}\bar z_{r,k}
+\sum_{(k,m)\in\mathcal{Q}_g}\beta_{g,k,m}\,R_{r,k,m}.
$$

$b_g$ 为目标类别先验；$w_{g,k}$ 为区域上下文提示权重；$\mathcal{Q}_g$ 是目标与上下文物体／区域的关系集合；$R_{r,k,m}$ 是由 3D 相对位置、邻接或包含关系得到的关系证据。权重须在离线数据上拟合或校准，不能手工按最终测试标签调节。缺失的上下文证据记为 unknown，不当作负分。

**权重的拟合口径**

对训练样本 $n=(g,r)$，组装特征向量 $x_n=[\bar z_{r,1:K},R_{r,*},\ell^{geo}_r]$，标签 $y_n=1$ 表示区域内存在 GT 目标实例，$y_n=0$ 表示场景标注确认该区域没有该类实例。拟合 logit 权重时最小化带正则的加权交叉熵：

$$
\theta^*=\arg\min_\theta\left\{-\sum_n\omega_n\left[y_n\log p_n+(1-y_n)\log(1-p_n)\right]
+\lambda\lVert\theta\rVert_2^2\right\},\qquad
p_n=\operatorname{sigmoid}(b_g+\theta^\top x_n).
$$

初始数据量有限时，跨目标类别共享上下文权重 $\theta$，只给目标类别单独拟合先验截距 $b_g$；样本充足后才评估类别专属关系权重，并对类别偏移加收缩正则。$\omega_n$ 用于补偿按空间区域抽样造成的正负类比例偏差；若目标是校准真实后验概率，不能简单把正负样本采成 1:1 后直接使用未修正的输出。

数据划分按 **scene** 分组，不能随机拆同一场景中的区域或 episode。先用训练场景拟合 $\theta$；在独立校准场景上选择正则强度 $\lambda$、温度 $\tau$ 或 Platt 校准参数，使 NLL / Brier score 最低；S5-final 四场景作为锁定评估集，不参与提示权重、门槛或公式选择。若现有数据没有额外场景，只能做 leave-one-scene-out 的探索性开发，不能把它当成独立最终验证。

区域存在标签与“视点是否看见”是两种不同监督：区域 GT 标签来自 Habitat/HM3D 场景目标实例标注（或现有与 `top_down_map.target_bboxes_mask` 一致的 TP 标注口径）；视点观测结果由 RGB-D 几何可见性与 GT 实例投影联合生成。$\rho_{a,r}$ 和 $f_{a,r}$ 在可见性分层内分别按真目标／非目标视点的实测正票率估计，可用 Beta 平滑：

$$
\hat\rho_c=\frac{n_{+,1,c}+\alpha}{n_{1,c}+\alpha+\beta},\qquad
\hat f_c=\frac{n_{+,0,c}+\alpha}{n_{0,c}+\alpha+\beta},
$$

其中 $c$ 是可见体积分数、距离和遮挡程度的分层，$n_{+,1,c}$ 是该层目标实际存在且视觉模型给出正证据的次数，$n_{1,c}$ 是该层目标实际存在的有效可见次数；负类同理。不可见或不充分可见样本不计入 $n_{1,c}/n_{0,c}$。随后在校准场景上检验概率校准与可靠性分层是否单调。

**(2) 目标位置后验 $B_t$**

对每个区域及“尚未解释的空间”假设 $u$ 建立目标位置假设集合 $\mathcal{H}_t=\mathcal{R}_t\cup\{u\}$：

$$
L_r=\ell^{sem}_r+\ell^{geo}_r,\qquad
b_t(r)=\frac{\exp(L_r/\tau)}{\exp(L_u/\tau)+\sum_{j\in\mathcal{R}_t}\exp(L_j/\tau)},
\qquad
b_t(u)=1-\sum_{r\in\mathcal{R}_t}b_t(r).
$$

$\ell^{geo}_r$ 只表达区域是否是合理的目标位置／可观察位置，不把“未探索”误当作“无目标”；$\tau$ 是温度参数。$b_t(r)$ 表示目标位置落在区域 $r$ 的当前后验质量，$b_t(u)$ 保留未建图或证据不足位置的概率质量，避免已观察区域因归一化而被虚假地判成确定答案。若一个任务中允许多个同类目标，$b_t$ 解释为“下一处可成功目标位置”的分布；区域间相关性在首轮近似中忽略并作为局限记录。

**(3) 当前不确定性 $H_t$**

$$
H_t=-\sum_{h\in\mathcal{H}_t} b_t(h)\log_2\bigl(b_t(h)+\epsilon\bigr).
$$

$H_t$ 的单位为 bit。它是目标位置分布的不确定性；不能用 `H1` 的自由高度层比例替代。

**(4) 视点几何可见性和观察模型**

对观察动作 $a=(v,\psi)$ 与区域 $r$，由 3D 深度／障碍地图估计视锥交叠和射线遮挡：

$$
\nu(a,r)=\frac{\operatorname{Vol}(\text{可见且落入 }r\text{ 的目标可观察体积})}
{\operatorname{Vol}(\text{可观察目标体积于 }r)+\epsilon}.
$$

再将 $\nu(a,r)$、视觉模型在相似观察条件下的召回率和误报率标定为：

$$
\rho_{a,r}=P(o=+\mid h=r,a),\qquad
f_{a,r}=P(o=+\mid h\ne r,a).
$$

观测结果 $o$ 至少分为正证据 $+$、有效未见 $-$、不可判定 $?$。只有 $\nu(a,r)$ 达到由标定集确定的可观测条件时，才允许使用 $-$ 更新；否则结果为 $?$，不降低该区域后验。遮挡、深度无效或视角覆盖不足时，负证据似然比设为 1，即不更新。

**(5) 观察后的后验及期望熵**

对每个假设 $h$ 与结果 $o$，用观察似然更新：

$$
b_{t+1}(h\mid o,a)=\frac{P(o\mid h,a)b_t(h)}{\sum_{h'}P(o\mid h',a)b_t(h')}.
$$

结果概率与预期后验熵为：

$$
P(o\mid a)=\sum_hP(o\mid h,a)b_t(h),\qquad
\mathbb{E}[H_{t+1}\mid a]=\sum_{o\in\{+, -, ?\}}P(o\mid a)H(B_{t+1}\mid o,a).
$$

**(6) 几何路径代价 $C(a)$**

$$
C(a)=d_{G_t}(p_t,v)+\lambda_{turn}\,\Delta\psi+\lambda_{blocked}\,P(\text{路径受阻}\mid G_t).
$$

$d_{G_t}$ 是 3D free-space 拓扑上的可达路径长度，不是机器人与目标质心的 2D 欧氏距离；$\Delta\psi$ 是预期转向量。若视点不可达，则 $C(a)=+\infty$，该动作不进入覆盖动作集合。

### 3.4 核心场分数与最终输出

**动作级信息增益**：

$$
\Delta H(a)=\max\left(0,\ H_t-\mathbb{E}[H_{t+1}\mid a]\right).
$$

**目标条件语义－空间信息场分数**：

$$
F_t(a)=\frac{\Delta H(a)}{C(a)+\epsilon}.
$$

$F_t(a)$ 的单位是 bit／米（若成本中加入时间，则解释为 bit／单位成本）。它是新的独立覆盖判据，不是目标置信度，也不与 `sigma_c` 相加。为便于跨 episode 比较，同时记录原始 $\Delta H$、$C(a)$ 和 episode 内 percentile；首轮不设固定覆盖阈值。

field 只输出区域／动作级覆盖分数；因此这里只保留 $F_t(a)$。

**每个时刻最终旁路输出**：

| 输出 | 含义 |
| :--- | :--- |
| $b_t(r),b_t(u)$ | 各区域及未知区域的目标位置后验 |
| $H_t$ | 当前目标位置不确定性 |
| $\Delta H(a)$ | 动作预期减少的目标位置不确定性 |
| $C(a)$ | 对应 3D 路径与观察代价 |
| $F_t(a)$ | 归一化信息增益，独立场分数 |
| `validity / provenance` | 可见性是否充分、使用的语义关系和几何证据来源、无效原因 |

场不输出 STOP、不创建 confirmed 状态、不修改 memory，也不把未检测区域伪装成精确目标坐标。对未覆盖区域，输出的是带概率与不确定性的搜索区域假设。

### 3.5 目标覆盖可行性验证

field 只验证目标覆盖，不评估动作选择。验证分两层，都不影响当前动作。

**A. 区域级目标先验（离线）**

在 1 m cell 上拟合“目标更可能位于哪些未覆盖区域”的概率模型，按 scene 分组做 held-out 验证。指标为 AUC、Brier、NLL；它只说明区域级目标先验是否可学，不能说明动作选择或 STOP 可靠。

**B. 覆盖回放**

在每个历史时刻的 frontier／可达区域观察动作上计算 $F_t(a)$，统计无 TP 的 episode 中，场 Top-K 区域是否落入 GT 目标的可视邻域、后续轨迹是否实际进入这些区域，以及到达后是否获得 TP。主指标为可行 no-TP episode 的 GT 区域 Recall@K、首次 TP 提前量和新增 FP 代价；likely-infeasible episode 单列。

**覆盖门**：

- 区域覆盖门：在可行 no-TP episode 中，场 Top-K 能以可接受覆盖命中 GT 目标可视邻域。
- 轨迹覆盖门：覆盖命中能转化为真实轨迹上的 TP，而不只是离线空间邻近。
- 稳定门：按场景和类别分层后无单一类别主导；不可见区域保持 unknown，不因假设负证据错误压低后验。
- 成本门：视觉语义推理与 3D 几何查询的延迟、显存和额外视点数量可接受。

当前证据见 `Results/VLVM-S7f0-exp.md`：区域模型 train AUC 0.8201，Top-5 条件覆盖 0.8726（no-TP 0.7166），全局 Top-5 31.90%。这些结果只支持目标覆盖，不构成动作选择依据。

**当前数据限制**：S5-final 的 JSONL 没有逐时刻 3D explored/free/occupied 快照、区域上下文语义向量、相机视锥可见体素或可达观察视点，不能计算完整的 $B_t$、$H_t$、$\Delta H$ 或 $F_t$。用现有 $S$、`occ_ratio`、`num_obs` 拼代理分数只能检验旧信号，不能作为 field 覆盖可行性结论。

> `region_semantics.jsonl` 对当前几何覆盖方案不是必需项，**已决定放弃全量生成**；生成器与 `/tmp` 样本仅保留为可选复现。

### 3.6 后续 field 在线导航实验（已移出 policy）

> 以下为历史设计。field shadow、原始观测归档和在线接入已从 policy 与 `_nav_s7.sh` 移除；当前仅保留离线分析模块。

field 只保留目标覆盖能力，后续在线实验分两部分：

1. **覆盖 shadow（历史）** 在未覆盖区域／frontier 上计算 $F_t(a)$，把 Top-K 区域写入 `detections/field_shadow.jsonl`，不改变动作。主指标是 no-TP episode 的 GT 区域覆盖、首次 TP 提前量、FP/steps 和轨迹变化。S7 baseline Top-5 条件覆盖 0.8726（no-TP 0.7166）。
2. **region_semantic（单独覆盖臂）** 不要求先生成全量离线侧车；将该机制作为独立的在线导航实验臂，验证区域语义上下文能否改善目标覆盖。只有该独立臂显示可行性后，才讨论与几何覆盖合并；失败时不影响几何覆盖。
3. **停止仍延后** 场的覆盖分数只表示继续观察的价值，不构成 STOP 资格；目标身份确认机制另行讨论。

> 当前不改代码、不启动实验；上述拆分只固定后续实验的边界。
>
> **历史边界**：field 原始观测曾以 ZIP 记录，覆盖 shadow 曾在动作选择之后写入 `detections/field_shadow.jsonl`。这些在线链路已移除；当前 `_get_target_object_location()` 不读取任何 field score / region semantic。

field 只用于区域搜索与目标覆盖；目标选择仍使用 `sigma_c` 与 TSP3D 证据。其设计借鉴 [GLMap](https://arxiv.org/html/2605.01736) 的实例／区域语义关联、[TrajRAG](https://arxiv.org/html/2605.01700) 的几何－语义轨迹表示、[Chain-of-Search](https://ojs.aaai.org/index.php/AAAI/article/view/37150) 的收益－成本权衡，但不移植这些论文的完整系统。

### 3.7 S7 实现与数据完整性

| 模块 | 文件 | 状态 |
| :--- | :--- | :--- |
| 原始观测采集 | `vlfm/vlvm/semantic_spatial_field/observation_logger.py` | **已移出 policy**；不再在线生成 ZIP |
| 区域/标签/观测率 | `vlfm/vlvm/semantic_spatial_field/offline_data.py` | 离线模块保留 |
| 语义侧车生成 | `vlfm/vlvm/semantic_spatial_field/region_semantics.py` | 已删除/不参与运行 |
| 离线拟合/回放 | `vlfm/vlvm/semantic_spatial_field/offline_analysis.py` | 离线 CLI 保留 |
| 独立推理 | `vlfm/vlvm/semantic_spatial_field/inference.py` | 离线模块保留 |
| 在线导航 | `vlfm/vlvm/semantic_spatial_field/shadow.py` | **已移出 policy**；不再写 `field_shadow.jsonl` |
| 原始观测归档 | `field_observations/*.zip` | 不再生成；历史实验保留 |
| 在线 shadow 输出 | `field_shadow.jsonl` | 不再生成 |

> `scripts/_nav_s7.sh` 已移除 field shadow、field 归档校验和 middle-tier 参数/拟合。当前 S7 入口只保留 floor manager/shadow、GD 与黑名单 STOP 逻辑。

历史 `field_observations/<scene>/<episode>_<id>.zip` 的内容（当前不再生成）：

- `episode.json`：`schema=semantic_spatial_field_observations_v1`、`scene_id`、`episode_id`、`target_object`、`frame_count`、`map_spec`、GT key 列表。
- `frames/*.npz`：`step`、`view_index`、`target_object`、`robot_xyz`、`robot_xy`、`robot_yaw_rad`、`rgb`（policy 当前视角 RGB）、`depth_normalized_filtered`（policy 深度滤波后的归一化深度）、`camera_to_episodic`、`depth_range_intrinsics`、`frontiers_3d/2d`、`map_pixel_origin`。
- `ground_truth/top_down_map.npz`：`target_bboxes_mask`、`upper_bound`、`lower_bound`、`grid_resolution`、`tf_episodic_to_global`。

注意：深度是 **policy 滤波后的归一化深度**，不是原始传感器深度；每帧同时记录深度范围和相机内参，足以反投影。离线分析器 README/CLI：

```bash
python -m vlfm.semantic_spatial_field.offline_analysis \
  --run-dir outputs/<run> --out outputs/<run>/field_analysis
```

无 `region_semantics.jsonl` 时只拟合几何基线，报告中明确标记 `semantic_sidecar_available=false`；全量语义侧车已决定不生成，生成器与 `/tmp` 样本仅保留为可选复现，不阻塞采集或导航。单集冒烟验证结果：399 帧、11785 个区域样本、429 个正样本、区域模型 AUC 0.910；证明当前 ZIP 可被离线分析器完整读取。

语义侧车生成器（BLIP-2 ITM，按区域投影 + RGB patch 打分）：

```bash
python -m vlfm.semantic_spatial_field.region_semantics \
  --run-dir outputs/<run> \
  --out outputs/<run>/detections/region_semantics.jsonl \
  --scenes <scene> --episode-ids <episode> \
  --max-frames 200 --every-n 2 --max-regions-per-frame 3 \
  --contexts "a bed,a chair,a couch,a toilet,a television"
```

生成器只读取已完成的 field ZIP，不启动导航、不修改 policy；`--scenes`/`--episode-ids` 用于先验证含目标的回合。侧车按 `region_id` 对齐，缺失区域在 `SemanticSidecar` 中标记为不可用而不是负证据。离线分析现在会展开逐上下文分数（`sem_context_<prompt>`），并加入目标条件分数（`sem_target_context` / `sem_target_margin`），避免把所有 prompt 压成单一均值。

注意：归档中的 3D frontier 点可能为空，但 `frontiers_2d` 有值；离线特征与探索回放必须同时使用两者，否则 `frontier_proximity` 会退化为常数 0。该问题已在 `offline_data.py` 修复。

`scripts/_nav_s7.sh` 已移除 `field_observations/*.zip` 的 schema/帧数/key 校验；历史离线标签仍可在完整 cell（含四角）上采样 GT mask。

> 2026-10-27 重启三场景实验：`outputs/s6_gdmech_3scene_2026-10-27_restart`（3 × 99）。本次使用单一 `upgrade_min_obs=4`、track-only gd-tsp、gd-only 可被后续 TSP 提升、增量 GD 选择优先；field 原始归档在线采集，`field_analysis` 离线分析已完成，但全量 `region_semantics.jsonl` 尚未生成，不阻塞导航。

## 4. 机制四：middle-tier 提升概率（已移出 policy）

> 状态：**已从 `TSP3DObjectNavPolicy`、Hydra 配置和 `_nav_s7.sh` 移除**。当前不再实例化 ranker、不写 `middle_rank_shadow.jsonl`、不跑在线/自动离线拟合。
>
> 下方内容保留为历史设计记录；如需恢复，应作为独立离线实验重新接入，而不是恢复当前 policy 的隐式依赖。
>
> 本次运行：`outputs/s7_baseline_3scene_2026-10-30_004608`（3 场景 × 99），由 `bash scripts/_nav_s7.sh` 启动；前两轮 `..._003618` / `..._004427` 分别因类别 one-hot 特征 bug、复用服务卡住而中止，归档为 `..._aborted_classfeat` / `..._aborted_servicehang`。
>
> 运行完成：成功 141/297（SR 47.47%），Oracle 46.46%，SPL .2182，steps 157.04；`middle_rank_shadow.jsonl` 16,032 行，`offline_fit` 样本 652（gd-only 600 / 正 48，tsp-confirmed 52 / 正 18），模型已写入 `<run>/middle_tier/middle_tier_model.json`。
>
> 结果文档：[VLVM-S7baseline-exp.md](VLVM-S7baseline-exp.md)、[VLVM-S7m0-exp.md](VLVM-S7m0-exp.md)、[VLVM-S7f0-exp.md](VLVM-S7f0-exp.md)。`false_positive` 与 `never_saw...likely_infeasible` 的详细分解见 baseline 文档 §2.2。

### 4.1 边界

- 三大档五小档：`gd-tsp`；中间档 `tsp-confirmed / gd-only`；低档 `tsp-provisional / stable`。
- `stable` 从导航选择中过滤；`tsp-provisional` 作为 TSP 内部中转站，不作为中间档独立竞争目标。
- STOP 仍只允许 `gd-tsp confirmed`；`gd-tsp` 与 `tsp-provisional` 的升级/merge 规则不变。
- 当前 baseline 排序：`gd-tsp > gd-only > confirmed > provisional`；中间档优先 `gd-only`。
- 后续中间档调整不用简单的标签顺序，改为估计目标后续成为 gd-tsp 的条件概率。

### 4.2 量化目标与特征口径

$$
P(\text{promote to gd-tsp within }K \mid \text{features at selection time})
$$

> 这是离线监督目标，不是在线闭式规则；在线只输出模型估计的 $p$。

**样本与标签**

- 样本单位：中间档目标第一次被选为 nav goal 的时刻 $t_0$。
- `gd-only`：$(t_0,t_0+K]$ 内出现 `upgrade` 事件且 `upgrade_reason=gd_only_tsp_merge`（或事件名 `gd_only_tsp_merge`）→ $y_K=1$。
- `tsp-confirmed`：$(t_0,t_0+K]$ 内出现 `gd_tsp_merge` 事件（或 `upgrade` + `upgrade_reason=gd_tsp`）→ $y_K=1$。
- 主窗口先固定一个 $K$；5 / 10 / 20 / 全程作为敏感性。窗口内 episode 结束且未出现事件 → 截尾样本，不作为负样本。
- 特征只取 $t_0$ 及之前；事件与 GT 只用于标签。

**四组特征**

| 特征组 | gd-only | tsp-confirmed |
| :--- | :--- | :--- |
| 跨模态支持 | TSP candidate/track 的最近 `dist_xy/dist_3d`、`iau`、`containment`/反向 containment、$t_0$ 前 `same_object`/近邻次数；排除 `lifecycle=gd` 轨道；未来 `gd_only_tsp_merge` 不作特征 | GD `match_track`/`match_candidate` 的 `same_object`、`dist/iau/containment`、GD `conf`、`gd_conf_max/gd_conf_last`、`gd_hits`；`gd_hits` 是累计新视角命中，不是严格连续帧 |
| 重复证据 | `gd_hits`、`gd_only_merge` 次数、`gd_conf_max/gd_conf_last` | `num_obs`、TSP merge 次数、`sigma_c`、age |
| 几何/距离 | `dist_robot`、depth、occ、目标类别 | 距离、occ、stable/provisional 历史 |
| 选择上下文 | 当前是否被选、`n_selectable`、距离桶；可加 `selection_tier`、`switch_reason` | 同左 |

若样本单位是目标首次被选，`当前是否被选` 恒为真，无区分度；应改为 `selected_before` / `selection_count`。只有候选级逐步样本保留二值 `selected`。

**四个子目标与合成**

每个特征组先产生子分数/子概率 $s_i=P(y=1\mid x_i)$，可用组内 logistic 或单调归一分；最终在 log-odds 空间合成：

$$
\operatorname{logit}p=\beta_0+\sum_{i\in\{cross,repeat,geo,ctx\}}\beta_i\operatorname{logit}s_i
$$

- $\beta$ 由离线数据拟合，不手工指定。
- 子模型与组合权重先在同批 sample 上拟合，summary 标记 `fit_mode=in_sample`；样本足够后再改为按 scene 分组 + out-of-fold 子分数，避免同一样本同时拟合子模型和权重。
- 缺失值用 missing indicator + 训练集中位数/分位数填充，不使用未来信息。

**排序口径**

- 采用后只替换中间档 `gd-only / tsp-confirmed` 的相对顺序，按 $p$ 降序；`gd-tsp` 仍最高、`provisional` 仍最低，外层档位不变。
- 同 $p$ 回退现有 `sigma_c` / `gd_hits` / 距离桶 / `num_obs` 键。
- $p$ 只做排序，不设阈值，不改变准入、生命周期或 STOP。
- gd-only 与 tsp-confirmed 基率不同；两类子模型输出必须跨类校准后再比较。

**评估**

- 离线：AUC、Brier/NLL、Top-k promotion 命中率、按 scene / target class 分层。
- 在线 shadow：$p$、四组子分数、排名与静态排序的 Top-1/Top-k 差异，只读不改动作。
- 最终：success、FP、steps、STOP 转化；不能只看 AUC 或提升率。

### 4.3 现有数据

目标级创建→gd-tsp：

| run | 类型 | 创建目标数 | 提升为 gd-tsp | 提升率 | 提升中位步数 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| S6gd3 | gd-only | 729 | 131 | 18.0% | 11 |
| S6gd3 | tsp-confirmed | 314 | 127 | 40.4% | 2 |
| S6gd3 | tsp-provisional | 171 | 64 | 37.4% | - |
| S6gd4 | gd-only | 751 | 132 | 17.6% | 13 |
| S6gd4 | tsp-confirmed | 342 | 140 | 40.9% | 2 |
| S6gd4 | tsp-provisional | 192 | 67 | 34.9% | - |

时间窗：

| run | 类型 | 5 步内 | 10 步内 | 20 步内 | 全程 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| S6gd3 | gd-only | 4.3% | 8.5% | 14.3% | 18.0% |
| S6gd3 | confirmed | 27.1% | 33.8% | 37.3% | 40.4% |
| S6gd4 | gd-only | 3.9% | 7.3% | 13.2% | 17.6% |
| S6gd4 | confirmed | 28.7% | 34.0% | 36.1% | 40.9% |

解释边界：

- `gd-only` 的优势是基数大、绝对提升数接近，不是单位目标提升率高；
- `tsp-confirmed` 单位目标提升率约 40%，高于 gd-only 的约 18%；
- `tsp-provisional` 有约 35–37% 的 gd-tsp 提升率，但最终目标只有 4–5 个且 success=0，保留为内部中转；
- 中间档排序应基于条件提升概率，而不是标签顺序。

### 4.4 实现模块与 shadow 契约（shadow 已接入）

**阶段**

1. **特征 shadow**：实现特征组装与只读日志，不加载模型；记录四组原始特征、静态排序结果。
2. **离线拟合**：用后续 merge 事件生成标签，先拟合四组子模型与组合权重（`fit_mode=in_sample`），输出模型；scene 分组/OOF 待样本足够后启用。
3. **概率 shadow**：加载模型，记录 $s_i$、$p$、排名；仍不改变动作。
4. **在线接入**：配置开关打开后，仅替换中间档排序；模型或特征异常回退静态顺序。

**历史实现文件（已不再被 policy 导入）**

- `vlfm/vlvm/semantic_spatial_field/middle_tier/features.py`：四组特征组装与计数；
- `vlfm/vlvm/semantic_spatial_field/middle_tier/ranker.py`：模型加载与 $s_i/p$ 计算；
- `vlfm/vlvm/semantic_spatial_field/middle_tier/shadow.py`：只读 shadow 行生成与排名；
- `vlfm/vlvm/semantic_spatial_field/middle_tier/offline_fit.py`：标签生成、四组子模型/组合权重拟合；
- `vlfm/policy/tsp3d_objectnav_policy.py`：**已移除 shadow 调用与相关配置字段**；
- 输出：当前不再生成 `detections/middle_rank_shadow.jsonl`。

> 当前状态：policy 只保留静态层级选择。`offline_fit` 如手动使用，需自行提供历史 run 的日志：
>
> ```bash
> python -m vlfm.semantic_spatial_field.middle_tier.offline_fit --run-dir outputs/<run> --K 10
> ```

**模块**

| 模块 | 职责 | 落点 |
| :--- | :--- | :--- |
| 特征组装器 | 按 `memory_id × step` 汇总 memory 状态、GD proposal、TSP candidate/track、选择上下文，输出四组特征 | 新增，policy 侧或 `vlfm/policy/utils/` |
| 概率计算 | 加载拟合模型，输出 $s_i$、$p$ | 新增轻量模块 |
| shadow logger | 历史只读 `middle_rank_shadow.jsonl` | 已移出 policy |
| 离线拟合 | 标签生成、子模型/权重/校准（首版 in-sample） | 新增离线脚本 |
| 在线排序接入 | 在 `_get_target_object_location` 中间档 comparator 使用 $p$；开关与回退 | `vlfm/policy/tsp3d_objectnav_policy.py` |

**Shadow 字段（至少）**：`episode_id`、`step`、`memory_id`、`target_class`、`centroid`、`type`、`state`、`selected`、`n_selectable`、`distance_bucket`、四组特征、`s_cross/s_repeat/s_geo/s_ctx`、`p`、`rank`、`static_rank`、`model_version`。`centroid` 用于 episode 级 `tp` 标注。

**边界**

- 不改 `gd-tsp`、`tsp-provisional`、`stable`、准入、生命周期或 STOP。
- 无模型或特征异常时回退当前静态顺序。
- 第一版只做 shadow 与离线拟合；S7 baseline 由 `bash scripts/_nav_s7.sh` 启动。


