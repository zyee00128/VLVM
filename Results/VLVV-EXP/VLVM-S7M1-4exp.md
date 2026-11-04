# VLVM-S7 M4：真实每层 frontier、floor transition 与 gd-only 门控结果

> 运行目录：`outputs/s7_floor2_2026-11-01_154307`
> 启动脚本：`scripts/_nav_s7.sh`
> 运行时间：2026-11-01 15:43 至 2026-11-02 04:25
> 场景：`5cdEh9F2hJL` / `mL8ThkuaVTM` / `QaLdnwvtxbs`，`3 × 99 = 297` episodes
> 对照：S7 baseline、M1、M2、M3/Gate2
>
> 本文整理两件事：
> 1. 真实 3D occupancy / 每层 2D frontier / transition node 的在线表现；
> 2. gd-only 几何门、时效门和 blacklist 状态门的实际数据。
>
> **后续变更（2026-11-02）：** 几何门和时效门代码均已删除。本文 §5.4/§5.5 记录的是删除前实验中的门控行为；当前代码只保留 floor 机制和 blacklist 状态门。所有独立证据门已弃用。本文运行本身仍是 floor shadow-only；之后代码已加入当前层 frontier 耗尽后的 entry goal 动作接入。
>
> **entry 重构（2026-11-02）：** 删除 `TransitionNode`，改为 `FloorEntry`；`entry_xyz + exit_xy` 记录连接，新增 `CROSS_FLOOR_CANDIDATE` / `FLOOR_ENTRANCE` / `STAIR`、垂直连通性验证、真实楼层面积判定和独立 `entry_blacklist`。本文中的 `transition_node_count` 是旧口径，对应新日志的 `entry_count`。

## 1. 摘要

- 导航结果：**137 / 297，SR 46.13%**，没有超过 S7 baseline 的 141 / 297。
- floor 机制已经在真实运行中产生多层和 transition node：
  - 最大 `floor_count=5`
  - 15 次楼层切换
  - 最大 `transition_node_count=31`
  - 211 个日志帧出现 transition node
- 但 floor 机制仍是 **shadow-only**，没有接入动作；`would_floor_transition` 仍为 0。
- **几何门基本在误杀 TP**：
  - 58 个被几何门阻止的候选事件中，57 个是 TP；
  - 唯一 8 个 memory track 中 7 个是 TP；
  - 全部失败原因是 `geom_unknown` + `out_of_bounds`，而 `depth_ok=True`、`occ_passed=True`。
- **时效门有微弱方向性收益，但证据不足**：
  - 122 个事件中只有 4 个 TP；
  - 唯一 6 个 track 中 2 TP / 4 非 TP；
  - 当前日志没有 `age_steps`、`gd_last_evidence_step` 和阈值，不能确认阈值 20 是否有效。
- **两者都没有带来 SR 提升**。由于门只影响 gd-only 到 gd-tsp 的升级，gd-only 本身仍可被选择，门对动作的影响天然有限。

## 2. 运行口径

| 项 | 值 |
| :--- | :--- |
| policy | `HabitatITMPolicyV1` |
| TSP3D checkpoint | `data/vlvm_v1/vlvm_v1.pth` |
| `sigma_tar` / `sigma_prop` | `0.60` / `0.45` |
| `provisional_sigma` | `0.55` |
| `upgrade_min_obs` | `4` |
| 目标选择 | `gd-tsp > gd-only > confirmed > provisional` |
| STOP | 仅 `gd-tsp confirmed`；`pointnav_stop_radius=0.90 m` |
| gd-only 深度要求 | `depth_ok=True` |
| gd-only 升级门 | 几何/状态 + 时效 `age<=20` |
| 主导航地图 | `ProbabilisticGrid`，固定 `camera_height=0.88` |
| floor 地图 | 独立 `ObstacleMap3D`，实际相机高度 |
| floor 动作接入 | 否，read-only shadow |

## 3. 导航结果

### 3.1 总体对比

| 指标 | S7 baseline | M1 | M2 | M3 / Gate2 | **M4（本文）** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| episodes | 297 | 297 | 297 | 297 | **297** |
| success | 141 | 142 | 137 | 142 | **137** |
| SR | 47.47% | 47.81% | 46.13% | 47.81% | **46.13%** |
| Oracle success | 138 | 138 | 138 | 138 | **138** |
| Oracle SR | 46.46% | 46.46% | 46.46% | 46.46% | **46.46%** |
| SPL | 0.2182 | 0.2207 | 0.2118 | 0.2202 | **0.2153** |
| Soft SPL | 0.2852 | 0.2838 | 0.2801 | 0.2828 | **0.2798** |
| steps mean | 157.04 | 265.87 | 268.19 | 266.23 | **271.77** |

> M4 的 step mean 与 M1–M3 接近，均明显高于旧 S7 baseline；不能把差异单独归因于 floor map。S7 baseline 与 M1–M4 的实现口径并不完全相同，当前表只能作为趋势参照，不是严格 A/B。

### 3.2 分场景

| 场景 | episodes | success | SR | Oracle | SPL | steps mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 99 | 49 | 49.49% | 62.63% | 0.2188 | 228.98 |
| `QaLdnwvtxbs` | 99 | 79 | 79.80% | 72.73% | 0.3647 | 167.28 |
| `mL8ThkuaVTM` | 99 | 9 | 9.09% | 4.04% | 0.0624 | 419.06 |

与 M3 相比：

- `5cd`：52 → 49
- `QaL`：81 → 79
- `mL8`：9 → 9

mL8 没有变化；下降主要集中在 5cd/QaL。

### 3.3 失败原因

| failure_cause | 数量 |
| :--- | ---: |
| success | 137 |
| `false_positive` | 109 |
| `never_saw_target_did_not_travel_stairs_likely_infeasible` | 43 |
| `never_saw_target_did_not_travel_stairs_feasible` | 3 |
| `bad_stop_true_positive` | 2 |
| `timeout_true_positive` | 2 |
| `false_negative` | 1 |

## 4. Floor 机制

### 4.1 实现

- 主地图仍是 `ProbabilisticGrid`，使用固定相机高度，服务导航、occupancy consistency 和 GD。
- floor 地图是独立的 `ObstacleMap3D`：
  - 使用实际 episode-relative 相机高度校正后的点云；
  - 维护自己的 3D occupancy；
  - 按楼层 `z_min/z_max` 切片，得到每层 2D `occupied/known`;
  - 从每层 free-unknown 边界检测 2D frontier；
  - frontier z 取所属楼层的 `z_ref`。
- 旧的 `frontier_sensor_3d` 已删除；它按当前 agent 高度 snap 2D frontier，所有 frontier z 相同，不能表达楼层。
- `FloorManager` 使用每层 frontier 发现跨层候选，记录 `TransitionNode`。
- 目前没有 floor 动作接入，floor 结果只进入 `floor_shadow` / `floor_manager` 日志。

### 4.2 在线统计

| 指标 | 值 |
| :--- | ---: |
| `floor_manager` 日志行数 | 80,717 |
| 最大 `floor_count` | 5 |
| 最大 `floor_map_count` | 5 |
| 楼层切换次数 | 15 |
| 最大 `transition_node_count` | 31 |
| 出现 transition node 的帧数 | 211 |
| 最大 `all_frontier_3d_count` | 17 |
| `floor_shadow` 最大 `floor_proxy_count` | 2 |
| `floor_shadow` 最大 `cross_floor_frontier_count` | 3 |
| `would_floor_transition=True` 帧数 | 0 |
| `frontier_z_span` 最大 | 1.3098 m |
| `frontier_z_span>0` 帧数 | 302 |
| frontier z 范围 | -1.31 至 0.50 m |

### 4.3 解读

- 真实 3D occupancy + 每层 2D frontier 已经能产生多层和 transition node，不再是 M3 的“全 z 相同、0 个节点”。
- `transition_node_count` 是“跨层 frontier candidate”，不等于已确认楼梯入口；当前 `node_type` 应视为 `CROSS_FLOOR_CANDIDATE`，没有几何或实际通行验证。
- `would_floor_transition` 仍为 0；floor manager 的 action hook 只在当前层 frontier 耗尽后被调用，不能把这个旧字段当作“没有动作”。
- floor 机制没有影响 SR；M4 的 137/297 不能证明 floor 机制有导航收益，只能证明 shadow 输入和跨层检测链路已工作。

### 4.4 transition 节点的两类语义

| 类型 | 定义 | 当前代码状态 |
| :--- | :--- | :--- |
| **楼层入口 / floor entrance** | 经过几何连通性、可达性或实际通行验证的楼梯/坡道/开口；可作为跨层路由入口 | 当前没有生成或验证，数量为 0 |
| **跨层 candidate** | 当前楼层之外的 2D occupancy 中，已知 free space 与 unknown 的边界；只说明该处可能有其他楼层空间 | 当前 31 个节点全部属于此类 |

当前 31 个节点的实际特征：

- `source` 全部是 `per_floor_2d_frontier`；
- `status` 全部是 `CANDIDATE`；
- `node_type` 当前语义是 `CROSS_FLOOR_CANDIDATE`；
- 多个 `from_floor` 会指向同一个 `exit_xy`，说明同一其他楼层 frontier 可从多个楼层/视角看到；
- `delta_z` 有成组的近似值（约 0.85、0.91、1.31、1.77 m），但这只是 frontier 高度差，不是楼梯坡度或可通行高度；
- 节点没有 `entry_xy` 到 `exit_xy` 的 3D 连通性验证。

工作流当前是：

1. 用实际相机高度更新独立 3D occupancy；
2. 按楼层 z 切片，生成每层 2D occupied/known；
3. 从 free-unknown 边界检测每层 2D frontier；
4. 把其他楼层的 frontier 作为 `CROSS_FLOOR_CANDIDATE` 写入 `TransitionNode`；
5. 当前层 frontier 耗尽时，policy 选择最近的 candidate `exit_xy` 做 PointNav；
6. 没有 candidate 时才回到转向/最后 frontier fallback。

主要问题：

1. **candidate 与真正入口没有分离**：31 个节点全部是 candidate，没有 `FLOOR_ENTRANCE` 验证。
2. **exit_xy 不等于楼梯入口**：它是其他楼层的 2D free-unknown 边界；PointNav 到该 xy 可能走到死路或不可达区域。
3. **没有垂直连通性验证**：缺少 entry→exit 的 3D free-space/楼梯几何检查。
4. **节点可重复且跨层重复**：同一 `exit_xy` 被多个 `from_floor` 记录，`transition_node_count` 会高估独立入口数。
5. **楼层 ID 可能包含楼梯中间层**：当前 `floor_count` 最大 5，但场景语义楼层不一定是 5；高度阈值/楼梯过程会生成中间 floor。
6. **没有 transition 成功/失败状态机**：节点不会因实际尝试失败而黑名单；当前只有 candidate 状态。
7. **动作目标未做可达性预检**：`_explore_safe()` 直接对 `exit_xy` 调 PointNav，失败后没有节点级回退。
8. **日志口径容易误读**：`transition_node_count` 是 episode 内累计节点数，不是当前帧可见入口数。

## 5. gd-only 门控分析

### 5.1 门的语义

门只在以下条件成立时检查：

- 一个已存在的 `gd-only` 记忆；
- 当前候选成功 merge；
- 候选不是纯 GD evidence；
- 候选生命周期是 confirmed 或 provisional。

通过后才会把 `gd-only` 提升为 `gd-tsp`。  
如果门失败，记忆仍保留为 `gd-only`，而 **gd-only 仍然可以参与目标选择**。因此门主要影响：

- 是否能获得 gd-tsp 的 STOP 资格；
- 是否能进入更强的确认链路；
- 不会直接删除 gd-only 候选。

### 5.2 候选级统计

`candidates.jsonl` 共 8,031 条候选。

| `gd_only_promotion_gate` | 候选事件 | TP | 非 TP |
| :--- | ---: | ---: | ---: |
| `None`（未触发/通过） | 7,753 | 1,810 | 5,943 |
| `temporal` | 122 | 4 | 118 |
| `blacklist` | 98 | 85 | 13 |
| `geom_state` | 58 | 57 | 1 |

> 候选事件会跨帧重复，不能把行数当作独立目标数。

### 5.3 Track 级统计

按 `scene + episode_id + memory_id` 去重后：

| 门原因 | 唯一 track | TP track | 非 TP track |
| :--- | ---: | ---: | ---: |
| `temporal` | 6 | 2 | 4 |
| `geom_state` | 8 | 7 | 1 |
| `blacklist` | 34 | 29 | 5 |

### 5.4 几何门

`geom_state` 的 58 条候选全部满足：

- `depth_ok=True`
- `occ_passed=True`
- `geom_status="geom_unknown"`
- `occ_reason="out_of_bounds"`
- `occ_rejected_at=None`

也就是说，几何门不是因为目标框与 occupancy 冲突而拒绝，而是因为 **目标落在 3D occupancy 地图边界外**：

- 候选级：57/58 是 TP；
- track 级：7/8 是 TP。

结论：**当前几何门是明显的误杀路径。** 它主要阻止正确的 gd-only 升级，而不是过滤错误目标。至少在 `depth_ok=True`、`occ_passed=True` 时，不应仅凭 `geom_unknown/out_of_bounds` 做硬拒绝；应改为软证据或直接移除该门。

### 5.5 时效门

`temporal` 的 122 条候选全部满足几何通过：

- `depth_ok=True`
- `occ_passed=True`
- `geom_status="geom_confirmed"`
- `occ_status="HARD_CONFIRMED"`

失效原因只有年龄超过 `GD_ONLY_PROMOTION_MAX_AGE_STEPS=20`。

数据：

- 候选级：4/122 TP（约 3.3%）
- track 级：2/6 TP，4/6 非 TP

方向上，时效门主要挡住较老的证据，确实偏向过滤负例；但：

- 唯一 track 只有 6 个，样本太小；
- 没有记录 `age_steps`、`gd_last_evidence_step`、`gd_last_step`；
- 不能判断阈值 20 是否合理；
- 门只影响升级，gd-only 仍可选择，因此对 SR 的作用有限。

结论：**时效门可能有小幅精度收益，但当前数据不足以证明有效，也不能解释整体 SR 下降。**

### 5.6 blacklist 状态门（附带观察）

`blacklist` 不是几何门或时效门，而是 STOP blacklist 状态门：

- 98 条候选；
- 34 个唯一 track；
- 29 个 TP / 5 个非 TP。

它同样在阻止大量 TP，说明当前 STOP blacklist 与 gd-only 升级之间存在较强耦合，后续需要单独分析，不能把它当作“错误目标过滤”的收益。

### 5.7 成功升级路径

`memory_events.jsonl` 中的 upgrade：

| `upgrade_reason` | 数量 | TP | 非 TP |
| :--- | ---: | ---: | ---: |
| `base_upgrade` | 41 | 17 | 24 |
| `gd_only_tsp_merge` | 93 | 71 | 22 |
| `gd_tsp` | 62 | 37 | 25 |

`gd_only_tsp_merge` 的 TP 率为 71/93 ≈ 76.3%，但被门阻止的 track 不会出现在升级日志中，因此不能用这 93 条直接评价门的效果。

## 6. 机制分析

### 6.1 为什么两个门对 SR 影响弱

1. **门只作用于 gd-only → gd-tsp 的升级**
   - gd-only 本身就是可选目标；
   - 门失败后目标仍可能被选中；
   - 主要变化是 STOP 资格和后续确认路径，而不是“目标是否被选”。

2. **几何门拒绝的是 `out_of_bounds`**
   - 这不是目标框与 occupied voxel 的矛盾；
   - 地图外/边界外并不等于目标错误；
   - 结果是大量 TP 被阻止升级。

3. **时效门只影响旧 GD 证据**
   - 当前样本只有 6 个唯一 track；
   - 失效原因没有 age 数值，无法做阈值敏感性分析；
   - 对动作链路的贡献天然有限。

4. **floor 机制仍是 shadow-only**
   - transition node 已产生，但没有接入 nav goal；
   - 因此 floor 发现不会改变 SR；
   - 当前 137/297 不能评价 floor 动作收益。

### 6.2 结果是否支持删除门

- **几何门：数据支持删除或放宽。**
  - 7/8 被阻止 track 是 TP；
  - 失败全部是 `geom_unknown/out_of_bounds`；
  - 当前硬门没有过滤 FP 的证据。

- **时效门：暂不支持直接判定无效，但也不支持保留。**
  - 方向是过滤旧证据/负例；
  - 样本太小；
  - 缺少 age 日志；
  - 建议先补日志做离线分析，再决定阈值或删除。

- **blacklist：单独处理。**
  - 29/34 被阻止 track 是 TP；
  - 不是几何/时效门，但同样存在误杀风险。

## 7. 产物与局限

### 7.1 产物

- `detections/episodes.jsonl`
- `detections/candidates.jsonl`
- `detections/memory_events.jsonl`
- `detections/floor_shadow.jsonl`
- `detections/floor_manager.jsonl`
- `s7_summary.json`
- `COMPLETED`

### 7.2 局限

- 当前对照不是严格 A/B；S7 baseline、M1–M3 与 M4 的实现和运行时长不同。
- `candidates.jsonl` 的 TP 是候选级标签；同一 memory track 可跨多帧出现，本文同时给出 track 级统计。
- `gd_only_promotion_gate` 只记录失败原因，未记录 `age_steps` 和证据步数。
- `transition_node_count` 是跨层 frontier candidate，不是已验证楼梯入口。
- floor manager 没有动作接入，`would_floor_transition=0` 不能解释为“没有跨层机会”。

## 8. 结论

1. 真实 per-floor occupancy/frontier 已经在线工作：5 层、15 次切换、31 个 transition node、302 个非零 z span 帧。
2. floor 机制仍是 shadow-only，当前不改变导航动作，因此不能从 137/297 评价其动作收益。
3. 几何门数据上明显负向：主要阻止 TP，且失败原因是 `out_of_bounds`，建议删除或放宽。
- 时效门有微弱的 FP 过滤倾向，但只有 6 个唯一 track，且缺少 age 日志，不能证明有效；该门已在后续代码中删除。
5. 当前整体结果没有超过 S7 baseline/M1/M3；两门没有带来可观测的 SR 收益。
