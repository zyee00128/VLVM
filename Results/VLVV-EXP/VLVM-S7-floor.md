# Project VLVM S7：Floor Manager 机制

> 目标：为多层场景维护每层独立的 occupancy/frontier 状态，显式记录楼层入口 entry，并区分“当前层探索耗尽”和“全局探索耗尽”。
>
> **铁律：跨层搜索不能影响任何同层搜索行为。**

## 0. 2026-11-03 当前设计

> 本节是当前目标设计；下文 §1–§12 为历史记录。`exit_xy`、`CROSS_FLOOR_CANDIDATE`、`FLOOR_ENTRANCE`、`STAIR`、垂直连通性采样、other-floor frontier probe 均已废弃。
>
> 当前 policy 侧：`act()` 每步只做被动 `_update_floor_state`；`_accumulate_floor_entries` 只在 explore 路径运行；只有同层 explorer 自己 STOP 后才进入 `_floor_control`。FloorManager、`FloorRecord`、`FloorEntry`、`CurrentFloorState`、每层地图构建与传感器保持不变。

### 0.0 铁律与边界（最高优先级）

- **同层搜索是默认路径；跨层搜索只是同层搜索完全结束后的扩展。**
- 同层探索算法保持 baseline：`_explore()` / `_explore_safe()` / stuck-fallback 逻辑不变。
- **frontier 输入按层隔离**：当前 floor map 有 frontier 时，同层探索使用该层 frontier；per-floor frontier 为空（初始化/瞬态）时，回退 Habitat 原始 frontier 作为 bootstrap，绝不把空数组交给 `_explore()`。
- 不允许把其他 floor 的 frontier 混入当前层；`_current_floor_frontiers_2d()` 只返回当前 `current_floor_id` 的 frontier。
- 不允许 coverage/frontier 预抢占同层探索；`_floor_control` 只能在 `_floor_handover=True` 时接管动作。
- 只有同层 `_explore()` 自己返回 STOP，或 stuck-fallback 重试也 STOP 后，才允许设置 `_floor_handover=True`。
- 无可用 entry 时不得提前 STOP；必须取消 handover、恢复同层 fallback，由正常 episode 上限决定结束。
- 楼层观察、entry 累积、floor 点云准备必须异常隔离；失败只能打印 `[FLOOR] ...`，不能中断同层导航。
- 只有已经完成的跨层 traversal 才可以重置同层探索状态；被动高度变化、level 变化不得重置。

### 0.1 术语

| 术语 | 含义 |
| :--- | :--- |
| mem-life | 目标记忆生命周期的可选状态：`gd-only` / `confirmed` / `provisional`；不再称 fallback |
| frontier fallback | 仅指同层探索中机器人卡在同一个位置、frontier 无法推进时的处理 |
| floor handover | 当前层探索结束后进入 entry 流程 |
| entry | 当前层 TSP3D `stairs` 检测到的楼梯入口；不是 exit，不建跨层边 |

### 0.2 总原则

- 同层探索与跨层探索分离：2D frontier 只负责当前层；floor manager 只负责层状态、entry 和换层。
- 只处理最常见情况：同层探索完全耗尽 → 最近 entry → 验证 → 实际高度变化 → 下一层继续探索。
- 不做其他楼层 frontier probe，不做 exit/STAIR 节点，不做垂直连通性采样。
- `level` 与 `true floor` 仅用于日志/统计：`level` 是按实际 `floor_z` 记录的当前高度层，包含楼梯平台；`true floor` 是旧的晋升标记。跨层动作只使用当前 level，不等待 true-floor 晋升，也不受 `min_new_floor_gap` 约束。

### 0.3 mem-life 状态与 STOP

| 状态 | 含义 | 可选为目标 | 可 STOP |
| :--- | :--- | :--- | :--- |
| `gd-only` | GD 单源，未获 TSP3D 确认 | 是 | 否 |
| `confirmed` | TSP3D 确认 | 是 | 仅 `origin=gd-tsp` 或 `promoted_from_gd_only=True` |
| `provisional` | 暂定 | 是 | 否 |
| `stable` | 删除池 | 否 | 否 |

- 选择顺序仍为：`gd-tsp > gd-only > confirmed > provisional`。
- `stable` 不参与选择、不升级。
- 三种可选状态统一称 mem-life，不再称 fallback。

### 0.4 同层探索与 frontier fallback

- **跨层隔离原则（2026-11-03）**：跨层动作不得改变同层探索算法；同层 frontier 输入按 floor 隔离。
  - 同层仍走 baseline `_explore()` / `_explore_safe()`，不改算法、不改 fallback 条件。
  - `_current_floor_frontiers_2d()` 优先提供当前 `current_floor_id` 的 per-floor frontier。
  - per-floor frontier 为空（冷启动/瞬态空图）时保留 Habitat 原始 `frontier_sensor` 做 bootstrap，避免把空 frontier 当成 STOP。
  - 不允许混入其他 floor 的 frontier；只有已完成的跨层 traversal 才切换 current floor。
  - 只有同层 explorer 自己返回 STOP（或其 stuck-fallback retry 也 STOP）后，才允许 `_floor_handover=True`。
  - `_floor_control` 不再通过 coverage/frontier 预抢占同层探索。
  - 楼层观察只写日志/floor manager，不重置 explore blacklist、stuck、last frontier 等状态；只有已完成的跨层 traversal 才重置。
  - floor 相关计算全部异常隔离；失败只打印 `[FLOOR] ...`，不得中断同层导航。
  - 无 entry 时不再提前 STOP，而是恢复同层 fallback，由正常 episode 上限决定结束。
- 同层仍使用 2D frontier explorer；算法不变，输入 frontier 来自当前层。
- `fallback` 只保留一个触发条件：机器人在 pursue frontier 时，连续 `frontier_stuck_steps=15` 步 XY 位移 ≤ `frontier_stuck_move_m=0.05 m`。
- 不再用 action 重复、BLIP cosine 不变、同一 frontier 连续选择次数作为 fallback 条件；这些只保留日志。
- 触发后：
  1. 把当前 frontier 加入 `_explore_frontier_blacklist`；
  2. 清空 `_last_frontier` 与卡死计数；
  3. 过滤 blacklist 后重跑同层 explorer；
  4. 若 explorer 返回非 STOP，继续同层探索；
  5. 若没有可用 frontier，进入 floor handover。
- fallback 不直接 STOP；无可用 entry 时也只取消 handover 并恢复同层 fallback，不提前结束 episode。

#### 0.4.1 per-floor frontier 规则

- `PerFloorFrontierMap` 继续按 floor band 维护独立 2D frontier。
- 同层探索入口统一调用 `_current_floor_frontiers_2d()`：
  - 当前 floor map 可用且 frontier 非空 → 使用该层 frontier；
  - 当前 floor frontier 为空 → 保留 Habitat `frontier_sensor` 作为 bootstrap；
  - 不允许读取其他 floor 的 frontier。
- 这样同层仍是 baseline explorer，只是 frontier 输入按层隔离，不会把 1 楼 frontier 用在 2 楼。
- 跨层成功后 current floor 切换，下一次 `_explore()` 自动读取新层 frontier；新层 frontier 仍为空时先走 Habitat bootstrap，等该层地图积累后再切换。
- per-floor frontier 的计算失败只降级为 raw bootstrap，不影响同层导航。

### 0.5 Floor manager 不变部分

- `FloorRecord`：每层 `floor_id`、`z_ref`、`z_min/z_max`、status、coverage、frontier、occupancy、entry_ids。
- `FloorEntry`：`entry_id`、`floor_id`、`entry_xy`、`entry_dz`（可选测量）、status、confidence、attempt/progress 字段。
- `CurrentFloorState`：当前/上一 level、`floor_z`、切换原因、最近切换步。
- per-floor 3D occupancy / 2D frontier 构建与 `FloorCameraHeight` 传感器不变。
- 不再使用 `exit_xy`、`CROSS_FLOOR_CANDIDATE`、`FLOOR_ENTRANCE`、`STAIR`、垂直连通性采样。

### 0.6 简化跨层状态机

| 状态 | 进入条件 | 行为 | 退出条件 |
| :--- | :--- | :--- | :--- |
| `EXPLORE_CURRENT` | episode 开始或换层完成 | 2D frontier 探索；每步累计 entry | 当前层无目标判定触发 |
| `FLOOR_DONE` | 同层探索耗尽 | 选择最近可用 entry | 有 entry → `APPROACH_ENTRY`；无 entry → `NO_ENTRY` |
| `APPROACH_ENTRY` | 选中 entry | PointNav 到 `entry_xy` | 距离 ≤ `entry_arrival_radius=0.90 m` |
| `VERIFY_ENTRY` | 到达 entry | 同时查询 TSP3D 与 GD；任一通过即 verified | 通过 → `TRAVERSE_ENTRY`；都失败 → blacklist，回 `FLOOR_DONE` |
| `TRAVERSE_ENTRY` | entry verified | 转向 + 直行爬楼；不标记 TRAVERSED | 实际高度变化 ≥1.00 m → `SWITCH_FLOOR`；超时 → blacklist |
| `SWITCH_FLOOR` | 高度变化确认且已完成 cross-floor traversal | 更新当前 level，重置同层探索状态 | 回到 `EXPLORE_CURRENT` |
| `NO_ENTRY` | 同层 explorer 已 STOP 且无可用 entry | 立即恢复同层 fallback，不扫描、不 STOP | 新 entry 出现或 episode 上限 |

### 0.7 Entry 条件与累计

- 来源：当前层 floor-only 点云上的 TSP3D `stairs` 检测。
- 创建条件（全部满足）：
  - `stairs` box 存在；
  - confidence ≥ `entry_conf_min=0.30`；
  - box 垂直跨度 ≥ `entry_min_dz=0.20 m`；
  - `entry_xy` 位于当前层 known free space；
  - 与机器人距离 ≤ `entry_max_dist_m=6.0 m`。
- 同一楼梯 3D 聚类/NMS 后生成一个 entry；同层 `entry_dedup_xy_m=0.50 m` 内去重。
- 不设每层 entry 数量上限：只按 0.50 m XY 去重，保留不同楼梯的 entry。
- 累计方式：`EXPLORE_CURRENT` 每步运行一次 `stairs` 查询，更新已有 entry 的 `last_seen_step/confidence`；不因暂时看不见而删除。
- 状态：`CANDIDATE` → `VERIFIED`；失败进独立 `entry_blacklist`。不再写 `TRAVERSED`；验证通过的 entry 可以重复使用。
- `entry_dz` 在 arrival 后测量并记录，不作为创建/验证门槛。
- 不存在 `at_entry_timeout`：到达 entry 附近只累计诊断计数，不因停留步数拉黑。

### 0.8 当前层“没有目标”的判定

不做 coverage / frontier 预判，也不使用当前层累计步数。唯一入口是：

- 同层 `_explore()` 自己返回 STOP；或
- stuck-fallback 已把失败 frontier 拉黑，重试 `_explore()` 仍返回 STOP。

只有这时才设置 `_floor_handover=True` 并进入 `FLOOR_DONE`。同层 explorer 仍返回动作时，floor 代码不得接管。

仅在 explore 模式生效；若已有目标记忆，`act()` 进入 navigate，floor handover 不介入。

### 0.9 Entry 可用性与验证

- 候选可用：status 不为 `BLACKLISTED`，属于当前 floor，`entry_xy` 有限。
- 选择：按 `entry_xy` 与机器人 XY 距离取最近。
- 到达后同时查询：
  - TSP3D：当前层 floor-only 点云，text=`stairs`；
  - GD：当前 RGB，text=`stairs`。
- 只要 TSP3D 或 GD 任一 confidence ≥ `entry_verify_conf_min=0.30` → `VERIFIED`；两者都失败才 `BLACKLISTED`。
- GD-only 验证没有 `entry_dz` 时也允许进入 `TRAVERSE_ENTRY`；`traverse_xy` 缺省用 entry + robot yaw 向前 1.5 m。
- 不存在 `at_entry_timeout`；到达后停留不会拉黑 entry。

### 0.10 进入下一层与初始化

- `TRAVERSE_ENTRY`：验证通过后不使用 PointNav。先按 `traverse_xy` 计算航向误差，`|theta|>0.35 rad` 时左/右转，否则持续 `MOVE_FORWARD`；PointNav 只用于接近 entry。原因是 PointNav 的 2D 目标会让机器人停在楼梯底部，实测 4 个 VERIFIED entry 都停在 `traverse_xy` 附近且高度不变。
- 换层判定：`abs(floor_z - 当前层 z_ref) ≥ floor_switch_delta_z=1.00 m`；向上、向下都有效。
- `floor_z` 本身保持有符号的 episode-relative 高度，不能取绝对值；否则上层与下层的 `z_ref` 会混淆。判定只对差值取绝对值。
- 1.00 m 的来源与依据：
  - 一段楼梯会产生多个 0.3–0.5 m 的 level；0.30 m 阈值会让 current level 在楼梯上每 8 步左右翻转一次；
  - 本轮 198 episodes 中单个 episode 出现 z=1.434↔1.836 的 46 次上下切换，说明 0.30 m 太低；
  - 1.00 m 高于单段踏步/休息平台，低于层高 2.7 m，既能避免台阶抖动，又能在进入下一层前切换；
  - `entry_min_dz=0.20 m` 是 stair box 的垂直跨度，衡量“这是楼梯体积”；1.00 m 是机器人实际 `floor_z` 变化，衡量“已经换层”，两者不是同一口径。
- 换层成功：
  - 不标记 `TRAVERSED`；只释放 `active_entry_id`。
  - `CurrentFloorState` 更新 current/previous level 与切换步；
  - 只有已完成的 cross-floor traversal 才清空当前层探索状态；被动 level 变化不重置；
  - 保留全局状态：目标记忆、mem-life、`_stop_blacklist`、`entry_blacklist`、episode 步数、PointNav；
  - 重新使用当前 level 作为新的探索层，进入 `EXPLORE_CURRENT`。
- 超时：`entry_traverse_timeout_steps=60` 内无高度变化 → blacklist entry，回 `FLOOR_DONE`。
- 新层初始化不重置 episode，不重建地图历史；只重置同层探索状态。
- `min_new_floor_gap_m` 只影响日志里的 `floor_count`/`is_true_floor`，不影响跨层动作；不再用 true-floor 晋升作为换层前提。

### 0.11 Floor fallback 与终止

- 当前层无可用 entry → `NO_ENTRY`：立即取消 handover，恢复未修改的同层 fallback；不原地扫描、不提前 STOP。
- 若仍无 entry，episode 只受正常 500 步上限约束；新 entry 出现时清除 failed 标记，允许再次 handover。
- entry 验证/穿越失败 → blacklist 并尝试下一个最近 entry；全部 blacklisted 后进入 `NO_ENTRY`。
- 不使用 other-floor frontier probe；不因为“还有别的层”而继续导航到不可达点。

### 0.12 日志与指标

- `floor_manager.jsonl`：每步 `floor_records`、`current_floor_id`、`frontier` 数、entry 状态与 blacklist。
- mem-life 事件：`new/merge/upgrade/delete`。
- frontier fallback 事件：触发步、frontier XY、重试结果。
- entry 事件：`detected/verified/blacklisted/traversed`。
- 评估保留 `traveled_stairs` 与 `cumulative_elevation_change`。

### 0.13 与旧版差异

- 删除：`exit_xy`、`CROSS_FLOOR_CANDIDATE`、`FLOOR_ENTRANCE`、`STAIR`、垂直连通性采样、other-floor frontier probe。
- 删除：action 重复 / cosine 不变 / frontier 重复选择作为 fallback 触发条件。
- 保留：FloorManager、`FloorRecord`、`FloorEntry`、`CurrentFloorState`、每层地图与传感器、同层 2D frontier。
- blacklist 保持硬门：同时阻断 gd-only 晋升与目标选择；不做任何 release。

### 0.14 2026-11-03 在线结果

运行：`outputs/s7_floor2_2scene_2026-11-03_005206`，2 场景 × 99 episodes。

| 指标 | 数值 |
| :--- | ---: |
| 总成功 | 29 / 198（14.65%） |
| 5cdEh9F2hJL | 24 / 99（24.24%） |
| mL8ThkuaVTM | 5 / 99（5.05%） |
| traveled_stairs episodes | 4（均失败） |
| `floor_manager.jsonl` rows | 39,842 |
| 最大 `floor_count` / `level_count` | 2 / 5 |
| `FLOOR-SWITCH` / `FLOOR-PROMOTE` | 146 / 1 |
| `FLOOR-ENTRY` detections / verify true | 80 / 4 |
| `FLOOR] approach entry` | 1,079 |

结论：机制已真实接入并产生动作（1079 次 approach、22 次验证），但 entry handover 仍未成功爬楼；4 个 verified entry 都停在 `traverse_xy` 后 `at_entry_timeout`。实际发生的 4 次 stair travel 全部来自普通 frontier 探索，不是 handover。另有一个 episode 在 1.434/1.836 m 之间往返 46 次，暴露 0.30 m 换层阈值与每层切换重置探索状态的问题。

已按本轮证据修正：换层阈值 1.00 m、true-floor 最小间距 2.00 m、仅 true-floor 切换重置探索状态、verified entry 改为转向+直行、fallback traverse 距离 0.8→1.5 m。

> 下文 §1–§12 为旧版设计与历史实验记录，仅作归档；与本节冲突处以本节为准。


## 1. 设计边界

- 不再使用一张覆盖全局高度的 3D map。
- 每层维护独立的 occupancy map，z 范围只覆盖该层高度。
- 楼层高度、层间高差、总高度范围由 floor manager 维护，不塞进单层 map。
- 同层探索继续以 **2D frontier** 为主；同层目标候选继续使用现有 STOP blacklist。
- 3D frontier 只用于发现/验证层间 entry，不直接替代同层 2D frontier。
- entry 只由 TSP3D `stairs` 检测创建：记录当前层的 `entry_xy`、`floor_id` 和初始 `entry_dz=0`，没有 exit、没有 STAIR/其他节点类型；一层允许多个 entry。
- 其他楼层的 frontier 只用于楼层候选/probe，不创建 entry；probe 本身不是 entry，只有 probe 附近再次检测到 stairs 才会生成 entry。
- entry 在到达后才计算 `entry_dz` 并同时做几何与 TSP3D 语义检查；它的唯一作用是切换楼层，不参与同层目标选择。
- `floor_count` 只统计被 promotion 的 true floor；`level_count` 统计所有内部 level。
- true floor promotion 条件为或关系：新的 candidate frontier（z 间距 + XY 新颖性），或 verified entry 造成的实际稳定高度变化。
- 当前 agent 所在 level 可以按实际高度变化切换，但只有 promotion 后对应的 level 才进入 `floor_count`。
- floor 动作优先在当前层 frontier 耗尽后的 explore fallback 中生效；不替代同层 frontier，也不改变 STOP。
- 为避免 frontier 残留/不可达导致永远 ACTIVE，explore 会在以下任一条件强制进入 floor fallback：当前 true floor 覆盖率连续 `floor_coverage_hold_steps` 步 >= `floor_coverage_stop_ratio`；连续 `explore_stuck_steps` 步位移小于 `explore_stuck_move_m`；同一 action + BLIP cosine 连续 `explore_action_repeat_steps` 步不变；或同一 frontier 连续 `explore_frontier_repeat_steps` 次被选择。
- coverage 只对 true floor 计算，楼梯中间 level 的 coverage 视为 0；coverage fallback 后若当前层没有 entry，则直接用其他未进入层的 candidate frontier 做 probe，只有 probe 附近检测到 stairs 才生成 entry。
- 跨层失败使用独立 `entry_blacklist`，记录不可用的 `entry_xy` / `entry_dz`，不与同层目标 blacklist 混用；失败后尝试下一个最近 entry。
- 当前层 entry 与 frontier 都耗尽后，先尝试其他未进入层的 candidate frontier probe；probe 也耗尽才结束 episode；环境 500 步上限仍然有效。

## 2. 数据来源

| 数据 | 来源 |
| :--- | :--- |
| 机器人位姿 / 高度 | `_observations_cache["robot_xy_z"]`、`robot_heading` |
| 同层 2D frontier | Habitat `frontier_sensor`；后续按楼层 map 生成 |
| 3D frontier | free-space 边界；后续由每层/层间 map 提取 |
| 深度点云 | `object_map_rgbd` 的 depth + `tf_camera_to_episodic` |
| 每层 occupancy | floor-only 深度点按 z 分配到对应 floor map |
| entry 候选 | floor-only 点云上的 TSP3D `stairs` 检测（聚类/NMS 后每个 stair 一个 entry） |
| 其他层 frontier | 每个 level 单独缓存的 candidate frontier；只作为 probe 位置 |
| 目标楼层提示 | 只读查找目标 memory centroid 最近的 level；不创建新 level |
| 实际跨层验证 | agent 实际 z 变化、楼梯/坡道通行结果 |
| GT 楼层标签 | 仅离线评估使用，不进入在线决策 |

离线 stairs 召回代理统计：

```bash
python scripts/stairs_recall.py outputs/<run_dir> --window-steps 30 --radius-m 2.0
```

脚本把 true-floor 变化视作跨层事件，统计事件前窗口内、半径内的 accepted stair detection；它是代理召回，不等同于有 stair bbox GT 的精确召回。

## 3. 代码结构

```text
vlfm/vlvm/floor_manager/
├── __init__.py
├── manager.py                  # FloorManager
├── state.py                    # FloorRecord / FloorEntry / CurrentFloorState
├── floor_shadow.py             # read-only floor shadow
├── floor_camera_height.py      # floor-only dynamic camera height
└── per_floor_frontier.py       # 真实 3D occupancy -> 每层 2D occupancy/frontier
```

Floor manager 使用 episode-relative 高度：`floor_camera_height` 返回 `(agent_base_z, camera_z)`。旧 `frontier_sensor_3d` 已删除，因为它按当前 agent 高度 snap 2D frontier，所有 z 相同。`per_floor_frontier.py` 维护独立的 3D occupancy（相机高度使用实际高度），按楼层 z 切片投影出每层 2D occupancy/frontier，frontier z 取该层 `z_ref`。现有导航栈的固定 `camera_height` 不变。`FloorRecord.z_ref` 只在创建楼层时写入，同层内动态高度不会改写参考高度。

## 4. Floor Manager 维护内容

### 4.1 每层 `FloorRecord`

| 字段 | 含义 |
| :--- | :--- |
| `floor_id` | 稳定楼层 ID |
| `z_ref` | 该层地面参考高度 |
| `z_min` / `z_max` | 单层 map 的垂直范围 |
| `status` | `ACTIVE / EXHAUSTED / COMPLETE / BLOCKED` |
| `coverage` | 该层覆盖率；初期可为空 |
| `frontier_2d` | 该层 2D frontier |
| `frontier_3d` | 该层/层间 3D frontier |
| `occupancy_map` | 该层独立 occupancy map | 
| `explored_area` / `free_area` | 该层探索与 free-space |
| `entry_ids` | 关联的 entry |
| `visit_count` | 进入该层次数 |
| `last_observed_step` | 最近观测步 |
| `last_frontier_step` | 最近 frontier 出现步 |
| `exhausted_steps` | 连续耗尽步数 |
| `is_true_floor` | 是否达到真实楼层面积阈值；中间层为 `False` |
| `level_kind` | `FLOOR` / `INTERMEDIATE` / `PROVISIONAL` |
| `known_area_m2` | 该层已探索 2D 面积，用于真实楼层判定 |

每层的 `z_ref` 由 `agent_z - camera_height` 初始化；相邻层 z 差由 `floor_height_m` 维护。

### 4.2 `FloorEntry`

`FloorEntry` 只表示当前层上的 stair 出/入口，不再有 exit、STAIR 节点或跨层边。

| 字段 | 含义 |
| :--- | :--- |
| `entry_id` | 稳定 entry ID |
| `floor_id` | entry 所属的当前 level/floor |
| `entry_xy` | 当前层 stair 入口 XY（由 floor-only 点云的 TSP3D stair box 计算） |
| `entry_dz` | 到达后由 stair box 离当前 `z_ref` 最远的角点测得；首次为 0 |
| `dz_measured` | `entry_dz` 是否已测量 |
| `target_floor_hint` | 可选目标层提示；当前不参与最近距离选择 |
| `status` | `CANDIDATE / VERIFIED / TRAVERSED / BLACKLISTED` |
| `source` | 固定为 `tsp3d_stairs` |
| `confidence` | TSP3D stair 置信度 |
| `verified` / `verify_confidence` / `verify_step` | 到达后 TSP3D `stairs` 查询结果 |
| `first_seen_step` / `last_seen_step` | 观测时间 |
| `first_attempt_step` / `last_attempt_step` | 尝试时间 |
| `last_progress_step` / `best_goal_distance_m` | 导航进展与黑名单判定 |
| `attempt_count` / `at_entry_steps` | 尝试次数/到达入口后的停留步数 |
| `query_fail_count` | TSP3D 查询不可评估的连续次数 |
| `blacklisted_step` / `blacklist_reason` | 独立 entry blacklist 记录 |

规则：

- 多个 entry 可属于同一 `floor_id`；同一 stair 的多个 TSP3D box 先做 3D 聚类/NMS，只生成一个 entry。
- 当前层 frontier 耗尽时，`nearest_entry()` 只按 `entry_xy` 距离选择。
- 到达后先计算 `entry_dz` 并做几何检查，再检查 TSP3D 楼梯语义；失败则 blacklist 并尝试下一个 entry。
- `TRAVERSED` entry 在再次检测到同一 stair 时可恢复为 `CANDIDATE`；只有实际使用的 entry 会被标记 traversed。

### 4.3 `CurrentFloorState`

| 字段 | 含义 |
| :--- | :--- |
| `current_floor_id` | 当前确认楼层/层 |
| `previous_floor_id` | 上一个楼层/层 |
| `agent_z` | 当前相机/机器人高度 |
| `floor_z` | 当前层地面高度 |
| `floor_confidence` | 当前层归属置信度 |
| `in_entry` | 是否处于 entry 过程 |
| `entry_id` | 当前 entry |
| `entry_start_step` | entry 开始步 |
| `observed_delta_z` | 实际垂直变化 |
| `last_switch_step` / `switch_reason` | 最近切换记录 |

切换规则：

- 不能因“计划去 entry”提前切换；
- 只有实际 z 变化、且 entry 一致时才更新；
- 失败时保留旧楼层；
- 成功后才把新层设为 current，并把对应 entry 标为 `TRAVERSED`。

### 4.4 楼层高度与 entry 参数

| 参数 | 含义 |
| :--- | :--- |
| `camera_height_m` | 相机高度，用于 floor-only 点云 z 校正 |
| `floor_z_min_offset_m` / `floor_z_max_offset_m` | 单层 map 的上下偏移 |
| `floor_z_tolerance_m` | level 归属容差 |
| `floor_height_m` | 相邻层名义高差 |
| `floor_switch_delta_z_m` | 当前 level 切换的最小高度变化 |
| `min_new_floor_gap_m` | 提升为 true floor 的最小 z 间距 |
| `candidate_frontier_dedup_xy_m` | candidate frontier 与已有 true floor 的 XY 去重距离 |
| `entry_dedup_xy_m` | 同层 entry 去重距离 |
| `entry_min_abs_dz_m` | entry 几何检查的最小绝对高差 |
| `entry_blacklist_after_steps` | 无进展超时步数 |
| `entry_blacklist_progress_m` | 判定“有进展”的最小距离改善 |
| `entry_blacklist_near_m` | 靠近 entry 时不触发 progress timeout |
| `entry_at_arrival_timeout_steps` | 到达入口后仍未换层的超时步数 |
| `entry_query_fail_limit` | TSP3D 查询不可评估的失败上限 |
| `entry_arrival_radius` | policy 到达 entry、触发 TSP3D 楼梯查询的距离 |
| `entry_stair_query_conf_min` | TSP3D `stairs` 查询的最低置信度 |
| `entry_stair_cluster_radius_m` / `entry_stair_cluster_z_m` | stair box 3D 聚类/NMS 半径与 z 容差 |
| `floor_coverage_stop_ratio` / `floor_coverage_hold_steps` | 当前层覆盖率 fallback 阈值和连续保持步数（只看覆盖占比） |
| `explore_stuck_steps` / `explore_stuck_move_m` | 连续无位移步数和位移阈值，用于 explore 卡死 fallback |
| `explore_action_repeat_steps` / `explore_action_cosine_eps` | 同一 action + 不变 BLIP cosine 的 frontier 更换阈值 |
| `explore_frontier_repeat_steps` | 同一个 frontier 连续被选择的更换阈值 |
| `probe_max_attempts` / `probe_arrival_radius_m` | 其他层 frontier probe 的尝试上限/到达半径 |
| `probe_blacklist_after_steps` | probe 无进展超时步数 |
| `exhausted_patience_steps` | 连续无 frontier 判定耗尽的步数 |
| `max_floors` | 最大维护层数 |

## 5. 每步更新流程

1. 读取 robot z、floor-only 深度点云和当前层 frontier。
2. 按 `z_ref ± tolerance` 维护内部 level；切换 level 只更新当前 level，不单独提升 true floor。
3. 刷新所有 level 的 occupancy/frontier；每个 level 单独保存 candidate frontier。
4. true floor promotion（或关系）：candidate frontier 满足 z 间距 + XY 新颖性，或 verified entry 产生实际稳定高度变化。
5. 当前层 frontier 耗尽时由 policy 按距离选择 entry；到达后计算 `entry_dz` 并做 TSP3D 语义检查。
6. entry 失败则 blacklist 并尝试下一个；entry/frontier 都耗尽时，用其他未进入层的 candidate frontier 做 probe。
7. probe 不创建 entry；只有 probe 附近再次检测到 TSP3D `stairs` 才生成 entry。
8. 所有 entry、当前层 frontier、其他层 probe 都耗尽后才结束 episode；环境 500 步上限仍然有效。
9. `scripts/stairs_recall.py` 读取 `floor_manager.jsonl` 的 stair detections 和 true-floor 变化，输出代理 recall。

## 6. 当前层与全局耗尽

- `current_floor.status == EXHAUSTED` 只表示当前层没有可用 frontier。
- 当前层耗尽时，floor manager 查询已验证的 `FLOOR_ENTRANCE` / `STAIR` entry。
- 只有所有已发现楼层都耗尽、且没有可尝试 entry 时，才标记全局探索耗尽。
- `likely_infeasible` 仍只是离线地图标签，不能直接作为在线终止条件。

## 7. 3D frontier 与 2D frontier 分工

### 同层探索

- 使用当前层的 2D frontier；
- 保持现有 `_explore()` / PointNav 行为；
- 3D frontier 不直接生成同层目标。

### 跨层探索

- 3D frontier 用于发现层间 free-space 边界；
- 作为 `CROSS_FLOOR_CANDIDATE` 的候选种子；
- 通过 `entry_xyz → exit_xy(+exit_z)` 垂直 free-space 采样验证；
- 验证后才成为 `FLOOR_ENTRANCE`；
- 不直接使用未验证 candidate 作为跨层 nav goal；`entry_goal_xy()` 根据当前所在端返回 `entry_xyz[:2]` 或 `exit_xy`。

## 8. 历史数据结论（旧固定高度 3D map）

| 场景 | 结论 |
| :--- | :--- |
| mL8 | 至少三层；起始高度约 `-2.6 / 0 / 2.8 m`，目标高度约 `-2.3 / 0.3–0.5 / 3.1–3.6 m` |
| 5cd | 当前 episode 单层 |
| QaL | 当前 episode 单层 |

旧固定高度 3D map 的全局 z 范围只有 `0.15–1.50 m`，mL8 的其他楼层直接落在地图外。2026-11-02 后已改为独立的 per-floor `ObstacleMap3D`，使用实际相机高度，因此该限制不再适用于当前 floor map；M4 在线统计见 §12。

## 9. 初步实现状态

已落地：

- `FloorRecord`、`FloorEntry`、`CurrentFloorState` 数据结构；
- `FloorManager.observe()` shadow 更新；
- 真实楼层面积判定与 `level_kind`；
- 按 frontier z 创建 `CROSS_FLOOR_CANDIDATE` / `FLOOR_ENTRANCE` / `STAIR`；
- entry→exit 垂直连通性采样验证；
- entry 尝试计数、同层超时 blacklist、首次实际跨层 `TRAVERSED`；
- `floor_count` / `level_count` / entry 类型与 blacklist 日志；
- `floor_manager.jsonl` 日志落盘；
- floor shadow、动态高度与 3D frontier 已扁平化到 `vlfm/vlvm/floor_manager/`，`shadow/` 目录已删除。

已落地的最小动作接入：

- 当前层 frontier 耗尽时，`_explore_safe()` 使用 `FloorManager.best_entry()`；
- `entry_goal_xy()` 在当前层为源层时返回 `entry_xyz[:2]`，在当前层为目标层时返回 `exit_xy`；
- 没有可用 entry 时才回退到原有转向/最后 frontier。

尚未落地：

- 更可靠的楼梯几何/坡道检测；
- 完整楼层图代价与目标层路由；
- 跨层成功率的长期统计（当前只记 `TRAVERSED`）。

## 10. 后续实现顺序

1. ~~每层独立 occupancy map（z 范围只覆盖一层）~~ 已实现。
2. ~~每层 2D frontier，保留同层 2D 探索~~ 已实现。
3. ~~3D vertical connectivity + 楼梯/开口节点检测~~ 已实现基础采样验证。
4. ~~entry 状态机、失败黑名单~~ 已实现最小版本；楼层图代价仍待补。
5. mL8 上验证真实 `traveled_stairs`、floor switch、never_saw。
6. ~~当前层耗尽后接入 entry goal~~ 已实现；后续用新 run 验证 entry 可达性与黑名单触发。

## 11. 2026-11-02 动作接入

- `FloorManager.best_entry(robot_xy)` 从当前楼层选择最近的已验证 `FLOOR_ENTRANCE` / `STAIR`。
- `_explore_safe()` 在 frontier STOP 被抑制后，调用 `entry_goal_xy()` 取得当前端的坐标，再用 `_pointnav(stop=False)`。
- 控制台打印 `[FLOOR] current-floor frontier exhausted: use entry ... entry_xyz=... exit_xy=...`。
- 到达源层 entry 后打印 `[ENTRY-STAIR]`，TSP3D 查询 `stairs`；确认后走向 `exit_xy`，未确认则 `[ENTRY-BLACKLIST] reason=stair_not_found`。
- entry 导航若长期没有进展，按 `last_progress_step` / `best_goal_distance_m` 拉黑，不使用固定 first-attempt 超时；靠近 entry 时等待楼梯查询，不拉黑。
- 如果没有可用 entry，才退回原有转向/最后 frontier fallback。
- floor manager 在 explore 决策前刷新，因此使用当前帧的 per-floor frontier；每步只更新一次。
- `frontier_sensor` 在 explore 前只写当前层 frontier；其他层 frontier 只写入 `floor_frontier_2d/3d` 和 entry 记录，避免 2D explorer 直接跨层导航。

## 12. M4 在线统计

`outputs/s7_floor2_2026-11-01_154307`：

历史 M4（旧 transition 口径）：

- 最大 `floor_count=5`（含中间层，新口径会拆成 `floor_count` + `level_count`）
- 15 次楼层切换
- 最大旧 `transition_node_count=31`（新口径对应 `entry_count`）
- `floor_shadow` 最大跨层 frontier 数 3
- `frontier_z_span` 最大 1.31 m，非零帧 302

新口径日志字段：

- `floor_count`：真实楼层数；`level_count`：全部层/中间层数
- `floor_records`：每层 `z_ref`、`is_true_floor`、`level_kind`、`known_area_m2`、frontier 数
- `entry_count`、`entry_type_counts`、`floor_entrance_count`、`cross_floor_candidate_count`、`stair_count`、`valid_entry_count`
- `entries`：`entry_xyz`、`entry_xyz_source`、`exit_xy`、`exit_z`、`delta_z`、`status`、`vertical_connected`、`vertical_evidence`、`stair_verified`、`stair_confidence`、`reachable`、尝试/进展计数
- `entry_blacklist_count` / `entry_blacklist`：不可用的 `entry_xyz` / `exit_xy`

这些结果证明 per-floor frontier 与 entry 能在线产生；动作接入后仍需新 run 验证 `traveled_stairs`、floor switch 和 SR。
