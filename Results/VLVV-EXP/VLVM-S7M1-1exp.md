# Project VLVM Stage 7 M1：1A 实验与 1B/1C shadow

> **后续口径（2026-11-02）：** 1C 的几何门、时效门和所有独立证据门均已弃用；当前只考虑 floor 机制和 blacklist。本文保留历史实验结果。

> 运行目录：`outputs/s7_baseline_3scene_2026-10-30_163357`
> 场景：`5cdEh9F2hJL`、`mL8ThkuaVTM`、`QaLdnwvtxbs`，3 × 99 回合
> 启动方式：`FLOOR_SHADOW_ENABLE=1 IDENTITY_SHADOW_ENABLE=1 bash scripts/_nav_s7.sh`
> 运行时间：11:28:46；`COMPLETED` 与 `s7_summary.json` 已生成
> 关键边界：1A 是唯一的动作改动；1B/1C 均为只读 shadow，不改变目标选择、动作或 STOP。

## 0. 结论摘要

- **1A 的 STOP 抑制不变量成立**：最终 `explore/0` 从 baseline 的 85 个降到 0；explore 的 STOP 被转向/回退替代。
- **1A 没有带来实质性能提升**：success 141 → 142，SR 47.47% → 47.81%；oracle 仍为 138。代价是平均步数 157.0 → 265.9（+69.3%），中位步数 102 → 204。
- **1A 主要是把失败换了形态**：never_saw 51 → 46（−5），false_positive 100 → 105（+5），timeout_true_positive 2 → 1，success +1。mL8 的 never_saw 仍然全部存在，只是从约 93 步结束变成 500 步耗尽。
- **1B shadow 本轮无效，不是负结果**：79k 行 floor shadow 中 `frontier_count_3d` 全为 0，`floor_proxy_count` 全为 0，`would_floor_transition` 全为 0；当前数据无法验证楼层/楼梯机制。必须先补 3D frontier/高度来源。
- **1C 的“不同视点独立命中”门可行性不足**：独立视点/`gd_hits` 对 TP 与非 TP 的 AUC 只有约 0.57–0.61；错误 STOP 记忆中 30/45 反而满足 `independent_ok`；在 gd-only 提升中，该门会阻断 22 个 TP、只阻断 4 个非 TP。当前数据不支持把独立视点门直接接入提升。
- **1C 下一步应做“语义冲突”门，而不是继续调 `gd_hits`**：只分冲突/不冲突；冲突保留 gd-only，不冲突再走其他证据门。主动复核没有证据支持，放弃在线接入。

## 1. 实验设置与口径
### 1.1 本轮改动

| 机制 | 状态 | 是否改变动作 |
| :--- | :--- | :--- |
| 1A：explore STOP 抑制 | 已启用 | 是；explore 动作 0 被替换为转向/回退 |
| 1B：楼层/楼梯 frontier | shadow | 否 |
| 1C：独立视点身份确认 | shadow | 否 |
| field shadow / middle-tier shadow | 保持原配置 | 否 |

### 1.2 指标口径

- `success`、`SR`、`oracle`、`SPL` 来自 `episodes.jsonl` / `s7_summary.json`。
- `explore/0` 按 `navigation.log` 每个 episode 最后一个 `Step | Mode | Action` 解析；baseline 为 85，旧文档按失败类别统计为 81，本文使用逐 episode 解析值。
- `explore_stop_suppressed` 是 1A 内部计数；floor shadow 中按 episode 取最大值，用于判断该 episode 是否发生过抑制。
- 1C 的 `gd_hits` 来自 GD calibration 的 `gd_hits_after`，不是 memory manager 的完整状态；该字段只在 `match_track.same_object` 成立时递增，因此 `would_block_promotion` 只是保守代理，不是最终线上门。
- 1C 统计单位：同一 episode 同一 `memory_id` 取最后一条 identity shadow 行；提升事件取事件发生前最近一条 memory 快照。

## 2. 1A 实验结果

### 2.1 总体指标

| 指标 | baseline | M1（1A） | 变化 |
| :--- | ---: | ---: | ---: |
| episodes | 297 | 297 | 0 |
| success | 141 | 142 | +1 |
| SR | 47.47% | 47.81% | +0.34 pp |
| oracle_success | 138 | 138 | 0 |
| SPL | 0.2182 | 0.2207 | +0.0026 |
| soft SPL | 0.2852 | 0.2838 | −0.0014 |
| steps mean | 157.04 | 265.87 | +108.83（+69.3%） |
| steps median | 102 | 204 | +102 |
| false_positive | 100 | 105 | +5 |
| never_saw（likely + feasible） | 48 + 3 | 43 + 3 | −5 |
| timeout_true_positive | 2 | 1 | −1 |
| bad_stop_true_positive | 2 | 2 | 0 |
| false_negative | 1 | 1 | 0 |

### 2.2 分场景

| 场景 | baseline SR | M1 SR | baseline steps | M1 steps | 观察 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 5cd | 52/99 = 52.53% | 52/99 = 52.53% | 223.4 | 223.4 | 完全不变 |
| QaL | 80/99 = 80.81% | 81/99 = 81.82% | 153.4 | 155.1 | +1 success |
| mL8 | 9/99 = 9.09% | 9/99 = 9.09% | 94.3 | 419.1 | success 不变，步数约 ×4.4 |

mL8 是 1A 成本最明显的场景：中位步数从 95 升到 500，但 success 仍为 9，oracle 仍为 4。

### 2.3 STOP 抑制

| 项目 | baseline | M1 |
| :--- | ---: | ---: |
| 最终 `explore/0` | 85 | 0 |
| 最终 `navigate/0` | 190 | 193 |
| 最终 `explore` 总数 | 107 | 104 |
| 最终 `Action=2`（左转） | 7 | 87 |
| 发生过 explore STOP 抑制的 episode | — | 85 |
| 抑制步数合计 | — | 31,983 |

按 episode 结果拆分：

| 结果 | episodes | 发生抑制的 episode | 抑制步数 |
| :--- | ---: | ---: | ---: |
| success | 142 | 2 | 8 |
| false_positive | 105 | 38 | 12,996 |
| never_saw | 46 | 45 | 18,979 |
| 其他失败 | 4 | 0 | 0 |

结论：1A 确实把“explore 终止”改成了“继续扫描/回退”。但在 mL8，frontier 耗尽后没有可用的跨楼层目标，策略只是在原地或旧 frontier 之间反复消耗预算，最终以 500 步结束。**这不是探索能力提升，只是终止语义修复。**

### 2.4 episode 级变化

baseline → M1 发生变化的核心回合：

| 场景/回合 | 目标 | baseline | M1 | 步数 |
| :--- | :--- | :--- | :--- | ---: |
| QaL 32 | couch | timeout_true_positive | success | 422 → 434 |
| mL8 64 | couch | never_saw | success | 13 → 39 |
| mL8 75 | couch | success | false_positive | 120 → 500 |
| mL8 0 | bed | never_saw | false_positive | 99 → 500 |
| mL8 17 | chair | never_saw | false_positive | 145 → 500 |
| mL8 37 | bed | never_saw | false_positive | 100 → 500 |
| mL8 59 | bed | never_saw | false_positive | 128 → 500 |

净效果是 +1 success，但其中 4 个 never_saw 只是变成了 false_positive，1 个原有 success 变成了 false_positive。**1A 单独使用时没有解决“找不到/确认不了目标”的根因。**

### 2.5 1A 判断

- 作为**动作流不变量**，1A 是成功的：explore 不再能发出 STOP。
- 作为**性能机制**，1A 不成立：SR 只 +1，oracle 不变，步数大幅增加，false_positive 反而增加。
- 1A 应该保留为安全约束，但必须和 1B 的楼层/楼梯探索或更强的目标发现机制组合，不能单独作为 S7 的收益来源。
- 下一步不建议继续调 1A 的 fallback 转向次数；应先把 frontier 耗尽后的“去哪里”交给 1B。

## 3. 1B shadow：楼层/楼梯

### 3.1 数据量与关键限制

- `floor_shadow.jsonl`：78,962 行。
- 所有行：`frontier_count_3d = 0`。
- 所有行：`floor_proxy_count = 0`，`floor_proxy_id/z = null`。
- 所有行：`visible_floor_ids = []`，`would_floor_transition = False`，`transition_candidate_* = null`。
- `frontier_count_2d` 在耗尽时常见为 1，但那是 `[0, 0]` 哨兵点，不是有效 frontier。

因此，本轮 **1B shadow 没有获得任何可用的楼层高度或跨层候选证据**。这是数据源问题，不是 1B 算法失败：当前任务只配置了 2D `frontier_sensor`，没有可用的 3D frontier。

### 3.2 仍然可用的 1A/探索信号

| 结果 | episodes | `explore_exhausted` episode | 抑制步数 |
| :--- | ---: | ---: | ---: |
| success | 142 | 2 | 8 |
| false_positive | 105 | 38 | 12,996 |
| never_saw | 46 | 45 | 18,979 |
| 其他失败 | 4 | 0 | 0 |

分场景：

- mL8：45/45 never_saw 全部耗尽；35/45 false_positive 也耗尽；success 只有 1/9 发生耗尽。
- QaL：3/16 false_positive 耗尽；success 1/81。
- 5cd：false_positive 和 success 均未耗尽。

这说明 1A 的“frontier 耗尽”主要发生在 mL8，且与 never_saw / FP 高度重合。1B 要解决的正是这些耗尽后的跨层探索；但本轮没有 3D 高度，不能判断“是否有楼梯候选”或“是否应该跨层”。

### 3.3 1B 判断与修复项

- **不能把本轮 1B shadow 当作负结果。** 它只证明当前 3D frontier 数据为空。
- 下一轮 1B shadow 前必须补齐以下之一：
  1. 增加真正的 3D frontier / 高度传感器；
  2. 用 2D frontier + 3D obstacle map 查询 frontier 的可达高度/楼层；
  3. 至少记录机器人实际楼层变化和楼梯入口可见性。
- 在数据修复前，不启动 1B 动作 A/B；否则无法归因。
- 1A 的 fallback 不应无限循环：当前 45 个 never_saw 都跑到 500 步，说明“原地转向 + 旧 frontier”没有产生新信息。

### 3.4 3D frontier 生成器（实验结果之后新增，未重跑实验）

本轮实验之后新增了真正的 3D frontier 提取，只给楼层管理器/floor shadow 使用，**不改变同层 2D 导航**。

**新增文件**

- `vlfm/shadow/frontier_3d.py`：`extract_frontiers_3d(obstacle_map)`。

**提取逻辑**

1. 从 `obstacle_map3d._map` 取 occupied，从 `obstacle_map3d.explored_area` 取 explored。
2. `free = explored & ~occupied`。
3. 对 `explored` 做 6 邻域 erosion，`frontier = free & ~erosion(explored)`，得到“已探索空闲体素且邻接未探索空间”的 3D 边界。
4. 过滤：
   - 上方至少 `0.80 m` 是 free，排除天花板/悬空边界；
   - 下方 `0.25 m` 内有 occupied 或 unknown，排除纯空中边界；
   - 按 voxel stride 下采样，最多保留 2048 个点。
5. 用 `obstacle_map3d._grid_index_to_xyz` 转成世界坐标 `(x,y,z)`。

**代码流向**

```text
Habitat observations
  └─ frontier_sensor (2D) ──► _explore() / PointNav
                               （完全不变）

obstacle_map3d
  ├─ _map (occupied)
  ├─ explored_area
  └─ _grid_index_to_xyz
        │
        ▼
extract_frontiers_3d()  [vlfm/shadow/frontier_3d.py]
        │  N×3 (x,y,z)
        ▼
TSP3DObjectNavPolicy._get_floor_frontiers_3d()
        │  cache; FLOOR_SHADOW_3D_EVERY_N=5
        ▼
TSP3DObjectNavPolicy._log_floor_shadow_frame()
        │  local variable only
        ▼
FloorShadowLogger.observe(frontiers_3d=...)
        │
        ▼
detections/floor_shadow.jsonl
```

关键边界：

- `_get_floor_frontiers_3d()` 只在 `_log_floor_shadow_frame()` 中调用；`_log_floor_shadow_frame()` 在动作选择之后执行。
- 3D 点**不写入** `_observations_cache["frontier_sensor_3d"]`，也不替换 `frontier_sensor`。
- `_explore()` 仍只读 2D `frontier_sensor`；PointNav 仍只接收 2D goal。
- floor shadow 新增 `valid_frontier_count_2d`、`frontier_z_source`、`floor_data_valid`，用于确认 3D 数据是否真的非空。
- 启动脚本新增 `FLOOR_SHADOW_3D_EVERY_N`（默认 5），只控制 shadow 提取频率。

因为这是实验结束后的代码修复，本文第 2/3 节仍描述旧运行结果；需要重新跑 1B shadow 才能得到新数据。

**冒烟验证**：`outputs/s7m1_3dfrontier_smoke_2026-10-31`（1 episode，500 行）中：

- `frontier_count_3d` 最大 1990，`frontier_z_source=3d_sensor`；
- `floor_data_valid=True`；
- `floor_proxy_id=0`、`floor_proxy_z=0.45`；
- `visible_floor_ids=[0]`、`would_floor_transition=False`。

该 episode 只验证了 3D frontier 提取和单层 floor proxy；它没有跨层，因此还不能验证楼梯候选。下一次应在 mL8 上专门跑 smoke。

## 4. 1C shadow：独立视点身份确认

### 4.1 数据规模

| 项目 | 数值 |
| :--- | ---: |
| identity shadow 行数 | 5,943 |
| 去重后的 memory 数 | 821 |
| TP 记忆 | 418 |
| 非 TP 记忆 | 403 |
| 提升事件（有 identity 快照） | 197 |
| `gd_only_tsp_merge`（有快照） | 125 |
| `gd_tsp`（有快照） | 58 |

> `s7_summary.json` 总 upgrades 为 229（`gd_only_tsp_merge` 136、`gd_tsp` 58、`base_upgrade` 35）；其中部分提升事件没有对应的 identity shadow 行，因此 1C 提升时点分析使用有快照的 197 次。

### 4.2 TP 与非 TP 的独立视点特征

| 特征 | TP（418） | 非 TP（403） | AUC（TP vs 非TP） |
| :--- | ---: | ---: | ---: |
| `gd_hits` 均值 | 6.18 | 5.02 | 0.586 |
| `view_count` 均值 | 6.30 | 5.28 | 0.570 |
| `distinct_pose_count` 均值 | 5.47 | 4.58 | 0.570 |
| `distinct_yaw_count` 均值 | 1.89 | 2.11 | 0.457 |
| `same_object_views` 均值 | 5.40 | 3.81 | 0.607 |
| `max_parallax_m` 均值 | 1.75 | 2.00 | 0.500 |
| `independent_ok` | 187/418 = 44.7% | 173/403 = 42.9% | — |
| `would_block_promotion` | 78/418 = 18.7% | 53/403 = 13.2% | — |

解释：

- `gd_hits`、`view_count`、`distinct_pose_count` 只有弱分离，AUC 约 0.57–0.59。
- `distinct_yaw_count` 几乎反向（AUC 0.457），`max_parallax_m` 完全无分离（AUC 0.500）。
- `same_object_views` 稍好（AUC 0.607），但它更像“被重复关联的次数”，仍不足以做身份门。
- `would_block_promotion` 在 TP 中反而更高：**独立视点门会更容易挡住正确目标，而不是错误目标。**

### 4.3 提升时点的行为

提升事件中：

| 组 | 数量 | `would_block_promotion` |
| :--- | ---: | ---: |
| TP promotions | 139 | 22（15.8%） |
| 非 TP promotions | 58 | 6（10.3%） |

其中有 identity 快照的 `gd_only_tsp_merge` 125 次提升：

| 组 | 数量 | 被独立视点门阻断 |
| :--- | ---: | ---: |
| TP | 93 | 22（23.7%） |
| 非 TP | 32 | 4（12.5%） |

这是本轮最重要的 1C 结论：

> **在 gd-only 提升时点，独立视点门会阻断 22 个正确提升，只阻断 4 个错误提升。**

原因是：正确目标通常被反复观察，`gd_hits`/`view_count`/`same_object_views` 更高；错误 gd-only 往往在早期就被选中、观测次数少，根本达不到 `gd_hits >= 3` 的触发条件。门只在“高命中尾部”生效，而高命中尾部本身更偏 TP。

### 4.4 对错误 STOP 的行为

按最终动作关联最后一条 STOP shadow 快照：

| 组 | 数量 | `independent_ok` | `would_block_promotion` |
| :--- | ---: | ---: | ---: |
| 错误 STOP（false_positive，final action 0） | 45 | 30/45 = 66.7% | 8/45 = 17.8% |
| TP STOP（success，final action 0） | 142 | 79/142 = 55.6% | 32/142 = 22.5% |

错误 STOP 的记忆平均 `gd_hits=7.38`、`distinct_pose=5.78`、`parallax=1.57`，与 TP STOP 的 `7.70`、`6.30`、`1.59` 几乎重合。**错误目标也可以被从多个视点反复看到；视点多样性不等于语义正确。**

因此，1C 的独立视点门：

- 不能可靠拦截错误 STOP；
- 会阻断更多 TP STOP；
- 不应直接接入 `gd-only → gd-tsp` 或 STOP。

### 4.5 为什么独立视点门失效

1. **它衡量的是观测多样性，不是类别正确性。** 错误目标可以被多次观察，正确目标也可能只被短暂看到；两者分布高度重叠。
2. **门的触发条件偏向高 `gd_hits`。** 高命中记忆更可能是 TP，因此门主要作用在正确目标上。
3. **`same_object` 是 GD/TSP 关联结果，不是独立语义验证。** 它可能重复同一错误关联。
4. **`gd_hits` 的当前计数本身是弱新信息门**（0.2 m / 10° / 3 帧），不是严格独立视点；1C shadow 的严格代理也没有显示出与 TP 的稳定关系。
5. **错误 STOP 的大量多视点证据说明问题在“目标类语义”，不在“看过几次”。** baseline 45 个错误 STOP 中 36 个是错误物体，这更符合语义冲突门的适用范围。

### 4.6 1C 具体行为建议

#### A. 目标类语义复核门（下一步）

位置：`gd-only → gd-tsp` 提升前，只做一次候选级检查，不逐帧运行。

输入：GD bbox crop、目标类 prompt、必要的上下文。

输出只允许两种状态：

- **冲突**：保留 `gd-only`，不提升；不删除记忆；记录冲突原因。
- **不冲突**：允许提升，但继续通过其他证据门。

补充规则：

- 不做细粒度属性一致性，不做多类概率排序；
- crop 质量差或无法判断时，按“不冲突”处理并记录 `low_quality`，避免过严；
- 该门是“语义一致性门”，不能称为“视点独立门”；两者解决不同失败模式。

#### B. 其他证据门

| 门 | 作用 | 当前建议 |
| :--- | :--- | :--- |
| 同目标关联门 | 要求 TSP merge 指向同一 `memory_id`/track，`same_object` 或等价关联成立 | 保留；不能只靠 0.5 m 距离 |
| 几何/状态门 | `depth_ok`、非 `geom_unknown`、非 `stable`、非 STOP blacklist | 保留 |
| 时效门 | GD 证据必须近期；旧 `gd_hits` 不参与提升 | 保留 |
| 现有 TSP merge / provisional gate | 同类别、0.5 m 内、`confirmed`/`provisional` | 不变 |
| 独立视点门 | 多视角/位移证据 | **只保留为 shadow；当前数据不支持接入** |
| 单一阈值 | `gd_hits`/`num_obs`/`sigma_c` 任一阈值 | 禁止单独决定提升 |

#### C. 主动复核

- 触发：当前目标已选中、已看到、`num_obs < 4`、尚不可 STOP，且存在竞争候选或分数接近；不由 `num_obs` 单独触发。
- 行为：有限预算内环绕/侧移/回看；最多 2 个新视点或固定步数；预算用完或没有新证据就恢复原选择，不永久禁止切换。
- 当前决策：**不接入在线动作**。现有 identity shadow 不能证明主动复核的收益；若后续没有专门的反事实 shadow 证明“减少错误切换/错误 STOP 且 timeout 可控”，直接放弃。
- 不做硬性“`num_obs < 4` 禁止切换”；该条件过严会卡住正确目标，过松会保留 FP。

### 4.7 1C 判断

- **独立视点门：当前数据不支持。** AUC 弱，提升时点误伤 TP，错误 STOP 反而高独立视点。
- **语义冲突门：下一步优先做 shadow。** 它针对“错误物体”这一主要失败模式，且只分冲突/不冲突，行为简单。
- **主动复核：放弃在线接入。** 保留为离线问题，不再成为 1C 主线。
- **不要调 `gd_hits` 阈值。** 当前问题不是阈值高低，而是该信号与目标正确性不相关。

## 5. 三个机制的总体判断

| 机制 | 本轮结果 | 判断 |
| :--- | :--- | :--- |
| 1A | explore/0 85 → 0；SR +0.34 pp；步数 +69% | 不变量成立，但单独无性能收益；保留为安全约束 |
| 1B | 3D frontier 全空，floor proxy 全空 | 数据源无效，不能评价；先修传感器/高度来源 |
| 1C 独立视点 | AUC 约 0.57–0.61；提升时阻断 TP 多于 FP | 不接入；保留 shadow |
| 1C 语义冲突 | 未实现，仅有设计 | 下一步做 shadow；只分冲突/不冲突 |
| 1C 主动复核 | 无证据 | 放弃在线接入 |

## 6. 下一步

1. **1A**：保留 STOP 抑制；但给 fallback 设预算/终止条件，避免 mL8 全部 500 步空转。不要把 1A 当作单独收益机制。
2. **1B**：先修 `frontier_sensor_3d` 或从 2D frontier + 3D map 生成高度；重跑 shadow，确认 `floor_proxy_count>0` 且存在跨层候选，再做动作 A/B。
3. **1C**：实现语义冲突 shadow，只在 gd-only 提升前记录冲突/不冲突和 `tp`；先验证冲突在 FP 中是否显著多于 TP，再考虑接入。
4. **1C 独立视点**：保留 `identity_shadow.jsonl` 作为诊断，不再作为提升门候选；不调 `gd_hits` 阈值。
5. **1C 主动复核**：暂不实现；没有反事实证据就不进入动作链路。

## 7. 产物

- `outputs/s7_baseline_3scene_2026-10-30_163357/s7_summary.json`
- `outputs/s7_baseline_3scene_2026-10-30_163357/detections/episodes.jsonl`
- `outputs/s7_baseline_3scene_2026-10-30_163357/detections/floor_shadow.jsonl`
- `outputs/s7_baseline_3scene_2026-10-30_163357/detections/identity_shadow.jsonl`
- `outputs/s7_baseline_3scene_2026-10-30_163357/s7m1_analysis.json`
- 分析脚本：`scripts/analyze_s7m1.py`
