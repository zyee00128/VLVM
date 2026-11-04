# VLVM-S7 baseline：S6-final 机制三场景结果

> 运行目录：`outputs/s7_baseline_3scene_2026-10-30_004608`  
> 启动脚本：`scripts/_nav_s7.sh`  
> 运行时间：`2026-10-30 00:46:36` 至 `2026-10-30 07:34:12`（6:47:36）  
> 场景：`5cdEh9F2hJL` / `QaLdnwvtxbs` / `mL8ThkuaVTM`，`3 × 99 = 297` episodes  
> 说明：本文只整理 S7 baseline 导航结果与运行产物；middle-tier 日志单独见 [VLVM-S7m0-exp.md](VLVM-S7m0-exp.md)，field 离线复算见 [VLVM-S7f0-exp.md](VLVM-S7f0-exp.md)。

## 1. 运行口径

| 项 | 值 |
| :--- | :--- |
| policy | `HabitatITMPolicyV1` |
| TSP3D checkpoint | `data/vlvm_v1/vlvm_v1.pth` |
| `sigma_tar` / `sigma_prop` | `0.60` / `0.45` |
| `provisional_sigma` | `0.55` |
| `upgrade_min_obs` | `4` |
| 目标选择 | `gd-tsp > gd-only > confirmed > provisional` |
| STOP | 仅 `gd-tsp confirmed`；`pointnav_stop_radius=0.90 m` |
| gd-only depth | 必须 `depth_ok=True` |
| gd-only conf gate / TTL | 关闭 / 不删除 |
| stable | 从选择中隐藏、不升级、不写 shadow |
| middle-tier | `enable_middle_tier_shadow=True`、`enable_middle_tier_ranking=False`、模型路径为空 |

## 2. 整体结果

| 指标 | 值 |
| :--- | ---: |
| episodes | 297 |
| success | **141 / 297** |
| SR | **47.47%** |
| Oracle success | 46.46% |
| SPL | 0.2182 |
| Soft SPL | 0.2852 |
| 平均 steps | 157.04 |

### 2.1 分场景

| 场景 | episodes | success | SR | Oracle | SPL | Soft SPL | steps |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 99 | 52 | 52.53% | 62.63% | 0.2273 | 0.2635 | 223.41 |
| `QaLdnwvtxbs` | 99 | 80 | 80.81% | 72.73% | 0.3699 | 0.4322 | 153.37 |
| `mL8ThkuaVTM` | 99 | 9 | 9.09% | 4.04% | 0.0572 | 0.1599 | 94.32 |

### 2.2 失败原因

| failure_cause | 数量 |
| :--- | ---: |
| `false_positive` | 100 |
| `never_saw_target_did_not_travel_stairs_likely_infeasible` | 48 |
| `never_saw_target_did_not_travel_stairs_feasible` | 3 |
| `bad_stop_true_positive` | 2 |
| `timeout_true_positive` | 2 |
| `false_negative` | 1 |
| `did_not_fail`（success） | 141 |

> 口径：`false_positive` 表示目标类在 memory 中存在，但最终 `nav_goal` 不在 GT 目标 bbox 内；它**不是单一的检测器误报**。`never_saw...likely_infeasible` 表示没有看到目标、没有走楼梯，且最终 `top_down_map.is_feasible=False`；`likely_infeasible` 是事后地图标签，不是 GT 不可达标签。

#### 2.2.1 `false_positive`（100）

按最后一步动作拆分：

| 终止方式 | n | 5cd / QaL / mL8 | oracle=1 | 有 TP 记忆 / TP 候选 | 最终目标来源 |
| :--- | ---: | :---: | ---: | :---: | :--- |
| `navigate` 后 STOP（动作 0） | 45 | 26 / 10 / 9 | 8 | 12 / 11 | `gd-tsp` 45 |
| `explore` 返回 STOP（动作 0） | 33 | 0 / 3 / 30 | 3 | 13 / 6 | `gd-only` 33 |
| 500 步超时 | 22 | 18 / 3 / 1 | 13 | 17 / 16 | `gd-only` 17，`tsp-provisional` 5 |

这 100 个里有 **24 个 oracle=1**（目标曾在探索范围内），但最终没有成功停止；另外 76 个从头到尾没有看到目标。按 oracle 再拆：

- oracle=1：8 个错误 STOP、3 个探索 STOP、13 个超时。
- oracle=0：37 个错误 STOP、30 个探索 STOP、9 个超时。

**A. 45 个策略 STOP：错误目标被确认成 `gd-tsp`**

- 45 个全部来自 `gd-tsp` confirmed；16 个由 `gd-only` 提升而来，29 个是直接的 gd-tsp 确认链路。
- 42 / 45 在选中目标点 0.9 m 内停止，说明 **STOP 半径不是主要问题**；失败发生在“停到了错误目标”。
- 45 个最终目标全部 `goal_tp=False`。其中 36 个质心也不在 GT 目标 bbox；9 个质心在 GT 目标附近，但选中的表面目标点在 bbox 外。
- 32 / 45 在整个 episode 中既没有 TP 候选也没有 TP 记忆；12 个曾有 TP 记忆、11 个曾有 TP 候选（有重叠），但最终仍选择了错误条目。
- 错误 STOP 的确认特征：`sigma_c` 中位数 0.613，`gd_hits` 中位数 6，`num_obs` 中位数 6，目标距离中位数 1.64 m；成功 episode 的最终 STOP（有 shadow 的 140 个）对应中位数为 `sigma_c=0.64`、`gd_hits=7`、`num_obs=6`。两组特征高度重叠，单纯调 STOP 阈值或半径不能分开。
- 9 个“质心 TP、目标点非 TP”的样本集中在床/沙发等大目标：7 个在 QaL，2 个在 mL8；目标点到质心距离 1.1–1.4 m。这是 **大目标表面点偏移**，不是错误物体识别。

**B. 33 个探索 STOP：frontier 耗尽后提前结束**

- 30 / 33 在 mL8；最终目标全部是 `gd-only`，不是策略 STOP（`stop_eligible=False`），最后一步是 `Mode: explore | Action: 0`。
- 中位步数 119；30 / 33 `oracle=0`，23 / 33 的最终质心不是 TP。
- 13 个曾有 TP 记忆，但目标没有升级到可停止状态；最终目标链均出现 `stop_blacklist`。
- 这说明探索阶段在“当前楼层没有 frontier”时直接返回 STOP，导致两类失败：完全没看到目标，或看到过但没来得及完成确认。

**C. 22 个超时：目标看到了但始终没有形成可 STOP 的确认**

- 18 / 22 在 5cd，13 / 22 `oracle=1`，17 / 22 曾有 TP 记忆，16 / 22 曾有 TP 候选。
- 最终目标为 `gd-only` 17 个、`tsp-provisional` 5 个；全部 `stop_eligible=False`，没有一次策略 STOP。
- 最终目标的 `num_obs` 和 `gd_hits` 中位数都只有 2；目标在视野中出现过，但没有积累到确认条件，随后不断切换目标直到 500 步。
- 这与 45 个错误 STOP 是同一个状态机的两端：**弱证据既可能过早确认错误目标，也可能迟迟确认不了正确目标**。

**形成原因**

1. **同轨重复 GD 命中被当成独立证据**：`gd_hits` 中位数 6，但这些命中大多来自同一错误 track 的连续帧；`gd-only → gd-tsp` 提升路径只要求重复命中，没有目标类语义复核或视点位移约束。
2. **STOP 证据与真实身份不独立**：错误 STOP 与成功 STOP 的 `sigma_c`、`gd_hits`、`num_obs` 分布几乎重合，当前 STOP 门只能确认“这条轨迹稳定”，不能确认“这是目标”。
3. **大目标表面点偏移**：`goal_use_surface=True` 时，床/沙发等大目标的表面点可能距质心 1.1–1.4 m，落在 GT bbox 外；即使质心已 TP，最终仍被记为 false positive。
4. **探索 STOP 没有目标条件**：`_explore()` 在当前层 frontier 耗尽时返回动作 0，policy 没有拦截；这会把“没找到目标”直接变成 episode 终止。
5. **确认不足与确认过早并存**：22 个超时中 17 个已有 TP 记忆，但 `num_obs=2`、没有 gd-tsp；需要主动复核，而不是继续全局切换。

#### 2.2.2 `never_saw_target_did_not_travel_stairs_likely_infeasible`（48）

- 48 个全部在 mL8；48 个最后一步都是 `Mode: explore | Action: 0`，中位步数 93（范围 13–159），没有一个是 500 步超时。
- 48 个的 TP 候选 = 0、TP 记忆 = 0；除 1 个例外，最终没有 nav goal，说明不是 candidate 被错误选择，而是**目标从未进入候选池**。
- 38 / 48 的起始点与目标高度差绝对值 > 1.5 m（10 个目标在上层、28 个在下层）；全部 48 个的 |Δy| 中位数 2.37 m，episode geodesic distance 3.57–13.85 m、中位 9.35 m。
- 目标类别：couch 18、toilet 13、bed 11、chair 6。
- mL8 共 99 集：9 success、40 false_positive、48 likely_infeasible、2 never_saw_feasible。也就是说，mL8 的失败不是少量难例，而是 **48.5% 的 episode 在探索阶段提前结束**。

这里要特别区分两件事：

- `top_down_map.is_feasible=False` 是**最终地图上的事后标签**，只说明当前已探索楼层没有形成到目标的可通行路径；不能据此断言 GT 目标不可达。
- 这些 episode 的 HM3D metadata 仍给出有限的 `geodesic_distance`，且 38/48 明确跨楼层。因此更准确的解释是：**frontier 探索只覆盖了当前楼层，找不到楼梯/下一层入口后返回 STOP**，而不是 48 个目标都不存在或物理上不可达。

**形成原因**

1. **frontier 探索没有楼层/楼梯语义**：当前层 frontier 耗尽就返回 STOP；没有“楼梯入口”“下一层 frontier”或楼层切换动作。
2. **没有目标时的 STOP 保护**：`_explore()` 的 STOP 没有被 `target_detected` 或 `stop_eligible` 拦截。
3. **跨楼层目标缺少先验**：mL8 的目标多在另一层；agent 没有到达目标层的机制，检测器自然没有机会产生候选。
4. **失败标签会掩盖可恢复性**：`likely_infeasible` 是地图标签，不是 GT 可行性；把它当作不可解样本会低估探索策略问题。

#### 2.2.3 后续改进

| 优先级 | 改动 | 直接覆盖的失败 | 说明 |
| :--- | :--- | :--- | :--- |
| P0 | 禁止 `_explore()` 在目标未确认时返回 STOP；frontier 耗尽时改走回退探索/楼梯入口 | 33 个探索 STOP + 48 个 never_saw | 这是最大的一组：81 个 episode 的最后动作都是 `explore/0` |
| P0 | 增加楼层感知 frontier / 楼梯节点；记录楼层切换和已探索楼层 | mL8 的 38/48 跨楼层样本 | 不能只看当前层 2D frontier |
| P0 | 将“重复 GD 命中”改为“不同视点/位移的独立证据”，并加入目标类语义复核后才允许 `gd-only → gd-tsp` | 45 个错误 STOP，尤其 16 个提升样本 | 不要把同一 track 的连续帧累加当成确认 |
| P1 | 大目标（bed/couch）使用受约束的目标点/质心邻域，保证目标点不越出目标 bbox | 9 个大目标表面点偏移样本 | 目标点偏移与错误物体要分开统计 |
| P1 | 对已看到但 `num_obs<4` 的目标做主动环绕/复核，不立即切换 | 22 个超时，其中 13 个 oracle=1 | 解决“正确目标迟迟不确认” |
| P1 | 失败统计拆成 `wrong_stop / explore_stop / timeout_unconfirmed / surface_goal_miss` | 全部失败 | 当前 `false_positive` 混了三种机制，不能直接当作检测器 FP 率 |
| P1 | 在评测中单独报告 mL8 与跨楼层子集；用 GT shortest path 核实 `likely_infeasible` | mL8 48 集 | 避免把“没探索到”误报成“不可达” |

**不要只调 STOP 阈值**：45 个错误 STOP 中 42 个已经在 0.9 m 内，且 `sigma_c/gd_hits/num_obs` 与成功 STOP 高度重叠；继续收紧阈值只会把错误 STOP 换成更多超时。应先解决目标身份证据、探索终止和跨楼层能力。

## 3. 候选与准入

| 项 | n | TP | FP |
| :--- | ---: | ---: | ---: |
| candidates | 5,688 | 1,628 | 4,060 |
| candidate precision | — | 28.62% | — |
| `s_penalty_admitted` | 687 | — | — |
| memory written | 2,181 | — | — |
| memory merged | 1,861 | — | — |

### 3.1 provisional gate outcomes

| outcome | n | TP | FP |
| :--- | ---: | ---: | ---: |
| `confirmed` | 146 | 62 | 84 |
| `merge_existing` | 1,861 | 1,192 | 669 |
| `provisional` | 174 | 57 | 117 |
| `missing` | 216 | 7 | 209 |
| `rejected` | 3,291 | 310 | 2,981 |

## 4. 记忆生命周期

### 4.1 事件

| event | n | TP |
| :--- | ---: | ---: |
| `gd_only_new` | 737 | 316 |
| `gd_only_merge` | 3,051 | 1,635 |
| `gd_tsp_merge` | 1,291 | 974 |
| `new` | 320 | 119 |
| `merge` | 1,861 | 1,192 |
| `upgrade` | 224 | 155 |
| `transition` | 123 | 35 |
| `delete` | 89 | 27 |
| `stop_blacklist` | 478 | 134 |

### 4.2 新条目状态

| state | n | TP | FP |
| :--- | ---: | ---: | ---: |
| `confirmed` | 146 | 62 | 84 |
| `provisional` | 174 | 57 | 117 |

### 4.3 升级来源

| upgrade_reason | n | TP |
| :--- | ---: | ---: |
| `base_upgrade` | 33 | 15 |
| `gd_only_tsp_merge` | 135 | 105 |
| `gd_tsp` | 56 | 35 |

### 4.4 删除来源

| delete_reason | n | TP |
| :--- | ---: | ---: |
| `stable_stale_purge` | 77 | 24 |
| `stable_total_miss` | 7 | 2 |
| `stable_miss_delete` | 5 | 1 |

### 4.5 stable transition

| transition | n |
| :--- | ---: |
| `confirmed->stable` | 75 |
| `provisional->stable` | 48 |

## 5. 选择与 STOP shadow

### 5.1 nav goal tier

| selection_tier_name | 行数 | TP |
| :--- | ---: | ---: |
| `gd-tsp` | 2,679 | 1,767 |
| `gd-only` | 8,362 | 2,630 |
| `confirmed` | 463 | 129 |
| `provisional` | 454 | 117 |

### 5.2 STOP shadow

| 项 | 值 |
| :--- | ---: |
| `stop_shadow` 行数 | 11,493 |
| `stop_eligible_current=true` | 2,668 |
| `tp=true` | 4,519 |
| `goal_tp=true` | 3,654 |

## 6. 与 S6 结果对照

| run | success | SR |
| :--- | ---: | ---: |
| S6gd3 | 139 / 297 | 46.80% |
| S6gd5 | 134 / 297 | 45.12% |
| **S7 baseline** | **141 / 297** | **47.47%** |

分场景：

| 场景 | S6gd3 | S6gd5 | S7 baseline |
| :--- | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 51 | 46 | 52 |
| `QaLdnwvtxbs` | 80 | 79 | 80 |
| `mL8ThkuaVTM` | 8 | 9 | 9 |

## 7. 产物

- `s7_summary.json`
- `detections/episodes.jsonl`
- `detections/candidates.jsonl`
- `detections/memory_events.jsonl`
- `detections/nav_goals.jsonl`
- `detections/stop_shadow.jsonl`
- `detections/middle_rank_shadow.jsonl`
- `detections/field_shadow.jsonl`
- `detections/field_observations/`：297 ZIP，约 45 GB
- `field_analysis/field_offline_report.json`
- `field_analysis/field_offline_summary.md`
- `middle_tier/middle_tier_fit_summary.json`
- `middle_tier/middle_tier_model.json`

## 8. 相关文档

- `Results/VLVM-S7m0-exp.md`：middle-tier 拟合与边界分析
- `Results/VLVM-S7f0-exp.md`：场机制目标覆盖复算与在线 shadow
- `Results/VLVM-S7.md`：S7 机制总览与后续实验拆分
