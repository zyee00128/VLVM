# Project VLVM Stage 7 M3：独立证据门、主动复核与目标关联门实验结果

> 说明：本文保留 M3 的历史实验结果；其中 identity、semantic、evidence、active recheck、目标关联门代码已在本轮清理中删除。**2026-11-02 后续口径：** gd-only 几何门、时效门也已删除；当前只考虑 floor 机制和 blacklist。当前保留的 floor shadow / floor manager 结果见 §7。

> 运行目录：`outputs/s7_m1a_gate2_2026-11-01_002946`
> 对照：S7 baseline `outputs/s7_baseline_3scene_2026-10-30_004608`；M1 `outputs/s7_m1a_3scene_2026-10-30_163357`；M2 `outputs/s7_gates_3scene_2026-10-31`
> 场景：`5cdEh9F2hJL`、`mL8ThkuaVTM`、`QaLdnwvtxbs`，3 × 99 回合
> 本轮动作门：目标关联门（`ASSOCIATION_GATE_ENABLE=1`，半径 1.0 m）
> 本轮 shadow：semantic、identity、evidence、active recheck、floor manager 等均为只读日志

## 1. 导航结果

### 1.1 总体

| 指标 | S7 baseline | M1 | M2 | M3 |
| :--- | ---: | ---: | ---: | ---: |
| episodes | 297 | 297 | 297 | 297 |
| success | 141 | 142 | 137 | 142 |
| SR | 47.47% | 47.81% | 46.13% | **47.81%** |
| oracle_success | 138 | 138 | 138 | 138 |
| Oracle SR | 46.46% | 46.46% | 46.46% | **46.46%** |
| SPL | 0.2182 | 0.2207 | 0.2118 | **0.2202** |
| soft SPL | 0.2852 | 0.2838 | 0.2801 | **0.2828** |
| steps mean | 157.04 | 265.87 | 268.19 | **266.23** |
| steps median | 102 | 204 | 208 | **204** |

### 1.2 分场景

| 场景 | episodes | success | SR | Oracle | SPL | steps mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5cdEh9F2hJL | 99 | 52 | 52.53% | 62.63% | 0.2273 | 224.23 |
| QaLdnwvtxbs | 99 | 81 | 81.82% | 72.73% | 0.3710 | 155.40 |
| mL8ThkuaVTM | 99 | 9 | 9.09% | 4.04% | 0.0624 | 419.06 |

### 1.3 失败原因

| failure_cause | 数量 |
| :--- | ---: |
| success | 142 |
| false_positive | 105 |
| never_saw（likely_infeasible） | 43 |
| never_saw（feasible） | 3 |
| bad_stop_true_positive | 2 |
| timeout_true_positive | 1 |
| false_negative | 1 |

### 1.4 分类别

| 目标类别 | episodes | success | SR | FP |
| :--- | ---: | ---: | ---: | ---: |
| chair | 68 | 46 | 67.6% | 16 |
| bed | 70 | 37 | 52.9% | 25 |
| couch | 73 | 32 | 43.8% | 23 |
| tv | 43 | 21 | 48.8% | 19 |
| toilet | 43 | 6 | 14.0% | 22 |

## 2. 候选与目标关联门

### 2.1 候选规模

| 项目 | 数量 |
| :--- | ---: |
| 候选总数 | 7,532 |
| 候选级 TP | 1,680 |
| 候选级 FP | 5,852 |
| 实际写入记忆 | 2,483 |

### 2.2 拒绝路径

| `reject` | 数量 | TP | FP | 被拒候选中 TP 占比 |
| :--- | ---: | ---: | ---: | ---: |
| `association_gate` | 4,117 | 178 | 3,939 | 4.32% |
| `provisional_gate` | 712 | 141 | 571 | 16.45% |
| `occ` | 220 | 7 | 213 | 3.18% |
| 无拒绝（写入） | 2,483 | 1,354 | 1,129 | 54.53% |
| 合计 | 7,532 | 1,680 | 5,852 | — |

### 2.3 关联状态

| `association_status` | 数量 |
| :--- | ---: |
| `associated` | 1,742 |
| `ambiguous` | 1,274 |
| `unassociated` | 4,516 |
| `association_blocked=True` | 4,312 |
| 其中实际 `reject=association_gate` | 4,117 |
| 被标记阻塞但仍写入 | 195 |

### 2.4 Episode 级观察

- 候选级被关联门拒绝的 4,117 条中，FP 占 95.68%，说明它确实在大量拒绝 FP。
- 但 M3 的 SR 为 142/297，与 M1 的 142/297 完全相同；FP 数也与 M1 的 105 相同。
- 出现“候选级 TP 被阻塞”的 episode 有 94 个；其中 77 个仍然成功。
- 只有“TP 被阻塞、没有 FP 被阻塞”的 episode 仅 1 个，且该 episode 仍然成功。
- 4,117 条 `reject=association_gate` 的候选全部满足 `provisional_gate_state=None`、`provisional_gate_outcome=rejected`、`provisional_gate_passed=False`；它们在原 provisional/new 路径中本来就不会被写入。关联门主要是在改拒绝标签，不是新增了一次有效的准入否决。
- 105 个 FP episode 中，首个错误记忆有 79 个来自 `gd-only`、6 个来自已确认的 `gd-tsp/tsp-confirmed`，8 个来自 `provisional`；关联门拒绝的 4,117 条候选全部是 `association_gd_only=False`，既没有触及这些 `gd-only` 记忆，也没有改写已确认目标。
- 结论：关联门有候选级 FP 拒绝量，但没有转化为 episode 级导航收益；它没有解决“已经升级但真假难判”的目标。

## 3. 语义独立证据门

语义 shadow 只在 `gd_only` 且候选状态为 `confirmed/provisional` 时触发，阈值配置为 `0.10`。

| 指标 | 数值 |
| :--- | ---: |
| semantic shadow 行数 | 190 |
| 覆盖 episode | 147 |
| 唯一候选 | 190 |
| 投影有效 | 119 |
| 投影无效 | 71 |
| `semantic_conflict_shadow=True` | **0** |
| `semantic_conflict_shadow=False` | **190** |
| 有效语义分数数量 | 119 |
| 有效分数最小值 | 0.1166 |
| 有效分数中位数 | 0.2140 |
| 有效分数最大值 | 0.4126 |
| 触发候选中 TP / FP | 138 / 52 |
| 有效投影中 TP / FP | 87 / 32 |

- 119 个有效投影全部高于阈值 `0.10`；没有任何候选被语义门判为冲突。
- TP 与 FP 的冲突数均为 0；语义分数没有形成可用于拒绝的分离。
- 71/190（37.4%）触发候选无法得到有效 crop，进一步降低了可执行性。
- 结论：**语义独立证据门在本轮没有产生任何有效拒绝，判定无效。**

## 4. Identity 独立证据门

`identity_shadow.jsonl` 共 5,951 行，覆盖 823 个 memory、250 个 episode。其判定规则包含 view/parallax 独立性与 `gd_hits>=3`；这里给出 memory 级反事实结果（每个 memory 取最后一条快照）。

| 指标 | 数值 |
| :--- | ---: |
| identity shadow 行数 | 5,951 |
| 覆盖 memory 数 | 823 |
| 覆盖 episode 数 | 250 |
| memory 级 `would_block_promotion=True` | 130 |
| 其中 TP memory | 77 |
| 其中 FP memory | 53 |
| `independent_ok`：TP | 189 / 418 = 45.2% |
| `independent_ok`：FP | 171 / 405 = 42.2% |
| 任一时点会触发 block 的 memory | 339（TP 200 / FP 139） |

- memory 级最终快照中，identity 门会阻塞 77 个 TP 和 53 个 FP，阻塞的 TP 反而更多。
- `independent_ok` 在 TP 与 FP 上仅差约 3 个百分点；identity 特征没有形成可用的真假分离。
- 该门是 shadow-only，但 `would_block_promotion` 已被实际计算；因此这不是“没有执行所以看不出效果”，而是反事实判断本身不利于启用。
- 结论：**Identity 独立证据门会误伤更多 TP，判定无效，放弃使用。**

## 5. 独立证据门（evidence 日志）

`evidence_shadow.jsonl` 覆盖 7,312 条 post-occ 候选。

| 字段 | 取值 | 数量 |
| :--- | :--- | ---: |
| `would_block_promotion` | `None` | 7,312 |
| `independent_ok` | `None` | 7,312 |
| `view_ok` | `None` | 7,312 |
| `depth_ok` | `None` | 7,312 |
| `semantic_conflict_shadow` | `False` | 190 |
| `semantic_conflict_shadow` | `None` | 7,122 |

- 7,312 条证据行中，没有任何一条形成 `would_block_promotion` 判定；独立证据门没有产生可执行的准入决策。
- 该门在本轮只是日志采集，没有可评估的拦截量、TP 保留率或 SR 影响。
- 结论：**独立证据门未形成有效判定，不能证明有效；按“无效或作用不明显即放弃”的标准，放弃使用。**

## 6. 主动复核

主动复核 shadow 的触发条件为：`navigate`、非 STOP eligible、`num_obs < 4`、`n_selectable > 1`。

| 指标 | 数值 |
| :--- | ---: |
| active recheck 行数 | 79,071 |
| `would_recheck=True` 行数 | 1,784 |
| `would_recheck=True` 占 navigate 行比例 | 14.9% |
| 触发 episode | 115 / 297（38.7%） |
| 触发 episode 中 success | 71 |
| 触发 episode 中 false_positive | 43 |
| 触发 episode 中 timeout_true_positive | 1 |
| 触发 episode 中 never_saw | 0 |
| true 行按 episode 结果：success | 934 |
| true 行按 episode 结果：false_positive | 820 |
| true 行按 episode 结果：timeout | 30 |
| 触发时 state=`gd` | 1,575 |
| 触发时 state=`confirmed` | 176 |
| 触发时 state=`provisional` | 33 |
| `n_selectable=2` | 1,197 |
| `n_selectable=3` | 389 |
| `num_obs=1 / 2 / 3` | 476 / 928 / 380 |
| 首次触发 step 中位数 | 76 |

- 触发并不集中于失败 episode：71 个成功 episode 也触发，true 行中 934/1,784（52.4%）来自成功 episode。
- 只覆盖 43/105 个 FP episode；对 46 个 never_saw episode 触发数为 0。
- 该 shadow 只记录“是否会复核”，没有执行复核，也没有记录复核后的记忆状态、停止决策或导航结果；因此无法证明能减少 FP 或提升 SR。
- 结论：**主动复核没有显示出有效的错误选择性，且没有反事实收益证据；判定作用不明显，放弃使用。**

## 7. Floor 机制 shadow（1B + floor manager）

> M3 旧 run 没有 `frontier_sensor_3d`，且 floor manager 误用了固定相机高度；因此以下数据是“输入缺失”的历史结果。新代码已加入 `frontier_sensor_3d` 和 floor-only 动态高度 `floor_camera_height`，需要重新跑一轮才能评估 floor 机制。

### 7.1 floor_shadow

`floor_shadow.jsonl` 共 79,071 行，覆盖 297 个 episode；它是只读的 frontier 高度/楼层代理，不参与动作。

| 指标 | 数值 |
| :--- | ---: |
| 行数 / episode | 79,071 / 297 |
| `frontier_count_2d` 均值 / 最大值 | 2.68 / 14 |
| `frontier_count_3d` 非零行 | **0** |
| `floor_proxy_count>0` 行 | **0** |
| `floor_proxy_changed=True` 行 | **0** |
| `visible_floor_ids` 非空行 | **0** |
| `cross_floor_frontier_count>0` 行 | **0** |
| `would_floor_transition=True` 行 | **0** |
| `transition_candidate_*` 非空行 | **0** |
| `explore_exhausted=True` 行 | 32,452 / 79,071（41.0%） |

- `frontier_z_min/max/median/span` 全部为空；本轮没有 3D frontier 进入 floor shadow。
- 虽然有 32,452 个 `explore_exhausted=True` 帧，但没有产生任何跨层候选或 `would_floor_transition=True`。
- 因此本轮 floor shadow 只能记录 2D frontier 数量，不能验证楼层切换机制。

### 7.2 floor_manager shadow

`floor_manager.jsonl` 共 79,071 行，覆盖 297 个 episode。

| 指标 | 数值 |
| :--- | ---: |
| `floor_count` | 1（全部行） |
| `current_floor_id=0` | 79,071 / 79,071 |
| `current_floor_status=ACTIVE` | 79,071 / 79,071 |
| `current_frontier_3d_count` | 0（全部行） |
| `transition_node_count` | 0（全部行） |
| `would_switch_floor=True` | **0** |
| `switch_reason` | `initial_floor`（全部行） |
| `target_floor_hint` | `None`（全部行） |

- `floor_manager` 在所有 episode 中始终只有 1 个 floor、`current_floor_id=0`，且 `agent_z` 固定为 0.88、`floor_z=0.0`。
- 没有 3D frontier、transition node 或 switch 信号；本轮 floor manager 结果同样是“单层代理”。

### 7.3 结论

- 本轮 floor shadow/floor manager **没有观测到跨层信号**，不能证明 floor 机制有效，也不能据此判定其无效；这是本轮数据不可评估，而不是 floor 机制动作失败。
- 如果后续要继续验证 floor 机制，必须先确保 3D frontier / floor proxy 数据进入 shadow，再进行单层与多层场景的对照。

## 8. 总结判定

| 机制 | 本轮结果 | 判定 |
| :--- | :--- | :--- |
| 语义独立证据门 | 190 次触发，0 次冲突；119 个有效投影全部高于阈值 | **无效，放弃** |
| Identity 独立证据门 | 5,951 行 / 823 memory；memory 级 block 130 个，其中 TP 77、FP 53 | **误伤 TP 更多，放弃** |
| 主动复核 | 1,784 次触发、115 个 episode；成功 episode 触发 71 个，FP episode 仅覆盖 43 个，无复核结果 | **作用不明显，放弃** |
| 目标关联门 | 拒绝 4,117 条候选，95.68% 为 FP；但拒绝项全部原本就不会写入，SR 与 M1 同为 142/297 | **没有增量作用，不作为有效机制保留** |

## 9. 产物

- `outputs/s7_m1a_gate2_2026-11-01_002946/s7_summary.json`
- `outputs/s7_m1a_gate2_2026-11-01_002946/s_evidence.json`
- `outputs/s7_m1a_gate2_2026-11-01_002946/gd_offline_cal.json`
- `outputs/s7_m1a_gate2_2026-11-01_002946/detections/candidates.jsonl`
- `outputs/s7_m1a_gate2_2026-11-01_002946/detections/semantic_shadow.jsonl`
- `outputs/s7_m1a_gate2_2026-11-01_002946/detections/evidence_shadow.jsonl`
- `outputs/s7_m1a_gate2_2026-11-01_002946/detections/active_recheck_shadow.jsonl`
- `outputs/s7_m1a_gate2_2026-11-01_002946/detections/memory_events.jsonl`
- `outputs/s7_m1a_gate2_2026-11-01_002946/detections/episodes.jsonl`
