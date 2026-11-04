# Project VLVM Stage 7 M2：同目标关联门 + 目标类语义复核门

> 运行目录：`outputs/s7_gates_3scene_2026-10-31`
> 对照：R0 `outputs/s7_baseline_3scene_2026-10-30_004608`；M1 `outputs/s7_m1a_3scene_2026-10-30_163357`
> 场景：`5cdEh9F2hJL`、`mL8ThkuaVTM`、`QaLdnwvtxbs`，3 × 99 回合
> 运行耗时：11:46:13（时:分:秒）；`COMPLETED` 与 `s7_summary.json` 已生成
> 本轮动作改动：1A + 同目标关联门 + 目标类语义复核门；其余机制保持 shadow。

## 1. 启动配置

```bash
RUN_DIR=outputs/s7_gates_3scene_2026-10-31 \
FLOOR_SHADOW_ENABLE=1 \
IDENTITY_SHADOW_ENABLE=1 \
ASSOCIATION_GATE_ENABLE=1 \
SEMANTIC_GATE_ENABLE=1 \
SEMANTIC_CONFLICT_THRESHOLD=0.10 \
bash scripts/_nav_s7.sh
```

启动日志中的 gate 配置：

```text
1B floor shadow: enable=1 3d_every_n=5 (read-only)
1C identity shadow: enable=1 (read-only)
1C gates: association=1 semantic=1 threshold=0.10
```

代码位置：

- `vlfm/shadow/semantic_gate.py`
- `vlfm/shadow/floor_shadow.py`
- `vlfm/shadow/frontier_3d.py`
- `vlfm/shadow/identity_shadow.py`
- `vlfm/policy/tsp3d_objectnav_policy.py`
- `vlfm/tsp3d_models/fussion_utils/target_memory_manager.py`

## 2. 导航结果

### 2.1 总体

| 指标 | R0 baseline | M1（1A + shadow） | M2（1A + gates） |
| :--- | ---: | ---: | ---: |
| episodes | 297 | 297 | 297 |
| success | 141 | 142 | 137 |
| SR | 47.47% | 47.81% | 46.13% |
| oracle_success | 138 | 138 | 138 |
| SPL | 0.2182 | 0.2207 | 0.2118 |
| soft SPL | 0.2852 | 0.2838 | 0.2801 |
| steps mean | 157.04 | 265.87 | 268.19 |
| steps median | 102 | 204 | 208 |

### 2.2 分场景

| 场景 | R0 SR | M1 SR | M2 SR | M2 success | M2 steps mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 5cd | 52.53% | 52.53% | 51.52% | 51/99 | 223.93 |
| QaL | 80.81% | 81.82% | 78.79% | 78/99 | 161.55 |
| mL8 | 9.09% | 9.09% | 8.08% | 8/99 | 419.08 |

### 2.3 失败类别

| failure_cause | R0 | M1 | M2 |
| :--- | ---: | ---: | ---: |
| false_positive | 100 | 105 | 110 |
| never_saw likely_infeasible | 48 | 43 | 43 |
| never_saw feasible | 3 | 3 | 3 |
| bad_stop_true_positive | 2 | 2 | 2 |
| timeout_true_positive | 2 | 1 | 1 |
| false_negative | 1 | 1 | 1 |

### 2.4 最终动作

| 项目 | R0 | M2 |
| :--- | ---: | ---: |
| 最终 `explore/0` | 85 | 0 |
| 最终 `explore` 总数 | 107 | 105 |
| 最终 `navigate` 总数 | 190 | 192 |
| `Action=0` 总数 | 272 | 188 |
| `Action=2` 总数 | 7 | 87 |
| `Action=3` 总数 | 0 | 2 |

## 3. Gate 触发记录

### 3.1 候选级统计

`candidates.jsonl`：7,822 条。

| 字段/值 | 数量 |
| :--- | ---: |
| `reject=provisional_gate` | 4,977 |
| `reject=occ` | 224 |
| `reject=promotion_gate` | 96 |
| `reject=null` | 2,525 |
| `association_ambiguous=True` | 251 |
| `promotion_blocked=True` | 251 |
| `semantic_conflict=True` | 0 |
| `semantic_score` 非空 | 0 |

### 3.2 记忆升级

| 升级原因 | 数量 |
| :--- | ---: |
| `base_upgrade` | 37 |
| `gd_tsp` | 59 |
| `gd_only_tsp_merge` | 136 |
| 合计 | 232 |

## 4. 1A/探索 shadow

### 4.1 STOP 抑制

| 结果 | episodes | 发生抑制的 episode | 抑制步数 |
| :--- | ---: | ---: | ---: |
| success | 137 | 1 | 7 |
| false_positive | 110 | 38 | 12,996 |
| never_saw | 46 | 45 | 18,979 |
| 其他失败 | 4 | 0 | 0 |

- 1A 抑制步数合计：31,982。
- 最终 `explore/0`：0。

### 4.2 最终模式/动作

- 最终 `explore`：105。
- 最终 `navigate`：192。
- 最终 `Action=0`：188。
- 最终 `Action=1`：20。
- 最终 `Action=2`：87。
- 最终 `Action=3`：2。

## 5. 1B floor shadow

### 5.1 数据规模

| 文件 | 行数 |
| :--- | ---: |
| `floor_shadow.jsonl` | 79,651 |
| `field_shadow.jsonl` | 79,651 |
| `identity_shadow.jsonl` | 6,048 |
| `stop_shadow.jsonl` | 12,251 |
| `middle_rank_shadow.jsonl` | 17,022 |
| `memory_events.jsonl` | 9,012 |
| `nav_goals.jsonl` | 12,793 |
| `track_frames.jsonl` | 101,631 |

### 5.2 3D frontier 有效性

| 项目 | 数值 |
| :--- | ---: |
| `floor_data_valid=True` 行数 | 79,406 |
| `frontier_z_source=3d_sensor` 行数 | 79,406 |
| `frontier_z_source=none` 行数 | 245 |
| `frontier_count_3d` 最大值 | 2,048 |
| `floor_proxy_count` 最大值 | 1 |
| `visible_floor_ids` 最大数量 | 1 |
| `would_floor_transition=True` 行数 | 0 |
| `floor_proxy_changed=True` 行数 | 0 |
| `cross_floor_frontier_count>0` 行数 | 0 |

### 5.3 按 episode 结果

| 结果 | episodes | `explore_exhausted` | 抑制 episode | 抑制步数 | 平均步数 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| success | 137 | 1 | 1 | 7 | 140.22 |
| false_positive | 110 | 38 | 38 | 12,996 | 327.59 |
| never_saw | 46 | 45 | 45 | 18,979 | 500.00 |
| 其他失败 | 4 | 0 | 0 | 0 | 351.50 |

## 6. 1C identity shadow

### 6.1 数据规模

| 项目 | 数值 |
| :--- | ---: |
| identity shadow 行数 | 6,048 |
| 去重 memory 数 | 835 |
| TP 记忆 | 422 |
| 非 TP 记忆 | 413 |
| 提升事件（有快照） | 199 |
| `stop_final` 行数 | 251 |
| `stop_events` 行数 | 230 |

### 6.2 TP / 非 TP

| 特征 | TP（422） | 非 TP（413） |
| :--- | ---: | ---: |
| `gd_hits` 均值 | 6.16 | 5.06 |
| `view_count` 均值 | 6.30 | 5.31 |
| `distinct_pose_count` 均值 | 5.48 | 4.62 |
| `distinct_yaw_count` 均值 | 1.89 | 2.12 |
| `same_object_views` 均值 | 5.40 | 3.83 |
| `max_parallax_m` 均值 | 1.74 | 2.01 |
| `independent_ok` | 191/422 = 45.3% | 178/413 = 43.1% |
| `would_block_promotion` | 77/422 = 18.2% | 54/413 = 13.1% |

AUC（TP vs 非 TP）：

| 特征 | AUC |
| :--- | ---: |
| `gd_hits` | 0.585 |
| `view_count` | 0.568 |
| `distinct_pose_count` | 0.568 |
| `distinct_yaw_count` | 0.456 |
| `same_object_views` | 0.606 |
| `max_parallax_m` | 0.497 |

### 6.3 提升时点

| 组 | 数量 | `would_block_promotion` |
| :--- | ---: | ---: |
| TP promotions | 143 | 21（14.7%） |
| 非 TP promotions | 56 | 6（10.7%） |

### 6.4 按来源

| 来源 | memories | TP | `independent_ok` | `would_block` |
| :--- | ---: | ---: | ---: | ---: |
| gd-only | 515 | 204 | 211 | 77 |
| gd-tsp | 136 | 100 | 64 | 34 |
| gd-only-promoted | 129 | 89 | 93 | 19 |
| tsp-confirmed | 33 | 18 | 0 | 1 |
| tsp-provisional | 22 | 11 | 1 | 0 |

### 6.5 STOP 快照

| 组 | 数量 | stop_tp | goal_tp | `independent_ok` | `would_block` | `gd_hits` 均值 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 错误 STOP（false_positive，final action 0） | 49 | 12 | 1 | 31 | 9 | 7.04 |
| TP STOP（success，final action 0） | 137 | 132 | 98 | 79 | 30 | 7.86 |
| 其他 STOP 快照（含 final action 非 0 的 FP） | 65 | 21 | 19 | 15 | 8 | 2.58 |

> 上表为 `stop_final` 快照行（251 行），不是全部 297 回合。

## 7. 产物

- `outputs/s7_gates_3scene_2026-10-31/s7_summary.json`
- `outputs/s7_gates_3scene_2026-10-31/s7m2_analysis.json`
- `outputs/s7_gates_3scene_2026-10-31/detections/episodes.jsonl`
- `outputs/s7_gates_3scene_2026-10-31/detections/candidates.jsonl`
- `outputs/s7_gates_3scene_2026-10-31/detections/floor_shadow.jsonl`
- `outputs/s7_gates_3scene_2026-10-31/detections/identity_shadow.jsonl`
- `outputs/s7_gates_3scene_2026-10-31/detections/stop_shadow.jsonl`
- `outputs/s7_gates_3scene_2026-10-31/detections/middle_rank_shadow.jsonl`
- 分析脚本：`scripts/analyze_s7m1.py`
