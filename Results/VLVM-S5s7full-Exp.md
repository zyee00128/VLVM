# S7 full 实验结果：evidence 选择 + `num_obs≥3` 停止 + 视角扩张复核

## 1. 口径

| 项 | 值 |
| :--- | :--- |
| run dir | `outputs/s5_s7full_2026-10-25_140835/` |
| 基准 | S6（merge 新信息门 + `stable` 删除池） |
| provisional 准入 | $0.50\le\sigma_c<0.60$，$1.0\le dist\le4.0$ m |
| 目标选择 | evidence 分层 → freshness → 距离 → $\sigma_c$；过滤 `stable` / `stop_blocked` / stale>30 |
| 停止门槛 | `confirmed` + $n_{\text{obs}}\ge3$ + 本帧重确认 |
| 视角扩张 | 转 30° → 后退 0.4 m（2 步）→ 复核；总动作 ≤5；每集 ≤4 次；失败 → `stop_blocked` + `stable` |
| merge / `stable` | 与 S6 相同 |
| 场景 / 集数 | 3 × 99 = 297 |
| 仿真 | 是 |

## 2. 验收

| 指标 | 目标 | S7 full | 判定 |
| :--- | :--- | ---: | :--- |
| FP | ≤80 | 75 | 达成 |
| success | ≥121 | 106 | 未达 |
| never_saw 合计 | ≤80 | 89（25 + 64） | 未达 |

## 3. 导航总体

| 配置 | success | SR ↑ | Oracle ↑ | SPL ↑ | SoftSPL ↑ | steps ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 G4 baseline | 116 | 39.06% | 43.43% | 0.1608 | 0.2369 | 172.5724 |
| S5 P1 全量 baseline | 398 | 40.20% | 54.04% | 0.1586 | — | 200.7600 |
| S5 P1 三场景 baseline | 114 | 38.38% | 43.43% | 0.1597 | 0.2364 | 173.5960 |
| S5-S1 | 115 | 38.72% | 43.43% | 0.1607 | 0.2382 | 171.1953 |
| S5-S4-1 | 114 | 38.38% | 43.43% | 0.1582 | 0.2346 | 171.8047 |
| S5-S2 | 121 | 40.74% | 42.09% | 0.1798 | 0.2690 | 144.8148 |
| S5 arm A | 121 | 40.74% | 43.43% | 0.1797 | 0.2591 | 156.6 |
| S5 arm B | 116 | 39.06% | 40.74% | 0.1788 | 0.2695 | 149.2 |
| S6 | 117 | 39.39% | 41.08% | 0.1803 | 0.2741 | 146.4007 |
| S7 full | 106 | 35.69% | 45.45% | 0.1380 | 0.2141 | 216.8620 |

## 4. 逐场景

| 场景 | S7 full success | S7 full SR | S7 0.52 success | S6 success | arm B success | arm A success |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 35 | 35.35% | 38 | 39 | 40 | 42 |
| `QaLdnwvtxbs` | 61 | 61.62% | 67 | 67 | 65 | 68 |
| `mL8ThkuaVTM` | 10 | 10.10% | 11 | 11 | 11 | 11 |

## 5. 逐类别

| 类别 | S7 full success / SR | S7 0.52 success / SR | S6 success / SR | arm B success / SR |
| :--- | ---: | ---: | ---: | ---: |
| `bed`（70） | 34 / 48.57% | 38 / 54.29% | 39 / 55.71% | 39 / 55.71% |
| `chair`（68） | 41 / 60.29% | 34 / 50.00% | 34 / 50.00% | 33 / 48.53% |
| `couch`（73） | 18 / 24.66% | 25 / 34.25% | 25 / 34.25% | 24 / 32.88% |
| `tv`（43） | 11 / 25.58% | 17 / 39.53% | 17 / 39.53% | 18 / 41.86% |
| `toilet`（43） | 2 / 4.65% | 2 / 4.65% | 2 / 4.65% | 2 / 4.65% |

## 6. 失败构成

| `failure_cause` | S7 full | S7 0.52 | S6 | arm B |
| :--- | ---: | ---: | ---: | ---: |
| `did_not_fail` | 106 | 116 | 117 | 116 |
| `false_positive` | 75 | 80 | 86 | 92 |
| `false_negative` | 24 | 15 | 15 | 13 |
| `never_saw_target_did_not_travel_stairs_feasible` | 25 | 22 | 20 | 18 |
| `never_saw_target_did_not_travel_stairs_likely_infeasible` | 64 | 64 | 59 | 58 |
| `timeout_true_positive` | 3 | 0 | 0 | 0 |

## 7. 逐集配对

| 对比 | 前者 success | 后者 success | 仅前者成功 | 仅后者成功 | 两者均成功 | 两者均失败 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| S7 full vs S7 0.52 | 106 | 116 | 17 | 27 | 89 | 164 |
| S7 full vs S6 | 106 | 117 | 18 | 29 | 88 | 162 |
| S7 full vs arm B | 106 | 116 | 19 | 29 | 87 | 162 |
| S7 full vs arm A | 106 | 121 | 13 | 28 | 93 | 163 |
| S7 full vs S2U | 106 | 121 | 16 | 31 | 90 | 160 |

## 8. 停止门槛

| 指标 | S7 full | S7 0.52 | S6 | arm B | arm A |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `stop_allowed=True` | 142 | 187 | 189 | 187 | 186 |
| `stop_allowed=False`（被门槛拦截） | 0 | 156 | 212 | 212 | 88 |
| 含实际停止的集数（`stops=1`） | 141 | 186 | 188 | 186 | 185 |

## 9. 生命周期

| 事件 | S7 full | S7 0.52 | S6 |
| :--- | ---: | ---: | ---: |
| `new` | 1,072 | 465 | 511 |
| — `confirmed` | 340 | 134 | 127 |
| — `provisional` | 732 | 331 | 384 |
| `merge` | 4,417 | 1,116 | 1,241 |
| `upgrade`（TP） | 1,586（830） | 220（110） | 254（112） |
| `expand`（pass / fail） | 504（97 / 407） | — | — |
| `transition` | 1,997 | 155 | 188 |
| `delete`（TP） | 635（232） | 83（26） | 111（35） |
| `revive`（TP） | 1,194（659） | 20（14） | 22（14） |

**new 状态**

| 状态 | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `confirmed` | 340 | 179 | 161 | 52.65% |
| `provisional` | 732 | 279 | 453 | 38.11% |

**merge 新信息门**

| `new_info` | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `True` | 3,592 | 2,256 | 1,336 | 62.81% |
| `False` | 825 | 312 | 513 | 37.82% |
| 去重轨道（`False`） | 205 | 103 | 102 | 50.24% |

**状态迁移**

| 迁移 / 原因 | n | TP |
| :--- | ---: | ---: |
| `confirmed → stable` / `expand_blocked` | 291 | 146 |
| `confirmed → stable` / `expand_budget_exhausted` | 1,239 | 661 |
| `confirmed → stable` / `near_miss_downgrade` | 130 | 57 |
| `confirmed → stable` / `stale_demote` | 43 | 15 |
| `provisional → stable` / `expand_blocked` | 90 | 21 |
| `provisional → stable` / `expand_budget_exhausted` | 78 | 29 |
| `provisional → stable` / `near_miss_downgrade` | 101 | 32 |
| `provisional → stable` / `stale_demote` | 25 | 5 |

**删除**

| `delete_reason` | n | TP | TP 占比 | 占 delete 总量 |
| :--- | ---: | ---: | ---: | ---: |
| `stable_stale_purge` | 385 | 135 | 35.06% | 60.63% |
| `stable_miss_delete` | 204 | 66 | 32.35% | 32.13% |
| `stable_total_miss` | 46 | 31 | 67.39% | 7.24% |
| 合计 | 635 | 232 | 36.54% | 100.00% |

**复活**

| 项 | 数值 |
| :--- | ---: |
| `revive` 事件 | 1,194 |
| 其中 TP | 659（55.19%） |
| `revive_count` 中位数 | 38 |
| `revive_count` 最大值 | 180 |

**`stable` 轨道级**

| 项 | TP | FP |
| :--- | ---: | ---: |
| 进入 `stable` | 352 | 520 |
| 复活 | 107 | 61 |
| 删除 | 179 | 375 |
| 停泊至集末 | 66 | 84 |

## 10. confirmed 池

| 项 | S7 full | S7 0.52 | S6 | arm B |
| :--- | ---: | ---: | ---: | ---: |
| `new` 中 `confirmed` FP | 161 | 65 | 61 | 61 |
| `upgrade` 事件 FP | 756 | 110 | 142 | 147 |
| `upgrade` 去重轨道 / TP | 505 / 242 | 212 / 106 | 244 / 109 | 249 / 102 |
| confirmed 池 FP（事件口径） | 917 | 175 | 203 | 208 |
| confirmed 池 FP（轨道去重） | 401 | 170 | 197 | 208 |
| 存活至集末的 confirmed FP | 156 | 148 | 168 | 154 |
| 已删除的 confirmed 轨道 | 427 | 37 | 48 | 82 |

注：`upgrade` 事件数 1,586 高于去重轨道 505，原因是同一轨道经 `stable → provisional → confirmed` 反复复活/升级；事件口径精度 52.33%，去重口径 47.92%。

## 11. nav goal 与准入

**准入漏斗**

| 阶段 | n | TP | precision |
| :--- | ---: | ---: | ---: |
| 候选 | 10,856 | 4,021 | 37.04% |
| provisional gate：`confirmed` | 340 | 179 | 52.65% |
| provisional gate：`provisional` | 732 | 279 | 38.11% |
| provisional gate：`merge_existing` | 4,417 | 2,568 | 58.14% |
| provisional gate：`rejected` | 5,128 | 989 | 19.29% |
| provisional gate：`missing` | 239 | 6 | 2.51% |
| 写入 / merge（`admitted`） | 5,489 | 3,026 | 55.13% |
| 其中 `s_penalty_admitted` | 1,985 | — | — |

**episode 分组**

| 分组 | S7 full 集数 | S7 full success | S6 集数 | S6 success |
| :--- | ---: | ---: | ---: | ---: |
| 有准入 TP、无准入 FP | 42 | 29 | 58 | 47 |
| 有准入 TP、也有准入 FP | 124 | 74 | 72 | 57 |
| 只有准入 FP | 55 | 1 | 91 | 11 |
| 无任何准入候选 | 76 | 2 | 76 | 2 |

**nav goal**

| 指标 | S7 full | S7 0.52 | S6 | arm B |
| :--- | ---: | ---: | ---: | ---: |
| nav goal 行数 | 10,951 | 4,465 | 4,797 | 5,777 |
| nav goal TP 占比 | 46.70% | 47.26% | 45.38% | 37.60% |
| 有 nav goal 的集数 | 221 | 211 | 221 | 221 |

**每集被选轨道数**

| 轨道数 | 0 | 1 | 2 | 3 | 4 | ≥5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 集数 | 76 | 56 | 36 | 23 | 23 | 83 |

**最后一条 nav goal 的状态**

| 状态 | 集数 | TP | 命中率 |
| :--- | ---: | ---: | ---: |
| `confirmed` | 170 | 116 | 68.24% |
| `provisional` | 30 | 9 | 30.00% |
| `stable` | 21 | 9 | 42.86% |

## 12. 离线标定摘要

| 项 | S7 full |
| :--- | ---: |
| 停止规则 `num_obs≥3`：允许 / TP / FP | 157 / 108 / 49 |
| 停止规则 `num_obs≥3 且 confirmed` | 148 / 103 / 45 |
| 停止规则 `num_obs≥3 且 recent3` | 150 / 106 / 44 |
| 停止规则 `num_obs≥3 且 recent15` | 143 / 102 / 41 |
| 选择回放：同轨道子集实际 TP | 4,030 |
| 选择回放：同轨道子集 evidence TP | 4,031 |
| 最后 nav goal：实际 TP | 134 |
| 最后 nav goal：evidence Top-1 TP | 122 |
| merge 严格门 0.30/15/4：provisional `num_obs≥2` | 293 条，TP 166，precision 56.66% |
| upgrade `sigma_c_latest≥0.52` | 1,370 条，TP 708，precision 51.68% |
| upgrade `sigma_c_ema_05≥0.52` | 1,395 条，TP 718，precision 51.47% |
| `expand`（pass / fail） | 504（97 / 407） |
| `expand` 相关 transition | `expand_blocked` 381 + `expand_budget_exhausted` 1,317 |
| `revive` 最大 `revive_count` | 180 |
| 准入 0.55 / 1.5–4.0 m | 2,568 条，TP 1,513，precision 58.9%，有 TP 集 149 |

完整回放见 `Results/VLVM-S5cal-Exp.md` §10。

## 13. 运行信息

| 项 | 值 |
| :--- | :--- |
| 完成标记 | launcher `S7 完成，用时 25135s`（无 `COMPLETED` 文件） |
| 耗时 | 25,135 s（6:58:55） |
| 检测日志 | `detections/{candidates,memory_events,nav_goals,track_frames,episodes}.jsonl` |
| `track_frames` 行数 | 46,237 |
| 离线标定输出 | `offline_cal.json`、`offline_cal_s7full.json` |
