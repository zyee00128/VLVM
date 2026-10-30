# VLVM-S6gd5：gd-only 优先排序 + gd-only depth 放开三场景结果

> 运行目录：`outputs/s6_gdfinal_gdonlyfirst_2026-10-29_162544`  
> 对照版本：S6gd3（`outputs/s6_gd3_restart_2026-10-28_182915`）  
> 场景：`5cdEh9F2hJL` / `mL8ThkuaVTM` / `QaLdnwvtxbs`，共 `3 × 99 = 297` episodes  
> 状态：`COMPLETED`

## 1. 本版配置

| 项 | S6gd5 实际运行 |
| :--- | :--- |
| 选择层级 | `gd-tsp（含 promoted gd-only） > gd-only > confirmed > provisional` |
| STOP | 仅 `gd-tsp confirmed` 可停；`pointnav_stop_radius=0.90 m` |
| `gd_depth_failed_track_merge` | `true` |
| `gd_depth_failed_gd_only` | `true` |
| `enable_gd_ttl` | `false`；gd-only 不删除 |
| `enable_gd_only_conf_gate` | `false` |
| `occ` gate | `occ_passed` |
| stable | 命中只写 `gd_stable_shadow`，不升级 |

与 S6gd3 相比，深度相关配置由全拒绝变为：

- confirmed/provisional track 可用 track centroid 合并；
- gd-only new/merge 可进入 gd-only。

## 2. 总体结果

| 指标 | S6gd3 | S6gd5 | 差值 |
| :--- | ---: | ---: | ---: |
| success | **139 / 297** | **134 / 297** | **−5** |
| SR | 46.80% | 45.12% | −1.68 pp |
| Oracle | 46.46% | 46.13% | −0.34 pp |
| SPL | 0.2157 | 0.2086 | −0.0071 |
| SoftSPL | 0.2867 | 0.2814 | −0.0052 |
| steps | 153.55 | 159.17 | +5.63 |

## 3. 分场景

| 场景 | S6gd3 success / Oracle / steps | S6gd5 success / Oracle / steps | success Δ |
| :--- | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 51 / 62 / 213.75 | 46 / 62 / 228.94 | **−5** |
| `QaLdnwvtxbs` | 80 / 72 / 152.92 | 79 / 73 / 153.62 | −1 |
| `mL8ThkuaVTM` | 8 / 4 / 93.97 | 9 / 4 / 94.96 | +1 |

## 4. 失败原因

| failure_cause | S6gd3 | S6gd5 | 差值 |
| :--- | ---: | ---: | ---: |
| `did_not_fail` | 139 | 134 | −5 |
| `false_positive` | 102 | 107 | +5 |
| `bad_stop_true_positive` | 2 | 1 | −1 |
| `timeout_true_positive` | 2 | 3 | +1 |
| `false_negative` | 1 | 1 | 0 |
| `never_saw_*_infeasible` | 48 | 48 | 0 |
| `never_saw_*_feasible` | 3 | 3 | 0 |

## 5. 最终目标来源

有最终 `nav_goal` 记录的 episode 为 245。

| 最终来源 | S6gd3 goals / TP / success | S6gd5 goals / TP / success |
| :--- | ---: | ---: |
| `gd-tsp` | 187 / 144 / 138 | 187 / 142 / 133 |
| `gd-only` | 53 / 16 / 0 | 52 / 17 / 0 |
| 纯 TSP `confirmed` | 1 / 0 / 0 | 2 / 0 / 0 |
| `provisional` | 4 / 0 / 0 | 4 / 0 / 0 |

gd-tsp 内部：

| 子类 | S6gd3 goals / TP / success | S6gd5 goals / TP / success |
| :--- | ---: | ---: |
| track gd-tsp | 92 / 67 / 60 | 89 / 64 / 57 |
| promoted gd-only | 95 / 77 / 78 | 98 / 78 / 76 |

## 6. depth 放开数据

S6gd5 的 gd-only depth 事件：

| 事件 | depth_relaxed | n | TP | FP |
| :--- | :---: | ---: | ---: | ---: |
| `gd_only_new` | false | 681 | 272 | 409 |
| `gd_only_new` | true | 105 | 55 | 50 |
| `gd_only_merge` | false | 3,361 | 1,820 | 1,541 |
| `gd_only_merge` | true | 142 | 61 | 81 |
| **合计** | true | **247** | **116** | **131** |

涉及 **90 个 episode**，事件按场景：

| 场景 | depth episodes | gd3 success | gd5 success | relaxed events | TP | FP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 47 | 19 | 14 | 149 | 34 | 115 |
| `QaLdnwvtxbs` | 38 | 31 | 30 | 91 | 78 | 13 |
| `mL8ThkuaVTM` | 5 | 2 | 3 | 7 | 4 | 3 |
| **合计** | **90** | **52** | **47** | **247** | **116** | **131** |

非 depth 事件 episode 的 success：

| 范围 | S6gd3 | S6gd5 | 差值 |
| :--- | ---: | ---: | ---: |
| depth-affected episodes | 52 | 47 | **−5** |
| 其他 episodes | 87 | 87 | 0 |

## 7. STOP 数据

| 指标 | S6gd3 | S6gd5 |
| :--- | ---: | ---: |
| 实际 `Action: 0` | 274 | 267 |
| `stop_shadow` 行数 | 10,960 | 12,496 |
| `stop_eligible_current=true` | 2,559 | 2,735 |
| eligible 来源 | 全部 gd-tsp | 全部 gd-tsp |
| `bad_stop_true_positive` | 2 | 1 |
| `timeout_true_positive` | 2 | 3 |

STOP 规则没有变化，仍是仅 `gd-tsp confirmed` 可停。停止相关失败数：

- S6gd3：`bad_stop 2 + timeout 2 = 4`
- S6gd5：`bad_stop 1 + timeout 3 = 4`

## 8. episode 翻转

| 场景 / episode | S6gd3 → S6gd5 | S6gd5 failure | wrong stop | depth events | S6gd5 最终来源 / TP |
| :--- | :--- | :--- | :---: | ---: | :--- |
| `5cdEh9F2hJL` / 11 | success → failure | false_positive | false | 1 | gd-tsp / TP |
| `5cdEh9F2hJL` / 21 | success → failure | false_positive | true | 4 | gd-tsp / FP |
| `5cdEh9F2hJL` / 24 | success → failure | false_positive | true | 9 | provisional / FP |
| `5cdEh9F2hJL` / 58 | success → failure | false_positive | false | 2 | gd-tsp / FP |
| `5cdEh9F2hJL` / 64 | success → failure | false_positive | true | 3 | gd-tsp / FP |
| `5cdEh9F2hJL` / 72 | success → failure | false_positive | true | 2 | gd-tsp / FP |
| `QaLdnwvtxbs` / 61 | success → failure | false_positive | false | 6 | gd-tsp / TP |
| `QaLdnwvtxbs` / 89 | success → failure | false_positive | false | 6 | gd-tsp / FP |
| `5cdEh9F2hJL` / 18 | failure → success | did_not_fail | — | 6 | gd-tsp / TP |
| `QaLdnwvtxbs` / 82 | failure → success | did_not_fail | — | 1 | gd-tsp / TP |
| `mL8ThkuaVTM` / 4 | failure → success | did_not_fail | — | 3 | gd-tsp / TP |

净变化：8 个 success → failure，3 个 failure → success，合计 **−5**。

## 9. 归因判断

| 因素 | 数据 |
| :--- | :--- |
| depth | 8 个 success→failure 均落在 depth-affected episodes（另有 3 个 failure→success）；depth-affected success 52→47，其他 episode 87→87 |
| stop | STOP 规则未变；stop 相关失败 4→4；8 个退化 episode 全部是 `false_positive`，无 `bad_stop`/`timeout` 退化 |
| 5cd 集中 | 5cd 的 depth 事件 149 条，FP 115；5cd success 51→46，贡献 −5 |
| 事件质量 | gd-only depth-relaxed 事件 TP 116 / FP 131；`gd_only_merge` 的 relaxed FP 81 条 |

**判断：这 5 集退化主要来自 depth 放开，不是 STOP 规则。**

但 S6gd5 同时包含目标排序内部 key 的代码校正（gd-only 用 `gd_hits`，confirmed 用 `sigma`）；它不是 STOP，也不是 depth。现有结果只能说明退化和 depth-affected episodes 完全重合，尚不能严格排除排序 key 校正与 depth 的交互。

## 10. 结论

1. S6gd5 为 134/297，较 S6gd3 的 139/297 少 5 集。
2. 分场景变化为：5cd −5、QaL −1、mL8 +1。
3. depth-relaxed gd-only 事件 247 条，涉及 90 个 episode；其中 gd-only merge 的 relaxed FP 为 81 条。
4. 所有净退化都发生在 depth-affected episodes；非 depth episodes 的 success 保持 87→87。
5. STOP 规则和停止相关失败数没有解释这 5 集退化；实际停止数 274→267，bad_stop 2→1，timeout 2→3。
6. 因此当前数据支持“depth 放开是主要退化因素，STOP 不是主要因素”。
