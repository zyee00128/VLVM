# S6 实验结果：merge 新信息门 + `stable` 删除池

## 1. 口径

| 项 | 值 |
| :--- | :--- |
| run dir | `outputs/s5_s6_2026-10-24_164308/` |
| 基准 | S5 arm B |
| provisional 准入 | $0.50\le\sigma_c<0.60$，$1.0\le dist\le4.0$ m |
| merge 新信息门 | 位姿基线 ≥0.2 m，或朝向差 ≥10°，或间隔 ≥3 步 |
| merge 到 `stable` | 禁止；只走复活 |
| merge 内升级 | `close_reobservation` 关闭 |
| 生命周期 | `confirmed` / `provisional` 3 次 `near_miss` → `stable`；无直接删除 |
| `stable` 删除 | 陈旧 ≥60 步 / 连续 `near_miss` ≥3 / 累计 `near_miss` ≥6 |
| `stable` 复活 | 0.5 m / 10 步 / $\sigma_c\ge0.50$，保留 `num_obs` 与分数 |
| 选择 | `stable` 不进 nav goal；`provisional` 可追、不可停 |
| 停止 | 仅 `confirmed` 可停；STOP 0.90 m |
| 场景 / 集数 | 3 × 99 = 297 |
| 仿真 | 是 |

## 2. 验收

| 指标 | 目标 | S6 | 判定 |
| :--- | :--- | ---: | :--- |
| 升级 precision | ≥50% | 44.09%（112 / 254） | 未达 |
| TP 升级数 | ≥85 | 112 | 达成 |
| confirmed 池 FP | ≤170 | 203 | 未达 |
| delete 总量 | ≤120 | 111 | 达成 |
| success | ≥116 | 117 | 达成 |
| never_saw 合计 | ≤76 | 79（20 + 59） | 未达 |

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

arm A 的 provisional 准入是 $$\sigma_c \in {0.55, 0.60}$$，arm B / S6 的 provisional 准入是 $$\sigma_c \in {0.50, 0.60}$$，这是相差几集的最主要原因，{0.50, 0.60} 引入了更多被选择的 TP 目标。

## 4. 逐场景

| 场景 | S6 success | S6 SR | arm B success | arm B SR | arm A success | S2U success |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 39 | 39.39% | 40 | 40.40% | 42 | 44 |
| `QaLdnwvtxbs` | 67 | 67.68% | 65 | 65.66% | 68 | 67 |
| `mL8ThkuaVTM` | 11 | 11.11% | 11 | 11.11% | 11 | 10 |

## 5. 逐类别

| 类别 | S6 success / SR | arm B success / SR | arm A success / SR | S2U success |
| :--- | ---: | ---: | ---: | ---: |
| `bed`（70） | 39 / 55.71% | 39 / 55.71% | 39 / 55.71% | 38 |
| `chair`（68） | 34 / 50.00% | 33 / 48.53% | 39 / 57.35% | 39 |
| `couch`（73） | 25 / 34.25% | 24 / 32.88% | 27 / 36.99% | 24 |
| `tv`（43） | 17 / 39.53% | 18 / 41.86% | 13 / 30.23% | 13 |
| `toilet`（43） | 2 / 4.65% | 2 / 4.65% | 3 / 6.98% | 7 |

## 6. 失败构成

| `failure_cause` | S6 | arm B | arm A | S2U |
| :--- | ---: | ---: | ---: | ---: |
| `did_not_fail` | 117 | 116 | 121 | 121 |
| `false_positive` | 86 | 92 | 72 | 79 |
| `false_negative` | 15 | 13 | 19 | 13 |
| `never_saw_target_did_not_travel_stairs_feasible` | 20 | 18 | 21 | 20 |
| `never_saw_target_did_not_travel_stairs_likely_infeasible` | 59 | 58 | 64 | 64 |

## 7. 逐集配对

| 对比 | 前者 success | 后者 success | 仅前者成功 | 仅后者成功 | 两者均成功 | 两者均失败 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| S6 vs arm B | 117 | 116 | 3 | 2 | 114 | 178 |
| S6 vs arm A | 117 | 121 | 15 | 19 | 102 | 161 |
| S6 vs S2U | 117 | 121 | 17 | 21 | 100 | 159 |

## 8. 停止门槛

| 指标 | S6 | arm B | arm A | S2U |
| :--- | ---: | ---: | ---: | ---: |
| `stop_allowed=True` | 189 | 187 | 186 | 193 |
| `stop_allowed=False`（被门槛拦截） | 212 | 212 | 88 | 0 |
| 含实际停止的集数（`stops=1`） | 188 | 186 | 185 | 192 |

## 9. 生命周期

| 事件 | S6 | arm B | arm A | S2U |
| :--- | ---: | ---: | ---: | ---: |
| `new` | 511 | 513 | 374 | 312 |
| — `confirmed` | 127 | 125 | 169 | 164 |
| — `provisional` | 384 | 388 | 205 | 148 |
| `merge` | 1,241 | 1,270 | 1,195 | 1,063 |
| `upgrade`（TP） | 254（112） | 249（102） | 156（83） | 70（45） |
| `transition` | 188 | 7 | 8 | 22 |
| `delete`（TP） | 111（35） | 151（58） | 68（28） | 6（1） |
| `revive`（TP） | 22（14） | — | — | — |

**new 状态**

| 状态 | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `confirmed` | 127 | 66 | 61 | 51.97% |
| `provisional` | 384 | 142 | 242 | 36.98% |

**merge 新信息门**

| `new_info` | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `True` | 1,161 | 712 | 449 | 61.33% |
| `False` | 80 | 47 | 33 | 58.75% |

| `new_info=False` 分组 | n | TP |
| :--- | ---: | ---: |
| `confirmed` | 48 | 33 |
| `provisional` | 32 | 14 |
| 去重轨道 | 51 | 28 |

**状态迁移**

| 迁移 / 原因 | n | TP |
| :--- | ---: | ---: |
| `confirmed → stable` / `near_miss_downgrade` | 85 | 35 |
| `provisional → stable` / `near_miss_downgrade` | 63 | 25 |
| `confirmed → stable` / `stale_demote` | 11 | 1 |
| `provisional → stable` / `stale_demote` | 29 | 7 |

**删除**

| `delete_reason` | n | TP | 占 delete 总量 |
| :--- | ---: | ---: | ---: |
| `stable_stale_purge` | 71 | 16 | 63.96% |
| `stable_miss_delete` | 28 | 10 | 25.23% |
| `stable_total_miss` | 12 | 9 | 10.81% |
| 合计 | 111 | 35 | 100.00% |

| `delete_reason` | 计数分布 |
| :--- | :--- |
| `stable_stale_purge` | `age_steps` = 60（32）/ 61（39） |
| `stable_miss_delete` | `near_miss` = 3（28）；`total_miss` = 6（28） |
| `stable_total_miss` | `total_miss` = 6（3）/ 7（1）/ 8（8）；`near_miss` = 1（11）/ 2（1） |

**复活**

| 项 | 数值 |
| :--- | ---: |
| `revive` 事件 | 22 |
| 其中 TP | 14（63.64%） |
| `revive_count=1` | 21 |
| `revive_count=2` | 1 |

## 10. confirmed 池

| 项 | S6 | arm B | arm A |
| :--- | ---: | ---: | ---: |
| `new` 中 `confirmed` FP | 61 | 61 | 83 |
| `upgrade` 事件 FP | 142 | 147 | 73 |
| confirmed 池 FP（事件口径） | 203 | 208 | 156 |
| confirmed 池 FP（轨道去重） | 197 | 208 | 156 |
| 存活至集末的 confirmed FP | 168 | 154 | 127 |
| 已删除的 confirmed 轨道 | 48 | 82 | 49 |

## 11. nav goal 与准入

**准入漏斗**

| 阶段 | n | TP | precision |
| :--- | ---: | ---: | ---: |
| 候选 | 4,671 | 1,330 | 28.47% |
| provisional gate：`confirmed` | 127 | 66 | 51.97% |
| provisional gate：`provisional` | 384 | 142 | 36.98% |
| provisional gate：`merge_existing` | 1,241 | 759 | 61.16% |
| provisional gate：`rejected` | 2,742 | 358 | 13.06% |
| provisional gate：`missing` | 177 | 5 | 2.82% |
| 写入 / merge（`admitted`） | 1,752 | 967 | 55.19% |
| 其中 `s_penalty_admitted` | 514 | — | — |

**episode 分组**

| 分组 | S6 集数 | S6 success | arm B 集数 | arm B success |
| :--- | ---: | ---: | ---: | ---: |
| 有准入 TP、无准入 FP | 58 | 47 | 58 | 47 |
| 有准入 TP、也有准入 FP | 72 | 57 | 71 | 56 |
| 只有准入 FP | 91 | 11 | 92 | 11 |
| 无任何准入候选 | 76 | 2 | 76 | 2 |

**nav goal**

| 指标 | S6 | arm B | arm A | S2U |
| :--- | ---: | ---: | ---: | ---: |
| nav goal 行数 | 4,797 | 5,777 | 3,951 | 3,858 |
| nav goal TP 占比 | 45.38% | 37.60% | 49.91% | 47.54% |
| 有 nav goal 的集数 | 221 | 221 | 200 | — |

**每集被选轨道数**

| 轨道数 | 0 | 1 | 2 | 3 | 4 | ≥5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 集数 | 76 | 109 | 64 | 28 | 8 | 12 |

**最后一条 nav goal 的状态**

| 状态 | 集数 | TP | 命中率 |
| :--- | ---: | ---: | ---: |
| `confirmed` | 197 | 112 | 56.85% |
| `provisional` | 24 | 5 | 20.83% |
| `stable` | 0 | 0 | — |

## 12. 离线标定摘要

**S7 参数判定（阈值取自 `VLVM-S5.md` §3.6）**

| 判据 | 触发阈值 | S6 | 结果 |
| :--- | :--- | ---: | :--- |
| TP 升级 | <85 | 112 | 未触发 M2 放宽 |
| confirmed 池 FP | >170 | 203（累计）/ 168（存活） | 触发停止判据硬项 |
| 复活 precision | <60% | 63.64% | 未触发收紧 |
| 陈旧清除 TP 占比 | >20% | 22.54% | 触发 60 → 90 步 |
| `stable` 内 miss 删除占比 | >50% | 25.23% | 未触发 3 → 4 |
| `stop_blocked` 集数 | >150 | 无此机制 | 不适用 |

**升级口径候选（事件级）**

| 规则 | n | TP | precision | recall |
| :--- | ---: | ---: | ---: | ---: |
| 现状 `num_obs≥2` | 254 | 112 | 44.09% | 78.87% |
| `sigma_c_latest≥0.52` | 171 | 89 | 52.05% | 62.68% |
| `sigma_c_ema_05≥0.52` | 180 | 92 | 51.11% | 64.79% |

完整回放见 `Results/VLVM-S5cal-Exp.md` §8。

## 13. 运行信息

| 项 | 值 |
| :--- | :--- |
| 完成标记 | launcher `S6 完成，用时 17118s`（无 `COMPLETED` 文件） |
| 耗时 | 17,118 s（4:45:18） |
| 检测日志 | `detections/{candidates,memory_events,nav_goals,track_frames,episodes}.jsonl` |
| `track_frames` 行数 | 12,823 |
| 离线标定输出 | `offline_cal.json`、`offline_cal_s6.json` |

# S7 实验结果：provisional 下界 0.52

## 1. 口径

| 项 | 值 |
| :--- | :--- |
| run dir | `outputs/s5_s7_2026-10-24_232335/` |
| 基准 | S6（merge 新信息门 + `stable` 删除池） |
| provisional 准入 | $0.52\le\sigma_c<0.60$，$1.0\le dist\le4.0$ m |
| 与 S6 差异 | 仅 `provisional_sigma` 0.50 → 0.52 |
| merge | 0.20 m / 10° / 3 步；禁止 merge 到 `stable` |
| `stable` 删除 | 陈旧 ≥60 步 / 连续 `near_miss` ≥3 / 累计 `near_miss` ≥6 |
| `stable` 复活 | 0.5 m / 10 步 / $\sigma_c\ge0.50$ |
| 停止 | 仅 `confirmed` 可停；STOP 0.90 m |
| S7 计划项 | evidence 排序、`num_obs≥3` 停止、视角扩张复核均未启用 |
| 场景 / 集数 | 3 × 99 = 297 |
| 仿真 | 是 |

## 2. 验收（沿用 S6 判定门）

| 指标 | 目标 | S7 | 判定 |
| :--- | :--- | ---: | :--- |
| 升级 precision | ≥50% | 50.00%（110 / 220） | 达成 |
| TP 升级数 | ≥85 | 110 | 达成 |
| confirmed 池 FP | ≤170 | 175（事件口径）/ 170（去重口径） | 事件口径未达；去重口径达成 |
| delete 总量 | ≤120 | 83 | 达成 |
| success | ≥116 | 116 | 达成 |
| never_saw 合计 | ≤76 | 86（22 + 64） | 未达 |

## 3. 导航总体

| 配置 | success | SR ↑ | Oracle ↑ | SPL ↑ | SoftSPL ↑ | steps ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| S5 臂 A | 121 | 40.74% | 43.43% | 0.1797 | 0.2591 | 156.6 |
| S5 臂 B | 116 | 39.06% | 40.74% | 0.1788 | 0.2695 | 149.2 |
| S6 | 117 | 39.39% | 41.08% | 0.1803 | 0.2741 | 146.4007 |
| S7 | 116 | 39.06% | 41.75% | 0.1798 | 0.2712 | 148.6633 |

## 4. 逐场景

| 场景 | S7 success | S7 SR | S6 success | S6 SR | 臂 B success | 臂 A success |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 38 | 38.38% | 39 | 39.39% | 40 | 42 |
| `QaLdnwvtxbs` | 67 | 67.68% | 67 | 67.68% | 65 | 68 |
| `mL8ThkuaVTM` | 11 | 11.11% | 11 | 11.11% | 11 | 11 |

## 5. 逐类别

| 类别 | S7 success / SR | S6 success / SR | 臂 B success / SR | 臂 A success / SR |
| :--- | ---: | ---: | ---: | ---: |
| `bed`（70） | 38 / 54.29% | 39 / 55.71% | 39 / 55.71% | 39 / 55.71% |
| `chair`（68） | 34 / 50.00% | 34 / 50.00% | 33 / 48.53% | 39 / 57.35% |
| `couch`（73） | 25 / 34.25% | 25 / 34.25% | 24 / 32.88% | 27 / 36.99% |
| `tv`（43） | 17 / 39.53% | 17 / 39.53% | 18 / 41.86% | 13 / 30.23% |
| `toilet`（43） | 2 / 4.65% | 2 / 4.65% | 2 / 4.65% | 3 / 6.98% |

## 6. 失败构成

| `failure_cause` | S7 | S6 | 臂 B | 臂 A |
| :--- | ---: | ---: | ---: | ---: |
| `did_not_fail` | 116 | 117 | 116 | 121 |
| `false_positive` | 80 | 86 | 92 | 72 |
| `false_negative` | 15 | 15 | 13 | 19 |
| `never_saw_target_did_not_travel_stairs_feasible` | 22 | 20 | 18 | 21 |
| `never_saw_target_did_not_travel_stairs_likely_infeasible` | 64 | 59 | 58 | 64 |

## 7. 逐集配对

| 对比 | 前者 success | 后者 success | 仅前者成功 | 仅后者成功 | 两者均成功 | 两者均失败 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| S7 vs S6 | 116 | 117 | 4 | 5 | 112 | 176 |
| S7 vs 臂 B | 116 | 116 | 7 | 7 | 109 | 174 |
| S7 vs 臂 A | 116 | 121 | 12 | 17 | 104 | 164 |
| S7 vs S2U | 116 | 121 | 15 | 20 | 101 | 161 |

## 8. 停止门槛

| 指标 | S7 | S6 | 臂 B | 臂 A |
| :--- | ---: | ---: | ---: | ---: |
| `stop_allowed=True` | 187 | 189 | 187 | 186 |
| `stop_allowed=False`（被门槛拦截） | 156 | 212 | 212 | 88 |
| 含实际停止的集数（`stops=1`） | 186 | 188 | 186 | 185 |

## 9. 生命周期

| 事件 | S7 | S6 | 臂 B | 臂 A | S2U |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `new` | 465 | 511 | 513 | 374 | 312 |
| — `confirmed` | 134 | 127 | 125 | 169 | 164 |
| — `provisional` | 331 | 384 | 388 | 205 | 148 |
| `merge` | 1,116 | 1,241 | 1,270 | 1,195 | 1,063 |
| `upgrade`（TP） | 220（110） | 254（112） | 249（102） | 156（83） | 70（45） |
| `transition` | 155 | 188 | 7 | 8 | 22 |
| `delete`（TP） | 83（26） | 111（35） | 151（58） | 68（28） | 6（1） |
| `revive`（TP） | 20（14） | 22（14） | — | — | — |

**new 状态**

| 状态 | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `confirmed` | 134 | 69 | 65 | 51.49% |
| `provisional` | 331 | 136 | 195 | 41.09% |

**merge 新信息门**

| `new_info` | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `True` | 1,045 | 691 | 354 | 66.13% |
| `False` | 71 | 34 | 37 | 47.89% |
| 去重轨道（`False`） | 47 | 23 | 24 | 48.94% |

| `new_info=False` 状态 | n | TP |
| :--- | ---: | ---: |
| `confirmed` | 43 | 24 |
| `provisional` | 28 | 10 |

**状态迁移**

| 迁移 / 原因 | n | TP |
| :--- | ---: | ---: |
| `confirmed → stable` / `near_miss_downgrade` | 74 | 35 |
| `provisional → stable` / `near_miss_downgrade` | 51 | 23 |
| `confirmed → stable` / `stale_demote` | 9 | 1 |
| `provisional → stable` / `stale_demote` | 21 | 3 |

**删除**

| `delete_reason` | n | TP | TP 占比 | 占 delete 总量 |
| :--- | ---: | ---: | ---: | ---: |
| `stable_stale_purge` | 56 | 12 | 21.43% | 67.47% |
| `stable_miss_delete` | 13 | 4 | 30.77% | 15.66% |
| `stable_total_miss` | 14 | 10 | 71.43% | 16.87% |
| 合计 | 83 | 26 | 31.33% | 100.00% |

**`stable` 轨道级**

| 项 | TP | FP |
| :--- | ---: | ---: |
| 进入 `stable` | 55 | 90 |
| 复活 | 13 | 6 |
| 删除 | 19 | 56 |
| 停泊至集末 | 23 | 28 |

**复活**

| 项 | 数值 |
| :--- | ---: |
| `revive` 事件 | 20 |
| 其中 TP | 14（70.00%） |
| `revive_count=1` | 19 |
| `revive_count=2` | 1 |

## 10. confirmed 池

| 项 | S7 | S6 | 臂 B | 臂 A |
| :--- | ---: | ---: | ---: | ---: |
| `new` 中 `confirmed` FP | 65 | 61 | 61 | 83 |
| `upgrade` 事件 FP | 110 | 142 | 147 | 73 |
| confirmed 池 FP（事件口径） | 175 | 203 | 208 | 156 |
| confirmed 池 FP（轨道去重） | 170 | 197 | 208 | 156 |
| 存活至集末的 confirmed FP | 148 | 168 | 154 | 127 |
| 已删除的 confirmed 轨道 | 37 | 48 | 82 | 49 |

## 11. nav goal 与准入

**准入漏斗**

| 阶段 | n | TP | precision |
| :--- | ---: | ---: | ---: |
| 候选 | 4,753 | 1,349 | 28.38% |
| provisional gate：`confirmed` | 134 | 69 | 51.49% |
| provisional gate：`provisional` | 331 | 136 | 41.09% |
| provisional gate：`merge_existing` | 1,116 | 725 | 64.96% |
| provisional gate：`rejected` | 2,993 | 413 | 13.80% |
| provisional gate：`missing` | 179 | 6 | 3.35% |
| 写入 / merge（`admitted`） | 1,581 | 930 | 58.82% |
| 其中 `s_penalty_admitted` | 522 | — | — |

**episode 分组**

| 分组 | S7 集数 | S7 success | S6 集数 | S6 success |
| :--- | ---: | ---: | ---: | ---: |
| 有准入 TP、无准入 FP | 64 | 52 | 58 | 47 |
| 有准入 TP、也有准入 FP | 68 | 52 | 72 | 57 |
| 只有准入 FP | 79 | 10 | 91 | 11 |
| 无任何准入候选 | 86 | 2 | 76 | 2 |

**nav goal**

| 指标 | S7 | S6 | 臂 B | 臂 A |
| :--- | ---: | ---: | ---: | ---: |
| nav goal 行数 | 4,465 | 4,797 | 5,777 | 3,951 |
| nav goal TP 占比 | 47.26% | 45.38% | 37.60% | 49.91% |
| 有 nav goal 的集数 | 211 | 221 | 221 | 200 |

**每集被选轨道数**

| 轨道数 | 0 | 1 | 2 | 3 | 4 | ≥5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 集数 | 86 | 105 | 64 | 26 | 12 | 4 |

**最后一条 nav goal 的状态**

| 状态 | 集数 | TP | 命中率 |
| :--- | ---: | ---: | ---: |
| `confirmed` | 194 | 114 | 58.76% |
| `provisional` | 17 | 6 | 35.29% |
| `stable` | 0 | 0 | — |

## 12. 离线标定摘要

| 项 | S6 | S7 |
| :--- | ---: | ---: |
| new provisional n / TP / FP | 384 / 142 / 242 | 331 / 136 / 195 |
| new provisional precision | 36.98% | 41.09% |
| upgrade 事件 / TP / precision | 254 / 112 / 44.09% | 220 / 110 / 50.00% |
| confirmed 池 FP（事件 / 去重） | 203 / 197 | 175 / 170 |
| delete 总量（TP） | 111（35） | 83（26） |
| success / SR | 117 / 39.39% | 116 / 39.06% |
| never_saw 合计 | 79 | 86 |

| S7 标定项 | 结果 |
| :--- | :--- |
| merge 严格门 0.30/15/4 | provisional `num_obs≥2` 129 条，TP 76，precision 58.91%；TP 低于 85 |
| upgrade `sigma_c_latest≥0.52` | 162 条，TP 87，precision 53.70% |
| upgrade `sigma_c_ema_05≥0.52` | 184 条，TP 96，precision 52.17% |
| 停止 `num_obs≥3` | 允许 110，TP 79，FP 31；confirmed 为 194 / 114 / 80 |
| 目标选择 evidence 排序 | 同轨道子集实际 TP 2,076；evidence 2,037；最后 nav goal 120 → 113 |
| 准入 0.55 / 1.5–4.0 m | 866 条，TP 559，precision 64.5%，有 TP 集 118 |

完整回放见 `Results/VLVM-S5cal-Exp.md` §9。

## 13. 运行信息

| 项 | 值 |
| :--- | :--- |
| 完成标记 | launcher `S7 完成，用时 17085s`（无 `COMPLETED` 文件） |
| 耗时 | 17,085 s（4:44:45） |
| 检测日志 | `detections/{candidates,memory_events,nav_goals,track_frames,episodes}.jsonl` |
| `track_frames` 行数 | 10,775 |
| 离线标定输出 | `offline_cal.json`、`offline_cal_s7.json` |
