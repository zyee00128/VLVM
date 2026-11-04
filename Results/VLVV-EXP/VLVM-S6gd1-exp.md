# VLVM-S6gd1：GD 完整机制三场景实验

> 运行目录：`outputs/s6_gdmech_3scene_2026-10-27_142116`
> 场景：`5cdEh9F2hJL` / `QaLdnwvtxbs` / `mL8ThkuaVTM`，共 3 × 99 = 297 episodes  

## 1. S6gd1 运行配置

| 项 | 值 |
| :--- | :--- |
| `provisional_sigma` | `0.55` |
| `upgrade_min_obs` | `4` |
| `upgrade_min_obs_close` | `3` |
| `upgrade_max_dist` | `4.0 m` |
| GD | `enable_gd_mechanism=true`，`gd_conf_min=0.40`，`gd_only_conf_min=0.50`，`gd_max_dist=5.0`，`gd_only_ttl=60` |
| gd-only 类别 | `bed,chair,couch,toilet` |

## 2. 总体结果

| 指标 | S6gd1 | S5-S6 基线 | Δ |
| :--- | ---: | ---: | ---: |
| episodes | 297 | 297 | 0 |
| success | 116 | 117 | −1 |
| SR | 39.06% | 39.39% | −0.34 pp |
| Oracle success | 117 | 122 | −5 |
| Oracle rate | 39.39% | 41.08% | −1.69 pp |
| SPL | 0.1798 | 0.1803 | −0.0005 |
| SoftSPL | 0.2832 | 0.2741 | +0.0091 |
| steps / episode | 130.29 | 146.40 | −16.11 |

| 成功配对（S6 / S5） | episodes |
| :--- | ---: |
| both success | 86 |
| only S6 success | 30 |
| only S5 success | 31 |
| both fail | 150 |

| Oracle 配对（S6 / S5） | episodes |
| :--- | ---: |
| both oracle | 106 |
| only S6 oracle | 11 |
| only S5 oracle | 16 |
| neither oracle | 164 |

## 3. 逐场景

| 场景 | n | S6 success | S6 SR | S6 Oracle | S6 SPL | S6 steps | S5 success | S5 SR | S5 Oracle | S5 SPL | S5 steps |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 99 | 33 | 33.33% | 53.54% | 0.1311 | 187.2 | 39 | 39.39% | 56.57% | 0.1605 | 224.4 |
| `QaLdnwvtxbs` | 99 | 76 | 76.77% | 60.61% | 0.3626 | 114.2 | 67 | 67.68% | 63.64% | 0.3105 | 123.1 |
| `mL8ThkuaVTM` | 99 | 7 | 7.07% | 4.04% | 0.0456 | 89.5 | 11 | 11.11% | 3.03% | 0.0698 | 91.7 |

## 4. 逐类别

| 类别 | n | S6 success | S6 SR | S6 Oracle | S5 success | S5 SR | S5 Oracle |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bed` | 70 | 29 | 41.43% | 45.71% | 39 | 55.71% | 60.00% |
| `chair` | 68 | 35 | 51.47% | 61.76% | 34 | 50.00% | 55.88% |
| `couch` | 73 | 26 | 35.62% | 36.99% | 25 | 34.25% | 36.99% |
| `toilet` | 43 | 6 | 13.95% | 37.21% | 2 | 4.65% | 34.88% |
| `tv` | 43 | 20 | 46.51% | 0.00% | 17 | 39.53% | 0.00% |

## 5. 失败原因

| `failure_cause` | S6gd1 | S5-S6 基线 |
| :--- | ---: | ---: |
| `did_not_fail` | 116 | 117 |
| `false_positive` | 97 | 86 |
| `false_negative` | 9 | 15 |
| `never_saw_target_did_not_travel_stairs_feasible` | 13 | 20 |
| `never_saw_target_did_not_travel_stairs_likely_infeasible` | 62 | 59 |

## 7. GD 机制统计

### 7.1 路由决策

| 指标 | 数量 |
| :--- | ---: |
| GD frames | 35,133 |
| proposals | 4,539 |
| `base_rejected` | 2,355 |
| `gd_tsp` | 1,143 |
| `gd_only` | 1,031 |
| `blocked` | 10 |

主要拒绝原因：

| reason | 数量 |
| :--- | ---: |
| `gd_only_class_disabled` | 894 |
| `depth_not_ok` | 798 |
| `gd_only_conf_below_min` | 661 |
| `stable` | 10 |
| `gd_occ_geom_unknown` | 1 |
| `candidate_occ_failed` | 1 |

### 7.2 写入与合并

| 事件 | n | TP | FP |
| :--- | ---: | ---: | ---: |
| `gd_only_new` | 230 | 144 | 86 |
| `gd_only_merge` | 800 | 643 | 157 |
| `gd_tsp_new` | 180 | 69 | 111 |
| `gd_tsp_merge` | 963 | 709 | 254 |

### 7.3 导航选择

| 来源 | nav_goal 行数 | 出现 episode 数 | 行 TP | 行 FP |
| :--- | ---: | ---: | ---: | ---: |
| `gd-only` | 1,609 | 119 | 1,016 | 593 |
| `gd-tsp` | 1,948 | 162 | 1,232 | 716 |

首个错误锁来源：

| 来源 | episodes |
| :--- | ---: |
| `gd-only` | 56 |
| `gd-tsp` | 37 |
| 无 GD 来源 | 204 |

### 7.4 主要问题

| 问题 | 证据 | 结果 |
| :--- | :--- | :--- |
| candidate 直接新建 confirmed 过早 | `gd_tsp_new` = 180（TP 69 / FP 111，precision 38.3%）；被选中的 38 集：S6/S5 success 14/18，Oracle 18/22，wrong stop 19/13 | 新 confirmed 是主要 FP 来源 |
| GD 前置覆盖不足 | `base_rejected` = 2,355 / 4,539；其中 class disabled 894、depth 798、conf 661 | 可进入 memory 的 GD 观测有限 |
| gd-only 只能作为 fallback | gd-only 被选中 119 集；只有 gd-only（无 gd-tsp）38 集：success 10/14 | 已有 confirmed/provisional 时无法纠正错误锁定 |
| candidate 匹配顺序 | GD 先在当前帧 `match_candidate` 中匹配，再检查 candidate occ；candidate 尚未先经过完整 admission 转成 track | 应先 `candidate → occ/S/admission → track`，再让 GD 匹配 track |
| `gd-only` 增量不足 | `gd_only_new` = 230（TP 144 / FP 86）；被选中 23 集：success 6/8。`gd_only_merge` = 800（TP 643 / FP 157）；被选中 81 集：success 45/44，Oracle 62/65 | 高 precision 的 merge 没有转化为足够增量 TP |
| 已有 track 的 GD 仲裁相对有效 | `gd_tsp_merge` = 963（TP 709 / FP 254）；被选中 141 集：success 96/85，Oracle 83/83 | 该类路径与正向结果关联 |
| upgrade 条件偏松 | 39 次 upgrade：TP 14 / FP 25（35.9%）；`close_reobservation` 34 次仅 12 TP，`num_obs=3` 32 次仅 10 TP | 机械条件满足，但精度不足 |

### 7.5 upgrade 日志核对

| 条件/分组 | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| 全部 upgrade | 39 | 14 | 25 | 35.9% |
| `close_reobservation` | 34 | 12 | 22 | 35.3% |
| `frame_upgrade` | 5 | 2 | 3 | 40.0% |
| `num_obs=3` | 32 | 10 | 22 | 31.3% |
| `num_obs>=4` | 7 | 4 | 3 | 57.1% |
| `3.0m <= dist < 4.0m` | 10 | 3 | 7 | 30.0% |

实现核对：

- 39 次 upgrade 的 `from_state` 全部为 `provisional`，`lifecycle_state` 全部为 `confirmed`；
- `frame_upgrade` 的 `num_obs` 均 ≥4；
- `close_reobservation` 的 `num_obs` 均 ≥3；
- 全部 `detect_dist <= 4.0 m`，最大约 3.945 m。
- `s6_summary.json` 的旧 contract 文本写成 `num_obs>=2`，与本次实际事件不符；启动脚本的汇总 contract 已改为单一 `upgrade_min_obs=4`。

## 8. 产物

- `detections/candidates.jsonl`
- `detections/memory_events.jsonl`
- `detections/nav_goals.jsonl`
- `detections/track_frames.jsonl`
- `detections/gd_frames.jsonl`
- `detections/gd_proposals.jsonl`
- `detections/field_observations/`：297 个 ZIP，约 38 GB
- `s6_summary.json`
- `gd_offline_cal.json`
- `COMPLETED`
