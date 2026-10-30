# VLVM-S6gd4：confirmed/gd-only 顺序对调 + depth track-relax 三场景结果

> 运行目录：`outputs/s6_gd4_depthrelax_2026-10-29_015327`  
> 对照版本：S6gd3（`outputs/s6_gd3_restart_2026-10-28_182915`）  
> 场景：`5cdEh9F2hJL` / `mL8ThkuaVTM` / `QaLdnwvtxbs`，共 `3 × 99 = 297` episodes  
> 本版状态：`COMPLETED`

## 1. 本版配置

| 项 | S6gd4 实际运行 |
| :--- | :--- |
| 选择层级 | `gd-tsp（含 promoted gd-only） > confirmed > gd-only > provisional` |
| 与 S6gd3 的排序差异 | 仅 `confirmed` 与 `gd-only` 对调；S6gd3 为 `gd-tsp > gd-only > confirmed > provisional` |
| STOP | 仅 `gd-tsp confirmed`（含 promoted gd-only）可停；纯 TSP confirmed、gd-only、provisional 不可停 |
| `gd_depth_failed_track_merge` | `true`；仅允许 depth 失败的 `confirmed/provisional` track 合并 |
| depth 例外 | 使用 track centroid；不新建 track；`gd-only/new`、candidate-only、stable 仍拒绝 |
| `enable_gd_only_conf_gate` | `false` |
| `enable_gd_ttl` | `false` |
| `gd_only_classes` | `""`（全部 target class） |
| `stable_revive_radius/window/min_sigma` | `0.8 / 20 / 0.50` |
| `pointnav_stop_radius` | `0.90 m` |

## 2. 总体结果

| 指标 | S6gd3 | S6gd4 | 差值 |
| :--- | ---: | ---: | ---: |
| success | 139 / 297 | **132 / 297** | **−7** |
| SR | 46.80% | **44.44%** | −2.36 pp |
| Oracle | 46.46% | 45.79% | −0.67 pp |
| SPL | 0.2157 | 0.2106 | −0.0051 |
| SoftSPL | 0.2867 | 0.2828 | −0.0039 |
| steps | 153.55 | 162.34 | +8.79 |
| candidates | 5,602 | 6,248 | +646 |
| candidate TP / FP | 1,605 / 3,997 | 1,932 / 4,316 | +327 / +319 |
| candidate precision | 28.65% | 30.92% | +2.27 pp |

## 3. 分场景

| 场景 | S6gd3 success / Oracle / steps | S6gd4 success / Oracle / steps | success Δ |
| :--- | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 51 / 62 / 213.75 | 50 / 62 / 220.25 | −1 |
| `QaLdnwvtxbs` | 80 / 72 / 152.92 | 73 / 71 / 173.30 | **−7** |
| `mL8ThkuaVTM` | 8 / 4 / 93.97 | 9 / 3 / 93.47 | +1 |

## 4. 失败原因

| failure_cause | S6gd3 | S6gd4 | 差值 |
| :--- | ---: | ---: | ---: |
| `did_not_fail` | 139 | 132 | −7 |
| `false_positive` | 102 | 105 | +3 |
| `bad_stop_true_positive` | 2 | 2 | 0 |
| `timeout_true_positive` | 2 | 6 | +4 |
| `false_negative` | 1 | 1 | 0 |
| `never_saw_target_did_not_travel_stairs_likely_infeasible` | 48 | 48 | 0 |
| `never_saw_target_did_not_travel_stairs_feasible` | 3 | 3 | 0 |

## 5. 最终目标来源

有最终 `nav_goal` 记录的 episode 为 245；其余 52 个 episode 没有最终目标行。

| 最终来源 | S6gd3 goals / TP / success | S6gd4 goals / TP / success |
| :--- | ---: | ---: |
| `gd-tsp` | 187 / 144（77.0%） / 138 | 187 / 143（76.5%） / 131 |
| `gd-only` | 53 / 16（30.2%） / 0 | 51 / 19（37.3%） / 0 |
| 纯 TSP `confirmed` | 1 / 0 / 0 | 2 / 1 / 0 |
| `provisional` | 4 / 0 / 0 | 5 / 0 / 0 |

### 5.1 gd-tsp 内部组成

| 子类 | S6gd3 goals / TP / success | S6gd4 goals / TP / success | 差值 goals / TP / success |
| :--- | ---: | ---: | ---: |
| track gd-tsp | 92 / 67 / 60 | 99 / 74 / 65 | +7 / +7 / +5 |
| promoted gd-only | 95 / 77 / 78 | 88 / 69 / 66 | **−7 / −8 / −12** |

## 6. depth-relax 事件

### 6.1 `gd_tsp_merge` 分组

| depth_relaxed | association | n | TP | FP | precision |
| :---: | :--- | ---: | ---: | ---: | ---: |
| false | `track_confirmed` | 1,400 | 1,096 | 304 | 78.3% |
| false | `track_provisional` | 54 | 34 | 20 | 63.0% |
| true | `track_confirmed` | 15 | 12 | 3 | 80.0% |
| true | `track_provisional` | 3 | 2 | 1 | 66.7% |

`depth_relaxed=true` 合计 18 条，TP 14 / FP 4，precision 77.8%。

### 6.2 涉及 episode

| 场景 / episode | relaxed 事件 | TP | S6gd3 success | S6gd4 success |
| :--- | ---: | ---: | :---: | :---: |
| `QaLdnwvtxbs` / 95 | 2 | 2 | 0 | 0 |
| `QaLdnwvtxbs` / 28 | 6 | 6 | 1 | 0 |
| `QaLdnwvtxbs` / 10 | 1 | 1 | 1 | 1 |
| `QaLdnwvtxbs` / 25 | 1 | 1 | 1 | 1 |
| `mL8ThkuaVTM` / 4 | 2 | 2 | 0 | 1 |
| `5cdEh9F2hJL` / 16 | 3 | 0 | 0 | 0 |
| `5cdEh9F2hJL` / 73 | 1 | 0 | 0 | 0 |
| `5cdEh9F2hJL` / 7 | 1 | 1 | 1 | 1 |
| `5cdEh9F2hJL` / 32 | 1 | 1 | 1 | 1 |
| **合计** | **18** | **14** | **5** | **5** |

## 7. 两版差异

### 7.1 配置差异

逐项比较两次 `resolved_config.yaml`，除输出路径外，行为配置只有一项不同：

| 配置项 | S6gd3 | S6gd4 |
| :--- | :---: | :---: |
| `gd_depth_failed_track_merge` | 不存在（默认 `false`） | `true` |

以下参数两版相同：`enable_gd_only_conf_gate=false`、`enable_gd_ttl=false`、`gd_only_classes=""`、`gd_conf_min=0.40`、`gd_max_dist=5.0`、`upgrade_min_obs=4`、`stable_revive_radius/window/min_sigma=0.8/20/0.50`、`pointnav_stop_radius=0.90`、`stop_gate_turns=3`、`stop_gate_confirm_radius=1.2`。

### 7.2 排序差异

| 版本 | 四挡顺序 |
| :--- | :--- |
| S6gd3 | `gd-tsp > gd-only > confirmed > provisional` |
| S6gd4 | `gd-tsp > confirmed > gd-only > provisional` |

两版差异仅为 `confirmed` 与 `gd-only` 的顺序；STOP 资格相同，均为 `gd-tsp confirmed`。

### 7.3 depth 与日志差异

| 项 | S6gd3 | S6gd4 |
| :--- | :--- | :--- |
| depth 失败 track merge | 全部拒绝，写 `gd_track_depth_shadow` | confirmed/provisional 可合并；`gd-only/new`、candidate-only、stable 仍拒绝 |
| `gd_track_depth_shadow` | 12 条 | 0 条 |
| `gd_tsp_merge` 附加字段 | 无 `depth_ok/depth_relaxed` | 写 `depth_ok`、`depth_relaxed` |
| `gd_tsp_merge` 数量 | 1,256 | 1,472 |
| `gd_only_new` / `gd_only_merge` | 729 / 2,937 | 751 / 3,201 |
| `new` / `merge` / `upgrade` | 314 / 1,825 / 222 | 341 / 2,112 / 228 |
| `delete` / `transition` | 81 / 116 | 108 / 152 |
| `gd_stable_shadow` | 8 | 8 |

实现层差异：

| 位置 | S6gd3 | S6gd4 |
| :--- | :--- | :--- |
| `GdMechanismConfig` | 无 depth 例外开关 | 新增 `gd_depth_failed_track_merge` |
| `mechanism.py` base gate | `depth_ok=false` 一律拒绝 | 仅 `track_same & confirmed/provisional` 可绕过 depth；使用 track centroid |
| `VLVMConfig` | 无该字段 | 新增字段并支持 Hydra 传入 |
| `vlvm_trainer.py` 配置快照 | 无该字段 | 写入 `gd_depth_failed_track_merge` |
| `_nav_s6_gdmech.sh` 头部与 summary | 无 depth 说明 | 打印 depth 放开范围 |
| `memory_events` / `gd_tsp_merge` | 无 `depth_ok/depth_relaxed` | 记录 `depth_ok/depth_relaxed` |

### 7.4 不属于 S6gd4 的后续调整

`_pointnav` 的 `stop=False` 动作屏蔽，以及 `[STOP]` 日志新增 `goal_origin`，是在 S6gd4 完成之后加入当前工作区的；S6gd4 本身未执行这两项调整，STOP 条件仍按旧代码运行。

## 8. 退化归因

### 8.1 episode 翻转

| 方向 | episode 数 | 主要 failure_cause |
| :--- | ---: | :--- |
| success → failure | 11 | `false_positive` 9、`timeout_true_positive` 2 |
| failure → success | 4 | `did_not_fail` 4 |
| 净变化 | **−7** | — |

11 个 success → failure 中，9 个在首个选择层级分歧处出现了 tier 变化；2 个没有 tier 变化：

| 场景 / episode | S6gd3 → S6gd4 | 首个 tier 分歧 | 最终来源（S6gd3 → S6gd4） | S6gd4 failure |
| :--- | :--- | :--- | :--- | :--- |
| `5cdEh9F2hJL` / 6 | success → failure | 无 | gd-tsp(promoted) → gd-tsp(track) | `false_positive` |
| `5cdEh9F2hJL` / 24 | success → failure | `gd_only → confirmed` | gd-tsp(track) → gd-tsp(track) | `false_positive` |
| `5cdEh9F2hJL` / 72 | success → failure | `gd_only → gd_tsp` | gd-tsp(promoted) → gd-tsp(promoted) | `timeout_true_positive` |
| `QaLdnwvtxbs` / 11 | success → failure | `gd_tsp → gd_only` | gd-tsp(promoted) → gd-only | `false_positive` |
| `QaLdnwvtxbs` / 26 | success → failure | `gd_only → provisional` | gd-tsp(promoted) → gd-only | `timeout_true_positive` |
| `QaLdnwvtxbs` / 28 | success → failure | `gd_tsp → gd_only` | gd-tsp(promoted) → gd-only | `false_positive` |
| `QaLdnwvtxbs` / 42 | success → failure | `gd_only → confirmed` | gd-tsp(promoted) → gd-only | `false_positive` |
| `QaLdnwvtxbs` / 49 | success → failure | `gd_only → confirmed` | gd-tsp(promoted) → 纯 TSP confirmed | `false_positive` |
| `QaLdnwvtxbs` / 51 | success → failure | 无 | gd-tsp(track) → gd-tsp(track) | `false_positive` |
| `QaLdnwvtxbs` / 85 | success → failure | `gd_only → confirmed` | gd-tsp(promoted) → gd-tsp(track) | `false_positive` |
| `QaLdnwvtxbs` / 93 | success → failure | `gd_only → confirmed` | gd-tsp(track) → gd-tsp(track) | `false_positive` |

4 个 failure → success：

| 场景 / episode | 首个 tier 分歧 | 最终来源（S6gd3 → S6gd4） |
| :--- | :--- | :--- |
| `5cdEh9F2hJL` / 3 | `gd_only → confirmed` | gd-only → gd-tsp(promoted) |
| `5cdEh9F2hJL` / 49 | `gd_only → gd_tsp` | gd-only → gd-tsp(track) |
| `QaLdnwvtxbs` / 21 | `gd_only → confirmed` | gd-tsp(track) → gd-tsp(track) |
| `mL8ThkuaVTM` / 4 | `confirmed → gd_tsp` | gd-tsp(track) → gd-tsp(track) |

### 8.2 首个选择层级分歧汇总

| 首个分歧（S6gd3 → S6gd4） | episodes | 两边都成功 | 两边都失败 | success → failure | failure → success | 净变化 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gd_only → confirmed` | 39 | 26 | 6 | 5 | 2 | −3 |
| `gd_tsp → gd_only` | 16 | 8 | 6 | 2 | 0 | −2 |
| `gd_only → gd_tsp` | 14 | 6 | 6 | 1 | 1 | 0 |
| `confirmed → gd_tsp` | 3 | 2 | 0 | 0 | 1 | +1 |
| `provisional → gd_tsp` | 2 | 2 | 0 | 0 | 0 | 0 |
| `gd_only → provisional` | 1 | 0 | 0 | 1 | 0 | −1 |
| `provisional → gd_only` | 1 | 0 | 1 | 0 | 0 | 0 |
| **合计** | **76** | **44** | **19** | **9** | **4** | **−5** |

另有 2 个 success → failure episode 没有 tier 分歧（`5cdEh9F2hJL/6`、`QaLdnwvtxbs/51`），合计净变化 −7。

### 8.3 退化来源判断

| 因素 | 数据 |
| :--- | :--- |
| `confirmed` / `gd-only` 对调 | 首个分歧为 `gd_only → confirmed` 的 39 个 episode 净 −3；5 个 success → failure、2 个 failure → success |
| promoted gd-tsp 减少 | promoted 最终目标 95 → 88，success 78 → 66（−12）；track gd-tsp success 60 → 65（+5） |
| 纯 confirmed 最终目标 | 1 → 2，success 仍为 0；其中 `QaLdnwvtxbs/49` 为 TP 但不可 STOP |
| depth-relax | 18 条事件、9 个 episode；事件 TP 14 / FP 4；涉及 episode 的 success 5 → 5，净 0 |
| 其他 tier 分歧 | `gd_tsp → gd_only` 净 −2；`gd_only → provisional` 净 −1；`confirmed → gd_tsp` 净 +1 |
| 无 tier 分歧的翻转 | 2 个 success → failure，净 −2 |
| 失败类型 | success → failure 的 11 个中，9 个 `false_positive`、2 个 `timeout_true_positive` |

主要退化集中在 `QaLdnwvtxbs`：success −7，包含 8 个 success → failure、1 个 failure → success；其中 7 个失败 episode 存在首个 tier 分歧。

## 9. 结论

1. S6gd4 的 success 从 139 降到 132，SR 从 46.80% 降到 44.44%，退化为 7 个 episode；`QaLdnwvtxbs` 贡献 −7。
2. 两版实际行为差异只有 `confirmed` / `gd-only` 顺序对调和 depth track-relax；STOP、conf gate、TTL、revive 和主要阈值未变。
3. depth-relax 只触发 18 条事件、涉及 9 个 episode，事件 TP 14 / FP 4；同一批 episode 的 success 为 5 → 5，净变化 0，不能解释整体 −7。
4. 退化主要跟随选择层级分歧：76 个 episode 出现首个 tier 分歧，合计净 −5；其中 `gd_only → confirmed` 净 −3，`gd_tsp → gd_only` 净 −2，`confirmed → gd_tsp` 净 +1。
5. 最终 gd-tsp 总数保持 187，但 promoted gd-only 从 95 降到 88，其 success 从 78 降到 66；track gd-tsp success 从 60 升到 65。退化集中在 promoted gd-tsp 的 success 损失，并与纯 confirmed 被提前选择、最终不可 STOP 同时出现。
6. 单次 run 无法把排序效应与随机波动完全分离；2 个无 tier 分歧的 success → failure 说明还存在非排序因素。

## 10. 本 run 之后的机制调整

以下调整在 S6gd4 之后确定，不属于本 run 数据：

- `gd-only` 不删除；`enable_gd_ttl=false`，只保留 TTL shadow；
- `gd_depth_failed_gd_only=true`：depth 失败允许进入 gd-only new/merge；
- GD 命中 `stable` 仍只写 `gd_stable_shadow`，不升级为 gd-tsp；
- `occ` gate 使用 `occ_passed`，保留 gd-only merge；
- 中间档 `tsp-confirmed / gd-only` 的 gd-tsp 提升概率量化方法见 [VLVM-S7.md](VLVM-S7.md) §2，尚未实现。
