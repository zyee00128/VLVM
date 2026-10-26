# VLVM Stage 5 Final：四场景最终实验与离线标定

> 定位：本文是 S5 最终机制的唯一结果归档，合并最终四场景导航结果与基于该运行日志的离线标定。历史标定细节保留在 [VLVM-S5cal-Exp.md](VLVM-S5cal-Exp.md)，本文不回填该文件。

## 1. 结论摘要

| 项 | 结果 |
| :--- | :--- |
| 导航 | 396 集，success **164**，SR **41.41%** |
| Oracle / SPL / SoftSPL | **58.08%** / **0.1803** / **0.2795** |
| 平均步数 | **184.78** |
| 同四场景位置 | 低于 P0/SP1（43.69%）与 P1（42.68%），高于 vlvm v1（36.61%）；与 P0/SP1 的配对净差为 **−9 集**（McNemar $p\approx0.336$） |
| 主要瓶颈 | 难类覆盖与目标身份：`toilet`、`tv`、`potted plant` 合计 152 集，仅 21 集成功；S-penalty 准入 precision 77.19%，但 TP recall 仅 41.38% |
| 停止 | 仅 `confirmed` 可停：离线回放保留 164 TP、丢失 6 TP、规避 20 FP；`num_obs≥3` 会丢失 59 TP |
| 已删除参数 | `upgrade_sigma_min` 已从代码/配置删除；本次 close 182 次升级全部 ≥0.52，frame 原本不检查，删除不影响本次事件；低 sigma frame 仍会升级 |
| 需单独复核 | `stable_total_miss` 删除 12 条中 11 条为 TP |

最终机制的方向是“高精度准入 + 只允许 confirmed 停止 + stable 池治理”，但本轮结果没有超过同四场景的 P0/SP1；它主要改善了步数与 SoftSPL，SR/Oracle 均略低。覆盖不足仍是第一约束，不能靠继续加严停止解决。

## 2. 运行口径

### 2.1 运行信息

| 项 | 值 |
| :--- | :--- |
| 运行时间 | 2026-10-26 10:29–18:32，约 8:02:37 |
| 运行目录 | `outputs/s5_final_4scene_2026-10-26_102914` |
| 场景 / 集数 | `5cdEh9F2hJL`、`TEEsavR23oF`、`mv2HUxq3B53`、`wcojb4TFT35`；4 × 99 = 396 |
| 检测模型 | TSP3D V1，`data/vlvm_v1/vlvm_v1.pth` |
| 语义复核 | BLIP2-ITM；GD `gdp_enable=false`，未参与选择 |
| 导航控制 | 仅 `confirmed` 可 STOP；`pointnav_stop_radius=0.90 m`；无 `num_obs` 停止门槛、无当前帧重确认、无视角扩张复核 |
| 生命周期 | `provisional` / `confirmed` / `stable`；stable 池删除与复活见 §2.2、§6.3 |
| 导航状态 | 396/396 集完成，`episodes.jsonl` 396 行 |

### 2.2 关键参数

| 参数 | 值 |
| :--- | ---: |
| `sigma_tar` / `sigma_prop` | 0.60 / 0.45 |
| `t_S` / `s_min` / `radius_m` / `uncovered_w` | 0.14 / 0.07 / 0.5 / 1.0 |
| `provisional_sigma` / `provisional_min_dist` / `provisional_max_dist` | 0.50 / 1.0 / 4.0 |
| `provisional_s_min` | 0.07 |
| `upgrade_min_obs` / `upgrade_max_dist` | 2 / 4.0 |
| `merge_dist_thresh` / `merge_new_info_dist` / `merge_new_info_yaw` / `merge_new_info_steps` | 0.5 / 0.2 / 10° / 3 |
| `confirmed_to_stable_misses` / `provisional_to_stable_misses` | 3 / 3 |
| `stable_miss_delete` / `stable_total_miss` / `stable_stale_purge` | 5 / 8 / 60 |
| `stable_revive_radius` / `stable_revive_window` / `stable_revive_min_sigma` | 0.5 / 10 / 0.50 |
| `stop_gate_confirm_radius` / `stop_gate_turns` | 1.2 m / 3 |
| `pointnav_stop_radius` | 0.90 m |
| `gdp_enable` | false |

### 2.3 运行版本与后处理说明

本次运行启动于仓库重构之前，`resolved_config.yaml` 仍保留 `enable_fb`、`fb_*`、`gdp_*`、`lock_*` 等旧键。可解析的 pick 日志中 `pick_gd=0`、`pick_both=0`，GD 未参与目标选择；运行日志中的生命周期决策由 provisional/stable 池驱动。运行结束后仓库已删除 GD/fb 死路径并重命名内部字段，**本次 396 集没有用重构后的代码重新跑**；本文结果以运行日志为准。

导航主进程完整结束；启动脚本在运行期间被编辑，bash 在后续解析到旧文件偏移时中断，导致自动汇总未执行。`s5_summary.json` 用启动脚本内同一段 Python 逻辑补跑，`s_evidence.json` 与 `offline_cal.json` 分别用对应工具补跑；`COMPLETED` 已手工补写。数据文件没有缺失。

运行后已删除 `upgrade_sigma_min`：本次运行 close 路径 182 次升级全部 ≥0.52、frame 原本不检查，因此删除不会改变本次日志中的升级事件；今后 close 路径的低 sigma 升级不再被拦截。

## 3. 总体结果

### 3.1 最终指标

| 指标 | 值 |
| :--- | ---: |
| episodes | 396 |
| success | 164 |
| SR ↑ | 41.41% |
| Oracle ↑ | 58.08%（230/396） |
| SPL ↑ | 0.1803 |
| SoftSPL ↑ | 0.2795 |
| steps / episode ↓ | 184.78 |

n=396 时 SR 的 95% 置信区间约为 **36.6%–46.3%**；单个 4–5 pp 的差异不宜直接解释为机制胜负。

### 3.2 同四场景历史工作点

以下都使用 `5cdEh9F2hJL`、`TEEsavR23oF`、`mv2HUxq3B53`、`wcojb4TFT35`，每场景 99 集，共 396 集。

| 工作点 | SR ↑ | Oracle ↑ | SPL ↑ | SoftSPL ↑ | steps ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| VLFM（2D 基线） | **45.71%**（181/396） | 52.53% | **0.2487** | **0.3020** | 174.73 |
| Raw VLVM ScanRefer | 44.44%（176/396） | 50.76% | 0.1812 | 0.2477 | **143.40** |
| P0/SP1 | 43.69%（173/396） | **63.13%** | 0.1737 | 0.2508 | 220.08 |
| P1 | 42.68%（169/396） | 61.87% | 0.1689 | 0.2495 | 217.50 |
| **S5 Final（本文）** | **41.41%（164/396）** | **58.08%** | **0.1803** | **0.2795** | **184.78** |
| vlvm v1 | 36.61% | 54.55% | 0.1607 | 0.2622 | 188.74 |

与 P0/SP1 相比：SR **−2.27 pp**、Oracle **−5.05 pp**、SPL **+0.0066**、SoftSPL **+0.0287**、steps **−35.30**。与 vlvm v1 相比：SR **+4.80 pp**、Oracle **+3.53 pp**、SPL **+0.0196**、SoftSPL **+0.0173**、steps **−3.96**。

同 episode 配对比较（Final vs P0/SP1）：

| 项 | 集数 |
| :--- | ---: |
| 两者均成功 | 134 |
| 仅 Final 成功 | 30 |
| 仅 P0/SP1 成功 | 39 |
| 两者均失败 | 193 |
| 净差 | **−9** |

净差 −9，discordant 30 vs 39，McNemar 精确检验 $p\approx0.336$。因此本轮不能声称最终机制优于 P0/SP1；可以确认的是它在更低步数下取得了略高的 SoftSPL/SPL，但 SR 与 Oracle 没有同步提高。

> 注意：S5 A/B、S6、S7 的三场景为 `5cdEh9F2hJL`、`QaLdnwvtxbs`、`mL8ThkuaVTM`，与最终四场景只有 `5cdEh9F2hJL` 重叠；不能把 41.41% 与 40.74%、39.39%、35.69% 直接比较。

## 4. 逐场景结果

| 场景 | episodes | success | SR ↑ | Oracle ↑ | SPL ↑ | SoftSPL ↑ | steps ↓ | ΔSR vs P0/SP1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv2HUxq3B53` | 99 | 50 | **50.51%** | 68.69% | **0.2157** | 0.2895 | 193.52 | +0.01 pp |
| `5cdEh9F2hJL` | 99 | 39 | 39.39% | 57.58% | 0.1572 | 0.2556 | 220.09 | +0.99 pp |
| `wcojb4TFT35` | 99 | 38 | 38.38% | 55.56% | 0.1828 | **0.2951** | **115.95** | −7.12 pp |
| `TEEsavR23oF` | 99 | 37 | 37.37% | **50.51%** | 0.1655 | 0.2779 | 209.56 | −3.03 pp |

逐场景失败结构：

| 场景 | FP | FN | never_saw（可行） | never_saw（可能不可行） |
| :--- | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 43 | 13 | 4 | 0 |
| `TEEsavR23oF` | 23 | 0 | 9 | 30 |
| `mv2HUxq3B53` | 37 | 6 | 6 | 0 |
| `wcojb4TFT35` | 29 | 6 | 13 | 13 |
| 合计 | **132** | **25** | **32** | **43** |

观察：

- `mv2HUxq3B53` 仍是最高 SR 场景；`wcojb4TFT35` 的 SR 比 P0/SP1 低 7.12 pp，是本轮最大场景回退。
- `TEEsavR23oF` 的 Oracle 只有 50.51%，其中 30 集为 likely-infeasible 的 never_saw；该场景的 SR 上限本身较低。
- `5cdEh9F2hJL` 的 Oracle 与 P0/SP1 基本持平（57.58% vs 57.60%），但 FP 43、FN 13，说明该场景的主要问题不是“看不到”，而是错误目标与错误停止。

## 5. 类别结果与失败结构

### 5.1 逐类别

| 类别 | episodes | success | SR ↑ | Oracle ↑ | FP | FN | never_saw | steps ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chair` | 86 | 59 | **68.60%** | 80.23% | 25 | 0 | 2 | **99.6** |
| `bed` | 76 | 41 | 53.95% | 75.00% | 22 | 1 | 12 | 131.8 |
| `couch` | 82 | 43 | 52.44% | 70.73% | 39 | 0 | 0 | 145.1 |
| `tv` | 60 | 12 | 20.00% | 26.67% | **38** | 6 | 4 | 202.5 |
| `toilet` | 71 | 9 | 12.68% | 42.25% | 8 | **18** | **36** | **338.3** |
| `potted plant` | 21 | 0 | 0.00% | 0.00% | 0 | 0 | **21** | 310.5 |

### 5.2 难类与易类

| 分组 | episodes | success | SR | 失败占全部失败 |
| :--- | ---: | ---: | ---: | ---: |
| 易类：`chair` + `bed` + `couch` | 244 | 143 | **58.61%** | 101 / 232（43.5%） |
| 难类：`toilet` + `tv` + `potted plant` | 152 | 21 | **13.82%** | 131 / 232（56.5%） |

难类的失败结构：

| 项 | 数值 | 占对应总量 |
| :--- | ---: | ---: |
| FP | 46 | 46 / 132 = 34.8% |
| FN | 24 | 24 / 25 = 96.0% |
| never_saw | 61 | 61 / 75 = 81.3% |

- `potted plant` 21 集全部 never_saw，且 Oracle=0；这批样本本身不可达或场景中无目标，不能作为检测/导航机制的失败证据。
- `toilet` 的主要问题是没看到或跟丢：36 次 never_saw、18 次 FN；成功 9 集，平均步数 338.3。
- `tv` 的主要问题是错误目标：FP 38，占全部 FP 的 28.8%；成功 12 集，Oracle 26.67%。

### 5.3 失败原因与停止

| 失败原因 | 集数 | 占比 |
| :--- | ---: | ---: |
| `false_positive` | 132 | 33.33% |
| `false_negative` | 25 | 6.31% |
| `never_saw_target ... feasible` | 32 | 8.08% |
| `never_saw_target ... likely_infeasible` | 43 | 10.86% |
| `did_not_fail`（成功） | 164 | 41.41% |

- 82/396 集（20.7%）在覆盖目标前错误停止；占 132 个 FP 的 **62.1%**。
- 46 集达到 500 步上限，全部失败；成功集平均 148.05 步，失败集平均 210.74 步，总体中位数 140 步。
- Oracle 成功但最终失败的 75 集（75/230 = 32.6%）中，FP 50、FN 25；说明“看到过目标”仍不能稳定转化为成功。

## 6. 候选、记忆与生命周期

### 6.1 候选与 provisional gate

| gate outcome | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `confirmed` | 209 | 115 | 94 | 55.02% |
| `merge_existing` | 1,730 | 1,217 | 513 | **70.35%** |
| `provisional` | 534 | 195 | 339 | 36.52% |
| `missing` | 298 | 27 | 271 | 9.06% |
| `rejected` | 3,380 | 387 | 2,993 | 11.45% |
| **合计** | **6,151** | **1,941** | **4,210** | **31.56%** |

- `merge_existing` 的 precision 最高；`rejected` 和 `missing` 的 precision 均低于 12%，gate 对 FP 的拦截有效。
- `provisional` 新档 precision 只有 36.52%，但保留了 195 TP，是后续升级与 stable 池的主要来源。

### 6.2 记忆事件与升级

| 事件 | n |
| :--- | ---: |
| `new` | 743 |
| `merge` | 1,730 |
| `upgrade` | 327 |
| `transition` | 254 |
| `delete` | 143 |

`s5_summary` 的 `memory_written=2,473` 等于 `new 743 + merge 1,730`；`s_penalty_admitted=1,026` 与 §7.1 的 admitted 为同一口径。

`new` 状态：

| 状态 | n | TP | precision |
| :--- | ---: | ---: | ---: |
| `provisional` | 534 | 195 | 36.52% |
| `confirmed` | 209 | 115 | 55.02% |

升级路径：

| 路径 | n | TP | precision | 说明 |
| :--- | ---: | ---: | ---: | :--- |
| `close_reobservation` | 182 | 97 | 53.30% | 全部满足 `sigma_c_latest ≥ 0.52` |
| `frame_upgrade` | 145 | 60 | 41.38% | 105 条低于 0.52（38 TP / 67 FP） |
| **合计** | **327** | **157** | **48.01%** | — |

删除前的 0.52 全局门历史回放：

| 口径 | n | TP | FP | precision | 相对实际 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 实际升级事件 | 327 | 157 | 170 | 48.01% | — |
| 全局 `sigma_c_latest ≥ 0.52` | 222 | 119 | 103 | **53.60%** | −38 TP / −67 FP |

`upgrade_sigma_min` 已删除。本次运行中 close 182 次全部 ≥0.52、frame 原本不检查，因此删除对本次事件无影响；表中的 0.52 仅作为删除前的历史回放，用来说明低 sigma frame 的精度风险。

### 6.3 删除与状态迁移

| 事件 / 原因 | n | 事件中 TP | TP 占比 | 说明 |
| :--- | ---: | ---: | ---: | :--- |
| `delete` 合计 | 143 | 35 | 24.48% | 删除中 TP 占比越低，说明越偏向清 FP |
| `stable_stale_purge` | 117 | 21 | 17.95% | 以 FP 为主，符合预期 |
| `stable_miss_delete` | 14 | 3 | 21.43% | 以 FP 为主 |
| `stable_total_miss` | 12 | 11 | **91.67%** | 几乎全是真目标，需复核 |

状态迁移：

| 迁移 / 原因 | n | TP | precision |
| :--- | ---: | ---: | ---: |
| `near_miss_downgrade` | 197 | 74 | 37.56% |
| `stale_demote` | 57 | 6 | 10.53% |
| `confirmed → stable` | 114 | 52 | 45.61% |
| `provisional → stable` | 140 | 28 | 20.00% |

`stable_stale_purge` 与 `stable_miss_delete` 主要清 FP，方向正确；但 `stable_total_miss=8` 在本次日志中 12 次删除有 11 次删到真目标。该阈值样本少（12 次），不能仅凭本轮断言应改，但它是目前最明确的删除侧风险点。

## 7. S-penalty 证据

### 7.1 原始检测与准入

| 口径 | n | TP | FP | precision | TP recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| raw detections | 5,853 | 1,914 | 3,939 | 32.70% | 100% |
| raw `c ≥ 0.60` | 1,307 | 923 | 384 | 70.62% | 48.22% |
| S-penalty admitted | 1,026 | 792 | 234 | **77.19%** | 41.38% |

> `s_evidence` 的 5,853 条是 proposal-shadow 可重建的检测；`s5_summary` 的候选总数为 6,151，差额 298 即 gate 的 `missing` 分支。

S-penalty 把 precision 从 32.70% 提高到 77.19%，但 TP recall 只有 41.38%；高精度准入与覆盖不足是同时存在的。

### 7.2 候选置信阈值

| `c` 阈值 | n | TP | FP | precision | TP recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `≥0.45` | 5,853 | 1,914 | 3,939 | 32.70% | 100% |
| `≥0.50` | 3,805 | 1,647 | 2,158 | 43.29% | 86.05% |
| `≥0.55` | 2,326 | 1,329 | 997 | 57.14% | 69.44% |
| `≥0.60` | 1,307 | 923 | 384 | 70.62% | 48.22% |

### 7.3 S 分支与分布

| 分支 | n | TP | FP | precision | admitted |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `covered-high` | 2,296 | 1,155 | 1,141 | 50.30% | 718 |
| `covered-low` | 2,711 | 361 | 2,350 | 13.32% | 98 |
| `uncovered` | 846 | 398 | 448 | **47.04%** | 210 |

- covered 候选中 TP 的 S 中位数 **0.1782**，FP **0.1188**；S 仍有区分度。
- 本次 covered 候选的 S 最小值约 **0.088**，高于 `s_min=0.07`，因此最终配置不会触发下限 clamp。
- `uncovered` 分支 precision 47.04%，高于 `covered-low` 的 13.32%；这支持 `uncovered_w=1.0`，即不因为“未覆盖”而额外重罚。

### 7.4 类别与距离

| 类别 | n | TP | precision |
| :--- | ---: | ---: | ---: |
| `toilet` | 198 | 99 | **50.00%** |
| `couch` | 1,472 | 543 | 36.89% |
| `chair` | 1,667 | 599 | 35.93% |
| `bed` | 1,415 | 469 | 33.14% |
| `tv` | 992 | 204 | 20.56% |
| `potted plant` | 109 | 0 | 0.00% |

近距离检测 precision 反而更低：0–1 m 为 15.51%，1–2 m 为 35.45%，2–3 m 为 44.56%，≥3 m 为 50.18%；但近距离 coverage 更高（0–1 m 94.94%，≥3 m 48.03%）。这说明近场候选数量多但噪声大，不能只靠置信度解决。

## 8. 最终运行离线标定

### 8.1 口径

| 项 | 值 |
| :--- | :--- |
| 输入 | `outputs/s5_final_4scene_2026-10-26_102914/detections/` |
| episode / success | 396 / 164 |
| 轨道数 | 743 |
| 停止标定 | 最后决策点的 nav goal 回放 |
| 工具 | `ft_tsp3d/eval_tools/nav_offline_cal.py` |
| 输出 | `offline_cal.json`、`offline_cal.log` |
| 启动仿真 | 否 |

### 8.2 停止规则回放

| 规则 | 允许 | TP 保留 | TP 丢失 | FP 保留 | FP 规避 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| baseline（最后 nav goal） | 318 | 170 | 0 | 148 | 0 |
| `num_obs≥2` | 224 | 139 | 31 | 85 | 63 |
| `num_obs≥3` | 160 | 111 | 59 | 49 | 99 |
| `state=confirmed` | 292 | **164** | **6** | 128 | 20 |
| 最近 3 步内重检 | 210 | 114 | 56 | 96 | 52 |
| `num_obs≥2` 且最近 3 步重检 | 151 | 96 | 74 | 55 | 93 |
| `num_obs≥2` 且 `max_sigma≥0.60` | 146 | 107 | 63 | 39 | 109 |
| `num_obs≥2` 且最近距离 ≤2.5 m | 209 | 130 | 40 | 79 | 69 |

结论：

- `state=confirmed` 保留 **164 TP**，与实际 success 完全一致；只丢失 6 TP，同时规避 20 FP。
- `num_obs≥2` 和 `num_obs≥3` 分别丢失 31 / 59 TP；加最近 3 步重检丢失 56 TP；加 `max_sigma≥0.60` 丢失 63 TP。停止侧继续加严的代价明显大于收益。
- 这与最终机制“仅 confirmed 可停、无 `num_obs` 门槛、无当前帧重确认、无视角扩张”一致。

### 8.3 升级规则回放

全部轨道：

| 规则 | n | TP | FP | precision | recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 743 | 315 | 428 | 42.40% | 100% |
| `num_obs≥2` | 439 | 239 | 200 | 54.44% | 75.87% |
| `num_obs≥3` | 273 | 170 | 103 | 62.27% | 53.97% |
| `max_sigma≥0.60` | 320 | 185 | 135 | 57.81% | 58.73% |
| `num_obs≥2` 且 `max_sigma≥0.60` | 232 | 156 | 76 | **67.24%** | 49.52% |
| `num_obs≥2` 且最近距离 ≤2.5 m | 381 | 217 | 164 | 56.96% | 68.89% |

provisional 子集：

| 规则 | n | TP | FP | precision | recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 534 | 199 | 335 | 37.27% | 100% |
| `num_obs≥2` | 316 | 151 | 165 | 47.78% | 75.88% |
| `num_obs≥3` | 181 | 101 | 80 | 55.80% | 50.75% |
| `max_sigma≥0.60` | 111 | 69 | 42 | 62.16% | 34.67% |
| `num_obs≥2` 且 `max_sigma≥0.60` | 109 | 68 | 41 | **62.39%** | 34.17% |

结论：

- `num_obs≥2` 是 recall/precision 的折中点；再加 `max_sigma≥0.60` 可把 precision 提到 62%–67%，但 recall 降到约 34%–50%，会丢掉大量中置信 TP。
- `frame_upgrade` 是低 sigma 升级的主要入口：105/145 条低于 0.52（38 TP / 67 FP），precision 仅 36.19%。全局加 0.52 门可把 precision 提高到 53.60%，但会丢掉 38 TP。该参数已删除，升级统一为 `num_obs`/距离；这条历史回放说明删除后需要接受低 sigma frame 的 FP 风险，若后续转为精度优先再单独实验升级门。

### 8.4 删除代理

| 口径 | TP 轨道 / FP 轨道 |
| :--- | ---: |
| 轨道数 | 315 / 428 |
| 最大连续未匹配 ≥2 帧 | 77.78% / 50.23% |
| 最大连续未匹配 ≥3 帧 | 62.54% / 32.48% |

TP 的连续未匹配比例反而高于 FP，不能把“连续未匹配帧数”直接当作删除门；实际删除结构见 §6.3。`stable_stale_purge` 与 `stable_miss_delete` 主要清 FP，`stable_total_miss` 则需要复核。

### 8.5 准入扩张

> 工具内“现状 0.55 / 1.5–4.0”是旧口径标签，不代表本次运行。本次实际为 `0.50 / 1.0–4.0`。

| 准入口径 | n | TP | precision | 有 TP 集数 |
| :--- | ---: | ---: | ---: | ---: |
| 0.55 / 1.5–4.0（旧标签） | 1,421 | 1,037 | 72.98% | 177 |
| 0.50 / 1.5–4.0 | 1,825 | 1,210 | 66.30% | 195 |
| 0.55 / 1.0–4.0 | 1,573 | 1,110 | 70.57% | 185 |
| **0.50 / 1.0–4.0（本次）** | **2,155** | **1,343** | **62.32%** | **202** |
| 0.45 / 1.0–4.0 | 2,937 | 1,562 | 53.18% | 214 |

- 相对 0.55/1.5–4.0，本次口径把有 TP 集数从 177 提到 202（+25），precision 从 72.98% 降到 62.32%。
- 相对 0.55/1.0–4.0，有 TP 集数 +17，precision −8.25 pp。
- 继续降到 0.45/1.0–4.0 只再增加 12 个有 TP 集，precision 降到 53.18%；这支持把 `provisional_sigma` 保持在 0.50，而不是继续降线。

### 8.6 标定结论到最终参数

| 机制 / 参数 | 最终值 | 标定依据 | 判定 |
| :--- | :--- | :--- | :--- |
| 停止资格 | 仅 `confirmed` | `state=confirmed` 保留 164 TP、丢 6、规避 20 FP | 保留 |
| 停止半径 | 0.90 m | 与 confirmed 停止配合；无 n_obs/重检/扩张 | 保留 |
| `t_S` / `s_min` | 0.14 / 0.07 | S 准入 precision 77.19%；covered 最小 S≈0.088 > 0.07 | 保留 |
| `uncovered_w` | 1.0 | uncovered precision 47.04% > covered-low 13.32% | 保留 |
| provisional | 0.50 / 1.0–4.0 m | 有 TP 集数 +25，precision 62.32% | 保留 |
| 升级 sigma | 已删除 | close 182 次全部 ≥0.52；frame 低 sigma 41.38% | 采用 num_obs + 距离，覆盖优先 |
| stable 删除 | stale 60 / miss 5 / total 8 | stale/miss 主要清 FP；total_miss 11/12 为 TP | 保留但复核 total_miss |

## 9. 与历史离线标定的关系

历史标定细节见 [VLVM-S5cal-Exp.md](VLVM-S5cal-Exp.md)；本文只保留与最终机制直接相关的结论，不修改该文件。

| 历史结论 | 对最终机制的影响 |
| :--- | :--- |
| S7 full 的 `num_obs≥3` 停止门槛保留 108 TP、丢 26 TP；`state=confirmed` 保留 116 TP、丢 18 TP | 最终改为仅 `confirmed` 可停，不设 n_obs 门槛 |
| S7 full 的最后 nav goal 从实际 134 TP 变为 evidence 排序 122 TP；S7 0.52 为 120→113 | 删除 evidence 目标选择 |
| S7 full 视角扩张 504 次（97 通过 / 407 失败），steps 216.86 | 删除扩张复核 |
| `s_min=0.07` 来自三场景 18,457 条 covered 候选最小值 0.087629 | 最终保留 0.07；本次 covered 最小约 0.088，未触发 clamp |
| `uncovered_w=1.0` 因 uncovered 与整体 precision 接近 | 本次 uncovered precision 47.04%，继续保留 |
| 历史准入从 0.55/1.5–4.0 扩到 0.50/1.0–4.0 会以 precision 换覆盖 | 本次 202 个有 TP 集、precision 62.32%，支持该折中 |

## 10. 判定与后续

### 10.1 已确认

- 最终机制在 4 场景 396 集上完整跑通，无 GD 参与；S5 的 provisional、stable、只允许 confirmed 停止、无 n_obs/重检/扩张的路径可以被日志复现。
- S-penalty 的 precision 提升明确（32.70% → 77.19%）；停止只保留 `confirmed` 时，离线回放保留 164 TP、丢 6 TP，优于 `num_obs≥2/3`。
- 覆盖是主瓶颈：S-penalty 只保留 41.38% 的 TP；难类 `toilet`、`tv`、`potted plant` 合计 152 集，仅 21 集成功。

### 10.2 未确认 / 风险

- 本轮没有超过同四场景 P0/SP1：SR −2.27 pp，配对净差 −9，统计上不显著；不能把“最终机制”写成全面优于历史工作点。
- 运行代码是重构前版本；当前仓库已删除 GD/fb 死路径并重命名内部字段，尚未用重构后代码重跑 396 集。
- `upgrade_sigma_min` 已删除；低 sigma frame 升级仍会发生（105 条，38 TP / 67 FP），这是覆盖优先的显式代价，FP 风险仍在。
- `stable_total_miss` 删除 11/12 为 TP；样本小，但属于明确的删除侧风险。
- 单次运行、单 seed、396 集，SR 置信区间约 ±4.9 pp；场景间差异不应过度归因。

### 10.3 后续优先级

1. **覆盖**：优先处理 `toilet`、`tv`、`potted plant` 的候选覆盖与中置信 TP 准入；GD 若要重新接入，应先在离线日志上过覆盖门，而不是直接改停止。
2. **目标身份/FP**：`tv` 贡献 38 FP，`couch` 39 FP；先解决分类与错误锁定，继续加严停止只会继续丢 TP。
3. **待复核机制**：`stable_total_miss` 的阈值；`upgrade_sigma_min` 已删除，若后续转为精度优先，再单独实验升级 sigma 门，不恢复当前半生效参数。
4. **复现**：用当前重构后代码重跑同一四场景，确认行为与本文日志一致，再决定是否把本文作为代码级基线。

## 11. 产物与复现

| 文件 | 内容 |
| :--- | :--- |
| `detections/episodes.jsonl` | 396 条 episode 结果与失败原因 |
| `detections/candidates.jsonl` | 6,151 条候选 |
| `detections/memory_events.jsonl` | 3,197 条 new/merge/upgrade/transition/delete |
| `detections/track_frames.jsonl` | 18,568 条轨道帧记录 |
| `s5_summary.json` | 导航、gate、memory、生命周期汇总 |
| `s_evidence.json` / `s_evidence.log` | S-penalty 与 proposal shadow 统计 |
| `offline_cal.json` / `offline_cal.log` | 停止、升级、删除代理、准入扩张回放 |
| `navigation.log` | 完整导航日志，396/396 |
| `resolved_config.yaml` | 实际 Hydra 配置 |
| `COMPLETED` | 导航完成与手工后处理标记 |

补跑命令：

```bash
cd /home/iair/vlvm/VLVM

/home/iair/miniconda3/envs/vlvm/bin/python -u ft_tsp3d/eval_tools/s_evidence.py \
  --jsonl outputs/s5_final_4scene_2026-10-26_102914/detections \
  --sigma-tar 0.60 --t-s 0.14 --require-proposal-shadow \
  --out outputs/s5_final_4scene_2026-10-26_102914/s_evidence.json

/home/iair/miniconda3/envs/vlvm/bin/python -u ft_tsp3d/eval_tools/nav_offline_cal.py \
  --run-dir outputs/s5_final_4scene_2026-10-26_102914 \
  --out outputs/s5_final_4scene_2026-10-26_102914/offline_cal.json
```

`scripts/_nav_s5final.sh` 当前 `bash -n` 通过；本次失败的原因是运行期间编辑了正在执行的 bash 文件，导致后续解析到旧偏移。以后重跑时不要在运行中编辑该脚本。
