# VLVM-S6gd3：GD 四挡目标排序与 promoted-as-gd-tsp 三场景结果

> 运行目录：`outputs/s6_gd3_restart_2026-10-28_182915`  
> 运行时间：`2026-10-28 18:30` 至 `2026-10-29 01:17`，约 `6:46:35`  
> 场景：`5cdEh9F2hJL` / `QaLdnwvtxbs` / `mL8ThkuaVTM`，共 `3 × 99 = 297` episodes  
> 对照基线：S6gd2（`outputs/s6_gd_2026-10-27_restart`，130/297）  
> 下一版结果：[VLVM-S6gd4-exp.md](VLVM-S6gd4-exp.md)

## 1. 本版机制

| 项 | S6gd3 实际运行 |
| :--- | :--- |
| `promoted_from_gd_only` | 由 TSP3D 合并提升时直接写 `origin="gd-tsp"`；不再作为独立 `gd-only-tsp` 类别。`promoted_from_gd_only=true` 只保留为 provenance |
| 选择层级 | **S6gd3 运行使用**：`gd-tsp（含 promoted） > gd-only > confirmed > provisional` |
| gd-only conf | `enable_gd_only_conf_gate=false`；`0.50/0.45` 只作为参考阈值并写 `would_reject` |
| STOP | 仅 `gd-tsp confirmed`（含 promoted gd-only）可停；纯 TSP confirmed、gd-only、provisional 不可停 |
| STOP shadow | 覆盖所有被选中的 confirmed（含 gd-tsp/promoted）与 gd-only；只读，不改变动作 |
| field shadow | `field_shadow.jsonl` 只读，不参与选择、准入或 STOP |

> **重要**：本文结果是 S6gd3 运行，使用 `gd-tsp > gd-only > confirmed > provisional`。根据本文结果，下一次收尾实验改为 `gd-tsp > confirmed > gd-only > provisional`；不要把两次排序混为同一 arm。

## 2. 总体结果

| 指标 | S6gd3 | S6gd2 | 差值 |
| :--- | ---: | ---: | ---: |
| success | **139 / 297** | 130 / 297 | **+9** |
| SR | **46.80%** | 43.77% | **+3.03 pp** |
| Oracle | **138 / 297（46.46%）** | 129 / 297（43.43%） | **+9** |
| SPL | **0.2157** | 0.2008 | +0.0149 |
| SoftSPL | **0.2867** | 0.2782 | +0.0085 |
| steps | 153.55 | 144.60 | +8.95 |

配对统计：

- success：both 114 / only S6gd3 25 / only S6gd2 16 / neither 142；McNemar 精确双侧 `p≈0.422`，**success 提升不显著**。
- Oracle：both 127 / only S6gd3 11 / only S6gd2 2 / neither 157；`p≈0.0449`，**Oracle 提升显著**。

结论：本版把更多 episode 带入目标可见区域，Oracle 增益明确；但 success 的 +9 仍不足以在配对检验下单独成立，不能把 SR +3.03 pp 解释为稳定机制增益。

## 3. 分场景

| 场景 | S6gd3 success / Oracle | S6gd2 success / Oracle | success Δ |
| :--- | ---: | ---: | ---: |
| `5cdEh9F2hJL` | **51 / 62** | 45 / 59 | **+6** |
| `QaLdnwvtxbs` | **80 / 72** | 76 / 66 | **+4** |
| `mL8ThkuaVTM` | 8 / 4 | 9 / 4 | **−1** |

- 增益主要来自 `5cd` 和 `QaL`；
- `mL8` 仍然几乎不可解：8 success，Oracle 仅 4；
- `mL8` 的 48 个失败仍归为 `never_saw_target_did_not_travel_stairs_likely_infeasible`，不应作为机制主 KPI。

## 4. 分目标类别

| 类别 | n | S6gd3 success / Oracle | S6gd2 success / Oracle | success Δ |
| :--- | ---: | ---: | ---: | ---: |
| bed | 70 | 37 / 43 | 43 / 43 | **−6** |
| chair | 68 | **46 / 45** | 41 / 43 | **+5** |
| couch | 73 | **30 / 33** | 26 / 27 | **+4** |
| toilet | 43 | 6 / 17 | 6 / 16 | 0 |
| tv | 43 | **20 / 0** | 14 / 0 | **+6** |

- `tv` 提升最明显，但 Oracle 仍为 0，说明 tv 的 success 主要来自命中/停止口径，不是目标覆盖；
- `bed` 回退 −6，和 gd-only 全类别开放、gd-only 最终目标增多同时出现；
- `chair`、`couch` 的增益与 gd-tsp/promoted 提升更一致。

## 5. 失败原因

| failure_cause | S6gd3 | S6gd2 | Δ |
| :--- | ---: | ---: | ---: |
| `did_not_fail` | 139 | 130 | +9 |
| `false_positive` | **102** | 81 | **+21** |
| `never_saw_target_did_not_travel_stairs_likely_infeasible` | 48 | 63 | −15 |
| `never_saw_target_did_not_travel_stairs_feasible` | 3 | 14 | −11 |
| `false_negative` | 1 | 9 | −8 |
| `bad_stop_true_positive` | 2 | 0 | +2 |
| `timeout_true_positive` | 2 | 0 | +2 |

关键判断：

- success 增加的同时，`false_positive` 从 81 增加到 102；不能用总 SR 掩盖 FP 回退；
- `never_saw` 总数从 77 降到 51，Oracle 增益与此一致；
- 新增 4 个 true-positive 停止异常（2 bad stop、2 timeout），数量很小，但值得在下一版继续观察。

## 6. 目标来源与排序效果

### 6.1 最终目标来源

| 最终来源 | S6gd3 goals / TP / success | S6gd2 goals / TP / success |
| :--- | ---: | ---: |
| `gd-tsp`（含 promoted） | **187 / 144（77.0%） / 138（73.8%）** | 142 / 108（76.1%） / 106（74.6%） |
| `gd-only` | **53 / 16（30.2%） / 0（0%）** | 21 / 15（71.4%） / 0（0%） |
| `confirmed`（纯 TSP） | **1 / 0 / 0** | 55 / 24（43.6%） / 23（41.8%） |
| `provisional` | 4 / 0 / 0 | 5 / 0 / 0 |

- S6gd3 的 success 主要来自 gd-tsp：187 个最终目标中 138 成功；
- gd-only 最终目标从 21 增至 53，但 **53 个全部失败**；其中 16 个最终目标本身是 TP，只是不可 STOP；
- 纯 confirmed 最终目标从 55 个降到 1 个，说明新排序把 confirmed 大量让位给 gd-only/gd-tsp；
- gd-only 的最终目标在 S6gd3 中不是“探索起点”，而是 episode 末尾仍无法升级/停止的残留目标。

### 6.2 gd-tsp 内部：track vs promoted

| 最终 gd-tsp 子类 | goals | TP | TP precision | success |
| :--- | ---: | ---: | ---: | ---: |
| track gd-tsp | 92 | 67 | 72.8% | 60（65.2%） |
| promoted gd-only | **95** | **77** | **81.1%** | **78（82.1%）** |

虽然 promoted 的 step-level precision 低于 track（57.7% vs 72.0%），但其最终目标 precision 和 episode success 反而更高。统一为 gd-tsp 并允许 STOP 是有数据支持的，不应重新拆成独立类别。

### 6.3 逐步目标行

| 来源 | nav_goal 行数 | TP / precision | 行级 success |
| :--- | ---: | ---: | ---: |
| `gd-tsp` | 2570 | 1655 / 64.4% | 1490 / 58.0% |
| `gd-only` | 7974 | 2554 / 32.0% | 3943 / 49.4% |
| `confirmed` | 427 | 159 / 37.2% | 297 / 69.6% |
| `provisional` | 407 | 109 / 26.8% | 205 / 50.4% |

- 行级 precision：gd-tsp > confirmed > gd-only > provisional；
- 行级 success：confirmed > gd-tsp > provisional > gd-only，说明 confirmed 即使不可 STOP，其目标选择仍较可靠；
- gd-only 行数最多（7974），但 precision 和行级 success 都最低，符合“探索池”的定位。

## 7. STOP shadow 分析

`stop_shadow.jsonl` 共 10,960 行；其中：

- `stop_eligible_current=true`：2,559 行；
- `stop_eligible_current=false`：8,401 行；
- 所有行的 `weights`/shadow 写入正常，未影响动作。

### 7.1 实际 gd-tsp STOP

对最终 gd-tsp episode 的最后一个 stop-eligible 样本：

| 口径 | episodes | goal TP | precision | success |
| :--- | ---: | ---: | ---: | ---: |
| 当前规则（距离 ≤0.9 m） | 181 | 94 | 51.9% | **136（75.1%）** |
| 再加 `num_obs>=3` | 141 | 76 | 53.9% | 109 |
| 再加 `num_obs>=5` | 100 | 58 | 58.0% | 77 |
| 再加 `gd_hits>=3` | 147 | 82 | 55.8% | 109 |
| 再加 `gd_hits>=5` | 115 | 69 | 60.0% | 86 |

提高 `num_obs`/`gd_hits` 能小幅提高 stop 目标 precision，但会损失大量成功 episode；当前 `pointnav_stop_radius=0.9 m` + gd-tsp 资格不应再加硬门。

### 7.2 纯 confirmed / gd-only 是否恢复 STOP

| 来源 | 距离 ≤0.9 m 的最终目标 | TP | success | 结论 |
| :--- | ---: | ---: | ---: | :--- |
| 纯 TSP confirmed | 6 | **0** | 5 | 不恢复 |
| gd-only | 11 | **3** | 3 | 不开放 |

纯 confirmed 的 0/6 TP 与 gd-only 的 3/11 TP 都不支持恢复 STOP；保持当前“仅 gd-tsp 可停”最安全。

## 8. gd-only 二次判据 conf

### 8.1 conf 与 TP

| 事件 | n | TP | precision | AUC（conf→TP） |
| :--- | ---: | ---: | ---: | ---: |
| `gd_only_new` | 729 | 313 | 42.9% | **0.574** |
| `gd_only_merge` | 2,937 | 1,630 | 55.5% | **0.555** |

分 band：

| conf | new n / TP | merge n / TP |
| :--- | ---: | ---: |
| `[0.40,0.45)` | 278 / 108 | 541 / 254 |
| `[0.45,0.50)` | 140 / 50 | 490 / 246 |
| `[0.50,0.55)` | 104 / 40 | 395 / 219 |
| `[0.55,0.60)` | 43 / 25 | 347 / 188 |
| `>=0.60` | 164 / 90 | 1164 / 723 |

### 8.2 如果开启 conf gate

以当前参考阈值回放：

- `gd_only_new`：拒绝 413 条（TP155），保留 316 条（TP158）；precision 从 37.5% 提升到 50.0%，但损失约一半 TP；
- `gd_only_merge`：拒绝 541 条（TP254），保留 2,396 条（TP1,376）；precision 从 47.0% 提升到 57.4%。

结论：**conf 只有弱排序能力（AUC≈0.55–0.57），且会直接损失大量 TP；gd-only 的核心作用是探索和后续 TSP3D 提升，不应开启二次 conf gate。**  
最终机制保持：

```yaml
enable_gd_only_conf_gate: false
gd_only_conf_min: 0.50        # reference only
gd_only_merge_conf_min: 0.45  # reference only
```

## 10. 局限

- 只有单一三场景 run，没有对四挡顺序、gd-only 类别、STOP 阈值做独立消融；
- success 的 McNemar 不显著，不能宣称稳定 +3 pp；
- gd-only 最终目标 53 个全部失败，说明部分 episode 的瓶颈仍是“无法从 gd-only 升级/停止”，不是单纯排序；
- `mL8` 仍然主要由 likely-infeasible 覆盖失败主导，不能把其低 SR 当作 GD 机制失败。
