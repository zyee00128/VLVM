# VLVM S5 实验结果

## 1. 实验口径

| 标识 | 场景 | episodes | `sigma_tar` | `sigma_prop` | `t_S` | `out_of_bounds` |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| P1 G4 baseline | `5cdEh9F2hJL`、`mL8ThkuaVTM`、`QaLdnwvtxbs` | 297 | 0.60 | 0.60 | 0.14 | reject |
| S5 P1 全量 baseline | 10 场景 | 990 | 0.60 | 0.45 | 0.14 | reject |
| S5 P1 三场景 baseline | 同三场景 | 297 | 0.60 | 0.45 | 0.14 | reject |
| S5-S1 | 同三场景 | 297 | 0.60 | 0.45 | 0.14 | `geom_unknown` 保留 |

| 固定项 | S5 P1 三场景 baseline | S5-S1 |
| :--- | :--- | :--- |
| checkpoint | `data/vlvm_v1/vlvm_v1.pth` | `data/vlvm_v1/vlvm_v1.pth` |
| S-penalty | 开启 | 开启 |
| `s_min` / radius / `uncovered_w` | 0.1 / 0.5 m / 1.0 | 0.1 / 0.5 m / 1.0 |
| occ | 开启，`min_voxels=8`、`min_ratio=0.001` | 开启，`min_voxels=8`、`min_ratio=0.001` |
| GD proposer | 关闭 | 关闭 |
| 其余 fallback / memory / 导航机制 | S5 P1 | 与 S5 P1 相同 |

## 2. 导航总体结果

| 配置 | 场景 | episodes | success | SR ↑ | Oracle ↑ | SPL ↑ | SoftSPL ↑ | steps ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 G4 baseline | 3 | 297 | 116 | 39.06% | 43.43% | 0.1608 | 0.2369 | 172.5724 |
| S5 P1 全量 baseline | 10 | 990 | 398 | 40.20% | 54.04% | 0.1586 | — | 200.7600 |
| S5 P1 三场景 baseline | 3 | 297 | 114 | 38.38% | 43.43% | 0.1597 | 0.2364 | 173.5960 |
| S5-S1 | 3 | 297 | 115 | 38.72% | 43.43% | 0.1607 | 0.2382 | 171.1953 |

| S5-S1 对照口径 | success Δ | SR Δ | Oracle Δ | SPL Δ | SoftSPL Δ | steps Δ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 对 S5 P1 三场景 baseline（同 `sigma_prop=0.45`） | +1 | +0.34 pp | 0.00 pp | +0.0010 | +0.0018 | -2.4007 |
| 对 P1 G4 baseline（`sigma_prop=0.60`） | -1 | -0.34 pp | +0.00 pp | -0.0001 | +0.0013 | -1.3771 |

## 3. 逐场景结果

| 场景 | 配置 | episodes | success | SR ↑ | Oracle ↑ | SPL ↑ | SoftSPL ↑ | steps ↓ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | P1 G4 | 99 | 38 | 38.38% | 57.58% | 0.1385 | 0.2113 | 256.9798 |
|  | S5 P1 | 99 | 38 | 38.38% | 57.58% | 0.1385 | 0.2111 | 256.9596 |
|  | S5-S1 | 99 | 38 | 38.38% | 57.58% | 0.1393 | 0.2137 | 252.9899 |
| `mL8ThkuaVTM` | P1 G4 | 99 | 10 | 10.10% | 3.03% | 0.0623 | 0.1506 | 90.2626 |
|  | S5 P1 | 99 | 10 | 10.10% | 3.03% | 0.0623 | 0.1506 | 90.2626 |
|  | S5-S1 | 99 | 10 | 10.10% | 3.03% | 0.0623 | 0.1506 | 90.2626 |
| `QaLdnwvtxbs` | P1 G4 | 99 | 68 | 68.69% | 69.70% | 0.2817 | 0.3489 | 170.4747 |
|  | S5 P1 | 99 | 66 | 66.67% | 69.70% | 0.2783 | 0.3476 | 173.5657 |
|  | S5-S1 | 99 | 67 | 67.68% | 69.70% | 0.2805 | 0.3502 | 170.3333 |

## 4. 逐目标类别结果

| 目标类别 | episodes | S5 P1 success | S5 P1 SR | S5-S1 success | S5-S1 SR | SR Δ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bed | 70 | 35 | 50.00% | 35 | 50.00% | 0.00 pp |
| chair | 68 | 39 | 57.35% | 39 | 57.35% | 0.00 pp |
| couch | 73 | 27 | 36.99% | 27 | 36.99% | 0.00 pp |
| toilet | 43 | 1 | 2.33% | 1 | 2.33% | 0.00 pp |
| tv | 43 | 12 | 27.91% | 13 | 30.23% | +2.33 pp |
| 合计 | 297 | 114 | 38.38% | 115 | 38.72% | +0.34 pp |

## 5. 失败构成

| 配置 | success | false positive | false negative | never-saw feasible | never-saw likely-infeasible | 合计 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 G4 | 116 | 63 | 25 | 28 | 65 | 297 |
| S5 P1 | 114 | 65 | 25 | 28 | 65 | 297 |
| S5-S1 | 115 | 66 | 25 | 26 | 65 | 297 |

## 6. 候选准入漏斗

| 阶段 | S5 P1 n | TP | FP | precision | S5-S1 n | TP | FP | precision | n Δ | TP Δ | FP Δ | precision Δ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-occ | 6,099 | 1,883 | 4,216 | 30.87% | 5,876 | 1,851 | 4,025 | 31.50% | -223 | -32 | -191 | +0.63 pp |
| `geom_confirmed` / pass | 5,701 | 1,823 | 3,878 | 31.98% | 5,532 | 1,809 | 3,723 | 32.70% | -169 | -14 | -155 | +0.72 pp |
| 硬几何 reject | 253 | 6 | 247 | 2.37% | 249 | 6 | 243 | 2.41% | -4 | 0 | -4 | +0.04 pp |
| `out_of_bounds` | 145 | 54 | 91 | 37.24% | 95 | 36 | 59 | 37.89% | -50 | -18 | -32 | +0.65 pp |
| post-occ | 5,701 | 1,823 | 3,878 | 31.98% | 5,627 | 1,845 | 3,782 | 32.79% | -74 | +22 | -96 | +0.81 pp |
| S-penalty admitted | 549 | 389 | 160 | 70.86% | 563 | 393 | 170 | 69.80% | +14 | +4 | +10 | -1.05 pp |
| 其中 `out_of_bounds` admitted | 0 | 0 | 0 | — | 28 | 13 | 15 | 46.43% | +28 | +13 | +15 | — |

## 7. occ 原因明细

| occ reason | 配置 | candidates | TP | FP | admitted | admitted TP | admitted FP |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `air_penetration_detected` | S5 P1 baseline | 19 | 0 | 19 | 0 | 0 | 0 |
|  | S5-S1 | 18 | 0 | 18 | 0 | 0 | 0 |
| `insufficient_occ_voxels` | S5 P1 baseline | 229 | 6 | 223 | 0 | 0 | 0 |
|  | S5-S1 | 226 | 6 | 220 | 0 | 0 | 0 |
| `low_occupancy_ratio` | S5 P1 baseline | 5 | 0 | 5 | 0 | 0 | 0 |
|  | S5-S1 | 5 | 0 | 5 | 0 | 0 | 0 |
| `out_of_bounds` | S5 P1 baseline | 145 | 54 | 91 | 0 | 0 | 0 |
|  | S5-S1 | 95 | 36 | 59 | 28 | 13 | 15 |
| `pass` | S5 P1 baseline | 5,701 | 1,823 | 3,878 | 549 | 389 | 160 |
|  | S5-S1 | 5,532 | 1,809 | 3,723 | 535 | 380 | 155 |

## 8. `out_of_bounds` episode 统计

| 指标 | S5 P1 baseline | S5-S1 | Δ |
| :--- | ---: | ---: | ---: |
| 存在 pre-occ TP 的 episode | 162 | 161 | -1 |
| 全部 TP 被 occ 删除的 episode | 2 | 0 | -2 |
| 含 `out_of_bounds` TP 的 episode | 8 | 7 | -1 |
| 含 `out_of_bounds` FP 的 episode | 24 | 24 | 0 |
| 含任意 `geom_unknown` 的 episode | — | 26 | — |
| 含 admitted `geom_unknown` 的 episode | — | 9 | — |
| 含 admitted TP `geom_unknown` 的 episode | — | 5 | — |
| 含 admitted FP `geom_unknown` 的 episode | — | 6 | — |

| S5-S1 episode 子集 | episodes | baseline 失败 → S5-S1 失败 | baseline 失败 → S5-S1 成功 | baseline 成功 → S5-S1 失败 | baseline 成功 → S5-S1 成功 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 含任意 `geom_unknown` | 26 | 20 | 1 | 0 | 5 |
| 含 admitted `geom_unknown` | 9 | 5 | 1 | 0 | 3 |
| 含 admitted TP `geom_unknown` | 5 | 1 | 1 | 0 | 3 |
| 含 admitted FP `geom_unknown` | 6 | 5 | 0 | 0 | 1 |

## 9. 配对 episode 结果

| baseline success | S5-S1 success | episodes |
| ---: | ---: | ---: |
| 0 | 0 | 182 |
| 0 | 1 | 1 |
| 1 | 0 | 0 |
| 1 | 1 | 114 |
| 合计 | — | 297 |

| 场景 | 失败→失败 | 失败→成功 | 成功→失败 | 成功→成功 | 合计 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 61 | 0 | 0 | 38 | 99 |
| `mL8ThkuaVTM` | 89 | 0 | 0 | 10 | 99 |
| `QaLdnwvtxbs` | 32 | 1 | 0 | 66 | 99 |
| 合计 | 182 | 1 | 0 | 114 | 297 |

| 唯一变化 episode | 目标类别 | baseline | S5-S1 | 含 `geom_unknown` | admitted `geom_unknown` | admitted TP `geom_unknown` |
| :--- | :--- | ---: | ---: | :---: | :---: | :---: |
| `QaLdnwvtxbs:78` | tv | 失败 | 成功 | 是 | 是 | 是 |

## 10. memory 与导航目标

| 层级 | 配置 | n | TP | FP | TP 占比 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| memory new | S5 P1 baseline | 226 | 132 | 94 | 58.41% |
|  | S5-S1 | 233 | 133 | 100 | 57.08% |
| memory merge | S5 P1 baseline | 323 | 257 | 66 | 79.57% |
|  | S5-S1 | 330 | 260 | 70 | 78.79% |
| memory delete | S5 P1 baseline | 7 | 6 | 1 | 85.71% |
|  | S5-S1 | 8 | 6 | 2 | 75.00% |
| memory 全事件 | S5 P1 baseline | 556 | 395 | 161 | 71.04% |
|  | S5-S1 | 571 | 399 | 172 | 69.88% |
| nav goal | S5 P1 baseline | 2,553 | 1,352 | 1,201 | 52.96% |
|  | S5-S1 | 2,493 | 1,247 | 1,246 | 50.02% |

## 11. S5-S1 判定门

| 判定项 | 对照 | 对照值 | S5-S1 | Δ | 门槛 | 是否命中 |
| :--- | :--- | ---: | ---: | ---: | :--- | :---: |
| SR | P1 G4 baseline | 39.06% | 38.72% | -0.34 pp | ≥39.06% | 否 |
| SR | S5 P1 三场景 baseline | 38.38% | 38.72% | +0.34 pp | 不下降 | 是 |
| 全部 TP 被 occ 删除的 episode | S5 P1 三场景 baseline | 2 | 0 | -2 | 下降 | 是 |
| false-positive failure rate | S5 P1 三场景 baseline | 21.89%（65/297） | 22.22%（66/297） | +0.34 pp | 增量 ≤1 pp | 是 |
| false-positive failure rate | P1 G4 baseline | 21.21%（63/297） | 22.22%（66/297） | +1.01 pp | 增量 ≤1 pp | 否 |

## 12. 结果分析

### 12.1 unknown 与 hard reject 的分离有效

硬几何 reject 共 249 个候选，其中仅 6 TP、243 FP，TP 占比为 2.41%；`geom_unknown` 共 95 个候选，其中 36 TP、59 FP，TP 占比为 37.89%。两类候选的 TP 密度相差 35.48 pp，说明 `out_of_bounds` 不等价于负几何证据，将其从 hard reject 中拆出具有明确的数据依据。

但 `geom_unknown` 也不等价于正常几何确认。S5-S1 中，正常 pass 候选经 S-penalty 后准入 535 个，其中 380 TP、155 FP，precision 为 71.03%；unknown 准入 28 个，其中 13 TP、15 FP，precision 仅为 46.43%。unknown 的质量处于 hard reject 与正常 pass 之间，呈现独立第三档证据，而不是可以直接并入任意一端。

### 12.2 候选 TP 恢复有限

S5-S1 直接放行的 unknown 中有 13 个 admitted TP，同时带入 15 个 admitted FP。完整在线结果中，总 admitted TP 从 389 增至 393，仅净增 4；总 admitted FP 从 160 增至 170，净增 10。正常 pass 路径的 admitted TP 同时从 389 降至 380，unknown 新增的 13 个 TP 抵消该变化后才形成最终的 TP +4。

该差异不能解释为固定候选集上的 `+4 TP / +10 FP`：occ 改动在首次 unknown 写入 memory 后改变了后续轨迹、帧数与 TSP3D 输入，pre-occ 总数也由 6,099 变为 5,876。因此候选总量变化包含闭环轨迹效应；能够直接确认的是，unknown 路径本身准入了 13 TP 与 15 FP。

### 12.3 TP 恢复只转化为一个新增成功 episode

S5 P1 baseline 中有 2 个 episode 的全部 TP 被 occ 删除，S5-S1 中该数量降为 0，说明 S1 完成了预定的局部机制目标。但机制实际覆盖面较小：297 集中仅 26 集出现 unknown，9 集存在 admitted unknown，5 集存在 admitted TP unknown。

这 5 集中，3 集在 baseline 与 S5-S1 中均成功，1 集由失败变为成功，1 集在两次实验中均失败。唯一新增成功为 `QaLdnwvtxbs:78`，目标类别为 tv，且该集确实包含 admitted TP unknown。全部 297 集的配对结果中，没有成功转失败，但也只有这一集由失败转为成功。

因此，unknown/reject 分离恢复了真实候选并产生了可验证的成功案例，但候选级 TP 恢复并未规模化转换为 episode SR：同参数 baseline 的 success 仅从 114 增至 115，SR 增量为 0.34 pp。

### 12.4 下游 memory 与 nav goal 质量未同步改善

| 下游指标 | S5 P1 baseline | S5-S1 | Δ |
| :--- | ---: | ---: | ---: |
| memory new TP 占比 | 58.41% | 57.08% | -1.33 pp |
| memory 全事件 TP 占比 | 71.04% | 69.88% | -1.16 pp |
| nav goal TP 占比 | 52.96% | 50.02% | -2.94 pp |
| nav goal TP 数 | 1,352 | 1,247 | -105 |
| nav goal FP 数 | 1,201 | 1,246 | +45 |

候选层新增 TP 没有带来 memory 与 nav goal 层的同步质量提升。memory 全事件新增 4 TP、11 FP，TP 占比下降 1.16 pp；nav goal 的 TP 占比下降 2.94 pp。该结果与 S5 已记录的目标选择问题一致：当前导航按最近质心选择，候选的质量状态不参与优先级，候选层恢复的 TP 不会自动转化为更高质量的导航目标。

由于两次实验的闭环轨迹与总步数不同，nav goal 数量变化不能全部归因于 unknown 本身；但本次结果至少没有提供“放行 unknown 可改善下游目标质量”的证据。

### 12.5 baseline 口径影响阶段判断

| 对照口径 | 配置差异 | success 变化 | SR 变化 | false-positive failure rate 变化 | 阶段门结果 |
| :--- | :--- | ---: | ---: | ---: | :--- |
| S5 P1 三场景 baseline | 与 S5-S1 同为 `sigma_prop=0.45` | 114 → 115 | +0.34 pp | +0.34 pp | SR 不下降；FP 增量 ≤1 pp |
| P1 G4 baseline | `sigma_prop=0.60`，S5-S1 为 0.45 | 116 → 115 | -0.34 pp | +1.01 pp | SR 未达到 39.06%；FP 增量超过 1 pp |

同参数主对照表明 S5-S1 没有造成 success episode 回退，并增加了 1 个成功；历史 G4 判定门则未通过。两种结论并不矛盾，因为 G4 与 S5-S1 的 proposal 候选池阈值不同，不能作为严格单变量配对。S5-S1 的严格配对证据应以 `sigma_prop=0.45` 的 S5 P1 三场景子集为主，G4 仅保留为 S5 文档规定的历史门槛。

### 12.6 S5-S1 阶段结论

| 结论项 | 结果 |
| :--- | :--- |
| `out_of_bounds` 是否应继续等同 hard reject | 否；unknown TP 占比 37.89%，hard reject TP 占比 2.41% |
| `geom_unknown` 是否可直接等同正常目标 | 数据不支持；unknown admitted precision 46.43%，正常 pass admitted precision 71.03% |
| 是否恢复被 occ 删除的 TP episode | 是；全部 TP 被 occ 删除的 episode 从 2 降至 0 |
| 是否形成稳定 SR 增益 | 未形成；同参数对照仅增加 1/297 个成功，+0.34 pp |
| 是否改善下游目标质量 | 否；memory 与 nav goal TP 占比均下降 |
| 是否通过历史 G4 判定门 | 否；SR 38.72% < 39.06%，FP failure rate 增量 1.01 pp |
| S1 机制证据 | 支持 `geom_confirmed / geom_unknown / geom_reject` 三档语义，不支持将 unknown 无条件按正常目标处理 |
