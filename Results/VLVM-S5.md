# Project VLVM Stage5：Rank V2 与 Gate 重构

## 阶段定位

S5 以 S4 最终权重 VLVM V1（原 R1v1 E3）及 G0、G1、G3/F0、F1 历史结果为基础重新设计 Rank 与 Gate。已有 Gate 结果只作为机制边界和训练方式对照，不直接延续为 S5 的最终 Gate 方案。

| S5 内容 | 定位 |
| :--- | :--- |
| 历史 R1v2 | 保留原始设计、训练结果与失败原因，作为 Rank V2 重构的反例 |
| Rank V2 | 从单 query 内部排序升级为同一 view 内跨 query、跨 bbox 的全局候选排序 |
| Gate V2 | 不在 VLVM V1 上继续追加旧 Gate；从 ScanRefer 起点重新进行 Rank V2 + Gate 联合微调 |
| Epoch 收敛 | 分别研究 base + Rank V2 的核心收敛，以及冻结核心后的 Gate-only 收敛 |

## 历史 R1v2 记录
### 1. 状态

| 项目 | 状态 |
| :--- | :--- |
| R1v2：$L_{cross}+L_{intra}$ | 已完成 E1–E3 训练与 E3 全量评测 |
| R1v2 结论 | 仅增强同一正 query 内高分候选的头部排序；整体识别、定位和高召回区分能力退化 |
| 主线状态 | Rank 实验暂停；训练代码已回退到 R1v1 $L_{cross}$ |
| 后续升级 | 同一 view 内聚合不同 query 的全部 bbox 候选做全局排序，保留 R1v1 pairwise softplus 形式 |
| 未定义项 | 正负配对、query 间分数归一化、hardest-negative 规则和 loss 归一化后续单独讨论 |

### 2. view、query 与 bbox 层级

当前采样粒度是 view。一个 view 包含多个 query record，但每次模型前向只输入同一份 view 点云和一个 query；一个训练 step 对该 view 的各 query 分别前向，再聚合损失。

| 层级 | R1v2 行为 |
| :--- | :--- |
| view | 一个训练 step；各 query 共享同一份局部点云 |
| query record | 每次前向只处理一个 query；训练集平均约 3.89 query/view |
| delivery Top-1 | 在线交付候选；$L_{cross}$ 使用各 query 的 Top-1 |
| training Top-R | R1v2 额外导出的训练候选；$L_{intra}$ 只在同一正 query 内配对 |

### 3. R1v2 设计

R1v2 从 `tsp3d_scanrefer.pth` 独立初始化，保持 U1、数据、优化器、3 epoch 和 R1v1 $L_{cross}$ 不变，从 epoch 1 新增同 query Top-R 分支：

$$
\mathcal L_{intra}=\operatorname{softplus}
\left(m-z_{gt}^{(q)}+z_{hard\_wrong}^{(q)}\right),
\qquad
\mathcal L_{rank}=1.0\,\mathcal L_{cross}+1.0\,\mathcal L_{intra}.
$$

`L_intra` 只在单个 `label=1` query 的 Top-R 候选内计算：选择 IoU 最大且达到 0.25 的候选作为正项，其余候选中 raw logit 最高者作为错误项。负 query 不构造 intra 正项。

| 控制项 | R1v1 对照 | R1v2 |
| :--- | :--- | :--- |
| 初始化 | `tsp3d_scanrefer.pth` | `tsp3d_scanrefer.pth` |
| 解冻边界 | U1 | U1 |
| epoch / seed | 3 / 0 | 3 / 0 |
| 基础学习率 | G1 `1e-4` / G2 `5e-5` / G3 `1e-5` | 相同 |
| weight decay / clip | `1e-4` / `0.1` | 相同 |
| rank margin / IoU 门 | `0.2` / `0.25` | 相同 |
| rank loss | $L_{cross}$ | $L_{cross}+L_{intra}$ |
| rank 权重 | `1.0` | `1.0 + 1.0` |
| 每轮训练步数 | 23,031 | 23,031 |

### 4. R1v2 E3 结果

| 档位 | 可训练参数 | Epoch | cls_base | bg | L1 | GIoU | rank_cross | rank_intra |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1v1 | 4,795,908 | 1 | 0.1337 | 0.0817 | 2.7453 | 1.1003 | 0.9416 | — |
| R1v1 | 4,795,908 | 2 | 0.1495 | 0.0736 | 2.7409 | 1.0915 | 0.9780 | — |
| R1v1 | 4,795,908 | 3 | 0.1463 | 0.0636 | 2.6290 | 1.0663 | 0.9285 | — |
| R1v2 | 4,795,908 | 1 | 0.1888 | 0.0513 | 3.0239 | 1.1534 | 0.4011 | 0.8725 |
| R1v2 | 4,795,908 | 2 | 0.1931 | 0.0526 | 2.8904 | 1.1162 | 0.3962 | 0.8629 |
| R1v2 | 4,795,908 | 3 | 0.2007 | 0.0390 | 2.8513 | 1.1108 | 0.3678 | 0.8606 |

全量 E3 使用 14 个 val 场景、2,638 个 view、2,590 个正查询和 2,638 个负查询；`sigma_tar=0`，共 5,228 次调用，`failures=0`。

| 指标组 | 指标 | R1v1 E3 | R1v2 E3 | R1v2 − R1v1 |
| :--- | :--- | ---: | ---: | ---: |
| 定位 | hit@0.25 | 33.63% | 27.30% | -6.33pp |
| 定位 | hit@0.50 | 14.59% | 12.74% | -1.85pp |
| 定位 | mean IoU | 0.1957 | 0.1553 | -0.0404 |
| 排序 | AUROC | 0.8363 | 0.8372 | +0.0009 |
| 排序 | AUPRC | 0.4511 | 0.5063 | +0.0552 |
| 排序 | pair acc | 0.9127 | 0.9109 | -0.0018 |
| 高召回拒识 | FPR@TPR90 / FPR@TPR95 | 0.4232 / 0.5901 | 0.5452 / 0.7297 | +0.1220 / +0.1396 |
| 高召回拒识 | Precision@R90 / R95 | 0.2983 / 0.2436 | 0.2054 / 0.1692 | -0.0929 / -0.0744 |
| 分数 | 正确均值 / 错误均值 / gap | 0.5198 / 0.3137 / 0.2061 | 0.5767 / 0.2859 / 0.2908 | +0.0569 / -0.0278 / +0.0847 |
| 校准 | ECE-15 / Brier / NLL | 0.1814 / 0.1428 / 0.4552 | 0.1900 / 0.1245 / 0.4145 | +0.0086 / -0.0183 / -0.0407 |

**5.1 六类别 hit@0.25**

| 权重 | bed | chair | couch | potted plant | toilet | tv |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1v1 E3 | 49.18% | 28.42% | 41.71% | 16.67% | 41.67% | 14.89% |
| R1v2 E3 | 38.25% | 27.04% | 31.05% | 0.60% | 39.58% | 12.94% |
| 变化 | -10.93pp | -1.38pp | -10.66pp | -16.07pp | -2.09pp | -1.95pp |

**5.2 命中迁移**

| R1v1 / R1v2 hit@0.25 状态 | 数量 |
| :--- | ---: |
| 两者均命中 | 619 |
| 仅 R1v1 命中 | 252 |
| 仅 R1v2 命中 | 88 |
| 两者均 miss | 1,631 |

| 类别 | R1v2 相对 R1v1 的命中数变化 |
| :--- | ---: |
| bed | -60 |
| chair | -13 |
| couch | -56 |
| potted plant | -27 |
| toilet | -2 |
| tv | -6 |

**5.3 分数分布**

| 类别 | R1v1 mean / p10 / p50 / p90 / ≥0.70 | R1v2 mean / p10 / p50 / p90 / ≥0.70 |
| :--- | :--- | :--- |
| hit | 0.5198 / 0.3170 / 0.5557 / 0.6510 / 3.6% | 0.5767 / 0.2313 / 0.6400 / 0.8266 / 43.1% |
| miss | 0.3986 / 0.1939 / 0.3871 / 0.6238 / 3.0% | 0.3454 / 0.1372 / 0.2843 / 0.7005 / 10.1% |
| absent | 0.2584 / 0.1188 / 0.2435 / 0.4218 / 0.08% | 0.2435 / 0.1337 / 0.2249 / 0.3717 / 0.45% |

| 补充指标 | R1v1 E3 | R1v2 E3 |
| :--- | ---: | ---: |
| hit score sd | 0.1342 | 0.2248 |
| Cohen's d（概率） | 1.4356 | 1.4833 |
| R90 阈值 | 0.31695 | 0.22988 |
| R95 阈值 | 0.25537 | 0.18382 |
| 视图级排序违例率 | 8.73% | 8.91% |

**结果分析**

R1v2 把高分命中进一步推高，因此 AUPRC、gap、Brier 和 NLL 改善；但 hit 分布明显变宽，p10 从 0.3170 降至 0.2313。为覆盖 90%/95% 真阳性，阈值必须显著下移，导致 FPR 上升、Precision 下降。

同视图 pair acc 基本持平，六类 hit@0.25 全部退化，potted plant 从 16.67% 降至 0.60%。这说明 $L_{intra}$ 的收益局限在同一正 query 内部的高分候选，没有提升正确目标、错误定位和 target-absent 之间的整体区分能力。R1v2 不能替代 R1v1 E3 作为 no-S 锚点。



## S5 Rank V2 方向

S5 不继续对单个正 query 内部的 Top-R 做局部补丁，而是将同一 view 中所有 query 产生的全部 bbox 聚合为统一候选池，直接学习跨 query、跨 bbox 的全局排序。损失形式暂时保留 R1v1 的 pairwise softplus：

$$
\mathcal L_{rank}^{global}=\operatorname{softplus}
\left(m-z_{pos}+z_{hard\_neg}\right).
$$

| 设计项 | S5 当前方向 / 待确定内容 |
| :--- | :--- |
| 排序范围 | 同一 view 内聚合所有 query 的全部 bbox，不再局限于单个正 query 的 Top-R |
| 正候选 | 以达到 IoU 门的正确定位为基础；每个 GT 只取最佳 bbox，还是保留多个达标 bbox仍需确定 |
| 负候选 | 同时覆盖 target-absent、正 query 错框和其他 query 错框 |
| hardest negative | 比较 view 全局最高负 logit 与按负样本类型分层选择两种规则 |
| 分数可比性 | 检查不同 query 的 logit 尺度；必要时增加 view 内归一化，但不以压缩分数动态范围为代价 |
| loss 归一化 | 比较按 view、正候选数和有效 pair 数归一化，避免 query 数量改变单个 view 的梯度权重 |
| 主指标约束 | Rank 改善必须同时满足 hit@0.25、hit@0.50、mean IoU 不发生系统性退化 |
| 计算预算 | 确定全部 bbox 导出、候选池构建与流式重放的显存和时间上限 |

Rank V2 的目标不是继续拉大少量高分命中的分数，而是改善正确目标、错误定位与 target-absent 之间的全局次序。历史 R1v2 已证明 AUPRC、gap 或校准改善不能替代定位和高召回拒识指标。

## S5 Gate V2 方向

### 重新开始联合微调

S5 Gate 不继续在 VLVM V1 上追加现有 Gate 模块，而是从 `tsp3d_scanrefer.pth` 重新初始化一轮完整训练，在 epoch 1 即同时引入 Rank V2 与 Gate。这里的“从头开始”指从 ScanRefer 预训练权重开始新的项目微调，不是随机初始化 TSP3D。

新的联合训练必须区别于 G1：G1 使用 `detach_inputs=True`，Gate loss 只训练 `presence_gate.*`，不构成 Gate 对共享表征的真正联合优化。S5 需要明确 Gate 梯度进入哪些共享层、梯度权重和启用时机，并保留不让 Gate 破坏定位能力的约束。TSP3D 仍是主模型，Gate 只负责 target-present / target-absent 的存在性判断和 veto。

### Gate 优化方向

| 方向 | S5 设计重点 |
| :--- | :--- |
| 优化目标 | 在固定 `recall_open≥0.99` 下提高 target-absent intercept，而不是只追求 Gate AUROC/AUPRC |
| 正样本保护 | 将 false-close、hit@0.25、hit@0.50 和 mean IoU 作为硬约束，防止分数治理收益掩盖定位损失 |
| Rank V2 协同 | Rank V2 负责 view 内候选次序，Gate 负责 query 是否存在；联合训练但保持任务边界清晰 |
| Gate 输入 | 在 TSP3D 3D 特征基础上评估 Top-1 分数、Top-1/Top-2 margin、候选分数离散度和跨 query 上下文 |
| 负样本 | F1 hard-v2 作为历史数据基线，后续加入冻结/在线核心产生的高置信 target-absent、类别混淆和近邻误检 |
| 数据课程 | 从普通负样本过渡到 hard negative，控制 replace-source 比例，避免早期困难负样本主导训练 |
| 损失形式 | 比较二分类损失、同 view 正负 pairwise 排序和显式召回约束；损失统一由训练 pipeline 消费 |
| 梯度治理 | 记录 Gate 与定位、Rank V2 梯度的冲突，限制 Gate 进入共享层的范围和权重 |
| 标定 | 每个 epoch 在固定召回约束下独立标定阈值，不使用测试结果反向选择训练参数 |
| 消融口径 | 同一 checkpoint 同时报告 `threshold=0` 的 core 与标定阈值下的 gate，单独计算 Gate 净贡献 |

当前结果只支持“旧 Gate 作用存在但有限，冻结核心的独立训练更稳健”，不能证明独立训练是 Gate 的理论最优解。S5 的联合微调用于验证共享表征是否能提供当前冻结特征中缺失的存在性信息；如果联合训练仍以定位退化换取小幅拒识收益，则不保留该 Gate 路线。

## Epoch 的两种收敛路线

### 路线 A：base + Rank V2 核心收敛

从 `tsp3d_scanrefer.pth` 开始新的 base + Rank V2 训练，使用完整学习率周期扩展 epoch，并保存每轮 checkpoint。该路线寻找的是验证集上的更优收敛点，不宣称能够证明或获得理论全局最优。checkpoint 选择首先约束 hit@0.25、hit@0.50 和 mean IoU，再比较 AUROC、AUPRC、FPR@TPR90/95、pair acc 与校准。

G0 从 R1v1 E3 尾端 `LR=1e-6` 继续训练 E4/E5，只能说明当前低学习率续训发生回退，不能代替从 ScanRefer 使用完整调度进行更大 epoch 的收敛实验。

### 路线 B：Gate-only 独立收敛

在路线 A 选出的核心候选 checkpoint 上冻结全部核心参数，只训练 Gate，直到 Gate validation 指标和固定召回阈值稳定。该路线用于测量给定核心表征下 Gate 的能力上限，并与从 ScanRefer 开始的 Rank V2 + Gate 联合训练形成严格对照。

| 对照 | 目的 |
| :--- | :--- |
| base + Rank V2 core | 确定不含 Gate veto 的核心定位与排序上限 |
| 同一 core + Gate-only | 测量冻结表征下 Gate 的净贡献与独立收敛边界 |
| ScanRefer → base + Rank V2 + Gate 联合训练 | 测量 Gate 参与共享表征优化后的完整路线收益 |
| 联合 checkpoint 的 `core` / `gate` | 分离核心训练轨迹变化与 Gate veto 的即时作用 |

两条收敛路线必须使用匹配的数据顺序、训练预算和评测集。最终路线判断同时比较完整训练路线和固定 checkpoint 的 Gate 净作用，不能把更强的初始化核心误判为 Gate 机制收益。

以上内容为 S5 设计记录，当前不实现、不启动实验。
