# VLVM Stage 4：S 语义仲裁与 TSP3D 置信度区分度优化

> S4 坚持 TSP3D 3D-native 主线。R1v1 与 R2 保留在本文；R1v2 的设计、实验、结果分析及后续 Rank/Gate 重构统一维护在 [VLVM-S5.md](VLVM-S5.md)。
> S4 的模型微调阶段已收敛并关闭。R3-G0/G1/G3/F1 均作为历史对照保留；最终模型锁定为 **VLVM V1（原 R1v1 E3）**。下一步在该权重上固定 `sigma_tar=0.60`，直接进入 S-penalty 机制优化；后续模型重构进入 [VLVM-S5.md](VLVM-S5.md)。


## Phase G：最终 S-penalty 参数与机制优化

最终权重为 `data/vlvm_v1/vlvm_v1.pth`，Presence Gate 关闭；直接复用 Phase D 已完成的四场景 no-S 搜索结果：

$$\text{VLVM V1, no-S:}\qquad \sigma_{tar}=0.60.$$

后续 S-penalty 实验不再安排 sigma grid、窄搜索或 Gate 阈值标定；`0.60` 是固定控制变量，不作为 S 参数联调项。固定 VLVM V1、Presence Gate 关闭、raw `sigma_tar=0.60`，不再进行模型训练或置信度阈值搜索。

### G1：先验证 S 是否值得使用

在相同候选池上比较 raw $c$ 与 S：

- S-only AUROC/AUPRC；
- covered 条件下 TP/FP 分布；
- covered-high、covered-low、uncovered 三路 precision/recall；
- S 对 raw $c$ 的增量区分度；
- 按场景、类别和目标距离分层。

若 S 在 covered 子集上没有稳定增量，最终配置保持 S-penalty 关闭，不进入复杂机制。

### G2：现有 penalty 参数收口

固定 `sigma_tar=0.60`，保持冲突门和在线 2D 检测源关闭，按单变量顺序搜索：

1. $t_S\in\{0.10,0.12,0.15,0.18\}$；
2. 在最佳 $t_S$ 上搜索 $floor\in\{0.3,0.5,0.7\}$；
3. 在最佳组合上搜索 $radius\in\{0.3,0.5,0.7\}\,\mathrm m$；
4. `uncovered_w` 首轮固定 1.0。

先用 detection replay 淘汰明显劣势档，只将 Pareto 档送入四场景导航。

### G3：S 主导仲裁升级

在相同 proposal pool 上比较：

| 档位 | 准入/排序方式 |
| :--- | :--- |
| P0 | S 关闭，仅 raw $c$ |
| P1 | 当前 $c\cdot w_S(S)$ penalty |
| P2 | 显式加权质量分 $q(c,S)$，S 权重高于 $c$ |
| P3 | P2 + 质量带内按距离选择 |

机制升级必须同时完成：

- 拆分候选生成阈值 $\sigma_{prop}$ 与记忆准入阈值 $\tau_{admit}$；
- covered-high / covered-low / uncovered 分支独立；
- occ gate 保持在 S 前；
- 2.5D region map 与 memory-life 保持不变；
- 质量优先、距离次级，禁止纯最近候选选择；
- 所有档使用完全相同的候选池，避免候选数差异污染结论。

最终 S 机制判定：

1. 四场景平均 SR 不低于 P0；
2. SPL 不出现系统性下降；
3. fp、覆盖前错误停止和入记忆 FP 明确减少；
4. fn 不因 covered-low 惩罚明显增加；
5. 收益至少跨多个场景成立；
6. 若复杂仲裁不优于 P1，则保留简单 penalty；若 P1 也不优于 P0，则关闭 S。


## 0. S4 微调收敛

S4 最终权重确定为 **VLVM V1**，即历史 R1v1 E3。权重已由 `data/tsp3d_ft/rank/r1_5080_ckpt_epoch3.pth` 重命名并迁移至 `data/vlvm_v1/vlvm_v1.pth`；对应训练 manifest、训练/验证日志、控制台日志、完成标记和可训练参数审计统一保存在 `data/vlvm_v1/training_logs/`。

选择依据是：R1v1 E3 已在离线定位、置信度排序和 no-S 导航之间形成当前最佳平衡；继续训练至 E4/E5 会改善部分分数分布，但总体定位、AUPRC 和高召回拒识未稳定超过 E3。联合 Gate、原始数据独立 Gate 以及 hard-data-v2 Gate 均只带来微小排序或拒识变化，同时伴随少量定位/正确目标损失，不足以替换主权重。因此 S4 不再追加 epoch，也不把 Gate 装入最终 VLVM V1。

仓库中的 Gate 训练代码只保留一种口径：以 VLVM V1 初始化，冻结全部核心，只使用原始 `local_view` 独立训练 `presence_gate.*`。联合 Gate 与 hard-data/replay Gate 的启动、损失接入和训练前向已删除；历史结果仅用于说明已完成的选择过程。S4 模型微调至此收敛；当前继续完成 S-penalty，后续 Rank V2、Gate V2 与两条 epoch 收敛路线统一在 S5 重新设计。

## 1. 结果分析与问题定责

### 在线前置检测：假阳仍是主因

历史完整在线统计中：

| 证据 | 数值 | 含义 |
| :--- | :--- | :--- |
| TSP3D detection precision | 28.41%（3545/12477） | 写入前候选纯度低 |
| TSP3D TP/FP 原始分数均值 | 0.781 / 0.773 | gap 仅 +0.008，几乎不可分 |
| GD TP/FP 分数均值 | 0.646 / 0.556 | gap +0.089，2D 信号更有区分性但仍不充分 |
| S 的 TP/FP 均值 | 0.171 / 0.130 | 正确 gap 是 **+0.041** |
| 记忆候选 TP 率 | 44.8% | 机制过滤后仍不足一半 |
| `both` / `gd` / `tsp3d` 单源 TP 率 | 78.9% / 48.8% / 27.0% | 交叉证据有效，TSP3D 单源最弱 |
| 失败中的 FP | 884（72.2%） | 假阳停靠是首要失败模式 |
| 真目标覆盖前已因 FP 停错 | 743 集 | 假阳会提前终止探索 |

- S 的均值差为正，说明它具备辅助价值；但 +0.041 的均值差不能证明它足以成为主排序器。

### E3：定位提升，区分度下降

| 指标 | base3090 | scanrefer | 判断 |
| :--- | ---: | ---: | :--- |
| hit@IoU 0.25 | 29.46% | 18.69% | base +10.77pp，识别/定位有效提升 |
| hit@IoU 0.50 | 11.58% | 5.21% | base +6.37pp |
| mean best IoU | 0.1634 | 0.1181 | base +0.0453 |
| 命中框分数均值 | 0.4743 | 0.6328 | base 整体左移 |
| 负样本分数均值 | 0.2232 | 0.3360 | base 的拒识也变强 |
| hit/miss 中位数间隔 | 0.168 | 0.310 | **base 排序间隔约缩小一半** |
| potted plant hit@0.25 | 8.93% | 14.88% | **少样本类发生退化** |

- base 是**框更接近目标，同时正确框的分数优势不足且整体分数标尺漂移**。$\sigma_{tar}$ 只能重新选择工作点，不能恢复排序间隔。
- E3 目前每个正视图只配一个负查询，负样本仍偏容易；它适合判断分数漂移和基础定位，不足以代表导航中会诱发停靠的高分硬 FP。

### 明确问题 / 前提

当前问题拆成两条相互独立、按顺序推进的路线：

1. **机制层：固定模型与原始门限后只优化 S。** 使用 VLVM V1，Presence Gate、GD 和 S 冲突门关闭，固定 `sigma_tar=0.60`；先优化现有乘法 penalty，再比较“S 主导仲裁”。
2. **模型层：直接训练候选置信度的排序能力** 以相同 scanrefer checkpoint 为初始化，做逐级解冻消融；主目标是正确候选压过在线硬假阳，而不是继续优先优化框或中间特征。Presence gate 只作为视图级 target-absent 拒绝器，不代替候选级区分损失。

明确现有前提：

- “3D 模型远弱于 2D 模型”只在**当前在线局部点云与类别短提示分布**下成立，不能推广为 3D 模型的普遍结论。[TSP3D 原论文]明确面向 ScanRefer/ReferIt3D 式完整场景 3D grounding，而非连续导航中的局部视锥壳。TSP3D 在完整扫描式 3DVG 基准上本身具有较强定位能力；本项目的主要矛盾是**训练分布与在线残缺点云不一致**，以及**语义置信度无法区分 TP/FP**。
- E3 能证明 base 的**局部识别/定位能力提升**，但也证明其**置信度排序能力变差**；两者并不矛盾。
- 首轮未进入主线的解耦架构已归档，S4 不再继续该分支。
- 当前 active S 来源是 VLFM 风格的 **2.5D region ValueMap**。保留的 surface 版本仍把 3D 表面点落入 2D $(x,y)$ 桶，并非真正保留完整 3D 语义场。现阶段没有证据支持重建价值地图。


## 2. S-penalty 路线修订

### 当前机制的真实行为

当前流程是：

$$
c \ge \sigma_{tar}
\rightarrow \text{occ gate}
\rightarrow c' = c\cdot w_S(S)
\rightarrow c' \ge \sigma_{tar}
\rightarrow \text{memory}
$$

其中：
- $S\ge t_S$ 时 $w_S=1$；
- $0<S<t_S$ 时 $w_S=\max(floor,S/t_S)$；
- 未覆盖 $S\le0$ 或 `None` 时 $w_S=uncovered\_w=1$，即免罚；
- 进入记忆后，导航目标在可锁候选中按**离机器人最近**选择，置信度与 S 不参与最终候选排序。

* 当前 S 只负责“拒绝谁”，不负责“多个已准入目标中选谁”。高分 FP 一旦通过，最近目标规则仍可能优先导航到它。

* TSP3D 服务端与 policy 都会在 S 查询前按原始 $\sigma_{tar}$ 过滤。

* 当前 $w_S\le1$ 时没有逻辑错误；但若新机制允许高 S 提升低 $c$ 候选，则这些候选在计算 S 前已经消失。

### 第一阶段：只调参数，不改公式

控制变量固定为：模型使用 VLVM V1、Presence Gate 关闭、GD 关闭、occ 开启且参数不变、2.5D region map 不变、memory-life 不变，raw `sigma_tar=0.60` 全程固定。

1. **固定 raw $\sigma_{tar}=0.60$：** 直接复用 VLVM V1 已完成的四场景 no-S 标定，不再进行 sigma grid 或窄搜索。
2. **只调 S：** 先调 $t_S$，再调 floor，最后调查询半径；避免多参数全排列掩盖因果。
3. **建议初始窄范围：** $t_S\in\{0.10,0.12,0.15,0.18\}$，$floor\in\{0.3,0.5,0.7\}$，$radius\in\{0.3,0.5,0.7\}\,m$。`uncovered_w` 首轮固定 1.0，防止把尚未建图误当成低语义证据。
4. **先离线 replay，再跑导航：** 从已有 detection log 计算每组的准入 TP/FP、召回、precision 与 episode 级首次错误锁定；只把 Pareto 档送入四场景 SR/Oracle/SPL。

这一阶段回答的是“S 能否在不改框架的情况下过滤 FP”，不回答“S 能否主导候选选择”。

### 第二阶段：从 penalty 改为语义仲裁

更清晰的机制应拆成四步：

1. **候选生成门：** 用较低的 $\sigma_{prop}$ 保留可能被 S 验证的 TSP3D 候选；occ gate 继续一票否决空腔框。
2. **证据标准化：** 分别把 $c$ 和 covered-S 映射到各自验证集上的可比较尺度。S 原值不是概率，不应直接解释为正确率。
3. **质量融合：**

   $$
   q=\sigma\left(\alpha\,\operatorname{logit}(\hat c)
   +\beta\,\operatorname{logit}(\hat S)+b\right),\quad \beta>\alpha
   $$

   其中 $\hat c$、$\hat S$ 是验证集标定后的分数。若只做无学习版本，可用 $q=\hat c^{\alpha}\hat S^{\beta}$。

4. **记忆仲裁：**$q\ge\tau_{admit}$ 才能进入记忆；多个候选并存时，先按 $q$ 选质量带，再在近似同分候选中用距离决定，不能继续完全按最近目标选择。**质量优先、距离次级**：纯 S 排序会把用于探索的区域级语义先验误当成实例检测分数；纯距离又会放大最近 FP。可采用：先保留 $q\ge q_{max}-\delta$ 的质量带，再从带内选最近候选。

### 未覆盖不是低 S

必须保留三个状态：

| 状态 | 语义 | 建议行为 |
| :--- | :--- | :--- |
| covered-high | 有地图证据且 S 高 | 允许提升排序优先级 |
| covered-low | 有地图证据但 S 低 | 明确惩罚或拒绝 |
| uncovered | 该位置尚无可靠语义观测 | 不当作反证；保留 TSP3D 最低证据门槛，但不可无条件压过 covered-high |

如果把 uncovered 直接当 $S=0$，探索早期召回会显著受损；如果继续完全免罚且按最近选择，它又会成为 FP 绕过 S 的通道。因此 uncovered 应当是单独分支，而不是一个数值。

### 暂不修改语义价值地图

VLFM 的 ValueMap 本来用于为 frontier 提供语言相关探索价值，而非提供实例级检测概率；其成功依赖 RGB/VLM 语义与几何 frontier 的组合。[VLFM 原论文]支持把它作为探索先验，但不支持直接把单点最大值解释为目标存在概率。

当前不改 2.5D region map，原因有三点：

- 既往实验已显示 region 版本优于 surface 路径；
- 当前 S 查询采用半径内最大值，地图构建误差与融合误差尚未和仲裁公式误差分离；
- 项目中的探索/导航图已经定稿，直接重建语义地图会同时改变探索覆盖和检测准入，无法归因。

只有当固定候选集的离线统计表明 S-only 区分度跨场景不稳定，且错误主要来自地图峰值污染/错位时，才进入地图候补实验。候补也应先做轻量可靠性层，而不是恢复“全 3D”路线：为 S 同时记录覆盖次数、观测新鲜度、视角/距离质量；用多次观测的稳健聚合替代单次 max；按目标框投影区域查询而非只查一个近表面点。

## 3. 模型微调路线修订

### 解冻层数实验的正确基线

当前 base 并不是“只微调头部”：它冻结 stem、layer1/2 和 RoBERTa，同时训练 layer3/4、text projector、跨模态/TGP/CBA 相关模块及最终 head。它代表较广泛的顶层解冻端点。

解冻消融必须满足：

- 每一档都从同一个 `tsp3d_scanrefer.pth` 初始化；
- 当前 base 仅作为最宽解冻档的结果对照，不能以 base checkpoint 为起点再冻结，因为此前的参数漂移已经写入权重；
- 固定数据、batch、epoch、优化器、随机种子和 LR 规则；
- 冻结子树保持 `eval()`，防止 BN running statistics 与 Dropout 继续漂移；
- 以候选排序指标早停，不以总 loss 或单一 $\sigma$ 下的 accuracy 早停。

“减少解冻层数”不是越少越好，而是在稳定性与适配能力之间找 Pareto 点。渐进解冻与分层学习率的核心目的正是降低灾难性遗忘；[ULMFiT 原论文](https://aclanthology.org/P18-1031/)给出了 gradual unfreezing 与 discriminative learning rates 的经典实践依据。

### 区分度必须直接监督候选置信度 R1v1

每个候选需要一个“是否为正确目标框”的监督标签：正视图中与目标 GT 匹配且达到 IoU 标准的候选为正；其余候选、负视图候选和在线高分幻觉为负。训练目标分三层：

1. **候选分类损失：** BCE/Focal/ASL 负责概率方向。Focal Loss 的价值是降低大量易负样本的支配，让训练聚焦困难样本；这与当前在线高分 FP 问题一致。
[Focal Loss 原论文](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)

2. **同视图排序损失：** 对正确候选与最高分错误候选施加 pairwise margin：

   $$
   \mathcal L_{rank}=\operatorname{softplus}
   \left(m-z_{pos}+z_{hard\_neg}\right)
   $$

   它直接优化“正确目标必须压过最危险 FP”，比只压低所有负样本更贴近导航 Top-1 决策。pairwise ranking 与 AUC 优化本质相关，可用阈值无关指标判定。
   [PMLR pairwise/AUC 论述](https://proceedings.mlr.press/v30/Agarwal13.html)

3. **在线硬负样本：** 从训练场景 rollout 中回收高分误框、同房间相似物、局部壳和贴墙结构，加入 replay pool。OHEM 的原始工作说明，检测训练的有效增益常来自少量高损失困难候选，而非继续增加大量易背景。
[OHEM 原论文](https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html)

现有 20 个导航评测场景的失败日志只能用于诊断，不能直接回灌训练，否则会产生场景泄漏。硬负例应从训练场景按相同在线流程生成，并按 scene 划分 train/val。

R1v2 的设计、实验、结果分析及后续全局候选排序升级已迁移至 [VLVM-S5.md](VLVM-S5.md)，当前暂停继续升级。


## 4. 统一评测指标

### 指标定义

| 记录字段 | 含义 |
| :--- | :--- |
| `label` | 数据集侧标签：1 = 正查询（该视图内查询类可见）；0 = 负查询（该视图内该类可见像素 < 40，target-absent） |
| `score` | TSP3D Top-1 交付置信度 `dets[0]["confidence"]`（`cls_conv` logit 过 sigmoid）；无框时为 `null`，不进任何均值 |
| `best_iou` | Top-1 框与 GT 的 3D IoU（仅正查询写入记录） |
| `hit25` | `best_iou ≥ 0.25`（仅正查询）；`hit50` 只进聚合，不落 jsonl |
| `has_box` / `pruned_all` | 是否出框 / 是否被剪枝清空 |

**样本类名与表列对照**：

| 类名 | 定义 | 对应表列 |
| :--- | :--- | :--- |
| hit | 正查询且 `hit25=True` | hit@0.25 的分子；**正确均值**的样本集 |
| miss | 正查询且 `hit25=False` | 计入**错误均值** |
| absent | `label=0` 的负查询 | **负查询均值**；同样计入**错误均值** |
| 错误池 | miss ∪ absent | **错误均值** = 该池的 `score` 均值 |

**指标定义**：

| 指标 | 定义 |
| :--- | :--- |
| hit@0.25 / hit@0.50 | 正查询中 `hit25` / `hit50` 的比例 |
| mean IoU | 正查询 Top-1 框与 GT 的 mean best IoU |
| 正确均值 / 错误均值 | hit 集 / miss∪absent 集的 `score` 均值 |
| 负查询均值 | absent 集的 `score` 均值 |
| gap | 正确均值 − 错误均值（绝对尺度读数） |
| AUROC | 随机取一正一错，正确样本分数更高的概率 |
| AUPRC | 精确率-召回率曲线下面积；正例率约 14.6% 时随机基线 ≈ 0.146 |
| FPR@TPR90/95 | 阈值取“正样本召回 ≥90% / ≥95%”的工作点，错误样本越阈比例 |
| Precision@R90/95 | 上述工作点的 TP/(TP+FP) |
| pair acc | 同一视图内“正确正查询分数 > target-absent 负查询分数”的成对比例（tie 计 0.5） |
| ECE-15 / Brier / NLL | 原始置信度（未做 temperature scaling）与 `hit25` 标签的校准误差 / 均方误差 / 对数损失 |
| logit 空间 | `score` 的 logit 变换 $\operatorname{logit}(p)=\ln\frac{p}{1-p}$；用于去掉 sigmoid 饱和导致的概率尺度压缩，与概率空间并列报告 |
| 类内 sd（概率 / logit） | 某一类样本集内部 `score`（或其 logit）的**总体标准差** $\sqrt{\frac{1}{n}\sum_i(x_i-\bar x)^2}$（÷ n，非 ÷ (n−1)）；hit 侧的“hit 概率 sd”即 hit 集在概率空间的类内 sd，“logit sd”即同一集合在 logit 空间的类内 sd；错误侧同法在 miss∪absent 上计算 |
| logit 差 | logit 空间下的“正确均值 − 错误均值”，与概率空间的 gap 互为对照 |
| 合并 sd / Cohen's d | $\sigma_{pooled}=\sqrt{(\sigma_{hit}^2+\sigma_{err}^2)/2}$；$d=\dfrac{\bar x_{hit}-\bar x_{err}}{\sigma_{pooled}}$（概率与 logit 两个空间各算一次），为尺度无关的差距读数 |

**跨权重比较的口径限制**：

- 正确/错误标签依赖各权重自己的 Top-1 框，正例集合规模与错误池构成随权重变化，差值中混入标签构成变化；
- gap、正确/错误均值、负查询均值都是绝对尺度读数：分数分布整体平移或收缩会机械改变它们，不能单独作为判据；
- FPR@TPR / Precision@R 依赖工作点的绝对阈值位置，跨权重比较必须说明阈值是各自标定还是固定；
- 校准指标不进判定门，只作为阈值可迁移性与尺度漂移的读数。

### 模型级

| 能力 | 必须报告的指标 | 说明 |
| :--- | :--- | :--- |
| 定位 | hit@IoU 0.25/0.50、mean best IoU | 验证 E3 的几何增益是否保留 |
| 候选排序 | AUROC、AUPRC、pairwise accuracy、TP/FP score gap | 判断区分度，不依赖单一阈值 |
| 高召回拒识 | FPR@TPR=90%/95%、precision@固定 recall | 对应“不能漏掉真目标”的导航约束 |
| 校准 | ECE、Brier、NLL | 决定阈值是否可迁移；不等同于排序能力 |
| 类别稳定性 | 六类分别统计，单列 potted plant | 防止总体均值掩盖少样本类退化 |

### S 证据级

必须分别统计 raw $c$、S-only、融合 $q$：

- TP/FP 的覆盖率与 covered 条件分布；
- AUROC/AUPRC 及其相对 raw $c$ 的增量；
- covered-high / covered-low / uncovered 三路的 precision、recall；
- 按场景、类别、目标距离分层；
- 首次候选、最终入记忆候选与最终被选导航候选三层统计。

### 导航级

SR、Oracle、SPL 之外必须保留 fp/fn/nv、首次错误锁定时间、真目标覆盖前错误停止数。只看 SR 无法区分“模型没看见”“看见但没检出”和“被近处 FP 抢先”。


## 5. 实验顺序与判定门

### 5.1 总体顺序

本轮固定采用以下顺序：

逐级解冻消融 → R1v1 / 历史 R2 结论冻结 → R3-G0/G1/G3/F1 对照完成 → 最终权重锁定为 VLVM V1 → 复用已确定的 `sigma_tar=0.60` → 直接优化 S-penalty。

核心原则：

1. **G0/G1 同起点、同轮次。** 两组都恢复 R1v1 E3 的模型、核心 optimizer 与 global step；G0 只延长 R1v1，G1 的唯一新增训练目标是 $\mathcal L_{gate}$。
2. **完成既定四个增量 epoch。** G0 E4/E5 已完成；当前 G1 保持原配置运行至 E5 结束，中间只保存 checkpoint 与 validation，不再提前中断或改参。
3. **G1 正式诊断固定为 E5。** G1 完成既定 E4/E5 训练后，统一使用最终 E5 checkpoint 的 `g1coree5` 与 `g1gatee5`；E4 只保留逐轮训练、validation 与 Gate 标定记录，不参与正式机制诊断或路线选择。
4. **续训 LR 不回升。** 核心参数保持 R1v1 E3 checkpoint 的末端 LR；G1 新增 Gate 使用独立参数组，在 E4 warmup、E5 cosine。
5. **G3 即 F0。** 从 R1v1 E3 冻结全部核心，只训练 `presence_gate.*`；G3 与 F0 不重复训练或续训。
6. **F1 与 G3/F0 严格配对。** 两者都从 R1v1 E3 新建 Gate optimizer，固定训练 3 epoch；F1 唯一变化是 5% target-absent hard-data-v2 `replace-source`，每轮总步数仍为 23,031。
7. **S-penalty 进入当前主线。** 最终 checkpoint 与 raw `sigma_tar=0.60` 均已锁定；不再训练模型或搜索 sigma，只优化 S 参数与仲裁方式。

### 5.2 全阶段固定控制变量

| 项目 | 固定设置 |
| :--- | :--- |
| 初始化权重 | G0/G1/G3/F1 历史实验均从现已命名为 `data/vlvm_v1/vlvm_v1.pth` 的 R1v1 E3 开始；G0/G1 恢复核心 optimizer/global step，G3/F1 以相同 seed 新建 Gate optimizer |
| 语义机制 | `enable_s_penalty=False` |
| S 冲突门 | 关闭 |
| 在线 2D 检测源 | 关闭 |
| 微调 gate | G0 关闭；G1 联合运行 `base + rank_cross + gate`，Gate 输入 detach；G3/F1 冻结核心，训练前向与反向只消费 `gate` 并更新 `presence_gate.*` |
| occ geometry | 开启，参数保持不变 |
| 语义地图 | 2.5D region map 保持不变 |
| memory-life | 保持不变 |
| 数据划分 | train/val 按 scene 隔离；导航评测场景不回灌训练 |
| 训练预算 | G0/G1 为 E4/E5；G3/F1 固定 3 epoch × 23,031 step，seed=0；第 1 epoch warmup，后两轮 cosine 至 `1e-6` |
| Gate-only optimizer | AdamW，Gate LR=`1e-4`，weight decay=`1e-4`，clip=`0.1`；只含 `presence_gate.*` |
| F1 数据 | hard-data-v2 仅保留 1,317 条 target-absent records（1,297 views），每轮抽取 1,152 views，占 5%，`replace-source`；source 顺序与 G3/F0 对齐 |
| 日志目录 | `ft_tsp3d/ft_log/s4_gate_only_2026-10-15/`；G3/F0、F1 的日志、manifest、Gate dump 与校准结果统一保存 |
| 冻结行为 | 冻结模块 `requires_grad=False` 且保持 `eval()` |
| 最终 no-S 导航锚点 | VLVM V1（R1v1 E3）：$\sigma_{tar}=0.60$ |

VLVM V1 的 `sigma_tar=0.60` 已由四场景 no-S grid 确定：平均 SR 36.61%、Oracle 54.55%、SPL 0.1607。该工作点直接作为 S-penalty 优化的固定 raw 门限，不再重复标定；Presence Gate 保持关闭，避免同时改变候选分布。

### 5.3 Phase A：冻结基线与评测口径

先冻结两个参照，不再修改：

| 编号 | 权重 | 用途 | S-penalty | Presence gate |
| :--- | :--- | :--- | :---: | :---: |
| B0 | scanrefer | 原始模型能力与分数标尺参照 | 关 | 关 |
| B1 | 当前 base | 已验证的局部点云适配参照 | 关 | 关 |

必须保存：

- E3 全量 per-record 结果；
- hit@0.25/0.50、mean best IoU；
- 正确候选、错误候选和负视图的原始 logit/score；
- 六类别独立统计；
- B1 在 $\sigma_{tar}=0.65$ 下的四场景导航结果；
- 在线 fp/fn/nv、首次错误锁定及真目标覆盖前错误停止数。

判定目的：B0 定义“未微调起点”，B1 定义“定位提升但区分度仍不足”的现有上限。后续档位必须同时与二者比较。

### 5.4 Phase B：逐级解冻消融

这一阶段只改变可训练层，不加入 Presence gate、pairwise loss 或新的 hard-negative pool，以单独确定“解冻到哪里”。

| 档位 | 可训练模块（代码参数前缀） | 冻结模块 | 与当前 base 的关系 |
| :--- | :--- | :--- | :--- |
| U0 | 最终候选置信度头：`head.cls_conv` | `head.bbox_conv`、其余 head、`text_projector`、全部 `vision_backbone`、`text_encoder`（RoBERTa） | 最小适配档；实际仅训练 `cls_conv` 的 weight/bias |
| U1 | U0 + `text_projector`；TGP：`head.keep_trans`、`head.keep_conv`；CBA：`head.com_trans`、`head.com_cls`、共享的 `head.pos_embed`；最终融合：`head.fuse`；多尺度重建/FPN：`head.up_block_*`、`head.lateral_block_*`、`head.out_block_*` | `head.bbox_conv`、全部 `vision_backbone`、`text_encoder` | 适配完整语义匹配路径；不是只解冻单个“最后一级跨模态”模块 |
| U2 | U1 + `vision_backbone.layer4` | `head.bbox_conv`、backbone stem/`layer1`/`layer2`/`layer3`、`text_encoder` | 在 U1 上增加最高层 3D 语义适配 |
| U3 | U2 + `vision_backbone.layer3`（即同时训练 `layer3`/`layer4`） | `head.bbox_conv`、backbone stem/`layer1`/`layer2`、`text_encoder` | gate 关闭时，与当前 base 的可训练范围只差 `head.bbox_conv`：base 训练它，U3 冻结它 |

代码中的 `head.pruning` 是无可训练参数的硬裁剪算子，因此不会出现在解冻白名单或参数审计中。这里的“最终跨模态”若单指靠近输出的一层，主要是 `head.com_trans`，其后还有紧邻 `cls_conv`/`bbox_conv` 的 `head.fuse`；U1 为保持语义路径完整，实际同时解冻上述 TGP、CBA、最终融合和 FPN 模块。

训练约束：

- 四档均从 scanrefer 初始化，不能从当前 base 继续冻结；
- 使用完全相同的现有 base loss、数据与训练预算；
- bbox 回归路径的训练状态在四档间保持一致；若 U0/U1 冻结 bbox，则 U2/U3 也冻结，避免把框回归变化混入解冻层数结论；
- 各层使用分层 LR，但同一模块在各档的 LR 必须相同；
- 训练过程同时记录 train/val，禁止用导航结果选择 epoch。

E3 晋级条件：

1. hit@0.25、hit@0.50 和 mean best IoU 不发生系统性退化；
2. AUPRC/AUROC、pairwise accuracy 至少有一项稳定提高；
3. FPR@TPR=90%/95% 下降；
4. 提升不能只来自所有分数整体下移；
5. potted plant 及其他少样本类不得再次出现明显单类崩塌。

输出：只保留 Pareto 前两档进入 Phase C；其余档位停止，不跑导航。

### 5.5 Phase C：R1v1 与 R2 候选区分度微调

#### R1v1 损失定义

一个 view 内包含多个 query record。每个 query 分别前向，只使用其在线交付 Top-1。达到 IoU 门的正 query Top-1，与同 view 其余错误 query 中分数最高的 Top-1 配对：

$$
\mathcal L_{rank}=\operatorname{softplus}
\left(m-z_{pos}+z_{hard\_neg}\right).
$$

错误池包含 target-absent query 的 Top-1，也包含 Top-1 定位失败的正 query。当前固定 `margin=0.2`、正项 `IoU≥0.25`、`rank_weight=1.0`。

| 控制项 | R1v1 设置 |
| :--- | :--- |
| 初始化 | `tsp3d_scanrefer.pth` |
| 解冻边界 | U1 |
| 损失 | base + R1v1 `rank_cross` |
| epoch / seed | 3 / 0 |
| 每轮步数 | 23,031 |
| G1 / G2 / G3 LR | `1e-4` / `5e-5` / `1e-5` |
| 优化器 | AdamW，β=`0.9/0.999`，ε=`1e-8` |
| weight decay / clip | `1e-4` / `0.1` |
| 调度 | E1 线性 warmup，随后 cosine 至 `1e-6` |
| 数据 | `data/ft_datasets/local_view`，train/val 按 scene 隔离 |

R1v1 E3 保持为当前已验证的 no-S 历史锚点。其 checkpoint、逐 epoch loss、E3 全量结果和导航结果见 `VLVM-S4-Exp.md`。

#### R2 hard-negative replay

R2 只验证 replay 数据供给，损失仍与 R1v1 完全相同。每轮总步数固定为 23,031，通过 source 配对替换将 10% 原始 view 换成 replay view，避免把额外训练量混入 hard data 的贡献。

| 控制项 | R1v1 | R2 |
| :--- | :--- | :--- |
| 初始化 | ScanRefer | ScanRefer |
| 解冻边界 | U1 | U1 |
| 损失 | base + `rank_cross` | 相同 |
| 每轮总步数 | 23,031 | 23,031 |
| 原始 / replay 步数 | 23,031 / 0 | 20,728 / 2,303 |
| Replay 占比 | 0% | 10% |
| 替换方式 | — | replay view 替换对应 `source_sample_id` 原始 view |
| 其他训练设置 | R1v1 | 与 R1v1 完全相同 |

R2 整体未超过 R1v1，仅在个别 epoch 的局部指标上改善，因此不替换 R1v1 E3。R1v1/R2 的实验记录保留在 S4，不迁移到 `VLVM-S5.md`。

R1v2 及后续 Rank/Gate 重构不再属于当前 S4 微调主线，统一见 [VLVM-S5.md](VLVM-S5.md)。

### 5.6 Phase D：R1v1 E3 / ScanRefer 无 S 导航对照

目的：确认 R1v1 E3 的离线提升能否转化为导航收益，并用 ScanRefer 在完全相同的 3D-native 框架下建立原始权重参照。

| 组别 | 权重 | 场景 | $\sigma_{tar}$ | S-penalty | GD | 其他框架设置 |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| N1 | `data/vlvm_v1/vlvm_v1.pth` | 固定四场景 | checkpoint 独立 grid search | 关 | 关 | VLVM V1 导航基线 |
| N0 | `data/tsp3d_models/tsp3d_scanrefer.pth` | 与 N1 相同 | 固定 `sigma_tar = 0.70`（已有标定结果，不重复 grid） | 关 | 关 | 除权重和对应 $\sigma_{tar}$ 外与 N1 完全一致 |

| 固定控制项 | 配置 |
| :--- | :--- |
| TSP3D 主模型 | 开启；不使用 2D 检测器替代 |
| Presence gate | 关闭 |
| S 冲突门 | 关闭 |
| occ gate | 开启，参数不变 |
| 2.5D region map | 不变 |
| memory-life / 探索 / 导航 | 不变 |
| episode 与 seed | 两组完全一致 |
| 候选阈值协议 | N1 使用 grid 生成规则并独立择点；N0 固定 `sigma_tar = 0.70` |

执行顺序：历史 N1（R1v1 E3）与 N0（ScanRefer）已全部完成；R3-G0/G1 同轮次实验完成后，从 R1v1 E3 串行运行 G3/F0 与 F1 Gate-only 配对训练。

#### 四场景评测指标

两组统一统计：

| 层级 | 指标 |
| :--- | :--- |
| 主指标 | 四场景平均 SR |
| 路径效率 | SPL |
| 覆盖诊断 | Oracle |
| 失败构成 | fp / fn / nv |
| 检测转换 | det precision、入记忆 TP 率、覆盖前错误停止 |
| 稳定性 | 四场景单独结果与六类别结果 |

两组判定规则：

1. 先比较平均 SR；
2. SR 接近时以 SPL 和 fp 数决定；
3. Oracle 只用于解释覆盖，不以 Oracle 单独选权重；
4. 不接受只靠一个场景大幅提升、其余场景普遍退化的档位；
5. R1v1 E3 已证明跨 query Top-1 ranking 的离线提升能够转化为导航收益；Phase E 先以 G0/G1 同轮次结果和 G3 独立结果锁定 Gate 路线，最终只对入选权重进行 no-S 导航复核。

### 5.7 Phase E：R3 收敛、联合 Gate 与独立 Gate

#### E1：三条分支与执行顺序

| 分支 | 起点 | 训练目标 | 核心参数 | 数据 | 回答的问题 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| G0 | R1v1 E3 | base + `rank_cross` | U1 继续训练 | local_view | R1v1 延长 epoch 的收敛边界 |
| G1 | R1v1 E3 | base + `rank_cross` + `gate` | U1 继续训练；Gate 输入 detach | local_view | $L_{gate}$ 与原目标同期训练是否有效，以及联合权重上的 Gate veto 是否有效 |
| G3 | R1v1 E3 | 优化目标只有 `gate` | 全部冻结，只更新 `presence_gate.*` | local_view | 独立 Gate 训练是否有效，以及是否优于 G1 联合路线 |

G0 与 G1 是同起点、同轮次配对对照，G1 不继承 G0 E4/E5。G3 从同一个 R1v1 E3 初始化，但不恢复或更新核心 optimizer；只为 `presence_gate.*` 新建 AdamW。G3 同时作为 F0，F1 与其共享 Gate 初始化 seed、优化器、学习率、三轮预算、source 顺序、损失参数和 validation/阈值标定规则。

历史执行固定为：保留已经完成的 G0/G1 E4/E5 及 E3 对照；当时由旧版 `scripts/ft_gate.sh` 串行运行 G3/F0、F1。当前脚本已收缩为原始数据独立 Gate-only，不再提供这些历史档位。G2 编号本阶段未启用。

#### E2：续训学习率与梯度边界

- 核心参数保持 R1v1 E3 checkpoint 的末端学习率，禁止因把总 epoch 改为 5 而重新抬升 cosine 曲线。
- G1 的 `presence_gate.*` 使用独立 optimizer 参数组：E4 从 `1e-6` warmup 到 G1 LR，E5 cosine 回到 `1e-6`。
- Gate 输入的视觉与文本特征 detach；$\mathcal L_{gate}$ 不回传到 U1 核心。
- 核心与 Gate 分别执行 `clip=0.1`，Gate 梯度范数不改变核心梯度缩放。
- Gate 初始化与前向 Dropout RNG 独立，G0/G1 核心随机轨迹保持可配对。
- G3/F1 冻结 R1v1 E3 的全部核心参数并保持冻结模块 `eval()`，optimizer 中只允许出现 `presence_gate.*`；训练前向不构建 core head 输出，反向总损失只有 `gate_main + gate_recall`。
- G3/F1 统一训练 3 epoch：第 1 epoch 从 `1e-6` warmup 至 `1e-4`，后两轮 cosine 回到 `1e-6`；逐轮保存 validation Gate dump 并按 `recall_open≥0.99` 标定阈值。

#### E3：Gate 损失

Presence gate 输出 query 在当前 view 中存在目标的概率：

$$
P_i=\sigma(z_i^{gate}),\qquad y_i\in\{0,1\}.
$$

损失使用正样本 BCE、负样本 focal 和正样本召回下界：

$$
\mathcal L_{gate}
=-\alpha y_i\log(P_i+\epsilon)
-(1-\alpha)(1-y_i)P_i^{\gamma_-}\log(1-P_i+\epsilon)
+\lambda_{rec}\,y_i[\max(0,m_+-P_i)]^2.
$$

| 参数 | 固定值 |
| :--- | :--- |
| $$\alpha$$ | 根据 primary train split 正负数量计算 |
| $$\gamma_{neg}(\gamma_-)$$ | `2.0` |
| `positive_margin` $$m_+$$ | `0.70` |
| `recall_weight` $$\lambda_{rec}$$ | `1.0` |
| Gate 初始化 seed | `0`，独立 RNG |
| Gate 阈值标定目标 | validation `recall_open ≥ 0.99` |

#### E4：Gate 阈值与 E3 对照矩阵

Gate 只做高召回 veto，不参与 bbox 排序，也不改变通过视图的核心 raw score：

$$
P_{exist}<\tau_{gate}\Rightarrow\text{no detection},
\qquad
P_{exist}\ge\tau_{gate}\Rightarrow\text{保留核心 bbox 与 score}.
$$

Gate 阈值统一由 `ft_tsp3d/eval_tools/calibrate_gate.py` 消费对应 validation Gate dump 标定，固定约束为 `recall_open≥0.99`。E3 不再临时搜索阈值，只消费已经固定的 $\tau_{gate}$。

G0 E5 与 R1v1 E3 已有 E3 数据全部复用，不重复前向。G1 的正式机制诊断和路线判断统一使用最终 E5 checkpoint；E4 只保留逐轮记录：

| E3 档位 | 权重 | Gate 推理 | 直接对照 | 唯一解释 |
| :--- | :--- | :--- | :--- | :--- |
| `g1coree5` | G1 E5 | `threshold=0`，强制全开 | G0 E5 | $L_{gate}$ 与 `L_base + L_rank` 同期训练这一整套最终训练策略的贡献；同时检查 Gate 分支是否意外扰动核心轨迹 |
| `g1gatee5` | 与 `g1coree5` 完全相同 | 使用 E5 标定后的 $\tau_{gate}$ | `g1coree5` | 固定最终联合权重后 Gate veto 的净贡献 |

两项 E5 对照相互正交但都是条件效应：`g1coree5 − G0 E5` 不能证明 Gate veto 有效，`g1gatee5 − g1coree5` 也不能证明冻结核心后单独训练 $L_{gate}$ 有效。

G3/F0 三轮训练完成后按下列口径判断：

1. G3 `threshold=0` 理论上必须复现历史 R1v1 E3；即比较 `g3coree3` 与 R1v1 E3。先用核心参数 SHA256、配置审计和少量逐记录推理验证，一致后不重复完整 E3。
2. G3 E3 启用标定阈值后跑一次完整 E3；`g3gatee3 − R1v1 E3` 是独立 Gate 最终路线相对原始锚点的净变化。
3. 比较 `g1gatee5 − g1coree5` 与 `g3gatee3 − g3coree3`，判断固定各自训练后权重时 Gate veto 机制的作用是否改变；两组都只在同一 checkpoint 内切换 Gate 阈值，不混入核心差异。
4. 比较 `g1gatee5 − G0 E5` 与 `g3gatee3 − R1v1 E3`，确定 S4 当前预设方案下采用联合训练还是独立训练。该比较衡量两条完整训练路线，包含核心是否续训及各自既定训练预算，不解释为等步数的训练方式单变量效应。
5. Gate 必须在 `recall_open≥0.99` 下改善 target-absent intercept/FPR，同时定位、AUROC/AUPRC、pair accuracy 和六类别稳定性不得系统退化。

#### E5：产物边界与 Gate 路线选择

历史 G0/G1/G3/F1 启动档位已随 S4 收敛删除。当前 `scripts/ft_gate.sh` 只允许 VLVM V1 上的原始数据独立 Gate-only，参数审计必须证明只有 `presence_gate.*` 可训练。

Phase E 最终必须分别给出三个结论：

1. G1 同期训练策略相对 G0 是否有效；
2. G1 联合权重上的 Gate veto 是否有效；
3. G3 独立 Gate 是否有效，以及各自计划最终 checkpoint 下 G1 E5 与 G3 E3 哪条完整路线更好。

G3/F0 与 F1 完成后，联合 G1 结果锁定 Gate 是否启用，并独立判断 hard-data-v2 是否保留。

### 5.8 Phase F：hard-negative replay

#### F1：取消纯 R1v1 + hard-v2 续训

直接确定不再执行以下实验：

> **R1v1 E3 → R1v1 同设置 + hard data v2 续训一轮。**

现有两组证据均为负面：

| 证据 | 对照 | 关键结果 | 含义 |
| :--- | :--- | :--- | :--- |
| 历史 R2 | R2 E3 vs R1v1 E3；同为 ScanRefer 起点、U1、base + `rank_cross`、三轮，R2 只增加 10% hard-data-v2 source replacement | hit@0.25 `32.39% vs 33.63%`；mean IoU `0.1850 vs 0.1957`；AUROC `0.8040 vs 0.8363`；AUPRC `0.3575 vs 0.4511` | hard-v2 进入 base/rank replay 没有带来整体收益，区分度尤其退化 |
| R3-G0 | G0 E4/E5 vs R1v1 E3；不加 hard data，仅延长原训练 | E4 hit@0.25 `−3.82 pp`、AUPRC `−0.0419`；E5 虽部分恢复，定位、AUPRC 与 FPR@TPR95 仍未回到 E3 | 从 R1v1 E3 继续更新 U1 核心本身已经存在过度微调风险 |

R2 与 G0 不能在严格因果意义上替代“R1v1 E3 tail + hard-v2”的同起点实验，因为起点和训练阶段存在交互；但二者已经给出足够强的工程负先验：继续更新核心再叠加曾经无整体收益的 replay，不值得消耗一轮完整训练预算。因此该分支从计划中删除，不再以补实验为由恢复。

#### F2：G3/F0 与 F1 Gate-only 配对

F0 不再是从已入选 Gate checkpoint 继续训练的额外控制臂；G3 本身就是 F0。F1 与 G3/F0 同时从 R1v1 E3 出发，因此不会把 hard-data 贡献与额外 Gate 训练步数混合。

| 分支 | 起点 | 数据 | 每轮步数 | 反向目标 | 用途 |
| :--- | :--- | :--- | ---: | :--- | :--- |
| G3 / F0 | R1v1 E3 + Gate seed 0 | 100% 原始 local_view | 23,031 | `gate_main + gate_recall` | Gate-only 基线 |
| F1 | 与 G3/F0 完全相同 | 95% 原始 + 5% target-absent hard-data-v2，按 `source_sample_id` 配对替换 | 23,031 | `gate_main + gate_recall` | Gate 专用 hard-data-v2 净贡献 |

两臂固定 3 epoch、AdamW、Gate LR=`1e-4`、weight decay=`1e-4`、clip=`0.1`、seed=0；source shuffle 在替换选择之前生成，使 F1 与 G3/F0 的 source 顺序逐位置对齐。Gate loss 的 `gamma_neg=2.0`、`positive_margin=0.70`、`recall_weight=1.0` 与 primary split 计算的 $\alpha$ 完全相同。两臂逐 epoch 使用 `ft_tsp3d/eval_tools/calibrate_gate.py` 在 `recall_open≥0.99` 下独立固定 $\tau_{gate}$。只有 F1 相对 G3/F0 改善 target-absent intercept/FPR，且定位、排序与六类别稳定性不退化，hard-data-v2 才进入最终模型。

<!-- **历史探索性 Gate E1**

该结果来自已中止的 `双分支 rank + Gate + hard-data-v2` 探索分支，只证明 Gate 方案具备可行性，不作为当前 G0/G1 的结果。当前主线已改为继承 R1v1 并重新从 ScanRefer 独立训练。

| 指标 | R1v1 E1 | 旧排序分支 E1 | 旧 Gate+hard E1 | Gate − 旧排序 | Gate − R1v1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| hit@0.25 | 31.74% | 20.69% | 25.29% | +4.60pp | -6.45pp |
| hit@0.50 | 12.90% | 10.50% | 11.16% | +0.66pp | -1.74pp |
| mean IoU | 0.1785 | 0.1253 | 0.1477 | +0.0224 | -0.0308 |
| AUROC | 0.7881 | 0.8160 | 0.8466 | +0.0306 | +0.0585 |
| AUPRC | 0.3223 | 0.3655 | 0.4461 | +0.0806 | +0.1238 |
| pair acc | 0.8759 | 0.8955 | 0.9374 | +0.0419 | +0.0615 |
| FPR@TPR90 | 0.4260 | 0.5812 | 0.4395 | -0.1417 | +0.0135 |
| FPR@TPR95 | 0.5978 | 0.7694 | 0.6090 | -0.1604 | +0.0112 |
| Precision@R90 | 0.2828 | 0.1505 | 0.2269 | +0.0764 | -0.0559 |
| Precision@R95 | 0.2287 | 0.1238 | 0.1828 | +0.0590 | -0.0459 |
| 正确均值 / 错误均值 / gap | 0.5284 / 0.3722 / 0.1563 | 0.6084 / 0.3670 / 0.2414 | 0.6114 / 0.3455 / 0.2659 | +0.0030 / -0.0215 / +0.0245 | +0.0830 / -0.0267 / +0.1096 |
| ECE / Brier / NLL | 0.2395 / 0.1736 / 0.5285 | 0.2892 / 0.1665 / 0.5135 | 0.2535 / 0.1520 / 0.4812 | -0.0357 / -0.0145 / -0.0323 | +0.0140 / -0.0216 / -0.0473 |
| 负查询无框 | 0 | 0 | 43（1.63%） | +43 | +43 |

| 旧 Gate E1 类别 hit@0.25 | bed | chair | couch | potted plant | toilet | tv |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1v1 E1 | 42.08% | 27.78% | 47.24% | 8.93% | 26.04% | 13.27% |
| 旧 Gate+hard E1 | 32.60% | 25.56% | 35.62% | 0.00% | 15.62% | 10.68% |
| Gate − R1v1 | -9.48pp | -2.22pp | -11.62pp | -8.93pp | -10.42pp | -2.59pp | -->

旧 `R1v2 + Gate + hard E1` 仅作为可行性旁证：它相对旧 R1v2 E1 的 hit@0.25 `+4.60 pp`、AUROC `+0.0306`、AUPRC `+0.0806`、pair accuracy `+0.0419`，但当时同时加入 $L_{gate}$、Gate veto 与 Gate hard data，不能把收益单独归因于 hard data。Phase F 的 F0/F1 配对用于补齐这一缺失对照。

### 5.10 执行清单与实验预算

| 顺序 | 实验组 | 离线评测 | 四场景导航 | 输出 |
| ---: | :--- | :---: | :---: | :--- |
| 0 | B0/B1、U0–U3、R1v1、历史 R2 | 已完成 | 复用已有结果 | 固定历史参照 |
| 1 | R3-G0：R1v1 E3 → E4/E5 | 每轮保存；两轮完成后判断 | 不跑 | 收敛边界与 G0 E4/E5 完整产物 |
| 2 | R3-G1：同一 R1v1 E3 → E4/E5 + Gate | 每轮保存 | 不跑 | Gate E4/E5、Gate dump 与校准表 |
| 3 | G1 E5 两项正式 E3：`g1coree5`、`g1gatee5` | 必须；复用已有 G0 E5/R1v1 E3 | 不跑 | 最终联合训练路线贡献与 Gate veto 净贡献 |
| 4 | R3-G3/F0：R1v1 E3 → 冻结核心、Gate-only 3 epoch | 最终 Gate 档一次完整 E3；强制全开只做一致性检查 | 不跑 | 独立 Gate 权重、Gate dump 与校准表 |
| 5 | F1：与 G3/F0 配对的 5% target-absent hard-v2 `replace-source` | 必须 | 不跑 | hard data v2 的单变量结论 |
| 6 | G1、G3/F0、F1 训练方式与门控增益比较 | 必须 | 不跑 | 锁定 Gate 路线与 hard data 是否启用 |
| 7 | 锁定 VLVM V1 / Gate 关闭 / `sigma_tar=0.60` | 已完成 | 复用四场景 no-S grid | S-penalty 固定基线 |
| 8 | 最终 no-S 导航 | 已完成 | 复用固定四场景 | SR、Oracle、SPL、fp/fn/nv 与错误停靠 |
| 9 | P0/P1/P2/P3 | 下一步 | 仅 Pareto 档 | 最终 S 机制 |

G0/G1、G3/F0 与 F1 均已完成并保留历史结果。旧版脚本当时串行运行 `g3 f1`，两臂日志统一保存在 `ft_tsp3d/ft_log/s4_gate_only_2026-10-15/`；当前现役脚本不再包含这两个档位。
