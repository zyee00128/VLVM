# TSP3D 微调数据处理总览

| 项目 | 内容 |
| :--- | :--- |
| 覆盖阶段 | Base → Unfreeze U1 → Rank R1 → Rank R2 / R3 |
| 主模型 | TSP3D / BeaUTyDETR |
| 原始数据根目录 | `data/ft_datasets/local_view` |
| 在线硬负样本 capture | `data/ft_datasets/r2_capture_v2` |
| 在线硬负样本 replay | `data/ft_datasets/hard_negative_replay_v2` |
| 数据粒度 | 一个优化步对应一个 view；同 view 内所有正、负 query records 共用一份点云 |
| 当前有效版本 | replay v2；旧版 `hard_negative_replay` 不再用于训练 |
| 数据验证状态 | replay v2 全量验证通过；capture/replay 映射错误 0，rejected 0，训练/验证泄漏 0 |

## 1. 微调阶段与数据关系

| 阶段 | 实验含义 | 初始化权重 | 解冻档位 | 训练数据 | 数据处理变化 | 损失 | Epoch | 当前状态 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | ---: | :--- |
| Base | 原始微调基线 | `data/tsp3d_models/tsp3d_scanrefer.pth` | `legacy` | 原始 local-view train | 无新增数据；直接按 view 读取 | base：candidate classification、background、L1、GIoU | 3 | 已完成 |
| Unfreeze U1 | 锁定后续主线解冻边界 | ScanRefer；与 Base 独立初始化 | `u1` | 与 Base 完全相同 | 无新增数据；只改变可训练参数边界 | 与 Base 相同 | 3 | 已完成 |
| Rank R1 | 增加候选级区分度学习 | ScanRefer；与 U1 独立初始化 | `u1` | 与 Base/U1 完全相同 | 不生成新数据文件；训练时在同 view 内在线构造 rank pair | base + candidate pairwise ranking | 3 | 已完成 |
| Rank R2 | hard-data 干净消融 | ScanRefer；独立初始化 | `u1` | 原始 train + replay v2 | 每轮按优化步 90:10 混合 | base + candidate pairwise ranking | 3 | 待训练 |
| Rank R3 | 性能导向续训 | `data/vlvm_v1/vlvm_v1.pth` | `u1` | 原始 train + replay v2 | 每轮按优化步 90:10 混合；降低学习率 | base + candidate pairwise ranking | 2 | 待训练 |

| 关系 | 固定口径 |
| :--- | :--- |
| 实验顺序 | Base → U1 → R1 → R2/R3 表示实验演进顺序，不表示所有阶段都逐级继承权重 |
| Base、U1、R1、R2 | 均从同一个 ScanRefer checkpoint 独立初始化，用于控制变量比较 |
| R3 | 从 R1 E3 继续训练，用于验证“已有 rank 能力 + hard replay”的最终性能上限 |
| R2 与 R3 | 脚本中串行执行，但 R3 不继承 R2 权重；R2 完成后才启动 R3 |
| 验证集 | 所有阶段只使用原始 local-view val；hard replay 不进入验证集 |

## 2. 原始 local-view 数据生成

### 2.1 场景划分

| 项目 | 数值 / 规则 |
| :--- | :--- |
| HM3D 场景池 | 137 个可用 annotated train 场景 |
| Split seed | 0 |
| Val 比例 | 0.1，按场景划分 |
| Train | 123 个场景 |
| Val | 14 个场景 |
| 导航评测场景 | 不参与 local-view 数据生成 |
| Split 文件 | `data/ft_datasets/local_view/splits.json` |
| 泄漏约束 | train、val、导航评测场景互斥；hard replay 只能来自原始 train |

### 2.2 传感器与局部视图

| 项目 | 配置 |
| :--- | :--- |
| 输入模态 | RGB + Depth + Semantic |
| 图像分辨率 | 640 × 480 |
| 水平视场角 | 79° |
| 相机高度 | 0.88 m |
| 有效深度 | 0.5–5.0 m |
| 单 view 帧数 | `K ∈ [1, 8]`，实际均值 4.812 |
| 短轨迹跨度 | 连续采样点需通过 0.15 m 位移或 15° yaw 去重门 |
| 查询位姿 | K 帧中的最后一帧 |
| 融合坐标系 | episodic canonical，`p_e=(x,-z,y)` |
| 世界融合半径 | 6.0 m |
| 世界融合体素 | 0.02 m |
| 最大融合帧数 | 8 |
| 近邻刷新半径 | 3.0 m |
| 单 view 最大点数 | 200,000 |

### 2.3 点云发送侧处理

| 距离区间 | 采样体素 |
| :--- | ---: |
| 距离 ≤ 1.5 m | 0.01 m |
| 1.5 m < 距离 ≤ 3.0 m | 0.02 m |
| 距离 > 3.0 m | 0.05 m |

| 项目 | 规则 |
| :--- | :--- |
| 点云通道 | `(x, y, z, r, g, b)`，shape 为 `(N, 6)` |
| 磁盘 dtype | `float16`，与在线 wire dtype 一致 |
| 训练 dtype | loader 读取后转换为连续 `float32` Tensor |
| 点云复用 | 同一 view 的全部 records 共用 `points[offsets[v]:offsets[v+1]]` |
| 场景缓存 | 场景级 LRU，默认缓存 3 个场景 |

### 2.4 类别、正负样本与 GT

| 项目 | 规则 |
| :--- | :--- |
| 类别集合 | `chair`、`bed`、`potted plant`、`toilet`、`tv`、`couch` |
| 类名处理 | alias、大小写和单复数归一到六个规范类名 |
| 正样本可见门槛 | 目标实例可见像素 ≥ 40 |
| 多实例选择 | 选择当前 view 内可见像素最多的同类实例 |
| GT | 完整实例 AABB，转换到 canonical 系后轴对齐重包裹，格式 `(cx,cy,cz,l,w,h)` |
| 目标点门槛 | GT 内目标点数 ≥ 40；不足时丢弃正 record |
| 负样本 | 当前 view 中该类所有实例可见像素均 < 40；GT 为 `None` |
| 正负生成比例 | 1:3 |
| Record 权重 | 正常正样本 1.0；中度截断正样本 0.5；负样本 1.0 |

### 2.5 截断与丢弃规则

| 条件 | 处理 | 实际数量 |
| :--- | :--- | ---: |
| 截断率 ≤ 0.70 | 保留，weight=1.0 | 22,545 |
| 0.70 < 截断率 ≤ 0.85 | 保留，weight=0.5 | 2,450 |
| 截断率 > 0.85 | 丢弃 | 1,920 |
| GT 内目标点不足 40 | 丢弃 | 19,839 |
| 总丢弃记录 | 写入 `dropped.jsonl` | 21,759 |

### 2.6 原始数据规模

| Split | 场景 | Views | Records | 正 records | 负 records | Points |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 123 | 23,031 | 89,620 | 22,405 | 67,215 | 905,046,037 |
| Val | 14 | 2,638 | 10,360 | 2,590 | 7,770 | 100,544,366 |
| 合计 | 137 | 25,669 | 99,980 | 24,995 | 74,985 | 1,005,590,403 |

| 点云统计 | 数值 |
| :--- | ---: |
| 平均点数 / view | 39,175.29 |
| 中位点数 / view | 37,157 |
| P10 | 15,582 |
| P90 | 65,443.6 |
| 原始数据目录总大小 | 6.4 GB |
| Train NPZ | 5.7 GB |
| Val NPZ | 644 MB |
| GD sidecar | 24 MB |
| Instance tables | 11 MB |

### 2.7 原始训练集类别分布

| 类别 | 正 records | 负 records |
| :--- | ---: | ---: |
| bed | 5,617 | 10,659 |
| chair | 7,807 | 6,366 |
| couch | 4,726 | 9,826 |
| potted plant | 1,078 | 13,870 |
| toilet | 941 | 15,012 |
| tv | 2,236 | 11,482 |
| 合计 | 22,405 | 67,215 |

### 2.8 原始验证集类别分布

| 类别 | 正 records | 负 records |
| :--- | ---: | ---: |
| bed | 549 | 1,322 |
| chair | 943 | 723 |
| couch | 525 | 1,150 |
| potted plant | 168 | 1,515 |
| toilet | 96 | 1,697 |
| tv | 309 | 1,363 |
| 合计 | 2,590 | 7,770 |

### 2.9 Local-view NPZ schema

| 字段 | 类型 | 含义 |
| :--- | :--- | :--- |
| `points` | `float16 (M,6)` | 场景内全部 view 点云串接结果 |
| `offsets` | `int64 (V+1,)` | 第 v 个连续 view 的点云范围 `[offsets[v], offsets[v+1])` |
| `records_json` | JSON scalar | 所有训练 records，按原始 view ID 连续分组 |
| `record.view` | int | 原始 view ID；可能稀疏，不等于 offsets 下标 |
| `record.k` | int | 当前 view 融合帧数 |
| `record.poses` | list | K 帧位姿，含 x、y、floor、yaw_deg |
| `record.query` | str | 六类规范查询词 |
| `record.label` | 0 / 1 | target absent / target present |
| `record.weight` | float | 样本聚合权重 |
| `record.target` | object / null | 正样本 canonical AABB；负样本为空 |

## 3. 各阶段训练期数据处理

### 3.1 Base

| 环节 | 数据处理 |
| :--- | :--- |
| 数据输入 | `local_view/train` 全量 23,031 views |
| 优化步 | 每轮 23,031 步；每步一个 view |
| View 内处理 | 逐 record 前向；按 record.weight 对相同损失分项归一聚合 |
| 正 record | 匈牙利匹配 IoU 最高候选；匹配项参与 classification、L1、GIoU |
| 负 record | 所有候选按负目标计算 classification，并增加 background 二次压分 |
| GD sidecar | 文件存在，但 Base 未启用 gate/distill，不作为监督项 |
| Instance table | 文件存在，但归档后的 S1/S2 不属于本链路，不消费 entities |
| 新增持久化数据 | 无 |

### 3.2 Unfreeze U1

| 环节 | 数据处理 |
| :--- | :--- |
| 数据输入 | 与 Base 完全相同的 train/val split、views、records 和采样顺序规则 |
| 优化步 | 每轮 23,031 步 |
| 数据新增 | 无 |
| 标签变化 | 无 |
| 采样变化 | 无 |
| 唯一实验变量 | 可训练参数白名单切换为 U1；bbox 与 RoBERTa 仍冻结 |

### 3.3 Rank R1

| 环节 | 数据处理 |
| :--- | :--- |
| 持久化数据 | 仍只使用原始 local-view；不生成 rank 专用 NPZ |
| Pair 构造时机 | 每个 view 训练前向期间在线构造 |
| 正候选 | 对每条正 query 取 IoU 最大候选，且最大 IoU 必须 ≥ 0.25 |
| 错误池 | 所有未被选为合格正候选的候选；包含正 query 内误框和 target-absent query 候选 |
| Hardest negative | 当前 view 错误池中 raw logit 最高的候选 |
| Pair 数量 | 每个合格正候选与同一个 view hardest negative 配成一对 |
| Rank margin | 0.2 |
| Rank loss weight | 1.0 |
| 无有效 pair | 当前 view 跳过 rank 分项，Base 损失仍正常计算 |
| 优化步 | 每轮仍为 23,031 步；rank 不改变数据规模 |

## 4. 在线硬负样本

### 4.1 版本与来源

| 项目 | replay v2 固定值 |
| :--- | :--- |
| 挖掘模型 | R1 E3 |
| 挖掘 checkpoint | `data/vlvm_v1/vlvm_v1.pth` |
| 挖掘输入 | `data/ft_datasets/local_view/train` |
| 挖掘场景 | 全部 123 个 train 场景 |
| 禁止来源 | 原始 val、导航评测场景、评测日志 |
| Capture 目录 | `data/ft_datasets/r2_capture_v2` |
| Replay 目录 | `data/ft_datasets/hard_negative_replay_v2` |
| 数据处理完成标记 | `ft_tsp3d/ft_log/r2_data_v2/r2_data.done` |
| 处理代码 | `ft_tsp3d/data_processing/hard_negative_replay.py` |
| 自动处理脚本 | `scripts/r2_after_r1.sh` |

### 4.2 View 映射修复

| 字段 | 含义 | 使用位置 |
| :--- | :--- | :--- |
| `view` | 原始 record view ID；可稀疏，例如 10、42 | 文件名、审计、sample ID、source step |
| `view_index` | 场景内连续序号 `0..V-1` | `offsets[view_index:view_index+1]` 点云切片 |
| `source_view_index` | capture 中持久化的连续序号 | 二次加载和完整性校验 |
| `points_source.view` | 指向原始 NPZ offsets 的连续序号 | replay 构建时读取源点云 |
| 交叉校验 | `sample(scene,view_index)` 必须等于 `dataset.index[index]` | 挖掘阶段立即失败保护 |
| v1 问题 | 将稀疏原始 `view` 错当 offsets 下标，造成越界或引用错误 view | v1 数据停用 |
| v2 结果 | 原始 ID 与连续索引彻底分离 | 9,377 个 capture 映射错误为 0 |

### 4.3 R1 E3 全量挖掘

| 统计项 | 数值 |
| :--- | ---: |
| 扫描 views | 23,031 |
| 扫描 records | 89,620 |
| 低于 score 门槛的候选 | 593,447 |
| 正 query 中 IoU 不属于误检的候选 | 53,849 |
| 原始 hard candidates | 69,664 |
| 含 hard candidate 的 views / capture | 9,377 |
| Hard candidate 平均分数 | 0.606131 |
| 挖掘耗时 | 4,605.14 秒 |
| 模型缺失权重 | 0 |
| 模型意外权重 | 0 |
| Capture 目录大小 | 38 MB |

### 4.4 Hard candidate 入选规则

| 顺序 | 条件 | 处理 |
| ---: | :--- | :--- |
| 1 | Capture schema version 必须为 1 | 不匹配则拒绝 |
| 2 | `split=train` 且 scene 在冻结的原始 train split 中 | 非 train 或未知场景拒绝 |
| 3 | 源 view 点数 ≥ 64 | 点数不足拒绝 |
| 4 | Query 必须属于六类规范词表 | 非法类别拒绝 |
| 5 | Label 必须为 0 或 1 | 非法标签拒绝 |
| 6 | 正 query 必须有 GT；负 query 强制 GT=None | GT 不一致拒绝 |
| 7 | Candidate score 必须有限且位于 `[0,1]` | 非法分数拒绝 |
| 8 | Candidate box 必须为有限 canonical `(cx,cy,cz,l,w,h)` | 非法框拒绝 |
| 9 | Candidate score ≥ 0.50 | 低分候选不进入 hard pool |
| 10 | 正 query 的 candidate IoU < 0.25 | 记为 `low_iou` hard negative |
| 11 | 负 query 的高分 candidate | 记为 `target_absent` hard negative |

### 4.5 去重与配平

| 项目 | 配置 / 行为 |
| :--- | :--- |
| 同 sample/query | 只保留最危险候选：分数更高、IoU 更低者优先 |
| 空间去重中心距离 | ≤ 0.50 m 视为重复 |
| 空间去重 IoU | ≥ 0.50 视为重复 |
| 时间窗口 | 相同步轨迹内 step 差 ≤ 8 |
| 类别轮转 | 六类 round-robin |
| 场景轮转 | 类别内部按场景 round-robin |
| 最大总量 | 6,000 |
| 每类上限 | 1,000 |
| 每场景每类上限 | 25 |
| 每场景上限 | 100 |
| Replay record weight | 1.0 |
| 原始 hard candidates | 69,664 |
| 去重后 | 10,411 |
| 配平后 | 3,805 |

### 4.6 Replay v2 数据规模

| 项目 | 数值 |
| :--- | ---: |
| 场景 | 123 |
| Views | 3,678 |
| Records | 3,805 |
| Points | 154,149,120 |
| Label=0 | 1,317 |
| Label=1 | 2,488 |
| `target_absent` | 1,317 |
| `low_iou` | 2,488 |
| 最低 mined score | 0.500043 |
| 平均 mined score | 0.635833 |
| 最高 mined score | 0.826821 |
| 数据目录大小 | 约 1,000 MB |
| Rejected | 0 |

### 4.7 Replay v2 类别分布

| 类别 | Records | 每类上限 |
| :--- | ---: | ---: |
| bed | 1,000 | 1,000 |
| chair | 1,000 | 1,000 |
| couch | 1,000 | 1,000 |
| potted plant | 81 | 1,000 |
| toilet | 202 | 1,000 |
| tv | 522 | 1,000 |
| 合计 | 3,805 | 6,000 总上限 |

### 4.8 Capture schema

| 字段 | 含义 |
| :--- | :--- |
| `meta_json.schema_version` | Capture schema 版本 |
| `sample_id` | `<scene>/v<原始 view ID>` |
| `scene` / `split` | 来源场景与固定 train split |
| `episode` / `step` | Local-view 审计标识与原始 view ID |
| `source_view_index` | 原始场景 NPZ 中连续 offsets 下标 |
| `points_source` | `{npz, view}` 轻量引用，其中 `view` 为连续 `view_index`；capture 不重复保存大点云 |
| `poses` | 当前 view 的 K 帧位姿 |
| `records` | 仅包含至少一个 hard candidate 的 query record |
| `candidates` | score、canonical box、candidate_rank、reason |
| `mined_with` | 产生候选的 R1 E3 checkpoint |

### 4.9 Replay NPZ schema

| 字段 | 类型 | 含义 |
| :--- | :--- | :--- |
| `points` | `float16 (M,6)` | 入选 view 的原始点云副本；逐元素与源 local-view 一致 |
| `offsets` | `int64 (V+1,)` | Replay 内重新连续编号后的点云范围 |
| `records_json` | JSON scalar | 入选 hard records |
| `record.view` | int | Replay 文件内从 0 开始的连续 view ID |
| `record.query/label/weight` | str / int / float | 原 query、原标签、固定 replay weight=1.0 |
| `record.target` | object / null | 原始正 query GT 或负 query 空 GT |
| `hard_negative.source_sample_id` | str | 对应 capture sample |
| `hard_negative.source_episode/source_step` | str / int | 原始 local-view 来源 |
| `hard_negative.mined_score/mined_iou` | float | R1 E3 候选分数和 IoU |
| `hard_negative.mined_box` | list[6] | 被判定为 hard negative 的候选框 |
| `hard_negative.reason` | str | `low_iou` 或 `target_absent` |
| `hard_negative.candidate_rank` | int | 原始模型候选序号 |

## 5. R2 / R3 原始数据与 Replay 混合

### 5.1 每轮混合算法

| 步骤 | 行为 |
| ---: | :--- |
| 1 | 原始训练集 23,031 个 views 每轮完整遍历一次 |
| 2 | 根据 `replay_ratio=0.10` 计算 replay 步数：`round(23031×0.10/0.90)=2559` |
| 3 | Replay 每轮随机抽取 2,559 / 3,678 views；耗尽前不重复 |
| 4 | 合并 23,031 个 original steps 与 2,559 个 replay steps |
| 5 | 使用 seed 派生的 RNG 对整轮计划重新混洗 |
| 6 | 每个优化步仍只处理一个 view；不按 record 数量决定配比 |
| 7 | 下一 epoch 重新洗牌，使未入选 replay views 有机会进入后续轮次 |

### 5.2 优化步预算

| 阶段 | Epoch | 每轮 original | 每轮 replay | 每轮总步数 | Replay 比例 | 全程 original | 全程 replay | 全程总步数 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2 | 3 | 23,031 | 2,559 | 25,590 | 10.00% | 69,093 | 7,677 | 76,770 |
| R3 | 2 | 23,031 | 2,559 | 25,590 | 10.00% | 46,062 | 5,118 | 51,180 |

### 5.3 R2 / R3 固定参数

| 参数 | R2 | R3 |
| :--- | :--- | :--- |
| 初始化 | ScanRefer | R1 E3 |
| 数据 | Original 90% + replay v2 10% | Original 90% + replay v2 10% |
| 解冻 | U1 | U1 |
| 损失 | base + rank | base + rank |
| Rank weight / margin / IoU 门槛 | 1.0 / 0.2 / 0.25 | 1.0 / 0.2 / 0.25 |
| G1 LR | `1e-4` | `2e-5` |
| G2 LR | `5e-5` | `1e-5` |
| G3 LR | `1e-5` | `2e-6` |
| Weight decay | `1e-4` | `1e-4` |
| Gradient clip | 0.1 | 0.1 |
| Seed | 0 | 0 |
| Validation | 原始 local-view val | 原始 local-view val |

## 6. 数据验证结果

| 检查项 | 检查范围 | 结果 |
| :--- | :--- | :--- |
| 原始统计一致性 | 123 train scenes、23,031 views、89,620 records | 通过 |
| Capture 文件数 | 9,377 个 NPZ | 与 mining stats 一致 |
| 原始 view ID / 连续 view_index | 全部 capture | 映射错误 0 |
| Source NPZ / offsets | 全部 capture | 越界 0、路径错误 0 |
| Capture record / 原始 record | query、label、GT | 差异 0 |
| Candidate 合法性 | 69,664 个 candidates | 非有限值 0、非法 box 0、阈值违规 0 |
| 去重确定性 | 69,664 → 10,411 | 与 stats 完全一致 |
| 配平确定性 | 10,411 → 3,805 | 与 `selection.jsonl` 完全一致 |
| 类别上限 | 最大 1,000 / 1,000 | 通过 |
| 场景上限 | 最大 43 / 100 | 通过 |
| 场景×类别上限 | 最大 23 / 25 | 通过 |
| Replay 文件 / splits | 123 个场景 NPZ | 完全一致 |
| Replay offsets | 全部 123 个 NPZ | 起点、终点、单调性全部正确 |
| Replay records | 3,805 条 | 与 selection 和 stats 完全一致 |
| Replay points | 154,149,120 点 | 与原始 local-view 切片逐元素一致 |
| Train/val 泄漏 | Replay scenes 与原始 val scenes | 交集 0 |
| Loader 全量读取 | 3,678 views / 3,805 records | 映射错误 0 |
| Rejected | 完整 replay v2 构建 | 0 |
| 冒烟测试 | rank pair、streamed gradient、replay mix | 全部通过 |

| Loader 提示 | 影响判断 |
| :--- | :--- |
| Replay 目录不复制 `instances/` | 当前 R2/R3 仅使用 base + rank，不消费 `entities`，不影响训练 |
| Replay 关闭 GD sidecar | 当前 R2/R3 不启用 gate/distill，不影响训练 |
| 原始数据仍保留 instances 和 GD sidecar | 后续若重新启用对应分支，可从原始数据读取，不污染 replay v2 |

## 7. 数据产物与用途

| 路径 | 用途 | 是否直接进入训练 |
| :--- | :--- | :---: |
| `data/ft_datasets/local_view/train/*.npz` | Base、U1、R1、R2、R3 原始训练数据 | 是 |
| `data/ft_datasets/local_view/val/*.npz` | 所有阶段验证集 | 是，仅验证 |
| `data/ft_datasets/local_view/splits.json` | 冻结 train/val 场景边界 | 是 |
| `data/ft_datasets/local_view/stats.json` | 原始数据统计 | 否 |
| `data/ft_datasets/local_view/dropped.jsonl` | 原始数据丢弃审计 | 否 |
| `data/ft_datasets/local_view/gd/` | GD 教师侧车；本链路未启用 | 否 |
| `data/ft_datasets/local_view/instances/` | 实例表；本链路未消费 | 否 |
| `data/ft_datasets/r2_capture_v2/*/*.npz` | R1 E3 hard candidate 轻量引用与审计 | 否 |
| `data/ft_datasets/r2_capture_v2/mining_stats.json` | 全量挖掘统计 | 否 |
| `data/ft_datasets/hard_negative_replay_v2/train/*.npz` | R2/R3 hard replay | 是 |
| `data/ft_datasets/hard_negative_replay_v2/selection.jsonl` | 最终入选记录审计 | 否 |
| `data/ft_datasets/hard_negative_replay_v2/stats.json` | Replay 配置与统计 | 启动前校验 |
| `data/ft_datasets/hard_negative_replay_v2/splits.json` | Replay train scene 清单；val 固定为空 | 是 |
| `ft_tsp3d/ft_log/r2_data_v2/r2_data.done` | Replay v2 完成门禁 | 启动前校验 |

## 8. 执行入口

| 任务 | 命令 | 产物 |
| :--- | :--- | :--- |
| 原始数据生成 | `python ft_tsp3d/data_preprocess.py` | `data/ft_datasets/local_view` |
| U1 训练 | `PROFILES=u1 bash scripts/ft_unfreeze.sh` | `data/tsp3d_ft/unfreeze/u1_*` |
| R1 训练 | `bash scripts/ft_rank.sh` | `data/tsp3d_ft/rank/r1_*` |
| Replay v2 生成 | `CAPTURE_DIR=.../r2_capture_v2 OUTPUT_DIR=.../hard_negative_replay_v2 LOG_DIR=.../r2_data_v2 bash scripts/r2_after_r1.sh` | capture v2、replay v2、完成标记 |
| R2→R3 串行训练 | `bash scripts/ft_rank_harddata.sh` | `data/tsp3d_ft/r2`、`data/tsp3d_ft/r3` |
| 仅检查 R2/R3 命令 | `DRY_RUN=1 bash scripts/ft_rank_harddata.sh` | 不启动训练 |
