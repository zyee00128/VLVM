# TSP3D Fine-Tuning Pipeline

## 1. 训练代码组成

`ft_tsp3d/` 全部为训练侧代码；模型侧只透传开关、不反向依赖。

| 文件 | 职责 |
| :--- | :--- |
| `axis_aligned_iou_loss.py` | 轴对齐 3D IoU/GIoU（训练侧独立实现，无 mmcv 依赖） |
| `eval_tools/calibrate_gate.py` | τ_gate 训练后校准（消费 `gate_val.npz`，保召回建议值） |
| `data_processing/data_preprocess.py` | 数据供给层：habitat 仿真局部视角采样、正/负样本与 canonical GT |
| `data_loader.py` | 读 `local_view/{split}/<scene>.npz`；视图级 LRU 缓存 |
| `init_from_pretrain.py` | 旧 ckpt 部分加载（`strict=False`）、冻结清单、Rank 分层参数组、Gate-only 参数组与冻结层评估 |
| `losses.py` | base / R1v1 `rank_cross` / 独立 Gate 损失、匈牙利匹配与视图聚合 |
| `model_bridge.py` | 模型桥接：模型侧 `ft_outputs`（逐样本列表结构）→ 单 record 损失契约 pred |
| `pipeline.py` | 训练主循环：warmup+cosine、tail-LR 续训、逐 view 训练/验证、ckpt 与 Gate dump 保存；Gate 只允许独立训练 |
| `smoke_test.py` | 离线自检：契约、损失、调度、数据、Gate-only 与干跑训练 |


**模型侧接口**（`vlfm/tsp3d_models/` 与 `vlfm/vlm/tsp3d.py`，供对照）：

| 文件 | 训练相关职责 |
| :--- | :--- |
| `bdetr.py` | `ft_config` 的 Gate 开关、独立初始化 RNG、Gate-only 前向与推理装配 |
| `multilevel_head.py` | `forward_train` 仅返回 pipeline 消费的可微预测；`forward_test` 保持推理行为 |
| `ft_utils/` | 当前 base / gate 模型工具：`PresenceGate`、预测契约与 3D IoU/GIoU |
| `trans_modules.py` | Transformer 编解码/注意力层；注意力快路径（`need_weights=False`、全 False bool mask→`None`）为大视图训练显存优化点，数值不变、训练/推理共用 |
| `vlm/tsp3d.py` | 在线服务包装、权重与 Gate 配置装配；`predict` 返回 detections 与 diagnostics |

退役实验代码统一保存在 `archive/retired_ft/`；现役模型与训练 pipeline 不导入该目录，删除整个归档不会影响运行。

## 2. 数据依赖

| 产物 | 位置 | 说明 |
| :--- | :--- | :--- |
| 主数据 | `data/ft_datasets/local_view/{train,val}/<scene>.npz` | `points`(f16 M,6) + `offsets`(V+1) + `records_json`；当前 loader 实测：train **23,031 views / 89,620 records**、val **2,638 views / 10,360 records** |
| 分组 | `splits.json`（train/val 冻结）；nav 20 场景不生成样本 | 评测场景不参与训练/校准 |
| 词表 | `vlfm/tsp3d_models/grounding_dino/vocab.py`（6 类） | prompt = record `query` 规范类名 |

## 3. 运行方法

```bash
# 当前唯一 Gate 训练口径：VLVM V1 + 原始 local_view + 冻结核心
DRY_RUN=1 bash scripts/ft_gate.sh
bash scripts/ft_gate.sh

# 干跑自检（stub 模型 + 合成数据）
python ft_tsp3d/pipeline.py --dry-run --epochs 2 --stub-steps 8
```

**当前 Gate-only 口径**：从 `data/vlvm_v1/vlvm_v1.pth` 初始化，核心完全冻结，只使用原始 `local_view` 训练 `presence_gate.*` 3 epoch。联合训练和 hard-data/replay Gate 已删除。

## 4. 产物（每档）

```
data/tsp3d_ft/gate_only_original/<tag>_ckpt_last.pth
data/tsp3d_ft/gate_only_original/<tag>_ckpt_epoch{1,2,3}.pth
ft_tsp3d/ft_log/gate_only_original/<tag>_train.log
ft_tsp3d/ft_log/gate_only_original/<tag>_train_console.log
ft_tsp3d/ft_log/gate_only_original/<tag>_val.log
ft_tsp3d/ft_log/gate_only_original/<tag>_gate_val_epoch{1,2,3}.npz
ft_tsp3d/ft_log/gate_only_original/<tag>_calibration_e{1,2,3}.json
ft_tsp3d/ft_log/gate_only_original/<tag>_checkpoints.sha256
```

`<GPU>` = 机器显卡编号（跑批脚本自动探测；可用 `GPU_TAG=...` 覆盖）。
