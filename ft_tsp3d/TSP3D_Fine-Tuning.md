# TSP3D Fine-Tuning Pipeline

## 1. 训练代码组成

`ft_tsp3d/` 全部为训练侧代码；模型侧只透传开关、不反向依赖。

| 文件 | 职责 |
| :--- | :--- |
| `axis_aligned_iou_loss.py` | 轴对齐 3D IoU/GIoU（训练侧独立实现，无 mmcv 依赖） |
| `calibrate_gate.py` | τ_gate 训练后校准（消费 `gate_val.npz`，保召回建议值） |
| `data_preprocess.py` | 数据供给层：habitat 仿真局部视角采样（与在线传感器同源）、正/负样本、canonical GT、GD 侧车同遍产出 |
| `data_loader.py` | 读 `local_view/{split}/<scene>.npz` + `gd/` 侧车 + `instances/` 实例表；视图级 LRU 缓存；视锥内可见实体计算 |
| `export_instances.py` | 逐场景实例 AABB 表导出（L_s1 标签面；分钟级、非重跑） |
| `ft_geometry.py` | 几何工具：canonical 系变换、视锥掩码、AABB 表面点、可见实体查询 |
| `gd_sidecar.py` | GD 蒸馏侧车生产：对已生成 `local_view` 回放 `records["poses"]`（同源 sensor 重渲）→ 本地 GD → `gd/{split}/<scene>.npz`（帧级 `ok`/dets）；可分片、断点续跑；`data_preprocess.py --gd-sidecar` 一步生成同链路 |
| `gd_teacher.py` | 蒸馏教师：GD 侧车 → `S_max` → 软标签 `Ŷ`（温度 T=1.5）+ 质检层（ok 帧、深度支撑、GT 投影验证） |
| `init_from_pretrain.py` | 旧 ckpt 部分加载（`strict=False`）、冻结清单、三组 LR + wd 双分组的参数组划分、冻结层评估 |
| `losses.py` | 三项损失完整实现（base / gate / distill）+ 匈牙利匹配 + 视图/批聚合（预测契约见文件头）；S1/S2（L_s1 / L_s2 含 TGP）随解耦分支移出，见 `ft_s12/` |
| `metrics.py` | 在线评测日志提取（SR / 拦截数 / 失败结构等，与历史口径核对） |
| `model_bridge.py` | 模型桥接：模型侧 `ft_outputs`（逐样本列表结构）→ 单 record 损失契约 pred |
| `pipeline.py` | 训练主循环：warmup+cosine 调度、逐 view 训练/验证、ckpt 保存、`--dump-gate` 校准数据、`--dry-run` 自检 |
| `smoke_test.py` | 离线自检：契约/损失/调度/几何/教师/数据/干跑训练等用例（S1/S2 用例见 `ft_s12/smoke_test_s12.py`） |
| `ft_s12/` | **S1/S2 归档区（训练侧）**，不被本管道导入：`losses_s12.py`（L_s1 / L_s2 含 TGP 分项 + `S12LossConfig`）、`ablation.py`（M0–M4 消融表 + 损失→`enable_gate`/`enable_prop`/`enable_refine` 装配推导）、`smoke_test_s12.py`（归档区自检）；接回步骤见其 `__init__.py` |


**模型侧接口**（`vlfm/tsp3d_models/` 与 `vlfm/vlm/tsp3d.py`，供对照）：

| 文件 | 训练相关职责 |
| :--- | :--- |
| `bdetr.py` | `ft_config` 开关（`enable_gate` + `gate_threshold` / `num_proposals`；S1/S2 的 `enable_prop` / `enable_refine` 已随解耦分支移出，入参被忽略）；训练模式 `forward` 返回 losses dict，`inputs["return_outputs"]=True` 时附 `ft_outputs` |
| `multilevel_head.py` | `forward_train`（`_base_ft_forward` 单候选路径）、`forward_test` 推理；模型侧保底损失仅 base / gate / distill 三项 |
| `ft_utils/` | base / gate / distill 微调工具包：`ft_modules.py`（门控头 `PresenceGate`、掩码/池化工具、盒转换、保底损失）、`prediction.py`（预测契约工具）、`boxes.py`（3D IoU/GIoU）；模型与本包都不导入 S1/S2 归档 |
| `ft_s12/` | **S1/S2 归档区**（不被模型导入）：`s12_decoupled.py`（提议/实体精修前向、结构化输出、逐样本损失 + 接口说明）、`axis_aligned_iou_loss.py`（vlfm 侧已无引用的旧 IoU 损失） |
| `trans_modules.py` | Transformer 编解码/注意力层；注意力快路径（`need_weights=False`、全 False bool mask→`None`）为大视图训练显存优化点，数值不变、训练/推理共用 |
| `vlm/tsp3d.py` | 在线服务包装（`TSP3D` / `TSP3DServer` / `TSP3DClient`）：权重与 `ft_config` 装配（env 链 `TSP3D_CHECKPOINT` / `TSP3D_MODE` / `TSP3D_GATE_THRESHOLD` + ckpt 回推，仅门控位）；ckpt 加载为宽松匹配（`strict=False`，旧 ckpt 的 S1/S2 键被忽略）；`predict` 返回 detections + diagnostics（`presence_prob` 等）；训练侧不调用 |

**开关映射（单变量一致性）**：训练侧按损失开关推导模型开关——
`gate/distill ⇒ enable_gate`。`prop/refine ⇒ enable_prop/enable_refine` 的推导同表还
保存在 `ft_s12/ablation.py::model_switches`（训练侧归档区）；模型侧对应装配说明见
`vlfm/tsp3d_models/ft_s12/s12_decoupled.py` 头部。

## 2. 数据依赖

| 产物 | 位置 | 说明 |
| :--- | :--- | :--- |
| 主数据 | `data/ft_datasets/local_view/{train,val}/<scene>.npz` | `points`(f16 M,6) + `offsets`(V+1) + `records_json`；训练规模（实测）：train **23,038 views / 89,624 records**、val **2,638 views / 10,360 records** |
| GD 侧车 | `data/ft_datasets/local_view/gd/{split}/` | 帧级 GD 证据（蒸馏教师；`--no-sidecar` 可关） |
| 实例表 | `data/ft_datasets/local_view/instances/{split}/` | L_s1 标签面（S1/S2 已归档，当前不消费；接回 L_s1 后需先补：`python ft_tsp3d/export_instances.py`） |
| 分组 | `splits.json`（train/val 冻结）；nav 20 场景不生成样本 | 评测场景不参与训练/校准 |
| 词表 | `vlfm/tsp3d_models/grounding_dino/vocab.py`（6 类） | prompt = record `query` 规范类名 |

## 3. 运行方法

```bash
# 单档（示例：gate 档 = base + L_gate；权重/日志分流 + 档位命名）
PYTHONPATH=. python ft_tsp3d/pipeline.py --losses gate \
    --ckpt data/tsp3d_models/tsp3d_scanrefer.pth \
    --out-dir ft_tsp3d/ft_log --weight-dir data/tsp3d_ft --log-dir ft_tsp3d/ft_log \
    --tag gate --dump-gate ft_tsp3d/ft_log/gate_gate_val.npz

# 三档单开关跑批 / 双机分批（见脚本头注释）
MACHINE=a bash scripts/ft_tsp3d.sh          # A 机：base → gate
MACHINE=b bash scripts/ft_tsp3d.sh          # B 机：base → distill
HEAD=2 MACHINE=a bash scripts/ft_tsp3d.sh   # 先跑队列前 2 档（看输出再续）
LIMIT=2 SCENES=00006-HkseAnWCgqk EPOCHS=1 VAL_EVERY=0 TASK=base \
    WEIGHT_DIR=/tmp/ft_smoke_w LOG_DIR=/tmp/ft_smoke_l bash scripts/ft_tsp3d.sh   # 冒烟（临时目录）

# 干跑自检（stub 模型 + 合成数据）
python ft_tsp3d/pipeline.py --dry-run --epochs 2 --stub-steps 8
```

**训练口径**：每步 = 1 view（正/负 records 共享点云，逐 record 前向，B=1）；
3 epoch；AdamW(β=0.9/0.999, ε=1e-8)；三组 LR（1e-4 / 5e-5 / 1e-5，各按 wd=0/1e-4 拆组）；
第 1 epoch 线性 warmup（1e-6 → 目标），其后 cosine → 1e-6；grad clip 0.1。

## 4. 产物（每档）

```
data/tsp3d_ft/<档>_<GPU>_ckpt_last.pth       # 最新权重（含 optimizer；键 "model" 与在线加载口径一致）
data/tsp3d_ft/<档>_<GPU>_ckpt_epoch<N>.pth
ft_tsp3d/ft_log/<档>_<GPU>_train.log         # 逐步分项损失 / LR / bad-step 计数
ft_tsp3d/ft_log/<档>_<GPU>_train_console.log # 控制台全量（含 [data]/[teacher]/[val] 等）
ft_tsp3d/ft_log/<档>_<GPU>_val.log           # 验证分项损失
ft_tsp3d/ft_log/<档>_<GPU>_gate_val.npz      # P_exist 校准数据（τ_gate 用）——**仅门控档
                                             # gate / distill 产出**；base 无门控头
                                             # （[val] 行 gate_pts=0）⇒ 本文件不存在属正常
ft_tsp3d/ft_log/<档>_<GPU>_train.done        # 跑批成功标记（SKIP_DONE 用）
```

`<GPU>` = 机器显卡编号（跑批脚本自动探测；可用 `GPU_TAG=...` 覆盖）。
