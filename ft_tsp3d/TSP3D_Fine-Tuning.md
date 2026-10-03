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
| `losses.py` | 五项损失完整实现（base / gate / s1 / s2(含 TGP) / distill）+ 匈牙利匹配 + 视图/批聚合（预测契约见文件头） |
| `metrics.py` | 在线评测日志提取（SR / 拦截数 / 失败结构等，与历史口径核对） |
| `model_bridge.py` | 模型桥接：模型侧 `ft_outputs`（逐样本列表结构）→ 单 record 损失契约 pred |
| `pipeline.py` | 训练主循环：warmup+cosine 调度、逐 view 训练/验证、ckpt 保存、`--dump-gate` 校准数据、`--dry-run` 自检 |
| `smoke_test.py` | 离线自检：契约/损失/调度/几何/教师/数据/干跑训练等用例 |


**模型侧接口**（`vlfm/tsp3d_models/` 与 `vlfm/vlm/tsp3d.py`，供对照）：

| 文件 | 训练相关职责 |
| :--- | :--- |
| `bdetr.py` | `ft_config` 开关（`enable_gate` / `enable_prop` / `enable_refine`）；训练模式 `forward` 返回 losses dict，`inputs["return_outputs"]=True` 时附 `ft_outputs` |
| `multilevel_head.py` | `forward_train`（含 `_baseline_train_forward` 单候选路径 / `_ft_forward` 解耦路径）、`forward_ft` 结构化输出、`forward_test` 推理 |
| `ft_modules.py` | 门控头（`PresenceGate`）与模型侧保底损失函数（训练侧不使用其数值） |
| `trans_modules.py` | Transformer 编解码/注意力层；注意力快路径（`need_weights=False`、全 False bool mask→`None`）为大视图训练显存优化点，数值不变、训练/推理共用 |
| `vlm/tsp3d.py` | 在线服务包装（`TSP3D` / `TSP3DServer` / `TSP3DClient`）：权重与 `ft_config` 装配（env 链 `TSP3D_CHECKPOINT` / `TSP3D_MODE` / `TSP3D_GATE_THRESHOLD` + ckpt 回推）、`predict` 返回 detections + diagnostics（`presence_prob` 等）；训练侧不调用 |

**开关映射（单变量一致性）**：训练侧按损失开关推导模型开关——
`gate/distill ⇒ enable_gate`；`prop/refine ⇒ enable_prop`；`refine ⇒ enable_refine`（含 TGP）。

## 2. 数据依赖

| 产物 | 位置 | 说明 |
| :--- | :--- | :--- |
| 主数据 | `data/ft_datasets/local_view/{train,val}/<scene>.npz` | `points`(f16 M,6) + `offsets`(V+1) + `records_json`；训练规模（实测）：train **23,038 views / 89,624 records**、val **2,638 views / 10,360 records** |
| GD 侧车 | `data/ft_datasets/local_view/gd/{split}/` | 帧级 GD 证据（蒸馏教师；`--no-sidecar` 可关） |
| 实例表 | `data/ft_datasets/local_view/instances/{split}/` | L_s1 标签面；**缺表时 s1 项自动跳过**——训练 s1 前先补：`python ft_tsp3d/export_instances.py` |
| 分组 | `splits.json`（train/val 冻结）；nav 20 场景不生成样本 | 评测场景不参与训练/校准 |
| 词表 | `vlfm/tsp3d_models/grounding_dino/vocab.py`（6 类） | prompt = record `query` 规范类名 |

## 3. 运行方法

```bash
# 单档（示例：s2 档 = base + L_s2 含 TGP；权重/日志分流 + 档位命名）
PYTHONPATH=. python ft_tsp3d/pipeline.py --losses s2,tgp \
    --ckpt data/tsp3d_models/tsp3d_scanrefer.pth \
    --out-dir ft_tsp3d/ft_log --weight-dir data/tsp3d_ft --log-dir ft_tsp3d/ft_log \
    --tag s2 --dump-gate ft_tsp3d/ft_log/s2_gate_val.npz

# 五档单开关跑批 / 双机分批（见脚本头注释）
MACHINE=a bash scripts/ft_tsp3d.sh          # A 机：base → gate → s1
MACHINE=b bash scripts/ft_tsp3d.sh          # B 机：base → s2 → distill
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
                                             # gate / distill 产出**；base/s1/s2 无门控头
                                             # （[val] 行 gate_pts=0）⇒ 本文件不存在属正常
ft_tsp3d/ft_log/<档>_<GPU>_train.done        # 跑批成功标记（SKIP_DONE 用）
```

`<GPU>` = 机器显卡编号（跑批脚本自动探测；可用 `GPU_TAG=...` 覆盖）。
