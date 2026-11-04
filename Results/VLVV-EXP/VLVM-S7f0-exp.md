# VLVM-S7f0：场机制目标覆盖、离线拟合与在线 shadow

> 合并说明：本文合并 S6gd2 离线 与 S6gd3 在线 shadow，并加入 S7 baseline 的离线复算与在线 shadow。
>
> **口径：field 只用于目标覆盖，不能做排序选择。** 本文只分析目标区域先验与覆盖/探索回放。

## 0. S7 baseline 离线复算

> 数据：`outputs/s7_baseline_3scene_2026-10-30_004608/detections/field_observations`（297 ZIP，约 45 GB）  
> 分析：`outputs/s7_baseline_3scene_2026-10-30_004608/field_analysis`  
> 配置：`cell_size=1.0 m`、`top_k=5`、`l2=1.0`、`min_visibility=0.15`、scene-grouped CV；semantic sidecar 未生成。

**数据源：**

| 数据 | run | 用途 |
| :--- | :--- | :--- |
| S6gd2 离线 | `outputs/s6_gd2_2026-10-27_restart` | §1–§7 的区域模型与覆盖回放 |
| S6gd3 在线 shadow | `outputs/s6_gd3_restart_2026-10-28_182915` | §8 的在线覆盖对照 |
| S7 baseline 离线 | `outputs/s7_baseline_3scene_2026-10-30_004608` | §0 的区域模型/覆盖复算；§8 在线 shadow |

配置：`cell_size=1.0 m`、`top_k=5`、`l2=1.0`、`min_visibility=0.15`、scene-grouped CV。  
说明：旧 `field_analysis_v2` 结果只在对照处保留。未生成全量 `region_semantics.jsonl`，主结果仍是**几何基线**；当前导航已接入**field 覆盖 shadow**（只读 `field_shadow.jsonl`）；后续 field 在线实验仍分两部分：目标覆盖，以及单独在线验证 `region_semantic`（见 §5.1），不把全量侧车作为前置。

### 0.1 输入数据

| 项 | S7 baseline | S6gd2 |
| :--- | ---: | ---: |
| field episodes | 297 | 297 |
| field frames | 46,640 | 42,946 |
| region samples | 1,242,782 | 1,070,940 |
| positive regions | 54,716（4.40%） | 40,470（3.78%） |
| negative regions | 1,188,066 | 1,030,470 |
| semantic sidecar | `false`（0 行） | `false`（0 行） |

### 0.2 区域模型

| 指标 | S7 baseline | S6gd2 |
| :--- | ---: | ---: |
| status | `ok` | `ok` |
| train AUC | 0.8201 | 0.8259 |
| train Brier | 0.03782 | 0.03099 |
| train NLL | 0.14636 | 0.12569 |
| intercept | -2.8981 | -3.1389 |

权重（`geom_log_distance`、`geom_frontier_proximity`、`geom_log_visits`、`geom_log_candidates`、`geom_observed`）：

| 数据 | weights |
| :--- | :--- |
| S7 baseline | `[-0.3956, 0.000053, 0.3279, 0.5981, -1.4769]` |
| S6gd2 | `[-0.2624, 0.0292, 0.1354, 0.9444, -1.1816]` |

Scene-grouped CV：

| held-out scene | S7 n | S7 AUC | S7 Brier | S7 NLL | S6 AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 750,354 | 0.8352 | 0.02653 | 0.11314 | 0.8774 |
| `QaLdnwvtxbs` | 380,662 | 0.7599 | 0.07356 | 0.28463 | 0.7824 |
| `mL8ThkuaVTM` | 111,766 | 0.6954 | 0.02828 | 0.12278 | 0.6615 |

### 0.3 覆盖回放

| 口径 | S7 可评估帧 | S7 目标区域在集合中 | S7 Top-5 命中 | S7 条件命中率 | S6 条件命中率 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| all | 46,640 | 17,050 | 14,877 | 0.8726 | 0.8315 |
| no-TP episodes | 2,209 | 2,209 | 1,583 | 0.7166 | 0.8407 |

- S7 baseline 全局 Top-5 命中：`14,877 / 46,640 = 31.90%`（目标区域只出现在 36.56% 的帧集合中）。
- S7 baseline 无 TP episode 122 个；S6gd2 为 132 个。
- 条件命中率与全局命中率不可混用。
- **解读**：S7 baseline 全局覆盖高于 S6gd2，但 no-TP 条件命中从 0.8407 降到 0.7166；全局提升不能直接归因于目标覆盖，接入前应按 no-TP 口径复核。

### 0.4 探索回放

| 指标 | S7 all | S7 no-TP | S6 all | S6 no-TP |
| :--- | ---: | ---: | ---: | ---: |
| Top-5 目标命中 | 0.8726 | 0.7166 | 0.8315 | 0.8407 |
| 目标区域进入 Top-1 | 0.5567 | 0.5192 | 0.5600 | 0.5256 |
| 命中目标且未被现有候选覆盖 | 0.9018 | 0.9918 | 0.8730 | 0.9867 |
| Top-5 新区域比例 | 0.9824 | 0.9966 | 0.9803 | 0.9954 |
| 前沿重叠 | 0.2036 | 0.2746 | 0.2324 | 0.3039 |
| 角向分散度 | 0.4835 | 0.4940 | 0.4749 | 0.4383 |
| Top-5 平均距离 | 2.689 m | 2.381 m | 2.677 m | 2.410 m |
| 后验熵 | 0.8465 | 0.8427 | 0.8128 | 0.7228 |
| Top-1/Top-2 margin | 0.0468 | 0.0717 | 0.0941 | 0.1662 |

### 0.5 处理产物

- `field_analysis/field_offline_report.json`
- `field_analysis/field_offline_summary.md`
- `field_analysis/field_inference_config.json`

## 1. 输入数据

| 项 | S6gd2 | 旧 `field_analysis_v2` |
| :--- | ---: | ---: |
| field episodes | 297 | 297 |
| field frames | 42,946 | 38,697 |
| region samples | 1,070,940 | 879,576 |
| positive regions | 40,470（3.78%） | 29,324（3.33%） |
| negative regions | 1,030,470 | 850,252 |
| semantic sidecar | `false`（0 行） | `false`（0 行） |

场归档完整：297 个 ZIP，约 42 GB。S6gd2 的导航步数更长（平均 144.60 步），因此 field frames 明显多于上一版；下面的区域模型与覆盖回放均基于新的归档重新拟合。

## 2. 区域模型

| 指标 | S6gd2 | 旧 `field_analysis_v2` |
| :--- | ---: | ---: |
| status | `ok` | `ok` |
| train AUC | **0.8259** | 0.8236 |
| train Brier | 0.03099 | 0.02617 |
| train NLL | 0.12569 | 0.11097 |
| intercept | -3.1389 | -3.4409 |

权重（`geom_log_distance`、`geom_frontier_proximity`、`geom_log_visits`、`geom_log_candidates`、`geom_observed`）：

```text
[-0.2624, 0.0292, 0.1354, 0.9444, -1.1816]
```

`geom_frontier_proximity` 在本版不再恒为 0，但权重仍接近 0；主要判别力来自 `geom_observed`（负向）和 `geom_log_candidates`（正向）。

Scene-grouped CV：

| held-out scene | n | positive | AUC | Brier | NLL |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 643,466 | 11,944 | 0.8774 | 0.01769 | 0.08766 |
| `QaLdnwvtxbs` | 326,158 | 24,990 | **0.7824** | 0.06440 | 0.25778 |
| `mL8ThkuaVTM` | 101,316 | 3,536 | 0.6615 | 0.03081 | 0.13578 |

区域模型整体没有退化：train AUC 与上一版基本持平，`QaL` held-out AUC 从 0.7484 升到 0.7824，`mL8` 从 0.6515 升到 0.6615；`5cd` 从 0.8876 降到 0.8774。Brier/NLL 变差与正样本率从 3.33% 升到 3.78% 同时出现；区域判别力仍由 AUC 支持，但概率校准不能视为变好。

**区域模型说明了什么**：它说明几何特征能在 1 m cell 级别区分“目标更可能存在的区域”和普通区域；主要权重是 `geom_observed` 负向、`geom_log_candidates` 正向，即目标区域更可能是“尚未充分观察、且已有候选线索”的区域。这支持**目标覆盖先验**，但不能说明动作选择或 STOP 可靠。`mL8` held-out AUC 只有 0.6615，也说明跨场景泛化并不均匀，不能把 0.8259 当成全局目标定位能力。

## 3. field 能力边界

field 只用于目标覆盖：在未被现有线索覆盖的区域与 frontier 上生成 Top-K 搜索区域，供只读 shadow 使用。**field 不能做排序选择**；其输出不进入目标选择、准入、记忆或 STOP。目标选择继续使用 `sigma_c`/TSP3D 与 GD 证据。

## 4. 覆盖回放

| 口径 | 可评估帧 | 目标区域在集合中 | Top-5 命中 | 命中率 |
| :--- | ---: | ---: | ---: | ---: |
| all | 42,946 | 14,458 | 12,022 | 0.8315 |
| no-TP episodes | 1,701 | 1,701 | 1,430 | 0.8407 |

补充：

- `all` 中有 28,488 / 42,946 帧目标区域不在当前区域集合中（66.3%），覆盖指标是**条件命中率**，不是全局覆盖；若按全部帧算，Top-5 目标命中只有 12,022 / 42,946 = **28.0%**。
- 165 个 episode 有 TP，因此无 TP episode 为 297 − 165 = 132；其中可评估帧 1,701。
- 与上一版比：`all` 从 0.8361 降到 0.8315，no-TP 口径从 0.8137 升到 0.8407。整体覆盖基本稳定，no-TP episode 反而更好；但目标不在区域集合的帧数更多，仍不能把条件覆盖率当作全局覆盖率。

## 4.1 探索回放

探索回放用于回答 field 是否把机器人带向更多目标、更分散的方向；指标如下：

| 指标 | S6gd2 all | S6gd2 no-TP | 旧 all | 旧 no-TP |
| :--- | ---: | ---: | ---: | ---: |
| Top-5 目标命中 | 0.8315 | 0.8407 | 0.8361 | 0.8137 |
| 目标区域进入 Top-1 | 0.5600 | 0.5256 | — | — |
| 命中目标且未被现有候选覆盖 | 0.8730 | 0.9867 | — | — |
| Top-5 新区域比例 | 0.9803 | 0.9954 | 0.9807 | 0.9947 |
| 前沿重叠 | 0.2324 | 0.3039 | 0.2731 | 0.3224 |
| 角向分散度 | 0.4749 | 0.4383 | 0.4834 | 0.4245 |
| Top-5 平均距离 | 2.677 m | 2.410 m | 2.536 m | 2.452 m |
| 后验熵 | 0.8128 | 0.7228 | 0.7911 | 0.6704 |
| Top-1/Top-2 margin | 0.0941 | 0.1662 | 0.1169 | 0.2082 |

补充：

- Top-5 目标命中的绝对值与上一版接近；在 no-TP episode 上还有提升。
- 目标区域进入 Top-1 的比例只有 0.560（all）/ 0.526（no-TP），说明场经常把目标区域放进 Top-5，但不一定排到第一位。该指标是本次报告新增，旧全量报告没有生成，不能与旧 20 集抽样的 Top-1 0.853 直接比较。
- `命中目标且未被现有候选覆盖` 为 0.873 / 0.987：Top-5 命中的目标区域绝大多数未被现有 TSP3D 候选覆盖，说明场确实在提出未覆盖区域探索方向；这是目前最有价值的信号。
- Top-5 新区域比例约 0.98，与上一版一致；前沿重叠从 0.273/0.322 降到 0.232/0.304，说明本版 Top-5 更少复读 frontier，但也可能更分散。
- 后验熵升高、margin 降低：场对“最该去的方向”比上一版更不确定；角向分散度仍在中等区间。
- **不能把 Top-5 覆盖强解释为“带向更确定的地方”**：目标进入 Top-1 只有 0.560/0.526，熵 0.8128/0.7228，margin 0.0941/0.1662；这些指标支持“覆盖更广、方向更分散”，不支持“更快收敛到确定目标”。
- 局限：上述仍是**几何场**；没有语义侧车，无法判断“目标上下文”是否带来额外覆盖。

## 5. 可行性结论

| 检查项 | 结果 |
| :--- | :--- |
| 数据采集/读取 | 通过：297 episodes、42,946 frames、1,070,940 regions |
| 区域模型 | 通过区域级判别：train AUC 0.8259；`5cd/QaL/mL8` held-out 0.8774/0.7824/0.6615。它说明“目标区域几何先验”可学；不能说明动作选择或 STOP 可靠 |
| 覆盖 | 条件覆盖 0.8315；no-TP 0.8407；但目标区域只在 33.7% 帧的集合中，全局 Top-5 命中 28.0%，不能宣称全局覆盖强 |
| 目标覆盖证据 | **field 的唯一能力**：Top-5 新区域比例 0.980，命中目标中 0.873–0.987 未被现有候选覆盖；前沿重叠 0.232–0.304；角向分散 0.44–0.47。它提供未覆盖区域探索方向，不提供收敛确定性 |
| 当前 field 定位 | **目标覆盖**：只用于 frontier/未覆盖区域的 Top-K 搜索；输出不进入目标选择、准入、记忆或 STOP |
| 语义侧车 | **不生成全量侧车作为前置**；`region_semantic` 改为后续独立在线导航实验臂，几何覆盖不依赖它 |

当前结论：**field 只保留目标覆盖能力。** 区域模型可作为区域级目标先验，覆盖/探索回放支持它能提出未覆盖区域探索方向；但目标进入 Top-1 仅 0.53–0.56，后验熵升高、margin 降低，说明它不能作为收敛器。目标选择仍由 `sigma_c`/TSP3D 与 GD 证据负责。

对后续工作的直接含义：

- 目标选择继续使用 `sigma_c`/现有 TSP3D 与 GD 证据；field 不进入该链路。
- field 的定位是：在现有目标覆盖之外生成 Top-K 搜索区域，优先选择未覆盖、非 frontier 复读、方向分散的区域。
- 覆盖指标必须同时报告条件覆盖和全局覆盖（当前 0.8315 vs 0.2800），避免把“目标已在区域集合中”的条件命中误写成全局目标覆盖。
- 全量 `region_semantics.jsonl` 对目标覆盖不是必需项；语义侧车不再作为几何覆盖的前置。

### 5.1 后续在线导航实验拆分

field 只保留目标覆盖部分，后续在线导航实验分两部分，不合并成一个机制开关：

1. **目标覆盖（当前只读 shadow）**：在线写入 frontier/未覆盖区域的 Top-K 覆盖 shadow，输出 `detections/field_shadow.jsonl`；不改变动作或目标选择。主指标是 no-TP episode 的 GT 区域覆盖、首次 TP 提前量、FP/steps 和轨迹变化。
2. **region_semantic（单独验证）**：当前不接入；待 GD 与 field 覆盖机制确定后，直接作为独立的在线导航 A/B 实验臂，验证区域语义上下文是否能改善目标覆盖。该臂通过前不与几何覆盖合并；失败时仍保留几何覆盖。

S7 baseline 已按该边界运行；field 仍只读，不触发上述在线臂。

### 5.2 导航 shadow 边界（当前实现）

- field 原始观测 shadow 在 `act()` 开头调用，只写 RGB-D/pose/frontier/map ZIP；异常被捕获，不改变动作、目标选择或 STOP。
- field 覆盖 shadow 在动作选择完成后调用，使用自己的增量区域网格和 `field_shadow.jsonl`；只记录 Top-K 区域、posterior、未覆盖区域比例、frontier 重叠、熵/margin 等，不写回 policy。
- `region_semantic` 当前仍不进入 policy：它由 `region_semantics.py` 离线读取已完成 ZIP 生成，`offline_analysis` 再消费；policy 不 import、不调用、不读取该结果。后续只在 GD 与 field 目标覆盖确定后单独做 A/B。
- `_get_target_object_location()` 只读取 TSP3D/GD memory 和生命周期字段，不读取 field shadow 或 `region_semantic`。因此 field shadow 不会改变导航框架。

## 6. 语义侧车：为什么没有生成，是否还需要

### 6.1 不是生成器失败

S6gd2 全量 run 没有 `<run>/detections/region_semantics.jsonl`，报告中的 warning 因此是：

```text
semantic region sidecar unavailable; fitting/evaluating geometry-only baseline
```

但生成器本身可用：BLIP-2 ITM 服务 12182 正在监听；样本 `QaLdnwvtxbs:46` 已生成 5 rows，`target_context_score≈0.5986`，端到端离线分析返回 `status=ok`。真正的原因是：

1. 全量侧车命令从未跑完；
2. 现有样本只写到 `/tmp/region_semantics_*.jsonl`，而 `offline_analysis` 默认只读取 `<run>/detections/region_semantics.jsonl`（或用 `--region-semantics` 显式指定）。样本文件没有进入 run 目录，所以分析仍然显示 `false`。

### 6.2 全量成本

生成器默认 `max_regions_per_frame=0`（全部区域）且 `every_n=1`。本 run 平均每帧约 24.94 个区域，5 个上下文、3 个投影高度：

| 配置 | 估算 ITM 调用量 | 说明 |
| :--- | ---: | :--- |
| 全区域 / 每帧 / 3 高度 / 5 prompt | **约 16.1M** | 数百小时量级，不适合直接跑 |
| 每 5 帧 / 每帧 2 区域 / 3 高度 / 5 prompt | 约 25.8 万 | 稀疏侧车，约数小时量级 |
| 每 5 帧 / 每帧 2 区域 / 3 高度 / 1 prompt | 约 5.2 万 | 仅适合按目标类拆分的专项实验 |

样本计时（10 帧、3 区域、5 prompt）约 38 秒，包含进程启动和 BLIP-2 调用；全区域全 prompt 的成本比样本高三个数量级，不是当前值得做的默认步骤。全量语义生成还会产生很大的 JSONL，并逐 episode 读取 42 GB field ZIP。

### 6.3 可选复现（不作为当前计划）

如果未来重新研究语义上下文，先写稀疏侧车并明确输出到 run 的 `detections/`：

```bash
python -m vlfm.semantic_spatial_field.region_semantics \
  --run-dir outputs/s6_gd2_2026-10-27_restart \
  --out outputs/s6_gd2_2026-10-27_restart/detections/region_semantics.jsonl \
  --every-n 5 --max-regions-per-frame 2 \
  --region-z-levels 0.4,0.8,1.2 \
  --contexts "a bed,a chair,a couch,a toilet,a television"
```

这只是**稀疏侧车**：没有覆盖到的区域在 `SemanticSidecar` 中是 unavailable，不是负证据。若要跑全区域，则去掉 `--every-n` 和 `--max-regions-per-frame`，但应按上面的成本表先确认机器和 BLIP-2 队列能承受。

样本验证命令（写到 `/tmp`，不进入离线分析主输出）：

```bash
python -m vlfm.semantic_spatial_field.region_semantics \
  --run-dir outputs/s6_gd2_2026-10-27_restart \
  --out /tmp/region_semantics_qal46.jsonl \
  --scenes QaLdnwvtxbs --episode-ids 46 \
  --max-frames 5 --every-n 1 --max-regions-per-frame 2 \
  --region-z-levels 0.4,0.8,1.2 \
  --contexts "a bed,a chair,a couch,a toilet,a television"
```

### 6.4 当前是否还需要

field 只保留目标覆盖，因此**不再生成全量语义侧车作为前置**；几何覆盖不依赖它：

- 几何场已经能给出未覆盖区域 Top-K、新区域比例和方向分散度；
- `region_semantic` 改为后续**独立在线导航实验臂**，直接验证其可行性，而不是先离线生成全量侧车；
- 不应因为侧车缺失而阻塞导航或把几何场结果标成无效。

限制仍然存在：

- 区域是 2D cell，投影高度只能候选化；目标区域在当前视角不可见时，该行会被跳过，而不是当作负证据。
- BLIP-2 上下文分数差异较小（样本约 0.56–0.65），是否足以改善探索判断需要实验，不能仅凭小样本断言。
- 侧车按 `region_id` 对齐；缺失行是 unavailable，不是负样本。

## 7. 产物

- `field_analysis/field_offline_report.json`
- `field_analysis/field_offline_summary.md`
- `field_analysis/field_inference_config.json`
- `detections/field_shadow.jsonl`：在线 field 覆盖只读 shadow（Top-K 区域、posterior、覆盖/多样性指标）
- `detections/field_observations/`：297 ZIP，约 42 GB

## 8. 在线目标覆盖 shadow（S6gd3 + S7 baseline）

> 本节合并原 S7f1 的 S6gd3 在线统计，并加入 S7 baseline。在线 shadow 只读取动作选择后的区域场，不改变动作或目标选择。

### 8.1 完整性

| 项 | S6gd3 | S7 baseline |
| :--- | ---: | ---: |
| run | `outputs/s6_gd3_restart_2026-10-28_182915` | `outputs/s7_baseline_3scene_2026-10-30_004608` |
| field shadow 行数 | 45,603 | 46,640 |
| episodes | 297 | 297 |
| `weights_loaded=true` | 45,603 / 45,603（100%） | 46,640 / 46,640（100%） |
| `top_k=5` 行数 | 41,460 | 42,495 |
| `top_k<5` 行数 | 4,143 | 4,145 |
| explore / navigate / initialize | 31,129 / 10,910 / 3,564 | 31,596 / 11,480 / 3,564 |
| 平均 `n_regions` | 24.13（max 87） | 24.45（max 87） |

### 8.2 全局步级覆盖

| 指标 | S6gd3 | S7 baseline |
| :--- | ---: | ---: |
| Top-1 命中率（全部帧） | 17.11% | 16.85% |
| Top-5 命中率（全部帧） | 28.50% | 28.03% |
| 至少一次 Top-1 命中的 episodes | 101 / 297（34.0%） | 100 / 297（33.67%） |
| 至少一次 Top-5 命中的 episodes | 165 / 297（55.6%） | 165 / 297（55.56%） |

### 8.3 分场景

| 场景 | S6gd3 rows | S6gd3 Top-1 | S6gd3 Top-5 | S7 rows | S7 Top-1 | S7 Top-5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 21,161 | 18.1% | 26.4% | 22,118 | 17.33% | 25.69% |
| `QaLdnwvtxbs` | 15,139 | 23.8% | 46.0% | 15,184 | 24.06% | 45.50% |
| `mL8ThkuaVTM` | 9,303 | 3.9% | 4.8% | 9,338 | 3.96% | 5.17% |

### 8.4 成功/失败与 no-TP episode

| 口径 | S6gd3 rows | S6gd3 Top-1 | S6gd3 Top-5 | S7 rows | S7 Top-1 | S7 Top-5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| success episodes | 18,889 | 17.1% | 31.0% | 19,446 | 16.60% | 30.59% |
| failed episodes | 26,714 | 17.1% | 26.7% | 27,194 | 17.02% | 26.20% |
| no-TP episodes | 15,004 | 4.3% | 5.1% | 15,535 | 4.18% | 4.94% |

- S6gd3：174 / 297 个 episode 出现过 TP，其余 123 个无 TP。
- S7 baseline：175 / 297 个 episode 出现过 TP，其余 122 个无 TP。
- 上表 no-TP 口径是**在线全局**；S7f0 §4 的 0.8407 / S7 baseline 离线复算是“目标区域在区域集合中”的条件覆盖率，两者不可直接比较。

### 8.5 探索特征

| 指标 | S6gd3 均值 | S7 baseline 均值 |
| :--- | ---: | ---: |
| `posterior_entropy` | 0.696 | 0.696 |
| `top1_top2_margin` | 0.123 | 0.121 |
| `candidate_novelty_rate`（新区域比例） | 0.704 | 0.698 |
| `frontier_overlap_rate` | 0.269 | 0.267 |
| `angular_dispersion` | 0.491 | 0.490 |
| `mean_topk_distance_m` | 2.595 m | 2.608 m |
| `top1_to_nav_goal_m` | 3.317 m（n=11,327） | 3.276 m（n=12,004） |

### 8.6 边界

- field shadow 写入正常；S6gd3 与 S7 baseline 的全局 Top-5 覆盖接近，Top-1 仍低。
- 在线 shadow 不能替代离线 ZIP 复算；条件覆盖、目标区域是否在集合中、no-TP 条件命中只在 §0/§4 的离线结果中计算。
- §0.3 离线全局 Top-5 31.90% 与 §8.2 在线 28.03% 的目标判定/可评估帧口径不同，不能直接视为机制退化，需按同一目标映射复核。
- 保持只读：field 只用于目标覆盖，不把 field 分数写入 `sigma_c`、STOP 或 GD 准入。

