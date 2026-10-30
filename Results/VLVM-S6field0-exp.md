# VLVM-S6field0：场机制离线拟合与可行性

> 数据：`outputs/s6_gdmech_3scene_2026-10-27_142116`  
> 分析：`outputs/s6_gdmech_3scene_2026-10-27_142116/field_analysis_v2`  
> 配置：`cell_size=1.0 m`、`top_k=5`、`l2=1.0`、`min_visibility=0.15`、scene-grouped CV。  
> 修订：修正 `frontiers_3d` 为空时未使用 `frontiers_2d` 的问题；旧版 `field_analysis` 的 `geom_frontier_proximity` 权重恒为 0，本文以 `field_analysis_v2` 为准。

## 1. 输入数据

| 项 | 值 |
| :--- | ---: |
| field episodes | 297 |
| field frames | 38,697 |
| region samples | 879,576 |
| positive regions | 29,324 |
| negative regions | 850,252 |
| semantic sidecar | `false`（0 行） |
| candidate rows | 4,269 |
| track rows | 14,084 |
| nav_goal rows | 4,837 |

场归档完整：297 个 ZIP，约 38 GB。该 run 没有生成全量 `region_semantics.jsonl`，因此下面的主表仍是**几何基线**；语义侧车生成器与接入代码已补齐，见第 6 节。

## 2. 区域模型

| 指标 | 值 |
| :--- | ---: |
| status | `ok` |
| train AUC | 0.8236 |
| train Brier | 0.02617 |
| train NLL | 0.11097 |
| `geom_frontier_proximity` 权重 | 0.0932（修复前为 0.0） |

Scene-grouped CV：

| held-out scene | n | AUC | Brier | NLL |
| :--- | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 557,936 | 0.8876 | 0.0149 | 0.0785 |
| `QaLdnwvtxbs` | 222,085 | 0.7484 | 0.0551 | 0.2264 |
| `mL8ThkuaVTM` | 99,555 | 0.6515 | 0.0313 | 0.1399 |

## 3. 候选排序

有效候选 4,088 / 4,269；无效 181。

| 排序器 | pairwise AUC | Top-1 TP（全组） | Top-1 TP（有 TP 组） |
| :--- | ---: | ---: | ---: |
| field | 0.7061 | 0.3050 | 1.0000 |
| baseline `sigma_c` | **0.7986** | 0.3050 | 1.0000 |

分场景：

| 场景 | n | positive | field AUC | baseline AUC |
| :--- | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 1,435 | 340 | 0.7013 | **0.7718** |
| `QaLdnwvtxbs` | 2,125 | 836 | 0.7236 | **0.8162** |
| `mL8ThkuaVTM` | 528 | 71 | 0.6362 | **0.9054** |

## 4. 覆盖回放

| 口径 | 可评估帧 | 目标区域在集合中 | Top-5 命中 | 命中率 |
| :--- | ---: | ---: | ---: | ---: |
| all | 38,697 | 12,059 | 10,083 | 0.8361 |
| no-TP-candidate episodes | 1,777 | 1,777 | 1,446 | 0.8137 |

补充：

- `all` 中有 26,638 / 38,697 帧目标区域不在当前区域集合中，覆盖指标是条件命中率，不是全局覆盖。
- no-TP candidate 的 episode 数为 145；其中可评估帧 1,777。
- `episodes_with_tp_candidate=152`。

## 4.1 探索回放（新增）

候选排序只回答“已有候选该先看哪个”，不回答“场是否把机器人带向更多目标/更分散的方向”。新增指标：

| 口径 | Top-5 目标命中 | Top-5 非候选比例 | 前沿重叠 | 角向分散度 | Top-5 平均距离 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| all | 0.8361 | 0.9807 | 0.2731 | 0.4834 | 2.536 m |
| no-TP-candidate episodes | 0.8137 | 0.9947 | 0.3224 | 0.4245 | 2.452 m |

补充：

- 后验熵 `all=0.7911`、no-TP `=0.6704`；Top-1/Top-2 margin `all=0.1169`、no-TP `=0.2082`。margin 越高表示场对“最该去的方向”越确定。
- 非候选比例约 0.98：Top-5 中绝大多数区域不在当前 TSP3D 候选/轨迹集合内，说明场有能力提出**候选集合之外的探索方向**；这不能替代候选排序，但说明“候选门未通过”不能直接推出“场没有探索价值”。
- 前沿重叠 all=0.273、no-TP=0.322：Top-5 与 frontier 有实质重叠，但并非只做 frontier 复读。
- 角向分散度基于 Top-5 相对机器人方位的 resultant length，`1` 表示完全分散，`0` 表示同一方向；当前约 0.42–0.48，属于中等分散。
- 20 集抽样（3,426 帧，660 帧目标可见）进一步给出：Top-5 命中 0.915、目标进入 Top-1 的比例 0.853、命中目标中 **0.833 的目标区域不在当前候选集合内**。这说明场不是只在已有候选里重排，而是能把目标方向带进探索集；该抽样与全量不可直接比较，只作为方向性证据。
- 局限：上述仍是**几何场**；没有语义侧车，无法判断“目标上下文”是否带来额外覆盖。

## 5. 可行性结论

| 检查项 | 结果 |
| :--- | :--- |
| 数据采集/读取 | 通过：297 episodes、38,697 frames、879,576 regions |
| 区域模型拟合 | 通过：train AUC 0.8236；但 mL8 held-out AUC 仅 0.6515 |
| 候选排序门 | 未通过：field AUC 0.7061 < baseline `sigma_c` 0.7986，且三个场景均低于 baseline |
| 覆盖门 | 条件覆盖 0.8361；no-TP 口径 0.8137，但 26,638 帧目标不在区域集合，不能当全局覆盖 |
| 探索门 | 有条件通过：Top-5 目标覆盖约 0.84，Top-5 非候选比例 0.98，前沿重叠 0.27–0.32；说明场能提出候选外方向 |
| 当前几何基线整体可行性 | **不足以作为候选门**：排序低于 `sigma_c`；但“无探索价值”不成立 |
| 完整语义场可行性 | **仍未判定**：全量 run 无 `region_semantics.jsonl`；生成器与逐上下文特征已补齐，尚未跑全量语义拟合 |

当前结论：**场机制的数据链路完整，几何基线在候选排序上仍不如 `sigma_c`，因此不能直接替换候选门；但探索回放显示它能覆盖更多目标区域、提出大量候选外区域并保持中等方向分散，不能因候选排序单项失败而否定场机制。** 语义场是否真正有效，需要生成全量 `region_semantics.jsonl` 后用同一 scene-grouped CV 对比几何场与语义场。

对升级条件的直接含义：

- 候选门仍应保留 `sigma_c`/现有 TSP3D 证据；场分数更适合作为**探索先验**（候选外 Top-K、frontier 排序、方向多样化），而不是候选准入分数。
- 若要把场用于候选准入，必须先完成 `candidate → tracking → GD match → field` 的顺序，而不是当前的 `candidate match → field`；否则 `gd_tsp_new` 类假阳性会继续被场放大。该顺序改造尚未实现。
- 语义侧车应优先补“目标类 prompt + 场景上下文”的逐上下文分数；只取均值会把床/电视等不同目标压成同一标量，无法提供多样判据。

完整报告见：

- `field_analysis_v2/field_offline_report.json`
- `field_analysis_v2/field_offline_summary.md`
- `field_analysis_v2/field_inference_config.json`
- `field_analysis_v2/field_candidate_scores.jsonl`
- `field_exploration_20c/field_offline_report.json`（20 集抽样，含 target_top1 / outside_candidates）

## 6. 语义侧车生成与接入

新增 `vlfm/semantic_spatial_field/region_semantics.py`：

- 从 field ZIP 读取 RGB、`camera_to_episodic`、深度内参和机器人位姿；
- 用与离线分析相同的 `IncrementalRegionBuilder` 生成区域 ID；
- 将区域中心投影到当前 RGB，裁剪 patch，调用 BLIP-2 ITM 对上下文 prompt 打分；
- 输出与 `SemanticSidecar` 对齐的 `region_semantics.jsonl`。

侧车按 `region_id` 对齐；缺失区域在 `SemanticSidecar` 中标记为不可用而不是负证据。逐上下文分数会展开为 `sem_context_<prompt>`，并额外给出目标条件分数 `sem_target_context` / `sem_target_margin`；`tv↔television`、`couch↔sofa` 自动映射，其他目标按子串匹配，匹配不到时目标特征置 0，不伪造成负证据。

用法（先跑含目标回合，确认投影/裁剪/分数后再全量）：

```bash
python -m vlfm.semantic_spatial_field.region_semantics \
  --run-dir outputs/s6_gdmech_3scene_2026-10-27_142116 \
  --out /tmp/region_semantics_qal46.jsonl \
  --scenes QaLdnwvtxbs --episode-ids 46 \
  --max-frames 200 --every-n 2 --max-regions-per-frame 3 \
  --region-z-levels 0.4,0.8,1.2 \
  --contexts "a bed,a chair,a couch,a toilet,a television"
```

已验证：

| 项 | 结果 |
| :--- | :--- |
| 单集小样本 | `5cdEh9F2hJL:0`，2 rows，`SemanticSidecar.available=true` |
| 目标条件字段 | `5cdEh9F2hJL:0`（target=`tv`）1 row：`target_context_prompt=a television`、`target_context_score=0.5877`、margin=0.0178；`sem_target_context` 可读 |
| 含目标回合 | `QaLdnwvtxbs:46`，11 rows（30 帧、每 5 帧、每帧 3 区域；大量区域不可见） |
| 多高度投影 | 5 帧 × 2 区域 × 3 高度共 30 次投影，17 次不可见、0 次裁剪失败、5 rows；逐行高度在 0.4/0.8/1.2 中按最高分选择 |
| 逐上下文特征 | `sem_context_<prompt>`、`sem_target_context`、`sem_target_margin`；5 个 prompt 时侧车特征维数 10，几何+语义总维数 15 |
| 端到端拟合 | `QaLdnwvtxbs:46` + 5 行旧侧车（无 target 字段）：模型 `ok`、`semantic_sidecar_available=true`、逐上下文权重可读；单回合 AUC 仅证明通路，不可当泛化结果 |

限制：

- 全量语义生成需要 BLIP-2 逐区域打分，成本高；调用量约为 `frames / every_n × regions_per_frame × heights × prompts`。本次没有全量生成，因此第 2–4 节仍是几何基线。
- 区域是 2D cell，投影高度只能候选化；目标区域若在当前视角不可见，该行会被跳过，而不是当作负证据。
- BLIP-2 上下文分数差异较小（冒烟样本约 0.56–0.65），是否足以支撑候选排序/探索门，需要全量数据验证，不能仅凭小样本断言。
