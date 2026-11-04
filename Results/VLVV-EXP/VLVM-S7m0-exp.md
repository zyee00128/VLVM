# VLVM-S7m0：middle-tier promotion 日志与离线拟合

> 运行目录：`outputs/s7_baseline_3scene_2026-10-30_004608`  
> 在线日志：`detections/middle_rank_shadow.jsonl`  
> 离线模型：`middle_tier/middle_tier_model.json`  
> 在线排序：`enable_middle_tier_ranking=False`；本文件只整理 shadow 与离线拟合，不改变导航动作。

## 1. 日志完整性

| 项 | 值 |
| :--- | ---: |
| shadow 行数 | 16,032 |
| 覆盖 episodes | 243 / 297 |
| step 范围 | 12–499 |
| `gd-only` 行数 | 13,906 |
| `tsp-confirmed` 行数 | 2,126 |
| `selected=true` 行数 | 8,825 |
| 在线 `model_available=true` 行数 | 0 |
| 在线 `p` / `rank` | `null` |

### 1.1 分场景

| 场景 | 行数 | 占比 | 覆盖 episodes | 有 selected 行 episodes |
| :--- | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 7,415 | 46.25% | 97 | 95 |
| `QaLdnwvtxbs` | 6,727 | 41.96% | 98 | 96 |
| `mL8ThkuaVTM` | 1,890 | 11.79% | 48 | 47 |

### 1.2 类型与状态

| type | state | 行数 | selected=true | TP |
| :--- | :--- | ---: | ---: | ---: |
| `gd-only` | `gd` | 13,906 | 8,362 | 4,473 |
| `tsp-confirmed` | `confirmed` | 2,126 | 463 | 567 |

- 至少一次 `gd-only` 被选中的 episodes：225
- 至少一次 `tsp-confirmed` 被选中的 episodes：40
- `selected_before`、`n_selectable`、`static_rank` 均已记录。

### 1.3 覆盖与占比解释

**mL8 为什么只有 48 集、11.79% 行**

| 场景 | 有 middle 的 episodes | rows / episode | 中位步数 | 中位候选数 | selected 行 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `5cdEh9F2hJL` | 97 | 76.4 | 144 | 15 | 3,621 |
| `QaLdnwvtxbs` | 98 | 68.6 | 87.5 | 25 | 4,099 |
| `mL8ThkuaVTM` | 48 | 39.4 | 101.5 | 8.5 | 1,105 |

mL8 的 99 集里只有 48 集产生 middle 行，另外 51 集没有可进入 middle 层的候选/选择：

- 47 个 `never_saw...likely_infeasible` 没有 middle 行（另 1 个有 16 行但无 TP）；
- 2 个 `never_saw...feasible` 没有 middle 行；
- 2 个 success episode 没有 middle 行；
- 40 个 `false_positive` 全部有 middle 行。

因此 mL8 的低占比首先来自 **没有候选可记录**，其次才是 episode 更短、候选更少：mL8 有 middle 的 episode 中位仅 8.5 个候选、15 个 nav goal，单集产生 39.4 行，低于 5cd 的 76.4 和 QaL 的 68.6。离线拟合中 mL8 只有 `63 + 3 = 66 / 652` 个样本（10.1%），其中 `tsp-confirmed` 仅 3 个；模型主要被 5cd/QaL 主导，不能据此认为 mL8 的 middle-tier 行为已被充分验证。

**`tsp-confirmed` 为什么远少于 `gd-only`**

| type | 行数 | 唯一条目 | 覆盖 episodes | selected 行 | selected episodes |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `gd-only` | 13,906 | 737 | 236 | 8,362 | 225 |
| `tsp-confirmed` | 2,126 | 146 | 87 | 463 | 40 |

`tsp-confirmed` 只占 13.3% 的行、5.2% 的 selected 行。原因是两层叠加，**不只是总量少**：

1. **供给更少**：`tsp-confirmed` 只有 146 个唯一条目、87 个 episode；`gd-only` 为 737 个条目、236 个 episode。confirmed 条目本身就会 transition 到 stable，持续可选的窗口更短。
2. **选择优先级压制**：S7 baseline 的顺序是 `gd-tsp > gd-only > confirmed > provisional`。在同时有 `gd-only` 和 `tsp-confirmed` 的 898 个 step 中，591 个 step 选了 `gd-only`，307 个 step 选了 middle 层之外的 `gd-tsp`，**0 个 step 选 `tsp-confirmed`**。`tsp-confirmed` 的 463 个 selected 行基本只出现在没有 `gd-only` 可选的 step。

所以离线拟合里的 52 个 `tsp-confirmed` 样本不是随机抽取的，而是“gd-only 不可用或未被选中”条件下的选择样本；正样本率 34.6% 也不能直接和 gd-only 的 8% 横向比较。

## 2. 类别与选择上下文

### 2.1 类别 one-hot

| 类别 | 行数 |
| :--- | ---: |
| `tv` | 6,701 |
| `chair` | 3,197 |
| `bed` | 2,883 |
| `couch` | 2,669 |
| `toilet` | 582 |

### 2.2 `n_selectable`

| `n_selectable` | 行数 |
| ---: | ---: |
| 1 | 5,684 |
| 2 | 4,044 |
| 3 | 3,362 |
| 4 | 1,636 |
| 5 | 573 |
| 6 | 182 |
| 7 | 165 |
| 8 | 285 |
| 9 | 97 |
| 10 | 4 |

### 2.3 `static_rank`

| `static_rank` | 行数 |
| ---: | ---: |
| 0 | 10,699 |
| 1 | 3,666 |
| 2 | 1,161 |
| 3 | 359 |
| 4 | 107 |
| 5 | 36 |
| 6 | 4 |

### 2.4 `switch_reason`

| switch reason | 行数 | 比例 |
| :--- | ---: | ---: |
| `held_near` | 14,734 | 91.9% |
| `initial` | 712 | 4.4% |
| `switched` | 387 | 2.4% |
| `held_far` | 199 | 1.2% |

### 2.5 `static_rank` 与 `switch_reason` 口径

**`static_rank`**

- 只在 middle 层候选（`gd-only` / `tsp-confirmed`）内部排序，不是全体候选的全局排名；`gd-tsp`、`tsp-provisional`、`stable` 不在这个排名里。
- 排序 key 依次为：`selection_tier`（gd-only=1 在 tsp-confirmed=2 前）、`tsp-confirmed` 的 `sigma_band`、`gd-only` 的 `-gd_hits`、`distance_bucket`、`num_obs>=3` 标记、`-num_obs`、`stale_steps`。
- `static_rank=0` 是当前 step 的 middle 层最优候选。8,825 个 selected 行的 `static_rank` 全部为 0；另有 1,874 个 `static_rank=0` 的行未选中，是因为该 step 实际选的是 middle 层之外的 `gd-tsp`，不是 middle 排序失败。
- `rank` 是模型概率排名，本 run 模型未加载，所以全为 `null`；当前只能看 `static_rank`，不能把它解释成模型排序或全局候选排名。

**`switch_reason`**

| 值 | 定义 |
| :--- | :--- |
| `initial` | 还没有 `_last_target_coord`，或 episode 第一次锁定目标；设置当前目标 |
| `held_near` | 新选择的目标点与上一目标点距离 `<0.1 m`，视为保持同一目标 |
| `held_far` | 目标点变化在 `0.1–0.5 m`，但新候选质心距机器人 `>2 m`；为避免小位移触发频繁重规划，继续保持旧目标 |
| `switched` | 目标点变化 `>=0.5 m`，或变化 `0.1–0.5 m` 但新候选距机器人 `<=2 m`；实际切换导航目标 |

`switch_other` 是未匹配以上值的兜底 one-hot；本 run 为 0。91.9% 的 `held_near` 说明 middle 日志绝大多数是同一目标的连续快照，而不是独立的目标重选样本；离线按行统计会显著高估样本独立性，训练/验证必须按 episode 或 scene 分组。实际 shadow JSONL 没有单独的 top-level `switch_reason` 字符串，它编码在 `features.ctx.switch_initial/held_near/held_far/switched/other` 中，本表是对这些 one-hot 的汇总。

## 3. 四组特征完整性

| 特征组 | 至少一个非空 | 全部非空 | 全部为空 |
| :--- | ---: | ---: | ---: |
| `cross` | 16,032 | 575 | 0 |
| `repeat` | 16,032 | 13,906 | 0 |
| `geo` | 16,032 | 120 | 0 |
| `ctx` | 16,032 | 16,032 | 0 |

说明：`cross`/`geo` 的可选字段（TSP 近邻、`depth_ok`、`occ_passed`）按候选实际可用性留空；`ctx` 全量完整；缺失值在离线拟合中使用 missing indicator + 训练集中位数填充。

## 4. 离线拟合

### 4.1 样本定义与拟合口径

这里的“样本”**不是 shadow 的一行，也不是一个 step**，而是一个唯一目标条目：

$$
(\text{scene},\ \text{episode\_id},\ \text{memory\_id})
$$

构造规则：

1. 对每个目标条目，取其在 `middle_rank_shadow.jsonl` 中**第一次** `selected=true` 的行，记为 $t_0$；
2. 读取同一目标在 $t_0$ 之后、$t_0+K$ 步以内的 `memory_events.jsonl`；
3. 如果出现 promotion 事件，标签为 1；否则为 0：
   - `gd-only`：`upgrade` 且 `upgrade_reason=gd_only_tsp_merge`，或 `gd_only_tsp_merge` 事件；
   - `tsp-confirmed`：`gd_tsp_merge` 事件，或 `upgrade` 且 `upgrade_reason=gd_tsp`；
4. 如果 $t_0+K$ 前 episode 已经结束且没有 promotion，状态未知，记为 `censored` 并跳过；
5. 模型特征只取 $t_0$ 这一行的 `features`，不是该目标后续行的聚合。

因此正样本表示 **“这个目标在未来 10 步内被策略提升为 gd-tsp”**，不是“这个目标是 TP”，也不是“导航最终成功”。同样的 `memory_id` 无论被选中多少步，只贡献一个样本。

| 项 | 值 / 含义 |
| :--- | ---: |
| 窗口 K | 10 步 |
| 唯一目标条目总数 | 883（`gd-only` 737 + `tsp-confirmed` 146） |
| 可用样本 | 652 |
| 跳过 `never_selected` | 214 个目标条目：有过 middle 行，但从未成为选中目标 |
| 跳过 `censored` | 17 个目标条目：到 episode 结束仍未满 K 步，无法判断未来是否 promotion |
| `fit_mode` | `in_sample`：同一批 652 个样本既拟合又计算 AUC，不是 held-out |

`652 + 214 + 17 = 883`，三者正好对应全部 middle 目标条目。`never_selected` 和 `censored` 不是负样本，不能把标签补成 0。

| type | 样本（目标条目） | 正样本 | 正样本率 | AUC（in-sample） |
| :--- | ---: | ---: | ---: | ---: |
| `gd-only` | 600 | 48 | 8.00% | 0.8752 |
| `tsp-confirmed` | 52 | 18 | 34.62% | 0.8088 |

上表的 `样本` 是“可用的唯一目标条目数”；`正样本` 是其中未来 10 步内发生 promotion 的条目数。`AUC (in-sample)` 用同一批样本计算，因此通常偏乐观，不能当作在线泛化指标；尤其 `tsp-confirmed` 只有 52 个条目、18 个正样本，且其选择本身受 `gd-only` 优先级影响。

### 4.2 样本场景分布

| 场景 | `gd-only` | `tsp-confirmed` |
| :--- | ---: | ---: |
| `5cdEh9F2hJL` | 261 | 22 |
| `QaLdnwvtxbs` | 276 | 27 |
| `mL8ThkuaVTM` | 63 | 3 |

### 4.3 样本类别分布

| 类别 | `gd-only` | `tsp-confirmed` |
| :--- | ---: | ---: |
| `bed` | 175 | 5 |
| `chair` | 108 | 24 |
| `couch` | 104 | 16 |
| `toilet` | 41 | 1 |
| `tv` | 172 | 6 |

### 4.4 模型结构

| type | cross 特征数 | repeat 特征数 | geo 特征数 | ctx 特征数 |
| :--- | ---: | ---: | ---: | ---: |
| `gd-only` | 10 | 9 | 16 | 10 |
| `tsp-confirmed` | 19 | 12 | 16 | 10 |

模型格式：`vlvm_middle_tier_ranker_v1`；组合层在 log-odds 空间使用四组子分数。每个 type 单独拟合：

$$
s_g=\sigma\left(b_g+\sum_j\beta_{gj}\frac{x_{gj}-\mu_{gj}}{\sigma_{gj}}+\sum_j\gamma_{gj}\,\mathrm{missing}_{gj}\right)
$$

$$
p=\sigma\left(b+\sum_{g\in\{cross,repeat,geo,ctx\}}w_g\,\mathrm{logit}(s_g)\right)
$$

其中：

- `β` 是组内 L2 逻辑回归权重，`l2=1.0`；缺失值先按训练集中位数填充，并加 missing indicator；
- `s_g` 是四组子分数，`w` 是第二层逻辑回归权重，`l2=0.1`；
- 权重全部由 `offline_fit.py` 从样本标签拟合，不是手工指定；当前标签是“未来 10 步内 promotion”，不是 TP 或 success；
- 当前 `p` 是 type-specific 的，`gd-only` 和 `tsp-confirmed` 分别有独立模型，**两个 p 不能直接当作同一标尺比较**。

当前 `middle_tier_model.json` 的 combine 权重（in-sample，且样本少，不能当稳定解释）：

| type | cross | repeat | geo | ctx | intercept |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `gd-only` | 0.820 | 1.525 | 1.337 | -0.326 | 5.751 |
| `tsp-confirmed` | ~0 | 0.348 | 1.218 | -0.150 | 0.211 |

`tsp-confirmed` 的 `cross` 权重接近 0、样本只有 52 / 18 正样本，说明当前权重很不稳定；不能把这组系数解释成“cross 特征无效”或“geo 特征最重要”的机制结论。

| 校验项 | 结果 |
| :--- | :--- |
| `MiddleTierRanker` 加载 | `available=true` |
| 单行 `gd-only` 评分 | `s_cross=0.054686`、`s_repeat=0.073696`、`s_geo=0.027800`、`s_ctx=0.055385`、`p=0.013728` |
| 在线使用 | 未启用（`enable_middle_tier_ranking=False`） |

### 4.5 样本量与泛化检查

用同一批 `_collect_samples` 做快速分组验证，不改变当前模型文件。这里的分组验证只是可行性诊断，不是正式评测：

| type | 样本 / 正样本 | in-sample AUC | leave-one-scene-out | episode-grouped 5-fold |
| :--- | ---: | ---: | ---: | ---: |
| `gd-only` | 600 / 48 | 0.8752 | **0.7520** | **0.8000** |
| `tsp-confirmed` | 52 / 18 | 0.8088 | **0.6225** | **0.6454** |

补充：

- `gd-only` 的 leave-one-scene-out 分场景 AUC：5cd 0.833、QaL 0.729、mL8 0.614（mL8 只有 4 个正样本）。
- `tsp-confirmed` 只有 52 个样本、18 个正样本；mL8 只有 3 个样本，分场景 AUC 没有解释力。
- 两类合计只有 66 个正样本 / 64 个正 episode；`gd-only` 使用 45 个特征、`tsp-confirmed` 使用 57 个特征。按正样本数看，模型复杂度明显偏高，in-sample AUC 偏乐观。
- 结论：当前日志只能支持“`gd-only` 可能存在中等强度的离线信号”；**不足以证明 `tsp-confirmed` 可学习，更不足以证明在线 middle-tier 排序有效**。特别是 mL8 和 tsp-confirmed 的样本量不足。

### 4.6 从离线到在线的落地顺序

1. **先固定标签**：当前 `p` 是 `P(promote within K)`，不是 `P(TP)`。如果导航目标是少追错目标，应增加 TP 或 `TP & promotion` 辅助标签；用当前 TP 标签快速检查时，跨 scene 泛化仍弱，不能只靠改标签解决。
2. **离线 gate**：按 episode/scene 分组做 nested CV；至少要求 `gd-only` 每折不崩、概率校准可用、Top-1 TP 有提升。当前 `gd-only` 约 0.75–0.80，`tsp-confirmed` 约 0.62–0.65，只能先做 `gd-only` 的探索性模型。
3. **先简化模型**：45 / 57 个特征对 48 / 18 个正样本过重；先压到 `repeat + geo + class` 等少量稳定特征，减少 missing indicator 依赖，再重新做分组验证。
4. **再接在线 comparator**：当前 policy 只保存了 `_enable_middle_tier_ranking`，实际 `_get_target_object_location()` 仍使用静态 `selection_tier` key；要在线生效，必须在选目标前构造 middle snapshot、给 tier 1/2 打分，并只在 tier 1/2 内使用 `p`，保持 `gd-tsp` 优先级和 STOP gate 不变。模型缺失或异常时回退静态顺序。
5. **单独 A/B**：不要直接改主 baseline 的 `gd-only > confirmed` 顺序；用独立 arm 比较 `static` vs `model ranking`，看 success、SPL、FP、steps 和 STOP 转化。若为了补 `tsp-confirmed` 数据而强制选择它，必须记录 propensity，且不要和 baseline 样本直接合并。

## 5. 产物与边界

- 在线日志：`detections/middle_rank_shadow.jsonl`
- 拟合摘要：`middle_tier/middle_tier_fit_summary.json`
- 模型：`middle_tier/middle_tier_model.json`
- 日志字段包含 `scene`、`episode_id`、`step`、`memory_id`、`candidate_id`、`centroid`、`type`、`state`、`selected`、`n_selectable`、`distance`、`distance_bucket`、`static_rank`、四组 `features`、`s_cross/s_repeat/s_geo/s_ctx`、`p`、`rank`、`model_version`。
- 本 run 只生成 shadow；不改变候选排序、记忆、STOP 或动作。模型为 in-sample 拟合，不能直接当作在线增益证据。
- 样本不是随机候选样本：`tsp-confirmed` 只在没有 `gd-only` 可选时才会被选中；mL8 只有 66 个拟合样本（其中 tsp-confirmed 3 个），场景和类型都不平衡。
- 91.9% 的行是 `held_near`，同一目标被连续记录；按行随机划分会泄漏，后续验证必须按 episode/scene 分组。
- 如果为了补 `tsp-confirmed` 样本而交换 `gd-only > confirmed` 的优先级，这会改变动作轨迹、目标选择和 promotion 标签分布，不能再和当前 baseline 日志直接合并；应作为独立 treatment arm 或带选择概率的探索采样，并记录 propensity。

## 6. 关联文档

- S7 baseline 导航结果：[VLVM-S7baseline-exp.md](VLVM-S7baseline-exp.md)
- field 离线复算与 shadow：[VLVM-S7f0-exp.md](VLVM-S7f0-exp.md)
