# VLVM Stage 5：机制调整

> 定位：S4 已冻结 TSP3D V1 与 S-penalty v1；S5 处理十场景 SR 从 40.20% 继续提升的系统性瓶颈。60% 作为目标上界，其中约 47 集依赖第二来源复评，本轮不承诺。  
> 约束：TSP3D 仍是主模型；2.5D ValueMap、探索/导航框架不改；GD 暂不定稿。

## 1. S4 最终选择（冻结）

| 项 | 定稿 |
| :--- | :--- |
| 模型 | VLVM V1，`data/vlvm_v1/vlvm_v1.pth` |
| S-penalty | v1：`w_S = 1`（uncovered）或 `min(1, max(S,0.1)/t_S)`；`c' = c · w_S` |
| 参数 | `t_S=0.14`、`s_min=0.1`、`radius=0.5 m`、`uncovered_w=1.0` |
| 准入 | `sigma_tar=0.60`；`sigma_prop=0.45` 只决定候选池，不改变准入集 |
| 关闭项 | H penalty、S conflict gate、`floor` 搜索、P2/S-v2 主线 初步实验无优势，归档保留，不进入主线和后续调参 |
| P1 基线 | 990 集 SR 40.20%（398/990）、Oracle 54.04%、SPL 0.1586、steps 200.76 |

## 2. 当前问题与 60% 可达性

分析范围：19,412 条候选、990 集、2,289 条 memory event、9,063 条 nav goal。

### 2.1 准入漏斗

| 阶段 | 候选 | TP | precision | 说明 |
| :--- | ---: | ---: | ---: | :--- |
| pre-occ | 19,412 | 5,916 | 30.48% | 原始候选 |
| post-occ | 17,417 | 5,448 | 31.28% | TP 保留 92.09% |
| raw `c≥0.60` | 3,732 | 2,381 | 63.80% | 未加 S |
| S-penalty admitted | 2,244 | 1,638 | **72.99%** | 高置信 TP 保留 68.8% |
| episode 有准入 TP | — | — | — | 399/990（40.3%） |

S-penalty 在高置信线上剔除 743 TP / 745 FP，precision +9.19 pp。它不是主要漏斗；`sigma_tar=0.60` 与候选置信分布才是。

### 2.2 Episode 分组

| 分组 | 集数 | SR | wrong lock | wrong stop | nav goal TP 占比 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 有准入 TP、无准入 FP | 308 | 92.86% | 0.0% | 2.3% | 100.00%（3,786/3,786） |
| 有准入 TP、也有准入 FP | 91 | 65.93% | 63.7% | 12.1% | 50.03%（915/1,829） |
| 只有准入 FP | 166 | 18.07% | 100.0% | 61.4% | 0.00%（0/3,448） |
| 无任何准入候选 | 425 | 5.18% | 0.0% | 0.0% | — |

**无准入 TP 的 591 集**

| 子类 | 集数 | 当前 success | 关键事实 |
| :--- | ---: | ---: | :--- |
| 高置信 TP 被 S 拦截 | 26 | 7（26.9%） | 被拒 TP 的 S 均值 0.1107，conf 均值 0.666 |
| 有 TP，但最高 conf 在 0.5–0.6 | 146 | 24（16.4%） | conf 中位数 0.555；TP 的 S 中位数 0.205，S 未惩罚 |
| 完全无 TP | 419 | 21（5.0%） | 其中 195 集为 likely-infeasible |

146 集中有 38 集落在“只有准入 FP”组、108 集落在“无任何准入候选”组。它们不是排序问题，而是中置信正确目标无法被确认与准入。

### 2.3 导航目标层

| 事实 | 数值 |
| :--- | :--- |
| nav goal 总数 / TP | 9,063 / 4,701（51.87%） |
| 候选准入 precision | 72.99% |
| 有准入 TP 的集中，最高分准入候选为 TP | 370/399（92.7%） |
| TP+FP 共存组：TP 分数更高（62 集） | wrong lock 64.5%，SR 64.5% |
| TP+FP 共存组：FP 分数更高（29 集） | wrong lock 62.1%，SR 69.0% |

导航选择当前是**纯最近质心**（`argmin(dists_2d)`），`quality_score` 只记录、不参与选择。因此候选层的 73% precision 在目标层退化为 51.87%；FP 是否存在于记忆里，比 TP/FP 的分数高低更能决定结果。

注：2.2/2.5/2.6 中的“TP+FP 共存组”指该 episode 同时有准入 TP 与准入 FP，与 GD 的 `both` 投票无关。P1 中 `gdp_enable=False`，memory `src` 全部为 `tsp3d`，不存在 `both` 投票事件。

### 2.4 fallback 与 GD 现状

| 事件 | 数量 | 说明 |
| :--- | ---: | :--- |
| memory new | 803 | 其中 795 条创建即 suspicious |
| memory merge | 1,441 | suspicion 清除 0 次 |
| memory delete | 45 | 全部 `hysteresis_exceeded`，全部为 suspicious |
| memory src | 2,289 | 全部 `tsp3d`（GD 关闭） |

fallback 目前只在近场连续 miss 时删除，删除量仅 45/803；GD 关闭导致没有任何第二票，suspicious 永远不会被清除。**当前 fallback 对持续 FP 轨道几乎不起作用，GD 也没有参与确认。**

### 2.5 硬上限与分项预算

有 post-occ TP 的 episode 为 571 集，无 TP 但当前成功 21 集；即使 571 集全部成功，SR 也只有 **592/990 = 59.8%**。因此 60% 必须扩大检测覆盖，不能只靠准入、排序或 FP 治理。

| 工作流 | 可回收集 | 当前 success | 目标 success | 增量 | 目标假设 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 目标选择质量化（TP+FP 共存组） | 91 | 60 | 85 | +25 | 达到 93% |
| 中置信 TP 确认 | 146 | 24 | 124 | +100 | 达到 85% |
| S 阻断 TP 确认 | 26 | 7 | 21 | +14 | 达到 81% |
| 干净 TP 转化 | 308 | 286 | 296 | +10 | 达到 96% |
| 前四类小计 | 571 | 377 | 526 | +149 | — |
| 本轮定稿（含无 TP 现状 21） | 990 | 398 | 547 | +149 | **SR 55.3%** |
| 第二来源召回（GD，待重新评估，不计入本轮） | 419（195 不可行） | 21 | 68 | +47 | 仅作 60% 待定项 |
| 若第二来源通过 | 990 | 398 | 594 | +196 | 60.0%（待定） |

- 本轮定稿上限为 **55.3%**；60% 依赖第二来源复评，不再作为本轮承诺。
- 前四项必须同时成立；任何单独一项都不足以把 SR 推到 55% 以上。
- 若不改目标选择，候选层 precision 无法传导到导航层，TP+FP 共存组的 25 集和仅 FP 组的恢复都无从兑现。

### 2.6 问题 → 方向

| 问题 | 关键证据 | 方向 | 阶段 |
| :--- | :--- | :--- | :--- |
| 1. 目标选择忽略质量 | nav goal TP 仅 51.87%；共存组中 SR 为 64.5%（TP 高分）/ 69.0%（FP 高分），无方向性差异 | `_get_target_object_location` 从 `argmin(dists_2d)` 改为按选定 $\sigma_c$、`num_obs`、距离排序；两态只决定删除，不参与选择 | S4 |
| 2. 中置信 TP 未准入 | 146 集，TP conf 0.5–0.6，S 中位 0.205，当前成功 24 | provisional：$0.55\le\sigma_c<0.60$、$1.5\le dist\le4.0$ m、不看 $S$ | S2 |
| 3. FP 轨道污染与停止无确认 | 166 集 FP-only，wrong lock 100%；91 集共存；120 集 wrong stop；22 集候选全干净仍 FP | provisional 提供待验证档，但 STOP 本轮不改 | S2 |
| 4. S 拦截少量高置信 TP | 26 集，当前成功 7 | provisional 不按 $S$ 判态，$S$ 只作用于准入 | S2 |
| 5. 无 TP 覆盖 | 419 集（195 集不可行），当前成功 21 | 第二来源召回方案待重新评估；本轮不定稿，60% 缺口暂挂 | — |
| 6. occ 误删 `out_of_bounds` | 393 TP / 710 FP；17 集 TP 被删光 | `out_of_bounds` → `geom_unknown`；硬几何保留 reject | S1 |

### 2.7 明确不做

- 不再调 S-penalty 权重、`t_S`、`sigma_tar`，不恢复 v2。
- 不通过全局降低准入线换召回。
- 不让 GD 分数进入 TSP3D 置信度或 S 融合公式。

## 3. 机制调整

### 3.1 occ 拆分（S1）

| occ 原因 | n | TP | FP | 处理 |
| :--- | ---: | ---: | ---: | :--- |
| `out_of_bounds` | 1,103 | 393 | 710 | `geom_unknown`：保留候选；$\sigma_c>0.60$ 时进入 `confirmed` |
| `insufficient_occ_voxels` | 815 | 70 | 745 | 保留 reject |
| `air_penetration` | 52 | 3 | 49 | 保留 reject |
| `low_occupancy_ratio` | 25 | 2 | 23 | 保留 reject |

- 现状：统一 reject 损失 468 TP；17 集 TP 被删光，全部由 `out_of_bounds` 造成。
- 调整：状态拆为 `geom_reject` 与 `geom_unknown`；前者继续拦截，后者在 occ 阶段只打标不拦截，后续仍受 s-penalty 准入约束。
- `geom_unknown` 保留 `geom_status` 标签；S1 阶段不设 provisional 上限：$\sigma_c>0.60$ 直接进入 `confirmed`，$\sigma_c\le0.60$ 不写入 memory。
- unknown 不再额外要求 covered；uncovered 按 S-penalty 的 `uncovered_w=1` 处理，与 S1 口径一致。
- 本轮 fallback 只保留 `confirmed` / `stable` 两态；状态只影响 delete，不影响停止。
- 日志保留 `occ_reason`，新增 `geom_status`；验收看 TP 被删 episode、FP 增量、SR。

### 3.2 fallback 三态（S2 重启，S3 跳过）

现状：803 new 中 795 suspicious；1,441 merge 中 0 次清除；45 delete（30 TP / 15 FP）。问题不是阈值不够低，而是 `suspicious` 同时承担“低置信”“远距离”“待验证”三种语义。

三态初始在 new 时确定；S2 之后只降不升，S3 允许 merge 重判。`stable` 只能由 `confirmed` / `provisional` 降转产生，不能在 new 直接创建。`s_min=0.1` 只约束 S-penalty 准入，不参与 `provisional` 的 new 判定。

写入顺序：每个检测先做 merge 判定（同类最近质心距离 `<0.5 m`）；不满足才 new 并按 new 条件定状态。

**S1 / S2 / S3 的差异**

| 维度 | S1 | S2 | S3 |
| :--- | :--- | :--- | :--- |
| 改动层 | 仅 occ 准入：`out_of_bounds` → `geom_unknown`，硬几何仍 reject | new 状态判定 + 准入/删除规则 | 仅 merge 状态判定；new 与 S2 相同 |
| new 状态 | 无三态；unknown 沿用原 fallback（suspicious / non-suspicious） | 引入三态：$\sigma_c\ge0.60$ 为 `confirmed`；$0.55\le\sigma_c<0.60$ 且 $1.5\le dist\le4.0$ m 为 `provisional`，不看 $S$；`geom_unknown` 最高 `provisional` | 与 S2 相同 |
| `geom_unknown` | 保留候选，无状态上限 | 需满足 provisional 条件且 $\sigma_c\ge0.55$；最高 `provisional` | 同 S2；merge 重判仍封顶 `provisional` |
| 准入边界 | 仍 $\sigma_c\ge0.60$ 的 memory gate | 0.55–0.60 的 `provisional` 需新增 0.55 gate，或把准入下限降到 0.55 | 与 S2 相同 |
| 状态迁移 | 无三态迁移；原 binary 逻辑 | 只降不升 | 可升可降 |
| 升级触发 | 无三态升级 | 无 | merge 重判达标，且连续 2 帧一致 |
| 降级触发 | 无三态降级；原 fallback 只在 `near_miss` 删除 | 仅 `near_miss` 达标 | `near_miss` 达标，或 merge 重判为 `stable` |
| merge 对状态 | 不改 | 不改状态 | 按选定口径重判 |
| 分数用途 | 无 5 列记录；`quality_score` 仍为历史 max | 记录 max / 最新 / EMA×3 共 5 列；new 用当前帧 $\sigma_c$ 定态，存储口径不参与行为 | 选定 1 列参与 merge 重判 |
| `stable` 回升 | 无 `stable` 概念 | 不允许 | 允许；`geom_unknown` 最高只到 `provisional` |
| 删除规则 | 原 binary：suspicious 2 次 / 非 suspicious 5 次删除 | `confirmed` 3 次 / `provisional` 2 次 → `stable`；`stable` 2 次删除 | 与 S2 相同 |
| 导航影响 | 不变 | 无；状态只影响 delete | 无新增；目标选择留给 S4 |
| 目的 | 恢复 occ 误删的 unknown 候选 | 验证三态与删除阈值，离线选定口径 | 验证 merge 重判是否值得 |

注：S1 实测仅改 occ 准入，未启用三态；`geom_unknown` 接入 `provisional` 及状态上限在 S2 生效。

| 状态 | new 条件 | 来源 |
| :--- | :--- | :--- |
| `confirmed` | $\sigma_c\ge0.60$ 且非 `geom_unknown` | new |
| `provisional` | $0.55\le\sigma_c<0.60$ 且 $1.5\ \mathrm{m}\le dist\le4.0\ \mathrm{m}$，不看 $S$；`geom_unknown` 同条件且 $\sigma_c\ge0.55$ | new |
| `stable` | — | **仅由 confirmed / provisional 降转** |

`geom_unknown` 不是绕过 $\sigma_c$ 的独立入口：进入 `provisional` 仍要求 $\sigma_c\ge0.55$ 且 $1.5\le dist\le4.0$ m；$\sigma_c<0.55$、$dist<1.5$ m、$dist>4.0$ m 均不进入 `provisional`。`provisional` 不看 $S$；S-penalty 只作用于 $\sigma_c$ 计算与准入，不参与 new 状态判定。它保留 `geom_status` 标签，但只改变状态上限（最高 `provisional`）。停止校验不读该标签，后续只按生命周期状态执行。

**merge（复用当前行为）**

merge 门槛：同类轨道质心距离 `< merge_dist_thresh = 0.5 m`；否则新建轨道。

| 字段 | merge 行为 |
| :--- | :--- |
| 质心 | `centroid_new = 0.8 × centroid_old + 0.2 × centroid_detection`（`ema_weight_old=0.8`） |
| `num_obs` | `+1` |
| 置信度 | S2：同时记录 max / 最新 / EMA，行为不依赖；S3：使用 S2 选定口径 |
| `near_miss` | 清零 |
| 生命周期状态 | S2 不变；S3 允许 merge 按选定分数重判（见 S1–S3 差异表） |
| `src` | 按 `merge_src_tag(old, new)` 合并；GD 关闭时仍为 `tsp3d` |
| surface | 不更新，保留首次写入的 near-surface 点 |
| `last_candidate_id` | 更新为本次候选 |

- S2 在 new 之后不按存储口径升级/降级，也不从 stable 回升；S3 例外。
- S2 中 max / 最新 / EMA×3 共 5 列分数只记录，不改变 fallback 与导航行为；离线比较后再选定口径。
- S3 用选定口径做 merge 重判；这是唯一允许状态升级的阶段。

**S3 的 merge 重判规则**

| 用于重判的 $\sigma_c$（S2 选定口径） | 重判状态 |
| :--- | :--- |
| $\sigma_c\ge0.60$ 且非 `geom_unknown` | `confirmed` |
| $0.55\le\sigma_c<0.60$ 且 $1.5\le dist\le4.0$ m（不看 $S$）；`geom_unknown` 同条件且 $\sigma_c\ge0.55$ | `provisional` |
| 其他 | `stable`（仅当当前为 confirmed/provisional） |

- 连续 2 帧证据一致才生效，避免单帧抖动。
- S3 是本轮唯一允许状态升级的阶段；S2 仍然只降不升。
- `geom_unknown` 即使 $\sigma_c\ge0.60$ 也只升到 `provisional`；该上限在 merge 重判中同样生效。
- 若 S2 选定 max（单调不减），S3 实际只升不降；此时跳过 S3，或规定状态重判固定用最新/EMA，max 只供 S4。

**delete：只看 near_miss，只有 stable 会删除**

| 当前状态 | near_miss 触发 | 动作 |
| :--- | ---: | :--- |
| `confirmed` | ≥3 | → `stable`，near_miss 清零 |
| `provisional` | ≥2 | → `stable`，near_miss 清零 |
| `stable` | ≥2 | **delete** |
| `stable` | <2 | 保持 stable |

- `confirmed` / `provisional` 永不直接 delete；只有 `stable` 会 delete。
- near_miss 只在近场 FOV 锥内累加；当前 nav goal 0.5 m 内豁免；merge 会清零。
- `fb_suspicious_conf=0.75` 退出定稿：它高于 $\sigma_c$ 的 p99=0.7044，几乎把全部轨道判成 suspicious；新状态不再由单一 confidence 阈值决定。
- 本轮 STOP 不改，也不读取状态；停止校验与 GD 一起留到最后。
- 验收：三态分布不再出现 99% suspicious；delete 中 TP 不增加；confirmed/provisional 无直接删除。

### 3.2.1 状态机、$\sigma_c$ 记录与目标选择（S4-1 / S2 实测）

三项机制已按实测冻结，不再作为待选方案；S3 跳过，S4-2 转为离线定列。

| 项 | 落地值 |
| :--- | :--- |
| S4-1 状态 | 仅 `confirmed` / `stable`；new $\sigma_c\ge0.60$ → `confirmed`（含 `geom_unknown`） |
| S2 (after S4) 状态 | `confirmed`：$\sigma_c\ge0.60$ 且非 `geom_unknown`；`provisional`：$0.55\le\sigma_c<0.60$ 且 $1.5\le dist\le4.0$ m，不看 $S$ |
| merge | 先判 merge（同类质心 `<0.5 m`）；命中只更新质心、`num_obs`、五列 $\sigma_c$、清 `near_miss`，不改状态 |
| 降转 / 删除 | `confirmed` 3 次 / `provisional` 2 次 `near_miss` → `stable`；`stable` 2 次 → delete |
| STOP | 不读状态，仍为 0.90 m |
| $\sigma_c$ 记录 | `sigma_c_max` / `latest` / `ema_03` / `ema_05` / `ema_08` 五列随 new / merge 落盘 |
| 目标选择 | S4-B：`num_obs≥2` 优先 → $\sigma_c$ 档（≥0.70 / 0.65–0.70 / 0.60–0.65 / 0.55–0.60）→ 档内最近；不读状态；latch 0.1 / 0.5 m |

| 指标 | S4-1 | S2 (after S4) |
| :--- | ---: | ---: |
| new（TP / precision） | 232（132 / 56.90%） | 312（161 / 51.60%） |
| new 状态构成 | `confirmed` 232 | `confirmed` 163（89 / 54.60%）、`provisional` 149（72 / 48.32%） |
| merge（TP / precision） | 360（275 / 76.39%） | 1,065（695 / 65.26%） |
| 降转 | `confirmed→stable` 14 | `provisional→stable` 19、`confirmed→stable` 6 |
| delete（TP / FP） | 6（4 / 2） | 7（2 / 5） |
| 直接删除 confirmed / provisional | 0 | 0 |
| 每集记录数均值 | 0.78 | 1.03 |
| nav goal TP 占比 | 49.91% | 48.09% |
| SR | 38.38% | 40.74% |

| nav goal TP 占比按 $\sigma_c$ 档 | S4-1 | S2 |
| :--- | ---: | ---: |
| ≥0.70 | 24.93% | 38.60% |
| 0.65–0.70 | 76.06% | 78.31% |
| 0.60–0.65 | 45.50% | 59.20% |
| 0.55–0.60 | — | 34.12% |

| 五列区分度（merge 事件，TP vs FP 的 AUC） | S4-1（n=360） | S2（n=1,065） |
| :--- | ---: | ---: |
| `sigma_c_max` | 0.516 | 0.642 |
| `sigma_c_latest` | 0.527 | 0.606 |
| `sigma_c_ema_03` | 0.514 | 0.587 |
| `sigma_c_ema_05` | 0.517 | 0.597 |
| `sigma_c_ema_08` | 0.522 | 0.604 |
| 候选级 $\sigma_c$（参照） | 0.782 | 0.802 |

- 五列在已准入记录上的区分度接近随机（S4-1 ≤0.527），候选级 0.78–0.80，差距来自 0.60 / 0.55 截断后的范围收缩；列选择改由离线在放宽后的分布上定，S4-2 不再单独跑导航。
- 落盘与回放：两轮五列缺失均为 0，回放违例 0。
- 选择层自由度：每集 0–1 条记录的集数占 S4-1 84.8%（119 + 133 集）、S2 75.4%（99 + 125 集）；S4-B 在这些集上与纯最近邻等价。
- 选择层效果：S4-1 对 S1 配对 114 vs 115（仅 S4-1 成功 1 集、仅 S1 成功 2 集）；S2 对 S4-1 为 121 vs 114，且同时改变准入与状态，不单独归因选择。
- 边界：$\sigma_c$ 排序只影响选中哪条记录，不改变 new 准入线、不改 `near_miss` / `stable` 规则、不改 0.90 m STOP。

### 3.3 GD 状态

GD 暂不进入本轮定稿，fallback 和 $\sigma_c$ 都不依赖 `both` / `gd`。GD 的触发方式、确认语义与是否保留需重新评估后另开实验，不在本文展开。

### 3.4 provisional 准入、升级与删除（S2U / S5）

顺序：S2U → 标定 2 → S5（A/B 同轮）→ 标定 3 → S6（merge + `stable` 池，新 baseline）→ S7（目标选择、停止门槛、视角扩张复核）→ S8（$\sigma_{prop}$ grid）→ GD（最后）。完整阶段表见文末「实验顺序与判定门」。

**3.4.1 S2U**

| 项 | 值 |
| :--- | :--- |
| 规模 | 3 场景 × 99 = 297 集 |
| 与 S2 相同 | new 条件 $0.55\le\sigma_c<0.60$、$1.5\le dist\le4.0$ m、不看 $S$；STOP 仍为 0.9 m 且不读状态；`confirmed` 3 / `provisional` 2 near_miss → `stable`、`stable` 2 → delete；nav goal 0.5 m 豁免照旧 |
| 与 S2 差异 1 | 运行时升级：merge 后 `num_obs≥2` 且 `detect_dist≤2.5 m` → `provisional → confirmed`，单独落 `upgrade` 事件 |
| 与 S2 差异 2 | 仪表化：`track_frames.jsonl` 记录每检测帧的位姿、距离、方位角、`in_fov`、`depth_ok`、`matched`、`num_obs`、状态与 `tp`；memory / nav goal 增加 `robot_xy`、`robot_yaw`、`detect_dist` |
| 不做的事 | 不禁止 provisional 停止；不改 `near_miss` 阈值；不改 lock 豁免 |
| 用途 | ① 单独测“升级”这一个变量；② 为标定 2 提供位姿与可见性数据，决定 S5 的两个开关 |

S2U 要回答的三个问题（直接决定 S5 口径）：

| 问题 | 判据 |
| :--- | :--- |
| 升级轨道能否停对 | `upgrade` 事件的 TP 占比、升级轨道触发的停止是否落在目标上 |
| provisional 是否保留“可直接停” | 升级轨道命中率明显高于直接停、且 FP 未同步下降 → S5 改为“未升级不可停”；否则保留直接停 |
| lock 豁免取消的范围 | 对比被豁免（nav goal 0.5 m 内）与未豁免轨道的 TP/FP 与 `near_miss` 计数：TP 大量落在豁免区 → 只取消 provisional 的豁免；TP 占比低 → 全部取消 |

**3.4.2 S5**

| 项 | 口径 |
| :--- | :--- |
| provisional 准入 | $0.55\le\sigma_c<0.60$，距离固定 $1.0\le dist\le4.0$ m（下界由 1.5 降到 1.0） |
| 状态升级 | `provisional → confirmed`：`num_obs≥2` 且质心距离 $\le4.0$ m；判定从“仅 merge 时”改为**每帧检查**，距离统一用质心 |
| 可停资格 | **只有 `confirmed` 可触发 STOP**；未升级的 `provisional` 只可追，接近到 1.2 m 仍未升级 → 转确认动作，超过 3 帧仍不满足则对该轨道拉黑并放弃 |
| 删除规则 | `provisional` 3 次 `near_miss` **直接 delete**；**由 provisional 升级而来的 `confirmed` 同规则**（保留 `origin=provisional` 标记）；原生 `confirmed` 仍为 3 → `stable`、`stable` 2 → delete |
| lock 豁免 | 只取消 `provisional` 与 `origin=provisional` 的 0.5 m 豁免；原生 `confirmed` 保留 |
| 日志要求 | `upgrade` 事件、`origin` 标记、delete 事件的 `from_state` 与触发计数、停止门槛拦截次数 |
| 验收 | FP ≤79；SR ≥40.74%（A 臂不低于 S2U）；B 臂看新增低分档的回收与 FP 代价 |

两个臂同轮一起测，不再逐项拆开验证：

| 臂 | provisional 准入 | 停止资格 | 目的 |
| :--- | :--- | :--- | :--- |
| A | 0.55 / 1.0–4.0 | 仅 confirmed 可停 | S5 基线 |
| B | 0.50 / 1.0–4.0 | 仅 confirmed 可停 | 测低分档的回收与 FP 代价 |

S5 实测（三场景 × 99，详见 `VLVM-S5s5-Exp.md`）：

| 指标 | 臂 A | 臂 B | 验收门 |
| :--- | ---: | ---: | :--- |
| success / SR | 121 / 40.74% | 116 / 39.06% | A 达成；B 未达 |
| FP | 72 | 92 | A 达成（≤79）；B 超限 |
| FN | 19 | 13 | — |
| never_saw（feasible / infeasible） | 21 / 64 | 18 / 58 | — |
| SPL / steps | 0.1797 / 156.6 | 0.1788 / 149.2 | — |
| 升级事件（TP / precision） | 156（83 / 53.21%） | 249（102 / 40.96%） | — |
| 停门槛（允许 / 拦截） | 186 / 88 | 187 / 212 | — |
| delete（TP） | 68（28） | 151（58） | — |

### 3.5 S6：merge 条件与 `stable` 删除池

基准 = S5 臂 B（$0.50\le\sigma_c<0.60$、$1.0\le dist\le4.0$ m、仅 `confirmed` 可停、STOP 0.90 m）。

只改 merge 与 fallback 的淘汰/复活路径，`stable` 退出目标选择；`provisional` 可以进入目标选择，但不作为目标停止。

**取代关系**

| 原条款 | 位置 | S6 起 |
| :--- | :--- | :--- |
| merge 命中只更新质心 / `num_obs` / 五列分数 / 清 `near_miss` | §3.2.1 | 增加新信息门；禁止 merge 到 `stable` |
| merge 内触发 `close_reobservation` 升级 | §3.2.1 / §3.4.2 | 取消；升级只由每帧检查 `num_obs`（新信息计数）与质心距离判定 |
| `provisional` 3 次 `near_miss` 直接 delete | §3.4.2 | 改为 3 次 `near_miss` → `stable`，与 `confirmed` 同阈值 |
| `stable` 2 次 `near_miss` → delete | §3.2.1 | 改为三条件取或：陈旧 ≥60 步 / `stable` 内连续 miss ≥3 / 累计 miss ≥6（复活不清零） |
| `stable` 与目标选择的关系 | 未定义 | 明确禁止：`stable` 不进 nav goal 候选、不可停 |

**merge 规则**

| 编号 | 规则 | 参数 |
| :--- | :--- | :--- |
| M1 | 不许 merge 到 `stable`；命中 `stable` 时先判复活 | `revive_radius=0.5 m`、`revive_window=10 步`、`revive_min_sigma=0.50` |
| M2 | merge 需带新信息才计入 `num_obs` 与分数 | 位姿基线 ≥0.2 m，或朝向差 ≥10°，或间隔 ≥3 步 |
| M3 | merge 只累加证据，不改生命周期状态 | `enable_close_reobservation_upgrade=false` |
| M4 | merge 半径保持 0.5 m | `merge_dist_thresh=0.5` |

不满足 M2 时仍更新 EMA 质心与 `last_seen_step`，不加 `num_obs`、不更新 `sigma_c_max`。

**`stable` 池**

| 迁移 | 触发 | 参数 |
| :--- | :--- | :--- |
| `provisional` → `stable` | 近场累计 3 次未匹配 | `provisional_to_stable_misses=3` |
| `confirmed` → `stable` | 近场累计 3 次未匹配 | `confirmed_to_stable_misses=3` |
| `stable` → `provisional` | 新候选在 0.5 m 内、距上次观测 ≤10 步、$\sigma_c\ge0.50$ | 复活，保留 `num_obs` 与分数 |
| 直接 delete | 不再存在 | `provisional_direct_delete=false` |

`stable` 的三个删除条件取或，任一命中即删：

| 条件 | 参数 | 覆盖场景 |
| :--- | :--- | :--- |
| 陈旧 | 距上次观测 ≥`stable_stale_purge=60` 步 | 离开近场后再不出现 |
| 近场反复看不到 | `stable` 内连续 `near_miss` ≥`stable_miss_delete=3` | 仍在近场但持续无匹配 |
| 累计无证据停留 | 累计 `near_miss` ≥`stable_total_miss=6`，复活不清零 | 反复进出 `stable` 但不产生新证据 |

不按复活次数设上限：反复被看到的条目应当继续复活并累积证据，被限制的只能是“没有证据的停留”。

三个条件的数据支撑强度不同：

| 条件 | 支撑强度 | 数据 |
| :--- | :--- | :--- |
| 陈旧 ≥60 步 | 直接 | 存活到集末且 ≥60 步未观测 32 条（TP 3 / FP 29，臂 B） |
| `stable` 内连续 miss ≥3 | 弱 | `stable` 态条目仅臂 A 8 / 臂 B 7 / S2U 22 条；近场帧匹配率 9.8% / 12.1% / 10.7%，最大连续 miss 达 4 的占比最高 → 阈值 2 与 3 只差 1 帧 |
| 累计 miss ≥6 | 设计性 | 再观测次数分布（0 / 1 / 2 / ≥3）= 96 / 17 / 10 / 28，≥3 档 TP 20 / 28（71%）→ 这组必须保留，故预算改按 miss 总量计，不按复活次数计 |

| `stable` 内约定 | 值 |
| :--- | :--- |
| 参与目标选择 / nav goal | 否 |
| 可触发 STOP | 否 |
| 吸收 merge 证据 | 否（只走复活） |
| 删除入口 | 只在这一层：`stable_stale_purge` / `stable_miss_delete` / `stable_total_miss` |

进 `stable` 已等价于旧 delete 的淘汰强度，`confirmed` 与 `provisional` 的淘汰路径完全一致，只差停止资格；`provisional` 不再有更短的删除预算。

**数据依据**
| 项 | 数值 |
| :--- | ---: |
| merge 写入 `stable` | 17 次（TP 6） |
| 升级事件（TP / precision） | 249（102 / 40.96%） |
| 升级位姿基线 <0.3 m | 194（TP 72 / 37.1%） |
| 升级位姿基线 ≥0.5 m | 44（TP 28 / 63.6%） |
| `provisional_miss_delete` | 145（TP 58） |
| 其中删除后 10 步内同物体再现 | 34（TP 24 / FP 10） |
| 其中删除后 20 步 / 1.0 m 内再观测 ≥3 次 | 28（TP 20 / FP 8） |
| 存活到集末且 ≥60 步未观测 | 32（TP 3 / FP 29） |

**验收**
| 指标 | 目标 | 参照（臂 B） |
| :--- | :--- | ---: |
| 升级 precision | ≥50% | 40.96% |
| TP 升级数 | ≥85 | 102 |
| confirmed 池 FP | ≤170 | 208 |
| delete 总量 | ≤120 | 151 |
| success | ≥116 | 116 |
| never_saw 合计 | ≤76 | 76 |

### 3.6 S7：目标选择、停止门槛与视角扩张复核

在 S6 基线上改三处；准入、merge、`stable` 池口径全部冻结。

**目标选择**

| 层级 | 判据 |
| :--- | :--- |
| 1 硬过滤 | 跳过 `stable`、`stop_blocked` 未复核、距上次观测 >30 步 |
| 2 主排序 | evidence score 分层 |
| 3 次排序 | 新鲜度（距上次观测步数，越到导航后期权重越高） |
| 4 次排序 | 质心距离 |
| 5 末位 | $\sigma_c$，仅同级 tiebreak |

$$\text{ev}=\log(1+n_{\text{obs}})\cdot\left(1+\min\!\left(\frac{b}{0.5},1\right)\right)\cdot\left(1+\min\!\left(\frac{\Delta\psi}{30°},1\right)\right)-0.05\,s_{\text{stale}}$$

$n_{\text{obs}}$ 为带新信息的观测次数，$b$ 为观测位姿基线，$\Delta\psi$ 为朝向跨度，$s_{\text{stale}}$ 为距上次观测步数。

**停止判据**

| 项 | 口径 |
| :--- | :--- |
| 资格 | `confirmed` |
| 证据 | $n_{\text{obs}}\ge3$ |
| 时点 | 本帧重确认且 $\rho\le1.2$ m |
| 未达 | 进入视角扩张复核，不再直接左转 + 黑名单 |

**视角扩张复核（原 3.4.3）**

| 项 | 口径 |
| :--- | :--- |
| 触发 | $\rho\le1.2$ m 且停止判据未达 |
| 动作 | 原地转 40° → 后退 0.4 m → 重新观测，≤5 步 |
| 通过 | 复核后 evidence 达标 → 允许停 |
| 不通过 | 打 `stop_blocked` 标签 → `stable`，回探索；不做永久黑名单 |
| 预算 | 每集 ≤4 次 |
| 日志 | `[EXPAND]`：`memory_id`、前后 evidence、基线、$\sigma_c$、结果 |
| 与旧口径差异 | 不再以“目标进入 0.3 m 或 STOP 首次满足”为触发；不做多视图三角化删除 |

**数据依据**

| 项 | 数值 |
| :--- | ---: |
| 末端目标 $\sigma_c$ 中位数（TP / FP，臂 B） | 0.620 / 0.603 |
| 末端目标 `num_obs` 中位数（TP / FP，臂 B） | 4 / 2 |
| `num_obs≥3` 停门槛回放精度（臂 A / 臂 B） | 67.2% / 65.5% |
| 同上 TP 覆盖 | 66.7% / 66.1% |
| 停门槛拦截次数（臂 A / 臂 B） | 88 / 212 |
| 有停止记录的集数 / 每集拦截均值 | 208 / 1.9 |
| 首目标来源 → success（臂 B：`new_confirmed` / `upgraded` / `provisional`） | 61.4% / 52.6% / 42.9% |

**验收**

| 指标 | 目标 | 参照（臂 B） |
| :--- | :--- | ---: |
| FP | ≤80 | 92 |
| success | ≥121 | 116 |
| never_saw 合计 | ≤80 | 76 |
| steps | ≤155 | 149.2 |

**据 S6 结果调整**

| S6 结果 | S7 调整 |
| :--- | :--- |
| TP 升级 <85 | M2 放宽为“基线 ≥0.15 m 或间隔 ≥2 步” |
| confirmed 池 FP >170 | 停止判据加“近距重确认 ≤1.5 m”硬项 |
| 复活精度 <60% | `revive_radius` 0.4 m、`revive_window` 6 步 |
| 陈旧清除中 TP 占比 >20% | `stable_stale_purge` 60 → 90 步 |
| `stable` 内 miss 删除占比 >50% | `stable_miss_delete` 3 → 4，并核对 `stable` 态帧的匹配率是否仍 <15% |
| `stop_blocked` 集数 >150 | 扩张预算 4 → 6，并允许 `provisional` 触发 |

### 3.7 $\sigma_{prop}$ grid search（S8，最后单独一轮）

> S5 定稿时取消独立 S8：`sigma_prop=0.45` 固定。下表保留为历史方案，不再执行。

| 项 | 口径 |
| :--- | :--- |
| 位置 | S7 之后、GD 之前 |
| 变量 | 只改 `sigma_prop`（服务端候选池下限）；准入、状态、停止规则全部冻结在 S5 定稿 |
| 网格 | 待定；日志里 `conf` 最小值正好是 0.450，只能证实当前截断位置，池外无任何数据 |
| 目的 | 只测候选池扩容，不改准入集与行为阈值 |
| 观测 | 新增候选的 TP 占比、无 TP 集的覆盖增量、FP 增量、服务端返回条数 |
| 约束 | 单独一轮；不与任何门槛改动同跑，否则无法区分“池子变大”与“门槛降低” |

### 3.8 离线标定（独立于在线实验）

历史离线标定共三轮，都不启动仿真；输入都是已完成运行的 `detections/` 日志，输出单独成文。最终四场景运行后补充第 4 轮离线标定，结果统一整理在 [VLVM-S5final.md](VLVM-S5final.md) §8，不再写入 `VLVM-S5cal-Exp.md`。

| 轮次 | 时机 | 输入 | 标定对象 | 输出 |
| :--- | :--- | :--- | :--- | :--- |
| 标定 1 | 在线运行 2 之前 | S2 297 集日志 | 停止决策点的可用判据、升级规则、删除阈值代理、准入扩张代价 | 停止规则基线与候选阈值；决定 3.4.1 升级条件 |
| 标定 2 | 在线运行 2 之后 | S2U 297 集日志（含逐步轨迹记录与位姿） | `near_miss` 阈值复核、升级条件复核、豁免影响 | 3.4.2 的删除与豁免口径（`VLVM-S5cal-Exp.md` §1–6） |
| 标定 3 | S5 A/B 之后 | 臂 A / 臂 B 297 集日志 | 升级路径与视角基线、删除阈值、停门槛、准入口径 | 3.5 与 3.6 的参数（`VLVM-S5cal-Exp.md` §7） |

标定 1 的能力边界：

| 能做 | 不能做 |
| :--- | :--- |
| 用最后一条 nav goal 作为停止决策点，评估“若该轨道不满足某判据就不许停”的 TP 保留 / FP 规避数量 | nav goal 只记录决策事件（每集中位 14 行，约占步数 15%），无法逐步回放“换个时机停” |
| 轨道级升级规则的 precision / recall（`num_obs`、最近距离、$\sigma_c$ 上限） | 不能模拟停被抑制后继续探索产生的新轨迹与新候选 |
| 用“该帧有检测但 0.5 m 内无匹配”作未匹配代理，给出连续 2 / 3 次的 TP-FP 分布 | 已证否：该代理在 TP 上反而更常见（≥3 帧 66.3% vs 51.0%），不能用于定初值 |

标定 2 的能力边界：

| 能做 | 不能做 |
| :--- | :--- |
| 用逐步 `track_frames` 记录（位姿、方位角、FOV、距离、是否匹配）复算近场 miss | 已复算：FOV+深度+未匹配在停止目标上 FP/TP ≈ 0.85；仅 `near_miss`（≤2.5 m且仅 provisional）为 2.2 : 1；遮挡仍未建模 |
| 用同一批轨迹比较视角扩张的触发距离（3.0 / 2.5 / 2.0 m）与确认通过率 | 不改变 STOP 判定，结论只用于设阈值 |

顺序约束：标定 1 的结论先于在线运行 2 落地；标定 2 已定删除与豁免口径；标定 3 结论用于定 3.5 与 3.6 的参数，先于 S6、S7 落地；GD 始终排在最后一步，不参与前两轮标定。

## 4. S5 最终机制（定稿）

> 本节是 S5 的最终口径。S7 的 evidence 选择、`num_obs≥3` 停止门槛、当前帧重确认与视角扩张复核均不进入最终机制；S8 不再单开阶段，`sigma_prop=0.45` 固定。GD proposer 保留在代码中，但本节基线不启用，转入 [VLVM-S6.md](VLVM-S6.md) 单独调整。

### 4.1 机制总表

| 层 | 最终机制 |
| :--- | :--- |
| 候选准入 | `sigma_tar=0.60`、`sigma_prop=0.45`；`c'=c·w_S`；`t_S=0.14`、`s_min=0.07`、`radius=0.5 m`、`uncovered_w=1.0` |
| 几何 | `insufficient_occ_voxels` / `air_penetration` / `low_occupancy_ratio` 继续 reject；`out_of_bounds` → `geom_unknown`，最高只到 `provisional` |
| merge | 同类质心 `<0.5 m`；新信息门 `≥0.2 m` 或 `≥10°` 或 `≥3 步`；不允许 merge 到 `stable` |
| provisional | `0.50 ≤ sigma_c < 0.60`，`1.0 ≤ dist ≤ 4.0 m` |
| stable | `confirmed/provisional` 各 3 次 near_miss → `stable`；`stable` 删除条件取或：`stale≥60` / 连续 miss≥5 / 累计 miss≥8；复活 `0.5 m / 10 步 / sigma_c≥0.50` |
| 升级 | `close_reobservation` / `frame_upgrade` 均只检查 `num_obs ≥ 2` 与质心距离 `≤4.0 m`；sigma 门已删除 |
| 目标选择 | 硬过滤 `stable`；$\sigma_c$ 带通档：`0.65–0.70 > 0.60–0.65 > 0.55–0.60 > ≥0.70`；同档 `confirmed > provisional`；再按距离；近距离 tiebreak：`num_obs`，`freshness` |
| 近场放弃 | `provisional` 追到 `1.2 m` 内，左转最多 3 帧仍未升级 → 拉黑该轨道 → 回 explore |
| 停止 | 仅 `confirmed` 可停；无 `num_obs` 停止门槛；无当前帧重确认；无视角扩张复核；`pointnav_stop_radius=0.90 m` |
| fusion | 只保留 `world` / `none`；删除 camera / panoramic / world+scan |
| GD | 代码独立保留在 `vlfm/tsp3d_models/grounding_dino`；policy 当前不导入、不调用 GD，后续重构见 [VLVM-S6.md](VLVM-S6.md) |

### 4.2 最终参数

| 参数 | 值 |
| :--- | ---: |
| `sigma_tar` / `sigma_prop` | 0.60 / 0.45 |
| `t_S` / `s_min` / `radius_m` / `uncovered_w` | 0.14 / 0.07 / 0.5 / 1.0 |
| `provisional_sigma` / `provisional_min_dist` / `provisional_max_dist` | 0.50 / 1.0 / 4.0 |
| `upgrade_min_obs` / `upgrade_max_dist` | 2 / 4.0 |
| `merge_dist_thresh` / `merge_new_info_dist` / `merge_new_info_yaw` / `merge_new_info_steps` | 0.5 / 0.2 / 10° / 3 |
| `confirmed_to_stable_misses` / `provisional_to_stable_misses` | 3 / 3 |
| `stable_miss_delete` / `stable_total_miss` / `stable_stale_purge` | 5 / 8 / 60 |
| `stable_revive_radius` / `stable_revive_window` / `stable_revive_min_sigma` | 0.5 / 10 / 0.50 |
| `stop_gate_confirm_radius` / `stop_gate_turns`（仅 provisional 近场放弃） | 1.2 m / 3 |
| `pointnav_stop_radius` | 0.90 m |
| `occ_min_voxels` / `occ_min_ratio` | 8 / 0.001 |

### 4.3 删除的机制

| 机制 | 处理 |
| :--- | :--- |
| evidence 目标选择 | 删除；可匹配子集 TP 基本不变，最后 nav goal TP 反降 |
| `num_obs≥3` 停止门槛 | 删除；S6 / S7 0.52 分别丢 37 / 35 TP |
| 当前帧重确认 | 删除；与 `n≥3` 叠加后近场仅约 40% 可停 |
| 视角扩张复核 / `stop_blocked` | 删除；S7 full 504 次扩张、407 失败，steps 216.9 |
| `camera` / `panoramic` / `world+scan` | 删除；fusion 只保留 `world` / `none` |
| 两态 `enable_two_state` / `s4_sigma_source` | 删除；最终固定 `legacy` $\sigma_c$ 口径 |
| `provisional` 直接删除 / close-reobservation 升级 | 删除；统一走 `stable` 池与每帧升级 |
| `enable_fb` / `fb_*` / suspicious 标记链 | 删除；近场 miss 只保留 stable 池预算 |
| `upgrade_sigma_min` | 删除；升级只保留 `num_obs` 与距离，覆盖优先，不再设 sigma 门 |
| GD policy 调用 / `gdp_*` / `gd_skip_classes` | 删除；GD 代码独立保留，policy 不再导入或调用 |

### 4.4 定稿依据

| 项 | 数据 |
| :--- | :--- |
| 准入基线 | S5 臂 A 121 / 40.74%，FP 72；高于 0.50 / 0.52 / S6 |
| 停止 | S6 / S7 0.52 的 `confirmed` 分别保留 112 / 114 TP；`n≥3` 仅 75 / 79 |
| evidence | S7 0.52 最后 nav goal TP 120→113；S7 full 134→122 |
| $\sigma_c$ 带通 | `0.65–0.70` 档 TP 率 64.8 / 73.0 / 80.9%；`≥0.70` 在 S6 / S7 0.52 / S4-1 / S2 均偏弱 |
| `s_min` | 三场景 18,457 条 covered 候选最小值 0.087629；取 0.07 作为无 clamp 下限 |
| `uncovered_w` | uncovered precision 33–35%，与候选池整体接近；下调只丢 TP，不提升 precision |
| stable 删除 | 连续 miss 删除发生在 3；累计 miss 6 档 FP 偏多，8 档 TP 稳定 |

### 4.5 当前验证

| 项 | 值 |
| :--- | :--- |
| 最终四场景 | `5cdEh9F2hJL`、`TEEsavR23oF`、`mv2HUxq3B53`、`wcojb4TFT35`，4 × 99 = 396 集 |
| 启动脚本 | `scripts/_nav_s5final.sh` |
| 运行目录 | `outputs/s5_final_4scene_2026-10-26_102914` |
| 结果 | 396 集，success 164，SR **41.41%**；Oracle 58.08%，SPL 0.1803，SoftSPL 0.2795，steps 184.78 |
| 状态 | 已完成；结果与最终运行离线标定见 [VLVM-S5final.md](VLVM-S5final.md) |

## 实验顺序与判定门

排序原则：先固定准入与状态机（臂 B 作基准），再合并 merge 与 fallback 改造（S6），最后改目标选择、停止门槛与视角扩张复核（S7）；候选池扩容（S8）与 GD 排在最后，不与阈值改动同跑。

| 阶段 | 单一改动 | 口径 | 判定门 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| S0 | 数据与离线重放：occ reason 拆分、$\sigma_c$ 分布、fallback 状态分布 | P1 990 集日志，不跑仿真 | 固定后续阶段的阈值与上界 | 已完成 |
| S1 | occ 拆分：`out_of_bounds` 改为 `geom_unknown`（不 reject）；其他 occ reason 保留 reject | 三场景 × 99 | SR 不低于 P1 G4（39.06%）；TP 被删 episode 下降；FP 增量 ≤1 pp | 已完成 |
| S2 | 重启 provisional：new 条件 $0.55\le\sigma_c<0.60$、$1.5\le dist\le4.0$ m、不看 $S$；三态 `confirmed` / `provisional` / `stable`；准入下限随 provisional 降到 0.55 | 三场景 × 99 | 每集记录数上升；nav goal TP 占比 ≥65%；SR 不低于 S4-1；delete 中 TP 不增加 | 已完成（SR 40.74%，见 `VLVM-S5s2-Exp.md`） |
| S3 | 跳过：不做 merge 重判与状态升级 | — | — | 跳过 |
| S4-1 | 两态 + unknown $>0.6$ → `confirmed`；S4-B 使用旧 $\sigma_c$：`verify_state[3]` 历史最大 $c'$ | 三场景 × 99 | 作为 S4-2 的 baseline；SR 不低于 S1 | 已完成（SR 38.38%，见 `VLVM-S5s4-Exp.md`） |
| S4-2 | 五列改为离线定列（候选级分布选口径），不单独跑导航；运行时验证并入 S2 之后 | 离线 + 三场景 × 99 | 对比 S4-1；nav goal TP 占比 ≥65%；SR 不低于 S4-1 | 离线定列 |
| S5 | 三场景 A/B：provisional 准入 0.55 / 0.50，仅 `confirmed` 可停 | 三场景 × 99 | A ≥40.74%、FP ≤79 | 已完成（121 / 116，见 `VLVM-S5s5-Exp.md`）；最终机制见 §4；四场景 164/396（41.41%），见 `VLVM-S5final.md` |
| S6 | merge 新信息门 + `stable` 删除池（基准臂 B） | 三场景 × 99 | 升级 precision ≥50%；confirmed 池 FP ≤170；delete ≤120；SR ≥116 | 已完成（117，SR 39.39%；precision 与 FP 门未达，见 `VLVM-S5s6-Exp.md`） |
| S7 | evidence 排序 + 停止 `num_obs≥3` + 视角扩张复核 | 三场景 × 99 | FP ≤80；SR ≥121；never_saw ≤80；steps ≤155 | 已完成（106，SR 35.69%；FP 75 达成，SR/never_saw/steps 未达；见 `VLVM-S5s7full-Exp.md`） |
| S8 | `sigma_prop` 网格 | 三场景 × 99 | 准入与阈值全部冻结，只测候选池扩容 | 取消独立阶段：`sigma_prop=0.45` 固定 |

S0/S1/S2/S2U/S4-1/S5/S6/S7 已完成；S3 跳过；S4-2 改为离线定列；S5 最终机制见 §4；S8 取消独立阶段，`sigma_prop=0.45` 固定。GD 进入 [VLVM-S6.md](VLVM-S6.md)（GD 阶段，与 §3.5 的 S5-S6 merge 实验不同）。
