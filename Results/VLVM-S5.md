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

### 3.2.1 S4-1 两态口径（历史，provisional 关闭）

仅用于 S4-1；S2 重启后按 3.2 的更新条件（$0.55\le\sigma_c<0.60$、$1.5\le dist\le4.0$ m、不看 $S$）。

| 项 | 当前定稿 |
| :--- | :--- |
| 状态 | 仅 `confirmed` / `stable`；不启用 `provisional` |
| new | $\sigma_c>0.60$ 进入 `confirmed`；$\sigma_c\le0.60$ 不写入 memory |
| `geom_unknown` | $\sigma_c>0.60$ 直接进入 `confirmed`；不再设 provisional 上限，也不再额外要求 covered |
| merge | 不改状态；只更新质心、`num_obs`、五列 $\sigma_c$，并清零 `near_miss` |
| 降转/删除 | `confirmed` 累计 3 次 `near_miss` → `stable`；`stable` 累计 2 次 → delete |
| STOP | 不读状态；仍为到目标 0.90 m 内停止 |
| 跳过 | S2 三态、S3 merge 重判均不执行；不新增 provisional 行为 |

### 3.3 $\sigma_c$ 记录口径（S4-1 / S4-2）

- 现状：$\sigma_c$ 在每次检测时用当时 $c$、$S$ 计算一次；memory 存 `max(old_conf, new_conf)`；$S$ 不再重查；`quality_score` 用这个历史最大值，但当前选择不使用。
- 问题：单帧高 $\sigma_c$ 永久抬高轨道，后续低 S/低 c 不会降低；FP 轨道会被历史峰值保护。
- 五种记录值如下；`current` 指当前帧检测的 $\sigma_c=c_{frame}\cdot w_S(S_{frame})$，不重查历史 S。

| 列名 | 定义 | new 初始化 | merge 更新 | 特性 |
| :--- | :--- | :--- | :--- | :--- |
| `sigma_c_max` | 历史最大 $\sigma_c$ | `current` | $\max(old,current)$ | 单调不减，保留峰值 |
| `sigma_c_latest` | 最近一帧 $\sigma_c$ | `current` | `current` | 可降，对单帧 S 敏感 |
| `sigma_c_ema_03` | $\alpha=0.3$ 的 EMA | `current` | $0.3\cdot current+0.7\cdot old$ | 最平滑，最稳 |
| `sigma_c_ema_05` | $\alpha=0.5$ 的 EMA | `current` | $0.5\cdot current+0.5\cdot old$ | 中等平滑 |
| `sigma_c_ema_08` | $\alpha=0.8$ 的 EMA | `current` | $0.8\cdot current+0.2\cdot old$ | 最接近最新帧 |

- 五种值随 `new` / `merge` memory event 一起记录；S2/S3 跳过后仍需保留，供 S4-2 使用，不改变 fallback 行为。
- EMA 的 $\alpha$ 为新样本权重；三档在实验前固定，避免补跑。
- **S4-1**：继续读 `verify_state[3]` 的历史最大 $c'$，即旧口径；B 方案用它排序。
- **S4-2**：从五列（`sigma_c_max` / `latest` / `ema_03` / `ema_05` / `ema_08`）选一列，替换 B 方案的 `valid_scores` 来源。
- 五列必须在 provisional 关闭时也落盘；否则 S4-2 无法回放或在线读取。
- 验收：同一轨迹可复现 5 列分数；S4-2 能读取选定列。

### 3.4 目标选择（S4）

- 现状：`_get_target_object_location` 用 `argmin(dists_2d)`，`quality_score/num_obs/src` 只记录。
- 两档：

| 方案 | 规则 |
| :--- | :--- |
| A 基线 | 纯最近质心（当前 `argmin(dists_2d)`） |
| B 分数+观测 | 先按 `num_obs` 分组，再按 $\sigma_c$ 分档，档内选最近 |

**S4-B 具体排序**

| 步骤 | 规则 |
| :--- | :--- |
| 1. `num_obs` | 优先 `num_obs>=2`；仅当没有 `num_obs>=2` 时退回 `num_obs=1` |
| 2. $\sigma_c$ 档 | 最高非空档：$\ge0.70$ / $0.65$–$0.70$ / $0.60$–$0.65$ / $<0.60$ |
| 3. 距离 | 档内 `argmin(dists_2d)`；保留 0.1 m / 0.5 m distance-latch |

- **S4-B 不读取 `confirmed` / `stable`**；两态只决定 delete，不参与选择。
- A 仅作为最近邻参考，不单独跑；S4-1/S4-2 均使用 B。
- S4-1 的 $\sigma_c$ 用旧口径：`verify_state[3]` 的历史最大 $c'$。
- S4-2 只替换 $\sigma_c$ 来源：从五种记忆值中选一列，接入 B 方案排序。
- $\sigma_c$ 计算改变只作用于 **nav goal 排序**：改变谁被选中；不改变 new 准入线、不改变 `near_miss`/stable、不改变 0.90 m STOP 半径。STOP 的间接变化仅来自所选目标不同。
- 保留 0.1 m / 0.5 m distance-latch，避免频繁切换。
- 验收：nav goal TP 占比（当前 51.87%）≥65%，SR 不下降。

### 3.5 GD 状态

GD 暂不进入本轮定稿，fallback 和 $\sigma_c$ 都不依赖 `both` / `gd`。GD 的触发方式、确认语义与是否保留需重新评估后另开实验，不在本文展开。

## 4. 实验顺序与判定门

排序原则：先固定准入与两态，再只改目标选择；S4-1 与 S4-2 之间只改变 $\sigma_c$ 计算一个变量。GD 不在本轮顺序内。

| 阶段 | 单一改动 | 口径 | 判定门 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| S0 | 数据与离线重放：occ reason 拆分、$\sigma_c$ 分布、fallback 状态分布 | P1 990 集日志，不跑仿真 | 固定后续阶段的阈值与上界 | 已完成 |
| S1 | occ 拆分：`out_of_bounds` 改为 `geom_unknown`（不 reject）；其他 occ reason 保留 reject | 三场景 × 99 | SR 不低于 P1 G4（39.06%）；TP 被删 episode 下降；FP 增量 ≤1 pp | 已完成 |
| S2 | 重启 provisional：new 条件 $0.55\le\sigma_c<0.60$、$1.5\le dist\le4.0$ m、不看 $S$；三态 `confirmed` / `provisional` / `stable`；准入下限随 provisional 降到 0.55 | 三场景 × 99 | 每集记录数上升；nav goal TP 占比 ≥65%；SR 不低于 S4-1；delete 中 TP 不增加 | 已完成（SR 40.74%，见 `S5s2.md`） |
| S3 | 跳过：不做 merge 重判与状态升级 | — | — | 跳过 |
| S4-1 | 两态 + unknown $>0.6$ → `confirmed`；S4-B 使用旧 $\sigma_c$：`verify_state[3]` 历史最大 $c'$ | 三场景 × 99 | 作为 S4-2 的 baseline；SR 不低于 S1 | 已完成（SR 38.38%，见 `S5s4.md`） |
| S4-2 | 五列改为离线定列（候选级分布选口径），不单独跑导航；运行时验证并入 S2 之后 | 离线 + 三场景 × 99 | 对比 S4-1；nav goal TP 占比 ≥65%；SR 不低于 S4-1 | 离线定列 |
| S5 | 十场景验证（不含 GD） | 990 集 | 本轮定稿目标 ≥55.3%；60% 待第二来源复评 | 待跑 |

S0/S1/S2/S4-1 已完成；S3 跳过；S4-2 改为离线定列；S5 待跑。GD 不在本轮顺序内，待重新评估后另开实验。
