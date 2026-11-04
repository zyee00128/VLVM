# Project VLVM Stage 6：GD Proposer 机制调整

> 定位：S5 最终机制已定稿，S6 只调整 GD proposer。本文按 nav pipeline 记录问题定位与 GD 调整顺序；GD 的具体机制、参数与实验结论在讨论定稿后写入 §4，未跑通前不写入机制表。

## 1. S5 定稿基线

S6 继承 [VLVM-S5.md](VLVM-S5.md) 的最终机制，不重新讨论准入、目标选择、停止和稳定池。基线固定为：

| 层 | 最终机制 |
| :--- | :--- |
| 候选准入 | `sigma_tar=0.60`、`sigma_prop=0.45`；`c'=c·w_S`；`t_S=0.14`、`s_min=0.07`、`radius=0.5 m`、`uncovered_w=1.0` |
| 几何 | `insufficient_occ_voxels` / `air_penetration` / `low_occupancy_ratio` 继续 reject；`out_of_bounds` → `geom_unknown`，最高只到 `provisional` |
| merge | 同类质心 `<0.5 m`；新信息门 `≥0.2 m` 或 `≥10°` 或 `≥3 步`；不允许 merge 到 `stable` |
| provisional | `0.50 ≤ sigma_c < 0.60`，`1.0 ≤ dist ≤ 4.0 m` |
| stable | `confirmed/provisional` 各 3 次 near_miss → `stable`；`stable` 删除条件取或：`stale≥60` / 连续 miss≥5 / 累计 miss≥8；复活 `0.5 m / 10 步 / sigma_c≥0.50` |
| 升级（S5 历史） | `close_reobservation` / `frame_upgrade` 均只检查 `num_obs ≥ 2` 与质心距离 `≤4.0 m`；S6 已改为 `num_obs=4`、无距离门，见下方 S6 生效值 |
| 目标选择 | 硬过滤 `stable`；$\sigma_c$ 带通档：`0.65–0.70 > 0.60–0.65 > 0.55–0.60 > ≥0.70`；同档 `confirmed > provisional`；再按距离；近距离 tiebreak：`num_obs`，`freshness` |
| 近场放弃 | `provisional` 追到 `1.2 m` 内，左转最多 3 帧仍未升级 → 拉黑该轨道 → 回 explore |
| 停止 | 仅 `confirmed` 可停；无 `num_obs` 停止门槛；无当前帧重确认；无视角扩张复核；`pointnav_stop_radius=0.90 m` |
| fusion | 只保留 `world` / `none`；删除 camera / panoramic / world+scan |

S5 最终四场景实测：164/396（SR 41.41%），结果与离线标定见 [VLVM-S5final.md](VLVM-S5final.md)。

关键参数：

| 参数 | 值 |
| :--- | ---: |
| `sigma_tar` / `sigma_prop` | 0.60 / 0.45 |
| `t_S` / `s_min` / `radius_m` / `uncovered_w` | 0.14 / 0.07 / 0.5 / 1.0 |
| `provisional_sigma` / `provisional_min_dist` / `provisional_max_dist` | 0.50 / 1.0 / 4.0 |
| `upgrade_min_obs` / `upgrade_max_dist`（S5 历史） | 2 / 4.0 |
| `merge_dist_thresh` / `merge_new_info_dist` / `merge_new_info_yaw` / `merge_new_info_steps` | 0.5 / 0.2 / 10° / 3 |
| `confirmed_to_stable_misses` / `provisional_to_stable_misses` | 3 / 3 |
| `stable_miss_delete` / `stable_total_miss` / `stable_stale_purge` | 5 / 8 / 60 |
| `stable_revive_radius` / `stable_revive_window` / `stable_revive_min_sigma` | 0.5 / 10 / 0.50（S5 历史基线） |
| `stop_gate_confirm_radius` / `stop_gate_turns`（仅 provisional 近场放弃） | 1.2 m / 3 |
| `pointnav_stop_radius` | 0.90 m |
| `occ_min_voxels` / `occ_min_ratio` | 8 / 0.001 |

> **S6 生效值**：`provisional_sigma=0.55`；普通升级只使用 `upgrade_min_obs=4`（不再有 close/dist 变体）；另一条升级路径是 `gd-tsp` 命中 confirmed/provisional track，并且只允许 merge 到既有 track。`enable_gd_mechanism=true`。当前实验使用 `stable_revive_radius/window=0.8/20`、gd-only 全类别、`occ_passed` 几何门（`occ_confirmed` 仅作诊断/排序特征）；`gd_only_conf_min`/`gd_only_merge_conf_min` 只作为记录阈值，**conf 门控关闭**；`gd_only_ttl=90` 只作为参考，**gd-only 不删除**；启用 `gd_stable_shadow` 与 `gd_depth_shadow`，stable 命中只写 shadow、**不升级**；`gd_depth_failed_track_merge=True` 允许 depth 失败的 confirmed/provisional track 合并；`gd_depth_failed_gd_only=True` 允许 depth 失败的 gd-only new/merge。上表 §1 是 S5 历史基线，不覆盖 §4 的 S6 参数，避免把实验前参数误读成当前配置。
>
> 日志兼容：S6gd2 的 merge-time 普通升级仍写 `upgrade_reason=close_reobservation`，这是历史标签；实际条件只有 `num_obs>=4`，不看距离。当前代码已统一为 `base_upgrade`，并用 `trigger=merge` / `trigger=frame` 区分触发时机；不再使用 `close_reobservation`、`merge_upgrade` 或 `frame_upgrade`。

## 2. 问题定位

> 当前目标来源只有 TSP3D；区分依据为 TSP3D 置信度、value map 语义价值与 occ 几何。

### 2.1 目标覆盖（第一约束）

| 事实 | 数值 |
| :--- | ---: |
| post-occ TP 集数 / 总计 | 571 / 990 |
| 现有输出支持的 SR 上限（571 全成功 + 21 无 TP 现成功） | 592 / 990 = 59.8% |
| 无 TP 集 | 419（195 likely-infeasible） |
| 中置信 TP 集（conf 0.5–0.6） | 146（TP conf 中位 0.555；S 中位 0.205；当前 success 24） |
| 高置信 TP 被 S 拦截 | 26（当前 success 7） |
| FP-only 集 | 166（wrong lock 100%；wrong stop 61.4%；success 30） |

判断：

- 覆盖是绑定约束：59.8% 已低于 60%，且为乐观上限；仅靠 TSP3D + occ 无法支撑 60%。
- 可回收池：224 集可行无 TP + 146 集中置信；26 集 S 阻断与 195 集不可行不在 GD 可及范围。
- 166 集 FP-only 本质是召回失败：真实目标没有进入候选，才对 FP 形成 100% wrong lock。

### 2.2 目标选择与排序（第二约束）

| 事实 | 数值 |
| :--- | ---: |
| 候选准入 precision | 72.99% |
| nav goal TP 占比 | 51.87% |
| TP+FP 共存集 | 91（SR 65.93%；TP 高分 64.5%、FP 高分 69.0%） |
| 轨道级 $\sigma_c$ 区分度（merge 事件 AUC） | 0.587–0.642 |
| 末端目标 $\sigma_c$ 中位（TP / FP） | 0.6244 / 0.5962 |
| 末端目标 `num_obs` 中位（TP / FP） | 5 / 3 |
| evidence 排序回放 | 同轨道 4,030→4,031 TP；末端 134→122 TP |

判断：

- 排序与分类高度相关；TP/FP 在现有特征上无方向性差异，换排序规则没有增益。
- 选择层的上限由分类质量决定；在覆盖不足时，排序只能决定“先错哪一个”。

### 2.3 最终目标停止（第三约束）

| 事实 | 数值 |
| :--- | ---: |
| 末端目标 `confirmed` 命中率 | 68.24%（116 / 170） |
| 末端目标 `provisional` 命中率 | 30.00%（9 / 30） |
| 末端目标 `stable` 命中率 | 42.86%（9 / 21） |
| 停止资格 | 仅 `confirmed`；STOP 0.90 m |
| 加严回放（S7 `num_obs≥3` vs `confirmed`） | −8 TP / −5 FP（S6 为 −37 TP / −47 FP） |
| expansion 复核（S7 full） | 504 次（97 通过 / 407 失败）；steps 216.9 |

判断：

- 停止精度是分类质量的下游；停止门加严只能做 TP/FP 换手，不能提升目标身份正确率。
- 停止不是当前主杠杆；先解决覆盖与分类，再讨论停止复核。

## 3. GD Proposer Outline

### 3.1 离线标定（前置，零行为）

- 形式：S5 定稿配置 + GD 开启但只落日志，不写记忆、不影响动作。
- 目的：在真实帧与位姿上测 GD 的召回率、TP/FP 正票率、类别差异与成本。
- 先测量：419 集无 TP 中 GT 目标的轨迹可视率（至少一帧进入 FOV + depth）。它是 GD 覆盖的真实上限；不可视部分 GD 无解，探索已冻结。
- 判定门：
  - 覆盖：224 集可行无 TP 中，GD proposal 命中 GT 0.5 m 内 ≥50%，才进入 3.2。
  - 区分：TP 轨道与 FP 轨道的 GD 正票率差 ≥0.3，才进入 3.3。
  - 停止：末端 `confirmed` TP/FP 的正票率差 ≥0.4，且预测 precision ≥85%，才进入 3.4。
- S6gd0 实测结论：覆盖门未过（无 post-occ TP 中 GD 命中仅 18/156，11.54%），因此 GD **不作为独立覆盖源**；只保留窄类别、近场、正向证据与仲裁语义。
- 未过门的处理：不进入对应阶段；覆盖门未过时 GD 只做窄类别/近场补召，不改主线。

### 3.2 目标覆盖（修正：GD-only 池是窄类别补召，不是独立覆盖源）
> 目标未出现时的搜索假设

- 实测约束：GD 在无 TP 集的覆盖率远低于 50% 门，不能承担“从零发现任意目标”的职责。
- 保留用途：当前 S6 实验已放开到全部 target class；GD 在近场、占据门通过、绕过 S-penalty 的条件下生成 `gd-only` 池条目，作为导航可选的补充观察点。
- 明确边界：`gd-only` 不产生 `confirmed`、不可停、不写 $\sigma_c$、不进入 TSP3D 的 admission / fusion / S-penalty；它不是第二覆盖源，也不对“目标一定存在”作声明。
- 验收：三场景 success 增量、FP 增量、never_saw 变化、steps；若 FP 或 steps 变差，回退 gd-only 池。

### 3.3 目标选择与排序（第二优先级）
> 探索行动的收益与成本比较

- 定位：GD 正票作为正交证据，接入选择、生命周期与 `confirmed` 仲裁。
- 约束：不替代 $\sigma_c$ 带通与距离排序；不进入 S-penalty / fusion；正票需带新信息门。
- `gd` 只命中 `stable` 时，当前**不升级** `stable`；只写 `gd_stable_shadow`，记录 GD proposal 的 GT/位置、stable track 的状态和后续 `track_frames` 轨迹。
- `gd-tsp`：GD 命中已存在的 TSP3D `provisional` 或 `confirmed` 时，直接进入 `confirmed`，已有轨道不增加 num_obs、不移动质心、不改 $$\sigma_c$$；
- `gd-only`：无命中时，当前**不再执行 conf 门控**；仍记录 `conf` 与参考阈值 `0.50/0.45`，用于后续决定是否启用及阈值。可能合并到已有的 `gd-only`，也可能新建 `gd-only`。
- 上述 `gd-only` conf、`gd-tsp` 的 track/candidate 定义与 confirmed 条件均为待实验结果调整项，见 §4.5.1。
- 验收：nav goal TP 占比、TP+FP 共存集 SR、confirmed 池 FP、gd-only 被选择后的 FP/steps。

### 3.4 最终目标停止（延后）

- 定位：延后。
- 当前决定：**不把 GD 接入 STOP**；`confirmed & gd_pos` 的 precision 81.25%，但 TP 损失不可接受；`gd_hits>=2` precision 87.72%，TP 只保留 50/171。
- 约束：S5 停止机制保持不变；不把 GD 缺失当负证据。
- 验收：停止 precision、wrong stop 集数、TP 损失、steps。

## 4. GD 在线机制

> 结论先行：GD 不是独立覆盖源，而是**窄类别近场补召 + TSP3D 正向仲裁**。

### 4.1 两条路径

| 路径 | 输入 | 关联条件 | 写记忆方式 | 结果状态 | 是否可停 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gd-only` | 未关联 TSP3D bbox 的 GD bbox | 全类别 + 近场 + `occ_passed`/`occ_confirmed` | `accumulate_gd_only(...)` | `lifecycle="gd"`、`origin="gd-only"` | 否 |
| `gd-tsp` | GD bbox 命中 TSP3D 的 `provisional/confirmed` track/candidate | `gd_pos` 或 `gd_candidate_pos` + 候选占据 | `accumulate(force_confirmed=True, origin="gd-tsp", block_stable=True)`，再 `apply_gd_evidence` | `confirmed`，`origin="gd-tsp"` | 是（继承 confirmed 的行为） |

GD bbox 命中 `stable` 时不写记忆、不升级，只写 `gd_stable_shadow` 事件；事件带 GD proposal 的 centroid（用于 `tp`/GT 标注）、stable track 的 centroid/状态及 `memory_id`，后续轨迹用 `track_frames.jsonl` 同一 `memory_id` 关联。

纯 TSP3D 条目的 `origin` 标记为 `tsp-confirmed` 或 `tsp-provisional`；`gd-only`/`gd-tsp` 表示 GD 参与了来源或仲裁。`promoted_from_gd_only` 一旦由 TSP3D 合并提升，就直接归一为 `origin="gd-tsp"`，不再作为独立的 `gd-only-tsp` 类别；该 flag 仅保留为 provenance。当前 `tsp-*` marker 主要用于 provenance/日志，选择由 `lifecycle_state`、`gd_only`、`gd_tsp` 等行为字段决定；`origin="gd-tsp"` 会直接驱动 `gd_tsp` 行为。

### 4.2 初始 gate（所有 GD proposal 共用）

按顺序执行，任一项失败即拒绝：

1. `canon` 必须是当前 episode 的目标类别（多类别目标用 `|` 分割，逐类过滤）。
2. `conf >= gd_conf_min`；默认 `0.40`。该阈值只做初筛，不承担可靠性判断。
3. `in_fov == True`：检测框中心在当前相机视场内。
4. `dist_robot <= gd_max_dist`；默认 `5.0 m`。离线数据显示 5–10 m 的 GD 收益低、噪声高，暂不放开。
5. **先关联 track**：读取 `match_track` / `match_candidate`，得到 `stable`、`confirmed`、`provisional`、`gd` 或 none。
6. **再判 depth**：默认 `depth_ok == True`；当前实验额外允许例外：
   - `track_same & confirmed/provisional & depth_ok=false` 且 `gd_depth_failed_track_merge=True`：使用 track centroid 继续合并到既有 confirmed/provisional track，写 `depth_relaxed=true`；不新建 track；
   - 无 confirmed/provisional track 的 `gd-only` new/merge 在 `gd_depth_failed_gd_only=True` 时允许继续走 gd-only，写 `depth_relaxed=true, depth_relaxed_gd_only=true`；
   - `track_same & stable & depth_ok=false` 仍只写 `gd_stable_shadow`、拒绝，不升级；
   - 未放开的 `track_same & confirmed/provisional & depth_ok=false` 仍写 `gd_track_depth_shadow`，只记录、不写记忆；
   - `stable & depth_ok=true` 仍走 `block_stable`，只写 shadow，不升级。
7. 之后按关联关系分流；未通过 base gate 的 proposal 只写 `mechanism.decisions` 的 `base_rejected`，不写记忆。

`gd_conf` 单独记录为 `gd_conf_max/gd_conf_last`，不映射为 $\sigma_c$，也不进入 S-penalty、fusion 或 TSP3D admission。

### 4.3 gd-only：全类别补召池

- 类别白名单已全部放开：`gd_only_classes=""`（空字符串表示全部 target class 都允许）；`tv` 和 `potted plant` 不再被类别门排除。
- 置信度：保留 `gd_only_conf_min=0.50`（新条目）和 `gd_only_merge_conf_min=0.45`（已有 gd-only 复观测）作为**参考阈值**。当前 `enable_gd_only_conf_gate=false`，不按 conf 拒绝；每条 decision/event 记录 `gd_only_conf_threshold/gd_only_conf_margin/gd_only_conf_would_reject`，后续按类别统计 TP/FP 再决定是否启用及阈值。
- 几何判据使用 `occ_passed=True`：`occ_passed=False` 一律拒绝；`occ_confirmed` 只作为诊断/排序特征保留。`geom_unknown` 但 `occ_passed=True` 的新目标允许进入 gd-only；`require_hard_occ_for_gd_only=True` 时才退回只接受 `HARD_CONFIRMED`。
- 不使用 TSP3D 候选的占据，也不使用 S-penalty 区域准入。
- 写入 `lifecycle="gd"`，`origin="gd-only"`，`gd_only=True`；`sigma_c` 不伪造，保持无 TSP3D 分数语义。
- 后续 GD 命中同一 gd 条目时合并 `gd_hits/gd_conf`，仍保持 `gd`，**不会**因 GD 累计而升级为 `confirmed`。
- `gd` 不可停止；可在没有 `confirmed/provisional` 时被选为导航目标。连续命中只提高 gd-only 内部排序。
- TTL / 删除：当前确定 **gd-only 不删除**；`enable_gd_ttl=false`，`gd_only_ttl=90` 仅保留为参考值。年龄跨过 `30/60/90/120/180/240/300` 步时写 `gd_ttl_shadow`，记录 `age_steps/gd_hits/gd_conf_max/gd_conf_last/num_obs/dist_to_robot`。`stable` 删除预算不适用于 gd 条目。

### 4.4 gd-tsp：TSP3D 正向仲裁

- 关联来源：
  - `match_track.same_object == True` 且不是 `stable`；或
  - `match_candidate.same_object == True`，且该候选 `occ_passed` 通过。
- `gd_pos == True`（已有 GD 正票）或候选匹配，才允许走 gd-tsp；单纯 `new_info` 不够。
- 分两种情况：
  - **已有 TSP3D 轨道**：`force_confirmed=True`，直接把 `provisional` 提升为 `confirmed`，并写 `origin="gd-tsp"`；若已是 `confirmed`，只补 `origin` 和 GD 证据。
  - **同物体候选尚未成轨**：用候选质心/表面写一条已确认轨道，记录 `gd-tsp` 升级事件；候选必须已通过占据。
- `block_stable=True`：命中 `stable` 时不写 GD 证据、不改状态、不复活、不覆盖 `stable` 预算；当前额外写 `gd_stable_shadow`。当前决定**不把 GD 命中的 stable 直接升级为 gd-tsp**。
- `apply_gd_evidence` 只写 `gd_hits/gd_conf_max/gd_conf_last/gd_last_step/gd_candidate_pos/gd_last_candidate_id`；不改质心、`num_obs`、$\sigma_c$ 或 TSP3D 分数。
- gd-tsp 在已有轨道合并时走 `gd_evidence_only=True`：只提升 `lifecycle_state` 和 `origin`，不把 GD 计作一次 TSP3D observation，也不移动质心或改写 $\sigma_c$。
- gd-tsp 是 S6 的保守取舍：只对“已经是 TSP3D 候选/轨道”的对象加速确认，不扩大候选空间；因此不会把 GD-only 的高 FP 直接带入 confirmed 池。

### 4.5 生命周期、排序与 TTL

- 目标层级采用三大档五小档：`gd-tsp`；中间档 `tsp-confirmed / gd-only`；低档 `tsp-provisional / stable`。`stable` 从选择中过滤；`tsp-provisional` 是 TSP 内部中转状态，不单列为中间档竞争目标。
- 同级排序（历史 policy key，按优先级从高到低）：
  1. `confirmed` 优先于 `provisional`，`provisional` 优先于 `gd`；
  2. gd-only 内部：`gd_hits >= 2` 优先于 `gd_hits == 1`；
  3. `confirmed/provisional` 内，真正增量的 GD 证据优先：`gd_upgraded`（gd-tsp 将 provisional/gd 提升为 confirmed）或 `promoted_from_gd_only`（TSP3D 合并 gd-only，随后归一为 gd-tsp）排在同层最前；
  4. 非 gd 条目按 $\sigma_c$ 带通 tier；
  5. `gd-tsp` 或 `gd_hits >= 2` 的正向条目优先；
  6. 距离（0.5 m 取整）近者优先；
  7. 再按 `num_obs` `freshness`。
- gd-only 的 `score` 不参与 $\sigma_c$ 带通；其 `stop_eligible=false`，永远不会触发 STOP。
- TSP3D 的 `provisional → confirmed` 只使用一个门槛：`num_obs >= upgrade_min_obs=4`；merge 与逐帧路径都统一记 `reason=base_upgrade`，用 `trigger=merge` / `trigger=frame` 区分；历史 S6gd2 日志仍可见 `close_reobservation`/`frame_upgrade`；没有 `num_obs=3` 路径，也不使用距离升级。
  - 另一条升级路径是 `gd-tsp`：只允许命中已入 memory 的 `confirmed/provisional` track，然后 `force_confirmed` 并记 `reason="gd_tsp"`；candidate-only 命中不再直接写 gd-tsp。gd-tsp 是 merge-only，若无法合并到目标 memory 则记 `blocked`，不会新建 track。
- `gd-only` 当前**不删除**：`enable_gd_ttl=false`，仅记录 shadow；`gd-tsp`/`confirmed` 继续使用 S5 的 stable/near-miss 预算。
- `stable` revive 当前为 `radius=0.8 m`、`window=20 步`、`min_sigma_c=0.50`。该半径独立于普通 `merge_dist_thresh=0.5 m`，最近 stable 在 0.8 m 内即可尝试 revive。merge 事件记录 `revive_sigma_c/revive_num_obs_before/revive_num_obs_after/revive_dist_to_old_centroid/revive_age_steps/revive_detect_dist` 以及实际阈值，便于后续判断 0.8/20 是否带来净收益。
- 当前**不把 GD 接入 STOP**；`confirmed & gd_pos` 仅作离线复核，原因见 §3.4。
- **停止资格（当前代码）**：只有 `gd-tsp` confirmed 可停；`promoted_from_gd_only` 在提升时已归一为 `gd-tsp`，因此同样可停。**纯 TSP confirmed 暂时取消可停**；`gd-only` / `provisional` 不可停。所有被选中的 `confirmed`（含 gd-tsp/promoted）与 `gd-only` 每步写 `stop_shadow.jsonl`，记录 `gd_hits/num_obs/sigma_c/goal_distance` 及多组假设阈值；不影响动作。
- **S6gd2 运行的排序**：`state tier (confirmed > provisional > gd) -> gd-only hits>=2 -> gd_incremental -> sigma_c band -> gd_tsp/gd_hits>=2 -> distance bucket -> num_obs -> gd_hits -> freshness`。
- **S6gd3 运行排序**：`gd-tsp（含 promoted gd-only） > gd-only > confirmed > provisional`。
- **S6gd4 运行排序**使用四挡：`gd-tsp（含 promoted gd-only） > confirmed > gd-only > provisional`；gd-tsp 内 `sigma band -> gd_hits -> distance bucket -> num_obs/freshness`，confirmed/provisional 内 `sigma band -> distance bucket -> num_obs/freshness`，gd-only 内 `gd_hits -> distance bucket -> num_obs/freshness`。`stable` 与 `stop_blacklist` 仍先过滤。
- **当前最新排序（S7 baseline）**：`gd-tsp（含 promoted gd-only） > gd-only > confirmed > provisional`；中间档只调整 `gd-only / confirmed` 顺序，其他层级和 STOP 不变。

### 4.5.1 待实验结果调整的判据

> 当前先保持现实现，不在本实验中途改规则；以下各项作为结果分析后的调参/改臂入口。

| 项 | 当前实现 | 待确认/调整 | 主要结果指标 |
| :--- | :--- | :--- | :--- |
| GD class 放开 |
| `gd depth-ok` | 已放开 `track_same & confirmed/provisional` 的 track merge，并使用 track centroid；也已放开 gd-only new/merge（`gd_depth_failed_gd_only=True`）；stable 仍只写 shadow、不升级 | 观察 gd-only depth 放开的 TP/FP、steps 和错误 stop | depth_relaxed 的 TP/FP、gd-only 提升率、steps |
| `gd-only occ gate` | 使用 `occ_passed`；`occ_confirmed` 仅作诊断/排序特征 | 保留 gd-only merge，不退回 hard-only | gd-only new/merge 的 TP/FP |
| `gd-only` conf | `enable_gd_only_conf_gate=false`；保留 new 0.50 / merge 0.45 参考值；每条 decision/event 记录 observed conf、阈值、margin、would_reject | 下一 run 按类别统计 `gd_only_new` / `gd_only_merge` 的 TP/FP，再决定是否启用及阈值 | gd-only 写入 precision、被选后 FP/steps、SR/SPL |
| `gd-only` TTL | `enable_gd_ttl=false`；在 30/60/90/120/180/240/300 步写 `gd_ttl_shadow`；**不删除** | 保留 shadow 仅作观察 | gd-only 存活步数、长期 FP、TP 保留率 |
| `gd-tsp` 定义 | **track-only + merge-only**：只匹配已入 memory 的 `confirmed/provisional` track；candidate-only 仅记日志，不写 gd-tsp；无法合并到目标 track 时记 `blocked` | 已决定；后续核对 `association=candidate_only` 的数量与 TP/FP | candidate-only 被挡掉的 FP、track-only gd-tsp 的 TP/FP、SR |
| `track_stable` | **不升级**；写 `gd_stable_shadow`，记录 GD proposal GT、stable centroid/状态和 `memory_id`，后续轨迹由 `track_frames` 关联 | 已确定不直接升级；继续观察 shadow | shadow precision、后续轨迹连续性 |
| `stable` revive | 当前实验 `0.8 m / 20 步 / sigma_c≥0.50`；merge 事件记录 sigma_c、num_obs、旧质心距离、age、detect_dist | 根据本轮 revive 判据数据决定是否回到 0.5/15 或调整 | revive TP/FP、revive 后续轨迹、FP/steps |

### 4.5.2 目标排序

当前讨论使用三大档五小档：`gd-tsp`；中间档 `tsp-confirmed / gd-only`；低档 `tsp-provisional / stable`。`stable` 从选择中过滤，`tsp-provisional` 作为 TSP 内部中转站，不单独参与中间档竞争。

历史 S6gd2 三场景实验使用旧排序；S6gd3 运行使用 `gd-tsp > gd-only > confirmed > provisional`；S6gd4 运行使用 `gd-tsp > confirmed > gd-only > provisional`。当前最新排序（S7 baseline）为：

| 优先级 | 类别 | 内部排序 |
| :---: | :--- | :--- |
| 1 | `gd-tsp`（track 或 `promoted_from_gd_only`，提升后 origin=gd-tsp） | `sigma_c band -> gd_hits 降序 -> 距离桶 -> num_obs>=3/数量 -> stale` |
| 2 | `gd-only`（`lifecycle=gd`） | `gd_hits 降序 -> 距离桶 -> num_obs -> stale`；`gd_hits>=3` 为强正向 |
| 3 | `confirmed`（纯 TSP3D，仍不可 STOP） | `sigma_c band -> 距离桶 -> num_obs>=3/数量 -> stale` |
| 4 | `provisional` | 同 confirmed；作为 TSP 内部中转/低档 |

阈值：

- `sigma_c band` 继续用 `0.65–0.70 > 0.60–0.65 > 0.55–0.60 > ≥0.70 > 其他`。
- 距离桶为 `round(distance_m / 0.5)`；同桶再比较 `num_obs`，`num_obs>=3` 优先于 1–2，随后 `num_obs` 多者优先。
- **`num_obs` 放在 `freshness` 前面**：S6gd2 同一步/同层/同距离桶的 TP-FP pair 中，`num_obs` 正确排序 185 对、`freshness` 85 对；限定同 band/hits/距离桶内仍是 29 vs 21。`freshness` 只在 `num_obs` 相同时作为 tiebreak。
- `gd-only` 的 `gd_hits` 直接降序；`gd_hits>=3` 在 S6gd2 中明显高于 1–2，但不设硬门槛，避免首次探索被切断。
- `stable` 仍从选择中过滤；`stop_blacklist` 仍按 memory_id 过滤。
- 数据提示：gd-only 与 confirmed 同时候选时，gd-only 的 step-level TP 覆盖更高（396 个共有 step：gd-only 295 vs confirmed 195），但会推迟 confirmed 的 STOP；建议下一轮再加“近距离 confirmed 的 STOP 例外”或单因素消融。

补充约定：

- **`gd-tsp` 已统一为 track-only**：只允许 `match_track.lifecycle_state in (confirmed, provisional)`；candidate-only 命中只保留 `association=candidate_only` 日志，不再写 `origin=gd-tsp`，因此 `gd_tsp=true` 的含义唯一。
- GD 路由固定为三条：`track_same & stable -> blocked`；`track_same & confirmed/provisional -> gd-tsp`；其余无 track 关联 -> gd-only（仍过 class/occ；conf 当前只记录、不拒绝）。`gd` track 的再次命中走 gd-only，不会自证成 confirmed。
- `gd-only` 的 `0.50` / `0.45` 当前只是参考阈值，不控制写入；conf gate 关闭。两者都不应与 `sigma_tar=0.60` 或 `provisional_sigma=0.55` 混写。
- 后续 TSP3D candidate 与 gd-only 条目空间匹配时，由 `TargetMemoryManager.accumulate` 提升为 `confirmed`（`reason=gd_only_tsp_merge`，保留 `promoted_from_gd_only=true`，并写入 `origin=gd-tsp`），不再永久停留在 gd 池，也不进入独立的 gd-only-tsp 分支。
- candidate 的 GD occ 只看 `occ_passed`；`geom_status="geom_unknown"` 不再单独拒绝，`occ_passed=false` 仍然拒绝。

### 4.5.3 中间档排序

中间档 `tsp-confirmed / gd-only` 的排序调整已移入 S7，见 [VLVM-S7.md](VLVM-S7.md) §2；S6 只保留 GD proposer 与当前选择链路，不在本节维护中间档 ranker。

### 4.6 保守参数（S6 默认）

| 参数 | 值 | 作用 |
| :--- | :---: | :--- |
| `enable_gd_mechanism` | `true` | 总开关；关闭时 policy 不创建机制 |
| `gd_conf_min` | `0.40` | 所有 GD proposal 初筛 |
| `gd_only_conf_min` / `gd_only_merge_conf_min` | `0.50` / `0.45` | 仅记录为参考阈值 |
| `enable_gd_only_conf_gate` | `false` | 关闭 conf 拒绝；保留 would-reject 日志 |
| `gd_max_dist` | `5.0 m` | GD 近场硬门 |
| `gd_only_ttl` / `enable_gd_ttl` | `90` / `false` | TTL 仅作参考；gd-only 不删除，只写 shadow |
| `gd_stable_shadow` / `gd_depth_shadow` | `true` / `true` | stable 命中、depth 失败 track merge 的只读 shadow |
| `gd_depth_failed_track_merge` | `true` | 仅允许 depth 失败的 confirmed/provisional track 合并 |
| `gd_depth_failed_gd_only` | `true` | 允许 depth 失败的 gd-only new/merge；stable 仍只写 shadow、不升级 |
| `gd_only_classes` | `""` | 空 = 全部 target class |
| `upgrade_min_obs` | `4` | 唯一普通升级门槛；无距离门 |
| `stable_revive_radius/window/min_sigma` | `0.8 / 20 / 0.50` | 当前实验 revive 判据 |
| `enable_gd_pool` | 内部跟随 `enable_gd_mechanism` | 是否允许创建 gd 条目 |

### 4.7 日志与可追溯字段

每次 GD 处理都会写 `mechanism` 子记录；建议按以下文件核对：

| 文件 | 关键字段 |
| :--- | :--- |
| `gd_frames.jsonl` | `step`、`robot_xyz`、`proposal_ids/n_proposals`、`mechanism.counts/decisions/events` |
| `gd_proposals.jsonl` | `proposal_id`、`canon/conf/dist/depth_ok/in_fov`、`gd_pos/gd_candidate_pos`、`match_track/match_candidate`、`gd_hits_after`、`new_info` |
| `memory_events.jsonl` | `event`（TSP3D `new/merge/delete/transition/upgrade`；GD `gd_only_new/gd_only_merge/gd_tsp_merge`；shadow `gd_stable_shadow/gd_track_depth_shadow`；TTL `gd_ttl_shadow`）、`origin`、`lifecycle_state`、`gd_hits`、`association`、`gd_incremental`、`upgrade_reason`（`gd_tsp`/`gd_only_tsp_merge`/`base_upgrade`，普通升级另带 `trigger=merge|frame`） |
| `memory_events.jsonl`（gd-only conf） | `gd_only_merge`、`gd_only_conf_threshold`、`gd_only_conf_margin`、`gd_only_conf_would_reject`、`gd_only_conf_gate_enabled` |
| `memory_events.jsonl`（shadow） | `shadow_id`、`shadow_kind`、`shadow_only`、`proposal_id`、`memory_id`、`centroid`（GD proposal，用于 TP）、`track_centroid`、`track_tp`（stable track 的 GT）、`track_lifecycle_state`、`track_num_obs`、`track_sigma_c_latest`、`depth_ok`、`conf`；后续轨迹按 `memory_id` 关联 `track_frames`，未来 GD 观测关联 `gd_proposals` |
| `memory_events.jsonl`（revive） | `revived=true`、`revive_sigma_c`、`revive_num_obs_before/after`、`revive_dist_to_old_centroid`、`revive_age_steps`、`revive_detect_dist`、`revive_radius/window/min_sigma` |
| `memory_events.jsonl`（stop_blacklist） | `event=stop_blacklist`、`memory_id`、`near_turns`、`distance`、`stop_gate_turns/stop_gate_confirm_radius`、`lifecycle_state`、`origin`、`gd_hits`、`num_obs`、`tp`；用于判断黑名单是否误伤可升级目标 |
| `stop_shadow.jsonl` | 被选中的 confirmed（含 gd-tsp / promoted gd-only）与 gd-only 的只读停止样本：`goal_distance`、`num_obs`、`gd_hits`、`sigma_c/band`、`stale_steps`、`tp/goal_tp`，以及 `would_stop_r09/r12`、`gdhits>=3/5`、`numobs>=3/5` 等假设阈值；`stop_eligible_current` 标明该步是否真的可停。注意 `[STOP]` 行里的 `tsp3d_confirm` 只是**本帧 TSP3D 共位**，不是目标 `origin`；判断 STOP 资格看 `stop_eligible_current` 与 `origin=gd-tsp`，不要用 `tsp3d_confirm` 推断来源 |
| `track_frames.jsonl` | 轨道级 `origin`、`gd_only`、`gd_tsp`、`gd_hits`、`gd_last_step`、`gd_age`、`gd_conf_max/gd_conf_last`、`lifecycle_state`、`stop_eligible` |
| `nav_goals.jsonl` | 被选目标的 `origin`、`gd_hits`、`gd_tsp`、`lifecycle_state`、`stop_eligible`、`switch_reason` |
| `episodes.jsonl` | `cfg` 中的 GD 参数、升级阈值、S-penalty 参数 |

**后续统计口径**：对 `memory_events.jsonl` 中 `event in (gd_only_new, gd_only_merge)` 的行，按 `target_class × event` 汇总 `n / tp / fp / precision`，并附 `conf` 分位数（0.40–0.45、0.45–0.50、0.50–0.55、≥0.55）。`gd_only_new` 与 `gd_only_merge` 分开看，不能把二次命中混入首次探索精度；TP/FP 使用 writer 写入的 `tp`，以事件 `centroid` 标注。

### 4.8 代码范围与非目标

- `target_memory_manager.py`：`GD` 状态、`origin`、`apply_gd_evidence`、gd TTL 开关/shadow、`force_confirmed/block_stable`、revive 判据日志。
- `grounding_dino/mechanism.py`：base gate、gd-only/gd-tsp 路由、depth 失败例外、机制日志。
- `tsp3d_objectnav_policy.py`：机制初始化、每步调用、gd-only 选择与日志、配置传递。
- `config/experiments/vlvm_objectnav_hm3d.yaml`：上述参数，包括 `gd_depth_failed_gd_only`。
- `scripts/_nav_s6_gdmech.sh`：三场景实验入口，配置以 YAML 为准。
- **结果文档**：[VLVM-S6gd1-exp.md](VLVM-S6gd1-exp.md)、[VLVM-S6gd2-exp.md](VLVM-S6gd2-exp.md)、[VLVM-S6gd3-exp.md](VLVM-S6gd3-exp.md)、[VLVM-S6gd4-exp.md](VLVM-S6gd4-exp.md)、[VLVM-S6gd5-finalexp.md](VLVM-S6gd5-finalexp.md)。
- **S6gd3 实验**：`outputs/s6_gd3_restart_2026-10-28_182915`（promoted gd-only 归一为 gd-tsp；STOP shadow 覆盖所有选中 confirmed/gd-only；排序 `gd-tsp > gd-only > confirmed > provisional`）。
- **S6gd4 收尾实验**：`outputs/s6_gd4_depthrelax_2026-10-29_015327`（confirmed 提到 gd-only 之前；gd-only conf gate 保持关闭；该 run 的 depth 失败只对 confirmed/provisional track merge 放开）。S6gd4 之后已开放 gd-only depth；见 §4.5.1；中间档排序见 [VLVM-S7.md](VLVM-S7.md) §2。
- **S7 延续**：[VLVM-S7.md](VLVM-S7.md)；field 离线结果：[VLVM-S7f0-exp.md](VLVM-S7f0-exp.md)。
- **非目标**：不把 GD 当独立覆盖源；不把 GD conf 写进 $\sigma_c$；不改变 TSP3D admission/fusion/S-penalty；不把 GD 缺失当负证据；不在本轮接入 STOP。

### 4.9 后续阶段边界

field 机制（目标条件语义－空间信息场）和轻量离线 ranker 已移入 [VLVM-S7.md](VLVM-S7.md)。S6 只保留 GD proposer 与导航链路，不包含 field 机制。
