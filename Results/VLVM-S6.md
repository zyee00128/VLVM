# Project VLVM Stage 6：GD Proposer 机制调整

> 定位：S5 最终机制已定稿，S6 只调整 GD proposer。本文按 nav pipeline 记录问题定位与 GD 调整顺序；GD 的具体机制、参数与实验结论在讨论定稿后写入 §4，未跑通前不写入机制表。

> 以 S5 定稿为唯一基线，只改 GD，不同时改 TSP3D 候选的准入、$\sigma_c$ 带通、停止门槛、稳定池或 fusion。policy 当前不调用 GD，S5 四场景结果作为对照；旧 GD 代码独立保留，待重新设计。

## 1. S5 定稿基线

S6 继承 [VLVM-S5.md](VLVM-S5.md) 的最终机制，不重新讨论准入、目标选择、停止和稳定池。基线固定为：

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

S5 最终四场景实测：164/396（SR 41.41%），结果与离线标定见 [VLVM-S5final.md](VLVM-S5final.md)。

关键参数：

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

## 2. 问题定位（按 nav pipeline 排序）

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

## 3. GD 调整顺序

> 只改 GD，不改 TSP3D 候选的准入、$\sigma_c$ 带通、停止门槛、稳定池与 fusion。以下为顺序与判定门；具体机制待讨论定稿后写入 §4，不在本文提前落参数。

### 3.1 离线标定（前置，零行为）

- 形式：S5 定稿配置 + GD 开启但只落日志，不写记忆、不影响动作。
- 目的：在真实帧与位姿上测 GD 的召回率、TP/FP 正票率、类别差异与成本。
- 先测量：419 集无 TP 中 GT 目标的轨迹可视率（至少一帧进入 FOV + depth）。它是 GD 覆盖的真实上限；不可视部分 GD 无解，探索已冻结。
- 判定门：
  - 覆盖：224 集可行无 TP 中，GD proposal 命中 GT 0.5 m 内 ≥50%，才进入 3.2。
  - 区分：TP 轨道与 FP 轨道的 GD 正票率差 ≥0.3，才进入 3.3。
  - 停止：末端 `confirmed` TP/FP 的正票率差 ≥0.4，且预测 precision ≥85%，才进入 3.4。
- 未过门的处理：不进入对应阶段；覆盖门未过时 GD 只做窄类别/近场补召，不改主线。

### 3.2 目标覆盖（第一优先级）

- 定位：GD 作为第二覆盖源，只负责“提议在哪看 / 查哪个区域”，确认权归 TSP3D。
- 约束：GD 单独不产生 `confirmed`，不可停；GD 不写 $\sigma_c$；不改变 TSP3D 候选的准入。GD-origin 轨道是否新增独立准入分支，待 §4 定稿。
- 验收：224 集可行无 TP 的 success 增量、FP 增量、never_saw 变化、steps。

### 3.3 目标选择与排序（第二优先级）

- 定位：GD 正票作为正交证据，只在 3.2 通过后接入选择/生命周期。
- 约束：不替代 $\sigma_c$ 带通与距离排序；不进入 S-penalty / fusion；正票需带新信息门。
- 验收：nav goal TP 占比、TP+FP 共存集 SR、confirmed 池 FP。

### 3.4 最终目标停止（延后）

- 定位：停止复核只在 3.1 的停止分离门通过后单独成臂。
- 约束：S5 停止机制保持不变；不把 GD 缺失当负证据。
- 验收：停止 precision、wrong stop 集数、TP 损失、steps。

## 4. GD 具体机制（待定稿）

当前 `vlfm/policy/tsp3d_objectnav_policy.py` 已移除 GD 的 import、参数和调用；`vlfm/tsp3d_models/grounding_dino` 作为独立代码保留，等待重新设计。具体机制尚未定稿，讨论确认后在本节追加。
