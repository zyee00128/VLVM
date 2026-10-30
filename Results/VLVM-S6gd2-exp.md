# VLVM-S6gd2：GD 机制三场景复跑

> 运行目录：`outputs/s6_gdmech_3scene_2026-10-27_restart`  
> 运行时间：`2026-10-27 23:02` 至 `2026-10-28 05:17`，约 6:18:14  
> 场景：`5cdEh9F2hJL` / `QaLdnwvtxbs` / `mL8ThkuaVTM`，共 3 × 99 = 297 episodes  
> field 原始归档：297 ZIP，约 42 GB

## 1. 本版机制

| 项 | S6gd2 |
| :--- | :--- |
| 普通升级 | 只保留 `upgrade_min_obs=4`，删除 close 变体与距离升级 |
| `gd-tsp` 关联 | track-only：只接受已进入 memory 的 `confirmed/provisional` track |
| candidate-only | 只记 `association=candidate_only`，不直接 confirmed |
| stable | S6gd2 直接 `blocked`；当前代码只写 `gd_stable_shadow`，仍不升级 |
| gd-only 提升 | 后续 admitted TSP3D candidate 匹配 gd-only 时提升为 confirmed |
| 选择判据 | 优先 `gd_upgraded` / `promoted_from_gd_only` 的增量证据 |
| 写入坐标 | track 命中使用 track centroid，避免 candidate/proposal 偏移造成重复条目 |

S6gd2 运行时：`provisional_sigma=0.55`、`gd_only_conf_min=0.50`、gd-only 类别为 `bed,chair,couch,toilet`。当前代码（下一轮实验，尚未启动）：`gd_only_classes=""`（全类别）；`enable_gd_only_conf_gate=false`，只记录 observed conf 与 0.50/0.45 参考阈值；`enable_gd_ttl=false`，只写 `gd_ttl_shadow`；`stable_revive_radius/window=0.8/20`；`gd_stable_shadow` / `gd_depth_shadow` 开启。

> 日志兼容：S6gd2 运行日志中 merge-time 普通升级仍标为 `close_reobservation`，逐帧路径为 `frame_upgrade`；这是历史标签，实际条件只有 `num_obs>=4`，没有距离门。当前代码已统一为 `base_upgrade`，用 `trigger=merge|frame` 区分触发时机。本 run 的 `gd_tsp_new=0`，当前代码已改为 gd-tsp 只能合并既有 track，不再产生该标记。


## 2. 总体结果

| 指标 | S6gd2 | S6gd1 | S5-S6 基线 |
| :--- | ---: | ---: | ---: |
| success | **130 / 297** | 116 / 297 | 117 / 297 |
| SR | **43.77%** | 39.06% | 39.39% |
| Oracle | **129 / 297（43.43%）** | 117 / 297（39.39%） | 122 / 297（41.08%） |
| SPL | **0.2008** | 0.1798 | 0.1803 |
| SoftSPL | **0.2782** | 0.2832 | 0.2741 |
| steps | 144.60 | 130.29 | 146.40 |

**结果对比**

与 S6gd1：

| 成功配对 | episodes |
| :--- | ---: |
| both success | 101 |
| only S6gd2 success | 29 |
| only S6gd1 success | 15 |
| both fail | 152 |

- McNemar 精确双侧约 `p=0.049`，success 提升处于边缘显著。

| Oracle 配对 | episodes |
| :--- | ---: |
| both oracle | 117 |
| only S6gd2 oracle | 12 |
| only S6gd1 oracle | 0 |
| neither oracle | 168 |

- 新 run 的 Oracle 完全覆盖 S6gd1：没有丢失任何旧 Oracle，新增 12 集（`p≈0.0005`）。

与 S5-S6：

- success 配对：both 98 / only S6gd2 32 / only S5 19 / neither 148（`p≈0.092`）。
- Oracle 配对：both 117 / only S6gd2 12 / only S5 5 / neither 163（`p≈0.143`）。

因此相对 S6gd1 的提升更明确；相对 S5 的 success/Oracle 提升尚未达到同样统计强度。

## 3. 分场景

| 场景 | S6gd2 success / Oracle | S6gd1 success / Oracle | S5-S6 success / Oracle |
| :--- | ---: | ---: | ---: |
| `5cdEh9F2hJL` | **45 / 59** | 33 / 53 | 39 / 56 |
| `QaLdnwvtxbs` | **76 / 66** | 76 / 60 | 67 / 63 |
| `mL8ThkuaVTM` | **9 / 4** | 7 / 4 | 11 / 3 |

- 增益主要来自 `5cd`：success +12（vs S6gd1）。
- `QaL` success 持平，但 Oracle +6。
- `mL8` 仍是最弱场景：9 success，且 Oracle 只有 4；其中 63 集归为 likely-infeasible。

## 4. 分目标类别

| 类别 | n | S6gd2 success / Oracle | S6gd1 success / Oracle | S5-S6 success / Oracle |
| :--- | ---: | ---: | ---: | ---: |
| bed | 70 | **43 / 43** | 29 / 32 | 39 / 42 |
| chair | 68 | **41 / 43** | 35 / 42 | 34 / 38 |
| couch | 73 | 26 / 27 | 26 / 27 | 25 / 27 |
| toilet | 43 | 6 / 16 | 6 / 16 | 2 / 15 |
| tv | 43 | **14 / 0** | 20 / 0 | 17 / 0 |

关键变化：

- bed：success 从 29 提升到 43，Oracle 从 32 到 43；
- chair：success 35 → 41，Oracle 42 → 43；
- couch / toilet：success 与 Oracle 均持平；
- tv：20 → 14，Oracle 仍为 0；这是本版的主要类别回退。

## 6. 失败原因与覆盖分解

| failure_cause | S6gd2 | S6gd1 | S5-S6 |
| :--- | ---: | ---: | ---: |
| `did_not_fail` | 130 | 116 | 117 |
| `false_positive` | **81** | 97 | 86 |
| `false_negative` | 9 | 9 | 15 |
| `never_saw_target_did_not_travel_stairs_feasible` | 14 | 13 | 20 |
| `never_saw_target_did_not_travel_stairs_likely_infeasible` | 63 | 62 | 59 |

`false_positive` 81 + `never_saw` 77 = 158 / 167 个失败（94.6%），两者仍是主因。若只算可达失败，FP 81 + feasible never_saw 14 = 95 / 167（56.9%）；63 个 likely-infeasible 不应作为机制主 KPI。它们不是同一层问题：`false_positive` 既可能是“看到目标但先停错”，也可能是“目标从未进入可达/已探索区域，只能追一个 FP”。需要把 FP 再拆开。

### 6.1 FP 的覆盖分解

以 `candidates.jsonl` 中是否存在 `tp=true` 候选、以及 `episodes.jsonl` 的 GT `oracle_success`（目标 bbox 是否进入过 fog-of-war 探索区）交叉统计：

| FP 口径 | episodes | 占 81 FP | 解释 |
| :--- | ---: | ---: | :--- |
| FP 总数 | 81 | 100.0% | — |
| 无 TP candidate | 51 | 63.0% | TSP3D 从未给出正确候选 |
| 无 TP track frame | 52 | 64.2% | 正确目标从未形成轨道 |
| 无 TP nav goal | 55 | 67.9% | 正确目标从未被选为导航目标 |
| oracle=0 | 66 | 81.5% | 目标从未进入已探索区域 |
| oracle=0 且无 TP candidate | **48** | **59.3%** | 覆盖 + 检测共同失败，最像“没见到目标” |
| oracle=0 但有 TP candidate | 18 | 22.2% | 有候选但目标仍未进入 fog；含 TP 判定/覆盖口径歧义 |
| oracle=1 且有 TP candidate | **12** | **14.8%** | 目标已探索、TP 已出现，仍选错/停错 |
| oracle=1 但无 TP candidate | 3 | 3.7% | 目标已探索，但检测器没给 TP |

`tv` 的 `oracle_success` 在本 run 和历史 S6gd1/S5-S6 中均为 0，不能把 `tv` 的 oracle=0 直接解释为“没看到目标”。排除 `tv` 后，57 个 FP 中仍有 **35 个 oracle=0 且无 TP candidate**（61.4%），12 个 oracle=1 且有 TP candidate（选择/停止），7 个 oracle=0 但有 TP candidate，3 个 oracle=1 但无 TP candidate。

按场景看“无 TP candidate 的 FP”：

| 场景 | FP 总数 | 无 TP candidate | 说明 |
| :--- | ---: | ---: | :--- |
| `5cdEh9F2hJL` | 42 | **33** | 覆盖失败最集中 |
| `QaLdnwvtxbs` | 21 | 5 | 主要是选择/停止，而非无目标 |
| `mL8ThkuaVTM` | 18 | **13** | 覆盖失败次集中 |

`never_saw` 77 集中：74/77 没有任何 TP candidate，3 集有 TP candidate 但仍未覆盖/未选中；其中 63 集为 likely-infeasible，14 集为 feasible（toilet 7、tv 5、couch 1、chair 1）。

结论：

**“主要问题转向目标覆盖”这个判断基本正确。** 81 个 FP 中至少 48 个（59.3%）同时满足 oracle=0 和无 TP candidate；若把 `never_saw` 也算入覆盖失败，覆盖/探索是第一位。

仍有 12 个 FP 是“目标已探索、TP 已出现但先停错”，3 个是“目标已探索但 TSP3D 未检出”，这部分不能靠继续压 FP 解决，需要目标选择、停止和检测覆盖分别处理。

## 7. GD / Fallback

### 7.0 Pipeline

```text
GD proposal
  -> base gate: target class / conf>=0.40 / depth / FOV / dist<=5.0 m
  -> association: track / candidate / none
  -> gd-tsp: 命中 confirmed/provisional track，merge-only
  -> gd-only secondary gate: class whitelist / conf>=0.50 / HARD_CONFIRMED
  -> memory lifecycle: new / merge / upgrade / transition / revive / delete
  -> navigation selection: state tier / gd_incremental / sigma band / gd_tsp / distance
```

本 run 共 5,493 个 GD decision。`depth_not_ok` 在关联前拒绝 964 个，进入关联判定 4,529 个；随后 gd-only 二次 gate 拒绝 2,187 个，最终 gd-only 1,245、gd-tsp 1,096、blocked 1。

### 7.1 Association 标记的含义与导航作用

`association` 由 `match_track.same_object` / `match_candidate.same_object` 决定：`gd_pos` 是 track 级正票，`gd_candidate_pos` 是 candidate 级正票；只有 track 级可以走 gd-tsp。它本身不是目标选择的直接排序键，而是通过 `lifecycle_state`、`origin`、`gd_hits`、`gd_upgraded` / `promoted_from_gd_only` 影响后续状态与选择键。

| association | n | 本 run 路由 | 含义 | 后续导航作用 |
| :--- | ---: | :--- | :--- | :--- |
| `none` | 2,956 | `base_rejected` 2,702；`gd_only` 254 | 没有同物体 track，也没有同物体 candidate | 无 track 证据；允许时只能进入 gd-only 探索池 |
| `track_gd` | 1,293 | `gd_only` 973；`base_rejected` 320 | GD 命中已有 `lifecycle=gd` 的 gd-only 条目 | 通常合并 gd-only 证据并保持 `gd`；本 run 959 merge / 14 new（见下）；只在没有 confirmed/provisional 时可选，永不可停 |
| `track_confirmed` | 1,016 | `gd_tsp` 1,016 | GD 命中已有 `confirmed` track | 只补 GD 证据、`origin=gd-tsp`、`gd_hits+1`；不改 `num_obs`/质心/分数；选择键 `gd_tsp` 生效 |
| `candidate_only` | 147 | `base_rejected` 129；`gd_only` 18 | GD 只命中尚未成轨的 TSP candidate | 不确认、不建 track；candidate 必须先通过 admission 成为 track，才能走 gd-tsp |
| `track_provisional` | 80 | `gd_tsp` 80 | GD 命中 `provisional` track | 强制提升 `confirmed`，写 `gd_upgraded`/`gd_incremental`；选择时排在同层 sigma band 前 |
| `track_stable` | 1 | `blocked` 1 | GD 命中 `stable` 条目 | 当前完全阻断，不吸收证据、不复活、不升级 |

补充：

- `none` 的 2,956 中有 964 是 depth 拒绝（尚未进入关联/二次判据），其余才进入后续分流。
- `track_gd` 的 973 次 gd-only 写入中有 959 次 merge、14 次 new；14 次 new 说明 track 命中的 gd 条目与 `accumulate_gd_only` 的最近邻没有对齐。后续应同样给 gd-only merge 加 expected memory-id 保护。
- `track_gd` 的 320 个 `base_rejected` 全部是 `gd_only_conf_below_min`，即已有 gd-only 条目的再次 GD 命中被同一个 0.50 门槛拒绝；不是类别拒绝。当前代码已关闭 conf 门控，这 320 条下一轮会进入 gd-only 路径并记录 conf/TP/FP。
- `candidate_only` 的 129 个拒绝全部是 `tv` 的 `gd_only_class_disabled`；另外 18 个非 tv candidate-only 进入了 gd-only。
- `track_stable` 只有 1 次，发生在 `QaLdnwvtxbs:0`（chair，episode 已 success），因此本 run 不足以判断“stable 也应该升级”是否有效。
- 代码还保留 `track_<state>` 兜底标签（本 run 未出现）；它不会进入 gd-tsp，只会按 gd-only 二次判据处理，不能当作已确认轨道。
- `association` 只解释“GD 和谁匹配”；真正的选择效果由 `state_tier`、`gd_incremental`、`gd_tsp`、`gd_hits` 决定。

### 7.2 gd-only 二次判据

GD base gate 之后，只有不属于 `track_confirmed/provisional` 的 proposal 才进入 gd-only 二次判据：类别白名单、`conf>=0.50`、`HARD_CONFIRMED` 几何。

| 二次判据 | n | 构成 | 说明 |
| :--- | ---: | :--- | :--- |
| `gd_only_class_disabled` | **1,349** | `tv` 1,349（none 1,248 + candidate_only 101） | S6gd2 旧白名单；当前代码空 whitelist = 全类别，不再有这类拒绝 |
| `gd_only_conf_below_min` | **831** | none 483 + `track_gd` 320 + candidate_only 28 | S6gd2 旧口径 new/merge 共用 0.50；当前关闭 conf gate，只记录 observed conf 与 new 0.50 / merge 0.45 参考阈值 |
| `gd_occ_geom_unknown` | 7 | chair 5、bed 1、couch 1 | S6gd2 旧硬占据门拒绝；当前代码允许 `occ_passed=True` 的 unknown，只拒绝 `occ_passed=False` |
| 通过并写 gd-only | **1,245** | none 254 + candidate_only 18 + `track_gd` 973 | 其中 new 286 = none 254 + candidate_only 18 + `track_gd` 14；merge 959 全部来自 `track_gd` |

gd-only 写入的目标精度：

| 类别 | `gd_only_new` TP/FP | precision | `gd_only_merge` TP/FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| bed | 56 / 47 | 54.4% | 223 / 124 | 64.3% |
| chair | 48 / 15 | **76.2%** | 176 / 12 | **93.6%** |
| couch | 47 / 36 | 56.6% | 100 / 61 | 62.1% |
| toilet | 30 / 7 | **81.1%** | 247 / 16 | **93.9%** |
| tv | — | — | — | — |
| 合计 | 181 / 105 | 63.3% | 746 / 213 | 77.8% |

上表是 S6gd2 旧四类白名单的结果。当前代码已全类别放开，`tv`/`potted plant` 的 gd-only precision 尚未被本 run 测量，必须在下一 run 单独监控；不能把 bed/chair 的精度直接外推到 tv。

按 gd-only 写入后的累计 `gd_hits` 分层：

| `gd_hits` | TP/FP | precision |
| :--- | ---: | ---: |
| ≥1 | 927 / 318 | 74.5% |
| ≥2 | 730 / 203 | 78.2% |
| ≥3 | 597 / 127 | 82.5% |
| ≥4 | 493 / 81 | 85.9% |
| ≥5 | 408 / 47 | 89.7% |

判读：

- gd-only 的类别差异明显：chair/toilet 在 new 阶段已有 76%–81% precision，bed/couch 只有 54%–57%；不能用一个全局阈值同等对待。
- `gd_hits` 越高精度越好，但**首次进入导航的 gd-only 目标一定只有 `gd_hits=1`**。因此 `gd_hits>=2/4` 更适合作为“继续追踪/提升优先级”的条件，不应作为“首次探索必须有多次命中”的硬门，否则会切断探索。
- 831 个 `gd_only_conf_below_min` 中有 320 个是已有 gd-only 的再观测。S6gd2 时 new/merge 共用 0.50；当前先关闭 conf gate，按类别记录 `gd_only_new` / `gd_only_merge` 的 conf 与 TP/FP，再决定是否启用及阈值。

### 7.3 gd-tsp：只做既有 track 的 merge

本 run 的 `gd_tsp_new=0`。gd-tsp 全部是既有 track merge；文档不再把 `gd_tsp_new` 作为有效路径，当前代码也增加了 merge-only 保护。

| association | n | TP | FP | precision |
| :--- | ---: | ---: | ---: | ---: |
| `track_confirmed` | 1,016 | 827 | 189 | 81.4% |
| `track_provisional` | 80 | 57 | 23 | 71.3% |
| 合计 | 1,096 | 884 | 212 | **80.7%** |

升级事件按 `memory_id` 关联 `track_frames.jsonl` 重新标注：

| upgrade_reason | n | TP | FP | precision | 实际触发 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `gd_tsp` | 80 | 57 | 23 | 71.3% | GD 命中 provisional track，强制 confirmed |
| `gd_only_tsp_merge` | 55 | 47 | 8 | **85.5%** | 后续 TSP3D candidate 合并 gd-only，提升 confirmed |
| `close_reobservation` | 32 | 21 | 11 | 65.6% | S6gd2 历史标签；实际是 merge-time 普通升级，32 条全部 `num_obs=4`，距离 1.32–4.60 m，无距离门；当前为 `base_upgrade(trigger=merge)` |
| `frame_upgrade` | 5 | 2 | 3 | 40.0% | S6gd2 历史标签；逐帧扫描命中 `num_obs>=4`；当前为 `base_upgrade(trigger=frame)` |
| 合计 | 172 | 127 | 45 | 73.8% | — |

`close_reobservation` 的“close”是旧机制遗留命名，不表示近距或距离升级；当前代码已把 merge-time 与逐帧路径统一为 `base_upgrade`，并用 `trigger=merge|frame` 区分。两者阈值都是 `num_obs>=4`，不再有 `close`/`frame` 两个独立 reason。按新口径合并后，`base_upgrade` 为 37 条（TP 23 / FP 14），总升级仍为 172 条。

### 7.4 fallback：transition / revive / delete 的目标数据

TP/FP 用 `memory_id` 关联 `track_frames.jsonl`；`gd_ttl` 是 gd-only 的 TTL，不是 stable 删除。

| 生命周期事件 | n | TP / FP | TP 率 | 主要目标结构 / 说明 |
| :--- | ---: | ---: | ---: | :--- |
| transition `near_miss_downgrade` | 48 | 12 / 36 | 25.0% | confirmed 10/13，provisional 2/23；chair FP15、tv FP10、couch FP9 |
| transition `stale_demote` | 25 | 9 / 16 | 36.0% | confirmed 8/3，provisional 1/13；tv FP8、chair TP7/FP4 |
| revive（`merge` 且 `revived=true`） | 8 | 4 / 4 | 50.0% | S6gd2 旧 revive 0.5 m/10 步；当前实验 0.8 m/20 步；chair 4/3、couch 0/1；stable→provisional，保留 num_obs/分数历史 |
| delete `gd_ttl` | 69 | 37 / 32 | 53.6% | S6gd2 旧 TTL=60；当前 `enable_gd_ttl=false`，不删除，只写 `gd_ttl_shadow`；bed FP22、toilet TP16、couch TP10、chair TP7/FP3 |
| delete `stable_stale_purge` | 40 | 11 / 29 | 27.5% | chair TP7/FP11、tv TP2/FP11、toilet TP2/FP4、bed FP2、couch FP1 |
| delete `stable_miss_delete` | 4 | 0 / 4 | 0.0% | chair FP2、couch FP1、tv FP1 |
| stable 删除合计 | 44 | 11 / 33 | 25.0% | `stable_total_miss` 本 run 为 0 |

本 run 阈值：`confirmed/provisional → stable` 均为 3 次 near_miss；`stale` 60 步；`stable` 删除为 stale 60 / 连续 miss 5 / 累计 miss 8。

stable 目标后续出现的 radius/window 模拟（73 个 stable transition；严格只统计 **stable transition 之后**、已准入且 `sigma_c>=0.50` 的候选；gap 从最后一次观测算；取半径窗口内最早候选；不是闭环反事实实验）：

| radius | window=10 | window=15 | window=20 |
| ---: | ---: | ---: | ---: |
| 0.4 m | 6 (4 TP / 2 FP) | 7 (4 / 3) | 7 (4 / 3) |
| 0.5 m | 9 (5 / 4) | **11 (6 / 5)** | **11 (6 / 5)** |
| 0.6 m | 9 (5 / 4) | 12 (6 / 6) | 12 (6 / 6) |
| 0.7 m | 9 (5 / 4) | 13 (7 / 6) | 13 (7 / 6) |
| 0.8 m | 10 (6 / 4) | 14 (8 / 6) | 15 (8 / 7) |
| 1.0 m | 12 (5 / 7) | 16 (7 / 9) | 17 (7 / 10) |

window=20 时，任意半径共 20 个 stable 目标有后续候选（9 TP / 11 FP）；其余 53 个没有：33 个 transition 的 `age_steps>=20`，窗口在 stable 建立前已耗尽；20 个 `age_steps<20` 但没有合格候选。window=20 的半径分布：≤0.5 m 11 个（6/5），0.5–0.8 m 4 个（2/2），0.8–1.0 m 2 个（0/2），>1.0 m 3 个（1/2）。

gap 从最后一次观测算，window=20 内首次候选的分布：

| gap | n | TP / FP |
| ---: | ---: | ---: |
| 1–5 | 6 | 3 / 3 |
| 6–10 | 6 | 2 / 4 |
| 11–15 | 7 | 4 / 3 |
| 16–20 | 1 | 0 / 1 |

按 stable 建立后的步数看，18 个在 1–5 步内出现，1 个在 6–10 步，1 个在 11–15 步；即“stable 后还能再看到”的目标主要是在 stable 建立后立刻出现。按类别：chair 31 个中 13 个（7 TP / 6 FP），tv 21 个中 3 个（0 / 3），couch 9 个中 3 个（1 / 2），bed 6 个中 1 个（1 / 0），toilet 6 个中 0 个。

首次后续候选 gap>20 的分布（73 个 stable 中 30 个首次出现在 20 步之后，23 个完全没有后续候选）：

| gap | n | TP / FP | 候选时 stable 仍存活 |
| ---: | ---: | ---: | ---: |
| 21–40 | 0 | 0 / 0 | 0 |
| 41–60 | 3 | 2 / 1 | 3 |
| 61–90 | 12 | 7 / 5 | 0 |
| >90 | 15 | 5 / 10 | 0 |

这 30 个 gap>20 的候选中，只有 3 个发生在 stable 删除之前（全部在 41–60 步），27 个在 stable 已删除后出现；半径≤0.5 m 的晚候选有 6 个（4 TP / 2 FP），但其中只有 1 个在删除前，且是 FP。因此单独把 window 从 20 放宽到 40 没有新增；放宽到 60 在当前 0.5 m 下只多 1 个 FP、没有 TP。要让晚候选真正可 revive，必须同时延长 stable 的 stale purge 生命周期，而不是只改 window；这会明显增加 FP，暂不采用。

实际 revive 8 条的最大距离为 0.449 m、最大 gap 为 9 步；8 条全部来自 `age_steps=2–5` 的 stable transition，stable 后 1–6 步出现。0.5 m/15 相比 0.5 m/10 多 1 TP、多 1 FP；15→20 在 0.5 m 下不再增加。半径放到 0.6 m/15 只多 1 FP、不多 TP；0.8 m/20 为 8 TP / 7 FP，比 0.5 m/15 多 2 TP / 2 FP。离线数据更支持 **0.5 m / 15 步**；但本轮代码按用户指定改为 **0.8 m / 20 步**，用于记录更大半径/窗口下 revive 的 `sigma_c`、`num_obs`、`dist`、`age` 等判据；是否回退由本轮闭环结果决定。

判读：

- transition 73 条里 52 条是 FP（71.2%），stable 删除 44 条里 33 条是 FP（75.0%），fallback 主要清理的是错误锁定，不是大量丢 TP。
- revive 8 条为 4 TP / 4 FP，样本小；chair 的 7 条 revive 为 4 TP / 3 FP，说明“stable 被再次看到后复活”不是纯噪声。
- `stable_stale_purge` 仍删除 11 个 TP；其中 chair 7、tv 2、toilet 2。若后续要继续提高覆盖率，这 11 个 TP 可以作为 stable 复核的 shadow 对象，而不是直接放宽全局删除阈值。


### 7.5 导航选择

**S6gd2 运行时的选择键**（从高到低）是：

```text
state tier (confirmed > provisional > gd)
  -> gd-only 内部 hits>=2
  -> gd_incremental (gd_upgraded / promoted_from_gd_only)
  -> sigma_c band
  -> gd_tsp 或 gd_hits>=2
  -> 距离
  -> num_obs / freshness
```

> **下一次启动的代码**改为四挡：`gd-tsp/gd-only-tsp > gd-only > confirmed > provisional`；内部 key 见 `VLVM-S6.md` §4.5。当前正在运行的三场景实验仍使用上表旧排序；四挡排序需要下一次启动才生效。

四挡优先级依据（S6gd2 日志，TP 为 nav_goal 行标签）：

| 类别 | nav_goal 行 TP / 总数 | 行 precision | 首次目标 TP/FP | 首次目标最终 success |
| :--- | ---: | ---: | ---: | ---: |
| `gd-tsp` | 1,332 / 1,736 | 76.7% | 10 / 7 | 9 / 17 |
| `gd-only-tsp` | 50 / 56 | 89.3% | 0 / 0 | 0 / 0 |
| `gd-only` | 1,316 / 2,089 | 63.0% | 65 / 61 | 80 / 126 |
| TSP `confirmed` | 321 / 742 | 43.3% | 17 / 13 | 17 / 30 |
| TSP `provisional` | 149 / 887 | 16.8% | 20 / 30 | 23 / 50 |

同一步同时有 gd-only 与 TSP confirmed 的 396 个 step 中，gd-only 任一条 TP 的 step 为 295，confirmed 任一条 TP 的 step 为 195；按 gd-only 优先会在 step-level 净增加约 100 个 TP 选择，但 396 个 step 的 confirmed 都可 STOP，因此需要额外测试“近距离 confirmed 的 STOP 例外”。

### 7.5.1 stop_blacklist 与可停复核（实测推断）

> 旧日志没有显式 `stop_blacklist` 事件；以下由 nav_goals 中非 stop-eligible 目标连续第 4 次进入 1.2 m 推断，属于近似统计。

- 推断 11 次拉黑，覆盖 10 个 episode：gd-only 7 次、provisional 4 次。
- 这些 episode 最终 success 6/10；gd-only 拉黑事件 5/7 最终 success，provisional 1/4。
- 拉黑后只有 1 个 provisional 条目后来变成 confirmed，0 个 gd-only 条目后来变成 confirmed；说明本轮黑名单主要是在止损，不是大量误伤可升级目标。
- 纯 TSP confirmed 最终目标共 44 个 episode：TP 15、FP 27、success 17；按 `num_obs` 拆分：
  - `num_obs<3`：29 个，TP 4、FP 25、success 8；
  - `num_obs>=3`：15 个，TP 11、FP 4、success 9；
  - `num_obs>=5`：10 个，TP 9、FP 1、success 6。
- 当前所有 confirmed 都可停，太宽；纯 TSP confirmed 至少应要求 `num_obs>=3`，更保守可用 `num_obs>=5`。gd-tsp/gd-only-tsp 继续优先可停。
- gd-only 当前不可停；最终 gd-only 目标 21 个，TP 15，但 success 0。若要尝试 gd-only 可停，建议只对 `gd_hits>=5` 做小范围消融，并要求进入 `pointnav_stop_radius=0.9 m` 后再停；不要全局放开。

**下一次启动的暂定停止规则**：只有 `gd-tsp` track 和 `gd-only-tsp` 可停；**纯 TSP confirmed 暂时取消可停**；`gd-only` / `provisional` 不可停。纯 confirmed 与 gd-only 的每步目标写 `stop_shadow.jsonl`，记录距离、`num_obs`、`gd_hits`、`sigma_c`、GT 命中及多组假设阈值，用于确定下一版可停条件。

**num_obs vs freshness**：在 track_frames 的同一步/同层/同 0.5 m 距离桶内比较 TP-FP pair，`num_obs` 正确排序 185 对，`freshness` 85 对；进一步限定同 sigma band / gd_hits 后为 29 vs 21。因此排序中 `num_obs` 在前，`freshness` 仅作同 `num_obs` 的次级 tiebreak。

| 目标来源 | nav_goal 行数 | 出现 episode 数 | 行 TP / FP | 行 precision | 首次目标 episode / success |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `tsp3d`（origin=None） | 1,629 | 127 | 470 / 1,159 | 28.9% | 80 / 40 |
| `gd-tsp` | 1,736 | 149 | 1,332 / 404 | **76.7%** | 17 / 9 |
| `gd-only` | 2,145 | 135 | 1,366 / 779 | 63.7% | **126 / 80** |

首次目标的细分：

| 首次来源 | 首次 TP 数 | 首次 FP 数 | 最终 success |
| :--- | ---: | ---: | ---: |
| `tsp3d` | 37 | 43 | 40 / 80 |
| `gd-tsp` | 10 | 7 | 9 / 17 |
| `gd-only` | 65 | 61 | 80 / 126 |

补充：

- 首次选 gd-only 的 61 个 FP 中，有 32 个最终仍 success；首次选 gd-tsp 的 7 个 FP 全部失败。gd-only 更像“探索起点”，gd-tsp 更像“确认后的高精度目标”。
- 含任意 gd-tsp 目标的 149 集 success 110（73.8%）；含任意 gd-only 目标的 135 集 success 84（62.2%）。这是相关性，成功轨迹本身也会产生更多 GD 机会，不能直接当因果。
- 行 precision 是重复出现的 nav goal 行，不是独立 episode；不能直接用 76.7% 推断因果增益。
- 分来源、分类别的行 precision：

| 类别 | gd-tsp | gd-only | tsp3d |
| :--- | ---: | ---: | ---: |
| bed | 94.4% | 40.4% | 68.5% |
| chair | 78.9% | 73.4% | 40.2% |
| couch | 54.0% | 62.0% | 21.6% |
| toilet | 100.0% | 91.2% | 0.0% |
| tv | 70.4% | — | 17.9% |

- `gd-tsp` 在 bed/tv 上精度高，但 couch 只有 54.0%；`gd-only` 在 toilet/chair 高，bed 只有 40.4%。因此“提高 gd-tsp 优先级”应按类别生效，不能全局把 gd-tsp 放在 sigma band 前。
- 当前全 run 仍有 131 集首次错误锁定、66 集 wrong stop before target coverage；这些是覆盖之外的选择/停止残余。

### 7.6 二次判据与 base gate 的数据含义

| 可调点 | 本 run 数据 | 判断 / 下一步 |
| :--- | :--- | :--- |
| `depth_not_ok` 在关联前拒绝 | 964；tv 582、couch 192、toilet 72、chair 61、bed 57 | gd-tsp 使用 track centroid，不需要 GD proposal 的 depth；当前代码已先关联 track 再判 depth，只对 `confirmed/provisional` track merge 写 `gd_track_depth_shadow`，仍不写记忆 |
| `gd_only_class_disabled` | S6gd2 旧数据 1,349；当前空 whitelist 后预计为 0 | 全类别已放开；tv/couch 的 gd-only precision 需要在下一 run 单独监控，不能沿用旧白名单结论 |
| `gd_only_conf_below_min` | S6gd2 旧数据 831；其中 `track_gd` 320 | 当前 conf gate 关闭，保留 new 0.50 / merge 0.45 参考值；下一 run 按类别核对 `gd_only_new` / `gd_only_merge` 的 conf 与 TP/FP |
| `gd_occ_geom_unknown` | S6gd2 旧数据 7；当前已改为 `occ_passed=True` 或 `occ_confirmed=True` | 7 条 unknown 的 `occ_passed` 均为 true（`out_of_bounds`），新代码会放行；`occ_passed=False` 仍拒绝，下一 run 需监控 unknown 条目的 TP/FP |
| `gd_hits` | ≥4 时 85.9%，≥5 时 89.7% | 适合做选择/提升优先级；不适合作为首次探索硬门 |
| `track_stable` 升级 | 仅 1 次（chair，episode 已 success） | 当前代码写 `gd_stable_shadow`，记录 GD proposal GT、stable 状态和 `memory_id`；后续轨迹用 `track_frames` 关联，仍不直接改状态 |
| `gd-tsp` 与 sigma band | gd-tsp 行 precision 76.7% | 可在同 state/同 band 内优先，或按类别（bed/tv/chair）提前；couch 不应全局提前 |

下一轮新增的只读日志：

- `gd_stable_shadow`：stable + GD 命中，记录 GD proposal `centroid/tp`、`track_centroid/track_tp`、`memory_id`、conf、depth；后续轨迹按 `memory_id` 关联 `track_frames`。
- `gd_track_depth_shadow`：先关联 track、再判 depth，只对 confirmed/provisional track merge 记录；不写记忆。
- `gd_ttl_shadow`：TTL 关闭时在 30/60/90/120/180/240/300 步记录 gd-only 年龄、conf、hits。
- revive merge：记录 `sigma_c`、`num_obs` before/after、旧质心距离、age、detect_dist。

### 7.7 机制分工（当前决定）

| 层 | 职责 | 不负责 |
| :--- | :--- | :--- |
| TSP3D + `sigma_c` | 候选准入与 TSP 候选分数 | 不负责候选外探索 |
| GD-tsp | 对既有 confirmed/provisional track 做确认/增强；按类别提高选择优先级 | 不做全局候选排序，不从 candidate-only 新建 track |
| GD-only | 候选外探索池；首次探索起点；`gd_hits` 用于后续排序 | 不可停、不参与 confirmed 判定 |
| field | 候选外区域 Top-K 探索覆盖先验 | 不参与候选排序、准入、记忆或 STOP |
| 语义侧车 | 可选的离线研究项 | 不作为当前机制前置，不生成全量 |

结论：**可以按“field 只做探索覆盖、排序选择交给 `sigma_c` + GD 证据”推进。** 但要把措辞限定为：GD-tsp 是“确认/增强 + 按类别的选择优先级”，不是完整的全候选 ranker；`gd-only` 是探索池，不是排序器。若后续要进一步提升排序，不建议回到 field candidate score；轻量离线 ranker 的初步想法已记录在 `VLVM-S7.md` §1.8，**当前明确延后实现**，先完成本轮 conf/TTL/shadow 判据统计。

## 8. 产物

- `detections/episodes.jsonl`
- `detections/candidates.jsonl`
- `detections/memory_events.jsonl`
- `detections/nav_goals.jsonl`
- `detections/track_frames.jsonl`
- `detections/gd_frames.jsonl`
- `detections/gd_proposals.jsonl`
- `detections/field_observations/`：297 ZIP，约 42 GB
- `field_analysis/field_offline_report.json`
- `field_analysis/field_offline_summary.md`
- `field_analysis/field_inference_config.json`
- `field_analysis/field_candidate_scores.jsonl`
- `s6_summary.json`
- `gd_offline_cal.json`
- `COMPLETED`
