# VLVM 机制候选：跨论文可借鉴项与内部联动

> 定位：从 ApexNav / ZONDA / GLMap / TrajRAG / ASCENT / MFNP 中筛出可迁移到 VLVM 的四项机制，说明落点、边界、与 S7 既有机制的联动。

## 0. 三点记录（M-0 前置记账层）

> 本项不改导航点选择、不改置信、不改准入/STOP，只补齐「每个 memory 记三个点」的记账，供 1.1–1.4 与离线标定使用。

### 0.1 三个点的定义与现状

| 点 | 语义 | 存储位置 | 用途 | 现状 |
| :--- | :--- | :--- | :--- | :--- |
| 质心 | EMA 合并后的目标中心 | `_target_3d_memory[cls][idx]` | 目标管理：merge 判定、排序、region 归属 | 已实现 |
| 首次表面点 | 首次写入的近表面点 | `_target_surface_memory[cls][idx]` | **当前导航终点**（`goal_use_surface=True`） | 已实现 |
| 最新表面点 | 最近一次观测的近表面点 | `state["surface_latest"]` + `surface_latest_step` | 暂无消费方，只记录 | 已实现（只记录） |

配套字段：`surface_first`、`surface_first_step`（首次写入快照，用于离线算 `latest − first` 位移）。EMA 表面点不再记录。

### 0.2 与 1.1–1.4 的耦合

| 机制 | 耦合 | 结论 |
| :--- | :--- | :--- |
| 1.1 多标签融合 + 负向证据 | 负向证据要求「簇在 FoV 内但本帧未检出」，判定锚点只能是**最新表面点**；首次点可能来自远距/斜视，早已不可见，用它判会全判为不可见 | 可见性判定以 latest 为准；点组需可扩到 label 级 |
| 1.2 多视角验证 + 视图质量打分 | first + latest 天然构成「远景 / 近景」两个尺度，对应 ZONDA 的 near/far 视图；但两点只是**下界**，按 `Q` 选 top-k 帧仍需帧级观测缓冲 | 三点给 1.2 兜底尺度对，不替代缓冲 |
| 1.3 instance + region 双尺度 | instance 单元的可视表示可取 first/latest 两张裁剪图；region 归属必须用**质心**（表面点随视角漂移会导致同一物体落进不同区域） | region 键 = 质心；instance 图 = first/latest |
| 1.4 易混类 + per-class 阈值 | 链条是「阈值 → 通过标签 → 最佳标签 → 导航点来源」；若点仍按 memory 一份，会出现导航点与最佳标签分属不同类 | 点组必须降到 label 级，否则 1.1/1.4 会返工 |

### 0.3 边界

- **不改善 SR**。导航仍用首次表面点，S7 §2.1 的 9 个大目标 near-miss（质心 TP、目标点非 TP，距离 1.1–1.4 m）不会被本项修掉；要修必须改「导航点用哪个点」或加 bbox 约束，属独立决策。
- 不写入 GD-only 路径：与既有不变量「GD 证据不改质心/几何」保持一致。
- 唯一即时价值是**离线可测**：`surface_latest − surface_first` 位移分布、latest 是否越出 bbox、first 过期比例。

### 0.4 前提与边界

- 四项全部落在**感知—记忆—确认**层（候选置信、簇结构、确认证据、查询词表），不改动作决策链。
- 跨论文 ΔSR 不可相加：ApexNav / ZONDA 的消融是从「无该模块」的基线掉出来的。
- 所有阈值都必须用 VLVM 自己的置信分布重标定（TSP3D grounding score 与 YOLO/GroundingDINO 检测置信不同量纲）。

## 1. 四项机制

### 1.1 目标级多标签置信融合 + 负向证据融合

**粒度决定（2026-11-04）**

- TSP3D 目前是单 query，只有 target 一个标签，label 级与 memory 级等价，因此 **1.1 先按 memory 级实现**：融合单位是 memory，权重仍用体积（box 体积/框内点数），负向证据只对 target 标签生效。
- **label 级推迟到 1.4 一起做**：只有引入易混类多 query 后，label 级才有多个标签与「类内对照」的意义；届时点组、置信、`argmax(c·n)` 一并升到 label 级。
- 因此本机制当前形态 = 「单标签体积加权 + 未检出负向降分」，不是 ApexNav 的完整多标签融合。

**来源与证据**：ApexNav target-centric semantic fusion。消融（HM3Dv1 / HM3Dv2 / MP3D SR）：

| 变体 | v1 | v2 | MP3D |
| :--- | ---: | ---: | ---: |
| 完整 | 59.60 | 76.20 | 39.23 |
| w/o Fusion（当前帧结果直接覆盖） | 44.50 | 55.00 | 27.84 |
| w/o Observation（不在检出时不给负向惩罚） | 50.48 | 70.20 | 35.58 |
| max-confidence 融合 | 51.43 | 67.90 | 32.11 |

**定义**

- 以「同一物体簇」为融合单位，簇内为每个标签维护 `(点云, 置信, 累计检测体积)`，多标签共存，不覆盖。
- 标签置信更新按体积加权：`c_new = (n_old·c_old + v_det·c_det) / (n_old + v_det)`，`n_new = n_old + v_det`；体积 `v` 取点云下采样后的点数（3D 场景下可直接用 box 体积或框内点数）。
- 最佳标签取 `argmax(c_o · n_o)`，即置信与观测量的乘积。
- 负向证据：簇在当前 FoV 内、但本帧该标签未检出时，以 `c_det=0`、体积取「当前深度点云中该标签点云最近邻可见点数」参与同一融合，使置信下降。

**VLVM 落点**

- `memory_life/target_memory_manager.py` 的置信更新与簇结构：现状是 `num_obs` 计次 + `sigma_c` + merge（距离/偏航/步数）三件套，缺少体积权重与多标签共存。
- 现状 `stable_miss_delete` 只做 stable 池删除，不参与置信融合，与「负向证据」不是同一件事，不可替代。

**需要的数据**：每个候选的 3D box 或点云（TSP3D 已有 `box_3d`）、每帧该簇的可见性判定（depth 最近邻即可，不需要新传感器）。

**风险**：低置信多标签会让簇内标签数膨胀，`argmax(c·n)` 可能被高频小物体刷掉真目标；需要限制每簇标签数并设体积下限。

**验收门**：`false_positive` 下降；`never_saw` 不上升；`gd-tsp` 提升 TP 数不显著下降（对应 S7 §E3 口径）。

---

### 1.2 多视角目标验证 + 视图质量打分

**来源与证据**：ZONDA multi-view target verification。消融（HM3D / MP3D SR）：66.5 → 57.8（去掉跨楼层，参照项）、66.5 → 41.5（去掉多视角验证，−25.0）；MP3D 48.2 → 30.2（−18.0）。这是所有已读论文中单模块杠杆最大的一项。

**定义**

- 每个候选维护观测缓冲，视图质量 `Q = S_raw · f_area · f_edge`：`S_raw` 为检测置信，`f_area` 为目标像素面积比的对数归一（偏好大而清晰的观测），`f_edge` 为离散惩罚（贴图像边界的观测降权）。
- 到达候选后，按 `Q` 取多个尺度的视图（近景 + 远景），一起送 VLM 交叉核对，判断是否为真目标。

**VLVM 落点**

- 确认层：STOP 前复核，位置与 `frontier doc` 机制 A（explore STOP → 候选复核）的「到达后双查询」同一处，可作为其第三条证据通道。
- 用已有的 BLIP2-ITM 服务做交叉核对，TSP3D 保持主判；符合「2D 模型只作辅助」的约束。
- 视图质量打分本身零成本，可先单独接入用于「选哪几帧送复核」。

**需要的数据**：每候选的历史观测帧与其 box 投影面积/边界接触标记。当前 `candidates.jsonl` / `track_frames.jsonl` 已有部分字段，需补 `f_area`、`f_edge`。

**风险**：新增在线 VLM 调用（BLIP2 单次百毫秒级），与 500 步预算和复核批次叠加会显著增加耗时；近距离 GD 高置信误报也可能骗过多视角核对（同一视点不同尺度不是独立证据）。

**验收门**：wrong stop 下降；`false_positive` 下降且 `timeout` 不上升；步数增幅有界。

---

### 1.3 instance + region 双尺度语义单元

**来源与证据**：GLMap。消融（HM3D SR / SPL）：VLFM 52.5/30.4 → +instance 57.4/31.3 → +region 56.2/30.9 → 两者 59.1/32.2；落到 ApexNav 基座 59.6 → 62.7。收益来自「用渲染图/文本描述而非特征向量喂 VLM/LLM」，不需要训练投影层。

**定义**

- 语义单元分两级：instance（物体/属性）与 region（功能区域/场景），每单元同时存**自然语言描述**与**可视表示**（GLMap 用 3D Gaussian 渲染，VLVM 可用 3D box 投影裁剪图）。
- 相似度可用 VLM 对渲染图打分，或 LLM 对文本描述打分，二者都可直接调用现成服务。

**VLVM 落点**

- 只作**确认层证据**：对已选候选，用其 box 裁剪图 + 所在区域描述做一次上下文核对，回答「这个区域像不像该目标该出现的地方」。
- 明确**不参与候选排序/准入/STOP**：与 `VLVM-S7-frontier.md` §3.2 弃用 field 选点的结论保持一致，避免复活已被否定的路径。
- region 单元的构建需从零实现：原 `semantic_spatial_field/` 已随 field 弃用于 2026-11-04 删除，没有可复用的离线区域结构。

**需要的数据**：候选 box → 图像裁剪的投影链路（已有 3D box 与相机 tf）、区域索引（需新建窗口 cell 结构）、一次 LLM 生成区域描述。

**风险**：region 语义若被用于打分，会与「field 只做覆盖」的定稿冲突；渲染/裁剪成本与 GLMap 的 3DGS 不在同一量级，需接受用裁剪图近似。语义侧车已决定不生成，区域描述只能按需小批量生成。

**验收门**：确认层 FP 下降；不得因 region 分数改变目标选择结果（日志需能证明 comparator 未变）。

---

### 1.4 易混类清单 + 自适应阈值

**来源与证据**：ApexNav LLM 离线推理。消融：去掉相似对象辅助检测，MP3D 59.60→28.70（−10.5）、HM3Dv1 −2.7、HM3Dv2 −1.2。

**定义**

- 离线、per-target、一次性生成：① 易混类 3–5 个（限定检测词表内）② 自适应阈值 `c_th`（原文用 apple=0.25 / bed=0.65 作锚点，区间 `[0.25,0.65]`，MP3D 用 `[0.25,0.35]`）③ 目标最可能出现的房间。
- 在线零 LLM 调用：`[target + 易混类]` 一起送检测，簇内多标签对照，`best==target && c_o>c_th` 才算 reliable，其余为 suspected，仅在 frontier 耗尽后启用。
- 原文 LangChain 式 CoT + 2 个 few-shot；模型 DeepSeek-V3（离线）。

**VLVM 落点**

- 查询层：TSP3D 现在是单 query，改成 `[target, 易混类…]` 多 query，各 query 的 box 落进同一 memory 并按标签区分；这是「输入适配」范围内的改动，不动权重。
- **本项是 label 级粒度的前置**：1.1 的多标签部分、点组升到 label 级、`argmax(c·n)` 选最佳标签都随本项一起实现；未做本项时 1.1 只能停留在 memory 级。
- 阈值层：把全局 `sigma_tar/sigma_prop` 换成 per-class 表。**必须重标定**：用 E3 口径（val 集 2638 视图、`sigma_tar=0` 取全量 raw Top-1）扫 per-class 阈值，不能搬 ApexNav 的 0.25–0.65。
- 词表对齐：HM3D 6 类 COCO + MP3D 21 类 open-vocab 需先做数据集标签→查询词映射（sofa→couch、tv monitor→tv、plant→potted plant 等）。

**风险**：多 query 使每帧 TSP3D 调用数 ×(1+易混类数)，延迟与显存同步上升；易混类若选得差会把 FP 引入候选池，需保留「只有 target 标签可提升」的硬约束。

**验收门**：per-class 阈值在 E3 离线口径上优于全局单一阈值；在线 FP 不升、`never_saw` 下降。

## 2. 复杂度与杠杆对照

| 序 | 机制 | 复杂度 | 改动面 | 外部证据（去掉后 ΔSR） | 与已定稿模块的冲突 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1.1 | 多标签置信融合 + 负向证据 | 中 | 记忆管理器置信/簇结构 | −15.1 / −21.2 / −11.4；−9.1 / −6.0 / −3.7（v1/v2/MP3D） | 无 |
| 1.2 | 多视角验证 + 视图质量打分 | 中高 | 确认层 + 新增在线 VLM 调用 | −25.0（v1）、−18.0（MP3D） | 无 |
| 1.3 | instance + region 双尺度语义单元 | 高 | 候选证据 + 区域索引 + 按需文本 | +6.6（对 VLFM）、+3.1（对 ApexNav） | 有（不得用于排序，见 §3.2） |
| 1.4 | 易混类清单 + 自适应阈值 | 低 | 查询词表 + 阈值配置（离线产物） | −10.5（MP3D）、−2.7（v1） | 无 |

## 3. memory 级置信融合 + 负向证据

> 本轮形态：**memory 级、单标签**（TSP3D 单 query）。label 级随 1.4 多 query 一起做。
> 不改动作的 shadow 先行，确定有效后再接入；阈值本轮不动。

### 3.1 记号与数据来源

| 记号 | 含义 | 来源 |
| :--- | :--- | :--- |
| `c_raw` | 本帧该候选的 TSP3D raw 置信（未乘 `w_S`） | `ObjectDetections` 的 logits / `conf_f` |
| `v_pts` | 体积口径 A：框内点数，3D box 内落在当前帧点云的点数 | `box_3d` + 当前帧 pcd |
| `v_box` | 体积口径 B：box 体积（m³） | `box_3d` |
| `v` | 实际代入融合公式的体积值；shadow 阶段两口径各算一遍，正式接入时择一 | `v_pts` 或 `v_box` |
| `n` | 该 memory 的累计观测量 | state（新增） |
| `c_fused` | 融合后置信，用于进入 `s_penalty` | 本机制输出 |
| `s_latest` | 该 memory 的最新表面点 | `state["surface_latest"]`（M-0 已有） |
| `N_hit` | `s_latest` 在当前帧点云中半径 `r_hit` 内的命中点数 | 当前帧 pcd |
| `d` | 机器人到 `s_latest` 的距离 | 机器人位姿 |

### 3.2 正向融合（体积加权累积）

首次写入：

```
n = v
c_fused = c_raw
```

后续每次匹配（同一 memory）：

```
c_fused = (n * c_fused + v * c_raw) / (n + v)
n       = n + v
```

体积口径（两套并行记录，shadow 比较后择一）：

| 口径 | 定义 | 优点 | 缺陷 | 截断 |
| :--- | :--- | :--- | :--- | :--- |
| A `v_pts` | 框内点数 | 与观测质量同源，远处小框不会被高估 | 受深度采样密度与点云降采样影响，近距离过密 | `v_min=20`，`v_cap=8000`（点） |
| B `v_box` | box 体积（m³） | 与距离无关、稳定可复算 | 大物体体积压过近距离小物体的高质量观测 | `v_min=0.005`，`v_cap=8`（m³） |

shadow 阶段两种口径各算一遍（`c_both_pts` / `c_both_box`），离线比较后再定正式口径；正式接入只用其中一个。

### 3.3 负向证据（未检出降分）

触发条件必须**同时**满足，否则不降分：

1. 该 memory 的本帧未产生任何匹配候选（这一帧没被任何检测吃到）；
2. `s_latest` 在当前帧点云中存在最近邻点，且最近邻距离 ≤ `r_hit`、命中点数 `N_hit ≥ N_min`；
3. 机器人到 `s_latest` 的距离 `d ≤ d_vis`；
4. 距该 memory 上一次降分已过 `neg_cooldown` 帧。

满足时按同一公式以 `c_raw = 0` 计入：

```
v_neg   = min(N_hit, v_cap)
c_fused = (n * c_fused + v_neg * 0) / (n + v_neg)
n       = n + v_neg
```

参数说明与建议值：

| 参数 | 含义 | 建议值 | 作用与调法 |
| :--- | :--- | :--- | :--- |
| `r_hit` | `s_latest` 与当前帧点云最近邻的匹配半径 | 0.2 m | 过大时邻近物体的点会被当成本物体的可见证据，造成误罚；过小时稍作位移就判不可见，负向证据失效 |
| `N_min` | 允许判「可见但未检出」的最小命中点数 | 5 | 主要节流阀：调大→只在真看得清时罚（漏罚增多、误罚减少）；调小→灵敏度高但遮挡也罚 |
| `d_vis` | 允许降分的最大机器人距离 | `0.95 * max_depth`（现 5.0 m → 4.75 m） | 超出视为过远，不构成可靠负证据；远处小目标不应被罚 |
| `neg_cooldown` | 同一 memory 两次降分的最小间隔帧数 | 3 帧 | 防连续遮挡把置信打到地板；调小→衰减更快、更贴近实时遮挡，但易误杀真目标 |
| `v_cap` | 单次负向体积上限 | 点口径 8000 点 / 体积口径 8 m³ | 防止一次大面积近距离遮挡把 `c_fused` 一次清零 |
| `c_floor` | `c_fused` 下限 | 0.0 | 只防数值下穿；若希望目标永不彻底消失，可设 0.1 |

`neg_cooldown` 的去抖实现：state 里记 `last_neg_step`，`step - last_neg_step < neg_cooldown` 时跳过该次负向更新；`c_fused = max(c_fused, c_floor)` 在每次更新后统一施加。

- 负向证据**不写** `total_miss` / `stable_*_miss`，与现有 miss 计数互不干扰（两者目的不同：miss 用于生命周期状态迁移，负向证据用于置信重估）。

### 3.4 与现有链路的关系

```
TSP3D raw c
  → occ gate（几何/占据一致性）
  → [本机制] c_fused（体积加权 + 未检出负向）
  → s_penalty: sigma_c = c_fused * w_S
  → sigma_tar / sigma_prop 门槛
  → provisional gate → 档位 → 选择 / STOP
```

- `s_penalty` 公式与 `w_S` 不变，只替换进入它的那个数；不把本机制并进 `s_penalty`（它是无状态逐帧乘子，无法表达跨帧累积）。
- 阈值（`sigma_tar=0.60`、`sigma_prop=0.45`、`provisional 0.50–0.60`、`sigma_band`、`upgrade_min_obs=4`）本轮沿用，重标定在 shadow 结果之后。
- GD 通道（`gd_hits` / `gd_conf_min`）不参与本机制。

### 3.5 阶段

| 阶段 | 内容 | 动作变化 | 产物 |
| :--- | :--- | :--- | :--- |
| S-1 | **全面 shadow**：同一次运行内同时输出「仅体积加权」「仅负向」「两者」三套 `c_fused`（两种体积口径各算一遍），并记录 `c_raw / v_pts / v_box / n / N_hit / d / sigma_c 前后 / 档位 / 命中TP-FP` | 否 | `detections/fusion_field.jsonl`，可离线扫阈值曲线与各配置的 TP/FP |
| S-2 | 按 S-1 结论选配置与阈值，正式接入（开关打开） | 是 | SR、FP、never_saw、timeout、steps |
| S-3 | 视角置信对比（max vs 平均 vs 平方加权，单独一批） | 是 | SR、FP、steps |

S-1 的判读口径：若「仅负向」使被降分项里 TP 占比明显偏高（近距离仍被降），说明可见性门槛不够严，调 `N_min` / `r_hit` / `neg_cooldown`；若三套配置的 FP 变化几乎相同，说明体积加权与负向证据在 VLVM 的漏检结构下都不起作用，S-2 直接跳过。

### 3.6 日志字段（`fusion_field.jsonl`）

`scene`、`episode_id`、`target_object`、`step`、`memory_id`、`distance`、`c_raw`、`v_pts`、`v_box`、`n_before`、`n_after`、`N_hit`、`neg_triggered`、`c_vol_only`、`c_neg_only`、`c_both_pts`、`c_both_box`、`sigma_c_before`、`sigma_c_after`、`tier`、`is_tp`（离线按目标 bbox 标）、`admitted`。

三点字段仍在 `memory_events.jsonl`（`vlfm/vlvm/utils/three_points.py`），与本文件按 `memory_id` + `step` 关联，不合并落盘。

> **2026-11-05 决定：本节的正向融合（`c_fused`）与「降分」全部弃用**，代码已删除（`volume_fusion.py`、`conf_override`、`fusion_field.jsonl` 通道）。实验依据见 [VLVM-S8-run1.md](VLVM-S8-run1.md)：`c_raw` 的 TP/FP AUC 0.7358 → `c_fused_pts` 0.6881 / `c_fused_box` 0.6034；准入集 TP 464 → 372 / 342；负向触发落在 TP 与 FP 记忆上等量（425 / 351），TP 记忆被压到 `c_neg` p50 0.133。保留下来的是**负证据的记账口径（§3.3 的可见性与冷却条件）**，它已迁移为 §4 的独立分数，不再改写任何置信。

## 4. STOP recheck V1（视图质量 + 负证据）——**已删除，仅存历史记录**

> **2026-11-06 决定：本节全部删除**（`verify.py`、`stop_check.py`、`StopRecheckMixin`、8 处 policy 调用点、`box_latest`、`verify_*` 配置与 `verify_stop.jsonl` 落盘）。实验依据见 [VLVM-S8-run3.md](VLVM-S8-run3.md)：297 集全配对 SR 51.18% → 50.84%（flip 3 / gain 2，扣噪后净 0），`failure_cause` 分布逐项不变（fp 89→90、never_saw_target 51→51）。
>
> 删除理由：107 集（36%）根本没有 `nav_gd_tsp` 出口，机制够不着；91% 的集从未被否决；被否决的 26 集每集只改 1 帧动作。瓶颈不在 STOP 判定，而在「目标从未被看见」与「锁错目标后一路走到底」。
>
> 下表内容仅作为设计与失败原因的存档，代码已不存在。

> 历史状态（2026-11-05）：已落代码并 `active`。
> 定位：本节是 1.2（多视角目标验证 + 视图质量打分）在 VLVM 的落地形态，也承接 1.1 保留下来的负证据。只作用于 **STOP 之前**，不改准入、排序、探索与导航。

代码分层：

| 文件 | 职责 | 依赖 |
| :--- | :--- | :--- |
| `vlfm/vlvm/recheck/verify.py` | 纯计算：投影、`f_area`/`f_edge`、视图缓冲、`Q_score`、`neg_score` | 不知道 policy 存在 |
| `vlfm/vlvm/recheck/expl_recheck.py` | `expl recheck`：explore frontier 耗尽后的巡回与受控 STOP（`ExplRecheckMixin`） | 读 memory 列表与选择器 |
| `vlfm/vlvm/recheck/stop_check.py` | `nav recheck` 与 policy 的联接：`StopRecheckMixin`（`_observe_views` / `_verify_shadow` / `_verify_itm_score` / `_stop_recheck_reset` / `_stop_recheck_drop`） | 读 `_target_3d_memory`、`_memory_manager`、当帧 RGB-D 与位姿 |
| `vlfm/policy/tsp3d_objectnav_policy.py` | 只在帧末调 `_observe_views`，在四个 STOP 出口调 `_verify_shadow` | 不实现任何判据 |

### 4.1 术语与 STOP 出口

| 名称 | 作用域 | 触发 | 出口 |
| :--- | :--- | :--- | :--- |
| `expl recheck` | explore | frontier 耗尽 | 队列走完 → 唯一受控 STOP；队列为空 → 直通 STOP |
| `nav recheck` | navigate | 目标进入 1.2 m 且未取得 STOP 资格 | 核验通过 → STOP；不通过 → 拉黑 |

现存的全部 STOP 出口：

| # | 出口 | 位置 |
| :--- | :--- | :--- |
| 1 | `nav gd-tsp` STOP | `_pointnav(goal, stop=True)` 内 `rho < 0.9` 或者 `stop=True` 但 `rho ≥ 0.9` |
| 2 | `nav not gd-tsp` 左转窗口 | 1.2 m 内 `_turn_left_action` 确定能否升级为 gd-tsp，计满 4 帧拉黑 |
| 3 | `expl recheck` 队列走完 | `recheck/expl_recheck._recheck_stop` |
| 4 | explore 队列为空 STOP | `_explore_safe` 末：_recheck_begin 返回 False → 直接 return self._stop_action |
| 5 | 环境步数上限 | 环境侧超时，不可控 |

### 4.2 接入范围

| 出口 | 状态 | 行为 |
| :--- | :--- | :--- |
| #1 `nav gd-tsp`（几何到位 + 网络自发 STOP） | **active** | 先核验：`tsp3d_confirm=True` 直接放行（`confirm_pass`）；否则通过 → 立刻 `_pointnav(stop=True)`，记 `verdict=pass` 并加入 `_verified_memories`（之后 `cached_pass` 免重评）；拒绝 → 本帧左转、下帧重新 act，**不拉黑**，但该 memory 降权 `verify_penalty_steps` 帧；同一 memory 拒满 `verify_max_reject=1` 次即放行（`reject_mem_cap`），全 episode 累计拒满 `verify_max_reject_total=6` 次后放行（`reject_cap`） |
| #2 `nav not gd-tsp` 左转窗口 | log-only | 不参与判定：1.2 m 内左转争取升级为 `gd-tsp`，升级后下一帧走 #1 的门控；不升级照常拉黑 |
| #3 `expl recheck` 到达 | **active** | 通过 → 就地 STOP、提前结束巡回；拒绝 → 下一个 waypoint（走完仍 STOP，停在最后一处） |
| #4 explore 队列为空 | **active** | 先用二级过滤（放宽 `stable` 与 `_recheck_attempted`）取 1 个 fallback waypoint 巡回，到达后按 #3 门控；二级也为空才直接 STOP |
| #5 环境步数上限 | 不可控 | 环境侧超时结束 |

三道不计/不计满的情形：`memory_id` 为空的出口（#4 无候选）只记录不计拒绝；`EXIT_NAV_NONGD_TURN` 恒放行；已通过的记忆不再消耗拒绝额度。

各处口的放行顺序（任一命中即放行）：`cached_pass` → `confirm_pass`（`tsp3d_confirm=True`，R-5）→ `reject_mem_cap`（该 memory 已拒满）→ `reject_cap`（本集已拒满）。

理由：`stop_eligible_current=true` 的 2749 条日志全部是 `origin=gd-tsp`（含 `promoted_gd_only` 1333 条），其中 `goal_tp=false` 占 1425 条（51.8%）；89 个 FP 集里 45 集经过 `gd-tsp` 的 `stop_eligible`。#2 当前不产生 STOP，因此只积累阈值依据。

### 4.3 候选级视图缓冲

每个 memory 一份定长环形缓冲，键为 `memory_id`，与 `_target_3d_memory` / `_target_surface_memory` / `_target_verify_state` 平行。

| 记录字段 | 来源 |
| :--- | :--- |
| `step` / `detected` / `in_fov` / `distance` / `azimuth_deg` | 当帧状态 |
| `sigma_c` / `sigma_c_raw` | 检出帧的 `c·w_S` 与原始 `c` |
| `box_px` / `corners_ahead` / `behind_ratio` | memory 的 `box_latest` 8 角点按当帧位姿与内参投影 |
| `area_ratio` / `f_area` / `f_edge` | 由 `box_px` 计算 |
| `q` / `q_raw` | `σ_c·f_area·f_edge` / `c_raw·f_area·f_edge` |
| `crop` | `box_px` 外扩 25% 后缩放到 128×128（只留在内存，不落盘） |

| 规则 | 值 |
| :--- | :--- |
| 缓冲深度 | `verify_buffer_cap=8` |
| 每 memory 保留帧 | `verify_k=4`：**首次观测 + 最新观测 + 按 `q` 取前 2** |
| 多样性 | 同一 memory 相邻入选项要求 `\|Δdist\|>0.3 m` 或 `\|Δazimuth\|>10°` |
| 未检出帧的 box | 用 memory 的 `box_latest`（新增字段），否则算不出 `f_area` |
| 清理 | memory 被删/复位、episode 结束清空 |
| 存储 | 128×128×3 uint8 ≈ 49 KB/帧，8 帧 ≈ 393 KB/候选；整帧 640×480 会到 7.4 MB/候选，禁用 |

### 4.4 两条判据

**判据 A：视图质量 `Q`**

```
f_area = clip( log(area_ratio / area_ref) / log(area_hi / area_ref), 0, 1 )
f_edge = clip( 0.5 + margin / (2 * edge_px), 0, 1 )      # 软化：完全贴边仍有 0.5
Q_frame = S_raw * f_area * f_edge,        S_raw = sigma_c = c_raw * w_S
Q_score = max( Q_frame )  over the k picked frames
usable  = f_area >= f_area_min            # 只看目标是否在画面中够大
```

两处与初稿不同，均由 2026-11-05 的诊断实验确定：

| 项 | 初稿 | 现行 | 原因 |
| :--- | :--- | :--- | :--- |
| `f_edge` | `clip(margin/edge_px)`，贴边即为 0 | `clip(0.5 + margin/(2·edge_px))`，贴边为 0.5 | STOP 距离上大目标（床/沙发/桌）必然填满画面、`margin=0`，硬归零会让核验在近距永远不通过 |
| `Q_score` | 取 `mean` | 取 `max` | 缓冲区里含远景帧，用均值会让被截断的近景拖低分数；取最佳一帧更贴近「有没有一个可信视图」 |
| `usable` | `f_area` 与 `f_edge` 双门槛 | 只看 `f_area` | `f_edge` 改为乘性权重后不再适合做硬门 |

**判据 B：负证据分数（只在 usable 帧上统计）**

```
usable  = f_area >= f_area_min                                    # 只看像素几何
visible = (N_hit >= neg_min_hits) and (d <= neg_d_vis)           # N_hit: surface_latest 球内当帧点数
neg_hit = (not detected) and visible and cooled                  # cooled: 距上次 >= neg_cooldown 帧
neg_score = n_neg_hit / n_usable
```

先 `usable` 再判未检出：`usable` 由视图几何决定，与检测置信无关，所以「太远/太偏/模糊」的帧既不进 `Q` 也不进负证据。

**判定**

```
q_abstain      = (Q_score is None)                              # 无可用视图 = 无证据
would_pass_q   = True if q_abstain else (Q_score >= q_stop)      # abstain 不否决（R-5）
would_pass_neg = (neg_score is None) or (neg_score <= neg_max)
would_pass_itm = (itm_score is None) or (itm_score >= itm_floor)
verdict        = would_pass_q and would_pass_neg      # itm_gate=False 时不参与
```

`would_pass_q_strict` 保留严格读法（`Q_score is not None and Q_score >= q_stop`）只用于离线分析。`itm_score` 为 BLIP2-ITM 对本候选最高 `Q` 视图的 `cosine(crop, "a photo of a {target}")`，每候选最多 1 次调用，TSP3D 仍是主判。

### 4.5 参数

| 参数 | 默认 | 含义 |
| :--- | ---: | :--- |
| `verify_mode` | `active` | `off` / `shadow` / `active` |
| `verify_turn_frames` | 3 | 预留（当前未使用，窗口由 `verify_max_reject` 控制） |
| `verify_k` / `verify_buffer_cap` | 4 / 8 | 保留帧数 / 环形缓冲深度 |
| `verify_crop_pad` / `verify_crop_size` | 0.25 / 128 | 裁剪外扩 / 边长 |
| `verify_area_ref` / `verify_area_ref_mode` | 0.02 / `fixed` | `f_area` 参考面积比 / `per_class` 仅记录 |
| `verify_area_hi` | 0.10 | 映射到 `f_area=1` 的面积比（10% 画面） |
| `verify_edge_px` | 8.0 | 映射到 `f_edge=1` 的边距 |
| `verify_f_area_min` / `verify_f_edge_min` | 0.15 / 0.50 | usable 门槛（**现只用 `f_area`**；`f_edge_min` 保留未用） |
| `verify_neg_r_hit` / `verify_neg_min_hits` | 0.20 m / 5 | 可见性探针半径 / 最小点数 |
| `verify_neg_d_vis` / `verify_neg_cooldown` | 4.75 m / 3 帧 | 负证据距离上限 / 冷却 |
| `verify_q_stop` / `verify_neg_max` | 0.25 / 0.30 | 判据阈值，**必须按 shadow 分布重标** |
| `verify_max_reject` | 1 | **单 memory**拒绝次数上限，达到后该 memory 放行（不拉黑） |
| `verify_max_reject_total` | 6 | **全 episode**拒绝次数上限，达到后整个门控放行 |
| `verify_confirm_override` | True | `tsp3d_confirm=True`（本帧 admitted 候选与目标共位 0.5 m）直接放行 STOP |
| `verify_penalty_steps` | 6 | 被拒 memory 在选择器里的降权帧数（`tier + 10`） |
| `verify_itm_enable` / `verify_itm_floor` | True / 0.30 | ITM 交叉核对 / cosine 门 |
| `verify_itm_gate` | False | False = ITM 只记录，不参与 `would_pass`（未定项） |
| `verify_log_frames` | False | 是否逐帧落 `verify_frames.jsonl` |

### 4.6 阶段

| 阶段 | 内容 | 动作变化 |
| :--- | :--- | :--- |
| R-1 | 全出口 shadow：四处出口都算 `Q_score` / `neg_score` / `itm_score` 并落日志 | 否（2026-11-04 run1 已完成，数据见 [VLVM-S8-run1.md](VLVM-S8-run1.md) 与 §4.4 诊断） |
| R-2 | 放开 #1/#3/#4：拒绝仅否决本次 STOP，带「已通过免重评」与「全 episode 拒满 3 次放行」两道边界 | 是（2026-11-05 run2，**判为亏**，见下） |
| R-5 | `Q` 不可计算时 abstain（不否决）+ `tsp3d_confirm=True` 直接放行 | 是（2026-11-05 run3） |
| R-6 | 拒绝次数改 per-memory（1 次）+ episode 总额度（6 次）+ 被拒 memory 在选择器降权 6 帧 | 是（2026-11-05 run3） |
| R-3 | 视结果放开 #2（非 `gd-tsp` 通过即 STOP） | 否（数据不支持，见下） |
| R-4 | ITM 从记录转为参与 `would_pass`（`verify_itm_gate=True`） | 否（同上） |

**R-5/R-6 结果（run3，297 集全配对，2026-11-05，详见 [VLVM-S8-run3.md](VLVM-S8-run3.md)）**

| 指标 | run1（shadow 基线） | run3（active，R-5/R-6） |
| :--- | ---: | ---: |
| SR | 152/297 = 51.18% | 151/297 = 50.84% |
| steps 合计 | 48273 | 48006 |
| flip / gain | — | 3 / 2（其中 2 flip + 1 gain 属于噪声） |
| `failure_cause` 分布 | fp 89 / never_saw 51 / timeout 3 | fp 90 / never_saw 51 / timeout 3 |

四处修复消除了系统性损失（第一场景 79/99 → 83/99，run1 为 82/99），但 **297 集净贡献为 0**：107 集（36%）根本没有 `nav_gd_tsp` 出口，91% 的集从未被否决，被否决的 26 集每集只改 1 帧动作。R-3/R-4 因此不做——瓶颈不在 STOP 判定，而在「目标从头未被看见」与「锁错目标后一路走到底」两类失败。

**R-2 结果（run2，68 集配对，2026-11-05，`outputs/s8_verify_active_3scene_2026-11-05_172315`）**

| 指标 | run1（shadow 基线） | run2（active） |
| :--- | ---: | ---: |
| SR | 53/68 = 77.9% | 49/68 = 72.1% |
| steps 合计 | 11383 | 11548 |
| 未受 reject 影响的 47 集 | 35 | 35（完全一致） |
| 受 reject 影响的 21 集 | 18 | 14（flip 4 / gain 0） |

四个 flip 集（ep3/31/50/67）全为 `false_positive`，步数仅 +6~7，即 3 帧左转造成的干扰。诊断出四个缺陷：

| # | 缺陷 | 证据 | 对应修复 |
| :--: | :--- | :--- | :--- |
| A | `Q_score=None` 被当作拒绝 | 60 个 reject 里 34 个（57%）`Q` 不可计算 | R-5 abstain |
| B | 无 `tsp3d_confirm` override | ep67 step49 `tsp3d_confirm=True` 仍被拒，`confirm` 随后永久丢失 | R-5 override |
| C | 拒绝额度是 episode 全局 | 21 个受影响集的 reject 数全恰好为 3，用满后整集无门控 | R-6 per-memory |
| D | 拒绝后无降权 | ep31 在 1.47 m 处用相同 `q=0.1314` 连拒 3 次 | R-6 penalty |

两条防死磕边界：同一 memory 通过后记入 `_verified_memories` 免重评；单 memory 与全 episode 两道拒绝额度。任一触发都直接 STOP，行为退回接入前。

R-6 判读口径：① `verdict` 分布——`reject` 应显著少于 `confirm_pass`+`cached_pass`，若 `reject` 仍 >15% 说明 `q_stop` 过高；② 同 `memory_id` 不应出现 2 次 `reject`（`reject_count_mem` 恒为 1）；③ SR 不低于 77.9%（run1 配对基线），步数增幅 ≤ 3 步/集；④ `reject` 行的 `would_pass_q_strict=False` 但 `q_abstain=False` 才是真几何否决，`q_abstain=True` 的拒绝应为 0。

run3 对上述四条的实测：①`reject` 占 0.9%（25/2693），远低；②成立，25 次全为各集第 1 次；③SR 50.84% vs run1 51.18%，步数 −0.9/集；④**未达成**——仍有 7 次 `q_abstain=True` 的拒绝，全部来自 `neg_score` 单独否决（`Q` 弃权了但负证据没有），其中 `5cdEh9F2hJL` ep0 因此超时失败。

### 4.7 日志字段（`detections/verify_stop.jsonl`）

`scene`、`episode_id`、`target_object`、`success`、`oracle_success`、`failure_cause`、`kind=verify_stop`、`exit`、`step`、`verify_mode`、`memory_id`、`memory_uid`、`target_class`、`distance`、`origin`、`state`、`sigma_c`、`gd_hits`、`turn`、`tier`、`queue_index`、`relaxed`、`buffer_frames`、`buffer_memories`、`n_frames`、`n_usable`、`n_miss_usable`、`n_picked`、`picked_steps`、`picked_flags`、`q_score`、`q_score_raw`、`f_area_mean`、`f_edge_mean`、`sigma_c_mean`、`neg_score`、`n_hits_last`、`q_abstain`、`would_pass_q_strict`、`itm_score`、`itm_step`、`would_pass_q`、`would_pass_neg`、`would_pass_itm`、`would_pass`、`q_stop`、`neg_max`、`itm_floor`、`verify_applied`、`verdict`（`pass` / `cached_pass` / `confirm_pass` / `reject` / `reject_mem_cap` / `reject_cap`）、`reject_count`、`reject_count_mem`、`tsp3d_confirm`、`tp`。

`episodes.jsonl` 增 `n_verify_stop`。`verify_log_frames=True` 时另出 `detections/verify_frames.jsonl`（含 `usable`、`neg_triggered`、`area_ratio`、`box_px`、`corners_ahead`、`behind_ratio`），默认关闭。

### 4.8 已知边界

| 项 | 说明 |
| :--- | :--- |
| 图像仅用于裁剪打分 | TSP3D 仍是主判；BLIP2-ITM 只在候选最高 `Q` 视图上调用 1 次 |
| 2D 框来自投影 | memory 存 `box_latest`（8 角点），主点按 `W/2, H/2`，与反投影函数严格互逆（已做往返验证） |
| 投影必须用传感器尺寸 | `_depth_image_shape` 是 PointNav 输入（224×224），像素几何必须用当帧 RGB 的 640×480；曾因此全部帧 `f_edge=0` |
| `f_edge` 软化 | 近距大目标必然贴边，硬归零将使核验在 STOP 距离永不通过 |
| `Q` 的绝对阈值无先例 | `q_stop=0.25` 来自 2026-11-05 诊断（检出帧 `q` p10 0.21 / p50 0.29），仍需本批数据重标 |
| `gd-only` 记忆无 `box_latest` | `accumulate_gd_only` 不记录 box，其未检出帧算不出 `Q`；R-5 后这类记忆 abstain 放行，不再被误拒 |
| `tsp3d_confirm` 依赖 stats | `verify_confirm_override` 读的是 `_stats.tsp3d_confirm()`（与 `[STOP]` 行同源）；`enable_stats=False` 时退回无 override |
| 降权只改选择器 tier | `verify_penalty_steps` 内该 memory 的 tier `+10`，在有其它候选时被排后；无其它候选时仍会被选中（不丢目标） |
| 负证据不改置信 | 只作 STOP 侧分数；`sigma_c`、准入、排序一律不动 |
| `expl` 空队列出口 | 拒绝后仍须 STOP，只能改位置，收益上限低 |
| 非 `gd-tsp` (#2) | 放开即新增 STOP 资格，FP 风险上升，需本批阈值先落地 |

> 上述四机制弃用，1.1 / 1.4 整合保留。

## 5. 多视角质量分数（MVS）验证 (STOP recheck)
> **定位**：取代 §4 的「视图质量 `Q` + 负证据分数」两条判据。**判据只保留一个 —— 多视角质量分数 `MVS`**；负证据降为 `MVS` 的一个**加性减项**，且本批只做 shadow（`w_neg=0`）。
>
> **状态**：#2 直接接入（active）；#3/#4 explore 与 mem-life（upgrade / trans / revive / delete）只记录。`#1 nav gd-tsp` 与 explore 失败重试一律不使用 `MVS`。

### 5.1 术语与 STOP 出口

| 名称 | 作用域 | 触发 | 出口 |
| :--- | :--- | :--- | :--- |
| `expl recheck` | explore | frontier 耗尽 | 队列走完 → 唯一受控 STOP；队列为空 → 直通 STOP |
| `nav MVS` | navigate | 目标进入 1.2 m 且未取得 STOP 资格（原左转窗口） | `MVS ≥ θ` → STOP；否则左转采集，计满即拉黑 |

现存的全部 STOP 出口与 `MVS` 的接入状态：

| # | 出口 | 位置 | `MVS` 状态 |
| :--- | :--- | :--- | :--- |
| 1 | `nav gd-tsp` STOP | `_pointnav(goal, stop=True)`：`rho < 0.9` 发 `_stop_action`，否则走 `_pointnav_policy.act` | **不使用**（保持原样） |
| 2 | `nav not gd-tsp` 左转窗口 | 1.2 m 内 `_turn_left_action`，计满 4 帧拉黑 | **active：由 `MVS` 替换**，不再要求升级为 `gd-tsp` |
| 3 | explore 队列为空 STOP | `_explore_safe` 末：`_recheck_begin` 返回 False → `return self._stop_action` | shadow only |
| 4 | `expl recheck` 队列走完 | `recheck/expl_recheck._recheck_stop` | shadow only |
| 5 | 环境步数上限 | 环境侧超时，不可控 | — |

### 5.2 为什么必须换算法（§4 的失败机理）

§4 的两条判据在 297 集上净贡献为 0（见 [VLVM-S8-run3.md](VLVM-S8-run3.md)），失败不是阈值问题，是**证据源选错**：

| 现象 | 数据 | 根因 |
| :--- | :--- | :--- |
| `Q` 大面积不可计算 | 60 次拒绝里 34 次（57%）`q_score=None` | `f_area` 依赖 `box_latest` 8 角点投影；`gd-only` 路径不写 box，未检出帧更无 box |
| 作用点太晚 | 25 次拒绝分布在 25 集，每集仅 1 帧 | 门控只能在「已经具备 STOP 资格」之后否决，而 21 个可修复 fp 里有 14 集**根本没拿到 STOP 资格** |
| 负证据无判别力 | 触发落在 TP / FP 上等量（425 / 351） | 分母选的是「usable 帧」，而 usable 由 box 面积决定，与「是否真看见」无关 |

因此本节的两条硬约束：

1. **只用不依赖 2D box 投影的证据**（点云、位姿、memory 自身统计、TSP3D 共位）；
2. **作用在「获得 STOP 资格」这一步**（#2 出口），而不是在资格之后否决。

### 5.3 多视角质量分数 `MVS`

**定义**：`MVS` 由三个正因子**加性**合成、负证据作为**加性减项**，全部来自本框架已有数据，不需要新传感器、不需要 2D 框投影、不需要额外 VLM 调用。

```
MVS = w_hit·f_hit + w_geo·f_geo + w_cooc·f_cooc  −  w_neg·neg_ratio
三因子 ∈ [0,1]；MVS ∈ [−w_neg, 1]
```

**为什么用加性而不是乘法**：乘法在任一分量低时会把整体压到接近 0，导致“单项低 = 全部否决”，阈值也不可解释（§4 的 `f_area·f_edge` 就是这个问题）；加性可让单一强证据搏回弱势项，且负证据可以自然地做成减项而不会与主项二次放大。

| 因子 | 语义 | 用到的框架证据 | 计算 |
| :--- | :--- | :--- | :--- |
| `f_hit` | **观测充分度**：该 memory 被独立观测了多少次 | `state["num_obs"]`（memory 已有字段） | `f_hit = clip(log(num_obs/n_ref) / log(n_hi/n_ref), 0, 1)`，`n_ref=2, n_hi=10` |
| `f_geo` | 多视角一致性：同一 memory 在多帧中的质心漂移（越小越可信） | 每帧检出质心的窗口（`geo_window=30`） | `σ_c = sqrt(trace(M2)/M)`；`f_geo = clip(1 − σ_c / (k_geo·max(d,1)), 0, 1)`；样本 < 2 时 `None` |
| `f_cooc` | 跨通道共位：TSP3D 与 GD 是否同时指向这里（本框架独有的独立证据） | 本帧该 memory 是否被 TSP3D 召回 + `gd_hits` | 检出 → 1.0；否则 `gd_hits ≥ 2` → 0.75；`= 1` → 0.5；否则 0.25 |

**`f_hit` 为什么不用点云密度**（2026-11-06 实测后改正）：

初稿把 `f_hit` 写成「质心球 `r_in` 内点数 / (1+d²)」的对数归一。冒烟实测推翻了这个设计：

| 证据 | 数据 |
| :--- | :--- |
| 球内点数与目标对错无关 | `5cdEh9F2hJL` ep78 的 `m000000`（`tp=False`，`num_obs=2`）球内 **26446 点**；`m000002`（`tp=False`）**44965 点** |
| 原因 | 点云是完整 640×480 反投影（307200 点），`r_in=0.6 m` 球同时包含墙面、地板与邻近家具 |
| 改用 `num_obs` 的分开度 | 21 个可修复 fp 里 `num_obs ≥ 6` 全部 `tp=True`，`num_obs ≤ 2` 全部 `tp=False` |

`r_in` / `rho_ref` / `rho_hi` 降为**诊断字段**（`n_in` 仍落盘供离线分析），不参与 `MVS`。

两点设计取舍：

- **不用「帧数」当多视角信号**。`num_obs` 已能把 11 集分干净（`≥6` 全 `tp=True`，`≤2` 全 `tp=False`），但它是「次数」不是「视角独立」。`f_geo` 用**漂移**衡量一致性，正是 ApexNav 与本框架的真实差别所在：Observation 反复认同一个点，还是多个独立视角收敛到同一个 3D 位置。
- **`f_hit` 用对数归一**，沿用 §4 `f_area` 的归一思路但换成 3D 点数：近距大目标 `ρ` 自然大、远距小目标自然小，且**不需要 box**，`gd-only` 记忆同样可算。

**判定**（#2 出口）：

```
stop_ok_mvs = (MVS >= mvs_stop[origin]) and (f_geo is None or f_geo >= mvs_geo_min)
```

`f_geo` 单独设硬门，因为「多视角一致性差」是身份可疑的强信号，不应被 `f_hit`/`f_cooc` 的高分抵消；`f_geo=None`（观测样本 < 2）时硬门跳过。其余三项全部走加性，**不设单项硬门**。

**通过 = 授予持续 STOP 资格**（实现时补上的关键语义）：

`_pointnav(goal, stop=True)` 内部仍要求 `rho < pointnav_stop_radius(0.90)` 才发 STOP。首帧判定发生在 `rho ≈ 1.1`，因此「通过 → 立即 `_pointnav(stop=True)`」在 1.0–1.2 m 处根本停不下来（实测：ep78 连续 4 帧 `action=stop` 但 rho 从 1.11 降到 0.69 都没停）。

所以通过时不是「本帧 STOP」，而是**给该 memory 一个与 `gd-tsp` 同等的持续 STOP 资格**（`state["passed"]`），之后每帧走 `_pointnav(goal, stop=True)`，自然在 `rho < 0.90` 时停。这也是「不需要升级 gd-tsp 也能停」的真正含义。

初始权重（均为待标定，先落这份值让第一次运行有定义）：

| 权重 | 值 | 理由 |
| :--- | ---: | :--- |
| `w_hit` | 0.5 | `f_hit` 与 §4 的 `f_area` 语义最接近、样本最多，作主项 |
| `w_geo` | 0.3 | 多视角一致性是本节新增的唯一独立证据 |
| `w_cooc` | 0.2 | 跨通道共位强但廉价，作辅项 |
| `w_neg` | **0.0** | 负证据本批只 shadow（TP/FP 等量），不参与 MVS |

### 5.4 负证据（本批 shadow）

负证据不再作为独立判据，降为 `MVS` 的**加性减项**（不与三因子相乘），且**先只记录、不参与判定**（因为 §3 已证明其触发在 TP/FP 上等量分布）。

**计算方式调整**（与 §4 的关键差别在分母）：

```
visible   = (N_hit >= neg_min_hits) and (d <= neg_d_vis)      # 质心球内有真实点 = 真的看得见
neg_hit   = visible and (not detected) and cooled              # 看得见但 TSP3D 本帧没召回
neg_ratio = n_neg / n_visible                                 # 分母 = 可见帧数，不是 usable 帧数
MVS       = w_hit·f_hit + w_geo·f_geo + w_cooc·f_cooc − w_neg·neg_ratio
```

| 项 | §4 | §5 | 理由 |
| :--- | :--- | :--- | :--- |
| `visible` 的判据 | `f_area ≥ f_area_min`（box 投影面积） | 质心球 `r_hit` 内点数 `≥ neg_min_hits`（纯点云） | 去掉 box 依赖，`gd-only` 与未检出帧都可算 |
| 分母 | `n_usable` | `n_visible` | 「看得见」才是负证据成立的前提；用面积筛出来的分母与语义无关 |
| 合成方式 | 独立判据 `would_pass_neg`（二元否决） | **加性减项** `− w_neg·neg_ratio` | 二元否决会把“轻微负面”与“重度负面”当同一回事 |
| 是否参与判定 | 是 | **否**（shadow，`w_neg=0`） | TP/FP 等量，需先在 #2 出口上重新测判别力 |

### 5.5 接入点与行为（#2 已回滚；**MVS 代码已于 2026-11-06 全部删除**）

**`#2 nav not gd-tsp`：原左转窗口已恢复，`MVS` 只记一行 shadow（2026-11-06 回滚）**

曾把 #2 改为「进入 1.2 m 首帧判定 → 通过即授予 STOP 资格 / 不过则降权 → 两次不过拉黑」，8 集探测 −1 SR（§5.9），已**整体回滚**。当前 #2 的行为与 S8 基线完全一致：

| 帧序 | 条件 | 行为 |
| :--- | :--- | :--- |
| 到 goal 距离 ≤ `stop_gate_confirm_radius`(1.2 m) | — | `turns += 1`，写 `_lookaround_trace`，动作 = `_turn_left_action`；同时写一行 `mvs.jsonl`（`exit=nav_not_gd_tsp`、`action_taken=shadow`） |
| 同上且 `turns > stop_gate_turns`(3) | — | 该 memory 进 `_stop_blacklist`，打 `[BLACKLIST] ... reason=stop_gate_exhausted`，转 `explore` |
| 其它 | — | `_pointnav(goal, stop=False)`，继续靠近 |

`MVS` 在此出口**不参与任何决策**：不授予 STOP 资格、不改变 `tier`、不写入 `_stop_blacklist`。行内的 `stop_ok_mvs` 只是「若拿它当判据会不会过」的离线诊断列。

回滚删除的代码（`vlfm/vlvm/recheck/mvs_check.py` / `tsp3d_objectnav_policy.py`）：

| 删除项 | 作用 |
| :--- | :--- |
| `_mvs_gate_exit2()` | 首帧判定 + 授予 `state["passed"]` 的入口 |
| `_mvs_passed()` 及 `stop_ok or self._mvs_passed(...)` | 持续 STOP 资格 |
| `_mvs_blacklist()` | 只有它会让 #2 写 `_stop_blacklist` |
| `_mvs_is_skipped()` 与选择器 `tier+10` 降权 | 拒绝后的候选降权 |
| `state{"fails","skip_until","judged","in_zone","passed"}` | 上述行为的状态 |
| `mvs_active_exit2` / `mvs_skip_steps` / `mvs_skip_max` / `exit2_radius` | 上述行为的参数（ctor + `VLVMConfig` + yaml 同步删除） |

`_stop_blacklist` 因此恢复为**只有一个写入点**（`stop_gate_exhausted`），与 S8 基线一致。

**`origin` 分层（仅作为 shadow 行里的诊断分组，不再决定阈值是否生效）**：

| `origin` / `state` | fp-oracle1 | success | fp-oracle0 | 本次实测 `MVS` 判别力 |
| :--- | ---: | ---: | ---: | :--- |
| `gd-only` / `gd` | 11（tp 6） | 8（tp 5） | 30（tp 2） | 与 `tp` 近乎独立（§5.9） |
| `tsp-confirmed` / `confirmed` | 0 | 0 | 1（tp 0） | 样本不足 |
| `tsp-provisional` / `provisional` | 3（tp 0） | 0 | 2（tp 0） | 高阈值 `0.75` 下全部拒掉，但也拒掉不了 fp 之外的收益 |
| `gd-tsp` / `confirmed`（含 promoted） | 7（tp 6） | 142（tp 137） | 36（tp 2） | 不评估 |

阈值参数 `mvs_stop` / `mvs_stop_prov` / `mvs_accept_origins` 保留，仅用于 `stop_ok_mvs` 这一诊断列。

**其它落点（全部 shadow，未变）**

| 落点 | 记录内容 | 不改的部分 |
| :--- | :--- | :--- |
| #3 `expl_recheck` 队列走完 | `MVS` 是否成立、`f_*` 分量 | 仍按原逻辑在队列末尾 STOP |
| #4 explore 队列为空 | 同上 + 当时位置是否有候选 | 仍按 1A 的抑制规则 |
| mem-life `upgrade`（`provisional→confirmed`、`gd-only→gd-tsp`） | `MVS` 与 `num_obs`/`gd_hits` 的联合分布 | 升级条件一律不动 |
| mem-life `trans` / `revive` / `delete` | 同上 | 状态机不动 |


### 5.6 参数

| 参数 | 默认 | 含义 |
| :--- | ---: | :--- |
| `mvs_enable` | True | MVS 总开关 |
| `mvs_shadow_explore` | True | #3/#4 是否记录 |
| `mvs_shadow_memlife` | True | mem-life 各迁移是否记录 |
| `mvs_r_in` | 0.6 m | `n_in` 诊断字段的球半径（`f_hit` 已改用 `num_obs`，不再用它） |
| `mvs_n_ref` / `mvs_n_hi` | 2 / 10 | `f_hit` 的 `num_obs` 对数归一参考点 |
| `mvs_k_geo` | 0.10 | `f_geo` 的容差尺度（漂移 / 距离） |
| `mvs_geo_min` | 0.35 | `f_geo` 硬门（唯一硬门） |
| `mvs_stop` | 0.55（初值） | 主阈值（`gd-only` / `tsp-confirmed`） |
| `mvs_stop_prov` | **0.75** | `tsp-provisional` 的高阈值 |
| `w_hit` / `w_geo` / `w_cooc` | 0.5 / 0.3 / 0.2 | 加性权重，和为 1.0 |
| `w_neg` | **0.0** | 负证据的加性权重；0 = 只记录不参与，标定后再开 |
| `mvs_geo_window` | 30 帧 | `f_geo` 的统计窗口 |
| `neg_min_hits` / `neg_r_hit` / `neg_d_vis` / `neg_cooldown` | 5 / 0.20 m / 4.75 m / 3 | 沿用 §3.3 的可见性口径 |
| `mvs_accept_origins` | `gd-only,tsp-confirmed,tsp-provisional` | `stop_ok_mvs` 诊断列按哪些 `origin` 计算（`gd-tsp` 不在内） |

已删除（回滚 + 死代码清理）：`mvs_active_exit2` / `mvs_skip_steps` / `mvs_skip_max` / `exit2_radius`（随 #2 回滚，§5.5）、`mvs_rho_ref` / `mvs_rho_hi`（`f_hit` 改用 `num_obs` 后无人读取）。保留的阈值参数只服务 `stop_ok_mvs` 这一离线诊断列，不再影响动作。

### 5.7 日志

`detections/mvs.jsonl`（合并原 `verify_stop.jsonl` 与 `verify_frames.jsonl`）：

`scene`、`episode_id`、`target_object`、`success`、`oracle_success`、`failure_cause`、`exit`（`nav_not_gd_tsp` / `expl_arrive` / `expl_empty` / `memlife_upgrade` / `memlife_change`）、`step`、`memory_id`、`origin`、`lifecycle_state`、`distance`、`goal`（2D，**必须有，用于离线打 `tp`**）、`centroid`、`num_obs`、`gd_hits`、`sigma_c`、`n_in`、`f_hit`、`f_geo`、`centroid_drift`、`f_cooc`、`tsp3d_confirm`、`neg_ratio`、`n_visible`、`n_neg`、`MVS`、`stop_ok_mvs`、`action_taken`（回滚后恒为 `shadow`）、`tp`。

`goal` / `centroid` 必须落盘 —— 这是 §4 的最大日志缺陷：`verify_stop.jsonl` 缺这两列，导致 `tp` 恒为 `False`（4892 行全部无信息）。

### 5.8 实现与冒烟发现（2026-11-06）

代码已落：`vlfm/vlvm/recheck/mvs.py`（纯计算）+ `mvs_check.py`（`MvsMixin`，policy 联接）+ `mvs.jsonl` 通道。

**五处实现时必须改的地方**（都在冒烟里暴露，已修正）：

| # | 问题 | 现象 | 处理 |
| :--: | :--- | :--- | :--- |
| 1 | `f_hit` 用点云密度无判别力 | 错目标 `m000000`（`tp=False`）球内 **26446 点**，与正确目标同量级 | 改为 `num_obs` 的对数归一（§5.3），`n_in` 降为诊断字段 |
| 2 | 通过 ≠ 本帧 STOP | `_pointnav(stop=True)` 仍要求 `rho < 0.90`；首帧判定在 `rho≈1.1`，实测连判 4 帧 `action=stop` 但 rho 一路降到 0.69 都没停 | 通过时写入 `state["passed"]`，授予与 `gd-tsp` 同等的**持续 STOP 资格** |
| 3 | zone 距离口径要一致 | `_mvs_observe` 用「到质心」距离、gate 用「到 goal」距离，两者差 0.3–0.8 m，导致同一次接近内 `judged` 被反复重置 → 连续判 2 次 → 过早拉黑 | `_mvs_goal_distance()` 统一用 goal 距离 |
| 4 | 只判首帧 + 唯一候选会卡死 | 判 skip 后再不重判，机器人走到目标跟前（`rho=0.15`）卡到 500 步 | `skip_steps` 到期后允许**再判一次**，`skip_max=2` 次不过才拉黑 |
| 5 | `f_geo` 对 `gd-only` 几乎无数据 | 冒烟 11 行 `f_geo` **全为 `None`**：window 只记 TSP3D 召回帧，而 `gd-only` 的 `num_obs` 来自 GD 累积、TSP3D 很少召回 | 记录于 §5.9，本批只能靠 `f_hit`+`f_cooc`，`f_geo` 实际只对 `gd-tsp` 类记忆生效 |

**冒烟结果（`mvs_active_exit2=True`，默认参数）**：

| episode | run1 基线 | 本实现 | 说明 |
| :--- | :--- | :--- | :--- |
| `5cdEh9F2hJL` ep78 | **329 步成功** | **225 步 `false_positive`** | step222 对 `m000003`（`num_obs=5`，`tp=False`）判 `stop` → 误通过 |
| `5cdEh9F2hJL` ep22 | 成功 | 240 步成功 | 4 次 `skip`/`blacklist` 后正常 |
| `5cdEh9F2hJL` ep27 | 成功 | 47 步成功 | 无 MVS 介入 |

`nav_not_gd_tsp` 11 行的判决分布：`skip` 5 / `blacklist` 5 / `stop` 1，其中 `stop` 那 1 次就是 ep78 的误通过。

**结论（当时的判断，后被 §5.9 推翻）**：实现可用、shadow 完整，但**默认 `mvs_stop=0.55` 偏低**。当时以为只要调高阈值即可，8 集探测证明调阈值救不了：错目标的证据本身就更强。

**8 集探测结果（#2 接管版，默认参数，`5cdEh9F2hJL` 前 8 集；该版本已回滚）**：

| episode | run1 基线 | 本实现 | 判读 |
| :--- | :--- | :--- | :--- |
| ep20 | 成功 380 | 成功 380 | SAME（无介入） |
| ep22 | 成功 294 | 成功 240 | SAME |
| ep25 | fp 39 | fp 39 | SAME（无介入） |
| ep27 | 成功 47 | 成功 47 | SAME（无介入） |
| ep38 | fp 78 | fp 78 | SAME（无介入） |
| **ep48** | **fp 500** | **成功 166** | **GAIN** |
| **ep65** | **成功 211** | **fp 497** | **FLIP** |
| **ep78** | **成功 329** | **fp 225** | **FLIP** |
| 合计 | **5/8** | **4/8** | −1 |

判决分布（24 行）：`skip` 9 / `blacklist` 8 / `stop` 3 / `shadow` 4。`stop` 3 次里 **2 次是误通过**（`tp=False`）。

### 5.9 决定性反例：证据强度与目标对错无关

两次误通过目标的所有证据都「很强」：

| 样本 | `MVS` | `sigma_c` | `num_obs` | `gd_hits` | `tp` |
| :--- | ---: | ---: | ---: | ---: | :---: |
| ep65 `m000004`（target=`tv`） | 0.8756 | **0.895** | **22** | **20** | **False** |
| ep48 `m000001`（target=`toilet`） | 0.9286 | 0.701 | 15 | 14 | True |
| ep78 `m000003`（target=`bed`） | 0.6209 | 0.441 | 5 | 5 | False |

ep65 的错目标在**每一项证据上都超过或接近正确样本**：置信 0.895 > 0.701、观测量 22 > 15、`f_geo` 0.752 也成立。它不是「看不清」或「只瞥一眼」，而是**另一台 `tv`**。

推论（本批最重要的结论）：

| 结论 | 依据 |
| :--- | :--- |
| **`MVS` 的天花板很低** | 无论怎么调权重/阈值，三因子的输入都与 `tp` 近乎独立：`tp=False` 的 MVS 分布 `[0.14×8, 0.21×4, 0.23, 0.39×4, 0.52, 0.62, 0.88]` 与 `tp=True` 的 `0.93` 完全重叠 |
| **fp 的主体是「同类不同实例」** | ep65 找到了另一台 `tv`；ep78 在多个 `bed` 之间切换（4 个候选全 `tp=False`） |
| **这类错误无法用「看得清不清楚、看了多少次」区分** | 需要的是语义先验：易混类对照、房间先验、per-class 阈值（即 §1.4），或标注 |

因此建议：**`MVS` 不进入 active 全量实验**，代码保留（shadow 可用、`mvs.jsonl` 可作为「同实例/不同实例」的对照数据）；SR 杠杆转向 §5.10 的两项。

### 5.10 阶段与判读口径

| 阶段 | 内容 | 状态 | 判读 |
| :--- | :--- | :--- | :--- |
| S9-A | #2 接管 + explore/mem-life 全 shadow（8 集探测） | **已回滚** | −1 SR（5/8 → 4/8）；判别力不足：`tp=False` 的 `MVS` 达 0.88（§5.9） |
| S9-B | 离线：`gd-only` 子集上 `MVS` 对 `tp` 的 AUC | **停做** | 由 §5.9 分布重叠已可判定 AUC ≈ 0.5，无需再算 |
| S9-C1 / C2 | `mvs_stop` 落地 / `w_neg` 打开 | **停做** | 前置门（§5.11-D）不通过 |
| S9-D | explore 自适应切模 + 最近 frontier | **代码已落、未跑实验**（§6.1） | SR、fp、timeout、steps |

`MVS` 现为纯 shadow 通道：`mvs.jsonl` 仍逐帧记录 #2 / explore / mem-life 各出口的分数，供「同实例 vs 不同实例」的离线对照，不再影响任何动作。

### 5.11 为什么「STOP 判据」这一类机制在这套框架里必然无效

§3（memory 级置信融合）、§4（视图质量 + 负证据 recheck）、§5（多视角质量分数）三次尝试都是同一件事：在 STOP 出口加一个判据。三次结果都是 net 0 或负收益，原因是**判据的输入与 fp 的成因在信息上就不相交**，不是参数没调好。

**A. 所有候选判据都是「观测质量」的函数，而 fp 是「实例身份」错误**

| 判据 | 输入 | 实际测的是 | 对「是不是目标实例」 |
| :--- | :--- | :--- | :--- |
| `f_hit` | `num_obs`、`rho` | 这个 memory 累计被检出多少帧 | 无关 |
| `f_geo` | 质心漂移 / 距离 | 换视角后质心是否稳定 | 无关（只说明点云是同一团，不说明是什么） |
| `f_cooc` | GD 命中与 TSP3D 召回的同帧率 | 两个模型是否盯着同一处 | 只排除「GD 幻觉框」和「TSP3D 乱猜」，对**真实的另一个同类物体**完全无感 |
| `sigma_c` | 置信 × 空间一致性 | 检测质量 | 弱相关（0.69–0.76 vs 0.43–0.51），但分布重叠 |
| `q_score`（§4） | BLIP-2 ITM 图文相似度 | 图里像不像这个类 | 弱相关，且 57% 的拒绝帧为 `None` |
| `neg_ratio` | 近距离未见检测 | 该处是否**什么都没有** | 只对「空气」有效，对「错的那一个」无效 |

它们都是 $\hat p(\text{观测} \mid \text{物体存在})$。判 STOP 需要的是

$$
p(\text{该物体} = \text{目标实例} \mid \text{观测}) \propto \underbrace{p(\text{观测} \mid \text{实例})}_{\text{上述全部判据}} \cdot \underbrace{p(\text{实例} \mid \text{场景})}_{\text{先验，框架里没有}}
$$

缺的正是后一项：这个房间里有几个同类、目标类别的易混类是什么、这类物体通常出现在哪。没有这一项，前一项估计得再准也只描述「看清楚了没有」。

**B. 三条结构性理由**

| # | 理由 | 后果 |
| :--: | :--- | :--- |
| 1 | 一旦某 memory 被锁成 goal，导航就把相机钉在它身上；此后**所有新证据都采自同一处** | 观测质量单调变好，「对错」完全不变。判据只能看到「越来越清楚」。ep65 的 0.895 不是噪声，是真实的清晰度 |
| 2 | fp 与 success 在这些维度上**分布重叠**（`|goal−centroid|`、`num_obs`、`sigma_c`、`MVS`） | 同分布上的阈值只能沿 ROC 滑动，收益与损失同阶。§4 修完 4 个 bug 后 297 集仍 net 0，就是这个必然性的实测 |
| 3 | STOP 不可回收，误拦截可回收 | 一次错误放行 = 该 episode 立即变 fp；一次错误拦截 = 多花几步。损失/收益不对称，任何 AUC < 1 的判据都会偏向损失 |
| 4 | **STOP 时刻的可用信息最少** | 1.2 m 处视野窄、只看到物体的一部分、多视角机会被导航终止（原左转只有 4 帧）。用最少的证据做最不可逆的决定 |

**C. 什么样的判据理论上才可能有效**

必须引入框架当前不具备的信息源，而不是把现有信号重新加权：

| 方向 | 为什么可能有效 | 代价 |
| :--- | :--- | :--- |
| 语义先验：易混类清单、per-class 阈值、房间先验（`bed` 在卧室） | 直接提供 $p(\text{实例}\mid\text{场景})$ 这一项 | 需要类别清单与统计，§1.4 |
| 指令侧属性验证（颜色/材质/邻近物体） | 「同类不同实例」的唯一可判据信息源 | 需要指令解析与属性检测器 |
| 主动反证据搜索：去同一房间找其它同类物体再比较 | 把「看了多少次」变成「和谁竞争」 | 改变了控制流程，不是 STOP 判据 |
| 标注 / oracle | 只用于离线分析，不能上线 | — |

**D. 由此得到的上线门槛（写进验收门，避免第四次重犯）**

任何 STOP 判据在跑全量前，先离线算两个数：

| 指标 | 门槛 |
| :--- | :--- |
| 该判据在「成功末态 vs fp 末态」上的 AUC | **≥ 0.85** 才允许 active；0.65–0.85 只允许 shadow；< 0.65 停做 |
| 在成功集上的误拦截率（本可 STOP 却没有） | **≤ 2%** |

`MVS` 的实测（probe8，`nav_not_gd_tsp` 20 行，`tp=True` 仅 1 行）：

| 口径 | 数值 | 说明 |
| :--- | :--- | :--- |
| 全样本 AUC | 1.00 | 假象：19 个 `tp=False` 里有 13 个 `MVS ≤ 0.23`，把分数拉低的是「只看到 1-2 帧的候选」，不是「看得很清楚的错物体」 |
| 决策区间（`MVS ≥ 0.55`）内的样本 | 3 个：0.9286(`tp`) / 0.8756(`fp`) / 0.6209(`fp`) | **precision = 1/3**，且最高分 fp 与唯一 tp 只差 0.05 |
| 把阈值提到 0.90 后 | 只剩 1 个样本（ep48 的 tp） | 等于「几乎全部拒掉」，退化成不接管；ep48 的收益其实来自「拒绝后选择器换了目标」，不是判据判对了 |

也就是说：整体排序看似可分，但**与决策相关的分数段上 tp/fp 混叠**，这正是「调阈值救不了」的定量证据。

**E. 回滚验证（8 集，`QaLdnwvtxbs`；运行目录已于 2026-11-06 清理，数据见下表）**

与 run1 基线（`outputs/s8_verify_r6_3scene_2026-11-05_200949`）同场景同 episode 逐集对照：

| episode | 回滚后 | run1 基线 | 判读 |
| :--- | :--- | :--- | :--- |
| 46 | fp | fp | SAME |
| 43 | 成功 | 成功 | SAME |
| 16 | 成功 | 成功 | SAME |
| 59 | 成功 | 成功 | SAME |
| 97 | fp | fp | SAME |
| 52 | 成功 | 成功 | SAME |
| 66 | 成功 | 成功 | SAME |
| 21 | fp | fp | SAME |
| 合计 | **5/8** | **5/8** | **8/8 SAME** |

`mvs.jsonl` 187 行，`action_taken` 全为 `shadow`（`nav_not_gd_tsp` 163 / `memlife_change` 20 / `expl_arrive` 4），确认 MVS 已完全退出动作链路，仅保留观测通道。

### 5.12 代码清理记录（2026-11-06，S9 全部回退）

S9 的 MVS 实验结束，代码整体移除，工作树回到「无 MVS、无 §4 verify」的 S8 形态。

**删除的文件**

| 文件 | 原职责 |
| :--- | :--- |
| `vlfm/vlvm/recheck/mvs.py` | 纯计算：`f_hit` / `f_geo` / `f_cooc` / 负证据 / 加性融合 |
| `vlfm/vlvm/recheck/mvs_check.py` | `MvsMixin`：per-memory 统计 + 各出口 shadow 行 |

**删除的接线**

| 位置 | 删除内容 |
| :--- | :--- |
| `recheck/__init__.py` | `MvsConfig` / `MvsMixin` 导出 |
| `recheck/expl_recheck.py` | `expl_arrive` 的 shadow 调用 |
| `policy/tsp3d_objectnav_policy.py` | `MvsMixin` 基类；16 个 `mvs_*` 构造参数；`_init_mvs` / `_mvs_reset`；`detected_ids` / `detected_centroids` 记账；`_mvs_target_centroid_3d`；逐帧 `_mvs_note_views` / `_mvs_observe`；`_mvs_goal_distance`；#2 / `expl_empty` / mem-life 三处 shadow 调用；`VLVMConfig` 的 16 个 `mvs_*` 字段 |
| `utils/episode_stats_logger.py` | `mvs_logs` 参数、`mvs.jsonl` 通道、`n_mvs` 字段、返回值 `mvs` 计数 |
| `utils/vlvm_trainer.py` | `mvs_logs=` 传参 |
| `config/experiments/vlvm_objectnav_hm3d.yaml` | `mvs_*` 配置块 |

**验证**

| 检查 | 结果 |
| :--- | :--- |
| `py_compile`（5 个改动文件） | 通过 |
| `python -m vlfm.run --cfg job` | 471 行解析成功，`mvs` 相关字段为 0 |
| `dataclasses.fields(VLVMConfig)` | 无 `mvs` 字段 |
| `TSP3DObjectNavPolicy.__mro__` | `ExplRecheckMixin` 之后直接 `BasePolicy`，无 `MvsMixin` |
| 冒烟（同 3 集跑两次） | 两次完全相同：`QaLd ep78 fp 32` / `ep22 成功 77` / `ep27 成功 65` |
| 输出目录产物 | 不再生成 `mvs.jsonl`，`episodes.jsonl` 无 `n_mvs` |
| 与历史 run 对照 | `ep78` 与 `s8_fusion_shadow_3scene_2026-11-04_215549`（无 verify、297 集）的 **fp 32 步完全一致**；`s8_verify_r6`（§4 verify active）的 42 步 success 属已删机制的效果 |

即：清理后的行为等于干净的 S8 基线，`ep78` 的 32 步 fp 是 S8 本身的失败，不是回退引入的回归。

### 5.13 状态核对：当前代码 vs `s8_fusion_shadow_3scene`（2026-11-06）

做 explore 改造前的基线核对。

**配置 diff**（当前 `--cfg job` vs 那次 run 的 `resolved_config.yaml`，排除输出路径 / episode 上限 / eval 覆盖）：

| 项 | 差异 |
| :--- | :--- |
| `volume_fusion_*`（9 行） | 那次有（§3 shadow 融合，只记录不改动作），现已随 §3 弃用删除 |
| 其它 | 无 |

**行为对照**（同 3 场景 8 集，逐集比对 `failure_cause` + `steps_count`）：

| scene / ep | 当前 | `s8_fusion_shadow` | 判读 |
| :--- | :--- | :--- | :--- |
| 5cdEh9F2hJL ep46 | 成功 417 | 成功 417 | SAME |
| ep43 | fp 38 | fp 38 | SAME |
| ep16 | fp 53 | fp 53 | SAME |
| ep59 | fp 173 | fp 173 | SAME |
| ep97 | 成功 332 | 成功 332 | SAME |
| ep52 | 成功 144 | 成功 144 | SAME |
| ep66 | fp 500 | fp 500 | SAME |
| ep21 | 成功 300 | 成功 300 | SAME |
| 合计 | 4/8 | 4/8 | **8/8 SAME，步数逐集完全相同** |

结论：当前工作树在**动作层面等价于 `s8_fusion_shadow_3scene_2026-11-04_215549`**，唯一差别是那次多一个 §3 的 `fusion_field.jsonl` shadow 日志通道（不参与决策）。

**过程记录**：核对时发现 18:03 有四个文件（policy / trainer / episode_stats_logger / yaml）被回退成 MVS active 版，而 `mvs.py` / `mvs_check.py` 已删 → `ImportError`。重新清理后又暴露出 `_query_and_process` 里 `detected_ids.add(...)` 的残留（`NameError`），一并删除后通过。两次问题都出现在「回退版本 + 已删模块」混搭状态，不属于设计缺陷。
