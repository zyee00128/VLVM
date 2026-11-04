# VLVM 机制候选：跨论文可借鉴项与内部联动

> 定位：从 ApexNav / ZONDA / GLMap / TrajRAG / ASCENT / MFNP 中筛出可迁移到 VLVM 的四项机制，说明落点、边界、与 S7 既有机制的联动。
> 前三项按**复杂度升序**排列；第四项（易混类清单 + 自适应阈值）实际复杂度最低，排在末尾仅为编号沿用。
>
> 状态：设计记录；M-0（三点记录 + 日志）已实现。**2026-11-04 决定：field 机制弃用；1.1 退化为 memory 级实现，label 级保留到随 1.4 一起做。** 实验由人工启动。
> 边界：不改 2.5D 语义价值场、几何障碍场、探索/导航机制、回退机制；不改 TSP3D 权重；floor 机制已删除；不回退 2D 框架。

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

## 3. 实验计划

### 3.1 约束

- **1.1 的实现位置**：`memory_life/` 下新文件，由 policy 在 **occ gate 之后、`sigma_c` 计算之前** 调用，输出 `c_fused` 替换进入 `s_penalty` 的 `conf_f`；`sigma_c = c_fused · w_S` 公式不变。不并入 `s_penalty`（它是无状态逐帧乘子，表达不了跨帧累积与未检出衰减）。
- **1.1 本轮为 memory 级**：单标签年龄加权 —— 体积（box 体积或框内点数）加权累积 + 未检出负向降分。label 级随 1.4 一起上。
- **阈值先不动**：沿用 `sigma_tar=0.60`、`sigma_prop=0.45`、`provisional 0.50–0.60`、`sigma_band`、`upgrade_min_obs=4`。重标定放到本轮出结果之后，用 E3 口径（val 2638 视图、`sigma_tar=0` 取 raw Top-1）叠加 `c_fused` 分布重扫。
- **视角置信先不动**：value map 保持现状（`use_max_confidence=True`）；平均加权 / 平方加权只以 shadow 记录分数，不改变 frontier 选择。
- 每批单独可回退；floor 已删除；field 已弃用，不再参与。

### 3.2 批次

| 批次 | 改动 | 动作是否变化 | 主指标 |
| :--- | :--- | :--- | :--- |
| M-0 ✅ | 三点记录 + 日志（`point_surface_first` / `point_surface_latest` / `point_first_latest_shift_m`） | 否 | `latest − first` 位移分布、越 bbox 比例、first 过期比例 |
| E-1 | 1.1 memory 级，**开关默认关**：只记录 `c_raw` / `c_fused` / `v` / `n` / 未检出事件 | 否 | `c_fused` 与 `c_raw` 分布位移；被降分项里 TP/FP 比例 |
| E-2 | 1.1 开关打开（`c_fused` 进 `s_penalty`），阈值沿用现状 | 是 | FP、never_saw、TP 提升数、timeout、steps |
| E-3 | value map 视角置信 shadow：同一次运行内额外记录 `max` / 平均 / 平方加权三种 S | 否 | 三种 S 在 frontier 上的 Top-1 一致率与分叉率 |
| E-4 | 阈值重标定（离线，基于 E-2 落盘的 `sigma_c` 分布） | 否 | 全局 vs per-class 的准入 TP/FP 曲线 |
| E-5 | 视角置信 A/B（按 E-3 结果决定是否替换 max） | 是 | SR、FP、steps |

顺序：M-0 → E-1 → E-2 →（E-3 与 E-2 可并行）→ E-4 → E-5。

### 3.3 验收门

- **E-1**：未检出降分不能主要落在真 target 上。若「近距离仍被降分」占比过高，说明该机制在 VLVM 的漏检结构下不成立，停止后续。
- **E-2**：FP 下降且 `never_saw` 不上升；`timeout` 不成为主要新增失败类型；步数增幅有界。若 SR 下降但准入 TP 数与被拒 TP 数的对比显示是阈值不匹配，记为「待 E-4」，不判机制无效。
- **E-3**：三种 S 的 Top-1 一致率接近 1 时，E-5 不必做。
- **E-4**：per-class 阈值在 E3 离线口径上优于全局单一阈值。
- **E-5**：SR 提升且 FP 不升。

### 3.4 日志口径

- 三点字段进 `memory_events.jsonl` 的 `new`/`merge` 事件，由 `vlfm/vlvm/utils/three_points.py` 生成（只读，不改动作）。
- 1.1 的 `c_raw` / `c_fused` / `v` / `n` / 未检出事件建议单独一个 jsonl（`detections/fusion_field.jsonl`），避免与 memory 事件按 `memory_id` 聚合时重复计数。

