# Project VLVM: 3D Native 主动视觉语言导航系统

> **阶段状态（2026-09-01）**：**Stage1 完结**——单场景 `5cdEh9F2hJL` 完成全部机制调优并定稿，定稿档 = `wm_near_refresh_val`（SR **0.4444** / spl 0.1538 / soft_spl 0.2029），全链路结果见 `Results_final.md` 与 `VLVM-S1/Results.md`。

## 1. 项目背景与技术范式演进 (Project Background & Paradigm Shift)

本项目名为 **VLVM**，基于开源视觉语言导航框架 **VLFM** (`https://github.com/rai-opensource/vlfm`) 进行重构。我们的核心目标是将传统的 **2D 投影启发式导航** 升级为**纯三维几何与语义原生（3D Native）主动导航系统**。

### 1.1 传统 2D 方案（VLFM）的局限性
* **视角降维与深度残缺**：高度依赖 2D 像素级目标检测（Grounding-DINO/YOLO）、实例分割（MobileSAM）及单目/深度图投影。在复杂室内环境中，2D 投影常因深度尺度压缩、边缘出血（Depth Edge Bleeding）和缺乏真实三维感知而导致空间定位失真、产生空中鬼影点。
* **跨层与遮挡失效**：2D 方案难以有效应对多层跨层（Stairs）、复杂遮挡及大场景下的几何尺度震荡。

### 1.2 我们的技术范式（3D Native Paradigm）
* **全三维语义空间闭环**：彻底抛弃 2D 像素掩码依赖，将 RGB-D 传感器输入直接反投影为真实米级尺度的彩色 3D 点云。
* **TSP3D 原生三维接地**：引入 **TSP3D**（Text-guided Sparse Voxel Pruning）在 3D 体素空间中直接回归物理边界框（3D BBox）。


## 2. 核心架构与系统设计 (Core Architecture & System Design)

### 2.1 三维几何障碍建图 (Geometric Obstacle Mapping)
* **后端演进**：对比测试了稠密三态网格 (`ObstacleMap3D`)、八叉树 (`OctoMap`，因建图慢、状态不稳定已**弃用**) 与概率稠密体素网格 (`ProbabilisticGrid`)。
* **最终方案**：采用 **`ProbabilisticGrid`（累积 explored 掩码 + log-odds 贝叶斯更新 + 消费层排除障碍 `explored_area & ~_map`）**。它能够有效容忍噪声、支持多帧观测的自我纠正，并在内存与计算上保持高效。

### 2.2 2.5D 语义价值场 (Semantic Value Field)
* **演进历程**：从初版的 3D 语义体素（过度设计且开销大）演进为**2.5D BEV 价值场**。
* **核心数学模型**：
  $$S_{\text{final}}(x,y) = S(x,y) + \lambda \cdot H_1(x,y)$$
  * $S(x,y)$：平面基准语义价值（cosine 相似度分数）。
  * $H_1(x,y)$：高度轴通行带内的自由空间占比（0/1 归一化语义价值分数）。
  * $\lambda$：高度轴奖励权重（定稿 $\lambda = 0.3$）。
* **样式策略**：定稿采用 **`ITMPolicyV1`（区域式 Region Style + H1）**，在保持与 VLFM 2D 策略兼容的同时引入了真三维高程奖励。

### 2.3 TSP3D 目标定位与输入适配 (TSP3D Localization & Input Adaptation)
* **时间融合滑窗**：支持 **Camera 路线**（8 帧滑窗 + 视角去重 `cam_B1`）与 **World 路线**（累积地图 + 帧窗口 `wm_f8`）。
* **距离自适应采样**：在发送至 TSP3D 前进行近密远疏的分档体素化，兼顾近场精度与远场效率。


## 3. 关键机制与优化历程 (Key Mechanisms & Optimization Journey)

### 3.1 耗时优化 (Time Optimization)
* **痛点**：早期版本单步耗时高达 ~1.85s，模型推理快但通信层（明文 JSON 传输 20 万点云）成为巨大瓶颈。
* **手段**：
  1. 点云二进制化 + `float16` 传输（payload 从 24.7MB 降至 ~6MB）。
  2. 服务端解析由 JSON 改为原生 `bytes`。
  3. **消除无效重试**（`enable_retry: False`）：移除了低效的空结果重试，使单步耗时降至 **~0.90s**，总耗时缩减近一半。

### 3.2 假阳性治理与 S-penalty 软 Gate 治理 (False Positive Governance)
* **痛点**：TSP3D 在在线局部点云上容易产生高置信“幻觉框”（假阳性占失败案例的 85% 以上）。
* **核心治理机制**：
  * **反幻觉回退机制 (Enable Fallback)**：引入近场重检测确认与滞回删除，锁定的目标若经接近后未被证实，主动放弃并**回退至 explore 状态**重新探索。
  * **S-penalty 软 Gate 治理**：利用 BLIP-2 ITM 独立语义场对 TSP3D 置信度进行交叉验证（$c' = c \cdot w_S(S)$）。
  * **未覆盖免罚 (No-cov Free)**：对语义场未覆盖的目标（$S \le 0$）不进行盲目惩罚，保障真目标首检召回。
  * **近表面点优化 (`goal_use_surface=True`)**：导航终点由质心优化为 bbox 近表面点，使机器人能直接停靠在物体表面，大幅提升路径效率（步数减少 ~17.3%）。
  * **WM 近场采用滑动更新机制**


## 4. 遗留的缩短推理时间 / 降低内存占用的机制

### 一、地图更新 / 查询降频类

> **共同前提**：`update_map` 在各模式均以 `explore=True` 每步无条件调用；实测瓶颈为 explored 连通域裁剪 `ndimage.label` 全量 ~45ms/步（占 `update_map` ~84%）。

| # | 机制 | 超参 | 预估收益 | 风险 / 注意 | 状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **explored 连通域裁剪降频**：`ndimage.label` 每步 ~45ms（当前最大 CPU 瓶颈），隔 N 步裁剪或仅 explore 阶段执行 | `explored_refresh_interval` | ~40ms/步 | 低-中；explored 略滞后/孤立噪声；**最敏感，需 A/B（前 15~20 集不恶化）** | 未实施 |
| 2 | **navigate 阶段降频更新**：目标锁定后传 `explore=False`（不做 label+射线标记） | `navigate_explored_interval` | ~50ms/步 | navigate 不消费 value map/H1，**启用回退**，降频 + 回 explore 前强制刷新一次 | 未实施 |
| 3 | **探索阶段地图更新降频**：几何地图隔 N 步重建 | `map_update_interval` | 整体再省 | 低-中；地图滞后；**较敏感，需 A/B** | 未实施 |
| 4 | **语义场更新降频**：`_update_value_map` 隔 N 步 | `value_map_update_interval` | 待测 | 需 A/B | 未实施 |
| 5 | **TSP3D 查询条件化/降频**：navigate 且离目标 > 某距离时跳过/降频查询（目标已锁定，TSP3D 只服务近场回退确认），近场恢复每步；探索阶段隔 N 步查一次 | `tsp3d_query_interval` | 每省一次查询省 ~0.4-0.7s；navigate 通常占 30-50% 步数 | 需 A/B | 回退靠"本帧新鲜 merge 证据"，降频延迟确认——近场必须每步查；探索阶段降频延迟目标发现，步数略增（净收益仍大） |


### 二、采样 / 降采样类

| # | 机制 | 超参 | 预估收益 | 风险 / 注意 | 状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 5 | **射线标记降采样**：`_mark_explored_along_rays` 1000→500 条 | `ray_max_samples` | ~3ms | 低 | 未实施 |
| 6 | **V2 表面点落格降采样**：`step=8²` 加大 | `surface_sample_step` | 待测 | 低（仅 V2 表面式，V1 区域式不涉及） | 不实施，V2暂不使用 |
| 7 | **TSP3D 融合采样规模**：`fuse_max_points` / `fuse_voxel_size` | 0.02/200k | — | ⚠️ **已 A/B 否决降档**（0.03/50k SR 0.1010、丢 5 集无挽回；0.02/50k 仍丢 {50,57}），保持 0.02/200k（=基线，保准确率） | 已否决 |


## 5. 参数网格搜索总览

> VLVM 全部可网格搜索的新增参数一览。

- 基础参数 `depth_image_shape` `fov_angle` `camera_height` `text_prompt` `visualize` `init_turn_steps` `min_depth` / `max_depth` `pointnav_stop_radius` `use_vqa` / `vqa_prompt` 与VLVM同值

- 回退机制 `fb_near_radius`= `max_depth×0.5` 对应 VLFM；
- 几何障碍场 `voxel_size=0.05m` 对应 VLFM 2D 顶视图 `pixels_per_meter=20`≈0.05m；`hole_area_thresh`，`obstacle_map_area_threshold`，`agent_radius` 对应 VLFM一致；`agent_height` 对应 VLFM 相机高度 0.88；
- 几何障碍场的障碍高度带 `min_obstacle_height` / `max_obstacle_height` = 0.15 / 1.50 对应 VLFM 0.61 / 0.88
- 语义价值场 `query_radius_m` 对应 VLFM `sort_waypoints(frontiers, 0.5)` 固定 0.5m

| 参数 | 机制（归属） | VLVM 当前值 |
| :--- | :--- | :--- |
**--- 回退机制 ---**
| `fb_hysteresis` | 可信目标删除滞回（帧） | 5 |
| `fb_suspicious_hysteresis` | 可疑目标删除滞回（帧） | 2 |
| `fb_suspicious_conf` | 可疑置信度分界 | 0.75 |
**--- 几何障碍场 ---**
| `log_odds_occ` / `log_odds_free` | 概率体素 log-odds 增量 | 2.0 / -2.0 |
| `occ_threshold` / `free_threshold` | 占/空判定阈值 | 0.0 / 0.0 |
| `nav_slice_height` | 导航高度切片（m） | 0.30 |
**--- 语义价值场 ---**
| `h_lam` | H1 高度轴奖励权重 | 0.3 |
| `h_norm_max` / `h_z_min` / `h_z_max` | H1 高度带参数 | 1.0 / 0.15 / 0.88 |
| `query_z_min` / `query_z_max` | 查询高度带（m） | 0.15 / 1.50 |
**--- TSP3D query（对应 VLFM 2D 检测阈值 `coco_threshold=0.8` / `non_coco_threshold=0.4`） ---**
| `sigma_tar` | 目标置信度准入阈值 | 0.70 |
| `sigma_sce` | TGP 体素保留阈值 | 0.15 |
| `tau` | CBA 补全阈值 | 0.15 |
| `near_field_dist` | 近场自适应距离（m） | 1.0 |
| `near_field_sigma_scale` | 近场 sigma 缩放 | 0.8 |
**--- 融合器机制 ---**
| `pcd_window_size` | 帧窗口上限（camera 时间融合） | 8 | 
| `wm_max_frames` | 帧窗口上限（world 时间融合） | 8 |
| `cam_radius` | 半径滑出（camera 发送前裁剪） | None |
| `wm_radius` | 半径滑出（world 地图体素滑出） | 6.0 |
| `cam_min_view_disp` / `cam_min_view_yaw` | 视角去重（camera 进入端） | 0.15 / 15° |
| `wm_min_view_disp` / `wm_min_view_yaw` | 视角去重（world 进入端） | 0.15 / 15° |
**--- S-penalty 机制 ---**
| `s_penalty_thresh` | S-惩罚轴：S 阈值（S 低于则罚 $w_S$<1） | 0.15 |
| `s_penalty_floor` | S-惩罚轴：惩罚下限（w_S 下界，保召回） | 0.3 |
| `s_penalty_radius_m` | S-惩罚轴：S 查询半径 | 0.5 |
**--- 距离自适应采样机制 ---**
| `near_dist` / `mid_dist` | 近/中/远带分界 | 1.5 / 3.0 |
| `near_voxel` / `mid_voxel` / `far_voxel` | 各带体素尺寸 | 0.01 / 0.02 / 0.05 |
