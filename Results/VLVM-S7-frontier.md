# Project VLVM Stage 7（续）：explore 终止后的延迟 STOP（recheck）

> 定位：承接 [VLVM-S7.md](VLVM-S7.md) 的机制一（目标确认与探索终止）。
> 本文只覆盖两件事：**A. explore STOP → 延迟 STOP（实现，主线保留）**、**B. field 机制（已弃用，代码已删除）**。
> 明确不做：floor / 跨层机制冻结；不回退 2D 框架；不改 TSP3D 权重训练。
>
> 状态：A 已实现并跑满 198 集（`201331`）；2026-11-04 做过一轮“自适应到达预算”实验（110 集后因退化停止），结论是**自适应无收益且自我制造超时**，已删除代码，回到单一固定值 `recheck_arrive_steps = 45`。**S7 到此结束，不再调 recheck 机制。** B（field）已弃用，代码不保留。

## 0. 前提与边界

- **VLFM / VLVM 基线都不能跨层搜索**。`frontier_sensor` 的 `top_down_map` 按 `ref_floor_height=agent_y`、`ceiling_height=2.0 m` 切片，只含当前楼层；输出 `(N,2)`，PointNav 只有 `rho/theta`，无楼层/楼梯语义。
- **floor 机制冻结**：已有显式通道但实测未跑通（`2026-11-03_005206`：198 集仅 4 集上楼；VERIFIED entry 后 `at_entry_timeout`）。冻结的代价是 `never_saw` 不可回收。
- **机制 A 不依赖 floor**：explore STOP 后的可用预算只能转为“对已有候选的确认”，不是“再探索”。
- 主代码位置：`vlfm/vlvm/recheck/expl_recheck.py::ExplRecheckMixin`（原 `mode.py::RecheckMixin`；策略通过多继承接入），`vlfm/policy/tsp3d_objectnav_policy.py::_explore_safe` 调用。

## 1. 失败结构（机制 A 的决策依据）

### 1.1 三场景基线（297 集，`s7_baseline_3scene_2026-10-30_004608`）

| 项目 | 数值 |
| :--- | ---: |
| success / SR | 141 / 47.47% |
| Oracle | 46.46% |
| SPL | .2182 |
| steps mean | 157.04 |
| `false_positive` | 100 |
| `never_saw`（likely + feasible） | 48 + 3 |
| `timeout_true_positive` | 2 |
| `bad_stop_true_positive` | 2 |
| `false_negative` | 1 |

终止方式拆分：

| 终止方式 | n | oracle=1 | 有 TP 记忆 | 最终目标 |
| :--- | ---: | ---: | ---: | :--- |
| `navigate` 后 STOP | 45 | 8 | 12 | 全部 `gd-tsp` |
| `explore` 返回 STOP | 33 | 3 | 13 | 全部 `gd-only` |
| 500 步超时 | 22 | 13 | 17 | `gd-only` 17 / `tsp-provisional` 5 |

关键事实：33 个 explore-STOP 失败里 13 个**已经有 TP 记忆**，22 个超时里 17 个已有 TP 记忆、13 个 `oracle=1`。目标被看到过，但没有达到可 STOP 的确认条件（`gd-tsp confirmed`）——这是机制 A 要吃的残留。

### 1.2 1A（STOP 抑制）的既有结论

| 指标 | baseline | M1（1A） |
| :--- | ---: | ---: |
| success / SR | 141 / 47.47% | 142 / 47.81% |
| steps mean | 157.04 | 265.87 |
| 最终 `explore/0` | 85 | 0 |
| `false_positive` | 100 | 105 |
| 抑制步数合计 | — | 31,983 |

事实：STOP 抑制本身已成立，但只是转向/回旧 frontier 不产生新信息，步数 +69.3%、SR +1。**机制 A 要补的是“STOP 之后去哪里”**，不是重复抑制。

## 2. 机制 A：explore STOP → 延迟 STOP（已实现）

> 2026-11-04 重构：只删除对动作零作用的成分（到达后双查询提升判定），保留原有行为。机制只做两件事：**把 STOP 推迟**、**把剩余步数花在已有记忆的航点上**；STOP 仍发生在队列耗尽那一刻、机器人所在的位置——**因此 SR 语义与 `s7_recheck_2scene_2026-11-03_201331` 完全一致，不引入预算与额外约束**。

### 2.1 问题

- 同层 frontier 耗尽时，同层动作已无信息增益；继续转向只是消耗预算。
- 但“立即 STOP”也不是好选择：STOP 判据是「1 m 内有目标实例」，早停等于把最后一点距离丢掉。
- 此时剩下的记忆（已被选择过、黑名单或 stable 之外）虽然确认失败，但它们的质心仍然是“目标曾出现过的位置”——把它们当作剩余步数的航点，是唯一还能产生近距离观测的手段。

### 2.2 触发条件

代码位置：`_explore_safe()` 的尾部。

1. `act()` 只在 `_get_target_object_location()` 返回 `None` 时进 explore（即当前没有任何可导航目标）；有 `gd-tsp confirmed` 时策略处于 navigate，复核**不可能**触发。因此“`gd-tsp` 不参与复核”不是额外过滤，而是触发条件本身的结果。
2. explore 路径内部，只有基线 `_explore()` 返回动作 0（STOP，即同层 frontier 为空）或 stuck-retry 后仍 STOP 时才进入复核判定；不再进入 1A 的转向/旧 frontier 分支。
3. `_recheck_begin()` 能凑出非空候选队列才真正进入复核；凑不出则直接放行 explore STOP。
4. 楼层/entry 不参与判定（floor 冻结）。

### 2.3 候选池与排序

`_recheck_candidates(robot_xyz)`：

- 类别来源：`self._target_object` 按 `|` 拆分后的每个类，遍历 `_target_3d_memory` / `_target_surface_memory` / `_target_verify_state`。
- **黑名单不排除**（关键结构事实，见 §2.6）；排除 `stable`、排除本集已尝试过的 `memory_id`（`_recheck_attempted`）。
- 目标点：`goal_xy` 取 `centroid[:2]`，当 `goal_use_surface=True` 且有表面点时改用表面点。
- 档位 `tier`：`0` gd-tsp（`origin=="gd-tsp"` 或 `promoted_from_gd_only`）、`1` gd-only、`2` confirmed、`3` provisional、`4` 其它。
- **排序键 = 复用目标选择 comparator**（与 `_get_target_object_location` 完全同构）：`tier → sigma_band → −gd_hits → ⌊distance/0.5⌋ → num_obs 满不足 → −num_obs → stale → memory_id`。其中 `sigma_band` 只在 tier ∈ {0,2,3} 参与、`gd_hits` 只在 tier ∈ {0,1} 参与，与选择侧一致。队列建好即固定，`_recheck_advance` **不再按当前位置重排**（比最近优先更早的版本已回滚）。

### 2.4 状态机（巡回 → 耗尽 → STOP）

| 步骤 | 方法 | 行为 | 职责 |
| :--- | :--- | :--- | :--- |
| 建队 | `_recheck_begin` | 构建队列、`_recheck_active=True`、记录 `recheck_start` | 把“要不要 STOP”变成“先走这段队列” |
| 逐点巡回 | `_recheck_act` | `_recheck_total_steps += 1`；`rho > pointnav_stop_radius` 则 `_pointnav`，否则记 `recheck_result{outcome=arrived}` 并推进 | 把机器人送到航点旁，制造近距离观测 |
| 到达超时 | `_recheck_act` | 单候选步数 > `recheck_arrive_steps` → `outcome=arrive_timeout` → 推进 | 防远候选卡死 |
| 收尾 | `_recheck_stop` | 队列耗尽 → `_recheck_active=False`、`_explore_exhausted=True`、记 `recheck_exhausted{total_steps, outcome="queue_exhausted", best_memory_id, best_distance}`、就地返回 `_stop_action` | 受控的 explore STOP 出口；与旧版同一步、同一位置 |

**已删除的无作用行为**：`_recheck_evaluate`（到达后重读 `origin/promoted_from_gd_only` 并判定提升）。在只读前提下这个结果不可能改变任何动作（`_stop_blacklist` 不释放 → `_get_target_object_location` 永不选中它），纯属日志开销。**也未引入预算闸与锚点回走**：两者都会改变 STOP 时机/位置，破坏与旧版的 SR 可比性；锚点作为可评估想法转到日志（`best_distance`，见 §2.9）。

### 2.5 不变量（为什么它是“只读”的）

- **STOP 不变量**：目标 STOP 仍只由 `gd-tsp confirmed` 触发；explore 的 STOP 只在「frontier 为空 且 复核队列已耗尽」时允许发出。1A 的“explore 永不 STOP”被改为“复核未耗尽前禁止 explore STOP”。
- **排序/准入/生命周期不变**：不改普通目标选择排序、不改准入/merge/delete/revive、不改 TSP3D 输出与阈值口径。
- **不解除黑名单**：机制代码里根本没有 `_stop_blacklist.discard`，也没有 `blacklist_released` 事件；因此被拉黑的记忆永远不会重新回到目标选择。机制不再“复核后决定是否恢复”，而是直接把队列当作走位路径。
- 异常隔离：`_recheck_begin/_recheck_act` 均 `try/except BaseException`，出错只打印 `[RECHECK] ...` 并退出复核，不影响导航。

### 2.6 为什么候选池必须包含黑名单条目（结构事实）

`act()` 只有 `_get_target_object_location()` 返回 `None` 才进 explore，而该函数本身**跳过 stable 与 `_stop_blacklist`**（但 gd-only / provisional 是可选中的）。也就是说：explore STOP 的那一刻，“可选中候选”已为空，剩下的只能是黑名单或 stable 条目。若队列也过滤黑名单，队列恒为空、机制等于不存在。

这也定下了能力上界：**队列里全是“已经被选过、已经到过、已经确认失败”的记忆**，所以机制只能“把 STOP 推迟并把落点挑好”，不能发现新目标；同理，它的命中率天花板就是黑名单条目的 TP 率（实测 20–21%，见 §2.9）。

### 2.7 参数（Hydra 超参数，原为环境变量）

| 参数 | 默认 | 含义 |
| :--- | ---: | :--- |
| `recheck_enable` | `True` | 机制总开关；关掉就是基线 explore STOP |
| `recheck_arrive_steps` | **`45`** | 单航点到达预算（固定值）；机制唯一节流阀 |

**为什么是 45**：所有已观测的到达步数 p100 = **38–39**（`201331` 与 `radapt` 两个 run），而“自适应造成的超时”那些航点实测需要 39–48 步（`used 25–36` + 剩余 `0.9–2.3 m × 6.94 步/m`）。取 45 可以救回 5/6 个这样的航点，只比已知可用的 40 多 5 步；取 50 能覆盖全部（含 47.6 的孤立样本），取 55 则覆盖 `201331` 里那 4 个真长途超时（需 ≈54）。选 45 作为成本/覆盖的平衡点。

不设队列总预算、不设候选上限 K、不设距离自适应（均已试验或评估后否定，见 §4）。

**`recheck_arrive_steps = 40` 的来源**：无标定依据，是实现时的拍定值。三个 run（`pre` / `r13` / `fsel2`）的实测到达步数分布：

| 分位 | 到达成功（n=82，首个候选） |
| :--- | ---: |
| p50 / p75 / p90 / p95 / max | 16–20 / 23–31 / 29–33 / 33–38 / 39 |

即 40 步恰好压在 p95 上：尾部的 `arrive_timeout` **全部在 41 步触发且剩余距离 p50 ≈ 1.9 m**。由数据拟合的步速约 **7.9 步/米**（含转向开销），对应 2 m→16 步、4 m→31 步、6 m→47 步——**预算 40 等于把“距离 ≳ 5 m”的候选直接判死**。更好的选择（未实现，待定）：

1. 距离自适应：`budget = ceil(d0 · 7.9 · 1.3) + 5`，既不放走近候选也不白等远候选。
2. 或设候选距离上限 `recheck_max_dist_m`（如 5 m）：远候选**直接不入队**，比给一个必然超时的预算更省步数。
3. 固定抬到 60 步是下策：会让近候选在多候选队列里平均多耗预算。

### 2.8 日志（写入 `detections/memory_events.jsonl`）

| 事件 / 字段 | 含义 |
| :--- | :--- |
| `recheck_start` | 进入复核：`n_candidates` + 每个候选的 `memory_id/tier/distance/blacklisted` |
| `recheck_result` | 航点结果：`outcome ∈ {arrived, arrive_timeout}`、`steps_used`、`distance`、`blacklisted` |
| `recheck_exhausted` | 机制结束：`total_steps`、`outcome="queue_exhausted"`、`best_memory_id`、`best_distance`（STOP 位置到 comparator 首名的距离，**仅日志**） |
| `stop_blacklist`（优化方向日志） | 新增 `sigma_c` / `sigma_tar` / `sigma_gap` / `lookaround_sigma_max|min|delta` / `lookaround_trace`（每帧 `turn/dist/sigma_c/gd_hits/state/origin`，保留最后 8 帧） |

`stop_blacklist` 的新增字段就是“拉黑那一刻就地复核（look-around+平移）”的判定依据：若 `lookaround_sigma_delta ≈ 0`，说明多转视角根本没改变检测分数，再做平移也救不回来；若 `sigma_gap` 很小而 `delta` 明显，则值得加平移。`best_distance` 是“STOP 锚定”方向的判定依据。

### 2.9 实测：机制的真实作用点

主结果（`s7_recheck_2scene_2026-11-03_201331`，198 集）：

| run | 场景 | 集数 | SR | 说明 |
| :--- | :--- | ---: | ---: | :--- |
| `s7_recheck_2scene_2026-11-03_201331` | mL8 + 5cd | 198 | **68 / 198 = 34.34%** | mL8 16/99（基线 9/99）、5cd 52/99（= 基线 52/99）；步数 103.4 / 223.9 |
| 基线子集（`s7_baseline_3scene_2026-10-30_004608` 的同两场景） | mL8 + 5cd | 198 | 61 / 198 = 30.81% | mL8 9/99、5cd 52/99 |

事件计数：`recheck_start` 33、`recheck_result` 43、`recheck_exhausted` 29。

**逐 episode 对齐 30 个进过复核的 episode 后，得到机制真实的因果键：**

| 观察项 | 数值 | 含义 |
| :--- | ---: | :--- |
| STOP 步 == `recheck_exhausted` 步 | **30 / 30** | 机制结束的那一步就发了 STOP |
| 复核期间产生新 memory | 0 | 不是“多看到新目标” |
| 复核期间恢复 navigate | 1 / 30 | 不是“升为 gd-tsp 后转 navigate” |
| 队列里至少有一个 TP 条目 | **3 / 30** | 27 个队列全是 FP |
| 全 FP 队列但仍成功 | **6 / 27** | 成功率与 TP 率同量级：碰运气 |
| 失败类型 | `false_positive` 23 + `timeout_tp` 1 | STOP 落在了 FP 候选旁 |
| 黑名单条目 TP 率 | 56 / 267 = 21.0% | 队列精度天花板 |

结论修正（2026-11-04）：**SR 增益不是来自“复核后确认”，而是来自“多走 ~27 步、接着在巡回末位候选旁 STOP”**。ObjectNav 的成功判据是「STOP 时 1 m 内有目标实例」，所以只要 STOP 落点靠近 GT 就算成功；而巡回末位恰好是 comparator 里最弱的候选（provisional/其它），于是 23 个 episode 变成 `false_positive`。

两条由此得出的结论：

1. **机制本体 = 延迟 STOP + 航点位移**，这部分有实测收益，保留。
2. **STOP 落点可优化**（把 STOP 挪到 comparator 首名旁），但这会改变 STOP 时机与位置、破坏与旧版的 SR 可比性，所以本版**只记录不实施**：`recheck_exhausted.best_distance` 给出了“STOP 位置距首名的距离”。先用这批字段回答“现有 STOP 离首名到底有多远，回走代价多大”，再决定是否开新批次。

### 2.10 方向 1/3 的对照实验（结论：中性偏负）

`pre` 版（comparator 排序 + 提升后解除黑名单）= `s7_recheck_2scene_2026-11-03_201331`；`r13` 版（最近优先 + 只读）= `s7_r13_field_2026-11-04_004818`（磁盘满中断，114 集：mL8 99 + 5cd 15）。配对比较：

| 指标（114 集交集） | r13（方向 1/3） | pre（comparator + release） |
| :--- | ---: | ---: |
| SR | 22/114 = 19.3% | 23/114 = 20.2% |
| 500 步超时 / 平均步数 | 5 / 123.5 | 4 / 122.5 |

- **方向 3（只读）的后果**：旧版靠 `promoted_observed` 记录提升、并（在更早版本里）解黑名单；但 `_stop_blacklist` 不释放时，提升结果永远不会被 `_get_target_object_location()` 采用，所以“双查询判定”这一行为对动作零作用（已在重构中删除）。
- **方向 1（最近优先）的预期收益没有出现**：队列只有 1.3–1.4 个候选/次、单候选平均 ≈21 步、`arrive_timeout` 占比 8–9%（与 pre 版相同），预算几乎不绑定；最近优先只改变“先看谁”，在一个 episode 的量级上就出现翻转（5cd ep0：426 步成功 → 500 步超时）。
- 因此**排序已回退为目标选择 comparator**（§2.3），只读（不解除黑名单）作为当前主线保留。

### 2.11 风险与已知边界

- **机制上限 = 黑名单条目 TP 率（20–21%）**。队列全是已被拒绝的候选，所以“就地 STOP”的期望成功率就是这个数。
- **无可用排序信号**：黑名单条目的 TP 率按 `lifecycle_state` / `origin` / `gd_hits` 分组后差异不稳定（`confirmed+tsp-confirmed` 在 `pre` 是 36%、在 `fsel2` 是 12%，n 只有 11/17），不能拿来做可信度门槛。
- **触发时机漏掉大多数可回收样本**：37 集有 TP 黑名单条目，但只有 10 集进入复核；27 集因 frontier 未耗尽而永远不触发（其中 16 集最终失败且几乎都烧满 500 步）。
- 远候选很贵：单候选平均 ≈21 步，40 步预算下距离 ≳5 m 的候选基本无法到达。
- 不设队列总预算，意味着长队列可能把接近 500 步的 episode 拖成 timeout；当前只靠 `recheck_arrive_steps` 限单个航点。

## 3. 机制 B：field（已弃用，仅保留设计思路）

> 实现已全部删除，不再有代码、配置或日志通道。以下只保留当初的设计意图与弃用依据，供后续判断是否值得以别的形式重启。

### 3.1 设计思路（历史记录）

- 来源：Multi-Scale Gaussian-Language Map（GLMap）的实例/区域语义关联、TrajRAG 的几何－语义检索与收益－成本权衡。
- 目标：在 explore 阶段为 frontier 提供一个**与语义置信无关**的“目标覆盖”偏好——优先去「目标先验高且尚未被观察」的区域。
- 形式：把 frontier 上“目标存在”的后验熵的期望下降按路径代价归一

$$
\Delta H(a)=\max\left(0,\ H_t-\mathbb{E}[H_{t+1}\mid a]\right),
\qquad
F_t(a)=\frac{\Delta H(a)}{C(a)+\epsilon}
$$

并以 `S`（ValueMap 半径查询）作相对分档、`F` 作档内排序，选择规则与 `gd-tsp > gd-only > confirmed > provisional` 同构（档位决定优先权，档内才比序键）。
- 只在 explore 分支生效；navigate 阶段完全不读取 field；无模型/特征时直接回退原 frontier 行为。

### 3.2 弃用依据（在线实测，198 集）

| 指标 | field 选点（`s7_fsel2_2scene_2026-11-04_101942`） | recheck（对照，同 198 集） |
| :--- | ---: | ---: |
| SR | **67 / 198 = 33.8%** | 68 / 198 = 34.3% |
| 500 步超时 | 13 | 23 |
| 平均步数 | 150.2 | 163.7 |
| mL8 | 16/99，步数 116.0 | 16/99，步数 103.4 |
| 5cd | 51/99，步数 184.4 | 52/99，步数 223.9 |
| 逐集翻转 | 31 集（15 得 / 16 失） | — |

机制层面的退化证据（来自落盘的 `coverage_field.jsonl`，15,121 个多 frontier 步）：

| 检验 | 数值 | 含义 |
| :--- | ---: | :--- |
| `argmax F == argmin route` | **87.2%** | F 排序等价于“选最近的 frontier” |
| Spearman(F, −route) 中位 | 0.900 | 同上 |
| CV(ΔH) 中位 / CV(F) 中位 | 0.241 / 0.591 | ΔH 几乎不区分，F 的方差主要来自 `1/route` |
| S 量程 / 分档 | S ∈ [0.000, 0.284]，相对分位分档；多 frontier 步中 100% 存在 ≥2 个档 | 相对分档保证“永远有最高档”，弱 S 的步也会被强行降序 |
| 覆盖集定义 | 1 m 内的**其它 frontier 点** | 目标覆盖量从未在线计算，离线 AUC 与在线目标函数无对应 |

- 离线判别力（region AUC .8259 / .8201）是**静态判别力**，其验证方式是“给定区域是否含目标”；它不蕴含“按 F 选 frontier 能提升 SR”，实测也确认了这一点。
- 唯一实打实的改善是 500 步超时 23→13、步数 163.7→150.2，来源是选点自带的 stall 黑名单守卫，与 field 打分无关。
- S 复用 ValueMap 的同一个语义值，没有引入新的目标证据；因此机制上界只是“把语义最大化换成了最近邻”，与基线 `_explore`（`S + λ·H1` + 粘滞 + acyclic 约束）相比不占优。

### 3.3 删除清单（2026-11-04）

| 位置 | 删除内容 |
| :--- | :--- |
| `vlfm/vlvm/semantic_spatial_field/` | 整目录（`coverage.py` / `coverage_log.py` / `core.py` / `inference.py` / `__init__.py`） |
| `vlfm/policy/tsp3d_objectnav_policy.py` | `_field_eval` / `_field_select_index` / `_log_field_eval`、`_field_logs`、`_field_sel_*` 状态、`_explore_safe` 中的 field 选点分支、`field_log_enable` / `field_select_enable` / `field_sel_stale_steps` / `field_sel_min_progress_m` 超参数 |
| `vlfm/vlvm/recheck/expl_recheck.py` | `_init_recheck` 中的 4 个 field 参数与对应属性 |
| `config/experiments/vlvm_objectnav_hm3d.yaml` | 4 个 field 字段 |
| `vlfm/utils/vlvm_trainer.py` / `episode_stats_logger.py` | `field_logs` 参数与 `coverage_field.jsonl` 输出通道 |
| `scripts/_nav_s7.sh` | `FIELD_SELECT_ENABLE` 开关与 banner 行 |

无备份保留。历史 run 的 `coverage_field.jsonl` 仍在 `outputs/` 下，仅供离线复查，不再有代码读取它。

## 4. 实验批次

| 批次 | 改动 | 状态 | 结论 |
| :--- | :--- | :--- | :--- |
| F-1 | S 分档 + F 排序（log-only 落盘） | 已完成（`s7_r13_field_2026-11-04_004818`） | 字段完整，可用于退化分析 |
| F-2 | F 分档 + S 排序 / field 选点接入 explore | 已完成（`s7_fsel2_2scene_2026-11-04_101942`） | **弃用**：SR 持平、机制退化为最近邻 |
| F-A1 | explore STOP → 候选巡回 + 就地 STOP | 已完成（`s7_recheck_2scene_2026-11-03_201331`） | SR 68/198；增益来自末位 STOP 的几何接近性 |
| F-A2 | 纯重构：删除双查询判定（行为不变） | 已完成 | 已包含在 F-A7 代码中 |
| F-A7 | 自适应到达预算 | **已中止并弃用** `s7_radapt_2scene_2026-11-04_162214`（110 集） | 配对 21/110 vs 22/110（只差 mL8 ep55）；步数完全一致 117.9；机制上自适应亲手制造了 4–5 个 40 步本可到达的超时 → 回退固定值 45 |
| F-A3 | 拉黑即复核（look-around + 平移） | **放弃** | 35 条拉黑中 8 条（23%）分数已 ≥0.60 但被状态卡住（加视角无用）；`delta>0.02` 仅 7 条（TP 2 条）→ 上界≈+1% 且会新增 FP |
| F-A4 | 候选距离上限 | **不做** | 110 集内 ≥4 m 候选只有 1 条，无作用面 |
| F-A5 | 锚点 STOP | **放弃** | `best_distance` p50 0.84 m、max 1.18 m，0 例 >1.5 m → 无优化空间 |
| F-A6 | 队列总预算闸 | **不做** | 会截断队列/改变 STOP 时机，需另开批次；S7 已结束 |
| ~~F-B1/F-B2~~ | field 覆盖偏好 | **作废** | 机制已删除 |

顺序约束（若重启 field）：必须先把覆盖集改成地图 cell、把分档改成绝对标定、并在目标函数里引入目标证据，否则会重复 §3.2 的退化。

## 5. 与既有文档的关系

| 文档 | 关系 |
| :--- | :--- |
| [VLVM-S7.md](VLVM-S7.md) §1.2 | 1A 的 STOP 抑制；本文机制 A 在其之后补“去哪里” |
| [VLVM-S7.md](VLVM-S7.md) §1.3 | floor 机制；本文冻结，不再推进 |
| [VLVM-S7.md](VLVM-S7.md) §3 | field 原始设计；本文 §3 记录弃用 |
| [VLVM-S7M1-1exp.md](VLVM-S7M1-1exp.md) | 1A 实测数据来源 |
| [VLVM-S7baseline-exp.md](VLVM-S7baseline-exp.md) | 三场景失败结构来源 |
| [VLVM-S7-floor.md](VLVM-S7-floor.md) | floor 设计与失败记录，仅作归档 |

## 6. 收尾状态（S7 结束）

机制 A 的最终形态（`2026-11-04` 定稿）：

- 唯一参数：`recheck_enable: True`、`recheck_arrive_steps: 45`；无自适应、无队列总预算、无候选上限。
- 行为：explore frontier 耗尽 → 按目标选择 comparator 巡回已有记忆 → 队列耗尽就地 STOP。
- 已删除：双查询提升判定 / `promoted_observed` / 黑名单释放通路 / 队列总预算 / 锚点回走 / 距离自适应预算。
- 保留的诊断日志（不改动作）：`recheck_start/result/exhausted`（含 `steps_used`、`best_distance`）、`stop_blacklist` 的 `sigma_c/sigma_gap/lookaround_*`。

已证伪或放弃的方向（详见 §4）：F-A3（拉黑即复核）、F-A5（锚点 STOP）、F-A7（自适应预算）、F-A4/F-A6、以及 field 的 B 线。

遗留但不再推进的观察：

- 23% 的被拉黑记忆其 `sigma_c` 已 ≥0.60，堵点在 `confirmed & gd-tsp` 的状态/来源判定，而不是分数或视角。若将来要提 SR，应从这里入手（对全部 episode 生效），而不是继续调整 recheck。
- 无队列总预算意味着长队列仍可能把接近 500 步的 episode 拖成 timeout；已知、接受。
