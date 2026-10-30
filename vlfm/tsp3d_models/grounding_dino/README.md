# `grounding_dino`：GD 3D 提议与在线机制

当前目录是 S6 GD 机制的**运行代码**，不是旧版“帧级合票”实验。

## 当前调用链

```text
TSP3DObjectNavPolicy
  -> GdCalibrationLogger           calibration.py
  -> GdProposer                    proposer.py
  -> GdMechanism                   mechanism.py
```

- `calibration.py`：每帧调用 GD 服务，把 2D 框转成 3D proposal，并记录
  `match_candidate` / `match_track`、GD 置信度、深度/FOV 和 new-info。
- `mechanism.py`：消费 calibration 的 JSON-safe record，执行三条路径：
  - `track=stable` -> `blocked`；
  - `track=confirmed/provisional` -> `gd-tsp`，按 track centroid 写入；
  - 其余无 track 关联 -> `gd-only`（class/conf/box-occ 过滤）。
  candidate-only 只保留 `association=candidate_only` 诊断，不再直接确认。
- `gd_offline_cal.py`：跑完后离线分析 GD 覆盖率、关联精度和 proposal 分布。

## 活跃文件

| 文件 | 作用 |
| :--- | :--- |
| `config.py` | `GdProposalConfig`，proposer 旋钮 |
| `types.py` | `CameraView` / `GdProposal` / `ProposalStats` |
| `vocab.py` | 目标类别文本对齐、caption 构造 |
| `client.py` | GD 服务访问；失败返回“未检查”，不产生负证据 |
| `region.py` | 2D 框 -> 深度反投 -> 3D 点簇 |
| `boxes.py` | 3D AABB / IaU / containment 几何 |
| `proposer.py` | `GdProposer` / `should_run` |
| `calibration.py` | 在线 GD 观测、关联与日志 |
| `mechanism.py` | gd-only / gd-tsp 路由、写入和事件 |
| `gd_offline_cal.py` | 离线标定/回放 CLI |

已删除：

- `provenance.py`：旧的 candidate-level `BOTH/GD/TSP3D` 合票与
  `attach_votes` / `admit_as_suspicious` / `is_lockable` 规则；当前机制已改为
  track-only gd-tsp + gd-only pool，不再调用该文件。
- `types.Tsp3dCandidate`：仅旧 provenance 使用的数据类型。
- `__pycache__/`：本地缓存。

## 运行入口

在线导航由 `scripts/_nav_s7.sh` 调用；离线汇总：

```bash
python -m vlfm.tsp3d_models.grounding_dino.gd_offline_cal --run-dir outputs/<run>
```

## 关键约束

- GD 失败或未运行时返回“未检查”，不写负证据。
- `gd-only` 不创建 STOP 资格；只有后续被 admitted TSP3D candidate 合并时才
  提升为 `confirmed`（`reason=gd_only_tsp_merge`）。
- `gd-tsp` 只接受已进入 memory 的 `confirmed/provisional` track。
- 普通升级只使用 `num_obs>=4`，不使用距离门。
