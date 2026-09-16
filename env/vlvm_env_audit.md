# VLVM 环境文件差异审计（2026-09-16）

> **背景**：`env/` 下原有三份文件（`vlvm环境配置指南.md` / `vlvm_env_export.yml` / `vlvm_requirements.txt`）源自
> 早期（约 2026-08 中）状态。本次以实机（conda env `vlfm`，**2026-09-16 用户定名统一为 `vlvm`**）逐项核对，记录**过期项与修正依据**，
> 并说明新增文件的作用。旧版原件保留于 `env/.bak_20260916/`。

---

## 一、审计环境（实机核对基线）

| 项 | 实测值 |
|---|---|
| 环境 | conda `vlfm`（待迁移为 `vlvm`），`/root/miniconda3/envs/vlfm`，Python 3.9.25 |
| 硬件 | RTX 3090 24 GB，驱动 570.124.04（CUDA 12.8），工具链 nvcc 11.8.89 / gcc 11.3.0 |
| 核心 | torch 1.12.1+cu113、torchvision 0.13.1+cu113、numpy 1.26.4、transformers 4.26.0、timm 0.4.12 |
| 三维算子 | MinkowskiEngine 0.5.4、mmcv 2.1.0、mmdet 3.2.0、mmdet3d 1.4.0、mmengine 0.10.4 |
| 仿真 | habitat-sim 0.2.4、habitat-lab 0.2.4、habitat-baselines 0.2.4（均 editable 源码） |
| MinkowskiEngine 基准 | `ME.MinkowskiConvolution(3,16,3,dimension=3)` 输出 `torch.Size([95, 16])` |

---

## 二、原文件的主要问题与修正

### 1. conda 环境名不一致（高危）

| 文件/位置 | 旧内容 | 实际 | 修正 |
|---|---|---|---|
| 指南 §1.1 | `conda create -n vlvm` | 运行时环境名 **`vlfm`** | **2026-09-16 用户定：环境名统一为 `vlvm`**；脚本改为优先 `vlvm`、自动回退 `vlfm`；迁移步骤见指南附录 C |
| `vlvm_env_export.yml` | `name: vlvm` | 同上 | `name:` 与 `-n vlvm` 一致（定名后无需再注） |
| 依据 | — | `conda env list` 只有 `base` / `vlfm`（实机环境尚未改名） | — |

> 影响：旧指南建 `vlvm`、实机跑 `vlfm` ⇒ 两套命名并存。现脚本两种名字都能识别（`CONDA_ENV_NAME` 可覆盖）。

### 2. 子模块路径过期（高危）

- **旧**：子模块（habitat-lab / frontier_exploration / depth_camera_filtering）视为 `$VLVM/` 内的目录，`pip install -e .` 直接就地安装。
- **实测**：`vlvm` 仓库**不含**这些目录，运行时全部来自外部目录 `/root/autodl-tmp/vlfm`：

| 包 | 实测源码路径 |
|---|---|
| `vlfm` | `/root/autodl-tmp/vlvm/vlfm` |
| `groundingdino` | `/root/autodl-tmp/vlvm/GroundingDINO/groundingdino` |
| `habitat` | `/root/autodl-tmp/vlfm/habitat-lab/habitat-lab/habitat` |
| `habitat_baselines` | `/root/autodl-tmp/vlfm/habitat-lab/habitat-baselines/habitat_baselines` |
| `frontier_exploration` | `/root/autodl-tmp/vlfm/frontier_exploration/frontier_exploration` |
| `depth_camera_filtering` | `/root/autodl-tmp/vlfm/depth_camera_filtering/depth_camera_filtering` |

- **后果**：新机若只拷 `vlvm` 仓库，`import habitat` / `import frontier_exploration` 全部失败；`frontier_exploration` 缺失还会导致 hydra 报
  `Could not load 'habitat/task/lab_sensors/frontier_sensor'`（`run.py` 依赖其注册副作用）。
- **修正**：指南新增 §3.1「外部依赖」章节 + `vlvm_setup_env.sh` 支持 `DEPS_ROOT` 重定向。

### 3. `vlvm_requirements.txt` 的大量路径条目过期

旧文件（`pip freeze` 产物）含 7 处 `/root/autodl-tmp/vlfm/...` 路径与 2 处 GitHub URL：

| 旧条目 | 状态 |
|---|---|
| `depth_camera_filtering @ file:///root/autodl-tmp/vlfm/...` | 路径需改为外部依赖（见上） |
| `-e /root/autodl-tmp/vlfm/frontier_exploration` | 同上 |
| `-e /root/autodl-tmp/vlfm/GroundingDINO` | 应为 `/root/autodl-tmp/vlvm/GroundingDINO` |
| `-e /root/autodl-tmp/vlfm/MobileSAM` | **已零引用**（2D 时代遗留），保留安装但降级为可选 |
| `-e /root/autodl-tmp/vlvm`（vlfm 本体） | 保留（新增 `pyproject.toml` 后可正常 `-e` 安装） |
| `habitat-lab / habitat-baselines @ git+...1639e1a` | 与实机 editable 源码同源，作为无 `$DEPS` 时的备选保留 |
| `magnum @ file:///opt/conda/envs/py39/conda-bld/...` | build 机私有路径，迁移必挂；实机由 `habitat_sim-0.2.4.egg` + `.pth` 提供 |
| `en-core-web-sm @ https://...spacy-models/...`（无安装） | 实机未装；代码零引用 spacy ⇒ 删除 |
| `detectron2 @ git+...` | 实机为 0.6（GD 链路依赖），保留 |

**新文件**改为**手工整理的直接依赖清单**（分区 + 版本 + 注意事项），不再是从一个装歪的环境导出；全量复刻交给 `vlvm_env_export.yml`。

### 4. `vlvm_env_export.yml` 是异构环境的快照

旧快照（297 行）与实机差异（`--no-builds` 比对）：

| 包 | 旧快照 | 实机 | 判定 |
|---|---|---|---|
| `transformers` | **4.57.6** | 4.26.0 | 旧快照严重过期——4.57.6 会破坏 BLIP-2/LAVIS，属**错误基线** |
| `setuptools` | 67.6.1 | 82.0.1 | 实机更新 |
| 条目路径 | 含 `/root/autodl-tmp/vlfm` 的 pip 行 | — | 重新导出后无路径条目（conda 只记包名版本） |
| `magnum` | 无 | `magnum==0.0.0`（非 pip 管理） | 由 habitat-sim 提供 |
| `pydantic` 等 | 无版本差异 | 一致 | — |

**新快照**：`conda env export -n vlfm --no-builds` 现采（298 行 = 40 conda 包 + 258 pip 包），name 归一为 `vlvm`，头部加用途/注意事项。

### 5. 代码侧缺打包配置（本次修复）

- **发现**：`vlvm` 仓库根**既无 `setup.py` 也无 `pyproject.toml`**，git 历史亦无（从未提交）。
  实机 `import vlfm` 之所以可用，靠 site-packages 中历史遗留的 easy-install 路径条目；
  一旦在**新机**执行 `pip install -e .` 会直接报 `does not appear to be a Python project`。
- **修复**：新增 `pyproject.toml`（`packages.find.where = ["vlfm"]`，与原 VLFM 布局一致；
  不声明 `dependencies`，避免 pip 解析时把锁定的 transformers 等升级）。
- **验证**：`pip install -e . --no-deps --dry-run` → `Would install vlfm-0.1`；正式安装后
  `import vlfm`/`vlfm.run` 正常，来源指向 `/root/autodl-tmp/vlvm/vlfm/__init__.py`。

### 6. 其他修正

| 项 | 旧 | 新 |
|---|---|---|
| 快照方案 | 指向 `/root/autodl-fs/vlvm_env_2026-08-16.tar.gz` | 该文件本机已不存在；改为"通用 tar.gz + 必跑 `vlvm_setup_env.sh`" |
| `python -m vlfm.utils.generate_dummy_policy` | 指南沿用 | 该脚本已于 2026-09-15 删除 → 改为"从备份拷贝 `data/dummy_policy.pth`" |
| `psutil` | 清单内 | **实机未安装**；代码 try/except 导入，标注为可选 |
| `seaborn` 依赖 | 旧 pyproject 声明 | 代码零引用（属旧 yolov7 链路），不列入 |
| 环境名口径（09-16 补） | 指南写 `vlvm`、实机 `vlfm` | **统一为 `vlvm`**；脚本优先 `vlvm` + 回退 `vlfm`；迁移见指南附录 C |
| 微调链路 | 未提 | 2026-09-15 已从仓库删除（`ft_pipeline` / `ft_tsp3d.sh`），环境侧无额外依赖 |
| 模型配置口径 | 未提 | 新增说明：TSP3D 主权重由 `TSP3D_CHECKPOINT` 指定（现档 `tsp3d_scanrefer.pth`）；GD 为辅助提议源（`gdp_enable` / `every_n=5` / `max_boxes=1`），不改主模型地位 |

---

## 三、本次产出文件

| 文件 | 状态 | 说明 |
|---|---|---|
| `vlvm环境配置指南.md` | **重写** | 新增实机速览、外部依赖章、路径重定向、故障速查表、差异附录 |
| `vlvm_requirements.txt` | **重写** | 直接依赖清单（9 区 + 版本 + 注意），不再含失效路径 |
| `vlvm_env_export.yml` | **重新导出** | 实机全量快照，258 pip 包，无私有路径；头部加使用说明 |
| `vlvm_setup_env.sh` | **新增** | 一键重定向 6 个 editable 包 + 14 项导入校验 + 权重/数据集/端口检查 |
| `vlvm_env_audit.md` | **新增** | 本文件 |
| `pyproject.toml`（仓库根） | **新增** | 补回丢失的打包配置，使 `pip install -e .` 可用 |
| `.bak_20260916/` | 新增 | 三份旧文件原件备份 |

---

## 四、校验记录（2026-09-16）

```
bash env/vlvm_setup_env.sh
  ① 6 个 editable 包路径全部正确
  ② 导入校验 13 项 OK（psutil 1 项 WARN=可选）
  ③ 权重 4 项 OK（tsp3d_scanrefer 669M / GD 662M / pointnav 33M / dummy 52K）
     数据集 3 项 OK（objectnav / hm3d / mp3d，软链均有效）
  ④ 环境变量与端口：12186 / 12182 / 12181 三服务 LISTEN（当时评测在跑）
  结果：全部通过
```

> 注：审计期间 `scripts/my_eval.sh`（档位 `d2_scan`）与三个 VLM 服务正在运行，
> 本次所有操作（含 `pip install -e .`）均为**磁盘侧**改动，不影响运行中进程。
