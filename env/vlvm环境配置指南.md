# VLVM 项目环境配置指南

> **适用对象**：在全新服务器 / 实例上为 **VLVM 项目**（`/root/autodl-tmp/vlvm`）搭建可运行环境。
>
> **项目定位**：3D Native 主动视觉语言导航 —— VLFM 导航框架重构 + **TSP3D** 三维目标定位（主模型）。
>
> **配置顺序**：先装 **vlfm 导航框架环境**（第一部分）→ 再装 **TSP3D 模型依赖**（第二部分）→ 最后做**外部依赖与路径重定向**（第三部分）。
>
> **目录约定**：本文用 `$VLVM` 代表项目根（`/root/autodl-tmp/vlvm`），`$DEPS` 代表外部依赖根（`/root/autodl-tmp/vlfm`，见 §3.1）。

---

## 0. 环境速览（实机核对信息，2026-09-16）

| 项 | 值 | 备注 |
|---|---|---|
| 机器 | AutoDL 实例，**RTX 3090 (24 GB)** ×1 | 驱动 570.124.04 / CUDA 12.8 |
| 系统工具链 | gcc 11.3.0、Ubuntu 22.04 | MinkowskiEngine 编译用 |
| 工具链 CUDA | `/usr/local/cuda-11.8`（nvcc 11.8.89） | 编译用 `TORCH_CUDA_ARCH_LIST=8.6` |
| conda 环境 | **`vlfm`**（`/root/miniconda3/envs/vlfm`） | ⚠️ 环境名是 `vlfm` 不是 `vlvm`，脚本与文档均按 `vlfm` 调用 |
| Python | 3.9.25 | |
| torch / torchvision | 1.12.1+cu113 / 0.13.1+cu113 | 锁死 |
| transformers | **4.26.0** | 锁死，>4.26.0 破坏 BLIP-2 |
| 三维算子 | MinkowskiEngine 0.5.4 / mmcv 2.1.0 / mmdet 3.2.0 / mmdet3d 1.4.0 | 见 §2 |
| 仿真 | habitat-sim / habitat-lab / habitat-baselines = 0.2.4 | lab / baselines 来自 editable 源码 |
| 磁盘 | 项目本体约 6.5 GB（其中 data 5.3 GB）+ 数据集约 43 GB | 数据集为软链，见 §3.4 |

**环境文件（本目录）**

| 文件 | 用途 |
|---|---|
| `vlvm_requirements.txt` | **直接依赖清单**（显式安装项 + 锁定版本 + 注意事项） |
| `vlvm_env_export.yml` | **全量快照**（258 个 pip 包 + conda 层），整环境复刻用 |
| `vlvm_setup_env.sh` | **一键环境重定向 / 校验脚本**（新机解压快照或 pip 装完后运行） |
| `vlvm_env_audit.md` | 旧版环境文件的**差异审计**（过期路径 / 版本、修正方式、最新实测） |

---

# 第一部分：vlfm 导航框架环境

## 1.1 创建 conda 环境

```bash
conda create -n vlfm python=3.9 -y
conda activate vlfm
```

> 环境名必须是 `vlfm`：`scripts/my_eval.sh` 中 `CONDA_ENV_NAME="vlfm"` 写死。

## 1.2 安装 PyTorch（CUDA 11.3 构建）

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 1.12.1+cu113 True
```

## 1.3 安装框架核心依赖

```bash
pip install numpy==1.26.4 transformers==4.26.0 tokenizers==0.13.3 timm==0.4.12 \
    salesforce-lavis==1.0.2 open3d==0.18.0 opencv-python==4.5.5.64 \
    Flask==3.1.3 openmim==0.3.9
```

> - `transformers==4.26.0` **务必锁定**（更高版本破坏 BLIP-2 的 LAVIS 调用）；
> - `timm==0.4.12` 锁定（BLIP-2 视觉骨干与 GroundingDINO 共用）；
> - `numpy==1.26.4` 锁 1.x（habitat-sim 0.2.4 / mmdet3d 1.4.0 不适配 numpy 2.x）。

## 1.4 安装 Habitat 仿真依赖

```bash
pip install habitat-sim==0.2.4
cd $DEPS/habitat-lab/habitat-lab       && pip install -e .
cd $DEPS/habitat-lab/habitat-baselines && pip install -e .
```

> habitat-lab 源码**不在 vlvm 仓库内**（见 §3.1）。若新机没有 `$DEPS`，可改用：
> ```bash
> pip install -e "git+https://github.com/facebookresearch/habitat-lab.git@1639e1ae732ba1e84199a1a04b79c7243c3f8586#egg=habitat_lab&subdirectory=habitat-lab"
> pip install -e "git+https://github.com/facebookresearch/habitat-lab.git@1639e1ae732ba1e84199a1a04b79c7243c3f8586#egg=habitat_baselines&subdirectory=habitat-baselines"
> ```

## 1.5 安装项目本体与子模块（editable）

```bash
cd $VLVM                        && pip install -e .      # vlfm 包（包名 vlfm，路径在 vlvm）
cd $DEPS/depth_camera_filtering && pip install -e .
cd $DEPS/frontier_exploration   && pip install -e .
cd $VLVM/GroundingDINO          && pip install -e .      # GD 辅助机制（端口 12181）依赖
```

> ⚠️ **`frontier_exploration` 不是可选包**：`vlfm/run.py` 顶部 `import frontier_exploration`
> 用于注册 habitat 的 `frontier_sensor` / `frontier_exploration_map` 配置组；缺失时 hydra 合成直接报
> `Could not load 'habitat/task/lab_sensors/frontier_sensor'`。
>
> ⚠️ `depth_camera_filtering` / `frontier_exploration` 源码在 `$DEPS` 下（**不在 vlvm 仓库内**），
> 无公开 pip 包，新机必须拷源码后再 editable 安装。

---

# 第二部分：TSP3D 模型环境

TSP3D 需要额外的**编译型依赖**：MinkowskiEngine（稀疏卷积）与 mmcv / mmdet3d（3D 检测算子）。
（注：模型微调链路 `ft_pipeline` 已于 2026-09-15 从仓库删除，环境侧无额外新增依赖。）

## 2.1 安装系统级 BLAS

```bash
apt-get install -y libopenblas-dev
```

> 不装会在编译时报 `BLAS not found from numpy.distutils.system_info.get_info`。

## 2.2 安装 MinkowskiEngine 0.5.4（源码编译）

PyPI **只有源码包（无预编译 wheel）**，必须现场编译：

**① 解决 setuptools/distutils**（新版 setuptools 缺 `msvccompiler`，二选一）：

```bash
pip install "setuptools<68"                 # 方案 1：降级 setuptools
export SETUPTOOLS_USE_DISTUTILS=stdlib      # 方案 2：强制用 Python 自带 distutils
```

**② 下载源码并写 BLAS 配置**：

```bash
cd /tmp && pip download MinkowskiEngine==0.5.4 --no-deps --no-binary :all: -d me_src
cd me_src && tar xzf MinkowskiEngine-0.5.4.tar.gz && cd MinkowskiEngine-0.5.4
cat > site.cfg <<'EOF'
[openblas]
libraries = openblas
library_dirs = /usr/lib/x86_64-linux-gnu
include_dirs = /usr/include/x86_64-linux-gnu/openblas-pthread
EOF
```

**③ 打 torch 1.12 兼容补丁（ATen.h 头顺序 + `py::self`）**：

- 在以下 8 个源文件中，于 `#include <ATen/cuda/CUDAUtils.h>` / `#include <ATen/cuda/CUDAContext.h>` 之前插入 `#include <ATen/ATen.h>`：
  `src/spmm.cu`、`src/broadcast_gpu.cu`、`src/convolution_gpu.cu`、`src/convolution_kernel.cu`、`src/convolution_transpose_gpu.cu`、`src/allocators.cuh`、`src/common.hpp`、`src/direct_max_pool.cpp`
- 将 `pybind/extern.hpp` 中 `.def(py::self == py::self)` 改为 `.def("__eq__", &minkowski::CoordinateMapKey::operator==, py::is_operator())`

**④ 编译并安装**：

```bash
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.6"
python setup.py bdist_wheel --blas=openblas
pip install dist/MinkowskiEngine-0.5.4-*.whl
```

> `TORCH_CUDA_ARCH_LIST` 按目标 GPU：RTX 3090 / 4090 用 `8.6`，A100 用 `8.0`。
> 单次编译约 15–30 分钟；产出的 wheel 建议留存复用（同 GPU 架构可跨机安装）。

### 验证 MinkowskiEngine

```bash
python -c "
import torch, MinkowskiEngine as ME
coords = torch.cat([torch.randint(0,10,(100,3)).int(), torch.zeros(100,1).int()],1).cuda()
sp = ME.SparseTensor(torch.randn(100,3).cuda(), coords)
out = ME.MinkowskiConvolution(3,16,3,dimension=3).cuda()(sp)
print('ME OK', out.shape)"      # 期望 ME OK torch.Size([95, 16])
```

## 2.3 安装 mmcv（预编译 wheel，免编译）

```bash
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```

> 该 wheel 对应 torch 1.12 / cu113 / py3.9，含 `nms3d` 等 CUDA 算子，**无需源码编译**。

## 2.4 安装 mmdet / mmdet3d

```bash
pip install mmdet==3.2.0 mmdet3d==1.4.0
```

> 自动带入 mmengine 0.10.4；`mmdet3d` 提供 TSP3D 的 3D 框结构（`DepthInstance3DBoxes` 等）。

## 2.5 验证 TSP3D 环境

```bash
cd $VLVM
python -c "from vlfm.tsp3d_models.bdetr import BeaUTyDETR; print('BeaUTyDETR OK')"
python -m vlfm.vlm.tsp3d --port 12186      # 需先备好权重，见 §3.2
# 期望: [TSP3D] Weights loaded from ...  / Running on http://localhost:12186
```

---

# 第三部分：外部依赖、权重与路径重定向

## 3.1 外部依赖（不在 vlvm 仓库内的 4 个源码目录）

`vlvm` 仓库**不自带**以下源码，本机全部取自 `$DEPS = /root/autodl-tmp/vlfm`（原 VLFM 参考仓库）：

| 目录 | 体积 | 被谁使用 | 可否省 |
|---|---|---|---|
| `$DEPS/habitat-lab/` | 41 MB | `habitat` / `habitat_baselines` 源码（editable 安装） | **必需** |
| `$DEPS/frontier_exploration/` | 428 KB | 注册 `frontier_sensor` 等配置组 + 备选探索器 | **必需** |
| `$DEPS/depth_camera_filtering/` | 68 KB | 深度相机滤波（habitat 观测处理） | **必需** |
| `$DEPS/MobileSAM/` | 86 MB | `mobile_sam` 包（2D 时代遗留，代码零引用） | 可省（亦可卸载） |

> 新机若没有 `$DEPS`：拷入上述 3 个必需目录到任意位置，改 `vlvm_setup_env.sh` 顶部 `DEPS_ROOT` 后运行。

## 3.2 模型权重与数据（`$VLVM/data/`）

| 路径 | 体积 | 用途 | 获取 |
|---|---|---|---|
| `data/tsp3d_models/tsp3d_scanrefer.pth` | 668 MB | **TSP3D 主权重**（当前定稿档） | 训练方提供 / 历史备份 |
| `data/tsp3d_models/`（BERT 文本编码器） | 约 3.6 GB | TSP3D 文本分支（`data_path` 指向此目录） | HuggingFace（`HF_ENDPOINT=https://hf-mirror.com`） |
| `data/groundingdino_swint_ogc.pth` | 662 MB | GD 辅助机制（端口 12181） | 公开权重 |
| `data/pointnav_weights.pth` | 33 MB | PointNav 底层策略 | VLFM 官方发布 |
| `data/dummy_policy.pth` | 52 KB | habitat-baselines eval 占位（缺失直接 exit） | 见下方说明 |
| `data/datasets/objectnav`、`data/scene_datasets/{hm3d,mp3d}` | — | 评测数据集（软链到条件数据目录） | 见 §3.4 |

> `data/dummy_policy.pth` 缺失时 `run.py` 会提示 `python -m vlfm.utils.generate_dummy_policy`，
> 但该脚本**已从仓库删除**（2026-09-15 清理）—— 直接从旧机备份拷贝该文件即可。
>
> **权重即模型配置**：TSP3D 主权重由 `TSP3D_CHECKPOINT` 指定（当前档 = `tsp3d_scanrefer.pth`，
> 即 scanrefer 预训练基线）；GD 辅助机制（V7.3 引入）以 `gdp_enable / gdp_every_n=5 / gdp_max_boxes=1`
> 控制，属**辅助提议源**，不改变 TSP3D 的主模型地位。换权重只改环境变量或 YAML，不动代码。

## 3.3 环境变量（运行前必须设置）

| 变量 | 建议值 | 说明 |
|---|---|---|
| `PYTHONPATH` | `$VLVM` | 保证 `import vlfm` 命中项目内源码 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像（模型下载） |
| `TSP3D_DATA_PATH` | `$VLVM/data/tsp3d_models/` | TSP3D 权重目录（服务端） |
| `TSP3D_CHECKPOINT` | `$VLVM/data/tsp3d_models/tsp3d_scanrefer.pth` | 切换权重用 |
| `TSP3D_PORT` / `BLIP2ITM_PORT` / `GROUNDING_DINO_PORT` | `12186` / `12182` / `12181` | 三个 VLM 服务端口（客户端代码默认值） |
| `EGL_PLATFORM` | `surfaceless` | habitat headless 渲染 |
| `MAGNUM_LOG` / `MAGNUM_GPU_VALIDATION` | `quiet` / `OFF` | 抑制 magnum 日志 |
| `TURN_LEFT_WIDE_TURNS` | 自动派生 | `run.py` 按 `panoramic_turn_steps` 计算写入，**不要手工设** |

`scripts/my_eval.sh` 已内置上述设置（含 `unset DISPLAY`、`CUDA_VISIBLE_DEVICES=0`）。

## 3.4 数据集

`data/datasets` 与 `data/scene_datasets` 在本机是**软链接**（指向 AutoDL 条件数据目录），
仓库中为空目录属正常。新机需重建软链或拷贝 HM3D / MP3D 数据（约 30 GB 量级），
否则评测启动即报 scene 缺失。

## 3.5 一键重定向 / 校验

```bash
bash $VLVM/env/vlvm_setup_env.sh            # 默认路径 = 本机布局
# 新机调整： DEPS_ROOT=/your/deps VLVM_ROOT=/your/vlvm bash vlvm_setup_env.sh
```

脚本做三件事：① 对 6 个 editable 包按当前 `$VLVM` / `$DEPS` 重新 `pip install -e`；
② 逐项校验关键包能否导入；③ 打印权重 / 环境变量 / 服务状态清单。

---

# 第四部分：运行与排障

## 4.1 启动流程（每次评测）

```bash
cd $VLVM
bash scripts/launch_vlm_servers.sh     # 起 3 个 VLM 服务（tmux 会话，加载约 60–90 s）
bash scripts/my_eval.sh                # BATCH=A / BATCH=B 分机组评测
```

> 改过服务端代码（`vlfm/vlm/*.py`、`vlfm/tsp3d_models/*`）后**必须重启服务**，
> 否则进程内仍是旧代码（历史上多次因此产生无效实验）。

## 4.2 常见故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `Could not load 'habitat/task/lab_sensors/frontier_sensor'` | `run.py` 缺 `import frontier_exploration` 副作用，或该包未安装 | §1.5 / §3.1 |
| `Dummy policy weights not found` | `data/dummy_policy.pth` 缺失 | 从备份拷贝（§3.2） |
| `BLAS not found ... get_info` | 未装 openblas | §2.1 |
| MinkowskiEngine 编译报 CUDA 头错误 | 未打 ATen 补丁 | §2.2 ③ |
| `Could not import lavis` / BLIP-2 服务起不来 | transformers 被升级 | 退回 4.26.0 |
| 客户端 `Connection refused` | VLM 服务未起或端口不符 | §3.3 |
| 渲染报 EGL / DISPLAY 错误 | 未设 headless 变量 | §3.3 |
| Pylance 报 `cv2` / `numpy` / `torch` 无法解析 | 编辑器解释器未选 `vlfm` env | 选 `/root/miniconda3/envs/vlfm/bin/python`，非代码问题 |
| 显存不足 / 抢卡 | 单卡需约 17 GB（3 服务 + 评测） | `nvidia-smi` 确认并停掉多余进程 |
| conda 环境名不一致报错 | 误用 `vlvm` | 统一用 `vlfm`（§1.1） |

---

# 附录 A：整环境快照方案（conda-pack）

若持有整环境快照包（tar.gz）：

```bash
mkdir -p /root/miniconda3/envs/vlfm
tar -xzf <snapshot>.tar.gz -C /root/miniconda3/envs/vlfm
/root/miniconda3/envs/vlfm/bin/conda-unpack        # 重写前缀
source /root/miniconda3/etc/profile.d/conda.sh && conda activate vlfm
bash $VLVM/env/vlvm_setup_env.sh                   # 重定向 6 个 editable 包 + 校验
```

> ⚠️ 旧版文档提到的 `/root/autodl-fs/vlvm_env_2026-08-16.tar.gz` 在本机已不存在；
> 且 conda-pack 快照会排除 editable 包 ⇒ 必须补跑 `vlvm_setup_env.sh`。

# 附录 B：与旧版指南的差异（2026-09-16 修正）

| 项 | 旧版 | 现版 | 依据 |
|---|---|---|---|
| conda 环境名 | `vlvm` | **`vlfm`** | `scripts/my_eval.sh` 实机生效值 |
| 子模块路径 | `$VLVM/habitat-lab` 等 | **`$DEPS = /root/autodl-tmp/vlfm`** | `python -c "import habitat"` 实机解析 |
| transformers | 正文 4.26.0，快照 4.57.6 | **4.26.0**（锁定） | 实机 `pip freeze` |
| 快照文件 | 含 `/root/autodl-tmp/vlfm` 路径条目与 `mobile-sam` | 重新导出（无路径条目） | `conda env export --no-builds` |
| 新增 | — | `vlvm_setup_env.sh` / `vlvm_env_audit.md` | 本次整理 |
