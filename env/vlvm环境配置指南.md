# VLVM 项目环境配置指南

> **配置顺序**：先装 **vlfm 导航框架环境**（第一部分）→ 再装 **TSP3D 模型依赖**（第二部分）→ 最后做**外部依赖与路径重定向**（第三部分）→ 用 `env/vlvm_setup_env.sh` 一键校验/清理（第四部分）。

> **目录约定**：`$WS` = 工作区根，`$VLVM` = 仓库根（`$WS/vlvm`），`$DEPS` = 外部源码根（含 `habitat-lab-<sha>/`、`MinkowskiEngine/`、`mmcv/`）。

> **一键脚本**：`bash $VLVM/env/vlvm_setup_env.sh --all`（= 自动配置 + 清理残余包 + 环境校验）。
> 另有 `--setup` / `--clean` / `--check` / `--fix` 单功能入口与 `--clean --dry-run` 预演；
> 路径/conda 根/环境名（`vlvm` 否则回退 `vlfm`）均自动探测。

---

## 开始之前：国内镜像与依赖下载源

**① HuggingFace 镜像（权重与 HF 模型，国内必用）**

```bash
export HF_ENDPOINT=https://hf-mirror.com      # huggingface_hub / transformers 自动走镜像
# 单文件下载（以 TSP3D 主权重为例）：
huggingface-cli download Zoe11111/VLVM tsp3d_scanrefer.pth --local-dir $VLVM/data/tsp3d_models
```
浏览器下载同理：把 `https://huggingface.co/<owner>/<repo>/...` 换成 `https://hf-mirror.com/<owner>/<repo>/...` 即可。

**② GitHub 镜像（habitat-lab / MinkowskiEngine / mmcv 等源码，国内直连 github.com 常超时）**

```bash
# 方式 A：git 前缀代理（配置一次，之后所有 github.com 的 clone/fetch 自动经代理）
git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/"
# 取消：git config --global --unset-all url."https://gh-proxy.com/https://github.com/".insteadOf

# 方式 B：手动拼前缀（适合下 zip / tar.gz）
git clone https://gh-proxy.com/https://github.com/open-mmlab/mmcv.git
wget https://gh-proxy.com/https://github.com/open-mmlab/mmcv/archive/refs/tags/v2.1.0.tar.gz
```
公共代理前缀（`gh-proxy.com` / `ghfast.top` / `gh-proxy.net` 等）的可用性会变化，失效时换一个；也可用 Gitee 镜像仓库（`gitee.com/mirrors/<repo>`）。

**③ PyPI 镜像（pip 包）**

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>
# 或全局：pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**④ 仓库外依赖一览（下载源 → 解压/克隆到 → 怎么装）**

| 依赖 | 下载源 | 解压 / 克隆到 | 安装 |
|---|---|---|---|
| habitat-lab 0.2.4（含 habitat-baselines） | GitHub `facebookresearch/habitat-lab` @ `1639e1ae732ba1e84199a1a04b79c7243c3f8586` | `$WS/habitat-lab-1639e1ae732ba1e84199a1a04b79c7243c3f8586/` | 两个子目录分别 `pip install -e .`（§1.4） |
| MinkowskiEngine 0.5.4 | GitHub `NVIDIA/MinkowskiEngine` tag `v0.5.4`（或 PyPI 源码包） | `$WS/MinkowskiEngine/` | 打补丁 → 编译 wheel → 安装（§2.2） |
| mmcv 2.1.0 | GitHub `open-mmlab/mmcv` tag `v2.1.0` | `$WS/mmcv/` | 编译 wheel → 安装（§2.3） |
| frontier_exploration / depth_camera_filtering | GitHub `naokiyokoyama/frontier_exploration`、`naokiyokoyama/depth_camera_filtering` | `$WS/frontier_exploration/`、`$WS/depth_camera_filtering/` | `pip install -e .`（§1.5） |
| GroundingDINO（GD 辅助源） | 已随本仓库提供（`$VLVM/GroundingDINO/`） | — | `pip install -e .`（§1.5） |

> `$WS` = 工作区根，`$VLVM` = 本仓库根（`$WS/vlvm`）。

## 0. 环境速览

| 项 | 机器 B（RTX 5080） | 机器 A（RTX 3090） |
|---|---|---|
| GPU / 驱动 / CUDA | RTX 5080 16 GB，驱动 580.178.04，CUDA 12.8 | RTX 3090 24 GB，驱动 570.124.04，CUDA 11.3 运行时 |
| 系统工具链 | gcc 11、Ubuntu 22.04 | gcc 9、Ubuntu 20.04 |
| 工具链 CUDA | `/usr/local/cuda-12.8`（nvcc 12.8.93），`TORCH_CUDA_ARCH_LIST=12.0` | `/usr/local/cuda-11.3`，`TORCH_CUDA_ARCH_LIST=8.6` |
| conda | `/home/iair/miniconda3`，环境 **`vlvm`**（Python 3.9.23） | `/root/miniconda3`，环境 **`vlfm`**（Python 3.9） |
| torch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 | 1.12.1+cu113 / 0.13.1+cu113 |
| transformers | **4.26.0**（锁死，>4.26.0 破坏 BLIP-2） | 同左 |
| 三维算子 | MinkowskiEngine 0.5.4 / mmcv 2.1.0 / mmdet 3.2.0 / mmdet3d 1.4.0 / **mmengine 0.10.7** | 同左，但 **mmengine 0.10.4**（0.10.7 未必要；0.10.4 与 torch 1.12 兼容） |
| 仿真 | habitat-sim 0.2.4；habitat-lab / habitat-baselines 0.2.4（editable 源码，`$DEPS` 下） | 同左；源码在 `$WS/vlfm/habitat-lab/` |
| editable 包 | vlfm、groundingdino、habitat-lab、habitat-baselines（另 2 个见 §3.1） | 同左（6 个均可 editable，源码齐备） |

**环境文件（`env/` 目录）**

| 文件 | 用途 |
|---|---|
| `vlvm_requirements.txt` | **直接依赖清单**（两套 CUDA 栈分别标注 + 锁定版本 + 注意事项） |
| `vlvm_setup_env.sh` | **一键脚本**：自动配置（editable/conda 变量）→ 清理残余包 → 环境校验（双栈版本表自动选） |
| `vlvm_env_export.yml` | **旧机全量快照**（RTX 3090/cu113 时代），仅作历史参考 |


# 第一部分：vlfm 导航框架环境

## 1.1 创建 conda 环境

```bash
conda create -n vlvm python=3.9 -y && conda activate vlvm
```

> 环境名共存约定：脚本优先 `vlvm`，不存在则回退探测 `vlfm`（`CONDA_ENV_NAME` 可覆盖）。

## 1.2 安装 PyTorch

```bash
# 机器 B（CUDA 12.8 / sm_120）：
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 机器 A（CUDA 11.3 / sm_86）：
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
     -f https://download.pytorch.org/whl/torch_stable.html

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 机器 B → 2.8.0+cu128 True；机器 A → 1.12.1+cu113 True

# 编译 TSP3D 算子：
# 机器 B
conda env config vars set -n vlvm \
    CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0

# 机器 A 对应为 
conda env config vars set -n vlvm \
    CUDA_HOME=/usr/local/cuda-11.3 TORCH_CUDA_ARCH_LIST=8.6

# 设置后重新 conda deactivate && conda activate <env>
```

> 驱动“向下兼容”不能补出 torch wheel 中不存在的 sm_120 kernel。
> 若本机不做算子编译（沿用预编译 wheel / 已装好的环境），可跳过 `conda env config vars`。

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

**① 下载 habitat-lab 源码（锁定 commit `1639e1ae`）**

```bash
cd $WS
# 方式 A（git clone，需已配置 GitHub 代理，见文首镜像说明）：
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab && git checkout 1639e1ae732ba1e84199a1a04b79c7243c3f8586 && cd ..
mv habitat-lab habitat-lab-1639e1ae732ba1e84199a1a04b79c7243c3f8586
# 方式 B（tar 包，经代理）：
# wget https://gh-proxy.com/https://github.com/facebookresearch/habitat-lab/archive/1639e1ae732ba1e84199a1a04b79c7243c3f8586.tar.gz
# tar xzf 1639e1ae732ba1e84199a1a04b79c7243c3f8586.tar.gz   # 解压出 habitat-lab-1639e1ae.../
```

> **解压/克隆到**：`$WS/habitat-lab-1639e1ae732ba1e84199a1a04b79c7243c3f8586/`（其内含 `habitat-lab/` 与 `habitat-baselines/` 两个子目录）。

**② 安装**（两个子目录分别 editable 安装）：

```bash
pip install habitat-sim==0.2.4
cd $WS/habitat-lab-1639e1ae732ba1e84199a1a04b79c7243c3f8586/habitat-lab       && pip install -e .
cd $WS/habitat-lab-1639e1ae732ba1e84199a1a04b79c7243c3f8586/habitat-baselines && pip install -e .
```

> 装完**保留源码目录**（editable 指向它，勿删）；若下载过 `.tar.gz` 包，此时可删。

## 1.5 安装项目本体与子模块（editable）

```bash
cd $VLVM && pip install -e .                  # vlfm 包（源码在 $VLVM/vlfm）
cd $VLVM/GroundingDINO && pip install -e .    # GD 辅助机制依赖
# VLFM 的两个依赖仓库（需先从 GitHub 克隆到 $WS，见下方说明）：
cd $WS/frontier_exploration   && pip install -e .
cd $WS/depth_camera_filtering && pip install -e .
```

> ⚠️ **`frontier_exploration` 不是可选包**：`vlfm/run.py` 顶部 `import frontier_exploration`
> 用于注册 habitat 的 `frontier_sensor` / `frontier_exploration_map` 配置组；缺失时 hydra 合成直接报
> `Could not load 'habitat/task/lab_sensors/frontier_sensor'`。

> ⚠️ `frontier_exploration` / `depth_camera_filtering` 是 VLFM 的两个**独立依赖仓库**，不在 `vlvm` 内：
> ```bash
> cd $WS
> git clone https://github.com/naokiyokoyama/frontier_exploration.git
> git clone https://github.com/naokiyokoyama/depth_camera_filtering.git
> ```
> 克隆后按上面的 `pip install -e` 安装（只做拷贝安装也能运行，但改不了源码）。

---

# 第二部分：TSP3D 模型环境

TSP3D 需要额外的**编译型依赖**：MinkowskiEngine（稀疏卷积）与 mmcv / mmdet3d（3D 检测算子）。

## 2.1 安装系统级 BLAS

```bash
apt-get install -y libopenblas-dev
```

> 不装会在编译时报 `BLAS not found from numpy.distutils.system_info.get_info`。

## 2.2 安装 MinkowskiEngine 0.5.4（源码编译）

PyPI **只有源码包（无预编译 wheel）**，必须现场编译：

**① 安装当前已验证的构建工具**：

```bash
pip install setuptools==78.1.0 ninja==1.13.0
```

> 不要为满足未参与运行链路的 `openxlab 0.1.3` 而把 setuptools 降到 60.2；该版本与
> 当前 pip 的构建元数据流程不兼容，会导致源码包在生成 wheel 前失败。

**② 下载源码、解压到 `$WS/MinkowskiEngine`，并写 BLAS 配置**：

```bash
# 方式 A（PyPI 源码包）：
cd /tmp && pip download MinkowskiEngine==0.5.4 --no-deps --no-binary :all: -d me_src
tar xzf me_src/MinkowskiEngine-0.5.4.tar.gz -C /tmp
# 方式 B（GitHub tag v0.5.4，经代理，见文首镜像说明）：
# wget https://gh-proxy.com/https://github.com/NVIDIA/MinkowskiEngine/archive/refs/tags/v0.5.4.tar.gz -O /tmp/ME-v0.5.4.tar.gz
# tar xzf /tmp/ME-v0.5.4.tar.gz -C /tmp          # 解压出 MinkowskiEngine-0.5.4/

# 固定放到工作区（后续补丁/编译都在这里做；换机可整目录复用）：
mv /tmp/MinkowskiEngine-0.5.4 $WS/MinkowskiEngine

cat > $WS/MinkowskiEngine/site.cfg <<'EOF'
[openblas]
libraries = openblas
library_dirs = /usr/lib/x86_64-linux-gnu
include_dirs = /usr/include/x86_64-linux-gnu/openblas-pthread
EOF
```

**③ 打 torch 2.8 / CUDA 12.8 兼容补丁**：

- 在以下 8 个源文件中，于 `#include <ATen/cuda/CUDAUtils.h>` / `#include <ATen/cuda/CUDAContext.h>` 之前插入 `#include <ATen/ATen.h>`：
  `src/spmm.cu`、`src/broadcast_gpu.cu`、`src/convolution_gpu.cu`、`src/convolution_kernel.cu`、`src/convolution_transpose_gpu.cu`、`src/allocators.cuh`、`src/common.hpp`、`src/direct_max_pool.cpp`
- 将 `pybind/extern.hpp` 中 `.def(py::self == py::self)` 改为 `.def("__eq__", &minkowski::CoordinateMapKey::operator==, py::is_operator())`
- CUDA 12.8 还需移除 `src/3rdparty/concurrent_unordered_map.cuh` 内旧 cuDF NVTX 包装，
  并为 Thrust 调用补齐 `execution_policy` / `remove` / `unique` / `sort` / `reduce` 等头文件；
  补丁对象 = 上一步解压到 `$WS/MinkowskiEngine` 的源码，补好后原样保留（换机/重编可复用）。

**④ 编译并安装**：

```bash
cd $WS/MinkowskiEngine
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4
python setup.py bdist_wheel --blas=openblas
pip install dist/minkowskiengine-0.5.4-*.whl
```

> `TORCH_CUDA_ARCH_LIST` 必须匹配目标 GPU；RTX 5080 使用 `12.0`。
> 机器 A 改为 `CUDA_HOME=/usr/local/cuda-11.3 TORCH_CUDA_ARCH_LIST=8.6`；
> torch 1.12 下额外需要“torch 头文件补丁”（`ATen/ATen.h` 插入、`py::self == py::self` 改写、
> 旧 cuDF NVTX 包装移除与 Thrust `execution_policy` 补齐均同样适用，见仓库历史笔记）。
> 单次编译约 15–30 分钟；**产出的 wheel 建议留存复用**（同 GPU 架构可跨机安装），位于
> `$WS/MinkowskiEngine/dist/minkowskiengine-0.5.4-cp39-cp39-linux_x86_64.whl`。
> 装好后可清理编译中间产物：`rm -rf $WS/MinkowskiEngine/build`（`dist/*.whl` 与源码目录保留）。

### 验证 MinkowskiEngine

```bash
python -c "
import torch, MinkowskiEngine as ME
coords = torch.cat([torch.randint(0,10,(100,3)).int(), torch.zeros(100,1).int()],1).cuda()
sp = ME.SparseTensor(torch.randn(100,3).cuda(), coords)
out = ME.MinkowskiConvolution(3,16,3,dimension=3).cuda()(sp)
print('ME OK', out.shape)"      # 期望 ME OK torch.Size([95, 16])
```

## 2.3 安装 mmcv

**下载 / 解压到 `$WS/mmcv`**（两条路选一）：

```bash
cd $WS
# 方式 A：git clone（tag v2.1.0）
git clone --branch v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git
# 方式 B：tar 包（经代理，见文首镜像说明）
# wget https://gh-proxy.com/https://github.com/open-mmlab/mmcv/archive/refs/tags/v2.1.0.tar.gz
# tar xzf v2.1.0.tar.gz && mv mmcv-2.1.0 mmcv
```

**编译安装**：

```bash
# 机器 B（CUDA 12.8 / sm_120）：源码编译完整 mmcv
cd $WS/mmcv
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4 MMCV_WITH_OPS=1 FORCE_CUDA=1
python setup.py bdist_wheel
pip uninstall -y mmcv-lite
pip install dist/mmcv-2.1.0-*.whl

# 机器 A（CUDA 11.3 / torch 1.12）：官方预编译 wheel，无需编译
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```

> OpenMMLab 没有 torch 2.8/cu128/sm_120 的 MMCV 2.1.0 官方 wheel，机器 B 必须源码编译完整
> `mmcv`；`mmcv-lite` 不含 mmdet3d 需要的 `_ext` CUDA 算子。机器 A 直接 pip 安装即可。
> 编译产物 `$WS/mmcv/dist/mmcv-2.1.0-cp39-cp39-linux_x86_64.whl` 建议留存（同架构可跨机复用）；
> 装好后可删中间产物：`rm -rf $WS/mmcv/build`。

## 2.4 安装 mmdet / mmdet3d

```bash
# 机器 B：
pip install mmdet==3.2.0 mmdet3d==1.4.0 mmengine==0.10.7

# 机器 A：
pip install mmdet==3.2.0 mmdet3d==1.4.0 mmengine==0.10.4
```

> torch 2.8 下使用 mmengine 0.10.7；0.10.4 会与新增的 `torch.optim.Adafactor`
> 重复注册。机器 A torch 1.12 下继续用 0.10.4。
> `mmdet3d` 提供 TSP3D 的 3D 框结构（`DepthInstance3DBoxes` 等）。

## 2.5 验证 TSP3D 环境

```bash
cd $VLVM
python -c "from vlfm.tsp3d_models.bdetr import BeaUTyDETR; print('BeaUTyDETR OK')"
python -m vlfm.vlm.tsp3d --port 12186
# 期望: [TSP3D] Weights loaded from ...  / Running on http://localhost:12186
```

---

# 第三部分：外部依赖、权重与路径重定向

## 3.1 外部依赖与放置位置（安装完成后核对）

前三部分从仓库外引入的依赖及目标位置（下载源见文首总表），安装完成后应存在：

| 位置 | 体积约 | 用途 | 备注 |
|---|---|---|---|
| `$WS/habitat-lab-1639e1ae732ba1e84199a1a04b79c7243c3f8586/` | 24 MB | `habitat` / `habitat_baselines` editable 源码 | **勿删**（editable 指向它） |
| `$WS/MinkowskiEngine/` | 55 MB（含 build 时更大） | 已打补丁源码 + 可复用 wheel（`dist/`） | 源码与 wheel 建议保留；`build/` 可删 |
| `$WS/mmcv/` | 140 MB | MMCV 源码 + 可复用 wheel（`dist/`） | 同上 |
| `$WS/frontier_exploration/`、`$WS/depth_camera_filtering/` | < 1 MB | habitat 配置组注册 + 深度相机滤波 | editable 安装（§1.5） |
| conda 环境 `site-packages` | — | 其余 pip 依赖（torch / lavis / habitat-sim 等） | 由第一、二部分 pip 安装 |

> 换机/重装提醒：`dist/*.whl`（MinkowskiEngine、mmcv）与源码目录可直接带到新机，装 wheel 即可跳过编译；
> `vlvm_setup_env.sh` 会自动探测这些目录的位置，无需手工改路径。

## 3.2 权重下载（`$VLVM/data/`）

**TSP3D 全部权重发布在 HuggingFace：https://huggingface.co/Zoe11111/VLVM/tree/main**
（含主权重 `tsp3d_scanrefer.pth`；备选档 `ckpt_sr3d.pth` / `ckpt_nr3d.pth` 同页）

```bash
export HF_ENDPOINT=https://hf-mirror.com        # 国内镜像，见文首说明
mkdir -p $VLVM/data/tsp3d_models
huggingface-cli download Zoe11111/VLVM tsp3d_scanrefer.pth --local-dir $VLVM/data/tsp3d_models
# 备选档（可选，历史 A/B 已判负）：
# huggingface-cli download Zoe11111/VLVM ckpt_sr3d.pth ckpt_nr3d.pth --local-dir $VLVM/data/tsp3d_models
# 无 CLI 时浏览器打开 https://hf-mirror.com/Zoe11111/VLVM/tree/main 逐个下载
```

| 路径 | 体积 | 说明 | 来源 |
|---|---|---|---|
| `data/tsp3d_models/tsp3d_scanrefer.pth` | 669 MB | **TSP3D 主权重（scanrefer 档；三档 A/B 中的唯一适用档）** | HuggingFace `Zoe11111/VLVM` |
| `data/tsp3d_models/ckpt_sr3d.pth` / `ckpt_nr3d.pth` | 各 668 MB | 备选档（sr3d 漏检 / nr3d 误检，历史判负） | 同上（可选） |
| `data/tsp3d_models/{model.safetensors, config.json, merges.txt, dict.txt, README.md}` | 约 476 MB | TSP3D 文本分支（RoBERTa-base），`data_path` 直接指向此目录 | HF 同 repo 随附（或 `roberta-base`） |
| `data/groundingdino_swint_ogc.pth` | 662 MB | GD 辅助提议源（端口 12181） | HF `ShilongLiu/GroundingDINO`（或 IDEA-Research GitHub release） |
| `data/pointnav_weights.pth` | 33 MB | PointNav 底层策略 | VLFM 官方发布物（下载指引见 `bdaiinstitute/vlfm` 仓库） |
| `data/dummy_policy.pth` | 52 KB | habitat-baselines eval 占位（缺失直接 exit） | VLFM 官方资源 / 项目内备份 |
| `data/bert-base-uncased/` | — | LAVIS（BLIP-2）的 BERT 目录（可选） | HF `bert-base-uncased`；不放置则走 HF 缓存 |

> ⚠️ **权重名已改为自动探测（2026-09-18）**：`scripts/launch_vlm_servers.sh` 与
> `vlfm/vlm/tsp3d.py` 不再写死档名，按在位文件依次探测
> `tsp3d_scanrefer.pth → ckpt_sr3d.pth → ckpt_nr3d.pth`：
> - 机器 A（RTX 3090）：在位 `tsp3d_scanrefer.pth` → 命中第一个（与历史实验一致，行为不变）
> - 机器 B（RTX 5080）：在位 `ckpt_sr3d.pth` → 回退命中第二个
>
> 需要切换档位时显式导出：`export TSP3D_CHECKPOINT=$VLVM/data/tsp3d_models/ckpt_nr3d.pth`。
> `vlvm_setup_env.sh` §3 会打印实际命中的档名，不命中时提示手动指定。
>
> **权重即模型配置**：TSP3D 主权重由 `TSP3D_CHECKPOINT` 指定；GD 辅助机制（V7.3 引入）以
> `gdp_enable / gdp_every_n=5 / gdp_max_boxes=1` 控制，属**辅助提议源**，不改变 TSP3D 的主模型地位。
> 换权重只改环境变量或 YAML，不动代码。

## 3.3 数据集（下载与软链）

数据集**不进仓库**：实体放数据盘（记为 `<DATA_ROOT>`），仓库 `data/` 下只建软链。

```bash
# ① ObjectNav 任务数据（episode JSON）
mkdir -p $VLVM/data/datasets/objectnav/hm3d
ln -s <DATA_ROOT>/objectnav_hm3d_v2 $VLVM/data/datasets/objectnav/hm3d/v1

# ② HM3D 场景（val 必需、train 可选；episode 内路径以 hm3d_v0.2/ 开头，必须保留该目录名）
mkdir -p $VLVM/data/scene_datasets/hm3d_v0.2
ln -s <DATA_ROOT>/hm3d-train-habitat-v0.2 $VLVM/data/scene_datasets/hm3d_v0.2/train
ln -s <DATA_ROOT>/hm3d-val-habitat-v0.2   $VLVM/data/scene_datasets/hm3d_v0.2/val
cd $VLVM/data/scene_datasets && ln -s hm3d_v0.2 hm3d

# ③ MP3D（可选，仅 MP3D 相关实验需要）
ln -s <DATA_ROOT>/mp3d_habitat/mp3d $VLVM/data/scene_datasets/mp3d

# ④ LAVIS BERT（可选；不建则 BLIP-2 走 HF 缓存）
ln -s <DATA_ROOT>/bert-base-uncased $VLVM/data/bert-base-uncased
```

> 下载源：HM3D 与 ObjectNav 数据从 AI Habitat 官方数据页（https://aihabitat.org/datasets/）获取；
> MP3D 按 pointnav 数据页的许可条款获取。**只装 val 场景 + ObjectNav v2 即可跑评测**。
> `vlvm_setup_env.sh` 会自动校验这些软链（含断链检测，两机布局均兼容）。

## 3.4 环境变量（运行前必须设置）

| 变量 | 建议值 | 说明 |
|---|---|---|
| `PYTHONPATH` | `$VLVM` | 保证 `import vlfm` 命中项目内源码 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像（模型下载） |
| `TSP3D_DATA_PATH` | `$VLVM/data/tsp3d_models/` | TSP3D 权重目录（服务端） |
| `TSP3D_CHECKPOINT` | 自动探测（见 §3.2） | 切换权重用；不设时按候选名探测 |
| `LAVIS_BERT_MODEL_PATH` | 可选（按在位探测） | 有 `data/bert-base-uncased` 才注入；无则 LAVIS 走 HF 缓存 |
| `TSP3D_PORT` / `BLIP2ITM_PORT` / `GROUNDING_DINO_PORT` | `12186` / `12182` / `12181` | 三个 VLM 服务端口（客户端代码默认值） |
| `EGL_PLATFORM` | `surfaceless` | habitat headless 渲染 |
| `MAGNUM_LOG` / `MAGNUM_GPU_VALIDATION` | `quiet` / `OFF` | 抑制 magnum 日志 |
| `TURN_LEFT_WIDE_TURNS` | 自动派生 | `run.py` 按 `panoramic_turn_steps` 计算写入，**不要手工设** |

`scripts/my_eval.sh` 与 `scripts/launch_vlm_servers.sh` 已内置上述设置（含 `unset DISPLAY`、`CUDA_VISIBLE_DEVICES=0`）。


## 3.5 一键配置 / 清理 / 校验

```bash
bash $VLVM/env/vlvm_setup_env.sh --all        # 自动配置 + 清理残余包 + 环境校验
bash $VLVM/env/vlvm_setup_env.sh              # 只校验（不改动，默认）
bash $VLVM/env/vlvm_setup_env.sh --setup      # 只做配置（editable 重定向 + conda 变量）
bash $VLVM/env/vlvm_setup_env.sh --clean      # 只做清理
bash $VLVM/env/vlvm_setup_env.sh --clean --dry-run   # 预演：只列出将删除的项
```

脚本自动探测 `WORKSPACE_ROOT` / `VLVM_ROOT` / `DEPS_ROOT` / `CONDA_ROOT` / `CONDA_ENV`
（`vlvm` 不存在时回退 `vlfm`），一般无需传参；确需覆盖时用环境变量：
`VLVM_ROOT=... DEPS_ROOT=... CONDA_ENV=... bash vlvm_setup_env.sh`。

**七段执行流程**

| 段 | 内容 |
|---|---|
| §0 | 路径/环境探测：python 版本、仓库结构、GPU、磁盘 |
| §1 | editable 包重定向（`--setup` 时按当前路径重装；否则只校验指向） |
| §2 | 关键包**导入 + 版本**双重校验（torch/CUDA、numpy、transformers、ME、mmcv 系、habitat 系、vlfm.run、TSP3D 模型） |
| §3 | 权重与数据：6 个权重文件 + 2 个备选权重 + 5 条数据软链（含断链检测） |
| §4 | 清理残余 pip 包（`--clean`） |
| §5 | 清理仓库构建残留（`--clean`） |
| §6–§7 | 环境变量提示、三个 VLM 服务端口状态 |

---

# 第四部分：安装后清理（只删"为本次安装下载/产生、之后不再需要"的内容）

| 时机 | 清理 | 务必保留 |
|---|---|---|
| MinkowskiEngine 编译 + 安装完成后 | `rm -rf $WS/MinkowskiEngine/build` | 源码目录、`dist/*.whl` |
| mmcv 编译 + 安装完成后 | `rm -rf $WS/mmcv/build` | 源码目录、`dist/*.whl` |
| GroundingDINO `pip install -e` 完成后 | `rm -rf $VLVM/GroundingDINO/build` | 预编译 `_C.*.so` 连同源码 |
| habitat-lab 装好后 | 下载的 `*.tar.gz` / `*.zip`（如有） | 解压出的 `$WS/habitat-lab-1639.../`（editable 指向） |
| 全部装完且磁盘紧张时（可选） | `pip cache purge`；`conda clean -a -y` | — |

> 仓库内的构建残留（`__pycache__` / `*.egg-info` / `.ipynb_checkpoints`）可随时清理：
> `bash $VLVM/env/vlvm_setup_env.sh --clean --dry-run` 预演，去掉 `--dry-run` 执行。
