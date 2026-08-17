#!/bin/bash

# ==============================================================================
# 脚本名称: my_eval.sh (For VLVM)
# 主要优化:
#   1. 启用 NVIDIA EGL 硬件加速渲染，大幅提升仿真速率
#   2. 启用 GPU 支持，并将数据直接分配给 3090 显卡处理
#   3. 配置国内 HuggingFace 镜像源，防止多模态大模型权重下载中断
# ==============================================================================

# ----------------- 项目基础路径配置 -----------------
PROJECT_ROOT="/root/autodl-tmp/vlvm"
CONDA_ENV_NAME="vlfm"

# 如果当前就在项目根目录下，自动获取当前路径
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(pwd)
fi

cd "$PROJECT_ROOT"

# 加载 Conda 环境
CONDA_PROFILE="/root/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate "$CONDA_ENV_NAME"
else
    echo "警告: 未找到 $CONDA_PROFILE，尝试使用默认 conda 命令..."
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
fi

# ----------------- 声明 GPU 与 EGL 渲染环境变量 -----------------
# 指定使用 0 号 GPU (RTX 3090)
export CUDA_VISIBLE_DEVICES=0

# 配置 EGL 无头服务器渲染
export EGL_PLATFORM=surfaceless
export FORCE_GLX_USE_EGL=1

# 关闭 Magnum 渲染器内部 GPU 校验警告及不必要的渲染日志
export MAGNUM_LOG=quiet
export MAGNUM_GPU_VALIDATION=OFF

# 设置 HuggingFace 国内镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 清空虚拟显示屏设置（无头服务器模式）
unset DISPLAY

echo "================================================="
echo "环境变量配置完毕！"
echo "项目路径: $PROJECT_ROOT"
echo "当前 Conda 环境: $CONDA_DEFAULT_ENV"
echo "使用 GPU 设备: RTX 3090 (CUDA:0)"
echo "渲染引擎模式: NVIDIA EGL 硬件加速"
echo "导航系统: VLVM (3D Native Paradigm)"
echo "================================================="
MP3D_PATH="data/datasets/objectnav/mp3d/val/val.json.gz"

# ----------------- 运行评测命令 -----------------
export PYTHONPATH="/root/autodl-tmp/vlvm:$PYTHONPATH"
set -o pipefail
# 运行 HM3D 数据集

python -m vlfm.run \
    habitat.dataset.content_scenes="[5cdEh9F2hJL]" \
    habitat_baselines.eval.split=val \
    habitat_baselines.num_environments=1 \
    habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
    habitat.simulator.habitat_sim_v0.gpu_gpu=False 2>&1 | tee "$PROJECT_ROOT/outputs/$(date +%Y-%m-%d)/eval_hm3d_$(date +%H%M%S).log"

# 运行 MP3D 数据集
# python -m vlfm.run \
#     habitat.dataset.data_path="$MP3D_PATH" \
#     habitat_baselines.eval.split=val \
#     habitat_baselines.num_environments=1 \
#     habitat_baselines.test_episode_count=700 \
#     habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
#     habitat.simulator.habitat_sim_v0.gpu_gpu=False 2>&1 | tee "$PROJECT_ROOT/outputs/$(date +%Y-%m-%d)/eval_hm3d_$(date +%H%M%S).log"

