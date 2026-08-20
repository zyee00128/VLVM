#!/bin/bash

# ==============================================================================
# my_eval.sh (For VLVM) -- serial A/B comparison for the 2.5D semantic value plane (BEV)
# Purpose:
#   1. Enable NVIDIA EGL hardware-accelerated rendering to speed up simulation
#   2. Serially run 3D baseline / V1 (region) / V2 (surface) across h_lam values
#   3. Dataset fixed to HM3D 5cdEh9F2hJL (unchanged); logs saved as <tag>_<time>.log
# Note: start the VLM servers first (run launch_vlm_servers.sh in tmux)
# ==============================================================================

# ----------------- Project base path config -----------------
PROJECT_ROOT="/root/autodl-tmp/vlvm"
CONDA_ENV_NAME="vlfm"

# Fall back to the current directory if the project root is missing
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(pwd)
fi

cd "$PROJECT_ROOT"

# Load the conda environment
CONDA_PROFILE="/root/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate "$CONDA_ENV_NAME"
else
    echo "Warning: $CONDA_PROFILE not found, trying the default conda command..."
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
fi

# ----------------- GPU & EGL rendering environment variables -----------------
export CUDA_VISIBLE_DEVICES=0
export EGL_PLATFORM=surfaceless
export FORCE_GLX_USE_EGL=1
export MAGNUM_LOG=quiet
export MAGNUM_GPU_VALIDATION=OFF
export HF_ENDPOINT=https://hf-mirror.com
unset DISPLAY
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
set -o pipefail

echo "================================================="
echo "环境变量配置完毕！"
echo "项目路径: $PROJECT_ROOT"
echo "当前 Conda 环境: $CONDA_DEFAULT_ENV"
echo "渲染引擎模式: NVIDIA EGL 硬件加速"
echo "================================================="

# ----------------- Output dir & dataset -----------------
OUT_DIR="$PROJECT_ROOT/outputs/$(date +%Y-%m-%d)"
mkdir -p "$OUT_DIR"
SCENES="[5cdEh9F2hJL]"   # HM3D scene (dataset fixed, do not change)

python -m vlfm.run \
    habitat.dataset.content_scenes="$SCENES" \
    habitat_baselines.eval.split=val \
    habitat_baselines.num_environments=1 \
    habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
    habitat.simulator.habitat_sim_v0.gpu_gpu=False 2>&1 \
    | tee "$OUT_DIR/eval_hm3d_$(date +%H%M%S).log"

echo ""
echo "================================================="
echo "All experiments completed!"
echo "Log dir: $OUT_DIR"
echo "================================================="
