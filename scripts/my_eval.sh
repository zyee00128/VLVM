#!/bin/bash

# ==============================================================================
# 脚本名称: my_eval.sh (For VLVM) —— VLVM4 5.2-2：TSP3D 权重 A/B（sr3d / nr3d）
# 用途:
#   1. 启用 NVIDIA EGL 硬件加速渲染
#   2. 基座/最优参数由 yaml 提供（σ_tar=0.70，enable_fb=True 回退开，psr=0.90）
#   3. 每组实验前用对应 TSP3D_CHECKPOINT 重启 tsp3d 服务（TSP3D 权重在服务端加载，切换必须重启）
#   4. 数据集固定 HM3D 5cdEh9F2hJL，日志存 outputs/VLVM-V4
# 注意: blip2itm 服务不重启（与权重无关）；需先启动 blip2itm（tmux 中跑 launch_vlm_servers.sh）
# ==============================================================================

# ----------------- 项目基础路径配置 -----------------
PROJECT_ROOT="/root/autodl-tmp/vlvm"
CONDA_ENV_NAME="vlfm"

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

# ----------------- 输出目录与数据集 -----------------
OUT_DIR="$PROJECT_ROOT/outputs/VLVM-V4.2"
mkdir -p "$OUT_DIR"
SCENES="[5cdEh9F2hJL]"

export TSP3D_DATA_PATH=${TSP3D_DATA_PATH:-/root/autodl-tmp/vlvm/data/tsp3d_models/}
TSP3D_PORT=${TSP3D_PORT:-12186}

# ----------------- 实验定义 -----------------
args=(
        habitat.dataset.content_scenes="$SCENES"
        habitat_baselines.eval.split=val
        habitat_baselines.num_environments=1
        habitat.simulator.habitat_sim_v0.gpu_device_id=0
        habitat.simulator.habitat_sim_v0.gpu_gpu=False
        habitat_baselines.rl.policy.name="HabitatITMPolicyV1"
    )

python -m vlfm.run "${args[@]}" 2>&1 \
        | tee "$OUT_DIR/eval_hm3d_$(date +%Y-%m-%d)_$(date +%H%M%S)_baseline.log"

echo ""
echo "================================================="
echo "实验完成！"
echo "日志目录: $OUT_DIR"
echo "================================================="
