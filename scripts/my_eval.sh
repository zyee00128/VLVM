#!/bin/bash

# ==============================================================================
# 脚本名称: my_eval.sh (For VLVM) —— 自动实验脚本（VLVM-V4.3 阶段）
# 用途:
#   1. 启用 NVIDIA EGL 硬件加速渲染
#   2. 基座参数由 yaml 提供（σ_tar=0.70, fb_suspicious_conf=0.75 激活, psr=0.90, 回退开）
#   3. 按 EXPERIMENTS 数组自动串行跑多档实验，每档用 hydra 覆盖参数（如 use_world_map）
#   4. 每档日志按「标签」命名，存 outputs/VLVM-V4.3
# 用法: bash scripts/my_eval.sh   （改 EXPERIMENTS 数组即可换实验）
# 注意: blip2itm 服务不重启；需先启动 VLM 服务（tmux 中跑 launch_vlm_servers.sh）
#       换 TSP3D 权重需重启 tsp3d 服务（见 VLVM4 权重 A/B 备注）
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

# ----------------- 输出目录与数据集 -----------------
OUT_DIR="$PROJECT_ROOT/outputs/VLVM-V4.3"
mkdir -p "$OUT_DIR"
SCENES="[5cdEh9F2hJL]"   # HM3D 场景（数据集固定）

export TSP3D_DATA_PATH=${TSP3D_DATA_PATH:-/root/autodl-tmp/vlvm/data/tsp3d_models/}
TSP3D_PORT=${TSP3D_PORT:-12186}

# ----------------- 实验定义 -----------------
# 格式: 标签 | 参数覆盖（多个 hydra 键=值，空格分隔；值含 { } 需单引号包裹）
# 基线 = cam_B1（yaml 默认：σ_tar=0.70 / fb_suspicious_conf=0.75 / psr=0.90 / 回退开 /
#        probabilistic / camera 路线 B1 视角去重 0.15/15°，s_penalty 全关）——复用已有 0.3030 结果
EXPERIMENTS=(
    # ===== 4.6.1 多候选 + top-1 网格（基座 = gate：enable_s_penalty + thresh=0.15 + gate_admission=True）=====
    "nms_gate_t03|habitat_baselines.rl.policy.enable_s_penalty=True habitat_baselines.rl.policy.s_penalty_thresh=0.15 habitat_baselines.rl.policy.s_penalty_gate_admission=True habitat_baselines.rl.policy.nms_score_thr=0.3 habitat_baselines.rl.policy.top1_write=True"
    "nms_gate_t05|habitat_baselines.rl.policy.enable_s_penalty=True habitat_baselines.rl.policy.s_penalty_thresh=0.15 habitat_baselines.rl.policy.s_penalty_gate_admission=True habitat_baselines.rl.policy.nms_score_thr=0.5 habitat_baselines.rl.policy.top1_write=True"
    "nms_gate_t07|habitat_baselines.rl.policy.enable_s_penalty=True habitat_baselines.rl.policy.s_penalty_thresh=0.15 habitat_baselines.rl.policy.s_penalty_gate_admission=True habitat_baselines.rl.policy.nms_score_thr=0.7 habitat_baselines.rl.policy.top1_write=True"
)

# ----------------- 运行 -----------------
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r LABEL OVERRIDE <<< "$exp"
    echo "================================================="
    echo ">>> 开始实验: $LABEL"
    echo ">>> 覆盖参数: $OVERRIDE"
    echo "================================================="

    args=(
        habitat.dataset.content_scenes="$SCENES"
        habitat_baselines.eval.split=val
        habitat_baselines.num_environments=1
        habitat.simulator.habitat_sim_v0.gpu_device_id=0
        habitat.simulator.habitat_sim_v0.gpu_gpu=False
        habitat_baselines.rl.policy.name="HabitatITMPolicyV1"
    )
    # 拆分为多个 hydra 覆盖（空格分隔，值无空格）
    for ov in $OVERRIDE; do
        args+=("$ov")
    done

    LOGFILE="$OUT_DIR/eval_hm3d_$(date +%Y-%m-%d)_$(date +%H%M%S)_${LABEL}.log"
    python -m vlfm.run "${args[@]}" 2>&1 | tee "$LOGFILE"
    echo ">>> 完成: $LABEL -> $(basename "$LOGFILE")"
done

echo ""
echo "================================================="
echo "全部实验完成！"
echo "日志目录: $OUT_DIR"
echo "================================================="
