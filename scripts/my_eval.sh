#!/bin/bash

# ==============================================================================
# 脚本名称: my_eval.sh (For VLVM) —— V3 几何障碍场最终方案实验
# 用途:
#   1. 启用 NVIDIA EGL 硬件加速渲染
#   2. 跑最终方案（#10）：ProbabilisticGrid 累积 explored 掩码 + 消费层排除障碍
#      （om_style=probabilistic），V1(区域式) + λ=0.3
#   3. 数据集固定为 HM3D 5cdEh9F2hJL，日志按 标签_时间 分开保存
# 注意: 运行前需先启动 VLM 服务（tmux 中跑 launch_vlm_servers.sh）
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
OUT_DIR="$PROJECT_ROOT/outputs/$(date +%Y-%m-%d)"
mkdir -p "$OUT_DIR"
SCENES="[5cdEh9F2hJL]"   # HM3D 场景（数据集固定，不改）

# ----------------- 实验定义 -----------------
# 格式: 标签 | 策略名 | vm_style | h_lam | om_style | log_odds_occ | log_odds_free | occ_threshold | free_threshold
# 08-21 保守抵消（方向 C）A/B：#10 基础上 log_odds_occ=2.0 / free=-1.5（不对称，抵消不归零）
#   + occ_thr=1.0 / free_thr=-1.0（滞回，(-1,+1) 为 Unknown 缓冲带）
EXPERIMENTS=(
  "v1_prob_consv_C|HabitatITMPolicyV1|region|0.3|probabilistic|2.0|-1.5|1.0|-1.0"
)

echo "共 ${#EXPERIMENTS[@]} 组实验，串行执行..."
echo "日志目录: $OUT_DIR"

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r TAG POLICY STYLE H_LAM OM_STYLE LO_OCC LO_FREE OCC_THR FREE_THR <<< "$exp"

    echo ""
    echo "==================== 实验: $TAG ===================="
    echo "  策略: $POLICY | 模式: ${STYLE:-默认} | λ: ${H_LAM:-默认} | om_style: ${OM_STYLE:-默认}"
    echo "  log-odds: occ=${LO_OCC:-默认} free=${LO_FREE:-默认} occ_thr=${OCC_THR:-默认} free_thr=${FREE_THR:-默认}"

    args=(
        habitat.dataset.content_scenes="$SCENES"
        habitat_baselines.eval.split=val
        habitat_baselines.num_environments=1
        habitat.simulator.habitat_sim_v0.gpu_device_id=0
        habitat.simulator.habitat_sim_v0.gpu_gpu=False
        habitat_baselines.rl.policy.name="$POLICY"
    )
    if [ -n "$STYLE" ]; then
        args+=(habitat_baselines.rl.policy.vm_style="$STYLE")
    fi
    if [ -n "$H_LAM" ]; then
        args+=(habitat_baselines.rl.policy.h_lam="$H_LAM")
    fi
    if [ -n "$OM_STYLE" ]; then
        args+=(habitat_baselines.rl.policy.om_style="$OM_STYLE")
    fi
    if [ -n "$LO_OCC" ]; then
        args+=(habitat_baselines.rl.policy.log_odds_occ="$LO_OCC")
    fi
    if [ -n "$LO_FREE" ]; then
        args+=(habitat_baselines.rl.policy.log_odds_free="$LO_FREE")
    fi
    if [ -n "$OCC_THR" ]; then
        args+=(habitat_baselines.rl.policy.occ_threshold="$OCC_THR")
    fi
    if [ -n "$FREE_THR" ]; then
        args+=(habitat_baselines.rl.policy.free_threshold="$FREE_THR")
    fi

    python -m vlfm.run "${args[@]}" 2>&1 \
        | tee "$OUT_DIR/eval_hm3d_${TAG}_$(date +%H%M%S).log"

    echo "===== 实验 $TAG 完成 ====="
done

echo ""
echo "================================================="
echo "全部 ${#EXPERIMENTS[@]} 组实验完成！"
echo "日志目录: $OUT_DIR"
echo "================================================="
