#!/bin/bash

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
OUT_DIR="$PROJECT_ROOT/outputs/VLVM-V7.1"
mkdir -p "$OUT_DIR"
# 三场景（方向在多场景复评时用）：SCENES="[TEEsavR23oF, mv2HUxq3B53, wcojb4TFT35]"
SCENES="[5cdEh9F2hJL]"

export TSP3D_DATA_PATH=${TSP3D_DATA_PATH:-/root/autodl-tmp/vlvm/data/tsp3d_models/}
TSP3D_PORT=${TSP3D_PORT:-12186}

# ----------------- 实验定义 -----------------

# 格式: 标签 | 参数覆盖（多个 hydra 键=值，空格分隔）
#
# =====================================================================
# VLVM-V7.1 前置对比实验（2026-09-08，本机；V7 最终组合档 g_m8_d1 /
# g_m8_d1_pitch 在另一台服务器跑，勿在本机重跑）
#
# P0 · 采样确定性化已落地（代码，非档位）：
#   - policy 每 env 步 np.random.seed(det_seed + 单调计数)（覆盖发送端 cap 与
#     density cluster 子采样）；det_seed 默认 0。
#   - TSP3D 服务端每次 predict 前 torch.manual_seed(TSP3D_SEED + 查询计数)，
#     TSP3D_SEED 默认 0（覆盖 forward_test 内 CBA/keep 的 torch.randperm）。
#   ⚠️ 运行前必须重启 VLM 服务加载新代码：bash scripts/launch_vlm_servers.sh
#   （否则服务端旧代码无 torch seed / conf_floor，双跑不可能 bit-stable）。
#
# 档位说明：
#   - p0_rep_A / p0_rep_B：纯 world 定稿，两次完全相同 -> 验证 det 总数/SR
#     bit-stable（det 级随机源全固定后应复现）。
#   - p0_diag：world + diag_enable=True（服务端 conf_floor=0.30）——P3 bed-fn
#     前置诊断（记录低于 sigma_tar / geom gate 丢弃的真检测），末尾打印 fn 集
#     detection trace；行为与 base 等价（客户端仍按 sigma_tar 过滤），可兼作
#     diag 无副作用校验。
#   - p0_gm8d1：occ 门 + density①（多视角 20°）确定性重跑 —— 保留机制在
#     确定性 cap 下的再锚定（对照 V7 g_m8_d1，预期 ~45±1）。
# 锚点（99 集纯 world，5cdEh9F2hJL，cap 确定性化前）：world 44 / g_m8 45 /
# density①(v1_view) 44 / pitch 45。
# =====================================================================
EXPERIMENTS=(
    "p0_rep_A|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.det_seed=0"
    "p0_rep_B|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.det_seed=0"
    "p0_diag|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.det_seed=0 habitat_baselines.rl.policy.diag_enable=True habitat_baselines.rl.policy.diag_conf_floor=0.30"
    "p0_gm8d1|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.enable_occ_consistency=True habitat_baselines.rl.policy.enable_density_gate=True habitat_baselines.rl.policy.density_min_view_span_deg=20.0"
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
# bash ./scripts/launch_vlm_servers.sh
# bash ./scripts/my_eval.sh
