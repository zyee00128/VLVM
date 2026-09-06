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
# V7 机制评测输出（模块一 M2/M4/M8/M3 / 模块二 M6/M7/M9 / 模块三 M5 的 A/B，vs world 基线）
OUT_DIR="$PROJECT_ROOT/outputs/VLVM-V7"
mkdir -p "$OUT_DIR"
# 三场景（方向在多场景复评时用）：SCENES="[TEEsavR23oF, mv2HUxq3B53, wcojb4TFT35]"
SCENES="[5cdEh9F2hJL]"

export TSP3D_DATA_PATH=${TSP3D_DATA_PATH:-/root/autodl-tmp/vlvm/data/tsp3d_models/}
TSP3D_PORT=${TSP3D_PORT:-12186}

# ----------------- 实验定义 -----------------
# 格式: 标签 | 参数覆盖（多个 hydra 键=值，空格分隔；值含 { } 需单引号包裹）
#
# =====================================================================
# 当前网格：V7 模块三 软前沿引力 A/B（M5，soft_frontier_bias.py）
#   - 全部纯 world 单变量叠加（enable_scan=False）；对照 = world_v7_base
#     （world 定稿 SR 0.4444(44/99) / Oracle 54.55%，同轮同 shuffle、已跑共用不重跑）；
#   - M5 机制（Soft->Hard switch）：suspicious 单帧记录不硬锁导航，转为 explore
#     打分高斯引力 bias = w_bias * c' * exp(-d^2/2sigma^2)（d <= radius），注入点
#     itm_policy._sort_frontiers_by_value（region V1 / surface V2 共用）；
#   - 默认参数 bias=0.6 / sigma=1.5m / radius=4.5m(=3sigma)，均启发初值：
#     基础语义场 S 量级 ~0.1-0.2，0.6 是否压过/不足需 A/B 标定；
#   - 判据：SR 净超 world >= 2 集；Oracle 不降；spl/avg_steps（引力"顺路验证"
#     预期治假检测劫持与折返，SR 持平但 spl 改善亦记正向）；fp/fn/nv 归因；
#     检测级 precision / tp；盯 couch/bed 位姿敏感集（不得被打崩）；
#   - 每档 99 集 ≈ 3:45h；批1 = m5/m5_bw03/m5_bw12（独立 + bias 强度），
#     批2 = m5_sg25（sigma 扩散）。组合档（M5+M3/M8）待独立档判读后再定。
# =====================================================================
EXPERIMENTS=(
    # ---- 批1 (独立机制 + bias_weight 强度 A/B，单变量) ----
    # M5 默认参数：bias=0.6，相对基础打分 S+lambda*H1（~0.1-0.2）属中强引力。
    "m5|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.m5_enable=True"
    # bias_weight=0.30：弱引力（保守，仅轻微重塑前沿序，soft 源不主导选择）。
    "m5_bw03|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.m5_enable=True habitat_baselines.rl.policy.m5_bias_weight=0.30"
    # bias_weight=1.20：强引力（激进，soft 源可主导最近前沿，验证上限）。
    "m5_bw12|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.m5_enable=True habitat_baselines.rl.policy.m5_bias_weight=1.20"

    # ---- 批2 (sigma 扩散 A/B；radius 跟随 3sigma 配平，维持完整核不截断) ----
    # sigma=2.5m / radius=7.5m：广域引力（soft 源热区更大，"顺路验证"范围更广）。
    "m5_sg25|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.m5_enable=True habitat_baselines.rl.policy.m5_gaussian_sigma=2.50 habitat_baselines.rl.policy.m5_max_influence_radius=7.50"
)

# =====================================================================
# 旧网格（注释保留）：V7 模块一 几何准入门控（M2/M4/M8/M3）
# - 档：gap90_opp0（opp 分离对照） / g_m8 / g_m4 / g_m2_m7 / g_m3 / g_m8_m3
# - M2 近距硬拒会扼杀近场 re-detect merge（回退误删风险）=> 必配 M7 保真；
#   M4 单独开 = 可疑透传记忆（2 帧更快清理），与 M7 配套更稳。
# =====================================================================
# EXPERIMENTS=(
#     # ---- opp A/B：纯前沿 gap90 ----
#     "gap90_opp0|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.enable_scan=True habitat_baselines.rl.policy.frontier_trigger=True habitat_baselines.rl.policy.scan_min_gap=90 habitat_baselines.rl.policy.scan_opportunistic_after=0"
#
#     # M8 Box-占据一致性门控：空框/无物理支撑检测单步拦截（源头减假，最安全）。
#     "g_m8|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.geom_m8_occ_gate=True"
#     # M4 视锥边缘/远端可疑标记（透传记忆 suspicious，2 帧更快清理）。
#     "g_m4|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.geom_m4_suspicious=True"
#     # M2 近距硬拒 + M7 近场豁免（必配：M2 无 M7 会误删近场真目标）。
#     "g_m2_m7|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.geom_m2_near_reject=True habitat_baselines.rl.policy.enable_mechanism_7=True"
#     # M3 Box 内点云时空累积密度硬门槛（>=150 点 & >=2 帧 才 HARD 入库）。
#     "g_m3|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.geom_m3_density=True"
#     # M8+M3：占据支撑 + 密度累积 双源头减假。
#     "g_m8_m3|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.geom_m8_occ_gate=True habitat_baselines.rl.policy.geom_m3_density=True"
# )

# =====================================================================
# 旧网格（注释保留）：V7 模块二 target-memory（M6/M7/M9）
# - 档：world_v7_base / v7_m7 / v7_m6 / v7_m6_m7 / v7_m9 / v7_m6_m7_m9
# =====================================================================
# EXPERIMENTS=(
#     "world_v7_base|habitat_baselines.rl.policy.fusion_style=world"
#     "v7_m7|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.enable_mechanism_7=True"
#     "v7_m6|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.enable_mechanism_6=True"
#     "v7_m6_m7|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.enable_mechanism_6=True habitat_baselines.rl.policy.enable_mechanism_7=True"
#     "v7_m9|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.enable_mechanism_9=True"
#     "v7_m6_m7_m9|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.enable_mechanism_6=True habitat_baselines.rl.policy.enable_mechanism_7=True habitat_baselines.rl.policy.enable_mechanism_9=True"
# )

# ---- 其他备用档（注释保留） ----
# M6 激进档：soft_only=False（全目标可擦除）。仅当 m6 保守档无 fn 反弹时再测。
# "v7_m6_all|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.enable_mechanism_6=True habitat_baselines.rl.policy.free_erasure_soft_only=False"
# M4 单独 + M7 保真配套（若 g_m4 暴露误删再上）。
# "g_m4_m7|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.geom_m4_suspicious=True habitat_baselines.rl.policy.enable_mechanism_7=True"
# M5 组合（批4，待 m5 独立档判读后再定）：M5+M3（soft 源协同）或 M5+M8。
# "m5_g_m3|habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.m5_enable=True habitat_baselines.rl.policy.geom_m3_density=True"

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
