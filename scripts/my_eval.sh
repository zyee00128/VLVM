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
OUT_DIR="$PROJECT_ROOT/outputs/VLVM-V7.2"
mkdir -p "$OUT_DIR"
# 单场景（5cdE 单场景调参基线）—— A4 对照批必须用它：
SCENES="[5cdEh9F2hJL]"
# 三场景：
# SCENES="[TEEsavR23oF, mv2HUxq3B53, wcojb4TFT35]"

# ----------------- 实验定义 -----------------
#
# 定稿（2026-09-15，代码已清理）：occ 门 + density 门 + GD 辅助机制（`gdp_enable=True` / `gdp_every_n=5`）
#   ① geometric admission module —— enable_occ_consistency + enable_density_gate
#   ② GD auxiliary mechanism     —— 独立提议源 + 两票制（gdp_* / 默认 on / every_n=5）

Stage2_1="habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.det_seed=0 habitat_baselines.rl.policy.enable_occ_consistency=True habitat_baselines.rl.policy.enable_density_gate=True"
EXPERIMENTS=()

# ============================================================================
# A3 批（2026-09-14，已跑完）—— 提议侧旋钮
# ----------------------------------------------------------------------------
# 结果：cap 43（−7.08）/ thr3 47（−3.04）/ thr5 49（−1.02）vs 对照 n5 50；mb1 待读。
# ⇒ 提议侧不列入收益项（详见 results/VLVM-V8.md §3.14 与 `### 启示`）。
# 注：档位中的 `gdp_indep_enable`（A4）/ `gdp_neg_enable`（R2）开关已随代码删除，
#     故本批整段停用，仅留作历史记录。
# ============================================================================
# GDP_N5="habitat_baselines.rl.policy.gdp_every_n=5"
# EXPERIMENTS+=(
#     "V73_a3_cap|$Stage2_1 $GDP_N5 habitat_baselines.rl.policy.gdp_caption_style=target"
#     "V73_a3_thr3|$Stage2_1 $GDP_N5 habitat_baselines.rl.policy.gdp_box_thr=0.3"
#     "V73_a3_thr5|$Stage2_1 $GDP_N5 habitat_baselines.rl.policy.gdp_box_thr=0.5"
#     "V73_a3_mb1|$Stage2_1 $GDP_N5 habitat_baselines.rl.policy.gdp_max_boxes=1"
# )

# ============================================================================
# 待排（方向 2；由用户手动开启）
# ----------------------------------------------------------------------------
# D2-5 R0 三态票型记账（量测，零 GPU）→ D2-8（GD 类别白名单，排除 couch）→
#   D2-1 + D2-2 + D2-6（单票停止前双确认 / 选点排序 / 确认窗口，同批）→ D2-4（`g_self` 量测后）
# ============================================================================
# EXPERIMENTS+=(
#     "V73_d2_8|$Stage2_1"          # 待实现开关后排档
# )
# ============================================================================
# A4 证据独立性批（2026-09-14；机制已删除，仅留历史记录）
# ----------------------------------------------------------------------------
# A4-0 = 影子计数（行为零改动）；A4-1 = 门槛生效（gdp_indep_enable=True）。
# 结论：n1 +3 SR / +5 Oracle；n5 零效应 ⇒ 已收缩，代码随本轮清理删除。
# ============================================================================
# EXPERIMENTS+=(
#     "V73_a4_n1|$Stage2_1 habitat_baselines.rl.policy.gdp_every_n=1"
#     "V73_a4_n5|$Stage2_1 habitat_baselines.rl.policy.gdp_every_n=5"
# )

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
