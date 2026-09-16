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
OUT_DIR="$PROJECT_ROOT/outputs/VLVM-V8_1"
mkdir -p "$OUT_DIR"
# 单场景（5cdE 单场景调参基线）：
SCENES="[5cdEh9F2hJL]"
# 三场景：
# SCENES="[TEEsavR23oF, mv2HUxq3B53, wcojb4TFT35]"

# ----------------- 实验定义（2026-09-16 · 双机两批 · 方向 3）-----------------
#
# 定稿基座（不变）：occ 门 + density 门 + GD 辅助机制（gdp_enable / every_n=5 / max_boxes=1）
#   单场景 5cdE 99 集；对照档 = `d2_anchor`（09-15，51/99，Oracle 56/99）——
#   本批不重跑本地 anchor（用户 09-16 定；d2_anchor 之后的代码改动均为零行为影子/量测）。
#
# 本轮全部实验（共 6 档）分两组、两机并行（每档 ~4 h；依据 §优化方向 3）：
#   用法：服务器 1  `BATCH=A bash scripts/my_eval.sh`
#         服务器 2  `BATCH=B bash scripts/my_eval.sh`
#   ※ 新增的 3-4 / 3-5 / 3-9 属"无法用前置判据确定是否使用"的机制（影子只能证明
#     作用面，判不了方向）⇒ 直接排 A/B 档；代码已实现（d2_fill_only / d2_geo_lock / d2_stop_gate）。
#
# 【A 组 · 服务器 1】
#   a1) d2_scan     scan 复跑（Oracle 通路，最高优先）：V5 `ws_scan_gap90` 曾 +4.04pp @
#                   SR 持平；当前管线（GD n5 + max_boxes=1）首测；gap=90 / opp=60 /
#                   frontier=True / turn=6 全为 YAML 默认（零代码）。
#                   判据：**Oracle ≥ +2（主）**、SR 不降、steps、逐类 fp。
#   a2) d2_stopg    停止闸 v2（3-9，§3.15.12 重设计形态）：`tsp3d` 源 + 类条件
#                   couch|toilet + n_obs<=1 才拦；二次证据（n_obs 增长 / 票型升级）即放行；
#                   窗口 30 步到期 ⇒ 释放路径 = 丢弃该条目回落 explore；配套
#                   `d2_lock_exempt`（navigate 下 nav goal 条目豁免 near_miss，防"丢条目"）。
#                   判据：`stop_blocked` 类分布命中 couch/toilet、这两类 SR↑、
#                   chair/bed 零回归、`stop_abandoned` 后是否重新找到目标。
#   a3) d2_geolock  几何独立票锁门（3-5）：`gd` 条目解锁要求 `gd_geo_ind >= 2`
#                   （位移 >= 0.5 m 或视角差 >= 30°），替代纯计数 num_obs。
#                   判据：gd 条目 tp 率 ↑、近距瞬锁组（stops_within5）↓、Oracle 不降。
#
# 【B 组 · 服务器 2】
#   b1) d2_lmo3     `lock_min_obs` 2→3（零代码，杠杆最大）：直接调"纯 gd 条目几票可锁"。
#                   判据：`[GD2]` 解锁事件数 ↓、逐类 fp ↓、Oracle 不降、SR（Δ≥2）。
#   b2) d2_merg3    `gdp_merge_dist` 0.5→0.3（零代码）：收紧 GD↔TSP3D 共位判据
#                   （`both` 是最精确群体 tp 63.2%；共位量 25 次/99 集）。
#                   判据：`both : gd` 比、逐类 fp、SR。
#   b3) d2_fill     帧级补框（3-4）：GD 仅在没有 TSP3D 目标类候选的帧新建条目，
#                   其余帧只作共位背书（`both`）；影子计数 `w_new_tsp/w_new` 随档产出。
#                   判据：`gd_gated` 量级、`both : gd` 比、逐类 fp/fn、SR。
#
# 两机同代码同配置；各档均对照 `d2_anchor`（跨机噪声带 ±1）。
# 后续批次：3-8（D2-2′，排序序已修正，待离线 what-if 过门）/ 3-2（scan 参数，依 a1）/
# 3-10（cap 触顶率）；3-9 的 release=stop 变体可作 a2 的补充档。
Stage2_1="habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.det_seed=0 habitat_baselines.rl.policy.enable_occ_consistency=True habitat_baselines.rl.policy.enable_density_gate=True"

BATCH="${BATCH:-A}"

if [ "$BATCH" = "B" ]; then
    # 服务器 2
    EXPERIMENTS=(
        "d2_lmo3|$Stage2_1 habitat_baselines.rl.policy.lock_min_obs=3"
        "d2_merg3|$Stage2_1 habitat_baselines.rl.policy.gdp_merge_dist=0.3"
        "d2_fill|$Stage2_1 habitat_baselines.rl.policy.d2_fill_only=True"
    )
else
    # 服务器 1
    EXPERIMENTS=(
        "d2_scan|$Stage2_1 habitat_baselines.rl.policy.enable_scan=True"
        "d2_stopg|$Stage2_1 habitat_baselines.rl.policy.d2_stop_gate=True habitat_baselines.rl.policy.d2_lock_exempt=True"
        "d2_geolock|$Stage2_1 habitat_baselines.rl.policy.d2_geo_lock=True"
    )
fi

echo ">>> 本机批次: BATCH=$BATCH （共 ${#EXPERIMENTS[@]} 档，顺序执行）"
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
