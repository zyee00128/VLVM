#!/bin/bash

# ==============================================================================
# VLVM 消融实验 · 基线 = 最终定稿档（数据集两档，默认 hm3d）
# ------------------------------------------------------------------------------
# 数据集（第 1 参数）：
#   hm3d（默认）= HM3D ObjectNav v1 val 全量 = 20 场景 / 2000 集
#   mp3d        = MP3D ObjectNav v1 val 全量 = 11 场景 / 2195 集
#
# 设计 = 五模块大档（默认队列，各撤销一个模块）：
#   no_memlife  memory-life：可疑标志链 + 近场滞回回退 + 锁保持（含身份豁免）
#               —— EMA 跨帧合并不消融；"可疑标志链"的行为出口（可疑条目
#                  2 帧滞回预算）随 enable_fb=False 一并失效
#   no_spen     S-penalty：软 gate + 近表面点（S 查询 / 导航终点）+ 冲突门
#   no_occ      occ geometry：occ 几何门
#   no_wfusion  world fusion：整条融合链（增量地图 / 去重 / 近场刷新 / D3）
#   no_gd       GD 辅助提议源：第二提议源 + 票型链
#   base        Base 档：定稿全机制复现（对拍基准，队列末尾自动跑）
#
# 小档（只注册设置，默认不跑；ARMS="ml_nofb" bash scripts/ablation.sh 手动选）：
#   memory-life : ml_nofb（仅回退）/ ml_nolockx（仅锁保持）
#                 / ml_nolockid（仅身份豁免）/ ml_nosusp（可疑预算拉平）
#   S-penalty   : sp_nogate（仅冲突门）/ sp_nosurf（仅 S 查询近表面点）
#                 / sp_nogoalsurf（仅导航终点近表面点）
#   world fusion: wf_nod3（仅距离自适应采样）/ wf_nodedup（仅视角去重）
#                 / wf_norefresh（仅近场刷新）/ wf_norefval（仅值层滑动）
#   GD          : gd_n1 / gd_n3（频率）/ gd_mb2（每帧框数）
#                 / gd_nowguard（弱证不锁）/ gd_capt（caption 风格 target）
#   （occ 单门，无内部小档）
#
# 用法：
#   bash scripts/ablation.sh                    # 六个档（五个大档 + Base，hm3d 全量）
#   bash scripts/ablation.sh mp3d               # 六个档（mp3d 全量）
#   ARMS="no_gd no_wfusion" bash scripts/ablation.sh
#   ARMS="ml_nofb" bash scripts/ablation.sh     # 小档（手动）
#   LIMIT=3 bash scripts/ablation.sh hm3d       # 冒烟：每档前 3 集
#   SKIP_DONE=1 bash scripts/ablation.sh        # 断点续跑（已有日志的档跳过）
# 跑前先起三个 VLM 服务：bash scripts/launch_vlm_servers.sh
# ==============================================================================

DATASET="${1:-hm3d}"

# ----------------- 项目基础路径配置 -----------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlvm}"

cd "$PROJECT_ROOT"

# 加载 Conda 环境（conda 根按 conda info --base → 常见路径顺序探测）
CONDA_PROFILE=""
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
for _c in "$CONDA_BASE" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /root/miniconda3; do
    if [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ]; then
        CONDA_PROFILE="$_c/etc/profile.d/conda.sh"
        break
    fi
done
if [ -n "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || {
        echo "警告: 激活 $CONDA_ENV_NAME 失败，回退环境名 vlfm"
        conda activate vlfm 2>/dev/null || true
    }
else
    echo "警告: 未找到 conda.sh，沿用当前 shell 解释器: $(command -v python)"
fi
echo ">>> 解释器: $(command -v python)"

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

# ----------------- 数据集两档 -----------------
case "$DATASET" in
    hm3d)
        DATA_PATH="data/datasets/objectnav/hm3d/v1/val/val.json.gz"
        N_SCENES=20; N_EPS=2000
        ;;
    mp3d)
        DATA_PATH="data/datasets/objectnav/mp3d/val/val.json.gz"
        N_SCENES=11; N_EPS=2195
        ;;
    *)
        echo "用法: bash scripts/ablation.sh [hm3d|mp3d]"
        exit 1
        ;;
esac

# ----------------- 输出目录 -----------------
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs}"
mkdir -p "$OUT_DIR"

# ----------------- 服务自检 -----------------
for port in 12186 12182 12181; do
    if ! timeout 2 bash -c ":</dev/tcp/127.0.0.1/$port" 2>/dev/null; then
        echo "警告: 端口 $port 未监听（TSP3D=12186 / BLIP2-ITM=12182 / Grounding-DINO=12181）"
        echo "      请先执行: bash scripts/launch_vlm_servers.sh"
    fi
done

# ----------------- 最终定稿基座（显式锚定，防 YAML 漂移） -----------------
# world fusion（融合器 / 输入适配）
BASE="habitat_baselines.rl.policy.fusion_style=world"
BASE="$BASE habitat_baselines.rl.policy.det_seed=0"
BASE="$BASE habitat_baselines.rl.policy.cap_style=random"
BASE="$BASE habitat_baselines.rl.policy.distance_sample=True"
BASE="$BASE habitat_baselines.rl.policy.wm_near_refresh_radius=3.0"
BASE="$BASE habitat_baselines.rl.policy.wm_near_refresh_value=True"
BASE="$BASE habitat_baselines.rl.policy.wm_min_view_disp=0.15"
BASE="$BASE habitat_baselines.rl.policy.wm_min_view_yaw=15.0"
# S-penalty
BASE="$BASE habitat_baselines.rl.policy.enable_s_penalty=True"
BASE="$BASE habitat_baselines.rl.policy.sigma_tar=0.70"
BASE="$BASE habitat_baselines.rl.policy.s_penalty_thresh=0.15"
BASE="$BASE habitat_baselines.rl.policy.s_penalty_floor=0.3"
BASE="$BASE habitat_baselines.rl.policy.s_penalty_uncovered_w=1.0"
BASE="$BASE habitat_baselines.rl.policy.s_penalty_use_surface=True"
BASE="$BASE habitat_baselines.rl.policy.goal_use_surface=True"
BASE="$BASE habitat_baselines.rl.policy.sx_conflict_gate=True"
# occ geometry
BASE="$BASE habitat_baselines.rl.policy.enable_occ_consistency=True"
# memory-life（含 EMA；lock_exempt_identity 定稿开）
BASE="$BASE habitat_baselines.rl.policy.enable_fb=True"
BASE="$BASE habitat_baselines.rl.policy.fb_hysteresis=5"
BASE="$BASE habitat_baselines.rl.policy.fb_suspicious_hysteresis=2"
BASE="$BASE habitat_baselines.rl.policy.merge_dist_thresh=0.5"
BASE="$BASE habitat_baselines.rl.policy.ema_weight_old=0.8"
BASE="$BASE habitat_baselines.rl.policy.lock_exempt=True"
BASE="$BASE habitat_baselines.rl.policy.lock_exempt_identity=True"
# GD 辅助提议源
BASE="$BASE habitat_baselines.rl.policy.gdp_enable=True"
BASE="$BASE habitat_baselines.rl.policy.gdp_every_n=5"
BASE="$BASE habitat_baselines.rl.policy.gdp_max_boxes=1"
BASE="$BASE habitat_baselines.rl.policy.gdp_weak_guard=True"
BASE="$BASE habitat_baselines.rl.policy.gdp_caption_style=vocab"

# 每档 delta（返回空串 = 未知档；大小档均为「撤销」语义）
mk_delta() {
    case "$1" in
        # ---- 大档：五模块各一（默认队列） ----
        no_memlife) echo "habitat_baselines.rl.policy.enable_fb=False habitat_baselines.rl.policy.lock_exempt=False habitat_baselines.rl.policy.lock_exempt_identity=False" ;;
        no_spen)    echo "habitat_baselines.rl.policy.enable_s_penalty=False habitat_baselines.rl.policy.s_penalty_use_surface=False habitat_baselines.rl.policy.goal_use_surface=False habitat_baselines.rl.policy.sx_conflict_gate=False" ;;
        no_occ)     echo "habitat_baselines.rl.policy.enable_occ_consistency=False" ;;
        no_wfusion) echo "habitat_baselines.rl.policy.fusion_style=none" ;;
        no_gd)      echo "habitat_baselines.rl.policy.gdp_enable=False" ;;
        # ---- 小档：模块内单机制（只注册，默认不跑） ----
        ml_nofb)      echo "habitat_baselines.rl.policy.enable_fb=False" ;;
        ml_nolockx)   echo "habitat_baselines.rl.policy.lock_exempt=False" ;;
        ml_nolockid)  echo "habitat_baselines.rl.policy.lock_exempt_identity=False" ;;
        ml_nosusp)    echo "habitat_baselines.rl.policy.fb_suspicious_hysteresis=5" ;;
        sp_nogate)    echo "habitat_baselines.rl.policy.sx_conflict_gate=False" ;;
        sp_nosurf)    echo "habitat_baselines.rl.policy.s_penalty_use_surface=False" ;;
        sp_nogoalsurf) echo "habitat_baselines.rl.policy.goal_use_surface=False" ;;
        wf_nod3)      echo "habitat_baselines.rl.policy.distance_sample=False" ;;
        wf_nodedup)   echo "habitat_baselines.rl.policy.wm_min_view_disp=0.0 habitat_baselines.rl.policy.wm_min_view_yaw=0.0" ;;
        wf_norefresh) echo "habitat_baselines.rl.policy.wm_near_refresh_radius=null" ;;
        wf_norefval)  echo "habitat_baselines.rl.policy.wm_near_refresh_value=False" ;;
        gd_n1)      echo "habitat_baselines.rl.policy.gdp_every_n=1" ;;
        gd_n3)      echo "habitat_baselines.rl.policy.gdp_every_n=3" ;;
        gd_mb2)     echo "habitat_baselines.rl.policy.gdp_max_boxes=2" ;;
        gd_nowguard) echo "habitat_baselines.rl.policy.gdp_weak_guard=False" ;;
        gd_capt)    echo "habitat_baselines.rl.policy.gdp_caption_style=target" ;;
        # ---- Base 档：定稿复现（队列末尾） ----
        base)       echo "" ;;
        *)          return 1 ;;
    esac
}

# 默认队列 = 五个大档 + Base 档（小档按需 ARMS 手动指定；选档/换序改 ARMS 即可）
ARMS="${ARMS:-no_memlife no_spen no_occ no_wfusion no_gd base}"

echo "================================================="
echo ">>> VLVM 五模块消融 · 数据集: $DATASET（$N_SCENES 场景 / $N_EPS 集，val 全量）"
echo ">>> 基准: 最终定稿档（五模块全开；锚定项见 BASE）"
echo ">>> 档位: $ARMS"
echo ">>> 输出: $OUT_DIR"
[ -n "${LIMIT:-}" ] && echo ">>> 冒烟模式: 每档 $LIMIT 集"
[ -n "${SKIP_DONE:-}" ] && echo ">>> 断点续跑: 已有日志的档将跳过"
echo "================================================="

TOTAL_START=$(date +%s)

for KEY in $ARMS; do
    LABEL="abl_$KEY"
    if ! DELTA=$(mk_delta "$KEY"); then
        echo ">>> 跳过未知档位: $KEY"
        continue
    fi
    OVERRIDE="$BASE $DELTA"

    EXIST=$(ls "$OUT_DIR"/eval_${DATASET}_*_"$LABEL".log 2>/dev/null | tail -1 || true)
    if [ -n "${SKIP_DONE:-}" ] && [ -n "$EXIST" ]; then
        echo ">>> 跳过 $LABEL（已存在日志: $(basename "$EXIST")）"
        continue
    fi

    echo "================================================="
    echo ">>> 开始实验: $LABEL"
    echo ">>> 覆盖参数: $OVERRIDE"
    echo "================================================="

    args=(
        habitat.dataset.data_path="$DATA_PATH"
        habitat.dataset.split=val
        habitat_baselines.eval.split=val
        habitat_baselines.num_environments=1
        habitat.simulator.habitat_sim_v0.gpu_device_id=0
        habitat.simulator.habitat_sim_v0.gpu_gpu=False
        habitat_baselines.rl.policy.name="HabitatITMPolicyV1"
    )
    for ov in $OVERRIDE; do
        args+=("$ov")
    done
    if [ -n "${LIMIT:-}" ]; then
        args+=("habitat_baselines.test_episode_count=$LIMIT")
    fi

    LOGFILE="$OUT_DIR/eval_${DATASET}_$(date +%Y-%m-%d)_$(date +%H%M%S)_${LABEL}.log"
    ARM_START=$(date +%s)

    python -m vlfm.run "${args[@]}" 2>&1 | tee "$LOGFILE"

    ARM_END=$(date +%s)
    ELAPSED=$((ARM_END - ARM_START))
    echo ">>> 完成: $LABEL -> $(basename "$LOGFILE")  用时 $(printf '%d:%02d:%02d' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))"
    echo ">>> 结果摘要（末行）:"
    grep -E "^(Success rate|Oracle Success Rate)" "$LOGFILE" | tail -2 | sed 's/^/    /'
done

TOTAL_END=$(date +%s)
TOTAL=$((TOTAL_END - TOTAL_START))
echo ""
echo "================================================="
echo "队列完成！总用时 $(printf '%d:%02d:%02d' $((TOTAL/3600)) $(((TOTAL%3600)/60)) $((TOTAL%60)))"
echo "日志目录: $OUT_DIR"
echo "--- 各档末行摘要 ---"
for f in "$OUT_DIR"/eval_${DATASET}_*_abl_*.log; do
    [ -e "$f" ] || continue
    printf "%-52s " "$(basename "$f")"
    grep -E "^(Success rate|Oracle Success Rate)" "$f" | tail -2 | tr '\n' ' '
    echo ""
done
echo "================================================="
# bash ./scripts/launch_vlm_servers.sh
# bash ./scripts/ablation.sh hm3d
# bash ./scripts/ablation.sh mp3d
