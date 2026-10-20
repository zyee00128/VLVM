#!/usr/bin/env bash
# ==============================================================================
# Phase G / SP1: validate whether semantic-field S is useful.
#
# - VLVM V1, Presence Gate absent/disabled, raw sigma_tar=0.60
# - GD proposal source disabled
# - existing multiplicative S-penalty enabled to evaluate c'=c*w_S and record raw S/c/c'
# - geometry, world fusion, value map and memory lifecycle unchanged
#
# Ten-scene panel selected from the previous 20-scene VLVM full evaluation:
#   - four Phase-D calibration anchors;
#   - three low-SR scenes (13.1%, 16.2%, 19.2%);
#   - three high-SR scenes (66.7%, 65.7%, 60.6%).
# This deliberately spans both tails instead of selecting only easy scenes.
#
# All run artifacts are forced under VLVM/outputs/phase_g_sp1_<timestamp>/:
#   launcher.log, navigation.log, resolved_config.yaml,
#   detections/s4_sp1_detections.jsonl, sp1_analysis.log, sp1_summary.json.
#
# Usage:
#   PHASE=P1 bash scripts/ft_nav.sh        # Phase-G P1-B 3-scene grid
#   DRY_RUN=1 bash scripts/ft_nav.sh       # preflight and print only
#   bash scripts/ft_nav.sh                 # full ten-scene SP1
#   LIMIT=10 bash scripts/ft_nav.sh        # smoke run (first 10 episodes total)
#   AUTO_START_SERVERS=0 bash scripts/ft_nav.sh  # require pre-started services
# ==============================================================================
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

# Keep the completed SP1 launcher reproducible while exposing the current P1
# entry point from the experiment script requested by the Phase-G workflow.
if [ "${PHASE:-SP1}" = "P1" ]; then
    exec bash "$HERE/phase_g_p1.sh" "$@"
fi

cd "$PROJECT_ROOT"

die() {
    echo ">>> [error] $*" >&2
    exit 1
}

port_open() {
    timeout 2 bash -c ":</dev/tcp/127.0.0.1/$1" 2>/dev/null
}

running_tsp3d_checkpoint() {
    local pid value
    for pid in $(pgrep -f 'python.*-m vlfm\.vlm\.tsp3d' 2>/dev/null || true); do
        [ -r "/proc/$pid/environ" ] || continue
        value="$(tr '\0' '\n' <"/proc/$pid/environ" \
            | sed -n 's/^TSP3D_CHECKPOINT=//p' | tail -1)"
        if [ -n "$value" ]; then
            realpath -e "$value" 2>/dev/null || printf '%s' "$value"
            return 0
        fi
    done
    return 1
}

# ---- Conda environment ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlvm}"
CONDA_PROFILE=""
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
for candidate in "$CONDA_BASE" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /root/miniconda3; do
    if [ -n "$candidate" ] && [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_PROFILE="$candidate/etc/profile.d/conda.sh"
        break
    fi
done
if [ -n "$CONDA_PROFILE" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_PROFILE"
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || die "无法激活 Conda 环境: $CONDA_ENV_NAME"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
[ -x "$PYTHON_BIN" ] || die "Python 不可执行: $PYTHON_BIN"

# ---- Runtime environment ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EGL_PLATFORM=surfaceless
export FORCE_GLX_USE_EGL=1
export MAGNUM_LOG=quiet
export MAGNUM_GPU_VALIDATION=OFF
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
unset DISPLAY

# ---- Fixed Phase-G inputs ----
VLVM_V1_CKPT="${VLVM_V1_CKPT:-$PROJECT_ROOT/data/vlvm_v1/vlvm_v1.pth}"
DATA_PATH="${DATA_PATH:-data/datasets/objectnav/hm3d/v1/val/val.json.gz}"
OUTPUTS_ROOT="$PROJECT_ROOT/outputs"
RUN_DIR="${RUN_DIR:-$OUTPUTS_ROOT/phase_g_sp1_$(date +%Y-%m-%d_%H%M%S)}"
RUN_DIR_ABS="$(realpath -m "$RUN_DIR")"
case "$RUN_DIR_ABS" in
    "$OUTPUTS_ROOT"/*) ;;
    *) die "RUN_DIR 必须位于 $OUTPUTS_ROOT 内: $RUN_DIR_ABS" ;;
esac

DET_LOG_DIR="$RUN_DIR_ABS/detections"
DET_JSONL="$DET_LOG_DIR/s4_sp1_detections.jsonl"
NAV_LOG="$RUN_DIR_ABS/navigation.log"
ANALYSIS_LOG="$RUN_DIR_ABS/sp1_analysis.log"
SUMMARY_JSON="$RUN_DIR_ABS/sp1_summary.json"
RESOLVED_CONFIG="$RUN_DIR_ABS/resolved_config.yaml"
LAUNCHER_LOG="$RUN_DIR_ABS/launcher.log"

ANCHOR_SCENES="5cdEh9F2hJL TEEsavR23oF mv2HUxq3B53 wcojb4TFT35"
LOW_SR_SCENES="mL8ThkuaVTM cvZr5TUy5C5 qyAac8rV8Zk"
HIGH_SR_SCENES="ziup5kvtCCR QaLdnwvtxbs svBbv1Pavdk"
SCENE_LIST="${SCENES:-$ANCHOR_SCENES $LOW_SR_SCENES $HIGH_SR_SCENES}"
SCENE_COUNT="$(wc -w <<<"$SCENE_LIST" | tr -d ' ')"
[ "$SCENE_COUNT" -eq 10 ] || die "SP1 要求 10 个场景，当前为 $SCENE_COUNT 个"
SCENES_CSV="$(tr ' ' '\n' <<<"$SCENE_LIST" | sed '/^$/d' | paste -sd, -)"
SCENES_OVERRIDE="[$SCENES_CSV]"

LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-}"
AUTO_START_SERVERS="${AUTO_START_SERVERS:-1}"

# ---- Read-only preflight ----
[ -s "$VLVM_V1_CKPT" ] || die "VLVM V1 checkpoint 不存在: $VLVM_V1_CKPT"
[ -s "$DATA_PATH" ] || die "HM3D ObjectNav val 不存在: $DATA_PATH"
[ -s data/pointnav_weights.pth ] || die "PointNav 权重不存在"
[ -s data/dummy_policy.pth ] || die "dummy policy 不存在"

"$PYTHON_BIN" - "$VLVM_V1_CKPT" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
    state = checkpoint["model"]
elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
    state = checkpoint["state_dict"]
else:
    state = checkpoint
keys = [str(key)[7:] if str(key).startswith("module.") else str(key) for key in state]
if any(key.startswith("presence_gate.") for key in keys):
    raise SystemExit("Phase G requires Presence Gate disabled, but checkpoint contains presence_gate.*")
cfg = checkpoint.get("ft_config", {}) if isinstance(checkpoint, dict) else {}
if bool(cfg.get("enable_gate", False)):
    raise SystemExit("Phase G requires ft_config.enable_gate=False")
PY

for scene in $SCENE_LIST; do
    [ -s "data/datasets/objectnav/hm3d/v1/val/content/$scene.json.gz" ] \
        || die "缺少场景 episode: $scene"
    find -L data/scene_datasets/hm3d_v0.2/val -mindepth 1 -maxdepth 1 \
        -type d -name "*-$scene" -print -quit | grep -q . \
        || die "缺少 HM3D 场景资产: $scene"
done

"$PYTHON_BIN" -m py_compile \
    ft_tsp3d/eval_tools/s_evidence.py \
    vlfm/tsp3d_models/fussion_utils/s_penalty.py \
    vlfm/policy/tsp3d_objectnav_policy.py \
    vlfm/utils/episode_stats_logger.py \
    vlfm/utils/vlvm_trainer.py

ARGS=(
    "habitat.dataset.data_path=$DATA_PATH"
    "habitat.dataset.split=val"
    "habitat.dataset.content_scenes=$SCENES_OVERRIDE"
    "habitat_baselines.eval.split=val"
    "habitat_baselines.num_environments=1"
    "habitat_baselines.rl.policy.name=HabitatITMPolicyV1"
    "habitat_baselines.rl.policy.sigma_tar=0.60"
    "habitat_baselines.rl.policy.sigma_prop=0.60"
    "habitat_baselines.rl.policy.enable_s_penalty=True"
    "habitat_baselines.rl.policy.s_penalty_thresh=0.15"
    "habitat_baselines.rl.policy.s_penalty_uncovered_w=1.0"
    "habitat_baselines.rl.policy.s_penalty_use_surface=True"
    "habitat_baselines.rl.policy.goal_use_surface=True"
    "habitat_baselines.rl.policy.gdp_enable=False"
    "habitat_baselines.rl.policy.enable_occ_consistency=True"
    "habitat_baselines.rl.policy.det_log_dir=$DET_LOG_DIR"
    "habitat_baselines.video_dir=$RUN_DIR_ABS/video"
    "habitat_baselines.tensorboard_dir=$RUN_DIR_ABS/tb"
    "habitat_baselines.checkpoint_folder=$RUN_DIR_ABS/checkpoints"
    "hydra.run.dir=$RUN_DIR_ABS/hydra"
)
if [ -n "$LIMIT" ]; then
    ARGS+=("habitat_baselines.test_episode_count=$LIMIT")
fi

echo "================================================="
echo ">>> Phase G / SP1: validate semantic-field S"
echo ">>> checkpoint : $VLVM_V1_CKPT"
echo ">>> sigma_tar  : 0.60 (fixed)"
echo ">>> sigma_prop : 0.60 (SP1 legacy candidate pool)"
echo ">>> S config   : t_S=0.15 floor=0.3 radius=0.5 uncovered_w=1.0"
echo ">>> disabled   : Presence Gate / GD proposer"
echo ">>> scenes     : $SCENE_COUNT"
for scene in $SCENE_LIST; do echo "      - $scene"; done
echo ">>> run dir    : $RUN_DIR_ABS"
[ -n "$LIMIT" ] && echo ">>> episode cap: $LIMIT total"
[ -n "$DRY_RUN" ] && echo ">>> DRY_RUN    : no services, navigation, or analysis will run"
echo "================================================="

if [ -n "$DRY_RUN" ]; then
    printf '>>> navigation command:'
    printf ' %q' "$PYTHON_BIN" -u -m vlfm.run "${ARGS[@]}"
    printf '\n'
    printf '>>> analysis command: %q -u %q --jsonl %q --sigma-tar 0.60 --out %q\n' \
        "$PYTHON_BIN" "ft_tsp3d/eval_tools/s_evidence.py" "$DET_LOG_DIR" "$SUMMARY_JSON"
    exit 0
fi

mkdir -p "$DET_LOG_DIR"
[ ! -e "$DET_JSONL" ] || die "检测日志已存在，拒绝混入旧记录: $DET_JSONL"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

# Start the standard service stack only when both required services are absent.
# GD is launched by the shared launcher but remains disabled in this SP1 run.
missing_required=0
for port in 12186 12182; do
    port_open "$port" || missing_required=1
done
if [ "$missing_required" -eq 1 ]; then
    [ "$AUTO_START_SERVERS" = "1" ] \
        || die "TSP3D/BLIP2-ITM 服务未就绪；先运行 scripts/launch_vlm_servers.sh"
    for port in 12186 12182 12181; do
        port_open "$port" && die "检测到部分旧服务占用端口 $port；先清理旧服务"
    done
    echo ">>> 启动标准 VLM 服务栈（TSP3D 固定 VLVM V1）"
    TSP3D_CHECKPOINT="$VLVM_V1_CKPT" VLVM_PYTHON="$PYTHON_BIN" \
        bash "$HERE/launch_vlm_servers.sh"
fi
port_open 12186 || die "TSP3D 服务未就绪: 12186"
port_open 12182 || die "BLIP2-ITM 服务未就绪: 12182"
EXPECTED_CKPT="$(realpath -e "$VLVM_V1_CKPT")"
if ACTIVE_CKPT="$(running_tsp3d_checkpoint)"; then
    [ "$ACTIVE_CKPT" = "$EXPECTED_CKPT" ] \
        || die "运行中的 TSP3D checkpoint 不匹配: $ACTIVE_CKPT"
else
    echo ">>> [warn] 无法从进程环境读取 TSP3D checkpoint；仅确认端口可用"
fi

echo ">>> 解析并保存最终 Hydra 配置"
"$PYTHON_BIN" -m vlfm.run --cfg job "${ARGS[@]}" >"$RESOLVED_CONFIG"
grep -q "det_log_dir: $DET_LOG_DIR" "$RESOLVED_CONFIG" \
    || die "resolved config 未命中 SP1 det_log_dir"

echo ">>> 开始十场景 SP1 导航采集"
NAV_START="$(date +%s)"
"$PYTHON_BIN" -u -m vlfm.run "${ARGS[@]}" 2>&1 | tee "$NAV_LOG"
NAV_END="$(date +%s)"
[ -s "$DET_JSONL" ] || die "导航结束但未生成检测日志: $DET_JSONL"

echo ">>> 开始 SP1 离线证据统计"
"$PYTHON_BIN" -u ft_tsp3d/eval_tools/s_evidence.py \
    --jsonl "$DET_LOG_DIR" \
    --sigma-tar 0.60 \
    --out "$SUMMARY_JSON" \
    2>&1 | tee "$ANALYSIS_LOG"
[ -s "$SUMMARY_JSON" ] || die "SP1 summary 未生成: $SUMMARY_JSON"

ELAPSED=$((NAV_END - NAV_START))
echo "================================================="
echo ">>> Phase G / SP1 完成"
echo ">>> navigation elapsed: $(printf '%d:%02d:%02d' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))"
echo ">>> run dir           : $RUN_DIR_ABS"
echo ">>> detections        : $DET_JSONL"
echo ">>> summary           : $SUMMARY_JSON"
echo "================================================="
