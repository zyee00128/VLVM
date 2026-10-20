#!/usr/bin/env bash
# Phase G / P1:
# - grid: five-point t_S search on the three-scene panel;
# - formal: fixed winning t_S on the ten-scene panel, with a 0.45 proposal
#   supply floor so sigma_prop=0.45/0.50/0.55/0.60 can be inspected without
#   changing the locked 0.60 policy-admission line.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
cd "$PROJECT_ROOT"

die() { echo ">>> [error] $*" >&2; exit 1; }
port_open() { timeout 2 bash -c ":</dev/tcp/127.0.0.1/$1" 2>/dev/null; }

tmux_session_for_pid() {
    local pid="$1" process_tty
    process_tty="$(ps -p "$pid" -o tty= 2>/dev/null | awk '{$1=$1; print}')"
    [ -n "$process_tty" ] && [ "$process_tty" != "?" ] || return 1
    tmux list-panes -a -F '#{session_name}|#{pane_tty}' 2>/dev/null \
        | awk -F '|' -v expected="/dev/$process_tty" '$2 == expected {print $1; exit}'
}

discover_vlm_server_session() {
    local pattern pid session expected="" pane_count
    for pattern in \
        'python.*-m vlfm\.vlm\.tsp3d' \
        'python.*-m vlfm\.vlm\.blip2itm' \
        'python.*-m vlfm\.vlm\.grounding_dino'; do
        pid="$(pgrep -f "$pattern" 2>/dev/null | head -n 1)"
        [ -n "$pid" ] || return 1
        session="$(tmux_session_for_pid "$pid")"
        [ -n "$session" ] || return 1
        if [ -z "$expected" ]; then
            expected="$session"
        elif [ "$session" != "$expected" ]; then
            return 1
        fi
    done
    pane_count="$(tmux list-panes -t "$expected" -F '#{pane_id}' 2>/dev/null | wc -l | tr -d ' ')"
    [ "$pane_count" -eq 3 ] || return 1
    printf '%s\n' "$expected"
}

start_vlm_services() {
    local session="$1"
    VLM_SESSION_NAME="$session" TSP3D_CHECKPOINT="$VLVM_V1_CKPT" \
        VLVM_PYTHON="$PYTHON_BIN" bash "$HERE/launch_vlm_servers.sh"
    port_open 12186 || die "TSP3D 服务启动失败: 12186"
    port_open 12182 || die "BLIP2-ITM 服务启动失败: 12182"
    port_open 12181 || die "GroundingDINO 服务启动失败: 12181"
}

restart_vlm_services() {
    local session="$1" attempt busy
    tmux has-session -t "$session" 2>/dev/null \
        || die "无法刷新模型服务；tmux 会话不存在: $session"
    echo ">>> RESTART model services: $session"
    tmux kill-session -t "$session"
    for attempt in $(seq 1 60); do
        busy=0
        for port in 12186 12182 12181; do
            port_open "$port" && busy=1
        done
        [ "$busy" -eq 0 ] && break
        sleep 1
    done
    [ "$busy" -eq 0 ] || die "模型服务停止后端口未释放"
    start_vlm_services "$session"
    echo ">>> RESTART complete: $session"
}

running_tsp3d_checkpoint() {
    local pid value
    for pid in $(pgrep -f 'python.*-m vlfm\.vlm\.tsp3d' 2>/dev/null || true); do
        [ -r "/proc/$pid/environ" ] || continue
        value="$(tr '\0' '\n' <"/proc/$pid/environ" | sed -n 's/^TSP3D_CHECKPOINT=//p' | tail -1)"
        if [ -n "$value" ]; then
            realpath -e "$value" 2>/dev/null || printf '%s' "$value"
            return 0
        fi
    done
    return 1
}

# Runtime and Python environment.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlvm}"
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
CONDA_PROFILE=""
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

VLVM_V1_CKPT="${VLVM_V1_CKPT:-$PROJECT_ROOT/data/vlvm_v1/vlvm_v1.pth}"
DATA_PATH="${DATA_PATH:-data/datasets/objectnav/hm3d/v1/val/val.json.gz}"
OUTPUTS_ROOT="$PROJECT_ROOT/outputs"
P1_MODE="${P1_MODE:-grid}"
case "$P1_MODE" in
    grid|formal) ;;
    *) die "P1_MODE 只能是 grid 或 formal: $P1_MODE" ;;
esac
if [ "$P1_MODE" = "formal" ]; then
    RUN_PREFIX="phase_g_p1_10scene"
    SIGMA_PROP="${SIGMA_PROP:-0.45}"
else
    RUN_PREFIX="phase_g_p1"
    SIGMA_PROP="${SIGMA_PROP:-0.60}"
fi
SIGMA_TAR="0.60"
RUN_ROOT="${RUN_ROOT:-$OUTPUTS_ROOT/${RUN_PREFIX}_$(date +%Y-%m-%d_%H%M%S)}"
RUN_ROOT_ABS="$(realpath -m "$RUN_ROOT")"
case "$RUN_ROOT_ABS" in
    "$OUTPUTS_ROOT"/*) ;;
    *) die "RUN_ROOT 必须位于 $OUTPUTS_ROOT 内: $RUN_ROOT_ABS" ;;
esac

# grid: calibration anchor + lowest-SR + highest-SR panel member.
# formal: the locked ten-scene panel used by P0 and the historical baselines.
if [ "$P1_MODE" = "formal" ]; then
    DEFAULT_SCENES="5cdEh9F2hJL TEEsavR23oF mv2HUxq3B53 wcojb4TFT35 mL8ThkuaVTM cvZr5TUy5C5 qyAac8rV8Zk ziup5kvtCCR QaLdnwvtxbs svBbv1Pavdk"
    EXPECTED_SCENE_COUNT=10
else
    DEFAULT_SCENES="5cdEh9F2hJL mL8ThkuaVTM QaLdnwvtxbs"
    EXPECTED_SCENE_COUNT=3
fi
SCENE_LIST="${SCENES:-$DEFAULT_SCENES}"
SCENE_COUNT="$(wc -w <<<"$SCENE_LIST" | tr -d ' ')"
[ "$SCENE_COUNT" -eq "$EXPECTED_SCENE_COUNT" ] \
    || die "P1 $P1_MODE 要求 $EXPECTED_SCENE_COUNT 个场景，当前为 $SCENE_COUNT 个"
SCENES_CSV="$(tr ' ' '\n' <<<"$SCENE_LIST" | sed '/^$/d' | paste -sd, -)"
SCENES_OVERRIDE="[$SCENES_CSV]"

# P0 offline replay supplies five representative points from high-recall through
# strong-penalty regimes.  s_min=0.1 and radius=0.5m remain fixed in code and
# are intentionally not Hydra parameters.
# name|t_S
if [ "$P1_MODE" = "formal" ]; then
    P1_CONFIGS=("p1_ts01400_sp045|0.1400")
else
    P1_CONFIGS=(
        "p1_ts01125|0.1125"
        "p1_ts01200|0.1200"
        "p1_ts01275|0.1275"
        "p1_ts01400|0.1400"
        "p1_ts01500|0.1500"
    )
fi

LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-}"
AUTO_START_SERVERS="${AUTO_START_SERVERS:-1}"
RESUME="${RESUME:-}"
if [ "$P1_MODE" = "formal" ]; then
    DEFAULT_RESTART_SERVERS=0
else
    DEFAULT_RESTART_SERVERS=1
fi
RESTART_SERVERS_BETWEEN_CONFIGS="${RESTART_SERVERS_BETWEEN_CONFIGS:-$DEFAULT_RESTART_SERVERS}"
VLM_SERVER_SESSION="${VLM_SERVER_SESSION:-phase_g_p1_servers}"
[ "$RESTART_SERVERS_BETWEEN_CONFIGS" = "0" ] \
    || [ "$RESTART_SERVERS_BETWEEN_CONFIGS" = "1" ] \
    || die "RESTART_SERVERS_BETWEEN_CONFIGS 只能是 0 或 1"

# Read-only preflight.
[ -s "$VLVM_V1_CKPT" ] || die "VLVM V1 checkpoint 不存在: $VLVM_V1_CKPT"
[ -s "$DATA_PATH" ] || die "HM3D ObjectNav val 不存在: $DATA_PATH"
[ -s data/pointnav_weights.pth ] || die "PointNav 权重不存在"
[ -s data/dummy_policy.pth ] || die "dummy policy 不存在"
if [ -n "$RESUME" ]; then
    [ -d "$RUN_ROOT_ABS" ] || die "续跑目录不存在: $RUN_ROOT_ABS"
    [ -f "$RUN_ROOT_ABS/manifest.tsv" ] || die "续跑目录缺少 manifest.tsv: $RUN_ROOT_ABS"
else
    [ ! -e "$RUN_ROOT_ABS" ] || die "输出根目录已存在，拒绝覆盖: $RUN_ROOT_ABS"
fi
for scene in $SCENE_LIST; do
    [ -s "data/datasets/objectnav/hm3d/v1/val/content/$scene.json.gz" ] || die "缺少 episode: $scene"
    find -L data/scene_datasets/hm3d_v0.2/val -mindepth 1 -maxdepth 1 \
        -type d -name "*-$scene" -print -quit | grep -q . || die "缺少场景资产: $scene"
done

"$PYTHON_BIN" -m py_compile \
    ft_tsp3d/eval_tools/s_evidence.py \
    ft_tsp3d/eval_tools/p1_nav_compare.py \
    vlfm/s_penalty_constants.py \
    vlfm/tsp3d_models/fussion_utils/s_penalty.py \
    vlfm/tsp3d_models/fussion_utils/target_geometric_gating.py \
    vlfm/tsp3d_models/fussion_utils/target_memory_manager.py \
    vlfm/policy/tsp3d_objectnav_policy.py \
    vlfm/utils/episode_stats_logger.py \
    vlfm/utils/vlvm_stats_logger.py \
    vlfm/utils/vlvm_trainer.py

print_plan() {
    echo "================================================="
    echo ">>> Phase G / P1 mode: $P1_MODE"
    echo ">>> checkpoint : $VLVM_V1_CKPT"
    echo ">>> scenes     : $SCENE_LIST"
    echo ">>> sigma_prop : $SIGMA_PROP (TSP3D service proposal floor)"
    echo ">>> sigma_tar  : $SIGMA_TAR (locked policy admission line)"
    echo ">>> s_min      : 0.10 (fixed in code)"
    echo ">>> S radius   : 0.50 m (fixed in code)"
    echo ">>> output root: $RUN_ROOT_ABS"
    echo ">>> configs    : ${#P1_CONFIGS[@]} (sequential)"
    echo ">>> svc refresh: $RESTART_SERVERS_BETWEEN_CONFIGS (between t_S configs)"
    [ -n "$RESUME" ] && echo ">>> RESUME     : skip COMPLETED configs; archive and rerun the interrupted config"
    [ -n "$LIMIT" ] && echo ">>> episode cap: $LIMIT per config"
    [ -n "$DRY_RUN" ] && echo ">>> DRY_RUN    : no directories, services or navigation"
    echo "================================================="
}
print_plan

build_args() {
    local run_dir="$1" det_dir="$2" t_s="$3"
    ARGS=(
        "habitat.dataset.data_path=$DATA_PATH"
        "habitat.dataset.split=val"
        "habitat.dataset.content_scenes=$SCENES_OVERRIDE"
        "habitat_baselines.eval.split=val"
        "habitat_baselines.num_environments=1"
        "habitat_baselines.rl.policy.name=HabitatITMPolicyV1"
        "habitat_baselines.rl.policy.sigma_tar=$SIGMA_TAR"
        "habitat_baselines.rl.policy.sigma_prop=$SIGMA_PROP"
        "habitat_baselines.rl.policy.enable_s_penalty=True"
        "habitat_baselines.rl.policy.s_penalty_thresh=$t_s"
        "habitat_baselines.rl.policy.s_penalty_uncovered_w=1.0"
        "habitat_baselines.rl.policy.s_penalty_use_surface=True"
        "habitat_baselines.rl.policy.goal_use_surface=True"
        "habitat_baselines.rl.policy.gdp_enable=False"
        "habitat_baselines.rl.policy.enable_occ_consistency=True"
        "habitat_baselines.rl.policy.det_log_dir=$det_dir"
        "habitat_baselines.video_dir=$run_dir/video"
        "habitat_baselines.tensorboard_dir=$run_dir/tb"
        "habitat_baselines.checkpoint_folder=$run_dir/checkpoints"
        "hydra.run.dir=$run_dir/hydra"
    )
    if [ -n "$LIMIT" ]; then
        ARGS+=("habitat_baselines.test_episode_count=$LIMIT")
    fi
    return 0
}

if [ -n "$DRY_RUN" ]; then
    for spec in "${P1_CONFIGS[@]}"; do
        IFS='|' read -r name t_s <<<"$spec"
        run_dir="$RUN_ROOT_ABS/$name"
        if [ -n "$RESUME" ] && [ -f "$run_dir/COMPLETED" ]; then
            echo ">>> SKIP $name: COMPLETED"
            continue
        fi
        if [ -n "$RESUME" ] && [ -e "$run_dir" ]; then
            echo ">>> ARCHIVE $name: existing incomplete directory"
        fi
        build_args "$run_dir" "$run_dir/detections" "$t_s"
        printf '>>> %s:' "$name"
        printf ' %q' "$PYTHON_BIN" -u -m vlfm.run "${ARGS[@]}"
        printf '\n'
    done
    exit 0
fi

# Services are checked once before the immutable run script starts its grid.
missing_required=0
for port in 12186 12182; do port_open "$port" || missing_required=1; done
if [ "$missing_required" -eq 1 ]; then
    [ "$AUTO_START_SERVERS" = "1" ] || die "TSP3D/BLIP2-ITM 服务未就绪"
    for port in 12186 12182 12181; do
        port_open "$port" && die "检测到部分旧服务占用端口 $port；先清理旧服务"
    done
    start_vlm_services "$VLM_SERVER_SESSION"
fi
port_open 12186 || die "TSP3D 服务未就绪: 12186"
port_open 12182 || die "BLIP2-ITM 服务未就绪: 12182"
EXPECTED_CKPT="$(realpath -e "$VLVM_V1_CKPT")"
if ACTIVE_CKPT="$(running_tsp3d_checkpoint)"; then
    [ "$ACTIVE_CKPT" = "$EXPECTED_CKPT" ] || die "TSP3D checkpoint 不匹配: $ACTIVE_CKPT"
else
    die "无法确认运行中的 TSP3D checkpoint"
fi
ACTIVE_VLM_SERVER_SESSION=""
if [ "$RESTART_SERVERS_BETWEEN_CONFIGS" = "1" ]; then
    ACTIVE_VLM_SERVER_SESSION="$(discover_vlm_server_session)" \
        || die "无法确认三项模型服务属于同一 tmux 会话；不能安全地按档刷新"
    echo ">>> managed model-service session: $ACTIVE_VLM_SERVER_SESSION"
fi

mkdir -p "$RUN_ROOT_ABS"
MASTER_LOG="$RUN_ROOT_ABS/launcher.log"
exec > >(tee -a "$MASTER_LOG") 2>&1
print_plan
GPU_MEMORY_LOG="$RUN_ROOT_ABS/gpu_memory.tsv"
if [ ! -e "$GPU_MEMORY_LOG" ]; then
    printf 'timestamp\tevent\tconfig\tgpu_used_mib\tgpu_free_mib\tprocesses\n' >"$GPU_MEMORY_LOG"
fi
log_gpu_memory() {
    local event="$1" config="$2" gpu_stats process_stats
    gpu_stats="$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
    process_stats="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | paste -sd ';' -)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" "$event" "$config" \
        "${gpu_stats%%,*}" "${gpu_stats#*,}" "${process_stats//$'\t'/ }" \
        >>"$GPU_MEMORY_LOG"
}
if [ -n "$RESUME" ]; then
    [ "$(head -n 1 "$RUN_ROOT_ABS/manifest.tsv")" = $'name\tt_S\tstatus' ] \
        || die "manifest.tsv 表头不匹配，拒绝续跑"
else
    printf 'name\tt_S\tstatus\n' >"$RUN_ROOT_ABS/manifest.tsv"
fi

for config_index in "${!P1_CONFIGS[@]}"; do
    spec="${P1_CONFIGS[$config_index]}"
    IFS='|' read -r name t_s <<<"$spec"
    run_dir="$RUN_ROOT_ABS/$name"
    manifest_row="$(printf '%s\t%s\tcomplete' "$name" "$t_s")"
    if [ -n "$RESUME" ] && [ -f "$run_dir/COMPLETED" ]; then
        grep -Fqx "$manifest_row" "$RUN_ROOT_ABS/manifest.tsv" \
            || die "$name 有 COMPLETED 标记但 manifest.tsv 无对应完成记录"
        echo ">>> SKIP $name: already complete"
        continue
    fi
    if [ -n "$RESUME" ] && grep -Fq "$(printf '%s\t' "$name")" "$RUN_ROOT_ABS/manifest.tsv"; then
        die "$name 未标记 COMPLETED，但 manifest.tsv 已有记录，拒绝自动续跑"
    fi
    if [ -n "$RESUME" ] && [ -e "$run_dir" ]; then
        interrupted_stamp="$(date +%Y-%m-%d_%H%M%S)"
        backup_dir="${run_dir}_interrupted_${interrupted_stamp}"
        backup_index=1
        while [ -e "$backup_dir" ]; do
            backup_dir="${run_dir}_interrupted_${interrupted_stamp}_${backup_index}"
            backup_index=$((backup_index + 1))
        done
        mv "$run_dir" "$backup_dir"
        echo ">>> ARCHIVED incomplete $name: $backup_dir"
    fi
    det_dir="$run_dir/detections"
    nav_log="$run_dir/navigation.log"
    config_log="$run_dir/resolved_config.yaml"
    analysis_log="$run_dir/p1_analysis.log"
    summary_json="$run_dir/p1_summary.json"
    mkdir -p "$det_dir"
    build_args "$run_dir" "$det_dir" "$t_s"

    echo ">>> START $name t_S=$t_s s_min=0.1 radius=0.5m"
    log_gpu_memory "before_config" "$name"
    "$PYTHON_BIN" -m vlfm.run --cfg job "${ARGS[@]}" >"$config_log"
    "$PYTHON_BIN" - "$config_log" "$t_s" "$det_dir" "$SIGMA_PROP" "$SIGMA_TAR" <<'PY'
import pathlib, re, sys

path = pathlib.Path(sys.argv[1])
expected_t_s = float(sys.argv[2])
expected_det_dir = sys.argv[3]
expected_sigma_prop = float(sys.argv[4])
expected_sigma_tar = float(sys.argv[5])
text = path.read_text(encoding="utf-8")

def scalar(key: str) -> str:
    matches = re.findall(rf"^\s*{re.escape(key)}:\s*(.*?)\s*$", text, re.MULTILINE)
    if not matches:
        raise SystemExit(f"resolved config missing key: {key}")
    return matches[-1]

if abs(float(scalar("sigma_prop")) - expected_sigma_prop) > 1e-12:
    raise SystemExit("sigma_prop mismatch")
if abs(float(scalar("sigma_tar")) - expected_sigma_tar) > 1e-12:
    raise SystemExit("sigma_tar mismatch")
if expected_sigma_prop > expected_sigma_tar:
    raise SystemExit("proposal floor must not exceed policy admission line")
if abs(float(scalar("s_penalty_thresh")) - expected_t_s) > 1e-12:
    raise SystemExit("s_penalty_thresh mismatch")
if scalar("det_log_dir") != expected_det_dir:
    raise SystemExit("det_log_dir mismatch")
for removed in ("s_penalty_floor", "s_penalty_radius_m"):
    if re.search(rf"^\s*{removed}:", text, re.MULTILINE):
        raise SystemExit(f"removed parameter is still exposed: {removed}")
print(
    f"validated resolved config: t_S={expected_t_s:g}, "
    f"sigma_prop={expected_sigma_prop:g}, sigma_tar={expected_sigma_tar:g}"
)
PY

    "$PYTHON_BIN" -u -m vlfm.run "${ARGS[@]}" 2>&1 | tee -a "$nav_log"

    "$PYTHON_BIN" - "$det_dir" "$LIMIT" "$SCENE_COUNT" "$SIGMA_PROP" "$SIGMA_TAR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
limit = sys.argv[2]
scene_count = int(sys.argv[3])
expected_sigma_prop = float(sys.argv[4])
expected_sigma_tar = float(sys.argv[5])
required = [
    "s4_sp1_detections.jsonl", "candidates.jsonl", "memory_events.jsonl",
    "nav_goals.jsonl", "episodes.jsonl",
]
for name in required:
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing/empty structured log: {path}")
episodes = [json.loads(line) for line in (root / "episodes.jsonl").open() if line.strip()]
expected = int(limit) if limit else scene_count * 99
if len(episodes) != expected:
    raise SystemExit(f"episode count mismatch: {len(episodes)} != {expected}")
candidate = json.loads(next(line for line in (root / "candidates.jsonl").open() if line.strip()))
for field in (
    "candidate_uid", "occ_passed", "s", "branch", "tp",
    "sigma_prop_floor", "sigma_tar_admit",
):
    if field not in candidate:
        raise SystemExit(f"candidate log missing field: {field}")
if abs(float(candidate["sigma_prop_floor"]) - expected_sigma_prop) > 1e-12:
    raise SystemExit("candidate sigma_prop provenance mismatch")
if abs(float(candidate["sigma_tar_admit"]) - expected_sigma_tar) > 1e-12:
    raise SystemExit("candidate sigma_tar provenance mismatch")
print(f"validated Phase-G logs: episodes={len(episodes)}")
PY

    ANALYSIS_ARGS=(
        --jsonl "$det_dir/s4_sp1_detections.jsonl"
        --sigma-tar "$SIGMA_TAR"
        --t-s "$t_s"
        --out "$summary_json"
    )
    if [ "$P1_MODE" = "formal" ]; then
        ANALYSIS_ARGS+=(--require-proposal-shadow)
    fi
    "$PYTHON_BIN" -u ft_tsp3d/eval_tools/s_evidence.py \
        "${ANALYSIS_ARGS[@]}" 2>&1 | tee "$analysis_log"
    printf '%s\t%s\tcomplete\n' "$name" "$t_s" >>"$RUN_ROOT_ABS/manifest.tsv"
    touch "$run_dir/COMPLETED"
    log_gpu_memory "after_config" "$name"
    echo ">>> COMPLETE $name"
    if [ "$RESTART_SERVERS_BETWEEN_CONFIGS" = "1" ] \
        && [ "$config_index" -lt "$((${#P1_CONFIGS[@]} - 1))" ]; then
        restart_vlm_services "$ACTIVE_VLM_SERVER_SESSION"
        log_gpu_memory "after_service_restart" "$name"
    fi
done

touch "$RUN_ROOT_ABS/ALL_COMPLETED"
if [ "$P1_MODE" = "grid" ]; then
    "$PYTHON_BIN" -u ft_tsp3d/eval_tools/p1_nav_compare.py \
        --run-root "$RUN_ROOT_ABS" 2>&1 | tee "$RUN_ROOT_ABS/p1_navigation_comparison.log"
    touch "$RUN_ROOT_ABS/COMPARISON_COMPLETED"
fi
echo ">>> Phase G / P1 $P1_MODE complete: $RUN_ROOT_ABS"
