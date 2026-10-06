#!/bin/bash
# ==============================================================================
# VLVM base 微调权重 · sigma_tar 四场景网格搜索
# ------------------------------------------------------------------------------
# 固定 base 微调权重，关闭 S-penalty 与 Grounding-DINO 辅助机制，只改变
# sigma_tar，逐档执行四个 HM3D val 场景并汇总导航与检测级指标。
#
# 初轮 68/99 集（因旧版稀疏特征融合故障而中断）用于范围初筛：
# 0.40/0.45 SR 仅 7.35%/14.71%，0.70 为 17.65%；0.50 与 0.65 均为
# 22.06%。正式四场景搜索据此缩到 [0.50, 0.65]，步长 0.05。
#
# 默认权重严格使用 E3 对应的 base_3090_ckpt_last.pth；可用 BASE_CKPT 覆盖。
# 默认四场景：5cdEh9F2hJL / TEEsavR23oF / mv2HUxq3B53 / wcojb4TFT35。
#
# 用法速览：
#   DRY_RUN=1 bash scripts/ft_sigma_search.sh
#   LIMIT=20 bash scripts/ft_sigma_search.sh
#   BASE_CKPT=data/tsp3d_ft/base_5080_ckpt_last.pth bash scripts/ft_sigma_search.sh
#   ARMS="s050 s055" bash scripts/ft_sigma_search.sh
#   SKIP_DONE=1 bash scripts/ft_sigma_search.sh
#
# 日志：
#   $OUT_DIR/ft_sigma_search_<时间戳>/eval_hm3d_<日期>_<时刻>_<sigma档>_<场景>.log
#   $OUT_DIR/ft_sigma_search_<时间戳>/summary.tsv
# ==============================================================================
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Conda 环境 ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlvm}"
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

# ---- 渲染 / GPU / HF 环境 ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EGL_PLATFORM=surfaceless
export FORCE_GLX_USE_EGL=1
export MAGNUM_LOG=quiet
export MAGNUM_GPU_VALIDATION=OFF
export HF_ENDPOINT=https://hf-mirror.com
unset DISPLAY
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# ---- 可调参数 ----
TSP3D_PORT="${TSP3D_PORT:-12186}"
BLIP2ITM_PORT="${BLIP2ITM_PORT:-12182}"
GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-12181}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs}"
RUN_DIR="${RUN_DIR:-$OUT_DIR/ft_sigma_search_$(date +%Y-%m-%d_%H%M%S)}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/data/tsp3d_ft}"
BASE_CKPT="${BASE_CKPT:-$CKPT_DIR/base_3090_ckpt_last.pth}"
DATA_PATH="${DATA_PATH:-data/datasets/objectnav/hm3d/v1/val/val.json.gz}"
GPU_TAG="${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -oE '[0-9]{4}' | head -1)}"
GPU_TAG="${GPU_TAG:-nogpu}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-}"
SKIP_DONE="${SKIP_DONE:-}"
SKIP_SERVERS="${SKIP_SERVERS:-}"
SKIP_CKPT_VERIFY="${SKIP_CKPT_VERIFY:-}"
FAIL_FAST="${FAIL_FAST:-}"
REQUIRE_ALL="${REQUIRE_ALL:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
MAX_SCENE_ATTEMPTS="${MAX_SCENE_ATTEMPTS:-2}"

# ---- sigma 网格；ARMS 可用于只跑部分档位 ----
QUEUE="${ARMS:-s050 s055 s060 s065}"

sigma_for_arm() {
    case "$1" in
        s050) printf '0.50' ;;
        s055) printf '0.55' ;;
        s060) printf '0.60' ;;
        s065) printf '0.65' ;;
        *)
            echo ">>> [error] 未知 sigma 档位: $1（允许 s050/s055/s060/s065）" >&2
            return 1 ;;
    esac
}

# ---- 四场景搜索集；SCENES 可覆盖 ----
SCENE_LIST="${SCENES:-5cdEh9F2hJL TEEsavR23oF mv2HUxq3B53 wcojb4TFT35}"

echo "================================================="
echo ">>> sigma 队列: $QUEUE"
echo ">>> 场景集   : $(echo $SCENE_LIST | wc -w) 个场景"
for _s in $SCENE_LIST; do echo "      - $_s"; done
echo ">>> base 权重 : $BASE_CKPT"
echo ">>> 实验条件 : enable_s_penalty=False + sx_conflict_gate=False + gdp_enable=False"
echo ">>> 保持开启 : BLIP2-ITM 语义场检测 + enable_occ_consistency=True"
echo ">>> 保持不变 : memory-life 与其余定稿机制"
echo ">>> 失败重试 : 每个 sigma×场景最多 $MAX_SCENE_ATTEMPTS 次，重试前重启服务"
echo ">>> 运行目录 : $RUN_DIR"
[ -n "$LIMIT" ] && echo ">>> 冒烟模式 : 每场景前 $LIMIT 集"
[ -n "$DRY_RUN" ] && echo ">>> DRY-RUN  : 只打印计划（不重启服务 / 不评测）"
echo "================================================="

port_open() { timeout 2 bash -c ":</dev/tcp/127.0.0.1/$1" 2>/dev/null; }

canonical_ckpt() {
    realpath -e -- "$1" 2>/dev/null
}

checkpoint_identity() {
    stat -Lc '%d:%i:%s:%Y' -- "$1" 2>/dev/null
}

checkpoint_sha256() {
    sha256sum -- "$1" | awk '{print $1}'
}

# 当前档位已核验的服务身份。每个场景开跑前都会再次检查，防止评测期间
# tmux 服务被外部重启或 checkpoint 文件被替换。
ACTIVE_TSP3D_SESSION=""
ACTIVE_CKPT=""
ACTIVE_CKPT_ID=""
ACTIVE_CKPT_SHA256=""

# ---- 所有 sigma 档固定使用同一份 base checkpoint ----
resolve_ckpt() {
    local resolved
    if [ ! -f "$BASE_CKPT" ]; then
        echo ">>> [warn] base checkpoint 不存在: $BASE_CKPT" >&2
        echo ">>>        用 BASE_CKPT=<路径> 显式指定" >&2
        return 1
    fi
    resolved="$(canonical_ckpt "$BASE_CKPT")" || return 1
    printf '%s' "$resolved"
}

find_single_vlm_session() {
    local sessions count
    sessions="$(tmux ls -F '#{session_name}' 2>/dev/null | grep -E '^vlm_servers_' || true)"
    count="$(printf '%s\n' "$sessions" | grep -c . || true)"
    if [ "$count" != "1" ]; then
        echo ">>> [error] 期望恰好 1 个 vlm_servers_* 会话，实际为 $count 个" >&2
        [ -n "$sessions" ] && printf '%s\n' "$sessions" >&2
        return 1
    fi
    printf '%s' "$sessions"
}

loaded_ckpt_from_session() {
    local sess="$1"
    # -J 合并 tmux 面板中的视觉换行。没有它时，长路径会在 pane 宽度处断行，
    # 导致“服务实际加载正确，但字符串核验误报不一致”。
    tmux capture-pane -p -J -S -1000 -t "${sess}:0.0" 2>/dev/null \
        | sed -nE 's/^.*\[TSP3D\] Weights loaded from (.*) \(matched [0-9]+ keys\).*$/\1/p' \
        | tail -1
}

loaded_ckpt_from_process() {
    local pid cmdline loaded
    for pid in $(pgrep -f 'python.*-m vlfm\.vlm\.tsp3d' 2>/dev/null || true); do
        [ -r "/proc/$pid/cmdline" ] || continue
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
        case "$cmdline" in
            *"-m vlfm.vlm.tsp3d --port $TSP3D_PORT"*) ;;
            *) continue ;;
        esac
        loaded="$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
            | sed -n 's/^TSP3D_CHECKPOINT=//p' | tail -1)"
        if [ -n "$loaded" ]; then
            printf '%s' "$loaded"
            return 0
        fi
    done
    return 1
}

verify_server_checkpoint() {
    local expected="$1" sess="$2" expected_id="${3:-}" expected_sha="${4:-}"
    local loaded loaded_real current_id current_sha

    expected="$(canonical_ckpt "$expected")" || {
        echo ">>> [error] 期望 checkpoint 不存在: $1"
        return 1
    }
    if [ -z "$sess" ] || ! tmux has-session -t "$sess" 2>/dev/null; then
        echo ">>> [error] TSP3D 服务会话不存在: ${sess:-<empty>}"
        return 1
    fi
    if ! port_open "$TSP3D_PORT"; then
        echo ">>> [error] TSP3D 端口 $TSP3D_PORT 未就绪"
        return 1
    fi

    # The startup line scrolls out of tmux history during a long scene.  Read
    # the live Python process environment first; keep pane parsing only as the
    # startup-time fallback.
    loaded="$(loaded_ckpt_from_process || true)"
    [ -n "$loaded" ] || loaded="$(loaded_ckpt_from_session "$sess")"
    if [ -z "$loaded" ]; then
        echo ">>> [error] 服务日志中未见完整的 '[TSP3D] Weights loaded from ...'"
        echo ">>> TSP3D pane 尾部："
        tmux capture-pane -p -J -S -80 -t "${sess}:0.0" 2>/dev/null | tail -40
        return 1
    fi
    loaded_real="$(canonical_ckpt "$loaded")" || {
        echo ">>> [error] 服务报告的 checkpoint 已不存在: $loaded"
        return 1
    }
    if [ "$loaded_real" != "$expected" ]; then
        echo ">>> [error] 服务实际加载权重与本档评测权重不一致（阻止测评）:"
        echo "      期望: $expected"
        echo "      实际: $loaded_real"
        return 1
    fi

    current_id="$(checkpoint_identity "$expected")"
    if [ -n "$expected_id" ] && [ "$current_id" != "$expected_id" ]; then
        echo ">>> [error] checkpoint 文件在服务启动后发生变化（阻止测评）: $expected"
        return 1
    fi
    if [ -n "$expected_sha" ]; then
        current_sha="$(checkpoint_sha256 "$expected")" || return 1
        if [ "$current_sha" != "$expected_sha" ]; then
            echo ">>> [error] checkpoint 内容在服务启动后发生变化（阻止测评）: $expected"
            return 1
        fi
    fi
    return 0
}

# ---- 以目标 ckpt 重启三 VLM 服务，并核验 TSP3D 真的加载了该权重 ----
# 注：会停掉所有 vlm_servers_* 会话（该前缀由 launch_vlm_servers.sh 创建）；
#     若想保留别的服务，先用 SKIP_SERVERS=1 自行管理服务。
restart_servers() {
    local ckpt="$1" s i sess ckpt_id ckpt_sha
    ckpt="$(canonical_ckpt "$ckpt")" || {
        echo ">>> [error] checkpoint 不存在: $1"
        return 1
    }
    ckpt_id="$(checkpoint_identity "$ckpt")"
    ckpt_sha="$(checkpoint_sha256 "$ckpt")" || return 1
    for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep -E '^vlm_servers_' || true); do
        echo ">>> 停止旧服务会话: $s"
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    for i in $(seq 1 15); do
        if ! port_open "$TSP3D_PORT" \
            && ! port_open "$BLIP2ITM_PORT" \
            && ! port_open "$GROUNDING_DINO_PORT"; then
            break
        fi
        sleep 2
    done
    echo ">>> 以新权重启动服务: TSP3D_CHECKPOINT=$ckpt"
    if ! TSP3D_CHECKPOINT="$ckpt" \
        TSP3D_PORT="$TSP3D_PORT" \
        BLIP2ITM_PORT="$BLIP2ITM_PORT" \
        GROUNDING_DINO_PORT="$GROUNDING_DINO_PORT" \
        bash "$PROJECT_ROOT/scripts/launch_vlm_servers.sh"; then
        echo ">>> [error] launch_vlm_servers.sh 退出码非 0（端口占用 / 启动失败）"
        return 1
    fi
    # S-penalty is disabled during sigma search, but its upstream semantic
    # field detector must remain live so the search preserves the same value
    # map observations needed by the later S-penalty stage.
    for i in $(seq 1 30); do
        if port_open "$TSP3D_PORT" && port_open "$BLIP2ITM_PORT"; then
            break
        fi
        sleep 2
    done
    if ! port_open "$TSP3D_PORT"; then
        echo ">>> [error] TSP3D 端口 $TSP3D_PORT 未就绪"
        return 1
    fi
    if ! port_open "$BLIP2ITM_PORT"; then
        echo ">>> [error] S-penalty 前置 BLIP2-ITM 端口 $BLIP2ITM_PORT 未就绪"
        return 1
    fi
    if [ -n "$SKIP_CKPT_VERIFY" ]; then
        echo ">>> [warn] SKIP_CKPT_VERIFY=1：跳过权重加载核验"
        return 0
    fi
    sess="$(find_single_vlm_session)" || return 1
    verify_server_checkpoint "$ckpt" "$sess" "$ckpt_id" "$ckpt_sha" || return 1
    ACTIVE_TSP3D_SESSION="$sess"
    ACTIVE_CKPT="$ckpt"
    ACTIVE_CKPT_ID="$ckpt_id"
    ACTIVE_CKPT_SHA256="$ckpt_sha"
    printf '%s\t%s\t%s\t%s\n' \
        "$(date '+%F %T')" "$ckpt" "$ckpt_sha" "$sess" >> "$RUN_DIR/checkpoints.tsv"
    echo ">>> 权重核验通过: $ckpt"
    echo ">>> SHA256: $ckpt_sha"
    return 0
}

adopt_existing_servers() {
    local ckpt="$1" sess ckpt_id ckpt_sha
    ckpt="$(canonical_ckpt "$ckpt")" || return 1
    sess="$(find_single_vlm_session)" || return 1
    ckpt_id="$(checkpoint_identity "$ckpt")"
    ckpt_sha="$(checkpoint_sha256 "$ckpt")" || return 1
    verify_server_checkpoint "$ckpt" "$sess" "$ckpt_id" "$ckpt_sha" || return 1
    ACTIVE_TSP3D_SESSION="$sess"
    ACTIVE_CKPT="$ckpt"
    ACTIVE_CKPT_ID="$ckpt_id"
    ACTIVE_CKPT_SHA256="$ckpt_sha"
    echo ">>> 现有服务权重核验通过: $ckpt"
}

# ---- 提取 habitat 指标行数值 ----
# 注：Success rate / Oracle 行无前缀（"Success rate: 38.75% (...) "）；Average episode *
# 行带 logging 时间戳前缀（"2026-09-20 19:38:29,989 Average episode spl: 0.1856"）。
# 因此按行内匹配、贪婪取最后冒号后的第一个 token；sr 需排除 Oracle 行。
metric_of() {  # $1=匹配正则 $2=日志 $3=排除正则（可选）
    grep -E "$1" "$2" 2>/dev/null | { if [ -n "${3:-}" ]; then grep -vE "$3"; else cat; fi; } \
        | tail -1 | sed -E 's/.*:[[:space:]]*//' | awk '{print $1}'
}

# ---- 提取检测级判读指标（评测尾部汇总块；缺行 → '-'） ----
# 返回（TAB 分隔）：det tp fp prec eps_tp entries entry_tp maxscore_mean gate_rejects
detect_metrics() {
    local log="$1" det tp fp prec eps entries etp mscore rej line
    line="$(grep -E 'total detections:' "$log" 2>/dev/null | tail -1)"
    det="$(printf '%s' "$line" | sed -E 's/.*total detections: ([0-9]+).*/\1/')"
    tp="$(printf '%s' "$line" | sed -E 's/.*tp=([0-9]+).*/\1/')"
    fp="$(printf '%s' "$line" | sed -E 's/.*fp=([0-9]+).*/\1/')"
    prec="$(grep -E '^precision:' "$log" 2>/dev/null | tail -1 | sed -E 's/^precision: ([0-9.]+)%.*/\1/')"
    line="$(grep -E 'episodes with >=1 true-positive detection:' "$log" 2>/dev/null | tail -1)"
    eps="$(printf '%s' "$line" | sed -E 's/.*detection: ([0-9]+\/[0-9]+).*/\1/')"
    line="$(grep -E 'entries: .*episodes=' "$log" 2>/dev/null | tail -1)"
    entries="$(printf '%s' "$line" | sed -E 's/^ *entries: ([0-9]+).*/\1/')"
    etp="$(printf '%s' "$line" | sed -E 's/^ *entries: [0-9]+ +tp=([0-9]+).*/\1/')"
    mscore="$(grep -E 'maxscore \(all\)' "$log" 2>/dev/null | tail -1 | sed -E 's/.*mean=([0-9.]+).*/\1/')"
    rej="$(grep -E 'gate rejects:' "$log" 2>/dev/null | tail -1 | sed -E 's/.*gate rejects: *//; s/ +/,/g')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
        "${det:--}" "${tp:--}" "${fp:--}" "${prec:--}" "${eps:--}" "${entries:--}" "${etp:--}" "${mscore:--}" "${rej:--}"
}

expected_episodes_for_scene() {
    local scene="$1"
    python - "$DATA_PATH" "$scene" "$LIMIT" <<'PY'
import gzip
import json
import sys
from pathlib import Path

path, scene, limit = sys.argv[1:]
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt", encoding="utf-8") as handle:
    payload = json.load(handle)
episodes = payload.get("episodes", [])
if not episodes:
    content_path = Path(path).parent / "content" / f"{scene}.json.gz"
    with gzip.open(content_path, "rt", encoding="utf-8") as handle:
        episodes = json.load(handle).get("episodes", [])
count = sum(scene in str(ep.get("scene_id", "")) for ep in episodes)
if limit:
    count = min(count, int(limit))
print(count)
PY
}

completed_episodes_in_log() {
    local log="$1" completed
    # Habitat prints the authoritative number of completed episodes in every
    # non-Oracle success line.  NAV summaries are diagnostic-only and can be
    # absent for episodes that finish before the normal policy summary hook.
    completed="$(
        tr '\r' '\n' < "$log" 2>/dev/null \
            | grep -E 'Success rate:.*\([0-9]+ out of [0-9]+\)' \
            | grep -v 'Oracle Success Rate:' \
            | tail -1 \
            | sed -nE 's/.*\([0-9]+ out of ([0-9]+)\).*/\1/p'
    )"
    if [[ "$completed" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$completed"
        return 0
    fi
    # Retain a fallback for logs produced by older evaluators that do not
    # print the success denominator.
    grep -c '^\[NAV\] episode summary:' "$log" 2>/dev/null || true
}

# ---- 单场景评测 ----
run_one() {
    local arm="$1" scene="$2" ckpt="$3"
    local sigma tag log donef status t0 t1 sr oracle spl soft steps expected completed
    sigma="$(sigma_for_arm "$arm")" || return 1
    expected="$(expected_episodes_for_scene "$scene")" || return 1
    if [ -z "$expected" ] || [ "$expected" -le 0 ]; then
        echo ">>> [error] 无法确定场景 $scene 的评测集数"
        return 1
    fi
    tag="${arm}_${scene}"
    log="$RUN_DIR/eval_hm3d_$(date +%Y-%m-%d)_$(date +%H%M%S)_${tag}.log"
    donef="$RUN_DIR/.done/${tag}.done"
    if [ -n "$SKIP_DONE" ] && [ -f "$donef" ]; then
        if grep -qx "episodes=$expected" "$donef"; then
            echo ">>> [skip] $tag（已完整完成 $expected 集，SKIP_DONE=1）"
            return 0
        fi
        echo ">>> [warn] $tag 的旧完成标记不含完整集数，按未完成重跑"
    fi

    local args=(
        habitat.dataset.data_path="$DATA_PATH"
        habitat.dataset.split=val
        habitat_baselines.eval.split=val
        habitat_baselines.num_environments=1
        habitat.simulator.habitat_sim_v0.gpu_device_id=0
        habitat.simulator.habitat_sim_v0.gpu_gpu=False
        habitat_baselines.rl.policy.name="HabitatITMPolicyV1"
        habitat_baselines.rl.policy.sigma_tar="$sigma"
        habitat_baselines.rl.policy.enable_s_penalty=False
        habitat_baselines.rl.policy.sx_conflict_gate=False
        habitat_baselines.rl.policy.enable_occ_consistency=True
        habitat_baselines.rl.policy.gdp_enable=False
        habitat.dataset.content_scenes="[$scene]"
    )
    if [ -n "$LIMIT" ]; then
        args+=("habitat_baselines.test_episode_count=$LIMIT")
    fi
    local ov
    for ov in $EXTRA_OVERRIDES; do args+=("$ov"); done

    if [ -n "$DRY_RUN" ]; then
        echo ">>> [dry-run] $tag  sigma_tar=$sigma  (TSP3D ckpt: $ckpt)"
        echo "    python -u -m vlfm.run ${args[*]}"
        return 0
    fi

    if [ -z "$SKIP_CKPT_VERIFY" ]; then
        if [ "$ACTIVE_CKPT" != "$ckpt" ] \
            || ! verify_server_checkpoint "$ckpt" "$ACTIVE_TSP3D_SESSION" \
                "$ACTIVE_CKPT_ID" "$ACTIVE_CKPT_SHA256"; then
            echo ">>> [error] $tag 开跑前权重一致性核验失败，未执行测评"
            return 1
        fi
    fi

    echo "-------------------------------------------------"
    echo ">>> 档位 $arm | sigma_tar=$sigma | 场景 $scene | 服务权重 $(basename "$ckpt")"
    echo ">>> 日志 $log"
    t0=$(date +%s)
    python -u -m vlfm.run "${args[@]}" 2>&1 | tee -a "$log"
    status="${PIPESTATUS[0]}"
    t1=$(date +%s)
    completed="$(completed_episodes_in_log "$log")"

    sr="$(metric_of 'Success rate:' "$log" 'Oracle')"
    oracle="$(metric_of 'Oracle Success Rate:' "$log")"
    spl="$(metric_of 'Average episode spl:' "$log")"
    soft="$(metric_of 'Average episode soft_spl:' "$log")"
    steps="$(metric_of 'Average episode steps_count:' "$log")"

    # 检测级判读行（无论成败都落盘，缺行记 '-'）
    printf '%s\t%s\t%s\t%s\n' "$arm" "$sigma" "$scene" "$(detect_metrics "$log")" >> "$RUN_DIR/detect_summary.tsv"

    if [ "$status" -eq 0 ] && [ "$completed" -eq "$expected" ]; then
        mkdir -p "$RUN_DIR/.done"
        printf 'episodes=%s\nlog=%s\n' "$expected" "$(basename "$log")" > "$donef"
        printf '%s\t%s\t%s\tOK(%s/%s)\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$arm" "$sigma" "$scene" "$completed" "$expected" "${sr:--}" "${oracle:--}" "${spl:--}" "${soft:--}" "${steps:--}" \
            "$((t1 - t0))" "$(basename "$log")" >> "$RUN_DIR/summary.tsv"
        echo ">>> 完成 $tag ($completed/$expected)  用时 $((t1 - t0))s  SR=${sr:--} Oracle=${oracle:--} spl=${spl:--}"
    else
        printf '%s\t%s\t%s\tINCOMPLETE(%s;%s/%s)\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$arm" "$sigma" "$scene" "$status" "$completed" "$expected" "${sr:--}" "${oracle:--}" "${spl:--}" "${soft:--}" "${steps:--}" \
            "$((t1 - t0))" "$(basename "$log")" >> "$RUN_DIR/summary.tsv"
        echo ">>> [fail] $tag 仅完成 $completed/$expected 集，退出码 $status（不写完成标记）"
        return 1
    fi
    return 0
}

# ---- 主流程 ----
if [ -z "$DRY_RUN" ]; then
    mkdir -p "$RUN_DIR/.done"
    if [ -z "$SKIP_DONE" ] || [ ! -f "$RUN_DIR/summary.tsv" ]; then
        printf 'arm\tsigma_tar\tscene\tstatus\tsuccess\toracle\tspl\tsoft_spl\tsteps\tsecs\tlog\n' > "$RUN_DIR/summary.tsv"
        printf 'arm\tsigma_tar\tscene\tdet\ttp\tfp\tprec\teps_tp\tentries\tentry_tp\tmaxscore_mean\tgate_rejects\n' > "$RUN_DIR/detect_summary.tsv"
        printf 'loaded_at\tcheckpoint\tsha256\ttmux_session\n' > "$RUN_DIR/checkpoints.tsv"
    else
        echo ">>> SKIP_DONE=1：保留并追加现有汇总文件"
    fi
fi

queue_failed=0
for arm in $QUEUE; do
    if ! sigma_for_arm "$arm" >/dev/null; then
        queue_failed=1
        [ -n "$FAIL_FAST" ] && break
        continue
    fi
    ckpt="$(resolve_ckpt "$arm")" || ckpt=""
    if [ -z "$ckpt" ]; then
        if [ -n "$DRY_RUN" ]; then
            ckpt="$BASE_CKPT"
            echo ">>> [dry-run] base ckpt 未找到（正式运行会 [skip]）: $ckpt"
        else
            echo ">>> [skip] $arm：未找到 base checkpoint：$BASE_CKPT"
            if [ -n "$REQUIRE_ALL" ]; then
                echo ">>> [error] REQUIRE_ALL=1：ckpt 缺失，退出"
                exit 1
            fi
            continue
        fi
    fi
    echo ""
    echo "############ 档位 $arm ############"
    echo ">>> ckpt: $ckpt"
    if [ -z "$SKIP_SERVERS" ]; then
        if [ -n "$DRY_RUN" ]; then
            echo ">>> [dry-run] 计划重启服务: kill vlm_servers_* → TSP3D_CHECKPOINT=$ckpt bash scripts/launch_vlm_servers.sh → 核验 [TSP3D] Weights loaded from"
        elif [ "$ACTIVE_CKPT" = "$ckpt" ] && [ -n "$ACTIVE_TSP3D_SESSION" ]; then
            # 全部 sigma 档使用同权重：服务无需重启，只做轻量复核。
            echo ">>> 复用已加载服务（同 ckpt，不重启）: $ckpt"
            if [ -z "$SKIP_CKPT_VERIFY" ] \
                && ! verify_server_checkpoint "$ckpt" "$ACTIVE_TSP3D_SESSION" "$ACTIVE_CKPT_ID" "$ACTIVE_CKPT_SHA256"; then
                echo ">>> [warn] 复用复核失败，改为重启服务"
                if ! restart_servers "$ckpt"; then
                    echo ">>> [error] $arm 服务重启/核验失败，跳过本档"
                    queue_failed=1
                    continue
                fi
            fi
        else
            if ! restart_servers "$ckpt"; then
                echo ">>> [error] $arm 服务重启/核验失败，跳过本档"
                queue_failed=1
                continue
            fi
        fi
    elif [ -z "$DRY_RUN" ] && [ -z "$SKIP_CKPT_VERIFY" ]; then
        if ! adopt_existing_servers "$ckpt"; then
            echo ">>> [error] SKIP_SERVERS=1，但现有服务未加载本档权重，跳过 $arm"
            queue_failed=1
            continue
        fi
    fi
    for scene in $SCENE_LIST; do
        attempt=1
        scene_ok=0
        while [ "$attempt" -le "$MAX_SCENE_ATTEMPTS" ]; do
            echo ">>> 场景尝试 $attempt/$MAX_SCENE_ATTEMPTS：$arm × $scene"
            if run_one "$arm" "$scene" "$ckpt"; then
                scene_ok=1
                break
            fi
            if [ "$attempt" -lt "$MAX_SCENE_ATTEMPTS" ] && [ -z "$DRY_RUN" ]; then
                echo ">>> $arm × $scene 未完整结束；重启服务后从该场景重新评测"
                if ! restart_servers "$ckpt"; then
                    echo ">>> [error] 重试前服务重启失败"
                    break
                fi
            fi
            attempt=$((attempt + 1))
        done
        if [ "$scene_ok" = "0" ]; then
            queue_failed=1
            if [ -n "$FAIL_FAST" ]; then
                echo ">>> FAIL_FAST=1：场景失败，停止队列"
                break 2
            fi
        fi
    done
done

echo ""
echo "================================================="
if [ -n "$DRY_RUN" ]; then
    echo "dry-run 完成（未重启服务 / 未评测）。去掉 DRY_RUN=1 即为正式运行。"
else
    echo "队列结束: $QUEUE   （GPU $GPU_TAG）"
    echo "日志目录: $RUN_DIR"
    if [ -f "$RUN_DIR/summary.tsv" ]; then
        echo ""
        echo "汇总 summary.tsv:"
        if command -v column >/dev/null 2>&1; then
            column -t -s $'\t' "$RUN_DIR/summary.tsv"
        else
            cat "$RUN_DIR/summary.tsv"
        fi
    fi
    echo ""
    echo "断点续跑: SKIP_DONE=1 RUN_DIR=$RUN_DIR bash scripts/ft_sigma_search.sh"
fi
echo "================================================="
if [ "$queue_failed" != "0" ]; then
    echo ">>> 注意：本次存在跳过的档或失败的场景（见上方 [skip] / [fail] / [error]）。"
fi
