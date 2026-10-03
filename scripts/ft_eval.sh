#!/bin/bash
# ==============================================================================
# VLVM 微调权重 · 分批多场景导航评测（HM3D val 子集）
# ------------------------------------------------------------------------------
# 与 scripts/ft_tsp3d.sh 配套：本脚本自动完成
#   「换档 ckpt → 重启 VLM 服务（并核验权重真的加载）→ 逐场景评测 → 汇总日志」。
#
# 双机分工：
#   A 机：base → gate                MACHINE=a bash scripts/ft_eval.sh
#   B 机：base → s1 → s2 → distill   MACHINE=b bash scripts/ft_eval.sh
#
# 场景集（HM3D val；依据 Results/Results_final.md 全量 20 场景实验的失败结构挑选，
# 覆盖 高/中/低 三档难度 × fp 主导 / nv 主导 / 转换损失 三类失效模式 + 历史对比场景）：
#   5cdEh9F2hJL  SR 56.6%  高难基准；Stage1–3 全程单场景，跨实验可比
#   TEEsavR23oF  SR 38.4%  中；VLFM/VLVM 三场景对比之一
#   mv2HUxq3B53  SR 43.4%  中；fp 主导 + 转换损失大（O-S +13.2）；对比场景
#   wcojb4TFT35  SR 39.4%  中；对比场景
#   zt1RVoi7PcG  SR 24.4%  低；fp 极端（tv 24 集全 fp，O-S +21.0）
#   mL8ThkuaVTM  SR 13.1%  低；nv 极端（Oracle 4.0%，覆盖盲区）
#   qyAac8rV8Zk  SR 19.2%  低；类别-场景耦合（bed 0/17 全错）
#   Dd4bFSTQ8gi  SR 42.4%  中；toilet fp 23 + 转换损失（O-S +12.1）
#   覆盖场景集：SCENES="5cdEh9F2hJL zt1RVoi7PcG wcojb4TFT35" bash scripts/ft_eval.sh
#
# 实验条件（固定覆盖）：
#   * 关闭 GD 机制：habitat_baselines.rl.policy.gdp_enable=False（第二提议源/帧级合票停用）；
#   * 其余机制沿用 config/experiments/vlvm_objectnav_hm3d.yaml 定稿档。
#     临时加开其它机制：EXTRA_OVERRIDES="habitat_baselines.rl.policy.enable_scan=True"
#
# 评测实验矩阵（同权重 × 多组准入参数 / 原版权重对照，2026-09-30 补充）：
#   * 枚举任意档名用 ARMS；每档独立 hydra 覆盖用 OVERRIDES_<档>（空格分隔）；
#     同权重多参数变体把 CKPT_<变体> 指向同一 ckpt 文件即可（自动复用已加载服务，不重启）；
#       ARMS="base_s60 ref" \
#         OVERRIDES_base_s60="habitat_baselines.rl.policy.sigma_tar=0.60 habitat_baselines.rl.policy.s_penalty_thresh=0.12" \
#         CKPT_base_s60=data/tsp3d_ft/base_3090_ckpt_last.pth \
#         bash scripts/ft_eval.sh
#   * ref 档 = 原版权重参照（默认 data/tsp3d_models/tsp3d_scanrefer.pth，TSP3D_REF_CKPT 可覆盖）；
#   * 检测级判读（det/tp/fp/entries/maxscore 等）自动落 $RUN_DIR/detect_summary.tsv。
#
# 用法速览：
#   DRY_RUN=1 MACHINE=a bash scripts/ft_eval.sh        # 只打印计划（不重启服务/不评测）
#   LIMIT=20 MACHINE=a bash scripts/ft_eval.sh         # 冒烟：每场景前 20 集
#   SKIP_DONE=1 RUN_DIR=... bash scripts/ft_eval.sh    # 断点续跑（跳过已完成的 档_场景）
#   SKIP_SERVERS=1 bash scripts/ft_eval.sh             # 服务已手动起好，只跑评测（不换 ckpt）
#   FAIL_FAST=1 bash scripts/ft_eval.sh                # 任一场景失败即停（默认记录后继续）
#   REQUIRE_ALL=1 bash scripts/ft_eval.sh              # ckpt 缺失即退出（默认跳过该档）
#   SKIP_CKPT_VERIFY=1 bash scripts/ft_eval.sh         # 跳过「服务实际加载权重」核验
#
# 日志：
#   $OUT_DIR/ft_eval_<时间戳>/eval_hm3d_<日期>_<时刻>_<档>_<场景>.log
#   $OUT_DIR/ft_eval_<时间戳>/summary.tsv   # 每场景一行：SR / Oracle / spl / soft_spl / steps / 用时
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
RUN_DIR="${RUN_DIR:-$OUT_DIR/ft_eval_$(date +%Y-%m-%d_%H%M%S)}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/data/tsp3d_ft}"
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
GATE_TAU="${GATE_TAU:-0.24}"                    # 占位：训练后校准值（见头注释，暂不生效）

# ---- 队列（双机分工；ARMS 可覆盖） ----
if [ -n "${ARMS:-}" ]; then
    QUEUE="$ARMS"
else
    case "${MACHINE:-}" in
        a|A|1) QUEUE="base gate" ;;
        b|B|2) QUEUE="base s1 s2 distill" ;;
        *)
            echo ""
            echo "用法: MACHINE=a|b bash scripts/ft_eval.sh"
            echo "      A 机: base→gate ；B 机: base→s1→s2→distill（ARMS=\"...\" 覆盖）"
            echo ""
            exit 1 ;;
    esac
fi

# ---- 场景集（默认 = HM3D val 代表性 8 场景，见头注释；SCENES 覆盖） ----
SCENE_LIST="${SCENES:-5cdEh9F2hJL TEEsavR23oF mv2HUxq3B53 wcojb4TFT35 zt1RVoi7PcG mL8ThkuaVTM qyAac8rV8Zk Dd4bFSTQ8gi}"

echo "================================================="
echo ">>> 机器队列 : $QUEUE"
echo ">>> 场景集   : $(echo $SCENE_LIST | wc -w) 个场景"
for _s in $SCENE_LIST; do echo "      - $_s"; done
echo ">>> 服务权重 : 每档自动重启服务（TSP3D_CHECKPOINT=<档 ckpt>，附加载核验）"
echo ">>> 实验条件 : gdp_enable=False（GD 关闭）+ 定稿档其余机制"
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

# ---- 解析某档 ckpt：CKPT_<档> > <档>_<GPU>_ckpt_last.pth > 唯一 <档>_*_ckpt_last.pth ----
resolve_ckpt() {
    local arm="$1" override match n resolved ref_ckpt
    eval "override=\${CKPT_${arm}:-}"
    if [ -n "$override" ]; then
        if [ -f "$override" ]; then
            resolved="$(canonical_ckpt "$override")" || return 1
            printf '%s' "$resolved"
            return 0
        fi
        echo ">>> [warn] CKPT_${arm}=$override 不存在" >&2
        return 1
    fi
    # ref 档 = 原版权重参照（权重归零对照；参数按当档 OVERRIDES_ref / EXTRA_OVERRIDES 走）
    if [ "$arm" = "ref" ]; then
        ref_ckpt="${TSP3D_REF_CKPT:-$PROJECT_ROOT/data/tsp3d_models/tsp3d_scanrefer.pth}"
        if [ -f "$ref_ckpt" ]; then
            resolved="$(canonical_ckpt "$ref_ckpt")" || return 1
            printf '%s' "$resolved"
            return 0
        fi
        echo ">>> [warn] ref 原版权重不存在: $ref_ckpt（TSP3D_REF_CKPT 可覆盖）" >&2
        return 1
    fi
    local f="$CKPT_DIR/${arm}_${GPU_TAG}_ckpt_last.pth"
    if [ -f "$f" ]; then
        resolved="$(canonical_ckpt "$f")" || return 1
        printf '%s' "$resolved"
        return 0
    fi
    match="$(ls "$CKPT_DIR"/${arm}_*_ckpt_last.pth 2>/dev/null | head -5)"
    n="$(printf '%s\n' "$match" | grep -c . || true)"
    if [ "$n" = "1" ]; then
        resolved="$(canonical_ckpt "$match")" || return 1
        printf '%s' "$resolved"
        return 0
    fi
    if [ "$n" != "0" ]; then
        echo ">>> [warn] ${arm} 命中多个候选 ckpt，请用 CKPT_${arm}=<路径> 指定:" >&2
        printf '%s\n' "$match" >&2
    fi
    return 1
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

    loaded="$(loaded_ckpt_from_session "$sess")"
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
    for i in $(seq 1 30); do
        port_open "$TSP3D_PORT" && break
        sleep 2
    done
    if ! port_open "$TSP3D_PORT"; then
        echo ">>> [error] TSP3D 端口 $TSP3D_PORT 未就绪"
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

# ---- 单场景评测 ----
run_one() {
    local arm="$1" scene="$2" ckpt="$3"
    local tag="${arm}_${scene}" log donef status t0 t1 sr oracle spl soft steps
    log="$RUN_DIR/eval_hm3d_$(date +%Y-%m-%d)_$(date +%H%M%S)_${tag}.log"
    donef="$RUN_DIR/.done/${tag}.done"
    if [ -n "$SKIP_DONE" ] && [ -f "$donef" ]; then
        echo ">>> [skip] $tag（已完成，SKIP_DONE=1）"
        return 0
    fi

    local args=(
        habitat.dataset.data_path="$DATA_PATH"
        habitat.dataset.split=val
        habitat_baselines.eval.split=val
        habitat_baselines.num_environments=1
        habitat.simulator.habitat_sim_v0.gpu_device_id=0
        habitat.simulator.habitat_sim_v0.gpu_gpu=False
        habitat_baselines.rl.policy.name="HabitatITMPolicyV1"
        habitat_baselines.rl.policy.gdp_enable=False
        habitat.dataset.content_scenes="[$scene]"
    )
    if [ -n "$LIMIT" ]; then
        args+=("habitat_baselines.test_episode_count=$LIMIT")
    fi
    local ov arm_ov
    eval "arm_ov=\${OVERRIDES_${arm}:-}"          # 每档独立覆盖（参数变体矩阵）
    for ov in $EXTRA_OVERRIDES $arm_ov; do args+=("$ov"); done

    if [ -n "$DRY_RUN" ]; then
        echo ">>> [dry-run] $tag  (TSP3D ckpt: $ckpt)"
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
    echo ">>> 档位 $arm | 场景 $scene | 服务权重 $(basename "$ckpt")"
    echo ">>> 日志 $log"
    t0=$(date +%s)
    python -u -m vlfm.run "${args[@]}" 2>&1 | tee -a "$log"
    status="${PIPESTATUS[0]}"
    t1=$(date +%s)

    sr="$(metric_of 'Success rate:' "$log" 'Oracle')"
    oracle="$(metric_of 'Oracle Success Rate:' "$log")"
    spl="$(metric_of 'Average episode spl:' "$log")"
    soft="$(metric_of 'Average episode soft_spl:' "$log")"
    steps="$(metric_of 'Average episode steps_count:' "$log")"

    # 检测级判读行（无论成败都落盘，缺行记 '-'）
    printf '%s\t%s\t%s\n' "$arm" "$scene" "$(detect_metrics "$log")" >> "$RUN_DIR/detect_summary.tsv"

    if [ "$status" -eq 0 ]; then
        mkdir -p "$RUN_DIR/.done" && touch "$donef"
        printf '%s\t%s\tOK\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$arm" "$scene" "${sr:--}" "${oracle:--}" "${spl:--}" "${soft:--}" "${steps:--}" \
            "$((t1 - t0))" "$(basename "$log")" >> "$RUN_DIR/summary.tsv"
        echo ">>> 完成 $tag  用时 $((t1 - t0))s  SR=${sr:--} Oracle=${oracle:--} spl=${spl:--}"
    else
        printf '%s\t%s\tFAIL(%s)\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$arm" "$scene" "$status" "${sr:--}" "${oracle:--}" "${spl:--}" "${soft:--}" "${steps:--}" \
            "$((t1 - t0))" "$(basename "$log")" >> "$RUN_DIR/summary.tsv"
        echo ">>> [fail] $tag 退出码 $status（日志已保留）"
        return 1
    fi
    return 0
}

# ---- 主流程 ----
if [ -z "$DRY_RUN" ]; then
    mkdir -p "$RUN_DIR/.done"
    printf 'arm\tscene\tstatus\tsuccess\toracle\tspl\tsoft_spl\tsteps\tsecs\tlog\n' > "$RUN_DIR/summary.tsv"
    printf 'arm\tscene\tdet\ttp\tfp\tprec\teps_tp\tentries\tentry_tp\tmaxscore_mean\tgate_rejects\n' > "$RUN_DIR/detect_summary.tsv"
    printf 'loaded_at\tcheckpoint\tsha256\ttmux_session\n' > "$RUN_DIR/checkpoints.tsv"
fi

queue_failed=0
for arm in $QUEUE; do
    ckpt="$(resolve_ckpt "$arm")" || ckpt=""
    if [ -z "$ckpt" ]; then
        if [ -n "$DRY_RUN" ]; then
            ckpt="$CKPT_DIR/${arm}_${GPU_TAG}_ckpt_last.pth"
            echo ">>> [dry-run] $arm ckpt 未找到 → 按默认命名展示（正式运行会 [skip]）: $ckpt"
        else
            echo ">>> [skip] $arm：未找到 ckpt（$CKPT_DIR/${arm}_*_ckpt_last.pth）——训练完成后再跑本档"
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
            # 同权重多参数变体（如 σ 扫描）：服务无需重启，只做轻量复核。
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
        if ! run_one "$arm" "$scene" "$ckpt"; then
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
    echo "断点续跑: SKIP_DONE=1 RUN_DIR=$RUN_DIR bash scripts/ft_eval.sh"
fi
echo "================================================="
if [ "$queue_failed" != "0" ]; then
    echo ">>> 注意：本次存在跳过的档或失败的场景（见上方 [skip] / [fail] / [error]）。"
fi
