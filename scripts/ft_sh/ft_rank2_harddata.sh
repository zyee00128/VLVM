#!/usr/bin/env bash
# 使用 hard-negative replay v2 训练 R2 单变量对照。
#
# R2：从 ScanRefer 独立初始化，U1 + base/rank，3 epoch；每轮固定 23,031 步，
#     以 2,303 个 replay view 配对替换其原始 source view，只改变 hard replay 数据供给。
# R3：已拆分为 C/H 配对续训，由 scripts/ft_r3.sh 单独管理；本脚本仅保留 R2。
#
# 仅打印命令：DRY_RUN=1 bash scripts/ft_rank_harddata.sh
# 正式训练：bash scripts/ft_rank_harddata.sh
# 阶段恢复：R2_RESUME=/abs/r2_epoch1.pth bash scripts/ft_rank_harddata.sh
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
DEFAULT_PY="$CONDA_BASE/envs/vlvm/bin/python"
if [ ! -x "$DEFAULT_PY" ]; then
    DEFAULT_PY="$(command -v python || true)"
fi
PY="${PY:-$DEFAULT_PY}"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/ft_datasets/local_view}"
REPLAY_DATA_DIR="${REPLAY_DATA_DIR:-$PROJECT_ROOT/data/ft_datasets/hard_negative_replay_v2}"
REPLAY_DONE_FILE="${REPLAY_DONE_FILE:-$PROJECT_ROOT/ft_tsp3d/ft_log/r2_data_v2/r2_data.done}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
REPLAY_RATIO="${REPLAY_RATIO:-0.10}"
REPLAY_MODE="${REPLAY_MODE:-replace-source}"

R2_INIT_CKPT="${R2_INIT_CKPT:-$PROJECT_ROOT/data/tsp3d_models/tsp3d_scanrefer.pth}"
R2_WEIGHT_DIR="${R2_WEIGHT_DIR:-$PROJECT_ROOT/data/tsp3d_ft/r2}"
R2_LOG_DIR="${R2_LOG_DIR:-$PROJECT_ROOT/ft_tsp3d/ft_log/r2}"
R2_EPOCHS="${R2_EPOCHS:-3}"
R2_LR1="${R2_LR1:-1e-4}"
R2_LR2="${R2_LR2:-5e-5}"
R2_LR3="${R2_LR3:-1e-5}"

VAL_EVERY="${VAL_EVERY:-1}"
SEED="${SEED:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLIP="${CLIP:-0.1}"
RANK_WEIGHT="${RANK_WEIGHT:-1.0}"
RANK_MARGIN="${RANK_MARGIN:-0.2}"
RANK_IOU_THRESHOLD="${RANK_IOU_THRESHOLD:-0.25}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

GPU_TAG="${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -oE '[0-9]{4}' | head -1 || true)}"
GPU_TAG="${GPU_TAG:-nogpu}"
R2_TAG="${R2_TAG:-r2_${GPU_TAG}}"
PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-$PROJECT_ROOT/ft_tsp3d/ft_log/rank_harddata}"
PIPELINE_DONE="$PIPELINE_LOG_DIR/r2.done"

if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
    if "$PY" -c "import sys, torch; v=torch.__version__.split('+')[0].split('.'); sys.exit(0 if (int(v[0]),int(v[1])) >= (2,1) else 1)" >/dev/null 2>&1; then
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    fi
fi

die() {
    echo ">>> [error] $*" >&2
    exit 1
}

common_preflight() {
    [ -x "$PY" ] || die "Python 不可执行: $PY"
    [ -s "$R2_INIT_CKPT" ] || die "R2 ScanRefer checkpoint 不存在: $R2_INIT_CKPT"
    [ -d "$DATA_DIR/train" ] || die "原始训练集不存在: $DATA_DIR/train"
    [ -d "$DATA_DIR/val" ] || die "原始验证集不存在: $DATA_DIR/val"
    "$PY" -m py_compile ft_tsp3d/pipeline.py ft_tsp3d/losses.py ft_tsp3d/model_bridge.py
}

wait_for_replay() {
    local waits=0
    while [ ! -s "$REPLAY_DONE_FILE" ]; do
        if ! pgrep -f "hard_negative_replay.py.*--output-dir ${REPLAY_DATA_DIR}" >/dev/null; then
            die "replay v2 未完成且没有处理进程: $REPLAY_DONE_FILE"
        fi
        waits=$((waits + 1))
        if [ $((waits % 5)) -eq 0 ]; then
            echo ">>> replay v2 仍在处理，已等待 $((waits * WAIT_SECONDS / 60)) 分钟"
        fi
        sleep "$WAIT_SECONDS"
    done
    [ -d "$REPLAY_DATA_DIR/train" ] || die "replay train 目录不存在: $REPLAY_DATA_DIR/train"
    [ -s "$REPLAY_DATA_DIR/stats.json" ] || die "replay stats.json 不存在: $REPLAY_DATA_DIR/stats.json"
}

append_common_args() {
    local -n cmd_ref="$1"
    cmd_ref+=(
        --data-dir "$DATA_DIR"
        --replay-data-dir "$REPLAY_DATA_DIR"
        --replay-ratio "$REPLAY_RATIO"
        --replay-mode "$REPLAY_MODE"
        --val-every "$VAL_EVERY"
        --seed "$SEED"
        --wd "$WEIGHT_DECAY"
        --clip "$CLIP"
        --unfreeze-profile u1
        --losses rank
        --rank-weight "$RANK_WEIGHT"
        --rank-margin "$RANK_MARGIN"
        --rank-iou-threshold "$RANK_IOU_THRESHOLD"
    )
    [ -n "${SCENES:-}" ] && cmd_ref+=(--scenes "$SCENES")
    [ -n "${VAL_SCENES:-}" ] && cmd_ref+=(--val-scenes "$VAL_SCENES")
    [ -n "${LIMIT:-}" ] && cmd_ref+=(--limit-views "$LIMIT")
    return 0
}

build_commands() {
    R2_CMD=(
        "$PY" -u ft_tsp3d/pipeline.py
        --ckpt "$R2_INIT_CKPT"
        --epochs "$R2_EPOCHS"
        --lr1 "$R2_LR1" --lr2 "$R2_LR2" --lr3 "$R2_LR3"
        --out-dir "$R2_LOG_DIR"
        --weight-dir "$R2_WEIGHT_DIR"
        --log-dir "$R2_LOG_DIR"
        --tag "$R2_TAG"
    )
    append_common_args R2_CMD
    [ -n "${R2_RESUME:-}" ] && R2_CMD+=(--resume "$R2_RESUME")

    return 0
}

print_command() {
    local label="$1"
    local -n cmd_ref="$2"
    printf '>>> %s command:' "$label"
    printf ' %q' "${cmd_ref[@]}"
    printf '\n'
}

write_manifest() {
    local dst="$1" status="$2" stage="$3" init="$4" result="$5"
    local epochs="$6" lr1="$7" lr2="$8" lr3="$9"
    "$PY" - "$dst" "$status" "$stage" "$init" "$result" \
        "$epochs" "$lr1" "$lr2" "$lr3" "$REPLAY_DATA_DIR/stats.json" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

dst, status, stage, init, result, epochs, lr1, lr2, lr3, replay_stats = sys.argv[1:]

def digest(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

payload = {
    "phase": f"S4-C-{stage}",
    "status": status,
    "initial_checkpoint": init,
    "initial_sha256": digest(init),
    "result_checkpoint": result if Path(result).is_file() else None,
    "result_sha256": digest(result),
    "unfreeze_profile": "u1",
    "losses": ["base", "rank"],
    "data_mix": {
        "original_dir": os.environ["DATA_DIR"],
        "replay_dir": os.environ["REPLAY_DATA_DIR"],
        "replay_ratio": float(os.environ["REPLAY_RATIO"]),
        "replay_mode": os.environ["REPLAY_MODE"],
        "replay_stats_sha256": digest(replay_stats),
    },
    "epochs": int(epochs),
    "learning_rates": [float(lr1), float(lr2), float(lr3)],
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
}
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
}

run_stage() {
    local stage="$1" cmd_name="$2" init="$3" weight_dir="$4" log_dir="$5" tag="$6"
    local epochs="$7" lr1="$8" lr2="$9" lr3="${10}" resume="${11}" overwrite="${12}"
    local -n cmd_ref="$cmd_name"
    local last_ckpt="$weight_dir/${tag}_ckpt_last.pth"
    local done_file="$log_dir/${tag}_train.done"
    local console_log="$log_dir/${tag}_train_console.log"
    local manifest="$log_dir/${tag}_manifest.json"

    if [ -s "$done_file" ] && [ -s "$last_ckpt" ]; then
        echo ">>> [skip] $stage 已完成: $last_ckpt"
        return 0
    fi
    if [ -e "$last_ckpt" ] && [ -z "$resume" ] && [ -z "$overwrite" ]; then
        die "$stage 已有权重但没有完成标记，拒绝覆盖: $last_ckpt"
    fi

    mkdir -p "$weight_dir" "$log_dir"
    write_manifest "$manifest" running "$stage" "$init" "$last_ckpt" \
        "$epochs" "$lr1" "$lr2" "$lr3"
    rm -f -- "$done_file"

    echo ">>> 开始 $stage"
    set +e
    "${cmd_ref[@]}" 2>&1 | tee "$console_log"
    local status="${PIPESTATUS[0]}"
    set -e
    if [ "$status" -ne 0 ]; then
        write_manifest "$manifest" failed "$stage" "$init" "$last_ckpt" \
            "$epochs" "$lr1" "$lr2" "$lr3"
        die "$stage 训练失败，退出码 $status；后续阶段未启动"
    fi

    [ -s "$last_ckpt" ] || die "$stage 返回成功但缺少 checkpoint: $last_ckpt"
    [ -s "$log_dir/${tag}_trainable.json" ] || die "$stage 缺少可训练参数审计文件"
    write_manifest "$manifest" complete "$stage" "$init" "$last_ckpt" \
        "$epochs" "$lr1" "$lr2" "$lr3"
    {
        echo "stage=$stage"
        echo "initial_checkpoint=$init"
        echo "checkpoint=$last_ckpt"
        echo "sha256=$(sha256sum "$last_ckpt" | awk '{print $1}')"
        echo "completed_at=$(date --iso-8601=seconds)"
    } > "$done_file"
    echo ">>> $stage 完成: $last_ckpt"
}

common_preflight
build_commands
print_command R2 R2_CMD

if [ -n "${DRY_RUN:-}" ]; then
    echo ">>> DRY_RUN 完成：R2 未启动。"
    exit 0
fi

wait_for_replay
"$PY" ft_tsp3d/smoke_test.py \
    --tests replay_mix_plan,candidate_pairwise_rank,streamed_rank_gradient
"$PY" -c "import torch, MinkowskiEngine; assert torch.cuda.is_available(); print('[preflight] torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0))"

export DATA_DIR REPLAY_DATA_DIR REPLAY_RATIO REPLAY_MODE
mkdir -p "$PIPELINE_LOG_DIR"
rm -f -- "$PIPELINE_DONE"

run_stage R2 R2_CMD "$R2_INIT_CKPT" "$R2_WEIGHT_DIR" "$R2_LOG_DIR" "$R2_TAG" \
    "$R2_EPOCHS" "$R2_LR1" "$R2_LR2" "$R2_LR3" "${R2_RESUME:-}" "${R2_OVERWRITE:-}"

{
    echo "r2_checkpoint=$R2_WEIGHT_DIR/${R2_TAG}_ckpt_last.pth"
    echo "completed_at=$(date --iso-8601=seconds)"
} > "$PIPELINE_DONE"
echo ">>> R2 微调完成: $PIPELINE_DONE"
