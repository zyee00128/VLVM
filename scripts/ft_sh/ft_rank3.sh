#!/usr/bin/env bash
# R3-C / R3-H 分阶段配对续训。
#
# C  : R1 E3 -> 总第 4 轮，100% 原训练集（原始数据续训控制）。
# H1 : R1 E2 -> 总第 3 轮，10% replay replace-source；与现有 R1 E3 配对。
# H2 : H1    -> 总第 4 轮，10% replay replace-source；与 C E1 配对。
#
# 为保持 R1 的学习率轨迹，H1 必须先以 --epochs 3 单独产出，H2 再以
# --epochs 4 从 H1 续训。禁止从 R1 E2 直接以 --epochs 4 连跑两轮。
#
# 默认只运行 C，不会在未检查 C 收敛性前自动进入 H：
#   DRY_RUN=1 bash scripts/ft_r3.sh
#   bash scripts/ft_r3.sh
# C 结果确认后运行 H：
#   STAGES="h1 h2" bash scripts/ft_r3.sh
# 仅打印全部计划：
#   DRY_RUN=1 STAGES="c h1 h2" bash scripts/ft_r3.sh
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
R1E2_CKPT="${R1E2_CKPT:-$PROJECT_ROOT/data/tsp3d_ft/rank/r1_5080_ckpt_epoch2.pth}"
R1E3_CKPT="${R1E3_CKPT:-$PROJECT_ROOT/data/tsp3d_ft/rank/r1_5080_ckpt_epoch3.pth}"

R3_WEIGHT_ROOT="${R3_WEIGHT_ROOT:-$PROJECT_ROOT/data/tsp3d_ft/r3}"
R3_LOG_ROOT="${R3_LOG_ROOT:-$PROJECT_ROOT/ft_tsp3d/ft_log/r3}"
STAGES="${STAGES:-c}"

REPLAY_RATIO="${REPLAY_RATIO:-0.10}"
REPLAY_MODE="${REPLAY_MODE:-replace-source}"
VAL_EVERY="${VAL_EVERY:-1}"
SEED="${SEED:-0}"
LR1="${LR1:-1e-4}"
LR2="${LR2:-5e-5}"
LR3="${LR3:-1e-5}"
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
C_TAG="${C_TAG:-r3c_${GPU_TAG}}"
H1_TAG="${H1_TAG:-r3h1_${GPU_TAG}}"
H2_TAG="${H2_TAG:-r3h2_${GPU_TAG}}"

if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
    if "$PY" -c "import sys, torch; v=torch.__version__.split('+')[0].split('.'); sys.exit(0 if (int(v[0]),int(v[1])) >= (2,1) else 1)" >/dev/null 2>&1; then
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    fi
fi

die() {
    echo ">>> [error] $*" >&2
    exit 1
}

contains_stage() {
    local wanted="$1" item
    for item in $STAGES; do
        [ "$item" = "$wanted" ] && return 0
    done
    return 1
}

validate_stages() {
    local item
    for item in $STAGES; do
        case "$item" in
            c|h1|h2) ;;
            *) die "未知阶段 '$item'；允许 c/h1/h2" ;;
        esac
    done
}

preflight() {
    [ -x "$PY" ] || die "Python 不可执行: $PY"
    [ -s "$R1E2_CKPT" ] || die "R1 E2 checkpoint 不存在: $R1E2_CKPT"
    [ -s "$R1E3_CKPT" ] || die "R1 E3 checkpoint 不存在: $R1E3_CKPT"
    [ -d "$DATA_DIR/train" ] || die "原始训练集不存在: $DATA_DIR/train"
    [ -d "$DATA_DIR/val" ] || die "原始验证集不存在: $DATA_DIR/val"
    if contains_stage h1 || contains_stage h2; then
        [ -d "$REPLAY_DATA_DIR/train" ] || die "replay 训练集不存在: $REPLAY_DATA_DIR/train"
        [ -s "$REPLAY_DATA_DIR/stats.json" ] || die "replay stats.json 不存在: $REPLAY_DATA_DIR/stats.json"
    fi
    "$PY" -m py_compile ft_tsp3d/pipeline.py ft_tsp3d/losses.py ft_tsp3d/model_bridge.py
}

append_common_args() {
    local -n cmd_ref="$1"
    cmd_ref+=(
        --data-dir "$DATA_DIR"
        --val-every "$VAL_EVERY"
        --seed "$SEED"
        --lr1 "$LR1" --lr2 "$LR2" --lr3 "$LR3"
        --wd "$WEIGHT_DECAY" --clip "$CLIP"
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

build_command() {
    local cmd_name="$1" resume="$2" epochs="$3" weight_dir="$4" log_dir="$5" tag="$6" replay="$7"
    local -n cmd_ref="$cmd_name"
    cmd_ref=(
        "$PY" -u ft_tsp3d/pipeline.py
        --ckpt "$resume"
        --resume "$resume"
        --epochs "$epochs"
        --out-dir "$log_dir"
        --weight-dir "$weight_dir"
        --log-dir "$log_dir"
        --tag "$tag"
    )
    append_common_args "$cmd_name"
    if [ "$replay" = "1" ]; then
        cmd_ref+=(
            --replay-data-dir "$REPLAY_DATA_DIR"
            --replay-ratio "$REPLAY_RATIO"
            --replay-mode "$REPLAY_MODE"
        )
    fi
}

print_command() {
    local label="$1"
    local -n cmd_ref="$2"
    printf '>>> %s command:' "$label"
    printf ' %q' "${cmd_ref[@]}"
    printf '\n'
}

run_stage() {
    local stage="$1" cmd_name="$2" resume="$3" epochs="$4" weight_dir="$5" log_dir="$6" tag="$7"
    local -n cmd_ref="$cmd_name"
    local last_ckpt="$weight_dir/${tag}_ckpt_last.pth"
    local epoch_ckpt="$weight_dir/${tag}_ckpt_epoch${epochs}.pth"
    local done_file="$log_dir/${tag}_train.done"
    local console_log="$log_dir/${tag}_train_console.log"

    print_command "$stage" "$cmd_name"
    if [ -n "${DRY_RUN:-}" ]; then
        return 0
    fi
    if [ -s "$done_file" ] && [ -s "$last_ckpt" ] && [ -s "$epoch_ckpt" ]; then
        echo ">>> [skip] $stage 已完成: $epoch_ckpt"
        return 0
    fi
    if [ -e "$last_ckpt" ] && [ -z "${OVERWRITE:-}" ]; then
        die "$stage 已有未标记完成的权重，拒绝覆盖: $last_ckpt"
    fi

    mkdir -p "$weight_dir" "$log_dir"
    rm -f -- "$done_file"
    echo ">>> 开始 $stage：resume=$(basename "$resume") total_epochs=$epochs"
    set +e
    "${cmd_ref[@]}" 2>&1 | tee "$console_log"
    local status="${PIPESTATUS[0]}"
    set -e
    [ "$status" -eq 0 ] || die "$stage 训练失败，退出码 $status"
    [ -s "$last_ckpt" ] || die "$stage 缺少 last checkpoint: $last_ckpt"
    [ -s "$epoch_ckpt" ] || die "$stage 缺少 epoch checkpoint: $epoch_ckpt"
    [ -s "$log_dir/${tag}_trainable.json" ] || die "$stage 缺少可训练参数审计文件"
    {
        echo "stage=$stage"
        echo "resume_checkpoint=$resume"
        echo "total_epochs=$epochs"
        echo "checkpoint=$epoch_ckpt"
        echo "sha256=$(sha256sum "$epoch_ckpt" | awk '{print $1}')"
        echo "completed_at=$(date --iso-8601=seconds)"
    } > "$done_file"
    echo ">>> $stage 完成: $epoch_ckpt"
}

validate_stages
preflight

C_WEIGHT_DIR="$R3_WEIGHT_ROOT/c"
C_LOG_DIR="$R3_LOG_ROOT/c"
H_WEIGHT_DIR="$R3_WEIGHT_ROOT/h"
H_LOG_DIR="$R3_LOG_ROOT/h"
H1_CKPT="$H_WEIGHT_DIR/${H1_TAG}_ckpt_epoch3.pth"

build_command C_CMD "$R1E3_CKPT" 4 "$C_WEIGHT_DIR" "$C_LOG_DIR" "$C_TAG" 0
build_command H1_CMD "$R1E2_CKPT" 3 "$H_WEIGHT_DIR" "$H_LOG_DIR" "$H1_TAG" 1
build_command H2_CMD "$H1_CKPT" 4 "$H_WEIGHT_DIR" "$H_LOG_DIR" "$H2_TAG" 1

echo ">>> R3 阶段: $STAGES"
echo ">>> 固定设置: U1 + base/rank; lr=($LR1,$LR2,$LR3); wd=$WEIGHT_DECAY; clip=$CLIP; seed=$SEED"
echo ">>> H 数据: replay_ratio=$REPLAY_RATIO replay_mode=$REPLAY_MODE（每轮总步数不变）"

if contains_stage c; then
    run_stage C C_CMD "$R1E3_CKPT" 4 "$C_WEIGHT_DIR" "$C_LOG_DIR" "$C_TAG"
fi
if contains_stage h1; then
    run_stage H1 H1_CMD "$R1E2_CKPT" 3 "$H_WEIGHT_DIR" "$H_LOG_DIR" "$H1_TAG"
fi
if contains_stage h2; then
    if [ -z "${DRY_RUN:-}" ] && [ ! -s "$H1_CKPT" ]; then
        die "H2 必须从 H1 epoch3 续训，但 checkpoint 不存在: $H1_CKPT"
    fi
    run_stage H2 H2_CMD "$H1_CKPT" 4 "$H_WEIGHT_DIR" "$H_LOG_DIR" "$H2_TAG"
fi

if [ -n "${DRY_RUN:-}" ]; then
    echo ">>> DRY_RUN 完成，未启动训练。"
else
    echo ">>> 指定的 R3 阶段已完成: $STAGES"
fi
