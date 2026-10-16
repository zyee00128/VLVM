#!/usr/bin/env bash
# 以现有 U1E3 为 R0 参照，自动训练 R1（U1 + candidate ranking）。
#
# 实验控制：
#   R0 = 已完成的 U1E3，不重复训练；
#   R1 = 从 scanrefer 独立初始化，保持 U1 解冻范围/数据/3 epoch/优化器不变，
#        唯一新增 candidate pairwise ranking loss。
#
# 仅检查命令：DRY_RUN=1 bash scripts/ft_rank.sh
# 正式启动：bash scripts/ft_rank.sh
# 断点恢复：RESUME=/abs/path/r1_5080_ckpt_epoch1.pth bash scripts/ft_rank.sh
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

R0_CKPT="${R0_CKPT:-$PROJECT_ROOT/data/tsp3d_ft/unfreeze/u1_5080_ckpt_epoch3.pth}"
INIT_CKPT="${INIT_CKPT:-$PROJECT_ROOT/data/tsp3d_models/tsp3d_scanrefer.pth}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/ft_datasets/local_view}"
WEIGHT_DIR="${WEIGHT_DIR:-$PROJECT_ROOT/data/tsp3d_ft/rank}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/ft_tsp3d/ft_log/rank}"

EPOCHS="${EPOCHS:-3}"
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
TAG="${TAG:-r1_${GPU_TAG}}"
LAST_CKPT="$WEIGHT_DIR/${TAG}_ckpt_last.pth"
DONE_FILE="$LOG_DIR/${TAG}_train.done"
CONSOLE_LOG="$LOG_DIR/${TAG}_train_console.log"
MANIFEST="$LOG_DIR/${TAG}_manifest.json"

if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
    if "$PY" -c "import sys, torch; v=torch.__version__.split('+')[0].split('.'); sys.exit(0 if (int(v[0]),int(v[1])) >= (2,1) else 1)" >/dev/null 2>&1; then
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    fi
fi

die() {
    echo ">>> [error] $*" >&2
    exit 1
}

preflight() {
    [ -x "$PY" ] || die "Python 不可执行: $PY"
    [ -s "$R0_CKPT" ] || die "R0/U1E3 checkpoint 不存在: $R0_CKPT"
    [ -s "$INIT_CKPT" ] || die "scanrefer checkpoint 不存在: $INIT_CKPT"
    [ -d "$DATA_DIR/train" ] || die "训练集目录不存在: $DATA_DIR/train"
    [ -d "$DATA_DIR/val" ] || die "验证集目录不存在: $DATA_DIR/val"

    "$PY" -m py_compile \
        ft_tsp3d/losses.py \
        ft_tsp3d/model_bridge.py \
        ft_tsp3d/pipeline.py \
        vlfm/tsp3d_models/multilevel_head.py
    "$PY" ft_tsp3d/smoke_test.py \
        --tests candidate_pairwise_rank,streamed_rank_gradient
    "$PY" -c "import torch, MinkowskiEngine; assert torch.cuda.is_available(); print('[preflight] torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0))"

    echo ">>> R0 参照      : $R0_CKPT"
    echo ">>> R1 初始化    : $INIT_CKPT"
    echo ">>> R1 解冻范围  : u1"
    echo ">>> R1 损失      : base + rank"
    echo ">>> rank 参数    : weight=$RANK_WEIGHT margin=$RANK_MARGIN iou=$RANK_IOU_THRESHOLD"
    echo ">>> 训练参数     : epochs=$EPOCHS seed=$SEED lr=($LR1,$LR2,$LR3) wd=$WEIGHT_DECAY"
    echo ">>> 权重目录     : $WEIGHT_DIR"
    echo ">>> 日志目录     : $LOG_DIR"
}

build_command() {
    CMD=(
        "$PY" -u ft_tsp3d/pipeline.py
        --data-dir "$DATA_DIR"
        --ckpt "$INIT_CKPT"
        --epochs "$EPOCHS"
        --val-every "$VAL_EVERY"
        --seed "$SEED"
        --lr1 "$LR1" --lr2 "$LR2" --lr3 "$LR3"
        --wd "$WEIGHT_DECAY" --clip "$CLIP"
        --unfreeze-profile u1
        --losses rank
        --rank-weight "$RANK_WEIGHT"
        --rank-margin "$RANK_MARGIN"
        --rank-iou-threshold "$RANK_IOU_THRESHOLD"
        --out-dir "$LOG_DIR"
        --weight-dir "$WEIGHT_DIR"
        --log-dir "$LOG_DIR"
        --tag "$TAG"
    )
    [ -n "${SCENES:-}" ] && CMD+=(--scenes "$SCENES")
    [ -n "${VAL_SCENES:-}" ] && CMD+=(--val-scenes "$VAL_SCENES")
    [ -n "${LIMIT:-}" ] && CMD+=(--limit-views "$LIMIT")
    [ -n "${RESUME:-}" ] && CMD+=(--resume "$RESUME")
    return 0
}

write_manifest() {
    local status="$1"
    "$PY" - "$MANIFEST" "$status" "$R0_CKPT" "$INIT_CKPT" "$LAST_CKPT" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

dst, status, r0, init, r1 = sys.argv[1:]

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
    "phase": "S4-C",
    "status": status,
    "r0": {"name": "u1e3", "checkpoint": r0, "sha256": digest(r0)},
    "r1": {
        "checkpoint": r1 if Path(r1).is_file() else None,
        "sha256": digest(r1),
        "initial_checkpoint": init,
        "initial_sha256": digest(init),
        "unfreeze_profile": "u1",
        "losses": ["base", "rank"],
        "rank_weight": float(os.environ["RANK_WEIGHT"]),
        "rank_margin": float(os.environ["RANK_MARGIN"]),
        "rank_iou_threshold": float(os.environ["RANK_IOU_THRESHOLD"]),
    },
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
}
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
}


preflight
build_command

if [ -n "${SKIP_DONE:-}" ] && [ -s "$DONE_FILE" ] && [ -s "$LAST_CKPT" ]; then
    echo ">>> [skip] R1 已完成: $LAST_CKPT"
    exit 0
fi
if [ -e "$LAST_CKPT" ] && [ -z "${RESUME:-}" ] && [ -z "${OVERWRITE:-}" ]; then
    die "R1 已有权重但没有完成标记，拒绝覆盖: $LAST_CKPT（使用 RESUME=... 或 OVERWRITE=1）"
fi

printf '>>> command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [ -n "${DRY_RUN:-}" ]; then
    echo ">>> DRY_RUN 完成，R0 未重复训练，R1 未启动。"
    exit 0
fi

mkdir -p "$WEIGHT_DIR" "$LOG_DIR"
export RANK_WEIGHT RANK_MARGIN RANK_IOU_THRESHOLD
write_manifest running
rm -f -- "$DONE_FILE"

set +e
"${CMD[@]}" 2>&1 | tee "$CONSOLE_LOG"
STATUS="${PIPESTATUS[0]}"
set -e
if [ "$STATUS" -ne 0 ]; then
    write_manifest failed
    die "R1 训练失败，退出码 $STATUS"
fi

[ -s "$LAST_CKPT" ] || die "R1 返回成功但缺少 checkpoint: $LAST_CKPT"
[ -s "$LOG_DIR/${TAG}_trainable.json" ] || die "R1 缺少可训练参数审计文件"
write_manifest complete
{
    echo "reference_r0=$R0_CKPT"
    echo "checkpoint_r1=$LAST_CKPT"
    echo "sha256_r1=$(sha256sum "$LAST_CKPT" | awk '{print $1}')"
    echo "completed_at=$(date --iso-8601=seconds)"
} > "$DONE_FILE"

echo ">>> R0/R1 完成：R0 复用 U1E3，R1 权重位于 $LAST_CKPT"
