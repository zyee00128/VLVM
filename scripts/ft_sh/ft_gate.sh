#!/usr/bin/env bash
# Presence Gate 唯一保留训练入口：VLVM V1 冻结核心，只使用原始 local_view
# 独立训练 presence_gate.*。不支持联合训练、hard-data 或 replay 混合。
set -Eeuo pipefail

HERE="$(cd "${BASH_SOURCE[0]%/*}" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
DEFAULT_PY="$CONDA_BASE/envs/vlvm/bin/python"
if [ ! -x "$DEFAULT_PY" ]; then DEFAULT_PY="$(command -v python || true)"; fi
PY="${PY:-$DEFAULT_PY}"

BASE_CKPT="${BASE_CKPT:-$PROJECT_ROOT/data/vlvm_v1/vlvm_v1.pth}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/ft_datasets/local_view}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/ft_tsp3d/ft_log/gate_only_original}"
WEIGHT_DIR="${WEIGHT_DIR:-$PROJECT_ROOT/data/tsp3d_ft/gate_only_original}"
EPOCHS="${EPOCHS:-3}"
VAL_EVERY="${VAL_EVERY:-1}"
SEED="${SEED:-0}"
GATE_LR="${GATE_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLIP="${CLIP:-0.1}"
GATE_GAMMA_NEG="${GATE_GAMMA_NEG:-2.0}"
GATE_POSITIVE_MARGIN="${GATE_POSITIVE_MARGIN:-0.70}"
GATE_RECALL_WEIGHT="${GATE_RECALL_WEIGHT:-1.0}"
GATE_TARGET_RECALL="${GATE_TARGET_RECALL:-0.99}"
GATE_INIT_SEED="${GATE_INIT_SEED:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

GPU_TAG="${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -oE '[0-9]{4}' | head -1 || true)}"
GPU_TAG="${GPU_TAG:-nogpu}"
TAG="${TAG:-gate_only_original_${GPU_TAG}}"

if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
    if "$PY" -c "import sys, torch; v=torch.__version__.split('+')[0].split('.'); sys.exit(0 if (int(v[0]),int(v[1])) >= (2,1) else 1)" >/dev/null 2>&1; then
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    fi
fi

die() { echo ">>> [error] $*" >&2; exit 1; }

preflight() {
    [ -x "$PY" ] || die "Python 不可执行: $PY"
    [ -s "$BASE_CKPT" ] || die "VLVM V1 checkpoint 不存在: $BASE_CKPT"
    [ -d "$DATA_DIR/train" ] || die "训练集不存在: $DATA_DIR/train"
    [ -d "$DATA_DIR/val" ] || die "验证集不存在: $DATA_DIR/val"
    [ -s "$DATA_DIR/stats.json" ] || die "训练数据 stats.json 不存在"
    [ "$VAL_EVERY" = "1" ] || die "Gate 独立训练要求逐 epoch validation"

    "$PY" -m py_compile \
        ft_tsp3d/losses.py ft_tsp3d/model_bridge.py ft_tsp3d/pipeline.py \
        ft_tsp3d/eval_tools/calibrate_gate.py ft_tsp3d/init_from_pretrain.py \
        vlfm/tsp3d_models/bdetr.py vlfm/tsp3d_models/multilevel_head.py
    "$PY" ft_tsp3d/smoke_test.py \
        --tests scheduler,gate_only_training,gate_initialization_rng_isolation,checkpoint_ft_config
    if [ -z "${DRY_RUN:-}" ]; then
        "$PY" -c "import torch, MinkowskiEngine; assert torch.cuda.is_available(); print('[preflight] torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0))"
    fi
}

CMD=("$PY" -u ft_tsp3d/pipeline.py --losses gate --gate-only
    --unfreeze-profile gate --ckpt "$BASE_CKPT" --epochs "$EPOCHS"
    --data-dir "$DATA_DIR" --val-every "$VAL_EVERY" --seed "$SEED"
    --lr1 "$GATE_LR" --lr2 "$GATE_LR" --lr3 "$GATE_LR"
    --wd "$WEIGHT_DECAY" --clip "$CLIP"
    --gate-gamma-neg "$GATE_GAMMA_NEG"
    --gate-positive-margin "$GATE_POSITIVE_MARGIN"
    --gate-recall-weight "$GATE_RECALL_WEIGHT"
    --gate-init-seed "$GATE_INIT_SEED"
    --out-dir "$LOG_DIR" --weight-dir "$WEIGHT_DIR" --log-dir "$LOG_DIR" --tag "$TAG")
[ -n "${RESUME:-}" ] && CMD+=(--resume "$RESUME")
[ -n "${SCENES:-}" ] && CMD+=(--scenes "$SCENES")
[ -n "${VAL_SCENES:-}" ] && CMD+=(--val-scenes "$VAL_SCENES")
[ -n "${LIMIT:-}" ] && CMD+=(--limit-views "$LIMIT")

preflight
printf '>>> Gate-only original command:'; printf ' %q' "${CMD[@]}"; printf '\n'
echo ">>> 初始化权重: $BASE_CKPT"
echo ">>> 日志目录: $LOG_DIR"
echo ">>> 权重目录: $WEIGHT_DIR"
if [ -n "${DRY_RUN:-}" ]; then
    echo ">>> DRY_RUN 完成，未启动训练。"
    exit 0
fi

mkdir -p "$LOG_DIR" "$WEIGHT_DIR"
"${CMD[@]}" 2>&1 | tee -a "$LOG_DIR/${TAG}_train_console.log"

for epoch in $(seq 1 "$EPOCHS"); do
    dump="$LOG_DIR/${TAG}_gate_val_epoch${epoch}.npz"
    report="$LOG_DIR/${TAG}_calibration_e${epoch}.json"
    [ -s "$dump" ] || die "Gate E$epoch 校准数据不存在: $dump"
    "$PY" ft_tsp3d/eval_tools/calibrate_gate.py --dump "$dump" \
        --target-recall "$GATE_TARGET_RECALL" --out "$report" \
        > "$LOG_DIR/${TAG}_calibration_e${epoch}.log"
done

final_ckpt="$WEIGHT_DIR/${TAG}_ckpt_epoch${EPOCHS}.pth"
last_ckpt="$WEIGHT_DIR/${TAG}_ckpt_last.pth"
[ -s "$final_ckpt" ] || die "缺少最终 checkpoint: $final_ckpt"
sha256sum "$final_ckpt" "$last_ckpt" > "$LOG_DIR/${TAG}_checkpoints.sha256"
echo ">>> 原始数据独立 Gate 训练完成。"
