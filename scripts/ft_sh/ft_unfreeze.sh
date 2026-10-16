#!/usr/bin/env bash
# TSP3D U0-U3 逐层解冻自动化训练。
#
# U0: 最终置信度 head
# U1: U0 + text projector + legacy 跨模态/FPN 路径
# U2: U1 + vision backbone layer4
# U3: U2 + vision backbone layer3
#
# 所有档位从同一 scanrefer checkpoint 初始化；bbox 与 RoBERTa 始终冻结，
# 使用同一 base loss、数据、epoch、seed 与优化器设置；Gate 不进入本轮。
#
# DRY_RUN=1 bash scripts/ft_tsp3d.sh
# PROFILES="u0 u1" LIMIT=2 SCENES=00006-HkseAnWCgqk EPOCHS=1 VAL_EVERY=0 bash scripts/ft_tsp3d.sh
# SKIP_DONE=1 bash scripts/ft_tsp3d.sh
# RESUME_U2=/abs/path/u2_5080_ckpt_epoch1.pth PROFILES=u2 bash scripts/ft_tsp3d.sh
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

CKPT="${CKPT:-$PROJECT_ROOT/data/tsp3d_models/tsp3d_scanrefer.pth}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/ft_datasets/local_view}"
WEIGHT_DIR="${WEIGHT_DIR:-$PROJECT_ROOT/data/tsp3d_ft/unfreeze}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/ft_tsp3d/ft_log/unfreeze}"
PROFILES="${PROFILES:-u0 u1 u2 u3}"
EPOCHS="${EPOCHS:-3}"
VAL_EVERY="${VAL_EVERY:-1}"
SEED="${SEED:-0}"
LR1="${LR1:-1e-4}"
LR2="${LR2:-5e-5}"
LR3="${LR3:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLIP="${CLIP:-0.1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

GPU_TAG="${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -oE '[0-9]{4}' | head -1 || true)}"
GPU_TAG="${GPU_TAG:-nogpu}"

if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
    if "$PY" -c "import sys, torch; v=torch.__version__.split('+')[0].split('.'); sys.exit(0 if (int(v[0]),int(v[1])) >= (2,1) else 1)" >/dev/null 2>&1; then
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    fi
fi

die() {
    echo ">>> [error] $*" >&2
    exit 1
}

is_profile() {
    case "$1" in u0|u1|u2|u3) return 0 ;; *) return 1 ;; esac
}

preflight() {
    [ -x "$PY" ] || die "Python 不可执行: $PY"
    [ -f "$CKPT" ] || die "scanrefer checkpoint 不存在: $CKPT"
    [ -d "$DATA_DIR/train" ] || die "训练集目录不存在: $DATA_DIR/train"
    [ -d "$DATA_DIR/val" ] || die "验证集目录不存在: $DATA_DIR/val"

    local n_train n_val profile
    n_train="$(find "$DATA_DIR/train" -maxdepth 1 -name '*.npz' -type f | wc -l)"
    n_val="$(find "$DATA_DIR/val" -maxdepth 1 -name '*.npz' -type f | wc -l)"
    [ "$n_train" -gt 0 ] || die "训练集为空"
    [ "$n_val" -gt 0 ] || die "验证集为空"
    for profile in $PROFILES; do
        is_profile "$profile" || die "未知 profile: $profile（仅支持 u0/u1/u2/u3）"
    done

    "$PY" -m py_compile \
        ft_tsp3d/init_from_pretrain.py \
        ft_tsp3d/model_bridge.py \
        ft_tsp3d/losses.py \
        ft_tsp3d/pipeline.py
    "$PY" -c "import torch, MinkowskiEngine; assert torch.cuda.is_available(); print('[preflight] torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0))"

    echo ">>> Python      : $PY"
    echo ">>> 初始化权重 : $CKPT"
    echo ">>> 数据         : train=$n_train scenes, val=$n_val scenes"
    echo ">>> 档位         : $PROFILES"
    echo ">>> 训练参数     : epochs=$EPOCHS seed=$SEED lr=($LR1,$LR2,$LR3) wd=$WEIGHT_DECAY"
    echo ">>> 权重目录     : $WEIGHT_DIR"
    echo ">>> 日志目录     : $LOG_DIR"
}

profile_resume() {
    local upper var
    upper="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
    var="RESUME_${upper}"
    printf '%s' "${!var:-}"
}

run_profile() {
    local profile="$1"
    local tag="${profile}_${GPU_TAG}"
    local done_file="$LOG_DIR/${tag}_train.done"
    local last_ckpt="$WEIGHT_DIR/${tag}_ckpt_last.pth"
    local console_log="$LOG_DIR/${tag}_train_console.log"
    local resume resume_var
    resume="$(profile_resume "$profile")"
    resume_var="RESUME_$(printf '%s' "$profile" | tr '[:lower:]' '[:upper:]')"

    if [ -n "${SKIP_DONE:-}" ] && [ -f "$done_file" ] && [ -s "$last_ckpt" ]; then
        echo ">>> [skip] $profile 已完成: $last_ckpt"
        return 0
    fi
    if [ -e "$last_ckpt" ] && [ -z "$resume" ] && [ -z "${OVERWRITE:-}" ]; then
        die "$profile 已有权重但没有完成标记，拒绝覆盖: $last_ckpt（用 $resume_var=... 或 OVERWRITE=1）"
    fi

    local -a cmd
    cmd=(
        "$PY" -u ft_tsp3d/pipeline.py
        --data-dir "$DATA_DIR"
        --ckpt "$CKPT"
        --epochs "$EPOCHS"
        --val-every "$VAL_EVERY"
        --seed "$SEED"
        --lr1 "$LR1" --lr2 "$LR2" --lr3 "$LR3"
        --wd "$WEIGHT_DECAY" --clip "$CLIP"
        --unfreeze-profile "$profile"
        --losses ""
        --out-dir "$LOG_DIR"
        --weight-dir "$WEIGHT_DIR"
        --log-dir "$LOG_DIR"
        --tag "$tag"
    )
    [ -n "${SCENES:-}" ] && cmd+=(--scenes "$SCENES")
    [ -n "${VAL_SCENES:-}" ] && cmd+=(--val-scenes "$VAL_SCENES")
    [ -n "${LIMIT:-}" ] && cmd+=(--limit-views "$LIMIT")
    [ -n "$resume" ] && cmd+=(--resume "$resume")

    echo "============================================================"
    echo ">>> 训练 $profile（tag=$tag）"
    "$PY" ft_tsp3d/init_from_pretrain.py --dry-run --unfreeze-profile "$profile"
    printf '>>> command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    echo "============================================================"
    if [ -n "${DRY_RUN:-}" ]; then
        return 0
    fi

    mkdir -p "$WEIGHT_DIR" "$LOG_DIR"
    rm -f -- "$done_file"
    set +e
    "${cmd[@]}" 2>&1 | tee "$console_log"
    local status="${PIPESTATUS[0]}"
    set -e
    if [ "$status" -ne 0 ]; then
        echo ">>> [error] $profile 训练失败，退出码 $status；队列停止" >&2
        return "$status"
    fi
    [ -s "$last_ckpt" ] || die "$profile 返回成功但缺少 ckpt: $last_ckpt"
    [ -s "$LOG_DIR/${tag}_trainable.json" ] || die "$profile 缺少可训练参数审计文件"
    {
        echo "profile=$profile"
        echo "checkpoint=$last_ckpt"
        echo "sha256=$(sha256sum "$last_ckpt" | awk '{print $1}')"
        echo "completed_at=$(date --iso-8601=seconds)"
    } > "$done_file"
}

preflight
for profile in $PROFILES; do
    run_profile "$profile" || exit 1
done

if [ -n "${DRY_RUN:-}" ]; then
    echo ">>> DRY_RUN 完成，未启动训练。"
else
    echo ">>> U0-U3 队列完成。权重: $WEIGHT_DIR"
    echo ">>> 下一步: ARMS=\"ref $PROFILES\" CKPT_DIR=\"$WEIGHT_DIR\" bash scripts/ft_e3.sh"
fi
