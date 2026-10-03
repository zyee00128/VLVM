#!/bin/bash
# ==============================================================================
# TSP3D 微调跑批 · 五档损失消融权重生成（base / gate / s1 / s2 / distill）
# ------------------------------------------------------------------------------
# 档位 = base + 单损失开关（单变量消融；模型开关由 pipeline 按损失自动推导）：
#   base     base（含 L_bg）                        --losses ""（默认即 base）
#   gate     base + L_gate                          --losses gate
#   s1       base + L_s1                            --losses s1
#   s2       base + L_s2（含 TGP 子项）              --losses s2,tgp
#   distill  base + L_gate_dist（门控仅蒸馏监督）     --losses distill
#
# 双机分工（权重生成分两部分；每部分 base 单独跑一次，其他消融分开跑）：
#   A 机：base → gate → s1        MACHINE=a bash scripts/ft_tsp3d.sh
#   B 机：base → s2 → distill     MACHINE=b bash scripts/ft_tsp3d.sh
#
# 常用覆盖：
#   TASK=s2  bash scripts/ft_tsp3d.sh                 # 只跑单档
#   TASKS="gate s1" bash scripts/ft_tsp3d.sh          # 自定队列
#   HEAD=2  MACHINE=a bash scripts/ft_tsp3d.sh        # 只跑队列前 2 档（先跑前几档、看输出再续）
#   LIMIT=2 SCENES=00006-HkseAnWCgqk EPOCHS=1 VAL_EVERY=0 TASK=base \
#       WEIGHT_DIR=/tmp/ft_smoke_w LOG_DIR=/tmp/ft_smoke_l bash scripts/ft_tsp3d.sh   # 冒烟（临时目录）
#   SKIP_DONE=1 bash scripts/ft_tsp3d.sh              # 断点续跑（已有 <档>_<GPU>_train.done 的档跳过）
#
# 产物落点（档位名 + 显卡编号命名；<GPU> 自动探测，本机 3090 / 另一台 5080，可 GPU_TAG=... 覆盖）：
#   权重：$WEIGHT_DIR/<档>_<GPU>_ckpt_last.pth、<档>_<GPU>_ckpt_epoch<N>.pth
#   日志：$LOG_DIR/<档>_<GPU>_train.log、<档>_<GPU>_train_console.log、<档>_<GPU>_val.log、
#         <档>_<GPU>_gate_val.npz、<档>_<GPU>_train.done
#
# 前置：
#   * s1 档需要全量实例表（local_view/instances/train/，L_s1 标签面）：python ft_tsp3d/export_instances.py
#     （缺失时脚本会硬报错退出，防止 L_s1 静默跳过导致白跑；s2 档不需要实例表）
#   * distill 档需要全量 GD 侧车（local_view/gd/，默认已含全量；缺失硬报错）
#   * 架构档 M0–M4（架构消融）直接调 pipeline：python ft_tsp3d/pipeline.py --ablation M3 ...
# ==============================================================================
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PY:-/root/miniconda3/envs/vlfm/bin/python}"
CKPT="${CKPT:-$PROJECT_ROOT/data/tsp3d_models/tsp3d_scanrefer.pth}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/ft_datasets/local_view}"
WEIGHT_DIR="${WEIGHT_DIR:-$PROJECT_ROOT/data/tsp3d_ft}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/ft_tsp3d/ft_log}"
# 显卡编号（产物命名用）：自动从 nvidia-smi 提取型号数字（本机 3090 / 另一台 5080）；GPU_TAG 可覆盖
GPU_TAG="${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -oE '[0-9]{4}' | head -1)}"
GPU_TAG="${GPU_TAG:-nogpu}"
EPOCHS="${EPOCHS:-3}"
VAL_EVERY="${VAL_EVERY:-1}"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1   # 训练打印实时落盘/上屏（避免 stdout 缓冲导致“看起来没动静”）

# 显存碎片保险：expandable_segments 仅 torch>=2.1 支持（更旧 torch 会直接报
# "Unrecognized CachingAllocator option" 崩掉），因此先探测版本再设置；
# 已显式设置 PYTORCH_CUDA_ALLOC_CONF 时不动。
if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
    if "$PY" -c "import sys, torch; v = torch.__version__.split('+')[0].split('.'); sys.exit(0 if (int(v[0]), int(v[1])) >= (2, 1) else 1)" >/dev/null 2>&1; then
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    fi
fi

COMMON=(--data-dir "$DATA_DIR" --ckpt "$CKPT" --epochs "$EPOCHS" --val-every "$VAL_EVERY")
[ -n "${SCENES:-}" ] && COMMON+=(--scenes "$SCENES")
[ -n "${LIMIT:-}" ] && COMMON+=(--limit-views "$LIMIT")
[ -n "${VAL_SCENES:-}" ] && COMMON+=(--val-scenes "$VAL_SCENES")

# ---- 档位 → 损失开关 ----
flags_of() {
    case "$1" in
        base)    echo "" ;;
        gate)    echo "gate" ;;
        s1)      echo "s1" ;;
        s2)      echo "s2,tgp" ;;   # L_s2 含 TGP 子项（§8·4）
        distill) echo "distill" ;;
        *)       echo "__UNKNOWN__" ;;
    esac
}

run_arm() {
    local key="$1"
    local losses
    losses="$(flags_of "$key")"
    if [ "$losses" = "__UNKNOWN__" ]; then
        echo ">>> 未知档位: $key（可选 base / gate / s1 / s2 / distill）"
        return 1
    fi
    local tag="${key}_${GPU_TAG}"
    local donef="$LOG_DIR/${tag}_train.done"
    if [ -n "${SKIP_DONE:-}" ] && [ -f "$donef" ]; then
        echo ">>> 跳过 $key（已有 ${tag}_train.done）"
        return 0
    fi
    echo "================================================="
    echo ">>> 档位: $key  损失: ${losses:-base}  显卡: $GPU_TAG"
    echo ">>> 权重 -> $WEIGHT_DIR/${tag}_ckpt_*.pth ；日志 -> $LOG_DIR/${tag}_*"
    echo "================================================="
    mkdir -p "$WEIGHT_DIR" "$LOG_DIR"
    "$PY" ft_tsp3d/pipeline.py "${COMMON[@]}" \
        --out-dir "$LOG_DIR" \
        --weight-dir "$WEIGHT_DIR" --log-dir "$LOG_DIR" --tag "$tag" \
        --losses "$losses" --dump-gate "$LOG_DIR/${tag}_gate_val.npz" \
        2>&1 | tee "$LOG_DIR/${tag}_train_console.log"
    local status="${PIPESTATUS[0]}"
    if [ "$status" -eq 0 ]; then
        touch "$donef"
    else
        echo ">>> 档位 $key 退出码 $status —— 停止队列（不自动进入下一档；修复后 SKIP_DONE=1 可续跑）"
    fi
    return "$status"
}

# ---- 队列选择（二选一：TASK/TASKS 显式指定，或 MACHINE 预置清单） ----
if [ -n "${TASK:-}" ]; then
    QUEUE="$TASK"
elif [ -n "${TASKS:-}" ]; then
    QUEUE="$TASKS"
else
    case "${MACHINE:-}" in
        a|A|1) QUEUE="base gate s1" ;;
        b|B|2) QUEUE="base s2 distill" ;;
        *)     echo ""
               echo "用法：MACHINE=a|b bash scripts/ft_tsp3d.sh   （A: base→gate→s1；B: base→s2→distill）"
               echo "      TASK=<档> | TASKS=\"<档 档…>\" bash scripts/ft_tsp3d.sh"
               echo ""
               exit 1 ;;
    esac
fi

# HEAD=N：只跑当前队列前 N 档（先跑前几档、按输出再续后档；对 TASK/TASKS 同样生效）
if [ -n "${HEAD:-}" ]; then
    QUEUE="$(echo $QUEUE | cut -d' ' -f1-"$HEAD")"
fi

# ---- 前置检查（全量跑：s1 缺实例表 / distill 缺侧车 → 硬报错，防白跑；SCENES 冒烟过滤时跳过） ----
NEED_INST=0; NEED_GD=0
for k in $QUEUE; do
    case "$k" in s1) NEED_INST=1 ;; distill) NEED_GD=1 ;; esac
done
if [ -z "${SCENES:-}" ]; then
    N_MAIN=$(ls "$DATA_DIR"/train/*.npz 2>/dev/null | wc -l)
    if [ "$NEED_INST" = 1 ]; then
        N_INST=$(ls "$DATA_DIR"/instances/train/*.json 2>/dev/null | wc -l)
        if [ "$N_INST" -lt "$N_MAIN" ]; then
            echo "[error] s1 档需要全量实例表：instances/train = $N_INST/$N_MAIN —— 先跑："
            echo "        $PY ft_tsp3d/export_instances.py   （缺失时 L_s1 会静默跳过 = 白跑）"
            exit 1
        fi
    fi
    if [ "$NEED_GD" = 1 ]; then
        N_GD=$(ls "$DATA_DIR"/gd/train/*.npz 2>/dev/null | wc -l)
        if [ "$N_GD" -lt "$N_MAIN" ]; then
            echo "[error] distill 档需要全量 GD 侧车：gd/train = $N_GD/$N_MAIN —— 先补侧车后再跑"
            exit 1
        fi
    fi
else
    if [ "$NEED_INST" = 1 ] && [ ! -d "$DATA_DIR/instances/train" ]; then
        echo "[warn] 冒烟模式：实例表目录缺失（$DATA_DIR/instances/train），s1 项将被跳过"
    fi
fi

for KEY in $QUEUE; do
    run_arm "$KEY" || exit 1
done

echo ""
echo "================================================="
echo "队列完成（GPU $GPU_TAG）。"
echo "  权重：$WEIGHT_DIR/<档>_${GPU_TAG}_ckpt_last.pth（及 _ckpt_epoch<N>.pth）"
echo "  日志：$LOG_DIR/<档>_${GPU_TAG}_train.log / _train_console.log / _val.log / _gate_val.npz"
echo "τ_gate 校准（门控有效档：gate / distill）："
echo "  $PY ft_tsp3d/calibrate_gate.py --dump $LOG_DIR/<档>_${GPU_TAG}_gate_val.npz --target-recall 0.995"
echo "================================================="
