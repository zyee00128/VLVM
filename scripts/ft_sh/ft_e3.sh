#!/usr/bin/env bash
# E3 离线同源检测自动化：依次评测并自动生成汇总与对比。
# 不启动 VLM 服务、不运行 Habitat；直接加载 checkpoint 调用在线同源 TSP3D.predict。
#
# DRY_RUN=1 bash scripts/ft_e3.sh
# ARMS="ref u0 u1" VIEWS_CAP=50 bash scripts/ft_e3.sh
# SKIP_DONE=1 RUN_DIR=/abs/path/e3_unfreeze_xxx bash scripts/ft_e3.sh
# CKPT_U0=/abs/path/custom.pth ARMS="ref u0" bash scripts/ft_e3.sh

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

CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/data/tsp3d_ft/unfreeze}"
REF_CKPT="${REF_CKPT:-$PROJECT_ROOT/data/tsp3d_models/tsp3d_scanrefer.pth}"
BASE_CKPT="${BASE_CKPT:-$PROJECT_ROOT/data/tsp3d_ft/base_3090_ckpt_last.pth}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/ft_tsp3d/ft_log}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/e3_unfreeze_$(date +%Y-%m-%d_%H%M%S)}"
ARMS="${ARMS:-ref base u0 u1 u2 u3}"
REFERENCE_ARM="${REFERENCE_ARM:-ref}"
SECOND_REFERENCE_ARM="${SECOND_REFERENCE_ARM:-base}"
SPLIT="${SPLIT:-val}"
NEG_PER_VIEW="${NEG_PER_VIEW:-1}"
VIEWS_CAP="${VIEWS_CAP:-0}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

GPU_TAG="${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -oE '[0-9]{4}' | head -1 || true)}"
GPU_TAG="${GPU_TAG:-nogpu}"

die() {
    echo ">>> [error] $*" >&2
    exit 1
}

canonical_file() {
    realpath -e -- "$1" 2>/dev/null
}

arm_override() {
    local upper var
    upper="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
    var="CKPT_${upper}"
    printf '%s' "${!var:-}"
}

gate_threshold_override() {
    local upper var
    upper="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
    var="GATE_THRESHOLD_${upper}"
    printf '%s' "${!var:-}"
}

resolve_ckpt() {
    local arm="$1" override exact matches count
    override="$(arm_override "$arm")"
    if [ -n "$override" ]; then
        canonical_file "$override"
        return
    fi
    case "$arm" in
        ref) canonical_file "$REF_CKPT"; return ;;
        base) canonical_file "$BASE_CKPT"; return ;;
    esac
    exact="$CKPT_DIR/${arm}_${GPU_TAG}_ckpt_last.pth"
    if [ -f "$exact" ]; then
        canonical_file "$exact"
        return
    fi
    matches="$(find "$CKPT_DIR" -maxdepth 1 -type f -name "${arm}_*_ckpt_last.pth" -print 2>/dev/null | sort)"
    count="$(printf '%s\n' "$matches" | grep -c . || true)"
    if [ "$count" -eq 1 ]; then
        canonical_file "$matches"
        return
    fi
    if [ "$count" -gt 1 ]; then
        echo ">>> [error] $arm 匹配多个 checkpoint，请设置 CKPT_${arm^^}:" >&2
        printf '%s\n' "$matches" >&2
    fi
    return 1
}

verify_checkpoint_profile() {
    local arm="$1" ckpt="$2"
    case "$arm" in u0|u1|u2|u3) ;; *) return 0 ;; esac
    "$PY" - "$arm" "$ckpt" <<'PY'
import sys, torch
expected, path = sys.argv[1:]
try:
    obj = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    obj = torch.load(path, map_location="cpu")
args = obj.get("args", {}) if isinstance(obj, dict) else {}
actual = str(args.get("unfreeze_profile", ""))
if actual != expected:
    raise SystemExit(f"checkpoint profile mismatch: expected={expected} actual={actual or '<missing>'} path={path}")
PY
}

summary_valid() {
    local file="$1"
    "$PY" - "$file" <<'PY'
import json, sys
try:
    d=json.load(open(sys.argv[1]))
    calls=int(d.get("n_calls",0))
    disc=d.get("discrimination") or {}
    has_disc=(int(disc.get("n",-1)) == calls and "auroc" in disc and
              "auprc" in disc and "calibration" in disc and "pair" in disc)
    has_gate=(not d.get("gate_enabled",False) or isinstance(d.get("presence_gate"),dict))
    ok=(int(d.get("views",0))>0 and calls>0 and int(d.get("failures",1))==0 and
        has_disc and has_gate)
except Exception:
    ok=False
raise SystemExit(0 if ok else 1)
PY
}

preflight() {
    [ -x "$PY" ] || die "Python 不可执行: $PY"
    "$PY" -m py_compile ft_tsp3d/eval_tools/diag_offline.py ft_tsp3d/data_loader.py vlfm/vlm/tsp3d.py
    "$PY" -c "import torch, MinkowskiEngine; assert torch.cuda.is_available(); print('[preflight] torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0))"
    local arm ckpt
    for arm in $ARMS; do
        ckpt="$(resolve_ckpt "$arm")" || die "找不到 $arm checkpoint"
        verify_checkpoint_profile "$arm" "$ckpt"
        echo ">>> $arm -> $ckpt"
    done
    echo ">>> E3 参数: split=$SPLIT neg_per_view=$NEG_PER_VIEW views_cap=$VIEWS_CAP scenes=${SCENES:-<all>}"
    echo ">>> 输出目录: $RUN_DIR"
}

run_arm() {
    local arm="$1" ckpt summary log gate_threshold
    ckpt="$(resolve_ckpt "$arm")" || die "找不到 $arm checkpoint"
    summary="$RUN_DIR/${arm}_summary.json"
    log="$RUN_DIR/${arm}.log"
    if [ -n "${SKIP_DONE:-}" ] && [ -f "$summary" ] && summary_valid "$summary"; then
        echo ">>> [skip] $arm 已有完整 summary"
        return 0
    fi

    local -a cmd
    cmd=(
        "$PY" -u ft_tsp3d/eval_tools/diag_offline.py
        --ckpt "$ckpt"
        --tag "$arm"
        --out-dir "$RUN_DIR"
        --split "$SPLIT"
        --neg-per-view "$NEG_PER_VIEW"
        --views-cap "$VIEWS_CAP"
        --device "$DEVICE"
    )
    gate_threshold="$(gate_threshold_override "$arm")"
    [ -n "$gate_threshold" ] && cmd+=(--gate-threshold "$gate_threshold")
    [ -n "${SCENES:-}" ] && cmd+=(--scenes "$SCENES")
    printf '>>> command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    if [ -n "${DRY_RUN:-}" ]; then
        return 0
    fi
    mkdir -p "$RUN_DIR"
    set +e
    "${cmd[@]}" 2>&1 | tee "$log"
    local status="${PIPESTATUS[0]}"
    set -e
    [ "$status" -eq 0 ] || return "$status"
    summary_valid "$summary" || die "$arm E3 输出不完整或存在推理失败: $summary"
}

write_manifest() {
    local manifest="$RUN_DIR/checkpoints.tsv" arm ckpt
    printf 'arm\tcheckpoint\tsha256\n' > "$manifest"
    for arm in $ARMS; do
        ckpt="$(resolve_ckpt "$arm")"
        printf '%s\t%s\t%s\n' "$arm" "$ckpt" "$(sha256sum "$ckpt" | awk '{print $1}')" >> "$manifest"
    done
}

write_summary() {
    "$PY" - "$RUN_DIR" $ARMS <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
arms=sys.argv[2:]
cols=[
    "arm","views","pos","neg","hit025","hit050","mean_iou",
    "hit_score_mean","neg_score_mean","hit_ge070","neg_ge070",
    "n_correct","correct_rate","correct_score_mean","correct_score_sd",
    "error_score_mean","error_score_sd","auroc","auprc",
    "threshold_r90","recall_r90","fpr_tpr90","precision_r90",
    "threshold_r95","recall_r95","fpr_tpr95","precision_r95",
    "pair_n","pair_acc","violation_rate","violation_gap_mean",
    "violation_gap_median","score_gap","logit_gap","cohen_d","logit_cohen_d",
    "ece15","brier","nll","gate_threshold","gate_auroc","gate_auprc",
    "gate_pos_mean","gate_neg_mean","gate_recall_open","gate_false_close",
    "gate_intercept","gate_target_absent_fpr","failures",
]
lines=["\t".join(cols)]
for arm in arms:
    d=json.loads((root/f"{arm}_summary.json").read_text())
    p,n=d["pos"],d["neg"]
    hs=p.get("score_hit",{})
    ns=n.get("score",{})
    disc=d.get("discrimination") or {}
    r90=disc.get("at_tpr90") or {}
    r95=disc.get("at_tpr95") or {}
    pair=disc.get("pair") or {}
    cal=disc.get("calibration") or {}
    correct=disc.get("correct_score") or {}
    error=disc.get("error_score") or {}
    gate=d.get("presence_gate") or {}
    gate_at=gate.get("at_threshold") or {}
    gate_pos=gate.get("positive_score") or {}
    gate_neg=gate.get("negative_score") or {}
    row=[
        arm,d.get("views"),d.get("n_pos"),d.get("n_neg"),
        p.get("hit025_pct"),p.get("hit050_pct"),p.get("mean_best_iou"),
        hs.get("mean","-"),ns.get("mean","-"),hs.get("ge070_pct","-"),
        ns.get("ge070_pct","-"),disc.get("n_correct","-"),disc.get("correct_rate","-"),
        correct.get("mean","-"),correct.get("sd","-"),error.get("mean","-"),
        error.get("sd","-"),disc.get("auroc","-"),disc.get("auprc","-"),
        r90.get("threshold","-"),r90.get("recall","-"),r90.get("fpr","-"),
        r90.get("precision","-"),r95.get("threshold","-"),r95.get("recall","-"),
        r95.get("fpr","-"),r95.get("precision","-"),pair.get("n_pairs","-"),
        pair.get("accuracy","-"),pair.get("violation_rate","-"),
        pair.get("violation_gap_mean","-"),pair.get("violation_gap_median","-"),
        disc.get("score_gap","-"),disc.get("logit_gap","-"),disc.get("cohen_d","-"),
        disc.get("logit_cohen_d","-"),cal.get("ece","-"),cal.get("brier","-"),
        cal.get("nll","-"),d.get("gate_threshold","-"),gate.get("auroc","-"),
        gate.get("auprc","-"),gate_pos.get("mean","-"),gate_neg.get("mean","-"),
        gate_at.get("recall_open","-"),gate_at.get("false_close","-"),
        gate_at.get("intercept","-"),gate_at.get("target_absent_fpr","-"),
        d.get("failures"),
    ]
    lines.append("\t".join("-" if value is None else str(value) for value in row))
(root/"summary.tsv").write_text("\n".join(lines)+"\n")
PY
}

compare_all() {
    local ref="$RUN_DIR/${REFERENCE_ARM}_summary.json" arm dst second
    [ -f "$ref" ] || die "reference summary 不存在: $ref"
    for arm in $ARMS; do
        [ "$arm" = "$REFERENCE_ARM" ] && continue
        dst="$RUN_DIR/compare_${arm}_vs_${REFERENCE_ARM}.txt"
        "$PY" -u ft_tsp3d/eval_tools/diag_offline.py --compare "$RUN_DIR/${arm}_summary.json" "$ref" | tee "$dst"
    done
    second="$RUN_DIR/${SECOND_REFERENCE_ARM}_summary.json"
    if [ -f "$second" ] && [ "$SECOND_REFERENCE_ARM" != "$REFERENCE_ARM" ]; then
        for arm in $ARMS; do
            case "$arm" in "$REFERENCE_ARM"|"$SECOND_REFERENCE_ARM") continue ;; esac
            dst="$RUN_DIR/compare_${arm}_vs_${SECOND_REFERENCE_ARM}.txt"
            "$PY" -u ft_tsp3d/eval_tools/diag_offline.py --compare "$RUN_DIR/${arm}_summary.json" "$second" | tee "$dst"
        done
    fi
}

preflight
if [ -n "${DRY_RUN:-}" ]; then
    for arm in $ARMS; do run_arm "$arm"; done
    echo ">>> DRY_RUN 完成，未执行 E3。"
    exit 0
fi

mkdir -p "$RUN_DIR"
write_manifest
for arm in $ARMS; do
    run_arm "$arm" || die "$arm E3 失败，队列停止"
done
write_summary
compare_all
echo ">>> E3 队列完成: $RUN_DIR"
echo ">>> 汇总: $RUN_DIR/summary.tsv"
