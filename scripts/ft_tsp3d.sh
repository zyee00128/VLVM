#!/usr/bin/env bash
# TSP3D HM3D Stage-1 fine-tune launcher (VLVM ft_pipeline).

# ---------- runtime knobs ----------
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export HF_ENDPOINT=https://hf-mirror.com

BASE_PTH=${BASE_PTH:-/root/autodl-tmp/vlvm/data/tsp3d_models/tsp3d_scanrefer.pth}
OUT_PTH=${OUT_PTH:-/root/autodl-tmp/vlvm/data/tsp3d_models/tsp3d_hm3d_ft.pth}
EPOCHS=${EPOCHS:-5}
CROP_RADIUS=${CROP_RADIUS:-6.0}
MAX_POINTS=${MAX_POINTS:-200000}
TEXT_STYLE=${TEXT_STYLE:-mixed}
LIMIT=${LIMIT:-0}
SEED=${SEED:-0}
FREEZE_SCOPE=${FREEZE_SCOPE:-prune}

[ -f "$BASE_PTH" ] || { echo "ERROR: base weights not found: $BASE_PTH (read-only, must exist)" >&2; exit 1; }

EXTRA=()
case "$FREEZE_SCOPE" in
    none|backbone|prune|output) ;;
    *) echo "ERROR: FREEZE_SCOPE must be none|backbone|prune|output (got '$FREEZE_SCOPE')" >&2; exit 1 ;;
esac
if [ "$FREEZE_SCOPE" != "none" ]; then EXTRA+=(--freeze_scope "$FREEZE_SCOPE"); fi
if [ "${ROTATE_Z:-1}" != "1" ]; then EXTRA+=(--no_rotate_z); fi
if [ "${ALIGN_ONLINE:-0}" = "1" ]; then EXTRA+=(--align_online); fi

echo "================================================="
echo ">>> TSP3D HM3D Stage-1 fine-tune"
echo ">>> base(ro) : $BASE_PTH"
echo ">>> output   : $OUT_PTH"
echo ">>> epochs   : $EPOCHS | crop : $CROP_RADIUS | max_points : $MAX_POINTS"
echo ">>> text     : $TEXT_STYLE | limit : $LIMIT | gpu : $CUDA_VISIBLE_DEVICES"
echo ">>> rotate_z : ${ROTATE_Z:-1} | align_online : ${ALIGN_ONLINE:-0} | freeze_scope : $FREEZE_SCOPE"
echo "================================================="

args=(
    --base_weights "$BASE_PTH"
    --output_weights "$OUT_PTH"
    --epochs "$EPOCHS"
    --crop_radius "$CROP_RADIUS"
    --max_points "$MAX_POINTS"
    --text_style "$TEXT_STYLE"
    --seed "$SEED"
)
if [ "$LIMIT" != "0" ]; then args+=(--limit "$LIMIT"); fi
args+=("${EXTRA[@]}")
args+=("$@")

python -m vlfm.ft_pipeline.train "${args[@]}"

echo ">>> done: weights at $OUT_PTH"
echo ">>> evaluate with (online servers):"
echo "    TSP3D_CHECKPOINT=$OUT_PTH bash ./scripts/launch_vlm_servers.sh"
