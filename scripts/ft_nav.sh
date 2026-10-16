#!/usr/bin/env bash
# 串行执行 N1 (VLVM V1) 和 N1 (ScanRefer)。
# N1：四场景 sigma grid（checkpoint 独立择点）。N0：固定 sigma_tar=0.70
# 两组共用场景与导航设置；S-penalty、
# S 冲突门、GD 全部关闭。
#
# 只检查计划：DRY_RUN=1 bash scripts/ft_nav_s4.sh
# 正式运行：bash scripts/ft_nav_s4.sh
# 断点续跑：SKIP_DONE=1 RUN_ROOT=<原运行目录> bash scripts/ft_nav_s4.sh
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
cd "$PROJECT_ROOT"

VLVM_V1_CKPT="${VLVM_V1_CKPT:-${R1E3_CKPT:-$PROJECT_ROOT/data/vlvm_v1/vlvm_v1.pth}}"
SCANREFER_CKPT="${SCANREFER_CKPT:-$PROJECT_ROOT/data/tsp3d_models/tsp3d_scanrefer.pth}"
# N1 sigma 区间按证据收缩：低端以 base 四场景导航 sweep 为准（<=0.50 明显劣，
# 0.50/0.55/0.60/0.65/0.70 的 SR=17.93/24.74/28.79/31.31/30.05），高端以 VLVM V1
# 的 E3 hit 分布为准（p95=0.684，故补一档 0.70；>0.70 召回崩塌）。
SIGMA_ARMS="${SIGMA_ARMS:-s050 s055 s060 s065 s070}"
N0_SIGMA_ARM="${N0_SIGMA_ARM:-s070}"
SCENES="${SCENES:-5cdEh9F2hJL TEEsavR23oF mv2HUxq3B53 wcojb4TFT35}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs}"
RUN_ROOT="${RUN_ROOT:-$OUT_DIR/ft_nav_s4_$(date +%Y-%m-%d_%H%M%S)}"

die() {
    echo ">>> [error] $*" >&2
    exit 1
}

if [ -z "${DRY_RUN:-}" ]; then
    [ -s "$VLVM_V1_CKPT" ] || die "VLVM V1 checkpoint 不存在: $VLVM_V1_CKPT"
    [ -s "$SCANREFER_CKPT" ] || die "ScanRefer checkpoint 不存在: $SCANREFER_CKPT"
fi

run_group() {
    local group="$1" ckpt="$2" run_dir="$3" arms="$4"
    echo "================================================="
    echo ">>> 启动 $group"
    echo ">>> checkpoint : $ckpt"
    echo ">>> sigma      : $arms"
    echo ">>> run dir    : $run_dir"
    echo "================================================="
    MODEL_TAG="$group" \
    MODEL_CKPT="$ckpt" \
    SIGMA_ARMS="$arms" \
    SCENES="$SCENES" \
    RUN_DIR="$run_dir" \
    REQUIRE_ALL=1 \
    FAIL_FAST=1 \
        bash "$HERE/ft_sigma_search.sh"
}

echo ">>> S4 Phase D 顺序: N1(VLVM V1) -> N0(ScanRefer)"
echo ">>> N1 跑 sigma grid; N0 固定 sigma_tar=0.70"
run_group n1_vlvm_v1 "$VLVM_V1_CKPT" "$RUN_ROOT/n1_vlvm_v1" "$SIGMA_ARMS"
run_group n0_scanrefer "$SCANREFER_CKPT" "$RUN_ROOT/n0_scanrefer" "$N0_SIGMA_ARM"

echo ">>> S4 Phase D 两组已完成: $RUN_ROOT"
