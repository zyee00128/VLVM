#!/bin/bash

# ==============================================================================
# my_eval.sh (For VLVM) -- serial A/B comparison for the 2.5D semantic value plane (BEV)
# Purpose:
#   1. Enable NVIDIA EGL hardware-accelerated rendering to speed up simulation
#   2. Serially run 3D baseline / V1 (region) / V2 (surface) across h_lam values
#   3. Dataset fixed to HM3D 5cdEh9F2hJL (unchanged); logs saved as <tag>_<time>.log
# Note: start the VLM servers first (run launch_vlm_servers.sh in tmux)
# ==============================================================================

# ----------------- Project base path config -----------------
PROJECT_ROOT="/root/autodl-tmp/vlvm"
CONDA_ENV_NAME="vlfm"

# Fall back to the current directory if the project root is missing
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(pwd)
fi

cd "$PROJECT_ROOT"

# Load the conda environment
CONDA_PROFILE="/root/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate "$CONDA_ENV_NAME"
else
    echo "Warning: $CONDA_PROFILE not found, trying the default conda command..."
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
fi

# ----------------- GPU & EGL rendering environment variables -----------------
export CUDA_VISIBLE_DEVICES=0
export EGL_PLATFORM=surfaceless
export FORCE_GLX_USE_EGL=1
export MAGNUM_LOG=quiet
export MAGNUM_GPU_VALIDATION=OFF
export HF_ENDPOINT=https://hf-mirror.com
unset DISPLAY
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
set -o pipefail

echo "================================================="
echo "Environment variables configured!"
echo "Project path: $PROJECT_ROOT"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Rendering: NVIDIA EGL hardware acceleration"
echo "================================================="

# ----------------- Output dir & dataset -----------------
OUT_DIR="$PROJECT_ROOT/outputs/$(date +%Y-%m-%d)"
mkdir -p "$OUT_DIR"
SCENES="[5cdEh9F2hJL]"   # HM3D scene (dataset fixed, do not change)

# ----------------- Experiment definitions -----------------
# Format: tag | policy name | value_map_style | h_lam
# V1 (region) / V2 (surface) each run lambda = 0 / 0.3 / 0.6 / 1.0 (lambda=0 == VLFM equivalent)
# The 3D baseline reuses the "time optimization 2" result (99 eps, SR 0.1414 / 2:57:21), not rerun here
EXPERIMENTS=(
#   "v1_region_lam0|HabitatITMPolicyV1|region|0.0"
#   "v1_region_lam03|HabitatITMPolicyV1|region|0.3"
#   "v1_region_lam06|HabitatITMPolicyV1|region|0.6"
#   "v1_region_lam10|HabitatITMPolicyV1|region|1.0"
  "v2_surface_lam0|HabitatITMPolicyV2|surface|0.0"
  "v2_surface_lam03|HabitatITMPolicyV2|surface|0.3"
  "v2_surface_lam06|HabitatITMPolicyV2|surface|0.6"
  "v2_surface_lam10|HabitatITMPolicyV2|surface|1.0"
)

echo "Total ${#EXPERIMENTS[@]} experiments, running serially..."
echo "Log dir: $OUT_DIR"

# ----------------- Run all experiments serially -----------------
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r TAG POLICY STYLE H_LAM <<< "$exp"

    echo ""
    echo "==================== Experiment: $TAG ===================="
    echo "  Policy: $POLICY | Style: ${STYLE:-default} | lambda: ${H_LAM:-default}"

    args=(
        habitat.dataset.content_scenes="$SCENES"
        habitat_baselines.eval.split=val
        habitat_baselines.num_environments=1
        habitat.simulator.habitat_sim_v0.gpu_device_id=0
        habitat.simulator.habitat_sim_v0.gpu_gpu=False
        habitat_baselines.rl.policy.name="$POLICY"
    )
    if [ -n "$STYLE" ]; then
        args+=(habitat_baselines.rl.policy.value_map_style="$STYLE")
    fi
    if [ -n "$H_LAM" ]; then
        args+=(habitat_baselines.rl.policy.h_lam="$H_LAM")
    fi

    python -m vlfm.run "${args[@]}" 2>&1 \
        | tee "$OUT_DIR/eval_hm3d_${TAG}_$(date +%H%M%S).log"

    echo "===== Experiment $TAG done ====="
done

echo ""
echo "================================================="
echo "All ${#EXPERIMENTS[@]} experiments completed!"
echo "Log dir: $OUT_DIR"
echo "================================================="
