#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.
# [Update 2026] Refactored for VLVM 3D Native Paradigm.
#
# Launch the ONLINE TSP3D + BLIP2-ITM model servers.

export HF_ENDPOINT=https://hf-mirror.com
export VLVM_PYTHON=${VLVM_PYTHON:-`which python`}
export TSP3D_DATA_PATH=${TSP3D_DATA_PATH:-/root/autodl-tmp/vlvm/data/tsp3d_models/}
export TSP3D_CHECKPOINT=${TSP3D_CHECKPOINT:-${TSP3D_DATA_PATH}/tsp3d_scanrefer.pth}
export TSP3D_PORT=${TSP3D_PORT:-12186}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}

# GD (grounding_dino.py) uses repo-relative paths for its config/weights.
cd "$(cd "$(dirname "$0")/.." && pwd)"

echo ">> TSP3D checkpoint : $TSP3D_CHECKPOINT"
echo ">> TSP3D port       : $TSP3D_PORT"
echo ">> BLIP2ITM port    : $BLIP2ITM_PORT"
echo ">> GroundingDINO    : $GROUNDING_DINO_PORT"

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically (3 panes: tsp3d | blip2itm | grounding-dino)
tmux split-window -h -t ${session_name}:0
tmux split-window -h -t ${session_name}:0

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 "${VLVM_PYTHON} -m vlfm.vlm.tsp3d --port ${TSP3D_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "${VLVM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "${VLVM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m

echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
echo "To stop: tmux kill-session -t ${session_name}"
echo "To switch weights, relaunch with: TSP3D_CHECKPOINT=<path to .pth> bash $0"
