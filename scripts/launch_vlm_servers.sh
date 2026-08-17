#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.
# [Update 2026] Refactored for VLVM 3D Native Paradigm


export VLVM_PYTHON=${VLVM_PYTHON:-`which python`}
export TSP3D_DATA_PATH=${TSP3D_DATA_PATH:-/root/autodl-tmp/vlvm/data/tsp3d_models/}
export TSP3D_CHECKPOINT=${TSP3D_CHECKPOINT:-data/tsp3d_models/tsp3d_scanrefer.pth}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}
export TSP3D_PORT=${TSP3D_PORT:-12186}

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically
tmux split-window -h -t ${session_name}:0

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 "${VLVM_PYTHON} -m vlfm.vlm.tsp3d --port ${TSP3D_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "${VLVM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m

# 提示连接方法
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
