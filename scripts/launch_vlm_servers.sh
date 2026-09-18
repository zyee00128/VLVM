#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.
# [Update 2026] Refactored for VLVM 3D Native Paradigm.
#
# Launch the ONLINE TSP3D / Grounding-DINO / BLIP2-ITM model servers.

export HF_ENDPOINT=https://hf-mirror.com
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export VLVM_PYTHON=${VLVM_PYTHON:-$(command -v python)}
export TSP3D_DATA_PATH=${TSP3D_DATA_PATH:-${PROJECT_ROOT}/data/tsp3d_models}
# 标准权重 = tsp3d_scanrefer.pth；
# 对比实验：TSP3D_CHECKPOINT=<path> bash $0 手动覆盖
# ckpt_sr3d.pth = ScanRefer 版对照    ckpt_nr3d.pth = NR3D 版对照
export TSP3D_CHECKPOINT=${TSP3D_CHECKPOINT:-${TSP3D_DATA_PATH}/tsp3d_scanrefer.pth}
export TSP3D_PORT=${TSP3D_PORT:-12186}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}

# GD (grounding_dino.py) uses repo-relative paths for its config/weights.
cd "$PROJECT_ROOT"

# --- 端口占用检查 ---
_ports="${TSP3D_PORT} ${BLIP2ITM_PORT} ${GROUNDING_DINO_PORT}"
_busy=""
for _p in $_ports; do
    if timeout 2 bash -c ":</dev/tcp/127.0.0.1/$_p" 2>/dev/null; then
        _busy="$_busy $_p"
    fi
done
if [ -n "$_busy" ]; then
    echo "错误: 端口已被占用:$_busy"
    echo "  现存会话: $(tmux ls 2>/dev/null | tr '\n' ' ')"
    echo "  先停旧服务: tmux kill-session -t <会话名>   （或指定其它端口: TSP3D_PORT=... bash $0）"
    exit 1
fi

echo ">> TSP3D checkpoint : $TSP3D_CHECKPOINT"
echo ">> TSP3D port       : $TSP3D_PORT"
echo ">> BLIP2ITM port    : $BLIP2ITM_PORT"
echo ">> GroundingDINO    : $GROUNDING_DINO_PORT"

# --- LAVIS BERT 目录：按在位探测；两机布局不同，缺失时不注入该变量 ---
LAVIS_BERT_DIR=""
for _d in "${PROJECT_ROOT}/data/bert-base-uncased" "${PROJECT_ROOT}/data/tsp3d_models/bert-base-uncased"; do
    [ -d "$_d" ] && { LAVIS_BERT_DIR="$_d"; break; }
done
BERT_ENV=""
[ -n "$LAVIS_BERT_DIR" ] && BERT_ENV="LAVIS_BERT_MODEL_PATH='${LAVIS_BERT_DIR}' "
[ -n "$LAVIS_BERT_DIR" ] && echo ">> LAVIS BERT dir   : $LAVIS_BERT_DIR"

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically (3 panes: tsp3d | blip2itm | grounding-dino)
tmux split-window -h -t ${session_name}:0
tmux split-window -h -t ${session_name}:0

# Run commands in each pane.
tmux send-keys -t ${session_name}:0.0 \
  "cd '${PROJECT_ROOT}' && HF_ENDPOINT='${HF_ENDPOINT}' TSP3D_DATA_PATH='${TSP3D_DATA_PATH}' TSP3D_CHECKPOINT='${TSP3D_CHECKPOINT}' ${VLVM_PYTHON} -m vlfm.vlm.tsp3d --port ${TSP3D_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 \
  "cd '${PROJECT_ROOT}' && HF_ENDPOINT='${HF_ENDPOINT}' ${BERT_ENV}${VLVM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 \
  "cd '${PROJECT_ROOT}' && ${VLVM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m

# --- 就绪等待：轮询三个端口，避免"脚本返回成功但服务仍在加载/已崩"的静默失败 ---
echo ">>> 等待三服务就绪（最多 150 s）..."
_ready=0
for _i in $(seq 1 30); do
    sleep 5
    _n=0
    for _p in $_ports; do
        timeout 2 bash -c ":</dev/tcp/127.0.0.1/$_p" 2>/dev/null && _n=$((_n + 1))
    done
    if [ "$_n" -eq 3 ]; then _ready=1; break; fi
done
if [ "$_ready" -eq 1 ]; then
    echo ">>> 三服务已就绪（${TSP3D_PORT} / ${BLIP2ITM_PORT} / ${GROUNDING_DINO_PORT}）"
else
    echo ">>> 警告: 150 s 内未全部就绪，检查各面板: tmux attach-session -t ${session_name}"
fi

echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
echo "To stop: tmux kill-session -t ${session_name}"
echo "对比实验权重（手动指定覆盖标准 tsp3d_scanrefer.pth）:"
echo "  TSP3D_CHECKPOINT=${TSP3D_DATA_PATH}/ckpt_sr3d.pth bash $0   # ScanRefer 版"
echo "  TSP3D_CHECKPOINT=${TSP3D_DATA_PATH}/ckpt_nr3d.pth bash $0   # NR3D 版"
