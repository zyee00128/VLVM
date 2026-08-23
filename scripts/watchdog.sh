#!/bin/bash
# ==============================================================================
# watchdog.sh —— VLVM4 5.1 实验看门狗（用户睡眠期间自主运行）
# 用法: bash scripts/watchdog.sh <TAG> <min_episodes_for_early_check> <hours_to_run>
#   例: bash scripts/watchdog.sh st080 10 6
# 行为:
#   1) 监视最新 <TAG> 日志：组完成（出现 Average episode success）→ 输出 DONE 并退出(0)
#   2) 早期异常：跑满 min_episodes 集且 Success rate 为 0.00% → 输出 SUSPECT 并退出(0)
#   3) 超时保护：超过 hours_to_run 小时 → 输出 TIMEOUT 并退出(0)
# 任何输出都代表"需要处理"，退出码 0 供 async 终端通知。
# ==============================================================================
LOG_DIR="/root/autodl-tmp/vlvm/outputs/VLVM-V4"
TAG="${1:-st080}"
MIN_EP="${2:-10}"
MAX_HOURS="${3:-6}"

# 找最新日志（同一 TAG 可能有多个，取最新）
LOG=$(ls -t "$LOG_DIR"/eval_hm3d_*_${TAG}_*.log 2>/dev/null | head -1)
if [ -z "$LOG" ]; then
    echo "ERROR: no log found for tag $TAG"
    exit 0
fi
echo "[watchdog] $TAG -> $LOG (early-check@${MIN_EP}eps, timeout=${MAX_HOURS}h)"

START=$(date +%s)
while true; do
    NOW=$(date +%s)
    ELAPSED_H=$(( (NOW - START) / 3600 ))
    if [ "$ELAPSED_H" -ge "$MAX_HOURS" ]; then
        echo "TIMEOUT: $TAG ran ${ELAPSED_H}h without completing"
        exit 0
    fi
    # 组完成？
    if grep -q "Average episode success" "$LOG" 2>/dev/null; then
        echo "DONE: $TAG completed"
        exit 0
    fi
    # 早期异常：已跑满 MIN_EP 集但成功率 0%
    EPISODES=$(grep -c "Success rate:" "$LOG" 2>/dev/null)
    if [ "$EPISODES" -ge "$MIN_EP" ]; then
        SR=$(grep "Success rate:" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+%")
        if [ "$SR" = "0.00%" ]; then
            echo "SUSPECT: $TAG has $SR success after $EPISODES episodes"
            exit 0
        fi
        # 若已跑满 MIN_EP 且成功>0，说明可用，改为只等完成（放宽异常检测）
        # 无需处理，继续等 DONE
    fi
    sleep 300
done
