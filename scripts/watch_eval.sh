#!/bin/bash

# ==============================================================================
# hm3d 全量实验守护（watchdog）· 崩溃/卡死 -> 立即以当前代码重启全量
# ------------------------------------------------------------------------------
# 检测策略（60 s 心跳）：
#   (a) `python -m vlfm.run` 进程存活
#   (b) 日志出现 "Error executing job" / "IndexError:"（hydra 级崩溃标志）
#   (c) 进程在但日志停滞 > STALE_MIN 分钟（视作卡死）
#   (d) 完成判定：集数 >= 2000 -> 正常退出
# 动作：kill 残留 -> nohup 重启 `bash scripts/my_eval.sh hm3d`（自动生成新日志文件）
# 打点：每跨 50 集输出一条健康检查记录
#
# 启动（脱离会话）：
#   tmux new-session -d -s watch_eval "bash scripts/watch_eval.sh > outputs/watch_eval.log 2>&1"
# 查看：tail -f outputs/watch_eval.log   /   tmux attach -t watch_eval
# 停止（先停守护再手动停实验，避免与守护互搏）：
#   tmux kill-session -t watch_eval
# ==============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

CHECK_EVERY_EPS=50        # 健康检查打点间隔（集）
STALE_MIN=45              # 日志停滞判定（分钟）
HEARTBEAT=60              # 心跳（秒）
TOTAL_EPS=2000            # hm3d val 全量集数（完成判定）

log() { echo "[watch $(date '+%F %T')] $*"; }

PAT="m vlfm.run .*hm3d/v1/val"
last_mark=-1

log "守护启动：心跳 ${HEARTBEAT}s · 每 ${CHECK_EVERY_EPS} 集打点 · 停滞阈值 ${STALE_MIN}min"

while true; do
    LOG=$(ls -t outputs/eval_hm3d_*.log 2>/dev/null | head -1)
    if [ -z "$LOG" ]; then
        log "未找到 hm3d 日志，等待…"
        sleep "$HEARTBEAT"; continue
    fi
    N=$(grep -c 'Success rate:' "$LOG" 2>/dev/null || echo 0)

    if [ "$N" -ge "$TOTAL_EPS" ]; then
        log "实验完成（集数=$N），守护退出"
        exit 0
    fi

    PID=$(pgrep -f "$PAT" | head -1 || true)

    why=""
    if [ -z "$PID" ]; then why="进程不存在"; fi
    if grep -qE "Error executing job|IndexError: " "$LOG" 2>/dev/null; then
        why="${why:+$why; }日志崩溃标志"
    fi
    if [ -n "$PID" ] && [ -f "$LOG" ]; then
        AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
        if [ "$AGE" -gt $((STALE_MIN * 60)) ]; then
            why="${why:+$why; }日志停滞 ${AGE}s"
        fi
    fi

    if [ -n "$why" ]; then
        log "异常（$why）@ $(basename "$LOG")，集数=$N -> 立即重启"
        pkill -f "$PAT" 2>/dev/null || true
        sleep 10
        pkill -9 -f "$PAT" 2>/dev/null || true
        nohup bash scripts/my_eval.sh hm3d >/dev/null 2>&1 &
        sleep 30
        NEW=$(ls -t outputs/eval_hm3d_*.log 2>/dev/null | head -1)
        log "已重启：$(basename "$NEW")"
        last_mark=-1
        sleep "$HEARTBEAT"; continue
    fi

    MARK=$((N / CHECK_EVERY_EPS))
    if [ "$MARK" -gt "$last_mark" ]; then
        log "健康检查：集数=$N（检查点 $((MARK * CHECK_EVERY_EPS))）· pid=$PID 正常"
        last_mark=$MARK
    fi
    sleep "$HEARTBEAT"
done
