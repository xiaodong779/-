#!/bin/bash
# monitor_train_and_shutdown.sh
# 监控 v3 训练进程（ucf101_discr_rwkv_split1_e50）。
# 当训练进程消失时——无论正常完成（日志含 [Training completed]）还是中断/崩溃——延迟若干秒后自动关机。
#
# 用法:
#   bash monitor_train_and_shutdown.sh [关机延迟秒数，默认 30]
#
# 取消监控（不关机）:
#   pkill -f monitor_train_and_shutdown.sh
#
# 注意: 只匹配 v3 的 checkpoint_dir，不会影响/误判其他训练进程。

SHUTDOWN_DELAY="${1:-30}"
LOG_FILE="/hy-tmp/Vita-CLIP/training/logs/ucf101_discr_rwkv_bs24_v3.log"
# 用 v3 特有的 checkpoint_dir 精确锁定训练进程，避免误伤其他训练
PROC_PATTERN="train_OPENAI_ALIGNED.py.*--checkpoint_dir checkpoints/ucf101_discr_rwkv_split1_e50"

log() { echo "$(date '+%F %T') [monitor] $*"; }

# 关机：兼容 shutdown / systemctl / poweroff
do_shutdown() {
    shutdown -h now 2>/dev/null \
        || systemctl poweroff 2>/dev/null \
        || poweroff 2>/dev/null \
        || log "关机命令执行失败，请手动关机"
}

log "开始监控训练进程，匹配模式: $PROC_PATTERN"
log "进程消失后 ${SHUTDOWN_DELAY} 秒自动关机"

# 防止脚本先于训练启动而误关机：先等待训练进程出现，最多等 10 分钟
WAITED=0
while ! pgrep -f "$PROC_PATTERN" > /dev/null 2>&1; do
    if [ "$WAITED" -ge 600 ]; then
        log "等待 10 分钟训练进程仍未出现，放弃监控（不关机）"
        exit 1
    fi
    log "训练进程尚未出现，等待中... (已等 ${WAITED}s)"
    sleep 30
    WAITED=$((WAITED + 30))
done
log "检测到训练进程，开始每 60 秒巡检一次"

# 主循环：进程存在则继续等
while pgrep -f "$PROC_PATTERN" > /dev/null 2>&1; do
    sleep 60
done

# 进程已消失
log "训练进程已消失"
if [ -f "$LOG_FILE" ] && grep -q "\[Training completed\]" "$LOG_FILE" 2>/dev/null; then
    log "日志含 [Training completed]，训练正常完成"
else
    log "日志未找到 [Training completed] 标记，判断为训练中断/崩溃"
fi

log "最后 15 行日志:"
tail -15 "$LOG_FILE" 2>/dev/null

log "${SHUTDOWN_DELAY} 秒后关机，如需取消请立即按 Ctrl+C 或执行: pkill -f monitor_train_and_shutdown.sh"
sleep "$SHUTDOWN_DELAY"

# 关机前最后确认：进程是否又出现（例如用户重启了训练）
if pgrep -f "$PROC_PATTERN" > /dev/null 2>&1; then
    log "检测到训练进程重新出现（可能已重启训练），取消关机"
    exit 0
fi

log "执行关机"
do_shutdown
