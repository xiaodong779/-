#!/bin/bash
# Diving48 3D 双分支专家训练后台启动脚本
# 用法: bash run_train_3d.sh
set -u
cd /hy-tmp/AMS-Net-main

CONFIG=configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py
WORK_DIR=work_dirs/ams_r50_3d_1x1x8_diving48_rgb_EVL_STG_ADSTA_stage1234_bs8_lr4e-3_v1
LOG="$WORK_DIR/train_3d.log"
PIDFILE=train_3d.pid

mkdir -p "$WORK_DIR"

# 防重复启动
if [ -f "$PIDFILE" ]; then
  OLD_PID=$(cat "$PIDFILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ERROR: training already running (PID $OLD_PID), log=$LOG"
    exit 1
  else
    echo "stale PID file ($OLD_PID), removing"
    rm -f "$PIDFILE"
  fi
fi

# 注意：
#  - 不可加 --deterministic：会关闭 cudnn.benchmark，触发 cuDNN 在 ADSTA conv 上的
#    illegal memory access（V100 + torch1.11 实测）
#  - config 已设 cudnn_benchmark=True（train.py 会据此开启 benchmark）
#  - PYTHONPATH 必须含项目根：python tools/train.py 时 sys.path[0]=tools/，找不到 mmaction
PYTHONPATH=/hy-tmp/AMS-Net-main:$PYTHONPATH \
nohup /hy-tmp/envs/ams_net/bin/python tools/train.py "$CONFIG" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "training started: PID=$PID"
echo "log:  $LOG"
echo "cmd:  /hy-tmp/envs/ams_net/bin/python tools/train.py $CONFIG"
