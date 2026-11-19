#!/bin/bash
# Fine-tune launcher for OSDaR23 k-shot adaptation experiments.
# Usage: bash train_ft.sh <k> <device> [epochs]
set -u
K="${1:?k}"; DEV="${2:-6}"; EP="${3:-40}"
PY="${PYTHON:-python}"
YOLO_BIN="$(dirname "$PY")/yolo"
ROOT="${ROOT:-$(pwd)}"
DATA="${DATA_ROOT:-./data}/osdar_ft/k${K}/osdar_ft.yaml"
INIT=$ROOT/runs/full_m/weights/best.pt
NAME="ft_k${K}"

[ -f "$DATA" ] || { echo "missing $DATA"; exit 1; }
[ -f "$INIT" ] || { echo "missing $INIT"; exit 1; }

cd "$ROOT"
setsid nohup "$YOLO_BIN" semantic train \
  model="$INIT" data="$DATA" \
  epochs="$EP" batch=4 imgsz=1024 device="$DEV" \
  lr0=0.003 lrf=0.01 warmup_epochs=1 \
  project="$ROOT/runs" name="$NAME" exist_ok=True \
  seed=0 workers=4 plots=True cache=False \
  > "$ROOT/logs/${NAME}.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$ROOT/logs/${NAME}.pid"
sleep 25
if kill -0 "$PID" 2>/dev/null; then
  echo "LAUNCHED $NAME pid=$PID device=$DEV"
else
  echo "START FAILED"; tail -25 "$ROOT/logs/${NAME}.log"; exit 1
fi
