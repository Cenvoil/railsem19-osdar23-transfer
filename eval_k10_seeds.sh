#!/bin/bash
# Evaluate k=10 seed 1/2 on 5.1 and 3.1 (same protocol as eval_ft.sh).
ROOT="${ROOT:-$(pwd)}"
PY="${PYTHON:-python}"
cd $ROOT
ev() { # model seq imgsz out
  $PY eval_cross_osdar.py --model $1 --seq $2 \
      --data-root ${DATA_ROOT:-./data}/osdar \
      --img-sub $( [ "$2" = "5.1" ] && echo unpacked_51/rgb_highres_center || echo others/3_fire_site_3.1/rgb_highres_center ) \
      --labels $( [ "$2" = "5.1" ] && echo unpacked_51/5_station_bergedorf_5.1_labels.json || echo others/3_fire_site_3.1/3_fire_site_3.1_labels.json ) \
      --imgsz $3 --device 1 --montage 6 --out $ROOT/$4 2>&1 | tail -3
}
for s in 1 2; do
  echo "== ft_k10_s${s} 5.1 1024 =="
  ev $ROOT/runs/ft_k10_s${s}/weights/best.pt 5.1 1024 eval_cross_ftk10_s${s}_51.md
  echo "== ft_k10_s${s} 3.1 1024 =="
  ev $ROOT/runs/ft_k10_s${s}/weights/best.pt 3.1 1024 eval_cross_ftk10_s${s}_31.md
done
echo K10_EVAL_DONE
