#!/bin/bash
# Evaluate fine-tuned k-shot models + 3.1 zero-shot@1024 baseline + confusion diag.
ROOT="${ROOT:-$(pwd)}"
PY="${PYTHON:-python}"
cd $ROOT
ev() { # model seq imgsz out
  $PY eval_cross_osdar.py --model $1 --seq $2 \
      --data-root ${DATA_ROOT:-./data}/osdar \
      --img-sub $( [ "$2" = "5.1" ] && echo unpacked_51/rgb_highres_center || echo others/3_fire_site_3.1/rgb_highres_center ) \
      --labels $( [ "$2" = "5.1" ] && echo unpacked_51/5_station_bergedorf_5.1_labels.json || echo others/3_fire_site_3.1/3_fire_site_3.1_labels.json ) \
      --imgsz $3 --device 6 --montage 6 --out $ROOT/$4 2>&1 | tail -3
}
echo "== 3.1 zero-shot 1024 (baseline) =="
ev $ROOT/runs/full_m/weights/best.pt 3.1 1024 eval_cross_31_full.md
echo "== ft_k10 5.1 1024 =="
ev $ROOT/runs/ft_k10/weights/best.pt 5.1 1024 eval_cross_ftk10_51.md
echo "== ft_k10 3.1 1024 =="
ev $ROOT/runs/ft_k10/weights/best.pt 3.1 1024 eval_cross_ftk10_31.md
echo "== ft_k30 5.1 1024 =="
ev $ROOT/runs/ft_k30/weights/best.pt 5.1 1024 eval_cross_ftk30_51.md
echo "== ft_k30 3.1 1024 =="
ev $ROOT/runs/ft_k30/weights/best.pt 3.1 1024 eval_cross_ftk30_31.md
echo "== ft_k30 5.1 2048 =="
ev $ROOT/runs/ft_k30/weights/best.pt 5.1 2048 eval_cross_ftk30_51_s2048.md
echo FT_EVAL_ALL_DONE
