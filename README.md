# RailSem19 → OSDaR23 zero-shot and k-shot transfer

Code, class mapping, run manifest, and derived tables for:

> **From RailSem19 to OSDaR23: Zero-Shot Transfer and k-Shot Adaptation for Railway Scene Understanding**
> Hai-Chuan Gan, Dan Tao — Beijing Jiaotong University

## What is included

- `eval_cross_osdar.py` — region-restricted L1 IoU evaluation on OSDaR23.
- `prep_osdar_ft.py`, `prep_osdar_ft_bal.py` — k-shot crop/mask preparation.
- `train_ft.sh`, `train_ft_seed.sh`, `eval_ft.sh`, `eval_k10_seeds.sh` — training/eval launchers.
- `YOLO_run_manifest.json` — source model, recipe, and checkpoint hashes.
- `class_mapping.csv` — OSDaR23 → RailSem19 mapping used in the paper.
- `results/` — CSV versions of Tables 2–11.
- `make_k_shot_figure.py`, `make_pipeline_figure.py` — figure scripts.

## Environment

`ultralytics==8.4.177`, `torch==2.14.1+cu130` (see `requirements.txt`).

## Quick start

```bash
bash train_ft.sh 30 0        # k=30, GPU 0
bash eval_ft.sh              # evaluate on 5.1/3.1
```

Set `DATA_ROOT` and `ROOT` if your layout differs. The scripts default to `./data`
and the current directory.

## Data and weights

RailSem19 and OSDaR23 are available from their providers under their respective
licenses. Raw images and annotations are not redistributed. YOLO weights are not
redistributed because of Ultralytics licensing; retrain from the manifest or use
your own checkpoint.

## License

The scripts are released under AGPL-3.0 because they depend on Ultralytics YOLO.
See `LICENSE`.
