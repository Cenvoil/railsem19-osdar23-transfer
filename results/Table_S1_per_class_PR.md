# Table S1. Per-class precision and recall

Protocol: ignore-unlabeled pixels (GT = 255 excluded); predictions inside a labelled polygon of another class count as false positives. Values are IoU / precision / recall (%). The class mean over GT>0 classes is the `mIoU_mapped` column in the main tables.

## 5.1

| Config | track (rail-track) | train (on-rails) | signal (traffic-light) |
|---|---:|---:|---:|
| zero-shot | 7.43 / 99.97 / 7.43 | 55.92 / 99.34 / 56.12 | 0.89 / 20.72 / 0.92 |
| k=10 | 79.79 / 99.56 / 80.07 | 52.41 / 99.54 / 52.54 | 0.00 / NaN / 0.00 |
| k=30 | 83.97 / 99.42 / 84.39 | 40.82 / 99.63 / 40.89 | 3.24 / 3.72 / 20.18 |
| k=50 | 78.21 / 94.07 / 82.26 | 9.47 / 99.86 / 9.47 | 3.59 / 3.93 / 29.46 |
| k=100 | 76.69 / 85.09 / 88.60 | 0.00 / NaN / 0.00 | 3.80 / 4.32 / 23.92 |

## 3.1

| Config | track (rail-track) | train (on-rails) | signal (traffic-light) |
|---|---:|---:|---:|
| zero-shot | 25.94 / 100.00 / 25.94 | 0.00 / 0.00 / NaN | 14.22 / 100.00 / 14.22 |
| k=10 | 47.54 / 100.00 / 47.54 | 0.00 / 0.00 / NaN | 3.89 / 94.68 / 3.90 |
| k=30 | 64.31 / 100.00 / 64.31 | 0.00 / 0.00 / NaN | 37.46 / 74.29 / 43.04 |
| k=50 | 76.44 / 99.97 / 76.46 | NaN / NaN / NaN | 48.91 / 74.69 / 58.62 |
| k=100 | 94.73 / 99.04 / 95.61 | NaN / NaN / NaN | 52.76 / 85.91 / 57.76 |

Source: masks saved by `eval_cross_osdar.py`; configs correspond to Tables 3 and 9. `—` means the class has no GT pixels in that sequence.
