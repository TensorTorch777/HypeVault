# Phase 36 watch-like open-set evaluation

WATCH_OOD_UNRESOLVED

Phase 35 Roboflow derivatives were excluded. Commons original files were scored with the frozen FP32 checkpoint.
No threshold was selected on this holdout.

## Reasons

- the independent holdout does not meet the pre-registered minimum
- holdout unseen luxury is 14, below 60
- fewer than 4 unseen brands have 8 holdout images
- holdout ordinary watches are 10, below 40
- holdout smart/digital watches are 1, below 25

## Holdout escape

- unseen_luxury: 14 images, authentic escape 0.7857142857142857
- ordinary_watch: 10 images, authentic escape 0.3
- smart_digital: 1 images, authentic escape 1.0
- wrist_store: 0 images, authentic escape 0.0
- overall: 0.6

## Phase 35 comparison

- unseen_luxury: phase35 0.8814432989690721 (194), phase36 0.7857142857142857 (14), difference -0.09572901325478644
- ordinary_watch: phase35 0.6818181818181818 (22), phase36 0.3 (10), difference -0.3818181818181818
- smart_digital: phase35 0.6956521739130435 (23), phase36 1.0 (1), difference 0.30434782608695654
- wrist_store: phase35 0.7777777777777778 (18), phase36 None (0), difference None
- overall_watch_like: phase35 0.8404669260700389 (257), phase36 0.6 (25), difference -0.24046692607003894

The final test was not opened. The checkpoint, temperature, and policy were not changed.
