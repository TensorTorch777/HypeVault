# Final model freeze

Validation metrics were tied.
Calibration was used only as the predeclared secondary criterion.
Paired calibration bootstrap was used.
The final test had not yet been opened at selection time.

Selected model: v2_dinov3_cls_patch_attention
Selected checkpoint: /home/tensortorch26/Desktop/HypeVault/ml_rtx5080/experiments/v2_dinov3_cls_patch_attention/epoch_018.pt
Selected temperature: 0.24038200410185356
Production threshold: null

Validation safety first: false-authentic rate <= 0.02. The corrected validation classification metrics are tied, so the predeclared secondary criterion is paired bootstrap of calibrated calibration-set NLL. A model is materially better only when its mean calibrated NLL is lower and the 95% percentile interval for NLL_A - NLL_B lies entirely below zero. Otherwise the models are calibration-tied. Ties then use lower calibrated ECE, lower calibrated Brier, cls_only before cls_patch_attention, lower measured peak allocated VRAM, higher measured training throughput, then the architecture name. Final-test metrics are not a selection input.

Calibration survivors: v2_dinov2_cls_only, v2_dinov2_cls_patch_attention, v2_dinov3_cls_patch_attention
Tie-break applied: True
