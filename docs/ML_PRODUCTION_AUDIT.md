# HypeVault ML production audit

Phase 0 inspection only. No training or production inference code was changed.

Audit date: 2026-10-04. Hardware present: NVIDIA GeForce RTX 5080, 16303 MiB.

## Current architecture

Two training paths exist and must stay separate until a replacement is verified.

| Path | Backbone | Resolution | Head input |
|---|---|---|---|
| `ml/train.py`, `ml/train_staged.py` | DINOv2-Giant (`dinov2_vitg14_reg` / timm `vit_giant_patch14_reg4_dinov2.lvd142m`) | 518×518 | CLS token |
| `ml_rtx5080/train.py` | DINOv2-Base (`vit_base_patch14_dinov2.lvd142m`) | 504×504 | CLS token |

The RTX 5080 classifier in `ml_rtx5080/train.py` is:

```
DINOv2 backbone
  → CLS token
  → LayerNorm
  → Dropout(p=dropout)          # default 0.2
  → Linear(embed_dim → 512)
  → GELU
  → Dropout(p=dropout/2)        # default 0.1
  → Linear(512 → 1)             # raw fake logit
```

`backend/inference/dinov2_model.py` copies that head. It does not match the shorter “LayerNorm → Linear 512 → GELU → Dropout → Linear 1” description: both dropouts are present.

The backbone forward uses the CLS token only. Patch tokens are discarded. Register tokens are not read explicitly; hub dict outputs use `x_norm_clstoken`.

Existing checkpoints were not modified:

- `ml_rtx5080/checkpoints/` — 7 epochs, best val F1 at epoch 2 (0.9935). Split still names a sneaker folder (`Label_1_Sneakers/Louis Vuitton`). Treat as historical, not the watches contract.
- `ml_rtx5080/checkpoints_watches/` — 11 epochs, best val F1 at epoch 6 (0.9903). Watches-only. No held-out `test` block in `history.json`.
- `ml/checkpoints/` — Giant / staged checkpoints. Leave untouched.

There is no project test suite. `upload_min_side_px` (400) is defined in `backend/config.py` and is not enforced in `backend/inference/routes.py`.

## Dataset pipeline

On disk today:

| Folder | Label | Brand directories | Images |
|---|---|---|---|
| `Label_0_Watches` | 0 AUTHENTIC | 5 | 15,000 |
| `Label_1_Watches` | 1 DEEPFAKE | 5 | 15,000 |

Brands, identical on both labels: A. Lange & Söhne, Audemars Piguet, Patek Philippe, Richard Mille, Vacheron Constantin. Each brand directory holds 3,000 images directly. There are no nested product, SKU, seller, or marketplace folders.

`load_hypevault_samples()` reads those two folders only. Corrupt files are replaced with a flat gray 224×224 image and still trained. That hides bad data inside the loss.

Available metadata:

| Field | Source |
|---|---|
| `label` | parent label folder |
| `brand` | immediate parent directory name |
| `image_path` | file path |
| `product_id` | not present |
| `marketplace` | not present |
| `seller` / source | not present |

Do not invent product, seller, or marketplace ids. Missing fields must stay `UNKNOWN`.

## Split strategy

`split_indices_by_product()` groups by `path.parent`. For this dataset that parent is the brand directory, not a product. The watches run is a brand-folder split:

- `checkpoints_watches/split_manifest.json`: 24,000 train images / 8 folders, 6,000 val images / 2 folders.
- Validation folders: `Label_0_Watches/Vacheron Constantin` and `Label_1_Watches/Richard Mille`.
- Those two folders are different brands and different classes. Validation is not a paired authentic/fake holdout of the same brand, and it is not a within-brand product holdout.
- `verify_product_split()` checks folder-path and file-path overlap only. It cannot detect source leakage because no source field exists.
- There is no `final_holdout` split. `--test_root` is optional and was null for both RTX runs. `history.json` has no test metrics.
- The older `ml_rtx5080/checkpoints/split_manifest.json` still lists sneaker folders (60,000 images, 20 folders). It does not describe the current watches tree.

## Model loading

Training tries `torch.hub` names first, then timm. The RTX default `vit_base_patch14_dinov2.lvd142m` is a timm id, so it loads through timm with `num_classes=0` and `dynamic_img_size=True`. Gradient checkpointing is enabled when the backbone exposes it.

Backend `DINOv2Classifier._load_backbone()` maps only Giant and Large hub/timm names. A timm id such as `vit_base_patch14_dinov2.lvd142m` falls through to `timm.create_model`. The default setting is still Giant:

```
dinov2_model_name = "dinov2_vitg14_reg"
inference_img_size = 518
```

`local_torch.py` loads checkpoints with `strict=False` and only logs a warning when keys are missing or unexpected. A Base/504 checkpoint can be attached to a Giant/518 module without a hard failure.

## Classifier head

See the architecture section. Output is a single raw logit. Training loss is `BCEWithLogitsLoss`. Sigmoid is applied only at metric and verdict time.

## Augmentation

Training (`build_transforms`, train=True), all stronger than the forensic starting point:

| Transform | Current |
|---|---|
| RandomResizedCrop | scale 0.55–1.0, ratio 0.8–1.25 |
| Horizontal flip | p=0.5 |
| Rotation | ±15° |
| Perspective | distortion 0.2, p=0.3 |
| ColorJitter | brightness/contrast 0.35, saturation 0.25, hue 0.08 |
| RandomGrayscale | p=0.03 (enabled) |
| GaussianBlur | kernel 7, sigma 0.1–3.0, p=0.2 |
| JPEG | quality 40–90, p=0.3 |
| MixUp | alpha 0.2, always on when `mixup_alpha > 0` |

Validation and the optional test loader use `Resize(img_size+16)` then `CenterCrop(img_size)`, bicubic, then ImageNet normalization:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## Loss

`BCEWithLogitsLoss(pos_weight=None)`.

`label_smoothing=0.05` is stored in `DEFAULTS` and in both RTX `config.json` files. It is never passed to the loss. Effective label smoothing is 0.

## Optimizer

AdamW, betas `(0.9, 0.999)`, eps `1e-8`.

Two parameter groups, not per-block decay:

| Group | Learning rate | Weight decay |
|---|---|---|
| every non-head parameter | `lr` (2e-5 in the RTX configs) | 0.05 |
| parameters whose name contains `head` | `lr * 10` (2e-4) | 0.0 |

Gradient clipping is 1.0, applied only on the batches that call `optimizer.step()`.

## Scheduler

`LambdaLR` cosine with linear warmup. `scheduler.step()` runs once per optimizer step.

`steps_per_epoch = len(train_loader) // accumulate_grad` uses floor division. A remainder micro-batch never steps, and the scheduler’s `total_steps` / warmup count ignore that remainder. Resume fast-forward uses the same floor count.

## Checkpoint selection

Best checkpoint is the highest validation F1 (`val_metrics["f1"] > best_val_f1`). Accuracy is recorded beside it and is not the selection key. False-authentic rate, PR-AUC, and ROC-AUC are not selection criteria. Plots compute ROC-AUC and PR-AUC only for the F1-winning epoch, at threshold 0.5.

Early stopping patience is 5 epochs without an F1 improvement (`early_stop_patience`). Epoch checkpoints are written every `save_every` (2) epochs. `final_model.pt` is the last epoch, not the selected checkpoint.

## Metrics

`compute_metrics()` thresholds sigmoid output at 0.5. It reports accuracy, precision, recall, and F1 for the positive class (deepfake = 1). It does not report ROC-AUC, PR-AUC, FPR, FNR, authentic recall, or false-authentic rate in `history.json`.

`confidence_threshold` (default 0.5) is parsed and written to `config.json` and is not used by training or validation.

Train metrics are computed on MixUp-softened labels after `.round()`, so train accuracy/F1 are not clean-image metrics.

## Preprocessing mismatch

| Stage | Resize | Normalize |
|---|---|---|
| Train aug | RandomResizedCrop to `img_size` | ImageNet |
| Val / test | Resize(`img_size+16`) + CenterCrop | ImageNet |
| `preprocess_chw()` in `triton_client.py` | direct BICUBIC resize to `inference_img_size` square | ImageNet, float32 |

Backend default `inference_img_size` is 518. The RTX checkpoints were trained at 504. `docs/DEPLOY_INFERENCE.md` already says the API must override this. `README.md` still documents Triton input `[1,3,518,518]` and `DINOV2_MODEL_NAME=dinov2_vitg14_reg`.

## ONNX export

`ml_rtx5080/train.py` `export_onnx()` and `ml_rtx5080/export_onnx_for_triton.py`:

- input `input__0`, output `output__0`
- spatial size from training `img_size` (504 for the RTX config)
- opset 18
- classifier returns `squeeze(1)`, so the ONNX output rank is `[batch]`, not `[batch, 1]`
- export errors are swallowed at the end of training

`scripts/export_triton_onnx.py` is a different path: backend `DINOv2Classifier`, default Giant, hardcoded 518, opset 17, and an unsqueeze wrapper so the output is `[batch, 1]`. It also uses `strict=False`.

No ONNX numeric parity check exists.

## TensorRT path

`ml/export_tensorrt.py` and `scripts/export_tensorrt.py` build an engine from ONNX. README training guide shows a Triton config with model name `hypevault_dinov2`, input name `input`, output name `logit`, and dims `[3, 518, 518]`. That contract does not match the backend defaults (`dinov2_classifier`, `input__0`, `output__0`).

No TensorRT latency/accuracy benchmark script exists under `ml_rtx5080/`.

## Triton path

`infra/docker-compose.yml` runs `nvcr.io/nvidia/tritonserver:23.10-py3` with `models/` mounted at `/models`. Host gRPC port is 18001.

Checked-in `models/dinov2_classifier/config.pbtxt`:

- name `dinov2_classifier`
- backend `onnxruntime`
- max batch 8
- input `input__0` FP32 `[3, 518, 518]`
- output `output__0` FP32 `[1]`

`models/dinov2_classifier/1/model.onnx` is present. Its spatial size is the 518 config, not the RTX 504 contract. Triton output is consumed as a logit and passed through `logits_to_verdict()`. Nothing in the Triton response is a calibrated probability.

## Backend decision policy

`logits_to_verdict()`:

1. `prob_fake = sigmoid(logit)`
2. if `prob_fake >= INFERENCE_FAKE_THRESHOLD` (default 0.50): verdict `FAKE`, confidence = `prob_fake`
3. else: verdict `AUTHENTIC`, confidence = `1 - prob_fake`

`apply_min_authentic_confidence()` then remaps AUTHENTIC to FAKE when authentic-class confidence is below `inference_min_authentic_confidence` (default 0.88). After the remap, confidence becomes `1 - c`, which is `prob_fake` again.

There is no REVIEW state. Ambiguous scores are forced to FAKE. The stored `confidence` is not a single calibrated P(fake); it flips meaning with the verdict. Local CUDA inference runs the forward in FP16 autocast. Training used BF16.

`REPORT_ENFORCE_TRITON` defaults to true, so the API refuses the torch backend unless that flag is turned off.

## Reproducibility

Saved next to RTX runs: `config.json`, `split_manifest.json`, `history.json`, `best_model.pt`, periodic `epoch_*.pt`, `final_model.pt`, plots.

Not recorded: git SHA, dataset hash, training-config hash, checkpoint SHA256, Python/PyTorch/torchvision/timm versions, CUDA version, GPU name, peak VRAM, training duration. Seed 42 is in the config. cuDNN benchmark mode is on, which is not strictly deterministic.

Installed training stack in `.venv` (not what the historical checkpoints necessarily used): PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, timm 1.0.26.

## Known bugs

1. Gradient accumulation drops a remainder. `train_epoch` steps only when `(batch_idx + 1) % accumulate_grad == 0`. With 5 batches and `accumulate_grad=2`, steps occur at batches 2 and 4. Batch 5’s gradients are never applied. The spec requires a step at batch 5 as well.
2. Scheduler length uses `len(train_loader) // accumulate_grad`, so it does not count a trailing partial accumulation.
3. `label_smoothing` is configured and unused.
4. `confidence_threshold` is configured and unused. Metrics and plots hardcode 0.5.
5. Checkpoint choice is max validation F1. An unsafe false-authentic checkpoint can be saved as `best_model.pt`.
6. `load_state_dict(..., strict=False)` in `local_torch.py` and `scripts/export_triton_onnx.py` hides architecture mismatches.
7. Corrupt images become gray placeholders instead of failing the sample.
8. README Triton tensor names (`input` / `logit`, model `hypevault_dinov2`) disagree with code and `config.pbtxt` (`input__0` / `output__0`, model `dinov2_classifier`).
9. RTX ONNX export emits a squeezed logit. Triton config expects a trailing dimension of 1. The Giant export wrapper unsqueezes; the RTX exporter does not.

## Known deployment mismatches

| Contract | RTX 5080 training | Backend default | Triton repo |
|---|---|---|---|
| Backbone | DINOv2-Base `vit_base_patch14_dinov2.lvd142m` | DINOv2-Giant `dinov2_vitg14_reg` | whatever was exported into `model.onnx` |
| Image size | 504 | 518 | 518 |
| Val geometry | resize + center crop | direct square resize | direct square resize |
| Precision | BF16 train | FP16 local infer | FP32 ONNX config |
| Output meaning | raw logit | sigmoid, then class-conditional confidence | raw logit assumed |
| Decision | none | AUTHENTIC or FAKE | n/a |

`docs/DEPLOY_INFERENCE.md` documents the 504 / Base override. Defaults and the checked-in Triton config do not apply it.

## Installed NVIDIA skills

Present: `.agents/skills/tao-train-nvdinov2/` (TAO NVDINOv2, skill version 0.1.0, Apache-2.0).

That skill is self-supervised teacher/student pretraining (`images_dir` of unlabeled images, metric `train_loss`, student checkpoints). It is not a labeled authenticity classifier. Default smoke settings in the skill use ViT-S at 224. It is optional experiment B-style adaptation, not the required supervised baseline.

Not installed anywhere searched under `~/.agents`, `~/.cursor`, or this repo: `tao-train-dinov3` or any `*dinov3*` / `*nvdinov3*` skill. Domain-adaptive DINOv3 via that skill cannot be run until the skill exists. Supervised DINOv3 fine-tune does not need it.

## DINOv2 vs DINOv3 feasibility

Required baseline remains supervised fine-tuning of a pretrained backbone on the labeled watch set.

DINOv3 ViT-B/16 is loadable in the current environment:

| Identifier | Role |
|---|---|
| `dinov3_vitb16` | official torch.hub name |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | Hugging Face weights |
| `vit_base_patch16_dinov3.lvd1689m` | timm name, `timm.is_model` true in timm 1.0.26 |

Official card: patch size 16, embed dim 768, 4 register tokens, about 86M parameters. ImageNet-style normalization matches DINOv2 `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`. timm’s pretrained config records `input_size=(3, 256, 256)` and `fixed_input_size=True`. Meta’s model card reports both 256 and 512 evaluations. 512×512 is the experiment target because 16 divides 512. 504 must not be used for DINOv3. A VRAM smoke test has to prove 512 works on this 16 GB GPU before any full run. Register tokens must be stripped before any patch-attention pool.

Both Base backbones are in the same parameter ballpark (~86M). DINOv3 is not assumed better. Experiment E (domain-adapted DINOv3) stays blocked on a missing skill and on a smoke test.

## DINOv3 license

License name: DINOv3 License (Meta), last updated 19 August 2025 on the GitHub `LICENSE.md`. It is not Apache-2.0. Full notes are in `docs/MODEL_LICENSE_REVIEW.md`.

Practical implication: a hosted authenticity API that does not ship weights can be compatible, with attribution and use-restriction duties still applying. Shipping ONNX, TensorRT, or fine-tuned weights to a customer is redistribution of a derivative and must carry the DINOv3 License. If that blocks proprietary distribution, production stays on DINOv2.

## Proposed file changes

Phase 1 touches the accumulation loop inside `ml_rtx5080/train.py` and adds a unit test. Later phases add the modules listed in `new_architecture.md` without deleting `ml/train.py` or overwriting `checkpoints/` or `checkpoints_watches/`. New runs go under `ml_rtx5080/experiments/`.

## Proposed experiment matrix

Same watches split, same seed policy, same selection rule (false-authentic constraint, then PR-AUC, then F1). No final-test tuning.

| ID | Backbone | Size | Head |
|---|---|---|---|
| A | DINOv2 ViT-B/14 `vit_base_patch14_dinov2.lvd142m` | 504 | CLS only |
| B | DINOv2 ViT-B/14 | 504 | CLS + patch attention |
| C | DINOv3 ViT-B/16 `vit_base_patch16_dinov3.lvd1689m` | 512 | CLS only |
| D | DINOv3 ViT-B/16 | 512 | CLS + patch attention |
| E | DINOv3 ViT-B/16 domain-adapted | 512 | CLS + patch attention, only if a 16 GB smoke test passes |

## Risks

- Brand-folder split will keep inflating or distorting “product” metrics until a real product id exists. With five brands, holding out one brand per class is a harsh and unbalanced generalization test.
- No marketplace/seller labels, so source-disjoint evaluation cannot be computed. Report that limitation instead of fabricating labels.
- Historical val F1 above 0.99 was selected without a false-authentic constraint and without a final test set.
- Serving the current Base checkpoint with Giant/518 defaults will score nonsense and can fail silently under `strict=False`.
- DINOv3 512 on 16 GB is unmeasured. timm marks the checkpoint fixed at 256 until a forward pass proves otherwise.
- DINOv3 license can forbid the deployment shape even if accuracy wins.

## Next action

Phase 1: fix remainder gradient accumulation, count scheduler steps from actual optimizer updates, and add the 5-batch / `accumulate_grad=2` test (steps at batches 2, 4, and 5). Do not start a training run in that phase.
