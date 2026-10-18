# Phase 52 — DINOv3 shortcut sensitivity

Measured sensitivity of the frozen GPU Triton DINOv3 path to controlled transforms.
Folder labels are directory names. They are not verified authenticity.

Protocol SHA-256 `c14c30cc148cda5a4e0324d6d7db6fec587ca52b4d0d31152879fe9c56bb614f`.

Checkpoint `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`. Temperature `0.24038200410185356`. Threshold `0.5`.

Triton `dinov3_authenticity_candidate` v1: `{"kinds": ["KIND_GPU"], "gpus": [0], "use_tf32": "0", "backend": "onnxruntime"}`.

Images scored: 40. Short cells: 0.

Primary decision is threshold-only P(fake) at 0.50. Policy flips include the quality gate and are secondary.
A shift is not evidence of what the training procedure learned.

All 40 original P(fake) values were within 0.001 of 0 or 1. None were within 0.05 of 0.50. Small logit moves therefore leave P(fake) and the threshold decision unchanged. The table reports max absolute logit change so that saturation is not mistaken for a zero effect.

Sample: 4 images × 5 brands × 2 folder labels = 40. Splits were train, calibration, and validation only. 0 of 40 paths are in the locked test membership. 20 source files were already 512×512 and 20 were not. No cell was short.

## Original repeat

Same tensor scored twice. Max abs delta logit `0`. Max abs delta P(fake) `0`. Threshold flips `0`.

## Variants

| Variant | Family | n | Max abs Δlogit | Max abs ΔP | Threshold flips | Policy flips | Pixels changed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `jpeg_q95` | encoding | 40 | 0.000641 | ~0 | 0 | 0 | 40 |
| `png` | encoding | 40 | 0 | 0 | 0 | 0 | 0 |
| `webp_q90` | encoding | 40 | 0.006255 | ~0 | 0 | 0 | 40 |
| `jpeg_q75` | compression | 40 | 0.004228 | ~0 | 0 | 0 | 40 |
| `jpeg_q50` | compression | 40 | 0.019577 | ~0 | 0 | 0 | 40 |
| `jpeg_q30` | compression | 40 | 9.043607 | 1.000000 | 1 | 1 | 40 |
| `long_side_256` | resizing | 40 | 0.032506 | ~0 | 0 | 36 | 40 |
| `stretch_512` | resizing | 40 | 0.032699 | ~0 | 0 | 0 | 20 |
| `pad_square_black` | padding | 40 | 0.011934 | ~0 | 0 | 4 | 17 |
| `pad_square_white` | padding | 40 | 0.007577 | ~0 | 0 | 4 | 17 |
| `pad_border_32_imagenet_mean` | padding | 40 | 0.012197 | ~0 | 0 | 3 | 40 |

PNG is the lossless control: decoded pixels and the logit were unchanged.

`stretch_512` changed pixels on the 20 files that were not already 512×512. On those, max abs Δlogit was 0.033 and no threshold flipped. On the 20 files that were already 512×512, the stretch was a no-op (max abs Δlogit 0).

## The one threshold flip

`jpeg_q30` on `Label_1_Watches/Patek Philippe/55f9ca55dbaa.jpg` (folder_fake, already 512×512): logit `4.603642` → `-4.439964` (Δlogit `-9.043607`), P(fake) `1.000000` → `9.5e-9`, threshold decision FAKE → AUTHENTIC. The original was not near 0.50. The other 39 images, including the other 19 folder_fake files, did not flip. The next-largest |Δlogit| at q30 was 2.53 and still left P(fake) saturated, so it did not flip.

## Policy flips are the quality gate

`long_side_256` flipped the shadow policy on 36/40 images to REVIEW because the resized image raised `LOW_RESOLUTION` (sometimes with another flag). The threshold decision did not change on any of them. The other 4 were already REVIEW from `LOW_CONTRAST` and stayed REVIEW.

`pad_square_black` and `pad_square_white` each moved 4 images from REVIEW to AUTHENTIC by clearing `LOW_CONTRAST` / `EXTREME_ASPECT_RATIO`. Their threshold decisions stayed AUTHENTIC.

`pad_border_32_imagenet_mean` moved 3 images from FAKE to REVIEW by adding `LOW_CONTRAST`. Their threshold decisions stayed FAKE.

Those policy changes are the frozen quality rule seeing a different image. They are not threshold flips of the authenticity head.

## Largest absolute P(fake) changes

- `jpeg_q30` on `Label_1_Watches/Patek Philippe/55f9ca55dbaa.jpg` (Patek Philippe, folder_fake): ΔP(fake) `-1.000000`, Δlogit `-9.043607`, threshold flip `True`, near 0.50 `False`, 512-square file `True`.
- `jpeg_q30` on `Label_1_Watches/A. Lange & Söhne/408b83c68357.jpg` (A. Lange & Söhne, folder_fake): ΔP(fake) `-0.000186`, Δlogit `-2.528692`, threshold flip `False`, near 0.50 `False`, 512-square file `True`.
- `jpeg_q30` on `Label_1_Watches/A. Lange & Söhne/150a19c41c19.jpg` (A. Lange & Söhne, folder_fake): ΔP(fake) `0.000000`, Δlogit `0.080619`, threshold flip `False`, near 0.50 `False`, 512-square file `True`.
- `long_side_256` on `Label_0_Watches/Patek Philippe/7e7f99a56ebf.jpg` (Patek Philippe, folder_authentic): ΔP(fake) `0.000000`, Δlogit `0.032506`, threshold flip `False`, near 0.50 `False`, 512-square file `False`.
- `jpeg_q30` on `Label_1_Watches/Vacheron Constantin/68f6e88bf9b8.jpg` (Vacheron Constantin, folder_fake): ΔP(fake) `-0.000000`, Δlogit `-0.026791`, threshold flip `False`, near 0.50 `False`, 512-square file `True`.
- `stretch_512` on `Label_0_Watches/A. Lange & Söhne/5531277ef4b5.jpg` (A. Lange & Söhne, folder_authentic): ΔP(fake) `-0.000000`, Δlogit `-0.032699`, threshold flip `False`, near 0.50 `False`, 512-square file `False`.
- `jpeg_q30` on `Label_1_Watches/Patek Philippe/1c06e835c3ac.jpg` (Patek Philippe, folder_fake): ΔP(fake) `0.000000`, Δlogit `0.026936`, threshold flip `False`, near 0.50 `False`, 512-square file `True`.
- `stretch_512` on `Label_0_Watches/A. Lange & Söhne/2cb473cabd01.jpg` (A. Lange & Söhne, folder_authentic): ΔP(fake) `-0.000000`, Δlogit `-0.028206`, threshold flip `False`, near 0.50 `False`, 512-square file `False`.
- `jpeg_q50` on `Label_1_Watches/A. Lange & Söhne/408b83c68357.jpg` (A. Lange & Söhne, folder_fake): ΔP(fake) `0.000000`, Δlogit `0.019577`, threshold flip `False`, near 0.50 `False`, 512-square file `True`.
- `jpeg_q50` on `Label_1_Watches/Vacheron Constantin/68f6e88bf9b8.jpg` (Vacheron Constantin, folder_fake): ΔP(fake) `0.000000`, Δlogit `0.019180`, threshold flip `False`, near 0.50 `False`, 512-square file `True`.
- `jpeg_q30` on `Label_1_Watches/Audemars Piguet/5ddcedf9a0e3.jpg` (Audemars Piguet, folder_fake): ΔP(fake) `0.000000`, Δlogit `0.018617`, threshold flip `False`, near 0.50 `False`, 512-square file `True`.
- `stretch_512` on `Label_0_Watches/Audemars Piguet/9d4f1a43c205.jpg` (Audemars Piguet, folder_authentic): ΔP(fake) `-0.000000`, Δlogit `-0.020518`, threshold flip `False`, near 0.50 `False`, 512-square file `False`.
- `jpeg_q50` on `Label_1_Watches/Patek Philippe/55f9ca55dbaa.jpg` (Patek Philippe, folder_fake): ΔP(fake) `0.000000`, Δlogit `0.017624`, threshold flip `False`, near 0.50 `False`, 512-square file `True`.
- `stretch_512` on `Label_0_Watches/Audemars Piguet/970df8dd5876.jpg` (Audemars Piguet, folder_authentic): ΔP(fake) `-0.000000`, Δlogit `-0.017878`, threshold flip `False`, near 0.50 `False`, 512-square file `False`.
- `long_side_256` on `Label_0_Watches/Vacheron Constantin/7706f59b5473.jpg` (Vacheron Constantin, folder_authentic): ΔP(fake) `0.000000`, Δlogit `0.015161`, threshold flip `False`, near 0.50 `False`, 512-square file `False`.

## Measured sensitivity versus a hypothesis

Measured: on this 40-image non-test draw, mild re-encoding and resizing moved logits by at most a few hundredths, and the saturated probabilities did not cross 0.50, except for one JPEG quality-30 recompression that moved a single logit by 9.04 and flipped FAKE to AUTHENTIC.

Not measured: whether training caused the network to use JPEG quantization, 512-square geometry, or padding as a class feature. One flip under heavy compression is compatible with ordinary input sensitivity. It does not show that folder labels are real, and it does not show that the 512-pixel catalog confound is what the frozen model relies on. Forcing non-square images to 512×512 did not flip a threshold decision in this draw.

## What this does not show

A measured shift is sensitivity of this serving path. It is not evidence that the network learned JPEG tables, 512-square geometry, or padding as an authenticity feature, and it is not a cross-brand or real-world authenticity result. Folder labels stay unverified.

`MODEL_CHANGED = false`

`CALIBRATION_CHANGED = false`

`FINAL_TEST_TOUCHED = false`

`DATASET_READY_FOR_TRAINING = false`

`PRODUCTION_PROMOTION_ALLOWED = false`
