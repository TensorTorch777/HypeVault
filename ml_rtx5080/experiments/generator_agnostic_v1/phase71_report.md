# Phase 71: generator-agnostic detector (frozen features + autoencoder reconstructions)

Run on 20 October 2026, following `protocol.json` (written before any feature extraction). The final evaluation ran once (`final_metrics.json`).

## Design

- **Real:** 5,000 Label_0 training photos (1,000 per brand).
- **Fake:** the same photos passed through the SD 1.5 and SDXL autoencoders. No prompts, no diffusion sampling. Label_1 and Kandinsky's MoVQ decoder were never used.
- **Distortions:** each image appears clean (JPEG q90) and with random resize, blur and JPEG q60–95, identically for real and fake.
- **Models:** two frozen backbones, each with a logistic-regression head: CLIP ViT-L/14 and pretrained DINOv3 ViT-B/16 (not the HypeVault fine-tune).

## Selection (validation only)

| Candidate | AUC, Label_0 vs Label_1 (unseen generator) | AUC, real vs reconstruction | Threshold (p99 of real validation) |
|---|---|---|---|
| CLIP ViT-L/14 linear | 0.788 | 0.804 | 0.879 |
| **DINOv3 ViT-B/16 linear (selected)** | **0.898** | 0.925 | 0.979 |

## Final evaluation (selected model, fixed threshold)

| Set | Generated detected | Real rejected | ROC-AUC |
|---|---|---|---|
| Kandinsky 2.2, unseen (primary) | **40 / 284 (14.1%, 95% CI 10.3–18.7%)** | 4 / 300 eBay (1.3%, CI 0.4–3.4%) | 0.664 |
| SDXL (decoder seen in training) | 45 / 150 (30.0%) | 4 / 300 (1.3%) | 0.854 |
| Label_1 test (unseen generator) | 552 / 1,500 (36.8%) | 12 / 1,506 Label_0 (0.8%) | 0.903 |

**Earlier models on the same benchmark images.** On Kandinsky, fine-tuned DINOv3 called 12 / 284 FAKE (4.2%, CI 2.2–7.3%; 102 REVIEW, ROC-AUC 0.83). DINOv2 called 13 / 284 FAKE (4.6%). On SDXL, fine-tuned DINOv3 called 119 / 150 FAKE and DINOv2 70 / 150.

## Against the pre-registered success criterion

- Kandinsky detection is higher than the best earlier model, and the 95% intervals do not overlap (10.3% > 7.3%): **met**.
- False-fake rate on the 300 eBay photos is ≤ 5% (1.3%): **met**.

## What it means

- The approach generalises a little better at a strict operating point. It catches about 3 times more Kandinsky images than the fine-tuned models, while rejecting only 1.3% of real eBay photos.
- **It is still not a usable detector for unseen generators:** 86% of Kandinsky images pass.
- Its ranking of Kandinsky images (ROC-AUC 0.664) is *worse* than the fine-tuned DINOv3's (0.83). The gain comes from the operating point, not from better separation.
- It is much weaker than the fine-tuned DINOv3 on the training generator (Label_1: 36.8% vs about 100%) and on SDXL (30% vs 79%).
- Reconstruction-only training does not replace real generated images. It is a candidate extra signal, not a replacement model.
