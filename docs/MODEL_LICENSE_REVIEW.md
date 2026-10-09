# DINOv3 license review

Reviewed 2026-10-04 from public sources, before any DINOv3 weights are wired into training or production.

This is an engineering reading of the published license, not a legal opinion. Additional legal review is appropriate before DINOv3 is selected as the production backbone.

## Sources

- Meta DINOv3 License, last updated 19 August 2025: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
- Meta hosted copy: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
- Meta maintainer clarification (patricklabatut, 21 August 2025): https://github.com/facebookresearch/dinov3/issues/28
- timm tag `vit_base_patch16_dinov3.lvd1689m` records `license='dinov3-license'`

## License name

DINOv3 License. It is a Meta custom license. It is not Apache-2.0, MIT, or BSD.

DINOv2 (`facebookresearch/dinov2`, and the timm `*.lvd142m` DINOv2 weights used by the current RTX 5080 run) stays the production fallback if this license blocks the intended distribution.

## Grant

Non-exclusive, worldwide, non-transferable, royalty-free limited license to use, reproduce, distribute, copy, create derivative works of, and modify the DINO Materials.

## Attribution

- Research publications that use the materials must acknowledge DINOv3.
- The Meta-hosted license text also requires a prominent “Built with DINOv3” credit on a related website, user interface, blog post, about page, or product documentation when materials or derivatives are distributed. The GitHub `LICENSE.md` snapshot and the Meta page are not identical on that display sentence. Treat the stricter public text as the one to clear with counsel.

## Redistribution

If DINOv3 materials or derivative works are distributed or made available to a third party, they may be distributed only under the DINOv3 License, and a copy of the license must ship with them.

Derivative works include fine-tuned checkpoints, ONNX files, and TensorRT engines built from DINOv3 weights.

A Meta maintainer stated that a commercial API or SaaS offering does not need to attach the license when weights and code are not redistributed, and that the offering must still comply with the license. A later public question about on-site demos that never hand the client model files was still open in the sources checked here. Do not treat “we only run it on our server” as settled for every commercial shape.

## Commercial and proprietary deployment

| Deployment shape | Engineering reading |
|---|---|
| HypeVault API scores images and does not ship weights, ONNX, or engines | Maintainer comment says the license file need not be attached to customers. Use restrictions still apply. Confirm with counsel. |
| Customer-installed model, downloaded checkpoint, or container that contains fine-tuned DINOv3 weights | Redistribution of a derivative. Ship the DINOv3 License. Proprietary-only terms on those weights conflict with “only under this Agreement.” |
| Publishing a paper or model card about the watch classifier | Acknowledge DINOv3. |

Acceptable use still includes: applicable law, privacy and data-protection rules, and trade controls. Military, weapons, nuclear, and espionage end uses called out in the license are prohibited. A luxury-watch authenticity marketplace is not one of those listed end uses. That does not remove the redistribution issue.

Meta can terminate the agreement for breach. On termination the materials must be deleted and use must stop.

## Production rule for this repo

Do not make DINOv3 the production model until:

1. Controlled experiments show it beats DINOv2 on the false-authentic constraint and held-out metrics.
2. Counsel confirms the intended shipping path (hosted API only, or downloadable weights).

If counsel rejects redistribution of fine-tuned DINOv3 artifacts, keep DINOv2-Base as the production backbone and leave DINOv3 as a research experiment.
