"""Phase 76: freeze the DINOv3 + Perception Encoder head, and score files with it.

Chosen from the exploratory fusion probe (fusion_probe.py, experiments/phase76_probe/fusion_compare.json).
Training uses every development set: the Phase 75 data, plus benchmark_v5 (spent) — both apps' watch
images (x10) and its 144 real eBay photos. The FAKE threshold is set on real watch photos only
(Label_0 validation + all benchmark eBay reals out-of-fold), at most 1% FAKE. Eligibility: at most
5% FAKE on the 1,000 held-out COCO photos (the Phase 75 content-shortcut rule).

    .venv/bin/python ml_rtx5080/phase76.py freeze
    .venv/bin/python ml_rtx5080/phase76.py score-files a.jpg b.png
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ML = Path(__file__).resolve().parent
REPO = ML.parent
OUT = ML / "experiments" / "phase76_fusion"
PROBE = ML / "experiments" / "phase76_probe"
RULE = OUT / "frozen_rule.json"
HEAD = OUT / "dinov3_pe_v6_head.joblib"
sys.path.insert(0, str(ML))

import numpy as np  # noqa: E402

PARTS = ("d3", "pe")
REPEAT = 10
FAKE_BUDGET = 0.01
COCO_FAKE_BUDGET = 0.05


def _pe_revision() -> str | None:
    snaps = Path.home() / ".cache/huggingface/hub/models--timm--vit_pe_core_large_patch14_336.fb/snapshots"
    found = sorted(p.name for p in snaps.iterdir()) if snaps.exists() else []
    return found[0] if len(found) == 1 else None


def freeze() -> None:
    import joblib
    import timm

    import fusion_probe as fp
    import frontier_v5 as f5
    import kandinsky_v4_design as d74

    if RULE.exists():
        raise SystemExit(f"{RULE.relative_to(REPO)} exists; the Phase 76 head is already frozen")
    feats75 = f5.CANDIDATES["front_pre+ft+coco"][0]

    def with_pe(d: dict, name: str) -> dict:
        z = np.load(fp.CACHE / f"{name}.npz")
        assert z["id"].tolist() == d["id"].tolist(), name
        return {**d, "d3": d74.view(d, feats75), "pe": z["pe"]}

    _, train, val_l0, _, _, frontier, coco = f5._dev()
    train, val_l0 = with_pe(train, "train_l0"), with_pe(val_l0, "val_l0")
    frontier, coco = with_pe(frontier, "frontier"), with_pe(coco, "coco")
    benches = [with_pe(d74.load(f"bench_{v}"), f"bench_{v}") for v in ("v1", "v2", "v3", "v4")]
    bench_gen = d74.concat([d74.subset(b, b["label"] == 1) for b in benches] + [with_pe(d74.load("k_train"), "k_train")])
    z = np.load(f5.FEATURES / "coco_holdout.npz")
    holdout = with_pe({k: z[k] for k in z.files}, "coco_holdout")

    v5z, bz = np.load(PROBE / "v5_features.npz"), np.load(PROBE / "backbone_features.npz")
    v5 = {k: v5z[k] for k in v5z.files}
    v5 = {"id": v5["id"], "group": v5["group"], "d3": d74.view(v5, feats75), "pe": bz["pe_core_large"]}
    g = v5["group"]
    keep = ("id", "d3", "pe")
    ebay = d74.concat([{k: d74.subset(b, b["label"] == 0)[k] for k in keep} for b in benches]
                      + [{k: d74.subset(v5, g == "real_ebay_v5")[k] for k in keep}])
    v5_gen = {k: np.concatenate([d74.subset(v5, np.isin(g, ["chatgpt_v5", "gemini_v5"]))[k]] * REPEAT) for k in keep}
    gens = [bench_gen, frontier, v5_gen]

    def x(d):
        return np.concatenate([d[p] for p in PARTS], axis=1)

    def fit(reals):
        return d74.fit_lr(np.concatenate([x(r) for r in reals + gens]),
                          np.concatenate([np.zeros(len(r["id"])) for r in reals] + [np.ones(len(q["id"])) for q in gens]), f5.C)

    model = fit([train, ebay, coco])
    folds = np.arange(len(ebay["id"])) % 5
    oof = np.zeros(len(ebay["id"]))
    for f in range(5):
        m = fit([train, d74.subset(ebay, folds != f), coco])
        oof[folds == f] = m.decision_function(x(d74.subset(ebay, folds == f)))
    real = np.concatenate([model.decision_function(x(val_l0)), oof])
    t_fake = d74.threshold(real)
    coco_fake = float((model.decision_function(x(holdout)) >= t_fake).mean())
    if coco_fake > COCO_FAKE_BUDGET:
        raise SystemExit(f"not eligible: {coco_fake:.1%} of held-out COCO photos FAKE (> {COCO_FAKE_BUDGET:.0%})")
    testtest = float(model.decision_function(x(d74.subset(v5, g == "motivating")))[0])

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, HEAD)
    rule = {
        "rule": "dinov3_pe_v6", "phase": 76, "features": "DINOv3 pre+ft (Phase 75 view) + Perception Encoder L/14-336 pooled",
        "pe_model": fp.PE_MODEL, "pe_hf_revision": _pe_revision(), "timm_version": timm.__version__,
        "dinov3_view": feats75, "head": f"StandardScaler + logistic regression (C = {f5.C}, balanced classes)",
        "model": {"path": str(HEAD.relative_to(REPO)), "sha256": hashlib.sha256(HEAD.read_bytes()).hexdigest()},
        "t_fake": t_fake,
        "training": {"reals": {"label0_train": len(train["id"]), "ebay_v1_v5": len(ebay["id"]), "coco": len(coco["id"])},
                     "generated": {"bench_v1_v4_and_kandinsky": len(bench_gen["id"]), "frontier": len(frontier["id"]),
                                   "chatgpt_v5_and_gemini_v5": f"{int(np.isin(g, ['chatgpt_v5', 'gemini_v5']).sum())} images x{REPEAT}"}},
        "threshold_source": f"real photos only: Label_0 validation + benchmark_v1-v5 eBay (out-of-fold), at most {FAKE_BUDGET:.0%} FAKE",
        "development_real_fake_rate": round(float((real >= t_fake).mean()), 4),
        "coco_holdout_fake_rate": round(coco_fake, 4),
        "motivating_case_score_in_training_context": round(testtest, 3),
        "normalisation": "RGB, Lanczos short side 512, centre square, JPEG q90",
        "frozen_before_any_v6_data": True,
    }
    RULE.write_text(json.dumps(rule, indent=2) + "\n")
    print(json.dumps({k: rule[k] for k in ("t_fake", "development_real_fake_rate", "coco_holdout_fake_rate",
                                            "motivating_case_score_in_training_context")}, indent=1))
    print("sha256", rule["model"]["sha256"])


class FrozenFusionHead:
    """The frozen Phase 76 rule: hash-checked head; FAKE if score >= t_fake."""

    def __init__(self) -> None:
        import joblib

        import fusion_probe as fp
        import kandinsky_v4 as k4
        import kandinsky_v4_design as d74

        self.rule = json.loads(RULE.read_text())
        path = REPO / self.rule["model"]["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.rule["model"]["sha256"]:
            raise RuntimeError("Phase 76 head does not match its recorded SHA-256")
        self.k4, self.d74 = k4, d74
        self.model = joblib.load(path)
        self.pre, self.ft = k4.Pretrained(), k4.FineTuned()
        self.enc = fp.Encoders(("pe",))

    def features_images(self, images) -> np.ndarray:
        """The head's input: DINOv3 view + Perception Encoder, one row per normalised image (Phase 78 reads it too)."""
        out = []
        for s in range(0, len(images), self.k4.BATCH):
            chunk = images[s:s + self.k4.BATCH]
            feats = {**self.pre.features(chunk), **self.ft.features(chunk)}
            out.append(np.concatenate([self.d74.view(feats, self.rule["dinov3_view"]), self.enc(chunk)["pe"]], axis=1))
        return np.concatenate(out)

    def scores_images(self, images) -> list[float]:
        return [float(v) for v in self.model.decision_function(self.features_images(images))]

    def normalise(self, img):
        """The benchmark normalisation: RGB, centre square, Lanczos 512 px, JPEG q90."""
        from PIL import Image

        img = img.convert("RGB")
        s = min(img.size)
        left, top = (img.width - s) // 2, (img.height - s) // 2
        return self.k4.jpeg90(img.crop((left, top, left + s, top + s)).resize((512, 512), Image.LANCZOS))

    def features_image(self, img) -> np.ndarray:
        return self.features_images([self.normalise(img)])[0]

    def score_image(self, img) -> float:
        """Score one PIL image with the benchmark normalisation."""
        return float(self.model.decision_function(self.features_image(img)[None])[0])

    def scores(self, paths: list[Path]) -> list[float]:
        return self.scores_images([self.k4.jpeg90(self.k4.square(p, 512)) for p in paths])

    def verdict(self, score: float) -> str:
        return "FAKE" if score >= self.rule["t_fake"] else "AUTHENTIC"


def score_files(paths: list[str]) -> None:
    head = FrozenFusionHead()
    for p, sc in zip(paths, head.scores([Path(x) for x in paths])):
        print(f"{Path(p).name:36s} score {sc:+.2f} (FAKE at >= {head.rule['t_fake']:+.2f}) -> {head.verdict(sc)}")


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        freeze()
    elif sys.argv[1] == "score-files":
        score_files(sys.argv[2:])
