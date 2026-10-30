"""Phase 77: freeze the candidate selected in experiments/phase77_probe/compare.json, and score files with it.

Same head as Phase 76 (StandardScaler + logistic regression, C = 0.05, balanced classes, app images
weighted x10), fitted with scikit-learn on the selected features and training data. The FAKE cut is
set on real watch photos only: Label_0 validation plus every benchmark_v1-v6 eBay photo out-of-fold,
at most 1% FAKE. Eligibility: at most 5% FAKE on the 1,000 held-out COCO photos.

    .venv/bin/python ml_rtx5080/phase77.py freeze
    .venv/bin/python ml_rtx5080/phase77.py score-files a.jpg b.png
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ML = Path(__file__).resolve().parent
REPO = ML.parent
OUT = ML / "experiments" / "phase77"
RULE = OUT / "frozen_rule.json"
HEAD = OUT / "phase77_head.joblib"
COMPARE = ML / "experiments" / "phase77_probe" / "compare.json"
sys.path.insert(0, str(ML))

import numpy as np  # noqa: E402

FAKE_BUDGET = 0.01
COCO_FAKE_BUDGET = 0.05
EBAY_SETS = ("bench_v1", "bench_v2", "bench_v3", "bench_v4", "v5", "v6")


def freeze() -> None:
    import joblib
    import timm
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import phase77_compare as pc
    from kandinsky_v4_design import subset

    if RULE.exists():
        raise SystemExit(f"{RULE.relative_to(REPO)} exists; the Phase 77 head is already frozen")
    compare = json.loads(COMPARE.read_text())
    override = compare.get("override")
    name = override["selected"] if override else compare["selected"]
    if name is None:
        raise SystemExit("compare.json selected no eligible candidate")
    pick = compare["results"][name]
    keys, data = tuple(pick["features"]), pick["data"]
    sets = pc.load()
    pool = pc._pool(sets, data)

    def fit(tr):
        x, y = pc._cat(tr, keys), np.concatenate([d["label"] for d in tr])
        w = np.concatenate([d["weight"] for d in tr])
        return make_pipeline(StandardScaler(), LogisticRegression(C=pc.C, class_weight="balanced", max_iter=5000)).fit(
            x, y, logisticregression__sample_weight=w)

    model = fit(pool)
    rng = np.random.default_rng(pc.SEED)
    ebay_ids = np.concatenate([sets[s]["id"][sets[s]["label"] == 0] for s in EBAY_SETS])
    fold = dict(zip(ebay_ids.tolist(), rng.integers(0, 5, len(ebay_ids)).tolist()))
    oof = []
    for f in range(5):
        def in_fold(d, f=f):
            return (d["label"] == 0) & np.array([fold.get(i, -1) == f for i in d["id"].tolist()])
        m = fit([subset(d, ~in_fold(d)) for d in pool])
        held = [subset(sets[s], in_fold(sets[s])) for s in EBAY_SETS]
        oof.append(m.decision_function(pc._cat([h for h in held if len(h["id"])], keys)))
    real = np.concatenate([model.decision_function(pc._cat([sets["val_l0"]], keys))] + oof)
    grid = np.unique(real)[::-1]
    t_fake = float(min(t for t in grid if (real >= t).mean() <= FAKE_BUDGET))
    coco_fake = float((model.decision_function(pc._cat([sets["coco_holdout"]], keys)) >= t_fake).mean())
    if coco_fake > COCO_FAKE_BUDGET:
        raise SystemExit(f"not eligible: {coco_fake:.1%} of held-out COCO photos FAKE (> {COCO_FAKE_BUDGET:.0%})")
    testtest = float(model.decision_function(pc._cat([sets["testtest"]], keys))[0])

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, HEAD)
    rule = {
        "rule": "phase77", "phase": 77, "candidate": name, "selection_override": override, "features": list(keys), "data": data,
        "pe_model": "vit_pe_core_large_patch14_336.fb", "timm_version": timm.__version__,
        "dinov3_view": "pre+ft", "pe512": {"side": 512, "mid_blocks_one_based": [12, 18], "mid_pool": "mean of patch tokens"},
        "head": f"StandardScaler + logistic regression (C = {pc.C}, balanced classes, app images weighted x{pc.REPEAT})",
        "model": {"path": str(HEAD.relative_to(REPO)), "sha256": hashlib.sha256(HEAD.read_bytes()).hexdigest()},
        "t_fake": t_fake,
        "training_rows": {"real": int(sum((d["label"] == 0).sum() for d in pool)),
                          "generated": int(sum((d["label"] == 1).sum() for d in pool))},
        "threshold_source": f"real watch photos only: Label_0 validation + benchmark_v1-v6 eBay (5-fold out-of-fold), at most {FAKE_BUDGET:.0%} FAKE",
        "development_real_fake_rate": round(float((real >= t_fake).mean()), 4),
        "coco_holdout_fake_rate": round(coco_fake, 4),
        "motivating_case_score_in_training_context": round(testtest, 3),
        "normalisation": "RGB, Lanczos short side 512, centre square, JPEG q90",
        "frozen_before_any_v7_data": True,
    }
    RULE.write_text(json.dumps(rule, indent=2) + "\n")
    print(json.dumps({k: rule[k] for k in ("candidate", "t_fake", "development_real_fake_rate", "coco_holdout_fake_rate",
                                            "motivating_case_score_in_training_context")}, indent=1))
    print("sha256", rule["model"]["sha256"])


class FrozenPhase77Head:
    """The frozen Phase 77 rule: hash-checked head; FAKE if score >= t_fake."""

    def __init__(self) -> None:
        import joblib

        import kandinsky_v4 as k4
        import kandinsky_v4_design as d74

        self.rule = json.loads(RULE.read_text())
        path = REPO / self.rule["model"]["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.rule["model"]["sha256"]:
            raise RuntimeError("Phase 77 head does not match its recorded SHA-256")
        self.k4, self.d74 = k4, d74
        self.model = joblib.load(path)
        keys = self.rule["features"]
        self.keys = keys
        self.pre = self.ft = self.pe336 = self.pe512 = None
        if "d3" in keys:
            self.pre, self.ft = k4.Pretrained(), k4.FineTuned()
        if "pe" in keys:
            import fusion_probe as fp

            self.pe336 = fp.Encoders(("pe",))
        if any(k.startswith("pe512") for k in keys):
            from phase77_probe import PE512

            self.pe512 = PE512()

    def features(self, images) -> np.ndarray:
        parts = {}
        if self.pre is not None:
            parts["d3"] = self.d74.view({**self.pre.features(images), **self.ft.features(images)}, self.rule["dinov3_view"])
        if self.pe336 is not None:
            parts["pe"] = self.pe336(images)["pe"]
        if self.pe512 is not None:
            parts.update(self.pe512(images))
        return np.concatenate([parts[k] for k in self.keys], axis=1)

    def scores_images(self, images) -> list[float]:
        out = []
        for s in range(0, len(images), self.k4.BATCH):
            out += self.model.decision_function(self.features(images[s:s + self.k4.BATCH])).tolist()
        return [float(v) for v in out]

    def score_image(self, img) -> float:
        """Score one PIL image with the benchmark normalisation (RGB, centre square, Lanczos 512 px, JPEG q90)."""
        from PIL import Image

        img = img.convert("RGB")
        s = min(img.size)
        left, top = (img.width - s) // 2, (img.height - s) // 2
        img = img.crop((left, top, left + s, top + s)).resize((512, 512), Image.LANCZOS)
        return self.scores_images([self.k4.jpeg90(img)])[0]

    def scores(self, paths: list[Path]) -> list[float]:
        return self.scores_images([self.k4.jpeg90(self.k4.square(p, 512)) for p in paths])

    def verdict(self, score: float) -> str:
        return "FAKE" if score >= self.rule["t_fake"] else "AUTHENTIC"


def score_files(paths: list[str]) -> None:
    head = FrozenPhase77Head()
    for p, sc in zip(paths, head.scores([Path(x) for x in paths])):
        print(f"{Path(p).name:36s} score {sc:+.2f} (FAKE at >= {head.rule['t_fake']:+.2f}) -> {head.verdict(sc)}")


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        freeze()
    elif sys.argv[1] == "score-files":
        score_files(sys.argv[2:])
