"""Phase 74 development: compare Kandinsky detectors built on DINOv3 features and MoVQ reconstructions.

Development data only (benchmark_v1-v3, Label_0 train/validation, kandinsky22_train). Each benchmark version
is held out in turn: detectors train without it, the FAKE threshold comes from real photos outside it (Label_0
validation plus the other two versions' eBay photos, at most 1% FAKE), and the held-out version is scored.

    .venv/bin/python ml_rtx5080/kandinsky_v4_design.py compare
    .venv/bin/python ml_rtx5080/kandinsky_v4_design.py freeze   # writes experiments/kandinsky_v4/fusion_rule.json
    .venv/bin/python ml_rtx5080/kandinsky_v4_design.py unseen   # leave-one-generator-out (after benchmark_v4)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from kandinsky_v4 import FEATURES, OUT  # noqa: E402
from normalize_surface import roc_auc  # noqa: E402

VERSIONS = ("v1", "v2", "v3")
KANDINSKY = {"heldout_kandinsky22", "kandinsky22_fresh", "kandinsky22_v3", "kandinsky22_train"}
FAKE_BUDGET = 0.01
REVIEW_BUDGET = 0.05
SELECTED = "gen_ft+ebay"
PRE = ("d3pre_cls", "d3pre_patch", "d3pre_cls_b4", "d3pre_cls_b8")
FT = ("d3ft_cls", "d3ft_pool")
MOVQ = ("movq_l1_512", "movq_cos_512", "movq_l1_768", "movq_cos_768")


def load(name: str) -> dict[str, np.ndarray] | None:
    path = FEATURES / f"{name}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


def movq_block(d: dict, prefix: str = "") -> np.ndarray:
    cols = [d[prefix + k] for k in MOVQ]
    return np.stack(cols + [np.log(c + 1e-6) for c in cols] + [d[prefix + "d3ft_logit"]], axis=1)


def block(d: dict, keys: tuple[str, ...], prefix: str = "") -> np.ndarray:
    return np.concatenate([d[prefix + k] for k in keys], axis=1)


def view(d: dict, feats: str, prefix: str = "") -> np.ndarray:
    parts = {"pre": lambda: block(d, PRE, prefix), "patch": lambda: d[prefix + "d3pre_patch"],
             "ft": lambda: block(d, FT, prefix), "movq": lambda: movq_block(d, prefix)}
    return np.concatenate([parts[p]() for p in feats.split("+")], axis=1)


def subset(d: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in d.items()}


def concat(ds: list[dict]) -> dict:
    keys = set.intersection(*(set(d) for d in ds))
    return {k: np.concatenate([d[k] for d in ds]) for k in keys}


def fit_lr(x: np.ndarray, y: np.ndarray, c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000)).fit(x, y)


def threshold(real_scores: np.ndarray) -> float:
    grid = np.unique(real_scores)[::-1]
    return float(min(t for t in grid if (real_scores >= t).mean() <= FAKE_BUDGET))


CANDIDATES = {
    # name: (features, fake sources, C, eBay reals in training); "aero" candidates are training-free
    "aero_l1_768": ("aero", "movq_l1_768", None, False),
    "aero_l1_512": ("aero", "movq_l1_512", None, False),
    "aligned_patch": ("patch", ("aligned",), 1.0, False),
    "aligned_pre": ("pre", ("aligned",), 1.0, False),
    "kshot_patch": ("patch", ("kandinsky",), 0.05, False),
    "kshot_pre": ("pre", ("kandinsky",), 0.05, False),
    "kshot_ft": ("ft", ("kandinsky",), 0.05, False),
    "kshot_pre+ft": ("pre+ft", ("kandinsky",), 0.05, False),
    "kshot_pre+aligned": ("pre", ("kandinsky", "aligned"), 0.05, False),
    "kshot_pre+ft+movq": ("pre+ft+movq", ("kandinsky",), 0.05, False),
    "kshot_ft+ebay": ("ft", ("kandinsky",), 0.05, True),
    "kshot_pre+ft+ebay": ("pre+ft", ("kandinsky",), 0.05, True),
    "gen_ft+ebay": ("ft", ("kandinsky", "othergen"), 0.05, True),
    "gen_pre+ft+ebay": ("pre+ft", ("kandinsky", "othergen"), 0.05, True),
    "gen_pre+ft+movq+ebay": ("pre+ft+movq", ("kandinsky", "othergen"), 0.05, True),
}


def training_set(feats: str, fakes: tuple[str, ...], others: list[dict], train: dict, k_train: dict | None,
                 ebay: dict | None) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [view(train, feats)], [np.zeros(len(train["id"]))]
    if ebay is not None:
        xs.append(view(ebay, feats))
        ys.append(np.zeros(len(ebay["id"])))
    if "aligned" in fakes:
        xs.append(view(train, feats, "aligned__"))
        ys.append(np.ones(len(train["id"])))
    gen = []
    if "kandinsky" in fakes:
        gen += [subset(d, np.isin(d["group"], list(KANDINSKY))) for d in others] + ([k_train] if k_train else [])
    if "othergen" in fakes:
        gen += [subset(d, (d["label"] == 1) & ~np.isin(d["group"], list(KANDINSKY))) for d in others]
    if gen:
        g = concat(gen)
        xs.append(view(g, feats))
        ys.append(np.ones(len(g["id"])))
    return np.concatenate(xs), np.concatenate(ys)


def fit_and_threshold(feats, fakes, c, use_ebay, others, train, k_train, val_l0):
    """Fit on everything outside the held-out version; the threshold uses Label_0 validation and the other
    versions' eBay photos, scored out-of-fold (5 folds) when those photos are also training reals."""
    other_real = concat([subset(d, d["label"] == 0) for d in others])
    if feats == "aero":
        score = lambda d: -d[fakes]  # noqa: E731  lower reconstruction error = more likely generated
        real_scores = np.concatenate([score(val_l0), score(other_real)])
        return score, threshold(real_scores), real_scores
    model = fit_lr(*training_set(feats, fakes, others, train, k_train, other_real if use_ebay else None), c)
    score = lambda d, m=model: m.decision_function(view(d, feats))  # noqa: E731
    ebay_scores = score(other_real)
    if use_ebay:
        folds = np.arange(len(other_real["id"])) % 5
        for f in range(5):
            m = fit_lr(*training_set(feats, fakes, others, train, k_train, subset(other_real, folds != f)), c)
            ebay_scores[folds == f] = m.decision_function(view(subset(other_real, folds == f), feats))
    real_scores = np.concatenate([score(val_l0), ebay_scores])
    return score, threshold(real_scores), real_scores


def compare(names: list[str]) -> None:
    bench = {v: load(f"bench_{v}") for v in VERSIONS}
    train, val_l0, val_l1, k_train = load("train_l0"), load("val_l0"), load("val_l1"), load("k_train")
    path = OUT / "design_compare.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        feats, fakes, c, use_ebay = CANDIDATES[name]
        results[name] = {"k_train_n": 0 if k_train is None else int(len(k_train["id"]))}
        for held in VERSIONS:
            others = [bench[v] for v in VERSIONS if v != held]
            score, t, real_scores = fit_and_threshold(feats, fakes, c, use_ebay, others, train, k_train, val_l0)
            test = bench[held]
            s = score(test)
            row = {"threshold_real_fake_rate": float((real_scores >= t).mean())}
            real = s[test["label"] == 0]
            row["real_fake"] = [int((real >= t).sum()), int(len(real))]
            for group in sorted(g for g in set(test["group"]) if not g.startswith("real")):
                gs = s[test["group"] == group]
                row[group] = [int((gs >= t).sum()), int(len(gs)), round(roc_auc(np.concatenate([real, gs]), np.concatenate([np.zeros(len(real)), np.ones(len(gs))])), 4)]
            l1 = score(val_l1)
            row["val_label1"] = [int((l1 >= t).sum()), int(len(l1))]
            results[name][held] = row
        k = [results[name][v][g] for v in VERSIONS for g in results[name][v] if g in KANDINSKY]
        r = [results[name][v]["real_fake"] for v in VERSIONS]
        results[name]["pooled"] = {"kandinsky_caught": [sum(a for a, *_ in k), sum(b for _, b, *_ in k)],
                                   "real_fake": [sum(a for a, _ in r), sum(b for _, b in r)]}
        p = results[name]["pooled"]
        others_txt = "  ".join(f"{g}:{results[name][v][g][0]}/{results[name][v][g][1]}" for v in VERSIONS
                               for g in results[name][v] if g not in KANDINSKY and g not in ("threshold_real_fake_rate", "real_fake"))
        print(f"{name:24s} Kandinsky {p['kandinsky_caught'][0]}/{p['kandinsky_caught'][1]} "
              f"({p['kandinsky_caught'][0] / p['kandinsky_caught'][1]:.1%})  real FAKE {p['real_fake'][0]}/{p['real_fake'][1]}  {others_txt}", flush=True)
        path.write_text(json.dumps(results, indent=2) + "\n")


def freeze() -> None:
    """Fit the selected candidate on all development data and fix its thresholds from real photos only."""
    import joblib

    feats, fakes, c, _ = CANDIDATES[SELECTED]
    bench = [load(f"bench_{v}") for v in VERSIONS]
    train, val_l0, k_train = load("train_l0"), load("val_l0"), load("k_train")
    ebay = concat([subset(d, d["label"] == 0) for d in bench])
    model = fit_lr(*training_set(feats, fakes, bench, train, k_train, ebay), c)
    # Real photos for thresholds: Label_0 validation (never trained on) + every dev eBay photo, out-of-fold.
    folds = np.arange(len(ebay["id"])) % 5
    ebay_oof = np.zeros(len(ebay["id"]))
    for f in range(5):
        m = fit_lr(*training_set(feats, fakes, bench, train, k_train, subset(ebay, folds != f)), c)
        ebay_oof[folds == f] = m.decision_function(view(subset(ebay, folds == f), feats))
    real = np.concatenate([model.decision_function(view(val_l0, feats)), ebay_oof])
    t_fake = threshold(real)
    grid = np.unique(real)[::-1]
    t_review = float(min(t for t in grid if t <= t_fake and ((real >= t) & (real < t_fake)).mean() <= REVIEW_BUDGET))
    path = OUT / "dinov3_kandinsky_v4_head.joblib"
    joblib.dump(model, path)
    compare_results = json.loads((OUT / "design_compare.json").read_text())
    rule = {
        "rule": "dinov3_kandinsky_v4",
        "phase": 74,
        "frozen_before_any_v4_scoring": True,
        "selected_candidate": SELECTED,
        "features": feats,
        "inputs": "fine-tuned DINOv3 epoch_018 (hash-checked, FP32): CLS token and attention-pooled patch embedding, "
                  "the two vectors its own head already uses; one forward pass, no extra model",
        "head": "StandardScaler + logistic regression (C = 0.05, balanced classes)",
        "trained_on": {
            "real": "5,000 Label_0 training photos + 900 benchmark_v1-v3 eBay photos",
            "generated": "Kandinsky 2.2: benchmark_v1-v3 (852) + kandinsky22_train (534, scenes T1-T8); "
                         "SDXL (150), aMUSEd (293), Stable Cascade lite (296) from benchmark_v1-v3",
        },
        "model": {"path": str(path.relative_to(_PKG.parent)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "thresholds": {"t_fake": t_fake, "t_review": t_review,
                       "source": "real photos only: 1,000 Label_0 validation + 900 eBay (out-of-fold); at most 1% FAKE and at most 5% more REVIEW"},
        "decision": "FAKE if score >= t_fake; REVIEW if score >= t_review or DINOv3 REVIEW; else AUTHENTIC (score = head decision function)",
        "development_results": {"leave_one_version_out": compare_results.get(SELECTED, {}).get("pooled"),
                                "scene_holdout": "trained on T1-T4 scenes: 100% of T5-T8 Kandinsky caught; trained on T5-T8: 97.9-100% of T1-T4"},
    }
    (OUT / "fusion_rule.json").write_text(json.dumps(rule, indent=2) + "\n")
    print(json.dumps({"t_fake": t_fake, "t_review": t_review, "real_fake_rate": float((real >= t_fake).mean())}))
    print("fusion_rule.json sha256", hashlib.sha256((OUT / "fusion_rule.json").read_bytes()).hexdigest())


def unseen() -> None:
    """Leave one generator out of training entirely (Kandinsky = all 1,386 development Kandinsky images) and score it.

    Run after benchmark_v4 to ask how the Phase 74 recipe does on a generator it has never seen. Development
    data only; the threshold uses real photos only (Label_0 validation + out-of-fold eBay), at most 1% FAKE.
    """
    bench = [load(f"bench_{v}") for v in VERSIONS]
    train, val_l0, k_train = load("train_l0"), load("val_l0"), load("k_train")
    allb = concat(bench)
    ebay = subset(allb, allb["label"] == 0)
    kand = concat([subset(allb, np.isin(allb["group"], list(KANDINSKY))), k_train])
    others = sorted(g for g in set(allb["group"]) if g not in KANDINSKY and not g.startswith("real"))
    folds = np.arange(len(ebay["id"])) % 5
    results = {}
    for feats in ("ft", "pre", "pre+ft"):
        for held in ("kandinsky", *others):
            gen = concat([subset(allb, allb["group"] == g) for g in others if g != held] + ([kand] if held != "kandinsky" else []))

            def fit(reals: dict):
                x = np.concatenate([view(train, feats), view(reals, feats), view(gen, feats)])
                y = np.r_[np.zeros(len(train["id"]) + len(reals["id"])), np.ones(len(gen["id"]))]
                return fit_lr(x, y, 0.05)

            model = fit(ebay)
            oof = np.zeros(len(ebay["id"]))
            for f in range(5):
                oof[folds == f] = fit(subset(ebay, folds != f)).decision_function(view(subset(ebay, folds == f), feats))
            t = threshold(np.concatenate([model.decision_function(view(val_l0, feats)), oof]))
            test = kand if held == "kandinsky" else subset(allb, allb["group"] == held)
            caught = int((model.decision_function(view(test, feats)) >= t).sum())
            results[f"{feats}|{held}"] = {"caught": caught, "n": int(len(test["id"])), "ebay_real_fake_oof": float((oof >= t).mean())}
            print(f"{feats:7s} held-out {held:20s} caught {caught}/{len(test['id'])} ({caught / len(test['id']):.1%})  "
                  f"eBay real FAKE (out-of-fold) {(oof >= t).mean():.1%}", flush=True)
    (OUT / "leave_one_generator_out.json").write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("compare", "freeze", "unseen"))
    parser.add_argument("--only", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "unseen":
        unseen()
    else:
        compare(args.only)


if __name__ == "__main__":
    main()
