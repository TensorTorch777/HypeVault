"""Statistics for the paired real-vs-generated benchmark. Pure functions, no I/O."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Sequence

from scipy.stats import beta, binom, binomtest


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided (1 - alpha) interval for a binomial proportion."""
    if n <= 0:
        return (math.nan, math.nan)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def one_sided_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided (1 - alpha) upper bound for a binomial proportion."""
    if n <= 0:
        return math.nan
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def required_discordant(split: float = 0.7, alpha: float = 0.05, power: float = 0.8, n_max: int = 2000) -> int:
    """Smallest discordant-pair count giving the requested power for an exact McNemar test."""
    for n in range(5, n_max + 1):
        # rejection region of the two-sided exact test under H0: p = 0.5
        reject = [k for k in range(n + 1) if binomtest(k, n, 0.5).pvalue <= alpha]
        achieved = sum(binom.pmf(k, n, split) for k in reject)
        if achieved >= power:
            return n
    raise ValueError("power not reached within n_max")


def cluster_bootstrap_diff(
    a_correct: Sequence[int],
    b_correct: Sequence[int],
    clusters: Sequence[str],
    *,
    iterations: int = 2000,
    seed: int = 20261016,
) -> tuple[float, float, float]:
    """Mean paired difference (a - b) in correctness with a percentile cluster-bootstrap 95% interval."""
    if not (len(a_correct) == len(b_correct) == len(clusters)) or not clusters:
        raise ValueError("inputs must be non-empty and the same length")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = list(groups)
    point = (sum(a_correct) - sum(b_correct)) / len(clusters)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in groups[k]]
        diffs.append(sum(a_correct[i] - b_correct[i] for i in idx) / len(idx))
    diffs.sort()
    lo = diffs[int(0.025 * (iterations - 1))]
    hi = diffs[int(0.975 * (iterations - 1))]
    return point, lo, hi


def binary_metrics(labels: Sequence[str], decisions: Sequence[str]) -> dict:
    """Positive class = GENERATED. A model's FAKE means 'generated', AUTHENTIC means 'real photo'.

    REVIEW stays in the coverage denominator and out of the binary metrics.
    """
    tp = fn = fp = tn = review_gen = review_real = 0
    for label, decision in zip(labels, decisions, strict=True):
        generated = label == "GENERATED"
        if decision == "REVIEW":
            review_gen += generated
            review_real += not generated
        elif decision == "FAKE":
            tp += generated
            fp += not generated
        elif decision == "AUTHENTIC":
            fn += generated
            tn += not generated
        else:
            raise ValueError(f"unexpected decision {decision!r}")
    n_gen, n_real = tp + fn + review_gen, fp + tn + review_real
    decided = tp + fn + fp + tn

    def ratio(k, n):
        return None if n == 0 else k / n

    tpr, tnr = ratio(tp, tp + fn), ratio(tn, tn + fp)
    precision = ratio(tp, tp + fp)
    f1 = None if not precision or not tpr else 2 * precision * tpr / (precision + tpr)
    return {
        "confusion": {"generated_as_fake": tp, "generated_as_authentic": fn, "real_as_fake": fp, "real_as_authentic": tn},
        "review": {"generated": review_gen, "real": review_real},
        "n": {"generated": n_gen, "real": n_real},
        "coverage": ratio(decided, n_gen + n_real),
        "accuracy": ratio(tp + tn, decided),
        "balanced_accuracy": None if tpr is None or tnr is None else (tpr + tnr) / 2,
        "precision_generated": precision,
        "recall_generated": tpr,
        "f1_generated": f1,
        # The costly error: a generated image passed as a real photo. Denominator: all generated images.
        "false_authentic": {"k": fn, "n": n_gen, "rate": ratio(fn, n_gen), "upper95": one_sided_upper(fn, n_gen)},
        "false_fake": {"k": fp, "n": n_real, "rate": ratio(fp, n_real), "upper95": one_sided_upper(fp, n_real)},
    }
