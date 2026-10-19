# Benchmark v1 — real listing photos vs AI-generated images

Pre-registered paired comparison of the frozen DINOv2 (live) and DINOv3 (candidate) checkpoints.
`protocol.json` was written before any data existed and fixes the classes, normalisation,
exclusions, sample sizes, analysis and the rule that decides the frontier model.

**Scope.** This answers "is this listing photo real or AI-generated?". It does not test whether a
watch is genuine, so it cannot lift the repository's `NO_GO_FOR_AUTHENTICITY_CLAIMS` status.

## Status (16 Oct 2026)

| Step | State |
|---|---|
| Protocol | Pre-registered |
| Pipeline smoke test (tiny test generator, CPU) | Passed: generation, normalisation, exclusion checks, paired Triton scoring, report |
| Real photos (eBay Browse API) | Waiting for eBay developer keys |
| Held-out generated set (Kandinsky 2.2) | Waiting for a free GPU |
| SDXL secondary set | Waiting for a free GPU |
| Confirmatory run | Not started |

## Run order

```bash
# 1. eBay keys in .env: EBAY_CLIENT_ID, EBAY_CLIENT_SECRET, MARKET_PSEUDONYM_SALT (any long random string)
.venv/bin/python ml_rtx5080/benchmark_v1_real_vs_generated/collect_real.py --per-brand 60

# 2. Generated images (GPU; about 6 GB with CPU offload)
.venv/bin/python ml_rtx5080/benchmark_v1_real_vs_generated/generate_synthetic.py --group heldout_kandinsky22 --per-brand 60
.venv/bin/python ml_rtx5080/benchmark_v1_real_vs_generated/generate_synthetic.py --group sdxl_base10 --per-brand 30

# 3. Score both models (Triton must serve both allow-listed models)
HYPEVAULT_DEPLOYMENT_MODE=research .venv/bin/python ml_rtx5080/benchmark_v1_real_vs_generated/run_benchmark.py
```

Each run writes `results/<run id>/metrics.json`, `report.md` and `predictions.csv`. A run that does
not meet the per-brand minimums is labelled `SMOKE_TEST`, and its numbers are not results.

## Files

| File | Role |
|---|---|
| `protocol.json` | Pre-registered design and frontier rule |
| `common.py` | Normalisation to 768 × 768 JPEG, hashing, exclusion against the stored training hashes |
| `collect_real.py` | REAL_PHOTO collection through the eBay Browse API |
| `generate_synthetic.py` | GENERATED collection with recorded generator, prompt and seed |
| `run_benchmark.py` | Paired scoring through the serving path and the analysis |
| `bench_stats.py` | Exact intervals, exact McNemar, power, cluster bootstrap |

Images in `data/` are ignored by git and are never redistributed.
