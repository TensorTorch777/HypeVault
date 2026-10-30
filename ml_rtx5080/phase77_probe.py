"""Phase 77 probe (exploratory): can the Phase 76 hybrid be made stronger on generators it never saw?

Candidate changes, compared on development data only (nothing here is a confirmatory result):
  features  - Perception Encoder at the stored 512 px (no downscale to 336), and its middle layers
              (mean patch tokens after blocks 12 and 18 of 24) next to the Phase 76 features;
  data      - benchmark_v6 (spent) watch images, and OpenFake development generators (phase77_openfake.py).

Selection is leave-one-family-out: each generator family is held out of training in turn and the
share of its images caught at the real-photo cut (at most 1% of real watch photos FAKE) is averaged.

    .venv/bin/python ml_rtx5080/phase77_probe.py extract
    .venv/bin/python ml_rtx5080/phase77_probe.py compare
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ML = Path(__file__).resolve().parent
REPO = ML.parent
OUT = ML / "experiments" / "phase77_probe"
CACHE = OUT / "features"
B5 = ML / "benchmark_v5_frontier"
B6 = ML / "benchmark_v6_fresh"
for p in (str(ML / "benchmark_v1_real_vs_generated"), str(ML)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

PE_MODEL = "vit_pe_core_large_patch14_336.fb"
PE_SIDE = 512
PE_MID_BLOCKS = (11, 17)  # zero-based: after blocks 12 and 18 of 24
BATCH = 32


class PE512:
    """Perception Encoder L/14 run at 512 px: pooled output plus mean patch tokens of two middle blocks."""

    def __init__(self) -> None:
        import timm
        import torch

        self.torch = torch
        self.model = timm.create_model(PE_MODEL, pretrained=True, num_classes=0, img_size=PE_SIDE).cuda().eval().half()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).cuda().half()
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).cuda().half()

    def __call__(self, images) -> dict[str, np.ndarray]:
        torch = self.torch
        x = torch.from_numpy(np.stack([np.asarray(i.resize((PE_SIDE, PE_SIDE)) if i.size != (PE_SIDE, PE_SIDE) else i)
                                       for i in images])).permute(0, 3, 1, 2).cuda().half() / 255.0
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            pooled = self.model(x)
            _, inter = self.model.forward_intermediates(x, indices=list(PE_MID_BLOCKS), output_fmt="NLC",
                                                        return_prefix_tokens=True, norm=False)
        out = {"pe512": pooled.float().cpu().numpy()}
        for b, (tokens, _prefix) in zip(PE_MID_BLOCKS, inter):
            out[f"pe512_m{b + 1}"] = tokens.mean(1).float().cpu().numpy()
        return out


def _v5_items():
    import kandinsky_v4 as k4
    from common import read_manifest

    z = np.load(ML / "experiments" / "phase76_probe" / "v5_features.npz")
    ids = z["id"].tolist()
    paths = {s.sample_id: (B5 / s.path if not Path(s.path).is_absolute() else Path(s.path))
             for s in read_manifest(B5 / "data" / "manifest.jsonl") if s.excluded is None}

    def load(i):
        return k4.jpeg90(k4.square(REPO / "testtest.jpeg", 512)) if i == "testtest" else k4.stored(paths[i])

    return ids, load


def _v6_rows() -> list[dict]:
    rows = []
    for line in (B6 / "data" / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("excluded") is None and r.get("path"):
            rows.append({"id": r["sample_id"], "group": r["group"], "brand": r["brand"], "path": r["path"],
                         "label": int(r["label"] != "REAL_PHOTO")})
    return rows


def sets() -> dict:
    """name -> (ids, loader, needs DINOv3 features too)."""
    import fusion_probe as fp
    import kandinsky_v4 as k4

    out = {name: (ids, load, False) for name, (ids, load) in fp._sets().items()}
    out["v5"] = (*_v5_items(), False)
    v6 = _v6_rows()
    paths = {r["id"]: r["path"] for r in v6}
    out["v6"] = ([r["id"] for r in v6], lambda i, p=paths: k4.stored(p[i]), True)
    try:
        import phase77_openfake as of

        for name, (ids, load) in of.sets().items():
            out[name] = (ids, load, True)
    except (ImportError, FileNotFoundError):
        pass
    return out


def extract(only: list[str] | None = None) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import kandinsky_v4 as k4

    CACHE.mkdir(parents=True, exist_ok=True)
    import fusion_probe as fp

    pe = PE512()
    d3 = None  # DINOv3 (pretrained, fine-tuned) and the Phase 76 PE at 336, for sets without Phase 76 caches
    with ThreadPoolExecutor(8) as pool:
        for name, (ids, load, needs_d3) in sets().items():
            target = CACHE / f"{name}.npz"
            if (only and name not in only) or target.exists():
                continue
            if needs_d3 and d3 is None:
                d3 = (k4.Pretrained(), k4.FineTuned(), fp.Encoders(("pe",)))
            store: dict[str, list] = {}
            for s in range(0, len(ids), BATCH):
                images = list(pool.map(load, ids[s:s + BATCH]))
                parts = [pe(images)]
                if needs_d3:
                    for b in range(0, len(images), k4.BATCH):
                        chunk = images[b:b + k4.BATCH]
                        parts += [d3[0].features(chunk), d3[1].features(chunk), d3[2](chunk)]
                for part in parts:
                    for k, v in part.items():
                        store.setdefault(k, []).append(np.asarray(v))
            np.savez(target, id=np.array(ids), **{k: np.concatenate(v) for k, v in store.items()})
            print(f"{name}: {len(ids)} images", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2:] or None)
    elif sys.argv[1] == "compare":
        import phase77_compare

        phase77_compare.main()
