"""Reference gallery: one fingerprint per eBay listing photo, labelled with the name in its title.

Built offline by scripts/build_watch_gallery.py. The photos themselves are not kept: only the
vector, the parsed name, the title and the listing link.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FIELDS = ("external_id", "brand", "model", "reference", "title", "item_url")


@dataclass(frozen=True)
class Neighbour:
    index: int
    similarity: float
    external_id: str
    brand: str
    model: str | None
    reference: str | None
    title: str
    item_url: str | None


class Gallery:
    def __init__(self, vectors: np.ndarray, meta: dict[str, list], *, pool: str, built_on: str | None = None) -> None:
        if vectors.ndim != 2 or any(len(meta[f]) != len(vectors) for f in _FIELDS):
            raise ValueError("gallery vectors and labels do not line up")
        self.vectors = vectors.astype(np.float32)
        self.meta = meta
        self.pool = pool
        self.built_on = built_on

    def __len__(self) -> int:
        return len(self.vectors)

    def neighbours(self, vec: np.ndarray, k: int = 15, *, exclude: set[int] | None = None) -> list[Neighbour]:
        sims = self.vectors @ vec.astype(np.float32)
        if exclude:
            sims = sims.copy()
            sims[list(exclude)] = -np.inf
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [self._neighbour(int(i), float(sims[i])) for i in top if np.isfinite(sims[i])]

    def _neighbour(self, i: int, sim: float) -> Neighbour:
        m = self.meta
        return Neighbour(i, sim, m["external_id"][i], m["brand"][i], m["model"][i] or None,
                         m["reference"][i] or None, m["title"][i], m["item_url"][i] or None)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors.astype(np.float16),
            pool=np.array(self.pool),
            built_on=np.array(self.built_on or ""),
            **{f: np.array([v or "" for v in self.meta[f]], dtype=str) for f in _FIELDS},
        )

    @classmethod
    def load(cls, path: Path) -> "Gallery":
        with np.load(path, allow_pickle=False) as z:
            meta = {f: z[f].tolist() for f in _FIELDS}
            return cls(z["vectors"], meta, pool=str(z["pool"]), built_on=str(z["built_on"]) or None)
