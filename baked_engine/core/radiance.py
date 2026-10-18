"""Look up baked radiance for any point of the original scene, given (triangle, barycentric).

Every original triangle is covered by one or more *pieces* (clipped parts) that
live either in the main atlas or in the auxiliary atlas. The lookup finds the
piece containing the point and bilinearly samples the right image.
"""

from __future__ import annotations

import numpy as np


def bilinear(img, px):
    """Sample ``img`` (H, W, C) at pixel coordinates ``px`` (n, 2), texel centres at i + 0.5."""
    h, w = img.shape[:2]
    x = px[:, 0] - 0.5
    y = px[:, 1] - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0 = np.clip(x0.astype(np.int64), 0, w - 1)
    y0 = np.clip(y0.astype(np.int64), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    b = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return a * (1 - fy) + b * fy


class PieceIndex:
    def __init__(self, tri_count, src, B, atlas, uv_px):
        """src (p,), B (p,3,3) corner barycentrics, atlas (p,) 0/1, uv_px (p,3,2) pixel coords."""
        src = np.asarray(src, dtype=np.int64)
        order = np.argsort(src, kind="stable")
        self.src = src[order]
        Bs = np.asarray(B, dtype=np.float64)[order]
        det = np.linalg.det(Bs)
        ok = np.abs(det) > 1e-14
        Binv = np.zeros_like(Bs)
        Binv[ok] = np.linalg.inv(Bs[ok])
        self.Binv = Binv
        self.valid = ok
        self.atlas = np.asarray(atlas, dtype=np.int8)[order]
        self.uv = np.asarray(uv_px, dtype=np.float64)[order]
        counts = np.bincount(self.src, minlength=tri_count)
        self.start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.count = counts
        self.max_count = int(counts.max()) if len(counts) else 0

    def locate(self, tri, bary):
        """Returns (piece index or -1, local barycentric (n,3))."""
        n = len(tri)
        best = np.full(n, -1, dtype=np.int64)
        best_score = np.full(n, -np.inf)
        best_lam = np.zeros((n, 3))
        cnt = self.count[tri]
        st = self.start[tri]
        for k in range(self.max_count):
            m = (cnt > k)
            if not m.any():
                break
            idx = np.nonzero(m)[0]
            p = st[idx] + k
            lam = np.einsum("ni,nij->nj", bary[idx], self.Binv[p])
            score = np.where(self.valid[p], lam.min(axis=1), -np.inf)
            better = score > best_score[idx]
            j = idx[better]
            best[j] = p[better]
            best_score[j] = score[better]
            best_lam[j] = lam[better]
        lam = np.maximum(best_lam, 0)
        lam /= np.maximum(lam.sum(axis=1, keepdims=True), 1e-12)
        return best, lam


class RadianceLookup:
    def __init__(self, pieces: PieceIndex, images):
        self.pieces = pieces
        self.images = images  # list indexed by atlas id

    def __call__(self, tri, bary, channels=3):
        n = len(tri)
        out = np.zeros((n, channels))
        if n == 0:
            return out
        p, lam = self.pieces.locate(tri, bary)
        found = p >= 0
        for a, img in enumerate(self.images):
            m = found & (self.pieces.atlas[np.maximum(p, 0)] == a)
            if not m.any() or img is None:
                continue
            uv = np.einsum("ni,nij->nj", lam[m], self.pieces.uv[p[m]])
            out[m] = bilinear(img, uv)[:, :channels]
        return out
