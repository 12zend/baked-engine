"""Conservative rasterisation of atlas triangles into texel records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Texels:
    x: np.ndarray  # (m,) column
    y: np.ndarray  # (m,) row (row 0 = v 0)
    tri: np.ndarray  # (m,) owning triangle
    bary: np.ndarray  # (m, 3) barycentric of the texel centre, clamped into the triangle
    inside: np.ndarray  # (m,) texel centre lies inside its triangle
    jac: np.ndarray  # (n_tris, 3, 2) d(bary)/d(pixel) per triangle

    def __len__(self):
        return len(self.x)

    def subset(self, mask):
        return Texels(self.x[mask], self.y[mask], self.tri[mask], self.bary[mask], self.inside[mask], self.jac)

    def jittered_bary(self, jitter):
        """Barycentrics for sample positions ``centre + jitter`` (m, 2), clamped to the triangle."""
        b = self.bary + np.einsum("mij,mj->mi", self.jac[self.tri], jitter)
        return clamp_bary(b)


def clamp_bary(b):
    b = np.maximum(b, 0.0)
    s = b.sum(axis=1, keepdims=True)
    return np.where(s > 1e-12, b / np.maximum(s, 1e-12), 1.0 / 3.0)


def rasterize(tri_px, width, height, conservative=0.75, chunk=1 << 21):
    tri_px = np.asarray(tri_px, dtype=np.float64)
    n = len(tri_px)
    a = tri_px[:, 0]
    e1 = tri_px[:, 1] - a
    e2 = tri_px[:, 2] - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    # (lambda1, lambda2) = M (p - a), M = inverse([e1 e2])
    M = np.empty((n, 2, 2))
    M[:, 0, 0] = e2[:, 1] * inv
    M[:, 0, 1] = -e2[:, 0] * inv
    M[:, 1, 0] = -e1[:, 1] * inv
    M[:, 1, 1] = e1[:, 0] * inv
    jac = np.zeros((n, 3, 2))
    jac[:, 1] = M[:, 0]
    jac[:, 2] = M[:, 1]
    jac[:, 0] = -(M[:, 0] + M[:, 1])
    # height of each vertex above its opposite edge, so lambda_i * h_i is a distance in pixels
    edge_len = np.stack(
        [
            np.linalg.norm(tri_px[:, 2] - tri_px[:, 1], axis=1),
            np.linalg.norm(tri_px[:, 0] - tri_px[:, 2], axis=1),
            np.linalg.norm(tri_px[:, 1] - tri_px[:, 0], axis=1),
        ],
        axis=1,
    )
    h = np.abs(det)[:, None] / np.maximum(edge_len, 1e-12)

    lo = tri_px.min(axis=1)
    hi = tri_px.max(axis=1)
    x0 = np.clip(np.ceil(lo[:, 0] - conservative - 0.5), 0, width - 1).astype(np.int64)
    x1 = np.clip(np.floor(hi[:, 0] + conservative - 0.5), -1, width - 1).astype(np.int64)
    y0 = np.clip(np.ceil(lo[:, 1] - conservative - 0.5), 0, height - 1).astype(np.int64)
    y1 = np.clip(np.floor(hi[:, 1] + conservative - 0.5), -1, height - 1).astype(np.int64)
    nx = np.maximum(x1 - x0 + 1, 0)
    ny = np.maximum(y1 - y0 + 1, 0)
    cnt = np.where(ok, nx * ny, 0)

    best_score = np.full(width * height, -np.inf, dtype=np.float32)
    best_tri = np.full(width * height, -1, dtype=np.int64)

    csum = np.cumsum(cnt)
    start = 0
    while start < n:
        base = csum[start - 1] if start > 0 else 0
        end = int(np.searchsorted(csum, base + chunk, side="right"))
        end = max(end, start + 1)
        ids = np.arange(start, end)
        c = cnt[ids]
        tot = int(c.sum())
        if tot:
            t = np.repeat(ids, c)
            k = np.arange(tot) - np.repeat(np.cumsum(c) - c, c)
            px = x0[t] + k % nx[t]
            py = y0[t] + k // nx[t]
            dx = px + 0.5 - a[t, 0]
            dy = py + 0.5 - a[t, 1]
            l1 = M[t, 0, 0] * dx + M[t, 0, 1] * dy
            l2 = M[t, 1, 0] * dx + M[t, 1, 1] * dy
            l0 = 1 - l1 - l2
            score = np.minimum(np.minimum(l0 * h[t, 0], l1 * h[t, 1]), l2 * h[t, 2])
            keep = score >= -conservative
            key = (py * width + px)[keep]
            score = score[keep].astype(np.float32)
            t = t[keep]
            o = np.lexsort((score, key))
            key, score, t = key[o], score[o], t[o]
            last = np.ones(len(key), dtype=bool)
            last[:-1] = key[1:] != key[:-1]
            key, score, t = key[last], score[last], t[last]
            better = score > best_score[key]
            best_score[key[better]] = score[better]
            best_tri[key[better]] = t[better]
        start = end

    texel = np.nonzero(best_tri >= 0)[0]
    tri = best_tri[texel]
    px = texel % width
    py = texel // width
    dx = px + 0.5 - a[tri, 0]
    dy = py + 0.5 - a[tri, 1]
    l1 = M[tri, 0, 0] * dx + M[tri, 0, 1] * dy
    l2 = M[tri, 1, 0] * dx + M[tri, 1, 1] * dy
    bary = np.stack([1 - l1 - l2, l1, l2], axis=1)
    inside = best_score[texel] >= 0
    return Texels(px, py, tri, clamp_bary(bary), inside, jac)
