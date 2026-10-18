"""Bounding volume hierarchy with vectorised (numpy) ray traversal.

Build: binned SAH, all nodes of one tree level are split together.
Traversal: every ray owns a small stack; each loop iteration pops one node
for every active ray, so the Python-level loop count is the longest
traversal path, not the number of rays.
"""

from __future__ import annotations

import numpy as np

_BINS = 16
_LEAF_MAX = 8
_MAX_DEPTH = 60
_STACK = 64
_CHUNK = 1 << 18


def _ranges(starts, counts):
    """Concatenate arange(s, s + c) for every (s, c)."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offs = np.repeat(np.asarray(starts, dtype=np.int64) - np.cumsum(counts) + counts, counts)
    return offs + np.arange(total)


def _box_area(lo, hi):
    e = np.maximum(hi - lo, 0)
    return 2 * (e[..., 0] * e[..., 1] + e[..., 1] * e[..., 2] + e[..., 2] * e[..., 0])


def build(P, leaf_target=4):
    """Return (order, nodes dict). ``order[slot]`` is the triangle stored at leaf slot ``slot``."""
    P = np.asarray(P, dtype=np.float64)
    T = len(P)
    lo = P.min(axis=1)
    hi = P.max(axis=1)
    cen = 0.5 * (lo + hi)
    order = np.arange(T, dtype=np.int64)
    cap = max(2 * T, 1)
    n_lo = np.zeros((cap, 3))
    n_hi = np.zeros((cap, 3))
    n_left = np.full(cap, -1, dtype=np.int32)
    n_axis = np.zeros(cap, dtype=np.int8)
    n_start = np.zeros(cap, dtype=np.int32)
    n_count = np.zeros(cap, dtype=np.int32)
    if T == 0:
        n_lo[0] = 1
        n_hi[0] = -1
        return order, dict(lo=n_lo[:1], hi=n_hi[:1], left=n_left[:1], axis=n_axis[:1], start=n_start[:1], count=n_count[:1])

    p_id = np.array([0], dtype=np.int64)
    p_start = np.array([0], dtype=np.int64)
    p_cnt = np.array([T], dtype=np.int64)
    next_id = 1
    depth = 0
    while len(p_id):
        pos = _ranges(p_start, p_cnt)
        tri = order[pos]
        seg = np.concatenate([[0], np.cumsum(p_cnt)[:-1]])
        b_lo = np.minimum.reduceat(lo[tri], seg, axis=0)
        b_hi = np.maximum.reduceat(hi[tri], seg, axis=0)
        c_lo = np.minimum.reduceat(cen[tri], seg, axis=0)
        c_hi = np.maximum.reduceat(cen[tri], seg, axis=0)
        n_lo[p_id] = b_lo
        n_hi[p_id] = b_hi
        n_start[p_id] = p_start
        n_count[p_id] = p_cnt
        ext = c_hi - c_lo
        axis = np.argmax(ext, axis=1)
        ext_ax = ext[np.arange(len(p_id)), axis]

        force_leaf = (p_cnt <= leaf_target) | (depth >= _MAX_DEPTH - 1)
        k_all = np.repeat(np.arange(len(p_id)), p_cnt)
        # ---- binned SAH for every node, evaluated jointly
        c = cen[tri, axis[k_all]]
        scale = np.where(ext_ax > 0, _BINS / np.maximum(ext_ax, 1e-300), 0.0)
        b = np.clip(((c - c_lo[k_all, axis[k_all]]) * scale[k_all]).astype(np.int64), 0, _BINS - 1)
        key = k_all * _BINS + b
        nn = len(p_id)
        cnt_bins = np.bincount(key, minlength=nn * _BINS).reshape(nn, _BINS)
        bin_lo = np.full((nn * _BINS, 3), np.inf)
        bin_hi = np.full((nn * _BINS, 3), -np.inf)
        np.minimum.at(bin_lo, key, lo[tri])
        np.maximum.at(bin_hi, key, hi[tri])
        bin_lo = bin_lo.reshape(nn, _BINS, 3)
        bin_hi = bin_hi.reshape(nn, _BINS, 3)
        L_lo = np.minimum.accumulate(bin_lo, axis=1)[:, :-1]
        L_hi = np.maximum.accumulate(bin_hi, axis=1)[:, :-1]
        L_cnt = np.cumsum(cnt_bins, axis=1)[:, :-1]
        R_lo = np.minimum.accumulate(bin_lo[:, ::-1], axis=1)[:, ::-1][:, 1:]
        R_hi = np.maximum.accumulate(bin_hi[:, ::-1], axis=1)[:, ::-1][:, 1:]
        R_cnt = np.cumsum(cnt_bins[:, ::-1], axis=1)[:, ::-1][:, 1:]
        with np.errstate(invalid="ignore"):
            cost = np.where(L_cnt > 0, _box_area(L_lo, L_hi) * L_cnt, 0) + np.where(R_cnt > 0, _box_area(R_lo, R_hi) * R_cnt, 0)
        cost = np.where((L_cnt > 0) & (R_cnt > 0), cost, np.inf)
        best = np.argmin(cost, axis=1)
        best_cost = cost[np.arange(nn), best]
        node_area = np.maximum(_box_area(b_lo, b_hi), 1e-300)
        sah_split_cost = 0.5 + best_cost / node_area
        sah_leaf = (p_cnt <= _LEAF_MAX) & (sah_split_cost >= p_cnt)
        leaf = force_leaf | sah_leaf
        use_median = ~np.isfinite(best_cost) & ~leaf
        split = ~leaf

        if not split.any():
            break
        # ---- partition the triangles of split nodes
        tri_split = split[k_all]
        side = b > best[k_all]  # bins <= best go left
        if use_median.any():
            med_mask = use_median[k_all]
            # Rank by centroid within node; split at half.
            idx = np.nonzero(med_mask)[0]
            perm = np.lexsort((c[idx], k_all[idx]))
            sorted_idx = idx[perm]
            first = np.searchsorted(k_all[sorted_idx], k_all[sorted_idx], side="left")
            rank = np.arange(len(sorted_idx)) - first
            side_med = np.zeros(len(sorted_idx), dtype=bool)
            side_med[:] = rank >= (p_cnt[k_all[sorted_idx]] // 2)
            side[sorted_idx] = side_med
        sel = np.nonzero(tri_split)[0]
        sort_key = k_all[sel] * 2 + side[sel]
        perm = np.argsort(sort_key, kind="stable")
        order[pos[sel]] = tri[sel][perm]
        left_cnt = np.bincount(k_all[sel], weights=(~side[sel]).astype(np.float64), minlength=nn).astype(np.int64)

        s_ids = np.nonzero(split)[0]
        n_new = len(s_ids)
        left_ids = next_id + 2 * np.arange(n_new)
        n_left[p_id[s_ids]] = left_ids
        n_axis[p_id[s_ids]] = axis[s_ids]
        n_count[p_id[s_ids]] = 0
        next_id += 2 * n_new
        lc = left_cnt[s_ids]
        p_id = np.stack([left_ids, left_ids + 1], axis=1).ravel()
        p_start = np.stack([p_start[s_ids], p_start[s_ids] + lc], axis=1).ravel()
        p_cnt = np.stack([lc, p_cnt[s_ids] - lc], axis=1).ravel()
        depth += 1

    m = next_id
    return order, dict(
        lo=n_lo[:m], hi=n_hi[:m], left=n_left[:m], axis=n_axis[:m], start=n_start[:m], count=n_count[:m]
    )


class Hits:
    __slots__ = ("t", "tri", "u", "v", "front")

    def __init__(self, n):
        self.t = np.full(n, np.inf, dtype=np.float64)
        self.tri = np.full(n, -1, dtype=np.int64)
        self.u = np.zeros(n, dtype=np.float64)
        self.v = np.zeros(n, dtype=np.float64)
        self.front = np.zeros(n, dtype=bool)

    @property
    def hit(self):
        return self.tri >= 0

    def bary(self):
        return np.stack([1 - self.u - self.v, self.u, self.v], axis=1)


class BVH:
    """Triangle BVH. ``alpha_test(tri_ids, u, v) -> keep_mask`` may reject hits (cut-out textures)."""

    def __init__(self, positions, alpha_tris=None, alpha_test=None):
        P = np.asarray(positions, dtype=np.float64).reshape(-1, 3, 3)
        self.count = len(P)
        order, nodes = build(P)
        self.order = order
        Q = P[order].astype(np.float32)
        self.v0 = [np.ascontiguousarray(Q[:, 0, k]) for k in range(3)]
        self.e1 = [np.ascontiguousarray(Q[:, 1, k] - Q[:, 0, k]) for k in range(3)]
        self.e2 = [np.ascontiguousarray(Q[:, 2, k] - Q[:, 0, k]) for k in range(3)]
        self.lo = [np.ascontiguousarray(nodes["lo"][:, k].astype(np.float32)) for k in range(3)]
        self.hi = [np.ascontiguousarray(nodes["hi"][:, k].astype(np.float32)) for k in range(3)]
        self.left = nodes["left"]
        self.axis = nodes["axis"].astype(np.int64)
        self.start = nodes["start"]
        self.cnt = nodes["count"]
        self.node_count = len(self.left)
        self.alpha_slots = None
        self.alpha_test = alpha_test
        if alpha_tris is not None and alpha_test is not None:
            flag = np.zeros(self.count, dtype=bool)
            flag[np.asarray(alpha_tris, dtype=np.int64)] = True
            self.alpha_slots = flag[order]
            if not self.alpha_slots.any():
                self.alpha_slots = None
        lo, hi = P.reshape(-1, 3).min(axis=0), P.reshape(-1, 3).max(axis=0)
        self.scene_scale = float(np.linalg.norm(hi - lo)) if self.count else 1.0

    # ------------------------------------------------------------------
    def intersect(self, orig, dirs, tmin=None, tmax=None, any_hit=False):
        """Closest hit (or any hit) along each ray. Directions need not be normalised; t is in units of |dir|."""
        orig = np.asarray(orig, dtype=np.float64)
        dirs = np.asarray(dirs, dtype=np.float64)
        n = len(orig)
        out = Hits(n)
        if n == 0 or self.count == 0:
            return out
        tmin = np.zeros(n) if tmin is None else np.broadcast_to(np.asarray(tmin, dtype=np.float64), (n,))
        tmax = np.full(n, np.inf) if tmax is None else np.broadcast_to(np.asarray(tmax, dtype=np.float64), (n,))
        for s in range(0, n, _CHUNK):
            e = min(n, s + _CHUNK)
            self._intersect_chunk(orig[s:e], dirs[s:e], tmin[s:e], tmax[s:e], any_hit, out, s)
        return out

    def occluded(self, orig, dirs, tmax, tmin=None):
        return self.intersect(orig, dirs, tmin=tmin, tmax=tmax, any_hit=True).hit

    def _intersect_chunk(self, orig, dirs, tmin, tmax, any_hit, out, base):
        n = len(orig)
        o = [orig[:, k].astype(np.float32) for k in range(3)]
        d = [dirs[:, k].astype(np.float32) for k in range(3)]
        with np.errstate(divide="ignore"):
            inv = [np.where(np.abs(dk) > 1e-30, 1.0 / dk, np.copysign(np.float32(1e30), dk)).astype(np.float32) for dk in d]
        dneg = np.stack([dk < 0 for dk in d], axis=1)
        tlo = tmin.astype(np.float32)
        tbest = np.where(np.isfinite(tmax), tmax, np.float32(3e38)).astype(np.float32)
        hit_slot = np.full(n, -1, dtype=np.int64)
        hu = np.zeros(n, dtype=np.float32)
        hv = np.zeros(n, dtype=np.float32)
        hfront = np.zeros(n, dtype=bool)
        stack = np.empty((n, _STACK), dtype=np.int32)
        stack[:, 0] = 0
        sp = np.ones(n, dtype=np.int32)
        active = np.arange(n)
        lo, hi = self.lo, self.hi
        v0, e1, e2 = self.v0, self.e1, self.e2
        while active.size:
            r = active
            s = sp[r] - 1
            sp[r] = s
            nd = stack[r, s]
            # slab test against the popped node
            t0x = (lo[0][nd] - o[0][r]) * inv[0][r]
            t1x = (hi[0][nd] - o[0][r]) * inv[0][r]
            t0y = (lo[1][nd] - o[1][r]) * inv[1][r]
            t1y = (hi[1][nd] - o[1][r]) * inv[1][r]
            t0z = (lo[2][nd] - o[2][r]) * inv[2][r]
            t1z = (hi[2][nd] - o[2][r]) * inv[2][r]
            tn = np.maximum(np.maximum(np.minimum(t0x, t1x), np.minimum(t0y, t1y)), np.minimum(t0z, t1z))
            tf = np.minimum(np.minimum(np.maximum(t0x, t1x), np.maximum(t0y, t1y)), np.maximum(t0z, t1z))
            ok = (tn <= tf * np.float32(1.00001) + np.float32(1e-7)) & (tf >= tlo[r]) & (tn <= tbest[r])
            rr = r[ok]
            nd = nd[ok]
            leaf = self.left[nd] < 0
            # inner nodes: push far child first so the near child is popped next
            ri = rr[~leaf]
            if ri.size:
                ni = nd[~leaf]
                lc = self.left[ni]
                neg = dneg[ri, self.axis[ni]]
                s = sp[ri]
                stack[ri, s] = np.where(neg, lc, lc + 1)
                stack[ri, s + 1] = np.where(neg, lc + 1, lc)
                sp[ri] = s + 2
            rl = rr[leaf]
            if rl.size:
                nl = nd[leaf]
                st = self.start[nl]
                ct = self.cnt[nl]
                for k in range(int(ct.max())):
                    m = ct > k
                    ry = rl[m]
                    slot = st[m] + k
                    ox, oy, oz = o[0][ry], o[1][ry], o[2][ry]
                    dx, dy, dz = d[0][ry], d[1][ry], d[2][ry]
                    ax, ay, az = e1[0][slot], e1[1][slot], e1[2][slot]
                    bx, by, bz = e2[0][slot], e2[1][slot], e2[2][slot]
                    px = dy * bz - dz * by
                    py = dz * bx - dx * bz
                    pz = dx * by - dy * bx
                    det = ax * px + ay * py + az * pz
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        idet = np.float32(1.0) / det
                        tx = ox - v0[0][slot]
                        ty = oy - v0[1][slot]
                        tz = oz - v0[2][slot]
                        u = (tx * px + ty * py + tz * pz) * idet
                        qx = ty * az - tz * ay
                        qy = tz * ax - tx * az
                        qz = tx * ay - ty * ax
                        v = (dx * qx + dy * qy + dz * qz) * idet
                        t = (bx * qx + by * qy + bz * qz) * idet
                        eps = np.float32(-1e-6)
                        good = (np.abs(det) > 1e-30) & (u >= eps) & (v >= eps) & (u + v <= 1 - eps) & (t > tlo[ry]) & (t < tbest[ry])
                    if self.alpha_slots is not None and good.any():
                        chk = good & self.alpha_slots[slot]
                        if chk.any():
                            ci = np.nonzero(chk)[0]
                            keep = self.alpha_test(self.order[slot[ci]], u[ci].astype(np.float64), v[ci].astype(np.float64))
                            good[ci[~keep]] = False
                    if not good.any():
                        continue
                    gi = ry[good]
                    tbest[gi] = t[good]
                    hit_slot[gi] = slot[good]
                    hu[gi] = u[good]
                    hv[gi] = v[good]
                    hfront[gi] = det[good] > 0
                    if any_hit:
                        sp[gi] = 0
            active = r[sp[r] > 0]
        sl = slice(base, base + n)
        found = hit_slot >= 0
        out.tri[sl] = np.where(found, self.order[np.maximum(hit_slot, 0)], -1)
        out.t[sl] = np.where(found, tbest, np.inf)
        out.u[sl] = hu
        out.v[sl] = hv
        out.front[sl] = hfront
