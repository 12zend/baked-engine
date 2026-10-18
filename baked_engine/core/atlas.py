"""Single-texture atlas: chart generation, rectangle packing and texel-density fitting.

Charts are flattened in world units so the whole atlas has (nearly) uniform
texel density. ``pack_atlas`` searches for the highest density at which all
charts fit into the requested image; when a requested density does not fit it
is lowered until it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import orthonormal_frames


def _bucket_directions(count):
    axes = [np.eye(3)[i] * s for i in range(3) for s in (1, -1)]
    if count <= 6:
        return np.asarray(axes)
    diag = [np.array([x, y, z]) / np.sqrt(3) for x in (1, -1) for y in (1, -1) for z in (1, -1)]
    dirs = axes + diag
    if count > 14:
        edges = []
        for i in range(3):
            for j in range(i + 1, 3):
                for si in (1, -1):
                    for sj in (1, -1):
                        e = np.zeros(3)
                        e[i] = si
                        e[j] = sj
                        edges.append(e / np.sqrt(2))
        dirs += edges
    return np.asarray(dirs)


def connected_components(n, a, b):
    """Label connected components of an undirected graph with ``n`` nodes and edges (a, b)."""
    parent = np.arange(n, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    while True:
        pa = parent[a]
        pb = parent[b]
        diff = pa != pb
        if not diff.any():
            break
        lo = np.minimum(pa[diff], pb[diff])
        hi = np.maximum(pa[diff], pb[diff])
        np.minimum.at(parent, hi, lo)
        while True:
            pp = parent[parent]
            if np.array_equal(pp, parent):
                break
            parent = pp
        keep = parent[a] != parent[b]
        a = a[keep]
        b = b[keep]
    _, labels = np.unique(parent, return_inverse=True)
    return labels


def weld_vertices(points, keys, eps):
    """Vertex ids for points (m, 3) that coincide within ``eps`` and share ``keys``."""
    q = np.floor(points / eps + 0.5).astype(np.int64)
    k = np.concatenate([np.asarray(keys, dtype=np.int64)[:, None], q], axis=1)
    _, ids = np.unique(k, axis=0, return_inverse=True)
    return ids.reshape(-1)


@dataclass
class Charts:
    chart: np.ndarray  # (n,) chart index per triangle
    local: np.ndarray  # (n, 3, 2) chart-space corner coordinates (world units * weight, or px if fixed)
    extent: np.ndarray  # (C, 2) chart size
    fixed: np.ndarray  # (C,) bool: extent is already in pixels

    @property
    def count(self):
        return len(self.extent)


def build_charts(P, group, weight, buckets=14, weld_eps=None):
    """Group triangles into charts and flatten them.

    P: (n, 3, 3) positions; group: (n,) triangles of different groups never share a chart;
    weight: (n,) texel-density multiplier per triangle (constant within a group).
    """
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    if n == 0:
        return Charts(np.zeros(0, np.int64), np.zeros((0, 3, 2)), np.zeros((0, 2)), np.zeros(0, bool))
    group = np.asarray(group, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.float64)
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area2 = np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(area2, 1e-30)[:, None]
    dirs = _bucket_directions(buckets)
    bucket = np.argmax(fn @ dirs.T, axis=1)
    if weld_eps is None:
        ext = np.ptp(P.reshape(-1, 3), axis=0).max()
        weld_eps = max(ext * 1e-6, 1e-9)
    vid = weld_vertices(P.reshape(-1, 3), np.repeat(group * 64 + bucket, 3), weld_eps).reshape(n, 3)
    # edges keyed by sorted welded vertex pairs
    e = np.stack([vid[:, [0, 1]], vid[:, [1, 2]], vid[:, [2, 0]]], axis=1).reshape(-1, 2)
    e.sort(axis=1)
    tri_of_edge = np.repeat(np.arange(n), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    es = e[order]
    same = (es[1:] == es[:-1]).all(axis=1)
    a = tri_of_edge[order][:-1][same]
    b = tri_of_edge[order][1:][same]
    chart = connected_components(n, a, b)
    C = int(chart.max()) + 1

    # area-weighted average normal per chart
    acc = np.zeros((C, 3))
    np.add.at(acc, chart, fn * area2[:, None])
    norm = np.linalg.norm(acc, axis=1)
    avg = np.where(norm[:, None] > 1e-30, acc / np.maximum(norm, 1e-30)[:, None], dirs[bucket[np.unique(chart, return_index=True)[1]]])
    # fall back to the bucket axis when some triangle would fold over the average plane
    min_dot = np.full(C, np.inf)
    np.minimum.at(min_dot, chart, np.einsum("ij,ij->i", fn, avg[chart]))
    first_tri = np.unique(chart, return_index=True)[1]
    avg = np.where((min_dot < 0.25)[:, None], dirs[bucket[first_tri]], avg)
    t, bt = orthonormal_frames(avg)
    tu = t[chart]
    tv = bt[chart]
    uv = np.stack([np.einsum("nij,nj->ni", P, tu), np.einsum("nij,nj->ni", P, tv)], axis=2)  # (n,3,2)

    # orientation: principal axis of the projected vertices (area weighted) vs. axis aligned
    w = np.repeat(area2, 3) + 1e-30
    pts = uv.reshape(-1, 2)
    cid = np.repeat(chart, 3)
    sw = np.bincount(cid, weights=w, minlength=C)
    mx = np.bincount(cid, weights=w * pts[:, 0], minlength=C) / sw
    my = np.bincount(cid, weights=w * pts[:, 1], minlength=C) / sw
    dx = pts[:, 0] - mx[cid]
    dy = pts[:, 1] - my[cid]
    sxx = np.bincount(cid, weights=w * dx * dx, minlength=C)
    syy = np.bincount(cid, weights=w * dy * dy, minlength=C)
    sxy = np.bincount(cid, weights=w * dx * dy, minlength=C)
    ang = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    best_ext = None
    best_rot = None
    for cand in (np.zeros(C), ang, ang + np.pi / 4):
        c, s = np.cos(cand)[cid], np.sin(cand)[cid]
        rx = c * dx + s * dy
        ry = -s * dx + c * dy
        lo = np.full((C, 2), np.inf)
        hi = np.full((C, 2), -np.inf)
        r = np.stack([rx, ry], axis=1)
        np.minimum.at(lo, cid, r)
        np.maximum.at(hi, cid, r)
        ex = hi - lo
        area = ex[:, 0] * ex[:, 1]
        if best_ext is None:
            best_ext, best_rot, best_lo, best_area = ex, cand.copy(), lo, area
        else:
            better = area < best_area * 0.98
            best_ext = np.where(better[:, None], ex, best_ext)
            best_lo = np.where(better[:, None], lo, best_lo)
            best_rot = np.where(better, cand, best_rot)
            best_area = np.where(better, area, best_area)
    c, s = np.cos(best_rot)[cid], np.sin(best_rot)[cid]
    rx = c * dx + s * dy - best_lo[cid, 0]
    ry = -s * dx + c * dy - best_lo[cid, 1]
    cw = np.zeros(C)
    cw[chart] = weight
    local = np.stack([rx, ry], axis=1).reshape(n, 3, 2) * weight[:, None, None]
    extent = best_ext * cw[:, None]
    return Charts(chart, local, extent, np.zeros(C, bool))


def merge_charts(a: Charts, b: Charts):
    return Charts(
        np.concatenate([a.chart, b.chart + a.count]),
        np.concatenate([a.local, b.local]),
        np.concatenate([a.extent, b.extent]),
        np.concatenate([a.fixed, b.fixed]),
    )


# ---------------------------------------------------------------------------
# Packing


class _Skyline:
    def __init__(self, width, height):
        self.W = width
        self.H = height
        self.x = np.array([0], dtype=np.int64)
        self.y = np.array([0], dtype=np.int64)
        self.w = np.array([width], dtype=np.int64)

    def _range_max_starts(self, rw):
        """For every segment i: (top y over [x_i, x_i + rw), valid)."""
        xs, ys = self.x, self.y
        S = len(xs)
        end = xs + rw
        valid = end <= self.W
        j = np.searchsorted(xs, end, side="left") - 1  # last segment overlapping
        j = np.maximum(j, np.arange(S))
        # sparse table for range max
        table = [ys]
        k = 1
        while (1 << k) <= S:
            prev = table[-1]
            half = 1 << (k - 1)
            nxt = prev.copy()
            nxt[: S - half] = np.maximum(prev[: S - half], prev[half:])
            table.append(nxt)
            k += 1
        length = j - np.arange(S) + 1
        lv = np.floor(np.log2(np.maximum(length, 1))).astype(np.int64)
        tops = np.empty(S, dtype=np.int64)
        for level in np.unique(lv):
            m = lv == level
            i = np.nonzero(m)[0]
            tbl = table[level]
            tops[m] = np.maximum(tbl[i], tbl[j[m] - (1 << level) + 1])
        return tops, valid

    def find(self, rw, rh):
        tops, valid = self._range_max_starts(rw)
        ok = valid & (tops + rh <= self.H)
        if not ok.any():
            return None
        score = np.where(ok, (tops + rh) * (self.W + 1) + self.x, np.iinfo(np.int64).max)
        i = int(np.argmin(score))
        return int(self.x[i]), int(tops[i]), int(tops[i] + rh)

    def place(self, x, top, rw):
        xs, ys, ws = self.x, self.y, self.w
        xe = xs + ws
        left = xe <= x
        right = xs >= x + rw
        # partially covered segments are trimmed
        nx, ny, nw = [], [], []
        for keep_mask in (left,):
            nx.append(xs[keep_mask])
            ny.append(ys[keep_mask])
            nw.append(ws[keep_mask])
        # segment straddling the left edge
        straddle_l = (xs < x) & (xe > x)
        if straddle_l.any():
            i = np.nonzero(straddle_l)[0][0]
            nx.append(np.array([xs[i]]))
            ny.append(np.array([ys[i]]))
            nw.append(np.array([x - xs[i]]))
        nx.append(np.array([x]))
        ny.append(np.array([top]))
        nw.append(np.array([rw]))
        straddle_r = (xs < x + rw) & (xe > x + rw)
        if straddle_r.any():
            i = np.nonzero(straddle_r)[0][-1]
            nx.append(np.array([x + rw]))
            ny.append(np.array([ys[i]]))
            nw.append(np.array([xe[i] - x - rw]))
        nx.append(xs[right])
        ny.append(ys[right])
        nw.append(ws[right])
        x2 = np.concatenate(nx)
        y2 = np.concatenate(ny)
        w2 = np.concatenate(nw)
        o = np.argsort(x2, kind="stable")
        x2, y2, w2 = x2[o], y2[o], w2[o]
        # merge equal heights
        keep = np.ones(len(x2), dtype=bool)
        keep[1:] = y2[1:] != y2[:-1]
        grp = np.cumsum(keep) - 1
        self.x = x2[keep]
        self.y = y2[keep]
        self.w = np.bincount(grp, weights=w2).astype(np.int64)


def _try_pack(sizes, width, height, allow_rotate=True):
    """Pack integer rectangles; returns (positions (C,2), rotated (C,)) or None."""
    C = len(sizes)
    order = np.lexsort((-sizes.min(axis=1), -sizes.max(axis=1)))
    pos = np.zeros((C, 2), dtype=np.int64)
    rot = np.zeros(C, dtype=bool)
    sky = _Skyline(width, height)
    for i in order:
        w, h = int(sizes[i, 0]), int(sizes[i, 1])
        cand = sky.find(w, h)
        if allow_rotate and w != h:
            cand_r = sky.find(h, w)
            if cand_r is not None and (cand is None or (cand_r[2], cand_r[0]) < (cand[2], cand[0])):
                sky.place(cand_r[0], cand_r[2], h)
                pos[i] = cand_r[:2]
                rot[i] = True
                continue
        if cand is None:
            return None
        sky.place(cand[0], cand[2], w)
        pos[i] = cand[:2]
    return pos, rot


@dataclass
class AtlasLayout:
    width: int
    height: int
    scale: float  # texels per chart unit (texels per metre for weight 1)
    margin: int
    tri_px: np.ndarray  # (n, 3, 2) pixel coordinates of triangle corners (texel centres at i + 0.5)
    chart_rect: np.ndarray  # (C, 4) x, y, w, h in pixels
    fill_ratio: float

    def uvs(self):
        return self.tri_px / np.array([self.width, self.height], dtype=np.float64)


def _rect_sizes(charts: Charts, scale, margin):
    ext = np.where(charts.fixed[:, None], charts.extent, charts.extent * scale)
    return (np.ceil(ext).astype(np.int64) + 1 + 2 * margin), ext


def pack_atlas(charts: Charts, width, height, padding=4, target_density=0.0, iterations=18):
    """Find the largest chart scale (<= target_density when given) that fits and lay the charts out."""
    if charts.count == 0:
        return AtlasLayout(width, height, 1.0, 0, np.zeros((0, 3, 2)), np.zeros((0, 4), np.int64), 0.0)
    margins = [max(1, int(np.ceil(padding / 2.0))) + 1, 2, 1]
    margins = sorted(set(m for m in margins if m <= max(1, int(np.ceil(padding / 2.0))) + 1), reverse=True)
    scalable = ~charts.fixed
    for margin in margins:
        area = (charts.extent[scalable, 0] * charts.extent[scalable, 1]).sum()
        fixed_px = ((charts.extent[~scalable] + 1 + 2 * margin) ** 2).sum() if (~scalable).any() else 0.0
        free = max(width * height - fixed_px, 1.0)
        upper = np.sqrt(free / max(area, 1e-30)) if area > 0 else 1.0
        max_side = charts.extent[scalable].max() if scalable.any() else 0.0
        if max_side > 0:
            upper = min(upper, (max(width, height) - 2 * margin - 2) / max_side)
        if target_density > 0:
            upper = min(upper, target_density)
        best = None
        sizes, _ = _rect_sizes(charts, upper, margin)
        res = _try_pack(sizes, width, height)
        if res is not None:
            best = (upper, res)
        else:
            lo, hi = 0.0, upper
            res0 = _try_pack(_rect_sizes(charts, 0.0, margin)[0], width, height)
            if res0 is None:
                continue
            best = (0.0, res0)
            for _ in range(iterations):
                mid = 0.5 * (lo + hi) if lo > 0 else hi * 0.6
                res = _try_pack(_rect_sizes(charts, mid, margin)[0], width, height)
                if res is not None:
                    lo = mid
                    best = (mid, res)
                else:
                    hi = mid
                if hi - lo < hi * 0.004:
                    break
        scale, (pos, rot) = best
        sizes, ext = _rect_sizes(charts, scale, margin)
        rect = np.concatenate([pos, np.where(rot[:, None], sizes[:, ::-1], sizes)], axis=1)
        tri_px = _place_triangles(charts, scale, margin, pos, rot)
        used = (sizes[:, 0] * sizes[:, 1]).sum() / float(width * height)
        return AtlasLayout(width, height, float(scale), margin, tri_px, rect, float(used))
    raise ValueError(
        f"{charts.count} charts cannot fit into {width}x{height} even at minimum size; use a larger texture"
    )


def _place_triangles(charts: Charts, scale, margin, pos, rot):
    c = charts.chart
    s = np.where(charts.fixed, 1.0, scale)[c][:, None]
    loc = charts.local * s[:, None, :]
    r = rot[c]
    loc = np.where(r[:, None, None], loc[:, :, ::-1], loc)
    off = pos[c].astype(np.float64) + margin + 0.5
    return loc + off[:, None, :]
