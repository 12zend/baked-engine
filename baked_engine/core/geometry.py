"""Triangle sets that remember where they came from, plus plane clipping and mirroring.

A :class:`TriSet` is a list of triangles where every corner stores both its
world position and its barycentric coordinate inside a *source* triangle of
the original scene. Clipping and mirroring only change positions and the
barycentric bookkeeping, so any attribute of the original scene (normal,
material uv, baked radiance) can be recovered for every output triangle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


@dataclass
class TriSet:
    P: np.ndarray  # (n, 3, 3) positions
    B: np.ndarray  # (n, 3, 3) barycentrics of each corner in the source triangle
    src: np.ndarray  # (n,) source triangle id
    tags: dict = field(default_factory=dict)  # name -> (n,) per-triangle arrays

    def __post_init__(self):
        self.P = np.asarray(self.P, dtype=np.float64).reshape(-1, 3, 3)
        self.B = np.asarray(self.B, dtype=np.float64).reshape(-1, 3, 3)
        self.src = np.asarray(self.src, dtype=np.int64).reshape(-1)

    def __len__(self):
        return len(self.src)

    @staticmethod
    def from_triangles(P, **tags):
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3, 3)
        n = len(P)
        B = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        return TriSet(P, B, np.arange(n), {k: np.asarray(v) for k, v in tags.items()})

    @staticmethod
    def empty(tag_names=(), like=None):
        tags = {}
        for k in tag_names:
            dt = like.tags[k].dtype if like is not None and k in like.tags else np.int64
            tags[k] = np.zeros(0, dtype=dt)
        return TriSet(np.zeros((0, 3, 3)), np.zeros((0, 3, 3)), np.zeros(0, np.int64), tags)

    def subset(self, mask):
        return TriSet(self.P[mask], self.B[mask], self.src[mask], {k: v[mask] for k, v in self.tags.items()})

    def with_tag(self, name, value):
        t = dict(self.tags)
        t[name] = np.broadcast_to(np.asarray(value), (len(self),)).copy()
        return TriSet(self.P, self.B, self.src, t)

    @staticmethod
    def concat(sets):
        sets = [s for s in sets if s is not None]
        if not sets:
            return TriSet.empty()
        names = set(sets[0].tags)
        for s in sets[1:]:
            names &= set(s.tags)
        return TriSet(
            np.concatenate([s.P for s in sets]),
            np.concatenate([s.B for s in sets]),
            np.concatenate([s.src for s in sets]),
            {k: np.concatenate([s.tags[k] for s in sets]) for k in names},
        )

    def face_normals(self):
        n = np.cross(self.P[:, 1] - self.P[:, 0], self.P[:, 2] - self.P[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)

    def areas(self):
        return 0.5 * np.linalg.norm(np.cross(self.P[:, 1] - self.P[:, 0], self.P[:, 2] - self.P[:, 0]), axis=1)

    def drop_degenerate(self, min_area=1e-14):
        return self.subset(self.areas() > min_area)


@dataclass(frozen=True)
class Plane:
    """Points x with ``dot(n, x) == d``; the *front* side is ``dot(n, x) > d``."""

    n: tuple
    d: float

    @staticmethod
    def from_point_normal(p, n):
        n = np.asarray(n, dtype=np.float64)
        n = n / np.linalg.norm(n)
        return Plane(tuple(n), float(np.dot(n, p)))

    @property
    def normal(self):
        return np.asarray(self.n, dtype=np.float64)

    def distance(self, x):
        return np.asarray(x) @ self.normal - self.d

    def flipped(self):
        return Plane(tuple(-self.normal), -self.d)

    def reflect_points(self, x):
        x = np.asarray(x, dtype=np.float64)
        n = self.normal
        return x - 2.0 * (x @ n - self.d)[..., None] * n

    def reflect_dirs(self, v):
        v = np.asarray(v, dtype=np.float64)
        n = self.normal
        return v - 2.0 * (v @ n)[..., None] * n

    def reflect_plane(self, other: "Plane") -> "Plane":
        """Image of ``other`` under reflection by this plane."""
        p0 = other.normal * other.d
        return Plane.from_point_normal(self.reflect_points(p0), self.reflect_dirs(other.normal))


def split_by_plane(ts: TriSet, plane: Plane, eps=1e-7):
    """Split triangles into (front, back) parts. Vertices within ``eps`` of the plane snap to it."""
    if len(ts) == 0:
        return ts, ts
    dist = plane.distance(ts.P)  # (n, 3)
    dist = np.where(np.abs(dist) < eps, 0.0, dist)
    pos = dist > 0
    neg = dist < 0
    all_front = ~neg.any(axis=1)  # every vertex >= 0 (triangles lying in the plane count as front)
    all_back = ~pos.any(axis=1) & ~all_front
    front_parts = [ts.subset(all_front)]
    back_parts = [ts.subset(all_back)]
    mixed = ~(all_front | all_back)
    if mixed.any():
        m = ts.subset(mixed)
        dm = dist[mixed]
        f, b = _split_mixed(m, dm)
        front_parts.append(f)
        back_parts.append(b)
    return TriSet.concat(front_parts).drop_degenerate(), TriSet.concat(back_parts).drop_degenerate()


def _split_mixed(ts: TriSet, dist):
    """Triangles with at least one vertex strictly on each side."""
    n = len(ts)
    pos = dist > 0
    neg = dist < 0
    npos = pos.sum(axis=1)
    # The lone vertex is the only one on its side (ties with zeros handled below).
    lone_is_pos = npos == 1
    lone_side = np.where(lone_is_pos[:, None], pos, neg)
    lone = np.argmax(lone_side, axis=1)
    idx = np.stack([lone, (lone + 1) % 3, (lone + 2) % 3], axis=1)
    rows = np.arange(n)[:, None]
    P = ts.P[rows, idx]
    B = ts.B[rows, idx]
    d = dist[rows, idx]
    # Edge intersections lone->1 and lone->2 (other vertices may be 0 -> t == 1).
    t1 = (d[:, 0] / (d[:, 0] - d[:, 1]))[:, None]
    t2 = (d[:, 0] / (d[:, 0] - d[:, 2]))[:, None]
    P1 = P[:, 0] + (P[:, 1] - P[:, 0]) * t1
    P2 = P[:, 0] + (P[:, 2] - P[:, 0]) * t2
    B1 = B[:, 0] + (B[:, 1] - B[:, 0]) * t1
    B2 = B[:, 0] + (B[:, 2] - B[:, 0]) * t2
    tri_lone_P = np.stack([P[:, 0], P1, P2], axis=1)
    tri_lone_B = np.stack([B[:, 0], B1, B2], axis=1)
    quad_a_P = np.stack([P1, P[:, 1], P[:, 2]], axis=1)
    quad_a_B = np.stack([B1, B[:, 1], B[:, 2]], axis=1)
    quad_b_P = np.stack([P1, P[:, 2], P2], axis=1)
    quad_b_B = np.stack([B1, B[:, 2], B2], axis=1)
    lone_set = TriSet(tri_lone_P, tri_lone_B, ts.src, dict(ts.tags))
    quad_set = TriSet(
        np.concatenate([quad_a_P, quad_b_P]),
        np.concatenate([quad_a_B, quad_b_B]),
        np.concatenate([ts.src, ts.src]),
        {k: np.concatenate([v, v]) for k, v in ts.tags.items()},
    )
    lp = lone_is_pos
    qp = np.concatenate([~lp, ~lp])
    front = TriSet.concat([lone_set.subset(lp), quad_set.subset(qp)])
    back = TriSet.concat([lone_set.subset(~lp), quad_set.subset(~qp)])
    return front, back


def clip_to_halfspaces(ts: TriSet, planes):
    """Return (inside, outside) where inside is the part on the front of every plane."""
    inside = ts
    outside = []
    for pl in planes:
        f, b = split_by_plane(inside, pl)
        outside.append(b)
        inside = f
        if len(inside) == 0:
            break
    return inside, TriSet.concat(outside) if outside else TriSet.empty(ts.tags.keys(), ts)


def mirror(ts: TriSet, plane: Plane) -> TriSet:
    """Reflect positions through ``plane`` and swap two corners to keep the winding front-facing."""
    P = plane.reflect_points(ts.P)
    P = P[:, [0, 2, 1]]
    B = ts.B[:, [0, 2, 1]]
    return TriSet(P, B, ts.src.copy(), {k: v.copy() for k, v in ts.tags.items()})


def interpolate(attr, ts: TriSet):
    """Evaluate a per-corner attribute ``attr`` (T, 3, k) of the source triangles at every TriSet corner."""
    a = attr[ts.src]  # (n, 3, k)
    return np.einsum("ncj,njk->nck", ts.B, a)


def convex_hull_2d(pts):
    """Monotone chain hull; returns hull vertices counter-clockwise."""
    pts = np.unique(np.round(np.asarray(pts, dtype=np.float64), 12), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.asarray(lower[:-1] + upper[:-1])


def plane_basis(n):
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def orthonormal_frames(n):
    """Vectorised branchless frame (Duff et al. 2017) for unit normals (N, 3)."""
    sign = np.where(n[:, 2] >= 0, 1.0, -1.0)
    a = -1.0 / (sign + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    t = np.stack([1 + sign * n[:, 0] ** 2 * a, sign * b, -sign * n[:, 0]], axis=1)
    bt = np.stack([b, sign + n[:, 1] ** 2 * a, -n[:, 1]], axis=1)
    return t, bt


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)
