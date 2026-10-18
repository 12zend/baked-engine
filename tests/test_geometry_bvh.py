import numpy as np

from baked_engine.core.bvh import BVH
from baked_engine.core.geometry import Plane, TriSet, clip_to_halfspaces, interpolate, mirror, split_by_plane


def brute_force(P, o, d):
    v0, v1, v2 = P[:, 0], P[:, 1], P[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    best_t = np.full(len(o), np.inf)
    best = np.full(len(o), -1)
    for i in range(len(P)):
        p = np.cross(d, e2[i])
        det = p @ e1[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1 / det
            tv = o - v0[i]
            u = np.einsum("ij,ij->i", tv, p) * inv
            q = np.cross(tv, e1[i])
            v = (d * q).sum(1) * inv
            t = (q @ e2[i]) * inv
        ok = (np.abs(det) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t < best_t)
        best_t[ok] = t[ok]
        best[ok] = i
    return best_t, best


def random_soup(rng, n):
    c = rng.uniform(-5, 5, (n, 1, 3))
    return c + rng.normal(0, 0.4, (n, 3, 3))


def test_bvh_matches_brute_force():
    rng = np.random.default_rng(1)
    P = random_soup(rng, 600)
    bvh = BVH(P)
    o = rng.uniform(-6, 6, (3000, 3))
    d = rng.normal(size=(3000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    h = bvh.intersect(o, d)
    t_ref, tri_ref = brute_force(P, o, d)
    hit_ref = tri_ref >= 0
    assert (h.hit == hit_ref).mean() > 0.998
    both = h.hit & hit_ref
    assert np.allclose(h.t[both], t_ref[both], rtol=1e-4, atol=1e-4)
    # any-hit agrees with closest-hit on whether something is there
    occ = bvh.occluded(o, d, tmax=np.full(len(o), np.inf))
    assert (occ == h.hit).mean() > 0.998


def test_bvh_tmax_and_front():
    P = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
    bvh = BVH(P)
    o = np.array([[0.2, 0.2, 1.0], [0.2, 0.2, -1.0], [0.2, 0.2, 1.0]])
    d = np.array([[0, 0, -1.0], [0, 0, 1.0], [0, 0, -1.0]])
    h = bvh.intersect(o, d, tmax=np.array([2.0, 2.0, 0.5]))
    assert h.hit.tolist() == [True, True, False]
    assert h.front.tolist()[:2] == [True, False]
    assert np.allclose(h.bary()[0], [0.6, 0.2, 0.2])


def test_alpha_test_rejects_hits():
    P = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [1, 0, -1], [0, 1, -1]]], dtype=float)
    bvh = BVH(P, alpha_tris=[0], alpha_test=lambda tri, u, v: np.zeros(len(tri), bool))
    h = bvh.intersect(np.array([[0.2, 0.2, 1.0]]), np.array([[0, 0, -1.0]]))
    assert h.tri[0] == 1


def test_split_preserves_area_and_side():
    rng = np.random.default_rng(3)
    ts = TriSet.from_triangles(random_soup(rng, 200))
    pl = Plane.from_point_normal([0.3, -0.2, 0.1], [0.2, 0.5, 1.0])
    f, b = split_by_plane(ts, pl)
    assert np.isclose(f.areas().sum() + b.areas().sum(), ts.areas().sum())
    assert (pl.distance(f.P) > -1e-6).all()
    assert (pl.distance(b.P) < 1e-6).all()
    # barycentric bookkeeping reproduces positions
    src_P = ts.P
    assert np.allclose(interpolate(src_P, f), f.P)
    assert np.allclose(interpolate(src_P, b), b.P)
    # winding preserved
    nf = f.face_normals()
    assert (np.einsum("ij,ij->i", nf, ts.face_normals()[f.src]) > 0.999).all()


def test_clip_convex_and_mirror():
    rng = np.random.default_rng(4)
    ts = TriSet.from_triangles(random_soup(rng, 100))
    planes = [Plane.from_point_normal([0, 0, 0], [0, 0, 1]), Plane.from_point_normal([1, 0, 0], [-1, 0, 0])]
    inside, outside = clip_to_halfspaces(ts, planes)
    assert np.isclose(inside.areas().sum() + outside.areas().sum(), ts.areas().sum())
    m = mirror(inside, planes[0])
    assert (m.P[..., 2] <= 1e-9).all()
    # mirrored winding: normal z flips sign relative to the reflected source normal
    n_src = ts.face_normals()[m.src]
    n_src[:, 2] *= -1
    assert (np.einsum("ij,ij->i", m.face_normals(), n_src) > 0.999).all()
    assert np.allclose(planes[0].reflect_points(interpolate(ts.P, m)), m.P)
