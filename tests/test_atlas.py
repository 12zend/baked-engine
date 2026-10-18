import numpy as np
import pytest

from baked_engine.core.atlas import Charts, build_charts, connected_components, pack_atlas
from baked_engine.core.raster import rasterize


def cube(center=(0, 0, 0), size=1.0):
    c = np.asarray(center, float)
    s = size / 2
    v = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)]) + c
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, cc, d in quads:
        tris += [(a, b, cc), (a, cc, d)]
    return v[np.array(tris)]


def grid(n=10, size=10.0, z=0.0):
    xs = np.linspace(0, size, n + 1)
    x, y = np.meshgrid(xs, xs)
    V = np.stack([x, y, np.full_like(x, z)], -1)
    a = V[:-1, :-1].reshape(-1, 3)
    b = V[:-1, 1:].reshape(-1, 3)
    c = V[1:, 1:].reshape(-1, 3)
    d = V[1:, :-1].reshape(-1, 3)
    return np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


def test_connected_components():
    lab = connected_components(6, [0, 1, 3], [1, 2, 4])
    assert lab[0] == lab[1] == lab[2]
    assert lab[3] == lab[4] != lab[0]
    assert len(set(lab.tolist())) == 3


def test_cube_gives_six_charts_and_plane_one():
    P = np.concatenate([cube(), grid() + [5, 5, -3]])
    group = np.zeros(len(P), int)
    ch = build_charts(P, group, np.ones(len(P)))
    assert ch.count == 7
    # flattening preserves area for planar charts
    loc = ch.local
    a2 = np.abs((loc[:, 1, 0] - loc[:, 0, 0]) * (loc[:, 2, 1] - loc[:, 0, 1]) - (loc[:, 1, 1] - loc[:, 0, 1]) * (loc[:, 2, 0] - loc[:, 0, 0])) / 2
    a3 = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    assert np.allclose(a2, a3)
    assert (ch.local.min(axis=(0, 1)) >= -1e-9).all()


def test_pack_fits_and_does_not_overlap():
    rng = np.random.default_rng(0)
    ext = rng.uniform(0.2, 3.0, (300, 2))
    ch = Charts(np.zeros(0, int), np.zeros((0, 3, 2)), ext, np.zeros(300, bool))
    lay = pack_atlas(ch, 512, 512, padding=4)
    r = lay.chart_rect
    assert (r[:, 0] >= 0).all() and (r[:, 1] >= 0).all()
    assert (r[:, 0] + r[:, 2] <= 512).all() and (r[:, 1] + r[:, 3] <= 512).all()
    occ = np.zeros((512, 512), int)
    for x, y, w, h in r:
        occ[y : y + h, x : x + w] += 1
    assert occ.max() == 1
    assert lay.fill_ratio > 0.6
    # density decreases when there is more to fit
    lay2 = pack_atlas(Charts(np.zeros(0, int), np.zeros((0, 3, 2)), ext * 2, np.zeros(300, bool)), 512, 512, padding=4)
    assert lay2.scale < lay.scale
    assert lay2.scale == pytest.approx(lay.scale / 2, rel=0.15)


def test_target_density_is_kept_when_it_fits_and_lowered_otherwise():
    ext = np.array([[2.0, 1.0]])
    ch = Charts(np.zeros(0, int), np.zeros((0, 3, 2)), ext, np.zeros(1, bool))
    assert pack_atlas(ch, 256, 256, target_density=10).scale == 10
    lowered = pack_atlas(ch, 256, 256, target_density=1000).scale
    assert 100 < lowered < 130


def test_rasterize_covers_triangles_once():
    P = grid(4, 4.0)
    ch = build_charts(P, np.zeros(len(P), int), np.ones(len(P)))
    lay = pack_atlas(ch, 64, 64, padding=2)
    tx = rasterize(lay.tri_px, 64, 64)
    inside = tx.inside.sum()
    covered = (lay.scale * 4) ** 2
    assert abs(inside - covered) <= 4 * lay.scale * 4 + 4
    keys = tx.y * 64 + tx.x
    assert len(np.unique(keys)) == len(keys)
    assert np.allclose(tx.bary.sum(1), 1) and (tx.bary >= 0).all()
    # jittered barycentrics move consistently with pixel offsets
    jb = tx.jittered_bary(np.zeros((len(tx), 2)))
    assert np.allclose(jb, tx.bary)
