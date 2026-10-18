import numpy as np
import pytest

from baked_engine.core.integrator import PathTracer
from baked_engine.core.scene import BakeScene, Environment, Light, Material


def quad(z=0.0, s=1.0, flip=False):
    a, b, c, d = [-s, -s, z], [s, -s, z], [s, s, z], [-s, s, z]
    t = [[a, b, c], [a, c, d]]
    if flip:
        t = [[a, c, b], [a, d, c]]
    return np.array(t, float)


def box_inside(s=1.0):
    """Closed cube with faces pointing inwards."""
    v = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in quads:
        tris += [(a, c, b), (a, d, c)]  # inward winding
    return v[np.array(tris)]


def scene_of(P, **kw):
    n = len(P)
    return BakeScene(P, None, None, np.zeros(n, int), np.zeros(n, int), **kw)


def bake_center(sc, spp=256, n=64, **kw):
    pt = PathTracer(sc, clamp=0, **kw)
    P = np.tile([0.0, 0.0, 0.0], (n, 1)) + np.c_[np.random.default_rng(0).uniform(-0.3, 0.3, (n, 2)), np.zeros(n)]
    N = np.tile([0.0, 0.0, 1.0], (n, 1))
    mean, var, valid = pt.bake(n, spp, lambda s, i: (P, N, N))
    return mean, var, valid


def test_constant_environment_furnace():
    sc = scene_of(quad(), environment=Environment(color=(0.3, 0.5, 0.7), strength=1.0))
    mean, _, valid = bake_center(sc)
    assert valid.all()
    assert np.allclose(mean.mean(0), [0.3, 0.5, 0.7], rtol=0.03)


def test_hdri_importance_sampling_matches_constant():
    tex = np.ones((32, 64, 3)) * 0.5
    tex[20:24, 10:14] = 40.0  # a bright "sun" blob in the upper hemisphere
    env = Environment(texture=tex)
    sc = scene_of(quad(), environment=env)
    mean, _, _ = bake_center(sc, spp=512)
    # reference: brute-force integral of the map over the upper hemisphere
    h, w = tex.shape[:2]
    v = (np.arange(h) + 0.5) / h
    u = (np.arange(w) + 0.5) / w
    uu, vv = np.meshgrid(u, v)
    d = env._uv_to_dir(uu.ravel(), vv.ravel())
    lat = (vv.ravel() - 0.5) * np.pi
    dA = np.cos(lat) * (2 * np.pi / w) * (np.pi / h)
    cosz = np.maximum(d[:, 2], 0)
    ref = (tex.reshape(-1, 3) * (cosz * dA)[:, None]).sum(0) / np.pi
    assert np.allclose(mean.mean(0), ref, rtol=0.05)


def test_sun_and_point_light_units():
    sun = Light("SUN", power=3.0, direction=(0, 0, -1), angle=0.01)
    sc = scene_of(quad(), lights=[sun], environment=Environment(color=(0, 0, 0)))
    mean, _, _ = bake_center(sc, spp=8)
    assert mean.mean() == pytest.approx(3.0 / np.pi, rel=1e-3)
    pl = Light("POINT", power=100.0, position=(0, 0, 2.0), radius=0.0)
    sc = scene_of(quad(), lights=[pl], environment=Environment(color=(0, 0, 0)))
    pt = PathTracer(sc, clamp=0)
    P = np.array([[0.0, 0, 0]])
    N = np.array([[0.0, 0, 1]])
    mean, _, _ = pt.bake(1, 4, lambda s, i: (P, N, N))
    assert mean[0, 0] == pytest.approx(100 / (4 * np.pi * 4) / np.pi, rel=1e-6)


def test_area_light_small_and_far():
    al = Light("AREA", power=50.0, position=(0, 0, 3.0), direction=(0, 0, -1), size=(0.05, 0.05))
    sc = scene_of(quad(), lights=[al], environment=Environment(color=(0, 0, 0)))
    pt = PathTracer(sc, clamp=0)
    P = np.array([[0.0, 0, 0]])
    N = np.array([[0.0, 0, 1]])
    mean, _, _ = pt.bake(1, 16, lambda s, i: (P, N, N))
    assert mean[0, 0] == pytest.approx(50 / (np.pi * 9) / np.pi, rel=1e-3)


def test_occlusion_by_roof():
    sun = Light("SUN", power=1.0, direction=(0, 0, -1), angle=0.0)
    P = np.concatenate([quad(), quad(1.0, 0.5, flip=True)])
    sc = scene_of(P, lights=[sun], environment=Environment(color=(0, 0, 0)))
    pt = PathTracer(sc, clamp=0, max_bounces=0)
    pts = np.array([[0.0, 0, 0], [0.9, 0.9, 0]])
    N = np.array([[0.0, 0, 1]] * 2)
    mean, _, _ = pt.bake(2, 4, lambda s, i: (pts, N, N))
    assert mean[0, 0] == 0
    assert mean[1, 0] == pytest.approx(1 / np.pi)


def test_closed_emissive_box_interreflection():
    P = box_inside()
    mat = Material(base_color=(0.5, 0.5, 0.5), roughness=1.0, specular=0.0, emission=(1, 1, 1), emission_strength=1.0)
    sc = scene_of(P, materials=[mat], environment=Environment(color=(0, 0, 0)))
    pt = PathTracer(sc, clamp=0, max_bounces=64)
    n = 256
    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(-0.8, 0.8, (n, 2)), np.full(n, -1.0)]
    N = np.tile([0.0, 0, 1], (n, 1))
    mean, _, valid = pt.bake(n, 64, lambda s, i: (pts, N, N))
    assert valid.all()
    # L_in = Le / (1 - rho) = 2 everywhere inside
    assert mean.mean() == pytest.approx(2.0, rel=0.03)


def test_leak_detection_flags_points_inside_geometry():
    P = box_inside(0.5)
    P = P[:, :, [0, 1, 2]][:, :, :] * 1.0
    # turn the box outward so a point inside sees back faces
    P = P[:, [0, 2, 1]]
    sc = scene_of(P, environment=Environment(color=(1, 1, 1)))
    pt = PathTracer(sc, clamp=0)
    pts = np.array([[0.0, 0.0, 0.0]])
    N = np.array([[0.0, 0.0, 1.0]])
    _, _, valid = pt.bake(1, 16, lambda s, i: (pts, N, N), leak_dist=np.array([2.0]))
    assert valid[0] == 0
