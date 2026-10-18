import numpy as np

from baked_engine.core import demo
from baked_engine.core.bvh import BVH
from baked_engine.core.imageops import encode
from baked_engine.core.pipeline import bake
from baked_engine.core.preview import reference_render, render
from baked_engine.core.reflections import KIND_COPY, KIND_DOME, KIND_MAIN, build_output_geometry, plan_reflections
from baked_engine.core.scene import BakeScene, Environment, Light, Material, ReflectorConfig, SceneObject
from baked_engine.core.settings import BakeSettings
from baked_engine.core.shading import MaterialTable


def _plan(sc, **kw):
    st = BakeSettings(**kw)
    mats = MaterialTable(sc)
    bvh = BVH(sc.positions)
    plan = plan_reflections(sc, st, mats, bvh)
    return st, plan, build_output_geometry(sc, plan, st)


def test_sea_becomes_window_with_mirrored_world():
    sc = demo.mountain_lake(resolution=24)
    st, plan, geo = _plan(sc)
    assert len(plan.reflectors) == 1
    r = plan.reflectors[0]
    assert np.allclose(r.plane.normal, [0, 0, 1]) and abs(r.plane.d) < 1e-9
    assert r.remove
    kind = geo.main.tags["kind"]
    P = geo.main.P
    # the sea surface itself is gone from the output, it lives in the aux set
    sea_tris = np.nonzero(sc.object_ids == 1)[0]
    assert not np.isin(geo.main.src[kind == KIND_MAIN], sea_tris).any()
    # nothing of the original scene remains below the water inside the reflection volume
    assert P[kind == KIND_MAIN][..., 2].min() > -1e-6
    # mirrored copies are exactly the reflection of what is above the water
    copies = geo.main.subset(kind == KIND_COPY)
    assert copies.P[..., 2].max() < 1e-6
    from baked_engine.core.geometry import interpolate

    src_pos = interpolate(sc.positions, copies)
    assert np.allclose(src_pos * [1, 1, -1], copies.P, atol=1e-6)
    assert (kind == KIND_DOME).sum() > 0


def test_mirrored_faces_wind_like_a_real_reflection():
    sc = demo.mountain_lake(resolution=12)
    st, plan, geo = _plan(sc)
    kind = geo.main.tags["kind"]
    copies = geo.main.subset(kind == KIND_COPY)
    src_n = sc.face_normals()[copies.src]
    expected = src_n * [1, 1, -1]
    assert (np.einsum("ij,ij->i", copies.face_normals(), expected) > 0.99).all()


def _corridor():
    """Two parallel mirrors facing each other with a coloured pillar between them."""
    def quad(c, u, v, n):
        c, u, v = map(np.asarray, (c, u, v))
        a, b, cc, d = c - u - v, c + u - v, c + u + v, c - u + v
        t = np.array([[a, b, cc], [a, cc, d]], float)
        fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        if (fn @ n < 0).any():
            t = t[:, [0, 2, 1]]
        return t

    floor = quad([0, 0, 0], [3, 0, 0], [0, 6, 0], [0, 0, 1])
    m1 = quad([-3, 0, 1.5], [0, 2, 0], [0, 0, 1.4], [1, 0, 0])
    m2 = quad([3, 0, 1.5], [0, 2, 0], [0, 0, 1.4], [-1, 0, 0])
    box = demo._box((0, 0, 0.5), (0.5, 0.5, 1.0))[0]
    P = np.concatenate([floor, m1, m2, box])
    obj = np.array([0, 0, 1, 1, 2, 2] + [3] * len(box))
    mat = np.array([0, 0, 1, 1, 1, 1] + [2] * len(box))
    mats = [Material("floor", (0.5, 0.5, 0.5)), Material("mirror", (0.9, 0.9, 0.9), metallic=1.0, roughness=0.0), Material("red", (0.8, 0.1, 0.1))]
    objs = [SceneObject("floor", ReflectorConfig(mode="OFF")), SceneObject("m1", ReflectorConfig(mode="GEOMETRY", remove_below="ON", mirror_depth=20.0)),
            SceneObject("m2", ReflectorConfig(mode="GEOMETRY", remove_below="ON", mirror_depth=20.0)), SceneObject("box")]
    return BakeScene(P, None, None, mat, obj, mats, [], [Light("SUN", power=3.0, direction=(0.2, 0.3, -1))], Environment(color=(0.3, 0.3, 0.3)), None, objs)


def test_mirror_in_mirror_recursion():
    sc = _corridor()
    st, plan, geo1 = _plan(sc, reflection_depth=1)
    assert len(plan.reflectors) == 2
    _, _, geo2 = _plan(sc, reflection_depth=2)
    chains1 = {c for c in geo1.chains.chains}
    chains2 = {c for c in geo2.chains.chains}
    assert max(len(c) for c in chains1) == 1
    assert (0, 1) in chains2 and (1, 0) in chains2
    n1 = (geo1.main.tags["kind"] == KIND_COPY).sum()
    n2 = (geo2.main.tags["kind"] == KIND_COPY).sum()
    assert n2 > n1
    # second-order copy of the pillar: mirrored by x=+3 then x=-3 -> translated by -12 in x
    k = geo2.main.tags["kind"]
    ch = geo2.main.tags["chain"]
    box_tris = np.nonzero(sc.object_ids == 3)[0]
    sel = (k == KIND_COPY) & np.isin(geo2.main.src, box_tris) & (ch == geo2.chains.get((0, 1)))
    assert sel.any()
    xs = geo2.main.P[sel][..., 0]
    assert np.allclose(xs.mean(), -12.0, atol=0.3)


def test_baked_reflection_matches_reference_from_new_viewpoints():
    sc = demo.mountain_lake(resolution=24)
    res = bake(sc, BakeSettings(width=256, height=256, samples=8, reflection_samples=8))
    W, H = 96, 54
    errs = []
    for p, tg in [((0, -58, 7), (0, 40, 12)), ((35, -20, 2.0), (-10, 50, 8))]:
        b = encode(render(res.positions, res.uvs, res.indices, res.image, p, tg, W, H, 70)[..., :3])
        r = encode(reference_render(sc, p, tg, W, H, 70, spp=8))
        # rows that show water (below the horizon) - this is where the reflection lives
        errs.append(np.abs(b - r)[H // 2 + 4 :].mean())
    assert max(errs) < 0.06, errs


def test_texture_mode_for_small_glossy_objects():
    sc = demo.mountain_lake(resolution=12)
    sc.objects[2].reflector = ReflectorConfig(mode="TEXTURE")
    sc.materials[2].metallic = 1.0
    st, plan, geo = _plan(sc)
    house = np.nonzero(sc.object_ids == 2)[0]
    assert plan.texture_spec[house].all()
