"""Procedural test scenes (no Blender needed)."""

from __future__ import annotations

import numpy as np

from .scene import BakeScene, Camera, Environment, Light, Material, ReflectorConfig, SceneObject


def _grid(n, size, height_fn, center=(0.0, 0.0)):
    xs = np.linspace(-size / 2, size / 2, n + 1) + center[0]
    ys = np.linspace(-size / 2, size / 2, n + 1) + center[1]
    x, y = np.meshgrid(xs, ys)
    z = height_fn(x, y)
    V = np.stack([x, y, z], -1)
    a = V[:-1, :-1].reshape(-1, 3)
    b = V[:-1, 1:].reshape(-1, 3)
    c = V[1:, 1:].reshape(-1, 3)
    d = V[1:, :-1].reshape(-1, 3)
    P = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    # smooth vertex normals from the height field gradient
    eps = size / n * 0.5
    def nrm(p):
        dzdx = (height_fn(p[..., 0] + eps, p[..., 1]) - height_fn(p[..., 0] - eps, p[..., 1])) / (2 * eps)
        dzdy = (height_fn(p[..., 0], p[..., 1] + eps) - height_fn(p[..., 0], p[..., 1] - eps)) / (2 * eps)
        nn = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], -1)
        return nn / np.linalg.norm(nn, axis=-1, keepdims=True)
    return P, nrm(P)


def _box(center, size):
    c = np.asarray(center, float)
    s = np.asarray(size, float) / 2
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s + c
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, cc, d in quads:
        tris += [(a, b, cc), (a, cc, d)]
    P = v[np.array(tris)]
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    # make faces point outwards
    out = np.einsum("ij,ij->i", fn, P.mean(1) - c) < 0
    P[out] = P[out][:, [0, 2, 1]]
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    return P, np.repeat(fn[:, None], 3, axis=1)


def sky_texture(h=64, w=128, sun_dir=(0.3, 0.6, 0.45)):
    v = (np.arange(h) + 0.5) / h
    lat = (v - 0.5) * np.pi
    t = np.clip(np.sin(lat), 0, 1)[:, None, None]
    horizon = np.array([0.75, 0.85, 1.0])
    zenith = np.array([0.18, 0.35, 0.85])
    ground = np.array([0.12, 0.11, 0.1])
    sky = horizon * (1 - t) ** 3 + zenith * (1 - (1 - t) ** 3)
    img = np.where((lat < 0)[:, None, None], ground, sky) * np.ones((h, w, 3))
    return img.astype(np.float32)


def mountain_lake(resolution=48, sea_size=160.0):
    """A mountain behind a calm sea, lit by a low sun: the canonical reflection test."""
    def height(x, y):
        m = 38 * np.exp(-((x - 5) ** 2 + (y - 45) ** 2) / (2 * 18.0**2))
        m += 22 * np.exp(-((x + 30) ** 2 + (y - 38) ** 2) / (2 * 12.0**2))
        ridge = 6 * np.sin(x * 0.11) * np.exp(-((y - 35) ** 2) / 300)
        shore = (y - 10) * 0.35  # the land dips below the sea towards the viewer
        return np.maximum(m + ridge + shore, -6.0) - 1.0

    terrain, tn = _grid(resolution, 120.0, height, center=(0, 25))
    sea = np.array([[[-sea_size / 2, -sea_size / 2 + 20, 0], [sea_size / 2, -sea_size / 2 + 20, 0], [sea_size / 2, sea_size / 2 + 20, 0]],
                    [[-sea_size / 2, -sea_size / 2 + 20, 0], [sea_size / 2, sea_size / 2 + 20, 0], [-sea_size / 2, sea_size / 2 + 20, 0]]])
    sn = np.tile([0.0, 0.0, 1.0], (2, 3, 1))
    house, hn = _box((12, 18, 3.5), (6, 5, 7))
    P = np.concatenate([terrain, sea, house])
    N = np.concatenate([tn, sn, hn])
    mat = np.concatenate([np.zeros(len(terrain), int), np.ones(2, int), np.full(len(house), 2)])
    obj = np.concatenate([np.zeros(len(terrain), int), np.ones(2, int), np.full(len(house), 2)])
    materials = [
        Material("rock", base_color=(0.32, 0.36, 0.22), roughness=0.9),
        Material("sea", base_color=(0.01, 0.03, 0.04), roughness=0.04, specular=0.5),
        Material("house", base_color=(0.75, 0.18, 0.12), roughness=0.6),
    ]
    to_sun = np.array([-0.55, -0.6, 0.5])
    sun = Light("SUN", power=4.0, color=(1.0, 0.9, 0.75), direction=tuple(-to_sun / np.linalg.norm(to_sun)), angle=np.radians(1.0))
    objects = [SceneObject("terrain", ReflectorConfig(mode="OFF")), SceneObject("sea", ReflectorConfig(mode="GEOMETRY")), SceneObject("house")]
    cam = Camera(position=(0.0, -25.0, 4.0), forward=(0.0, 1.0, 0.0))
    return BakeScene(P, N, None, mat, obj, materials, [], [sun], Environment(texture=sky_texture()), cam, objects)


def cornell_box():
    """Closed room with a ceiling light: tests interreflection and colour bleeding."""
    def quad(a, b, c, d):
        return [[a, b, c], [a, c, d]]
    s = 1.0
    walls = {
        "floor": quad([-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s]),
        "ceiling": quad([-s, -s, s], [-s, s, s], [s, s, s], [s, -s, s]),
        "back": quad([-s, s, -s], [s, s, -s], [s, s, s], [-s, s, s]),
        "left": quad([-s, -s, -s], [-s, s, -s], [-s, s, s], [-s, -s, s]),
        "right": quad([s, -s, -s], [s, -s, s], [s, s, s], [s, s, -s]),
        "front": quad([-s, -s, -s], [-s, -s, s], [s, -s, s], [s, -s, -s]),
    }
    P = []
    mat = []
    ids = {"floor": 0, "ceiling": 0, "back": 0, "front": 0, "left": 1, "right": 2}
    for k, q in walls.items():
        P += q
        mat += [ids[k]] * 2
    lamp = quad([-0.25, -0.25, s - 0.01], [-0.25, 0.25, s - 0.01], [0.25, 0.25, s - 0.01], [0.25, -0.25, s - 0.01])
    P += lamp
    mat += [3, 3]
    P = np.array(P, float)
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    walls_n = len(P) - 2
    inward = np.einsum("ij,ij->i", fn[:walls_n], -P[:walls_n].mean(1)) > 0
    P[:walls_n][~inward] = P[:walls_n][~inward][:, [0, 2, 1]]
    if fn[-1, 2] > 0:  # the lamp faces down
        P[-2:] = P[-2:][:, [0, 2, 1]]
    box, _ = _box((0.3, 0.2, -0.6), (0.6, 0.6, 0.8))
    P = np.concatenate([P, box])
    mat += [0] * len(box)
    materials = [
        Material("white", base_color=(0.73, 0.73, 0.73), roughness=1.0, specular=0.0),
        Material("red", base_color=(0.63, 0.06, 0.05), roughness=1.0, specular=0.0),
        Material("green", base_color=(0.14, 0.45, 0.09), roughness=1.0, specular=0.0),
        Material("lamp", base_color=(0.8, 0.8, 0.8), emission=(1.0, 0.85, 0.6), emission_strength=12.0),
    ]
    n = len(P)
    return BakeScene(P, None, None, np.array(mat), np.zeros(n, int), materials, [], [], Environment(color=(0, 0, 0)), Camera((0, -0.9, 0), (0, 1, 0)), [SceneObject("room")])
