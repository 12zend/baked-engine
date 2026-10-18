"""Render a baked result (triangles + one texture, unlit) from any camera.

This mimics what a consumer engine draws and is used to check that baked
reflections stay correct away from the reference view.
"""

from __future__ import annotations

import numpy as np

from .bvh import BVH
from .geometry import normalize
from .radiance import bilinear


def look_at_rays(position, target, width, height, fov_deg=60.0):
    pos = np.asarray(position, dtype=np.float64)
    fwd = normalize((np.asarray(target, dtype=np.float64) - pos)[None])[0]
    up = np.array([0.0, 0.0, 1.0])
    right = normalize(np.cross(fwd, up)[None])[0]
    up = np.cross(right, fwd)
    t = np.tan(np.radians(fov_deg) / 2)
    xs = ((np.arange(width) + 0.5) / width * 2 - 1) * t
    ys = ((np.arange(height) + 0.5) / height * 2 - 1) * t * height / width
    X, Y = np.meshgrid(xs, ys[::-1])
    d = fwd[None, :] + X.reshape(-1, 1) * right[None, :] + Y.reshape(-1, 1) * up[None, :]
    return np.broadcast_to(pos, d.shape).copy(), normalize(d)


def render(positions, uvs, indices, texture_rgba, position, target, width=320, height=180, fov_deg=60.0, background=(0, 0, 0), backface_cull=True, bvh=None):
    """texture_rgba: (H, W, C) with row 0 = bottom (uv v = 0). Returns (height, width, C) row 0 = top."""
    P = np.asarray(positions, dtype=np.float64)[np.asarray(indices, dtype=np.int64).reshape(-1, 3)]
    UV = np.asarray(uvs, dtype=np.float64)[np.asarray(indices, dtype=np.int64).reshape(-1, 3)]
    bvh = bvh or BVH(P)
    o, d = look_at_rays(position, target, width, height, fov_deg)
    tex = np.asarray(texture_rgba, dtype=np.float64)
    out = np.tile(np.asarray(background, dtype=np.float64)[: tex.shape[2]], (len(o), 1))
    if tex.shape[2] > len(background):
        out = np.concatenate([out, np.ones((len(o), tex.shape[2] - out.shape[1]))], axis=1)
    alive = np.arange(len(o))
    tmin = np.zeros(len(o))
    for _ in range(4):  # step through back faces / alpha cut-outs
        h = bvh.intersect(o[alive], d[alive], tmin=tmin[alive])
        hit = h.hit
        uv = np.einsum("ni,nij->nj", h.bary()[hit], UV[h.tri[hit]])
        px = uv * np.array([tex.shape[1], tex.shape[0]])
        col = bilinear(tex, px)
        ok = np.ones(int(hit.sum()), dtype=bool)
        if backface_cull:
            ok &= h.front[hit]
        if tex.shape[2] == 4:
            ok &= col[:, 3] >= (0.5 * tex[..., 3].max())
        idx = alive[hit]
        out[idx[ok]] = col[ok]
        retry = np.zeros(len(alive), dtype=bool)
        retry[np.nonzero(hit)[0][~ok]] = True
        tmin[alive[retry]] = h.t[retry] + 1e-4
        alive = alive[retry]
        if not len(alive):
            break
    return out.reshape(height, width, -1)


def reference_render(scene, position, target, width=320, height=180, fov_deg=60.0, spp=16, max_specular=2, background=(0, 0, 0), tracer=None):
    """Path-traced view of the *original* scene (diffuse lighting + up to ``max_specular`` sharp
    reflection bounces on smooth surfaces). Used as ground truth for baked reflections."""
    from .integrator import PathTracer
    from .sampling import Sampler
    from .shading import fresnel_schlick

    tracer = tracer or PathTracer(scene, max_bounces=4, clamp=10.0)
    mats = tracer.mats
    o, d = look_at_rays(position, target, width, height, fov_deg)
    n = len(o)
    out = np.zeros((n, 3))
    weight = np.ones((n, 3))
    alive = np.arange(n)
    for bounce in range(max_specular + 1):
        h = tracer.bvh.intersect(o[alive], d[alive])
        miss = ~h.hit
        out[alive[miss]] += weight[alive[miss]] * (tracer.env.eval(d[alive[miss]]) if bounce else np.asarray(background, float))
        hit = np.nonzero(h.hit)[0]
        idx = alive[hit]
        tri = h.tri[hit]
        bary = h.bary()[hit]
        surf = mats.surface(tri, bary)
        wo = -d[idx]
        ng = np.where((np.einsum("ij,ij->i", surf["Ng"], wo) < 0)[:, None], -surf["Ng"], surf["Ng"])
        ns = np.where((np.einsum("ij,ij->i", surf["Ns"], ng) < 0)[:, None], -surf["Ns"], surf["Ns"])
        smooth = (surf["rough"] <= 0.25) & (bounce < max_specular)
        F = np.where(smooth[:, None], fresnel_schlick(surf["f0"], np.clip(np.einsum("ij,ij->i", wo, ns), 0, 1)), 0.0)
        # diffuse radiance at the hit (averaged path-traced lighting)
        sampler = Sampler(len(idx), spp, seed=bounce)
        acc = np.zeros((len(idx), 3))
        for i in range(spp):
            L, _ = tracer.trace_from_surface(surf["P"], ng, ns, sampler, i)
            acc += L
        diffuse = surf["kd"] * acc / spp + mats.emission(tri, bary)
        out[idx] += weight[idx] * (1 - F) * diffuse
        weight[idx] *= F
        refl = wo * -1 + 2 * np.einsum("ij,ij->i", wo, ns)[:, None] * ns
        o[idx] = surf["P"] + ng * tracer.eps * 4
        d[idx] = refl
        alive = idx[smooth & (F.max(axis=1) > 1e-3)]
        if not len(alive):
            break
    return out.reshape(height, width, 3)
