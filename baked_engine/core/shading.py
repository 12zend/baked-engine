"""Vectorised material evaluation and BSDF (Lambert + GGX) for the path tracer."""

from __future__ import annotations

import numpy as np

from .geometry import normalize, orthonormal_frames
from .scene import BakeScene, luminance

INV_PI = 1.0 / np.pi


class MaterialTable:
    def __init__(self, scene: BakeScene):
        mats = scene.materials
        self.scene = scene
        self.textures = scene.textures
        self.base = np.array([m.base_color for m in mats], dtype=np.float64).reshape(-1, 3)
        self.base_tex = np.array([m.base_color_texture for m in mats], dtype=np.int64)
        self.alpha = np.array([m.alpha for m in mats], dtype=np.float64)
        self.alpha_clip = np.array([m.alpha_clip for m in mats], dtype=np.float64)
        self.rough = np.clip(np.array([m.roughness for m in mats], dtype=np.float64), 0.0, 1.0)
        self.metal = np.clip(np.array([m.metallic for m in mats], dtype=np.float64), 0.0, 1.0)
        self.spec = np.array([m.specular for m in mats], dtype=np.float64)
        self.emis = np.array([np.asarray(m.emission) * m.emission_strength for m in mats], dtype=np.float64).reshape(-1, 3)
        self.emis_tex = np.array([m.emission_texture for m in mats], dtype=np.int64)
        tex_alpha = np.array([t.has_alpha for t in self.textures] + [False], dtype=bool)
        self.cutout = (self.alpha < self.alpha_clip) | ((self.base_tex >= 0) & tex_alpha[self.base_tex])
        tri_mat = scene.material_ids
        self.tri_mat = tri_mat
        self.cutout_tris = np.nonzero(self.cutout[tri_mat])[0]
        self.emissive_mats = (self.emis.max(axis=1) > 0)

    # ------------------------------------------------------------------
    def _sample_tex(self, tex_ids, uv):
        out = np.ones((len(tex_ids), 4), dtype=np.float64)
        for t in np.unique(tex_ids):
            if t < 0:
                continue
            m = tex_ids == t
            out[m] = self.textures[t].sample(uv[m])
        return out

    def uv_at(self, tri, bary):
        return np.einsum("ni,nij->nj", bary, self.scene.uvs[tri])

    def base_color(self, tri, bary, with_alpha=False):
        mat = self.tri_mat[tri]
        col = self.base[mat].copy()
        alpha = self.alpha[mat].copy()
        tex = self.base_tex[mat]
        has = tex >= 0
        if has.any():
            s = self._sample_tex(tex[has], self.uv_at(tri[has], bary[has]))
            col[has] *= s[:, :3]
            alpha[has] *= s[:, 3]
        if with_alpha:
            return col, alpha
        return col

    def emission(self, tri, bary):
        mat = self.tri_mat[tri]
        e = self.emis[mat].copy()
        tex = self.emis_tex[mat]
        has = (tex >= 0) & (e.max(axis=1) > 0)
        if has.any():
            e[has] *= self._sample_tex(tex[has], self.uv_at(tri[has], bary[has]))[:, :3]
        return e

    def alpha_keep(self, tri, u, v):
        """BVH callback: keep hits whose alpha is above the material's clip threshold."""
        bary = np.stack([1 - u - v, u, v], axis=1)
        _, a = self.base_color(tri, bary, with_alpha=True)
        return a >= self.alpha_clip[self.tri_mat[tri]]

    def surface(self, tri, bary):
        """Shading data at hit points. Returns a dict of arrays."""
        sc = self.scene
        P = np.einsum("ni,nij->nj", bary, sc.positions[tri])
        Ns = normalize(np.einsum("ni,nij->nj", bary, sc.normals[tri]))
        p = sc.positions[tri]
        Ng = normalize(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
        mat = self.tri_mat[tri]
        base = self.base_color(tri, bary)
        metal = self.metal[mat]
        rough = self.rough[mat]
        f0 = (0.08 * self.spec[mat])[:, None] * (1 - metal[:, None]) + base * metal[:, None]
        kd = base * (1 - metal[:, None])
        return dict(P=P, Ns=Ns, Ng=Ng, kd=kd, f0=f0, rough=rough, mat=mat)


# ----------------------------------------------------------------------
# GGX helpers (all vectors in world space, unit length)


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def ggx_alpha(rough):
    return np.maximum(rough * rough, 1e-3)


def ggx_D(nh, a):
    a2 = a * a
    d = nh * nh * (a2 - 1) + 1
    return a2 / (np.pi * d * d)


def smith_G1(nv, a):
    a2 = a * a
    nv = np.maximum(nv, 1e-6)
    return 2 * nv / (nv + np.sqrt(a2 + (1 - a2) * nv * nv))


def fresnel_schlick(f0, c):
    c = np.clip(c, 0, 1)[:, None]
    return f0 + (1 - f0) * (1 - c) ** 5


def sample_vndf(wo, n, a, u1, u2):
    """Heitz 2018 visible-normal sampling of an isotropic GGX lobe; returns half vectors."""
    t, b = orthonormal_frames(n)
    vx, vy, vz = _dot(wo, t), _dot(wo, b), _dot(wo, n)
    vhx, vhy, vhz = a * vx, a * vy, vz
    ln = np.sqrt(vhx**2 + vhy**2 + vhz**2)
    vhx, vhy, vhz = vhx / ln, vhy / ln, vhz / ln
    lensq = vhx * vhx + vhy * vhy
    safe = lensq > 1e-12
    inv = np.where(safe, 1 / np.sqrt(np.where(safe, lensq, 1)), 0)
    t1 = np.where(safe[:, None], np.stack([-vhy * inv, vhx * inv, np.zeros_like(vhx)], 1), np.array([1.0, 0, 0]))
    t2 = np.cross(np.stack([vhx, vhy, vhz], 1), t1)
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    p1 = r * np.cos(phi)
    p2 = r * np.sin(phi)
    s = 0.5 * (1 + vhz)
    p2 = (1 - s) * np.sqrt(np.maximum(0, 1 - p1 * p1)) + s * p2
    p3 = np.sqrt(np.maximum(0, 1 - p1 * p1 - p2 * p2))
    nh = p1[:, None] * t1 + p2[:, None] * t2 + p3[:, None] * np.stack([vhx, vhy, vhz], 1)
    hx, hy, hz = a * nh[:, 0], a * nh[:, 1], np.maximum(nh[:, 2], 1e-6)
    h = hx[:, None] * t + hy[:, None] * b + hz[:, None] * n
    return normalize(h)


def cosine_sample(n, u1, u2):
    t, b = orthonormal_frames(n)
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.sqrt(np.maximum(0, 1 - u1))
    return x[:, None] * t + y[:, None] * b + z[:, None] * n


class BSDF:
    """Lambert + GGX microfacet lobe for a batch of shading points."""

    def __init__(self, n, wo, kd, f0, rough):
        self.n = n
        self.wo = wo
        self.kd = kd
        self.f0 = f0
        self.a = ggx_alpha(rough)
        self.nv = np.maximum(_dot(n, wo), 1e-6)
        fs = luminance(fresnel_schlick(f0, self.nv))
        fd = luminance(kd)
        tot = fs + fd
        self.p_spec = np.where(tot > 0, np.clip(fs / np.maximum(tot, 1e-12), 0.1, 0.9), 0.5)
        self.p_spec = np.where(fd <= 0, 1.0, self.p_spec)

    def eval(self, wi):
        """Returns (f * cos, pdf)."""
        n, wo, a = self.n, self.wo, self.a
        nl = _dot(n, wi)
        valid = nl > 0
        nl = np.maximum(nl, 1e-6)
        h = normalize(wi + wo)
        nh = np.maximum(_dot(n, h), 0)
        vh = np.maximum(_dot(wo, h), 1e-6)
        D = ggx_D(nh, a)
        G1o = smith_G1(self.nv, a)
        G = G1o * smith_G1(nl, a)
        F = fresnel_schlick(self.f0, vh)
        spec = F * (D * G / (4 * self.nv * nl))[:, None]
        diff = self.kd * INV_PI
        fcos = (spec + diff) * nl[:, None]
        pdf_s = G1o * D / (4 * self.nv)
        pdf_d = nl * INV_PI
        pdf = self.p_spec * pdf_s + (1 - self.p_spec) * pdf_d
        fcos[~valid] = 0
        pdf = np.where(valid, pdf, 0)
        return fcos, pdf

    def sample(self, u0, u1, u2):
        """Returns (wi, weight = f cos / pdf, pdf)."""
        choose_spec = u0 < self.p_spec
        wi = np.empty_like(self.n)
        if choose_spec.any():
            m = choose_spec
            h = sample_vndf(self.wo[m], self.n[m], self.a[m], u1[m], u2[m])
            wo = self.wo[m]
            wi[m] = 2 * _dot(wo, h)[:, None] * h - wo
        if (~choose_spec).any():
            m = ~choose_spec
            wi[m] = cosine_sample(self.n[m], u1[m], u2[m])
        fcos, pdf = self.eval(wi)
        weight = np.where(pdf[:, None] > 0, fcos / np.maximum(pdf, 1e-12)[:, None], 0)
        return wi, weight, pdf
