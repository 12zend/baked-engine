"""Light sampling for next-event estimation.

Units follow Cycles: point/spot power P gives intensity P / (4 pi) W/sr; sun
strength is irradiance in W/m^2; area lights of power P emit radiance
P / (4 A) from their -Z side.
"""

from __future__ import annotations

import numpy as np

from .geometry import normalize, orthonormal_frames
from .scene import BakeScene, Light, luminance


def _concentric_disk(u1, u2):
    a = 2 * u1 - 1
    b = 2 * u2 - 1
    r = np.where(np.abs(a) > np.abs(b), a, b)
    phi = np.where(np.abs(a) > np.abs(b), (np.pi / 4) * b / np.where(a == 0, 1, a), np.pi / 2 - (np.pi / 4) * a / np.where(b == 0, 1, b))
    phi = np.where((a == 0) & (b == 0), 0, phi)
    return r * np.cos(phi), r * np.sin(phi)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / np.maximum(e1 - e0, 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


class LightSample:
    __slots__ = ("wi", "dist", "Li", "pdf", "delta")

    def __init__(self, wi, dist, Li, pdf, delta):
        self.wi = wi  # (k, 3) direction to the light
        self.dist = dist  # (k,) distance (inf for distant lights)
        self.Li = Li  # (k, 3) incident radiance already divided by the sampling pdf
        self.pdf = pdf  # (k,) solid-angle pdf (for MIS; 0 for delta-like lights)
        self.delta = delta  # bool: BSDF sampling can never hit this light


def sample_light(light: Light, P, u1, u2):
    k = len(P)
    col = np.asarray(light.color, dtype=np.float64)
    if light.kind == "SUN":
        axis = -normalize(np.asarray(light.direction, dtype=np.float64)[None, :])[0]
        half = max(light.angle * 0.5, 1e-6)
        cos_max = np.cos(half)
        z = 1 - u1 * (1 - cos_max)
        r = np.sqrt(np.maximum(0, 1 - z * z))
        phi = 2 * np.pi * u2
        t, b = orthonormal_frames(np.broadcast_to(axis, (k, 3)).copy())
        wi = (r * np.cos(phi))[:, None] * t + (r * np.sin(phi))[:, None] * b + z[:, None] * axis
        Li = np.broadcast_to(col * light.power, (k, 3)).copy()
        return LightSample(wi, np.full(k, np.inf), Li, np.zeros(k), True)

    pos = np.asarray(light.position, dtype=np.float64)
    if light.kind in ("POINT", "SPOT"):
        to_l = pos - P
        d0 = np.linalg.norm(to_l, axis=1)
        w0 = to_l / np.maximum(d0, 1e-12)[:, None]
        if light.radius > 0:
            t, b = orthonormal_frames(w0)
            dx, dy = _concentric_disk(u1, u2)
            q = pos + light.radius * (dx[:, None] * t + dy[:, None] * b)
        else:
            q = np.broadcast_to(pos, (k, 3))
        to_q = q - P
        d = np.linalg.norm(to_q, axis=1)
        wi = to_q / np.maximum(d, 1e-12)[:, None]
        inten = light.power / (4 * np.pi)
        atten = np.ones(k)
        if light.kind == "SPOT":
            axis = normalize(np.asarray(light.direction, dtype=np.float64)[None, :])[0]
            cos_a = -(wi @ axis)
            cos_half = np.cos(light.spot_size * 0.5)
            smooth = (1 - cos_half) * light.spot_blend
            atten = np.where(cos_a > cos_half, 1.0, 0.0)
            if smooth > 0:
                atten = _smoothstep(0.0, 1.0, (cos_a - cos_half) / smooth)
        Li = col[None, :] * (inten * atten / np.maximum(d * d, 1e-12))[:, None]
        return LightSample(wi, d, Li, np.zeros(k), True)

    if light.kind == "AREA":
        au = normalize(np.asarray(light.axis_u, dtype=np.float64)[None, :])[0]
        av = normalize(np.asarray(light.axis_v, dtype=np.float64)[None, :])[0]
        nl = normalize(np.asarray(light.direction, dtype=np.float64)[None, :])[0]
        su, sv = light.size
        if light.shape in ("DISK", "ELLIPSE"):
            dx, dy = _concentric_disk(u1, u2)
            q = pos + (0.5 * su * dx)[:, None] * au + (0.5 * sv * dy)[:, None] * av
            area = np.pi * 0.25 * su * sv
        else:
            q = pos + ((u1 - 0.5) * su)[:, None] * au + ((u2 - 0.5) * sv)[:, None] * av
            area = su * sv
        to_q = q - P
        d = np.linalg.norm(to_q, axis=1)
        wi = to_q / np.maximum(d, 1e-12)[:, None]
        cos_l = -(wi @ nl)
        L = light.power / (4 * max(area, 1e-12))
        g = np.where(cos_l > 0, cos_l * area / np.maximum(d * d, 1e-12), 0.0)
        Li = col[None, :] * (L * g)[:, None]
        return LightSample(wi, d, Li, np.zeros(k), True)
    raise ValueError(f"unknown light kind {light.kind}")


def light_importance(light: Light, scene_extent):
    """Rough power estimate used to choose a light when there are many."""
    lum = float(luminance(np.asarray(light.color)))
    if light.kind == "SUN":
        return lum * light.power * scene_extent**2
    return lum * light.power


class EmissiveTriangles:
    """Area sampling of emissive triangles (two-sided), proportional to power."""

    def __init__(self, scene: BakeScene, mats):
        emissive = mats.emissive_mats[scene.material_ids]
        self.tris = np.nonzero(emissive)[0]
        self.count = len(self.tris)
        if self.count == 0:
            return
        areas = scene.areas()[self.tris]
        c = np.full((self.count, 3), 1 / 3)
        avg = mats.emission(self.tris, c)
        power = luminance(avg) * areas + 1e-12
        self.prob = power / power.sum()
        self.cdf = np.cumsum(self.prob)
        self.area = areas
        self.prob_of_tri = np.zeros(scene.triangle_count)
        self.prob_of_tri[self.tris] = self.prob
        self.area_of_tri = np.zeros(scene.triangle_count)
        self.area_of_tri[self.tris] = areas
        self.scene = scene
        self.mats = mats

    def sample(self, P, u0, u1, u2):
        i = np.minimum(np.searchsorted(self.cdf, u0 * self.cdf[-1], side="right"), self.count - 1)
        tri = self.tris[i]
        su = np.sqrt(u1)
        bary = np.stack([1 - su, su * (1 - u2), su * u2], axis=1)
        p = self.scene.positions[tri]
        q = np.einsum("ni,nij->nj", bary, p)
        ng = normalize(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
        to_q = q - P
        d = np.linalg.norm(to_q, axis=1)
        wi = to_q / np.maximum(d, 1e-12)[:, None]
        cos_l = np.abs(np.einsum("ij,ij->i", wi, ng))
        pdf = self.prob[i] / self.area[i] * d * d / np.maximum(cos_l, 1e-8)
        Le = self.mats.emission(tri, bary)
        Li = np.where((pdf > 0)[:, None] & (cos_l > 1e-8)[:, None], Le / np.maximum(pdf, 1e-20)[:, None], 0)
        return LightSample(wi, d, Li, pdf, False), tri

    def pdf_hit(self, tri, dist, wi):
        """Solid-angle pdf of having sampled the hit point (tri, dist) from the origin of ray wi."""
        p = self.scene.positions[tri]
        ng = normalize(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
        cos_l = np.abs(np.einsum("ij,ij->i", wi, ng))
        return self.prob_of_tri[tri] / np.maximum(self.area_of_tri[tri], 1e-20) * dist * dist / np.maximum(cos_l, 1e-8)
