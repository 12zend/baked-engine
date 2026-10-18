"""Wavefront path tracer used to bake view-independent (diffuse) lighting at surface points.

For every bake point the integrator estimates the radiance a white Lambertian
surface would reflect there (irradiance / pi). The caller multiplies by the
surface albedo afterwards, which keeps texture detail out of the noise and
lets the denoiser work on lighting alone.
"""

from __future__ import annotations

import numpy as np

from .bvh import BVH
from .lights import EmissiveTriangles, light_importance, sample_light
from .sampling import Sampler
from .scene import BakeScene, luminance
from .shading import BSDF, INV_PI, MaterialTable, cosine_sample


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _power_mis(a, b):
    a2 = a * a
    return np.where(a2 > 0, a2 / np.maximum(a2 + b * b, 1e-30), 0.0)


class PathTracer:
    def __init__(self, scene: BakeScene, max_bounces=6, clamp=10.0, seed=0, all_lights_threshold=8, bvh=None, mats=None):
        self.scene = scene
        self.mats = mats or MaterialTable(scene)
        if bvh is None:
            cut = self.mats.cutout_tris
            bvh = BVH(scene.positions, cut if len(cut) else None, self.mats.alpha_keep)
        self.bvh = bvh
        self.max_bounces = max(int(max_bounces), 0)
        self.clamp = float(clamp) if clamp and clamp > 0 else np.inf
        self.seed = seed
        self.env = scene.environment
        self.env.build_sampling()
        self.env_nee = self.env.texture is not None
        self.emissive = EmissiveTriangles(scene, self.mats)
        self.extent = scene.extent()
        self.eps = 2e-6 * self.extent
        self.lights = list(scene.lights)
        self.all_lights = len(self.lights) <= all_lights_threshold
        if self.lights and not self.all_lights:
            imp = np.array([light_importance(l, self.extent) for l in self.lights]) + 1e-12
            self.light_prob = imp / imp.sum()
            self.light_cdf = np.cumsum(self.light_prob)
        self.rng = np.random.default_rng(seed + 7919)

    # ------------------------------------------------------------------
    def offset(self, P, Ng, w):
        side = np.where(_dot(Ng, w) >= 0, 1.0, -1.0)
        mag = self.eps + 4e-7 * np.abs(P).max(axis=1)
        return P + Ng * (side * mag)[:, None]

    def _rand(self, n, k):
        return self.rng.random((n, k)).T

    def direct(self, P, Ng, bsdf_eval, u_light, u_extra):
        """Next-event estimation (lights, emissive triangles, environment) with MIS.

        bsdf_eval(wi, sel) -> (f*cos (k,3), pdf (k,)) for the points ``sel``.
        Returns (k, 3) radiance contribution.
        """
        k = len(P)
        out = np.zeros((k, 3))
        o_list, d_list, t_list, c_list, i_list = [], [], [], [], []

        def add(sel, wi, dist, contrib):
            good = (contrib.max(axis=1) > 0) & (_dot(Ng[sel], wi) > 0)
            if not good.any():
                return
            sel = sel[good]
            wi = wi[good]
            org = self.offset(P[sel], Ng[sel], wi)
            o_list.append(org)
            d_list.append(wi)
            t_list.append(np.where(np.isfinite(dist[good]), dist[good] * (1 - 1e-4) - 2 * self.eps, np.inf))
            c_list.append(contrib[good])
            i_list.append(sel)

        all_idx = np.arange(k)
        if self.lights:
            if self.all_lights:
                for light in self.lights:
                    ls = sample_light(light, P, u_light[0], u_light[1])
                    fcos, _ = bsdf_eval(ls.wi, all_idx)
                    add(all_idx, ls.wi, ls.dist, fcos * ls.Li)
            else:
                pick = np.minimum(np.searchsorted(self.light_cdf, u_extra[2] * self.light_cdf[-1], side="right"), len(self.lights) - 1)
                for li in np.unique(pick):
                    sel = np.nonzero(pick == li)[0]
                    ls = sample_light(self.lights[li], P[sel], u_light[0][sel], u_light[1][sel])
                    fcos, _ = bsdf_eval(ls.wi, sel)
                    add(sel, ls.wi, ls.dist, fcos * ls.Li / self.light_prob[li])
        if self.emissive.count:
            ls, _ = self.emissive.sample(P, u_extra[0], u_extra[1], u_extra[3])
            fcos, pdf_b = bsdf_eval(ls.wi, all_idx)
            w = _power_mis(ls.pdf, pdf_b)
            add(all_idx, ls.wi, ls.dist, fcos * ls.Li * w[:, None])
        if self.env_nee:
            wi, pdf_e = self.env.sample(u_extra[4], u_extra[5], u_extra[6])
            fcos, pdf_b = bsdf_eval(wi, all_idx)
            w = _power_mis(pdf_e, pdf_b)
            Le = self.env.eval(wi)
            add(all_idx, wi, np.full(k, np.inf), fcos * Le * (w / np.maximum(pdf_e, 1e-20))[:, None])
        if o_list:
            org = np.concatenate(o_list)
            dirs = np.concatenate(d_list)
            tmax = np.concatenate(t_list)
            contrib = np.concatenate(c_list)
            idx = np.concatenate(i_list)
            occ = self.bvh.occluded(org, dirs, tmax)
            vis = ~occ
            np.add.at(out, idx[vis], contrib[vis])
        return out

    def _clamp(self, c):
        if not np.isfinite(self.clamp):
            return c
        lum = luminance(c)
        s = np.where(lum > self.clamp, self.clamp / np.maximum(lum, 1e-30), 1.0)
        return c * s[:, None]

    def _env_miss(self, dirs, pdf_prev):
        Le = self.env.eval(dirs).astype(np.float64)
        if self.env_nee:
            w = _power_mis(pdf_prev, self.env.pdf(dirs))
            Le = Le * w[:, None]
        return Le

    # ------------------------------------------------------------------
    def trace_from_surface(self, P, Ng, Ns, sampler: Sampler, i, leak_dist=None):
        """One path sample per bake point. Returns (lighting (n,3), valid (n,))."""
        n = len(P)
        Ns = np.where((_dot(Ns, Ng) > 0)[:, None], Ns, Ng)
        L = np.zeros((n, 3))
        u_light = sampler.pair(i, 1).T
        u_dir = sampler.pair(i, 2).T
        u_x = np.concatenate([sampler.pair(i, 3).T, self._rand(n, 5)])

        def white_eval(wi, sel):
            c = _dot(Ns[sel], wi)
            ok = (c > 0) & (_dot(Ng[sel], wi) > 0)
            c = np.where(ok, c, 0.0)
            return np.repeat((c * INV_PI)[:, None], 3, axis=1), c * INV_PI

        L += self.direct(P, Ng, white_eval, u_light, u_x)
        valid = np.ones(n, dtype=bool)
        if self.max_bounces == 0:
            return L, valid

        wi = cosine_sample(Ns, u_dir[0], u_dir[1])
        up = _dot(Ng, wi) > 0
        pdf_prev = _dot(Ns, wi) * INV_PI
        idx = np.nonzero(up)[0]
        org = self.offset(P[idx], Ng[idx], wi[idx])
        dirs = wi[idx]
        thr = np.ones((len(idx), 3))
        pdf_prev = pdf_prev[idx]
        depth = 1
        while len(idx):
            h = self.bvh.intersect(org, dirs)
            miss = ~h.hit
            if miss.any():
                c = thr[miss] * self._env_miss(dirs[miss], pdf_prev[miss])
                if depth >= 2:
                    c = self._clamp(c)
                np.add.at(L, idx[miss], c)
            hit = h.hit
            if depth == 1 and leak_dist is not None:
                leak = hit & ~h.front & (h.t < leak_dist[idx])
                valid[idx[leak]] = False
                hit = hit & ~leak
            sel = np.nonzero(hit)[0]
            if not len(sel):
                break
            idx = idx[sel]
            thr = thr[sel]
            dirs = dirs[sel]
            pdf_prev = pdf_prev[sel]
            tri = h.tri[sel]
            t = h.t[sel]
            bary = np.stack([1 - h.u[sel] - h.v[sel], h.u[sel], h.v[sel]], axis=1)
            surf = self.mats.surface(tri, bary)
            wo = -dirs
            ng = surf["Ng"]
            ns = surf["Ns"]
            flip = _dot(ng, wo) < 0
            ng = np.where(flip[:, None], -ng, ng)
            ns = np.where(flip[:, None], -ns, ns)
            ns = np.where((_dot(ns, wo) > 1e-4)[:, None], ns, ng)
            # emission seen by the previous vertex (MIS against emissive-triangle NEE)
            if self.emissive.count:
                Le = self.mats.emission(tri, bary)
                em = Le.max(axis=1) > 0
                if em.any():
                    pl = self.emissive.pdf_hit(tri[em], t[em], dirs[em])
                    w = _power_mis(pdf_prev[em], pl)
                    c = thr[em] * Le[em] * w[:, None]
                    if depth >= 2:
                        c = self._clamp(c)
                    np.add.at(L, idx[em], c)
            if depth >= self.max_bounces:
                break
            Pp = np.einsum("ni,nij->nj", bary, self.scene.positions[tri])
            bsdf = BSDF(ns, wo, surf["kd"], surf["f0"], surf["rough"])
            m = len(idx)
            u_l = self._rand(m, 2)
            u_e = self._rand(m, 7)

            def eval_sel(wi_, s_):
                if len(s_) == m:
                    return bsdf.eval(wi_)
                sub = BSDF(ns[s_], wo[s_], surf["kd"][s_], surf["f0"][s_], surf["rough"][s_])
                return sub.eval(wi_)

            d = self.direct(Pp, ng, eval_sel, u_l, u_e)
            np.add.at(L, idx, self._clamp(thr * d))
            u = self._rand(m, 4)
            wi, weight, pdf = bsdf.sample(u[0], u[1], u[2])
            ok = (pdf > 0) & (_dot(ng, wi) > 0)
            thr = thr * weight
            if depth >= 3:
                q = np.clip(thr.max(axis=1), 0.05, 0.95)
                ok &= u[3] < q
                thr = thr / q[:, None]
            keep = np.nonzero(ok)[0]
            idx = idx[keep]
            thr = thr[keep]
            dirs = wi[keep]
            pdf_prev = pdf[keep]
            org = self.offset(Pp[keep], ng[keep], dirs)
            depth += 1
        return L, valid

    def bake(self, n, spp, sample_points, leak_dist=None, progress=None):
        """Estimate lighting at ``n`` bake points with ``spp`` samples each.

        sample_points(sampler, i) -> (P, Ng, Ns) for sample ``i`` (lets callers jitter positions).
        Returns (mean (n,3), variance of the mean (n,), valid_fraction (n,)).
        """
        sampler = Sampler(n, spp, seed=self.seed)
        s1 = np.zeros((n, 3))
        s2 = np.zeros(n)
        cnt = np.zeros(n)
        for i in range(spp):
            P, Ng, Ns = sample_points(sampler, i)
            L, ok = self.trace_from_surface(P, Ng, Ns, sampler, i, leak_dist)
            L = np.where(ok[:, None], L, 0)
            s1 += L
            lum = luminance(L)
            s2 += lum * lum
            cnt += ok
            if progress:
                progress((i + 1) / spp)
        c = np.maximum(cnt, 1)
        mean = s1 / c[:, None]
        ml = luminance(mean)
        var = np.maximum(s2 / c - ml * ml, 0) / c
        return mean, var, cnt / max(spp, 1)
