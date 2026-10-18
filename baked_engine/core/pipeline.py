"""End-to-end bake: scene -> (triangles with xyz + uv, one texture)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .atlas import Charts, build_charts, pack_atlas
from .bvh import BVH
from .geometry import TriSet, normalize
from .imageops import atrous_denoise, dilate, encode
from .integrator import PathTracer
from .radiance import PieceIndex, RadianceLookup
from .raster import Texels, clamp_bary, rasterize
from .reflections import KIND_COPY, KIND_DOME, KIND_MAIN, ReflectionShader, build_output_geometry, plan_reflections
from .scene import BakeScene
from .settings import BakeSettings
from .shading import MaterialTable


@dataclass
class BakeResult:
    positions: np.ndarray  # (V, 3) float32, world space (Blender axes, z up)
    uvs: np.ndarray  # (V, 2) float32, v = 0 at the bottom of the image
    indices: np.ndarray  # (F, 3) uint32, counter-clockwise front faces
    image: np.ndarray  # (H, W, 4) float32 scene-linear RGB + alpha
    stats: dict = field(default_factory=dict)

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    def encoded(self, view="AGX", exposure=0.0):
        """8-bit RGBA (row 0 = bottom)."""
        rgb = self.image[..., :3] if view == "NONE" else encode(self.image[..., :3], view, exposure)
        if view == "NONE":
            rgb = np.clip(rgb * 2.0**exposure, 0, 1)
        a = np.clip(self.image[..., 3:4], 0, 1)
        return np.round(np.concatenate([rgb, a], axis=2) * 255).astype(np.uint8)


class _Progress:
    def __init__(self, cb):
        self.cb = cb
        self.t0 = time.time()
        self.timings = {}
        self._stage = None
        self._t = self.t0

    def stage(self, name):
        now = time.time()
        if self._stage:
            self.timings[self._stage] = round(now - self._t, 3)
        self._stage = name
        self._t = now
        self(0.0)

    def __call__(self, frac):
        if self.cb:
            self.cb(self._stage, float(frac))

    def done(self):
        self.stage(None)
        return self.timings


@dataclass
class LitTarget:
    """Texels whose lighting is path traced: surfaces in their original position."""

    name: str
    width: int
    height: int
    ts: TriSet  # lit triangles (positions are on the original surface)
    tri_px: np.ndarray  # (n, 3, 2)
    texels: Texels  # tri indices refer to ``ts``
    texel_size: np.ndarray  # (m,) world size of each texel


def _texel_surface(scene, ts: TriSet, tri, lam, src_normals=None):
    """Source triangle, source barycentric, world position, shading and geometric normal."""
    src = ts.src[tri]
    sb = np.einsum("ni,nij->nj", lam, ts.B[tri])
    sb = clamp_bary(sb)
    P = np.einsum("ni,nij->nj", sb, scene.positions[src])
    Ns = normalize(np.einsum("ni,nij->nj", sb, scene.normals[src]))
    p = scene.positions[src]
    Ng = normalize(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
    return src, sb, P, Ns, Ng


class PathTracerBackend:
    """Default lighting backend: the built-in numpy path tracer."""

    name = "BUILTIN"

    def bake(self, scene, targets, settings, tracer, progress):
        out = []
        total = sum(len(t.texels) for t in targets)
        done = 0
        for t in targets:
            m = len(t.texels)
            light = np.zeros((m, 3))
            var = np.zeros(m)
            valid = np.zeros(m)
            chunk = max(int(settings.texel_chunk), 1024)
            for s in range(0, m, chunk):
                e = min(m, s + chunk)
                tx = t.texels.subset(np.arange(s, e))

                def points(sampler, i, tx=tx):
                    jit = sampler.pair(i, 0) - 0.5
                    lam = tx.jittered_bary(jit)
                    _, _, P, Ns, Ng = _texel_surface(scene, t.ts, tx.tri, lam)
                    return P, Ng, Ns

                leak = t.texel_size[s:e] * 1.0
                base = done

                def prog(f, base=base, n=e - s):
                    progress((base + f * n) / max(total, 1))

                L, v, ok = tracer.bake(e - s, settings.samples, points, leak_dist=leak, progress=prog)
                light[s:e] = L
                var[s:e] = v
                valid[s:e] = ok
                done += e - s
            out.append((light, var, valid))
        return out


def _dome_local(geo, main: TriSet, dome_px):
    """Azimuthal-equidistant (polar) chart coordinates in pixels for dome triangles."""
    ids = main.tags["dome"]
    sel = np.nonzero(main.tags["kind"] == KIND_DOME)[0]
    local = np.zeros((len(sel), 3, 2))
    chart_of = np.zeros(len(sel), dtype=np.int64)
    uniq = np.unique(ids[sel])
    for ci, d in enumerate(uniq):
        m = ids[sel] == d
        dm = geo.domes[d]
        pole = dm.pole / np.linalg.norm(dm.pole)
        u = dm.u - (dm.u @ pole) * pole
        u /= np.linalg.norm(u)
        v = np.cross(pole, u)
        dirs = normalize(main.P[sel[m]] - dm.center)
        z = np.clip(dirs @ pole, -1, 1)
        theta = np.arccos(z)
        r = np.clip(theta / (np.pi / 2), 0, 1)
        phi = np.arctan2(dirs @ v, dirs @ u)
        lx = (r * np.cos(phi) * 0.5 + 0.5) * (dome_px - 1)
        ly = (r * np.sin(phi) * 0.5 + 0.5) * (dome_px - 1)
        local[m] = np.stack([lx, ly], axis=-1)
        chart_of[m] = ci
    return sel, local, chart_of, len(uniq)


def _charts_for_main(geo, main: TriSet, settings, dome_px):
    kind = main.tags["kind"]
    chain = main.tags["chain"]
    scal = np.nonzero(kind != KIND_DOME)[0]
    group = kind[scal] * (1 << 24) + (chain[scal] + 1)
    weight = np.where(kind[scal] == KIND_COPY, settings.mirror_density, 1.0)
    ch = build_charts(main.P[scal], group, weight, settings.chart_buckets)
    n = len(main)
    chart = np.zeros(n, dtype=np.int64)
    local = np.zeros((n, 3, 2))
    chart[scal] = ch.chart
    local[scal] = ch.local
    extent = [ch.extent]
    fixed = [ch.fixed]
    dsel, dloc, dchart, nd = _dome_local(geo, main, dome_px)
    if len(dsel):
        chart[dsel] = dchart + ch.count
        local[dsel] = dloc
        extent.append(np.full((nd, 2), float(dome_px - 1)))
        fixed.append(np.ones(nd, dtype=bool))
    return Charts(chart, local, np.concatenate(extent), np.concatenate(fixed))


def _weld(P, UV):
    key = np.concatenate([P.reshape(-1, 3), UV.reshape(-1, 2)], axis=1).astype(np.float32)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    return uniq[:, :3], uniq[:, 3:], inv.reshape(-1, 3).astype(np.uint32)


def _scatter(texels, values, h, w, fill=0.0):
    shape = (h, w) + values.shape[1:]
    img = np.full(shape, fill, dtype=np.float64)
    img[texels.y, texels.x] = values
    return img


def bake(scene: BakeScene, settings: BakeSettings | None = None, progress=None, backend=None) -> BakeResult:
    settings = settings or BakeSettings()
    prog = _Progress(progress)
    backend = backend or PathTracerBackend()
    W, H = int(settings.width), int(settings.height)
    stats = {"notes": []}

    prog.stage("prepare")
    mats = MaterialTable(scene)
    cut = mats.cutout_tris
    bvh = BVH(scene.positions, cut if len(cut) else None, mats.alpha_keep)
    tracer = PathTracer(scene, settings.max_bounces, settings.clamp_indirect, settings.seed, bvh=bvh, mats=mats)

    prog.stage("reflections")
    plan = plan_reflections(scene, settings, mats, bvh)
    geo = build_output_geometry(scene, plan, settings)
    stats["notes"] += plan.notes + geo.notes
    main, aux = geo.main, geo.aux
    kind = main.tags["kind"]

    prog.stage("atlas")
    dome_px = settings.dome_size if settings.dome_size > 0 else max(32, W // 8)
    dome_px = min(dome_px, W - 8, H - 8)
    charts = _charts_for_main(geo, main, settings, dome_px)
    layout = pack_atlas(charts, W, H, settings.padding, settings.texel_density)
    AW = settings.aux_size if settings.aux_size > 0 else max(128, W // 2)
    AH = AW
    aux_layout = None
    if len(aux):
        ach = build_charts(aux.P, aux.tags["refl"] + 1, np.ones(len(aux)), settings.chart_buckets)
        aux_layout = pack_atlas(ach, AW, AH, 2, layout.scale * settings.aux_density)

    prog.stage("rasterize")
    tex_main = rasterize(layout.tri_px, W, H)
    tex_kind = kind[tex_main.tri]
    lit_main = tex_main.subset(tex_kind == KIND_MAIN)
    targets = [LitTarget("main", W, H, main, layout.tri_px, lit_main, np.full(len(lit_main), 1.0 / layout.scale))]
    if aux_layout is not None:
        tex_aux = rasterize(aux_layout.tri_px, AW, AH)
        targets.append(LitTarget("aux", AW, AH, aux, aux_layout.tri_px, tex_aux, np.full(len(tex_aux), 1.0 / aux_layout.scale)))

    prog.stage("lighting")
    lit = backend.bake(scene, targets, settings, tracer, prog)

    prog.stage("compose")
    diffuse_imgs = []
    masks = []
    lit_info = []
    for t, (light, var, valid) in zip(targets, lit):
        tx = t.texels
        src, sb, P, Ns, Ng = _texel_surface(scene, t.ts, tx.tri, tx.bary)
        kd = np.zeros((len(tx), 3))
        em = np.zeros((len(tx), 3))
        alpha = np.zeros(len(tx))
        ss = [(0.0, 0.0), (-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        for jx, jy in ss:
            lam = tx.jittered_bary(np.tile([jx, jy], (len(tx), 1)))
            s2, b2, _, _, _ = _texel_surface(scene, t.ts, tx.tri, lam)
            col, a = mats.base_color(s2, b2, with_alpha=True)
            kd += col * (1 - mats.metal[mats.tri_mat[s2]])[:, None]
            em += mats.emission(s2, b2)
            alpha += a
        kd /= len(ss)
        em /= len(ss)
        alpha /= len(ss)
        ok = valid >= 0.5
        mask = np.zeros((t.height, t.width), dtype=bool)
        mask[tx.y[ok], tx.x[ok]] = True
        L_img = _scatter(tx, light, t.height, t.width)
        if settings.denoise and settings.samples > 1:
            prog.stage(f"denoise {t.name}")
            v_img = _scatter(tx, var, t.height, t.width)
            p_img = _scatter(tx, P, t.height, t.width)
            n_img = _scatter(tx, Ns, t.height, t.width)
            s_img = _scatter(tx, t.texel_size, t.height, t.width, fill=1.0)
            L_img = atrous_denoise(L_img, v_img, mask, p_img, n_img, s_img, settings.denoise_iterations)
            prog.stage("compose")
        light_d = L_img[tx.y, tx.x]
        rad = kd * light_d + em
        lit_info.append(dict(src=src, sb=sb, P=P, Ns=Ns, rad=rad, diffuse=rad.copy(), alpha=alpha, ok=ok))
        img = _scatter(tx, rad, t.height, t.width)
        diffuse_imgs.append(dilate(img, mask))
        masks.append(mask)

    # pieces of the original scene with their atlas location (for radiance lookups)
    is_lit_main = kind == KIND_MAIN
    srcs = [main.src[is_lit_main]]
    Bs = [main.B[is_lit_main]]
    atl = [np.zeros(int(is_lit_main.sum()), dtype=np.int8)]
    uvs = [layout.tri_px[is_lit_main]]
    if aux_layout is not None:
        srcs.append(aux.src)
        Bs.append(aux.B)
        atl.append(np.ones(len(aux), dtype=np.int8))
        uvs.append(aux_layout.tri_px)
    pieces = PieceIndex(scene.triangle_count, np.concatenate(srcs), np.concatenate(Bs), np.concatenate(atl), np.concatenate(uvs))
    lookup_diffuse = RadianceLookup(pieces, diffuse_imgs + [None] * (2 - len(diffuse_imgs)))

    # ---- view-dependent gloss baked into the texture (metals, curved / small reflectors)
    full_imgs = [im.copy() for im in diffuse_imgs]
    shader = ReflectionShader(scene, plan, bvh, None, lookup_diffuse, tracer.eps, settings.reflection_samples, settings.seed)
    if plan.texture_spec.any():
        prog.stage("glossy")
        shader.full = lookup_diffuse
        for i, (t, info) in enumerate(zip(targets, lit_info)):
            sel = np.nonzero(plan.texture_spec[info["src"]])[0]
            if not len(sel):
                continue
            src = info["src"][sel]
            mat = mats.tri_mat[src]
            base = mats.base_color(src, info["sb"][sel])
            metal = mats.metal[mat][:, None]
            f0 = (0.08 * mats.spec[mat])[:, None] * (1 - metal) + base * metal
            spec = shader.texture_specular(info["P"][sel], info["Ns"][sel], f0, mats.rough[mat], plan.camera)
            info["rad"][sel] += spec
            img = _scatter(t.texels, info["rad"], t.height, t.width)
            full_imgs[i] = dilate(img, masks[i])
    lookup_full = RadianceLookup(pieces, full_imgs + [None] * (2 - len(full_imgs)))

    # ---- mirrored geometry and sky domes
    out_rgb = np.zeros((len(tex_main), 3))
    out_a = np.ones(len(tex_main))
    out_ok = np.ones(len(tex_main), dtype=bool)
    lit_idx = np.nonzero(tex_kind == KIND_MAIN)[0]
    out_rgb[lit_idx] = lit_info[0]["rad"]
    out_a[lit_idx] = lit_info[0]["alpha"]
    out_ok[lit_idx] = lit_info[0]["ok"]
    mir = np.nonzero(tex_kind != KIND_MAIN)[0]
    if len(mir):
        prog.stage("reflection texels")
        if len(targets) > 1:
            atx = targets[1].texels
            refl_tag = aux.tags["refl"][atx.tri]
            for r in plan.reflectors:
                m = refl_tag == r.index
                r.mean_diffuse = lit_info[1]["diffuse"][m].mean(axis=0) if m.any() else np.zeros(3)
        for r in plan.reflectors:
            if r.mean_diffuse is None:
                r.mean_diffuse = np.zeros(3)
        shader.full = lookup_full
        tx = tex_main.subset(mir)
        Q = np.einsum("ni,nij->nj", tx.bary, main.P[tx.tri])
        k = kind[tx.tri]
        src = main.src[tx.tri]
        sb = clamp_bary(np.einsum("ni,nij->nj", tx.bary, main.B[tx.tri]))
        tsize = np.where(k == KIND_COPY, 1.0 / (layout.scale * settings.mirror_density), (np.pi / 2) / (0.5 * dome_px))
        chain = main.tags["chain"][tx.tri]
        chunk = 1 << 16
        for s in range(0, len(tx), chunk):
            e = min(len(tx), s + chunk)
            out_rgb[mir[s:e]] = shader.shade(Q[s:e], chain[s:e], k[s:e], src[s:e], sb[s:e], tsize[s:e], geo.chains)
            prog((e) / len(tx))
        copy = k == KIND_COPY
        if copy.any():
            _, a = mats.base_color(src[copy], sb[copy], with_alpha=True)
            out_a[mir[copy]] = a

    prog.stage("finalize")
    mask = np.zeros((H, W), dtype=bool)
    mask[tex_main.y[out_ok], tex_main.x[out_ok]] = True
    img = np.zeros((H, W, 4))
    img[tex_main.y, tex_main.x, :3] = out_rgb
    img[tex_main.y, tex_main.x, 3] = out_a
    img = dilate(img, mask)
    img[..., 3] = np.clip(img[..., 3], 0, 1)

    P, UV, F = _weld(main.P, layout.uvs())
    stats.update(
        texel_density=layout.scale,
        atlas_fill=layout.fill_ratio,
        triangles=int(len(main)),
        triangles_scene=int((kind == KIND_MAIN).sum()),
        triangles_mirrored=int((kind == KIND_COPY).sum()),
        triangles_dome=int((kind == KIND_DOME).sum()),
        reflectors=[
            dict(
                index=r.index,
                normal=[round(float(x), 4) for x in r.plane.normal],
                offset=round(r.plane.d, 4),
                area=round(r.area, 3),
                roughness=round(r.roughness, 3),
                removes_behind=r.remove,
                mirror_depth=round(float(r.mirror_depth), 3),
            )
            for r in plan.reflectors
        ],
        texture_reflection_triangles=int(plan.texture_spec.sum()),
        backend=getattr(backend, "name", "custom"),
    )
    stats["timings"] = prog.done()
    return BakeResult(P.astype(np.float32), UV.astype(np.float32), F, img.astype(np.float32), stats)
