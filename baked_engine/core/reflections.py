"""Reflections baked into geometry.

A planar mirror is view independent when expressed as geometry: the scene
mirrored through the plane is exactly what any camera sees "inside" the
mirror. Baked Engine therefore

1. finds glossy planar patches (a sea, a lake, a polished floor, a mirror),
2. removes the patch from the output (it becomes a window) together with the
   geometry behind it that the window would otherwise reveal (e.g. the sea bed),
3. adds the scene in front of the plane mirrored behind it - clipped to the
   reflection volume so it can only be seen through the window - plus a
   mirrored sky dome, recursively for mirrors seen in mirrors,
4. bakes the texture of the mirrored copies as "what the reflection looks
   like": source radiance blurred by roughness (a ray gather around the
   reflected direction, so silhouettes blend into what is behind them),
   mixed with the reflector's own colour by Fresnel, with light glints.

Curved or small glossy surfaces fall back to baking their reflection into the
texture for a reference view point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Plane, TriSet, clip_to_halfspaces, convex_hull_2d, mirror, normalize, orthonormal_frames, plane_basis
from .scene import BakeScene, ReflectorConfig, luminance

KIND_MAIN = 0
KIND_COPY = 1
KIND_DOME = 2
KIND_AUX = 3


@dataclass
class Reflector:
    index: int
    plane: Plane
    tris: np.ndarray
    centroid: np.ndarray
    u: np.ndarray
    v: np.ndarray
    hull: np.ndarray
    radius: float
    area: float
    config: ReflectorConfig
    f0: np.ndarray
    roughness: float
    sigma: float
    dist_tol: float
    volume: list = field(default_factory=list)  # behind-mirror region: inside = front of every plane
    removal_volume: list = field(default_factory=list)  # cleared region behind the mirror
    remove: bool = False
    mirror_depth: float = 0.0
    viewer: np.ndarray = None
    dome: bool = True
    mean_diffuse: np.ndarray = None

    @property
    def front_volume(self):
        """Image of the reflection volume in front of the mirror (what gets mirrored)."""
        return [self.plane.reflect_plane(p) for p in self.volume]


@dataclass
class ReflectionPlan:
    reflectors: list
    texture_spec: np.ndarray  # (T,) bool: bake glossy reflection into the texture
    camera: np.ndarray = None
    notes: list = field(default_factory=list)


def _material_arrays(scene: BakeScene, mats):
    rough = mats.rough[scene.material_ids]
    metal = mats.metal[scene.material_ids]
    return rough, metal


def plan_reflections(scene: BakeScene, settings, mats, bvh) -> ReflectionPlan:
    T = scene.triangle_count
    extent = scene.extent()
    rough, metal = _material_arrays(scene, mats)
    modes = np.array([o.reflector.mode for o in scene.objects] + ["OFF"])
    tri_mode = modes[np.minimum(scene.object_ids, len(scene.objects))]
    texture_spec = np.zeros(T, dtype=bool)
    camera = None
    if scene.camera is not None and settings.reference_view in ("AUTO", "CAMERA"):
        camera = np.asarray(scene.camera.position, dtype=np.float64)
    notes = []
    if not settings.reflections:
        if settings.specular_bake:
            texture_spec = (tri_mode != "OFF") & ((metal >= 0.5) | (tri_mode == "TEXTURE"))
        return ReflectionPlan([], texture_spec, camera, notes)

    glossy = rough <= 0.25
    candidate = (tri_mode == "GEOMETRY") | ((tri_mode == "AUTO") & glossy)
    if settings.reflection_depth <= 0:
        candidate[:] = False
    fn = scene.face_normals()
    areas = scene.areas()
    cent = scene.positions.mean(axis=1)
    remaining = np.nonzero(candidate & (areas > 0))[0]
    patches = []

    # 1) objects that are planar as a whole but not flat in detail (wavy water): one mirror plane,
    #    the wave slopes widen the reflection blur instead of shattering it into many patches
    for oi in np.unique(scene.object_ids[remaining]):
        sel = remaining[scene.object_ids[remaining] == oi]
        if len(sel) < 8:
            continue
        cfg = scene.objects[oi].reflector
        w = areas[sel]
        n0 = normalize((fn[sel] * w[:, None]).sum(axis=0)[None])[0]
        ang = np.arccos(np.clip(fn[sel] @ n0, -1, 1))
        rms = float(np.sqrt((w * ang * ang).sum() / max(w.sum(), 1e-30)))
        if rms <= cfg.planar_tolerance or rms > np.radians(20) or (ang > np.radians(60)).any():
            continue
        d0 = float((cent[sel] @ n0 * w).sum() / max(w.sum(), 1e-30))
        heights = np.abs(scene.positions[sel] @ n0 - d0)
        pts = scene.positions[sel].reshape(-1, 3)
        radius = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max())
        if heights.max() > 0.05 * max(radius, 1e-9):
            continue
        patches.append(dict(members=sel, n=n0, d=d0, cfg=cfg, dist_tol=float(heights.max()) + 2e-4 * extent, extra=1.5 * rms))
        notes.append(f"object '{scene.objects[oi].name}': wavy surface reflected as one plane (wave slope {np.degrees(rms):.1f} deg adds blur)")
        remaining = np.setdiff1d(remaining, sel)

    # 2) greedy planar patches over the remaining glossy triangles
    remaining = remaining[np.argsort(-areas[remaining])]
    while len(remaining) and len(patches) < 4 * settings.max_reflectors:
        seed = remaining[0]
        cfg = scene.objects[scene.object_ids[seed]].reflector
        cos_tol = np.cos(cfg.planar_tolerance)
        dist_tol = 2e-4 * extent + 0.05 * np.sin(cfg.planar_tolerance) * extent
        n0 = fn[seed]
        d0 = float(n0 @ cent[seed])
        members = None
        for _ in range(2):
            cand_n = fn[remaining] @ n0 >= cos_tol
            dist = np.abs(scene.positions[remaining] @ n0 - d0).max(axis=1)
            sel = remaining[cand_n & (dist <= dist_tol)]
            if len(sel) == 0:
                sel = np.array([seed])
            w = areas[sel]
            n0 = normalize((fn[sel] * w[:, None]).sum(axis=0)[None])[0]
            d0 = float((cent[sel] @ n0 * w).sum() / max(w.sum(), 1e-30))
            members = sel
        remaining = np.setdiff1d(remaining, members, assume_unique=False)
        remaining = remaining[np.argsort(-areas[remaining])]
        patches.append(dict(members=members, n=n0, d=d0, cfg=cfg, dist_tol=dist_tol, extra=0.0))

    # 3) accept patches as geometry reflectors
    convex = _convex_glossy_boundary(scene, candidate, patches, fn)
    reflectors = []
    min_area = (0.02 * extent) ** 2
    for pi, pt in enumerate(patches):
        members = pt["members"]
        area = float(areas[members].sum())
        forced = (tri_mode[members] == "GEOMETRY").mean() > 0.5
        if len(reflectors) >= settings.max_reflectors:
            texture_spec[members] = True
            notes.append(f"{len(members)} glossy triangles beyond max_reflectors use texture reflections")
            continue
        if not forced and area < min_area:
            texture_spec[members] = True
            continue
        if not forced and convex[pi] > 0.1:
            # a planar facet of a curved / faceted glossy object: neighbouring mirror worlds would
            # show through each other, so bake its reflection into the texture instead
            texture_spec[members] = True
            continue
        r = _make_reflector(len(reflectors), scene, members, pt["n"], pt["d"], pt["cfg"], mats, extent, pt["dist_tol"], pt["extra"])
        _finish_volume(r, scene, bvh, settings, extent, camera, notes, forced)
        if r.volume is None:
            texture_spec[members] = True
            continue
        reflectors.append(r)
    if len(remaining):
        texture_spec[remaining] = True
    if settings.specular_bake:
        texture_spec |= (tri_mode != "OFF") & ((metal >= 0.5) | (tri_mode == "TEXTURE"))
        in_geo = np.zeros(T, dtype=bool)
        for r in reflectors:
            in_geo[r.tris] = True
        # geometry reflector pieces also get a texture reflection: it is used where they are seen
        # opaque inside other mirrors (recursion limit) and nowhere else.
        texture_spec |= in_geo
    else:
        texture_spec[:] = False
    return ReflectionPlan(reflectors, texture_spec, camera, notes)


def _convex_glossy_boundary(scene, candidate, patches, fn):
    """Per patch: fraction of its boundary that meets another glossy patch at a convex edge."""
    from .atlas import weld_vertices

    out = np.zeros(len(patches))
    if not patches:
        return out
    T = scene.triangle_count
    patch_of = np.full(T, -1, dtype=np.int64)
    for i, p in enumerate(patches):
        patch_of[p["members"]] = i
    cand = np.nonzero(candidate)[0]
    if len(cand) == 0:
        return out
    ext = scene.extent()
    vid = weld_vertices(scene.positions[cand].reshape(-1, 3), np.zeros(len(cand) * 3, np.int64), max(ext * 1e-6, 1e-9)).reshape(-1, 3)
    e = np.stack([vid[:, [0, 1]], vid[:, [1, 2]], vid[:, [2, 0]]], axis=1).reshape(-1, 2)
    e.sort(axis=1)
    tri = np.repeat(cand, 3)
    p0 = scene.positions[cand][:, [0, 1, 2]].reshape(-1, 3)
    p1 = scene.positions[cand][:, [1, 2, 0]].reshape(-1, 3)
    length = np.linalg.norm(p1 - p0, axis=1)
    order = np.lexsort((e[:, 1], e[:, 0]))
    es = e[order]
    same = np.zeros(len(es), dtype=bool)
    same[1:] = (es[1:] == es[:-1]).all(axis=1)
    a = tri[order][:-1][same[1:]]
    b = tri[order][1:][same[1:]]
    ln = length[order][1:][same[1:]]
    pa, pb = patch_of[a], patch_of[b]
    differ = pa != pb
    a, b, pa, pb, ln = a[differ], b[differ], pa[differ], pb[differ], ln[differ]
    cent = scene.positions.mean(axis=1)
    boundary = np.zeros(len(patches))
    bad = np.zeros(len(patches))
    for x, y, px in ((a, b, pa), (b, a, pb)):
        ok = px >= 0
        n = fn[x[ok]]
        # the neighbour bends away behind this facet's plane -> convex edge
        behind = np.einsum("ij,ij->i", cent[y[ok]] - cent[x[ok]], n) < -1e-9
        np.add.at(boundary, px[ok], ln[ok])
        np.add.at(bad, px[ok], ln[ok] * behind)
    # true perimeter of every patch: edge occurrences not shared with another triangle of the patch
    pe = patch_of[tri]
    key = np.concatenate([e, pe[:, None]], axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    single = (cnt[inv.reshape(-1)] == 1) & (pe >= 0)
    perim = np.bincount(pe[single], weights=length[single], minlength=len(patches))
    out = bad / np.maximum(np.maximum(perim, boundary), 1e-12)
    return out


def _make_reflector(index, scene, members, n, d, cfg, mats, extent, dist_tol, extra_sigma=0.0):
    areas = scene.areas()[members]
    w = areas / max(areas.sum(), 1e-30)
    plane = Plane(tuple(n), d)
    pts = scene.positions[members].reshape(-1, 3)
    u, v = plane_basis(n)
    centroid = (scene.positions[members].mean(axis=1) * w[:, None]).sum(axis=0)
    centroid = centroid - (centroid @ n - d) * n
    uv = np.stack([(pts - centroid) @ u, (pts - centroid) @ v], axis=1)
    hull = convex_hull_2d(uv)
    radius = float(np.linalg.norm(hull, axis=1).max()) if len(hull) else 0.0
    mat = scene.material_ids[members]
    c = np.full((len(members), 3), 1 / 3)
    base = mats.base_color(members, c)
    metal = mats.metal[mat][:, None]
    f0 = (0.08 * mats.spec[mat])[:, None] * (1 - metal) + base * metal
    f0 = (f0 * w[:, None]).sum(axis=0)
    rough = float((mats.rough[mat] * w).sum())
    alpha = max(rough * rough, 1e-3)
    sigma = 0.7 * alpha + max(cfg.blur, 0.0) + extra_sigma
    return Reflector(index, plane, members, centroid, u, v, hull, radius, float(areas.sum()), cfg, f0, rough, sigma, dist_tol)


def _finish_volume(r: Reflector, scene, bvh, settings, extent, camera, notes, forced):
    cfg = r.config
    n = r.plane.normal
    depth = cfg.mirror_depth if cfg.mirror_depth > 0 else 2.0 * extent
    if cfg.remove_below == "ON":
        r.remove = True
    elif cfg.remove_below == "OFF":
        r.remove = False
    else:
        r.remove = bool(n[2] >= 0.85)
    if not r.remove:
        # how much free space is behind the mirror? mirrored geometry deeper than that collides
        rng = np.random.default_rng(r.index)
        k = 96
        pick = rng.choice(r.tris, size=k, p=scene.areas()[r.tris] / scene.areas()[r.tris].sum())
        su = np.sqrt(rng.random(k))
        b = np.stack([1 - su, su * (1 - rng.random(k)), np.zeros(k)], axis=1)
        b[:, 2] = 1 - b[:, 0] - b[:, 1]
        p = np.einsum("ni,nij->nj", b, scene.positions[pick])
        org = p - n * (r.dist_tol * 2 + 1e-5 * extent)
        h = bvh.intersect(org, np.broadcast_to(-n, (k, 3)).copy())
        free = float(np.percentile(np.where(h.hit, h.t, 1e30), 10))
        if free < depth:
            if free < 0.2 * max(r.radius, 1e-6) and not forced and cfg.mode != "GEOMETRY":
                notes.append(f"reflector {r.index}: only {free:.3g} m free behind it, using texture reflection")
                r.volume = None
                return
            depth = min(depth, free)
    r.mirror_depth = depth
    # reference view point for fresnel / blur: the camera when it looks at the mirror's front
    has_cam = camera is not None and r.plane.distance(camera) > 1e-4 * extent
    if has_cam:
        r.viewer = camera
    else:
        h = max(0.1 * r.radius, 1e-3 * extent)
        r.viewer = r.centroid + n * h
    planes = [r.plane.flipped(), Plane(tuple(n), r.plane.d - depth)]
    wide = list(planes)
    flare = cfg.flare if cfg.flare >= 0 else np.radians(80.0 if r.remove else 50.0)
    flare = float(np.clip(flare, 0.0, np.radians(89.9)))
    margin = np.radians(8.0)
    hull = r.hull

    def side_plane(t3, m, e0, tilt):
        behind = -np.cos(tilt) * n + np.sin(tilt) * m
        N = np.cross(t3, behind)
        N /= np.linalg.norm(N)
        if N @ (r.centroid - n * 1e-3 * extent - e0) < 0:
            N = -N
        return Plane(tuple(N), float(N @ e0))

    if len(hull) >= 3:
        for i in range(len(hull)):
            a2 = hull[i]
            b2 = hull[(i + 1) % len(hull)]
            e = b2 - a2
            ln = np.linalg.norm(e)
            if ln < 1e-12:
                continue
            out2 = np.array([e[1], -e[0]]) / ln  # outward for a counter-clockwise hull
            m = out2[0] * r.u + out2[1] * r.v
            t3 = (e[0] * r.u + e[1] * r.v) / ln
            e0 = r.centroid + a2[0] * r.u + a2[1] * r.v
            e1 = r.centroid + b2[0] * r.u + b2[1] * r.v
            if flare < np.radians(89.5):
                wide.append(side_plane(t3, m, e0, flare))
            tilt = flare
            if has_cam and not _edge_framed(r, bvh, e0, e1, m, camera, extent):
                # nothing hides this side of the mirror: only keep what the reference view can see
                # through the window (plus a margin for nearby view points)
                mid = 0.5 * (e0 + e1)
                bv = mid - camera
                bv -= (bv @ t3) * t3
                view_tilt = np.arctan2(bv @ m, -(bv @ n)) + margin
                tilt = min(tilt, max(view_tilt, 0.0))
            if tilt < np.radians(89.5):
                planes.append(side_plane(t3, m, e0, tilt))
    r.volume = planes
    # what gets cleared behind the mirror: anything there is only visible through the window,
    # so the generous flare volume is used regardless of the reference view
    r.removal_volume = wide
    env = scene.environment
    sun = any(l.kind == "SUN" for l in scene.lights)
    r.dome = bool(luminance(env.average()) > 0 or sun or scene.lights)


def _edge_framed(r: Reflector, bvh, e0, e1, m, viewer, extent):
    """Is the plane just outside this window edge hidden behind opaque geometry from the viewer?"""
    k = 7
    t = (np.arange(k) + 0.5) / k
    pts = e0[None] + (e1 - e0)[None] * t[:, None]
    off = max(0.02 * r.radius, 4 * r.dist_tol, 1e-4 * extent)
    q = pts + m[None] * off
    d = q - viewer[None]
    h = bvh.intersect(np.broadcast_to(viewer, d.shape).copy(), d, tmax=np.full(k, 1.0 + 1e-3))
    blocked = h.hit & ~np.isin(h.tri, r.tris)
    return blocked.mean() >= 0.6


# ---------------------------------------------------------------------------
# output geometry


class ChainTable:
    def __init__(self):
        self.chains = []
        self._ids = {}

    def get(self, chain):
        chain = tuple(int(c) for c in chain)
        if chain not in self._ids:
            self._ids[chain] = len(self.chains)
            self.chains.append(chain)
        return self._ids[chain]


@dataclass
class Dome:
    center: np.ndarray
    pole: np.ndarray  # unit vector from the centre to the dome's deepest point
    u: np.ndarray


@dataclass
class OutputGeometry:
    main: TriSet  # tags: kind, chain, dome
    aux: TriSet  # tags: kind, refl
    chains: ChainTable
    domes: list
    notes: list


def _tag(ts, **kw):
    for k, v in kw.items():
        ts = ts.with_tag(k, v)
    return ts


def _dome_tris(r: Reflector, radius, segments):
    n = r.plane.normal
    pole = -n
    u, v = r.u, r.v
    rings = max(segments // 4, 3)
    seg = max(segments, 8)
    th = np.linspace(0, np.pi / 2, rings + 1)  # 0 at the horizon ring... pole at pi/2
    ph = np.linspace(0, 2 * np.pi, seg + 1)
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    # elevation below the plane: 0 = horizon, pi/2 = pole
    pts = r.centroid + radius * (
        np.cos(T)[..., None] * (np.cos(Ph)[..., None] * u + np.sin(Ph)[..., None] * v) + np.sin(T)[..., None] * pole
    )
    a = pts[:-1, :-1].reshape(-1, 3)
    b = pts[:-1, 1:].reshape(-1, 3)
    c = pts[1:, 1:].reshape(-1, 3)
    d = pts[1:, :-1].reshape(-1, 3)
    P = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    # faces must look towards the centre (inside of the dome)
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    to_c = r.centroid - P.mean(axis=1)
    flip = np.einsum("ij,ij->i", fn, to_c) < 0
    P[flip] = P[flip][:, [0, 2, 1]]
    ts = TriSet.from_triangles(P)
    ts = ts.drop_degenerate()
    ts.src[:] = -1
    return ts


def build_output_geometry(scene: BakeScene, plan: ReflectionPlan, settings) -> OutputGeometry:
    T = scene.triangle_count
    notes = []
    base = TriSet.from_triangles(scene.positions)
    base = _tag(base, kind=np.int64(KIND_MAIN), chain=np.int64(-1), dome=np.int64(-1), refl=np.int64(-1))
    out_flag = np.array([o.output for o in scene.objects] + [True])[np.minimum(scene.object_ids, len(scene.objects))]
    aux_parts = [base.subset(~out_flag)]
    main = base.subset(out_flag)
    refl_of_tri = np.full(T, -1, dtype=np.int64)
    for r in plan.reflectors:
        refl_of_tri[r.tris] = r.index
    rt = refl_of_tri[main.src]
    aux_parts.append(_tag(main.subset(rt >= 0), refl=rt[rt >= 0]))
    main = main.subset(rt < 0)
    for r in plan.reflectors:
        if r.remove:
            inside, outside = clip_to_halfspaces(main, r.removal_volume)
            aux_parts.append(inside)
            main = outside
    aux = TriSet.concat(aux_parts)
    aux = aux.with_tag("kind", KIND_AUX)

    chains = ChainTable()
    domes = []
    extent = scene.extent()
    all_pts = scene.positions.reshape(-1, 3)
    reflectors = plan.reflectors
    reflector_pieces = {r.index: aux.subset(aux.tags["refl"] == r.index) for r in reflectors}
    memo = {}
    budget = settings.triangle_budget

    def world(a, depth):
        key = (a, depth)
        if key in memo:
            return memo[key]
        r = reflectors[a]
        parts = [main]
        for b in reflectors:
            if b.index == a:
                continue
            if depth > 1:
                parts.append(world(b.index, depth - 1))
            else:
                op = reflector_pieces[b.index]
                parts.append(_tag(op, kind=np.int64(KIND_MAIN), chain=np.int64(-1), dome=np.int64(-1)))
        src = TriSet.concat(parts)
        front, _ = clip_to_halfspaces(src, r.front_volume)
        m = mirror(front, r.plane)
        old_chain = m.tags["chain"]
        new_chain = np.empty(len(m), dtype=np.int64)
        for oc in np.unique(old_chain):
            tail = () if oc < 0 else chains.chains[oc]
            new_chain[old_chain == oc] = chains.get((a,) + tail)
        kind = np.where(m.tags["kind"] == KIND_DOME, KIND_DOME, KIND_COPY)
        dome_ids = m.tags["dome"].copy()
        for od in np.unique(dome_ids[dome_ids >= 0]):
            dm = domes[od]
            domes.append(Dome(r.plane.reflect_points(dm.center), r.plane.reflect_dirs(dm.pole), r.plane.reflect_dirs(dm.u)))
            dome_ids[dome_ids == od] = len(domes) - 1
        m.tags["chain"] = new_chain
        m.tags["kind"] = kind
        m.tags["dome"] = dome_ids
        parts_out = [m]
        if r.dome:
            # mirroring keeps distances to points on the plane, so this radius encloses every
            # mirrored triangle while staying as close as possible (far clip planes!)
            radius = settings.dome_radius
            if radius <= 0:
                radius = 1.05 * float(np.linalg.norm(all_pts - r.centroid, axis=1).max()) + 1e-3 * extent
            dt = _dome_tris(r, radius, settings.dome_segments)
            if not r.remove:
                # a mirror with real geometry behind it: keep the dome inside the reflection volume.
                # (water-like reflectors keep the full dome so grazing views still see the sky)
                dt, _ = clip_to_halfspaces(dt, r.volume[2:])
            domes.append(Dome(r.centroid.copy(), -r.plane.normal, r.u.copy()))
            dt = _tag(dt, kind=np.int64(KIND_DOME), chain=np.int64(chains.get((a,))), dome=np.int64(len(domes) - 1), refl=np.int64(-1))
            parts_out.append(dt)
        res = TriSet.concat(parts_out)
        memo[key] = res
        return res

    copies = []
    total = len(main)
    depth = max(int(settings.reflection_depth), 1)
    # keep within the triangle budget by lowering the recursion depth
    while depth > 1 and len(reflectors) > 1 and total + len(main) * len(reflectors) ** depth > budget:
        depth -= 1
        notes.append(f"reflection depth lowered to {depth} to respect the triangle budget")
    for r in reflectors:
        w = world(r.index, depth)
        total += len(w)
        if total > budget:
            notes.append(f"triangle budget exceeded; reflections of reflector {r.index} and later skipped")
            break
        copies.append(w)
    main_all = TriSet.concat([main] + copies)
    return OutputGeometry(main_all, aux, chains, domes, notes)


# ---------------------------------------------------------------------------
# shading of mirrored texels


def fresnel(r: Reflector, cos_i):
    cos_i = np.clip(cos_i, 0, 1)[:, None]
    cfg = r.config
    f0 = r.f0[None, :] if cfg.strength < 0 else np.full((1, 3), cfg.strength)
    if not cfg.fresnel:
        return np.broadcast_to(f0, (len(cos_i), 3)).copy()
    # roughness damps the grazing peak (rough surfaces never become perfect mirrors)
    peak = 1.0 - 0.9 * min(r.roughness, 1.0)
    return f0 + (peak - f0) * np.clip(1 - cos_i, 0, 1) ** 5 * (peak > f0)


def cone_directions(aim, sigma, k, rng, normal=None):
    """``k`` directions per row around ``aim`` (m,3) with gaussian angular spread ``sigma`` (m,)."""
    m = len(aim)
    t, b = orthonormal_frames(aim)
    i = np.arange(k) + 0.5
    rad = np.sqrt(-2 * np.log(1 - 0.985 * i / k))
    ang = i * 2.399963229728653
    rot = rng.random(m) * 2 * np.pi
    a = ang[None, :] + rot[:, None]
    r = rad[None, :] * sigma[:, None]
    tr = np.tan(np.minimum(r, 1.4))
    d = aim[:, None, :] + tr[..., None] * (np.cos(a)[..., None] * t[:, None, :] + np.sin(a)[..., None] * b[:, None, :])
    d = normalize(d)
    if normal is not None:
        dn = np.einsum("mkj,mj->mk", d, normal)
        d = d - 2 * np.minimum(dn, 0)[..., None] * normal[:, None, :]
    return d


class LightGlints:
    """Analytic, roughness-blurred reflection of light sources (sun disk, lamps)."""

    def __init__(self, scene: BakeScene, bvh, eps):
        self.lights = scene.lights
        self.bvh = bvh
        self.eps = eps

    def __call__(self, X, w, sigma, max_dist=None, normal=None):
        n = len(X)
        out = np.zeros((n, 3))
        for light in self.lights:
            col = np.asarray(light.color, dtype=np.float64)
            if light.kind == "SUN":
                s = -normalize(np.asarray(light.direction, dtype=np.float64)[None])[0]
                s = np.broadcast_to(s, (n, 3))
                E = np.full(n, light.power)
                ang = np.full(n, max(light.angle * 0.5, 1e-4))
                dist = np.full(n, np.inf)
            else:
                to = np.asarray(light.position, dtype=np.float64) - X
                dist = np.linalg.norm(to, axis=1)
                s = to / np.maximum(dist, 1e-12)[:, None]
                if light.kind == "AREA":
                    nl = normalize(np.asarray(light.direction, dtype=np.float64)[None])[0]
                    su, sv = light.size
                    area = su * sv * (np.pi / 4 if light.shape in ("DISK", "ELLIPSE") else 1.0)
                    cos_l = np.maximum(-(s @ nl), 0)
                    E = light.power / (np.pi * max(area, 1e-12)) * area * cos_l / np.maximum(dist * dist, 1e-12)
                    ang = np.sqrt(area / np.pi) / np.maximum(dist, 1e-12)
                else:
                    E = light.power / (4 * np.pi) / np.maximum(dist * dist, 1e-12)
                    if light.kind == "SPOT":
                        axis = normalize(np.asarray(light.direction, dtype=np.float64)[None])[0]
                        cos_a = -(s @ axis)
                        cos_half = np.cos(light.spot_size * 0.5)
                        smooth = max((1 - cos_half) * light.spot_blend, 1e-6)
                        tt = np.clip((cos_a - cos_half) / smooth, 0, 1)
                        E = E * tt * tt * (3 - 2 * tt)
                    ang = np.maximum(light.radius, 1e-3) / np.maximum(dist, 1e-12)
            cos_t = np.clip(np.einsum("ij,ij->i", w, s), -1, 1)
            theta = np.arccos(cos_t)
            s2 = sigma * sigma + 0.5 * ang * ang
            g = np.exp(-theta * theta / (2 * s2)) / (2 * np.pi * s2)
            sig = (theta < 4 * np.sqrt(s2)) & (E > 0)
            if max_dist is not None:
                sig &= dist < max_dist
            if normal is not None:
                sig &= np.einsum("ij,ij->i", s, normal) > 0
            idx = np.nonzero(sig)[0]
            if not len(idx):
                continue
            org = X[idx] + s[idx] * self.eps * 4
            tmax = np.where(np.isfinite(dist[idx]), dist[idx] * (1 - 1e-4), np.inf)
            vis = ~self.bvh.occluded(org, np.ascontiguousarray(s[idx]), tmax)
            idx = idx[vis]
            out[idx] += col[None, :] * (E[idx] * g[idx])[:, None]
        return out


class ReflectionShader:
    def __init__(self, scene, plan, bvh, lookup_full, lookup_diffuse, eps, samples, seed=0):
        self.scene = scene
        self.plan = plan
        self.bvh = bvh
        self.full = lookup_full
        self.diffuse = lookup_diffuse
        self.env = scene.environment
        self.eps = eps
        self.samples = max(int(samples), 1)
        self.rng = np.random.default_rng(seed + 101)
        self.glints = LightGlints(scene, bvh, eps)

    def sky(self, dirs):
        return self.env.eval(dirs).astype(np.float64)

    def gather(self, X, aim, sigma, tmin=None, normal=None, chunk=1 << 15):
        """Average radiance along a cone of rays: baked surfaces where rays hit, the sky elsewhere."""
        m = len(X)
        out = np.zeros((m, 3))
        k = self.samples
        for s in range(0, m, chunk):
            e = min(m, s + chunk)
            dirs = cone_directions(aim[s:e], sigma[s:e], k, self.rng, None if normal is None else normal[s:e])
            o = np.repeat(X[s:e], k, axis=0)
            d = dirs.reshape(-1, 3)
            tm = None if tmin is None else np.repeat(tmin[s:e], k)
            h = self.bvh.intersect(o, d, tmin=tm)
            rad = np.zeros((len(o), 3))
            hit = h.hit
            if hit.any():
                rad[hit] = self.full(h.tri[hit], h.bary()[hit])
            if (~hit).any():
                rad[~hit] = self.sky(d[~hit])
            out[s:e] = rad.reshape(e - s, k, 3).mean(axis=1)
        return out

    def base_color(self, r: Reflector, X):
        n = r.plane.normal
        k = len(X)
        lift = r.dist_tol * 2 + 4 * self.eps
        org = X + n * lift
        h = self.bvh.intersect(org, np.broadcast_to(-n, (k, 3)).copy(), tmax=np.full(k, lift * 2))
        on = h.hit & np.isin(h.tri, r.tris)
        out = np.broadcast_to(r.mean_diffuse, (k, 3)).astype(np.float64).copy()
        if on.any():
            out[on] = self.diffuse(h.tri[on], h.bary()[on])
        return out

    def shade(self, Q, chain_ids, kinds, src_tri, src_bary, texel_size, chains):
        """Radiance for mirrored texels (KIND_COPY / KIND_DOME) located at output points Q."""
        n = len(Q)
        out = np.zeros((n, 3))
        refl = self.plan.reflectors
        for cid in np.unique(chain_ids):
            sel = np.nonzero(chain_ids == cid)[0]
            chain = chains.chains[cid]
            Qs = Q[sel]
            V = refl[chain[0]].viewer
            cur_o = np.broadcast_to(V, Qs.shape).astype(np.float64).copy()
            cur_w = normalize(Qs - cur_o)
            Qc = Qs.copy()
            Fs, bases = [], []
            sig2 = np.zeros(len(sel))
            for a in chain:
                r = refl[a]
                nrm = r.plane.normal
                denom = cur_w @ nrm
                t = (r.plane.d - cur_o @ nrm) / np.where(np.abs(denom) > 1e-12, denom, -1e-12)
                X = cur_o + cur_w * t[:, None]
                bad = (denom > -1e-9) | (t <= 0) | ~np.isfinite(t)
                if bad.any():
                    proj = Qc[bad] - (Qc[bad] @ nrm - r.plane.d)[:, None] * nrm
                    X[bad] = proj
                Fs.append(fresnel(r, np.abs(denom)))
                bases.append(self.base_color(r, X))
                sig2 += r.sigma**2
                cur_w = r.plane.reflect_dirs(cur_w)
                Qc = r.plane.reflect_points(Qc)
                cur_o = X
            sigma = np.sqrt(sig2)
            # leave the innermost reflector's surface so gather rays cannot hit it again
            last = refl[chain[-1]]
            Xk = cur_o + last.plane.normal * (last.dist_tol * 2 + 4 * self.eps)
            kind = kinds[sel]
            content = np.zeros((len(sel), 3))
            copy = kind == KIND_COPY
            if copy.any():
                ci = np.nonzero(copy)[0]
                to = Qc[ci] - Xk[ci]
                dist = np.linalg.norm(to, axis=1)
                aim = to / np.maximum(dist, 1e-12)[:, None]
                exact = sigma[ci] * dist < 0.5 * texel_size[sel][ci]
                direct = self.full(src_tri[sel][ci], src_bary[sel][ci])
                content[ci] = direct
                bl = ~exact
                if bl.any():
                    bi = ci[bl]
                    slack = np.clip(np.maximum(4 * sigma[bi] * dist[bl], 3 * texel_size[sel][bi]), 0, 0.5 * dist[bl])
                    g = self.gather(Xk[bi], aim[bl], sigma[bi], tmin=np.maximum(dist[bl] - slack, 0))
                    content[bi] = g
                glint_sigma = np.maximum(sigma[ci], texel_size[sel][ci] / np.maximum(dist, 1e-9) * 1.5)
                content[ci] += self.glints(Xk[ci], aim, glint_sigma, max_dist=dist)
            dome = ~copy
            if dome.any():
                di = np.nonzero(dome)[0]
                w = cur_w[di]
                ang_texel = texel_size[sel][di]
                sg = np.maximum(sigma[di], 1e-4)
                content[di] = self.gather(Xk[di], w, sg)
                content[di] += self.glints(Xk[di], w, np.maximum(sigma[di], ang_texel * 1.5))
            col = content
            for F, b in zip(reversed(Fs), reversed(bases)):
                col = (1 - F) * b + F * col
            out[sel] = col
        return out

    def texture_specular(self, P, Ns, f0, rough, viewer):
        """View-dependent gloss for the reference view, baked into ordinary texels."""
        if viewer is not None:
            wo = normalize(viewer[None, :] - P)
            ok = np.einsum("ij,ij->i", wo, Ns) > 0.05
            wo = np.where(ok[:, None], wo, Ns)
        else:
            wo = Ns.copy()
        cos_o = np.clip(np.einsum("ij,ij->i", wo, Ns), 0, 1)
        r = 2 * cos_o[:, None] * Ns - wo
        alpha = np.maximum(rough * rough, 1e-3)
        sigma = np.minimum(0.7 * alpha, 0.8)
        org = P + Ns * self.eps * 4
        content = self.gather(org, r, sigma, normal=Ns)
        content += self.glints(org, r, np.maximum(sigma, 2e-3), normal=Ns)
        F = f0 + (1 - f0) * ((1 - cos_o) ** 5)[:, None] * (1 - 0.9 * np.clip(rough, 0, 1))[:, None]
        return F * content
