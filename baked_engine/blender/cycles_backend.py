"""Lighting backend that lets Cycles path trace the bake points.

The lit triangles of every atlas are written into one temporary proxy mesh
whose active UV map is the atlas, carrying copies of the original materials
(so every Blender shader feature is respected). The original meshes are
hidden from rendering during the bake; the proxy is the same surface, so it
provides identical occlusion and bounce light. Cycles bakes diffuse lighting
(direct + indirect, without albedo) twice with different seeds so the
denoiser gets a per-texel variance estimate.
"""

from __future__ import annotations

import bpy
import numpy as np

from ..core.geometry import interpolate

ATLAS_UV = "BE_Atlas"
_BAKE_IMAGE = "BE_bake_canvas"


class CyclesBackend:
    name = "CYCLES"

    def __init__(self, context, bpy_materials, uv_name, run_on_main=None):
        # ``context`` is accepted for API symmetry but never stored: an operator's context is only
        # valid during the call, and the bake may run later from a modal handler.
        self.bpy_materials = bpy_materials
        self.uv_name = uv_name or "UVMap"
        self.run_on_main = run_on_main or (lambda fn: fn())

    def bake(self, scene, targets, settings, tracer, progress):
        return self.run_on_main(lambda: self._bake(scene, targets, settings, progress))

    # ------------------------------------------------------------------
    def _bake(self, scene, targets, settings, progress):
        ctx = bpy.context
        bscene = ctx.scene
        CW = max(t.width for t in targets)
        CH = sum(t.height for t in targets)
        yoff = np.cumsum([0] + [t.height for t in targets])[:-1]

        P = np.concatenate([t.ts.P for t in targets]).astype(np.float32)
        atlas_uv = np.concatenate(
            [(t.tri_px + np.array([0.0, yo])) / np.array([CW, CH]) for t, yo in zip(targets, yoff)]
        ).astype(np.float32)
        mat_uv = np.concatenate([interpolate(scene.uvs, t.ts) for t in targets]).astype(np.float32)
        nrm = np.concatenate([interpolate(scene.normals, t.ts) for t in targets])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-12)
        mids = np.concatenate([scene.material_ids[t.ts.src] for t in targets])

        state = _SceneState(ctx)
        image = None
        mesh = None
        obj = None
        mat_copies = []
        try:
            image = bpy.data.images.new(_BAKE_IMAGE, CW, CH, alpha=True, float_buffer=True)
            mesh = _build_mesh(P, atlas_uv, mat_uv, nrm, mids, self.uv_name)
            used = np.unique(mids)
            for i in range(int(mids.max()) + 1 if len(mids) else 0):
                ref = self.bpy_materials[i] if i < len(self.bpy_materials) else None
                src = bpy.data.materials.get(ref) if isinstance(ref, str) else ref
                m = _bake_material(src, image) if i in used else None
                mat_copies.append(m)
                mesh.materials.append(m)
            obj = bpy.data.objects.new("BE_proxy", mesh)
            bscene.collection.objects.link(obj)
            state.isolate(obj)
            state.configure(settings)
            halves = []
            spp = max(1, settings.samples // 2)
            for k in range(2):
                bscene.cycles.samples = spp
                bscene.cycles.seed = int(settings.seed) * 2 + k
                bpy.ops.object.bake(
                    type="DIFFUSE",
                    pass_filter={"DIRECT", "INDIRECT"},
                    margin=2,
                    margin_type="EXTEND",
                    use_clear=True,
                    uv_layer=ATLAS_UV,
                    target="IMAGE_TEXTURES",
                    save_mode="INTERNAL",
                )
                buf = np.empty(CW * CH * 4, dtype=np.float32)
                image.pixels.foreach_get(buf)
                halves.append(buf.reshape(CH, CW, 4).astype(np.float64))
                progress((k + 1) / 2)
            a, b = halves
            out = []
            for t, yo in zip(targets, yoff):
                tx = t.texels
                va = a[tx.y + yo, tx.x]
                vb = b[tx.y + yo, tx.x]
                mean = 0.5 * (va[:, :3] + vb[:, :3])
                d = (va[:, :3] - vb[:, :3]).mean(axis=1)
                # Cycles writes alpha 1 where it baked (incl. its margin); the rest is filled later
                covered = (va[:, 3] > 0.5).astype(np.float64)
                out.append((mean, 0.25 * d * d, covered))
            return out
        finally:
            state.restore()
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None:
                bpy.data.meshes.remove(mesh)
            for m in mat_copies:
                if m is not None:
                    bpy.data.materials.remove(m)
            if image is not None:
                bpy.data.images.remove(image)


def _build_mesh(P, atlas_uv, mat_uv, nrm, mids, uv_name):
    nt = len(P)
    me = bpy.data.meshes.new("BE_proxy")
    me.vertices.add(nt * 3)
    me.vertices.foreach_set("co", P.reshape(-1))
    me.loops.add(nt * 3)
    me.loops.foreach_set("vertex_index", np.arange(nt * 3, dtype=np.int32))
    me.polygons.add(nt)
    me.polygons.foreach_set("loop_start", np.arange(0, nt * 3, 3, dtype=np.int32))
    try:
        me.polygons.foreach_set("loop_total", np.full(nt, 3, dtype=np.int32))
    except (AttributeError, TypeError, RuntimeError):
        pass  # derived from loop_start in newer Blender versions
    me.polygons.foreach_set("material_index", mids.astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(nt, dtype=bool))
    me.update()
    src_layer = me.uv_layers.new(name=uv_name)
    src_layer.data.foreach_set("uv", mat_uv.reshape(-1))
    atlas = me.uv_layers.new(name=ATLAS_UV)
    atlas.data.foreach_set("uv", atlas_uv.reshape(-1))
    src_layer.active_render = True
    me.uv_layers.active = atlas
    if hasattr(me, "use_auto_smooth"):
        me.use_auto_smooth = True
    me.normals_split_custom_set(nrm.reshape(-1, 3).tolist())
    me.update()
    return me


def _bake_material(src, image):
    if src is None:
        m = bpy.data.materials.new("BE_bake_default")
        m.use_nodes = True
    else:
        m = src.copy()
        m.name = "BE_bake_" + src.name
        if not m.use_nodes:
            col = tuple(src.diffuse_color)
            m.use_nodes = True
            bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = col
    nt = m.node_tree
    for n in nt.nodes:
        n.select = False
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = image
    node.select = True
    nt.nodes.active = node
    return m


class _SceneState:
    """Temporarily hide the original geometry and switch to Cycles; restores everything."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.scene = ctx.scene
        self.hidden = []
        self.selection = []
        self.active = None
        self.render = {}

    def isolate(self, proxy):
        vl = self.ctx.view_layer
        self.active = vl.objects.active
        self.selection = [o for o in self.ctx.selected_objects]
        for o in self.scene.objects:
            if o is proxy or o.type not in ("MESH", "CURVE", "SURFACE", "META", "FONT"):
                continue
            if not o.hide_render:
                self.hidden.append(o)
                o.hide_render = True
        for o in self.selection:
            o.select_set(False)
        proxy.hide_render = False
        proxy.select_set(True)
        vl.objects.active = proxy

    def configure(self, settings):
        sc = self.scene
        self.render = dict(
            engine=sc.render.engine,
            samples=sc.cycles.samples,
            seed=sc.cycles.seed,
            margin=sc.render.bake.margin,
            use_clear=sc.render.bake.use_clear,
            max_bounces=sc.cycles.max_bounces,
            clamp_indirect=sc.cycles.sample_clamp_indirect,
        )
        sc.render.engine = "CYCLES"
        sc.cycles.max_bounces = max(int(settings.max_bounces), 0)
        sc.cycles.sample_clamp_indirect = float(settings.clamp_indirect)

    def restore(self):
        sc = self.scene
        for o in self.hidden:
            try:
                o.hide_render = False
            except ReferenceError:
                pass
        if self.render:
            sc.render.engine = self.render["engine"]
            sc.cycles.samples = self.render["samples"]
            sc.cycles.seed = self.render["seed"]
            sc.render.bake.margin = self.render["margin"]
            sc.render.bake.use_clear = self.render["use_clear"]
            sc.cycles.max_bounces = self.render["max_bounces"]
            sc.cycles.sample_clamp_indirect = self.render["clamp_indirect"]
        vl = self.ctx.view_layer
        for o in self.selection:
            try:
                o.select_set(True)
            except ReferenceError:
                pass
        if self.active is not None:
            try:
                vl.objects.active = self.active
            except ReferenceError:
                pass
