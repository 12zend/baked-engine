"""Turn a BakeResult into Blender data and files."""

from __future__ import annotations

import os
import tempfile

import bpy
import numpy as np

from ..core.export import export_result

COLLECTION = "Baked Engine"


def result_image(name, res):
    h, w = res.image.shape[:2]
    img = bpy.data.images.get(name)
    if img is not None and (img.size[0] != w or img.size[1] != h or not img.is_float):
        bpy.data.images.remove(img)
        img = None
    if img is None:
        img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    img.pixels.foreach_set(np.ascontiguousarray(res.image, dtype=np.float32).reshape(-1))
    img.update()
    img.pack()
    return img


def display_image(name, rgba):
    """8-bit sRGB image holding the display colours (the same pixels as ``name.png``).

    ``rgba`` is (H, W, 4) uint8 with row 0 at the bottom, like Blender's pixel order. Kept as a byte
    image so that ``Image.pixels`` returns exactly these values to exporters.
    """
    h, w = rgba.shape[:2]
    iname = f"{name}_display"
    img = bpy.data.images.get(iname)
    if img is not None and (img.size[0] != w or img.size[1] != h or img.is_float):
        bpy.data.images.remove(img)
        img = None
    if img is None:
        img = bpy.data.images.new(iname, w, h, alpha=True, float_buffer=False)
    try:
        img.colorspace_settings.name = "sRGB"
    except TypeError:
        pass
    img.pixels.foreach_set((np.ascontiguousarray(rgba, dtype=np.float32) / 255.0).reshape(-1))
    img.update()
    img.pack()
    return img


def add_export_shader(mat, img):
    """Add a Principled BSDF whose Base Color is the display texture.

    The preview surface stays the unlit Emission/Transparent mix (driven by the linear HDR image).
    This extra node is not connected to the output; it only tells exporters that read the Principled
    BSDF's Base Color (glTF, custom .txt exporters, ...) which texture and colours the mesh has.
    """
    nt_ = mat.node_tree
    for n in [n for n in nt_.nodes if n.get("baked_engine_export")]:
        nt_.nodes.remove(n)
    tex = nt_.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"
    tex.label = "Display colours (for exporters)"
    tex["baked_engine_export"] = True
    bsdf = nt_.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.label = "Export: Base Color = baked texture"
    bsdf["baked_engine_export"] = True
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)   # texture x white
    nt_.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt_.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    for key, val in (("Roughness", 1.0), ("Metallic", 0.0), ("Specular IOR Level", 0.0), ("Specular", 0.0), ("Emission Strength", 0.0)):
        sock = bsdf.inputs.get(key)
        if sock is not None and not sock.is_linked:
            sock.default_value = val
    tex.location = (-500, -350)
    bsdf.location = (-200, -350)
    return bsdf


def create_result_object(context, res, name, rgba=None):
    """Mesh with one UV map and one unlit material showing the baked texture."""
    img = result_image(name, res)
    old = bpy.data.objects.get(name)
    if old is not None:
        old_mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_mesh is not None and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    me = bpy.data.meshes.new(name)
    nv = len(res.positions)
    nt = len(res.indices)
    me.vertices.add(nv)
    me.vertices.foreach_set("co", res.positions.reshape(-1).astype(np.float32))
    me.loops.add(nt * 3)
    me.loops.foreach_set("vertex_index", res.indices.reshape(-1).astype(np.int32))
    me.polygons.add(nt)
    me.polygons.foreach_set("loop_start", np.arange(0, nt * 3, 3, dtype=np.int32))
    try:
        me.polygons.foreach_set("loop_total", np.full(nt, 3, dtype=np.int32))
    except (AttributeError, TypeError, RuntimeError):
        pass
    me.update()
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", res.uvs[res.indices.reshape(-1)].reshape(-1).astype(np.float32))
    me.update()
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt_ = mat.node_tree
    nt_.nodes.clear()
    out = nt_.nodes.new("ShaderNodeOutputMaterial")
    tex = nt_.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"
    emit = nt_.nodes.new("ShaderNodeEmission")
    transp = nt_.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt_.nodes.new("ShaderNodeMixShader")
    nt_.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt_.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt_.links.new(transp.outputs[0], mix.inputs[1])
    nt_.links.new(emit.outputs[0], mix.inputs[2])
    nt_.links.new(mix.outputs[0], out.inputs["Surface"])
    tex.location = (-500, 0)
    emit.location = (-200, -100)
    transp.location = (-200, 100)
    mix.location = (0, 0)
    out.location = (200, 0)
    if hasattr(mat, "blend_method"):
        try:
            mat.blend_method = "CLIP"
        except TypeError:
            pass
    if rgba is not None:
        add_export_shader(mat, display_image(name, rgba))
    me.materials.append(mat)
    obj = bpy.data.objects.new(name, me)
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
        context.scene.collection.children.link(coll)
    elif coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(coll)
    coll.objects.link(obj)
    obj["baked_engine_result"] = True
    obj["baked_engine_stats"] = str({k: v for k, v in res.stats.items() if k != "timings"})
    return obj, img


def _scene_encoded_rgba(context, img, path):
    """Encode through the scene's view transform by saving with Blender's colour management."""
    sc = context.scene
    fmt = sc.render.image_settings
    old = (fmt.file_format, fmt.color_mode, fmt.color_depth)
    try:
        fmt.file_format = "PNG"
        fmt.color_mode = "RGBA"
        fmt.color_depth = "8"
        img.save_render(path, scene=sc)
    finally:
        try:
            fmt.file_format, fmt.color_mode, fmt.color_depth = old
        except TypeError:
            pass
    loaded = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = loaded.size
        px = np.empty(w * h * 4, dtype=np.float32)
        loaded.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(loaded)
    return np.round(px.reshape(h, w, 4) * 255).astype(np.uint8)


def resolve_output_dir(path):
    """Absolute, writable output folder.

    ``//`` paths are relative to the .blend file; while the file is unsaved (and for any path that
    stays relative) they go under ``~/BakedEngine`` instead of Blender's working directory, which
    is ``/`` (read-only) when Blender is started from Finder.
    """
    raw = path or "//baked/"
    home_base = os.path.join(os.path.expanduser("~"), "BakedEngine")
    if raw.startswith("//") and not bpy.data.filepath:
        out = os.path.join(home_base, raw[2:])
    else:
        out = bpy.path.abspath(raw)
        if not os.path.isabs(out):
            out = os.path.join(home_base, out)
    return os.path.normpath(out)


def display_rgba(context, res, props, img=None):
    """Display-referred RGBA8 (row 0 = bottom): the colours written to ``name.png``."""
    if props.view_transform != "SCENE":
        return res.encoded(props.view_transform, props.exposure)
    tmp = None
    if img is None:
        img = tmp = result_image("BE_display_tmp", res)
    try:
        with tempfile.TemporaryDirectory(prefix="baked_engine_") as d:
            return _scene_encoded_rgba(context, img, os.path.join(d, "display.png"))
    finally:
        if tmp is not None:
            bpy.data.images.remove(tmp)


def export_files(context, res, props, img=None, rgba=None):
    directory = resolve_output_dir(props.output_dir)
    name = bpy.path.clean_name(props.output_name) or "baked"
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"cannot write to output folder '{directory}' ({exc.strerror}); choose another Output folder") from exc
    formats = [f for f, on in (("glb", props.export_glb), ("obj", props.export_obj), ("bkd", props.export_bkd)) if on]
    if rgba is None and props.view_transform == "SCENE":
        rgba = display_rgba(context, res, props, img)
    return export_result(
        res,
        directory,
        name,
        formats=formats,
        view=props.view_transform if props.view_transform != "SCENE" else "AGX",
        exposure=props.exposure,
        up=props.up_axis,
        hdr=props.export_hdr,
        rgba=rgba,
    )
