"""Turn a BakeResult into Blender data and files."""

from __future__ import annotations

import os

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


def create_result_object(context, res, name):
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


def _scene_encoded_rgba(context, img, directory, name):
    """Encode through the scene's view transform by saving with Blender's colour management."""
    path = os.path.join(directory, f"{name}.png")
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


def export_files(context, res, props, img=None):
    directory = bpy.path.abspath(props.output_dir)
    name = bpy.path.clean_name(props.output_name) or "baked"
    os.makedirs(directory, exist_ok=True)
    formats = [f for f, on in (("glb", props.export_glb), ("obj", props.export_obj), ("bkd", props.export_bkd)) if on]
    rgba = None
    if props.view_transform == "SCENE":
        if img is None:
            img = result_image(name + "_tmp", res)
        rgba = _scene_encoded_rgba(context, img, directory, name)
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
