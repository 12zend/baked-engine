"""Helpers for tests that run inside Blender (``blender -b --factory-startup --python ...``)."""

import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import bpy  # noqa: E402


def register_addon():
    import baked_engine

    try:
        baked_engine.register()
    except ValueError:
        pass  # already registered
    return baked_engine


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)


def principled(name, color, roughness=0.5, metallic=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Metallic"].default_value = metallic
    return m


def add_mesh(name, verts, faces, mat=None, location=(0, 0, 0)):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.uv_layers.new(name="UVMap")
    ob = bpy.data.objects.new(name, me)
    ob.location = location
    bpy.context.scene.collection.objects.link(ob)
    if mat:
        me.materials.append(mat)
    return ob


def plane(name, size, z=0.0, mat=None, center=(0, 0)):
    s = size / 2
    cx, cy = center
    return add_mesh(name, [(cx - s, cy - s, z), (cx + s, cy - s, z), (cx + s, cy + s, z), (cx - s, cy + s, z)], [(0, 1, 2, 3)], mat)


def sun(strength=3.0, rot=(math.radians(40), 0, math.radians(30)), angle=0.01):
    ld = bpy.data.lights.new("Sun", "SUN")
    ld.energy = strength
    ld.angle = angle
    ob = bpy.data.objects.new("Sun", ld)
    ob.rotation_euler = rot
    bpy.context.scene.collection.objects.link(ob)
    return ob


def world_color(color, strength=1.0):
    w = bpy.data.worlds.new("World") if bpy.context.scene.world is None else bpy.context.scene.world
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (*color, 1.0)
    bg.inputs["Strength"].default_value = strength
    return w


def device_cpu():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
