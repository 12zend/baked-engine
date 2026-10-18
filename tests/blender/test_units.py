"""Calibrate the built-in path tracer against Cycles on simple lighting setups.

Run:  blender -b --factory-startup --python tests/blender/test_units.py
"""

import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402

failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def setup(kind):
    common.clear_scene()
    for l in list(bpy.data.lights):
        bpy.data.lights.remove(l)
    common.device_cpu()
    white = common.principled("White", (0.8, 0.8, 0.8), roughness=1.0)
    white.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.0
    common.plane("Floor", 4.0, 0.0, white)
    common.world_color((0, 0, 0), 0.0)
    if kind == "sun":
        common.sun(2.0, rot=(math.radians(30), 0, 0), angle=0.0)
    elif kind == "point":
        ld = bpy.data.lights.new("P", "POINT")
        ld.energy = 200.0
        ld.shadow_soft_size = 0.0
        ob = bpy.data.objects.new("P", ld)
        ob.location = (0.5, 0, 2.0)
        bpy.context.scene.collection.objects.link(ob)
    elif kind == "spot":
        ld = bpy.data.lights.new("S", "SPOT")
        ld.energy = 300.0
        ld.shadow_soft_size = 0.0
        ld.spot_size = math.radians(60)
        ld.spot_blend = 0.3
        ob = bpy.data.objects.new("S", ld)
        ob.location = (0, 0, 2.0)
        bpy.context.scene.collection.objects.link(ob)
    elif kind == "area":
        ld = bpy.data.lights.new("A", "AREA")
        ld.energy = 300.0
        ld.size = 1.0
        ob = bpy.data.objects.new("A", ld)
        ob.location = (0, 0, 1.5)
        bpy.context.scene.collection.objects.link(ob)
    elif kind == "world":
        common.world_color((0.5, 0.6, 0.7), 1.5)
    elif kind == "emissive":
        em = bpy.data.materials.new("Lamp")
        em.use_nodes = True
        nt = em.node_tree
        nt.nodes.clear()
        e = nt.nodes.new("ShaderNodeEmission")
        e.inputs["Strength"].default_value = 5.0
        o = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(e.outputs[0], o.inputs["Surface"])
        lamp = common.plane("Lamp", 1.0, 1.2, em)
        lamp.rotation_euler = (math.pi, 0, 0)
        lamp.location = (0, 0, 2.4)


def bake_both(samples=256):
    from baked_engine.blender.cycles_backend import CyclesBackend
    from baked_engine.blender.extract import extract_scene
    from baked_engine.core.pipeline import bake
    from baked_engine.core.settings import BakeSettings

    bs, info = extract_scene(bpy.context)
    st = BakeSettings(width=64, height=64, samples=samples, reflections=False, specular_bake=False, denoise=False, max_bounces=4)
    a = bake(bs, st)
    b = bake(bs, st, backend=CyclesBackend(bpy.context, info["bpy_materials"], info["uv_name"]))
    return a, b


def main():
    common.register_addon()
    for kind in ("sun", "point", "spot", "area", "world", "emissive"):
        setup(kind)
        a, b = bake_both()
        ia = a.image[..., :3]
        ib = b.image[..., :3]
        floor = ia.mean(axis=2) > 1e-4
        ratio = ib[floor].mean() / max(ia[floor].mean(), 1e-9)
        va, vb = ia[floor].mean(axis=1), ib[floor].mean(axis=1)
        flat = va.std() < 1e-3 * max(va.mean(), 1e-9)
        corr = 1.0 if flat else np.corrcoef(va, vb)[0, 1]
        print(f"{kind}: builtin={ia[floor].mean():.4f} cycles={ib[floor].mean():.4f} ratio={ratio:.3f} corr={corr:.3f}")
        check(abs(ratio - 1) < 0.08, f"{kind} intensity matches Cycles (ratio {ratio:.3f})")
        check(corr > 0.9, f"{kind} spatial distribution matches Cycles (corr {corr:.3f})")


try:
    main()
except Exception:
    traceback.print_exc()
    failures.append("exception")
print("RESULT:", "OK" if not failures else f"{len(failures)} FAILED")
sys.exit(1 if failures else 0)
