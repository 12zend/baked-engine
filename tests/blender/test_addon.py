"""End-to-end add-on test inside Blender.

Run:  blender -b --factory-startup --python tests/blender/test_addon.py
"""

import math
import os
import sys
import tempfile
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


def build_lake_scene():
    common.clear_scene()
    common.device_cpu()
    sea_mat = common.principled("Sea", (0.01, 0.03, 0.04), roughness=0.05)
    rock = common.principled("Rock", (0.35, 0.33, 0.25), roughness=0.9)
    red = common.principled("Red", (0.7, 0.1, 0.08), roughness=0.5)
    sea = common.plane("Sea", 60, 0.0, sea_mat, center=(0, 10))
    sea.baked_engine.reflection_mode = "GEOMETRY"
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=12, depth=14, location=(0, 22, 5))
    hill = bpy.context.object
    hill.data.materials.append(rock)
    bpy.ops.mesh.primitive_cube_add(size=3, location=(6, 6, 1))
    cube = bpy.context.object
    cube.data.materials.append(red)
    common.sun(3.0, rot=(math.radians(50), 0, math.radians(200)))
    common.world_color((0.4, 0.55, 0.8), 1.0)
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    cam.location = (0, -15, 3)
    cam.rotation_euler = (math.radians(85), 0, 0)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return sea, hill, cube


def main():
    common.register_addon()
    build_lake_scene()
    from baked_engine.blender.extract import extract_scene

    bs, info = extract_scene(bpy.context)
    check(bs.triangle_count > 40, f"extracted {bs.triangle_count} triangles")
    check(len(bs.lights) == 1 and bs.lights[0].kind == "SUN", "sun light extracted")
    check(bs.camera is not None, "camera extracted")
    names = [o.name for o in bs.objects]
    check(any(o.reflector.mode == "GEOMETRY" for o in bs.objects), f"reflector config extracted {names}")

    out = tempfile.mkdtemp(prefix="be_test_")
    p = bpy.context.scene.baked_engine
    p.width = p.height = 256
    p.samples = 16
    p.output_dir = out
    p.export_bkd = True
    for backend in ("CYCLES", "BUILTIN"):
        p.backend = backend
        p.output_name = f"lake_{backend.lower()}"
        r = bpy.ops.baked_engine.bake()
        check(r == {"FINISHED"}, f"bake operator finished with {backend}")
        obj = bpy.data.objects.get(p.output_name)
        check(obj is not None and len(obj.data.polygons) > 0, f"result object created ({backend})")
        for ext in ("png", "glb", "obj", "bkd", "json"):
            path = os.path.join(out, f"{p.output_name}.{ext}")
            check(os.path.exists(path) and os.path.getsize(path) > 0, f"wrote {ext} ({backend})")
        check("BE_proxy" not in bpy.data.objects, "proxy cleaned up")
        check(all(not o.hide_render for o in bpy.context.scene.objects if o.type == "MESH" and not o.name.startswith("lake_")), "source objects restored")
        img = bpy.data.images.get(p.output_name)
        px = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        check(np.isfinite(px).all() and px.reshape(-1, 4)[:, :3].mean() > 0.01, f"texture has content ({backend}) mean={px.reshape(-1,4)[:, :3].mean():.3f}")


try:
    main()
except Exception:
    traceback.print_exc()
    failures.append("exception")
print("RESULT:", "OK" if not failures else f"{len(failures)} FAILED")
sys.exit(1 if failures else 0)
