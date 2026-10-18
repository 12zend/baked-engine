"""World conversion matches Cycles.

Run:  blender -b --factory-startup --python tests/blender/test_world.py
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


def direction_world():
    w = bpy.data.worlds.new("Dir")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = next(n for n in nt.nodes if n.type == "BACKGROUND")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vm = nt.nodes.new("ShaderNodeVectorMath")
    vm.operation = "MULTIPLY_ADD"
    vm.inputs[1].default_value = (0.5, 0.5, 0.5)
    vm.inputs[2].default_value = (0.5, 0.5, 0.5)
    nt.links.new(tc.outputs["Generated"], vm.inputs[0])
    nt.links.new(vm.outputs[0], bg.inputs["Color"])


def env_texture_world(rot_z):
    h, w = 32, 64
    img = bpy.data.images.new("EnvTest", w, h, float_buffer=True)
    yy, xx = np.mgrid[0:h, 0:w]
    px = np.zeros((h, w, 4), np.float32)
    px[..., 0] = xx / w
    px[..., 1] = yy / h
    px[..., 2] = ((xx // 8 + yy // 8) % 2) * 0.5
    px[..., 3] = 1
    img.pixels.foreach_set(px.ravel())
    wd = bpy.data.worlds.new("Env")
    bpy.context.scene.world = wd
    wd.use_nodes = True
    nt = wd.node_tree
    bg = next(n for n in nt.nodes if n.type == "BACKGROUND")
    et = nt.nodes.new("ShaderNodeTexEnvironment")
    et.image = img
    et.interpolation = "Closest"
    nt.links.new(et.outputs["Color"], bg.inputs["Color"])
    if rot_z:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value = (0, 0, rot_z)
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], et.inputs["Vector"])


def main():
    common.register_addon()
    from baked_engine.blender.extract import _TextureCache, _world
    from baked_engine.blender.world_capture import capture_world

    rng = np.random.default_rng(0)
    d = rng.normal(size=(400, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    direction_world()
    env = capture_world(bpy.context.scene, 128, 64, 1)
    err = np.abs(env.eval(d) - (d * 0.5 + 0.5)).mean()
    check(err < 0.02, f"captured world follows the direction convention (err {err:.4f})")
    for rot in (0.0, 0.7):
        env_texture_world(rot)
        parsed = _world(bpy.context.scene, _TextureCache(), set(), None)
        check(parsed.texture is not None, f"environment texture parsed directly (rot {rot})")
        cap = capture_world(bpy.context.scene, 256, 128, 1)
        a = parsed.eval(d)[:, :2]
        b = cap.eval(d)[:, :2]
        # compare where both are away from the wrap-around seam
        ok = (a[:, 0] > 0.05) & (a[:, 0] < 0.95) & (b[:, 0] > 0.05) & (b[:, 0] < 0.95)
        err = np.abs(a[ok] - b[ok]).mean()
        check(err < 0.03, f"parsed environment matches Cycles (rot {rot}, err {err:.4f})")


try:
    main()
except Exception:
    traceback.print_exc()
    failures.append("exception")
print("RESULT:", "OK" if not failures else f"{len(failures)} FAILED")
sys.exit(1 if failures else 0)
