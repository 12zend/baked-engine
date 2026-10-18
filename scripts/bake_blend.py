"""Bake a .blend file from the command line.

    blender -b scene.blend --python scripts/bake_blend.py -- --size 2048 --samples 256 --out ./baked

Every option maps to a property of the Baked Engine render panel; unspecified
options keep the values saved in the .blend file.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bpy  # noqa: E402


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, help="square texture size (or use --width/--height)")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--density", type=float, help="texels per metre (lowered automatically if it does not fit)")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--engine", choices=["CYCLES", "BUILTIN"])
    ap.add_argument("--no-reflections", action="store_true")
    ap.add_argument("--reflection-depth", type=int)
    ap.add_argument("--view", choices=["SCENE", "AGX", "ACES", "STANDARD", "NONE"])
    ap.add_argument("--formats", default=None, help="comma separated: glb,obj,bkd,hdr")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--name", help="output base name")
    ap.add_argument("--save-blend", help="also save the .blend with the preview object here")
    args = ap.parse_args(argv)

    import baked_engine

    if not hasattr(bpy.types.Scene, "baked_engine"):
        baked_engine.register()
    p = bpy.context.scene.baked_engine
    if args.size:
        p.width = p.height = args.size
    if args.width:
        p.width = args.width
    if args.height:
        p.height = args.height
    if args.density is not None:
        p.texel_density = args.density
    if args.samples:
        p.samples = args.samples
    if args.engine:
        p.backend = args.engine
    if args.no_reflections:
        p.reflections = False
    if args.reflection_depth:
        p.reflection_depth = args.reflection_depth
    if args.view:
        p.view_transform = args.view
    if args.formats is not None:
        fm = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
        p.export_glb, p.export_obj, p.export_bkd, p.export_hdr = ("glb" in fm, "obj" in fm, "bkd" in fm, "hdr" in fm)
    if args.out:
        p.output_dir = os.path.abspath(args.out)
    if args.name:
        p.output_name = args.name
    result = bpy.ops.baked_engine.bake()
    if result != {"FINISHED"}:
        sys.exit(1)
    print(p.status)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend), copy=True)


main()
