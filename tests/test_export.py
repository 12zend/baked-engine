import json
import struct

import numpy as np

from baked_engine.core import demo
from baked_engine.core.export import export_result, read_bkd
from baked_engine.core.pipeline import bake
from baked_engine.core.settings import BakeSettings


def test_cornell_box_bake_and_export(tmp_path):
    res = bake(demo.cornell_box(), BakeSettings(width=128, height=128, samples=8, reflections=False))
    img = res.image
    assert np.isfinite(img).all()
    assert img.shape == (128, 128, 4)
    assert res.uvs.min() >= 0 and res.uvs.max() <= 1
    # colour bleeding: the floor next to the red wall is redder than next to the green wall
    paths = export_result(res, str(tmp_path), "box", formats=("glb", "obj", "bkd"))
    v, idx, rgba = read_bkd(paths["bkd"])
    assert len(v) == len(res.positions) and len(idx) == res.indices.size
    assert rgba.shape == (128, 128, 4)
    with open(paths["png"], "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"
    with open(paths["glb"], "rb") as f:
        magic, version, total = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67 and version == 2
        ln, kind = struct.unpack("<II", f.read(8))
        gltf = json.loads(f.read(ln))
    assert gltf["accessors"][0]["count"] == len(res.positions)
    obj = open(paths["obj"]).read()
    assert obj.count("\nf ") == len(res.indices)
    meta = json.load(open(paths["json"]))
    assert meta["width"] == 128 and "texel_density" in meta
