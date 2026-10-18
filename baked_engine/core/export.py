"""Writers for the baked result: PNG, Radiance HDR, OBJ+MTL, binary glTF and a raw binary format.

Raw format ``.bkd`` (little endian)::

    char[4]  magic "BKD1"
    uint32   vertex_count
    uint32   index_count
    uint32   image_width
    uint32   image_height
    float32  vertices[vertex_count][5]   # x y z u v
    uint32   indices[index_count]        # triangles, counter-clockwise
    uint8    rgba[height][width][4]      # top row first

Positions use the axis convention chosen at export (``up='Z'`` keeps Blender's).
"""

from __future__ import annotations

import json
import os
import struct
import zlib

import numpy as np


def convert_axes(positions, up="Y"):
    p = np.asarray(positions, dtype=np.float32)
    if up == "Y":  # Blender (x, y, z) -> glTF/OBJ (x, z, -y)
        return np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1)
    return p.copy()


def write_png(path, rgba_bottom_up):
    """Write an 8-bit RGBA or RGB image whose row 0 is the bottom (Blender convention)."""
    img = np.ascontiguousarray(np.asarray(rgba_bottom_up, dtype=np.uint8)[::-1])
    h, w, c = img.shape
    color_type = {3: 2, 4: 6}[c]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
    return png


def write_hdr(path, rgb_bottom_up):
    """Radiance RGBE (uncompressed scanlines), linear values."""
    img = np.asarray(rgb_bottom_up, dtype=np.float32)[::-1, :, :3]
    h, w = img.shape[:2]
    m = img.max(axis=2)
    e = np.zeros_like(m, dtype=np.int32)
    mant, e = np.frexp(m)
    scale = np.where(m > 1e-32, mant * 256.0 / np.maximum(m, 1e-32), 0)
    rgbe = np.zeros((h, w, 4), dtype=np.uint8)
    rgbe[..., :3] = np.clip(img * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(m > 1e-32, e + 128, 0).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {h} +X {w}\n".encode())
        f.write(rgbe.tobytes())


def write_obj(path, positions, uvs, indices, texture_name, up="Y"):
    base = os.path.splitext(path)[0]
    mtl_path = base + ".mtl"
    name = os.path.basename(base)
    p = convert_axes(positions, up)
    lines = [f"# Baked Engine: {len(p)} vertices, {len(indices)} triangles", f"mtllib {os.path.basename(mtl_path)}", f"o {name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in p]
    lines += [f"vt {u:.7f} {v:.7f}" for u, v in uvs]
    lines.append(f"usemtl {name}")
    lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in indices]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {name}\nKa 0 0 0\nKd 1 1 1\nKs 0 0 0\nillum 0\nmap_Kd {texture_name}\nmap_d {texture_name}\n")


def write_glb(path, positions, uvs, indices, png_bytes, up="Y", name="baked"):
    """Single mesh, single unlit material, embedded PNG (KHR_materials_unlit)."""
    p = convert_axes(positions, up).astype(np.float32)
    # glTF uv origin is the top-left of the image
    t = np.asarray(uvs, dtype=np.float32).copy()
    t[:, 1] = 1.0 - t[:, 1]
    idx = np.asarray(indices, dtype=np.uint32).reshape(-1)
    blobs = []
    views = []
    offset = 0

    def add(data, target=None):
        nonlocal offset
        b = data.tobytes() if hasattr(data, "tobytes") else data
        pad = (-len(b)) % 4
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(b)}
        if target:
            view["target"] = target
        views.append(view)
        blobs.append(b + b"\x00" * pad)
        offset += len(b) + pad
        return len(views) - 1

    vp = add(p, 34962)
    vt = add(t, 34962)
    vi = add(idx, 34963)
    vimg = add(png_bytes)
    lo = p.min(axis=0).tolist() if len(p) else [0, 0, 0]
    hi = p.max(axis=0).tolist() if len(p) else [0, 0, 0]
    gltf = {
        "asset": {"version": "2.0", "generator": "Baked Engine"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [
            {
                "name": name,
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0},
                "alphaMode": "MASK",
                "alphaCutoff": 0.5,
                "extensions": {"KHR_materials_unlit": {}},
            }
        ],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": vimg, "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": vp, "componentType": 5126, "count": len(p), "type": "VEC3", "min": lo, "max": hi},
            {"bufferView": vt, "componentType": 5126, "count": len(t), "type": "VEC2"},
            {"bufferView": vi, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    binary = b"".join(blobs)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binary), 0x004E4942) + binary)


def write_bkd(path, positions, uvs, indices, rgba_bottom_up, up="Y"):
    p = convert_axes(positions, up).astype(np.float32)
    v = np.concatenate([p, np.asarray(uvs, dtype=np.float32)], axis=1)
    idx = np.asarray(indices, dtype=np.uint32).reshape(-1)
    img = np.ascontiguousarray(np.asarray(rgba_bottom_up, dtype=np.uint8)[::-1])
    h, w = img.shape[:2]
    with open(path, "wb") as f:
        f.write(b"BKD1" + struct.pack("<IIII", len(v), len(idx), w, h))
        f.write(v.tobytes())
        f.write(idx.tobytes())
        f.write(img.tobytes())


def read_bkd(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"BKD1"
    nv, ni, w, h = struct.unpack("<IIII", data[4:20])
    o = 20
    v = np.frombuffer(data, np.float32, nv * 5, o).reshape(nv, 5)
    o += nv * 20
    idx = np.frombuffer(data, np.uint32, ni, o)
    o += ni * 4
    img = np.frombuffer(data, np.uint8, w * h * 4, o).reshape(h, w, 4)
    return v, idx, img


def export_result(result, directory, name="baked", formats=("glb", "obj", "bkd"), view="AGX", exposure=0.0, up="Y", hdr=True):
    """Write every requested format. Returns a dict of written paths."""
    os.makedirs(directory, exist_ok=True)
    rgba = result.encoded(view, exposure)
    paths = {}
    png_path = os.path.join(directory, f"{name}.png")
    png = write_png(png_path, rgba)
    paths["png"] = png_path
    if hdr:
        hp = os.path.join(directory, f"{name}.hdr")
        write_hdr(hp, result.image[..., :3])
        paths["hdr"] = hp
    if "obj" in formats:
        op = os.path.join(directory, f"{name}.obj")
        write_obj(op, result.positions, result.uvs, result.indices, os.path.basename(png_path), up)
        paths["obj"] = op
    if "glb" in formats:
        gp = os.path.join(directory, f"{name}.glb")
        write_glb(gp, result.positions, result.uvs, result.indices, png, up, name)
        paths["glb"] = gp
    if "bkd" in formats:
        bp = os.path.join(directory, f"{name}.bkd")
        write_bkd(bp, result.positions, result.uvs, result.indices, rgba, up)
        paths["bkd"] = bp
    meta = dict(result.stats)
    meta.update(width=result.width, height=result.height, vertices=int(len(result.positions)), up_axis=up, view_transform=view)
    mp = os.path.join(directory, f"{name}.json")
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2, default=float)
    paths["json"] = mp
    return paths
