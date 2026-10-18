"""Convert the evaluated Blender scene into a :class:`BakeScene`."""

from __future__ import annotations

import math

import numpy as np
from mathutils import Matrix, Vector

from ..core.scene import BakeScene, Camera, Environment, Light, Material, ReflectorConfig, SceneObject, Texture


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


class _TextureCache:
    def __init__(self):
        self.textures = []
        self.index = {}

    def get(self, image, max_size=4096):
        if image is None:
            return -1
        key = image.name
        if key in self.index:
            return self.index[key]
        w, h = image.size
        if w == 0 or h == 0:
            self.index[key] = -1
            return -1
        px = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(px)
        px = px.reshape(h, w, 4)
        cs = image.colorspace_settings.name.lower()
        if not image.is_float and ("srgb" in cs and "linear" not in cs):
            px[..., :3] = srgb_to_linear(px[..., :3])
        step = max(1, int(math.ceil(max(w, h) / max_size)))
        if step > 1:
            px = px[::step, ::step]
        self.textures.append(Texture(px, image.name))
        self.index[key] = len(self.textures) - 1
        return self.index[key]


def _input(node, *names):
    for n in names:
        if n in node.inputs:
            return node.inputs[n]
    return None


def _upstream_image(socket, depth=0):
    """First Image Texture node feeding ``socket`` (through simple colour nodes)."""
    if socket is None or not socket.is_linked or depth > 6:
        return None
    node = socket.links[0].from_node
    if node.type == "TEX_IMAGE":
        return node
    for inp in node.inputs:
        if inp.type in ("RGBA", "VALUE", "VECTOR") and inp.is_linked:
            found = _upstream_image(inp, depth + 1)
            if found is not None:
                return found
    return None


def _value(socket, default):
    if socket is None:
        return default
    v = socket.default_value
    try:
        return tuple(v)[:3] if len(v) >= 3 else tuple(v)
    except TypeError:
        return float(v)


def _surface_node(mat):
    if not mat.use_nodes or mat.node_tree is None:
        return None
    out = None
    for node in mat.node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL" and node.target in ("ALL", "CYCLES"):
            if out is None or node.is_active_output:
                out = node
    if out is None:
        return None
    surf = out.inputs.get("Surface")
    if surf is None or not surf.is_linked:
        return None
    return surf.links[0].from_node


def convert_material(mat, cache: _TextureCache, warnings):
    m = Material(mat.name if mat else "default")
    if mat is None:
        return m
    m.base_color = tuple(mat.diffuse_color[:3])
    m.roughness = float(mat.roughness)
    m.metallic = float(mat.metallic)
    node = _surface_node(mat)
    if node is None:
        return m
    shaders = [node]
    if node.type in ("MIX_SHADER", "ADD_SHADER"):
        # use the most significant branch; mixing BSDFs is not representable exactly
        branch = [s.links[0].from_node for s in node.inputs if s.type == "SHADER" and s.is_linked]
        shaders = branch or shaders
        warnings.add(f"material '{mat.name}': mix/add shader approximated by its first BSDF")
    emission_nodes = [s for s in shaders if s.type == "EMISSION"]
    bsdf = next((s for s in shaders if s.type != "EMISSION"), None)
    if bsdf is not None:
        t = bsdf.type
        col = _input(bsdf, "Base Color", "Color")
        if col is not None:
            m.base_color = _value(col, m.base_color)
            img = _upstream_image(col)
            if img is not None and img.image is not None:
                m.base_color_texture = cache.get(img.image)
                if img.image is not None and col.links[0].from_node is img:
                    m.base_color = (1.0, 1.0, 1.0)
        r = _input(bsdf, "Roughness")
        if r is not None:
            m.roughness = float(_value(r, m.roughness))
        if t == "BSDF_PRINCIPLED":
            m.metallic = float(_value(_input(bsdf, "Metallic"), 0.0))
            spec = _input(bsdf, "Specular IOR Level", "IOR Level", "Specular")
            level = float(_value(spec, 0.5))
            ior_in = _input(bsdf, "IOR")
            if ior_in is not None and spec is not None and spec.name != "Specular":
                ior = max(float(_value(ior_in, 1.5)), 1.0)
                f0 = ((ior - 1) / (ior + 1)) ** 2 * 2 * level
                m.specular = f0 / 0.08
            else:
                m.specular = level
            a = _input(bsdf, "Alpha")
            m.alpha = float(_value(a, 1.0))
            ec = _input(bsdf, "Emission Color", "Emission")
            es = _input(bsdf, "Emission Strength")
            if ec is not None:
                m.emission = _value(ec, (0, 0, 0))
                m.emission_strength = float(_value(es, 1.0)) if es is not None else 1.0
                eimg = _upstream_image(ec)
                if eimg is not None and eimg.image is not None:
                    m.emission_texture = cache.get(eimg.image)
                    m.emission = (1.0, 1.0, 1.0)
        elif t == "BSDF_GLOSSY" or t == "BSDF_ANISOTROPIC":
            m.metallic = 1.0
        elif t in ("BSDF_GLASS", "BSDF_REFRACTION", "BSDF_TRANSPARENT"):
            warnings.add(f"material '{mat.name}': transmission is baked as an opaque glossy surface")
            m.metallic = 0.0
            m.specular = 0.5
        elif t == "BSDF_DIFFUSE":
            m.roughness = 1.0
            m.specular = 0.0
        elif t not in ("BSDF_PRINCIPLED",):
            warnings.add(f"material '{mat.name}': shader '{bsdf.bl_label}' approximated as diffuse")
    for en in emission_nodes:
        c = _input(en, "Color")
        m.emission = _value(c, (1, 1, 1))
        m.emission_strength = float(_value(_input(en, "Strength"), 1.0))
        eimg = _upstream_image(c)
        if eimg is not None and eimg.image is not None:
            m.emission_texture = cache.get(eimg.image)
            m.emission = (1.0, 1.0, 1.0)
        if bsdf is None:
            m.base_color = (0.0, 0.0, 0.0)
            m.specular = 0.0
    return m


def reflector_config(obj):
    props = getattr(obj, "baked_engine", None)
    if props is None:
        return ReflectorConfig(), True
    cfg = ReflectorConfig(
        mode=props.reflection_mode,
        strength=props.strength if props.override_strength else -1.0,
        blur=props.blur,
        fresnel=props.fresnel,
        planar_tolerance=props.planar_tolerance,
        flare=props.flare if props.override_flare else -1.0,
        mirror_depth=props.mirror_depth,
        remove_below=props.remove_below,
    )
    return cfg, bool(props.include_in_output)


def _mesh_arrays(mesh):
    mesh.calc_loop_triangles()
    nt = len(mesh.loop_triangles)
    nv = len(mesh.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(nv, 3)
    tri_v = np.empty(nt * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tri_v)
    tri_l = np.empty(nt * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("loops", tri_l)
    mat_idx = np.empty(nt, dtype=np.int64)
    mesh.loop_triangles.foreach_get("material_index", mat_idx)
    nl = len(mesh.loops)
    normals = np.empty(nl * 3, dtype=np.float32)
    if hasattr(mesh, "corner_normals"):
        mesh.corner_normals.foreach_get("vector", normals)
    else:  # Blender < 4.1
        mesh.calc_normals_split()
        mesh.loops.foreach_get("normal", normals)
    normals = normals.reshape(nl, 3)
    uv = np.zeros((nl, 2), dtype=np.float32)
    layer = None
    for l in mesh.uv_layers:
        if l.active_render:
            layer = l
    if layer is None and len(mesh.uv_layers):
        layer = mesh.uv_layers.active
    if layer is not None:
        buf = np.empty(nl * 2, dtype=np.float32)
        layer.data.foreach_get("uv", buf)
        uv = buf.reshape(nl, 2)
    return co[tri_v.reshape(nt, 3)], normals[tri_l.reshape(nt, 3)], uv[tri_l.reshape(nt, 3)], mat_idx, (layer.name if layer else None)


def _light(obj, matrix):
    d = obj.data
    mw = Matrix(matrix)
    pos = tuple(mw.translation)
    rot = mw.to_3x3()
    fwd = (rot @ Vector((0, 0, -1))).normalized()
    col = tuple(d.color)
    energy = float(d.energy) * 2.0 ** float(getattr(d, "exposure", 0.0))
    if d.type == "SUN":
        return Light("SUN", col, energy, pos, tuple(fwd), angle=float(getattr(d, "angle", 0.00918)))
    radius = float(getattr(d, "shadow_soft_size", 0.0))
    if d.type == "POINT":
        return Light("POINT", col, energy, pos, tuple(fwd), radius=radius)
    if d.type == "SPOT":
        return Light("SPOT", col, energy, pos, tuple(fwd), radius=radius, spot_size=float(d.spot_size), spot_blend=float(d.spot_blend))
    if d.type == "AREA":
        sx = rot.col[0].length
        sy = rot.col[1].length
        size_x = float(d.size) * sx
        size_y = float(d.size_y if d.shape in ("RECTANGLE", "ELLIPSE") else d.size) * sy
        au = tuple(rot.col[0].normalized())
        av = tuple(rot.col[1].normalized())
        if not getattr(d, "normalize", True):
            # un-normalised area lights keep their radiance when resized: power grows with area
            area = size_x * size_y * (math.pi / 4 if d.shape in ("DISK", "ELLIPSE") else 1.0)
            energy *= area
        return Light("AREA", col, energy, pos, tuple(fwd), size=(size_x, size_y), shape=d.shape, axis_u=au, axis_v=av)
    return None


def _world(scene, cache, warnings, capture):
    world = scene.world
    if world is None:
        return Environment(color=(0.0, 0.0, 0.0))
    env = Environment(color=tuple(world.color), strength=1.0)
    if not world.use_nodes or world.node_tree is None:
        return env
    out = next((n for n in world.node_tree.nodes if n.type == "OUTPUT_WORLD" and n.is_active_output), None)
    out = out or next((n for n in world.node_tree.nodes if n.type == "OUTPUT_WORLD"), None)
    if out is None or not out.inputs["Surface"].is_linked:
        return Environment(color=(0.0, 0.0, 0.0))
    bg = out.inputs["Surface"].links[0].from_node
    if bg.type != "BACKGROUND":
        warnings.add("world: unsupported shader, capturing it with Cycles")
        return capture(scene) if capture else env
    col = bg.inputs["Color"]
    strength = float(bg.inputs["Strength"].default_value) if not bg.inputs["Strength"].is_linked else 1.0
    if not col.is_linked:
        return Environment(color=tuple(col.default_value)[:3], strength=strength)
    src = col.links[0].from_node
    if src.type == "TEX_ENVIRONMENT" and src.image is not None and not src.inputs["Vector"].is_linked:
        idx = cache.get(src.image, max_size=2048)
        if idx >= 0:
            tex = cache.textures[idx].data[..., :3]
            return Environment(texture=tex, strength=strength)
    if src.type == "TEX_ENVIRONMENT" and src.image is not None and src.inputs["Vector"].is_linked:
        mp = src.inputs["Vector"].links[0].from_node
        if mp.type == "MAPPING" and not mp.inputs["Rotation"].is_linked:
            idx = cache.get(src.image, max_size=2048)
            rot = mp.inputs["Rotation"].default_value
            if idx >= 0 and abs(rot[0]) < 1e-6 and abs(rot[1]) < 1e-6:
                return Environment(texture=cache.textures[idx].data[..., :3], strength=strength, rotation_z=float(rot[2]))
    # anything else (sky texture, gradients, ...) is captured into an equirectangular map
    return capture(scene) if capture else env


def extract_scene(context, settings_props=None, capture_world="AUTO", only_selected=False):
    """Returns (BakeScene, info dict). Worlds other than a colour / HDRI are captured with Cycles
    unless ``capture_world`` is None."""
    if capture_world == "AUTO":
        from .world_capture import capture_world
    scene = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    cache = _TextureCache()
    warnings = set()
    materials = []
    mat_index = {}
    bpy_materials = []

    def mat_id(mat):
        key = mat.name if mat else None
        if key not in mat_index:
            mat_index[key] = len(materials)
            materials.append(convert_material(mat, cache, warnings))
            bpy_materials.append(mat.name if mat else None)
        return mat_index[key]

    P_list, N_list, UV_list, M_list, O_list = [], [], [], [], []
    objects = []
    obj_index = {}
    lights = []
    uv_names = {}
    mesh_cache = {}
    selected = {o.name for o in context.selected_objects} if only_selected else None
    for inst in depsgraph.object_instances:
        ob = inst.object
        if ob.type == "LIGHT":
            lt = _light(ob, inst.matrix_world)
            if lt is not None:
                lights.append(lt)
            continue
        if ob.type not in ("MESH", "CURVE", "SURFACE", "META", "FONT"):
            continue
        orig = ob.original
        if orig.hide_render or orig.get("baked_engine_result"):
            continue
        key = (ob.name, getattr(ob.data, "name", ""), bool(inst.is_instance))
        if key not in mesh_cache:
            try:
                me = ob.to_mesh()
            except RuntimeError:
                continue
            if me is None:
                continue
            arrays = _mesh_arrays(me)
            slots = [s.material for s in ob.material_slots]
            ob.to_mesh_clear()
            mesh_cache[key] = (arrays, slots)
        (tp, tn, tuv, tmi, uvname), slots = mesh_cache[key]
        if len(tp) == 0:
            continue
        mw = np.array(inst.matrix_world, dtype=np.float64)
        R = mw[:3, :3]
        p = tp.reshape(-1, 3).astype(np.float64) @ R.T + mw[:3, 3]
        p = p.reshape(-1, 3, 3)
        nrm_m = np.linalg.inv(R).T
        n = tn.reshape(-1, 3).astype(np.float64) @ nrm_m.T
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        n = n.reshape(-1, 3, 3)
        uv = tuv.astype(np.float64)
        if np.linalg.det(R) < 0:
            p = p[:, [0, 2, 1]]
            n = n[:, [0, 2, 1]]
            uv = uv[:, [0, 2, 1]]
        if slots:
            local = np.array([mat_id(s) for s in slots], dtype=np.int64)
            mids = local[np.clip(tmi, 0, len(local) - 1)]
        else:
            mids = np.full(len(tp), mat_id(None), dtype=np.int64)
        oname = orig.name if not inst.is_instance else f"{inst.parent.original.name}/{orig.name}" if inst.parent else orig.name
        cfg_src = orig
        if oname not in obj_index:
            cfg, include = reflector_config(cfg_src)
            if selected is not None and orig.name not in selected and not (inst.parent and inst.parent.original.name in selected):
                include = False
            obj_index[oname] = len(objects)
            objects.append(SceneObject(oname, cfg, include))
        P_list.append(p)
        N_list.append(n)
        UV_list.append(uv)
        M_list.append(mids)
        O_list.append(np.full(len(p), obj_index[oname], dtype=np.int64))
        if uvname:
            uv_names[uvname] = uv_names.get(uvname, 0) + len(p)
    if not P_list:
        raise RuntimeError("No renderable mesh geometry found")
    cam = None
    if scene.camera is not None:
        mw = scene.camera.matrix_world
        fwd = (mw.to_3x3() @ Vector((0, 0, -1))).normalized()
        cam = Camera(tuple(mw.translation), tuple(fwd))
    env = _world(scene, cache, warnings, capture_world)
    bs = BakeScene(
        np.concatenate(P_list),
        np.concatenate(N_list),
        np.concatenate(UV_list),
        np.concatenate(M_list),
        np.concatenate(O_list),
        materials,
        cache.textures,
        lights,
        env,
        cam,
        objects,
    )
    info = dict(
        warnings=sorted(warnings),
        bpy_materials=bpy_materials,
        uv_name=max(uv_names, key=uv_names.get) if uv_names else "UVMap",
    )
    return bs, info
