"""Scene description consumed by the bake pipeline.

Everything is stored as plain numpy arrays in world space so the pipeline can
run without Blender. Positions are in metres, colours are linear (scene
referred) RGB.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def luminance(rgb):
    rgb = np.asarray(rgb)
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


@dataclass
class Texture:
    """A linear float image, ``data[y, x]`` with ``y = 0`` at the bottom (v = 0)."""

    data: np.ndarray
    name: str = ""

    def __post_init__(self):
        d = np.asarray(self.data, dtype=np.float32)
        if d.ndim == 2:
            d = d[:, :, None]
        if d.shape[2] == 1:
            d = np.concatenate([d, d, d, np.ones_like(d)], axis=2)
        elif d.shape[2] == 3:
            d = np.concatenate([d, np.ones_like(d[:, :, :1])], axis=2)
        self.data = np.ascontiguousarray(d[:, :, :4])

    @property
    def has_alpha(self) -> bool:
        return bool((self.data[:, :, 3] < 0.999).any())

    def sample(self, uv: np.ndarray) -> np.ndarray:
        """Bilinear sample with repeat wrapping. ``uv`` is (N, 2); returns (N, 4)."""
        h, w = self.data.shape[:2]
        x = np.asarray(uv[:, 0], dtype=np.float64) * w - 0.5
        y = np.asarray(uv[:, 1], dtype=np.float64) * h - 0.5
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = (x - x0)[:, None].astype(np.float32)
        fy = (y - y0)[:, None].astype(np.float32)
        x0 = x0.astype(np.int64) % w
        y0 = y0.astype(np.int64) % h
        x1 = (x0 + 1) % w
        y1 = (y0 + 1) % h
        d = self.data
        top = d[y0, x0] * (1 - fx) + d[y0, x1] * fx
        bot = d[y1, x0] * (1 - fx) + d[y1, x1] * fx
        return top * (1 - fy) + bot * fy


@dataclass
class Material:
    name: str = "material"
    base_color: tuple = (0.8, 0.8, 0.8)
    base_color_texture: int = -1  # index into BakeScene.textures
    alpha: float = 1.0
    alpha_clip: float = 0.5  # texels with alpha below this are holes
    roughness: float = 0.5
    metallic: float = 0.0
    specular: float = 0.5  # Blender's "IOR level"; 0.5 -> F0 = 0.04
    emission: tuple = (0.0, 0.0, 0.0)
    emission_strength: float = 0.0
    emission_texture: int = -1

    @property
    def emissive(self) -> bool:
        return self.emission_strength > 0 and max(self.emission) > 0

    def f0(self) -> np.ndarray:
        dielectric = np.full(3, 0.08 * self.specular)
        return dielectric * (1 - self.metallic) + np.asarray(self.base_color) * self.metallic


@dataclass
class Light:
    kind: str  # 'POINT', 'SPOT', 'SUN', 'AREA'
    color: tuple = (1.0, 1.0, 1.0)
    power: float = 10.0  # W for point/spot/area, W/m^2 (irradiance) for sun
    position: tuple = (0.0, 0.0, 0.0)
    direction: tuple = (0.0, 0.0, -1.0)  # direction the light points / travels
    radius: float = 0.1  # soft shadow radius for point/spot
    angle: float = 0.00918  # sun angular diameter in radians (Blender default 0.526 deg)
    spot_size: float = np.pi / 4  # full cone angle
    spot_blend: float = 0.15
    size: tuple = (1.0, 1.0)  # area light extents along axis_u / axis_v
    shape: str = "RECTANGLE"  # 'RECTANGLE', 'SQUARE', 'DISK', 'ELLIPSE'
    axis_u: tuple = (1.0, 0.0, 0.0)
    axis_v: tuple = (0.0, 1.0, 0.0)


@dataclass
class Environment:
    """World lighting: constant colour or an equirectangular map (z up)."""

    color: tuple = (0.05, 0.05, 0.05)
    strength: float = 1.0
    texture: np.ndarray | None = None  # (H, W, 3) linear, row 0 = bottom (-z), Blender equirect
    rotation_z: float = 0.0

    def __post_init__(self):
        self._cdf = None
        if self.texture is not None:
            self.texture = np.ascontiguousarray(np.asarray(self.texture, dtype=np.float32)[:, :, :3])

    def eval(self, dirs: np.ndarray) -> np.ndarray:
        dirs = np.asarray(dirs, dtype=np.float64)
        if self.texture is None:
            return np.broadcast_to(np.asarray(self.color, np.float32) * self.strength, dirs.shape).astype(np.float32)
        u, v = self._dir_to_uv(dirs)
        tex = Texture(self.texture)
        out = tex.sample(np.stack([u, v], axis=1))[:, :3]
        return out * np.float32(self.strength)

    # Equirect convention of Cycles' Environment Texture: u = 0.5 - atan2(y, x) / 2pi.
    # ``rotation_z`` mimics a Mapping node rotating the lookup vector about +Z.
    def _dir_to_uv(self, dirs):
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        c, s = np.cos(self.rotation_z), np.sin(self.rotation_z)
        x, y = c * x - s * y, s * x + c * y
        u = 0.5 - np.arctan2(y, x) / (2 * np.pi)
        v = 0.5 + np.arcsin(np.clip(z / np.maximum(np.sqrt(x * x + y * y + z * z), 1e-30), -1, 1)) / np.pi
        return u, v

    def _uv_to_dir(self, u, v):
        phi = (0.5 - u) * 2 * np.pi
        theta = (v - 0.5) * np.pi
        x = np.cos(phi) * np.cos(theta)
        y = np.sin(phi) * np.cos(theta)
        z = np.sin(theta)
        c, s = np.cos(-self.rotation_z), np.sin(-self.rotation_z)
        x, y = c * x - s * y, s * x + c * y
        return np.stack([x, y, z], axis=1)

    @property
    def is_constant(self) -> bool:
        return self.texture is None

    def average(self) -> np.ndarray:
        if self.texture is None:
            return np.asarray(self.color, np.float64) * self.strength
        h = self.texture.shape[0]
        lat = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
        w = np.cos(lat)[:, None, None]
        return (self.texture * w).sum(axis=(0, 1)) / (w.sum() * self.texture.shape[1]) * self.strength

    def build_sampling(self):
        """Prepare a piecewise-constant importance sampling table (luminance x sin(theta))."""
        if self.texture is None or self._cdf is not None:
            return
        h, w = self.texture.shape[:2]
        lat = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
        lum = luminance(self.texture).astype(np.float64) + 1e-8
        pdf = lum * np.cos(lat)[:, None]
        flat = pdf.ravel()
        total = flat.sum()
        self._cdf = np.cumsum(flat) / total
        # pdf w.r.t. solid angle for each pixel: p_pix / (dA) where dA = cos(lat) dphi dtheta
        dphi = 2 * np.pi / w
        dtheta = np.pi / h
        self._pdf_sa = (pdf / total) / (np.maximum(np.cos(lat)[:, None], 1e-6) * dphi * dtheta)

    def sample(self, xi1, xi2, xi3):
        """Importance sample directions. Returns (dirs, pdf_solid_angle)."""
        n = len(xi1)
        if self.texture is None:
            z = 1 - 2 * xi1
            r = np.sqrt(np.maximum(0, 1 - z * z))
            phi = 2 * np.pi * xi2
            d = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
            return d, np.full(n, 1 / (4 * np.pi))
        self.build_sampling()
        h, w = self.texture.shape[:2]
        idx = np.minimum(np.searchsorted(self._cdf, xi1, side="right"), h * w - 1)
        py, px = idx // w, idx % w
        u = (px + xi2) / w
        v = (py + xi3) / h
        return self._uv_to_dir(u, v), self._pdf_sa[py, px]

    def pdf(self, dirs):
        n = len(dirs)
        if self.texture is None:
            return np.full(n, 1 / (4 * np.pi))
        self.build_sampling()
        h, w = self.texture.shape[:2]
        u, v = self._dir_to_uv(dirs)
        px = np.clip((u * w).astype(np.int64), 0, w - 1)
        py = np.clip((v * h).astype(np.int64), 0, h - 1)
        return self._pdf_sa[py, px]


@dataclass
class Camera:
    position: tuple = (0.0, 0.0, 0.0)
    forward: tuple = (0.0, 1.0, 0.0)


@dataclass
class ReflectorConfig:
    """How an object's glossy reflections are baked.

    mode:
      'AUTO'     - geometry reflections for glossy planar surfaces, texture otherwise
      'GEOMETRY' - planar patches become holes that show mirrored geometry
      'TEXTURE'  - reflection baked into the texture for the reference view
      'OFF'      - diffuse only
    """

    mode: str = "AUTO"
    strength: float = -1.0  # < 0: physically based (Fresnel from material)
    blur: float = 0.01  # extra blur cone half-angle in radians on top of roughness
    fresnel: bool = True
    planar_tolerance: float = 0.035  # radians of normal deviation allowed in a patch
    flare: float = -1.0  # side-plane tilt (radians) of the reflection volume; < 0: 80 deg for water, 50 deg otherwise
    mirror_depth: float = 0.0  # 0 = auto (scene size)
    remove_below: str = "AUTO"  # 'AUTO', 'ON', 'OFF': drop geometry behind the mirror


@dataclass
class SceneObject:
    name: str
    reflector: ReflectorConfig = field(default_factory=ReflectorConfig)
    output: bool = True  # False: only casts shadows / bounces light


@dataclass
class BakeScene:
    positions: np.ndarray  # (T, 3, 3) world positions
    normals: np.ndarray  # (T, 3, 3) shading normals per corner
    uvs: np.ndarray  # (T, 3, 2) material uv
    material_ids: np.ndarray  # (T,)
    object_ids: np.ndarray  # (T,)
    materials: list = field(default_factory=list)
    textures: list = field(default_factory=list)
    lights: list = field(default_factory=list)
    environment: Environment = field(default_factory=Environment)
    camera: Camera | None = None
    objects: list = field(default_factory=list)

    def __post_init__(self):
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float64).reshape(-1, 3, 3)
        t = len(self.positions)
        if self.normals is None:
            self.normals = np.repeat(self.face_normals()[:, None, :], 3, axis=1)
        self.normals = np.ascontiguousarray(self.normals, dtype=np.float64).reshape(t, 3, 3)
        if self.uvs is None:
            self.uvs = np.zeros((t, 3, 2))
        self.uvs = np.ascontiguousarray(self.uvs, dtype=np.float64).reshape(t, 3, 2)
        self.material_ids = np.asarray(self.material_ids, dtype=np.int32).reshape(t)
        self.object_ids = np.asarray(self.object_ids, dtype=np.int32).reshape(t)
        if not self.materials:
            self.materials = [Material()]
        if not self.objects:
            n_obj = int(self.object_ids.max()) + 1 if t else 0
            self.objects = [SceneObject(f"object_{i}") for i in range(n_obj)]

    @property
    def triangle_count(self) -> int:
        return len(self.positions)

    def face_normals(self) -> np.ndarray:
        p = self.positions
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)

    def areas(self) -> np.ndarray:
        p = self.positions
        return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)

    def bounds(self):
        if self.triangle_count == 0:
            return np.zeros(3), np.zeros(3)
        p = self.positions.reshape(-1, 3)
        return p.min(axis=0), p.max(axis=0)

    def extent(self) -> float:
        lo, hi = self.bounds()
        return float(max(np.linalg.norm(hi - lo), 1e-6))
