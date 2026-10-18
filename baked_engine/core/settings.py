"""User-facing bake settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BakeSettings:
    # atlas
    width: int = 1024
    height: int = 1024
    texel_density: float = 0.0  # texels per metre; 0 = as high as fits. Lowered automatically if it does not fit.
    padding: int = 4  # texels between charts
    chart_buckets: int = 14  # projection directions used to split charts
    # lighting
    samples: int = 32
    max_bounces: int = 6
    clamp_indirect: float = 10.0  # 0 disables firefly clamping
    denoise: bool = True
    denoise_iterations: int = 5
    seed: int = 0
    texel_chunk: int = 1 << 18  # bake points traced together
    # reflections
    reflections: bool = True
    reflection_depth: int = 1  # mirror-in-mirror recursion for geometry reflections
    reflection_samples: int = 24  # gather rays per texel for glossy blur
    mirror_density: float = 0.5  # texel density of mirrored geometry relative to the scene
    aux_density: float = 0.25  # density of hidden/removed surfaces (used only for lookups)
    aux_size: int = 0  # auxiliary atlas size; 0 = half the main width
    dome_size: int = 0  # pixel size of each mirrored-sky dome chart; 0 = width / 8
    dome_radius: float = 0.0  # 0 = 4 x scene extent
    dome_segments: int = 32
    max_reflectors: int = 16
    reference_view: str = "AUTO"  # 'AUTO' (camera when usable), 'CAMERA', 'SYNTHETIC'
    specular_bake: bool = True  # bake view-dependent gloss for metals / texture-mode reflectors
    triangle_budget: int = 4_000_000
    # output
    view_transform: str = "AGX"  # 'STANDARD', 'ACES', 'AGX', 'NONE' (linear)
    exposure: float = 0.0

    def to_dict(self):
        return asdict(self)
