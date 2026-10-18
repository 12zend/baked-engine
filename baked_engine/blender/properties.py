import math

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import PropertyGroup

from ..core.settings import BakeSettings

_D = BakeSettings()


class BakedEngineSceneProps(PropertyGroup):
    width: IntProperty(name="Width", default=1024, min=16, max=16384, description="Texture width in pixels")
    height: IntProperty(name="Height", default=1024, min=16, max=16384, description="Texture height in pixels")
    texel_density: FloatProperty(
        name="Texel Density",
        default=0.0,
        min=0.0,
        description="Wanted texels per metre. 0 = as high as fits. Lowered automatically when the scene does not fit",
    )
    padding: IntProperty(name="Padding", default=4, min=1, max=64, description="Texels between atlas charts")
    backend: EnumProperty(
        name="Engine",
        items=[
            ("CYCLES", "Cycles", "Path trace lighting with Cycles (all Blender materials, GPU)"),
            ("BUILTIN", "Built-in", "Path trace lighting with Baked Engine's own numpy path tracer"),
        ],
        default="CYCLES",
    )
    samples: IntProperty(name="Samples", default=128, min=1, max=65536)
    max_bounces: IntProperty(name="Bounces", default=6, min=0, max=64)
    clamp_indirect: FloatProperty(name="Clamp Indirect", default=10.0, min=0.0, description="Firefly clamp, 0 disables")
    denoise: BoolProperty(name="Denoise", default=True)
    rough_specular: BoolProperty(
        name="Rough Specular", default=False, description="Bake the view-averaged specular layer of rough non-metals"
    )
    only_selected: BoolProperty(
        name="Selected Only", default=False, description="Output only selected objects; everything still contributes light"
    )

    reflections: BoolProperty(name="Geometry Reflections", default=True, description="Bake planar reflections as mirrored geometry")
    reflection_depth: IntProperty(name="Mirror Depth", default=1, min=1, max=4, description="Mirrors seen inside mirrors")
    reflection_samples: IntProperty(name="Blur Samples", default=_D.reflection_samples, min=1, max=1024)
    mirror_density: FloatProperty(name="Reflection Resolution", default=_D.mirror_density, min=0.05, max=2.0, description="Texel density of mirrored geometry relative to the scene")
    specular_bake: BoolProperty(name="Texture Reflections", default=True, description="Bake gloss of metals and curved reflectors for the reference view")
    reference_view: EnumProperty(
        name="Reference View",
        items=[("AUTO", "Camera", "Use the scene camera when it sees the reflector"), ("SYNTHETIC", "Automatic", "Place a viewer above each reflector")],
        default="AUTO",
    )

    view_transform: EnumProperty(
        name="View Transform",
        items=[
            ("SCENE", "Scene", "Use the scene's colour management (exactly like a render)"),
            ("AGX", "AgX", ""),
            ("ACES", "ACES (fitted)", ""),
            ("STANDARD", "Standard", ""),
            ("NONE", "Linear", "Clip scene-linear values"),
        ],
        default="SCENE",
    )
    exposure: FloatProperty(name="Exposure", default=0.0, soft_min=-10, soft_max=10)
    output_dir: StringProperty(name="Output", subtype="DIR_PATH", default="//baked/")
    output_name: StringProperty(name="Name", default="baked")
    export_glb: BoolProperty(name="glTF (.glb)", default=True)
    export_obj: BoolProperty(name="OBJ", default=True)
    export_bkd: BoolProperty(name="Raw (.bkd)", default=False)
    export_hdr: BoolProperty(name="HDR texture", default=False)
    up_axis: EnumProperty(name="Up Axis", items=[("Y", "Y Up", "glTF / OBJ convention"), ("Z", "Z Up", "Blender convention")], default="Y")
    create_object: BoolProperty(name="Create Preview Object", default=True)
    hide_sources: BoolProperty(name="Hide Source Objects", default=False)
    status: StringProperty(name="Status", default="")

    def to_settings(self):
        return BakeSettings(
            width=self.width,
            height=self.height,
            texel_density=self.texel_density,
            padding=self.padding,
            samples=self.samples,
            max_bounces=self.max_bounces,
            clamp_indirect=self.clamp_indirect,
            denoise=self.denoise,
            rough_specular=self.rough_specular,
            reflections=self.reflections,
            reflection_depth=self.reflection_depth,
            reflection_samples=self.reflection_samples,
            mirror_density=self.mirror_density,
            specular_bake=self.specular_bake,
            reference_view=self.reference_view,
            view_transform="AGX" if self.view_transform == "SCENE" else self.view_transform,
            exposure=self.exposure,
        )


class BakedEngineObjectProps(PropertyGroup):
    include_in_output: BoolProperty(name="Include in Output", default=True, description="Off: the object only casts shadows and bounces light")
    reflection_mode: EnumProperty(
        name="Reflection",
        items=[
            ("AUTO", "Auto", "Geometry reflection for glossy planar surfaces, texture reflection otherwise"),
            ("GEOMETRY", "Geometry", "Mirrored geometry seen through this surface (correct from every view point)"),
            ("TEXTURE", "Texture", "Reflection baked into the texture for the reference view"),
            ("OFF", "Off", "Diffuse only"),
        ],
        default="AUTO",
    )
    override_strength: BoolProperty(name="Override Strength", default=False)
    strength: FloatProperty(name="Strength", default=0.3, min=0.0, max=1.0, description="Reflectance at normal incidence (replaces F0)")
    fresnel: BoolProperty(name="Fresnel", default=True, description="Stronger reflection at grazing angles")
    blur: FloatProperty(name="Extra Blur", default=0.01, min=0.0, max=0.5, subtype="ANGLE", description="Blur cone added to the material roughness")
    planar_tolerance: FloatProperty(name="Planarity", default=math.radians(2.0), min=0.0, max=math.radians(30), subtype="ANGLE", description="Normal deviation allowed in one mirror plane (raise for wavy water)")
    override_flare: BoolProperty(name="Override Volume", default=False)
    flare: FloatProperty(name="Volume Flare", default=math.radians(60.0), min=0.0, max=math.radians(89.9), subtype="ANGLE", description="How far mirrored geometry may extend sideways behind the mirror")
    mirror_depth: FloatProperty(name="Max Depth", default=0.0, min=0.0, subtype="DISTANCE", description="How deep the mirrored world reaches behind the mirror (0 = auto)")
    remove_below: EnumProperty(
        name="Clear Behind",
        items=[("AUTO", "Auto", "Water-like (upward facing) reflectors clear what is under them"), ("ON", "On", ""), ("OFF", "Off", "")],
        default="AUTO",
        description="Remove original geometry behind the mirror so the reflection is visible through it (e.g. the sea bed)",
    )


classes = (BakedEngineSceneProps, BakedEngineObjectProps)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.baked_engine = PointerProperty(type=BakedEngineSceneProps)
    bpy.types.Object.baked_engine = PointerProperty(type=BakedEngineObjectProps)


def unregister():
    del bpy.types.Object.baked_engine
    del bpy.types.Scene.baked_engine
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
