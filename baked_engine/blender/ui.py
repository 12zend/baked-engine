import bpy
from bpy.types import Panel


class BAKED_ENGINE_PT_main(Panel):
    bl_label = "Baked Engine"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):
        p = context.scene.baked_engine
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        col = lay.column(align=True)
        col.prop(p, "width")
        col.prop(p, "height")
        lay.prop(p, "texel_density")
        lay.prop(p, "padding")
        lay.prop(p, "only_selected")
        row = lay.row()
        row.scale_y = 1.6
        row.operator("baked_engine.bake", icon="RENDER_STILL")
        if p.status:
            lay.label(text=p.status)


class BAKED_ENGINE_PT_lighting(Panel):
    bl_label = "Lighting"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    bl_parent_id = "BAKED_ENGINE_PT_main"

    def draw(self, context):
        p = context.scene.baked_engine
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        lay.prop(p, "backend")
        lay.prop(p, "samples")
        lay.prop(p, "max_bounces")
        lay.prop(p, "clamp_indirect")
        lay.prop(p, "denoise")
        lay.prop(p, "rough_specular")


class BAKED_ENGINE_PT_reflections(Panel):
    bl_label = "Reflections"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    bl_parent_id = "BAKED_ENGINE_PT_main"

    def draw_header(self, context):
        self.layout.prop(context.scene.baked_engine, "reflections", text="")

    def draw(self, context):
        p = context.scene.baked_engine
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        col = lay.column()
        col.active = p.reflections
        col.prop(p, "reflection_depth")
        col.prop(p, "mirror_density")
        col.prop(p, "reflection_samples")
        col.prop(p, "reference_view")
        lay.prop(p, "specular_bake")


class BAKED_ENGINE_PT_output(Panel):
    bl_label = "Output"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    bl_parent_id = "BAKED_ENGINE_PT_main"

    def draw(self, context):
        p = context.scene.baked_engine
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        lay.prop(p, "output_dir")
        if p.output_dir.startswith("//") and not bpy.data.filepath:
            from .result import resolve_output_dir

            lay.label(text=f"Unsaved file: writes to {resolve_output_dir(p.output_dir)}", icon="INFO")
        lay.prop(p, "output_name")
        col = lay.column(heading="Formats", align=True)
        col.prop(p, "export_glb")
        col.prop(p, "export_obj")
        col.prop(p, "export_bkd")
        col.prop(p, "export_hdr")
        lay.prop(p, "up_axis")
        lay.prop(p, "view_transform")
        sub = lay.column()
        sub.active = p.view_transform != "SCENE"
        sub.prop(p, "exposure")
        lay.prop(p, "create_object")
        row = lay.row()
        row.prop(p, "hide_sources")
        row.operator("baked_engine.show_sources", text="", icon="HIDE_OFF")


class BAKED_ENGINE_PT_object(Panel):
    bl_label = "Baked Engine"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type in ("MESH", "CURVE", "SURFACE", "META", "FONT")

    def draw(self, context):
        p = context.object.baked_engine
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        lay.prop(p, "include_in_output")
        lay.prop(p, "reflection_mode")
        col = lay.column()
        col.active = p.reflection_mode != "OFF"
        row = col.row(heading="Strength")
        row.prop(p, "override_strength", text="")
        sub = row.row()
        sub.active = p.override_strength
        sub.prop(p, "strength", text="")
        col.prop(p, "fresnel")
        col.prop(p, "blur")
        geo = col.column()
        geo.active = p.reflection_mode in ("AUTO", "GEOMETRY")
        geo.prop(p, "planar_tolerance")
        geo.prop(p, "remove_below")
        geo.prop(p, "mirror_depth")
        row = geo.row(heading="Volume Flare")
        row.prop(p, "override_flare", text="")
        sub = row.row()
        sub.active = p.override_flare
        sub.prop(p, "flare", text="")


classes = (BAKED_ENGINE_PT_main, BAKED_ENGINE_PT_lighting, BAKED_ENGINE_PT_reflections, BAKED_ENGINE_PT_output, BAKED_ENGINE_PT_object)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
