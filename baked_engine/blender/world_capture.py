"""Capture any world shader (sky texture, gradients, node setups) as an equirectangular map.

A temporary scene that shares the world but contains no geometry is rendered
with a panoramic Cycles camera. The camera is oriented so the result follows
the convention of :class:`baked_engine.core.scene.Environment`.
"""

from __future__ import annotations

import os
import tempfile

import bpy
import numpy as np

from ..core.scene import Environment

# Euler rotation (XYZ) of the panoramic camera so that image u/v match Environment's convention.
CAPTURE_ROTATION = (np.pi / 2, 0.0, -np.pi / 2)


def capture_world(scene, width=512, height=256, samples=16):
    world = scene.world
    if world is None:
        return Environment(color=(0.0, 0.0, 0.0))
    tmp = bpy.data.scenes.new("BE_world_capture")
    cam_data = bpy.data.cameras.new("BE_world_capture")
    cam = bpy.data.objects.new("BE_world_capture", cam_data)
    path = os.path.join(tempfile.mkdtemp(prefix="baked_engine_"), "world.exr")
    try:
        tmp.world = world
        tmp.collection.objects.link(cam)
        tmp.camera = cam
        cam.rotation_euler = CAPTURE_ROTATION
        cam_data.type = "PANO"
        if hasattr(cam_data, "panorama_type"):
            cam_data.panorama_type = "EQUIRECTANGULAR"
        else:  # Blender < 4.0
            cam_data.cycles.panorama_type = "EQUIRECTANGULAR"
        r = tmp.render
        r.engine = "CYCLES"
        tmp.cycles.samples = samples
        if hasattr(tmp.cycles, "use_denoising"):
            tmp.cycles.use_denoising = False
        r.resolution_x = width
        r.resolution_y = height
        r.resolution_percentage = 100
        r.film_transparent = False
        r.image_settings.file_format = "OPEN_EXR"
        r.image_settings.color_depth = "32"
        tmp.view_settings.view_transform = "Standard" if "Standard" in [i.identifier for i in tmp.view_settings.bl_rna.properties["view_transform"].enum_items] else tmp.view_settings.view_transform
        r.filepath = path
        bpy.ops.render.render(write_still=True, scene=tmp.name)
        img = bpy.data.images.load(path)
        try:
            px = np.empty(width * height * 4, dtype=np.float32)
            img.pixels.foreach_get(px)
        finally:
            bpy.data.images.remove(img)
        tex = px.reshape(height, width, 4)[..., :3]
        return Environment(texture=tex, strength=1.0)
    finally:
        bpy.data.objects.remove(cam, do_unlink=True)
        bpy.data.cameras.remove(cam_data)
        bpy.data.scenes.remove(tmp)
        try:
            os.remove(path)
        except OSError:
            pass
