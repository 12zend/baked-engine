"""Baked Engine: bake a static Blender scene into triangles (xyz + uv) and one texture.

The ``core`` sub-package is pure numpy and never imports ``bpy`` so it can be
tested and run headless. The ``blender`` sub-package holds the add-on UI.
"""

bl_info = {
    "name": "Baked Engine",
    "author": "12zend",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "Properties > Render > Baked Engine",
    "description": "Path-traced bake of a static scene into triangles plus one texture, "
    "including view-correct planar reflections baked as geometry",
    "category": "Render",
}


def register():
    from .blender import register as _register

    _register()


def unregister():
    from .blender import unregister as _unregister

    _unregister()
