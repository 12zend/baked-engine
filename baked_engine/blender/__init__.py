"""Blender integration (properties, operators, panels)."""

from . import operators, properties, ui

_modules = (properties, operators, ui)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()
