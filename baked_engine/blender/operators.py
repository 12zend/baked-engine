from __future__ import annotations

import queue
import threading
import traceback

import bpy
from bpy.types import Operator

from ..core.pipeline import bake as run_bake
from .cycles_backend import CyclesBackend
from .extract import extract_scene
from .result import create_result_object, export_files
from .world_capture import capture_world


class Cancelled(Exception):
    pass


def _prepare(context):
    props = context.scene.baked_engine
    bscene, info = extract_scene(context, capture_world=capture_world, only_selected=props.only_selected)
    settings = props.to_settings()
    return props, bscene, info, settings


def _finish(op, context, props, res, info, source_objects):
    name = bpy.path.clean_name(props.output_name) or "baked"
    img = None
    if props.create_object:
        obj, img = create_result_object(context, res, name)
        if props.hide_sources:
            for o in source_objects:
                o.hide_set(True)
                o.hide_render = True
                o["baked_engine_hidden"] = True
    paths = {}
    if props.export_glb or props.export_obj or props.export_bkd or props.export_hdr:
        paths = export_files(context, res, props, img)
    st = res.stats
    msg = (
        f"Baked {st['triangles']} triangles ({st['triangles_mirrored']} mirrored) into "
        f"{res.width}x{res.height} at {st['texel_density']:.2f} texels/m"
    )
    props.status = msg
    op.report({"INFO"}, msg + (f"; wrote {paths.get('png')}" if paths else ""))
    for w in info["warnings"] + st.get("notes", []):
        op.report({"WARNING"}, w)


def _sources(context):
    return [o for o in context.scene.objects if o.type in ("MESH", "CURVE", "SURFACE", "META", "FONT") and not o.hide_render and not o.get("baked_engine_result")]


class BAKED_ENGINE_OT_bake(Operator):
    """Bake the scene into triangles (xyz + uv) and one texture"""

    bl_idname = "baked_engine.bake"
    bl_label = "Bake"
    bl_options = {"REGISTER"}

    _timer = None
    _thread = None

    @classmethod
    def poll(cls, context):
        return context.scene is not None and context.mode == "OBJECT"

    def _backend(self, context, info, run_on_main=None):
        props = context.scene.baked_engine
        if props.backend == "CYCLES":
            return CyclesBackend(context, info["bpy_materials"], info["uv_name"], run_on_main)
        return None

    # synchronous (scripts, command line)
    def execute(self, context):
        try:
            props, bscene, info, settings = _prepare(context)
            sources = _sources(context)
            res = run_bake(bscene, settings, None, self._backend(context, info))
            _finish(self, context, props, res, info, sources)
        except Exception as exc:  # report instead of crashing the UI
            traceback.print_exc()
            self.report({"ERROR"}, f"Baked Engine: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}

    # interactive: worker thread + modal timer; Cycles calls are marshalled to the main thread
    def invoke(self, context, event):
        try:
            self.props, self.bscene, self.info, self.settings = _prepare(context)
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, f"Baked Engine: {exc}")
            return {"CANCELLED"}
        self.sources = _sources(context)
        self.requests = queue.Queue()
        self.cancel = False
        self.progress = ("starting", 0.0)
        self.result = None
        self.error = None

        def run_on_main(fn):
            ev = threading.Event()
            box = {}
            self.requests.put((fn, ev, box))
            ev.wait()
            if "error" in box:
                raise box["error"]
            return box["result"]

        backend = self._backend(context, self.info, run_on_main)

        def progress(stage, frac):
            if self.cancel:
                raise Cancelled()
            self.progress = (stage or "", frac)

        def work():
            try:
                self.result = run_bake(self.bscene, self.settings, progress, backend)
            except Cancelled:
                self.error = "cancelled"
            except Exception as exc:
                traceback.print_exc()
                self.error = str(exc)

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        wm.progress_begin(0, 100)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            self.cancel = True
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        while not self.requests.empty():
            fn, ev, box = self.requests.get()
            try:
                box["result"] = fn()
            except Exception as exc:
                box["error"] = exc
            finally:
                ev.set()
        stage, frac = self.progress
        self.props.status = f"{stage} {frac * 100:.0f}%"
        context.window_manager.progress_update(int(frac * 100))
        for area in context.screen.areas if context.screen else []:
            if area.type == "PROPERTIES":
                area.tag_redraw()
        if self._thread.is_alive():
            return {"RUNNING_MODAL"}
        self._cleanup(context)
        if self.error:
            self.props.status = f"Error: {self.error}"
            self.report({"ERROR"} if self.error != "cancelled" else {"WARNING"}, f"Baked Engine: {self.error}")
            return {"CANCELLED"}
        try:
            _finish(self, context, self.props, self.result, self.info, self.sources)
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, f"Baked Engine: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}

    def _cleanup(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()


class BAKED_ENGINE_OT_show_sources(Operator):
    """Show the objects hidden by a previous bake"""

    bl_idname = "baked_engine.show_sources"
    bl_label = "Show Source Objects"

    def execute(self, context):
        for o in context.scene.objects:
            if o.get("baked_engine_hidden"):
                o.hide_set(False)
                o.hide_render = False
                del o["baked_engine_hidden"]
        return {"FINISHED"}


classes = (BAKED_ENGINE_OT_bake, BAKED_ENGINE_OT_show_sources)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
