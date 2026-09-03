import copy
import importlib.util
import sys
import threading
import uuid
from pathlib import Path

import panel as pn

from DMPS_inversion_gui import online_app as global_app


APP_PATH = Path(__file__).resolve().parent / "DMPS_inversion_gui" / "online_app.py"
SESSION_SETTINGS_DIR = Path(__file__).resolve().parent / ".session_inversion_settings"


def _load_session_app():
    session_id = uuid.uuid4().hex
    module_name = f"DMPS_inversion_gui.online_app_session_{session_id}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    module.SHARED_STATE_KEY = f"online_inversion_viewer_session_state_{session_id}"
    module.shared_state = pn.state.cache.setdefault(
        module.SHARED_STATE_KEY,
        {
            "lock": threading.Lock(),
            "version": 0,
            "raw_fig": None,
            "inversion_fig": None,
            "residual_fig": None,
            "smps_timing_fig": None,
            "difference_fig": None,
            "difference_diagnostics": None,
            "latest_inversion": None,
            "status": "Status: idle",
        },
    )
    module.local_shared_version = 0

    SESSION_SETTINGS_DIR.mkdir(exist_ok=True)
    module.SETTINGS_FILE = SESSION_SETTINGS_DIR / f"settings_{session_id}.json"
    try:
        module.save_settings()
    except Exception:
        pass
    return module


def _global_live_tab():
    status = pn.pane.Markdown("Status: waiting for global inversion state")
    metadata = pn.pane.Markdown()
    settings_json = pn.pane.JSON({}, depth=2, sizing_mode="stretch_width")
    raw_plot = pn.pane.Plotly(height=750, width=1300)
    inversion_plot = pn.pane.Plotly(width=1300)
    residual_plot = pn.pane.Plotly(width=1300, height=1200)
    smps_timing_plot = pn.pane.Plotly(width=1300, height=1100)
    difference_plot = pn.pane.Plotly(width=1300)
    refresh_button = pn.widgets.Button(name="Refresh global view", button_type="primary")
    controls_status = pn.pane.Markdown(
        "Global controls are loaded on demand so the live page opens quickly."
    )
    load_controls_button = pn.widgets.Button(
        name="Load global controls",
        button_type="primary",
    )
    controls_container = pn.Column(controls_status, load_controls_button, width=1400)
    controls_loaded = {"value": False}

    local = {"version": -1}

    def refresh(event=None):
        with global_app.shared_state["lock"]:
            version = global_app.shared_state.get("version", 0)
            status_text = global_app.shared_state.get("status", "Status: idle")

        status.object = str(status_text)
        metadata.object = f"Version: `{global_app.APP_VERSION}`  |  Shared update: `{version}`"
        try:
            settings_json.object = global_app.load_settings()
        except Exception as exc:
            settings_json.object = {"error": str(exc)}

        if version == local["version"]:
            return
        local["version"] = version

        with global_app.shared_state["lock"]:
            raw_fig = copy.deepcopy(global_app.shared_state.get("raw_fig"))
            inversion_fig = copy.deepcopy(global_app.shared_state.get("inversion_fig"))
            residual_fig = copy.deepcopy(global_app.shared_state.get("residual_fig"))
            smps_timing_fig = copy.deepcopy(global_app.shared_state.get("smps_timing_fig"))
            difference_fig = copy.deepcopy(global_app.shared_state.get("difference_fig"))

        raw_plot.object = raw_fig
        inversion_plot.object = inversion_fig
        residual_plot.object = residual_fig
        smps_timing_plot.object = smps_timing_fig
        difference_plot.object = difference_fig

    def load_controls(event=None):
        if controls_loaded["value"]:
            return
        controls_loaded["value"] = True
        controls_container.objects = [global_app.controls]

    refresh_button.on_click(refresh)
    load_controls_button.on_click(load_controls)
    refresh()
    pn.state.add_periodic_callback(refresh, period=2000, start=True)

    live_tabs = pn.Tabs(
        ("Current Inversion", pn.Column(inversion_plot)),
        ("Current Raw Data", pn.Column(raw_plot)),
        ("Residuals", pn.Column(residual_plot)),
        ("SMPS Timing", pn.Column(smps_timing_plot)),
        ("Difference Diagnostics", pn.Column(difference_plot)),
        ("Global Controls", controls_container),
        ("Settings", pn.Column(settings_json)),
        dynamic=True,
    )

    def load_controls_on_tab(event):
        if event.new == 5:
            load_controls()

    live_tabs.param.watch(load_controls_on_tab, "active")

    return pn.Column(
        "# Global live inversion",
        pn.Row(refresh_button, status),
        metadata,
        live_tabs,
        width=1400,
    )


def start_multi_app():
    print(f"DMPS inversion viewer {global_app.APP_VERSION}: {APP_PATH}", flush=True)
    if not pn.state.cache.get("online_inversion_viewer_global_started", False):
        global_app.start_app()
        pn.state.cache["online_inversion_viewer_global_started"] = True

    global_live = _global_live_tab()

    explorer_status = pn.pane.Markdown(
        "Personal explorer is loaded on demand so the live page opens quickly."
    )
    load_explorer_button = pn.widgets.Button(
        name="Load my personal explorer",
        button_type="primary",
    )
    explorer_container = pn.Column(explorer_status, load_explorer_button, width=1400)
    explorer_loaded = {"value": False}

    def load_explorer(event=None):
        if explorer_loaded["value"]:
            return
        explorer_loaded["value"] = True
        explorer_status.object = "Loading personal explorer..."
        session_app = _load_session_app()
        explorer_container.objects = [session_app.start_app()]

    load_explorer_button.on_click(load_explorer)

    tabs = pn.Tabs(
        ("Global Live", global_live),
        ("My Explorer", explorer_container),
        dynamic=True,
    )

    def load_explorer_on_tab(event):
        if event.new == 1:
            load_explorer()

    tabs.param.watch(load_explorer_on_tab, "active")
    return tabs


if __name__ == "__main__" and "--auto-worker" in sys.argv:
    global_app.run_auto_worker()
else:
    layout = start_multi_app() if pn.state.curdoc is not None else global_app.layout
    layout.servable()
