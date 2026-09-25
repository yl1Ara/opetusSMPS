import copy
import importlib.util
import json
import shutil
import sys
import threading
import uuid
from pathlib import Path

import panel as pn

from DMPS_inversion_gui import online_app as global_app
from DMPS_inversion_gui.source_comparison import build_comparison_figure


APP_PATH = Path(__file__).resolve().parent / "DMPS_inversion_gui" / "online_app.py"
SESSION_SETTINGS_DIR = Path(__file__).resolve().parent / ".session_inversion_settings"
PROFILE_SETTINGS_DIR = Path.home() / ".local/share/opetusSMPS/inversion-settings"
SOURCE_PROFILE_KEYS = {
    "Bipolar Pi (CSC)": "bipolar-pi",
    "Monopolar Pi (CSC)": "monopolar-pi",
    "SMEAR III UFSMPS (CSC)": "smeariii-ufsmps",
    "SMEAR III SMPS (CSC)": "smeariii-smps",
}


def _profile_settings_file(profile_key, source_label=None):
    if profile_key not in {*SOURCE_PROFILE_KEYS.values(), "explorer"}:
        raise ValueError(f"Unknown inversion settings profile: {profile_key}")
    PROFILE_SETTINGS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    profile = PROFILE_SETTINGS_DIR / f"{profile_key}.json"
    if profile.exists():
        return profile

    # Migrate the most recently edited tab from the previous session-only
    # scheme. Keep the old file as an additional backup.
    if source_label is not None and SESSION_SETTINGS_DIR.exists():
        sessions = sorted(
            SESSION_SETTINGS_DIR.glob("settings_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for session in sessions:
            try:
                settings = json.loads(session.read_text())
            except (OSError, ValueError):
                continue
            if global_app.scan_source_for_root(settings.get("scan_root", "")) == source_label:
                shutil.copyfile(session, profile)
                break

    if not profile.exists():
        if global_app.SETTINGS_FILE.exists():
            shutil.copyfile(global_app.SETTINGS_FILE, profile)
        else:
            profile.write_text(json.dumps(global_app.DEFAULT_SETTINGS, indent=2))
    profile.chmod(0o600)
    return profile


def _comparison_settings_file():
    PROFILE_SETTINGS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    return PROFILE_SETTINGS_DIR / "comparison.json"


def _save_comparison_settings(path, method, polarity, maximum):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps({
            "method": method, "polarity": polarity, "max_concentration": maximum,
        }, indent=2))
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_session_app(initial_scan_source=None, profile_key="explorer"):
    session_id = uuid.uuid4().hex
    module_name = f"DMPS_inversion_gui.online_app_session_{session_id}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    module.SETTINGS_FILE = _profile_settings_file(profile_key, initial_scan_source)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

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
            "aerosol_fig": None,
            "difference_fig": None,
            "difference_diagnostics": None,
            "growth_diagnostics": [],
            "growth_signal_fig": None,
            "growth_settings": {},
            "aerosol_properties": [],
            "latest_inversion": None,
            "status": "Status: idle",
        },
    )
    module.local_shared_version = 0

    if initial_scan_source is not None and module.scan_source.value != initial_scan_source:
        module.scan_source.value = initial_scan_source
    module.save_settings()

    def cleanup_session(session_context):
        for callback_name in ("auto_callback", "shared_sync_callback"):
            callback = getattr(module, callback_name, None)
            if callback is not None:
                callback.stop()
        module.inversion_executor.shutdown(wait=False, cancel_futures=True)
        pn.state.cache.pop(module.SHARED_STATE_KEY, None)
        sys.modules.pop(module_name, None)

    pn.state.on_session_destroyed(cleanup_session)
    return module


def _global_live_tab():
    status = pn.pane.Markdown("Status: waiting for global inversion state")
    metadata = pn.pane.Markdown()
    settings_json = pn.pane.JSON({}, depth=2, sizing_mode="stretch_width")
    raw_plot = pn.pane.Plotly(height=750, width=1300)
    inversion_plot = pn.pane.Plotly(width=1300)
    growth_signal_plot = pn.pane.Plotly(width=1300)
    growth_status = pn.pane.Markdown()
    roi_feedback = pn.pane.Markdown("Drag across heatmap cells to inspect an ROI.")
    roi_tool = pn.widgets.Select(
        name="ROI tool", options={"Rectangle": "select", "Freehand": "lasso"},
        value=global_app.roi_selection_tool.value,
    )
    residual_plot = pn.pane.Plotly(width=1300, height=1200)
    smps_timing_plot = pn.pane.Plotly(width=1300, height=1100)
    aerosol_plot = pn.pane.Plotly(width=1300, height=2000)
    difference_plot = pn.pane.Plotly(width=1300)
    modal_plot = pn.pane.Plotly(width=1000, height=650)
    modal_status = pn.pane.Markdown("Click an inverted heatmap to inspect its size distribution.")
    roi_plot = pn.pane.Plotly(width=1000, height=650)
    roi_growth_plot = pn.pane.Plotly(width=1000, height=450)
    roi_status = pn.pane.Markdown(
        "Draw a rectangle or freehand selection on the inversion heatmap to inspect an ROI."
    )
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

    local = {"version": -1, "result": None}

    def refresh(event=None):
        with global_app.shared_state["lock"]:
            version = global_app.shared_state.get("version", 0)
            status_text = global_app.shared_state.get("status", "Status: idle")
            growth_diagnostics = list(global_app.shared_state.get("growth_diagnostics", []))
            if version != local["version"]:
                raw_fig = copy.deepcopy(global_app.shared_state.get("raw_fig"))
                inversion_fig = copy.deepcopy(global_app.shared_state.get("inversion_fig"))
                growth_signal_fig = copy.deepcopy(global_app.shared_state.get("growth_signal_fig"))
                residual_fig = copy.deepcopy(global_app.shared_state.get("residual_fig"))
                smps_timing_fig = copy.deepcopy(global_app.shared_state.get("smps_timing_fig"))
                aerosol_fig = copy.deepcopy(global_app.shared_state.get("aerosol_fig"))
                difference_fig = copy.deepcopy(global_app.shared_state.get("difference_fig"))
                inversion_result = copy.deepcopy(global_app.shared_state.get("latest_inversion"))

        status.object = str(status_text)
        metadata.object = f"Version: `{global_app.APP_VERSION}`  |  Shared update: `{version}`"
        try:
            settings_json.object = global_app.load_settings()
        except Exception as exc:
            settings_json.object = {"error": str(exc)}

        if version == local["version"]:
            return
        local["version"] = version

        local["result"] = inversion_result
        if inversion_fig is not None:
            inversion_fig.update_layout(dragmode=roi_tool.value)
        if inversion_result is None:
            growth_status.object = "Run an inversion to evaluate growth tracks."
        elif growth_diagnostics:
            stationary = sum(bool(item.get("track_caveat")) for item in growth_diagnostics)
            growth_status.object = (
                f"Automatic growth tracks: **{len(growth_diagnostics)}**. "
                f"{stationary} D50 candidate(s) have a stationary component peak "
                "(possible broadening). The Growth Signal tab shows full "
                "enhancement and the component the tracker selected."
            )
        else:
            growth_status.object = "No automatic growth track accepted for the current result."
        raw_plot.object = raw_fig
        inversion_plot.object = inversion_fig
        growth_signal_plot.object = growth_signal_fig
        residual_plot.object = residual_fig
        smps_timing_plot.object = smps_timing_fig
        aerosol_plot.object = aerosol_fig
        difference_plot.object = difference_fig
        modal_plot.object = None
        modal_status.object = "Click an inverted heatmap to inspect its size distribution."
        roi_plot.object = None
        roi_growth_plot.object = None
        roi_status.object = "Draw on the inversion heatmap to inspect an ROI."
        roi_feedback.object = "Drag across heatmap cells to inspect an ROI."

    def load_controls(event=None):
        if controls_loaded["value"]:
            return
        controls_loaded["value"] = True
        controls_container.objects = [global_app.controls]

    refresh_button.on_click(refresh)
    load_controls_button.on_click(load_controls)

    def inspect_click(event):
        result = local["result"]
        if result is None:
            return
        analysis = global_app.analyze_heatmap_click(
            event.new,
            inversion_plot.object,
            result,
            global_app.modal_fit_modes.value,
            global_app.modal_fit_min_nm.value,
            global_app.modal_fit_max_nm.value,
        )
        global_app.render_modal_analysis(analysis, modal_plot, modal_status)

    inversion_plot.param.watch(inspect_click, "click_data")

    def inspect_selection(event):
        result = local["result"]
        if result is None:
            return
        analysis = global_app.analyze_heatmap_roi(
            event.new, inversion_plot.object, result,
            global_app.modal_fit_modes.value,
        )
        global_app.render_roi_analysis(
            analysis, plot_pane=roi_plot, status_pane=roi_status,
            growth_plot_pane=roi_growth_plot, store=False,
        )
        if analysis is not None:
            roi_feedback.object = "ROI selected — open the Selected ROI tab for distributions and growth."

    inversion_plot.param.watch(inspect_selection, "selected_data")

    def update_roi_tool(event):
        if inversion_plot.object is not None:
            inversion_plot.object.update_layout(dragmode=event.new)
            inversion_plot.param.trigger("object")

    roi_tool.param.watch(update_roi_tool, "value")
    refresh()
    pn.state.add_periodic_callback(refresh, period=2000, start=True)

    live_tabs = pn.Tabs(
        ("Current Inversion", pn.Column(pn.Row(roi_tool, roi_feedback), growth_status, inversion_plot)),
        ("Clicked Distribution", pn.Column(modal_status, modal_plot)),
        ("Aerosol Properties", pn.Column(aerosol_plot)),
        ("Current Raw Data", pn.Column(raw_plot)),
        ("Residuals", pn.Column(residual_plot)),
        ("SMPS Timing", pn.Column(smps_timing_plot)),
        ("Difference Diagnostics", pn.Column(difference_plot)),
        ("Global Controls", controls_container),
        ("Settings", pn.Column(settings_json)),
        ("Selected ROI", pn.Column(roi_status, roi_plot, roi_growth_plot)),
        ("Growth Signal", pn.Column(growth_status, growth_signal_plot)),
        dynamic=True,
    )

    def load_controls_on_tab(event):
        if event.new == 7:
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

    instrument_sources = (
        ("Bipolar Pi", "Bipolar Pi (CSC)"),
        ("Monopolar Pi", "Monopolar Pi (CSC)"),
        ("SMEAR III UFSMPS", "SMEAR III UFSMPS (CSC)"),
        ("SMEAR III SMPS", "SMEAR III SMPS (CSC)"),
    )
    source_apps = {}
    source_tabs = []
    source_loaders = {}
    for name, source_label in instrument_sources:
        container = pn.Column(
            pn.pane.Markdown(f"Load the **{name}** inversion controls on demand."),
            width=1400,
        )

        def load_source(event=None, name=name, source_label=source_label, container=container):
            if name in source_apps:
                return
            container.objects = [pn.pane.Markdown(f"Loading **{name}**...")]
            module = _load_session_app(
                initial_scan_source=source_label,
                profile_key=SOURCE_PROFILE_KEYS[source_label],
            )
            source_apps[name] = module
            container.objects = [module.start_app()]

        source_loaders[name] = load_source
        source_tabs.append((name, container))

    comparison_status = pn.pane.Markdown(
        "Invert scans in instrument tabs, then refresh this comparison. "
        "Each instrument retains its own settings and result."
    )
    comparison_plot = pn.pane.Plotly(width=1300)
    comparison_settings_file = _comparison_settings_file()
    try:
        comparison_settings = json.loads(comparison_settings_file.read_text())
    except (OSError, ValueError):
        comparison_settings = {}
    saved_method = comparison_settings.get("method")
    saved_polarity = comparison_settings.get("polarity")
    saved_maximum = comparison_settings.get("max_concentration")
    valid_maximum = (
        isinstance(saved_maximum, (int, float))
        and not isinstance(saved_maximum, bool)
        and 0 < saved_maximum < float("inf")
    )
    comparison_method = pn.widgets.Select(
        name="Charging model", options={label: method for method, label in global_app.INVERSION_METHODS.items()},
        value=saved_method if saved_method in global_app.INVERSION_METHODS else "gunn woessner mod",
    )
    comparison_polarity = pn.widgets.Select(
        name="DMA voltage sign", options=["positive", "negative"],
        value=saved_polarity if saved_polarity in {"positive", "negative"} else "positive",
    )
    comparison_clip = pn.widgets.FloatInput(
        name="Common color maximum (dN/dlog10Dp)",
        value=float(saved_maximum) if valid_maximum else 20000.0,
        step=1000.0,
    )

    def save_comparison(event=None):
        _save_comparison_settings(
            comparison_settings_file, comparison_method.value,
            comparison_polarity.value, comparison_clip.value,
        )

    for widget in (comparison_method, comparison_polarity, comparison_clip):
        widget.param.watch(save_comparison, "value")
    refresh_comparison_button = pn.widgets.Button(
        name="Refresh comparison", button_type="primary",
    )

    def refresh_comparison(event=None):
        results = {}
        time_offsets = {}
        for name, _ in instrument_sources:
            module = source_apps.get(name)
            if module is None:
                continue
            with module.shared_state["lock"]:
                latest = module.shared_state.get("latest_inversion")
            if latest is not None:
                results[name] = latest
                time_offsets[name] = float(module.smear_comparison_time_offset_sec.value)
        figure, message = build_comparison_figure(
            results, comparison_method.value, comparison_polarity.value,
            comparison_clip.value,
            time_offsets_sec=time_offsets,
        )
        comparison_plot.object = figure
        comparison_status.object = message

    refresh_comparison_button.on_click(refresh_comparison)
    comparison_tab = pn.Column(
        "# Cross-instrument comparison",
        pn.Row(comparison_method, comparison_polarity, comparison_clip, refresh_comparison_button),
        comparison_status, comparison_plot, width=1400,
    )

    tabs = pn.Tabs(
        ("Global Live", global_live),
        *source_tabs,
        ("Comparison", comparison_tab),
        ("My Explorer", explorer_container),
        dynamic=True,
    )

    def load_active_tab(event):
        index = event.new
        if 1 <= index <= len(instrument_sources):
            source_loaders[instrument_sources[index - 1][0]]()
        elif index == len(instrument_sources) + 1:
            refresh_comparison()
        elif index == len(instrument_sources) + 2:
            load_explorer()

    tabs.param.watch(load_active_tab, "active")
    return tabs


if __name__ == "__main__" and "--auto-worker" in sys.argv:
    global_app.run_auto_worker()
else:
    layout = start_multi_app() if pn.state.curdoc is not None else global_app.layout
    layout.servable()
