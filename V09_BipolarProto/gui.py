import panel as pn
import pandas as pd
import time
import DmpsControl as ctl
import numpy as np
import csv
import json
from pathlib import Path
from datetime import datetime
from collections import deque
from plotly.subplots import make_subplots
import threading
import traceback
import os
import subprocess
import atexit
import shutil
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from DmpsControl.tuning import aerosol_factor_from_pressures

pn.config.session_key_func = lambda request: request.path

SETTINGS_FILE = Path("settings.json")
STATE_DIR = Path(os.environ.get("DMPS_STATE_DIR", ".")).resolve()
HEALTH_FILE = STATE_DIR / "health.json"
HEALTH_INTERVAL_SEC = 2.0

DEFAULT_SETTINGS = {
    "cpc_com_port": "/dev/ttyAMA0",
    "range1": [1, 40],
    "range1_sheath": 20,
    "range1_steps": 20,
    "range2": [20, 400],
    "range2_sheath": 5,
    "range2_steps": 20,
    "meas_time": 15,
    "sleep_time": 5,
    "cpc_poll_interval": 0.5,
    "ntot_every_n_scans": 1,
    "ntot_rest_time": 10,
    "n_scans_plot": 5,
    "settling_time": 10,
    "final_point_extra_hold": 0,
    "smps_plot_step_shift": 0,
    "hv_source": "Bipolar DAC",
    "spellman_port": "/dev/ttyUSB0",
    "spellman_baud": 9600,
    "spellman_max_voltage": 30000,
    "aerosol_flow_enabled": False,
    "aerosol_flow_i2c_bus": 1,
    "aerosol_flow_i2c_address": "0x25",
    "aerosol_flow_calibration": 0.15228426395939088,
    "sheath_pid_kp": 0.008,
    "sheath_pid_ki": 0.015,
    "sheath_pid_kd": 0.0,
    "sheath_tune_output_low_v": 1.0,
    "sheath_tune_output_high_v": 2.5,
    "sheath_tune_settle_sec": 20,
    "sheath_tune_step_sec": 40,
    "sheath_tune_sample_interval_sec": 0.5,
    "sheath_tune_min_response_lpm": 0.5,
    "cpc_diag_auto_enabled": False,
    "cpc_diag_interval_sec": 60,
    "cpc_diag_command": "RD",
    "polarity_switch_time": 0,
    "Bipolar_toggle": True,
    "Ntot_time": 60,
    "CPC_type": "3010",
}

hardware_executor = ThreadPoolExecutor(max_workers=1)
cpc_diag_executor = ThreadPoolExecutor(max_workers=1)
spellman_executor = ThreadPoolExecutor(max_workers=1)
tuning_executor = ThreadPoolExecutor(max_workers=1)
calibration_executor = ThreadPoolExecutor(max_workers=1)
git_executor = ThreadPoolExecutor(max_workers=1)

# Run manually or from a service with panel serve, for example:
# uv run panel serve gui.py --address 0.0.0.0 --port 5006

APP_VERSION = (Path(__file__).resolve().parent / "VERSION").read_text().strip()

flowmeter = None
blower = None
flow_controller = None
cpc = None
inletValve = None
dac = None
hv_device = None
active_hv_config = None
aerosol_flowmeter = None
aerosol_active_config = None
hv_target_voltage = 0.0
hv_runtime_status = "uninitialized"
hv_io_lock = threading.RLock()
last_scan_saved = None
last_runtime_error = None
last_runtime_error_lock = threading.Lock()

measurement_running = threading.Event()
measurement_thread = None
hardware_stop_pending = False
init_running = False
settings_file_lock = threading.Lock()
tuning_cancel_event = threading.Event()
tuning_running = threading.Event()
tuning_result_lock = threading.Lock()
pending_tuning_result = None
tool_ui_updates = deque()
calibration_running = threading.Event()
git_check_running = threading.Event()
last_git_check_time = 0.0
GIT_CHECK_INTERVAL_SEC = 15 * 60

pn.extension("plotly")

APP_STOP_EVENT_KEY = "tdmps_gui_app_stop_event"
FLOW_CONTROLLER_CACHE_KEY = "tdmps_gui_flow_controller"
TUNING_CANCEL_EVENT_KEY = "tdmps_gui_tuning_cancel_event"
previous_app_stop_event = pn.state.cache.get(APP_STOP_EVENT_KEY)
if previous_app_stop_event is not None:
    previous_app_stop_event.set()
previous_tuning_cancel_event = pn.state.cache.get(TUNING_CANCEL_EVENT_KEY)
if previous_tuning_cancel_event is not None:
    previous_tuning_cancel_event.set()
previous_flow_controller = pn.state.cache.get(FLOW_CONTROLLER_CACHE_KEY)
if previous_flow_controller is not None:
    try:
        previous_flow_controller.stop()
    except Exception:
        pass
app_stop_event = threading.Event()
pn.state.cache[APP_STOP_EVENT_KEY] = app_stop_event
pn.state.cache[TUNING_CANCEL_EVENT_KEY] = tuning_cancel_event

#### Widgets ####
cpc_com_port = pn.widgets.TextInput(name="CPC COM port", value="/dev/ttyAMA0")
cpc_type = pn.widgets.Select(name="CPC Type", options=["3010", "HY09"], value="3010")
hv_source = pn.widgets.Select(
    name="HV source",
    options=["Bipolar DAC", "Monopolar Spellman"],
    value=DEFAULT_SETTINGS["hv_source"],
    width=180,
)
spellman_port = pn.widgets.TextInput(
    name="Spellman port",
    value=DEFAULT_SETTINGS["spellman_port"],
    width=150,
)
spellman_baud = pn.widgets.IntInput(
    name="Spellman baud",
    value=DEFAULT_SETTINGS["spellman_baud"],
    step=1,
    width=130,
)
spellman_max_voltage = pn.widgets.IntInput(
    name="Spellman max V",
    value=DEFAULT_SETTINGS["spellman_max_voltage"],
    step=100,
    width=150,
)
aerosol_flow_enabled = pn.widgets.Checkbox(
    name="Enable aerosol flowmeter", value=DEFAULT_SETTINGS["aerosol_flow_enabled"]
)
aerosol_flow_i2c_bus = pn.widgets.IntInput(
    name="Aerosol I2C bus", value=DEFAULT_SETTINGS["aerosol_flow_i2c_bus"], width=130
)
aerosol_flow_i2c_address = pn.widgets.TextInput(
    name="Aerosol I2C address", value=DEFAULT_SETTINGS["aerosol_flow_i2c_address"], width=150
)
aerosol_flow_calibration = pn.widgets.FloatInput(
    name="Aerosol calibration (L/min/Pa)",
    value=DEFAULT_SETTINGS["aerosol_flow_calibration"],
    step=0.001,
    width=210,
)

sheath_pid_kp = pn.widgets.FloatInput(name="Current Kp", value=DEFAULT_SETTINGS["sheath_pid_kp"], disabled=True, width=140)
sheath_pid_ki = pn.widgets.FloatInput(name="Current Ki", value=DEFAULT_SETTINGS["sheath_pid_ki"], disabled=True, width=140)
sheath_pid_kd = pn.widgets.FloatInput(name="Current Kd", value=DEFAULT_SETTINGS["sheath_pid_kd"], disabled=True, width=140)
sheath_tune_output_low_v = pn.widgets.FloatInput(name="Low DAC (V)", value=DEFAULT_SETTINGS["sheath_tune_output_low_v"], step=0.1, width=130)
sheath_tune_output_high_v = pn.widgets.FloatInput(name="High DAC (V)", value=DEFAULT_SETTINGS["sheath_tune_output_high_v"], step=0.1, width=130)
sheath_tune_settle_sec = pn.widgets.FloatInput(name="Low settle (s)", value=DEFAULT_SETTINGS["sheath_tune_settle_sec"], step=1, width=130)
sheath_tune_step_sec = pn.widgets.FloatInput(name="High step (s)", value=DEFAULT_SETTINGS["sheath_tune_step_sec"], step=1, width=130)
sheath_tune_sample_interval_sec = pn.widgets.FloatInput(name="Sample interval (s)", value=DEFAULT_SETTINGS["sheath_tune_sample_interval_sec"], step=0.1, width=150)
sheath_tune_min_response_lpm = pn.widgets.FloatInput(name="Min response (L/min)", value=DEFAULT_SETTINGS["sheath_tune_min_response_lpm"], step=0.1, width=170)
sheath_tune_start_button = pn.widgets.Button(name="Run sheath step tuner", button_type="warning")
sheath_tune_cancel_button = pn.widgets.Button(name="Cancel tuning", button_type="danger", disabled=True)
sheath_tune_apply_button = pn.widgets.Button(name="Apply result", button_type="success", disabled=True)
sheath_tune_progress = pn.indicators.Progress(name="Tuning progress", value=0, max=100, width=350)
sheath_tune_status = pn.pane.Markdown("Sheath tuning: idle")

aerosol_calibration_confirm = pn.widgets.Checkbox(
    name="External actual aerosol flow is already 1.0 L/min", value=False
)
aerosol_calibration_samples = pn.widgets.IntInput(name="Pressure samples", value=20, start=5, step=1, width=140)
aerosol_calibration_button = pn.widgets.Button(name="Calibrate aerosol flow", button_type="warning")
aerosol_calibration_status = pn.pane.Markdown("Aerosol calibration: idle")

start_button = pn.widgets.Toggle(name="Start measurement", button_type="success")
init_button = pn.widgets.Button(name="Initialize hardware", button_type="primary")
stop_button = pn.widgets.Button(name="Stop and zero HV", button_type="danger")
Bipolar_toggle = pn.widgets.Toggle(
    name="Bipolar scan", button_type="primary", value=True
)


n_scans_plot = pn.widgets.IntInput(
    name="Number of completed scans to plot", value=3, step=1
)

range1 = pn.widgets.ArrayInput(
    name="Range 1 [min,max] nm",
    value=np.array([1, 40]),
    max_array_size=2,
)
sheath1 = pn.widgets.IntInput(name="Sheath 1 (L/min)", value=20, step=1)
steps1 = pn.widgets.IntInput(name="Steps 1", value=20, step=1)

range2 = pn.widgets.ArrayInput(
    name="Range 2 [min,max] nm",
    value=np.array([20, 400]),
    max_array_size=2,
)
sheath2 = pn.widgets.IntInput(name="Sheath 2 (L/min)", value=5, step=1)
steps2 = pn.widgets.IntInput(name="Steps 2", value=20, step=1)


meas_time = pn.widgets.IntInput(name="Measurement time per size (s)", value=15, step=1)
sleep_time = pn.widgets.IntInput(name="Legacy sleep time (s)", value=5, step=1)
cpc_poll_interval = pn.widgets.FloatInput(
    name="CPC poll interval (s)",
    value=DEFAULT_SETTINGS["cpc_poll_interval"],
    step=0.1,
    width=150,
)
Ntot_time = pn.widgets.IntInput(name="Ntot measurement time (s)", value=60, step=1)
ntot_every_n_scans = pn.widgets.IntInput(
    name="Ntot every N scans (0 off)",
    value=DEFAULT_SETTINGS["ntot_every_n_scans"],
    step=1,
    width=170,
)
ntot_rest_time = pn.widgets.IntInput(
    name="Ntot rest time (s)",
    value=DEFAULT_SETTINGS["ntot_rest_time"],
    step=1,
    width=150,
)
settling_time = pn.widgets.IntInput(
    name="Settling time between size changes (s) ", value=10, step=1
)
final_point_extra_hold = pn.widgets.IntInput(
    name="Final point extra hold (s)",
    value=DEFAULT_SETTINGS["final_point_extra_hold"],
    step=1,
    width=170,
)
smps_plot_step_shift = pn.widgets.IntInput(
    name="SMPS plot step shift",
    value=DEFAULT_SETTINGS["smps_plot_step_shift"],
    step=1,
    width=170,
)
polarity_switch_time = pn.widgets.IntInput(
    name="Polarity switch time (s) ", value=0, step=1
)
status_text = pn.pane.Markdown("Status: idle")
git_update_status = pn.pane.Alert(
    "Software update: checking Git...", alert_type="info", sizing_mode="stretch_width"
)
git_update_button = pn.widgets.Button(name="Check updates", button_type="light", width=130)
current_cpc_pane = pn.pane.Markdown("Current CPC: -")
current_flow_pane = pn.pane.Markdown("Current sheath flow: -")
current_hv_pane = pn.pane.Markdown("Current HV: -")
current_aerosol_flow_pane = pn.pane.Markdown("Current aerosol flow: disabled")
latest_ntot_pane = pn.pane.Markdown("Latest Ntot: -")
scan_progress_pane = pn.pane.Markdown("Scan progress: idle")
pending_settings_pane = pn.pane.Markdown("Settings: current")
last_row_pane = pn.pane.Str("Last measurement: -")
scan_pane = pn.pane.Str("Scan program: -")

CPC_DIAGNOSTIC_COMMANDS = {
    "3010": {
        "Read liquid status (R0)": "R0",
        "Read condenser temperature (R1)": "R1",
        "Read saturator temperature (R2)": "R2",
        "Read readiness (R5)": "R5",
        "Read 6-second count (RA)": "RA",
        "Read 1-second count (RB)": "RB",
        "Read concentration (RD)": "RD",
        "Read temperature difference (RT)": "RT",
        "Read vacuum status (RV)": "RV",
        "Custom": "",
    },
    "HY09": {
        "Read concentration (RB)": "RB",
        "Custom": "",
    },
}
cpc_diag_command = pn.widgets.Select(
    name="CPC command",
    options=CPC_DIAGNOSTIC_COMMANDS["3010"],
    value=DEFAULT_SETTINGS["cpc_diag_command"],
    width=220,
)
cpc_diag_custom_command = pn.widgets.TextInput(name="Custom CPC command", value="", width=180)
cpc_diag_read_lines = pn.widgets.IntInput(name="Read lines", value=1, step=1, width=110)
cpc_diag_send_button = pn.widgets.Button(name="Send CPC command", button_type="primary")
cpc_diag_auto_enabled = pn.widgets.Checkbox(
    name="Auto log CPC diagnostics",
    value=DEFAULT_SETTINGS["cpc_diag_auto_enabled"],
)
cpc_diag_interval_sec = pn.widgets.IntInput(
    name="Auto interval (s)",
    value=DEFAULT_SETTINGS["cpc_diag_interval_sec"],
    step=1,
    width=130,
)
cpc_diag_status = pn.pane.Markdown("CPC diagnostics: idle")


def update_cpc_diagnostic_options(event=None):
    options = CPC_DIAGNOSTIC_COMMANDS[cpc_type.value]
    previous = cpc_diag_command.value
    cpc_diag_command.options = options
    cpc_diag_command.value = previous if previous in options.values() else next(iter(options.values()))
cpc_diag_table = pn.widgets.DataFrame(
    pd.DataFrame(columns=["time", "command", "response", "error"]),
    height=260,
    width=1100,
)

table_pane = pn.widgets.DataFrame(
    pd.DataFrame(
        columns=[
            "time",
            "scan_range",
            "size_nm",
            "cpc_count",
            "sheath_flow",
            "sheath_setpoint",
        ]
    ),
    height=220,
    width=900,
)
qc_table = pn.widgets.DataFrame(
    pd.DataFrame(columns=["scan_number", "result", "completeness", "unique_samples", "repeated_fraction"]),
    height=220,
    width=1450,
)

rows = []
current_size_index = 0
phase = "idle"
phase_start_time = time.time()
point_set_time = phase_start_time
scan_rows = []
completed_scans = []
scan_number = 0
active_point_key = None
next_sample_time = 0.0
last_point_set_duration_sec = np.nan
scan_program_cache_key = None
scan_program_cache = []
active_scan_settings = None
scan_started_monotonic = None
scan_started_wall = None
scan_serial_error_start = 0
final_hold_start_sample_id = 0
final_hold_sample_ids = set()
completed_qc_rows = deque(maxlen=100)
ui_rows_pending = deque(maxlen=500)
cpc_diag_rows = deque(maxlen=200)
cpc_diag_ui_pending = deque(maxlen=50)
cpc_diag_query_pending = False
last_cpc_diag_auto_time = 0.0
last_plot_update_time = 0.0
PLOT_REFRESH_INTERVAL_SEC = 1.0
latest_cpc_lock = threading.Lock()
latest_cpc = {
    "value": np.nan,
    "time": None,
    "duration_sec": np.nan,
    "error": None,
    "sample_id": 0,
    "sample_time": None,
}
cpc_reader_thread = None
cpc_reader_stop = threading.Event()
ui_callback = None
spellman_cache_lock = threading.Lock()
spellman_cache = {"voltage": np.nan, "status": None, "time": None, "error": None, "pending": False, "serial_errors": 0}
last_spellman_query_time = 0.0
aerosol_cache_lock = threading.Lock()
aerosol_cache = {"flow": np.nan, "pressure": np.nan, "temperature": np.nan, "error": None}
aerosol_poll_thread = None
health_thread = None


def record_runtime_error(error):
    global last_runtime_error
    with last_runtime_error_lock:
        last_runtime_error = str(error)


def ensure_settings_file():
    if not SETTINGS_FILE.exists():
        atomic_write_json(SETTINGS_FILE, DEFAULT_SETTINGS)


def atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w") as f:
        json.dump(value, f, indent=2)
        f.flush()
    temporary.replace(path)


def _int_list(value):
    return [int(x) for x in np.array(value).ravel()]


def save_settings():
    settings = {
        "cpc_com_port": str(cpc_com_port.value),
        "range1": _int_list(range1.value),
        "range1_sheath": int(sheath1.value),
        "range1_steps": int(steps1.value),
        "range2": _int_list(range2.value),
        "range2_sheath": int(sheath2.value),
        "range2_steps": int(steps2.value),
        "meas_time": int(meas_time.value),
        "sleep_time": int(sleep_time.value),
        "cpc_poll_interval": float(cpc_poll_interval.value),
        "ntot_every_n_scans": int(ntot_every_n_scans.value),
        "ntot_rest_time": int(ntot_rest_time.value),
        "n_scans_plot": int(n_scans_plot.value),
        "settling_time": int(settling_time.value),
        "final_point_extra_hold": int(final_point_extra_hold.value),
        "smps_plot_step_shift": int(smps_plot_step_shift.value),
        "hv_source": str(hv_source.value),
        "spellman_port": str(spellman_port.value),
        "spellman_baud": int(spellman_baud.value),
        "spellman_max_voltage": int(spellman_max_voltage.value),
        "aerosol_flow_enabled": bool(aerosol_flow_enabled.value),
        "aerosol_flow_i2c_bus": int(aerosol_flow_i2c_bus.value),
        "aerosol_flow_i2c_address": str(aerosol_flow_i2c_address.value),
        "aerosol_flow_calibration": float(aerosol_flow_calibration.value),
        "sheath_pid_kp": float(sheath_pid_kp.value),
        "sheath_pid_ki": float(sheath_pid_ki.value),
        "sheath_pid_kd": float(sheath_pid_kd.value),
        "sheath_tune_output_low_v": float(sheath_tune_output_low_v.value),
        "sheath_tune_output_high_v": float(sheath_tune_output_high_v.value),
        "sheath_tune_settle_sec": float(sheath_tune_settle_sec.value),
        "sheath_tune_step_sec": float(sheath_tune_step_sec.value),
        "sheath_tune_sample_interval_sec": float(sheath_tune_sample_interval_sec.value),
        "sheath_tune_min_response_lpm": float(sheath_tune_min_response_lpm.value),
        "cpc_diag_auto_enabled": bool(cpc_diag_auto_enabled.value),
        "cpc_diag_interval_sec": int(cpc_diag_interval_sec.value),
        "cpc_diag_command": str(cpc_diag_command.value),
        "polarity_switch_time": int(polarity_switch_time.value),
        "Bipolar_toggle": bool(Bipolar_toggle.value),
        "Ntot_time": int(Ntot_time.value),
        "CPC_type": str(cpc_type.value),
    }

    with settings_file_lock:
        atomic_write_json(SETTINGS_FILE, settings)


def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
    except json.JSONDecodeError:
        broken = SETTINGS_FILE.with_name("settings_broken.json")
        SETTINGS_FILE.rename(broken)
        ensure_settings_file()
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)

    settings_changed = False
    new_setting_keys = (
        "sheath_pid_kp", "sheath_pid_ki", "sheath_pid_kd",
        "sheath_tune_output_low_v", "sheath_tune_output_high_v",
        "sheath_tune_settle_sec", "sheath_tune_step_sec",
        "sheath_tune_sample_interval_sec", "sheath_tune_min_response_lpm",
    )
    if any(key not in settings for key in new_setting_keys):
        settings.update({key: DEFAULT_SETTINGS[key] for key in new_setting_keys if key not in settings})
        settings_changed = True

    # Earlier releases mislabeled the 3010's 9600/7-E-1 protocol as "3771".
    if settings.get("CPC_type") == "3771":
        settings["CPC_type"] = "3010"
        settings_changed = True

    if settings_changed:
        with settings_file_lock:
            atomic_write_json(SETTINGS_FILE, settings)

    cpc_com_port.value = settings.get("cpc_com_port", DEFAULT_SETTINGS["cpc_com_port"])
    cpc_type.value = settings.get("CPC_type", DEFAULT_SETTINGS["CPC_type"])
    update_cpc_diagnostic_options()

    range1.value = np.array(settings.get("range1", DEFAULT_SETTINGS["range1"]))
    sheath1.value = settings.get("range1_sheath", DEFAULT_SETTINGS["range1_sheath"])
    steps1.value = settings.get("range1_steps", DEFAULT_SETTINGS["range1_steps"])

    range2.value = np.array(settings.get("range2", DEFAULT_SETTINGS["range2"]))
    sheath2.value = settings.get("range2_sheath", DEFAULT_SETTINGS["range2_sheath"])
    steps2.value = settings.get("range2_steps", DEFAULT_SETTINGS["range2_steps"])

    meas_time.value = settings.get("meas_time", DEFAULT_SETTINGS["meas_time"])
    sleep_time.value = settings.get("sleep_time", DEFAULT_SETTINGS["sleep_time"])
    cpc_poll_interval.value = settings.get(
        "cpc_poll_interval",
        DEFAULT_SETTINGS["cpc_poll_interval"],
    )
    ntot_every_n_scans.value = settings.get(
        "ntot_every_n_scans",
        DEFAULT_SETTINGS["ntot_every_n_scans"],
    )
    ntot_rest_time.value = settings.get(
        "ntot_rest_time",
        DEFAULT_SETTINGS["ntot_rest_time"],
    )
    n_scans_plot.value = settings.get("n_scans_plot", DEFAULT_SETTINGS["n_scans_plot"])
    Bipolar_toggle.value = settings.get(
        "Bipolar_toggle", DEFAULT_SETTINGS["Bipolar_toggle"]
    )

    settling_time.value = settings.get(
        "settling_time", DEFAULT_SETTINGS["settling_time"]
    )
    final_point_extra_hold.value = settings.get(
        "final_point_extra_hold",
        DEFAULT_SETTINGS["final_point_extra_hold"],
    )
    smps_plot_step_shift.value = settings.get(
        "smps_plot_step_shift",
        int(round(settings.get("smps_plot_time_shift_sec", DEFAULT_SETTINGS["smps_plot_step_shift"]))),
    )
    polarity_switch_time.value = settings.get(
        "polarity_switch_time", DEFAULT_SETTINGS["polarity_switch_time"]
    )
    Ntot_time.value = settings.get("Ntot_time", DEFAULT_SETTINGS["Ntot_time"])
    hv_source.value = settings.get("hv_source", DEFAULT_SETTINGS["hv_source"])
    spellman_port.value = settings.get("spellman_port", DEFAULT_SETTINGS["spellman_port"])
    spellman_baud.value = settings.get("spellman_baud", DEFAULT_SETTINGS["spellman_baud"])
    spellman_max_voltage.value = settings.get(
        "spellman_max_voltage",
        DEFAULT_SETTINGS["spellman_max_voltage"],
    )
    aerosol_flow_enabled.value = settings.get("aerosol_flow_enabled", DEFAULT_SETTINGS["aerosol_flow_enabled"])
    aerosol_flow_i2c_bus.value = settings.get("aerosol_flow_i2c_bus", DEFAULT_SETTINGS["aerosol_flow_i2c_bus"])
    aerosol_flow_i2c_address.value = settings.get("aerosol_flow_i2c_address", DEFAULT_SETTINGS["aerosol_flow_i2c_address"])
    aerosol_flow_calibration.value = settings.get("aerosol_flow_calibration", DEFAULT_SETTINGS["aerosol_flow_calibration"])
    sheath_pid_kp.value = settings.get("sheath_pid_kp", DEFAULT_SETTINGS["sheath_pid_kp"])
    sheath_pid_ki.value = settings.get("sheath_pid_ki", DEFAULT_SETTINGS["sheath_pid_ki"])
    sheath_pid_kd.value = settings.get("sheath_pid_kd", DEFAULT_SETTINGS["sheath_pid_kd"])
    sheath_tune_output_low_v.value = settings.get("sheath_tune_output_low_v", DEFAULT_SETTINGS["sheath_tune_output_low_v"])
    sheath_tune_output_high_v.value = settings.get("sheath_tune_output_high_v", DEFAULT_SETTINGS["sheath_tune_output_high_v"])
    sheath_tune_settle_sec.value = settings.get("sheath_tune_settle_sec", DEFAULT_SETTINGS["sheath_tune_settle_sec"])
    sheath_tune_step_sec.value = settings.get("sheath_tune_step_sec", DEFAULT_SETTINGS["sheath_tune_step_sec"])
    sheath_tune_sample_interval_sec.value = settings.get("sheath_tune_sample_interval_sec", DEFAULT_SETTINGS["sheath_tune_sample_interval_sec"])
    sheath_tune_min_response_lpm.value = settings.get("sheath_tune_min_response_lpm", DEFAULT_SETTINGS["sheath_tune_min_response_lpm"])
    cpc_diag_auto_enabled.value = settings.get(
        "cpc_diag_auto_enabled",
        DEFAULT_SETTINGS["cpc_diag_auto_enabled"],
    )
    cpc_diag_interval_sec.value = settings.get(
        "cpc_diag_interval_sec",
        DEFAULT_SETTINGS["cpc_diag_interval_sec"],
    )
    saved_cpc_diag_command = settings.get(
        "cpc_diag_command",
        DEFAULT_SETTINGS["cpc_diag_command"],
    )
    cpc_diag_command.value = (
        saved_cpc_diag_command
        if saved_cpc_diag_command in cpc_diag_command.options.values()
        else next(iter(cpc_diag_command.options.values()))
    )

ensure_settings_file()
load_settings()


def bipolar_log_sizes(size_range_value, n, order="negative_then_positive", bipolar=None, source=None):
    lo, hi = np.array(size_range_value).ravel().astype(float)
    lo, hi = abs(lo), abs(hi)

    if lo <= 0 or hi <= 0:
        raise ValueError("Use positive nonzero limits, e.g. [20, 400]")
    if hi < lo:
        lo, hi = hi, lo
    if int(n) < 2:
        raise ValueError("steps must be >= 2")

    pos = np.logspace(np.log10(lo), np.log10(hi), int(n)).tolist()
    bipolar = Bipolar_toggle.value if bipolar is None else bipolar
    source = hv_source.value if source is None else source
    if source == "Bipolar DAC" and bipolar:
        neg = [-x for x in pos]
    else:
        neg = []

    if order == "positive_then_negative":
        return pos + neg
    return neg + pos


def save_completed_scan(scan_rows, scan_number):
    global last_scan_saved
    if not scan_rows:
        return

    t0 = pd.to_datetime(scan_rows[0]["time"])
    scan_id = t0.strftime("%Y%m%d_%H%M%S")

    run_day = t0.strftime("%Y%m%d")
    path = Path("logs/scans") / run_day / f"{scan_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")
    pd.DataFrame(scan_rows).to_csv(temporary_path, index=False)
    temporary_path.replace(path)
    last_scan_saved = str(path)
    print(f"Saved completed scan: {path}", flush=True)


def get_recent_completed_scans(n=None):

    if n is None:
        n = int(n_scans_plot.value)

    root = Path("logs/scans")
    if not root.exists():
        print(f"No scan root found: {root}", flush=True)
        return pd.DataFrame()

    csv_files = sorted(
        root.glob("*/*.csv"),
        key=lambda p: p.stem,
    )[-n:]

    dfs = []
    for f in csv_files:
        try:
            d = pd.read_csv(f)
            d["scan_id"] = f.stem
            dfs.append(d)
        except Exception as e:
            print(f"Could not read scan file {f}: {e}", flush=True)

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame()


def load_initial_scans_to_table():
    df0 = get_recent_completed_scans(int(n_scans_plot.value))
    if df0 is not None and not df0.empty:
        table_pane.value = df0.tail(100)
    else:
        print("No completed scans found on startup", flush=True)


def current_scan_settings():
    return {
        "range1": np.array(range1.value, dtype=float).tolist(), "sheath1": float(sheath1.value), "steps1": int(steps1.value),
        "range2": np.array(range2.value, dtype=float).tolist(), "sheath2": float(sheath2.value), "steps2": int(steps2.value),
        "meas_time": float(meas_time.value), "cpc_poll_interval": float(cpc_poll_interval.value),
        "settling_time": float(settling_time.value), "final_point_extra_hold": float(final_point_extra_hold.value),
        "polarity_switch_time": float(polarity_switch_time.value), "bipolar": bool(Bipolar_toggle.value),
        "hv_source": str(hv_source.value), "spellman_port": str(spellman_port.value),
        "spellman_baud": int(spellman_baud.value), "spellman_max_voltage": float(spellman_max_voltage.value),
        "ntot_time": float(Ntot_time.value), "ntot_every": int(ntot_every_n_scans.value),
        "ntot_rest_time": float(ntot_rest_time.value),
        "aerosol_flow_enabled": bool(aerosol_flow_enabled.value),
        "aerosol_flow_i2c_bus": int(aerosol_flow_i2c_bus.value),
        "aerosol_flow_i2c_address": str(aerosol_flow_i2c_address.value),
        "aerosol_flow_calibration": float(aerosol_flow_calibration.value),
    }


def build_scan_points(include_dac_codes=False, settings=None):
    settings = settings or current_scan_settings()
    scan = []

    for scan_range, limits, sheath, steps in [
        (1, settings["range1"], settings["sheath1"], settings["steps1"]),
        (2, settings["range2"], settings["sheath2"], settings["steps2"]),
    ]:
        minimum_steps = 1 if scan_range == 1 else 2
        if steps > minimum_steps:
            sizes = bipolar_log_sizes(
                limits, steps, bipolar=settings["bipolar"], source=settings["hv_source"]
            )
            for dp in sizes:
                point = {"scan_range": scan_range, "dp": float(dp), "sheath": sheath}
                if include_dac_codes and settings["hv_source"] == "Bipolar DAC":
                    point["dac_code"] = ctl.HV.dac_code_from_size(float(dp), Q_sh_lpm=sheath)
                point["hv_target_v"] = abs(ctl.HV.voltage_from_size(float(dp), Q_sh_lpm=sheath)) if settings["hv_source"] == "Monopolar Spellman" else ctl.HV.voltage_from_size(float(dp), Q_sh_lpm=sheath)
                scan.append(point)

    return scan


def get_scan_program():
    global scan_program_cache_key, scan_program_cache

    cache_key = (
        tuple(np.array(range1.value).ravel().astype(float)),
        int(steps1.value),
        float(sheath1.value),
        tuple(np.array(range2.value).ravel().astype(float)),
        int(steps2.value),
        float(sheath2.value),
        bool(Bipolar_toggle.value),
        hv_source.value,
    )
    if scan_program_cache_key == cache_key:
        return scan_program_cache

    scan = build_scan_points(include_dac_codes=True)
    scan_program_cache_key = cache_key
    scan_program_cache = scan
    return scan


def update_scan_preview():
    global scan_program_cache_key
    scan_program_cache_key = None
    try:
        scan = build_scan_points(include_dac_codes=False)
        sizes = [f"{p['dp']:.3g}" for p in scan]
        scan_pane.object = f"Scan points ({len(sizes)}): {sizes}"
    except Exception as e:
        scan_pane.object = f"Scan parse error: {e}"


def set_status_threadsafe(text):
    doc = pn.state.curdoc
    if doc is not None:
        doc.add_next_tick_callback(lambda: setattr(status_text, "object", text))
    else:
        status_text.object = text


def setup_hv_source(settings=None):
    global hv_device, active_hv_config, hv_target_voltage, hv_runtime_status

    settings = settings or current_scan_settings()
    if settings["hv_source"] == "Monopolar Spellman":
        config = (
            settings["hv_source"], settings["spellman_port"],
            settings["spellman_baud"], settings["spellman_max_voltage"],
        )
    else:
        config = (settings["hv_source"],)
    if active_hv_config == config:
        return

    if active_hv_config and active_hv_config[0] == "Monopolar Spellman" and hv_device is not None:
        try:
            hv_device.zero()
            hv_device.disable()
        except Exception as e:
            print(f"Previous Spellman shutdown failed: {e}", flush=True)
    elif active_hv_config and active_hv_config[0] == "Bipolar DAC":
        try:
            ctl.HV.zero()
            ctl.HV.cleanup()
        except Exception as e:
            print(f"Previous bipolar HV shutdown failed: {e}", flush=True)

    if settings["hv_source"] == "Monopolar Spellman":
        hv_device = ctl.SpellmanHV(
            port=settings["spellman_port"], baud=settings["spellman_baud"],
            max_voltage=settings["spellman_max_voltage"],
        )
        try:
            hv_device.clear_faults()
        except Exception as e:
            print(f"Spellman clear faults failed: {e}", flush=True)
        hv_device.enable()
        hv_device.zero()
        hv_target_voltage = 0.0
        hv_runtime_status = "zeroed/enabled"
        active_hv_config = config
        return

    hv_device = None
    ctl.setup()
    ctl.HV.zero()
    hv_target_voltage = 0.0
    hv_runtime_status = "zeroed"
    active_hv_config = config


def zero_hv(disable=False):
    global hv_target_voltage, hv_runtime_status

    with hv_io_lock:
        if active_hv_config and active_hv_config[0] == "Monopolar Spellman" and hv_device is not None:
            hv_device.zero()
            if disable:
                try:
                    hv_device.disable()
                except Exception as e:
                    print(f"Spellman disable failed: {e}", flush=True)
            hv_target_voltage = 0.0
            hv_runtime_status = "zeroed/disabled" if disable else "zeroed"
            return

        ctl.HV.zero()
        hv_target_voltage = 0.0
        hv_runtime_status = "zeroed"


def set_hv_for_point(point):
    global hv_target_voltage, hv_runtime_status
    with hv_io_lock:
        if app_stop_event.is_set() or not measurement_running.is_set():
            return
        dp = point["dp"]
        q_sheath = point["sheath"]

        if active_scan_settings and active_scan_settings["hv_source"] == "Monopolar Spellman":
            if hv_device is None:
                raise RuntimeError("Monopolar Spellman HV is not initialized")
            hv_device.voltage_set(abs(float(dp)), Q_sh_lpm=q_sheath)
            hv_target_voltage = abs(float(point.get("hv_target_v", 0.0)))
            hv_runtime_status = "enabled"
            return

        if "dac_code" in point:
            ctl.HV.write_dac8551(point["dac_code"])
        else:
            ctl.HV.voltage_set(dp, Q_sh_lpm=q_sheath)
        hv_target_voltage = float(point.get("hv_target_v", 0.0))
        hv_runtime_status = "enabled"


def hardware_stop_and_zero():
    global hardware_stop_pending
    import traceback as _traceback

    try:
        try:
            if inletValve is not None:
                inletValve.off()
        except Exception as e:
            print(f"Valve off failed during stop: {e}", flush=True)

        try:
            zero_hv(disable=True)
        except OSError:
            ctl.setup()
            zero_hv(disable=True)

        set_status_threadsafe("Status: stopped, HV zeroed")
    except Exception as e:
        _traceback.print_exc()
        record_runtime_error(e)
        set_status_threadsafe(f"Stop/zero failed: {e}")
    finally:
        hardware_stop_pending = False


def stop_and_zero():
    global phase, current_size_index, phase_start_time, dac, active_point_key, hardware_stop_pending

    measurement_running.clear()
    tuning_cancel_event.set()

    if start_button.value:
        start_button.value = False

    phase = "idle"
    current_size_index = 0
    phase_start_time = time.time()
    active_point_key = None
    status_text.object = "Status: stopping, zeroing HV..."

    if not hardware_stop_pending:
        hardware_stop_pending = True
        hardware_executor.submit(hardware_stop_and_zero)


def init_hardware_blocking():
    global flowmeter, blower, flow_controller, cpc, inletValve, dac, hv_device, aerosol_flowmeter
    import time as _time

    if flow_controller is not None:
        return

    #dac = ctl.DacOut()
    #dac.block()
    flowmeter = ctl.Flowmeter()
    blower = ctl.BlowerDAC()
    cpc = ctl.CPC(cpc_com_port.value, cpc_type.value)
    inletValve = ctl.PicoValve()

    setup_aerosol_flowmeter(current_scan_settings())

    
    setup_hv_source()
    _time.sleep(0.3)
    #dac.allow()

    flow_controller = ctl.blower.FlowController(
        flowmeter,
        blower,
        flow_lpm=float(sheath1.value),
        kp=float(sheath_pid_kp.value),
        ki=float(sheath_pid_ki.value),
        kd=float(sheath_pid_kd.value),
    )
    pn.state.cache[FLOW_CONTROLLER_CACHE_KEY] = flow_controller
    flow_controller.start()
    ensure_cpc_reader_thread()


def init_done_callback(fut, start_after=False):
    global init_running
    import traceback as _traceback

    try:
        fut.result()
        set_status_threadsafe("Status: hardware initialized")
        if start_after and start_button.value:
            ensure_measurement_thread()
            measurement_running.set()
            set_status_threadsafe("Status: running")
    except Exception as e:
        _traceback.print_exc()
        record_runtime_error(e)
        measurement_running.clear()
        set_status_threadsafe(f"Hardware init failed: {e}")
    finally:
        init_running = False


def init(start_after=False):
    global init_running

    if tuning_running.is_set() or calibration_running.is_set():
        status_text.object = "Status: hardware initialization refused during tuning/calibration"
        return

    if flow_controller is not None:
        ensure_cpc_reader_thread()
        settings = active_scan_settings or current_scan_settings()
        setup_hv_source(settings)
        setup_aerosol_flowmeter(settings)
        apply_idle_flow_setpoint()
        if start_after:
            ensure_measurement_thread()
            measurement_running.set()
            status_text.object = "Status: running"
        else:
            status_text.object = "Status: hardware initialized"
        return

    if init_running:
        status_text.object = "Status: hardware initialization already running"
        return

    init_running = True
    status_text.object = "Status: initializing hardware..."
    fut = hardware_executor.submit(init_hardware_blocking)
    fut.add_done_callback(lambda future: init_done_callback(future, start_after=start_after))


def apply_idle_flow_setpoint():
    if flow_controller is None or measurement_running.is_set() or tuning_running.is_set():
        return
    try:
        flow_controller.setpoint(float(sheath1.value))
    except Exception as e:
        print(f"Idle flow setpoint update failed: {e}", flush=True)


def measurement_loop(_stop_event=app_stop_event, _measurement_running=measurement_running):
    import time as _time
    import traceback as _traceback

    while not _stop_event.is_set():
        try:
            if _measurement_running.is_set():
                measurement_step()
        except Exception as e:
            _traceback.print_exc()
            try:
                set_status_threadsafe(f"Measurement loop error: {e}")
            except Exception:
                pass
        _time.sleep(0.05)


def cpc_reader_loop(
    _stop_event=app_stop_event,
    _cpc_reader_stop=cpc_reader_stop,
    _latest_cpc_lock=latest_cpc_lock,
    _latest_cpc=latest_cpc,
    _globals=globals(),
    _np=np,
    _default_settings=DEFAULT_SETTINGS,
):
    import time as _time

    while not _stop_event.is_set() and not _cpc_reader_stop.is_set():
        cpc_obj = _globals.get("cpc")
        if cpc_obj is None:
            _time.sleep(0.1)
            continue

        started = _time.monotonic()
        error = None
        got_response = False
        try:
            value = cpc_obj.read_instrument()
            got_response = _np.isfinite(float(value))
            if not got_response:
                error = "CPC returned no valid numeric concentration"
        except Exception as e:
            value = _np.nan
            error = str(e)
            print(f"CPC read failed: {e}", flush=True)
            record_runtime_error(e)

        if _stop_event.is_set() or _cpc_reader_stop.is_set():
            break

        finished = _time.monotonic()
        wall_time = datetime.now().isoformat()
        with _latest_cpc_lock:
            if got_response:
                _latest_cpc["sample_id"] += 1
                _latest_cpc["sample_time"] = wall_time
            _latest_cpc["value"] = value
            _latest_cpc["time"] = finished
            _latest_cpc["duration_sec"] = finished - started
            _latest_cpc["error"] = error

        try:
            sleep_sec = cpc_poll_interval_seconds()
        except Exception:
            sleep_sec = float(_default_settings["cpc_poll_interval"])
        _time.sleep(max(0.0, sleep_sec - (finished - started)))


def ensure_cpc_reader_thread():
    global cpc_reader_thread
    if cpc_reader_thread is None or not cpc_reader_thread.is_alive():
        cpc_reader_stop.clear()
        cpc_reader_thread = threading.Thread(target=cpc_reader_loop, daemon=True)
        cpc_reader_thread.start()


def latest_cpc_snapshot():
    import time as _time

    with latest_cpc_lock:
        value = latest_cpc["value"]
        sample_time = latest_cpc["time"]
        duration = latest_cpc["duration_sec"]
        error = latest_cpc["error"]
        sample_id = latest_cpc["sample_id"]
        sample_wall_time = latest_cpc["sample_time"]

    age = np.nan if sample_time is None else _time.monotonic() - sample_time
    return value, age, duration, error, sample_id, sample_wall_time


def aerosol_poll_loop(meter, _stop_event=app_stop_event):
    while not _stop_event.is_set() and aerosol_flowmeter is meter:
        try:
            meter.step()
            flow, pressure, temperature = meter.snapshot()
            with aerosol_cache_lock:
                aerosol_cache.update(flow=flow, pressure=pressure, temperature=temperature, error=None)
        except Exception as e:
            with aerosol_cache_lock:
                aerosol_cache["error"] = str(e)
            print(f"Aerosol flowmeter read failed: {e}", flush=True)
            record_runtime_error(e)
        _stop_event.wait(1.0)


def ensure_aerosol_poll_thread():
    global aerosol_poll_thread
    if aerosol_flowmeter is not None and (aerosol_poll_thread is None or not aerosol_poll_thread.is_alive()):
        aerosol_poll_thread = threading.Thread(target=aerosol_poll_loop, args=(aerosol_flowmeter,), daemon=True)
        aerosol_poll_thread.start()


def setup_aerosol_flowmeter(settings, raise_errors=False):
    global aerosol_flowmeter, aerosol_active_config, aerosol_poll_thread
    config = (
        settings["aerosol_flow_enabled"], settings["aerosol_flow_i2c_bus"],
        settings["aerosol_flow_i2c_address"], settings["aerosol_flow_calibration"],
    )
    if config == aerosol_active_config:
        return
    if not settings["aerosol_flow_enabled"]:
        if aerosol_flowmeter is not None:
            previous_meter = aerosol_flowmeter
            aerosol_flowmeter = None
            previous_meter.close()
        with aerosol_cache_lock:
            aerosol_cache.update(flow=np.nan, pressure=np.nan, temperature=np.nan, error=None)
        aerosol_active_config = config
        return
    try:
        if aerosol_flowmeter is not None:
            previous_meter = aerosol_flowmeter
            aerosol_flowmeter = None
            previous_meter.close()
        aerosol_flowmeter = ctl.AerosolFlowmeter(
            bus=settings["aerosol_flow_i2c_bus"],
            address=int(settings["aerosol_flow_i2c_address"], 0),
            calibration_lpm_per_pa=settings["aerosol_flow_calibration"],
        )
        aerosol_active_config = config
        aerosol_poll_thread = None
        ensure_aerosol_poll_thread()
    except Exception as e:
        aerosol_flowmeter = None
        with aerosol_cache_lock:
            aerosol_cache.update(flow=np.nan, pressure=np.nan, temperature=np.nan, error=str(e))
        print(f"Aerosol flowmeter initialization failed: {e}", flush=True)
        record_runtime_error(e)
        if raise_errors:
            raise


def aerosol_snapshot():
    with aerosol_cache_lock:
        return aerosol_cache["flow"], aerosol_cache["pressure"], aerosol_cache["temperature"], aerosol_cache["error"]


def current_tuning_config():
    return {
        "output_low_v": float(sheath_tune_output_low_v.value),
        "output_high_v": float(sheath_tune_output_high_v.value),
        "settle_sec": float(sheath_tune_settle_sec.value),
        "step_sec": float(sheath_tune_step_sec.value),
        "sample_interval_sec": float(sheath_tune_sample_interval_sec.value),
        "min_response_lpm": float(sheath_tune_min_response_lpm.value),
        "flow_min_lpm": 0.0,
        "flow_max_lpm": 25.0,
    }


def run_sheath_tuning_blocking(config):
    global pending_tuning_result

    log_path = Path("logs/sheath_tuning") / f"sheath_step_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        # HV must be confirmed off before open-loop blower control starts.
        zero_hv(disable=True)

        def log_sample(row):
            append_row_csv(log_path, row)

        def report(progress, phase_name, flow):
            tool_ui_updates.append({
                "kind": "tuning_progress",
                "progress": progress,
                "message": f"Sheath tuning: {phase_name} plateau, {flow:.2f} L/min",
            })

        result = flow_controller.run_step_tuning(
            config, tuning_cancel_event, sample_callback=log_sample,
            progress_callback=report,
        )
        result["log_path"] = str(log_path)
        with tuning_result_lock:
            pending_tuning_result = result
        tool_ui_updates.append({"kind": "tuning_result", "result": result})
    except Exception as e:
        tool_ui_updates.append({
            "kind": "tuning_error",
            "message": f"Sheath tuning failed: {e}. Raw samples: `{log_path}`",
        })
    finally:
        tuning_running.clear()
        tool_ui_updates.append({"kind": "tuning_finished"})


def start_sheath_tuning(event=None):
    global pending_tuning_result

    if measurement_running.is_set() or start_button.value:
        sheath_tune_status.object = "Sheath tuning refused: stop measurement first"
        return
    if calibration_running.is_set():
        sheath_tune_status.object = "Sheath tuning refused: aerosol calibration is active"
        return
    if hardware_stop_pending:
        sheath_tune_status.object = "Sheath tuning refused: wait for HV stop/zero to finish"
        return
    if flow_controller is None or flowmeter is None:
        sheath_tune_status.object = "Sheath tuning refused: initialize hardware first"
        return
    if type(flowmeter) is not ctl.Flowmeter:
        sheath_tune_status.object = "Sheath tuning refused: real sheath SFM3000 Flowmeter required"
        return
    if tuning_running.is_set():
        return

    pending_tuning_result = None
    tuning_cancel_event.clear()
    tuning_running.set()
    sheath_tune_start_button.disabled = True
    sheath_tune_cancel_button.disabled = False
    sheath_tune_apply_button.disabled = True
    sheath_tune_progress.value = 0
    sheath_tune_status.object = "Sheath tuning: zeroing/disabling HV before open-loop step..."
    tuning_executor.submit(run_sheath_tuning_blocking, current_tuning_config())


def cancel_sheath_tuning(event=None):
    if tuning_running.is_set():
        tuning_cancel_event.set()
        sheath_tune_status.object = "Sheath tuning: cancellation requested; restoring closed loop..."


def apply_sheath_tuning_result(event=None):
    if measurement_running.is_set() or start_button.value or tuning_running.is_set():
        sheath_tune_status.object = "Apply refused: measurement and tuning must be stopped"
        return
    with tuning_result_lock:
        result = None if pending_tuning_result is None else dict(pending_tuning_result)
    if result is None:
        sheath_tune_status.object = "Apply refused: no successful tuning result"
        return
    try:
        flow_controller.set_pid_params(result["Kp"], result["Ki"], result["Kd"])
        sheath_pid_kp.value = result["Kp"]
        sheath_pid_ki.value = result["Ki"]
        sheath_pid_kd.value = result["Kd"]
        save_settings()
        sheath_tune_apply_button.disabled = True
        sheath_tune_status.object = (
            f"Applied and saved Kp={result['Kp']:.6g}, Ki={result['Ki']:.6g}, "
            f"Kd={result['Kd']:.6g}"
        )
    except Exception as e:
        sheath_tune_status.object = f"Could not apply tuning result: {e}"


def persist_aerosol_factor(factor):
    with settings_file_lock:
        settings = {}
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r") as file:
                settings = json.load(file)
        settings["aerosol_flow_calibration"] = float(factor)
        atomic_write_json(SETTINGS_FILE, settings)


def run_aerosol_calibration_blocking(meter, settings, sample_count):
    global aerosol_active_config

    try:
        pressures = meter.read_pressure_samples(
            count=sample_count, interval_sec=0.5, cancel_event=app_stop_event
        )
        result = aerosol_factor_from_pressures(pressures)
        factor = result["factor_lpm_per_pa"]
        persist_aerosol_factor(factor)
        settings["aerosol_flow_calibration"] = factor
        aerosol_active_config = None
        setup_aerosol_flowmeter(settings, raise_errors=True)
        tool_ui_updates.append({"kind": "calibration_result", "result": result})
    except Exception as e:
        record_runtime_error(e)
        tool_ui_updates.append({"kind": "calibration_error", "message": str(e)})
    finally:
        calibration_running.clear()
        tool_ui_updates.append({"kind": "calibration_finished"})


def start_aerosol_calibration(event=None):
    if measurement_running.is_set() or start_button.value:
        aerosol_calibration_status.object = "Aerosol calibration refused: stop measurement first"
        return
    if tuning_running.is_set():
        aerosol_calibration_status.object = "Aerosol calibration refused: sheath tuning is active"
        return
    if not aerosol_calibration_confirm.value:
        aerosol_calibration_status.object = (
            "Calibration refused: confirm external actual flow is already 1.0 L/min"
        )
        return
    if aerosol_flowmeter is None:
        aerosol_calibration_status.object = "Calibration refused: initialize and enable aerosol sensor first"
        return
    if calibration_running.is_set():
        return
    sample_count = max(5, int(aerosol_calibration_samples.value))
    settings = current_scan_settings()
    calibration_running.set()
    aerosol_calibration_button.disabled = True
    aerosol_calibration_status.object = f"Aerosol calibration: averaging {sample_count} pressure readings..."
    calibration_executor.submit(
        run_aerosol_calibration_blocking, aerosol_flowmeter, settings, sample_count
    )


def query_spellman_cache(device):
    global last_spellman_query_time
    error = None
    try:
        voltage = device.get_voltage()
        status = device.get_status()
    except Exception as e:
        voltage, status, error = np.nan, None, str(e)
        print(f"Spellman background query failed: {e}", flush=True)
        record_runtime_error(e)
    with spellman_cache_lock:
        spellman_cache.update(voltage=voltage, status=status, time=datetime.now().isoformat(), error=error, pending=False)
        if error:
            spellman_cache["serial_errors"] += 1
    last_spellman_query_time = time.monotonic()


def maybe_query_spellman():
    if not active_hv_config or active_hv_config[0] != "Monopolar Spellman" or hv_device is None:
        return
    with spellman_cache_lock:
        if spellman_cache["pending"] or time.monotonic() - last_spellman_query_time < 10.0:
            return
        spellman_cache["pending"] = True
    spellman_executor.submit(query_spellman_cache, hv_device)


def spellman_snapshot():
    with spellman_cache_lock:
        return dict(spellman_cache)


_software_identity = None


def software_identity():
    global _software_identity
    if _software_identity is None:
        try:
            commit = subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                check=True, capture_output=True, text=True, timeout=2,
            ).stdout.strip()
        except Exception:
            commit = os.environ.get("DMPS_COMMIT", "unknown")
        _software_identity = {
            "version": os.environ.get("DMPS_VERSION", APP_VERSION),
            "commit": commit,
        }
    return dict(_software_identity)


def build_health_payload():
    cpc_value, cpc_age, _, cpc_error, _, cpc_sample_time = latest_cpc_snapshot()
    aerosol_flow, aerosol_pressure, aerosol_temperature, aerosol_error = aerosol_snapshot()
    spellman = spellman_snapshot()
    flow_diagnostics = flowmeter.diagnostics() if flowmeter is not None else {
        "connected": False, "error_count": 0, "consecutive_errors": 0,
        "crc_error_count": 0, "reconnect_count": 0, "last_error": None,
        "serial_number": None, "article_number": None, "sample_age_sec": None,
    }
    try:
        sheath_flow = float(flowmeter.get_flow()) if flowmeter is not None else np.nan
    except Exception as error:
        sheath_flow = np.nan
        flow_diagnostics["last_error"] = str(error)
    try:
        sheath_setpoint = float(flow_controller.pid.setpoint) if flow_controller is not None else np.nan
    except Exception:
        sheath_setpoint = np.nan
    disk = shutil.disk_usage(STATE_DIR)
    with last_runtime_error_lock:
        runtime_error = last_runtime_error
    active_source = active_hv_config[0] if active_hv_config else str(hv_source.value)
    hv_status = spellman["status"] if active_source == "Monopolar Spellman" else hv_runtime_status
    return {
        **software_identity(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scan_active": bool(measurement_running.is_set() and start_button.value),
        "phase": "tuning" if tuning_running.is_set() else "calibration" if calibration_running.is_set() else phase,
        "scan_number": int(scan_number),
        "hardware_initialized": all(
            item is not None for item in (flowmeter, blower, flow_controller, cpc, inletValve)
        ) and active_hv_config is not None and bool(flow_diagnostics.get("connected")),
        "cpc": {
            "type": cpc_type.value,
            "value_cm3": cpc_value, "sample_age_sec": cpc_age,
            "sample_timestamp": cpc_sample_time, "error": cpc_error,
        },
        "sheath": {
            "flow_lpm": sheath_flow, "setpoint_lpm": sheath_setpoint,
            "error_lpm": sheath_flow - sheath_setpoint,
        },
        "flowmeter": flow_diagnostics,
        "aerosol": {
            "enabled": bool(aerosol_flow_enabled.value), "flow_lpm": aerosol_flow,
            "pressure_pa": aerosol_pressure, "temperature_c": aerosol_temperature,
            "error": aerosol_error,
        },
        "hv": {
            "source": active_source, "target_v": hv_target_voltage,
            "status": hv_status, "error": spellman["error"] if active_source == "Monopolar Spellman" else None,
        },
        "last_scan_saved": last_scan_saved,
        "disk_free_bytes": disk.free,
        "last_error": runtime_error,
    }


def write_health():
    ctl.atomic_write_runtime_json(HEALTH_FILE, build_health_payload())


def health_loop(_stop_event=app_stop_event):
    while not _stop_event.is_set():
        try:
            write_health()
        except Exception as error:
            record_runtime_error(f"health write failed: {error}")
            print(f"Health write failed: {error}", flush=True)
        _stop_event.wait(HEALTH_INTERVAL_SEC)


def _safe_shutdown_impl(reason):
    global phase, aerosol_flowmeter, cpc, inletValve, flowmeter, blower, flow_controller
    global hv_runtime_status

    print(f"Safe shutdown started: {reason}", flush=True)
    measurement_running.clear()
    tuning_cancel_event.set()
    app_stop_event.set()
    cpc_reader_stop.set()
    phase = "shutdown"

    errors = []

    def attempt(label, callback):
        try:
            callback()
        except Exception as error:
            errors.append(f"{label}: {error}")
            print(f"Safe shutdown {label} failed: {error}", flush=True)

    def safe_outputs(prefix):
        global hv_runtime_status
        controller = flow_controller
        if controller is not None:
            attempt(f"{prefix} flow controller emergency stop", controller.emergency_stop)
            attempt(f"{prefix} flow controller thread stop", controller.stop)
        if inletValve is not None:
            attempt(f"{prefix} inlet valve off", inletValve.off)
        with hv_io_lock:
            if hv_device is not None:
                attempt(f"{prefix} Spellman zero", hv_device.zero)
                attempt(f"{prefix} Spellman disable", hv_device.disable)
            # Independently zero the SPI DAC even when another HV source was selected.
            if not active_hv_config or active_hv_config[0] != "Bipolar DAC":
                attempt(f"{prefix} bipolar SPI setup", ctl.HV.setup)
            attempt(f"{prefix} bipolar HV zero", ctl.HV.zero)
            hv_runtime_status = "zeroed/disabled"
        if blower is not None:
            attempt(f"{prefix} blower zero", lambda: blower.set_voltage(0.0))

    safe_outputs("initial")

    for executor in (hardware_executor, tuning_executor, calibration_executor, cpc_diag_executor, spellman_executor):
        attempt("executor stop", lambda executor=executor: executor.shutdown(wait=True, cancel_futures=True))
    attempt("git executor stop", lambda: git_executor.shutdown(wait=False, cancel_futures=True))
    measurement_running.clear()
    tuning_running.clear()
    calibration_running.clear()

    for thread in (measurement_thread, cpc_reader_thread, aerosol_poll_thread, health_thread):
        if thread is not None and thread is not threading.current_thread():
            attempt(f"thread {thread.name}", lambda thread=thread: thread.join(timeout=2.0))

    # Initialization or measurement I/O may have been in flight during the first pass.
    safe_outputs("final")
    attempt("bipolar SPI close", ctl.HV.cleanup)

    meter, aerosol_flowmeter = aerosol_flowmeter, None
    for label, resource in (
        ("aerosol flowmeter close", meter), ("CPC close", cpc),
        ("inlet valve close", inletValve), ("flowmeter close", flowmeter),
        ("blower close", blower), ("output gate close", dac),
    ):
        close = getattr(resource, "close", None)
        if close is not None:
            attempt(label, close)

    flow_controller = None
    cpc = None
    inletValve = None
    flowmeter = None
    blower = None
    if errors:
        record_runtime_error("; ".join(errors))
    attempt("final health write", write_health)
    print("Safe shutdown finished", flush=True)


shutdown_coordinator = ctl.ShutdownCoordinator(_safe_shutdown_impl)


def safe_shutdown(reason="requested"):
    return shutdown_coordinator.run(reason)


def queue_ui_row(row):
    ui_rows_pending.append(row)


def drain_ui_updates():
    rows_changed = False
    if ui_rows_pending:
        latest_df = pd.DataFrame(rows[-100:])
        table_pane.value = latest_df
        last_row_pane.object = str(ui_rows_pending[-1])
        ui_rows_pending.clear()
        rows_changed = True

    if cpc_diag_ui_pending:
        latest = cpc_diag_ui_pending[-1]
        if "status" in latest:
            cpc_diag_status.object = latest["status"]
        cpc_diag_table.value = pd.DataFrame(cpc_diag_rows)
        cpc_diag_ui_pending.clear()

    while tool_ui_updates:
        update = tool_ui_updates.popleft()
        kind = update["kind"]
        if kind == "tuning_progress":
            sheath_tune_progress.value = int(100 * update["progress"])
            sheath_tune_status.object = update["message"]
        elif kind == "tuning_result":
            result = update["result"]
            sheath_tune_progress.value = 100
            sheath_tune_apply_button.disabled = False
            sheath_tune_status.object = (
                f"Tuning result (not applied): Kp={result['Kp']:.6g}, "
                f"Ki={result['Ki']:.6g}, Kd={result['Kd']:.6g}; "
                f"K={result['process_gain_lpm_per_v']:.3g} L/min/V, "
                f"tau={result['tau_sec']:.2f}s, theta={result['theta_sec']:.2f}s. "
                f"Raw samples: `{result['log_path']}`"
            )
        elif kind == "tuning_error":
            sheath_tune_status.object = update["message"]
            sheath_tune_apply_button.disabled = True
        elif kind == "tuning_finished":
            sheath_tune_start_button.disabled = False
            sheath_tune_cancel_button.disabled = True
        elif kind == "calibration_result":
            result = update["result"]
            aerosol_flow_calibration.value = result["factor_lpm_per_pa"]
            aerosol_calibration_confirm.value = False
            aerosol_calibration_status.object = (
                f"Calibration saved and sensor reinitialized: mean pressure "
                f"{result['mean_pressure_pa']:.4f} Pa from {result['sample_count']} samples; "
                f"factor={result['factor_lpm_per_pa']:.8g} L/min/Pa"
            )
        elif kind == "calibration_error":
            aerosol_calibration_status.object = f"Aerosol calibration failed: {update['message']}"
        elif kind == "calibration_finished":
            aerosol_calibration_button.disabled = False
        elif kind == "git_status":
            git_update_status.object = update["message"]
            git_update_status.alert_type = update["alert_type"]
            git_update_button.disabled = False

    value, age, duration, error, sample_id, _ = latest_cpc_snapshot()
    cpc_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if np.isfinite(cpc_value):
        current_cpc_pane.object = f"Current CPC: {cpc_value:.0f} cm^-3 | sample {sample_id} | age {age:.2f}s | read {duration:.3f}s"
    elif error:
        current_cpc_pane.object = f"Current CPC: nan | error: {error}"
    else:
        current_cpc_pane.object = "Current CPC: nan"

    if flow_controller is not None and flowmeter is not None:
        flow = read_flow_value()
        flow_diag = flowmeter.diagnostics()
        try:
            setpoint = float(flow_controller.pid.setpoint)
        except Exception:
            setpoint = np.nan
        try:
            blower_v = float(flow_controller.out)
        except Exception:
            blower_v = np.nan
        if np.isfinite(flow):
            error_value = flow - setpoint if np.isfinite(setpoint) else np.nan
            if np.isfinite(error_value):
                current_flow_pane.object = (
                    f"Current sheath flow: {flow:.2f} L/min | "
                    f"set {setpoint:.2f} | error {error_value:+.2f} | blower {blower_v:.2f} V | "
                    f"SFM {'connected' if flow_diag['connected'] else 'DISCONNECTED'}, "
                    f"errors {flow_diag['error_count']} (CRC {flow_diag['crc_error_count']}), "
                    f"reconnects {flow_diag['reconnect_count']}, serial {flow_diag['serial_number']}"
                )
            else:
                current_flow_pane.object = f"Current sheath flow: {flow:.2f} L/min"
        else:
            current_flow_pane.object = "Current sheath flow: nan"
    else:
        current_flow_pane.object = "Current sheath flow: not initialized"

    maybe_query_spellman()
    if active_hv_config and active_hv_config[0] == "Monopolar Spellman":
        if hv_device is None:
            current_hv_pane.object = "Current HV: Spellman not initialized"
        else:
            cache = spellman_snapshot()
            voltage = cache["voltage"]
            current_hv_pane.object = f"Current HV: Spellman {voltage:.0f} V" if np.isfinite(voltage) else f"Current HV: Spellman unavailable ({cache['error'] or 'waiting'})"
    else:
        current_hv_pane.object = "Current HV: bipolar DAC"

    aerosol_flow, aerosol_dp, aerosol_temp, aerosol_error = aerosol_snapshot()
    if aerosol_flowmeter is not None and np.isfinite(aerosol_flow):
        flow_error = aerosol_flow - 1.0
        state = "OK" if abs(flow_error) <= 0.1 else "CHECK"
        current_aerosol_flow_pane.object = (
            f"Current aerosol flow: {aerosol_flow:.3f} L/min | target 1.000 | "
            f"error {flow_error:+.3f} | {state} | {aerosol_dp:.2f} Pa | {aerosol_temp:.1f} C"
        )
    elif aerosol_flow_enabled.value:
        current_aerosol_flow_pane.object = f"Current aerosol flow: unavailable ({aerosol_error or 'waiting'})"
    else:
        current_aerosol_flow_pane.object = "Current aerosol flow: disabled"

    update_scan_progress()

    if rows_changed:
        refresh_live_plot()

    maybe_run_auto_cpc_diagnostics()
    maybe_check_git_update()


def git_output(*arguments):
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes -o ConnectTimeout=10"
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    return result.stdout.strip()


def check_git_update_blocking():
    try:
        git_output("fetch", "--quiet", "--prune", "origin")
        branch = git_output("branch", "--show-current") or "main"
        local_revision = git_output("rev-parse", "--short", "HEAD")
        remote_ref = f"origin/{branch}"
        remote_revision = git_output("rev-parse", "--short", remote_ref)
        ahead, behind = [
            int(value)
            for value in git_output("rev-list", "--left-right", "--count", f"HEAD...{remote_ref}").split()
        ]
        if behind:
            message = (
                f"**Software update available:** `{local_revision}` -> `{remote_revision}` "
                f"({behind} commit{'s' if behind != 1 else ''}). Stop measurement and use "
                "**Stop and zero HV**, then run `dmps update` and refresh this page."
            )
            alert_type = "warning"
        elif ahead:
            message = f"Software: local `{local_revision}` is {ahead} commit(s) ahead of `{remote_ref}`."
            alert_type = "warning"
        else:
            message = f"Software is up to date: `{local_revision}`."
            alert_type = "success"
        tool_ui_updates.append({"kind": "git_status", "message": message, "alert_type": alert_type})
    except Exception as error:
        tool_ui_updates.append({
            "kind": "git_status",
            "message": f"Software update check unavailable: `{error}`",
            "alert_type": "light",
        })
    finally:
        git_check_running.clear()


def maybe_check_git_update(force=False):
    global last_git_check_time

    now = time.monotonic()
    if git_check_running.is_set() or (not force and now - last_git_check_time < GIT_CHECK_INTERVAL_SEC):
        return
    last_git_check_time = now
    git_check_running.set()
    git_update_button.disabled = True
    git_executor.submit(check_git_update_blocking)


def check_git_update_now(event=None):
    git_update_status.object = "Software update: checking Git..."
    git_update_status.alert_type = "info"
    maybe_check_git_update(force=True)


def ensure_measurement_thread():
    global measurement_thread
    if measurement_thread is None or not measurement_thread.is_alive():
        measurement_thread = threading.Thread(target=measurement_loop, daemon=True)
        measurement_thread.start()


polarity_switch = 1
measurement_finished = True
Ntot = False


def interruptible_sleep(
    seconds,
    _stop_event=app_stop_event,
    _measurement_running=measurement_running,
    _start_button=start_button,
):
    import time as _time

    deadline = _time.monotonic() + max(0.0, float(seconds))
    while _time.monotonic() < deadline:
        if _stop_event.is_set() or not _measurement_running.is_set() or not _start_button.value:
            return False
        _time.sleep(min(0.05, deadline - _time.monotonic()))
    return True


def apply_scan_point(point):
    global active_point_key

    dp = point["dp"]
    q_sheath = point["sheath"]
    key = (float(dp), float(q_sheath))
    if active_point_key == key:
        return

    flow_controller.setpoint(q_sheath)
    set_hv_for_point(point)
    active_point_key = key


def read_cpc_count():
    value, _, _, _, _, _ = latest_cpc_snapshot()
    return value


def read_flow_value():
    try:
        return flowmeter.get_flow()
    except Exception as e:
        print(f"Flow read failed: {e}", flush=True)
        return np.nan


def cpc_poll_interval_seconds():
    try:
        value = active_scan_settings["cpc_poll_interval"] if active_scan_settings else cpc_poll_interval.value
        return max(0.05, float(value))
    except Exception:
        return float(DEFAULT_SETTINGS["cpc_poll_interval"])


def append_measurement_row(point, scan_number_value, is_ntot=False, extra=None):
    sample_started = time.monotonic()
    cpc_count, cpc_age, cpc_read_duration, cpc_error, cpc_sample_id, cpc_sample_time = latest_cpc_snapshot()
    flow = read_flow_value()
    aerosol_flow, aerosol_dp, aerosol_temp, _ = aerosol_snapshot()
    hv_cache = spellman_snapshot()
    extra = extra or {}

    row = {
        "time": datetime.now().isoformat(),
        "scan_range": point["scan_range"],
        "size_nm": point["dp"],
        "cpc_count": cpc_count,
        "sheath_flow": flow,
        "sheath_setpoint": point["sheath"],
        "scan_number": scan_number_value,
        "Ntot": is_ntot,
        "sample_duration_sec": time.monotonic() - sample_started,
        "cpc_age_sec": cpc_age,
        "cpc_read_duration_sec": cpc_read_duration,
        "cpc_error": cpc_error,
        "cpc_sample_id": cpc_sample_id,
        "cpc_sample_time": cpc_sample_time,
        "hv_target_v": 0.0 if is_ntot else point.get("hv_target_v", np.nan),
        "spellman_measured_v": hv_cache["voltage"] if active_hv_config and active_hv_config[0] == "Monopolar Spellman" else np.nan,
        "spellman_status": hv_cache["status"] if active_hv_config and active_hv_config[0] == "Monopolar Spellman" else None,
        "aerosol_flow_lpm": aerosol_flow,
        "aerosol_dp_pa": aerosol_dp,
        "aerosol_temp_c": aerosol_temp,
        **extra,
    }

    rows.append(row)
    queue_ui_row(row)
    local_log = Path("logs") / f"measurement_{datetime.now().strftime('%Y%m%d')}.csv"
    legacy_columns = [
        "time", "scan_range", "size_nm", "cpc_count", "sheath_flow",
        "sheath_setpoint", "scan_number", "Ntot", "sample_duration_sec",
        "cpc_age_sec", "cpc_read_duration_sec", "cpc_error",
        "point_elapsed_sec", "point_set_duration_sec", "phase",
    ]
    log_row({column: row.get(column) for column in legacy_columns}, local_log=local_log, cloud_log=None)
    return row


def sample_due():
    global next_sample_time

    monotonic_now = time.monotonic()
    if monotonic_now < next_sample_time:
        return False
    next_sample_time = monotonic_now + cpc_poll_interval_seconds()
    return True


def expected_scan_duration(settings, scan, include_ntot=False):
    switches = sum(np.sign(scan[i]["dp"]) != np.sign(scan[i - 1]["dp"]) for i in range(1, len(scan)))
    duration = len(scan) * (settings["settling_time"] + settings["meas_time"])
    duration += switches * settings["polarity_switch_time"] + settings["final_point_extra_hold"]
    if include_ntot:
        duration += settings["settling_time"] + settings["ntot_time"] + settings["ntot_rest_time"]
    return duration


def update_scan_progress():
    if not active_scan_settings or scan_started_monotonic is None or phase == "idle":
        scan_progress_pane.object = "Scan progress: idle"
        return
    scan = active_scan_settings["scan"]
    point = scan[min(current_size_index, len(scan) - 1)]
    elapsed = time.monotonic() - scan_started_monotonic
    expected = expected_scan_duration(active_scan_settings, scan)
    eta = max(0.0, expected - elapsed)
    scan_progress_pane.object = (
        f"Scan {scan_number + 1} | range {point['scan_range']} | point "
        f"{min(current_size_index + 1, len(scan))}/{len(scan)} | {phase} | ETA {eta:.0f} s"
    )


def scan_qc(scan_data, settings, actual_duration, serial_errors):
    data = pd.DataFrame(scan_data)
    measured = data[data["Ntot"] == False].copy() if not data.empty else data
    expected = {(p["scan_range"], p["dp"]) for p in settings["scan"]}
    observed = set(zip(measured.get("scan_range", []), measured.get("size_nm", [])))
    completeness = len(expected & observed) / len(expected) if expected else 0.0
    ids = pd.to_numeric(measured.get("cpc_sample_id", pd.Series(dtype=float)), errors="coerce").dropna()
    ids = ids[ids > 0]
    unique_samples = int(ids.nunique())
    repeated_fraction = 1.0 - unique_samples / len(measured) if len(measured) else 1.0
    valid_samples = measured[pd.to_numeric(measured["cpc_sample_id"], errors="coerce") > 0] if len(measured) else measured
    unique_by_size = valid_samples.groupby(["scan_range", "size_nm"])["cpc_sample_id"].nunique() if len(valid_samples) else pd.Series(dtype=float)
    flow = pd.to_numeric(measured.get("sheath_flow", pd.Series(dtype=float)), errors="coerce")
    setpoint = pd.to_numeric(measured.get("sheath_setpoint", pd.Series(dtype=float)), errors="coerce")
    flow_rmse = float(np.sqrt(np.nanmean((flow - setpoint) ** 2))) if len(flow) else np.nan
    aerosol = pd.to_numeric(measured.get("aerosol_flow_lpm", pd.Series(dtype=float)), errors="coerce")
    aerosol_mean = float(aerosol.mean()) if aerosol.notna().any() else np.nan
    aerosol_error = aerosol_mean - 1.0 if np.isfinite(aerosol_mean) else np.nan
    hold_rows = measured[measured.get("phase", pd.Series(index=measured.index, dtype=object)) == "final_hold"]
    hold_ids = pd.to_numeric(hold_rows.get("cpc_sample_id", pd.Series(dtype=float)), errors="coerce")
    hold_unique_samples = int(hold_ids[hold_ids > 0].nunique())
    required_hold_samples = (
        max(1, abs(int(settings["smps_plot_step_shift"])))
        if settings["final_point_extra_hold"] > 0 else 0
    )
    expected_duration = expected_scan_duration(settings, settings["scan"], include_ntot=settings.get("did_ntot", False))
    duration_ratio = actual_duration / expected_duration if expected_duration else np.nan
    result = "Good"
    cpc_errors = int(measured.get("cpc_error", pd.Series(dtype=object)).notna().sum())
    serial_errors += cpc_errors
    if completeness < 1.0 or repeated_fraction > 0.5 or serial_errors or hold_unique_samples < required_hold_samples or (np.isfinite(flow_rmse) and flow_rmse > 0.5) or (np.isfinite(aerosol_error) and abs(aerosol_error) > 0.1) or (np.isfinite(duration_ratio) and not 0.8 <= duration_ratio <= 1.2):
        result = "Warning"
    return {
        "scan_number": scan_number, "result": result, "completeness": completeness,
        "unique_samples": unique_samples,
        "unique_samples_per_size_min": int(unique_by_size.min()) if len(unique_by_size) else 0,
        "repeated_fraction": repeated_fraction, "flow_rmse_lpm": flow_rmse,
        "aerosol_flow_mean_lpm": aerosol_mean, "aerosol_flow_error_lpm": aerosol_error,
        "final_hold_unique_samples": hold_unique_samples,
        "final_hold_required_samples": required_hold_samples,
        "serial_error_count": serial_errors,
        "actual_duration_sec": actual_duration, "expected_duration_sec": expected_duration,
    }


def begin_scan():
    global active_scan_settings, scan_started_monotonic, scan_started_wall, scan_serial_error_start
    active_scan_settings = current_scan_settings()
    active_scan_settings["scan"] = build_scan_points(include_dac_codes=True, settings=active_scan_settings)
    setup_hv_source(active_scan_settings)
    setup_aerosol_flowmeter(active_scan_settings)
    scan_started_monotonic = time.monotonic()
    scan_started_wall = datetime.now().isoformat()
    scan_serial_error_start = spellman_snapshot()["serial_errors"]
    pending_settings_pane.object = "Settings: active scan snapshot"


def complete_scan(do_ntot, last_point):
    global active_point_key, active_scan_settings, phase
    settings = active_scan_settings
    settings["did_ntot"] = do_ntot
    if do_ntot:
        scan_rows.extend(run_ntot_measurement(last_point["scan_range"], scan_number, last_point["sheath"], settings))
        active_point_key = None
    save_completed_scan(scan_rows, scan_number)
    completed_scans.append(pd.DataFrame(scan_rows.copy()))
    serial_errors = spellman_snapshot()["serial_errors"] - scan_serial_error_start
    qc = scan_qc(scan_rows, settings, time.monotonic() - scan_started_monotonic, serial_errors)
    completed_qc_rows.append(qc)
    qc_table.value = pd.DataFrame(completed_qc_rows)
    scan_rows.clear()
    active_scan_settings = None
    phase = "idle"
    pending_settings_pane.object = "Settings: next scan snapshot will be applied"


def run_ntot_measurement(scan_range, scan_number, q_sheath, settings):
    global inletValve, active_point_key, polarity_switch
    if inletValve is None:
        return []

    ntot_rows = []

    zero_hv()
    active_point_key = None
    inletValve.on()
    dp=1

    if not interruptible_sleep(settings["settling_time"]):
        inletValve.off()
        return ntot_rows

    t_start = time.time()

    while time.time() - t_start < settings["ntot_time"]:
        row = append_measurement_row(
            {"scan_range": scan_range, "dp": dp, "sheath": q_sheath},
            scan_number,
            is_ntot=True,
            extra={"phase_elapsed_sec": time.time() - t_start, "phase": "ntot"},
        )

        print(row, flush=True)
        ntot_rows.append(row)

        if not interruptible_sleep(cpc_poll_interval_seconds()):
            break

    #inletValve.valveoff()
    inletValve.off()
    interruptible_sleep(max(0.0, settings["ntot_rest_time"]))
    polarity_switch = 0

    ntot_values = pd.to_numeric(
        pd.DataFrame(ntot_rows).drop_duplicates("cpc_sample_id").get("cpc_count", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if not ntot_values.empty:
        latest_ntot_pane.object = f"Latest measured Ntot: {ntot_values.mean():.0f}"

    return ntot_rows

def measurement_step(debug=True):
    import traceback as _traceback

    global \
        current_size_index, \
        phase, \
        phase_start_time, \
        scan_number, \
        polarity_switch, \
        measurement_finished, \
        Ntot, \
        active_point_key, \
        next_sample_time, \
        last_point_set_duration_sec, \
        point_set_time, \
        final_hold_start_sample_id

    if not start_button.value:
        return

    if flow_controller is None:
        init()
        return

    now = time.time()

    try:
        if phase == "idle":
            begin_scan()
            phase = "measuring"
            phase_start_time = now
            current_size_index = 0
            measurement_finished = True
            active_point_key = None

        scan = active_scan_settings["scan"]
        if not scan:
            status_text.object = "Status: no scan points defined"
            return
        meas_sec = active_scan_settings["meas_time"]

        if phase == "measuring":
            point = scan[current_size_index]
            dp = point["dp"]
            q_sheath = point["sheath"]
            scan_range = point["scan_range"]

            if measurement_finished:
                new_sign = int(np.sign(dp))
                point_set_started = time.monotonic()
                point_set_time = time.time()
                apply_scan_point(point)
                last_point_set_duration_sec = time.monotonic() - point_set_started
                if polarity_switch != 0 and new_sign != polarity_switch:
                    if not interruptible_sleep(active_scan_settings["polarity_switch_time"]):
                        return
                polarity_switch = new_sign
                if not interruptible_sleep(active_scan_settings["settling_time"]):
                    return
                measurement_finished = False
                phase_start_time = time.time()
                next_sample_time = 0.0

            if not sample_due():
                return

            row = append_measurement_row(
                point,
                scan_number,
                is_ntot=False,
                extra={
                    "point_index": current_size_index,
                    "point_start_time": datetime.fromtimestamp(point_set_time).isoformat(),
                    "point_elapsed_sec": time.time() - phase_start_time,
                    "point_set_duration_sec": last_point_set_duration_sec,
                    "phase": "measuring",
                },
            )

            if debug:
                print(row, flush=True)

            scan_rows.append(row)

            now = time.time()
            if now - phase_start_time >= meas_sec:
                phase_start_time = now
                current_size_index += 1
                measurement_finished = True
                if current_size_index >= len(scan):
                    hold_sec = max(0.0, active_scan_settings["final_point_extra_hold"])
                    if hold_sec > 0:
                        phase = "final_hold"
                        phase_start_time = now
                        current_size_index = len(scan) - 1
                        measurement_finished = False
                        sample_id = scan_rows[-1].get("cpc_sample_id", 0) if scan_rows else 0
                        final_hold_start_sample_id = int(sample_id) if pd.notna(sample_id) else 0
                        final_hold_sample_ids.clear()
                        next_sample_time = time.monotonic() + cpc_poll_interval_seconds()
                        return

                    current_size_index = 0
                    Ntot = True

                    ntot_every = max(0, active_scan_settings["ntot_every"])
                    do_ntot = ntot_every > 0 and (scan_number + 1) % ntot_every == 0
                    complete_scan(do_ntot, point)
                    scan_number += 1

        if phase == "final_hold":
            point = scan[-1]
            apply_scan_point(point)

            if sample_due():
                _, _, _, _, sample_id, _ = latest_cpc_snapshot()
                if pd.notna(sample_id) and int(sample_id) > final_hold_start_sample_id and sample_id not in final_hold_sample_ids:
                    final_hold_sample_ids.add(sample_id)
                    row = append_measurement_row(
                        point,
                        scan_number,
                        is_ntot=False,
                        extra={
                            "point_index": current_size_index,
                            "point_start_time": datetime.fromtimestamp(point_set_time).isoformat(),
                            "point_elapsed_sec": time.time() - phase_start_time,
                            "point_set_duration_sec": last_point_set_duration_sec,
                            "phase": "final_hold",
                            "final_hold_sample_index": len(final_hold_sample_ids),
                        },
                    )
                    if debug:
                        print(row, flush=True)
                    scan_rows.append(row)

            hold_sec = max(0.0, active_scan_settings["final_point_extra_hold"])
            required_samples = max(1, abs(int(active_scan_settings["smps_plot_step_shift"])))
            hold_elapsed = time.time() - phase_start_time
            grace_sec = max(2.0, 2.0 * cpc_poll_interval_seconds())
            if hold_elapsed < hold_sec or (
                len(final_hold_sample_ids) < required_samples and hold_elapsed < hold_sec + grace_sec
            ):
                return

            current_size_index = 0
            phase = "measuring"
            measurement_finished = True
            Ntot = True

            ntot_every = max(0, active_scan_settings["ntot_every"])
            do_ntot = ntot_every > 0 and (scan_number + 1) % ntot_every == 0
            complete_scan(do_ntot, point)
            scan_number += 1
    except Exception as e:
        try:
            if inletValve is not None:
                inletValve.off()
        except Exception:
            pass

        '''try:
            if dac is not None:
                dac.block()
        except Exception:
            pass'''
        _traceback.print_exc()
        status_text.object = f"Measurement error: {e}"
        print(f"Measurement error: {e}", flush=True)
        record_runtime_error(e)


def append_row_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_row(row, local_log, cloud_log=None):
    append_row_csv(local_log, row)
    if cloud_log is not None:
        try:
            append_row_csv(cloud_log, row)
        except Exception as e:
            status_text.object = f"Cloud log failed, local OK: {e}"


def selected_cpc_diag_command():
    command = cpc_diag_command.value
    if command == "":
        command = cpc_diag_custom_command.value
    return str(command).strip()


def log_cpc_diagnostic(command, response, error=None):
    row = {
        "time": datetime.now().isoformat(),
        "cpc_type": cpc_type.value,
        "command": command,
        "response": response,
        "error": error,
    }
    cpc_diag_rows.append(row)
    path = Path("logs/cpc_status") / f"cpc_status_{datetime.now().strftime('%Y%m%d')}.csv"
    append_row_csv(path, row)
    return row


def run_cpc_diagnostic_command_blocking(command, read_lines, auto=False):
    if cpc is None:
        cpc_diag_ui_pending.append({
            "status": "CPC diagnostics: CPC not initialized",
        })
        return

    if not command:
        cpc_diag_ui_pending.append({
            "status": "CPC diagnostics: command is empty",
        })
        return

    try:
        response = cpc.query(command, read_lines=read_lines)
        row = log_cpc_diagnostic(command, response, None)
        mode = "auto" if auto else "manual"
        status = f"CPC diagnostics: {mode} {command} -> {response or '<empty>'}"
    except Exception as e:
        row = log_cpc_diagnostic(command, "", str(e))
        status = f"CPC diagnostics failed: {e}"
    cpc_diag_ui_pending.append({"row": row, "status": status})


def cpc_diag_done_callback(fut):
    global cpc_diag_query_pending
    cpc_diag_query_pending = False
    try:
        fut.result()
    except Exception as e:
        cpc_diag_ui_pending.append({"status": f"CPC diagnostics failed: {e}"})


def run_cpc_diagnostic_command(event=None, auto=False):
    global cpc_diag_query_pending

    if cpc_diag_query_pending:
        return

    command = selected_cpc_diag_command()
    read_lines = max(1, int(cpc_diag_read_lines.value))
    cpc_diag_query_pending = True
    cpc_diag_status.object = f"CPC diagnostics: querying {command or '<empty>'}..."
    fut = cpc_diag_executor.submit(run_cpc_diagnostic_command_blocking, command, read_lines, auto)
    fut.add_done_callback(cpc_diag_done_callback)


def maybe_run_auto_cpc_diagnostics():
    global last_cpc_diag_auto_time

    if not cpc_diag_auto_enabled.value:
        return
    now = time.monotonic()
    interval = max(1.0, float(cpc_diag_interval_sec.value))
    if now - last_cpc_diag_auto_time < interval:
        return
    last_cpc_diag_auto_time = now
    run_cpc_diagnostic_command(auto=True)


def on_start_change(event):
    global phase, phase_start_time, current_size_index, active_scan_settings, scan_started_monotonic

    if event.new:
        if tuning_running.is_set() or calibration_running.is_set():
            start_button.value = False
            status_text.object = "Status: start refused while tuning/calibration is active"
            return
        scan_rows.clear()
        active_scan_settings = None
        scan_started_monotonic = None
        phase = "idle"
        phase_start_time = time.time()
        current_size_index = 0
        init(start_after=True)

    else:
        measurement_running.clear()
        status_text.object = "Status: stopped"


def apply_smps_plot_step_shift_to_bins(grouped):
    grouped = grouped.copy()
    grouped["plot_abs_size_nm"] = grouped["abs_size_nm"]
    shift = int(smps_plot_step_shift.value)

    if shift == 0 or grouped.empty:
        return grouped

    shifted_parts = []
    group_cols = [col for col in ["scan_id", "scan_number", "scan_range", "polarity"] if col in grouped.columns]

    for _, group in grouped.groupby(group_cols, dropna=False) if group_cols else [(None, grouped)]:
        group = group.sort_values("abs_size_nm").copy()
        if len(group) < 2:
            shifted_parts.append(group)
            continue

        sizes = group["abs_size_nm"].to_numpy(dtype=float)
        source_indices = np.clip(np.arange(len(group)) - shift, 0, len(group) - 1)
        group["plot_abs_size_nm"] = sizes[source_indices]
        shifted_parts.append(group)

    return pd.concat(shifted_parts, ignore_index=True) if shifted_parts else grouped


def make_plot(df):
    df2 = get_recent_completed_scans(int(n_scans_plot.value))

    if (df is None or df.empty) and (df2 is None or df2.empty):
        return pn.pane.Markdown("No data")

    if df is None or df.empty:
        df = pd.DataFrame(
            columns=["time", "size_nm", "cpc_count", "sheath_setpoint", "sheath_flow"]
        )
    else:
        df = df.copy()

    df = df.copy()
    df["cpc_float"] = pd.to_numeric(df["cpc_count"], errors="coerce")

    hover_strings = df["size_nm"].astype(str).to_list()

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=False,
        row_heights=[0.18, 0.70, 0.45, 0.95, 0.45, 0.45],
        vertical_spacing=0.05,
    )

    fig.add_scatter(
        x=df["time"],
        y=df["size_nm"],
        mode="lines+markers",
        name="Particle size (nm)",
        customdata=hover_strings,
        hovertemplate="Size: %{y}<br>dp: %{customdata} nm<extra></extra>",
        row=1,
        col=1,
    )

    fig.add_scatter(
        x=df["time"],
        y=df["cpc_float"],
        mode="lines+markers",
        name="CPC (#/cm³)",
        customdata=hover_strings,
        hovertemplate="Conc: %{y}<br>dp: %{customdata} nm<extra></extra>",
        row=2,
        col=1,
    )

    flow = pd.to_numeric(df.get("sheath_flow", pd.Series(dtype=float)), errors="coerce")
    flow_setpoint = pd.to_numeric(df.get("sheath_setpoint", pd.Series(dtype=float)), errors="coerce")
    flow_error = flow - flow_setpoint
    flow_rmse = np.sqrt(np.nanmean(flow_error.to_numpy(dtype=float) ** 2)) if len(flow_error) else np.nan
    fig.add_scatter(
        x=df["time"],
        y=flow_error,
        mode="lines+markers",
        name=f"Sheath flow error (RMSE {flow_rmse:.3f} L/min)" if np.isfinite(flow_rmse) else "Sheath flow error",
        customdata=np.column_stack((flow, flow_setpoint)) if len(df) else None,
        hovertemplate=(
            "error=%{y:.3f} L/min<br>"
            "flow=%{customdata[0]:.3f}<br>setpoint=%{customdata[1]:.3f}<extra></extra>"
        ),
        row=3,
        col=1,
    )

    if df2 is not None and not df2.empty:
        df2 = df2.copy()
        df2["cpc_float"] = pd.to_numeric(df2["cpc_count"], errors="coerce")
        df2["size_float"] = pd.to_numeric(df2["size_nm"], errors="coerce")
        df2["abs_size_nm"] = df2["size_float"].abs()
        df2["polarity"] = np.where(df2["size_float"] > 0, "pos", "neg")
        df2["time"] = pd.to_datetime(df2["time"], errors="coerce")
        if "Ntot" in df2.columns:
            df2_scan = df2[df2["Ntot"] == False].copy()
        else:
            df2_scan = df2.copy()
        df2_scan = df2_scan.dropna(subset=["abs_size_nm", "cpc_float"])
        if "cpc_sample_id" in df2_scan.columns:
            sample_ids = pd.to_numeric(df2_scan["cpc_sample_id"], errors="coerce")
            with_ids = df2_scan[sample_ids.notna() & (sample_ids > 0)].drop_duplicates(
                subset=["scan_id", "scan_range", "abs_size_nm", "polarity", "cpc_sample_id"]
            )
            df2_scan = pd.concat([df2_scan[~(sample_ids.notna() & (sample_ids > 0))], with_ids])

        scan_key = "scan_id" if "scan_id" in df2_scan.columns else "scan_number"

        grouped = (
            df2_scan.groupby([scan_key, "scan_range", "abs_size_nm", "polarity"])
            .agg(
                cpc_float=("cpc_float", "mean"),
                time=("time", "median"),
                n_samples=("cpc_float", "size"),
            )
            .reset_index()
            .sort_values([scan_key, "polarity", "abs_size_nm"])
        )
        grouped = apply_smps_plot_step_shift_to_bins(grouped)

        latest_scan = grouped[scan_key].max() if not grouped.empty else None
        previous = grouped[grouped[scan_key] != latest_scan]
        if not previous.empty:
            background = (
                previous.groupby(["scan_range", "polarity", "abs_size_nm"])
                .agg(
                    cpc_float=("cpc_float", "mean"),
                    plot_abs_size_nm=("plot_abs_size_nm", "mean"),
                )
                .reset_index()
                .sort_values(["scan_range", "polarity", "plot_abs_size_nm"])
            )
            for (scan_range, polarity), g in background.groupby(["scan_range", "polarity"]):
                fig.add_scatter(
                    x=g["plot_abs_size_nm"],
                    y=g["cpc_float"],
                    mode="lines+markers",
                    name=f"Previous scans avg range {scan_range} {polarity}",
                    opacity=0.25,
                    line=dict(width=4),
                    row=4,
                    col=1,
                )

        peak_rows = []
        for (sn, scan_range, polarity), g in grouped.groupby([scan_key, "scan_range", "polarity"]):
            g = g.sort_values("plot_abs_size_nm")
            if g.empty:
                continue

            is_latest = sn == latest_scan

            fig.add_scatter(
                x=g["plot_abs_size_nm"],
                y=g["cpc_float"],
                mode="lines+markers",
                name=f"Latest scan range {scan_range} {polarity}" if is_latest else f"Scan {sn} range {scan_range} {polarity}",
                opacity=1.0 if is_latest else 0.15,
                line=dict(width=3 if is_latest else 1),
                row=4,
                col=1,
            )

            values = g["cpc_float"].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if np.any(finite):
                idx = np.flatnonzero(finite)[int(np.nanargmax(values[finite]))]
                peak_rows.append({
                    "scan_number": sn,
                    "scan_range": scan_range,
                    "polarity": polarity,
                    "time": pd.to_datetime(g["time"], errors="coerce").median(),
                    "peak_dp": float(g["plot_abs_size_nm"].iloc[idx]),
                    "peak_cpc": float(values[idx]),
                })

        if peak_rows:
            peak_df = pd.DataFrame(peak_rows).sort_values("time")
            for (scan_range, polarity), g in peak_df.groupby(["scan_range", "polarity"]):
                fig.add_scatter(
                    x=g["time"],
                    y=g["peak_dp"],
                    mode="lines+markers",
                    name=f"range {scan_range} {polarity} peak Dp",
                    row=5,
                    col=1,
                )
                fig.add_scatter(
                    x=g["time"],
                    y=g["peak_cpc"],
                    mode="lines+markers",
                    name=f"range {scan_range} {polarity} peak CPC",
                    row=6,
                    col=1,
                )

    fig.update_yaxes(title_text="Flow error L/min", row=3, col=1)
    fig.update_yaxes(title_text="CPC", row=4, col=1)
    fig.update_xaxes(type="log", title_text="shifted |dp| (nm)", row=4, col=1)
    fig.update_yaxes(type="log", title_text="Peak |dp| (nm)", row=5, col=1)
    fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=5, col=1)
    fig.update_yaxes(title_text="Peak CPC", row=6, col=1)
    fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=6, col=1)
    fig.update_yaxes(title_text="Size (nm)", row=1, col=1)
    fig.update_yaxes(title_text="CPC", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    fig.update_layout(
        title="Live DMPS scan",
        margin=dict(l=20, r=260, t=40, b=20),
        height=1050,
        width=1500,
        showlegend=True,
        legend=dict(x=1.18, y=1.0),
        autosize=False,
        uirevision="dmps",
    )

    return pn.pane.Plotly(
        fig,
        config={"responsive": False},
        height=1050,
        width=1500,
        sizing_mode="fixed",
    )

plot_box = pn.Column(
    pn.pane.Markdown("No data"),
    height=1200,
    width=1550,
    sizing_mode="fixed",
    scroll=False,
)


def refresh_live_plot(force=False):
    global last_plot_update_time

    now = time.monotonic()
    if not force and now - last_plot_update_time < PLOT_REFRESH_INTERVAL_SEC:
        return

    try:
        plot_box[:] = [make_plot(table_pane.value)]
        last_plot_update_time = now
    except Exception as e:
        print(f"Plot refresh failed: {e}", flush=True)


def startup_load():
    global ui_callback, last_scan_saved

    df0 = get_recent_completed_scans(int(n_scans_plot.value))
    saved_scans = sorted(Path("logs/scans").glob("*/*.csv"))
    if saved_scans:
        last_scan_saved = str(saved_scans[-1])

    if df0 is not None and not df0.empty:
        print(f"Startup loaded {len(df0)} rows", flush=True)

        table_pane.value = df0.tail(100)

        # also populate memory cache
        completed_scans.clear()

        group_key = "scan_id" if "scan_id" in df0.columns else "scan_number"

        for sn, g in df0.groupby(group_key):
            completed_scans.append(g.copy())

    else:
        print("No startup scans found", flush=True)

    refresh_live_plot(force=True)
    maybe_check_git_update(force=True)

    if ui_callback is None:
        ui_callback = pn.state.add_periodic_callback(drain_ui_updates, period=500, start=True)


pn.state.onload(startup_load)


def on_scan_setting_change(event):
    save_settings()
    update_scan_preview()
    apply_idle_flow_setpoint()
    refresh_live_plot(force=True)
    if active_scan_settings is not None:
        pending_settings_pane.object = "**Settings changed: saved, effective next scan**"
    else:
        pending_settings_pane.object = "Settings: current"


cpc_type.param.watch(update_cpc_diagnostic_options, "value")

for widget in [
    cpc_com_port,
    range1,
    sheath1,
    steps1,
    range2,
    sheath2,
    steps2,
    meas_time,
    sleep_time,
    cpc_poll_interval,
    n_scans_plot,
    ntot_rest_time,
    settling_time,
    final_point_extra_hold,
    smps_plot_step_shift,
    polarity_switch_time,
    Bipolar_toggle,
    Ntot_time,
    ntot_every_n_scans,
    cpc_type,
    hv_source,
    spellman_port,
    spellman_baud,
    spellman_max_voltage,
    aerosol_flow_enabled,
    aerosol_flow_i2c_bus,
    aerosol_flow_i2c_address,
    aerosol_flow_calibration,
    sheath_tune_output_low_v,
    sheath_tune_output_high_v,
    sheath_tune_settle_sec,
    sheath_tune_step_sec,
    sheath_tune_sample_interval_sec,
    sheath_tune_min_response_lpm,
    cpc_diag_command,
    cpc_diag_auto_enabled,
    cpc_diag_interval_sec,
]:
    widget.param.watch(on_scan_setting_change, "value")

start_button.param.watch(on_start_change, "value")
init_button.on_click(lambda event: init(start_after=False))
stop_button.on_click(lambda event: stop_and_zero())
cpc_diag_send_button.on_click(lambda event: run_cpc_diagnostic_command(event, auto=False))
sheath_tune_start_button.on_click(start_sheath_tuning)
sheath_tune_cancel_button.on_click(cancel_sheath_tuning)
sheath_tune_apply_button.on_click(apply_sheath_tuning_result)
aerosol_calibration_button.on_click(start_aerosol_calibration)
git_update_button.on_click(check_git_update_now)

update_scan_preview()

#### Layout ####
control_layout = pn.Column(
    f"# DMA / CPC Control GUI v{APP_VERSION}",
    pn.Row(git_update_status, git_update_button, sizing_mode="stretch_width"),
    pn.Row(cpc_com_port, cpc_type),
    pn.Row(hv_source, spellman_port, spellman_baud, spellman_max_voltage),
    pn.Row(aerosol_flow_enabled, aerosol_flow_i2c_bus, aerosol_flow_i2c_address, aerosol_flow_calibration),
    "# CPC / DMA control panel",
    pn.Row(start_button, status_text, init_button, stop_button),
    pn.Row(scan_progress_pane, pending_settings_pane),
    "### Scan range 1",
    pn.Row(range1, sheath1, steps1),
    "### Scan range 2",
    pn.Row(range2, sheath2, steps2),
    scan_pane,
    pn.Row(meas_time, cpc_poll_interval, Ntot_time, ntot_every_n_scans, ntot_rest_time, n_scans_plot),
    pn.Row(settling_time, final_point_extra_hold, smps_plot_step_shift, polarity_switch_time),
    pn.Row(current_cpc_pane, current_flow_pane, current_hv_pane, current_aerosol_flow_pane, latest_ntot_pane),
    "### Sheath flow PID step tuner",
    "Runs only while measurement is stopped. HV is zeroed/disabled first. Set low/high DAC values to bracket the normal steady blower output; flow above 25 L/min aborts the test. Gains are not changed until **Apply result**.",
    pn.Row(sheath_pid_kp, sheath_pid_ki, sheath_pid_kd),
    pn.Row(sheath_tune_output_low_v, sheath_tune_output_high_v, sheath_tune_settle_sec, sheath_tune_step_sec, sheath_tune_sample_interval_sec, sheath_tune_min_response_lpm),
    pn.Row(sheath_tune_start_button, sheath_tune_cancel_button, sheath_tune_apply_button, sheath_tune_progress),
    sheath_tune_status,
    "### Aerosol flow calibration",
    "Set and verify the external actual aerosol flow at **1.0 L/min before calibrating**.",
    pn.Row(aerosol_calibration_confirm, aerosol_calibration_samples, aerosol_calibration_button),
    aerosol_calibration_status,
    "### Completed-scan QC",
    qc_table,
    "### Live data",
    last_row_pane,
    table_pane,
    "### Live plot",
    plot_box,
    sizing_mode="fixed",
    width=1600,
)

cpc_diagnostics_layout = pn.Column(
    "# CPC Diagnostics",
    "Manual and optional periodic CPC serial queries. Commands use the same CPC serial lock as measurement reads.",
    pn.Row(cpc_diag_command, cpc_diag_custom_command, cpc_diag_read_lines, cpc_diag_send_button),
    pn.Row(cpc_diag_auto_enabled, cpc_diag_interval_sec),
    cpc_diag_status,
    cpc_diag_table,
    sizing_mode="fixed",
    width=1200,
)

layout = pn.Tabs(
    ("Control", control_layout),
    ("CPC Diagnostics", cpc_diagnostics_layout),
)

health_thread = threading.Thread(target=health_loop, daemon=True, name="health-writer")
health_thread.start()


def _termination_handler(signum, frame):
    safe_shutdown(signal.Signals(signum).name)
    raise SystemExit(128 + signum)


if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGTERM, _termination_handler)
    signal.signal(signal.SIGINT, _termination_handler)
atexit.register(safe_shutdown, "process exit")

layout.servable()
