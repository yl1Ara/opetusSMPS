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
from concurrent.futures import ThreadPoolExecutor

SETTINGS_FILE = Path("settings.json")

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
    "n_scans_plot": 5,
    "settling_time": 10,
    "final_point_extra_hold": 0,
    "smps_plot_step_shift": 0,
    "polarity_switch_time": 0,
    "Bipolar_toggle": True,
    "Ntot_time": 60,
    "CPC_type": "3771",
}

hardware_executor = ThreadPoolExecutor(max_workers=1)

# Run manually or from a service with panel serve, for example:
# uv run panel serve gui.py --address 0.0.0.0 --port 5006

flowmeter = None
blower = None
flow_controller = None
cpc = None
inletValve = None
dac = None

measurement_running = threading.Event()
measurement_thread = None
hardware_stop_pending = False
init_running = False

pn.extension("plotly")

#### Widgets ####
cpc_com_port = pn.widgets.TextInput(name="CPC COM port", value="/dev/ttyAMA0")
cpc_type = pn.widgets.Select(name="CPC Type", options=["3771", "HY09"], value="3771")

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
current_cpc_pane = pn.pane.Markdown("Current CPC: -")
latest_ntot_pane = pn.pane.Markdown("Latest Ntot: -")
last_row_pane = pn.pane.Str("Last measurement: -")
scan_pane = pn.pane.Str("Scan program: -")

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

rows = []
current_size_index = 0
phase = "idle"
phase_start_time = time.time()
scan_rows = []
completed_scans = []
scan_number = 0
active_point_key = None
next_sample_time = 0.0
last_point_set_duration_sec = np.nan
scan_program_cache_key = None
scan_program_cache = []
ui_rows_pending = deque(maxlen=500)
latest_cpc_lock = threading.Lock()
latest_cpc = {
    "value": np.nan,
    "time": None,
    "duration_sec": np.nan,
    "error": None,
}
cpc_reader_thread = None
cpc_reader_stop = threading.Event()
ui_callback = None


def ensure_settings_file():
    if not SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, "w") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2)


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
        "n_scans_plot": int(n_scans_plot.value),
        "settling_time": int(settling_time.value),
        "final_point_extra_hold": int(final_point_extra_hold.value),
        "smps_plot_step_shift": int(smps_plot_step_shift.value),
        "polarity_switch_time": int(polarity_switch_time.value),
        "Bipolar_toggle": bool(Bipolar_toggle.value),
        "Ntot_time": int(Ntot_time.value),
        "CPC_type": str(cpc_type.value),
    }

    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


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

    cpc_com_port.value = settings.get("cpc_com_port", DEFAULT_SETTINGS["cpc_com_port"])
    cpc_type.value = settings.get("CPC_type", DEFAULT_SETTINGS["CPC_type"])

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

ensure_settings_file()
load_settings()


def bipolar_log_sizes(size_range_value, n, order="negative_then_positive"):
    global Bipolar_toggle

    lo, hi = np.array(size_range_value).ravel().astype(float)
    lo, hi = abs(lo), abs(hi)

    if lo <= 0 or hi <= 0:
        raise ValueError("Use positive nonzero limits, e.g. [20, 400]")
    if hi < lo:
        lo, hi = hi, lo
    if int(n) < 2:
        raise ValueError("steps must be >= 2")

    pos = np.round(np.logspace(np.log10(lo), np.log10(hi), int(n))).astype(int)
    pos = list(dict.fromkeys([int(x) for x in pos if int(x) != 0]))
    if Bipolar_toggle.value:
        neg = [-x for x in pos]
    else:
        neg = []

    if order == "positive_then_negative":
        return pos + neg
    return neg + pos


def save_completed_scan(scan_rows, scan_number):
    if not scan_rows:
        return

    t0 = pd.to_datetime(scan_rows[0]["time"])
    scan_id = t0.strftime("%Y%m%d_%H%M%S")

    run_day = t0.strftime("%Y%m%d")
    path = Path("logs/scans") / run_day / f"{scan_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(scan_rows).to_csv(path, index=False)
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


def build_scan_points(include_dac_codes=False):
    scan = []

    if int(steps1.value) > 1:
        for dp in bipolar_log_sizes(range1.value, int(steps1.value)):
            sheath = float(sheath1.value)
            point = {"scan_range": 1, "dp": int(dp), "sheath": sheath}
            if include_dac_codes:
                point["dac_code"] = ctl.HV.dac_code_from_size(int(dp), Q_sh_lpm=sheath)
            scan.append(point)

    if int(steps2.value) > 2:
        for dp in bipolar_log_sizes(range2.value, int(steps2.value)):
            sheath = float(sheath2.value)
            point = {"scan_range": 2, "dp": int(dp), "sheath": sheath}
            if include_dac_codes:
                point["dac_code"] = ctl.HV.dac_code_from_size(int(dp), Q_sh_lpm=sheath)
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
        sizes = [p["dp"] for p in scan]
        scan_pane.object = f"Scan points ({len(sizes)}): {sizes}"
    except Exception as e:
        scan_pane.object = f"Scan parse error: {e}"


def set_status_threadsafe(text):
    doc = pn.state.curdoc
    if doc is not None:
        doc.add_next_tick_callback(lambda: setattr(status_text, "object", text))
    else:
        status_text.object = text


def hardware_stop_and_zero():
    global hardware_stop_pending

    try:
        try:
            if inletValve is not None:
                inletValve.off()
        except Exception as e:
            print(f"Valve off failed during stop: {e}", flush=True)

        try:
            ctl.HV.zero()
        except OSError:
            ctl.setup()
            ctl.HV.zero()

        set_status_threadsafe("Status: stopped, HV zeroed")
    except Exception as e:
        traceback.print_exc()
        set_status_threadsafe(f"Stop/zero failed: {e}")
    finally:
        hardware_stop_pending = False


def stop_and_zero():
    global phase, current_size_index, phase_start_time, dac, active_point_key, hardware_stop_pending

    measurement_running.clear()

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
    global flowmeter, blower, flow_controller, cpc, inletValve, dac

    if flow_controller is not None:
        return

    #dac = ctl.DacOut()
    #dac.block()
    flowmeter = ctl.Flowmeter()
    blower = ctl.BlowerDAC()
    cpc = ctl.CPC(cpc_com_port.value, cpc_type.value)
    inletValve = ctl.PicoValve()

    
    ctl.setup()
    ctl.HV.zero()
    time.sleep(0.3)
    #dac.allow()

    flow_controller = ctl.blower.FlowController(
        flowmeter,
        blower,
        flow_lpm=float(sheath1.value),
    )
    flow_controller.start()
    ensure_cpc_reader_thread()


def init_done_callback(fut, start_after=False):
    global init_running

    try:
        fut.result()
        set_status_threadsafe("Status: hardware initialized")
        if start_after and start_button.value:
            ensure_measurement_thread()
            measurement_running.set()
            set_status_threadsafe("Status: running")
    except Exception as e:
        traceback.print_exc()
        measurement_running.clear()
        set_status_threadsafe(f"Hardware init failed: {e}")
    finally:
        init_running = False


def init(start_after=False):
    global init_running

    if flow_controller is not None:
        ensure_cpc_reader_thread()
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


def measurement_loop():
    while True:
        if measurement_running.is_set():
            measurement_step()
        time.sleep(0.05)


def cpc_reader_loop():
    while not cpc_reader_stop.is_set():
        if cpc is None:
            time.sleep(0.1)
            continue

        started = time.monotonic()
        error = None
        try:
            value = cpc.read_instrument()
        except Exception as e:
            value = np.nan
            error = str(e)
            print(f"CPC read failed: {e}", flush=True)

        finished = time.monotonic()
        with latest_cpc_lock:
            latest_cpc["value"] = value
            latest_cpc["time"] = finished
            latest_cpc["duration_sec"] = finished - started
            latest_cpc["error"] = error

        time.sleep(cpc_poll_interval_seconds())


def ensure_cpc_reader_thread():
    global cpc_reader_thread
    if cpc_reader_thread is None or not cpc_reader_thread.is_alive():
        cpc_reader_stop.clear()
        cpc_reader_thread = threading.Thread(target=cpc_reader_loop, daemon=True)
        cpc_reader_thread.start()


def latest_cpc_snapshot():
    with latest_cpc_lock:
        value = latest_cpc["value"]
        sample_time = latest_cpc["time"]
        duration = latest_cpc["duration_sec"]
        error = latest_cpc["error"]

    age = np.nan if sample_time is None else time.monotonic() - sample_time
    return value, age, duration, error


def queue_ui_row(row):
    ui_rows_pending.append(row)


def drain_ui_updates():
    if ui_rows_pending:
        latest_df = pd.DataFrame(rows[-100:])
        table_pane.value = latest_df
        last_row_pane.object = str(ui_rows_pending[-1])
        ui_rows_pending.clear()

    value, age, duration, error = latest_cpc_snapshot()
    cpc_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if np.isfinite(cpc_value):
        current_cpc_pane.object = f"Current CPC: {cpc_value:.0f} cm^-3 | age {age:.2f}s | read {duration:.3f}s"
    elif error:
        current_cpc_pane.object = f"Current CPC: nan | error: {error}"
    else:
        current_cpc_pane.object = "Current CPC: nan"


def ensure_measurement_thread():
    global measurement_thread
    if measurement_thread is None or not measurement_thread.is_alive():
        measurement_thread = threading.Thread(target=measurement_loop, daemon=True)
        measurement_thread.start()


polarity_switch = 1
measurement_finished = True
Ntot = False


def interruptible_sleep(seconds):
    deadline = time.monotonic() + max(0.0, float(seconds))
    while time.monotonic() < deadline:
        if not measurement_running.is_set() or not start_button.value:
            return False
        time.sleep(min(0.05, deadline - time.monotonic()))
    return True


def apply_scan_point(point):
    global active_point_key

    dp = point["dp"]
    q_sheath = point["sheath"]
    key = (float(dp), float(q_sheath))
    if active_point_key == key:
        return

    flow_controller.setpoint(q_sheath)
    if "dac_code" in point:
        ctl.HV.write_dac8551(point["dac_code"])
    else:
        ctl.HV.voltage_set(dp, Q_sh_lpm=q_sheath)
    active_point_key = key


def read_cpc_count():
    value, _, _, _ = latest_cpc_snapshot()
    return value


def read_flow_value():
    try:
        return flowmeter.get_flow()
    except Exception as e:
        print(f"Flow read failed: {e}", flush=True)
        return np.nan


def cpc_poll_interval_seconds():
    try:
        return max(0.05, float(cpc_poll_interval.value))
    except Exception:
        return float(DEFAULT_SETTINGS["cpc_poll_interval"])


def append_measurement_row(point, scan_number_value, is_ntot=False, extra=None):
    sample_started = time.monotonic()
    cpc_count, cpc_age, cpc_read_duration, cpc_error = latest_cpc_snapshot()
    flow = read_flow_value()
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
        **extra,
    }

    rows.append(row)
    queue_ui_row(row)
    local_log = Path("logs") / f"measurement_{datetime.now().strftime('%Y%m%d')}.csv"
    log_row(row, local_log=local_log, cloud_log=None)
    return row


def sample_due():
    global next_sample_time

    monotonic_now = time.monotonic()
    if monotonic_now < next_sample_time:
        return False
    next_sample_time = monotonic_now + cpc_poll_interval_seconds()
    return True


def run_ntot_measurement(scan_range, scan_number, q_sheath):
    global inletValve, active_point_key, polarity_switch
    if inletValve is None:
        return []

    ntot_rows = []

    ctl.HV.zero()
    active_point_key = None
    inletValve.on()
    dp=1

    if not interruptible_sleep(float(settling_time.value)):
        inletValve.off()
        return ntot_rows

    t_start = time.time()

    while time.time() - t_start < float(Ntot_time.value):
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
    interruptible_sleep(float(settling_time.value))
    polarity_switch = 0

    ntot_values = pd.to_numeric(
        pd.DataFrame(ntot_rows).get("cpc_count", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if not ntot_values.empty:
        latest_ntot_pane.object = f"Latest measured Ntot: {ntot_values.mean():.0f}"

    return ntot_rows

def measurement_step(debug=True):
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
        last_point_set_duration_sec

    if not start_button.value:
        return

    if flow_controller is None:
        init()

    scan = get_scan_program()
    if not scan:
        status_text.object = "Status: no scan points defined"
        return

    now = time.time()
    meas_sec = float(meas_time.value)
    
  

    try:
        if phase == "idle":
            phase = "measuring"
            phase_start_time = now
            current_size_index = 0
            measurement_finished = True
            active_point_key = None

        if phase == "measuring":
            point = scan[current_size_index]
            dp = point["dp"]
            q_sheath = point["sheath"]
            scan_range = point["scan_range"]

            if measurement_finished:
                new_sign = int(np.sign(dp))
                point_set_started = time.monotonic()
                apply_scan_point(point)
                last_point_set_duration_sec = time.monotonic() - point_set_started
                if polarity_switch != 0 and new_sign != polarity_switch:
                    if not interruptible_sleep(float(polarity_switch_time.value)):
                        return
                polarity_switch = new_sign
                if not interruptible_sleep(float(settling_time.value)):
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
                    hold_sec = max(0.0, float(final_point_extra_hold.value))
                    if hold_sec > 0:
                        phase = "final_hold"
                        phase_start_time = now
                        current_size_index = len(scan) - 1
                        measurement_finished = False
                        next_sample_time = 0.0
                        return

                    current_size_index = 0
                    Ntot = True

                    scan_number += 1
                    ntot_every = max(0, int(ntot_every_n_scans.value))
                    do_ntot = ntot_every > 0 and scan_number % ntot_every == 0
                    Ntot_rows = run_ntot_measurement(scan_range, scan_number, q_sheath) if do_ntot else []
                    scan_rows.extend(Ntot_rows)

                    save_completed_scan(scan_rows, scan_number)
                    completed_scans.append(pd.DataFrame(scan_rows.copy()))

                    scan_rows.clear()
                    if do_ntot:
                        active_point_key = None

        if phase == "final_hold":
            point = scan[-1]
            apply_scan_point(point)

            if sample_due():
                row = append_measurement_row(
                    point,
                    scan_number,
                    is_ntot=False,
                    extra={
                        "point_elapsed_sec": time.time() - phase_start_time,
                        "point_set_duration_sec": last_point_set_duration_sec,
                        "phase": "final_hold",
                    },
                )
                if debug:
                    print(row, flush=True)
                scan_rows.append(row)

            if time.time() - phase_start_time < max(0.0, float(final_point_extra_hold.value)):
                return

            current_size_index = 0
            phase = "measuring"
            measurement_finished = True
            Ntot = True

            scan_number += 1
            ntot_every = max(0, int(ntot_every_n_scans.value))
            do_ntot = ntot_every > 0 and scan_number % ntot_every == 0
            Ntot_rows = run_ntot_measurement(point["scan_range"], scan_number, point["sheath"]) if do_ntot else []
            scan_rows.extend(Ntot_rows)

            save_completed_scan(scan_rows, scan_number)
            completed_scans.append(pd.DataFrame(scan_rows.copy()))

            scan_rows.clear()
            if do_ntot:
                active_point_key = None
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
        traceback.print_exc()
        status_text.object = f"Measurement error: {e}"
        print(f"Measurement error: {e}", flush=True)


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


def on_start_change(event):
    global phase, phase_start_time, current_size_index

    if event.new:
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
    group_cols = [col for col in ["scan_number", "scan_range", "polarity"] if col in grouped.columns]

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

        grouped = (
            df2_scan.groupby(["scan_number", "scan_range", "abs_size_nm", "polarity"])
            .agg(
                cpc_float=("cpc_float", "mean"),
                time=("time", "median"),
                n_samples=("cpc_float", "size"),
            )
            .reset_index()
            .sort_values(["scan_number", "polarity", "abs_size_nm"])
        )
        grouped = apply_smps_plot_step_shift_to_bins(grouped)

        latest_scan = grouped["scan_number"].max() if not grouped.empty else None
        previous = grouped[grouped["scan_number"] != latest_scan]
        if not previous.empty:
            background = (
                previous.groupby(["polarity", "abs_size_nm"])
                .agg(
                    cpc_float=("cpc_float", "mean"),
                    plot_abs_size_nm=("plot_abs_size_nm", "mean"),
                )
                .reset_index()
                .sort_values(["polarity", "plot_abs_size_nm"])
            )
            for polarity, g in background.groupby("polarity"):
                fig.add_scatter(
                    x=g["plot_abs_size_nm"],
                    y=g["cpc_float"],
                    mode="lines+markers",
                    name=f"Previous scans avg {polarity}",
                    opacity=0.25,
                    line=dict(width=4),
                    row=4,
                    col=1,
                )

        peak_rows = []
        for (sn, polarity), g in grouped.groupby(["scan_number", "polarity"]):
            g = g.sort_values("plot_abs_size_nm")
            if g.empty:
                continue

            is_latest = sn == latest_scan

            fig.add_scatter(
                x=g["plot_abs_size_nm"],
                y=g["cpc_float"],
                mode="lines+markers",
                name=f"Latest scan {polarity}" if is_latest else f"Scan {sn} {polarity}",
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
                    "polarity": polarity,
                    "time": pd.to_datetime(g["time"], errors="coerce").median(),
                    "peak_dp": float(g["plot_abs_size_nm"].iloc[idx]),
                    "peak_cpc": float(values[idx]),
                })

        if peak_rows:
            peak_df = pd.DataFrame(peak_rows).sort_values("time")
            for polarity, g in peak_df.groupby("polarity"):
                fig.add_scatter(
                    x=g["time"],
                    y=g["peak_dp"],
                    mode="lines+markers",
                    name=f"{polarity} peak Dp",
                    row=5,
                    col=1,
                )
                fig.add_scatter(
                    x=g["time"],
                    y=g["peak_cpc"],
                    mode="lines+markers",
                    name=f"{polarity} peak CPC",
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

plot_pane = pn.bind(make_plot, table_pane.param.value)

plot_box = pn.Column(
    plot_pane,
    height=1200,
    width=1550,
    sizing_mode="fixed",
    scroll=False,
)


def startup_load():
    global ui_callback

    df0 = get_recent_completed_scans(int(n_scans_plot.value))

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

    if ui_callback is None:
        ui_callback = pn.state.add_periodic_callback(drain_ui_updates, period=500, start=True)


pn.state.onload(startup_load)


def on_scan_setting_change(event):
    save_settings()
    update_scan_preview()


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
    settling_time,
    final_point_extra_hold,
    smps_plot_step_shift,
    polarity_switch_time,
    Bipolar_toggle,
    Ntot_time,
    ntot_every_n_scans,
    cpc_type,
]:
    widget.param.watch(on_scan_setting_change, "value")

start_button.param.watch(on_start_change, "value")
init_button.on_click(lambda event: init(start_after=False))
stop_button.on_click(lambda event: stop_and_zero())

update_scan_preview()

#### Layout ####
layout = pn.Column(
    "# DMA / CPC Control GUI",
    pn.Row(cpc_com_port, cpc_type),
    "# CPC / DMA control panel",
    pn.Row(start_button, status_text, init_button, stop_button),
    "### Scan range 1",
    pn.Row(range1, sheath1, steps1),
    "### Scan range 2",
    pn.Row(range2, sheath2, steps2),
    scan_pane,
    pn.Row(meas_time, cpc_poll_interval, Ntot_time, ntot_every_n_scans, n_scans_plot),
    pn.Row(settling_time, final_point_extra_hold, smps_plot_step_shift, polarity_switch_time),
    pn.Row(current_cpc_pane, latest_ntot_pane),
    "### Live data",
    last_row_pane,
    table_pane,
    "### Live plot",
    plot_box,
    sizing_mode="fixed",
    width=1600,
)

layout.servable()
