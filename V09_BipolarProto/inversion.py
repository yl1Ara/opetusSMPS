import panel as pn
import pandas as pd
import time
import inv_funcs as inv
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from plotly.subplots import make_subplots
import threading
import traceback
from scipy.optimize import nnls
from numpy.polynomial.legendre import leggauss

_GL_NODES, _GL_WEIGHTS = leggauss(5)
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

SETTINGS_FILE = Path("settings_inversion.json")

DEFAULT_SETTINGS = {}

inversion_executor = ThreadPoolExecutor(max_workers=1)
latest_inversion = None
latest_inversion_signature = None
inversion_running = False
inversion_lock = threading.Lock()

# Run with systemd service, or manually:
# source ./venv/bin/activate
# python gui.py

pn.extension("plotly")
refresh_button = pn.widgets.Button(name="Refresh scan list", button_type="primary")
plot_button = pn.widgets.Button(name="Plot selected scans", button_type="success")

plot_output = pn.pane.Plotly(height=700, width=1200)


scan_root = pn.widgets.TextInput(
    name="Scan folder",
    value="/home/yliara/Desktop/Projects/opetusSMPS/New/logs/scans",
)
scan_files = pn.widgets.MultiChoice(
    name="Select scan CSVs",
    options=[],
    value=[],
    width=700,
)


def refresh_scan_files(event=None):

    root = Path(scan_root.value)
    files = sorted(root.glob("*/*.csv"), key=lambda p: p.stem)
    print(files)

    scan_files.options = [str(p) for p in files]
    if files:
        scan_files.value = [str(files[-1])]


def load_selected_scans():
    dfs = []

    for f in scan_files.value:
        p = Path(f)
        d = pd.read_csv(p)
        d["scan_id"] = p.stem
        dfs.append(d)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def plot_selected_scans(event=None):
    df = load_selected_scans()

    if df.empty:
        return

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["cpc_float"] = pd.to_numeric(df["cpc_count"], errors="coerce")
    df["abs_size_nm"] = df["size_nm"].abs()
    df["polarity"] = np.where(df["size_nm"] > 0, "positive", "negative")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[
            "CPC concentration vs size",
            "CPC concentration vs time",
            "Sheath flow",
        ],
    )

    for (scan_id, polarity), g in df.groupby(["scan_id", "polarity"]):
        fig.add_scatter(
            x=g["abs_size_nm"],
            y=g["cpc_float"],
            mode="markers",
            name=f"{scan_id} {polarity}",
            row=1,
            col=1,
        )

        fig.add_scatter(
            x=g["time"],
            y=g["cpc_float"],
            mode="lines+markers",
            name=f"{scan_id} {polarity} time",
            row=2,
            col=1,
            showlegend=False,
        )

    for scan_id, g in df.groupby("scan_id"):
        fig.add_scatter(
            x=g["time"],
            y=g["sheath_flow"],
            mode="lines",
            name=f"{scan_id} sheath",
            row=3,
            col=1,
        )

        fig.add_scatter(
            x=g["time"],
            y=g["sheath_setpoint"],
            mode="lines",
            name=f"{scan_id} setpoint",
            row=3,
            col=1,
        )

    fig.update_xaxes(type="log", title_text="|dp| (nm)", row=1, col=1)
    fig.update_yaxes(title_text="CPC", row=1, col=1)
    fig.update_yaxes(title_text="CPC", row=2, col=1)
    fig.update_yaxes(title_text="Flow L/min", row=3, col=1)

    fig.update_layout(
        height=700,
        width=1200,
        title="Selected DMPS scans",
        showlegend=True,
    )

    plot_output.object = fig


plot_button.on_click(plot_selected_scans)

refresh_button.on_click(refresh_scan_files)


#### Widgets ####
def cunningham_correction(
    dp, T=293.15, P=101325, a=1.142, b=0.558, c=0.999, test_new_lambda=False
):
    if test_new_lambda:
        lambda_0 = 38.5e-9
        a = 1.996
        b = 0.975
        c = 1.746

    lambda_0 = 67.3e-9
    T0 = 273.15
    P0 = 101325
    lambda_air = lambda_0 * (T / T0) * (P0 / P)
    return 1 + (2 * lambda_air / dp) * (a + b * np.exp(-c * dp / (2 * lambda_air)))


def voltage_from_size(dp_nm, Q_sh_lpm=14.0, T_C=24.0, P=101325, debug=False):
    mu = 1.81e-5
    e = 1.602e-19
    negative = False

    dma = SimpleNamespace(L=0.28, r1=0.025, r2=0.033)
    r1 = dma.r1
    r2 = dma.r2
    L = dma.L

    if dp_nm <= 0:
        dp_nm = np.abs(dp_nm)
        negative = True

    dp = dp_nm * 1e-9
    Q_sh = Q_sh_lpm / 60000
    ln_r = np.log(r2 / r1)
    T_K = T_C + 273.15

    Cc = cunningham_correction(dp, T=T_K, P=P)

    V = (3 * mu * Q_sh * ln_r * dp) / (2 * L * e * Cc)

    if debug:
        if negative:
            print(f"dp: {dp_nm} nm, Cc: {Cc:.3f}, HV: {V:.1f} V")
        else:
            print(f"dp: {dp_nm} nm, Cc: {Cc:.3f}, HV: {V:.1f} V")

    if negative:
        V = -V

    return V


def ensure_settings_file():
    if not SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, "w") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2)


def save_settings():
    settings = {}

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


ensure_settings_file()
load_settings()


def invert_one_scan(d, polarity, scan_range, zn_over_zp=None, temp=293.15, press=101325):

    d = d.copy()
    d["cpc_float"] = pd.to_numeric(d["cpc_count"], errors="coerce")
    d["abs_size_nm"] = d["size_nm"].abs()
    d = d.sort_values("abs_size_nm")

    y = d.groupby("abs_size_nm")["cpc_float"].mean()
    dp_meas_nm = y.index.to_numpy(dtype=float)
    y = y.to_numpy(dtype=float)

    dp_grid_nm = dp_meas_nm.copy()
    dp_grid_m = dp_grid_nm * 1e-9
    ldp = np.log10(dp_grid_m)

    limits = np.empty(len(ldp) + 1)
    limits[0] = ldp[0] - (ldp[1] - ldp[0]) / 2
    limits[1:-1] = 0.5 * (ldp[1:] + ldp[:-1])
    limits[-1] = ldp[-1] + (ldp[-1] - ldp[-2]) / 2

    # Precompute GL quadrature points for all grid cells (shape: n_grid × n_GL)
    mids = 0.5 * (limits[:-1] + limits[1:])
    halfs = 0.5 * (limits[1:] - limits[:-1])
    # pts[j, k] = k-th GL node mapped into cell j; ravel to pass as one array
    gl_pts = (mids[:, None] + halfs[:, None] * _GL_NODES[None, :]).ravel()

    dma = SimpleNamespace(L=0.28, r1=0.025, r2=0.033)
    A = np.zeros((len(dp_meas_nm), len(dp_grid_nm)))

    qa = 1.0 / 60000.0
    qs = 1.0 / 60000.0
    q_sheath = float(d["sheath_setpoint"].median())
    qc = q_sheath / 60000.0
    qm = qc + qa - qs

    if polarity == "positive":
        p = np.arange(-1, -6, -1, dtype=float)
    else:
        p = np.arange(1, 6, 1, dtype=float)

    if zn_over_zp is None or not np.isfinite(zn_over_zp):
        zn_over_zp = 1.60e-4 / 1.35e-4

    Zp = 1e-4
    Zn = zn_over_zp * Zp

    for i, dp_nm in enumerate(dp_meas_nm):
        voltage = voltage_from_size(
            dp_nm if polarity == "positive" else -dp_nm,
            Q_sh_lpm=q_sheath,
        )

        args = (
            temp,
            press,
            p,
            voltage,
            dma.L,
            dma.r2,
            dma.r1,
            qa,
            qc,
            qm,
            qs,
            1.0,
            qa,
            1,
            Zp,
            Zn,
            140,
            101,
            1e13,
            1e13,
            "gunn woessner mod",
            0,
        )

        # Single vectorized call over all grid cells × GL nodes
        vals = inv.intfun(gl_pts, *args).reshape(len(dp_grid_nm), len(_GL_NODES))
        # Gauss-Legendre: integral over [a,b] = (b-a)/2 * dot(weights, f)
        # divided by (b-a) for the normalised kernel → just 0.5 * dot
        A[i, :] = 0.5 * vals @ _GL_WEIGHTS

    x, rnorm = nnls(A, y)

    return pd.DataFrame(
        {
            "abs_size_nm": dp_grid_nm,
            "N_GWalpha": x,
        }
    )


layout = pn.Column(
    "# Offline DMPS inversion / scan viewer",
    pn.Row(scan_root, refresh_button),
    scan_files,
    plot_button,
    plot_output,
    width=1300,
)

refresh_scan_files()
layout.servable()
# To host via Tailscale. Update the websocket_origin list with your Tailscale IPs to get access.
