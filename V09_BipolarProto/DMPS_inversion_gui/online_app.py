from datetime import date
import copy
import json
from statistics import LinearRegression
import sys
import time
import traceback
import threading
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import panel as pn
from plotly.subplots import make_subplots
from scipy.integrate import trapezoid
from scipy.optimize import nnls
from numpy.polynomial.legendre import leggauss

_GL_NODES, _GL_WEIGHTS = leggauss(5)

import inv_funcs as inv
from DMPS_inversion_gui import diagnostics as diag
from DMPS_inversion_gui.cpc_delay import (
    assign_cpc_samples_to_setpoints,
    build_response_kernel,
    deduplicate_cpc_rows,
    response_kernel_ill_conditioned,
    response_kernel_rejection_reason,
    solve_response_kernel_nnls,
)
from inv_funcs.cpc_loss import cpc_loss1
from inv_funcs.dmps_loss import dmps_loss1
from inv_funcs.ltubefl import ltubefl


# ---------------------------------------------------------------------
# Settings / constants
# ---------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = APP_ROOT / "settings_inversion.json"
APP_VERSION = (APP_ROOT / "VERSION").read_text().strip()

DEFAULT_SETTINGS = {
    "scan_root": "logs/scans",
    "save_root": "~/OneDrive/DMPS_inversions",
    "n_scans_plot": 200,
    "scan_selection_mode": "Newest N",
    "scan_start_date": None,
    "scan_start_time": "00:00",
    "scan_end_date": None,
    "scan_end_time": "23:59",
    "loaded_time_window_min": [0, 1440],
    "auto_interval_min": 30,
    "auto_file_age_sec": 2,
    "daily_overwrite": True,
    "dma_L": 0.28,
    "dma_r1": 0.025,
    "dma_r2": 0.033,
    "qa_lpm": 1.0,
    "qs_lpm": 1.0,
    "temp_K": 293.15,
    "press_Pa": 101325,
    "zratio": 1.35e-4 / 1.60e-4,
    "zratio_min": 0.3,
    "zratio_max": 3.0,
    "zratio_smoothing_step": 0.2,
    "zratio_min_size_nm": 10.0,
    "zratio_estimate_offset": 0.2,
    "ntot_plot_max": 10000,
    "heatmap_clip": 20000,
    "raw_uncertainty": "percentile 10-90",
    "growth_models": ["Lower edge D25", "Center D50", "Upper edge D75", "Ridge peak"],
    "growth_min_size_nm": 6.5,
    "growth_max_size_nm": 30.0,
    "growth_threshold_fraction": 0.35,
    "growth_max_gap_minutes": 90.0,
    "growth_min_event_scans": 4,
    "growth_max_rate_nm_h": 15.0,
    "difference_peak_min_size_nm": 30.0,
    "smallest_size": 6.5,
    "scan_inversion_type": "DMPS",
    "smps_settling_time_sec": 30.0,
    "smps_correction_mode": "Transport delay",
    "smps_transport_delay_sec": 30.0,
    "smps_response_window_sec": 1.0,
    "smps_dwell_sec": 30.0,
    "smps_kernel_timing_override": False,
    "smps_kernel_smoothness": 0.1,
    "smps_size_step_shift": 0,
    "smps_timing_offset_min_sec": -300.0,
    "smps_timing_offset_max_sec": 300.0,
    "smps_timing_offset_step_sec": 10.0,
    "smps_timing_match_tolerance_min": 15.0,
    "inversion_size_bin_decimals": 1,
    "cpc_gap_interpolation_enabled": True,
    "low_value_lift_enabled": False,
    "low_value_lift_ratio": 0.85,
    "low_value_lift_alpha": 1.0,
    "inversion_methods": ["gunn woessner mod"],
    "tube_segments": "tubediameter,tubelength,aflow,angle\n0,1.93,qa,0\n0,2.80,8,0\n0,5.21,1.3,0",
}

INVERSION_METHODS = {
    "gunn woessner mod": "Gunn-Woessner modified",
    "wiedensohler": "Wiedensohler",
    "fuchs": "Fuchs",
}
INVERSION_METHOD_LABELS = {label: method for method, label in INVERSION_METHODS.items()}

pn.extension("plotly")

inversion_executor = ThreadPoolExecutor(max_workers=1)
inversion_lock = threading.Lock()
inversion_running = False
latest_inversion = None
latest_difference_diagnostics = None
latest_growth_diagnostics = []
latest_growth_settings = {}
auto_pending_signature = None
AUTO_STATE_FILE = APP_ROOT / "auto_inversion_state.json"
scan_time_cache = {}
SHARED_STATE_KEY = "online_inversion_viewer_shared_state"
shared_state = pn.state.cache.setdefault(
    SHARED_STATE_KEY,
    {
        "lock": threading.Lock(),
        "version": 0,
        "raw_fig": None,
        "inversion_fig": None,
        "residual_fig": None,
        "smps_timing_fig": None,
        "difference_fig": None,
        "difference_diagnostics": None,
        "growth_diagnostics": [],
        "growth_settings": {},
        "latest_inversion": None,
        "status": "Status: idle",
    },
)
shared_state.setdefault("residual_fig", None)
shared_state.setdefault("smps_timing_fig", None)
shared_state.setdefault("growth_diagnostics", [])
shared_state.setdefault("growth_settings", {})
with shared_state["lock"]:
    local_shared_version = (
        shared_state["version"]
        if shared_state["raw_fig"] is None
        and shared_state["inversion_fig"] is None
        and shared_state["residual_fig"] is None
        and shared_state["smps_timing_fig"] is None
        and shared_state["difference_fig"] is None
        and shared_state["latest_inversion"] is None
        else -1
    )


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def ensure_settings_file():
    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(json.dumps(DEFAULT_SETTINGS, indent=2))


def load_settings():
    ensure_settings_file()
    try:
        settings = json.loads(SETTINGS_FILE.read_text())
        if "smps_correction_mode" not in settings and "smps_size_step_shift" in settings:
            settings["smps_correction_mode"] = "Integer step shift"
            SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
        return settings
    except json.JSONDecodeError:
        broken = SETTINGS_FILE.with_name("settings_inversion_broken.json")
        SETTINGS_FILE.rename(broken)
        ensure_settings_file()
        return json.loads(SETTINGS_FILE.read_text())


def save_settings():
    settings = {
        "scan_root": scan_root.value,
        "save_root": save_root.value,
        "n_scans_plot": int(n_scans_plot.value),
        "scan_selection_mode": scan_selection_mode.value,
        "scan_start_date": as_date(scan_start_date.value).isoformat() if as_date(scan_start_date.value) else None,
        "scan_start_time": scan_start_time.value,
        "scan_end_date": as_date(scan_end_date.value).isoformat() if as_date(scan_end_date.value) else None,
        "scan_end_time": scan_end_time.value,
        "loaded_time_window_min": [int(x) for x in loaded_time_window_min.value],
        "auto_interval_min": int(auto_interval_min.value),
        "auto_file_age_sec": int(auto_file_age_sec.value),
        "daily_overwrite": bool(daily_overwrite_checkbox.value),
        "dma_L": float(dma_L.value),
        "dma_r1": float(dma_r1.value),
        "dma_r2": float(dma_r2.value),
        "qa_lpm": float(qa_lpm.value),
        "qs_lpm": float(qs_lpm.value),
        "temp_K": float(temp_K.value),
        "press_Pa": float(press_Pa.value),
        "zratio": float(zratio_widget.value),
        "zratio_min": float(zratio_min_widget.value),
        "zratio_max": float(zratio_max_widget.value),
        "zratio_smoothing_step": float(zratio_smoothing_step.value),
        "zratio_min_size_nm": float(zratio_min_size_nm.value),
        "zratio_estimate_offset": float(zratio_estimate_offset.value),
        "ntot_plot_max": float(ntot_plot_max.value),
        "heatmap_clip": float(heatmap_clip.value),
        "raw_uncertainty": raw_uncertainty.value,
        "growth_models": list(growth_models.value),
        "growth_min_size_nm": float(growth_min_size_nm.value),
        "growth_max_size_nm": float(growth_max_size_nm.value),
        "growth_threshold_fraction": float(growth_threshold_fraction.value),
        "growth_max_gap_minutes": float(growth_max_gap_minutes.value),
        "growth_min_event_scans": int(growth_min_event_scans.value),
        "growth_max_rate_nm_h": float(growth_max_rate_nm_h.value),
        "difference_peak_min_size_nm": float(difference_peak_min_size_nm.value),
        "smallest_size": float(smallest_size.value),
        "scan_inversion_type": scan_inversion_type.value,
        "smps_settling_time_sec": float(smps_settling_time_sec.value),
        "smps_correction_mode": smps_correction_mode.value,
        "smps_transport_delay_sec": float(smps_transport_delay_sec.value),
        "smps_response_window_sec": float(smps_response_window_sec.value),
        "smps_dwell_sec": float(smps_dwell_sec.value),
        "smps_kernel_timing_override": bool(smps_kernel_timing_override.value),
        "smps_kernel_smoothness": float(smps_kernel_smoothness.value),
        "smps_size_step_shift": int(smps_size_step_shift.value),
        "smps_timing_offset_min_sec": float(smps_timing_offset_min_sec.value),
        "smps_timing_offset_max_sec": float(smps_timing_offset_max_sec.value),
        "smps_timing_offset_step_sec": float(smps_timing_offset_step_sec.value),
        "smps_timing_match_tolerance_min": float(smps_timing_match_tolerance_min.value),
        "inversion_size_bin_decimals": int(inversion_size_bin_decimals.value),
        "cpc_gap_interpolation_enabled": bool(cpc_gap_interpolation_enabled.value),
        "low_value_lift_enabled": bool(low_value_lift_enabled.value),
        "low_value_lift_ratio": float(low_value_lift_ratio.value),
        "low_value_lift_alpha": float(low_value_lift_alpha.value),
        "inversion_methods": selected_inversion_methods(),
        "tube_segments": tube_segments.value,
    }
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Index, pd.Series)):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def update_log_size_axis(fig, row, sizes):
    sizes = np.asarray(sizes, dtype=float)
    finite_sizes = sizes[np.isfinite(sizes)]
    if len(finite_sizes) == 0:
        fig.update_yaxes(type="log", title_text="dp (nm)", row=row, col=1)
        return
    y_min = max(float(smallest_size.value), np.nanmin(finite_sizes))
    y_max = np.nanmax(finite_sizes)
    if np.isfinite(y_min) and np.isfinite(y_max) and y_min > 0 and y_max > y_min:
        fig.update_yaxes(
            type="log",
            range=[np.log10(y_min), np.log10(y_max)],
            title_text="dp (nm)",
            row=row,
            col=1,
        )
    else:
        fig.update_yaxes(type="log", title_text="dp (nm)", row=row, col=1)


def update_log_size_x_axis(fig, row, sizes):
    sizes = np.asarray(sizes, dtype=float)
    finite_sizes = sizes[np.isfinite(sizes)]
    if len(finite_sizes) == 0:
        fig.update_xaxes(type="log", title_text="Dp (nm)", row=row, col=1)
        return
    x_min = max(float(smallest_size.value), np.nanmin(finite_sizes))
    x_max = np.nanmax(finite_sizes)
    if np.isfinite(x_min) and np.isfinite(x_max) and x_min > 0 and x_max > x_min:
        fig.update_xaxes(
            type="log",
            range=[np.log10(x_min), np.log10(x_max)],
            title_text="Dp (nm)",
            row=row,
            col=1,
        )
    else:
        fig.update_xaxes(type="log", title_text="Dp (nm)", row=row, col=1)


def app_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return APP_ROOT / path


def parse_saved_date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def parse_time_text(value, fallback):
    text = str(value or "").strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return pd.to_datetime(text, format=fmt).time()
        except Exception:
            pass
    return pd.to_datetime(fallback, format="%H:%M").time()


def as_date(value):
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()

def save_data(event=None):
    if latest_inversion is None:
        status.object = "No inversion data to save yet."
        return
    outdir = app_path(save_root.value)
    outdir.mkdir(parents=True, exist_ok=True)
    if daily_overwrite_checkbox.value:
        stamp = pd.Timestamp.now().strftime("%Y%m%d")
    else:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if raw_plot.object is not None:
        raw_plot.object.write_html(outdir / f"raw_plot_{stamp}.html")
    if inversion_plot.object is not None:
        inversion_plot.object.write_html(outdir / f"inversion_plot_{stamp}.html")
    if residual_plot.object is not None:
        residual_plot.object.write_html(outdir / f"residual_diagnostics_{stamp}.html")
    if difference_plot.object is not None:
        difference_plot.object.write_html(outdir / f"difference_diagnostics_{stamp}.html")
    if smps_timing_plot.object is not None:
        smps_timing_plot.object.write_html(outdir / f"smps_timing_diagnostics_{stamp}.html")
    ntot_tables = []
    measured_ntot_saved = False
    for tr in latest_inversion:
        if tr["kind"] == "heatmap":
            method = tr.get("method", "gunn woessner mod")
            method_name = method.replace(" ", "_")
            z = np.asarray(tr["Z"], dtype=float)
            heatmap_df = pd.DataFrame(
                z.T,
                index=pd.to_datetime(tr["x"]),
                columns=np.asarray(tr["y"], dtype=float),
            )
            heatmap_df.index.name = "time"
            heatmap_df.columns.name = "size_nm"
            heatmap_df.to_csv(
                outdir / f"heatmap_{method_name}_{tr['polarity']}_{stamp}.csv"
            )
        elif tr["kind"] == "ntot":
            polarity = tr["polarity"]
            method = tr.get("method", "gunn woessner mod")
            method_name = method.replace(" ", "_")
            d = pd.DataFrame({
                "time": pd.to_datetime(tr["x"]),
                f"Ntot_{method_name}_{polarity}_inverted": tr["y"],
            })
            if polarity == "positive" and "y_measured" in tr and not measured_ntot_saved:
                d["Ntot_measured"] = tr["y_measured"]
                measured_ntot_saved = True
            ntot_tables.append(d)
            
        elif tr["kind"] == "ion_ratio":
            ion_ratio_df = pd.DataFrame({
                "time": pd.to_datetime(tr["x"]),
                "Zp_Zn_raw": tr["y"],
                "Zp_Zn_smoothed": tr.get("y_smoothed", tr["y"]),
                "selected_dp_nm": tr["selected_dp"],
            })
            ion_ratio_df = ion_ratio_df.set_index("time").sort_index()
            ion_ratio_df.to_csv(outdir / f"estimated_z_ratio_{stamp}.csv")
        elif tr["kind"] in {
            "cpc_assignment_diagnostics",
            "range_overlap_diagnostics",
            "ntot_closure_diagnostics",
            "kernel_sample_residuals",
        }:
            diagnostic_rows = tr.get("rows", [])
            if diagnostic_rows:
                pd.DataFrame(diagnostic_rows).to_csv(
                    outdir / f"{tr['kind']}_{stamp}.csv", index=False
                )
                (outdir / f"{tr['kind']}_{stamp}.json").write_text(
                    json.dumps(_json_safe(diagnostic_rows), indent=2)
                )
    if ntot_tables:
        ntot_df = ntot_tables[0]
        for d in ntot_tables[1:]:
            ntot_df = pd.merge(ntot_df, d, on="time", how="outer")
        ntot_df = ntot_df.sort_values("time")
        try:
            t0 = ntot_df["time"].min()
            t1 = ntot_df["time"].max()

            smear_cpc = load_smeariii_cpc_concentration(
                t0 - pd.Timedelta(hours=1),
                t1 - pd.Timedelta(hours=1),
            )
            smear_cpc["time"] = smear_cpc["time"] + pd.Timedelta(hours=1)
            smear_cpc = smear_cpc[smear_cpc["time"].between(t0, t1)].copy()

            if not smear_cpc.empty:
                ntot_df = pd.merge_asof(
                    ntot_df.sort_values("time"),
                    smear_cpc[["time", "SMEARIII_CPC"]].sort_values("time"),
                    on="time",
                    direction="nearest",
                    tolerance=pd.Timedelta(minutes=15),
                )
        except Exception as e:
            print(f"Could not save SMEAR III CPC: {e}", flush=True)
        ntot_df = ntot_df.set_index("time")
        ntot_df.to_csv(outdir / f"ntot_{stamp}.csv")
    if latest_difference_diagnostics is not None:
        save_difference_diagnostics_data(outdir, stamp, latest_difference_diagnostics)
    growth_diagnostics = latest_growth_diagnostics
    if growth_diagnostics:
        summary_rows = []
        track_rows = []
        for growth in growth_diagnostics:
            summary_rows.append({
                key: value for key, value in growth.items()
                if key not in {"time", "dp", "fit"}
            })
            for timestamp, diameter, fitted in zip(
                growth["time"], growth["dp"], growth["fit"]
            ):
                track_rows.append({
                    "event_id": growth["event_id"],
                    "model": growth["model"],
                    "method": growth["source_method"],
                    "polarity": growth["polarity"],
                    "time": timestamp,
                    "diameter_nm": diameter,
                    "fitted_diameter_nm": fitted,
                })
        pd.DataFrame(summary_rows).to_csv(
            outdir / f"npf_growth_model_summary_{stamp}.csv", index=False
        )
        pd.DataFrame(track_rows).to_csv(
            outdir / f"npf_growth_model_tracks_{stamp}.csv", index=False
        )
        (outdir / f"npf_growth_models_{stamp}.json").write_text(
            json.dumps(_json_safe({
                "settings": latest_growth_settings,
                "diagnostics": growth_diagnostics,
            }), indent=2)
        )
    elif daily_overwrite_checkbox.value:
        for filename in (
            f"npf_growth_model_summary_{stamp}.csv",
            f"npf_growth_model_tracks_{stamp}.csv",
            f"npf_growth_models_{stamp}.json",
        ):
            (outdir / filename).unlink(missing_ok=True)
    status.object = f"Saved plots and data to `{outdir}`."


def save_difference_diagnostics_data(outdir, stamp, diagnostics):
    matches = diagnostics.get("matches")
    if matches is not None and not matches.empty:
        matches.to_csv(outdir / f"difference_matches_{stamp}.csv", index=False)

    ratio_rows = []
    for item in diagnostics.get("ratios", []):
        for size_nm, ratio in zip(item["size_nm"], item["ratio_median"]):
            ratio_rows.append({
                "method": item["method"],
                "polarity": item["polarity"],
                "size_nm": size_nm,
                "our_smear_ratio_median": ratio,
                "n_matches": item["n_matches"],
            })
    if ratio_rows:
        pd.DataFrame(ratio_rows).to_csv(outdir / f"difference_ratio_curves_{stamp}.csv", index=False)

    shape_rows = []
    for item in diagnostics.get("shapes", []):
        for size_nm, our, smear in zip(item["size_nm"], item["our_median"], item["smear_median"]):
            shape_rows.append({
                "method": item["method"],
                "polarity": item["polarity"],
                "size_nm": size_nm,
                "our_median": our,
                "smear_median": smear,
                "n_matches": item["n_matches"],
            })
    if shape_rows:
        pd.DataFrame(shape_rows).to_csv(outdir / f"difference_shape_curves_{stamp}.csv", index=False)

def cunningham_correction(dp, T=293.15, P=101325, a=1.142, b=0.558, c=0.999):
    lambda_0 = 67.3e-9
    T0 = 273.15
    P0 = 101325
    lambda_air = lambda_0 * (T / T0) * (P0 / P)
    return 1 + (2 * lambda_air / dp) * (a + b * np.exp(-c * dp / (2 * lambda_air)))


def voltage_from_size(dp_nm, q_sh_lpm, dma, temp_K=293.15, press_Pa=101325):
    mu = 1.81e-5
    e = 1.602176634e-19

    sign = -1 if dp_nm < 0 else 1
    dp = abs(float(dp_nm)) * 1e-9
    q_sh = float(q_sh_lpm) / 60000.0
    ln_r = np.log(dma.r2 / dma.r1)
    cc = cunningham_correction(dp, T=temp_K, P=press_Pa)

    v = (3 * mu * q_sh * ln_r * dp) / (2 * dma.L * e * cc)
    return sign * v


def get_dma():
    return SimpleNamespace(
        L=float(dma_L.value),
        r1=float(dma_r1.value),
        r2=float(dma_r2.value),
    )


def inversion_size_column(df):
    if "inversion_size_nm" in df.columns:
        return "inversion_size_nm"
    return "size_nm"


def inversion_size_bin_decimals_value():
    try:
        decimals = int(inversion_size_bin_decimals.value)
    except Exception:
        decimals = int(DEFAULT_SETTINGS["inversion_size_bin_decimals"])
    return max(0, decimals)


def merged_abs_size_nm(values):
    sizes = pd.to_numeric(values, errors="coerce").abs()
    return sizes.round(inversion_size_bin_decimals_value())


def get_scan_size_axis(df):
    size_col = inversion_size_column(df)
    sizes = sorted(merged_abs_size_nm(df[size_col]).dropna().unique())
    return np.asarray(sizes, dtype=float)


def one_sided_low_value_lift(values):
    y = np.asarray(values, dtype=float)
    lifted = y.copy()
    valid = np.isfinite(y) & (y > 0)
    if np.count_nonzero(valid) < 3:
        return lifted

    try:
        ratio = float(low_value_lift_ratio.value)
    except Exception:
        ratio = float(DEFAULT_SETTINGS["low_value_lift_ratio"])
    ratio = float(np.clip(ratio, 0.05, 1.0))

    try:
        blend = float(low_value_lift_alpha.value)
    except Exception:
        blend = float(DEFAULT_SETTINGS["low_value_lift_alpha"])
    blend = float(np.clip(blend, 0.05, 1.0))

    for _ in range(3):
        previous = np.full(len(lifted), np.nan)
        next_value = np.full(len(lifted), np.nan)

        last = np.nan
        for i, value in enumerate(lifted):
            previous[i] = last
            if np.isfinite(value) and value > 0:
                last = value

        last = np.nan
        for i in range(len(lifted) - 1, -1, -1):
            next_value[i] = last
            value = lifted[i]
            if np.isfinite(value) and value > 0:
                last = value

        neighbor_ok = np.isfinite(previous) & np.isfinite(next_value) & (previous > 0) & (next_value > 0)
        expected = np.full(len(lifted), np.nan)
        expected[neighbor_ok] = np.sqrt(previous[neighbor_ok] * next_value[neighbor_ok])
        floor = expected * ratio
        low = neighbor_ok & (~np.isfinite(lifted) | (lifted <= 0) | (lifted < floor))
        if not np.any(low):
            break

        current = np.where(np.isfinite(lifted) & (lifted > 0), lifted, 0.0)
        lifted[low] = current[low] + blend * (floor[low] - current[low])

    return lifted


def interpolate_cpc_gaps_for_inversion(size_nm, cpc_values):
    sizes = np.asarray(size_nm, dtype=float)
    y = np.asarray(cpc_values, dtype=float)
    keep = np.isfinite(sizes) & (sizes > 0)
    sizes = sizes[keep]
    repaired = y[keep].copy()
    valid = np.isfinite(repaired) & (repaired > 0)
    if np.count_nonzero(valid) == 0:
        return sizes[:0], repaired[:0]

    try:
        ratio = float(low_value_lift_ratio.value)
    except Exception:
        ratio = float(DEFAULT_SETTINGS["low_value_lift_ratio"])
    ratio = float(np.clip(ratio, 0.05, 1.0))

    log_sizes = np.log10(sizes)
    if np.count_nonzero(valid) == 1:
        expected = np.full(len(sizes), np.nan, dtype=float)
        expected[valid] = repaired[valid]
    else:
        expected = np.exp(np.interp(log_sizes, log_sizes[valid], np.log(repaired[valid])))
        expected[(log_sizes < log_sizes[valid].min()) | (log_sizes > log_sizes[valid].max())] = np.nan
    floor = expected * ratio
    low = ~np.isfinite(repaired) | (repaired <= 0) | (repaired < floor)
    repaired[low] = floor[low]
    measured = np.isfinite(repaired)
    return sizes[measured], repaired[measured]


def apply_smps_size_shift(df):
    df = df.copy()
    df["inversion_size_nm"] = pd.to_numeric(df["size_nm"], errors="coerce")
    return df


def smps_size_step_shift_value():
    try:
        return int(smps_size_step_shift.value)
    except Exception:
        return int(DEFAULT_SETTINGS["smps_size_step_shift"])


def build_cpc_series_for_inversion(d):
    if scan_inversion_type.value != "SMPS":
        return d.groupby("abs_size_nm")["cpc_float"].mean()

    if smps_correction_mode.value == "Transport delay":
        assignment = assign_cpc_samples_to_setpoints(
            d,
            delay_seconds=float(smps_transport_delay_sec.value),
            settling_seconds=float(smps_settling_time_sec.value),
        )
        assignment.cpc_by_size.attrs["assignment_diagnostics"] = assignment.diagnostics
        return assignment.cpc_by_size

    if smps_correction_mode.value == "None":
        return d.groupby("abs_size_nm")["cpc_float"].mean()

    shift = smps_size_step_shift_value()
    if shift == 0:
        return d.groupby("abs_size_nm")["cpc_float"].mean()

    sizes = np.asarray(sorted(d["abs_size_nm"].dropna().unique()), dtype=float)
    if len(sizes) < 2:
        return d.groupby("abs_size_nm")["cpc_float"].mean()

    size_to_index = {size: i for i, size in enumerate(sizes)}
    parts = []

    if "phase" in d.columns:
        final_hold = d[d["phase"].astype(str).str.lower() == "final_hold"].copy()
        normal = d.drop(index=final_hold.index)
    else:
        final_hold = d.iloc[0:0].copy()
        normal = d

    averaged = normal.groupby("abs_size_nm")["cpc_float"].mean()
    for size, value in averaged.items():
        source_index = size_to_index.get(float(size))
        if source_index is None:
            continue
        target_index = int(np.clip(source_index - shift, 0, len(sizes) - 1))
        parts.append({"abs_size_nm": sizes[target_index], "cpc_float": value})

    if not final_hold.empty:
        if "cpc_sample_id" in final_hold.columns:
            hold_ids = pd.to_numeric(final_hold["cpc_sample_id"], errors="coerce")
            final_hold = final_hold[hold_ids.notna() & (hold_ids > 0)].copy()
            if "cpc_sample_id" in normal.columns:
                normal_ids = set(pd.to_numeric(normal["cpc_sample_id"], errors="coerce").dropna())
                hold_ids = pd.to_numeric(final_hold["cpc_sample_id"], errors="coerce")
                final_hold = final_hold[~hold_ids.isin(normal_ids)]
            final_hold = final_hold.drop_duplicates("cpc_sample_id", keep="first")
        sort_column = "cpc_sample_time" if "cpc_sample_time" in final_hold.columns else "time"
        final_hold = final_hold.sort_values(sort_column)
        final_index = len(sizes) - 1
        for hold_index, row in enumerate(final_hold.itertuples(index=False), start=1):
            source_size = float(getattr(row, "abs_size_nm"))
            source_index = size_to_index.get(source_size, final_index)
            if shift > 0 and source_index == final_index:
                applied_shift = max(0, shift - hold_index)
            else:
                applied_shift = shift
            target_index = int(np.clip(source_index - applied_shift, 0, len(sizes) - 1))
            parts.append({"abs_size_nm": sizes[target_index], "cpc_float": getattr(row, "cpc_float")})

    if not parts:
        return averaged

    shifted = pd.DataFrame(parts).groupby("abs_size_nm")["cpc_float"].mean()
    return shifted.reindex(sizes).combine_first(averaged.reindex(sizes))


def normalize_inversion_methods(values):
    if isinstance(values, str):
        values = [values]

    methods = []
    for value in values:
        if value in INVERSION_METHODS:
            method = value
        else:
            method = INVERSION_METHOD_LABELS.get(value)
        if method is not None and method not in methods:
            methods.append(method)
    return methods


def selected_inversion_methods():
    methods = normalize_inversion_methods(inversion_methods.value)
    return methods or DEFAULT_SETTINGS["inversion_methods"]


def parse_tube_segments(text, qa, qs=None, qc=None, qm=None):
    flow_names = {
        "qa": qa,
        "aflow": qa,
        "aerosol": qa,
        "aerosolflow": qa,
        "qs": qs,
        "sample": qs,
        "sampleflow": qs,
        "qc": qc,
        "sheath": qc,
        "sheathflow": qc,
        "qm": qm,
    }
    segments = []

    for line_no, raw_line in enumerate(str(text).splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        parts = [part.strip() for part in line.replace(";", ",").split(",")]
        if len(parts) < 3:
            raise ValueError(f"Tube segment line {line_no}: expected diameter,length,flow[,angle]")

        try:
            diameter = float(parts[0])
            length = float(parts[1])
        except ValueError:
            if line_no == 1:
                continue
            raise ValueError(f"Tube segment line {line_no}: diameter and length must be numbers")

        flow_key = parts[2].lower().replace("_", "")
        if flow_key in flow_names and flow_names[flow_key] is not None:
            flow = float(flow_names[flow_key])
        else:
            try:
                flow = float(parts[2]) / 60000.0
            except ValueError as exc:
                raise ValueError(
                    f"Tube segment line {line_no}: flow must be L/min or one of qa, qs, qc, qm"
                ) from exc

        try:
            angle = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
        except ValueError as exc:
            raise ValueError(f"Tube segment line {line_no}: angle must be numeric") from exc

        if length <= 0 or flow <= 0:
            raise ValueError(f"Tube segment line {line_no}: length and flow must be positive")

        segments.append((diameter, length, flow, angle))

    if not segments:
        raise ValueError("Tube segments cannot be empty")

    return tuple(segments)


def method_label(method):
    return INVERSION_METHODS.get(method, method)


def dmps_loss_correction_factor_from_distribution(
    dp_nm, n_inv, tube_segments, qa, temp, press, cpc_type="3010"
):
    dp_nm = np.asarray(dp_nm, dtype=float)
    n_inv = np.asarray(n_inv, dtype=float)
    mask = np.isfinite(dp_nm) & np.isfinite(n_inv) & (dp_nm > 0) & (n_inv > 0)
    if not np.any(mask):
        return 1.0

    dp_m = dp_nm[mask] * 1e-9
    n = n_inv[mask]
    order = np.argsort(dp_m)
    dp_m = dp_m[order]
    n = n[order]

    tube_loss = np.ones_like(dp_m, dtype=float)
    for _, length, flow, *_ in tube_segments:
        tube_loss *= ltubefl(dp_m, length, flow, temp, press)

    loss = tube_loss * cpc_loss1(dp_m, temp, press, cpc_type=cpc_type) * dmps_loss1(dp_m, qa, temp, press)
    if len(n) == 1:
        if not np.isfinite(loss[0]) or loss[0] <= 0:
            return 1.0
        return 1.0 / loss[0]

    total = trapezoid(n, np.log(dp_m))
    if not np.isfinite(total) or total <= 0:
        return 1.0

    effective_transmission = trapezoid(n * loss, np.log(dp_m)) / total
    if not np.isfinite(effective_transmission) or effective_transmission <= 0:
        return 1.0
    return 1.0 / effective_transmission

import requests

SMEAR_API = "https://smear-backend-avaa-smear-prod.2.rahtiapp.fi"
def load_smeariii_cpc_concentration(start, end):
    url = f"{SMEAR_API}/search/timeseries"
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    params = {
        "from": start.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "tablevariable": "KUM_AERO.cn",
        "quality": "ANY",
        "aggregation": "NONE",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    df = pd.DataFrame(payload["data"])
    if df.empty:
        print(
            f"No SMEAR III CPC API data for {params['from']} to {params['to']}",
            flush=True,
        )
        return pd.DataFrame(columns=["time", "SMEARIII_CPC"])
    df = df.rename(columns={
        "samptime": "time",
        "KUM_AERO.cn": "SMEARIII_CPC",
    })
    df["time"] = pd.to_datetime(df["time"])
    df["SMEARIII_CPC"] = pd.to_numeric(df["SMEARIII_CPC"], errors="coerce")
    return df[["time", "SMEARIII_CPC"]]


def load_smeariii_sum_file(path):
    rows = []
    with Path(path).open() as file:
        lines = [line.strip() for line in file if line.strip()]

    i = 2
    while i + 1 < len(lines):
        size_parts = lines[i].split()
        conc_parts = lines[i + 1].split()
        i += 2

        if len(size_parts) < 10 or len(conc_parts) != len(size_parts):
            continue

        try:
            scan_time = pd.Timestamp(
                year=int(size_parts[0]),
                month=int(size_parts[1]),
                day=int(size_parts[2]),
                hour=int(size_parts[3]),
                minute=int(size_parts[4]),
                second=int(size_parts[5]),
            )
            sizes = np.asarray(size_parts[9:], dtype=float)
            concs = np.asarray(conc_parts[9:], dtype=float)
        except ValueError:
            continue

        for size_nm, conc in zip(sizes, concs):
            rows.append((scan_time, size_nm, conc))

    return pd.DataFrame(rows, columns=["time", "size_nm", "smear_conc"])


def load_smeariii_sum_range(start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    root = APP_ROOT / "SMEARIII"
    tables = []

    for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        path = root / f"DMPS007_{day:%Y%m%d}.sum"
        if path.exists():
            tables.append(load_smeariii_sum_file(path))

    if not tables:
        return pd.DataFrame(columns=["time", "size_nm", "smear_conc"])

    df = pd.concat(tables, ignore_index=True)
    df = df[df["time"].between(start, end)].copy()
    if df.empty:
        return pd.DataFrame(columns=["time", "size_nm", "smear_conc"])

    return df.sort_values(["time", "size_nm"])


def result_time_range(result):
    times = []
    for tr in result:
        if tr.get("kind") in {"heatmap", "ntot"} and len(tr.get("x", [])) > 0:
            times.extend(pd.to_datetime(tr["x"]))
    if not times:
        return None, None
    times = pd.to_datetime(times)
    return times.min(), times.max()


def three_day_median_window(result):
    t0, t1 = result_time_range(result)
    if t0 is None:
        return None, None
    start = t0.normalize()
    end = start + pd.Timedelta(days=3)
    return start, end


def build_median_distributions(result):
    start, end = three_day_median_window(result)
    medians = []

    for tr in result:
        if tr.get("kind") != "heatmap":
            continue
        z = np.asarray(tr["Z"], dtype=float)
        if z.size == 0:
            continue
        flow_rel_rmse = np.asarray(tr.get("flow_rel_rmse", []), dtype=float)
        if len(flow_rel_rmse) != z.shape[1]:
            flow_rel_rmse = np.full(z.shape[1], np.nan)
        if start is not None:
            times = pd.to_datetime(tr["x"])
            mask = (times >= start) & (times < end)
            if not np.any(mask):
                continue
            z = z[:, mask]
            flow_rel_rmse = flow_rel_rmse[mask]
        stats = diag.nan_stats_by_row(z)
        median_flow_rel_rmse = np.nanmedian(flow_rel_rmse)
        if not np.isfinite(median_flow_rel_rmse):
            median_flow_rel_rmse = 0.0
        flow_error = stats["median"] * median_flow_rel_rmse
        medians.append({
            "label": f"{method_label(tr.get('method', 'gunn woessner mod'))} {tr['polarity']}",
            "dp": np.asarray(tr["y"], dtype=float),
            "median": stats["median"],
            "p10": stats["p10"],
            "p90": stats["p90"],
            "flow_error": flow_error,
            "flow_rel_rmse": median_flow_rel_rmse,
            "n_scans": int(z.shape[1]),
        })

    if start is None:
        return medians

    smear = load_smeariii_sum_range(start, end)
    if not smear.empty:
        scan_sizes = []
        scan_concs = []
        for _, smear_scan in smear.groupby("time"):
            smear_scan = smear_scan.sort_values("size_nm")
            size_nm = smear_scan["size_nm"].to_numpy(dtype=float)
            conc = smear_scan["smear_conc"].to_numpy(dtype=float)
            if len(size_nm) > 0:
                scan_sizes.append(size_nm)
                scan_concs.append(conc)

        if scan_concs:
            channel_count = max(set(map(len, scan_concs)), key=list(map(len, scan_concs)).count)
            scan_sizes = [size_nm for size_nm in scan_sizes if len(size_nm) == channel_count]
            scan_concs = [conc for conc in scan_concs if len(conc) == channel_count]
            if scan_concs:
                smear_stats = diag.nan_stats_by_row(np.vstack(scan_concs).T)
                medians.append({
                    "label": "SMEAR III SMPS",
                    "dp": np.nanmedian(np.vstack(scan_sizes), axis=0),
                    "median": smear_stats["median"],
                    "p10": smear_stats["p10"],
                    "p90": smear_stats["p90"],
                    "flow_error": np.full(len(smear_stats["median"]), np.nan),
                    "flow_rel_rmse": np.nan,
                    "n_scans": int(len(scan_concs)),
                })

    return medians


def load_smeariii_cpc_for_times(times):
    times = pd.to_datetime(times)
    if len(times) == 0:
        return pd.DataFrame(columns=["time", "SMEARIII_CPC"])

    t0 = times.min()
    t1 = times.max()
    totalconc = load_smeariii_cpc_concentration(
        t0 - pd.Timedelta(hours=1),
        t1 - pd.Timedelta(hours=1),
    )
    totalconc["time"] = totalconc["time"] + pd.Timedelta(hours=1)
    return totalconc[totalconc["time"].between(t0, t1)].copy()


def match_to_smeariii_cpc(times, values, smear_cpc):
    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "value": pd.to_numeric(values, errors="coerce"),
    }).dropna(subset=["time", "value"])
    if df.empty or smear_cpc.empty:
        return pd.DataFrame(columns=["time", "value", "SMEARIII_CPC"])

    matched = pd.merge_asof(
        df.sort_values("time"),
        smear_cpc[["time", "SMEARIII_CPC"]].dropna().sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=15),
    )
    matched = matched.dropna(subset=["value", "SMEARIII_CPC"])
    matched["ratio"] = np.divide(
        matched["value"],
        matched["SMEARIII_CPC"],
        out=np.full(len(matched), np.nan),
        where=matched["SMEARIII_CPC"].to_numpy(dtype=float) > 0,
    )
    matched["delta"] = matched["value"] - matched["SMEARIII_CPC"]
    return matched


def build_scan_smeariii_comparison_heatmaps(result):
    heatmaps = [tr for tr in result if tr["kind"] == "heatmap"]
    if not heatmaps:
        return {}

    times = pd.to_datetime([t for tr in heatmaps for t in tr["x"]])
    if len(times) == 0:
        return {}

    smear = load_smeariii_sum_range(
        times.min() - pd.Timedelta(minutes=15),
        times.max() + pd.Timedelta(minutes=15),
    )
    if smear.empty:
        return {}

    smear_times = pd.to_datetime(sorted(smear["time"].unique()))
    comparisons = {}
    for tr in heatmaps:
        our_rows = []
        sizes = np.asarray(tr["y"], dtype=float)
        z = np.asarray(tr["Z"], dtype=float)
        for t, col in zip(pd.to_datetime(tr["x"]), z.T):
            for size_nm, conc in zip(sizes, col):
                if np.isfinite(size_nm) and np.isfinite(conc):
                    our_rows.append((t, size_nm, conc))

        if not our_rows:
            continue

        our = pd.DataFrame(our_rows, columns=["time", "size_nm", "our_conc"])
        our = our.groupby(["time", "size_nm"], as_index=False)["our_conc"].mean()
        our_times = sorted(our["time"].unique())
        sizes = np.asarray(sorted(our["size_nm"].unique()), dtype=float)
        ratio_cols = []

        for t in our_times:
            t = pd.Timestamp(t)
            deltas = np.abs(smear_times - t)
            if len(deltas) == 0 or deltas.min() > pd.Timedelta(minutes=15):
                ratio_cols.append(np.full(len(sizes), np.nan))
                continue

            smear_scan = smear[smear["time"] == smear_times[np.argmin(deltas)]]
            smear_sizes = smear_scan["size_nm"].to_numpy(dtype=float)
            smear_conc = smear_scan["smear_conc"].to_numpy(dtype=float)
            order = np.argsort(smear_sizes)
            smear_sizes = smear_sizes[order]
            smear_conc = smear_conc[order]
            valid = np.isfinite(smear_sizes) & np.isfinite(smear_conc) & (smear_conc > 0)
            if np.count_nonzero(valid) < 2:
                ratio_cols.append(np.full(len(sizes), np.nan))
                continue

            our_scan = our[our["time"] == t].set_index("size_nm").reindex(sizes)
            our_conc = our_scan["our_conc"].to_numpy(dtype=float)
            interp = np.interp(
                sizes,
                smear_sizes[valid],
                smear_conc[valid],
                left=np.nan,
                right=np.nan,
            )
            ratio_cols.append(our_conc / interp)

        comparisons[(tr.get("method", "gunn woessner mod"), tr["polarity"])] = {
            "x": [pd.Timestamp(t) for t in our_times],
            "y": sizes,
            "z": np.clip(np.column_stack(ratio_cols), 0, 2),
        }

    return comparisons


def list_scan_files(min_age_sec=0):
    root = app_path(scan_root.value)
    files = root.glob("*/*.csv")

    if min_age_sec > 0:
        now = pd.Timestamp.now().timestamp()
        files = [p for p in files if now - p.stat().st_mtime >= min_age_sec]

    return sorted(files, key=lambda p: (p.parent.name, p.stem))


def scan_date_bounds():
    start = as_date(scan_start_date.value)
    end = as_date(scan_end_date.value)
    start_ts = None
    end_ts = None
    if start:
        start_time = parse_time_text(scan_start_time.value, DEFAULT_SETTINGS["scan_start_time"])
        start_ts = pd.Timestamp.combine(start, start_time)
    if end:
        end_time = parse_time_text(scan_end_time.value, DEFAULT_SETTINGS["scan_end_time"])
        end_ts = pd.Timestamp.combine(end, end_time)
    return start_ts, end_ts


def scan_file_time_range(path):
    path = Path(path)
    try:
        stamp = pd.to_datetime(path.stem, format="%Y%m%d_%H%M%S", errors="raise")
        return stamp, stamp
    except Exception:
        return pd.NaT, pd.NaT


def filter_scan_files_by_date(files):
    start_ts, end_ts = scan_date_bounds()
    if start_ts is None and end_ts is None:
        return files

    selected = []
    for path in files:
        t0, t1 = scan_file_time_range(path)
        if pd.isna(t0) or pd.isna(t1):
            continue
        if start_ts is not None and t1 < start_ts:
            continue
        if end_ts is not None and t0 > end_ts:
            continue
        selected.append(path)
    return selected


def files_for_selection(min_age_sec=0, files=None):
    files = list_scan_files(min_age_sec=min_age_sec) if files is None else list(files)
    mode = scan_selection_mode.value
    if mode in {"Date range", "Date range + newest N"}:
        files = filter_scan_files_by_date(files)
    if mode in {"Newest N", "Date range + newest N"}:
        n = max(1, int(n_scans_plot.value))
        files = files[-n:]
    return files


def load_auto_state():
    if not AUTO_STATE_FILE.exists():
        return {"last_saved_signature": None}

    try:
        return json.loads(AUTO_STATE_FILE.read_text())
    except json.JSONDecodeError:
        return {"last_saved_signature": None}


def save_auto_state(state):
    AUTO_STATE_FILE.write_text(json.dumps(state, indent=2))


def selected_files_signature():
    parts = []

    for f in scan_files.value:
        p = Path(f)
        try:
            stat = p.stat()
            parts.append(f"{p}:{stat.st_mtime_ns}:{stat.st_size}")
        except FileNotFoundError:
            parts.append(str(p))

    return "|".join(parts)


def apply_loaded_time_window(df):
    if df.empty or "time" not in df.columns:
        return df

    start_min, end_min = loaded_time_window_min.value
    start_min = max(0, int(start_min))
    end_min = max(start_min, int(end_min))
    if start_min == 0 and end_min >= int(loaded_time_window_min.end):
        return df

    times = pd.to_datetime(df["time"], errors="coerce")
    t0 = times.min()
    if pd.isna(t0):
        return df

    start = t0 + pd.Timedelta(minutes=start_min)
    end = t0 + pd.Timedelta(minutes=end_min)
    return df[(times >= start) & (times <= end)].copy()


def load_selected_scans():
    dfs = []

    for f in scan_files.value:
        p = Path(f)
        try:
            d = pd.read_csv(p)
            d["scan_id"] = p.stem
            dfs.append(d)
        except Exception as e:
            status.object = f"Could not read {p}: {e}"
            print(f"Could not read {p}: {e}", flush=True)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = apply_loaded_time_window(df)
    df["cpc_float"] = pd.to_numeric(df["cpc_count"], errors="coerce")
    df["abs_size_nm"] = pd.to_numeric(df["size_nm"], errors="coerce").abs()
    df["polarity"] = np.where(df["size_nm"] > 0, "positive", "negative")
    return df


# ---------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------

settings = load_settings()

scan_root = pn.widgets.TextInput(
    name="Scan folder",
    value=settings.get("scan_root", DEFAULT_SETTINGS["scan_root"]),
    width=700,
)

save_root = pn.widgets.TextInput(
    name="Save folder",
    value=settings.get("save_root", DEFAULT_SETTINGS["save_root"]),
    width=700,
)

n_scans_plot = pn.widgets.IntInput(
    name="Auto-select last N",
    value=int(settings.get("n_scans_plot", DEFAULT_SETTINGS["n_scans_plot"])),
    step=1,
    width=160,
)
scan_selection_mode = pn.widgets.Select(
    name="Scan selection",
    options=["Newest N", "Date range", "Date range + newest N"],
    value=settings.get("scan_selection_mode", DEFAULT_SETTINGS["scan_selection_mode"]),
    width=190,
)
scan_start_date = pn.widgets.DatePicker(
    name="Start date",
    value=parse_saved_date(settings.get("scan_start_date", DEFAULT_SETTINGS["scan_start_date"])),
    width=150,
)
scan_start_time = pn.widgets.TextInput(
    name="Start time",
    value=str(settings.get("scan_start_time", DEFAULT_SETTINGS["scan_start_time"])),
    width=100,
    placeholder="HH:MM",
)
scan_end_date = pn.widgets.DatePicker(
    name="End date",
    value=parse_saved_date(settings.get("scan_end_date", DEFAULT_SETTINGS["scan_end_date"])),
    width=150,
)
scan_end_time = pn.widgets.TextInput(
    name="End time",
    value=str(settings.get("scan_end_time", DEFAULT_SETTINGS["scan_end_time"])),
    width=100,
    placeholder="HH:MM",
)

auto_interval_min = pn.widgets.IntInput(
    name="Auto interval min",
    value=int(settings.get("auto_interval_min", DEFAULT_SETTINGS["auto_interval_min"])),
    step=1,
    width=160,
)

auto_file_age_sec = pn.widgets.IntInput(
    name="Min file age sec",
    value=int(settings.get("auto_file_age_sec", DEFAULT_SETTINGS["auto_file_age_sec"])),
    step=10,
    width=160,
)

scan_files = pn.widgets.MultiChoice(
    name="Select scan CSVs",
    options=[],
    value=[],
    width=900,
)
saved_loaded_window = settings.get(
    "loaded_time_window_min",
    DEFAULT_SETTINGS["loaded_time_window_min"],
)
loaded_time_window_min = pn.widgets.IntRangeSlider(
    name="Loaded time window (min from first row)",
    start=0,
    end=24 * 60,
    value=(int(saved_loaded_window[0]), int(saved_loaded_window[1])),
    step=1,
    width=900,
)

save_button = pn.widgets.Button(name="Save plots/data", button_type="primary")
auto_checkbox = pn.widgets.Checkbox(name="Auto-run", value=False)
daily_overwrite_checkbox = pn.widgets.Checkbox(
    name="Daily overwrite files",
    value=bool(settings.get("daily_overwrite", DEFAULT_SETTINGS["daily_overwrite"])),
)

refresh_button = pn.widgets.Button(name="Refresh scan list", button_type="primary")
select_last_button = pn.widgets.Button(name="Select last N", button_type="primary")
plot_button = pn.widgets.Button(name="Plot raw selected scans", button_type="success")
invert_button = pn.widgets.Button(name="Run inversion", button_type="danger")
smps_timing_button = pn.widgets.Button(name="Update SMPS timing plot", button_type="primary")

dma_L = pn.widgets.FloatInput(name="DMA L (m)", value=float(settings.get("dma_L", 0.28)), step=0.01)
dma_r1 = pn.widgets.FloatInput(name="DMA r1 (m)", value=float(settings.get("dma_r1", 0.025)), step=0.001)
dma_r2 = pn.widgets.FloatInput(name="DMA r2 (m)", value=float(settings.get("dma_r2", 0.033)), step=0.001)

qa_lpm = pn.widgets.FloatInput(name="Aerosol flow qa (L/min)", value=float(settings.get("qa_lpm", 1.0)), step=0.1)
qs_lpm = pn.widgets.FloatInput(name="Sample flow qs (L/min)", value=float(settings.get("qs_lpm", 1.0)), step=0.1)

temp_K = pn.widgets.FloatInput(name="T (K)", value=float(settings.get("temp_K", 293.15)), step=1)
press_Pa = pn.widgets.FloatInput(name="P (Pa)", value=float(settings.get("press_Pa", 101325)), step=100)

zratio_widget = pn.widgets.FloatInput(
    name="Zn/Zp",
    value=float(settings.get("zratio", DEFAULT_SETTINGS["zratio"])),
    step=0.01,
)
zratio_min_widget = pn.widgets.FloatInput(
    name="Zn/Zp min",
    value=float(settings.get("zratio_min", DEFAULT_SETTINGS["zratio_min"])),
    step=0.01,
)
zratio_max_widget = pn.widgets.FloatInput(
    name="Zn/Zp max",
    value=float(settings.get("zratio_max", DEFAULT_SETTINGS["zratio_max"])),
    step=0.01,
)
zratio_smoothing_step = pn.widgets.FloatInput(
    name="Zn/Zp max step",
    value=float(settings.get(
        "zratio_smoothing_step",
        DEFAULT_SETTINGS["zratio_smoothing_step"],
    )),
    step=0.05,
)
zratio_min_size_nm = pn.widgets.FloatInput(
    name="Z-ratio min dp (nm)",
    value=float(settings.get(
        "zratio_min_size_nm",
        DEFAULT_SETTINGS["zratio_min_size_nm"],
    )),
    step=0.5,
)
zratio_estimate_offset = pn.widgets.FloatInput(
    name="Zn/Zp offset",
    value=float(settings.get(
        "zratio_estimate_offset",
        DEFAULT_SETTINGS["zratio_estimate_offset"],
    )),
    step=0.05,
)
use_zratio_checkbox = pn.widgets.Checkbox(name="Use Zn/Zp from settings", value=False)

smallest_size = pn.widgets.FloatInput(
    name="Smallest size (nm)",
    value=float(settings.get("smallest_size", DEFAULT_SETTINGS["smallest_size"])),
    step=0.1,
)
scan_inversion_type = pn.widgets.Select(
    name="Scan inversion type",
    options=["DMPS", "SMPS"],
    value=(
        settings.get("scan_inversion_type", DEFAULT_SETTINGS["scan_inversion_type"])
        if settings.get("scan_inversion_type", DEFAULT_SETTINGS["scan_inversion_type"]) in ["DMPS", "SMPS"]
        else DEFAULT_SETTINGS["scan_inversion_type"]
    ),
    width=140,
)
smps_settling_time_sec = pn.widgets.FloatInput(
    name="SMPS system settling (s)",
    value=float(settings.get(
        "smps_settling_time_sec",
        DEFAULT_SETTINGS["smps_settling_time_sec"],
    )),
    step=1.0,
    width=180,
)
smps_correction_mode = pn.widgets.Select(
    name="SMPS CPC correction",
    options=["Transport delay", "Response kernel (experimental)", "Integer step shift", "None"],
    value=(
        settings.get("smps_correction_mode", DEFAULT_SETTINGS["smps_correction_mode"])
        if settings.get("smps_correction_mode", DEFAULT_SETTINGS["smps_correction_mode"])
        in ["Transport delay", "Response kernel (experimental)", "Integer step shift", "None"]
        else DEFAULT_SETTINGS["smps_correction_mode"]
    ),
    width=180,
)
smps_transport_delay_sec = pn.widgets.FloatInput(
    name="CPC transport delay (s)",
    value=float(settings.get(
        "smps_transport_delay_sec",
        DEFAULT_SETTINGS["smps_transport_delay_sec"],
    )),
    step=1.0,
    width=180,
)
smps_response_window_sec = pn.widgets.FloatInput(
    name="CPC response window fallback (s)",
    value=float(settings.get(
        "smps_response_window_sec",
        DEFAULT_SETTINGS["smps_response_window_sec"],
    )),
    step=0.1,
    width=220,
)
smps_dwell_sec = pn.widgets.FloatInput(
    name="DMA dwell fallback (s)",
    value=float(settings.get("smps_dwell_sec", DEFAULT_SETTINGS["smps_dwell_sec"])),
    step=1.0,
    width=180,
)
smps_kernel_timing_override = pn.widgets.Checkbox(
    name="Override scan-row CPC timing",
    value=bool(settings.get(
        "smps_kernel_timing_override",
        DEFAULT_SETTINGS["smps_kernel_timing_override"],
    )),
)
smps_kernel_smoothness = pn.widgets.FloatInput(
    name="Kernel smoothness",
    value=float(settings.get(
        "smps_kernel_smoothness",
        DEFAULT_SETTINGS["smps_kernel_smoothness"],
    )),
    start=0.0,
    step=0.05,
    width=170,
)
smps_size_step_shift = pn.widgets.IntInput(
    name="SMPS size step shift",
    value=int(settings.get(
        "smps_size_step_shift",
        DEFAULT_SETTINGS["smps_size_step_shift"],
    )),
    step=1,
    width=170,
)
smps_timing_offset_min_sec = pn.widgets.FloatInput(
    name="Timing fit min offset (s)",
    value=float(settings.get(
        "smps_timing_offset_min_sec",
        DEFAULT_SETTINGS["smps_timing_offset_min_sec"],
    )),
    step=10.0,
    width=180,
)
smps_timing_offset_max_sec = pn.widgets.FloatInput(
    name="Timing fit max offset (s)",
    value=float(settings.get(
        "smps_timing_offset_max_sec",
        DEFAULT_SETTINGS["smps_timing_offset_max_sec"],
    )),
    step=10.0,
    width=180,
)
smps_timing_offset_step_sec = pn.widgets.FloatInput(
    name="Timing fit step (s)",
    value=float(settings.get(
        "smps_timing_offset_step_sec",
        DEFAULT_SETTINGS["smps_timing_offset_step_sec"],
    )),
    step=1.0,
    width=150,
)
smps_timing_match_tolerance_min = pn.widgets.FloatInput(
    name="SMEAR match tolerance (min)",
    value=float(settings.get(
        "smps_timing_match_tolerance_min",
        DEFAULT_SETTINGS["smps_timing_match_tolerance_min"],
    )),
    step=1.0,
    width=190,
)
inversion_size_bin_decimals = pn.widgets.IntInput(
    name="Merge size decimals",
    value=int(settings.get(
        "inversion_size_bin_decimals",
        DEFAULT_SETTINGS["inversion_size_bin_decimals"],
    )),
    step=1,
    width=160,
)
cpc_gap_interpolation_enabled = pn.widgets.Checkbox(
    name="Interpolate CPC gaps before inversion",
    value=bool(settings.get(
        "cpc_gap_interpolation_enabled",
        DEFAULT_SETTINGS["cpc_gap_interpolation_enabled"],
    )),
)
low_value_lift_enabled = pn.widgets.Checkbox(
    name="Lift low/zero artifacts",
    value=bool(settings.get(
        "low_value_lift_enabled",
        DEFAULT_SETTINGS["low_value_lift_enabled"],
    )),
)
low_value_lift_ratio = pn.widgets.FloatInput(
    name="Lift floor ratio",
    value=float(settings.get(
        "low_value_lift_ratio",
        DEFAULT_SETTINGS["low_value_lift_ratio"],
    )),
    step=0.05,
    width=150,
)
low_value_lift_alpha = pn.widgets.FloatInput(
    name="Lift correction fraction",
    value=float(settings.get(
        "low_value_lift_alpha",
        DEFAULT_SETTINGS["low_value_lift_alpha"],
    )),
    step=0.05,
    width=170,
)
ntot_plot_max = pn.widgets.FloatInput(
    name="Ntot plot max",
    value=float(settings.get("ntot_plot_max", DEFAULT_SETTINGS["ntot_plot_max"])),
    step=1000,
)
heatmap_clip = pn.widgets.FloatInput(
    name="Heatmap clip",
    value=float(settings.get("heatmap_clip", 20000)),
    step=1000,
)
raw_uncertainty = pn.widgets.Select(
    name="Raw uncertainty",
    options=["percentile 10-90", "std", "sem", "min-max", "none"],
    value=settings.get("raw_uncertainty", DEFAULT_SETTINGS["raw_uncertainty"]),
    width=180,
)
GROWTH_MODEL_OPTIONS = [
    "Lower edge D25",
    "Center D50",
    "Upper edge D75",
    "Ridge peak",
    "Appearance time",
]
GROWTH_MODEL_COLORS = {
    "Lower edge D25": "#2a9d8f",
    "Center D50": "#f4a261",
    "Upper edge D75": "#e76f51",
    "Ridge peak": "#7b2cbf",
    "Appearance time": "#277da1",
}
saved_growth_models = diag.growth_models_from_settings(
    settings, GROWTH_MODEL_OPTIONS, DEFAULT_SETTINGS["growth_models"]
)
growth_models = pn.widgets.MultiChoice(
    name="NPF growth tracks",
    options=GROWTH_MODEL_OPTIONS,
    value=saved_growth_models,
    width=500,
)
growth_min_size_nm = pn.widgets.FloatInput(
    name="Growth min dp (nm)",
    value=float(settings.get("growth_min_size_nm", DEFAULT_SETTINGS["growth_min_size_nm"])),
    step=0.5,
)
growth_max_size_nm = pn.widgets.FloatInput(
    name="Growth max dp (nm)",
    value=float(settings.get("growth_max_size_nm", DEFAULT_SETTINGS["growth_max_size_nm"])),
    step=1.0,
)
growth_threshold_fraction = pn.widgets.FloatInput(
    name="Growth threshold frac",
    value=float(settings.get(
        "growth_threshold_fraction",
        DEFAULT_SETTINGS["growth_threshold_fraction"],
    )),
    step=0.05,
)
growth_max_gap_minutes = pn.widgets.FloatInput(
    name="Growth max gap (min)",
    value=float(settings.get(
        "growth_max_gap_minutes", DEFAULT_SETTINGS["growth_max_gap_minutes"]
    )),
    step=5.0,
    width=180,
)
growth_min_event_scans = pn.widgets.IntInput(
    name="Growth min scans/event",
    value=int(settings.get(
        "growth_min_event_scans", DEFAULT_SETTINGS["growth_min_event_scans"]
    )),
    start=3,
    step=1,
    width=190,
)
growth_max_rate_nm_h = pn.widgets.FloatInput(
    name="Growth max plausible rate (nm/h)",
    value=float(settings.get(
        "growth_max_rate_nm_h", DEFAULT_SETTINGS["growth_max_rate_nm_h"]
    )),
    step=1.0,
    width=220,
)
difference_peak_min_size_nm = pn.widgets.FloatInput(
    name="Diff peak min dp (nm)",
    value=float(settings.get(
        "difference_peak_min_size_nm",
        DEFAULT_SETTINGS["difference_peak_min_size_nm"],
    )),
    step=1.0,
)

tube_segments = pn.widgets.TextAreaInput(
    name="Tube segments: tubediameter,tubelength,aflow[,angle]",
    value=str(settings.get("tube_segments", DEFAULT_SETTINGS["tube_segments"])),
    height=120,
    width=700,
    placeholder="tubediameter,tubelength,aflow,angle\n0,1.93,qa,0\n0,2.80,8,0",
)

saved_methods = settings.get("inversion_methods", DEFAULT_SETTINGS["inversion_methods"])
saved_methods = normalize_inversion_methods(saved_methods)
if not saved_methods:
    saved_methods = DEFAULT_SETTINGS["inversion_methods"]

inversion_methods = pn.widgets.MultiChoice(
    name="Inversion methods",
    options=list(INVERSION_METHODS.values()),
    value=[method_label(method) for method in saved_methods],
    width=700,
)

status = pn.pane.Markdown("Status: idle")

raw_plot = pn.pane.Plotly(height=750, width=1300)
inversion_plot = pn.pane.Plotly(width=1300)
residual_plot = pn.pane.Plotly(width=1300, height=1200)
difference_plot = pn.pane.Plotly(width=1300)
smps_timing_plot = pn.pane.Plotly(width=1300, height=1100)


def publish_shared_state(
    *,
    raw_fig=None,
    inversion_fig=None,
    residual_fig=None,
    smps_timing_fig=None,
    difference_fig=None,
    difference_diagnostics=None,
    growth_diagnostics=None,
    growth_settings=None,
    inversion_result=None,
    status_text=None,
):
    with shared_state["lock"]:
        if raw_fig is not None:
            shared_state["raw_fig"] = raw_fig
        if inversion_fig is not None:
            shared_state["inversion_fig"] = inversion_fig
        if residual_fig is not None:
            shared_state["residual_fig"] = residual_fig
        if smps_timing_fig is not None:
            shared_state["smps_timing_fig"] = smps_timing_fig
        if difference_fig is not None:
            shared_state["difference_fig"] = difference_fig
        if difference_diagnostics is not None:
            shared_state["difference_diagnostics"] = difference_diagnostics
        if growth_diagnostics is not None:
            shared_state["growth_diagnostics"] = copy.deepcopy(growth_diagnostics)
        if growth_settings is not None:
            shared_state["growth_settings"] = copy.deepcopy(growth_settings)
        if inversion_result is not None:
            shared_state["latest_inversion"] = inversion_result
        if status_text is not None:
            shared_state["status"] = status_text
        shared_state["version"] += 1


def sync_shared_state():
    global latest_inversion, latest_difference_diagnostics
    global latest_growth_diagnostics, latest_growth_settings, local_shared_version

    with shared_state["lock"]:
        version = shared_state["version"]
        if version == local_shared_version:
            return
        raw_fig = shared_state["raw_fig"]
        inversion_fig = shared_state["inversion_fig"]
        residual_fig = shared_state["residual_fig"]
        smps_timing_fig = shared_state["smps_timing_fig"]
        difference_fig = shared_state["difference_fig"]
        difference_diagnostics = shared_state["difference_diagnostics"]
        growth_diagnostics = copy.deepcopy(shared_state["growth_diagnostics"])
        growth_settings = copy.deepcopy(shared_state["growth_settings"])
        inversion_result = shared_state["latest_inversion"]
        status_text = shared_state["status"]

    if raw_fig is not None:
        raw_plot.object = copy.deepcopy(raw_fig)
    if inversion_fig is not None:
        inversion_plot.object = copy.deepcopy(inversion_fig)
    if residual_fig is not None:
        residual_plot.object = copy.deepcopy(residual_fig)
    if smps_timing_fig is not None:
        smps_timing_plot.object = copy.deepcopy(smps_timing_fig)
    if difference_fig is not None:
        difference_plot.object = copy.deepcopy(difference_fig)
    if difference_diagnostics is not None:
        latest_difference_diagnostics = difference_diagnostics
    latest_growth_diagnostics = growth_diagnostics
    latest_growth_settings = growth_settings
    if inversion_result is not None:
        latest_inversion = inversion_result
    if status_text is not None:
        status.object = status_text

    local_shared_version = version


# ---------------------------------------------------------------------
# Scan file browser
# ---------------------------------------------------------------------

def refresh_scan_files(event=None):
    root = app_path(scan_root.value)
    all_files = list_scan_files()
    files = files_for_selection(files=all_files)

    print("cwd:", Path.cwd(), flush=True)
    print("scan root:", root.resolve(), flush=True)
    print("found files:", len(all_files), flush=True)

    scan_files.options = [str(p) for p in all_files]

    if files:
        scan_files.value = [str(p) for p in files]
    else:
        scan_files.value = []

    status.object = f"Found **{len(all_files)}** scan CSV files; selected **{len(files)}** by `{scan_selection_mode.value}`."


def select_last_n(event=None):
    all_files = list_scan_files()
    files = files_for_selection(files=all_files)
    scan_files.options = [str(p) for p in all_files]
    scan_files.value = [str(p) for p in files]
    status.object = f"Selected **{len(scan_files.value)}** scan files by `{scan_selection_mode.value}`."


def apply_scan_selection(event=None):
    select_last_n()


refresh_button.on_click(refresh_scan_files)
select_last_button.on_click(select_last_n)
for w in [scan_selection_mode, scan_start_date, scan_start_time, scan_end_date, scan_end_time, n_scans_plot]:
    w.param.watch(apply_scan_selection, "value")


# ---------------------------------------------------------------------
# Raw scan plot
# ---------------------------------------------------------------------

def plot_selected_scans(event=None):
    df = load_selected_scans()
    df = df[df["Ntot"] == False]  
    if df.empty:
        status.object = "No selected scan data."
        return

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[
            "CPC concentration vs selected size",
            "CPC concentration vs time",
            "Sheath flow / setpoint",
            "Positive / negative CPC ratio",
        ],
    )

    df_for_size_summary = df.copy()
    df_for_size_summary["size_bin_nm"] = df_for_size_summary["abs_size_nm"].round(1)
    size_summary = (
        df_for_size_summary.groupby(["size_bin_nm", "polarity"])
        .agg(
            abs_size_nm=("abs_size_nm", "mean"),
            mean=("cpc_float", "mean"),
            std=("cpc_float", "std"),
            sem=("cpc_float", "sem"),
            p10=("cpc_float", lambda x: x.quantile(0.10)),
            p90=("cpc_float", lambda x: x.quantile(0.90)),
            ymin=("cpc_float", "min"),
            ymax=("cpc_float", "max"),
            count=("cpc_float", "size"),
        )
        .reset_index()
        .sort_values("abs_size_nm")
    )
    mode = raw_uncertainty.value
    if mode == "std":
        size_summary["err_low"] = size_summary["std"].fillna(0)
        size_summary["err_high"] = size_summary["std"].fillna(0)
    elif mode == "sem":
        size_summary["err_low"] = size_summary["sem"].fillna(0)
        size_summary["err_high"] = size_summary["sem"].fillna(0)
    elif mode == "min-max":
        size_summary["err_low"] = (size_summary["mean"] - size_summary["ymin"]).clip(lower=0).fillna(0)
        size_summary["err_high"] = (size_summary["ymax"] - size_summary["mean"]).clip(lower=0).fillna(0)
    elif mode == "none":
        size_summary["err_low"] = 0.0
        size_summary["err_high"] = 0.0
    else:
        size_summary["err_low"] = (size_summary["mean"] - size_summary["p10"]).clip(lower=0).fillna(0)
        size_summary["err_high"] = (size_summary["p90"] - size_summary["mean"]).clip(lower=0).fillna(0)

    for polarity, g in size_summary.groupby("polarity"):
        error_y = None
        if mode != "none":
            error_y = dict(
                type="data",
                array=g["err_high"],
                arrayminus=g["err_low"],
                visible=True,
                thickness=2.5,
                width=8,
            )

        fig.add_scatter(
            x=g["abs_size_nm"],
            y=g["mean"],
            error_y=error_y,
            mode="lines+markers",
            marker=dict(size=np.where(g["count"] > 1, 8, 11), symbol=np.where(g["count"] > 1, "circle", "circle-open")),
            name=f"{polarity} mean ({mode})",
            customdata=np.column_stack((g["err_low"], g["err_high"], g["count"])),
            hovertemplate=(
                "dp=%{x:.2f} nm<br>"
                "mean=%{y:.2f}<br>"
                "err -%{customdata[0]:.2f} / +%{customdata[1]:.2f}<br>"
                "n=%{customdata[2]:.0f}<extra></extra>"
            ),
            row=1,
            col=1,
        )

    for (scan_id, polarity), g in df.groupby(["scan_id", "polarity"]):
        g = g.sort_values("abs_size_nm")

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
        g = g.sort_values("time")

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

        grouped = (
            g.groupby(["abs_size_nm", "polarity"])["cpc_float"]
            .mean()
            .reset_index()
        )

        pos = grouped[grouped["polarity"] == "positive"].rename(columns={"cpc_float": "cpc_pos"})
        neg = grouped[grouped["polarity"] == "negative"].rename(columns={"cpc_float": "cpc_neg"})

        m = pd.merge(
            pos[["abs_size_nm", "cpc_pos"]],
            neg[["abs_size_nm", "cpc_neg"]],
            on="abs_size_nm",
            how="inner",
        ).sort_values("abs_size_nm")

        if not m.empty:
            ratio = np.divide(
                m["cpc_pos"].to_numpy(dtype=float),
                m["cpc_neg"].to_numpy(dtype=float),
                out=np.full(len(m), np.nan),
                where=m["cpc_neg"].to_numpy(dtype=float) > 0,
            )

            fig.add_scatter(
                x=m["abs_size_nm"],
                y=np.clip(ratio, 0, 4),
                mode="lines+markers",
                name=f"{scan_id} CPC + / -",
                row=4,
                col=1,
            )

    fig.update_xaxes(type="log", title_text="|dp| (nm)", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_xaxes(type="log", title_text="|dp| (nm)", row=4, col=1)

    fig.update_yaxes(title_text="CPC", row=1, col=1)
    fig.update_yaxes(title_text="CPC", row=2, col=1)
    fig.update_yaxes(title_text="Flow L/min", row=3, col=1)
    fig.update_yaxes(title_text="+ / -", row=4, col=1)

    fig.update_layout(
        height=750,
        width=1300,
        title="Selected DMPS scans",
        showlegend=True,
        margin=dict(l=50, r=260, t=60, b=30),
        legend=dict(x=1.02, y=1.0),
    )

    status_text = f"Plotted **{df['scan_id'].nunique()}** scan(s)."
    raw_plot.object = fig
    status.object = status_text
    publish_shared_state(raw_fig=fig, status_text=status_text)


plot_button.on_click(plot_selected_scans)


# ---------------------------------------------------------------------
# Ion ratio + inversion
# ---------------------------------------------------------------------

def estimate_ion_mobility_ratio_for_scan(g_scan, temp=293.15, press=101325):
    d = g_scan.copy()
    d["cpc_float"] = pd.to_numeric(d["cpc_count"], errors="coerce")
    d["abs_size_nm"] = merged_abs_size_nm(d["size_nm"])
    d["polarity"] = np.where(d["size_nm"].astype(float) > 0, "pos", "neg")
    grouped = (
        d.groupby(["abs_size_nm", "polarity"])["cpc_float"]
        .mean()
        .reset_index()
    )

    pos = grouped[grouped["polarity"] == "pos"].rename(columns={"cpc_float": "R_pos"})
    neg = grouped[grouped["polarity"] == "neg"].rename(columns={"cpc_float": "R_neg"})

    m = pd.merge(
        pos[["abs_size_nm", "R_pos"]],
        neg[["abs_size_nm", "R_neg"]],
        on="abs_size_nm",
        how="inner",
    ).sort_values("abs_size_nm")

    min_size_nm = float(zratio_min_size_nm.value)
    if np.isfinite(min_size_nm) and min_size_nm > 0:
        m = m[m["abs_size_nm"] >= min_size_nm]

    if len(m) < 3:
        return np.nan, np.nan

    dp = m["abs_size_nm"].to_numpy(dtype=float)
    Rp = m["R_pos"].to_numpy(dtype=float)
    Rn = m["R_neg"].to_numpy(dtype=float)

    # start from largest-size peak
    start = np.argmax(Rp + Rn)

    for i in range(start, len(dp)):
        if Rp[i] <= 0 or Rn[i] <= 0:
            continue

        dp_i_m = dp[i] * 1e-9

        # singly charged mobility at dp_i
        mob_i = (
            1.602176634e-19
            * cunningham_correction(dp_i_m, T=temp, P=press)
            / (3 * np.pi * 1.81e-5 * dp_i_m)
        )

        # doubly charged contaminant: same mobility => particle mobility is half
        dp_g_m = inv.min_mob(np.array([0.5 * mob_i]), temp, press)[0]
        dp_g_nm = dp_g_m * 1e9

        if dp_g_nm > np.nanmax(dp):
            return np.sqrt(Rp[i] / Rn[i]), dp[i]

        Rg_pos = np.interp(dp_g_nm, dp, Rp)
        Rg_neg = np.interp(dp_g_nm, dp, Rn)

        zratio_default = float(zratio_widget.value)
        Zn = 1e-4
        Zp = zratio_default * Zn

        fw_pos_1 = inv.gunn_woessner_modified(
            1,
            np.array([dp_g_m]),
            temp,
            Zp,
            Zn,
            140,
            101,
            1e13,
            1e13,
            0,
        )
        
        fw_pos_2 = inv.gunn_woessner_modified(
            2,
            np.array([dp_g_m]),
            temp,
            Zp,
            Zn,
            140,
            101,
            1e13,
            1e13,
            0,
        )

        fw_neg_1 = inv.gunn_woessner_modified(
            -1,
            np.array([dp_g_m]),
            temp,
            Zp,
            Zn,
            140,
            101,
            1e13,
            1e13,
            0,
        )
        
        fw_neg_2 = inv.gunn_woessner_modified(
            -2,
            np.array([dp_g_m]),
            temp,
            Zp,
            Zn,
            140,
            101,
            1e13,
            1e13,
            0,
        )

        double_pos = Rg_pos * fw_pos_2 / fw_pos_1
        double_neg = Rg_neg * fw_neg_2 / fw_neg_1

        ok_pos = double_pos < 0.10 * Rp[i]
        ok_neg = double_neg < 0.10 * Rn[i]

        if ok_pos and ok_neg:
            return np.sqrt(Rp[i] / Rn[i]), dp[i]

    return np.nan, np.nan


def smooth_ion_ratio_points(ion_points):
    max_step = float(zratio_smoothing_step.value)
    offset = float(zratio_estimate_offset.value)
    fallback = float(zratio_widget.value)
    zmin = float(zratio_min_widget.value)
    zmax = float(zratio_max_widget.value)
    if zmin > zmax:
        zmin, zmax = zmax, zmin

    def ratio_for_use(raw):
        if not np.isfinite(raw):
            return fallback
        if np.isfinite(zmin) and raw < zmin:
            return fallback
        if np.isfinite(zmax) and raw > zmax:
            return fallback
        return raw

    if max_step <= 0:
        return [
            (t, raw + offset if np.isfinite(raw) else raw, ratio_for_use(raw + offset if np.isfinite(raw) else raw), dp, scan_id)
            for t, raw, dp, scan_id in ion_points
        ]

    smoothed_points = []
    previous = np.nan
    for t, raw, dp, scan_id in sorted(ion_points, key=lambda x: x[0]):
        if np.isfinite(raw):
            raw = raw + offset
        used = ratio_for_use(raw)
        if np.isfinite(previous):
            delta = used - previous
            smoothed = previous + np.clip(delta, -max_step, max_step)
        else:
            smoothed = used

        smoothed_points.append((t, raw, smoothed, dp, scan_id))
        previous = smoothed

    return smoothed_points


def invert_one_scan(
    d,
    polarity,
    zratio=None,
    temp=293.15,
    press=101325,
    inversion_method="gunn woessner mod",
    cpc_series_override=None,
    assignment_diagnostics=None,
    response_kernel=None,
):
    d = d.copy()
    d = d[d["Ntot"] == False]
    d["cpc_float"] = pd.to_numeric(d["cpc_count"], errors="coerce")
    size_col = inversion_size_column(d)
    d["abs_size_nm"] = merged_abs_size_nm(d[size_col])
    d = d.dropna(subset=["abs_size_nm"])
    d = d[d["abs_size_nm"] > smallest_size.value]
    d = d.sort_values("abs_size_nm")

    if response_kernel is not None:
        dp_meas_nm = response_kernel.sizes_nm.copy()
        y = response_kernel.sample_values.copy()
        kernel_smoothness = float(smps_kernel_smoothness.value)
        rejection_reason = response_kernel_rejection_reason(response_kernel)
        if rejection_reason:
            result = pd.DataFrame(columns=["abs_size_nm", "N_GWalpha"])
            result.attrs["assignment_diagnostics"] = {
                **(assignment_diagnostics or {}),
                **response_kernel.diagnostics,
                "kernel_usable": False,
                "kernel_rejection_reason": rejection_reason,
            }
            return result
    else:
        y_series = (
            build_cpc_series_for_inversion(d)
            if cpc_series_override is None
            else cpc_series_override.copy()
        )
        if assignment_diagnostics is None:
            assignment_diagnostics = y_series.attrs.get("assignment_diagnostics")
        if cpc_gap_interpolation_enabled.value:
            dp_meas_nm, y = interpolate_cpc_gaps_for_inversion(
                y_series.index.to_numpy(dtype=float),
                y_series.to_numpy(dtype=float),
            )
        else:
            if smps_correction_mode.value == "Transport delay":
                y_series = y_series[np.isfinite(y_series)]
            else:
                y_series = y_series[y_series > 0]
            dp_meas_nm = y_series.index.to_numpy(dtype=float)
            y = y_series.to_numpy(dtype=float)

    if len(dp_meas_nm) < 2 or len(y) == 0:
        return pd.DataFrame(columns=["abs_size_nm", "N_GWalpha"])

    dp_grid_nm = dp_meas_nm.copy()
    dp_grid_m = dp_grid_nm * 1e-9
    ldp = np.log10(dp_grid_m)

    limits = np.empty(len(ldp) + 1)
    limits[0] = ldp[0] - (ldp[1] - ldp[0]) / 2
    limits[1:-1] = 0.5 * (ldp[1:] + ldp[:-1])
    limits[-1] = ldp[-1] + (ldp[-1] - ldp[-2]) / 2

    mids = 0.5 * (limits[:-1] + limits[1:])
    halfs = 0.5 * (limits[1:] - limits[:-1])
    gl_pts = (mids[:, None] + halfs[:, None] * _GL_NODES[None, :]).ravel()

    dma = get_dma()
    A = np.zeros((len(dp_meas_nm), len(dp_grid_nm)))

    qa = float(qa_lpm.value) / 60000.0
    qs = float(qs_lpm.value) / 60000.0
    q_sheath_lpm = float(d["sheath_setpoint"].median())
    cpc_type_values = d.get("cpc_type", pd.Series("3010", index=d.index)).dropna().astype(str)
    scan_cpc_type = cpc_type_values.mode().iloc[0] if len(cpc_type_values) else "3010"
    qc = q_sheath_lpm / 60000.0
    qm = qc + qa - qs
    parsed_tube_segments = parse_tube_segments(tube_segments.value, qa=qa, qs=qs, qc=qc, qm=qm)

    if polarity == "positive":
        p = np.arange(-1, -6, -1, dtype=float)
    else:
        p = np.arange(1, 6, 1, dtype=float)

    if zratio is None or not np.isfinite(zratio) or use_zratio_checkbox.value:
        zratio = float(zratio_widget.value)

    zp = 1e-4
    zn = zratio * zp

    for i, dp_nm in enumerate(dp_meas_nm):
        voltage = voltage_from_size(
            dp_nm if polarity == "positive" else -dp_nm,
            q_sh_lpm=q_sheath_lpm,
            dma=dma,
            temp_K=temp,
            press_Pa=press,
        )

        args = (
            temp, press, p, voltage,
            dma.L, dma.r2, dma.r1,
            qa, qc, qm, qs,
            1.93, qa, 1,
            zp, zn,
            140, 101,
            1e13, 1e13,
            inversion_method,
            0,
            parsed_tube_segments,
            scan_cpc_type,
        )

        vals = inv.intfun(gl_pts, *args).reshape(len(dp_grid_nm), len(_GL_NODES))
        A[i, :] = halfs * (vals @ _GL_WEIGHTS)

    if response_kernel is not None:
        x, sample_fit, solve_diagnostics = solve_response_kernel_nnls(
            response_kernel.matrix,
            A,
            y,
            smoothness=kernel_smoothness,
            correlation=response_kernel.correlation,
        )
        if solve_diagnostics["rank"] < len(dp_meas_nm):
            result = pd.DataFrame(columns=["abs_size_nm", "N_GWalpha"])
            result.attrs["assignment_diagnostics"] = {
                **(assignment_diagnostics or {}),
                **response_kernel.diagnostics,
                "kernel_usable": False,
                "kernel_rejection_reason": (
                    f"observational kernel/transfer rank {solve_diagnostics['rank']} "
                    f"is below {len(dp_meas_nm)} bins"
                ),
            }
            return result
        coverage = response_kernel.matrix.sum(axis=0)
        sample_residual = y - sample_fit
        y_binned = np.divide(
            response_kernel.matrix.T @ y,
            coverage,
            out=np.full(len(dp_meas_nm), np.nan),
            where=coverage > 0,
        )
        y_fit = np.divide(
            response_kernel.matrix.T @ sample_fit,
            coverage,
            out=np.full(len(dp_meas_nm), np.nan),
            where=coverage > 0,
        )
        y = y_binned
        assignment_diagnostics = {
            **(assignment_diagnostics or {}),
            **response_kernel.diagnostics,
            "rank": solve_diagnostics["rank"],
            "condition_number": solve_diagnostics["condition_number"],
            "minimum_singular_value": solve_diagnostics["minimum_singular_value"],
            "augmented_rank": solve_diagnostics["augmented_rank"],
            "augmented_condition_number": solve_diagnostics["augmented_condition_number"],
            "residual_norm": solve_diagnostics["residual_norm"],
            "kernel_smoothness": solve_diagnostics["smoothness"],
            "solution_roughness": solve_diagnostics["solution_roughness"],
            "sample_residual_rmse": float(np.sqrt(np.mean(sample_residual ** 2))),
            "sample_residual_bias": float(np.mean(sample_residual)),
            "sample_residual_max_abs": float(np.max(np.abs(sample_residual))),
            "ill_conditioned": response_kernel_ill_conditioned(solve_diagnostics),
        }
    else:
        x, _ = nnls(A, y)
        y_fit = A @ x
    residual = y - y_fit
    residual_rel = np.divide(
        residual,
        y,
        out=np.full(len(y), np.nan),
        where=y != 0,
    )

    result = pd.DataFrame({
        "abs_size_nm": dp_grid_nm,
        "N_GWalpha": x,
        "measured_cpc": y,
        "fitted_cpc": y_fit,
        "residual_cpc": residual,
        "residual_rel": residual_rel,
    })
    if assignment_diagnostics is not None:
        result.attrs["assignment_diagnostics"] = assignment_diagnostics
    if response_kernel is not None:
        result.attrs["sample_residual_rows"] = [
            {
                "cpc_sample_id": sample_id,
                "sample_time": pd.to_datetime(sample_time, unit="s", utc=True).isoformat(),
                "support_start": pd.to_datetime(support_start, unit="s", utc=True).isoformat(),
                "support_end": pd.to_datetime(support_end, unit="s", utc=True).isoformat(),
                "measured_cpc": float(measured),
                "fitted_cpc": float(fitted),
                "residual_cpc": float(measured - fitted),
            }
            for sample_id, sample_time, support_start, support_end, measured, fitted
            in zip(
                response_kernel.sample_ids,
                response_kernel.sample_times,
                response_kernel.support_starts,
                response_kernel.support_ends,
                response_kernel.sample_values,
                sample_fit,
            )
        ]
    return result


def run_inversion_calculation(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["abs_size_nm"] = pd.to_numeric(df["size_nm"], errors="coerce").abs()
    df["polarity"] = np.where(df["size_nm"] > 0, "positive", "negative")
    df = apply_smps_size_shift(df)

    size_axis = get_scan_size_axis(df[df["Ntot"] == False])
    output = []
    ion_points = []
    assignment_diagnostics = []
    kernel_sample_residuals = []
    range_overlap_diagnostics = []
    ntot_closure_diagnostics = []

    group_key = "scan_id" if "scan_id" in df.columns else "scan_number"
    transport_assignments = {}
    response_kernel_result = None
    if scan_inversion_type.value == "SMPS" and smps_correction_mode.value == "Transport delay":
        for scan_id, full_scan in df.groupby(group_key):
            assignment_rows = full_scan[full_scan["Ntot"] == False].copy()
            assignment_rows["cpc_float"] = pd.to_numeric(
                assignment_rows["cpc_count"], errors="coerce"
            )
            assignment_rows["abs_size_nm"] = merged_abs_size_nm(
                assignment_rows[inversion_size_column(assignment_rows)]
            )
            assignment_rows = assignment_rows[
                assignment_rows["abs_size_nm"] > smallest_size.value
            ]
            transport_assignments[scan_id] = assign_cpc_samples_to_setpoints(
                assignment_rows,
                delay_seconds=float(smps_transport_delay_sec.value),
                settling_seconds=float(smps_settling_time_sec.value),
                group_columns=("polarity", "scan_range"),
                override_timing=bool(smps_kernel_timing_override.value),
            )
    if scan_inversion_type.value == "SMPS" and smps_correction_mode.value == "Response kernel (experimental)":
        kernel_rows = df[df["Ntot"] == False].copy()
        kernel_rows["cpc_float"] = pd.to_numeric(kernel_rows["cpc_count"], errors="coerce")
        kernel_rows["abs_size_nm"] = merged_abs_size_nm(
            kernel_rows[inversion_size_column(kernel_rows)]
        )
        kernel_rows = kernel_rows[kernel_rows["abs_size_nm"] > smallest_size.value]
        response_kernel_result = build_response_kernel(
            kernel_rows,
            delay_seconds=float(smps_transport_delay_sec.value),
            response_window_seconds=float(smps_response_window_sec.value),
            dwell_seconds=float(smps_dwell_sec.value),
            group_columns=(group_key, "polarity", "scan_range"),
            override_timing=bool(smps_kernel_timing_override.value),
        )

    zratios = {}
    for scan_id, g_scan in df.groupby(group_key):
        zratio, selected_dp = estimate_ion_mobility_ratio_for_scan(
            g_scan,
            temp=float(temp_K.value),
            press=float(press_Pa.value),
        )
        if np.isfinite(zratio):
            ion_points.append((g_scan["time"].median(), zratio, selected_dp, scan_id))

    ion_points = smooth_ion_ratio_points(ion_points)
    for _, _, smoothed_zratio, _, scan_id in ion_points:
        if np.isfinite(smoothed_zratio) and smoothed_zratio != 0:
            zratios[scan_id] = smoothed_zratio

    inversion_method_values = selected_inversion_methods()
    for inversion_method in inversion_method_values:
        for polarity in ["positive", "negative"]:
            dd = df[df["polarity"] == polarity].copy()

            heat_cols = []
            heat_times = []
            heat_flow_rel_rmse = []
            ntot_vals = []
            ntot_measured = []
            residual_rows = []

            for scan_id, g_scan in dd.groupby(group_key):
                zratio = zratios.get(scan_id, np.nan)
                scan_parts = []
                ntot_scan = 0.0
                temp = float(temp_K.value)
                press = float(press_Pa.value)
                qa = float(qa_lpm.value) / 60000.0
                qs = float(qs_lpm.value) / 60000.0
                q_sheath_lpm = float(g_scan["sheath_setpoint"].median())
                _, flow_rel_rmse = diag.sheath_flow_relative_rmse(g_scan[g_scan["Ntot"] == False])
                qc = q_sheath_lpm / 60000.0
                qm = qc + qa - qs
                parsed_tube_segments = parse_tube_segments(
                    tube_segments.value,
                    qa=qa,
                    qs=qs,
                    qc=qc,
                    qm=qm,
                )

                ntot_rows = g_scan[g_scan["Ntot"] == True].copy()
                ntot_rows["cpc_float"] = pd.to_numeric(ntot_rows["cpc_count"], errors="coerce")
                ntot_rows, ntot_duplicates, ntot_duplicate_scope = deduplicate_cpc_rows(ntot_rows)
                measured_ntot_raw = ntot_rows["cpc_float"].mean()
                cpc_type_values = g_scan.get(
                    "cpc_type", pd.Series("3010", index=g_scan.index)
                ).dropna().astype(str)
                scan_cpc_type = cpc_type_values.mode().iloc[0] if len(cpc_type_values) else "3010"

                for _, g_range in g_scan.groupby("scan_range"):
                    cpc_series_override = None
                    group_assignment_diagnostics = None
                    response_kernel = None
                    if scan_id in transport_assignments:
                        assignment = transport_assignments[scan_id]
                        range_value = g_range["scan_range"].iloc[0]
                        try:
                            cpc_series_override = assignment.cpc_by_size.xs(
                                (polarity, range_value),
                                level=("polarity", "scan_range"),
                            )
                            group_assignment_diagnostics = {
                                **assignment.diagnostics,
                                "scan_id": str(scan_id),
                                "scan_range": range_value,
                                "polarity": polarity,
                            }
                        except KeyError:
                            cpc_series_override = pd.Series(dtype=float)
                    if response_kernel_result is not None:
                        range_value = g_range["scan_range"].iloc[0]
                        response_kernel = response_kernel_result.groups.get(
                            (scan_id, polarity, range_value)
                        )
                        if response_kernel is None:
                            continue
                        group_assignment_diagnostics = {
                            **response_kernel_result.diagnostics,
                            "scan_id": str(scan_id),
                            "scan_range": range_value,
                            "polarity": polarity,
                        }
                    invdf = invert_one_scan(
                        g_range,
                        polarity=polarity,
                        zratio=zratio,
                        temp=temp,
                        press=press,
                        inversion_method=inversion_method,
                        cpc_series_override=cpc_series_override,
                        assignment_diagnostics=group_assignment_diagnostics,
                        response_kernel=response_kernel,
                    )
                    assignment = invdf.attrs.get("assignment_diagnostics")
                    for sample_row in invdf.attrs.get("sample_residual_rows", []):
                        kernel_sample_residuals.append({
                            **sample_row,
                            "scan_id": str(scan_id),
                            "scan_range": g_range["scan_range"].iloc[0],
                            "method": inversion_method,
                            "polarity": polarity,
                        })
                    if assignment is not None and inversion_method == inversion_method_values[0]:
                        assignment = {
                            **assignment,
                            "scan_id": str(scan_id),
                            "scan_range": g_range["scan_range"].iloc[0],
                            "polarity": polarity,
                        }
                        assignment_diagnostics.append(assignment)
                        print(f"SMPS CPC assignment: {assignment}", flush=True)

                    if invdf.empty:
                        continue

                    dp_inv = invdf["abs_size_nm"].to_numpy(dtype=float)
                    n_inv = invdf["N_GWalpha"].to_numpy(dtype=float)
                    if low_value_lift_enabled.value:
                        n_inv = one_sided_low_value_lift(n_inv)

                    for row in invdf.itertuples(index=False):
                        residual_rows.append({
                            "time": g_scan["time"].median(),
                            "scan_id": scan_id,
                            "scan_range": g_range["scan_range"].iloc[0],
                            "method": inversion_method,
                            "polarity": polarity,
                            "abs_size_nm": float(row.abs_size_nm),
                            "measured_cpc": float(row.measured_cpc),
                            "fitted_cpc": float(row.fitted_cpc),
                            "residual_cpc": float(row.residual_cpc),
                            "residual_rel": float(row.residual_rel),
                        })

                    order = np.argsort(dp_inv)
                    scan_parts.append((dp_inv[order], n_inv[order]))

                if not scan_parts:
                    continue

                full_sum = np.zeros(len(size_axis), dtype=float)
                full_count = np.zeros(len(size_axis), dtype=float)
                part_columns = []

                for dp_inv, n_inv in scan_parts:
                    mask = (size_axis >= np.nanmin(dp_inv)) & (size_axis <= np.nanmax(dp_inv))
                    interpolated = np.interp(
                        np.log10(size_axis[mask]),
                        np.log10(dp_inv),
                        n_inv,
                    )
                    valid_interp = np.isfinite(interpolated)
                    mask_indices = np.flatnonzero(mask)
                    full_sum[mask_indices[valid_interp]] += interpolated[valid_interp]
                    full_count[mask_indices[valid_interp]] += 1
                    part_column = np.full(len(size_axis), np.nan)
                    part_column[mask_indices[valid_interp]] = interpolated[valid_interp]
                    part_columns.append(part_column)

                full_col = np.divide(
                    full_sum,
                    full_count,
                    out=np.full(len(size_axis), np.nan),
                    where=full_count > 0,
                )
                if low_value_lift_enabled.value:
                    full_col = one_sided_low_value_lift(full_col)

                ntot_scan = diag.integrate_number_distribution(
                    size_axis, full_col, part_columns=part_columns
                )

                if len(part_columns) >= 2 and inversion_method == inversion_method_values[0]:
                    overlap_metrics = diag.range_overlap_metrics(part_columns)
                    if overlap_metrics is not None:
                        range_overlap_diagnostics.append({
                            "scan_id": str(scan_id),
                            "polarity": polarity,
                            **overlap_metrics,
                        })

                measured_ntot = measured_ntot_raw * dmps_loss_correction_factor_from_distribution(
                    size_axis,
                    full_col,
                    parsed_tube_segments,
                    qa,
                    temp,
                    press,
                    cpc_type=scan_cpc_type,
                )
                ntot_closure_diagnostics.append({
                    "scan_id": str(scan_id),
                    "method": inversion_method,
                    "polarity": polarity,
                    "inverted_ntot": ntot_scan,
                    "measured_ntot_raw": measured_ntot_raw,
                    "measured_ntot": measured_ntot,
                    "cpc_type": scan_cpc_type,
                    "cpc_efficiency_model": (
                        scan_cpc_type if scan_cpc_type in {"3010", "HY09"} else "unity (no validated curve)"
                    ),
                    "ntot_duplicate_samples_ignored": ntot_duplicates,
                    "ntot_duplicate_scope": ntot_duplicate_scope,
                    "closure_ratio": (
                        float(ntot_scan / measured_ntot)
                        if np.isfinite(ntot_scan) and np.isfinite(measured_ntot) and measured_ntot > 0
                        else np.nan
                    ),
                })

                heat_cols.append(full_col)
                heat_times.append(g_scan["time"].median())
                heat_flow_rel_rmse.append(flow_rel_rmse)
                ntot_limit = float(ntot_plot_max.value)
                if np.isfinite(ntot_limit) and ntot_limit > 0 and ntot_scan > ntot_limit:
                    ntot_scan = np.nan
                ntot_vals.append(ntot_scan)
                ntot_measured.append(measured_ntot)

            if heat_cols:
                output.append({
                    "kind": "heatmap",
                    "method": inversion_method,
                    "polarity": polarity,
                    "Z": np.column_stack(heat_cols),
                    "x": heat_times,
                    "y": size_axis,
                    "flow_rel_rmse": heat_flow_rel_rmse,
                })

                output.append({
                    "kind": "ntot",
                    "method": inversion_method,
                    "polarity": polarity,
                    "x": heat_times,
                    "y": ntot_vals,
                    "y_measured": ntot_measured,
                })

            output.append({
                "kind": "residuals",
                "method": inversion_method,
                "polarity": polarity,
                "rows": residual_rows,
            })

    output.append({
        "kind": "ion_ratio",
        "x": [x[0] for x in ion_points],
        "y": [x[1] for x in ion_points],
        "y_smoothed": [x[2] for x in ion_points],
        "selected_dp": [x[3] for x in ion_points],
        "scan_id": [x[4] for x in ion_points],
    })

    output.append({
        "kind": "scan_health",
        "rows": diag.build_scan_health(df, group_key),
    })
    output.append({
        "kind": "cpc_assignment_diagnostics",
        "rows": assignment_diagnostics,
    })
    output.append({
        "kind": "range_overlap_diagnostics",
        "rows": range_overlap_diagnostics,
    })
    output.append({
        "kind": "ntot_closure_diagnostics",
        "rows": ntot_closure_diagnostics,
    })
    output.append({
        "kind": "kernel_sample_residuals",
        "rows": kernel_sample_residuals,
    })

    return output

def fitLinearFit(x, y):
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    y_fit = m * x + c
    residuals = y - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    return m, c, r_squared

def linear(x, m, c):
    return m * x + c

def plot_inversion_result(result):
    global latest_growth_diagnostics, latest_growth_settings
    heatmaps = [tr for tr in result if tr["kind"] == "heatmap"]
    heatmap_keys = [(tr.get("method", "gunn woessner mod"), tr["polarity"]) for tr in heatmaps]
    comparisons = build_scan_smeariii_comparison_heatmaps(result)
    comparison_keys = [key for key in heatmap_keys if key in comparisons]
    median_distributions = build_median_distributions(result)
    growth_diagnostics = diag.build_growth_rate_diagnostics(
        result,
        growth_min_size_nm=float(growth_min_size_nm.value),
        growth_max_size_nm=float(growth_max_size_nm.value),
        growth_threshold_fraction=float(growth_threshold_fraction.value),
        growth_models=growth_models.value,
        method_label=method_label,
        growth_max_gap_minutes=float(growth_max_gap_minutes.value),
        growth_min_event_scans=int(growth_min_event_scans.value),
        growth_max_rate_nm_h=float(growth_max_rate_nm_h.value),
    )
    latest_growth_diagnostics = growth_diagnostics
    latest_growth_settings = {
        "models": list(growth_models.value),
        "min_size_nm": float(growth_min_size_nm.value),
        "max_size_nm": float(growth_max_size_nm.value),
        "threshold_fraction": float(growth_threshold_fraction.value),
        "max_gap_minutes": float(growth_max_gap_minutes.value),
        "min_event_scans": int(growth_min_event_scans.value),
        "max_rate_nm_h": float(growth_max_rate_nm_h.value),
        "app_version": APP_VERSION,
    }
    formation_diagnostics = diag.build_formation_rate_diagnostics(
        result,
        growth_min_size_nm=float(growth_min_size_nm.value),
        growth_max_size_nm=float(growth_max_size_nm.value),
        ntot_limit=float(ntot_plot_max.value),
        method_label=method_label,
    )
    polarity_differences = diag.build_polarity_difference_heatmaps(result)
    scan_health = next((tr.get("rows", []) for tr in result if tr.get("kind") == "scan_health"), [])
    result_t0, result_t1 = result_time_range(result)
    smear_cpc = pd.DataFrame(columns=["time", "SMEARIII_CPC"])
    if result_t0 is not None:
        try:
            smear_cpc = load_smeariii_cpc_for_times([result_t0, result_t1])
        except Exception as e:
            print(f"Could not load SMEAR III CPC for scatter plots: {e}", flush=True)

    subplot_titles = [
        f"{method_label(method)} {polarity} inverted heatmap"
        for method, polarity in heatmap_keys
    ]
    subplot_titles.extend(["Ntot", "Estimated Zn/Zp ratio"])
    if growth_diagnostics:
        rates = ", ".join(
            f"{growth_diag['model']} {growth_diag['polarity']}: "
            f"{growth_diag['growth_rate']:.2f} nm/h"
            for growth_diag in growth_diagnostics
        )
        subplot_titles.append(f"NPF growth-model comparison ({rates})")
    if formation_diagnostics:
        subplot_titles.append("NPF formation-rate / onset diagnostic")
    if scan_health:
        subplot_titles.append("Scan health")
    subplot_titles.extend([
        "Three-day median dN/dlog10Dp",
        "Our CPC Ntot vs SMEAR III CPC",
        "Inverted Ntot vs SMEAR III CPC",
    ])
    subplot_titles.extend(
        f"{method_label(diff['method'])} positive / negative inversion ratio"
        for diff in polarity_differences
    )
    subplot_titles.extend(
        f"{method_label(method)} {polarity} our / SMEAR III SMPS ratio"
        for method, polarity in comparison_keys
    )

    ntot_row = len(heatmap_keys) + 1
    ion_ratio_row = ntot_row + 1
    growth_row = ion_ratio_row + 1 if growth_diagnostics else None
    formation_row = ion_ratio_row + 1 + int(bool(growth_diagnostics)) if formation_diagnostics else None
    scan_health_row = ion_ratio_row + 1 + int(bool(growth_diagnostics)) + int(bool(formation_diagnostics)) if scan_health else None
    median_row = ion_ratio_row + 1 + int(bool(growth_diagnostics)) + int(bool(formation_diagnostics)) + int(bool(scan_health))
    cpc_scatter_row = median_row + 1
    inversion_scatter_row = cpc_scatter_row + 1
    polarity_difference_start_row = inversion_scatter_row + 1
    comparison_start_row = polarity_difference_start_row + len(polarity_differences)
    rows = len(subplot_titles)
    vertical_spacing = min(0.008, 0.14 / max(1, rows - 1))

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=vertical_spacing,
        subplot_titles=subplot_titles,
    )

    heatmap_rows = {key: i + 1 for i, key in enumerate(heatmap_keys)}
    comparison_rows = {
        key: comparison_start_row + i for i, key in enumerate(comparison_keys)
    }
    polarity_difference_rows = {
        diff["method"]: polarity_difference_start_row + i
        for i, diff in enumerate(polarity_differences)
    }
    measured_ntot_added = False
    smear_cpc_added = False
    cpc_scatter_added = False
    scatter_max = 0.0
    inversion_scatter_max = 0.0

    for tr in result:
        if tr["kind"] == "heatmap":
            method = tr.get("method", "gunn woessner mod")
            row = heatmap_rows[(method, tr["polarity"])]
            z = np.clip(tr["Z"], 0, float(heatmap_clip.value))

            fig.add_heatmap(
                z=z,
                x=tr["x"],
                y=tr["y"],
                zmin=0,
                zmax=float(heatmap_clip.value),
                name=f"{method_label(method)} {tr['polarity']} heatmap",
                colorbar=dict(title="dN/dlog10Dp", len=0.35),
                hovertemplate=(
                    "time=%{x|%Y-%m-%d %H:%M}<br>"
                    "dp=%{y:.2f} nm<br>"
                    "dN/dlog10Dp=%{z:.2f}<extra></extra>"
                ),
                row=row,
                col=1,
            )

            for growth_diag in growth_diagnostics:
                if (
                    growth_diag["source_method"] == method
                    and growth_diag["polarity"] == tr["polarity"]
                ):
                    track_color = GROWTH_MODEL_COLORS[growth_diag["model"]]
                    fig.add_scatter(
                        x=growth_diag["time"],
                        y=growth_diag["dp"],
                        mode="lines+markers",
                        marker=dict(size=5, color=track_color),
                        line=dict(width=2, color=track_color),
                        name=(
                            f"Event {growth_diag['event_number']} {growth_diag['model']} "
                            f"{tr['polarity']} "
                            f"({growth_diag['growth_rate']:.2f} nm/h)"
                        ),
                        customdata=np.column_stack((
                            growth_diag["fit"],
                            np.full(len(growth_diag["dp"]), growth_diag["r2"]),
                        )),
                        hovertemplate=(
                            "time=%{x|%Y-%m-%d %H:%M}<br>"
                            "track dp=%{y:.2f} nm<br>fit dp=%{customdata[0]:.2f} nm<br>"
                            f"model={growth_diag['model']}<br>"
                            f"GR={growth_diag['growth_rate']:.2f} nm/h<br>"
                            "R2=%{customdata[1]:.3f}<extra></extra>"
                        ),
                        row=row,
                        col=1,
                    )
                    fig.add_scatter(
                        x=growth_diag["time"],
                        y=growth_diag["fit"],
                        mode="lines",
                        line=dict(width=2, dash="dash", color=track_color),
                        name=(
                            f"Event {growth_diag['event_number']} "
                            f"{growth_diag['model']} robust fit"
                        ),
                        hovertemplate=(
                            "time=%{x|%Y-%m-%d %H:%M}<br>"
                            "fitted dp=%{y:.2f} nm<extra></extra>"
                        ),
                        row=row,
                        col=1,
                    )

            update_log_size_axis(fig, row, tr["y"])
            fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=row, col=1)

        elif tr["kind"] == "ntot":
            method = tr.get("method", "gunn woessner mod")
            y_ntot = diag.guard_diagnostic_values(
                tr["y"],
                float(ntot_plot_max.value),
                use_ntot_limit=True,
                multiplier=8.0,
            )
            fig.add_scatter(
                x=tr["x"],
                y=y_ntot,
                mode="lines+markers",
                name=f"{method_label(method)} Ntot {tr['polarity']}",
                hovertemplate=(
                    f"inversion={method_label(method)} {tr['polarity']}<br>"
                    "time=%{x|%Y-%m-%d %H:%M}<br>"
                    "inverted Ntot=%{y:.2f}<extra></extra>"
                ),
                row=ntot_row,
                col=1,
            )

            if "y_measured" in tr and tr["polarity"] == "positive" and not measured_ntot_added:
                y_measured = diag.guard_diagnostic_values(
                    tr["y_measured"],
                    float(ntot_plot_max.value),
                    use_ntot_limit=True,
                    multiplier=8.0,
                )
                fig.add_scatter(
                    x=tr["x"],
                    y=y_measured,
                    mode="markers",
                    marker_symbol="x",
                    marker_size=10,
                    name=f"Measured Ntot",
                    hovertemplate=(
                        f"source=our CPC Ntot ({method_label(method)} {tr['polarity']} scan times)<br>"
                        "time=%{x|%Y-%m-%d %H:%M}<br>"
                        "our CPC Ntot=%{y:.2f}<extra></extra>"
                    ),
                    row=ntot_row,
                    col=1,
                )
                measured_ntot_added = True

                if not smear_cpc_added and not smear_cpc.empty:
                    smear_y = diag.guard_diagnostic_values(
                        smear_cpc["SMEARIII_CPC"],
                        float(ntot_plot_max.value),
                        use_ntot_limit=True,
                        multiplier=8.0,
                    )
                    fig.add_scatter(
                        x=smear_cpc["time"],
                        y=smear_y,
                        mode="lines+markers",
                        name="SMEAR III CPC",
                        hovertemplate="time=%{x|%Y-%m-%d %H:%M}<br>SMEAR III CPC=%{y:.2f}<extra></extra>",
                        row=ntot_row,
                        col=1,
                    )
                    smear_cpc_added = True

                if not cpc_scatter_added and "y_measured" in tr and tr["polarity"] == "positive":
                    matched = match_to_smeariii_cpc(tr["x"], tr["y_measured"], smear_cpc)
                    matched = diag.filter_ntot_matches(matched, float(ntot_plot_max.value))
                    if not matched.empty:
                        scatter_max = max(
                            scatter_max,
                            float(matched[["value", "SMEARIII_CPC"]].max().max()),
                        )
                        fig.add_scatter(
                            x=matched["value"],
                            y=matched["SMEARIII_CPC"],
                            mode="markers",
                            name="Our CPC vs SMEAR III CPC",
                            customdata=diag.plotly_customdata(
                                matched["time"],
                                matched["ratio"],
                                matched["delta"],
                                [f"{method_label(method)} {tr['polarity']}"] * len(matched),
                            ),
                            hovertemplate=(
                                "time=%{customdata[0]|%Y-%m-%d %H:%M}<br>"
                                "our CPC=%{x:.2f}<br>"
                                "SMEAR III CPC=%{y:.2f}<br>"
                                "our/SMEAR=%{customdata[1]:.3f}<br>"
                                "delta=%{customdata[2]:.2f}<br>"
                                "inversion method=%{customdata[3]}<extra></extra>"
                            ),
                            row=cpc_scatter_row,
                            col=1,
                        )
                        cpc_scatter_added = True

            matched = match_to_smeariii_cpc(tr["x"], tr["y"], smear_cpc)
            matched = diag.filter_ntot_matches(matched, float(ntot_plot_max.value))
            if not matched.empty:
                inversion_scatter_max = max(
                    inversion_scatter_max,
                    float(matched[["value", "SMEARIII_CPC"]].max().max()),
                )

                fig.add_scatter(
                    x=matched["value"],
                    y=matched["SMEARIII_CPC"],
                    mode="markers",
                    name=f"{method_label(method)} {tr['polarity']} vs SMEAR III CPC",
                    customdata=diag.plotly_customdata(
                        matched["time"],
                        matched["ratio"],
                        matched["delta"],
                        [f"{method_label(method)} {tr['polarity']}"] * len(matched),

                    ),
                    hovertemplate=(
                        "time=%{customdata[0]|%Y-%m-%d %H:%M}<br>"
                        "inverted Ntot=%{x:.2f}<br>"
                        "SMEAR III CPC=%{y:.2f}<br>"
                        "inverted/SMEAR=%{customdata[1]:.3f}<br>"
                        "delta=%{customdata[2]:.2f}<br>"
                        "inversion method=%{customdata[3]}<extra></extra>"

                    ),
                    row=inversion_scatter_row,
                    col=1,
                )
                m, c, r2 = fitLinearFit(matched["value"], matched["SMEARIII_CPC"])
                if np.isfinite(m) and np.isfinite(c):
                    fit_label = f"{method_label(method)} {tr['polarity']}"
                    x_fit = np.array([matched["value"].min(), matched["value"].max()])
                    y_fit = linear(x_fit, m, c)
                    fig.add_scatter(
                        x=x_fit,
                        y=y_fit,
                        mode="lines",
                        name=f"{fit_label} vs SMEAR III CPC fit (m={m:.3g}, R2={r2:.3f})",
                        hovertemplate=(
                            f"fitted inversion method={fit_label}<br>"
                            f"m={m:.3g}<br>"
                            "inverted Ntot=%{x:.2f}<br>"
                            "fit SMEAR III CPC=%{y:.2f}<extra></extra>"
                        ),
                        row=inversion_scatter_row,
                        col=1,
                    )

        elif tr["kind"] == "ion_ratio":
            if len(tr["x"]) == 0:
                continue
            zmin = float(zratio_min_widget.value)
            zmax = float(zratio_max_widget.value)
            if zmin > zmax:
                zmin, zmax = zmax, zmin
            fig.add_scatter(
                x=tr["x"],
                y=np.clip(tr["y"], zmin, zmax),
                mode="lines+markers",
                name="Zn/Zp raw",
                customdata=np.column_stack((tr["y"], tr["selected_dp"])),
                hovertemplate="raw Zn/Zp=%{customdata[0]:.3f}<br>dp=%{customdata[1]:.1f} nm<extra></extra>",
                row=ion_ratio_row,
                col=1,
            )
            if "y_smoothed" in tr:
                fig.add_scatter(
                    x=tr["x"],
                    y=np.clip(tr["y_smoothed"], zmin, zmax),
                    mode="lines+markers",
                    name="Zn/Zp smoothed (used)",
                    customdata=np.column_stack((tr["y_smoothed"], tr["selected_dp"])),
                    hovertemplate="smoothed Zn/Zp=%{customdata[0]:.3f}<br>dp=%{customdata[1]:.1f} nm<extra></extra>",
                    row=ion_ratio_row,
                    col=1,
                )

    if growth_row is not None:
        for growth_diag in growth_diagnostics:
            fig.add_scatter(
                x=[
                    f"Event {growth_diag['event_number']} {growth_diag['model']}<br>"
                    f"{growth_diag['polarity']} / {method_label(growth_diag['source_method'])}"
                ],
                y=[growth_diag["growth_rate"]],
                mode="markers",
                marker=dict(size=12, color=GROWTH_MODEL_COLORS[growth_diag["model"]]),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[max(0.0, growth_diag["slope_p90"] - growth_diag["growth_rate"])],
                    arrayminus=[max(0.0, growth_diag["growth_rate"] - growth_diag["slope_p10"])],
                ),
                name=(
                    f"Event {growth_diag['event_number']} "
                    f"{growth_diag['label']} {growth_diag['model']}"
                ),
                customdata=[[
                    growth_diag["r2"], growth_diag["rmse_nm"], growth_diag["n_points"],
                    growth_diag["duration_hours"], growth_diag["fit_quality"],
                    growth_diag["background_quality"],
                ]],
                hovertemplate=(
                    "model=%{x}<br>GR=%{y:.2f} nm/h<br>"
                    "R2=%{customdata[0]:.3f}<br>RMSE=%{customdata[1]:.2f} nm<br>"
                    "fit points=%{customdata[2]:.0f}<br>duration=%{customdata[3]:.2f} h<br>"
                    "fit quality=%{customdata[4]}<br>background=%{customdata[5]}<br>"
                    "whisker=pairwise-slope P10-P90 (not confidence)<extra></extra>"
                ),
                row=growth_row,
                col=1,
            )

    if formation_row is not None:
        for formation_diag in formation_diagnostics:
            customdata = np.column_stack((
                formation_diag["formation_rate"],
                np.full(len(formation_diag["time"]), formation_diag["threshold"]),
                np.full(len(formation_diag["time"]), formation_diag["spike_limit"]),
            ))
            fig.add_scatter(
                x=formation_diag["time"],
                y=formation_diag["concentration"],
                mode="lines+markers",
                name=f"{formation_diag['label']} {formation_diag['size_range']}",
                customdata=customdata,
                hovertemplate=(
                    "time=%{x|%Y-%m-%d %H:%M}<br>"
                    "N(%{fullData.name})=%{y:.2f}<br>"
                    "dN/dt=%{customdata[0]:.2f} cm-3 h-1<br>"
                    "onset threshold=%{customdata[1]:.2f}<br>"
                    "spike guard=%{customdata[2]:.2f}<extra></extra>"
                ),
                row=formation_row,
                col=1,
            )
            if pd.notna(formation_diag["onset_time"]):
                onset_idx = int(np.nanargmin(np.abs(pd.to_datetime(formation_diag["time"]) - formation_diag["onset_time"])))
                fig.add_scatter(
                    x=[formation_diag["onset_time"]],
                    y=[formation_diag["concentration"][onset_idx]],
                    mode="markers",
                    marker=dict(symbol="diamond", size=12),
                    name=f"{formation_diag['label']} onset",
                    hovertemplate="onset=%{x|%Y-%m-%d %H:%M}<br>N=%{y:.2f}<extra></extra>",
                    row=formation_row,
                    col=1,
                )

    if scan_health_row is not None:
        health_df = pd.DataFrame(scan_health).sort_values("time")
        fig.add_scatter(
            x=health_df["time"],
            y=100 * health_df["nan_fraction"],
            mode="lines+markers",
            name="CPC NaN fraction",
            customdata=diag.plotly_customdata(
                health_df["scan_id"],
                health_df["flow_rmse"],
                health_df["flow_rel_rmse"],
                health_df["missing_polarity"],
            ),
            hovertemplate=(
                "scan=%{customdata[0]}<br>"
                "time=%{x|%Y-%m-%d %H:%M}<br>"
                "CPC NaN=%{y:.1f}%<br>"
                "flow RMSE=%{customdata[1]:.3f} L/min<br>"
                "flow relative RMSE=%{customdata[2]:.3%}<br>"
                "missing polarity=%{customdata[3]}<extra></extra>"
            ),
            row=scan_health_row,
            col=1,
        )
        fig.add_scatter(
            x=health_df["time"],
            y=health_df["flow_rmse"],
            mode="lines+markers",
            name="Sheath flow RMSE",
            hovertemplate="time=%{x|%Y-%m-%d %H:%M}<br>flow RMSE=%{y:.3f} L/min<extra></extra>",
            row=scan_health_row,
            col=1,
        )

    for median in median_distributions:
        customdata = np.column_stack((
            median.get("p10", np.full(len(median["dp"]), np.nan)),
            median.get("p90", np.full(len(median["dp"]), np.nan)),
            np.full(len(median["dp"]), median.get("n_scans", np.nan)),
            np.full(len(median["dp"]), median.get("flow_rel_rmse", np.nan)),
            median.get("flow_error", np.full(len(median["dp"]), np.nan)),
        ))
        flow_error = np.asarray(median.get("flow_error", []), dtype=float)
        error_y = None
        if len(flow_error) == len(median["dp"]) and np.any(np.isfinite(flow_error) & (flow_error > 0)):
            error_y = dict(
                type="data",
                array=flow_error,
                visible=True,
                thickness=1.5,
                width=4,
            )
        fig.add_scatter(
            x=median["dp"],
            y=median["median"],
            mode="lines+markers",
            name=f"Median {median['label']}",
            error_y=error_y,
            customdata=customdata,
            hovertemplate=(
                "dp=%{x:.2f} nm<br>"
                "median=%{y:.2f}<br>"
                "p10=%{customdata[0]:.2f}<br>"
                "p90=%{customdata[1]:.2f}<br>"
                "scans=%{customdata[2]:.0f}<br>"
                "sheath rel RMSE=%{customdata[3]:.3%}<br>"
                "flow error bar=+/- %{customdata[4]:.2f}<extra></extra>"
            ),
            row=median_row,
            col=1,
        )

    if scatter_max > 0:
        fig.add_scatter(
            x=[0, scatter_max],
            y=[0, scatter_max],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="1:1 CPC",
            row=cpc_scatter_row,
            col=1,
        )

    if inversion_scatter_max > 0:
        fig.add_scatter(
            x=[0, inversion_scatter_max],
            y=[0, inversion_scatter_max],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="1:1 inversion",
            row=inversion_scatter_row,
            col=1,
        )

    fig.update_yaxes(title_text="Ntot", row=ntot_row, col=1)
    fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=ntot_row, col=1)
    fig.update_yaxes(title_text="Zn/Zp", row=ion_ratio_row, col=1)
    fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=ion_ratio_row, col=1)
    if growth_row is not None:
        fig.update_yaxes(
            title_text="Growth rate (nm/h); whiskers: pairwise-slope P10-P90",
            row=growth_row, col=1,
        )
        fig.update_xaxes(title_text="Banana-track model", row=growth_row, col=1)
    if formation_row is not None:
        fig.update_yaxes(title_text="N in event range", row=formation_row, col=1)
        fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=formation_row, col=1)
    if scan_health_row is not None:
        fig.update_yaxes(title_text="NaN % / flow RMSE", row=scan_health_row, col=1)
        fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=scan_health_row, col=1)
    median_sizes = np.concatenate([
        np.asarray(median["dp"], dtype=float)
        for median in median_distributions
    ]) if median_distributions else np.array([])
    update_log_size_x_axis(fig, median_row, median_sizes)
    fig.update_yaxes(title_text="dN/dlog10Dp", row=median_row, col=1)
    fig.update_xaxes(title_text="Our CPC Ntot", row=cpc_scatter_row, col=1)
    fig.update_yaxes(title_text="SMEAR III CPC Ntot", row=cpc_scatter_row, col=1)
    fig.update_xaxes(title_text="Inverted Ntot", row=inversion_scatter_row, col=1)
    fig.update_yaxes(title_text="SMEAR III CPC Ntot", row=inversion_scatter_row, col=1)

    for diff in polarity_differences:
        row = polarity_difference_rows[diff["method"]]
        fig.add_heatmap(
            z=diff["z"],
            x=diff["x"],
            y=diff["y"],
            zmin=0,
            zmax=2,
            colorscale=[
                [0.0, "#2c7bb6"],
                [0.5, "#ffffbf"],
                [1.0, "#d7191c"],
            ],
            name=f"{method_label(diff['method'])} positive / negative",
            colorbar=dict(title="+/-", len=0.25),
            hovertemplate=(
                "time=%{x|%Y-%m-%d %H:%M}<br>"
                "dp=%{y:.2f} nm<br>"
                "positive/negative=%{z:.3f}<extra></extra>"
            ),
            row=row,
            col=1,
        )
        fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=row, col=1)
        update_log_size_axis(fig, row, diff["y"])

    for key, row in comparison_rows.items():
        method, polarity = key
        comparison = comparisons.get(key)
        if comparison is not None:
            fig.add_heatmap(
                z=comparison["z"],
                x=comparison["x"],
                y=comparison["y"],
                zmin=0,
                zmax=2,
                colorscale=[
                    [0.0, "#2c7bb6"],
                    [0.5, "#ffffbf"],
                    [1.0, "#d7191c"],
                ],
                name=f"{method_label(method)} {polarity} / SMEAR III",
                colorbar=dict(title="ratio", len=0.25),
                hovertemplate=(
                    "time=%{x|%Y-%m-%d %H:%M}<br>"
                    "dp=%{y:.2f} nm<br>"
                    "our/SMEAR III=%{z:.3f}<extra></extra>"
                ),
                row=row,
                col=1,
            )
        fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=row, col=1)
        if comparison is not None:
            update_log_size_axis(fig, row, comparison["y"])
        else:
            fig.update_yaxes(type="log", title_text="dp (nm)", row=row, col=1)

    fig.update_layout(
        height=max(1200, 520 * rows),
        width=1300,
        title="Online inversion result",
        showlegend=True,
        margin=dict(l=50, r=260, t=60, b=30),
        legend=dict(x=1.02, y=1.0),
    )

    inversion_plot.object = fig
    return fig


def plot_residual_diagnostics(result):
    residual_rows = []
    sample_residual_rows = []
    for tr in result:
        if tr.get("kind") == "residuals" and tr.get("rows"):
            residual_rows.extend(tr["rows"])
        elif tr.get("kind") == "kernel_sample_residuals" and tr.get("rows"):
            sample_residual_rows.extend(tr["rows"])

    if not residual_rows:
        residual_plot.object = None
        return None

    df = pd.DataFrame(residual_rows)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in ["abs_size_nm", "measured_cpc", "fitted_cpc", "residual_cpc", "residual_rel"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "abs_size_nm", "measured_cpc", "fitted_cpc"])
    if df.empty:
        residual_plot.object = None
        return None

    summaries = (
        df.groupby(["method", "polarity", "scan_id", "time"], as_index=False)
        .agg(
            rmse=("residual_cpc", lambda x: float(np.sqrt(np.nanmean(np.square(x))))),
            median_abs_rel=("residual_rel", lambda x: float(np.nanmedian(np.abs(x)))),
            n=("residual_cpc", "size"),
        )
        .sort_values("time")
    )

    heatmap_keys = list(df.groupby(["method", "polarity"]).groups.keys())
    rows = 3 + len(heatmap_keys)
    fig = make_subplots(
        rows=rows,
        cols=1,
        vertical_spacing=min(0.03, 0.16 / max(1, rows - 1)),
        subplot_titles=(
            [
                "Measured CPC vs fitted CPC",
                "Residual CPC vs size",
                "Per-scan inversion residual summary",
            ]
            + [f"{method_label(method)} {polarity} relative residual heatmap" for method, polarity in heatmap_keys]
        ),
    )

    scatter_max = float(np.nanmax(df[["measured_cpc", "fitted_cpc"]].to_numpy(dtype=float)))
    sample_df = pd.DataFrame(sample_residual_rows)
    if not sample_df.empty:
        for column in ["measured_cpc", "fitted_cpc", "residual_cpc"]:
            sample_df[column] = pd.to_numeric(sample_df[column], errors="coerce")
        sample_df["sample_time"] = pd.to_datetime(sample_df["sample_time"], errors="coerce")
        sample_df = sample_df.dropna(subset=["sample_time", "measured_cpc", "fitted_cpc"])
        if not sample_df.empty:
            scatter_max = max(
                scatter_max,
                float(np.nanmax(sample_df[["measured_cpc", "fitted_cpc"]].to_numpy(dtype=float))),
            )
            for (method, polarity), g in sample_df.groupby(["method", "polarity"]):
                fig.add_scatter(
                    x=g["measured_cpc"],
                    y=g["fitted_cpc"],
                    mode="markers",
                    marker=dict(symbol="x", size=8),
                    name=f"{method_label(method)} {polarity} sample-domain fit",
                    customdata=diag.plotly_customdata(
                        g["sample_time"], g["scan_id"], g["cpc_sample_id"], g["residual_cpc"]
                    ),
                    hovertemplate=(
                        "sample=%{customdata[2]}<br>"
                        "time=%{customdata[0]|%Y-%m-%d %H:%M:%S}<br>"
                        "scan=%{customdata[1]}<br>"
                        "measured=%{x:.2f}<br>fitted=%{y:.2f}<br>"
                        "sample residual=%{customdata[3]:.2f}<extra></extra>"
                    ),
                    row=1,
                    col=1,
                )
    for (method, polarity), g in df.groupby(["method", "polarity"]):
        label = f"{method_label(method)} {polarity}"
        fig.add_scatter(
            x=g["measured_cpc"],
            y=g["fitted_cpc"],
            mode="markers",
            name=f"{label} fit closure",
            customdata=diag.plotly_customdata(g["time"], g["scan_id"], g["abs_size_nm"], g["residual_rel"]),
            hovertemplate=(
                "time=%{customdata[0]|%Y-%m-%d %H:%M}<br>"
                "scan=%{customdata[1]}<br>"
                "dp=%{customdata[2]:.2f} nm<br>"
                "measured=%{x:.2f}<br>"
                "fitted=%{y:.2f}<br>"
                "relative residual=%{customdata[3]:.2%}<extra></extra>"
            ),
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=g["abs_size_nm"],
            y=g["residual_cpc"],
            mode="markers",
            name=f"{label} residual",
            customdata=diag.plotly_customdata(g["time"], g["scan_id"], g["measured_cpc"], g["fitted_cpc"]),
            hovertemplate=(
                "time=%{customdata[0]|%Y-%m-%d %H:%M}<br>"
                "scan=%{customdata[1]}<br>"
                "dp=%{x:.2f} nm<br>"
                "residual=%{y:.2f}<br>"
                "measured=%{customdata[2]:.2f}<br>"
                "fitted=%{customdata[3]:.2f}<extra></extra>"
            ),
            row=2,
            col=1,
        )

    if np.isfinite(scatter_max) and scatter_max > 0:
        fig.add_scatter(
            x=[0, scatter_max],
            y=[0, scatter_max],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="1:1 fitted CPC",
            row=1,
            col=1,
        )

    for (method, polarity), g in summaries.groupby(["method", "polarity"]):
        label = f"{method_label(method)} {polarity}"
        fig.add_scatter(
            x=g["time"],
            y=g["rmse"],
            mode="lines+markers",
            name=f"{label} RMSE",
            customdata=diag.plotly_customdata(g["scan_id"], g["median_abs_rel"], g["n"]),
            hovertemplate=(
                "scan=%{customdata[0]}<br>"
                "time=%{x|%Y-%m-%d %H:%M}<br>"
                "RMSE=%{y:.2f}<br>"
                "median |rel residual|=%{customdata[1]:.2%}<br>"
                "points=%{customdata[2]:.0f}<extra></extra>"
            ),
            row=3,
            col=1,
        )

    for i, (method, polarity) in enumerate(heatmap_keys, start=4):
        g = df[(df["method"] == method) & (df["polarity"] == polarity)].copy()
        pivot = (
            g.pivot_table(
                index="abs_size_nm",
                columns="time",
                values="residual_rel",
                aggfunc="median",
            )
            .sort_index()
        )
        if pivot.empty:
            continue
        z = np.clip(pivot.to_numpy(dtype=float), -1.0, 1.0)
        fig.add_heatmap(
            z=z,
            x=list(pivot.columns),
            y=pivot.index.to_numpy(dtype=float),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="rel residual", len=0.25),
            name=f"{method_label(method)} {polarity} rel residual",
            hovertemplate="time=%{x|%Y-%m-%d %H:%M}<br>dp=%{y:.2f} nm<br>relative residual=%{z:.2%}<extra></extra>",
            row=i,
            col=1,
        )
        update_log_size_axis(fig, i, pivot.index.to_numpy(dtype=float))
        fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=i, col=1)

    fig.update_xaxes(title_text="Measured CPC", row=1, col=1)
    fig.update_yaxes(title_text="Fitted CPC", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Dp (nm)", row=2, col=1)
    fig.update_yaxes(title_text="Measured - fitted CPC", row=2, col=1)
    fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=3, col=1)
    fig.update_yaxes(title_text="CPC RMSE", row=3, col=1)
    fig.update_layout(
        height=max(1200, 430 * rows),
        width=1300,
        title="Inversion residual diagnostics",
        showlegend=True,
        margin=dict(l=50, r=260, t=70, b=40),
        legend=dict(x=1.02, y=1.0),
    )
    residual_plot.object = fig
    return fig


def plot_difference_diagnostics(result):
    global latest_difference_diagnostics

    t0, t1 = result_time_range(result)
    if t1 is None:
        difference_plot.object = None
        latest_difference_diagnostics = None
        return None

    smear = load_smeariii_sum_range(t1 - pd.Timedelta(hours=3, minutes=30), t1 + pd.Timedelta(minutes=15))
    diagnostics = diag.build_last_hours_smear_difference(
        result,
        smear,
        min_size_nm=float(smallest_size.value),
        peak_min_size_nm=float(difference_peak_min_size_nm.value),
        ntot_limit=float(ntot_plot_max.value),
        hours=3,
    )
    latest_difference_diagnostics = diagnostics
    fig = make_subplots(
        rows=4,
        cols=1,
        vertical_spacing=0.06,
        subplot_titles=[
            "Last 3 h median our / SMEAR III distribution ratio",
            "Integrated concentration match",
            f"Peak diameter shift over time (dp >= {float(difference_peak_min_size_nm.value):.1f} nm)",
            "Last 3 h median peak shape compared with SMEAR III",
        ],
    )

    for item in diagnostics["ratios"]:
        label = f"{method_label(item['method'])} {item['polarity']}"
        fig.add_scatter(
            x=item["size_nm"],
            y=item["ratio_median"],
            mode="lines+markers",
            name=f"{label} ratio",
            hovertemplate="dp=%{x:.2f} nm<br>median ratio=%{y:.3f}<extra></extra>",
            row=1,
            col=1,
        )

    smear_shape_added = False
    for item in diagnostics.get("shapes", []):
        label = f"{method_label(item['method'])} {item['polarity']}"
        fig.add_scatter(
            x=item["size_nm"],
            y=item["our_median"],
            mode="lines+markers",
            name=f"{label} shape",
            hovertemplate="dp=%{x:.2f} nm<br>our median=%{y:.2f}<extra></extra>",
            row=4,
            col=1,
        )
        if not smear_shape_added:
            fig.add_scatter(
                x=item["size_nm"],
                y=item["smear_median"],
                mode="lines+markers",
                line=dict(color="black", dash="dash"),
                name="SMEAR III shape",
                hovertemplate="dp=%{x:.2f} nm<br>SMEAR median=%{y:.2f}<extra></extra>",
                row=4,
                col=1,
            )
            smear_shape_added = True

    matches = diagnostics["matches"]
    if not matches.empty:
        for (method, polarity), g in matches.groupby(["method", "polarity"]):
            label = f"{method_label(method)} {polarity}"
            fig.add_scatter(
                x=g["smear_ntot"],
                y=g["our_ntot"],
                mode="markers",
                name=f"{label} N",
                customdata=diag.plotly_customdata(g["time"], g["ntot_ratio"], g["time_delta_min"]),
                hovertemplate=(
                    "time=%{customdata[0]|%Y-%m-%d %H:%M}<br>"
                    "SMEAR N=%{x:.2f}<br>our N=%{y:.2f}<br>"
                    "our/SMEAR=%{customdata[1]:.3f}<br>"
                    "time offset=%{customdata[2]:.1f} min<extra></extra>"
                ),
                row=2,
                col=1,
            )
            fig.add_scatter(
                x=g["time"],
                y=g["peak_shift_pct"],
                mode="lines+markers",
                name=f"{label} peak shift",
                customdata=diag.plotly_customdata(g["our_peak_nm"], g["smear_peak_nm"]),
                hovertemplate=(
                    "time=%{x|%Y-%m-%d %H:%M}<br>"
                    "peak shift=%{y:.1f}%<br>"
                    f"peak min dp={float(difference_peak_min_size_nm.value):.1f} nm<br>"
                    "our peak=%{customdata[0]:.2f} nm<br>"
                    "SMEAR peak=%{customdata[1]:.2f} nm<extra></extra>"
                ),
                row=3,
                col=1,
            )

        scatter_max = float(matches[["our_ntot", "smear_ntot"]].max().max())
        if np.isfinite(scatter_max) and scatter_max > 0:
            fig.add_scatter(
                x=[0, scatter_max],
                y=[0, scatter_max],
                mode="lines",
                line=dict(color="black", dash="dash"),
                name="1:1",
                row=2,
                col=1,
            )

    fig.update_xaxes(type="log", title_text="Dp (nm)", row=1, col=1)
    fig.update_yaxes(title_text="our / SMEAR", row=1, col=1)
    fig.update_xaxes(title_text="SMEAR integrated N", row=2, col=1)
    fig.update_yaxes(title_text="our integrated N", row=2, col=1)
    fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=3, col=1)
    fig.update_yaxes(title_text="peak shift %", row=3, col=1)
    fig.update_xaxes(type="log", title_text="Dp (nm)", row=4, col=1)
    fig.update_yaxes(title_text="dN/dlog10Dp", row=4, col=1)
    fig.update_layout(
        height=1800,
        width=1300,
        title="Last 3 h DMPS vs SMEAR III difference diagnostics",
        showlegend=True,
        margin=dict(l=50, r=260, t=70, b=40),
        legend=dict(x=1.02, y=1.0),
    )
    difference_plot.object = fig
    return fig


def _integrated_heatmap_series(tr, min_size_nm):
    sizes = np.asarray(tr["y"], dtype=float)
    z = np.asarray(tr["Z"], dtype=float)
    size_mask = np.isfinite(sizes) & (sizes >= float(min_size_nm))
    rows = []
    if np.count_nonzero(size_mask) < 2:
        return pd.DataFrame(columns=["time", "ntot"]), sizes[size_mask]

    event_sizes = sizes[size_mask]
    for t, col in zip(pd.to_datetime(tr["x"]), z.T):
        ntot = diag.integrate_distribution(event_sizes, np.asarray(col, dtype=float)[size_mask])
        rows.append((t, ntot))
    return pd.DataFrame(rows, columns=["time", "ntot"]).dropna(), event_sizes


def _modal_heatmap_series(tr, min_size_nm):
    sizes = np.asarray(tr["y"], dtype=float)
    z = np.asarray(tr["Z"], dtype=float)
    size_mask = np.isfinite(sizes) & (sizes >= float(min_size_nm))
    event_sizes = sizes[size_mask]
    rows = []

    if len(event_sizes) == 0:
        return pd.DataFrame(columns=["time", "mode_dp_nm", "mode_conc"])

    for t, col in zip(pd.to_datetime(tr["x"]), z.T):
        values = np.asarray(col, dtype=float)[size_mask]
        valid = np.isfinite(values)
        if not np.any(valid):
            continue
        valid_indices = np.flatnonzero(valid)
        peak_idx = valid_indices[int(np.nanargmax(values[valid]))]
        rows.append((t, event_sizes[peak_idx], values[peak_idx]))

    return pd.DataFrame(rows, columns=["time", "mode_dp_nm", "mode_conc"]).dropna()


def _representative_timing_offsets(offset_min, offset_max):
    offsets = [float(offset_min)]
    if offset_min <= 0 <= offset_max:
        offsets.append(0.0)
    else:
        offsets.append(float(offset_min + (offset_max - offset_min) / 2.0))
    offsets.append(float(offset_max))
    unique = []
    for offset in offsets:
        rounded = round(offset, 6)
        if rounded not in unique:
            unique.append(rounded)
    return unique


def _plot_smps_timing_without_smear(heatmaps, offset_min, offset_max):
    offsets = _representative_timing_offsets(offset_min, offset_max)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[
            "Inverted Ntot shifted by representative timing offsets",
            "Modal particle diameter shifted by representative timing offsets",
            "Median inverted particle size distribution",
        ],
    )
    summaries = []

    for tr in heatmaps:
        label = f"{method_label(tr.get('method', 'gunn woessner mod'))} {tr.get('polarity', 'unknown')}"
        our, event_sizes = _integrated_heatmap_series(tr, float(smallest_size.value))
        modes = _modal_heatmap_series(tr, float(smallest_size.value))
        z = np.asarray(tr["Z"], dtype=float)
        size_mask = np.isfinite(np.asarray(tr["y"], dtype=float)) & (np.asarray(tr["y"], dtype=float) >= float(smallest_size.value))

        if not our.empty:
            summaries.append(label)
            for offset in offsets:
                fig.add_scatter(
                    x=our["time"] + pd.to_timedelta(offset, unit="s"),
                    y=our["ntot"],
                    mode="lines+markers",
                    name=f"{label} Ntot {offset:.0f}s",
                    hovertemplate="shifted time=%{x|%Y-%m-%d %H:%M:%S}<br>Ntot=%{y:.2f}<extra></extra>",
                    row=1,
                    col=1,
                )

        if not modes.empty:
            for offset in offsets:
                fig.add_scatter(
                    x=modes["time"] + pd.to_timedelta(offset, unit="s"),
                    y=modes["mode_dp_nm"],
                    mode="lines+markers",
                    name=f"{label} mode {offset:.0f}s",
                    customdata=modes["mode_conc"],
                    hovertemplate=(
                        "shifted time=%{x|%Y-%m-%d %H:%M:%S}<br>"
                        "mode dp=%{y:.2f} nm<br>"
                        "mode dN/dlog10Dp=%{customdata:.2f}<extra></extra>"
                    ),
                    row=2,
                    col=1,
                )

        if z.size > 0 and np.any(size_mask):
            stats = diag.nan_stats_by_row(z[size_mask, :])
            fig.add_scatter(
                x=event_sizes,
                y=stats["median"],
                mode="lines+markers",
                name=f"Median {label}",
                customdata=np.column_stack((stats["p10"], stats["p90"])),
                hovertemplate=(
                    "dp=%{x:.2f} nm<br>median=%{y:.2f}<br>"
                    "p10=%{customdata[0]:.2f}<br>p90=%{customdata[1]:.2f}<extra></extra>"
                ),
                row=3,
                col=1,
            )

    if not summaries:
        smps_timing_plot.object = None
        status.object = "No valid inverted scans available for no-SMEAR SMPS timing diagnostics."
        return None

    fig.update_yaxes(title_text="Ntot", row=1, col=1)
    fig.update_xaxes(title_text="Shifted time", tickformat="%H:%M", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Mode Dp (nm)", row=2, col=1)
    fig.update_xaxes(title_text="Shifted time", tickformat="%H:%M", row=2, col=1)
    fig.update_xaxes(type="log", title_text="Dp (nm)", row=3, col=1)
    fig.update_yaxes(title_text="dN/dlog10Dp", row=3, col=1)
    fig.update_layout(
        height=1200,
        width=1300,
        title=(
            "SMPS timing diagnostics without SMEAR: external timing score unavailable; "
            "offsets only shift scan timestamps"
        ),
        showlegend=True,
        margin=dict(l=50, r=260, t=90, b=40),
        legend=dict(x=1.02, y=1.0),
    )
    smps_timing_plot.object = fig
    status.object = "SMPS timing diagnostics updated without SMEAR reference. Showing shifted Ntot, shifted mode diameter, and median distributions."
    return fig


def _smear_integrated_series(smear, event_sizes):
    rows = []
    for t, smear_scan in smear.groupby("time"):
        smear_scan = smear_scan.sort_values("size_nm")
        smear_sizes = smear_scan["size_nm"].to_numpy(dtype=float)
        smear_conc = smear_scan["smear_conc"].to_numpy(dtype=float)
        valid = np.isfinite(smear_sizes) & np.isfinite(smear_conc) & (smear_conc > 0)
        if np.count_nonzero(valid) < 2:
            continue
        interp = np.interp(
            event_sizes,
            smear_sizes[valid],
            smear_conc[valid],
            left=np.nan,
            right=np.nan,
        )
        rows.append((pd.Timestamp(t), diag.integrate_distribution(event_sizes, interp)))
    return pd.DataFrame(rows, columns=["time", "smear_ntot"]).dropna()


def _match_smps_timing(our, smear_ntot, offset_sec, tolerance_min):
    shifted = our.copy()
    shifted["match_time"] = shifted["time"] + pd.to_timedelta(float(offset_sec), unit="s")
    smear_for_match = smear_ntot.rename(columns={"time": "smear_time"}).copy()
    smear_for_match["match_time"] = smear_for_match["smear_time"]
    matched = pd.merge_asof(
        shifted.sort_values("match_time"),
        smear_for_match.sort_values("match_time"),
        on="match_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=float(tolerance_min)),
    ).dropna(subset=["ntot", "smear_ntot"])
    matched = matched[(matched["ntot"] > 0) & (matched["smear_ntot"] > 0)].copy()
    if matched.empty:
        return matched, np.nan, np.nan

    log_ratio = np.log10(matched["ntot"].to_numpy(dtype=float) / matched["smear_ntot"].to_numpy(dtype=float))
    rmse = float(np.sqrt(np.nanmean(log_ratio ** 2)))
    corr = np.nan
    if len(matched) >= 3:
        corr = float(np.corrcoef(
            np.log10(matched["ntot"].to_numpy(dtype=float)),
            np.log10(matched["smear_ntot"].to_numpy(dtype=float)),
        )[0, 1])
    return matched, rmse, corr


def plot_smps_timing_diagnostics(result=None, event=None):
    result = latest_inversion if result is None else result
    if result is None:
        status.object = "Run an inversion before plotting SMPS timing diagnostics."
        return None

    heatmaps = [tr for tr in result if tr.get("kind") == "heatmap"]
    t0, t1 = result_time_range(result)
    if not heatmaps or t0 is None:
        smps_timing_plot.object = None
        status.object = "No inversion heatmaps available for SMPS timing diagnostics."
        return None

    offset_min = float(smps_timing_offset_min_sec.value)
    offset_max = float(smps_timing_offset_max_sec.value)
    offset_step = abs(float(smps_timing_offset_step_sec.value))
    tolerance_min = max(0.1, float(smps_timing_match_tolerance_min.value))
    if offset_step <= 0 or not np.isfinite(offset_step):
        offset_step = 10.0
    if offset_min > offset_max:
        offset_min, offset_max = offset_max, offset_min

    offsets = np.arange(offset_min, offset_max + 0.5 * offset_step, offset_step)
    smear = load_smeariii_sum_range(
        t0 - pd.Timedelta(minutes=tolerance_min) + pd.to_timedelta(offset_min, unit="s"),
        t1 + pd.Timedelta(minutes=tolerance_min) + pd.to_timedelta(offset_max, unit="s"),
    )
    if smear.empty:
        return _plot_smps_timing_without_smear(heatmaps, offset_min, offset_max)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        subplot_titles=[
            "Our inverted Ntot and SMEAR Ntot at best timing offset",
            "Timing fit error vs offset",
            "Timing fit correlation vs offset",
            "Best-offset Ntot scatter",
        ],
    )
    summaries = []

    for tr in heatmaps:
        label = f"{method_label(tr.get('method', 'gunn woessner mod'))} {tr.get('polarity', 'unknown')}"
        our, event_sizes = _integrated_heatmap_series(tr, float(smallest_size.value))
        if our.empty or len(event_sizes) < 2:
            continue
        smear_ntot = _smear_integrated_series(smear, event_sizes)
        if smear_ntot.empty:
            continue

        scores = []
        matches_by_offset = {}
        for offset in offsets:
            matched, rmse, corr = _match_smps_timing(our, smear_ntot, offset, tolerance_min)
            scores.append((offset, rmse, corr, len(matched)))
            matches_by_offset[float(offset)] = matched

        score_df = pd.DataFrame(scores, columns=["offset_sec", "log_rmse", "corr", "n"])
        valid_scores = score_df[np.isfinite(score_df["log_rmse"])]
        if valid_scores.empty:
            continue
        best = valid_scores.sort_values(["log_rmse", "offset_sec"]).iloc[0]
        best_offset = float(best["offset_sec"])
        best_matched = matches_by_offset[best_offset]
        summaries.append(f"{label}: best {best_offset:.0f} s, logRMSE {best['log_rmse']:.3f}, r {best['corr']:.3f}, n {int(best['n'])}")

        fig.add_scatter(
            x=smear_ntot["time"],
            y=smear_ntot["smear_ntot"],
            mode="lines",
            name=f"SMEAR {label}",
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=our["time"],
            y=our["ntot"],
            mode="lines+markers",
            line=dict(dash="dot"),
            name=f"Our raw time {label}",
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=our["time"] + pd.to_timedelta(best_offset, unit="s"),
            y=our["ntot"],
            mode="lines+markers",
            name=f"Our best {best_offset:.0f}s {label}",
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=score_df["offset_sec"],
            y=score_df["log_rmse"],
            mode="lines+markers",
            name=f"RMSE {label}",
            customdata=np.column_stack((score_df["n"], score_df["corr"])),
            hovertemplate="offset=%{x:.0f}s<br>logRMSE=%{y:.3f}<br>matches=%{customdata[0]:.0f}<br>r=%{customdata[1]:.3f}<extra></extra>",
            row=2,
            col=1,
        )
        fig.add_scatter(
            x=[best_offset],
            y=[best["log_rmse"]],
            mode="markers",
            marker=dict(symbol="diamond", size=12),
            name=f"Best RMSE {label}",
            row=2,
            col=1,
        )
        fig.add_scatter(
            x=score_df["offset_sec"],
            y=score_df["corr"],
            mode="lines+markers",
            name=f"Correlation {label}",
            row=3,
            col=1,
        )
        if not best_matched.empty:
            fig.add_scatter(
                x=best_matched["ntot"],
                y=best_matched["smear_ntot"],
                mode="markers",
                name=f"Best scatter {label}",
                customdata=diag.plotly_customdata(best_matched["time"], best_matched["smear_time"]),
                hovertemplate=(
                    "our time=%{customdata[0]|%Y-%m-%d %H:%M:%S}<br>"
                    "matched SMEAR time=%{customdata[1]|%Y-%m-%d %H:%M:%S}<br>"
                    "our N=%{x:.2f}<br>SMEAR N=%{y:.2f}<extra></extra>"
                ),
                row=4,
                col=1,
            )

    if not summaries:
        return _plot_smps_timing_without_smear(heatmaps, offset_min, offset_max)

    fig.update_yaxes(title_text="Ntot", row=1, col=1)
    fig.update_xaxes(title_text="Time", tickformat="%H:%M", row=1, col=1)
    fig.update_yaxes(title_text="log10 ratio RMSE", row=2, col=1)
    fig.update_xaxes(title_text="Offset applied to our data (s)", row=2, col=1)
    fig.update_yaxes(title_text="log10 N correlation", row=3, col=1)
    fig.update_xaxes(title_text="Offset applied to our data (s)", row=3, col=1)
    fig.update_xaxes(title_text="Our inverted Ntot", row=4, col=1)
    fig.update_yaxes(title_text="SMEAR integrated Ntot", row=4, col=1)
    fig.update_layout(
        height=1300,
        width=1300,
        title="SMPS timing diagnostics: " + " | ".join(summaries),
        showlegend=True,
        margin=dict(l=50, r=260, t=90, b=40),
        legend=dict(x=1.02, y=1.0),
    )
    smps_timing_plot.object = fig
    status.object = "SMPS timing diagnostics updated. " + " | ".join(summaries)
    return fig


def update_smps_timing_plot(event=None):
    return plot_smps_timing_diagnostics()


def run_inversion(event=None):
    global inversion_running

    df = load_selected_scans()
    if df.empty:
        status.object = "No selected scan data for inversion."
        return

    with inversion_lock:
        if inversion_running:
            status.object = "Inversion already running."
            return
        inversion_running = True

    status.object = "Running inversion..."

    fut = inversion_executor.submit(run_inversion_calculation, df)

    doc = pn.state.curdoc
    def done_callback(future):
        global inversion_running, latest_inversion, auto_pending_signature
        try:
            result = future.result()
            latest_inversion = result
            def finish_success():
                global auto_pending_signature

                fig = plot_inversion_result(result)
                residual_fig = plot_residual_diagnostics(result)
                diff_fig = plot_difference_diagnostics(result)
                smps_timing_fig = plot_smps_timing_diagnostics(result)

                if auto_checkbox.value:
                    save_data()
                    if auto_pending_signature is not None:
                        save_auto_state({"last_saved_signature": auto_pending_signature})
                        auto_pending_signature = None
                    status_text = "Auto-run: inversion finished and saved."
                else:
                    status_text = "Inversion finished."

                assignment_rows = next(
                    (item["rows"] for item in result if item.get("kind") == "cpc_assignment_diagnostics"),
                    [],
                )
                if assignment_rows:
                    fallback_bins = sum(
                        len(row.get("source_fallback_bins", []))
                        + len(row.get("interpolated_bins", []))
                        + len(row.get("edge_fallback_bins", []))
                        for row in assignment_rows
                    )
                    kernel_rows = [
                        row for row in assignment_rows
                        if row.get("mode") == "Response kernel (experimental)"
                    ]
                    duplicate_ids = (
                        max(row.get("duplicate_cpc_ids_ignored", 0) for row in kernel_rows)
                        if kernel_rows
                        else sum(row.get("duplicate_cpc_ids_ignored", 0) for row in assignment_rows)
                    )
                    boundary_discards = (
                        max(row.get("mixed_boundary_discards", 0) for row in kernel_rows)
                        if kernel_rows
                        else 0
                    )
                    rejected_kernels = sum(
                        row.get("kernel_usable") is False for row in kernel_rows
                    )
                    ill_conditioned = sum(
                        bool(row.get("ill_conditioned", False)) for row in kernel_rows
                    )
                    status_text += (
                        f" CPC assignment: {fallback_bins} fallback bins, "
                        f"{duplicate_ids} duplicate sample IDs ignored, "
                        f"{boundary_discards} mixed-boundary samples discarded, "
                        f"{rejected_kernels} kernels rejected, "
                        f"{ill_conditioned} conditioning warnings."
                    )

                overlap_rows = next(
                    (item["rows"] for item in result if item.get("kind") == "range_overlap_diagnostics"),
                    [],
                )
                finite_seams = np.asarray([
                    row.get("median_relative_seam", np.nan) for row in overlap_rows
                ], dtype=float)
                finite_seams = finite_seams[np.isfinite(finite_seams)]
                if len(finite_seams):
                    status_text += f" Median range seam {100 * np.median(finite_seams):.1f}%."

                closure_rows = next(
                    (item["rows"] for item in result if item.get("kind") == "ntot_closure_diagnostics"),
                    [],
                )
                closure = np.asarray([
                    row.get("closure_ratio", np.nan) for row in closure_rows
                ], dtype=float)
                closure = closure[np.isfinite(closure)]
                if len(closure):
                    status_text += f" Median Ntot closure {np.median(closure):.2f}."

                status.object = status_text
                publish_shared_state(
                    inversion_fig=fig,
                    residual_fig=residual_fig,
                    smps_timing_fig=smps_timing_fig,
                    difference_fig=diff_fig,
                    difference_diagnostics=latest_difference_diagnostics,
                    growth_diagnostics=latest_growth_diagnostics,
                    growth_settings=latest_growth_settings,
                    inversion_result=result,
                    status_text=status_text,
                )

            if doc is not None:
                doc.add_next_tick_callback(finish_success)
            else:
                finish_success()
        except Exception:
            auto_pending_signature = None
            traceback.print_exc()
            if doc is not None:
                doc.add_next_tick_callback(
                    lambda: setattr(status, "object", "Inversion failed. Check terminal.")
                )
            else:
                status.object = "Inversion failed. Check terminal."
        finally:
            with inversion_lock:
                inversion_running = False

    fut.add_done_callback(done_callback)


def auto_refresh_invert_save():
    global auto_pending_signature

    if not auto_checkbox.value:
        return

    signature = prepare_auto_selection()
    if signature is None:
        return

    with inversion_lock:
        running = inversion_running

    if running:
        status.object = "Auto-run: inversion already running."
        return

    auto_pending_signature = signature
    status.object = "Auto-run: new scans detected, running inversion."
    run_inversion()


def prepare_auto_selection():
    min_age = max(0, int(auto_file_age_sec.value))
    files = files_for_selection(min_age_sec=min_age)

    scan_files.options = [str(p) for p in list_scan_files(min_age_sec=min_age)]

    if not files:
        status.object = "Auto-run: no completed scan files found."
        return

    scan_files.value = [str(p) for p in files]

    signature = selected_files_signature()
    state = load_auto_state()

    if signature == state.get("last_saved_signature"):
        status.object = "Auto-run: no new selected scans."
        return None

    return signature


def run_auto_worker():
    global latest_inversion

    print("Auto inversion worker started. Press Ctrl+C to stop.", flush=True)

    while True:
        try:
            signature = prepare_auto_selection()

            if signature is not None:
                print("Auto-run: new scans detected, running inversion.", flush=True)
                plot_selected_scans()
                df = load_selected_scans()
                if df.empty:
                    print("Auto-run: selected scans could not be loaded.", flush=True)
                    continue

                result = run_inversion_calculation(df)

                latest_inversion = result
                fig = plot_inversion_result(result)
                residual_fig = plot_residual_diagnostics(result)
                diff_fig = plot_difference_diagnostics(result)
                smps_timing_fig = plot_smps_timing_diagnostics(result)
                save_data()
                save_auto_state({"last_saved_signature": signature})
                publish_shared_state(
                    inversion_fig=fig,
                    residual_fig=residual_fig,
                    smps_timing_fig=smps_timing_fig,
                    difference_fig=diff_fig,
                    difference_diagnostics=latest_difference_diagnostics,
                    growth_diagnostics=latest_growth_diagnostics,
                    growth_settings=latest_growth_settings,
                    inversion_result=result,
                    status_text=str(status.object),
                )
                print(str(status.object), flush=True)

        except KeyboardInterrupt:
            print("Auto inversion worker stopped.", flush=True)
            return
        except Exception:
            traceback.print_exc()

        time.sleep(max(1, int(auto_interval_min.value)) * 60)

save_button.on_click(save_data)
invert_button.on_click(run_inversion)
smps_timing_button.on_click(update_smps_timing_plot)


for w in [
    scan_root,
    save_root,
    n_scans_plot,
    scan_selection_mode,
    scan_start_date,
    scan_start_time,
    scan_end_date,
    scan_end_time,
    loaded_time_window_min,
    auto_interval_min,
    auto_file_age_sec,
    daily_overwrite_checkbox,
    dma_L,
    dma_r1,
    dma_r2,
    qa_lpm,
    qs_lpm,
    temp_K,
    press_Pa,
    zratio_widget,
    zratio_min_widget,
    zratio_max_widget,
    zratio_smoothing_step,
    zratio_min_size_nm,
    zratio_estimate_offset,
    ntot_plot_max,
    heatmap_clip,
    raw_uncertainty,
    growth_models,
    growth_min_size_nm,
    growth_max_size_nm,
    growth_threshold_fraction,
    growth_max_gap_minutes,
    growth_min_event_scans,
    growth_max_rate_nm_h,
    difference_peak_min_size_nm,
    smallest_size,
    scan_inversion_type,
    smps_settling_time_sec,
    smps_correction_mode,
    smps_transport_delay_sec,
    smps_response_window_sec,
    smps_dwell_sec,
    smps_kernel_timing_override,
    smps_kernel_smoothness,
    smps_size_step_shift,
    smps_timing_offset_min_sec,
    smps_timing_offset_max_sec,
    smps_timing_offset_step_sec,
    smps_timing_match_tolerance_min,
    inversion_size_bin_decimals,
    cpc_gap_interpolation_enabled,
    low_value_lift_enabled,
    low_value_lift_ratio,
    low_value_lift_alpha,
    tube_segments,
    inversion_methods,
]:
    w.param.watch(lambda event: save_settings(), "value")


selection_controls = pn.Column(
    pn.Row(scan_root, refresh_button, select_last_button, n_scans_plot),
    pn.Row(scan_selection_mode, scan_start_date, scan_start_time, scan_end_date, scan_end_time),
    loaded_time_window_min,
    pn.Accordion(("Selected scan CSVs", scan_files), active=[]),
)

inversion_controls = pn.Column(
    pn.Row(dma_L, dma_r1, dma_r2),
    pn.Row(qa_lpm, qs_lpm, temp_K, press_Pa),
    pn.Row(zratio_widget, zratio_min_widget, zratio_max_widget, zratio_smoothing_step),
    pn.Row(zratio_min_size_nm, zratio_estimate_offset, use_zratio_checkbox, smallest_size),
    pn.Row(scan_inversion_type, smps_settling_time_sec, smps_correction_mode, smps_transport_delay_sec),
    pn.Row(
        smps_response_window_sec, smps_dwell_sec,
        smps_kernel_timing_override, smps_kernel_smoothness,
    ),
    pn.Row(smps_size_step_shift, inversion_methods),
    pn.Row(inversion_size_bin_decimals, cpc_gap_interpolation_enabled, low_value_lift_enabled),
    pn.Row(low_value_lift_ratio, low_value_lift_alpha),
    tube_segments,
)

diagnostic_controls = pn.Column(
    pn.Row(ntot_plot_max, heatmap_clip, raw_uncertainty),
    pn.Row(growth_models),
    pn.Row(
        growth_min_size_nm, growth_max_size_nm, growth_threshold_fraction,
        growth_max_gap_minutes, growth_min_event_scans,
        growth_max_rate_nm_h,
    ),
    pn.Row(difference_peak_min_size_nm),
    pn.Row(smps_timing_offset_min_sec, smps_timing_offset_max_sec, smps_timing_offset_step_sec, smps_timing_match_tolerance_min),
)

automation_controls = pn.Column(
    pn.Row(save_root),
    pn.Row(auto_checkbox, daily_overwrite_checkbox, auto_interval_min, auto_file_age_sec),
)

controls = pn.Column(
    pn.Tabs(
        ("Scan Selection", selection_controls),
        ("Inversion Settings", inversion_controls),
        ("Diagnostics", diagnostic_controls),
        ("Automation / Save", automation_controls),
        dynamic=True,
    ),
    pn.Row(plot_button, invert_button, smps_timing_button, save_button, status),
)

plot_tabs = pn.Tabs(
    ("Raw Data", pn.Column(raw_plot)),
    ("Inversion", pn.Column(inversion_plot)),
    ("Residuals", pn.Column(residual_plot)),
    ("SMPS Timing", pn.Column(smps_timing_plot)),
    ("Difference Diagnostics", pn.Column(difference_plot)),
    dynamic=True,
)

layout = pn.Column(
    f"# Online DMPS inversion / scan viewer v{APP_VERSION}",
    controls,
    plot_tabs,
    width=1400,
)

auto_callback = None
shared_sync_callback = None


def start_app():
    global auto_callback, shared_sync_callback

    print(f"DMPS inversion GUI {APP_VERSION}: {Path(__file__).resolve()}", flush=True)
    refresh_scan_files()
    if auto_callback is None:
        auto_callback = pn.state.add_periodic_callback(
            auto_refresh_invert_save,
            period=max(1, int(auto_interval_min.value)) * 60 * 1000,
            start=True,
        )
    if shared_sync_callback is None:
        shared_sync_callback = pn.state.add_periodic_callback(
            sync_shared_state,
            period=2000,
            start=True,
        )
    return layout


def update_auto_period(event):
    if auto_callback is not None:
        auto_callback.period = max(1, int(auto_interval_min.value)) * 60 * 1000


auto_interval_min.param.watch(update_auto_period, "value")
