"""Compare independently inverted instrument heatmaps on measured support."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from DMPS_inversion_gui import diagnostics as diag


def select_heatmaps(results_by_source, method, polarity):
    selected = {}
    for source, result in results_by_source.items():
        trace = next((
            item for item in result
            if item.get("kind") == "heatmap"
            and item.get("method", "gunn woessner mod") == method
            and item.get("polarity") == polarity
        ), None)
        if trace is None:
            continue
        sizes = np.asarray(trace["y"], dtype=float)
        concentrations = np.asarray(trace["Z"], dtype=float)
        if (
            concentrations.ndim != 2
            or concentrations.shape != (len(sizes), len(trace["x"]))
            or not np.any(np.isfinite(concentrations))
        ):
            continue
        selected[source] = trace
    return selected


def common_diameter_range(heatmaps):
    bounds = []
    for trace in heatmaps.values():
        sizes = np.asarray(trace["y"], dtype=float)
        observed = np.any(np.isfinite(np.asarray(trace["Z"], dtype=float)), axis=1)
        measured = sizes[observed & np.isfinite(sizes) & (sizes > 0)]
        if measured.size < 2:
            return None
        bounds.append((float(measured.min()), float(measured.max())))
    if not bounds:
        return None
    lower = max(bound[0] for bound in bounds)
    upper = min(bound[1] for bound in bounds)
    return (lower, upper) if lower < upper else None


def number_on_common_support(trace, lower_nm, upper_nm):
    """Integrate dN/dlog10Dp only over measured bins in a shared size range."""
    sizes = np.asarray(trace["y"], dtype=float)
    z = np.asarray(trace["Z"], dtype=float)
    measured_range = np.isfinite(sizes) & (sizes >= lower_nm) & (sizes <= upper_nm)
    part_columns = trace.get("part_columns", [])
    numbers = []
    for index in range(z.shape[1]):
        parts = part_columns[index] if index < len(part_columns) else None
        widths = diag.distribution_support_widths(sizes, parts)
        valid = (
            measured_range & np.isfinite(z[:, index]) & (z[:, index] >= 0)
            & np.isfinite(widths) & (widths > 0)
        )
        numbers.append(float(np.sum(z[valid, index] * widths[valid])) if np.count_nonzero(valid) >= 2 else np.nan)
    return numbers


def nearest_number_ratio(times, numbers, reference_times, reference_numbers):
    """Pair scans within 15 minutes; leave unpaired scans blank, without interpolation."""
    target = pd.DataFrame({"time": pd.to_datetime(times), "N": numbers})
    reference = pd.DataFrame({
        "time": pd.to_datetime(reference_times), "reference_N": reference_numbers,
    })
    valid_reference = reference[
        np.isfinite(reference["reference_N"]) & (reference["reference_N"] > 0)
    ].sort_values("time")
    if valid_reference.empty:
        return [np.nan] * len(target)
    paired = pd.merge_asof(
        target.reset_index().sort_values("time"), valid_reference,
        on="time", direction="nearest", tolerance=pd.Timedelta(minutes=15),
    ).sort_values("index")
    return np.divide(
        paired["N"].to_numpy(dtype=float), paired["reference_N"].to_numpy(dtype=float),
        out=np.full(len(target), np.nan),
        where=paired["reference_N"].notna().to_numpy(),
    ).tolist()


def build_comparison_figure(results_by_source, method, polarity, max_concentration=20000):
    heatmaps = select_heatmaps(results_by_source, method, polarity)
    if not heatmaps:
        return None, "Run an inversion with this method and polarity in an instrument tab first."

    bounds = common_diameter_range(heatmaps) if len(heatmaps) >= 2 else None
    titles = [f"{source} — {method} {polarity}-voltage scan" for source in heatmaps]
    if bounds:
        titles.append(f"Measured-range N ({bounds[0]:g}–{bounds[1]:g} nm; no extrapolation)")
        titles.append("N ratio to reference (nearest scan within 15 min)")
    figure = make_subplots(
        rows=len(titles), cols=1, shared_xaxes=True,
        vertical_spacing=min(0.08, 0.18 / max(1, len(titles) - 1)),
        subplot_titles=titles,
    )
    number_series = {}
    for row, (source, trace) in enumerate(heatmaps.items(), start=1):
        figure.add_trace(go.Heatmap(
            x=pd.to_datetime(trace["x"]), y=trace["y"], z=trace["Z"],
            coloraxis="coloraxis", name=source,
            hovertemplate="time=%{x}<br>Dp=%{y:.2f} nm<br>dN/dlog10Dp=%{z:.1f}<extra>" + source + "</extra>",
        ), row=row, col=1)
        figure.update_yaxes(type="log", title_text="Dp (nm)", row=row, col=1)
        if bounds:
            numbers = number_on_common_support(trace, *bounds)
            number_series[source] = (trace["x"], numbers)
            figure.add_trace(go.Scatter(
                x=pd.to_datetime(trace["x"]),
                y=numbers,
                mode="lines+markers", name=source,
            ), row=len(heatmaps) + 1, col=1)

    if bounds:
        number_row = len(heatmaps) + 1
        ratio_row = number_row + 1
        figure.update_yaxes(title_text="N (cm-3)", row=number_row, col=1)
        reference_source = "Bipolar Pi" if "Bipolar Pi" in number_series else next(iter(number_series))
        reference_times, reference_numbers = number_series[reference_source]
        for source, (times, numbers) in number_series.items():
            if source == reference_source:
                continue
            figure.add_trace(go.Scatter(
                x=pd.to_datetime(times),
                y=nearest_number_ratio(times, numbers, reference_times, reference_numbers),
                mode="lines+markers", name=f"{source} / {reference_source}",
            ), row=ratio_row, col=1)
        figure.add_hline(y=1, row=ratio_row, col=1, line_dash="dash", line_color="gray")
        figure.update_yaxes(title_text="N ratio", row=ratio_row, col=1)
    figure.update_xaxes(title_text="Time", row=len(titles), col=1)
    figure.update_layout(
        title="Independent instrument inversions (same color scale)",
        height=max(650, 420 * len(titles)), width=1300,
        coloraxis=dict(colorscale="Viridis", cmin=0, cmax=max(float(max_concentration), 1.0), colorbar=dict(title="dN/dlog10Dp")),
        margin=dict(l=85, r=130, t=85, b=55),
    )
    note = f"Showing {len(heatmaps)} independently inverted source(s)."
    if not bounds:
        note += " Run at least two sources with overlapping measured diameters for the common-range N comparison."
    else:
        note += (
            f" N compares only measured bins between {bounds[0]:g} and {bounds[1]:g} nm; "
            "ratios use nearest scans within 15 minutes, without time interpolation."
        )
    return figure, note
