import numpy as np
import pandas as pd


def plotly_customdata(*columns):
    return np.asarray(list(zip(*columns)), dtype=object)


def nan_stats_by_row(values):
    values = np.asarray(values, dtype=float)
    stats = {
        "median": np.full(values.shape[0], np.nan),
        "p10": np.full(values.shape[0], np.nan),
        "p90": np.full(values.shape[0], np.nan),
    }
    valid = np.any(np.isfinite(values), axis=1)
    if np.any(valid):
        stats["median"][valid] = np.nanmedian(values[valid], axis=1)
        stats["p10"][valid] = np.nanpercentile(values[valid], 10, axis=1)
        stats["p90"][valid] = np.nanpercentile(values[valid], 90, axis=1)
    return stats


def robust_upper_limit(values, multiplier=8.0):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 4:
        return np.inf
    median = np.nanmedian(finite)
    mad = np.nanmedian(np.abs(finite - median))
    if mad > 0:
        return median + multiplier * 1.4826 * mad
    p75, p25 = np.nanpercentile(finite, [75, 25])
    iqr = p75 - p25
    if iqr > 0:
        return p75 + multiplier * iqr
    return np.nanmax(finite) * 3


def suppress_isolated_spikes(values, multiplier=8.0):
    values = np.asarray(values, dtype=float).copy()
    limit = robust_upper_limit(values, multiplier=multiplier)
    if not np.isfinite(limit):
        return values, limit
    high = np.isfinite(values) & (values > limit)
    for i in np.flatnonzero(high):
        prev_ok = i == 0 or not np.isfinite(values[i - 1]) or values[i - 1] <= limit
        next_ok = i == len(values) - 1 or not np.isfinite(values[i + 1]) or values[i + 1] <= limit
        if prev_ok and next_ok:
            values[i] = np.nan
    return values, limit


def filter_ntot_matches(matched, ntot_limit):
    if matched.empty:
        return matched
    filtered = matched.copy()
    if np.isfinite(ntot_limit) and ntot_limit > 0:
        filtered = filtered[
            (filtered["value"] <= ntot_limit)
            & (filtered["SMEARIII_CPC"] <= ntot_limit)
        ]
    if filtered.empty:
        return filtered
    value_limit = robust_upper_limit(filtered["value"], multiplier=8.0)
    smear_limit = robust_upper_limit(filtered["SMEARIII_CPC"], multiplier=8.0)
    return filtered[
        (filtered["value"] <= value_limit)
        & (filtered["SMEARIII_CPC"] <= smear_limit)
    ]


def sheath_flow_relative_rmse(scan_rows):
    flow_source = scan_rows["sheath_flow"] if "sheath_flow" in scan_rows else pd.Series(np.nan, index=scan_rows.index)
    setpoint_source = scan_rows["sheath_setpoint"] if "sheath_setpoint" in scan_rows else pd.Series(np.nan, index=scan_rows.index)
    flow = pd.to_numeric(flow_source, errors="coerce")
    setpoint = pd.to_numeric(setpoint_source, errors="coerce")
    flow_error = flow - setpoint
    flow_rmse = np.sqrt(np.nanmean(np.square(flow_error)))
    setpoint_median = np.nanmedian(setpoint)
    if not np.isfinite(flow_rmse) or not np.isfinite(setpoint_median) or setpoint_median == 0:
        return np.nan, np.nan
    return float(flow_rmse), float(abs(flow_rmse / setpoint_median))


def log_diameter_bin_widths(size_nm):
    """Return inferred dlog10Dp bin widths for positive diameter-bin centers."""
    size_nm = np.asarray(size_nm, dtype=float)
    widths = np.full(size_nm.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(size_nm) & (size_nm > 0))
    if len(valid_indices) < 2:
        return widths
    order = valid_indices[np.argsort(size_nm[valid_indices])]
    log_sizes = np.log10(size_nm[order])
    if np.any(np.diff(log_sizes) <= 0):
        return widths
    edges = np.empty(len(log_sizes) + 1, dtype=float)
    edges[1:-1] = 0.5 * (log_sizes[:-1] + log_sizes[1:])
    edges[0] = log_sizes[0] - 0.5 * (log_sizes[1] - log_sizes[0])
    edges[-1] = log_sizes[-1] + 0.5 * (log_sizes[-1] - log_sizes[-2])
    widths[order] = np.diff(edges)
    return widths


def distribution_support_widths(size_nm, part_columns=None):
    """Return bin widths without bridging independently measured ranges."""
    size_nm = np.asarray(size_nm, dtype=float)
    if part_columns is None:
        return log_diameter_bin_widths(size_nm)

    support = np.zeros(size_nm.shape, dtype=float)
    valid_sizes = np.isfinite(size_nm) & (size_nm > 0)
    order = np.flatnonzero(valid_sizes)[np.argsort(size_nm[valid_sizes])]
    if len(order) < 2:
        support[~valid_sizes] = np.nan
        return support
    intervals = np.diff(np.log10(size_nm[order]))
    interval_supported = np.zeros(len(intervals), dtype=bool)
    for column in part_columns:
        part = np.asarray(column, dtype=float)
        available = np.isfinite(part[order])
        interval_supported |= available[:-1] & available[1:]
    valid_intervals = interval_supported & np.isfinite(intervals) & (intervals > 0)
    support[order[:-1]] += 0.5 * intervals * valid_intervals
    support[order[1:]] += 0.5 * intervals * valid_intervals
    support[~valid_sizes] = np.nan
    return support


def distribution_bin_coverage(size_nm, concentration, part_columns=None):
    size_nm = np.asarray(size_nm, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    widths = distribution_support_widths(size_nm, part_columns)
    available = np.isfinite(size_nm) & (size_nm > 0) & np.isfinite(concentration)
    finite_width = np.isfinite(widths) & (widths > 0)
    full_widths = log_diameter_bin_widths(size_nm)
    full = np.isfinite(full_widths) & (full_widths > 0)
    total_width = float(np.sum(full_widths[full]))
    if total_width <= 0:
        return np.nan
    return float(np.sum(widths[finite_width & available]) / total_width)


def integrate_number_distribution(size_nm, concentration, part_columns=None):
    size_nm = np.asarray(size_nm, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    valid = np.isfinite(size_nm) & (size_nm > 0) & np.isfinite(concentration)
    if np.count_nonzero(np.isfinite(size_nm) & (size_nm > 0)) < 2:
        return np.nan
    if part_columns is None:
        widths = log_diameter_bin_widths(size_nm)
        usable = valid & np.isfinite(widths) & (widths > 0)
        if not np.any(usable):
            return np.nan
        return float(np.sum(concentration[usable] * widths[usable]))

    valid_sizes = np.isfinite(size_nm) & (size_nm > 0)
    order = np.flatnonzero(valid_sizes)[np.argsort(size_nm[valid_sizes])]
    intervals = np.diff(np.log10(size_nm[order]))
    supported = np.zeros(len(intervals), dtype=bool)
    for column in part_columns:
        part = np.asarray(column, dtype=float)
        available = np.isfinite(part[order])
        supported |= available[:-1] & available[1:]
    connected = supported & valid[order[:-1]] & valid[order[1:]]
    if not np.any(connected):
        return np.nan
    trapezoids = 0.5 * (
        concentration[order[:-1]] + concentration[order[1:]]
    ) * intervals
    return float(np.sum(trapezoids[connected]))


def range_overlap_metrics(part_columns):
    if len(part_columns) < 2:
        return None
    values = np.vstack([np.asarray(column, dtype=float) for column in part_columns])
    overlap = np.sum(np.isfinite(values), axis=0) >= 2
    if not overlap.any():
        return None
    overlap_values = values[:, overlap]
    span = np.nanmax(overlap_values, axis=0) - np.nanmin(overlap_values, axis=0)
    scale = np.nanmean(np.abs(overlap_values), axis=0)
    relative = np.divide(
        span,
        scale,
        out=np.full(len(span), np.nan),
        where=scale > 0,
    )
    finite = relative[np.isfinite(relative)]
    return {
        "overlap_bin_count": int(overlap.sum()),
        "median_relative_seam": float(np.median(finite)) if len(finite) else np.nan,
        "p90_relative_seam": float(np.percentile(finite, 90)) if len(finite) else np.nan,
    }


def guard_diagnostic_values(values, ntot_limit, *, use_ntot_limit=True, multiplier=8.0):
    values = np.asarray(values, dtype=float).copy()
    if use_ntot_limit and np.isfinite(ntot_limit) and ntot_limit > 0:
        values[values > ntot_limit] = np.nan
    values, _ = suppress_isolated_spikes(values, multiplier=multiplier)
    limit = robust_upper_limit(values, multiplier=multiplier)
    if np.isfinite(limit):
        values[values > limit] = np.nan
    return values


def build_scan_health(df, group_key):
    rows = []
    for scan_id, g in df.groupby(group_key):
        scan_rows = g[g["Ntot"] == False].copy()
        if scan_rows.empty:
            continue

        cpc = pd.to_numeric(scan_rows["cpc_count"], errors="coerce")
        flow_rmse, flow_rel_rmse = sheath_flow_relative_rmse(scan_rows)
        polarities = set(scan_rows["polarity"].dropna())
        missing = []
        if "positive" not in polarities:
            missing.append("positive")
        if "negative" not in polarities:
            missing.append("negative")

        rows.append({
            "scan_id": scan_id,
            "time": scan_rows["time"].median(),
            "nan_fraction": float(cpc.isna().mean()),
            "flow_rmse": float(flow_rmse) if np.isfinite(flow_rmse) else np.nan,
            "flow_rel_rmse": float(flow_rel_rmse) if np.isfinite(flow_rel_rmse) else np.nan,
            "missing_polarity": ", ".join(missing) if missing else "none",
        })

    return rows


def build_formation_rate_diagnostics(
    result,
    *,
    growth_min_size_nm,
    growth_max_size_nm,
    ntot_limit,
    method_label,
):
    diagnostics = []
    min_size = float(growth_min_size_nm)
    max_size = min(float(growth_max_size_nm), 10.0)
    if min_size > max_size:
        min_size, max_size = max_size, min_size

    for tr in result:
        if tr.get("kind") != "heatmap":
            continue

        sizes = np.asarray(tr["y"], dtype=float)
        z = np.asarray(tr["Z"], dtype=float)
        times = pd.to_datetime(tr["x"])
        size_mask = np.isfinite(sizes) & (sizes >= min_size) & (sizes <= max_size)
        if z.size == 0 or np.count_nonzero(size_mask) < 2 or len(times) < 3:
            continue

        event_sizes = sizes[size_mask]
        event_z = z[size_mask, :]
        order = np.argsort(event_sizes)
        event_sizes = event_sizes[order]
        event_z = event_z[order, :]
        conc = np.array([
            integrate_number_distribution(event_sizes, col) for col in event_z.T
        ])
        conc = guard_diagnostic_values(conc, ntot_limit, use_ntot_limit=True, multiplier=8.0)
        conc, spike_limit = suppress_isolated_spikes(conc, multiplier=8.0)
        hours = (times - times[0]).total_seconds() / 3600.0
        finite = np.isfinite(hours) & np.isfinite(conc)
        if np.count_nonzero(finite) < 3:
            continue

        rate = np.full(len(conc), np.nan)
        rate[finite] = np.gradient(conc[finite], hours[finite])
        finite_idx = np.flatnonzero(finite)
        dt = np.diff(hours[finite])
        close = np.flatnonzero(dt < 1 / 60)
        if len(close):
            rate[finite_idx[close]] = np.nan
            rate[finite_idx[close + 1]] = np.nan
        rate = guard_diagnostic_values(np.abs(rate), ntot_limit, use_ntot_limit=True, multiplier=6.0) * np.sign(rate)
        baseline_n = max(3, int(np.ceil(np.count_nonzero(finite) * 0.2)))
        baseline = conc[finite][:baseline_n]
        threshold = np.nanmedian(baseline) + 3 * np.nanstd(baseline)
        onset_candidates = np.flatnonzero(finite & (conc > threshold))
        onset_time = times[onset_candidates[0]] if len(onset_candidates) else pd.NaT

        diagnostics.append({
            "label": f"{method_label(tr.get('method', 'gunn woessner mod'))} {tr['polarity']}",
            "time": times,
            "concentration": conc,
            "formation_rate": rate,
            "rate_semantics": "apparent accumulation rate dN/dt; excludes growth flux and losses",
            "onset_time": onset_time,
            "threshold": threshold,
            "spike_limit": spike_limit,
            "size_range": f"{min_size:.1f}-{max_size:.1f} nm",
        })

    return diagnostics


def build_polarity_difference_heatmaps(result):
    heatmaps = [tr for tr in result if tr.get("kind") == "heatmap"]
    by_method = {}
    for tr in heatmaps:
        by_method.setdefault(tr.get("method", "gunn woessner mod"), {})[tr["polarity"]] = tr

    comparisons = []
    for method, pair in by_method.items():
        pos = pair.get("positive")
        neg = pair.get("negative")
        if pos is None or neg is None:
            continue

        pos_z = np.asarray(pos["Z"], dtype=float)
        neg_z = np.asarray(neg["Z"], dtype=float)
        sizes = np.asarray(pos["y"], dtype=float)
        if pos_z.shape != neg_z.shape or len(sizes) == 0:
            continue

        ratio = np.divide(
            pos_z,
            neg_z,
            out=np.full_like(pos_z, np.nan, dtype=float),
            where=neg_z > 0,
        )
        comparisons.append({
            "method": method,
            "x": pos["x"],
            "y": sizes,
            "z": np.clip(ratio, 0, 2),
        })

    return comparisons


def growth_models_from_settings(settings, options, defaults):
    if "growth_models" in settings and isinstance(settings["growth_models"], list):
        return [model for model in settings["growth_models"] if model in options]
    legacy = {
        "weighted centroid": ["Center D50"],
        "peak size": ["Ridge peak"],
        "appearance time": ["Appearance time"],
        "quantile D50": ["Center D50"],
        "sliding window": ["Center D50"],
        "log-size fit": ["Center D50"],
    }
    return list(legacy.get(settings.get("growth_method"), defaults))


def weighted_log_diameter_quantile(sizes, concentration, quantile):
    sizes = np.asarray(sizes, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    usable = np.isfinite(sizes) & (sizes > 0) & np.isfinite(concentration) & (concentration >= 0)
    sizes = sizes[usable]
    concentration = concentration[usable]
    if len(sizes) < 2:
        return np.nan
    order = np.argsort(sizes)
    sizes = sizes[order]
    concentration = concentration[order]
    log_sizes = np.log10(sizes)
    interval_mass = 0.5 * (concentration[:-1] + concentration[1:]) * np.diff(log_sizes)
    cumulative = np.r_[0.0, np.cumsum(interval_mass)]
    if cumulative[-1] <= 0:
        return np.nan
    crossing = np.interp(np.clip(float(quantile), 0.0, 1.0) * cumulative[-1], cumulative, log_sizes)
    return float(10 ** crossing)


def build_growth_rate_diagnostics(
    result,
    *,
    growth_min_size_nm,
    growth_max_size_nm,
    growth_threshold_fraction,
    growth_models,
    method_label,
    growth_max_gap_minutes=90.0,
    growth_min_event_scans=4,
    growth_max_rate_nm_h=15.0,
):
    """Extract and fit several physically distinct tracks through an NPF banana."""
    diagnostics = []
    selected_models = set(growth_models or [])
    minimum_scans = max(3, int(growth_min_event_scans))

    def fit_track(label, polarity, source_method, event_number, model, times, diameters):
        track_times = pd.to_datetime(times)
        track_dp = np.asarray(diameters, dtype=float)
        finite = ~pd.isna(track_times) & np.isfinite(track_dp) & (track_dp > 0)
        track_times = track_times[finite]
        track_dp = track_dp[finite]
        if len(track_dp) < minimum_scans or len(pd.unique(track_times)) < minimum_scans:
            return None
        order = np.argsort(track_times)
        track_times = track_times[order]
        track_dp = track_dp[order]
        hours = np.asarray((track_times - track_times[0]).total_seconds(), dtype=float) / 3600.0
        pair_slopes = []
        for i in range(len(hours) - 1):
            dt = hours[i + 1:] - hours[i]
            usable = dt > 1 / 60
            pair_slopes.extend(((track_dp[i + 1:][usable] - track_dp[i]) / dt[usable]).tolist())
        pair_slopes = np.asarray(pair_slopes, dtype=float)
        pair_slopes = pair_slopes[np.isfinite(pair_slopes)]
        if not len(pair_slopes):
            return None
        slope = float(np.median(pair_slopes))
        intercept = float(np.median(track_dp - slope * hours))
        fitted = intercept + slope * hours
        residual = track_dp - fitted
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((track_dp - np.mean(track_dp)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        duration = float(hours[-1] - hours[0])
        diameter_span = float(np.max(track_dp) - np.min(track_dp))
        if (
            slope <= 0
            or slope > float(growth_max_rate_nm_h)
            or duration < 0.25
            or diameter_span < 2.0
            or not np.isfinite(r2)
            or r2 < 0.3
        ):
            return None
        slope_spread = float(np.percentile(pair_slopes, 90) - np.percentile(pair_slopes, 10))
        if (
            r2 >= 0.8 and len(track_dp) >= 6 and duration >= 1.0
            and slope_spread <= max(2.0, slope)
        ):
            fit_quality = "strong"
        elif r2 >= 0.6 and len(track_dp) >= 5:
            fit_quality = "acceptable"
        else:
            fit_quality = "marginal"
        return {
            "label": f"{label} {polarity}",
            "source_method": source_method,
            "polarity": polarity,
            "event_number": int(event_number),
            "event_id": f"{source_method}:{polarity}:event-{event_number}",
            "model": model,
            "time": track_times,
            "dp": track_dp,
            "fit": fitted,
            "growth_rate": slope,
            "slope_p10": float(np.percentile(pair_slopes, 10)),
            "slope_p90": float(np.percentile(pair_slopes, 90)),
            "r2": r2,
            "rmse_nm": float(np.sqrt(np.mean(residual ** 2))),
            "n_points": int(len(track_dp)),
            "duration_hours": duration,
            "diameter_span_nm": diameter_span,
            "fit_quality": fit_quality,
            "slope_interval_semantics": "10th-90th percentile of pairwise slopes",
        }

    def interpolated_ridge(sizes, concentration):
        index = int(np.nanargmax(concentration))
        if index == 0 or index == len(sizes) - 1:
            return float(sizes[index])
        local_y = concentration[index - 1:index + 2]
        if np.any(~np.isfinite(local_y)) or np.any(local_y <= 0):
            return float(sizes[index])
        local_x = np.log(sizes[index - 1:index + 2])
        a, b, _ = np.polyfit(local_x, np.log(local_y), 2)
        if not np.isfinite(a) or a >= 0:
            return float(sizes[index])
        ridge = np.exp(-b / (2 * a))
        return float(ridge) if sizes[index - 1] <= ridge <= sizes[index + 1] else float(sizes[index])

    def integrated_enhancement(column, log_sizes):
        finite = np.isfinite(column)
        connected = finite[:-1] & finite[1:]
        if np.count_nonzero(connected) < 1 or np.mean(finite) < 0.7:
            return np.nan
        interval = 0.5 * (column[:-1] + column[1:]) * np.diff(log_sizes)
        return float(np.sum(interval[connected]))

    def event_segments(active, missing_activity, times):
        active = np.asarray(active, dtype=bool).copy()
        time_seconds = np.asarray(times.view("int64"), dtype=float) / 1e9
        cadence = np.diff(time_seconds)
        cadence = cadence[np.isfinite(cadence) & (cadence > 0)]
        cadence_limit = 3.0 * np.median(cadence) if len(cadence) else np.inf
        gap_limit = min(max(60.0, float(growth_max_gap_minutes) * 60.0), cadence_limit)
        for index in range(1, len(active) - 1):
            if (
                not active[index]
                and bool(missing_activity[index])
                and active[index - 1]
                and active[index + 1]
                and time_seconds[index + 1] - time_seconds[index - 1] <= gap_limit
            ):
                active[index] = True
        indices = np.flatnonzero(active)
        if not len(indices):
            return []
        split_at = [0]
        for position in range(1, len(indices)):
            if (
                indices[position] != indices[position - 1] + 1
                or time_seconds[indices[position]] - time_seconds[indices[position - 1]] > gap_limit
            ):
                split_at.append(position)
        split_at.append(len(indices))
        return [
            indices[split_at[i]:split_at[i + 1]]
            for i in range(len(split_at) - 1)
            if split_at[i + 1] - split_at[i] >= minimum_scans
        ]

    for tr in result:
        if tr.get("kind") != "heatmap":
            continue

        sizes = np.asarray(tr["y"], dtype=float)
        z = np.asarray(tr["Z"], dtype=float)
        times = pd.DatetimeIndex(pd.to_datetime(tr["x"], errors="coerce"))
        if z.shape != (len(sizes), len(times)) or z.size == 0:
            continue
        valid_times = ~times.isna()
        z = z[:, valid_times]
        times = times[valid_times]
        time_order = np.argsort(times)
        times = times[time_order]
        z = z[:, time_order]

        min_size = float(growth_min_size_nm)
        max_size = float(growth_max_size_nm)
        if min_size > max_size:
            min_size, max_size = max_size, min_size
        size_mask = np.isfinite(sizes) & (sizes >= min_size) & (sizes <= max_size)
        if np.count_nonzero(size_mask) < 2:
            continue

        event_sizes = sizes[size_mask]
        event_z = z[size_mask, :]
        order = np.argsort(event_sizes)
        event_sizes = event_sizes[order]
        event_z = event_z[order, :]
        threshold_fraction = np.clip(float(growth_threshold_fraction), 0.0, 1.0)

        label = method_label(tr.get('method', 'gunn woessner mod'))
        source_method = tr.get('method', 'gunn woessner mod')
        polarity = tr['polarity']
        background = np.nanpercentile(event_z, 20, axis=1)
        enhancement = np.maximum(event_z - background[:, None], 0.0)
        log_sizes = np.log10(event_sizes)
        integrated = np.asarray([
            integrated_enhancement(col, log_sizes) for col in enhancement.T
        ])
        finite_activity = integrated[np.isfinite(integrated)]
        if len(finite_activity) < minimum_scans:
            continue
        temporal_difference = np.diff(event_z, axis=1)
        difference_center = np.nanmedian(temporal_difference, axis=1)
        # The lower difference quartile remains representative when an event
        # occupies much of the interval; 0.4506 is the Gaussian conversion for
        # the 25th percentile of |x1-x2| to one-sample standard deviation.
        detection_noise = (
            np.nanpercentile(
                np.abs(temporal_difference - difference_center[:, None]), 25, axis=1
            )
            / 0.4506
        )
        detection_threshold_snr = 4.0
        coherent_signal = np.zeros(len(times), dtype=bool)
        for time_index in range(len(times)):
            significant = (
                np.isfinite(enhancement[:, time_index])
                & np.isfinite(detection_noise)
                & (
                    enhancement[:, time_index]
                    > detection_threshold_snr * detection_noise
                )
            )
            run_length = 0
            for enabled in significant:
                run_length = run_length + 1 if enabled else 0
                if run_length >= 3:
                    coherent_signal[time_index] = True
                    break
        active_activity = np.isfinite(integrated) & coherent_signal
        segments = event_segments(
            active_activity, ~np.isfinite(integrated), times
        )

        background_scans = np.isfinite(integrated) & ~active_activity
        if np.count_nonzero(background_scans) >= 3:
            background_residual = event_z[:, background_scans] - background[:, None]
        else:
            cutoff = np.nanpercentile(np.abs(event_z - background[:, None]), 30, axis=1)
            background_residual = np.where(
                np.abs(event_z - background[:, None]) <= cutoff[:, None],
                event_z - background[:, None],
                np.nan,
            )
        size_noise = 1.4826 * np.nanmedian(
            np.abs(background_residual), axis=1
        )
        for event_number, segment in enumerate(segments, start=1):
            model_tracks = {
                "Lower edge D25": ([], []),
                "Center D50": ([], []),
                "Upper edge D75": ([], []),
                "Ridge peak": ([], []),
            }
            previous_ridge = None
            previous_time = None
            ridge_history = []
            diameter_resolution = float(np.nanmedian(np.diff(event_sizes)))
            for time_index in segment:
                col = enhancement[:, time_index]
                finite = np.isfinite(col) & (col > 0)
                if np.count_nonzero(finite) < 3:
                    continue
                component_mask = finite & (col >= threshold_fraction * np.nanmax(col[finite]))
                components = []
                component_start = None
                for size_index, enabled in enumerate(np.r_[component_mask, False]):
                    if enabled and component_start is None:
                        component_start = size_index
                    elif not enabled and component_start is not None:
                        left, right = component_start, size_index - 1
                        if right - left + 1 >= 3:
                            peak_index = left + int(np.nanargmax(col[left:right + 1]))
                            ridge = interpolated_ridge(
                                event_sizes[max(0, peak_index - 1):min(len(event_sizes), peak_index + 2)],
                                col[max(0, peak_index - 1):min(len(col), peak_index + 2)],
                            )
                            components.append((ridge, peak_index, left, right))
                        component_start = None
                if not components:
                    continue
                if previous_ridge is None:
                    ridge, peak_index, left, right = min(components, key=lambda item: item[0])
                else:
                    elapsed_hours = max(
                        1 / 60,
                        (times[time_index] - previous_time).total_seconds() / 3600.0,
                    )
                    expected_ridge = previous_ridge
                    if len(ridge_history) >= 2:
                        previous_rate = (
                            ridge_history[-1][1] - ridge_history[-2][1]
                        ) / max(
                            1 / 60,
                            (ridge_history[-1][0] - ridge_history[-2][0]).total_seconds() / 3600.0,
                        )
                        expected_ridge += previous_rate * elapsed_hours
                    ridge, peak_index, left, right = min(
                        components, key=lambda item: abs(item[0] - expected_ridge)
                    )
                    grid_tolerance = 2.0 * diameter_resolution
                    maximum_step = (
                        float(growth_max_rate_nm_h) * elapsed_hours + grid_tolerance
                    )
                    prediction_tolerance = (
                        0.5 * float(growth_max_rate_nm_h) * elapsed_hours
                        + grid_tolerance
                    )
                    if (
                        abs(ridge - previous_ridge) > maximum_step
                        or abs(ridge - expected_ridge) > prediction_tolerance
                    ):
                        continue
                previous_ridge = ridge
                previous_time = times[time_index]
                ridge_history.append((previous_time, previous_ridge))
                active_sizes = event_sizes[left:right + 1]
                active_concentration = col[left:right + 1]
                if left > 0 and right < len(event_sizes) - 1:
                    for model, quantile in (
                        ("Lower edge D25", 0.25),
                        ("Center D50", 0.50),
                        ("Upper edge D75", 0.75),
                    ):
                        model_tracks[model][0].append(times[time_index])
                        model_tracks[model][1].append(
                            weighted_log_diameter_quantile(
                                active_sizes, active_concentration, quantile
                            )
                        )
                if 0 < peak_index < len(event_sizes) - 1:
                    model_tracks["Ridge peak"][0].append(times[time_index])
                    model_tracks["Ridge peak"][1].append(ridge)

            for model, (track_times, track_sizes) in model_tracks.items():
                if model not in selected_models:
                    continue
                diagnostic = fit_track(
                    label, polarity, source_method, event_number,
                    model, track_times, track_sizes,
                )
                if diagnostic is not None:
                    diagnostic.update({
                        "event_start": times[segment[0]],
                        "event_end": times[segment[-1]],
                        "event_threshold_snr": detection_threshold_snr,
                        "background_method": "per-size 20th percentile",
                        "background_scan_fraction": float(np.mean(background_scans)),
                        "background_quality": (
                            "adequate" if np.count_nonzero(background_scans) >= max(3, int(0.2 * len(times)))
                            else "limited"
                        ),
                    })
                    if diagnostic["background_quality"] == "limited":
                        diagnostic["fit_quality"] = "marginal"
                    diagnostics.append(diagnostic)

            if "Appearance time" in selected_models:
                appearance_sizes = []
                appearance_times = []
                previous_time = None
                appearance_window = np.arange(
                    max(0, int(segment[0]) - 2),
                    min(len(times), int(segment[-1]) + 3),
                )
                for size_index, size_nm in enumerate(event_sizes):
                    row = enhancement[size_index, appearance_window]
                    finite = np.isfinite(row)
                    if np.count_nonzero(finite) < 3:
                        continue
                    finite_positions = np.flatnonzero(finite)
                    peak_position = int(np.nanargmax(np.where(finite, row, np.nan)))
                    if peak_position in (finite_positions[0], finite_positions[-1]):
                        continue
                    peak = float(np.nanmax(row))
                    threshold = max(
                        threshold_fraction * peak,
                        3.0 * float(size_noise[size_index]),
                    )
                    crossings = np.flatnonzero(
                        finite[1:] & finite[:-1]
                        & (row[1:] >= threshold) & (row[:-1] < threshold)
                    ) + 1
                    if not len(crossings):
                        continue
                    crossing = int(crossings[0])
                    appearance_index = appearance_window[crossing]
                    # Crossings in the first event scan are left-censored by event onset.
                    if appearance_index <= segment[0] or appearance_index > segment[-1]:
                        continue
                    previous_index = appearance_window[crossing - 1]
                    rise = row[crossing] - row[crossing - 1]
                    fraction = (
                        float(np.clip((threshold - row[crossing - 1]) / rise, 0.0, 1.0))
                        if rise > 0 else 1.0
                    )
                    appearance_time = times[previous_index] + fraction * (
                        times[appearance_index] - times[previous_index]
                    )
                    appearance_sizes.append(size_nm)
                    appearance_times.append(appearance_time)
                diagnostic = fit_track(
                    label, polarity, source_method, event_number,
                    "Appearance time", appearance_times, appearance_sizes,
                )
                if diagnostic is not None:
                    diagnostic.update({
                        "event_start": times[segment[0]],
                        "event_end": times[segment[-1]],
                        "event_threshold_snr": detection_threshold_snr,
                        "background_method": "per-size 20th percentile",
                        "background_scan_fraction": float(np.mean(background_scans)),
                        "background_quality": (
                            "adequate" if np.count_nonzero(background_scans) >= max(3, int(0.2 * len(times)))
                            else "limited"
                        ),
                    })
                    if diagnostic["background_quality"] == "limited":
                        diagnostic["fit_quality"] = "marginal"
                    diagnostics.append(diagnostic)

    return diagnostics


def integrate_distribution(size_nm, conc):
    return integrate_number_distribution(size_nm, conc)


def peak_diameter(size_nm, conc):
    size_nm = np.asarray(size_nm, dtype=float)
    conc = np.asarray(conc, dtype=float)
    valid = np.isfinite(size_nm) & np.isfinite(conc) & (conc > 0)
    if not np.any(valid):
        return np.nan
    valid_idx = np.flatnonzero(valid)
    return float(size_nm[valid_idx[int(np.nanargmax(conc[valid]))]])


def build_last_hours_smear_difference(result, smear, *, min_size_nm, peak_min_size_nm, ntot_limit, hours=3):
    heatmaps = [tr for tr in result if tr.get("kind") == "heatmap"]
    if not heatmaps or smear.empty:
        return {"ratios": [], "matches": pd.DataFrame(), "summary": pd.DataFrame()}

    all_times = pd.to_datetime([t for tr in heatmaps for t in tr.get("x", [])])
    if len(all_times) == 0:
        return {"ratios": [], "matches": pd.DataFrame(), "summary": pd.DataFrame()}

    end = all_times.max()
    start = end - pd.Timedelta(hours=hours)
    smear_times = pd.to_datetime(sorted(smear["time"].dropna().unique()))
    ratios = []
    shapes = []
    matches = []

    for tr in heatmaps:
        method = tr.get("method", "gunn woessner mod")
        polarity = tr.get("polarity", "unknown")
        sizes = np.asarray(tr["y"], dtype=float)
        z = np.asarray(tr["Z"], dtype=float)
        size_mask = np.isfinite(sizes) & (sizes >= float(min_size_nm))
        if np.count_nonzero(size_mask) < 2:
            continue

        ratio_cols = []
        our_cols = []
        smear_cols = []
        ratio_times = []
        for t, our_col in zip(pd.to_datetime(tr["x"]), z.T):
            if t < start or t > end or len(smear_times) == 0:
                continue
            deltas = np.abs(smear_times - t)
            if len(deltas) == 0 or deltas.min() > pd.Timedelta(minutes=15):
                continue
            smear_time = smear_times[np.argmin(deltas)]
            smear_scan = smear[smear["time"] == smear_time].sort_values("size_nm")
            smear_sizes = smear_scan["size_nm"].to_numpy(dtype=float)
            smear_conc = smear_scan["smear_conc"].to_numpy(dtype=float)
            valid_smear = np.isfinite(smear_sizes) & np.isfinite(smear_conc) & (smear_conc > 0)
            if np.count_nonzero(valid_smear) < 2:
                continue

            event_sizes = sizes[size_mask]
            our = np.asarray(our_col, dtype=float)[size_mask]
            smear_interp = np.interp(
                event_sizes,
                smear_sizes[valid_smear],
                smear_conc[valid_smear],
                left=np.nan,
                right=np.nan,
            )
            ratio = np.divide(
                our,
                smear_interp,
                out=np.full(len(event_sizes), np.nan),
                where=smear_interp > 0,
            )
            ratio_cols.append(ratio)
            our_cols.append(our)
            smear_cols.append(smear_interp)
            ratio_times.append(t)

            our_ntot = integrate_distribution(event_sizes, our)
            smear_ntot = integrate_distribution(event_sizes, smear_interp)
            if np.isfinite(ntot_limit) and ntot_limit > 0:
                if our_ntot > ntot_limit:
                    our_ntot = np.nan
                if smear_ntot > ntot_limit:
                    smear_ntot = np.nan
            peak_mask = event_sizes >= float(peak_min_size_nm)
            our_peak = peak_diameter(event_sizes[peak_mask], our[peak_mask])
            smear_peak = peak_diameter(event_sizes[peak_mask], smear_interp[peak_mask])
            peak_shift_pct = np.nan
            if np.isfinite(our_peak) and np.isfinite(smear_peak) and smear_peak > 0:
                peak_shift_pct = 100 * (our_peak / smear_peak - 1)
            matches.append({
                "method": method,
                "polarity": polarity,
                "time": t,
                "smear_time": smear_time,
                "time_delta_min": abs((t - smear_time).total_seconds()) / 60,
                "our_ntot": our_ntot,
                "smear_ntot": smear_ntot,
                "ntot_ratio": our_ntot / smear_ntot if np.isfinite(smear_ntot) and smear_ntot > 0 else np.nan,
                "our_peak_nm": our_peak,
                "smear_peak_nm": smear_peak,
                "peak_shift_pct": peak_shift_pct,
            })

        if ratio_cols:
            ratio_arr = np.vstack(ratio_cols)
            our_arr = np.vstack(our_cols)
            smear_arr = np.vstack(smear_cols)
            ratio_median = np.full(len(event_sizes), np.nan)
            valid_rows = np.any(np.isfinite(ratio_arr), axis=0)
            if np.any(valid_rows):
                ratio_median[valid_rows] = np.nanmedian(ratio_arr[:, valid_rows], axis=0)
            our_median = np.full(len(event_sizes), np.nan)
            smear_median = np.full(len(event_sizes), np.nan)
            valid_our = np.any(np.isfinite(our_arr), axis=0)
            valid_smear = np.any(np.isfinite(smear_arr), axis=0)
            if np.any(valid_our):
                our_median[valid_our] = np.nanmedian(our_arr[:, valid_our], axis=0)
            if np.any(valid_smear):
                smear_median[valid_smear] = np.nanmedian(smear_arr[:, valid_smear], axis=0)
            ratios.append({
                "method": method,
                "polarity": polarity,
                "size_nm": event_sizes,
                "ratio_median": np.clip(ratio_median, 0, 5),
                "n_matches": len(ratio_times),
            })
            shapes.append({
                "method": method,
                "polarity": polarity,
                "size_nm": event_sizes,
                "our_median": our_median,
                "smear_median": smear_median,
                "n_matches": len(ratio_times),
            })

    matches = pd.DataFrame(matches)
    if matches.empty:
        return {"ratios": ratios, "shapes": shapes, "matches": matches, "summary": pd.DataFrame()}

    summary = (
        matches.groupby(["method", "polarity"])
        .agg(
            n=("ntot_ratio", "size"),
            median_ntot_ratio=("ntot_ratio", "median"),
            median_peak_shift_pct=("peak_shift_pct", "median"),
            median_time_delta_min=("time_delta_min", "median"),
        )
        .reset_index()
    )
    return {"ratios": ratios, "shapes": shapes, "matches": matches, "summary": summary}
