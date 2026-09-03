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


def integrate_number_distribution(size_nm, concentration, part_columns=None):
    size_nm = np.asarray(size_nm, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    valid = np.isfinite(size_nm) & (size_nm > 0) & np.isfinite(concentration)
    if valid.sum() < 2:
        return np.nan
    order = np.argsort(size_nm)
    size_nm = size_nm[order]
    concentration = concentration[order]
    valid = valid[order]
    if part_columns is None:
        connected = valid[:-1] & valid[1:]
    else:
        parts = np.vstack([np.asarray(column, dtype=float)[order] for column in part_columns])
        connected = valid[:-1] & valid[1:] & np.any(
            np.isfinite(parts[:, :-1]) & np.isfinite(parts[:, 1:]), axis=0
        )
    widths = np.diff(np.log10(size_nm))
    trapezoids = 0.5 * (concentration[:-1] + concentration[1:]) * widths
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
            np.trapezoid(col, np.log10(event_sizes))
            if np.count_nonzero(np.isfinite(col)) >= 2 else np.nan
            for col in event_z.T
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


def build_growth_rate_diagnostics(
    result,
    *,
    growth_min_size_nm,
    growth_max_size_nm,
    growth_threshold_fraction,
    growth_method,
    method_label,
):
    diagnostics = []

    def fit_track(label, polarity, method, times, sizes):
        mode_times = pd.to_datetime(times)
        mode_sizes = np.asarray(sizes, dtype=float)
        hours = (mode_times - mode_times[0]).total_seconds() / 3600.0
        finite = np.isfinite(hours) & np.isfinite(mode_sizes) & (mode_sizes > 0)
        if np.count_nonzero(finite) < 3 or np.nanmax(hours[finite]) <= np.nanmin(hours[finite]):
            return None
        fit_y = np.log(mode_sizes[finite]) if method == "log-size fit" else mode_sizes[finite]
        slope, intercept = np.polyfit(hours[finite], fit_y, 1)
        fit = np.exp(intercept + slope * hours[finite]) if method == "log-size fit" else intercept + slope * hours[finite]
        residual = mode_sizes[finite] - fit
        ss_res = np.nansum(residual ** 2)
        ss_tot = np.nansum((mode_sizes[finite] - np.nanmean(mode_sizes[finite])) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if method == "log-size fit":
            growth_rate = float(np.nanmedian(slope * fit))
        else:
            growth_rate = float(slope)
        return {
            "label": f"{label} {polarity}",
            "time": mode_times[finite],
            "dp": mode_sizes[finite],
            "fit": fit,
            "growth_rate": growth_rate,
            "r2": r2,
            "n_points": int(np.count_nonzero(finite)),
        }

    for tr in result:
        if tr.get("kind") != "heatmap":
            continue

        sizes = np.asarray(tr["y"], dtype=float)
        z = np.asarray(tr["Z"], dtype=float)
        times = pd.to_datetime(tr["x"])
        if z.size == 0 or len(sizes) == 0 or len(times) == 0:
            continue

        min_size = float(growth_min_size_nm)
        max_size = float(growth_max_size_nm)
        if min_size > max_size:
            min_size, max_size = max_size, min_size
        size_mask = np.isfinite(sizes) & (sizes >= min_size) & (sizes <= max_size)
        if np.count_nonzero(size_mask) < 2:
            continue

        event_sizes = sizes[size_mask]
        event_z = z[size_mask, :]
        threshold_fraction = np.clip(float(growth_threshold_fraction), 0.0, 1.0)

        label = method_label(tr.get('method', 'gunn woessner mod'))
        polarity = tr['polarity']

        if growth_method == "appearance time":
            mode_sizes = []
            mode_times = []
            baseline_n = max(1, int(np.ceil(event_z.shape[1] * 0.2)))
            for size_nm, row in zip(event_sizes, event_z):
                row = np.asarray(row, dtype=float)
                finite = np.isfinite(row) & (row > 0)
                if np.count_nonzero(finite) < 3:
                    continue
                baseline = np.nanmedian(row[:baseline_n])
                peak = np.nanmax(row)
                threshold = baseline + threshold_fraction * (peak - baseline)
                hits = np.flatnonzero(finite & (row >= threshold))
                if len(hits) and np.isfinite(threshold) and peak > baseline:
                    mode_sizes.append(size_nm)
                    mode_times.append(times[hits[0]])

            if len(mode_sizes) < 3:
                continue
            diagnostic = fit_track(label, polarity, growth_method, mode_times, mode_sizes)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
            continue

        mode_sizes = []
        mode_times = []

        for t, col in zip(times, event_z.T):
            col = np.asarray(col, dtype=float)
            finite = np.isfinite(col) & (col > 0)
            if not np.any(finite):
                continue
            peak = np.nanmax(col[finite])
            if peak <= 0:
                continue

            active = finite & (col >= threshold_fraction * peak)
            if not np.any(active):
                continue

            if growth_method == "peak size":
                dp = event_sizes[int(np.nanargmax(col))]
            elif growth_method == "quantile D50":
                order = np.argsort(event_sizes[active])
                active_sizes = event_sizes[active][order]
                active_weights = col[active][order]
                cumulative = np.cumsum(active_weights)
                dp = np.interp(0.5 * cumulative[-1], cumulative, active_sizes)
            else:
                weights = col[active]
                dp = np.average(event_sizes[active], weights=weights)

            if np.isfinite(dp):
                mode_times.append(t)
                mode_sizes.append(dp)

        if len(mode_sizes) < 3:
            continue
        if growth_method == "sliding window":
            mode_times = pd.to_datetime(mode_times)
            mode_sizes = np.asarray(mode_sizes, dtype=float)
            hours = (mode_times - mode_times[0]).total_seconds() / 3600.0
            finite = np.isfinite(hours) & np.isfinite(mode_sizes)
            local_fit = np.full(len(mode_sizes), np.nan)
            local_rates = []
            for i in range(len(mode_sizes)):
                lo = max(0, i - 2)
                hi = min(len(mode_sizes), i + 3)
                local = finite[lo:hi]
                if np.count_nonzero(local) < 3:
                    continue
                slope, intercept = np.polyfit(hours[lo:hi][local], mode_sizes[lo:hi][local], 1)
                local_fit[i] = intercept + slope * hours[i]
                local_rates.append(slope)
            if np.count_nonzero(np.isfinite(local_fit)) < 3:
                continue
            diagnostics.append({
                "label": f"{label} {polarity}",
                "time": mode_times[np.isfinite(local_fit)],
                "dp": mode_sizes[np.isfinite(local_fit)],
                "fit": local_fit[np.isfinite(local_fit)],
                "growth_rate": float(np.nanmedian(local_rates)) if local_rates else np.nan,
                "r2": np.nan,
                "n_points": int(np.count_nonzero(np.isfinite(local_fit))),
            })
            continue

        diagnostic = fit_track(label, polarity, growth_method, mode_times, mode_sizes)
        if diagnostic is not None:
            diagnostics.append(diagnostic)

    return diagnostics


def integrate_distribution(size_nm, conc):
    size_nm = np.asarray(size_nm, dtype=float)
    conc = np.asarray(conc, dtype=float)
    valid = np.isfinite(size_nm) & np.isfinite(conc) & (size_nm > 0)
    if np.count_nonzero(valid) < 2:
        return np.nan
    order = np.argsort(size_nm[valid])
    return float(np.trapezoid(conc[valid][order], np.log10(size_nm[valid][order])))


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
