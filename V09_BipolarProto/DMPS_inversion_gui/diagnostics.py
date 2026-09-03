import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks


GAS_CONSTANT = 8.314462618
BOLTZMANN_CONSTANT = 1.380649e-23


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
    available = (
        np.isfinite(size_nm) & (size_nm > 0)
        & np.isfinite(concentration) & (concentration >= 0)
    )
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
    valid = (
        np.isfinite(size_nm) & (size_nm > 0)
        & np.isfinite(concentration) & (concentration >= 0)
    )
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


def normalized_log_gaussian(log_dp, area, mean, sigma):
    log_dp = np.asarray(log_dp, dtype=float)
    if area < 0 or sigma <= 0:
        return np.full(log_dp.shape, np.nan)
    return area / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(
        -0.5 * ((log_dp - mean) / sigma) ** 2
    )


def fit_lognormal_modes(size_nm, concentration, number_of_modes, part_columns=None):
    """Fit deterministic constrained Gaussian modes in log10 diameter space."""
    size_nm = np.asarray(size_nm, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    modes_requested = int(number_of_modes)
    support_widths = distribution_support_widths(size_nm, part_columns)
    valid = (
        np.isfinite(size_nm) & (size_nm > 0)
        & np.isfinite(concentration) & (concentration >= 0)
        & np.isfinite(support_widths) & (support_widths > 0)
    )
    if modes_requested < 1 or modes_requested > 3 or np.count_nonzero(valid) < max(7, 3 * modes_requested):
        return {"status": "failed", "reason": "insufficient data or unsupported mode count"}
    order = np.argsort(size_nm[valid])
    x = np.log10(size_nm[valid][order])
    y = concentration[valid][order]
    if np.any(np.diff(x) <= 0) or np.nanmax(y) <= 0:
        return {"status": "failed", "reason": "non-positive or duplicate-bin distribution"}

    smooth = np.convolve(y, np.ones(3) / 3.0, mode="same")
    peak_indices, properties = find_peaks(smooth, prominence=0.03 * np.nanmax(smooth))
    ranked = peak_indices[np.argsort(properties.get("prominences", np.array([])))[::-1]]
    peak_means = list(x[np.sort(ranked[:modes_requested])])
    fitted_support_widths = support_widths[valid][order]
    weights = np.maximum(y, 0) * fitted_support_widths
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return {"status": "failed", "reason": "zero integrated concentration"}
    quantiles = (np.arange(modes_requested) + 0.5) / modes_requested
    quantile_means = list(np.interp(quantiles * cumulative[-1], cumulative, x))
    if len(peak_means) < modes_requested:
        peak_means = quantile_means

    total_area = float(np.sum(y * fitted_support_widths))
    log_span = float(x[-1] - x[0])
    spacing = float(np.nanmedian(np.diff(x)))
    sigma_min = max(0.01, 0.5 * spacing)
    sigma_max = max(0.08, log_span)
    lower = []
    upper = []
    for _ in range(modes_requested):
        lower.extend([0.0, x[0], sigma_min])
        upper.extend([max(2 * total_area, 1.0), x[-1], sigma_max])

    scale = np.sqrt(np.maximum(y, max(1.0, 0.01 * np.nanmax(y))))

    def model(parameters):
        fitted = np.zeros_like(x)
        for index in range(modes_requested):
            area, mean, sigma = parameters[3 * index:3 * index + 3]
            fitted += normalized_log_gaussian(x, area, mean, sigma)
        return fitted

    mean_starts = [
        peak_means,
        quantile_means,
        list(np.linspace(x[0], x[-1], modes_requested + 2)[1:-1]),
    ]
    fits = []
    for means in mean_starts:
        initial = []
        for mean in means:
            initial.extend([
                max(total_area / modes_requested, 1e-12), mean,
                max(0.04, log_span / (5 * modes_requested)),
            ])
        candidate = least_squares(
            lambda parameters: (model(parameters) - y) / scale,
            initial,
            bounds=(lower, upper),
            max_nfev=20000,
            x_scale="jac",
        )
        if candidate.success and np.all(np.isfinite(candidate.x)):
            fits.append(candidate)
    if not fits:
        return {"status": "failed", "reason": "all deterministic optimizer starts failed"}
    fit = min(fits, key=lambda candidate: candidate.cost)
    fitted = model(fit.x)
    residual = y - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    components = []
    for index in range(modes_requested):
        area, mean, sigma = fit.x[3 * index:3 * index + 3]
        components.append({
            "component": index + 1,
            "area_cm3": float(area),
            "mean_log10_nm": float(mean),
            "mode_diameter_nm": float(10 ** mean),
            "sigma_log10": float(sigma),
            "geometric_std": float(10 ** sigma),
            "fit_range_area_cm3": float(
                np.sum(
                    normalized_log_gaussian(x, area, mean, sigma)
                    * fitted_support_widths
                )
            ),
        })
    components.sort(key=lambda item: item["mode_diameter_nm"])
    for index, component in enumerate(components, start=1):
        component["component"] = index
    dense_log_dp = np.linspace(x[0], x[-1], 400)
    dense_components = [
        normalized_log_gaussian(
            dense_log_dp, component["area_cm3"],
            component["mean_log10_nm"], component["sigma_log10"],
        )
        for component in components
    ]
    n = len(y)
    k = 3 * modes_requested
    weighted_ssr = float(np.sum(((fitted - y) / scale) ** 2))
    bic = n * np.log(max(weighted_ssr / n, np.finfo(float).tiny)) + k * np.log(n)
    area_fractions = np.asarray([
        component["fit_range_area_cm3"] for component in components
    ]) / max(total_area, 1e-12)
    centers = np.asarray([component["mean_log10_nm"] for component in components])
    sigmas = np.asarray([component["sigma_log10"] for component in components])
    separated = bool(
        len(components) == 1
        or np.all(np.diff(centers) >= np.maximum(spacing, 0.5 * np.minimum(sigmas[:-1], sigmas[1:])))
    )
    return {
        "status": "ok",
        "reason": "",
        "components": components,
        "diameter_nm": size_nm[valid][order],
        "observed": y,
        "fitted": fitted,
        "curve_diameter_nm": 10 ** dense_log_dp,
        "curve_components": dense_components,
        "curve_total": np.sum(dense_components, axis=0),
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bic": float(bic),
        "number_of_modes": modes_requested,
        "minimum_area_fraction": float(np.min(area_fractions)),
        "modes_separated": separated,
    }


def select_lognormal_mode_fit(size_nm, concentration, maximum_modes=3, part_columns=None):
    fits = [
        fit_lognormal_modes(size_nm, concentration, count, part_columns)
        for count in range(1, int(maximum_modes) + 1)
    ]
    accepted = [
        fit for fit in fits
        if fit.get("status") == "ok"
        and fit.get("minimum_area_fraction", 0) >= 0.05
        and fit.get("modes_separated", False)
    ]
    if not accepted:
        return fits[0]
    best = min(accepted, key=lambda item: item["number_of_modes"])
    for candidate in sorted(accepted, key=lambda item: item["number_of_modes"]):
        if candidate["bic"] < best["bic"] - 30.0:
            best = candidate
    return best


def distribution_moments(size_nm, concentration, part_columns=None):
    size_nm = np.asarray(size_nm, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    widths = distribution_support_widths(size_nm, part_columns)
    valid = (
        np.isfinite(size_nm) & (size_nm > 0) & np.isfinite(concentration)
        & (concentration >= 0) & np.isfinite(widths) & (widths > 0)
    )
    if np.count_nonzero(valid) < 2:
        return None
    dp = size_nm[valid]
    number_bins = concentration[valid] * widths[valid]
    number = float(np.sum(number_bins))
    if number <= 0:
        return None
    log_dp = np.log10(dp)
    geometric_mean = float(10 ** (np.sum(number_bins * log_dp) / number))
    geometric_std = float(10 ** np.sqrt(np.sum(number_bins * (log_dp - np.log10(geometric_mean)) ** 2) / number))
    return {
        "number_cm3": number,
        "number_mean_nm": float(np.sum(number_bins * dp) / number),
        "geometric_mean_nm": geometric_mean,
        "geometric_std": geometric_std,
        "surface_um2_cm3": float(np.pi * np.sum(number_bins * (dp / 1000.0) ** 2)),
        "volume_um3_cm3": float(np.pi / 6.0 * np.sum(number_bins * (dp / 1000.0) ** 3)),
        "bin_coverage": distribution_bin_coverage(size_nm, concentration, part_columns),
        "diameter_min_nm": float(np.min(dp)),
        "diameter_max_nm": float(np.max(dp)),
    }


def sulfuric_acid_condensation_sink(
    size_nm, concentration, temperature_k=293.15, pressure_pa=101325.0,
    part_columns=None,
):
    size_nm = np.asarray(size_nm, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    if not np.isfinite(temperature_k) or temperature_k <= 0 or not np.isfinite(pressure_pa) or pressure_pa <= 0:
        return np.nan
    widths = distribution_support_widths(size_nm, part_columns)
    valid = (
        np.isfinite(size_nm) & (size_nm > 0) & np.isfinite(concentration)
        & (concentration >= 0) & np.isfinite(widths) & (widths > 0)
    )
    if np.count_nonzero(valid) < 2:
        return np.nan
    dp_m = size_nm[valid] * 1e-9
    diffusivity = (
        1.013e-2 * float(temperature_k) ** 1.75
        * np.sqrt(1 / 98.08 + 1 / 28.965)
        / (float(pressure_pa) * (51.96 ** (1 / 3) + 19.7 ** (1 / 3)) ** 2)
    )
    molecular_speed = np.sqrt(
        8 * GAS_CONSTANT * float(temperature_k) / (np.pi * 98.08e-3)
    )
    vapor_mean_free_path = 3 * diffusivity / molecular_speed
    knudsen = 2 * vapor_mean_free_path / dp_m
    beta = (1 + knudsen) / (1 + 1.677 * knudsen + 1.333 * knudsen ** 2)
    number_bins_m3 = concentration[valid] * widths[valid] * 1e6
    return float(2 * np.pi * diffusivity * np.sum(beta * dp_m * number_bins_m3))


def brownian_coagulation_kernel(target_nm, collector_nm, temperature_k=293.15, pressure_pa=101325.0, density_kg_m3=1000.0):
    collectors = np.asarray(np.atleast_1d(collector_nm), dtype=float)
    if (
        not np.isfinite(target_nm) or target_nm <= 0
        or not np.isfinite(temperature_k) or temperature_k <= 0
        or not np.isfinite(pressure_pa) or pressure_pa <= 0
        or not np.isfinite(density_kg_m3) or density_kg_m3 <= 0
        or np.any(~np.isfinite(collectors)) or np.any(collectors <= 0)
    ):
        invalid = np.full(collectors.shape, np.nan)
        return float(invalid[0]) if np.ndim(collector_nm) == 0 else invalid
    diameter = np.asarray([target_nm, *collectors], dtype=float) * 1e-9
    temperature = float(temperature_k)
    pressure = float(pressure_pa)
    viscosity = 18.203e-6 * ((293.15 + 110.4) / (temperature + 110.4)) * (temperature / 293.15) ** 1.5
    air_mean_free_path = viscosity / pressure * np.sqrt(
        np.pi * GAS_CONSTANT * temperature / (2 * 0.02897)
    )
    slip = 1 + 2 * air_mean_free_path / diameter * (
        1.246 + 0.420 * np.exp(-0.87 * diameter / (2 * air_mean_free_path))
    )
    diffusivity = BOLTZMANN_CONSTANT * temperature * slip / (3 * np.pi * viscosity * diameter)
    mass = float(density_kg_m3) * np.pi * diameter ** 3 / 6
    thermal_speed = np.sqrt(8 * BOLTZMANN_CONSTANT * temperature / (np.pi * mass))
    particle_path = 8 * diffusivity / (np.pi * thermal_speed)
    g = ((diameter + particle_path) ** 3 - (diameter ** 2 + particle_path ** 2) ** 1.5) / (
        3 * diameter * particle_path
    ) - diameter
    d1 = diameter[0]
    d2 = diameter[1:]
    dsum = diffusivity[0] + diffusivity[1:]
    diameter_sum = d1 + d2
    denominator = diameter_sum / (diameter_sum + 2 * np.sqrt(g[0] ** 2 + g[1:] ** 2)) + (
        8 * dsum / (np.sqrt(thermal_speed[0] ** 2 + thermal_speed[1:] ** 2) * diameter_sum)
    )
    kernel = 2 * np.pi * dsum * diameter_sum / denominator
    return float(kernel[0]) if np.ndim(collector_nm) == 0 else kernel


def brownian_coagulation_sink(
    size_nm, concentration, target_nm, temperature_k=293.15,
    pressure_pa=101325.0, density_kg_m3=1000.0, part_columns=None,
):
    size_nm = np.asarray(size_nm, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    if (
        not np.isfinite(target_nm) or target_nm <= 0
        or not np.isfinite(temperature_k) or temperature_k <= 0
        or not np.isfinite(pressure_pa) or pressure_pa <= 0
        or not np.isfinite(density_kg_m3) or density_kg_m3 <= 0
    ):
        return np.nan
    widths = distribution_support_widths(size_nm, part_columns)
    valid = (
        np.isfinite(size_nm) & (size_nm >= float(target_nm))
        & np.isfinite(concentration) & (concentration >= 0)
        & np.isfinite(widths) & (widths > 0)
    )
    if np.count_nonzero(valid) < 1:
        return np.nan
    kernel = brownian_coagulation_kernel(
        target_nm, size_nm[valid], temperature_k, pressure_pa, density_kg_m3
    )
    return float(np.sum(kernel * concentration[valid] * widths[valid] * 1e6))


def build_aerosol_property_diagnostics(
    result,
    *,
    temperature_k=293.15,
    pressure_pa=101325.0,
    coagulation_targets_nm=(3.0, 10.0),
    particle_density_kg_m3=1000.0,
):
    diagnostics = []
    for trace in result:
        if trace.get("kind") != "heatmap":
            continue
        sizes = np.asarray(trace["y"], dtype=float)
        z = np.asarray(trace["Z"], dtype=float)
        times = pd.to_datetime(trace["x"])
        rows = []
        support_by_scan = trace.get("part_columns", [None] * len(times))
        for scan_index, (time, column) in enumerate(zip(times, z.T)):
            part_columns = (
                support_by_scan[scan_index]
                if scan_index < len(support_by_scan) else None
            )
            moments = distribution_moments(sizes, column, part_columns)
            if moments is None:
                moments = {
                    "number_cm3": np.nan,
                    "number_mean_nm": np.nan,
                    "geometric_mean_nm": np.nan,
                    "geometric_std": np.nan,
                    "surface_um2_cm3": np.nan,
                    "volume_um3_cm3": np.nan,
                    "bin_coverage": distribution_bin_coverage(
                        sizes, column, part_columns
                    ),
                    "diameter_min_nm": np.nan,
                    "diameter_max_nm": np.nan,
                }
            rows.append({
                "time": time,
                **moments,
                "invalid_or_negative_bin_count": int(np.count_nonzero(
                    ~np.isfinite(column) | (np.asarray(column, dtype=float) < 0)
                )),
                "condensation_sink_s1": sulfuric_acid_condensation_sink(
                    sizes, column, temperature_k, pressure_pa, part_columns
                ),
                **{
                    f"coagulation_sink_{target:g}nm_s1": brownian_coagulation_sink(
                        sizes, column, target, temperature_k, pressure_pa,
                        particle_density_kg_m3,
                        part_columns,
                    )
                    for target in coagulation_targets_nm
                },
            })
        if rows:
            diagnostics.append({
                "method": trace.get("method", "gunn woessner mod"),
                "polarity": trace.get("polarity", "unknown"),
                "scan_polarity_semantics": "DMA voltage sign; selected particle charge has opposite sign",
                "temperature_k": float(temperature_k),
                "pressure_pa": float(pressure_pa),
                "particle_density_kg_m3": float(particle_density_kg_m3),
                "condition_source": "configured inversion conditions",
                "range_semantics": "measured-range; no size extrapolation",
                "rows": rows,
            })
    return diagnostics


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


def build_mcc_growth_cross_checks(
    result,
    growth_diagnostics,
    *,
    method_label,
    maximum_growth_rate_nm_h=15.0,
    minimum_correlation=0.6,
):
    """Cross-check event growth using guarded native-time cross-correlation."""
    heatmaps = {
        (trace.get("method", "gunn woessner mod"), trace.get("polarity", "unknown")): trace
        for trace in result if trace.get("kind") == "heatmap"
    }
    events = {}
    for row in growth_diagnostics:
        key = row.get("event_id")
        if key is None:
            continue
        preferred = events.get(key)
        if preferred is None or row.get("model") == "Ridge peak":
            events[key] = row

    checks = []
    for event_id, event in events.items():
        trace = heatmaps.get((event["source_method"], event["polarity"]))
        if trace is None:
            continue
        sizes = np.asarray(trace["y"], dtype=float)
        z = np.asarray(trace["Z"], dtype=float)
        times = pd.DatetimeIndex(pd.to_datetime(trace["x"], errors="coerce"))
        valid_times = ~times.isna()
        times = times[valid_times]
        z = z[:, valid_times]
        order = np.argsort(times)
        times = times[order]
        z = z[:, order]
        event_mask = (times >= event["event_start"]) & (times <= event["event_end"])
        event_indices = np.flatnonzero(event_mask)
        if len(event_indices) < 6:
            continue
        track_dp = np.asarray(event["dp"], dtype=float)
        dlow, dhigh = np.nanpercentile(track_dp, [30, 70])
        if not np.isfinite(dlow) or not np.isfinite(dhigh) or dhigh - dlow < 2:
            continue
        size_order = np.argsort(sizes)
        sorted_sizes = sizes[size_order]
        sorted_z = z[size_order]
        if dlow < sorted_sizes[0] or dhigh > sorted_sizes[-1]:
            continue
        background = np.nanpercentile(sorted_z, 20, axis=1)
        enhancement = np.maximum(sorted_z - background[:, None], 0.0)
        log_sizes = np.log10(sorted_sizes)
        lower = np.asarray([
            np.interp(np.log10(dlow), log_sizes, enhancement[:, index])
            for index in event_indices
        ])
        upper = np.asarray([
            np.interp(np.log10(dhigh), log_sizes, enhancement[:, index])
            for index in event_indices
        ])
        event_times = times[event_indices]
        seconds = np.asarray((event_times - event_times[0]).total_seconds(), dtype=float)
        cadence = np.diff(seconds)
        cadence = cadence[np.isfinite(cadence) & (cadence > 0)]
        if len(cadence) == 0:
            continue
        median_cadence = float(np.median(cadence))
        maximum_gap = 3.0 * median_cadence
        duration = float(seconds[-1] - seconds[0])
        if duration <= 4 * median_cadence:
            continue
        lag_candidates = np.linspace(median_cadence, 0.75 * duration, min(80, len(event_times) * 3))

        def lag_score(lag_seconds, shifted_upper=upper):
            query = seconds + lag_seconds
            inside = query <= seconds[-1]
            query = query[inside]
            x_values = lower[inside]
            right = np.searchsorted(seconds, query, side="left")
            right = np.clip(right, 1, len(seconds) - 1)
            left = right - 1
            brackets_ok = (seconds[right] - seconds[left]) <= maximum_gap
            gap_boundaries = seconds[1:][np.diff(seconds) > maximum_gap]
            if len(gap_boundaries):
                source = seconds[inside]
                brackets_ok &= ~np.any(
                    (gap_boundaries[:, None] > source[None, :])
                    & (gap_boundaries[:, None] <= query[None, :]),
                    axis=0,
                )
            fraction = (query - seconds[left]) / (seconds[right] - seconds[left])
            y_values = shifted_upper[left] + fraction * (shifted_upper[right] - shifted_upper[left])
            finite = brackets_ok & np.isfinite(x_values) & np.isfinite(y_values)
            minimum_pairs = max(8, int(np.ceil(0.3 * len(event_times))))
            if np.count_nonzero(finite) < minimum_pairs:
                return None
            x_values = x_values[finite]
            y_values = y_values[finite]
            if np.std(x_values) <= 0 or np.std(y_values) <= 0:
                return None
            correlation = float(np.corrcoef(x_values, y_values)[0, 1])
            overlap = len(x_values) / len(event_times)
            return correlation * overlap ** 0.25, correlation, len(x_values), overlap

        zero_finite = np.isfinite(lower) & np.isfinite(upper)
        if np.count_nonzero(zero_finite) < 8:
            continue
        zero_correlation = float(np.corrcoef(lower[zero_finite], upper[zero_finite])[0, 1])
        candidates = []
        for lag_seconds in lag_candidates:
            score = lag_score(lag_seconds)
            if score is not None:
                candidates.append((score[0], score[1], lag_seconds, score[2], score[3]))
        if len(candidates) < 3:
            continue
        candidates.sort(key=lambda item: item[2])
        best_index = int(np.argmax([item[0] for item in candidates]))
        score, correlation, lag_seconds, pairs, overlap = candidates[best_index]
        if best_index in (0, len(candidates) - 1):
            continue
        alternatives = [
            item[0] for item in candidates
            if abs(item[2] - lag_seconds) > 2 * median_cadence
        ]
        prominence = score - max(alternatives) if alternatives else 0.0
        lower_peak_index = int(np.nanargmax(lower))
        upper_peak_index = int(np.nanargmax(upper))
        observed_peak_lag = seconds[upper_peak_index] - seconds[lower_peak_index]
        if (
            correlation < float(minimum_correlation)
            or correlation < zero_correlation + 0.05
            or prominence < 0.02
            or observed_peak_lag <= 0
            or abs(observed_peak_lag - lag_seconds) > 2 * median_cadence
        ):
            continue

        rng = np.random.default_rng(20260904)
        null_maxima = []
        surrogate_upper = upper.copy()
        finite_upper = np.isfinite(surrogate_upper)
        if np.count_nonzero(finite_upper) < 8:
            continue
        surrogate_upper[~finite_upper] = np.interp(
            np.flatnonzero(~finite_upper), np.flatnonzero(finite_upper),
            surrogate_upper[finite_upper],
        )
        centered_upper = surrogate_upper - np.mean(surrogate_upper)
        spectrum_amplitude = np.abs(np.fft.rfft(centered_upper))
        for _ in range(199):
            phases = rng.uniform(0, 2 * np.pi, len(spectrum_amplitude))
            phases[0] = 0.0
            if len(upper) % 2 == 0:
                phases[-1] = 0.0
            permuted = np.fft.irfft(
                spectrum_amplitude * np.exp(1j * phases), n=len(upper)
            )
            scores = [lag_score(lag, permuted) for lag in lag_candidates]
            finite_scores = [item[0] for item in scores if item is not None]
            if finite_scores:
                null_maxima.append(max(finite_scores))
        p_value = (1 + np.count_nonzero(np.asarray(null_maxima) >= score)) / (1 + len(null_maxima))
        if p_value > 0.05:
            continue
        lag_hour = lag_seconds / 3600.0
        rate = float((dhigh - dlow) / lag_hour)
        if rate <= 0 or rate > float(maximum_growth_rate_nm_h):
            continue
        track_rate = float(event["growth_rate"])
        relative_difference = abs(rate - track_rate) / max(0.5 * (rate + track_rate), 1e-12)
        agreement = "supportive" if abs(rate - track_rate) <= 1.0 or relative_difference <= 0.35 else "discordant"
        upper_peak_time = event_times[upper_peak_index]
        lower_peak_time = upper_peak_time - pd.Timedelta(hours=lag_hour)
        checks.append({
            "label": f"{method_label(event['source_method'])} {event['polarity']}",
            "source_method": event["source_method"],
            "polarity": event["polarity"],
            "event_number": event["event_number"],
            "event_id": event_id,
            "model": "MCC cross-check (experimental)",
            "time": pd.DatetimeIndex([lower_peak_time, upper_peak_time]),
            "dp": np.array([dlow, dhigh]),
            "fit": np.array([dlow, dhigh]),
            "growth_rate": rate,
            "slope_p10": np.nan,
            "slope_p90": np.nan,
            "r2": np.nan,
            "correlation": correlation,
            "zero_lag_correlation": zero_correlation,
            "peak_prominence": prominence,
            "permutation_p_value": p_value,
            "overlap_fraction": overlap,
            "lag_hours": lag_hour,
            "rmse_nm": np.nan,
            "n_points": int(pairs),
            "duration_hours": float((event_times[-1] - event_times[0]).total_seconds() / 3600),
            "diameter_span_nm": float(dhigh - dlow),
            "fit_quality": "cross-check",
            "background_quality": event.get("background_quality", "unknown"),
            "event_start": event["event_start"],
            "event_end": event["event_end"],
            "agreement": agreement,
            "relative_difference": relative_difference,
            "reference_track_model": event.get("model"),
            "slope_interval_semantics": "not estimated for MCC cross-check",
        })
    return checks


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
