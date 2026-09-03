from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import nnls


POINT_INDEX_COLUMNS = ("point_index", "scan_point_index", "size_index")
POINT_TIME_COLUMNS = (
    "point_valid_from",
    "point_start_time",
    "point_set_time",
    "point_time",
)
POINT_END_TIME_COLUMNS = (
    "point_valid_until",
    "point_end_time",
)
TRANSPORT_DELAY_COLUMNS = (
    "cpc_transport_delay_sec",
    "transport_delay_sec",
    "smps_transport_delay_sec",
)
RESPONSE_WINDOW_COLUMNS = (
    "cpc_response_window_sec",
    "response_window_sec",
)
DWELL_COLUMNS = (
    "point_dwell_sec",
    "dwell_sec",
    "smps_dwell_sec",
    "measurement_dwell_sec",
)


@dataclass(frozen=True)
class AssignmentResult:
    cpc_by_size: pd.Series
    diagnostics: dict


@dataclass(frozen=True)
class ResponseKernelGroup:
    sizes_nm: np.ndarray
    sample_values: np.ndarray
    matrix: np.ndarray
    sample_ids: np.ndarray
    diagnostics: dict
    sample_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    support_starts: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    support_ends: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    correlation: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))


@dataclass(frozen=True)
class ResponseKernelResult:
    groups: dict
    diagnostics: dict


def _epoch_seconds(values):
    values = pd.Series(values)
    if isinstance(values.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_dtype(values.dtype):
        return values.map(lambda value: value.timestamp() if pd.notna(value) else np.nan)
    numeric = pd.to_numeric(values, errors="coerce")
    result = numeric.astype(float)
    nonnumeric = numeric.isna() & values.notna()
    if nonnumeric.any():
        # Parse independently because scan timestamps may mix whole and fractional seconds.
        parsed = values[nonnumeric].map(
            lambda value: pd.to_datetime(value, errors="coerce", utc=True)
        )
        result.loc[nonnumeric] = parsed.map(
            lambda value: value.timestamp() if pd.notna(value) else np.nan
        )
    return result


def _first_present_column(df, candidates):
    return next((column for column in candidates if column in df.columns), None)


def _coalesced_epoch_seconds(df, candidates):
    result = pd.Series(np.nan, index=df.index, dtype=float)
    used_columns = []
    for column in candidates:
        if column not in df.columns:
            continue
        values = _epoch_seconds(df[column])
        usable = ~np.isfinite(result) & np.isfinite(values)
        if usable.any():
            result.loc[usable] = values.loc[usable]
            used_columns.append(column)
    return result, used_columns


def _timing_values(df, candidates, fallback, override, label):
    column = _first_present_column(df, candidates)
    fallback = max(0.0, float(fallback))
    if override:
        return pd.Series(fallback, index=df.index, dtype=float), f"widget override ({label})"
    if column is None:
        return pd.Series(fallback, index=df.index, dtype=float), f"widget fallback ({label})"

    values = pd.to_numeric(df[column], errors="coerce")
    usable = np.isfinite(values) & (values >= 0)
    result = values.where(usable, fallback).astype(float)
    source = f"scan rows ({column})"
    if not usable.all():
        source += f" + widget fallback ({int((~usable).sum())} rows)"
    return result, source


def deduplicate_cpc_rows(rows):
    d = rows.copy()
    if "cpc_sample_id" not in d.columns:
        return d, 0, "none"

    ids = pd.to_numeric(d["cpc_sample_id"], errors="coerce")
    valid_ids = np.isfinite(ids) & (ids > 0)
    if "acquisition_session_id" in d.columns:
        sessions = d["acquisition_session_id"].fillna("").astype(str)
        scope = "session:" + sessions
        missing_sessions = sessions.str.len() == 0
        if missing_sessions.any() and "cpc_sample_time" in d.columns:
            fallback_times = _epoch_seconds(d["cpc_sample_time"])
            scope.loc[missing_sessions] = "sample-time:" + fallback_times.loc[
                missing_sessions
            ].astype(str)
        scan_column = _first_present_column(d, ("scan_id", "scan_number"))
        if missing_sessions.any() and scan_column:
            still_missing = missing_sessions & scope.str.contains("nan", case=False)
            scope.loc[still_missing] = "scan:" + d.loc[still_missing, scan_column].astype(str)
        source = "acquisition session/sample-time fallback + cpc_sample_id"
    elif "cpc_sample_time" in d.columns:
        scope = _epoch_seconds(d["cpc_sample_time"])
        source = "cpc_sample_time + cpc_sample_id"
    else:
        scan_column = _first_present_column(d, ("scan_id", "scan_number"))
        scope = d[scan_column] if scan_column else pd.Series("", index=d.index)
        source = f"{scan_column} + cpc_sample_id" if scan_column else "cpc_sample_id"

    duplicate_keys = pd.DataFrame({"scope": scope, "id": ids}, index=d.index)
    duplicate = valid_ids & duplicate_keys.duplicated(keep="first")
    return d.loc[~duplicate].copy(), int(duplicate.sum()), source


def build_response_kernel(
    rows,
    delay_seconds,
    response_window_seconds,
    dwell_seconds,
    group_columns=("polarity", "scan_range"),
    override_timing=False,
    tolerance_seconds=1e-7,
):
    """Build boxcar CPC sample equations over chronological DMA windows."""
    d = rows.copy()
    d["_size"] = pd.to_numeric(d["abs_size_nm"], errors="coerce").abs()
    d["_value"] = pd.to_numeric(d["cpc_float"], errors="coerce")
    d["_row_time"] = _epoch_seconds(d["time"])
    d = d[np.isfinite(d["_size"]) & (d["_size"] > 0)].copy()
    group_columns = tuple(column for column in group_columns if column in d.columns)
    if d.empty:
        return ResponseKernelResult({}, {"event_count": 0, "accepted_samples": 0})

    d = d.sort_values("_row_time", kind="stable").reset_index(drop=True)
    point_index_column = _first_present_column(d, POINT_INDEX_COLUMNS)
    if point_index_column and pd.to_numeric(d[point_index_column], errors="coerce").notna().all():
        point_indices = pd.to_numeric(d[point_index_column], errors="coerce")
        changed = point_indices.ne(point_indices.shift())
        for column in group_columns:
            changed |= d[column].ne(d[column].shift())
        d["_event"] = changed.cumsum()
        event_source = point_index_column
    else:
        changed = d["_size"].ne(d["_size"].shift())
        for column in group_columns:
            changed |= d[column].ne(d[column].shift())
        d["_event"] = changed.cumsum()
        event_source = "contiguous size runs"

    d["_event_start"], point_time_columns = _coalesced_epoch_seconds(d, POINT_TIME_COLUMNS)
    event_timing_source = (
        f"scan rows ({' -> '.join(point_time_columns)})"
        if point_time_columns else "scan rows (time)"
    )

    d["_dwell"], dwell_source = _timing_values(
        d, DWELL_COLUMNS, dwell_seconds, override_timing, "dwell"
    )
    point_end_time_column = _first_present_column(d, POINT_END_TIME_COLUMNS)
    if point_end_time_column:
        d["_event_end"] = _epoch_seconds(d[point_end_time_column])
        event_end_source = f"scan rows ({point_end_time_column})"
    else:
        d["_event_end"] = np.nan
        event_end_source = "inferred from next setpoint/dwell"
    aggregations = {
        "size": ("_size", "first"),
        "start": ("_event_start", "min"),
        "observed": ("_row_time", "min"),
        "dwell": ("_dwell", "first"),
        "explicit_end": ("_event_end", "max"),
    }
    aggregations.update({column: (column, "first") for column in group_columns})
    events = d.groupby("_event", sort=False).agg(**aggregations).sort_values("observed", kind="stable")
    missing_event_starts = int(events["start"].isna().sum())
    events["start"] = events["start"].fillna(events["observed"])
    if missing_event_starts:
        event_timing_source += f" + row time fallback ({missing_event_starts} events)"
    events["end"] = events["explicit_end"]
    missing_event_ends = ~np.isfinite(events["end"])
    events.loc[missing_event_ends, "end"] = (
        events.loc[missing_event_ends, "start"] + events.loc[missing_event_ends, "dwell"]
    )
    if missing_event_ends.any() and len(events) > 1:
        # Every setpoint remains active until the next chronological setpoint.
        # Dwell supplies the otherwise unknowable end of the final window.
        sequence_column = next(
            (column for column in ("scan_id", "scan_number") if column in group_columns),
            None,
        )
        for index in range(len(events) - 1):
            if sequence_column is None or (
                events.iloc[index][sequence_column] == events.iloc[index + 1][sequence_column]
            ):
                if missing_event_ends.iloc[index]:
                    events.iloc[index, events.columns.get_loc("end")] = events.iloc[index + 1]["start"]
    if point_end_time_column and missing_event_ends.any():
        event_end_source += f" + inferred fallback ({int(missing_event_ends.sum())} events)"

    if "cpc_sample_time" in d.columns:
        sample_times = _epoch_seconds(d["cpc_sample_time"])
        sample_time_fallbacks = int((~np.isfinite(sample_times)).sum())
        d["_sample_time"] = sample_times.where(np.isfinite(sample_times), d["_row_time"])
        sample_time_source = "scan rows (cpc_sample_time)"
        if sample_time_fallbacks:
            sample_time_source += f" + row time fallback ({sample_time_fallbacks} rows)"
    else:
        d["_sample_time"] = d["_row_time"]
        sample_time_source = "scan rows (time)"
    d["_delay"], delay_source = _timing_values(
        d, TRANSPORT_DELAY_COLUMNS, delay_seconds, override_timing, "transport delay"
    )
    d["_response_window"], response_source = _timing_values(
        d, RESPONSE_WINDOW_COLUMNS, response_window_seconds, override_timing, "response window"
    )

    d, duplicate_count, duplicate_scope = deduplicate_cpc_rows(d)
    d["_sample_id"] = (
        pd.to_numeric(d["cpc_sample_id"], errors="coerce")
        if "cpc_sample_id" in d.columns else np.nan
    )

    samples = d[
        np.isfinite(d["_sample_time"])
        & np.isfinite(d["_value"])
        & np.isfinite(d["_delay"])
        & np.isfinite(d["_response_window"])
    ].copy()
    starts = events["start"].to_numpy(dtype=float)
    ends = events["end"].to_numpy(dtype=float)
    event_groups = [
        tuple(getattr(row, column) for column in group_columns)
        for row in events.itertuples(index=False)
    ]
    event_sizes = events["size"].to_numpy(dtype=float)
    group_sizes = {}
    for group, size in zip(event_groups, event_sizes):
        group_sizes.setdefault(group, [])
        if size not in group_sizes[group]:
            group_sizes[group].append(size)
    group_sizes = {group: sorted(sizes) for group, sizes in group_sizes.items()}

    accepted = {group: [] for group in group_sizes}
    mixed_boundary_discards = 0
    outside_window_discards = 0
    sample_columns = ["_sample_time", "_delay", "_response_window", "_value", "_sample_id"]
    for sample_time, delay, width, value, sample_id in samples[sample_columns].itertuples(
        index=False, name=None
    ):
        support_end = float(sample_time - delay)
        width = float(width)
        support_start = support_end - width
        if width == 0:
            candidates = np.flatnonzero(
                (starts - tolerance_seconds <= support_end)
                & (support_end <= ends + tolerance_seconds)
            )
            if len(candidates):
                candidates = np.asarray([candidates[np.argmax(starts[candidates])]])
            weights = np.ones(len(candidates), dtype=float)
        else:
            overlap = np.maximum(0.0, np.minimum(ends, support_end) - np.maximum(starts, support_start))
            candidates = np.flatnonzero(overlap > tolerance_seconds)
            weights = overlap[candidates] / width

        groups_hit = {event_groups[index] for index in candidates}
        covered = float(np.sum(weights))
        if len(groups_hit) > 1:
            mixed_boundary_discards += 1
            continue
        if not candidates.size or covered < 1.0 - tolerance_seconds:
            outside_window_discards += 1
            continue

        group = next(iter(groups_hit))
        matrix_row = np.zeros(len(group_sizes[group]), dtype=float)
        size_indices = {size: index for index, size in enumerate(group_sizes[group])}
        for event_index, weight in zip(candidates, weights):
            matrix_row[size_indices[event_sizes[event_index]]] += weight
        accepted[group].append(
            (float(value), sample_id, matrix_row, sample_time, support_start, support_end, width)
        )

    groups = {}
    for group, sizes in group_sizes.items():
        rows_for_group = accepted[group]
        matrix = np.vstack([item[2] for item in rows_for_group]) if rows_for_group else np.empty((0, len(sizes)))
        values = np.asarray([item[0] for item in rows_for_group], dtype=float)
        sample_ids = np.asarray([item[1] for item in rows_for_group], dtype=float)
        sample_times = np.asarray([item[3] for item in rows_for_group], dtype=float)
        support_starts = np.asarray([item[4] for item in rows_for_group], dtype=float)
        support_ends = np.asarray([item[5] for item in rows_for_group], dtype=float)
        support_widths = np.asarray([item[6] for item in rows_for_group], dtype=float)
        covered = np.any(matrix > tolerance_seconds, axis=0) if len(matrix) else np.zeros(len(sizes), dtype=bool)
        coverage = matrix.sum(axis=0) if len(matrix) else np.zeros(len(sizes), dtype=float)
        if len(rows_for_group):
            correlation = np.eye(len(rows_for_group), dtype=float)
            for i in range(len(rows_for_group)):
                for j in range(i + 1, len(rows_for_group)):
                    if support_widths[i] > 0 and support_widths[j] > 0:
                        overlap = max(
                            0.0,
                            min(support_ends[i], support_ends[j])
                            - max(support_starts[i], support_starts[j]),
                        )
                        value = overlap / np.sqrt(support_widths[i] * support_widths[j])
                        correlation[i, j] = correlation[j, i] = value
            effective_samples = float(len(rows_for_group) ** 2 / correlation.sum())
        else:
            effective_samples = 0.0
        singular_values = np.linalg.svd(matrix, compute_uv=False) if matrix.size else np.array([])
        condition_number = (
            float(singular_values[0] / singular_values[-1])
            if len(singular_values) and singular_values[-1] > 0 else np.inf
        )
        diagnostics = {
            "kernel_rank": int(np.linalg.matrix_rank(matrix)) if matrix.size else 0,
            "kernel_condition_number": condition_number,
            "sample_count": int(len(values)),
            "effective_independent_sample_count": effective_samples,
            "bin_count": int(len(sizes)),
            "per_bin_coverage": {str(size): float(value) for size, value in zip(sizes, coverage)},
            "endpoint_coverage": {
                "first": bool(covered[0]) if len(covered) else False,
                "last": bool(covered[-1]) if len(covered) else False,
            },
        }
        groups[group] = ResponseKernelGroup(
            sizes_nm=np.asarray(sizes, dtype=float),
            sample_values=values,
            matrix=matrix,
            sample_ids=sample_ids,
            diagnostics=diagnostics,
            sample_times=sample_times,
            support_starts=support_starts,
            support_ends=support_ends,
            correlation=correlation if len(rows_for_group) else np.empty((0, 0)),
        )

    diagnostics = {
        "mode": "Response kernel (experimental)",
        "event_count": int(len(events)),
        "event_source": event_source,
        "accepted_samples": int(sum(len(values) for values in accepted.values())),
        "duplicate_cpc_ids_ignored": duplicate_count,
        "duplicate_scope": duplicate_scope,
        "mixed_boundary_discards": mixed_boundary_discards,
        "outside_window_discards": outside_window_discards,
        "timestamp_uncertainty_sec": {
            "median": (
                float(pd.to_numeric(d["cpc_timestamp_uncertainty_sec"], errors="coerce").median())
                if "cpc_timestamp_uncertainty_sec" in d.columns else np.nan
            ),
            "max": (
                float(pd.to_numeric(d["cpc_timestamp_uncertainty_sec"], errors="coerce").max())
                if "cpc_timestamp_uncertainty_sec" in d.columns else np.nan
            ),
        },
        "metadata_provenance": {
            "setpoint_start": event_timing_source,
            "setpoint_end": event_end_source,
            "sample_time": sample_time_source,
            "transport_delay": delay_source,
            "response_window": response_source,
            "dwell": dwell_source,
        },
    }
    return ResponseKernelResult(groups, diagnostics)


def solve_response_kernel_nnls(
    matrix, transfer_matrix, sample_values, smoothness=0.0, correlation=None
):
    """Solve y = M @ A @ x with optional dimensionless smooth NNLS."""
    matrix = np.asarray(matrix, dtype=float)
    transfer_matrix = np.asarray(transfer_matrix, dtype=float)
    sample_values = np.asarray(sample_values, dtype=float)
    design = matrix @ transfer_matrix
    observational_design = design
    observational_values = sample_values
    covariance_floor = 0.0
    correlation_whitening = False
    if correlation is not None:
        correlation = np.asarray(correlation, dtype=float)
        if correlation.shape == (len(sample_values), len(sample_values)) and len(sample_values):
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
            covariance_floor = max(1e-6, 1e-6 * float(np.max(eigenvalues)))
            eigenvalues = np.maximum(eigenvalues, covariance_floor)
            whitener = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
            observational_design = whitener @ design
            observational_values = whitener @ sample_values
            correlation_whitening = True
    smoothness = max(0.0, float(smoothness))
    augmented_design = observational_design
    augmented_values = observational_values
    regularization_rows = 0
    if smoothness > 0 and design.shape[1] >= 3:
        regularizer = np.diff(np.eye(design.shape[1]), n=2, axis=0)
        column_norms = np.linalg.norm(observational_design, axis=0)
        positive_norms = column_norms[column_norms > 0]
        design_scale = float(np.median(positive_norms)) if len(positive_norms) else 1.0
        penalty = smoothness * design_scale * regularizer
        augmented_design = np.vstack([observational_design, penalty])
        augmented_values = np.r_[observational_values, np.zeros(len(penalty))]
        regularization_rows = len(penalty)
    solution, _ = nnls(augmented_design, augmented_values)
    fitted = design @ solution
    singular_values = (
        np.linalg.svd(observational_design, compute_uv=False)
        if observational_design.size else np.array([])
    )
    augmented_singular_values = (
        np.linalg.svd(augmented_design, compute_uv=False) if augmented_design.size else np.array([])
    )

    def condition(values):
        return float(values[0] / values[-1]) if len(values) and values[-1] > 0 else np.inf

    return solution, design @ solution, {
        "rank": int(np.linalg.matrix_rank(observational_design)),
        "condition_number": condition(singular_values),
        "minimum_singular_value": float(singular_values[-1]) if len(singular_values) else 0.0,
        "augmented_rank": int(np.linalg.matrix_rank(augmented_design)),
        "augmented_condition_number": condition(augmented_singular_values),
        "residual_norm": float(np.linalg.norm(sample_values - fitted)),
        "smoothness": smoothness,
        "regularization_rows": regularization_rows,
        "correlation_whitening": correlation_whitening,
        "covariance_eigenvalue_floor": covariance_floor,
        "solution_roughness": float(np.linalg.norm(np.diff(solution, n=2))) if len(solution) >= 3 else 0.0,
    }


def response_kernel_rejection_reason(group):
    bin_count = len(group.sizes_nm)
    rank = int(np.linalg.matrix_rank(group.matrix)) if group.matrix.size else 0
    endpoints = group.diagnostics.get("endpoint_coverage", {})
    coverage = group.diagnostics.get("per_bin_coverage", {})
    if coverage and any(float(value) <= 0 for value in coverage.values()):
        return "one or more size bins have no sample coverage"
    if rank < bin_count:
        return f"kernel rank {rank} is below {bin_count} bins"
    if not endpoints.get("first", False) or not endpoints.get("last", False):
        return "first or last size bin has no sample coverage"
    return None


def response_kernel_ill_conditioned(solve_diagnostics, threshold=1e8):
    value = float(solve_diagnostics.get("condition_number", np.inf))
    return not np.isfinite(value) or value > float(threshold)


def _fill_missing_bins(series):
    values = series.to_numpy(dtype=float, copy=True)
    missing = ~np.isfinite(values)
    source_used = np.zeros(len(values), dtype=bool)
    interpolation_used = np.zeros(len(values), dtype=bool)
    edge_used = np.zeros(len(values), dtype=bool)
    valid = np.isfinite(values)
    if missing.any() and valid.any():
        x = np.log(series.index.to_numpy(dtype=float))
        valid_indices = np.flatnonzero(valid)
        internal = missing & (np.arange(len(values)) > valid_indices[0]) & (
            np.arange(len(values)) < valid_indices[-1]
        )
        values[internal] = np.interp(x[internal], x[valid], values[valid])
        interpolation_used[internal] = True
        edge_used[missing & ~internal] = True
    elif missing.any():
        edge_used[missing] = True

    return (
        pd.Series(values, index=series.index, dtype=float),
        series.index[source_used].tolist(),
        series.index[interpolation_used].tolist(),
        series.index[edge_used].tolist(),
    )


def assign_cpc_samples_to_setpoints(
    rows, delay_seconds, settling_seconds=0.0, group_columns=(), override_timing=False
):
    """Assign unique CPC samples to contiguous DMA setpoint validity windows."""
    d = rows.copy()
    d["_size"] = pd.to_numeric(d["abs_size_nm"], errors="coerce").abs()
    d["_value"] = pd.to_numeric(d["cpc_float"], errors="coerce")
    d["_row_time"] = _epoch_seconds(d["time"])
    d = d[np.isfinite(d["_size"]) & (d["_size"] > 0)].copy()
    group_columns = tuple(column for column in group_columns if column in d.columns)
    expected_sizes = np.asarray(sorted(d["_size"].unique()), dtype=float)
    empty = pd.Series(index=expected_sizes, dtype=float)
    if d.empty:
        return AssignmentResult(empty, {"expected_bins": 0, "assigned_bins": 0})

    d = d.sort_values("_row_time", kind="stable").reset_index(drop=True)
    point_index_column = _first_present_column(d, POINT_INDEX_COLUMNS)
    if point_index_column and pd.to_numeric(d[point_index_column], errors="coerce").notna().all():
        point_indices = pd.to_numeric(d[point_index_column], errors="coerce")
        d["_event"] = point_indices.ne(point_indices.shift()).cumsum()
        event_source = point_index_column
    else:
        changed = d["_size"].ne(d["_size"].shift())
        for column in group_columns:
            changed |= d[column].ne(d[column].shift())
        d["_event"] = changed.cumsum()
        event_source = "contiguous size runs"

    d["_event_start"], point_time_columns = _coalesced_epoch_seconds(d, POINT_TIME_COLUMNS)
    missing_starts = ~np.isfinite(d["_event_start"])
    if missing_starts.any():
        elapsed_source = (
            d["point_elapsed_sec"]
            if "point_elapsed_sec" in d
            else pd.Series(0.0, index=d.index)
        )
        elapsed = pd.to_numeric(elapsed_source, errors="coerce").fillna(0.0)
        fallback_starts = d["_row_time"] - elapsed - max(0.0, float(settling_seconds))
        d.loc[missing_starts, "_event_start"] = fallback_starts.loc[missing_starts]
    timing_source = " -> ".join(point_time_columns) if point_time_columns else "row time/point elapsed"
    if point_time_columns and missing_starts.any():
        timing_source += " + row time/point elapsed fallback"

    aggregations = {
        "size": ("_size", "first"),
        "start": ("_event_start", "min"),
        "observed": ("_row_time", "min"),
    }
    aggregations.update({column: (column, "first") for column in group_columns})
    events = d.groupby("_event", sort=False).agg(**aggregations).sort_values("observed", kind="stable")
    events["start"] = events["start"].fillna(events["observed"])
    starts = events["start"].to_numpy(dtype=float)
    timing_repairs = 0
    for index in range(1, len(starts)):
        if not np.isfinite(starts[index]) or starts[index] <= starts[index - 1]:
            starts[index] = max(events["observed"].iloc[index], starts[index - 1] + 1e-6)
            timing_repairs += 1
    events["start"] = starts

    point_end_time_column = _first_present_column(d, POINT_END_TIME_COLUMNS)
    if point_end_time_column:
        d["_event_end"] = _epoch_seconds(d[point_end_time_column])
        explicit_ends = d.groupby("_event", sort=False)["_event_end"].max().reindex(events.index)
        ends = explicit_ends.to_numpy(dtype=float)
    else:
        ends = np.full(len(events), np.inf, dtype=float)
    if len(events) > 1:
        inferred_ends = np.r_[starts[1:], np.inf]
        ends = np.where(np.isfinite(ends), ends, inferred_ends)

    if "cpc_sample_time" in d.columns:
        sample_time = _epoch_seconds(d["cpc_sample_time"])
        d["_sample_time"] = sample_time.where(np.isfinite(sample_time), d["_row_time"])
    else:
        d["_sample_time"] = d["_row_time"]

    d["_delay"], delay_source = _timing_values(
        d, TRANSPORT_DELAY_COLUMNS, delay_seconds, override_timing, "transport delay"
    )

    duplicate_count = 0
    if "cpc_sample_id" in d.columns:
        sample_ids = pd.to_numeric(d["cpc_sample_id"], errors="coerce")
        valid_ids = np.isfinite(sample_ids) & (sample_ids > 0)
        duplicate_ids = valid_ids & sample_ids.duplicated(keep="first")
        duplicate_count = int(duplicate_ids.sum())
        samples = d[~duplicate_ids].copy()
    else:
        samples = d.copy()

    samples = samples[
        np.isfinite(samples["_sample_time"])
        & np.isfinite(samples["_value"])
        & np.isfinite(samples["_delay"])
    ].copy()
    effective_times = (
        samples["_sample_time"].to_numpy(dtype=float)
        - samples["_delay"].to_numpy(dtype=float)
    )
    event_indices = np.searchsorted(starts, effective_times, side="right") - 1
    assigned = event_indices >= 0
    assigned_indices = event_indices[assigned]
    assigned[assigned] &= effective_times[assigned] <= ends[assigned_indices] + 1e-7
    samples = samples.loc[assigned].copy()
    samples["_assigned_event"] = events.index.to_numpy()[event_indices[assigned]]
    samples["_assigned_size"] = events["size"].to_numpy(dtype=float)[event_indices[assigned]]
    for column in group_columns:
        samples[column] = events.loc[samples["_assigned_event"], column].to_numpy()

    repaired_parts = []
    source_bins = []
    interpolated_bins = []
    edge_bins = []
    groups = events.groupby(list(group_columns), dropna=False, sort=False) if group_columns else [((), events)]
    for group_key, group_events in groups:
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        sample_mask = pd.Series(True, index=samples.index)
        for column, value in zip(group_columns, group_key):
            sample_mask &= samples[column].eq(value) | (samples[column].isna() & pd.isna(value))
        sizes = np.asarray(sorted(group_events["size"].unique()), dtype=float)
        assigned_series = samples.loc[sample_mask].groupby("_assigned_size")["_value"].mean().reindex(sizes)
        repaired, source, interpolated, edge = _fill_missing_bins(assigned_series)
        if group_columns:
            repaired.index = pd.MultiIndex.from_tuples(
                [group_key + (size,) for size in repaired.index],
                names=group_columns + ("abs_size_nm",),
            )
            source_bins.extend([group_key + (size,) for size in source])
            interpolated_bins.extend([group_key + (size,) for size in interpolated])
            edge_bins.extend([group_key + (size,) for size in edge])
        else:
            source_bins.extend(source)
            interpolated_bins.extend(interpolated)
            edge_bins.extend(edge)
        repaired_parts.append(repaired)
    repaired = pd.concat(repaired_parts) if repaired_parts else empty
    diagnostics = {
        "expected_bins": int(len(events) if group_columns else len(expected_sizes)),
        "assigned_bins": int(samples["_assigned_event"].nunique()),
        "assigned_samples": int(len(samples)),
        "duplicate_cpc_ids_ignored": duplicate_count,
        "samples_before_first_window": int((event_indices < 0).sum()),
        "samples_outside_validity_windows": int(
            np.sum((event_indices >= 0) & ~assigned)
        ),
        "source_fallback_bins": source_bins,
        "interpolated_bins": interpolated_bins,
        "edge_fallback_bins": edge_bins,
        "edge_missing_bins": edge_bins,
        "event_count": int(len(events)),
        "event_source": event_source,
        "timing_source": timing_source,
        "timing_repairs": timing_repairs,
        "delay_seconds": float(delay_seconds),
        "metadata_provenance": {"transport_delay": delay_source},
    }
    if not group_columns:
        repaired.index.name = "abs_size_nm"
    return AssignmentResult(repaired, diagnostics)
