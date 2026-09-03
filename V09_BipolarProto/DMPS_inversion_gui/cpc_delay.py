from dataclasses import dataclass

import numpy as np
import pandas as pd


POINT_INDEX_COLUMNS = ("point_index", "scan_point_index", "size_index")
POINT_TIME_COLUMNS = (
    "point_start_time",
    "point_set_time",
    "point_valid_from",
    "point_time",
)


@dataclass(frozen=True)
class AssignmentResult:
    cpc_by_size: pd.Series
    diagnostics: dict


def _epoch_seconds(values):
    values = pd.Series(values)
    if isinstance(values.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_dtype(values.dtype):
        return values.map(lambda value: value.timestamp() if pd.notna(value) else np.nan)
    numeric = pd.to_numeric(values, errors="coerce")
    result = numeric.astype(float)
    nonnumeric = numeric.isna() & values.notna()
    if nonnumeric.any():
        parsed = pd.to_datetime(values[nonnumeric], errors="coerce", utc=True)
        result.loc[nonnumeric] = parsed.map(
            lambda value: value.timestamp() if pd.notna(value) else np.nan
        )
    return result


def _first_present_column(df, candidates):
    return next((column for column in candidates if column in df.columns), None)


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
    rows, delay_seconds, settling_seconds=0.0, group_columns=()
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

    point_time_column = _first_present_column(d, POINT_TIME_COLUMNS)
    if point_time_column:
        d["_event_start"] = _epoch_seconds(d[point_time_column])
        timing_source = point_time_column
    else:
        elapsed_source = (
            d["point_elapsed_sec"]
            if "point_elapsed_sec" in d
            else pd.Series(0.0, index=d.index)
        )
        elapsed = pd.to_numeric(elapsed_source, errors="coerce").fillna(0.0)
        d["_event_start"] = d["_row_time"] - elapsed - max(0.0, float(settling_seconds))
        timing_source = "row time/point elapsed"

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

    if "cpc_sample_time" in d.columns:
        sample_time = _epoch_seconds(d["cpc_sample_time"])
        d["_sample_time"] = sample_time.where(np.isfinite(sample_time), d["_row_time"])
    else:
        d["_sample_time"] = d["_row_time"]

    duplicate_count = 0
    if "cpc_sample_id" in d.columns:
        sample_ids = pd.to_numeric(d["cpc_sample_id"], errors="coerce")
        valid_ids = np.isfinite(sample_ids) & (sample_ids > 0)
        duplicate_ids = valid_ids & sample_ids.duplicated(keep="first")
        duplicate_count = int(duplicate_ids.sum())
        samples = d[~duplicate_ids].copy()
    else:
        samples = d.copy()

    samples = samples[np.isfinite(samples["_sample_time"]) & np.isfinite(samples["_value"])].copy()
    effective_times = samples["_sample_time"].to_numpy(dtype=float) - float(delay_seconds)
    event_indices = np.searchsorted(starts, effective_times, side="right") - 1
    assigned = event_indices >= 0
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
        "samples_before_first_window": int((~assigned).sum()),
        "source_fallback_bins": source_bins,
        "interpolated_bins": interpolated_bins,
        "edge_fallback_bins": edge_bins,
        "edge_missing_bins": edge_bins,
        "event_count": int(len(events)),
        "event_source": event_source,
        "timing_source": timing_source,
        "timing_repairs": timing_repairs,
        "delay_seconds": float(delay_seconds),
    }
    if not group_columns:
        repaired.index.name = "abs_size_nm"
    return AssignmentResult(repaired, diagnostics)
