"""Validation and loading for opt-in anonymized real-event regressions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_METADATA = {
    "fixture_version", "event_id", "time_origin", "expected",
    "anonymization",
}


def load_real_event_fixture(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    missing = REQUIRED_METADATA - set(metadata)
    if missing:
        raise ValueError(f"fixture metadata missing: {sorted(missing)}")
    data = np.load(path)
    required_arrays = {"elapsed_minutes", "diameter_nm", "concentration_cm3"}
    missing_arrays = required_arrays - set(data.files)
    if missing_arrays:
        raise ValueError(f"fixture arrays missing: {sorted(missing_arrays)}")
    elapsed = np.asarray(data["elapsed_minutes"], dtype=float)
    diameter = np.asarray(data["diameter_nm"], dtype=float)
    concentration = np.asarray(data["concentration_cm3"], dtype=float)
    if concentration.shape != (len(diameter), len(elapsed)):
        raise ValueError("concentration_cm3 must have shape (diameter, time)")
    if np.any(np.diff(elapsed) <= 0) or np.any(np.diff(diameter) <= 0):
        raise ValueError("elapsed_minutes and diameter_nm must increase strictly")
    time = pd.Timestamp(metadata["time_origin"]) + pd.to_timedelta(elapsed, unit="min")
    return metadata, {
        "kind": "heatmap",
        "method": "real-event-fixture",
        "polarity": metadata.get("polarity", "positive"),
        "x": pd.DatetimeIndex(time),
        "y": diameter,
        "Z": concentration,
    }
