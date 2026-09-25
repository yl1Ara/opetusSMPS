import hashlib
import json
import os
import shutil
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CACHE_FORMAT_VERSION = 1


def _normalized(value):
    if value is pd.NaT or value is pd.NA:
        return None
    if is_dataclass(value):
        return _normalized(asdict(value))
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(column) for column in value.columns],
            "dtypes": [str(dtype) for dtype in value.dtypes],
            "rows": _normalized(value.to_numpy(dtype=object).tolist()),
        }
    if isinstance(value, pd.Series):
        return {
            "name": str(value.name),
            "dtype": str(value.dtype),
            "index": _normalized(value.index.tolist()),
            "values": _normalized(value.tolist()),
        }
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _normalized(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return {"__inversion_cache_float__": "nan"}
        if np.isposinf(numeric):
            return {"__inversion_cache_float__": "inf"}
        if np.isneginf(numeric):
            return {"__inversion_cache_float__": "-inf"}
        return numeric
    if isinstance(value, dict):
        return {
            str(key): _normalized(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if value is None or isinstance(value, (str, int)):
        return value
    if hasattr(value, "__dict__"):
        return _normalized(vars(value))
    return str(value)


def _restored(value):
    if isinstance(value, dict):
        if set(value) == {"__inversion_cache_float__"}:
            return {
                "nan": float("nan"),
                "inf": float("inf"),
                "-inf": float("-inf"),
            }[value["__inversion_cache_float__"]]
        return {key: _restored(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restored(item) for item in value]
    return value


def cache_key(*values):
    payload = json.dumps(
        _normalized(values), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def files_fingerprint(paths, root=None):
    digest = hashlib.sha256()
    root = Path(root) if root is not None else None
    for value in sorted(Path(path) for path in paths):
        path = value.resolve()
        name = path.relative_to(root.resolve()) if root is not None else path
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class InversionCache:
    def __init__(self, root):
        self.root = Path(root)

    def path_for(self, key):
        return self.root / f"{key}.npz"

    def load(self, key):
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as artifact:
                metadata = json.loads(str(artifact["metadata"].item()))
                if (
                    metadata.get("format_version") != CACHE_FORMAT_VERSION
                    or metadata.get("key") != key
                ):
                    return None
                columns = metadata["columns"]
                frame = pd.DataFrame({
                    column: artifact[f"column_{index}"]
                    for index, column in enumerate(columns)
                })
                frame.attrs.update(_restored(metadata.get("attrs", {})))
                if "counting_covariance" in artifact:
                    frame.attrs["counting_covariance"] = artifact["counting_covariance"]
                return frame
        except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def store(self, key, frame):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        temporary = self.root / f".{key}.{os.getpid()}.{threading.get_ident()}.tmp"
        attrs = {
            name: value for name, value in frame.attrs.items()
            if name != "counting_covariance"
        }
        metadata = {
            "format_version": CACHE_FORMAT_VERSION,
            "key": key,
            "columns": list(frame.columns),
            "attrs": _normalized(attrs),
        }
        arrays = {
            f"column_{index}": frame[column].to_numpy(dtype=float)
            for index, column in enumerate(frame.columns)
        }
        if "counting_covariance" in frame.attrs:
            arrays["counting_covariance"] = np.asarray(
                frame.attrs["counting_covariance"], dtype=float,
            )
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    metadata=np.asarray(json.dumps(metadata, separators=(",", ":"))),
                    **arrays,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def clear(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def stats(self):
        if not self.root.exists():
            return {"entries": 0, "bytes": 0}
        files = list(self.root.glob("*.npz"))
        return {
            "entries": len(files),
            "bytes": sum(path.stat().st_size for path in files if path.is_file()),
        }
