import json
import math
import os
import threading
from pathlib import Path


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    payload = json_safe(value)
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"), allow_nan=False)
        file.write("\n")
        file.flush()
    os.replace(temporary, path)


class ShutdownCoordinator:
    """Run one process-wide shutdown sequence, regardless of its trigger."""

    def __init__(self, callback):
        self.callback = callback
        self._lock = threading.Lock()
        self._started = False

    @property
    def started(self):
        with self._lock:
            return self._started

    def run(self, reason="unknown"):
        with self._lock:
            if self._started:
                return False
            self._started = True
        self.callback(reason)
        return True
