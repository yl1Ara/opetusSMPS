#!/usr/bin/env python3
"""Exit 0 when health reports idle, 10 when busy, and 20 when unknown."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


BUSY_STATES = {"initializing", "running", "stopping", "tuning", "calibration"}


def measurement_state(path, maximum_age_seconds=10.0, expected_pid=None):
    health = json.loads(Path(path).read_text())
    timestamp = datetime.fromisoformat(health["timestamp"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds()
    if age < 0 or age > float(maximum_age_seconds):
        raise ValueError(f"health file age is {age:.1f}s")
    if not health.get("runtime_id") or int(health.get("pid", 0)) <= 0:
        raise ValueError("health lacks runtime identity")
    if expected_pid is not None and int(health["pid"]) != int(expected_pid):
        raise ValueError(
            f"health PID {health['pid']} does not match service PID {expected_pid}"
        )
    state = str(health.get("runtime_state", "unknown"))
    if health.get("scan_active") or state in BUSY_STATES:
        return 10, f"instrument runtime is busy ({state})"
    if state == "idle":
        return 0, "instrument runtime is idle"
    raise ValueError(f"unknown instrument runtime state: {state}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "health.json"
    maximum_age = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    expected_pid = int(sys.argv[3]) if len(sys.argv) > 3 else None
    try:
        code, message = measurement_state(path, maximum_age, expected_pid)
        print(message, file=sys.stderr if code else sys.stdout)
        return code
    except Exception as error:
        print(f"Could not verify measurement state: {error}", file=sys.stderr)
        return 20


if __name__ == "__main__":
    raise SystemExit(main())
