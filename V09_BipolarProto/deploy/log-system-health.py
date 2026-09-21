#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path


THROTTLE_FLAGS = {
    0: "under_voltage_now",
    1: "frequency_capped_now",
    2: "throttled_now",
    3: "soft_temperature_limit_now",
    16: "under_voltage_occurred",
    17: "frequency_capped_occurred",
    18: "throttling_occurred",
    19: "soft_temperature_limit_occurred",
}


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu_temperature_celsius(thermal_root=Path("/sys/class/thermal")):
    for zone in sorted(thermal_root.glob("thermal_zone*")):
        if read_text(zone / "type") == "cpu-thermal":
            value = read_text(zone / "temp")
            return float(value) / 1000.0 if value is not None else None
    return None


def throttling_status():
    try:
        result = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True,
            timeout=5, check=True,
        ).stdout.strip()
        raw = int(result.split("=", 1)[1], 16)
        return {"raw": f"0x{raw:x}", **{
            name: bool(raw & (1 << bit)) for bit, name in THROTTLE_FLAGS.items()
        }}
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError) as error:
        return {"raw": None, "error": str(error)}


def memory_status(meminfo_path=Path("/proc/meminfo")):
    values = {}
    try:
        for line in meminfo_path.read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
    }


def tailscale_status():
    try:
        status = json.loads(subprocess.run(
            ["tailscale", "status", "--json"], capture_output=True, text=True,
            timeout=5, check=True,
        ).stdout)
        return {
            "backend_state": status.get("BackendState"),
            "online": status.get("Self", {}).get("Online"),
        }
    except (FileNotFoundError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        return {"backend_state": None, "online": None, "error": str(error)}


def health_record():
    try:
        load = list(os.getloadavg())
    except OSError:
        load = [None, None, None]
    uptime = read_text("/proc/uptime")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "boot_id": read_text("/proc/sys/kernel/random/boot_id"),
        "cpu_temperature_c": cpu_temperature_celsius(),
        "throttling": throttling_status(),
        "uptime_sec": float(uptime.split()[0]) if uptime else None,
        "load_1m": load[0],
        "load_5m": load[1],
        "load_15m": load[2],
        "memory": memory_status(),
        "tailscale": tailscale_status(),
    }


def append_record(log_directory, record):
    log_directory.mkdir(parents=True, exist_ok=True)
    day = record["timestamp"][:10].replace("-", "")
    path = log_directory / f"system-health-{day}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_directory", type=Path)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: stop.set())

    while True:
        append_record(args.log_directory, health_record())
        if args.once or stop.wait(args.interval):
            return


if __name__ == "__main__":
    main()
