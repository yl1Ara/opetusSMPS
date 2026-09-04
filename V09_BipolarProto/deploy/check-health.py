#!/usr/bin/env python3
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    maximum_age = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    expected_pid = int(sys.argv[3]) if len(sys.argv) > 3 else None
    try:
        health = json.loads(path.read_text())
        timestamp = datetime.fromisoformat(health["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if age < -5 or age > maximum_age:
            raise ValueError(f"health file age is {age:.1f}s")
        for key in ("runtime_id", "runtime_state", "pid", "version", "commit", "scan_active", "phase", "hardware_initialized", "cpc", "flowmeter", "hv"):
            if key not in health:
                raise ValueError(f"missing {key}")
        if not health["runtime_id"] or int(health["pid"]) <= 0:
            raise ValueError("health lacks runtime identity")
        if expected_pid is not None and int(health["pid"]) != expected_pid:
            raise ValueError(
                f"health PID {health['pid']} does not match service PID {expected_pid}"
            )
        if health["runtime_state"] not in {
            "idle", "initializing", "running", "stopping", "tuning", "calibration",
        }:
            raise ValueError(f"invalid runtime state: {health['runtime_state']}")
        if (health["runtime_state"] == "running") != bool(health["scan_active"]):
            raise ValueError("runtime state and scan_active are inconsistent")
        if health["scan_active"]:
            cpc_age = health["cpc"].get("sample_age_sec")
            flow_age = health["flowmeter"].get("sample_age_sec")
            if not health["hardware_initialized"]:
                raise ValueError("scan active without initialized hardware")
            if not isinstance(cpc_age, (int, float)) or not math.isfinite(cpc_age) or cpc_age > maximum_age:
                raise ValueError(f"active scan CPC sample is stale ({cpc_age})")
            if health["cpc"].get("error"):
                raise ValueError(f"active scan CPC error: {health['cpc']['error']}")
            if not health["flowmeter"].get("connected"):
                raise ValueError("active scan sheath flowmeter is disconnected")
            if not isinstance(flow_age, (int, float)) or not math.isfinite(flow_age) or flow_age > maximum_age:
                raise ValueError(f"active scan sheath sample is stale ({flow_age})")
            if health["flowmeter"].get("last_error"):
                raise ValueError(f"active scan sheath flowmeter error: {health['flowmeter']['last_error']}")
            hv = health["hv"]
            if hv.get("error"):
                raise ValueError(f"active scan HV error: {hv['error']}")
            if hv.get("source") == "Bipolar DAC":
                if hv.get("status") != "enabled":
                    raise ValueError(f"active scan HV is not enabled ({hv.get('status')})")
            else:
                bits = (hv.get("status") or {}).get("bits", {})
                non_fault_bits = {"Enabled", "HW enable", "SW enable"}
                faults = [name for name, active in bits.items() if active and name not in non_fault_bits]
                if not bits.get("Enabled") or faults:
                    raise ValueError(f"active scan Spellman status is unsafe ({hv.get('status')})")
        print(f"instrument health current ({age:.1f}s old, phase={health['phase']})")
        return 0
    except Exception as error:
        print(f"invalid instrument health {path}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
