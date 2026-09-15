#!/usr/bin/env python3
"""Set the bipolar DAC to its calibrated electrical-zero midpoint."""


def should_skip_bipolar_zero():
    import json
    import os
    from pathlib import Path

    state_dir = Path(os.environ.get("DMPS_STATE_DIR", "."))
    try:
        settings = json.loads((state_dir / "settings.json").read_text())
    except Exception:
        return False
    return settings.get("hv_source") == "Monopolar Spellman"


def zero_bipolar(hv):
    hv.setup()
    try:
        hv.zero()
    finally:
        hv.cleanup()


def main():
    import DmpsControl as ctl

    if should_skip_bipolar_zero():
        print("Bipolar DAC safe-zero skipped: monopolar Spellman configuration", flush=True)
        return 0
    zero_bipolar(ctl.HV)
    print(f"Bipolar DAC set to safe midpoint code {ctl.HV.BIPOLAR_ZERO_CODE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
