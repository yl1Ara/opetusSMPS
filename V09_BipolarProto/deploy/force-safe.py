#!/usr/bin/env python3
"""Best-effort hardware safing, intended only for systemd ExecStopPost."""

import json
import os
from pathlib import Path

import DmpsControl as ctl


def main():
    state_dir = Path(os.environ.get("DMPS_STATE_DIR", "."))
    try:
        settings = json.loads((state_dir / "settings.json").read_text())
    except Exception:
        settings = {}
    errors = []

    def attempt(label, callback):
        try:
            callback()
            print(f"Force-safe: {label}: OK", flush=True)
        except Exception as error:
            errors.append(f"{label}: {error}")
            print(f"Force-safe: {label}: {error}", flush=True)

    blower = None

    def zero_blower():
        nonlocal blower
        blower = ctl.BlowerDAC()
        blower.set_voltage(0.0)

    attempt("blower zero", zero_blower)
    if blower is not None:
        attempt("blower close", blower.close)

    valve = None

    def close_valve():
        nonlocal valve
        valve = ctl.PicoValve(settings.get("pico_valve_port", "/dev/ttyACM0"))
        valve.off()

    attempt("inlet valve off", close_valve)
    if valve is not None:
        attempt("inlet valve close", valve.close)

    # Always safe the locally connected bipolar DAC before optional serial HV.
    attempt("bipolar SPI setup", ctl.HV.setup)
    attempt("bipolar HV zero", ctl.HV.zero)
    attempt("bipolar SPI close", ctl.HV.cleanup)

    spellman = None

    def connect_spellman():
        nonlocal spellman
        spellman = ctl.SpellmanHV(
            port=settings.get("spellman_port", "/dev/ttyUSB0"),
            baud=settings.get("spellman_baud", 9600),
            max_voltage=settings.get("spellman_max_voltage", 30000),
        )

    if settings.get("hv_source") == "Monopolar Spellman":
        attempt("Spellman connect", connect_spellman)
        if spellman is not None:
            attempt("Spellman zero", spellman.zero)
            attempt("Spellman disable", spellman.disable)

    if errors:
        print("Force-safe completed with errors: " + "; ".join(errors), flush=True)
    else:
        print("Force-safe completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
