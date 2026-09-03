#!/usr/bin/env python3
"""Set the bipolar DAC to its calibrated electrical-zero midpoint."""


def zero_bipolar(hv):
    hv.setup()
    try:
        hv.zero()
    finally:
        hv.cleanup()


def main():
    import DmpsControl as ctl

    zero_bipolar(ctl.HV)
    print(f"Bipolar DAC set to safe midpoint code {ctl.HV.BIPOLAR_ZERO_CODE}", flush=True)


if __name__ == "__main__":
    main()
