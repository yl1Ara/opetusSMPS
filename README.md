# opetUSMPS

The active instrument and inversion implementation is `V09_BipolarProto`. The hardware GUI runs on an instrument Raspberry Pi; online inversion is designed to run on a separate analysis computer. See `V09_BipolarProto/deploy/README.md` for installation and update procedures.

## Hardware profiles

- Bipolar systems use the SPI DAC, can scan both voltage polarities, and may enable the Pico inlet valve for Ntot measurements.
- Monopolar Spellman systems use positive scan points only, do not initialize the bipolar SPI DAC, and force Ntot valve measurements off.
- The verified `MPS` monopolar wiring uses the CPC on `/dev/serial0` (GPIO14/15) and Spellman on `/dev/ttyAMA3` (GPIO4/5 with `dtoverlay=uart3`).

## Porting older systems

Do not copy old GUI files over a deployment checkout. Commit reviewed changes, update through Git, and preserve settings/logs in the configured state directory so `dmps update` can require a clean fast-forward checkout.

Completed V09 scan CSVs include `scan_complete` and `expected_scan_points`. The inversion viewer rejects interrupted or time-window-truncated scans carrying this metadata, reports what it skipped, and continues processing other valid scans. It also isolates malformed files and individual range inversion failures instead of requiring operators to remove them manually.

Counting uncertainty is not `sqrt(cpc_count)` because `cpc_count` contains concentration in cm-3, not raw events. V09 converts concentration to effective counts using the configured CPC sample flow and each row's response/counting interval, applies Poisson `sqrt(N)` noise, and linearly propagates one-standard-deviation uncertainty through the active NNLS solution. Saved outputs include per-bin `heatmap_1sigma_*.csv` files and `Ntot_*_1sigma` columns. Verify CPC flow and counting-interval settings when porting this logic to older instruments.
