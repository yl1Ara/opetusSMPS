# Anonymized real-event fixtures

Real-event regressions are opt-in because no approved historical event is
currently stored in the repository. Add matching `EVENT.npz` and `EVENT.json`
files here after checking data-release permission.

The NPZ file must contain:

- `elapsed_minutes`: strictly increasing one-dimensional offsets.
- `diameter_nm`: strictly increasing one-dimensional bin centers.
- `concentration_cm3`: `dN/dlog10Dp`, shaped `(diameter, time)`.

The JSON sidecar must contain:

- `fixture_version`: currently `1`.
- `event_id`: a non-identifying stable name.
- `time_origin`: a synthetic ISO-8601 timestamp used only to reconstruct cadence.
- `polarity`: `positive` or `negative` DMA-voltage sign.
- `anonymization`: statement covering shifted time, removed location/instrument IDs,
  and whether concentrations were scaled.
- `expected`: accepted ranges for event count, growth rate, onset, modal tracks,
  and integrated concentration. Ranges must come from a reviewed baseline, not
  from the code under test in the same run.

Use `tests.real_event_fixtures.load_real_event_fixture` in regression tests.
Never include raw filenames, absolute timestamps, coordinates, operator names,
or acquisition-session identifiers.
