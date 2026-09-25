# Acquisition Service Boundary

The measurement runtime must not be owned by a browser session. Panel may render controls, plots, snapshots, and status, but opening or closing a browser tab must never start, stop, reset, or safe the instrument.

## Runtime Ownership

- One acquisition service process owns CPC, DMA, blower, valve, flowmeter, HV, scan state, active data buffers, and output files.
- The service starts and stops under systemd, not under a Panel/Bokeh session lifecycle.
- Shutdown and hardware safing happen only on explicit service shutdown, fatal runtime failure, or explicit stop/safe command.
- The UI polls or subscribes to runtime snapshots and submits commands to the service.

The current GUI already has useful concepts to preserve: `runtime_snapshot()`, `publish_runtime_snapshot()`, and command forwarding. The unsafe part to remove during the refactor is the owner Panel document being the hardware owner.

## Shared Control Model

Every authorized viewer may access the same control page. The UI is shared: all viewers see the same current state, same active scan, same settings, same latest data, and same command history.

To avoid conflicting edits:

- Commands are serialized by the acquisition service.
- Each command carries a `requester_id`, `requester_label`, `command_id`, timestamp, command type, and payload.
- A short control lease marks the currently active controller.
- The same controller can continue making changes while the lease is active.
- Another viewer can take control after the lease expires or through an explicit takeover action.
- Measurement-critical commands are validated by the service, not only by disabled UI widgets.

The lease is a coordination tool, not authentication. Network access control still belongs to the lab network, Tailscale ACLs, or an authenticated reverse proxy.

## Command Rules

- `start_scan`: accepted only when the runtime is initialized, idle, and settings validate.
- `stop_scan`: accepted while running or stopping; it remains available even when another viewer has the lease.
- `setting`: accepted only while idle; changes during an active scan are staged for the next scan or rejected explicitly.
- `initialize`: accepted only while idle.
- `manual_hv`: accepted only while idle or in a dedicated maintenance mode.
- `snapshot`: always read-only.

The service response should include whether the command was accepted, rejected, or queued, plus the reason. The UI should show this response to every connected viewer.

## Snapshot Content

Snapshots should include at least:

- Stable instrument ID.
- Runtime process ID/session ID.
- Runtime state: `starting`, `idle`, `running`, `stopping`, `error`, or `shutdown`.
- Whether a scan is active.
- Current phase/progress/status text.
- Active controller label and lease expiry.
- Last accepted command and requester.
- Current settings and which ones are editable.
- Latest CPC, flow, HV, aerosol, Ntot, table, QC, and live plot source data.
- Last completed scan metadata and file path.
- Last runtime error.

## Refactor Steps

1. Keep the current Panel UI and introduce a runtime client object behind existing callbacks.
2. Move command validation into service-side code using `DmpsControl.runtime_contract`.
3. Move hardware initialization and scan threads into a service object that can run without a Panel document.
4. Replace `pn.state.cache` owner/follower hardware ownership with service snapshots and command calls.
5. Remove browser-session shutdown of hardware. Session close should only remove that browser's UI callbacks.
6. Add an HTTP/WebSocket transport only after the in-process service boundary is stable.

This preserves the workflow students need: everyone sees live progress, snapshots, starts, stops, and status, while the measurement itself is protected from browser reconnects and accidental concurrent edits.
