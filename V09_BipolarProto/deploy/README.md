# Customer deployment

This deployment keeps all source files in one SSH-cloned monorepo checkout, but runs only the hardware GUI on an instrument Raspberry Pi:

- Hardware GUI: `gui.py`, systemd `tdmps@USER.service`, localhost port 5006 by default
- Online inversion files remain available for deployment on a separate analysis computer; they are not started on the Pi.

The hardware service restarts after process failures so its web interface remains available. Restarting the web process does not automatically initialize hardware or resume a scan. The server binds only to localhost and accepts only its configured exact websocket origin; wildcards are not used.

The templated hardware unit conflicts with the legacy `tdmps.service`, checks that the configured panel port is free before touching hardware, and rate-limits failed starts. Installation disables inactive legacy service names. If a legacy service is active, `dmps start`, `dmps restart`, and `dmps update` fail closed until it is stopped and disabled.

## SSH deploy key

Create a key on the customer system and leave its passphrase empty for unattended `git fetch`:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/opetus_smps_deploy -C "customer-system-name deploy"
cat ~/.ssh/opetus_smps_deploy.pub
```

Add the public key to the repository as a read-only GitHub deploy key. Configure SSH to use it:

```sshconfig
Host github.com-opetus-smps
    HostName github.com
    User git
    IdentityFile ~/.ssh/opetus_smps_deploy
    IdentitiesOnly yes
```

Verify host identity and access interactively before installation:

```bash
ssh -T git@github.com-opetus-smps
```

## Clone and install

Clone the repository, not `V09_BipolarProto` by itself. The installer verifies this layout:

```text
<repository-root>/V09_BipolarProto
```

```bash
git clone git@github.com-opetus-smps:OWNER/opetusSMPS.git ~/opetusSMPS
cd ~/opetusSMPS/V09_BipolarProto
```

Install `uv` if it is not already available, then install with the system's exact Tailscale MagicDNS name. Origins contain only `host[:port]`, never `https://`, a path, or `*`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
deploy/install-services.sh --origin customer-host.tailnet-name.ts.net
```

Use `--port 5008` when an installation should use a non-default local Panel port. The systemd port guard, Panel launcher, health command, and Tailscale proxy all read the resulting `DMPS_PANEL_PORT` setting.

By default, runtime settings and logs remain under `V09_BipolarProto`. To preserve an existing absolute collection path while running code from a clean clone, pass an existing writable state directory:

```bash
deploy/install-services.sh --origin customer-host.tailnet-name.ts.net --state-dir /home/pi/Desktop/TDMPS
```

The services then load code and dependencies from the Git checkout but retain `settings.json`, `settings_inversion.json`, `logs/`, and viewer state under the state directory.

## Bipolar and monopolar installations

The bipolar installation uses the SPI DAC and may use the Pico inlet valve for Ntot measurements. Keep SPI enabled, select `Bipolar DAC`, and enable the Ntot valve only when that valve is physically installed.

The monopolar installation uses only positive scan points, does not use the bipolar SPI DAC, and never uses the Ntot valve. Disable SPI and enable the CPC and Spellman UARTs in `/boot/firmware/config.txt`:

```ini
## Monopolar MPS
#dtparam=spi=on
enable_uart=1
dtoverlay=disable-bt-pi5
dtoverlay=uart3
```

After rebooting, verify the serial devices before starting the GUI:

```bash
ls -l /dev/serial* /dev/ttyAMA* /dev/ttyS*
```

The verified `MPS` monopolar installation uses `/dev/serial0 -> /dev/ttyS0` on GPIO14/15 for the CPC and `/dev/ttyAMA3` on GPIO4/5 for the Spellman supply. Set `cpc_com_port` to `/dev/serial0`, `spellman_port` to `/dev/ttyAMA3`, `hv_source` to `Monopolar Spellman`, and the Hauke DMA length to `0.11 m` before initializing hardware. Selecting that source forces positive-only scans and disables Ntot controls in the GUI. Other installations retain the `0.28 m` DMA default unless explicitly changed.

After changing overlays, reboot and confirm GPIO4/5 report `TXD3`/`RXD3`. Device aliases can vary with firmware, so verify the actual devices instead of assuming `/dev/serial1` exists.

The installer acquires the instrument maintenance lock and refuses a busy or unverifiable running service before changing dependencies. It creates `.venv`, synchronizes locked application dependencies plus Raspberry Pi hardware dependencies, syntax-checks only the instrument application and hardware modules, installs `dmps`, enables only the hardware service, and verifies both its localhost endpoint and fresh process-matched `health.json` heartbeat. Before Panel starts, a bipolar installation writes calibrated midpoint code `32705` to its DAC; monopolar installations skip that startup action because SPI is disabled. Inversion code is not executed on the instrument.

## Operations and updates

```bash
dmps status
dmps health
dmps system-health
dmps diagnose-reboot
dmps update
```

`dmps update` operates on the complete monorepo. It acquires an exclusive maintenance lock, refuses a dirty tree or non-fast-forward pull, and reads a fresh `health.json` whose PID must match systemd's current process; it never opens a Panel session to determine whether measurement is idle. Measurement, initialization, tuning, and calibration hold shared maintenance leases, so they cannot begin or remain active during an update. If state cannot be verified, the update fails closed. After its final idle check, it stops the hardware service before changing the checkout or environment, runs dependency synchronization and Python syntax checks, installs runtime files, starts the service if it was previously active, and requires the new PID's localhost endpoint and heartbeat to pass. A failed update leaves the service stopped rather than running mixed old and new files. Do not copy tracked source files into an installed checkout: publish changes to GitHub, keep runtime files under the configured state directory, and use `dmps update` so the checkout remains clean.

Stop a measurement in the GUI and confirm it is idle before updating. Do not schedule `dmps update` from cron or a systemd timer. Do not manually run a second hardware GUI beside `tdmps@USER.service`.

Service logs are available with `dmps log`. Installation overrides Raspberry Pi OS's volatile-journal default and retains up to 200 MB or three months of system journals across reboots. The independent `tdmps-health-log@USER.service` fsyncs CPU temperature and frequency, Raspberry Pi throttling history, uptime, load, memory, boot identity, and Tailscale version/state every minute under `logs/system/`; `dmps system-health` shows its latest records. After an unexplained reboot or outage, `dmps diagnose-reboot` prints those records alongside boot history, watchdog policy, current Pi throttling flags, current and previous boot power/reset kernel messages, and retained previous-boot TDMPS service logs. `dmps events` shows the latest persistent JSONL runtime events from `logs/runtime/`, including panel sessions, measurement transitions, one-minute hardware heartbeats, scan QC, failures, and shutdown results. The existing `tdmps@USER.service` name is retained for compatibility.

Stopping the service first invokes the application's idempotent safe shutdown. After the process exits, `ExecStopPost` independently attempts to command the inlet valve off, both HV outputs safe, and the blower DAC to zero, regardless of the currently saved profile. Missing hardware is reported but does not prevent the remaining safing attempts. This second layer never runs alongside the application.

## Tailscale exposure

The hardware service remains localhost-only. To expose it to the tailnet, install and authenticate Tailscale, then enable the supplied proxy unit:

```bash
sudo tailscale up
sudo systemctl enable --now "tdmps-serve@$(whoami).service"
tailscale serve status
dmps url
```

The route is:

```text
https://customer-host.tailnet-name.ts.net/gui
```

Use Tailscale ACLs/grants to limit customer access. Do not open the local Panel port in the host firewall and do not bind Panel to `0.0.0.0`.

## One-time migration from the copied TDMPS directory

Older systems may have `/home/pi/Desktop/TDMPS` as a standalone, dirty checkout. Do not run `git pull` there and do not delete it. Stop measurement and zero HV in the GUI first, then create the clean monorepo clone described above. Preserve local state before replacing the legacy service:

```bash
mkdir -p ~/opetusSMPS/V09_BipolarProto/logs
rsync -a ~/Desktop/TDMPS/logs/ ~/opetusSMPS/V09_BipolarProto/logs/
cp ~/Desktop/TDMPS/settings.json ~/opetusSMPS/V09_BipolarProto/settings.json
cp ~/Desktop/TDMPS/settings_inversion.json ~/opetusSMPS/V09_BipolarProto/settings_inversion.json
sudo systemctl disable --now tdmps.service
cd ~/opetusSMPS/V09_BipolarProto
deploy/install-services.sh --origin "$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
```

The installer intentionally refuses to proceed while the legacy service is active, preventing two hardware controllers from binding the same port or opening the same devices. Keep the old directory as a backup until the new main GUI, online viewer, settings, and historical logs have been verified.

The repository no longer tracks Python bytecode. Commit the accompanying `__pycache__` deletions and `.gitignore` update before cloning customer systems; otherwise imported bytecode can make future pulls dirty.

## Online inversion computer

Run inversion on a separate analysis computer, not on the instrument Pi. The repository includes `online_inversion_viewer.py`, `DMPS_inversion_gui/`, and `run_online_inversion_viewer.sh`. On the analysis computer:

```bash
cd ~/opetusSMPS/V09_BipolarProto
./run_online_inversion_viewer.sh
```

Set `ONLINE_VIEWER_ORIGIN=analysis-host.example:5007` when accessing it through another exact hostname. The launcher binds only to localhost by default; use a separate authenticated proxy or SSH forwarding for remote access.

On the CSC analysis host, install the tracked service helper after cloning the repository:

```bash
cd /home/ubuntu/opetusSMPS/V09_BipolarProto
sudo install -o root -g root -m 0755 deploy/inversion /usr/local/bin/inversion
sudo install -o root -g root -m 0644 deploy/inversion-completion.bash /etc/bash_completion.d/inversion
```

The production defaults are service `opetus-panel.service`, checkout
`/home/ubuntu/opetusSMPS/V09_BipolarProto`, and local health URL
`http://127.0.0.1:5008/online_inversion_viewer`. They can be overridden with
`INVERSION_SERVICE`, `INVERSION_APP_DIR`, and `INVERSION_HEALTH_URL`.

Use `inversion status`, `inversion health`, and `inversion tail` for routine
operation. `inversion update` refuses a dirty, detached, locally advanced, or
diverged checkout. It fetches and fast-forwards only from `origin/main`, runs a
locked dependency sync and Python/shell syntax checks, refreshes the helper,
and health-checks the restarted viewer. A stopped viewer remains stopped.

There is no reliable external signal for all per-session inversion jobs. When
the viewer is active, update therefore requires an interactive `UPDATE`
confirmation after all users have been notified and all jobs have finished.
Do not automate this command. It stops the viewer before changing source or
dependencies. A failed validation leaves it stopped; correct the problem,
rerun `inversion update`, then use `inversion start`. The Pi-specific
measurement-idle check does not protect online inversion work.

### CSC instrument data pull

The CSC `sync-dmps.timer` runs `/usr/local/bin/sync-dmps` as `ubuntu` every
30 minutes. The tracked script pulls `/home/pi/Desktop/TDMPS/logs/` from
`pi@varjo-dmps` into `/home/ubuntu/dmps/logs/` without deleting CSC files.
It uses `/home/ubuntu/.ssh/raspberrypi_sync` and the Pi's already-verified
`raspberrypi` SSH host-key entry (via `HostKeyAlias`). To install and check it
on CSC:

```bash
cd ~/opetusSMPS/V09_BipolarProto
sudo install -o root -g root -m 0755 deploy/sync-dmps /usr/local/bin/sync-dmps
sync-dmps --dry-run
sudo systemctl start sync-dmps.service
systemctl status sync-dmps.service --no-pager
systemctl list-timers sync-dmps.timer --no-pager
```

The first successful run may copy a backlog of scans. `journalctl -u
sync-dmps.service` shows the transfer summary or an SSH/rsync error.

The separate SMEAR III monopolar Pi (`pi@mps`) writes to
`/home/pi/opetusSMPS/V09_BipolarProto/logs/`. Its data belongs under
`/home/ubuntu/mps/logs/` on CSC, not in the bipolar `dmps` tree. Once CSC
can SSH to `pi@mps` without an interactive Tailscale check, install and
enable the independent MPS job:

```bash
cd ~/opetusSMPS/V09_BipolarProto
sudo install -o root -g root -m 0755 deploy/sync-mps /usr/local/bin/sync-mps
sudo install -o root -g root -m 0644 deploy/sync-mps.service /etc/systemd/system/sync-mps.service
sudo install -o root -g root -m 0644 deploy/sync-mps.timer /etc/systemd/system/sync-mps.timer
sync-mps --dry-run
sudo systemctl daemon-reload
sudo systemctl enable --now sync-mps.timer
sudo systemctl start sync-mps.service
systemctl status sync-mps.service --no-pager
```

If CSC gets a Tailscale SSH web-approval prompt, arrange noninteractive
access for this host in the tailnet policy before enabling the timer; an
interactive approval alone will not make unattended sync reliable.

### University SMB data through the desktop

The university SMB share is mounted on the local desktop using eduVPN's
**Split-tunnel** profile. CSC does not need eduVPN: a local user timer stages
only `*.scan` from UFSMPS 2026 and `*.scan`/`*.sum` from SMPS dated June 2026
onward, then pushes completed files to CSC over the existing SSH connection.
The local stage is `~/.local/share/opetusSMPS/university/2026/{ufsmps,smps}`;
CSC receives `/home/ubuntu/university/2026/{ufsmps,smps}`. No SMB login or
VPN configuration is copied to CSC, and no source or CSC files are deleted.

On the desktop, while logged in with eduVPN connected and `h527` mounted in
the file manager:

```bash
cd ~/Desktop/Projects/opetusSMPS/V09_BipolarProto
install -m 0755 deploy/sync-university-local ~/.local/bin/sync-university-local
install -m 0644 deploy/sync-university-local.service ~/.config/systemd/user/
install -m 0644 deploy/sync-university-local.timer ~/.config/systemd/user/
sync-university-local --dry-run
systemctl --user daemon-reload
systemctl --user enable --now sync-university-local.timer
systemctl --user start --no-block sync-university-local.service
```

The first run stages several gigabytes and may take a while. The timer retries
every 30 minutes while the desktop's user session is running; when eduVPN or
the SMB mount is unavailable it skips SMB and still sends previously staged
files to CSC. Check `systemctl --user status sync-university-local.service`
and `journalctl --user -u sync-university-local.service` for the outcome.
The script checks that the university server's route uses `eduVPN`; it does
not change the default route or connect/disconnect any VPN.

### Inverting the synced scans on CSC

The online viewer loads independent **Bipolar Pi**, **Monopolar Pi**,
**SMEAR III UFSMPS**, and **SMEAR III SMPS** instrument tabs on demand. Each
tab has its own inversion controls and result. The **Comparison** tab plots
their independently inverted heatmaps on the same color scale and compares
measured-range N only over their common diameter interval (when one exists).
Each instrument tab and **My Explorer** saves its settings under
`~/.local/share/opetusSMPS/inversion-settings/` on the analysis host. The
global tab continues to use `settings_inversion.json`. These files survive
viewer restarts and `inversion update`; the first load of an instrument tab
imports its latest legacy per-session settings if available. Keep the settings
directory when replacing the Git checkout. The **Comparison** tab also saves
its model, polarity, and color-scale maximum in this directory.

Use **Refresh comparison** after new inversions finish; tabs and results are
personal to that browser session. The **Scan source** selector within each
tab also supports a custom folder. Select a small number of days before
running inversion (university `.scan` files contain many scans per day); the
university tabs start at one day. The SMEAR III SMPS `.sum`
reference folder defaults to `/home/ubuntu/university/2026/smps` on CSC;
it can be edited in the diagnostics controls for the median and ratio plots.
Only `.scan` files enter the inversion; `.sum` files are comparison data.
Exports from UFSMPS, SMPS, and the monopolar Pi are kept in separate
subdirectories under the configured save folder. Check each instrument's
flows, CPC, size cutoff, timing correction, and inlet/tube-loss settings
before interpreting or comparing inverted distributions.

In an instrument's **Inversion** tab, choose Rectangle or Freehand and drag
across heatmap cells to inspect a region. The **Saved ROIs** tab shows the
selected distributions and, for at least four scans spanning 15 minutes, a
descriptive D50-vs-time slope. On **Global Live**, selection appears in its
**Selected ROI** tab instead. The growth status above each inversion heatmap
explains when automatic tracks were rejected; marginal tracks show points
without a fitted line. The **Growth Signal** tab plots full per-size
background-subtracted enhancement and a separate view of the component the
tracker selected. Its color scales are independent; positions on the raw
inversion heatmap are markers, not a raw-peak fit. A moving D50 without a
moving component peak is marked marginal, since it may reflect broadening.
The main 72-hour SMEAR median pairs scans within 15 minutes before comparing
distributions, while Difference Diagnostics uses paired scans from the last
three hours.
