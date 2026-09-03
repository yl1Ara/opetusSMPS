# Customer deployment

This deployment keeps all source files in one SSH-cloned monorepo checkout, but runs only the hardware GUI on an instrument Raspberry Pi:

- Hardware GUI: `gui.py`, systemd `tdmps@USER.service`, localhost port 5006
- Online inversion files remain available for deployment on a separate analysis computer; they are not started on the Pi.

The hardware service restarts after process failures so its web interface remains available. Restarting the web process does not automatically initialize hardware or resume a scan. The server binds only to localhost and accepts only its configured exact websocket origin; wildcards are not used.

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

By default, runtime settings and logs remain under `V09_BipolarProto`. To preserve an existing absolute collection path while running code from a clean clone, pass an existing writable state directory:

```bash
deploy/install-services.sh --origin customer-host.tailnet-name.ts.net --state-dir /home/pi/Desktop/TDMPS
```

The services then load code and dependencies from the Git checkout but retain `settings.json`, `settings_inversion.json`, `logs/`, and viewer state under the state directory.

The installer creates `.venv`, synchronizes locked application dependencies plus Raspberry Pi hardware dependencies, syntax-checks only the instrument application and hardware modules, installs `dmps`, enables only the hardware service, and verifies both its localhost endpoint and fresh `health.json` heartbeat. Before Panel starts, systemd writes calibrated midpoint code `32705` to the bipolar DAC; a failed midpoint write prevents the GUI from starting. Inversion code is not executed on the instrument.

## Operations and updates

```bash
dmps status
dmps health
dmps update
```

`dmps update` operates on the complete monorepo. It refuses a dirty tree, refuses a non-fast-forward pull, and repeatedly inspects the shared live Panel document to refuse an active measurement. If measurement state cannot be verified, it fails closed. It then runs `git fetch --prune origin`, `git pull --ff-only`, dependency synchronization, and Python syntax checks before restarting the hardware service if it was already running. The restarted service must pass its localhost HTTP health check. A failed dependency or syntax check leaves the existing process running and does not restart it. If a measurement is started during the update, the final guard leaves the updated main service process running without restarting it and reports failure.

Stop a measurement in the GUI and confirm it is idle before updating. Do not schedule `dmps update` from cron or a systemd timer. Do not manually run a second hardware GUI beside `tdmps@USER.service`.

Service logs are available with `dmps log`. The existing `tdmps@USER.service` name is retained for compatibility.

Stopping the service first invokes the application's idempotent safe shutdown. After the process exits, `ExecStopPost` independently commands the inlet valve off, both HV outputs safe, and the blower DAC to zero. This second layer never runs alongside the application.

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

Use Tailscale ACLs/grants to limit customer access. Do not open ports 5006 or 5007 in the host firewall and do not bind Panel to `0.0.0.0`.

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
