# Customer deployment

This deployment keeps both applications in one SSH-cloned monorepo checkout:

- Hardware GUI: `gui.py`, systemd `tdmps@USER.service`, localhost port 5006
- Offline viewer: `offline_inversion_viewer.py`, systemd `tdmps-viewer@USER.service`, localhost port 5007

Both services restart after process failures so their web interfaces remain available. Restarting the hardware web process does not automatically initialize hardware or resume a scan. Both servers bind only to localhost and accept only their configured exact websocket origins; wildcards are not used.

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

The default viewer origin is the same host on port 8443. Override it with `--viewer-origin HOST:PORT` if the proxy uses another exact origin. The installer creates `.venv`, synchronizes locked application dependencies plus Raspberry Pi hardware dependencies, syntax-checks both applications, installs the units and `dmps`, enables both application services, and verifies both localhost health endpoints.

## Operations and updates

```bash
dmps status
dmps health
dmps viewer status
dmps viewer health
dmps viewer stop
dmps viewer start
dmps update
```

`dmps update` operates on the complete monorepo. It refuses a dirty tree, refuses a non-fast-forward pull, and repeatedly inspects the shared live Panel document to refuse an active measurement. If measurement state cannot be verified, it fails closed. It then runs `git fetch --prune origin`, `git pull --ff-only`, dependency synchronization, and Python syntax checks before restarting only services that were already running. Every restarted service must pass its localhost HTTP health check. A failed dependency or syntax check leaves existing service processes running and does not restart them. If a measurement is started during the update, the final guard leaves the updated main service process running without restarting it and reports failure.

Stop a measurement in the GUI and confirm it is idle before updating. Do not schedule `dmps update` from cron or a systemd timer. Do not manually run a second hardware GUI beside `tdmps@USER.service`.

Service logs are available with `dmps log` and `dmps viewer log`. The existing `tdmps@USER.service` name is retained for compatibility; the viewer follows it as `tdmps-viewer@USER.service`.

## Tailscale exposure

The application services remain localhost-only. To expose them to the tailnet, install and authenticate Tailscale, then enable the supplied proxy unit:

```bash
sudo tailscale up
sudo systemctl enable --now "tdmps-serve@$(whoami).service"
tailscale serve status
dmps url
```

The default routes are:

```text
https://customer-host.tailnet-name.ts.net/gui
https://customer-host.tailnet-name.ts.net:8443/offline_inversion_viewer
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

The installer intentionally refuses to proceed while the legacy service is active, preventing two hardware controllers from binding the same port or opening the same devices. Keep the old directory as a backup until the new main GUI, offline viewer, settings, and historical logs have been verified.

The repository no longer tracks Python bytecode. Commit the accompanying `__pycache__` deletions and `.gitignore` update before cloning customer systems; otherwise imported bytecode can make future pulls dirty.
