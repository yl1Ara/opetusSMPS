#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s --origin HOST[:PORT] [--state-dir PATH] [--user USER]\n' "$0"
}

user_name="${SUDO_USER:-$USER}"
main_origin=""
state_dir=""
while (($#)); do
    case "$1" in
        --origin) main_origin="${2:?missing origin}"; shift 2 ;;
        --state-dir) state_dir="${2:?missing state directory}"; shift 2 ;;
        --user) user_name="${2:?missing user}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

if [[ ! "${main_origin}" =~ ^([A-Za-z0-9-]+\.)*[A-Za-z0-9-]+(:[0-9]{1,5})?$ ]]; then
    printf 'Use an exact websocket origin such as host.example.ts.net (no scheme, path, or wildcard).\n' >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
app_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_root="$(git -C "${app_dir}" rev-parse --show-toplevel)"
state_dir="${state_dir:-${app_dir}}"
if [[ "${app_dir}" != "${repo_root}/V09_BipolarProto" || ! -f "${app_dir}/gui.py" ]]; then
    printf 'Expected this checkout at repository-root/V09_BipolarProto; found %s\n' "${app_dir}" >&2
    exit 1
fi
if [[ "${state_dir}" != /* || ! -d "${state_dir}" || ! -w "${state_dir}" ]]; then
    printf 'State directory must be an existing writable absolute path: %s\n' "${state_dir}" >&2
    exit 1
fi
if [[ "$(id -un)" != "${user_name}" ]]; then
    printf 'Run this script as %s (it invokes sudo only for system files).\n' "${user_name}" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is required; install it before running this installer.\n' >&2
    exit 1
fi
if systemctl is-active --quiet tdmps.service; then
    printf 'Legacy tdmps.service is running. Stop the measurement, zero HV, then disable that service before installing.\n' >&2
    exit 1
fi

cd "${app_dir}"
uv sync --locked
uv pip install --python "${app_dir}/.venv/bin/python" -r requirements-hardware.txt
"${app_dir}/.venv/bin/python" deploy/compile-check.py gui.py DmpsControl deploy/force-safe.py deploy/zero-bipolar.py deploy/check-health.py

config_tmp="$(mktemp)"
trap 'rm -f "${config_tmp}"' EXIT
printf 'APP_DIR="%s"\nDMPS_STATE_DIR="%s"\nDMPS_WEBSOCKET_ORIGIN_MAIN="%s"\n' \
    "${app_dir}" "${state_dir}" "${main_origin}" >"${config_tmp}"

sudo install -d -o root -g root -m 0755 /etc/dmps /usr/local/libexec
sudo install -o root -g root -m 0644 "${config_tmp}" "/etc/dmps/${user_name}.env"
sudo install -o root -g root -m 0755 "${script_dir}/run-panel.sh" /usr/local/libexec/dmps-run-panel
sudo install -o root -g root -m 0755 "${script_dir}/run-force-safe.sh" /usr/local/libexec/dmps-force-safe
sudo install -o root -g root -m 0755 "${script_dir}/run-zero-bipolar.sh" /usr/local/libexec/dmps-zero-bipolar
sudo install -o root -g root -m 0644 "${script_dir}/tdmps@.service" /etc/systemd/system/tdmps@.service
sudo install -o root -g root -m 0644 "${script_dir}/tdmps-serve@.service" /etc/systemd/system/tdmps-serve@.service
install -d -m 0755 "${HOME}/bin"
install -m 0755 "${script_dir}/dmps" "${HOME}/bin/dmps"
install -m 0644 "${script_dir}/dmps-completion.bash" "${HOME}/.dmps-complete"
if ! command grep -qxF 'source ~/.dmps-complete' "${HOME}/.bashrc"; then
    printf '\nsource ~/.dmps-complete\n' >>"${HOME}/.bashrc"
fi

sudo systemctl daemon-reload
sudo systemctl disable --now "tdmps-viewer@${user_name}.service" >/dev/null 2>&1 || true
sudo systemctl enable --now "tdmps@${user_name}.service"
"${HOME}/bin/dmps" health

printf 'Installed TDMPS from %s\n' "${app_dir}"
printf 'Main:   http://127.0.0.1:5006/gui\n'
