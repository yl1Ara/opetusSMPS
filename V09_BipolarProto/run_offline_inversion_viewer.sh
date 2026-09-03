#!/usr/bin/env bash
set -euo pipefail

config="/etc/dmps/${USER}.env"
if [[ ! -r "${config}" ]]; then
    printf 'Missing %s; run deploy/install-services.sh first.\n' "${config}" >&2
    exit 1
fi
source "${config}"
exec "${APP_DIR}/deploy/run-panel.sh" viewer
