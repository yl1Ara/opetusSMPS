#!/usr/bin/env bash
set -euo pipefail

: "${APP_DIR:?APP_DIR is not configured}"
cd "${DMPS_STATE_DIR:-${APP_DIR}}"
export PYTHONPATH="${APP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${APP_DIR}/.venv/bin/python" "${APP_DIR}/deploy/zero-bipolar.py"
