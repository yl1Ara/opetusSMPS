#!/usr/bin/env bash
set -u

if [[ "${SERVICE_RESULT:-}" == "exec-condition" ]]; then
    printf 'Force-safe skipped: service startup was rejected before hardware access.\n'
    exit 0
fi

: "${APP_DIR:?APP_DIR is not configured}"
cd "${DMPS_STATE_DIR:-${APP_DIR}}" || exit 0
export PYTHONPATH="${APP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${APP_DIR}/.venv/bin/python" "${APP_DIR}/deploy/force-safe.py"
