#!/usr/bin/env bash
set -u

: "${APP_DIR:?APP_DIR is not configured}"
cd "${DMPS_STATE_DIR:-${APP_DIR}}" || exit 0
export PYTHONPATH="${APP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${APP_DIR}/.venv/bin/python" "${APP_DIR}/deploy/force-safe.py"
