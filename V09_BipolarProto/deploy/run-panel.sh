#!/usr/bin/env bash
set -euo pipefail

: "${APP_DIR:?APP_DIR is not configured}"

case "${1:-}" in
    main)
        app="gui.py"
        port=5006
        origin="${DMPS_WEBSOCKET_ORIGIN_MAIN:?main websocket origin is not configured}"
        extra=(--keep-alive 10000 --reuse-sessions --check-unused-sessions 60000 --unused-session-lifetime 3600000)
        ;;
    *)
        printf 'Usage: %s main\n' "$0" >&2
        exit 2
        ;;
esac

cd "${DMPS_STATE_DIR:-${APP_DIR}}"
exec "${APP_DIR}/.venv/bin/panel" serve "${APP_DIR}/${app}" \
    --address 127.0.0.1 \
    --port "${port}" \
    --allow-websocket-origin "127.0.0.1:${port}" \
    --allow-websocket-origin "${origin}" \
    "${extra[@]}"
