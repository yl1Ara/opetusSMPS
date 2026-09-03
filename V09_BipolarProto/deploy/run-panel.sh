#!/usr/bin/env bash
set -euo pipefail

: "${APP_DIR:?APP_DIR is not configured}"

case "${1:-}" in
    main)
        app="gui.py"
        port=5006
        origin="${DMPS_WEBSOCKET_ORIGIN_MAIN:?main websocket origin is not configured}"
        extra=(--reuse-sessions --keep-alive 10000 --check-unused-sessions 60000 --unused-session-lifetime 604800000)
        ;;
    viewer)
        app="offline_inversion_viewer.py"
        port=5007
        origin="${DMPS_WEBSOCKET_ORIGIN_VIEWER:?viewer websocket origin is not configured}"
        extra=()
        ;;
    *)
        printf 'Usage: %s {main|viewer}\n' "$0" >&2
        exit 2
        ;;
esac

cd "${APP_DIR}"
exec "${APP_DIR}/.venv/bin/panel" serve "${app}" \
    --address 127.0.0.1 \
    --port "${port}" \
    --allow-websocket-origin "127.0.0.1:${port}" \
    --allow-websocket-origin "${origin}" \
    "${extra[@]}"
