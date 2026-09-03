#!/usr/bin/env bash
set -euo pipefail

app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
origin="${ONLINE_VIEWER_ORIGIN:-localhost:5007}"

cd "${app_dir}"
exec uv run panel serve online_inversion_viewer.py \
    --address 127.0.0.1 \
    --port 5007 \
    --allow-websocket-origin "127.0.0.1:5007" \
    --allow-websocket-origin "${origin}"
