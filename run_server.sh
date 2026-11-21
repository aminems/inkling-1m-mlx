#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OMLX_PATH="${OMLX_PATH:-$DIR/omlx}" exec "$DIR/.venv/bin/python" "$DIR/server.py" "$@"
