#!/usr/bin/env bash
# One-command installer for inkling-1m-mlx on Apple Silicon (macOS).
# Sets up Python venv, installs MLX + server deps, clones oMLX (inkling2_loop loader),
# and downloads mlx-community/Inkling-2-Mini-oQ4e-mtp (~16.9 GB) from Hugging Face.
set -euo pipefail

MODEL_ID="${MODEL_ID:-mlx-community/Inkling-2-Mini-oQ4e-mtp}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

echo "=== inkling-1m-mlx Installer ==="

# 1. Verify Apple Silicon macOS
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: inkling-1m-mlx requires Apple Silicon (macOS arm64)." >&2
  exit 1
fi

# 2. Find Python >= 3.11
PY=""
for cand in python3.13 python3.12 python3.11 /opt/homebrew/bin/python3 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver="$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    major="${ver%%.*}"; minor="${ver##*.}"
    if (( major == 3 && minor >= 11 )); then
      PY="$(command -v "$cand")"
      break
    fi
  fi
done
if [[ -z "$PY" ]]; then
  echo "ERROR: Python >= 3.11 is required. Install via: brew install python@3.13" >&2
  exit 1
fi
echo "[1/4] Using Python: $PY ($("$PY" --version))"

# 3. Create venv and install dependencies
if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
echo "[2/4] Installing MLX & server dependencies..."
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

# 4. Clone oMLX (provides the inkling2_loop MLX loader)
if [[ ! -d omlx ]]; then
  echo "[3/4] Cloning jundot/omlx (shallow)..."
  git clone --depth 1 -q https://github.com/jundot/omlx.git omlx
else
  echo "[3/4] omlx/ already present."
fi

# 5. Pre-download weights from Hugging Face
if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  echo "[4/4] Downloading $MODEL_ID (~16.9 GB, resumes automatically if cached)..."
  .venv/bin/python -c "from huggingface_hub import snapshot_download; p = snapshot_download('$MODEL_ID'); print('Cached at:', p)"
else
  echo "[4/4] Skipping weight download (SKIP_DOWNLOAD=1)."
fi

chmod +x chat.py server.py run_server.sh 2>/dev/null || true

cat <<'EOF'

=== Installation Complete! ===

1) Start the OpenAI-compatible server (for OpenCode, Cursor, etc.):
   ./run_server.sh                       # 256K ctx, 2.06-bit SRFT KV, port 8080
   ./run_server.sh --max-ctx 1048576     # Full 1M ctx in ~19.5 GiB on 32 GB Mac!

2) Or chat directly in your terminal:
   OMLX_PATH=./omlx .venv/bin/python chat.py --kv lm2_srft --max-ctx 1048576

3) For OpenCode:
   Copy opencode.json.example -> opencode.json (or ~/.config/opencode/opencode.json),
   start ./run_server.sh, and select "Inkling-2-Mini (2-bit SRFT, 1M ctx)" in /models.
EOF
