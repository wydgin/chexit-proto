#!/usr/bin/env bash
# Start the Chexit API locally (creates .venv and installs deps on first run).
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "Creating .venv …"
  python3 -m venv .venv
fi

echo "Installing dependencies (first run may take a few minutes) …"
.venv/bin/pip install -q -r requirements.txt

echo "Starting http://127.0.0.1:8000 (Ctrl+C to stop)"
echo "Note: /predict can take minutes (TensorFlow + Score-CAM). Default is NO --reload so file saves do not restart the server mid-request."
echo "For auto-reload while editing API code: CHEXIT_UVICORN_RELOAD=1 $0"
if [[ "${CHEXIT_UVICORN_RELOAD:-}" == "1" ]]; then
  exec .venv/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
fi
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
