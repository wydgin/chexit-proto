#!/usr/bin/env bash
# Local API: http://127.0.0.1:8000  —  Vite proxies /api → here (see vite.config.ts).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
export PYTHONPATH=.
exec uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
