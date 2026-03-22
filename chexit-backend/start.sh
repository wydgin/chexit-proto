#!/usr/bin/env bash
# Render: use Start Command `bash start.sh` (Root Directory = chexit-backend).
# Avoids `cd chexit-backend` when cwd is already chexit-backend (would enter the
# pip shim folder chexit-backend/chexit-backend/ and break `import app`).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH=.
# CPU-only servers (Render): TensorFlow otherwise probes CUDA and logs cuInit errors.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:?}"
