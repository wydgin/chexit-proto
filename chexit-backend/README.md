# Chexit API (FastAPI)

## Local setup

1. **Python 3.11** (see `.python-version`). From this directory:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Assets** — U-Net weights under `../assets/models/` (gitignored). Either copy `unet_lung_seg_best.keras` locally or let startup download from Google Drive (leave `CHEXIT_SKIP_GDOWN` unset). MobileNet fold weights live in `../assets/mobilenet_tb_output/weights/` (tracked in git).

3. **Run the server**

   ```bash
   ./run_dev.sh
   ```

   Or from the **monorepo root**:

   ```bash
   npm run dev:api
   ```

   **One command (UI + API):** from monorepo root, `npm run dev:stack`.

4. **Frontend** — `npm run dev` (or `dev:stack`). Open **http://localhost:5173/dashboard**. The app calls `/api/predict`, which Vite proxies to `http://127.0.0.1:8000/predict`. `npm run preview` uses the same proxy on localhost.

5. **Docs** — Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The `/predict` **heatmap** PNG is composited at the **same width×height** as the uploaded study (CAM is computed at 224² then upsampled and blended on the full-resolution grayscale base).

### Optional env

| Variable | Purpose |
|----------|---------|
| `CHEXIT_SKIP_GDOWN=1` | Do not download U-Net from Drive (must have weights on disk). |
| `CHEXIT_SKIP_SCORECAM=1` | Fast lung-mask heatmap only (avoids long Score-CAM on CPU). |
| `CHEXIT_MAX_CXR_EDGE=2048` | Downscale large CXRs before the pipeline (longest side). |

### Tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

From monorepo root: `npm run test:backend`.
