export type PredictResponse = {
  diagnosis: string;
  risk_score: number;
  confidence_label: string;
  heatmap: string;
};

export type PredictUiState = {
  loading: boolean;
  error: string | null;
  data: PredictResponse | null;
};

/**
 * Predict URL:
 * - No VITE_CHEXIT_API_URL → always same-origin `/api/predict` (Vite dev + preview proxy → :8000).
 *   Works for any dev hostname (localhost, 127.0.0.1, Cursor tunnel, etc.).
 * - VITE_CHEXIT_API_URL set → direct URL (required for Vercel / production API).
 */
function predictUrl(): string {
  const trimmed = import.meta.env.VITE_CHEXIT_API_URL?.trim();
  if (trimmed) {
    return `${trimmed.replace(/\/$/, '')}/predict`;
  }
  return '/api/predict';
}

function apiLabelForErrors(): string {
  const trimmed = import.meta.env.VITE_CHEXIT_API_URL?.trim();
  if (trimmed) {
    return trimmed.replace(/\/$/, '');
  }
  return '/api (Vite → http://127.0.0.1:8000)';
}

function parseErrorDetail(body: unknown): string {
  if (!body || typeof body !== 'object') return 'Request failed';
  const d = (body as { detail?: unknown }).detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d
      .map((e) => (typeof e === 'object' && e && 'msg' in e ? String((e as { msg: string }).msg) : String(e)))
      .join(', ');
  }
  return 'Request failed';
}

function networkErrorHint(label: string): string {
  const httpsPage =
    typeof window !== 'undefined' && window.location.protocol === 'https:';
  const directHttp = (import.meta.env.VITE_CHEXIT_API_URL?.trim() ?? '').startsWith(
    'http://',
  );
  if (httpsPage && directHttp) {
    return (
      `Cannot call ${label} from an HTTPS page (mixed content). ` +
      `Remove VITE_CHEXIT_API_URL from .env so dev uses the Vite /api proxy, or use an https:// API URL, or open the app at http://localhost:5173.`
    );
  }
  return (
    `Cannot reach the API (${label}). Start FastAPI: cd chexit-backend && ./run_dev.sh — check http://127.0.0.1:8000/docs. ` +
    `Use npm run dev or npm run preview (not opening dist/ directly) so /api proxies to port 8000. ` +
    `Unset VITE_CHEXIT_API_URL for local proxy. Avoid uvicorn --reload during long /predict.`
  );
}

/** Score-CAM + TF can run several minutes; browsers rarely cancel, but proxies might. */
const PREDICT_TIMEOUT_MS = 10 * 60 * 1000;

function predictAbortSignal(): AbortSignal {
  if (typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function') {
    return AbortSignal.timeout(PREDICT_TIMEOUT_MS);
  }
  const c = new AbortController();
  setTimeout(() => c.abort(), PREDICT_TIMEOUT_MS);
  return c.signal;
}

function isAbortError(e: unknown): boolean {
  if (e instanceof DOMException && e.name === 'AbortError') {
    return true;
  }
  return e instanceof Error && e.name === 'AbortError';
}

export async function predictImage(file: File): Promise<PredictResponse> {
  const url = predictUrl();
  const label = apiLabelForErrors();
  const formData = new FormData();
  formData.append('file', file);

  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      body: formData,
      signal: predictAbortSignal(),
    });
  } catch (e) {
    if (isAbortError(e)) {
      throw new Error(
        `Analyze timed out after ${Math.round(PREDICT_TIMEOUT_MS / 60000)} minutes, or the request was cancelled. ` +
          `Try a smaller image, or run the API without --reload (see chexit-backend/run_dev.sh).`,
      );
    }
    const failedFetch =
      e instanceof TypeError &&
      (e.message === 'Failed to fetch' || e.message.includes('Load failed'));
    if (failedFetch) {
      throw new Error(networkErrorHint(label));
    }
    throw e;
  }

  if (!res.ok) {
    let message = res.statusText;
    try {
      message = parseErrorDetail(await res.json());
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }

  return res.json() as Promise<PredictResponse>;
}
