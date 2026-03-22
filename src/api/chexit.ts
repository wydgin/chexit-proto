const defaultBase = 'http://127.0.0.1:8000';

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

function apiBase(): string {
  const base = import.meta.env.VITE_CHEXIT_API_URL?.trim();
  return base && base.length > 0 ? base.replace(/\/$/, '') : defaultBase;
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

function networkErrorHint(base: string): string {
  const httpsPage =
    typeof window !== 'undefined' && window.location.protocol === 'https:';
  const httpApi = base.startsWith('http://');
  if (httpsPage && httpApi) {
    return (
      `Cannot call ${base} from an HTTPS site (browser blocks mixed content). ` +
      `Use an HTTPS API URL in VITE_CHEXIT_API_URL, or test Analyze from local dev (http://localhost:5173).`
    );
  }
  return (
    `Cannot reach ${base}/predict. Start the API (chexit-backend): ./run_dev.sh — then ` +
    `VITE_CHEXIT_API_URL should be http://127.0.0.1:8000. Open ${base}/docs to verify. ` +
    `If Analyze worked then failed: avoid uvicorn --reload while running long /predict (saving a file restarts the server and drops the request).`
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
  const base = apiBase();
  const formData = new FormData();
  formData.append('file', file);

  let res: Response;
  try {
    res = await fetch(`${base}/predict`, {
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
      throw new Error(networkErrorHint(base));
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
