import * as React from 'react';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import Collapse from '@mui/material/Collapse';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import LinearProgress from '@mui/material/LinearProgress';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import ChevronLeft from '@mui/icons-material/ChevronLeft';
import ChevronRight from '@mui/icons-material/ChevronRight';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import FlagOutlinedIcon from '@mui/icons-material/FlagOutlined';
import type { Theme } from '@mui/material/styles';
import type { PredictUiState } from '../../api/chexit';
import { fetchLatestUpload, type UploadRecord } from '../../api/chexit';

type FlagCategory = 'false-positive' | 'false-negative' | 'wrong-region' | 'image-quality' | 'other';

type AnomalyFlag = {
  categories: FlagCategory[];
  note: string;
  savedAt: string;
  snapshot: {
    fileName: string;
    diagnosis: string;
    riskScore: number | null;
    confidence: string;
  };
};

type FlagMap = Record<string, AnomalyFlag>;

const FLAG_STORAGE_KEY = 'chexit:anomaly-flags:v1';

const FLAG_CATEGORIES: { id: FlagCategory; label: string }[] = [
  { id: 'false-positive', label: 'False positive' },
  { id: 'false-negative', label: 'False negative' },
  { id: 'wrong-region', label: 'Wrong region' },
  { id: 'image-quality', label: 'Image quality' },
  { id: 'other', label: 'Other' },
];

const FLAG_CATEGORY_IDS = FLAG_CATEGORIES.map((c) => c.id);

function categoryLabel(id: FlagCategory): string {
  return FLAG_CATEGORIES.find((c) => c.id === id)?.label ?? id;
}

function isFlagCategory(value: unknown): value is FlagCategory {
  return typeof value === 'string' && (FLAG_CATEGORY_IDS as readonly string[]).includes(value);
}

/** Accept new schema (`categories[]`) and the legacy single-`category` shape side-by-side. */
function migrateFlag(raw: unknown): AnomalyFlag | null {
  if (!raw || typeof raw !== 'object') return null;
  const f = raw as Record<string, unknown>;
  let categories: FlagCategory[] = [];
  if (Array.isArray(f.categories)) {
    categories = f.categories.filter(isFlagCategory);
  } else if (isFlagCategory(f.category)) {
    categories = [f.category];
  }
  if (categories.length === 0) return null;
  const snapshotRaw = (f.snapshot && typeof f.snapshot === 'object'
    ? (f.snapshot as Record<string, unknown>)
    : {}) as Record<string, unknown>;
  return {
    categories,
    note: typeof f.note === 'string' ? f.note : '',
    savedAt: typeof f.savedAt === 'string' ? f.savedAt : new Date().toISOString(),
    snapshot: {
      fileName: typeof snapshotRaw.fileName === 'string' ? snapshotRaw.fileName : 'image',
      diagnosis: typeof snapshotRaw.diagnosis === 'string' ? snapshotRaw.diagnosis : '',
      riskScore:
        typeof snapshotRaw.riskScore === 'number' && Number.isFinite(snapshotRaw.riskScore)
          ? snapshotRaw.riskScore
          : null,
      confidence: typeof snapshotRaw.confidence === 'string' ? snapshotRaw.confidence : '',
    },
  };
}

function loadFlags(): FlagMap {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.localStorage.getItem(FLAG_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== 'object') return {};
    const out: FlagMap = {};
    for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
      const migrated = migrateFlag(value);
      if (migrated) out[key] = migrated;
    }
    return out;
  } catch {
    return {};
  }
}

function saveFlagsToStorage(map: FlagMap): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(FLAG_STORAGE_KEY, JSON.stringify(map));
  } catch {
    /* localStorage full / disabled — silently ignore in prototype */
  }
}

/** Empty image slot: glass-style panel for light + dark (replaces solid black placeholders). */
function liquidGlassImagePlaceholderSx(theme: Theme) {
  return {
    borderRadius: 2,
    width: '100%',
    aspectRatio: '3 / 4',
    position: 'relative' as const,
    overflow: 'hidden',
    border: '1px solid',
    borderColor: 'rgba(148, 163, 184, 0.38)',
    backgroundColor: 'rgba(255, 255, 255, 0.38)',
    backgroundImage:
      'linear-gradient(135deg, rgba(255,255,255,0.72) 0%, rgba(255,255,255,0.12) 42%, rgba(186, 230, 253, 0.18) 100%)',
    backdropFilter: 'blur(14px)',
    WebkitBackdropFilter: 'blur(14px)',
    boxShadow:
      '0 10px 40px rgba(15, 23, 42, 0.07), inset 0 1px 0 rgba(255,255,255,0.75), inset 0 -1px 0 rgba(148,163,184,0.18)',
    ...theme.applyStyles('dark', {
      borderColor: 'rgba(148, 163, 184, 0.22)',
      backgroundColor: 'rgba(15, 23, 42, 0.42)',
      backgroundImage:
        'linear-gradient(145deg, rgba(255,255,255,0.14) 0%, rgba(30,41,59,0.55) 45%, rgba(59,130,246,0.1) 100%)',
      boxShadow:
        '0 12px 48px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.1), inset 0 -1px 0 rgba(0,0,0,0.35)',
    }),
  };
}

/**
 * Truncate long strings in the MIDDLE — useful for DICOM UIDs where the
 * leading scheme + trailing identifier are both informative.
 *   "raw_1.2.840.114062.2.192.168.196.13.2013.9.27.9.58.39.43467890"
 *   → "raw_1.2.840.114…39.43467890"  (max=28)
 */
function truncateMiddle(value: string, max: number): string {
  if (value.length <= max) return value;
  const head = Math.ceil((max - 1) / 2);
  const tail = max - 1 - head;
  return `${value.slice(0, head)}…${value.slice(-tail)}`;
}

function formatUploadedAt(isoText: string | null): string {
  if (!isoText) return 'Uploaded • recently';
  const d = new Date(isoText);
  const now = Date.now();
  const diffMs = now - d.getTime();
  if (!Number.isFinite(diffMs)) return 'Uploaded • recently';
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return 'Uploaded • just now';
  if (diffMins === 1) return 'Uploaded • 1 min ago';
  return `Uploaded • ${diffMins} min ago`;
}

type FeaturesProps = {
  /** When set, this URL is shown in the Input X-Ray preview (e.g. right after upload). */
  previewImageUrl?: string | null;
  /** Local `blob:` URL from file picker — shown immediately before any Firebase upload. */
  localPreviewUrl?: string | null;
  predictUi: PredictUiState;
  onNavigateIndex?: (nextIndex: number) => void;
};

// function contributionBars(riskScore: number) {
//   const r = Math.min(100, Math.max(0, Math.round(riskScore)));
//   return [
//     { label: 'MobileNet-V2', value: Math.min(100, Math.round(r * 0.88)), barColor: '#6366f1' },
//     { label: 'ResNet-50', value: Math.min(100, Math.round(r * 1.02)), barColor: '#3b82f6' },
//     { label: 'DenseNet-121', value: r, barColor: '#22c55e' },
//   ];
// }

// function contributionBars(_riskScore: number) {
//   // Backend currently runs only MobileNetV2.
//   // Keep other planned models visible but zeroed until backend supports them.
//   return [
//     { label: 'MobileNetV2', value: 100, barColor: '#6366f1' },
//     { label: 'EfficientNetB2', value: 0, barColor: '#f59e0b' },
//     { label: 'DenseNet121', value: 0, barColor: '#22c55e' },
//   ];
// }

function contributionBars(contrib?: {
  mobilenetv2: number;
  efficientnetb2: number;
  densenet121: number;
}) {
  if (!contrib) {
    return [
      { label: 'MobileNetV2', value: 0, barColor: '#6366f1' },
      { label: 'EfficientNetB2', value: 0, barColor: '#f59e0b' },
      { label: 'DenseNet121', value: 0, barColor: '#22c55e' },
    ];
  }

  return [
    { label: 'MobileNetV2', value: Math.round(contrib.mobilenetv2), barColor: '#6366f1' },
    { label: 'EfficientNetB2', value: Math.round(contrib.efficientnetb2), barColor: '#f59e0b' },
    { label: 'DenseNet121', value: Math.round(contrib.densenet121), barColor: '#22c55e' },
  ];
}

export default function Features({
  previewImageUrl,
  localPreviewUrl,
  predictUi,
  onNavigateIndex,
}: FeaturesProps) {
  const pageBg = 'background.default';
  const cardBg = 'background.default';
  const cardBorder = 'divider';
  const mutedText = '#9ca3af';

  const [latestUpload, setLatestUpload] = React.useState<UploadRecord | null>(null);
  const [previewLoadFailed, setPreviewLoadFailed] = React.useState(false);

  React.useEffect(() => {
    let active = true;
    fetchLatestUpload()
      .then((record) => {
        if (active) {
          setLatestUpload(record);
        }
      })
      .catch(() => {
        if (active) {
          setLatestUpload(null);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const remotePreviewSrc = previewImageUrl ?? latestUpload?.downloadURL ?? null;
  const hasBatch = predictUi.items.length > 0;
  const safeIndex = Math.max(0, Math.min(predictUi.currentIndex, Math.max(0, predictUi.items.length - 1)));
  const selectedItem = hasBatch ? predictUi.items[safeIndex] : null;
  const previewSrc = selectedItem?.localPreviewUrl ?? localPreviewUrl ?? remotePreviewSrc;
  React.useEffect(() => {
    setPreviewLoadFailed(false);
  }, [previewSrc]);
  const showPreview = Boolean(previewSrc) && !previewLoadFailed;
  const hasRemoteImage = Boolean(remotePreviewSrc);
  const previewSubheader = hasBatch
    ? `Image ${safeIndex + 1} of ${predictUi.items.length} • ${truncateMiddle(
        selectedItem?.fileName ?? 'selected',
        30,
      )}`
    : previewLoadFailed
      ? 'No image uploaded'
    : localPreviewUrl && !previewImageUrl
      ? 'Preview • not uploaded yet'
      : previewImageUrl
        ? 'Uploaded • just now'
        : hasRemoteImage
          ? formatUploadedAt(latestUpload?.uploadedAt ?? null)
          : 'No image uploaded';

  const pred = selectedItem?.result ?? predictUi.data;
  const heatmapB64 = pred?.heatmap?.trim() ?? '';
  const hasHeatmap = heatmapB64 !== '';
  const heatmapSrc = hasHeatmap ? `data:image/png;base64,${heatmapB64}` : '';
  const riskPct =
    pred != null && Number.isFinite(Number(pred.risk_score))
      ? Math.round(Number(pred.risk_score))
      : null;
  const diagnosisLine = pred?.diagnosis?.trim() ?? '';
  const confidenceLine = pred?.confidence_label?.trim() ?? '';
  const isHighRisk = Boolean(
    /positive/i.test(diagnosisLine) ||
      /high/i.test(confidenceLine) ||
      (riskPct != null && riskPct >= 50),
  );
  /** Remount diagnosis UI when a new API payload arrives so labels/scores never appear stale. */
  const analysisVersionKey = pred
    ? `${diagnosisLine}|${riskPct ?? ''}|${confidenceLine}`
    : 'no-analysis';
  const canGoPrev = hasBatch && safeIndex > 0;
  const canGoNext = hasBatch && safeIndex < predictUi.items.length - 1;
  // const modelRows = pred && riskPct != null ? contributionBars(Number(pred.risk_score)) : [
  //   { label: 'MobileNet-V2', value: 60, barColor: '#6366f1' },
  //   { label: 'ResNet-50', value: 80, barColor: '#3b82f6' },
  //   { label: 'DenseNet-121', value: 75, barColor: '#22c55e' },
  // ];
  // const modelRows = pred && riskPct != null
  const modelRows = contributionBars(pred?.model_contributions);

  React.useEffect(() => {
    if (!hasBatch || !onNavigateIndex) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'ArrowLeft' && canGoPrev) {
        event.preventDefault();
        onNavigateIndex(safeIndex - 1);
      } else if (event.key === 'ArrowRight' && canGoNext) {
        event.preventDefault();
        onNavigateIndex(safeIndex + 1);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [hasBatch, onNavigateIndex, canGoPrev, canGoNext, safeIndex]);

  // ---- Anomaly flags (simplest opt-in, persisted to localStorage) ----
  /** Keyed by filename: stable enough for a prototype, lets re-running the same image keep its note. */
  const currentFlagKey: string | null = pred
    ? (selectedItem?.fileName ?? latestUpload?.fileName ?? null)
    : null;
  const [flags, setFlags] = React.useState<FlagMap>(() => loadFlags());
  const [flagFormOpen, setFlagFormOpen] = React.useState(false);
  const [flagFormCategories, setFlagFormCategories] = React.useState<FlagCategory[]>([]);
  const [flagFormNote, setFlagFormNote] = React.useState('');
  const [logOpen, setLogOpen] = React.useState(false);
  const currentFlag: AnomalyFlag | null = currentFlagKey ? (flags[currentFlagKey] ?? null) : null;
  const flagCount = Object.keys(flags).length;

  /** Close the inline form whenever the user navigates to a different image. */
  React.useEffect(() => {
    setFlagFormOpen(false);
  }, [currentFlagKey]);

  const openFlagForm = React.useCallback(() => {
    if (currentFlag) {
      setFlagFormCategories(currentFlag.categories);
      setFlagFormNote(currentFlag.note);
    } else {
      setFlagFormCategories([]);
      setFlagFormNote('');
    }
    setFlagFormOpen(true);
  }, [currentFlag]);

  const toggleFormCategory = React.useCallback((id: FlagCategory) => {
    setFlagFormCategories((prev) =>
      prev.includes(id) ? prev.filter((c) => c !== id) : [...prev, id],
    );
  }, []);

  const saveCurrentFlag = React.useCallback(() => {
    if (!currentFlagKey || !pred || flagFormCategories.length === 0) return;
    /** Preserve the user-facing order of FLAG_CATEGORIES regardless of click order. */
    const orderedCategories = FLAG_CATEGORIES.map((c) => c.id).filter((id) =>
      flagFormCategories.includes(id),
    );
    const next: FlagMap = {
      ...flags,
      [currentFlagKey]: {
        categories: orderedCategories,
        note: flagFormNote.trim(),
        savedAt: new Date().toISOString(),
        snapshot: {
          fileName: selectedItem?.fileName ?? latestUpload?.fileName ?? 'image',
          diagnosis: diagnosisLine,
          riskScore: riskPct,
          confidence: confidenceLine,
        },
      },
    };
    setFlags(next);
    saveFlagsToStorage(next);
    setFlagFormOpen(false);
  }, [
    currentFlagKey,
    pred,
    flags,
    flagFormCategories,
    flagFormNote,
    selectedItem,
    latestUpload,
    diagnosisLine,
    riskPct,
    confidenceLine,
  ]);

  const removeFlag = React.useCallback(
    (key: string) => {
      const next = { ...flags };
      delete next[key];
      setFlags(next);
      saveFlagsToStorage(next);
      if (key === currentFlagKey) {
        setFlagFormOpen(false);
      }
    },
    [flags, currentFlagKey],
  );

  return (
    <Box sx={{ bgcolor: pageBg }}>
      <Container id="features" maxWidth="lg" sx={{ pt: { xs: 4, md: 5 }, pb: { xs: 6, md: 7 } }}>
        <Stack
          direction="row"
          alignItems="center"
          justifyContent="space-between"
          spacing={1}
          sx={{ mb: 3, flexWrap: 'wrap', rowGap: 1 }}
        >
          <Typography
            variant="h6"
            sx={{
              color: 'text.primary',
              fontWeight: 600,
              letterSpacing: 0.5,
            }}
          >
            AI-assisted TB overview
          </Typography>
          {flagCount > 0 ? (
            <Button
              size="small"
              variant="outlined"
              startIcon={<FlagOutlinedIcon fontSize="small" />}
              onClick={() => setLogOpen(true)}
            >
              Anomaly log ({flagCount})
            </Button>
          ) : null}
        </Stack>

        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            gap: { xs: 2.5, md: 3 },
            alignItems: 'stretch',
          }}
        >
          {/* Input image */}
          <Card
            variant="outlined"
            sx={{
              flex: 1,
              // Critical: lets the flex item shrink below its content's intrinsic
              // width so long DICOM UID filenames in the subheader wrap inside
              // the card instead of pushing the layout (and the chevrons) out.
              minWidth: 0,
              bgcolor: cardBg,
              borderColor: cardBorder,
              color: 'text.primary',
              borderRadius: 2,
            }}
          >
            <CardHeader
              title={
                <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
                  <Typography
                    component="span"
                    sx={{ fontSize: 14, fontWeight: 600, flexShrink: 0 }}
                  >
                    Input X-Ray
                  </Typography>
                  {hasBatch ? (
                    <Stack
                      direction="row"
                      alignItems="center"
                      spacing={0.5}
                      sx={{ flexShrink: 0 }}
                    >
                      <IconButton
                        size="small"
                        disabled={!canGoPrev}
                        onClick={() => onNavigateIndex?.(safeIndex - 1)}
                      >
                        <ChevronLeft fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        disabled={!canGoNext}
                        onClick={() => onNavigateIndex?.(safeIndex + 1)}
                      >
                        <ChevronRight fontSize="small" />
                      </IconButton>
                    </Stack>
                  ) : null}
                </Stack>
              }
              subheader={previewSubheader}
              sx={{
                pb: 1,
                // Lets the content column shrink so long DICOM UIDs wrap inside
                // the card instead of stretching it out.
                '& .MuiCardHeader-content': {
                  minWidth: 0,
                },
                '& .MuiCardHeader-title': {
                  width: '100%',
                },
                '& .MuiCardHeader-subheader': {
                  color: mutedText,
                  fontSize: 12,
                  // Keep the filename inline with "Image N of M". Long DICOM
                  // UIDs are already middle-truncated to ~30 chars upstream;
                  // this is the belt-and-suspenders fallback if a card is
                  // narrower than expected.
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                },
              }}
            />
            <CardContent sx={{ pt: 1 }}>
              {showPreview ? (
                <Box
                  component="img"
                  key={previewSrc}
                  src={previewSrc ?? undefined}
                  alt="Input chest X-ray"
                  onError={() => setPreviewLoadFailed(true)}
                  sx={{
                    borderRadius: 2,
                    bgcolor: 'background.default',
                    border: '1px solid',
                    borderColor: cardBorder,
                    width: '100%',
                    maxHeight: { xs: '52vh', md: '62vh' },
                    height: 'auto',
                    objectFit: 'contain',
                    display: 'block',
                    mx: 'auto',
                  }}
                />
              ) : (
                <Box aria-hidden sx={(theme) => liquidGlassImagePlaceholderSx(theme)} />
              )}
            </CardContent>
          </Card>
{/* Diagnosis */}
          <Card
            variant="outlined"
            sx={{
              flex: 1.1,
              minWidth: 0,
              bgcolor: cardBg,
              borderColor: cardBorder,
              color: 'text.primary',
              borderRadius: 2,
            }}
          >
  <CardContent
    sx={{
      pb: 2,
      display: 'flex',
      flexDirection: 'column',
      gap: 3, // uniform vertical spacing between blocks
    }}
  >
    {predictUi.loading ? (
      <LinearProgress sx={{ borderRadius: 1 }} />
    ) : null}
    {/* Top: diagnosis + score — key forces fresh DOM when a new /predict result lands */}
    <Box key={analysisVersionKey}>
      <Typography
        variant="overline"
        sx={{ color: mutedText, letterSpacing: 1.5 }}
      >
        DIAGNOSIS
      </Typography>

      <Box sx={{ mt: 1 }}>
        <Typography
          component="h2"
          sx={{
            fontWeight: 900,
            lineHeight: 1.05,
            fontSize: { xs: '2.8rem', md: '3.4rem' },
            minHeight: { xs: 48, md: 58 },
            display: 'flex',
            alignItems: 'center',
          }}
        >
          {diagnosisLine ? (
            diagnosisLine
          ) : predictUi.loading ? (
            <CircularProgress size={30} thickness={5} />
          ) : (
            '—'
          )}
        </Typography>
        {confidenceLine.trim() ? (
          <Chip
            label={confidenceLine}
            size="small"
            sx={{
              mt: 1,
              alignSelf: 'flex-start',
              bgcolor: isHighRisk ? 'rgba(239,68,68,0.12)' : 'rgba(34,197,94,0.12)',
              border: isHighRisk ? '1px solid #ef4444' : '1px solid #22c55e',
              color: isHighRisk ? '#ef4444' : '#16a34a',
              fontSize: 11,
              height: 24,
              borderRadius: 999,
              px: 1.5,
            }}
          />
        ) : null}
      </Box>

      <Box sx={{ mt: 3 }}>
        <Typography
          variant="subtitle2"
          sx={{ color: mutedText, mb: 0.75 }}
        >
          TB risk score
        </Typography>
        <Typography
          component="p"
          sx={{
            fontWeight: 900,
            lineHeight: 1,
            fontSize: { xs: '3.8rem', md: '4.4rem' }, // big 70%
            minHeight: { xs: 64, md: 72 },
            display: 'flex',
            alignItems: 'center',
          }}
        >
          {riskPct != null ? (
            `${riskPct}%`
          ) : predictUi.loading ? (
            <CircularProgress size={34} thickness={5} />
          ) : (
            '—'
          )}
        </Typography>
        <Typography variant="caption" sx={{ color: mutedText, display: 'block', mt: 1 }}>
          {pred ? 'Estimated TB probability from the screening model.' : 'Run Analyze on a chest X-ray to see results.'}
        </Typography>
      </Box>
    </Box>

    {/* Anomaly flag (opt-in note from the clinician for the image on screen) */}
    {pred && currentFlagKey ? (
      <Box>
        {currentFlag ? (
          <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: 'wrap', rowGap: 1 }}>
            <FlagOutlinedIcon fontSize="small" sx={{ color: 'warning.main' }} />
            {currentFlag.categories.map((c) => (
              <Chip
                key={c}
                label={categoryLabel(c)}
                size="small"
                sx={{
                  bgcolor: 'rgba(245, 158, 11, 0.14)',
                  border: '1px solid rgba(245, 158, 11, 0.55)',
                  color: 'warning.main',
                  borderRadius: 999,
                  fontSize: 11,
                  height: 24,
                }}
              />
            ))}
            <Button size="small" variant="text" onClick={openFlagForm}>
              {flagFormOpen ? 'Close' : 'Edit note'}
            </Button>
            <Button
              size="small"
              variant="text"
              color="error"
              onClick={() => removeFlag(currentFlagKey)}
            >
              Remove
            </Button>
          </Stack>
        ) : (
          <Button
            size="small"
            variant="outlined"
            startIcon={<FlagOutlinedIcon fontSize="small" />}
            onClick={openFlagForm}
          >
            {flagFormOpen ? 'Cancel' : 'Flag anomaly'}
          </Button>
        )}

        <Collapse in={flagFormOpen} unmountOnExit>
          <Box
            sx={(theme) => ({
              mt: 1.5,
              p: 2,
              borderRadius: 2,
              border: '1px solid',
              borderColor: 'rgba(148, 163, 184, 0.35)',
              bgcolor: 'rgba(255, 255, 255, 0.55)',
              backdropFilter: 'blur(10px)',
              WebkitBackdropFilter: 'blur(10px)',
              ...theme.applyStyles('dark', {
                bgcolor: 'rgba(30, 41, 59, 0.55)',
                borderColor: 'rgba(148, 163, 184, 0.22)',
              }),
            })}
          >
            <Typography
              variant="caption"
              sx={{ color: mutedText, display: 'block', mb: 1 }}
            >
              Note an anomaly
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mb: 1.5 }}>
              {FLAG_CATEGORIES.map((c) => {
                const active = flagFormCategories.includes(c.id);
                return (
                  <Chip
                    key={c.id}
                    label={c.label}
                    size="small"
                    clickable
                    /** Use outlined for unselected so MUI's filled-default bg
                     *  doesn't override our transparent background. */
                    variant={active ? 'filled' : 'outlined'}
                    onClick={() => toggleFormCategory(c.id)}
                    aria-pressed={active}
                    /**
                     * The shared theme's `MuiChip` variant for `color: 'default'`
                     * paints filled chips with `gray[800]` bg + `gray[300]` label
                     * color in dark mode (and a light gray version in light mode).
                     * That variant generates `.MuiChip-root.MuiChip-colorDefault`
                     * selectors which match this Chip and tie the specificity of
                     * a plain `sx` override — meaning the theme can win on
                     * source order, especially once focus shifts away from the
                     * chip and re-triggers its resting state.
                     *
                     * We bump specificity by anchoring rules to `&.MuiChip-root`
                     * (and `&.MuiChip-root .MuiChip-label`) so the selected
                     * white-on-dark / dark-on-white look HOLDS regardless of
                     * focus, hover, or where the user clicks next.
                     */
                    sx={{
                      borderRadius: 999,
                      fontSize: 11,
                      height: 24,
                      transition:
                        'background-color 120ms ease, color 120ms ease, border-color 120ms ease',
                      ...(active
                        ? {
                            '&.MuiChip-root': {
                              backgroundColor: 'text.primary',
                              borderColor: 'text.primary',
                              border: '1px solid',
                            },
                            '&.MuiChip-root .MuiChip-label': {
                              color: 'background.default',
                            },
                            '&.MuiChip-root:hover, &.MuiChip-root:focus, &.MuiChip-root:focus-visible, &.MuiChip-root:active':
                              {
                                backgroundColor: 'text.primary',
                                borderColor: 'text.primary',
                                opacity: 1,
                              },
                          }
                        : {
                            '&.MuiChip-root': {
                              backgroundColor: 'transparent',
                              borderColor: 'divider',
                              border: '1px solid',
                            },
                            '&.MuiChip-root .MuiChip-label': {
                              color: 'text.primary',
                            },
                            '&.MuiChip-root:hover': {
                              backgroundColor: 'action.hover',
                              borderColor: 'divider',
                            },
                          }),
                    }}
                  />
                );
              })}
            </Box>
            <TextField
              fullWidth
              multiline
              minRows={2}
              maxRows={6}
              placeholder="Note (e.g., missed nodule on right apex)"
              value={flagFormNote}
              onChange={(e) => setFlagFormNote(e.target.value)}
              sx={{
                // The shared theme pins MuiOutlinedInput's root to a fixed height
                // for `size: small`, which squeezes the multiline textarea against
                // the top edge. Override height + padding for breathing room.
                '& .MuiOutlinedInput-root': {
                  height: 'auto',
                  py: 1.25,
                  px: 1.5,
                  alignItems: 'flex-start',
                },
                '& .MuiOutlinedInput-input': {
                  padding: 0,
                },
              }}
            />
            <Stack direction="row" spacing={1} sx={{ mt: 1.5, justifyContent: 'flex-end' }}>
              <Button size="small" onClick={() => setFlagFormOpen(false)}>
                Cancel
              </Button>
              <Button
                size="small"
                variant="contained"
                onClick={saveCurrentFlag}
                disabled={flagFormCategories.length === 0}
                sx={(theme) => ({
                  // Keep the label visible when disabled — MUI's default disabled
                  // styling fades both bg and text, which on dark mode reads as a
                  // near-blank white tile. Use a clearly-disabled-but-readable look.
                  '&.Mui-disabled': {
                    opacity: 1,
                    color: 'rgba(15, 23, 42, 0.55)',
                    backgroundColor: 'rgba(148, 163, 184, 0.28)',
                    backgroundImage: 'none',
                    boxShadow: 'none',
                    borderColor: 'rgba(148, 163, 184, 0.35)',
                    ...theme.applyStyles('dark', {
                      color: 'rgba(241, 245, 249, 0.7)',
                      backgroundColor: 'rgba(148, 163, 184, 0.18)',
                      borderColor: 'rgba(148, 163, 184, 0.28)',
                    }),
                  },
                })}
              >
                {currentFlag ? 'Update note' : 'Save note'}
              </Button>
            </Stack>
          </Box>
        </Collapse>
      </Box>
    ) : null}

    {/* Bottom: contributions (pulled closer to 70%) */}
    <Box>
                <Divider sx={{ borderColor: cardBorder, mb: 3 }} />

      <Typography
        variant="subtitle2"
        sx={{ color: mutedText, mb: 1.5 }}
      >
        Model contributions
      </Typography>

      {modelRows.map((item) => (
        <Box key={item.label} sx={{ mb: 1.5 }}>
          <Stack
            direction="row"
            justifyContent="space-between"
            sx={{ mb: 0.5 }}
          >
            <Typography variant="body2">{item.label}</Typography>
            <Typography variant="body2" sx={{ color: mutedText }}>
              {item.value}%
            </Typography>
          </Stack>
                    <LinearProgress
                      variant="determinate"
                      value={item.value}
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        bgcolor: 'action.hover',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 3,
                          bgcolor: item.barColor,
                        },
                      }}
                    />
        </Box>
      ))}
    </Box>
  </CardContent>
</Card>



          {/* Heatmap */}
          <Card
            variant="outlined"
            sx={{
              flex: 1,
              minWidth: 0,
              bgcolor: cardBg,
              borderColor: cardBorder,
              color: 'text.primary',
              borderRadius: 2,
            }}
          >
            <CardHeader
              title={
                <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
                  <Typography
                    component="span"
                    sx={{ fontSize: 14, fontWeight: 600, flexShrink: 0 }}
                  >
                    Prediction Heat Map
                  </Typography>
                  {hasBatch ? (
                    <Stack
                      direction="row"
                      alignItems="center"
                      spacing={0.5}
                      sx={{ flexShrink: 0 }}
                    >
                      <IconButton
                        size="small"
                        disabled={!canGoPrev}
                        onClick={() => onNavigateIndex?.(safeIndex - 1)}
                      >
                        <ChevronLeft fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        disabled={!canGoNext}
                        onClick={() => onNavigateIndex?.(safeIndex + 1)}
                      >
                        <ChevronRight fontSize="small" />
                      </IconButton>
                    </Stack>
                  ) : null}
                </Stack>
              }
              subheader="Highlighted TB-suspect regions"
              sx={{
                pb: 1,
                '& .MuiCardHeader-content': {
                  minWidth: 0,
                },
                '& .MuiCardHeader-title': {
                  width: '100%',
                },
                '& .MuiCardHeader-subheader': {
                  color: mutedText,
                  fontSize: 12,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                },
              }}
            />
            <CardContent sx={{ pt: 1 }}>
              {hasHeatmap ? (
                <Box
                  component="img"
                  key={`${analysisVersionKey}|${heatmapSrc.slice(0, 64)}`}
                  src={heatmapSrc}
                  alt="Saliency overlay on input study"
                  sx={{
                    borderRadius: 2,
                    width: '100%',
                    maxHeight: { xs: '52vh', md: '62vh' },
                    height: 'auto',
                    border: '1px solid',
                    borderColor: cardBorder,
                    objectFit: 'contain',
                    display: 'block',
                    mx: 'auto',
                    bgcolor: 'background.default',
                  }}
                />
              ) : (
                <Box aria-hidden sx={(theme) => liquidGlassImagePlaceholderSx(theme)} />
              )}
            </CardContent>
          </Card>
        </Box>
        <Dialog open={logOpen} onClose={() => setLogOpen(false)} fullWidth maxWidth="sm">
          <DialogTitle>Anomaly log ({flagCount})</DialogTitle>
          <DialogContent dividers>
            {flagCount === 0 ? (
              <Typography variant="body2" sx={{ color: mutedText }}>
                No anomalies noted yet.
              </Typography>
            ) : (
              <Stack spacing={1.5}>
                {Object.entries(flags)
                  .sort(([, a], [, b]) => (a.savedAt < b.savedAt ? 1 : -1))
                  .map(([key, flag]) => (
                    <Box
                      key={key}
                      sx={{
                        p: 1.5,
                        borderRadius: 1.5,
                        border: '1px solid',
                        borderColor: 'divider',
                      }}
                    >
                      <Stack
                        direction="row"
                        justifyContent="space-between"
                        alignItems="flex-start"
                        sx={{ mb: 0.75 }}
                      >
                        <Box sx={{ minWidth: 0, pr: 1 }}>
                          <Typography
                            variant="body2"
                            sx={{ fontWeight: 600, wordBreak: 'break-word' }}
                          >
                            {flag.snapshot.fileName}
                          </Typography>
                          <Typography
                            variant="caption"
                            sx={{ color: mutedText, display: 'block' }}
                          >
                            Model: {flag.snapshot.diagnosis || '—'}
                            {flag.snapshot.riskScore != null ? ` • ${flag.snapshot.riskScore}%` : ''}
                            {flag.snapshot.confidence ? ` • ${flag.snapshot.confidence}` : ''}
                          </Typography>
                        </Box>
                        <IconButton
                          size="small"
                          aria-label="Delete note"
                          onClick={() => removeFlag(key)}
                        >
                          <DeleteOutlineIcon fontSize="small" />
                        </IconButton>
                      </Stack>
                      <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap', rowGap: 0.5 }}>
                        {flag.categories.map((c) => (
                          <Chip
                            key={c}
                            label={categoryLabel(c)}
                            size="small"
                            sx={{
                              bgcolor: 'rgba(245, 158, 11, 0.14)',
                              border: '1px solid rgba(245, 158, 11, 0.55)',
                              color: 'warning.main',
                              borderRadius: 999,
                              fontSize: 11,
                              height: 22,
                            }}
                          />
                        ))}
                      </Stack>
                      {flag.note ? (
                        <Typography variant="body2" sx={{ mt: 0.75, whiteSpace: 'pre-wrap' }}>
                          {flag.note}
                        </Typography>
                      ) : null}
                      <Typography
                        variant="caption"
                        sx={{ color: mutedText, display: 'block', mt: 0.75 }}
                      >
                        {new Date(flag.savedAt).toLocaleString()}
                      </Typography>
                    </Box>
                  ))}
              </Stack>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setLogOpen(false)}>Close</Button>
          </DialogActions>
        </Dialog>
        {hasBatch ? (
          <Stack direction="row" justifyContent="center" spacing={0.75} sx={{ mt: 2 }}>
            {predictUi.items.map((item, idx) => {
              const isActive = idx === safeIndex;
              const color =
                item.status === 'error'
                  ? '#ef4444'
                  : item.status === 'done'
                    ? '#22c55e'
                    : item.status === 'processing'
                      ? '#3b82f6'
                      : '#6b7280';
              return (
                <Box
                  key={item.id}
                  component="button"
                  type="button"
                  aria-label={`Jump to image ${idx + 1}`}
                  onClick={() => onNavigateIndex?.(idx)}
                  sx={{
                    width: isActive ? 12 : 10,
                    height: isActive ? 12 : 10,
                    borderRadius: '50%',
                    border: 'none',
                    p: 0,
                    cursor: 'pointer',
                    bgcolor: color,
                    opacity: isActive ? 1 : 0.45,
                    transform: isActive ? 'scale(1.05)' : 'none',
                    transition: 'all 120ms ease',
                  }}
                />
              );
            })}
          </Stack>
        ) : null}
      </Container>
    </Box>
  );
}
