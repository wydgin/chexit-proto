import * as React from 'react';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import LinearProgress from '@mui/material/LinearProgress';
import CircularProgress from '@mui/material/CircularProgress';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import ChevronLeft from '@mui/icons-material/ChevronLeft';
import ChevronRight from '@mui/icons-material/ChevronRight';
import type { PredictUiState } from '../../api/chexit';
import { fetchLatestUpload, type UploadRecord } from '../../api/chexit';

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
    ? `Image ${safeIndex + 1} of ${predictUi.items.length} • ${selectedItem?.fileName ?? 'selected'}`
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

  return (
    <Box sx={{ bgcolor: pageBg }}>
      <Container id="features" maxWidth="lg" sx={{ pt: { xs: 4, md: 5 }, pb: { xs: 6, md: 7 } }}>
        <Typography
          variant="h6"
          sx={{
            mb: 3,
            color: 'text.primary',
            fontWeight: 600,
            letterSpacing: 0.5,
          }}
        >
          AI-assisted TB overview
        </Typography>

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
              bgcolor: cardBg,
              borderColor: cardBorder,
              color: 'text.primary',
              borderRadius: 2,
            }}
          >
            <CardHeader
              title={
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                  <Typography component="span" sx={{ fontSize: 14, fontWeight: 600 }}>
                    Input X-Ray
                  </Typography>
                  {hasBatch ? (
                    <Stack direction="row" alignItems="center" spacing={0.5}>
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
                '& .MuiCardHeader-title': {
                  width: '100%',
                },
                '& .MuiCardHeader-subheader': {
                  color: mutedText,
                  fontSize: 12,
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
                <Box
                  sx={{
                    borderRadius: 2,
                    bgcolor: '#000',
                    border: '1px dashed',
                    borderColor: cardBorder,
                    width: '100%',
                    aspectRatio: '3 / 4',
                  }}
                />
              )}
            </CardContent>
          </Card>
{/* Diagnosis */}
          <Card
            variant="outlined"
            sx={{
              flex: 1.1,
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

      {pred ? (
        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5, mb: 0.5 }}>
          TB screening:{' '}
          <Box component="span" sx={{ fontWeight: 700, color: 'text.primary' }}>
            {isHighRisk ? 'Positive call' : 'Negative call'}
          </Box>
          {riskPct != null ? ` • model score ${riskPct}%` : null}
        </Typography>
      ) : null}

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
        {confidenceLine || !predictUi.loading ? (
          <Chip
            label={confidenceLine || 'Run Analyze'}
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
              ...(!pred &&
                !predictUi.loading && {
                  bgcolor: 'action.hover',
                  borderColor: 'divider',
                  color: 'text.secondary',
                }),
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
              bgcolor: cardBg,
              borderColor: cardBorder,
              color: 'text.primary',
              borderRadius: 2,
            }}
          >
            <CardHeader
              title={
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                  <Typography component="span" sx={{ fontSize: 14, fontWeight: 600 }}>
                    Prediction Heatmap
                  </Typography>
                  {hasBatch ? (
                    <Stack direction="row" alignItems="center" spacing={0.5}>
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
                '& .MuiCardHeader-title': {
                  width: '100%',
                },
                '& .MuiCardHeader-subheader': {
                  color: mutedText,
                  fontSize: 12,
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
                <Box
                  aria-hidden
                  sx={{
                    borderRadius: 2,
                    width: '100%',
                    aspectRatio: '3 / 4',
                    bgcolor: '#000',
                    border: '1px solid',
                    borderColor: cardBorder,
                  }}
                />
              )}
            </CardContent>
          </Card>
        </Box>
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
