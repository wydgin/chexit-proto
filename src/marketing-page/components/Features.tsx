import * as React from 'react';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import LinearProgress from '@mui/material/LinearProgress';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import { doc, onSnapshot, Timestamp } from 'firebase/firestore';
import type { PredictUiState } from '../../api/chexit';
import { db } from '../../firebase';

function formatUploadedAt(seconds: number | null): string {
  if (seconds == null) return 'Uploaded • 2 min ago';
  const d = new Date(seconds * 1000);
  const now = Date.now();
  const diffMs = now - d.getTime();
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
};

// function contributionBars(riskScore: number) {
//   const r = Math.min(100, Math.max(0, Math.round(riskScore)));
//   return [
//     { label: 'MobileNet-V2', value: Math.min(100, Math.round(r * 0.88)), barColor: '#6366f1' },
//     { label: 'ResNet-50', value: Math.min(100, Math.round(r * 1.02)), barColor: '#3b82f6' },
//     { label: 'DenseNet-121', value: r, barColor: '#22c55e' },
//   ];
// }

function contributionBars(_riskScore: number) {
  // Backend currently runs only MobileNetV2.
  // Keep other planned models visible but zeroed until backend supports them.
  return [
    { label: 'MobileNetV2', value: 100, barColor: '#6366f1' },
    { label: 'EfficientNetB2', value: 0, barColor: '#f59e0b' },
    { label: 'DenseNet121', value: 0, barColor: '#22c55e' },
  ];
}

export default function Features({ previewImageUrl, localPreviewUrl, predictUi }: FeaturesProps) {
  const pageBg = 'background.default';
  const cardBg = 'background.default';
  const cardBorder = 'divider';
  const mutedText = '#9ca3af';

  const [latestUpload, setLatestUpload] = React.useState<{
    downloadURL: string;
    fileName: string;
    uploadedAt: number | null;
  } | null>(null);

  React.useEffect(() => {
    const latestRef = doc(db, 'uploads', 'latest');
    const unsub = onSnapshot(
      latestRef,
      (snap) => {
        const data = snap.data();
        if (data?.downloadURL) {
          const uploadedAt = data.uploadedAt instanceof Timestamp
            ? data.uploadedAt.seconds
            : typeof data.uploadedAt?.seconds === 'number'
              ? data.uploadedAt.seconds
              : null;
          setLatestUpload({
            downloadURL: data.downloadURL,
            fileName: data.fileName ?? 'image',
            uploadedAt,
          });
        } else {
          setLatestUpload(null);
        }
      },
      () => setLatestUpload(null),
    );
    return () => unsub();
  }, []);

  const remotePreviewSrc = previewImageUrl ?? latestUpload?.downloadURL ?? null;
  const previewSrc = localPreviewUrl ?? remotePreviewSrc;
  const hasRemoteImage = Boolean(remotePreviewSrc);
  const previewSubheader = localPreviewUrl && !previewImageUrl
    ? 'Preview • not uploaded yet'
    : previewImageUrl
      ? 'Uploaded • just now'
      : hasRemoteImage
        ? formatUploadedAt(latestUpload?.uploadedAt ?? null)
        : 'No image uploaded';

  const pred = predictUi.data;
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
  // const modelRows = pred && riskPct != null ? contributionBars(Number(pred.risk_score)) : [
  //   { label: 'MobileNet-V2', value: 60, barColor: '#6366f1' },
  //   { label: 'ResNet-50', value: 80, barColor: '#3b82f6' },
  //   { label: 'DenseNet-121', value: 75, barColor: '#22c55e' },
  // ];
  const modelRows = pred && riskPct != null
  ? contributionBars(Number(pred.risk_score))
  : [
      { label: 'MobileNetV2', value: 100, barColor: '#6366f1' },
      { label: 'EfficientNetB2', value: 0, barColor: '#f59e0b' },
      { label: 'DenseNet121', value: 0, barColor: '#22c55e' },
    ];
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

        {predictUi.error ? (
          <Typography variant="body2" color="error" sx={{ mb: 2 }}>
            {predictUi.error}
          </Typography>
        ) : null}

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
              title="Input X-Ray"
              subheader={previewSubheader}
              sx={{
                pb: 1,
                '& .MuiCardHeader-title': {
                  fontSize: 14,
                  fontWeight: 600,
                },
                '& .MuiCardHeader-subheader': {
                  color: mutedText,
                  fontSize: 12,
                },
              }}
            />
            <CardContent sx={{ pt: 1 }}>
              {previewSrc ? (
                <Box
                  component="img"
                  key={previewSrc}
                  src={previewSrc}
                  alt="Input chest X-ray"
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
              {previewSrc ? (
                <Box sx={{ mt: 1.5 }}>
                  <Typography variant="caption" sx={{ color: mutedText }}>
                    Input study (same resolution as uploaded).
                  </Typography>
                </Box>
              ) : null}
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
          }}
        >
          {diagnosisLine || (predictUi.loading ? '…' : '—')}
        </Typography>
        <Chip
          label={
            confidenceLine || (predictUi.loading ? 'Analyzing…' : 'Run Analyze')
          }
          size="small"
          sx={{
            mt: 1,
            alignSelf: 'flex-start',
            bgcolor: isHighRisk ? 'rgba(248,113,113,0.12)' : 'rgba(34,197,94,0.12)',
            border: isHighRisk ? '1px solid #f87171' : '1px solid #4ade80',
            color: isHighRisk ? '#fecaca' : '#bbf7d0',
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
          }}
        >
          {riskPct != null ? `${riskPct}%` : predictUi.loading ? '…' : '—'}
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
              title="Prediction Heatmap"
              subheader="Highlighted TB-suspect regions"
              sx={{
                pb: 1,
                '& .MuiCardHeader-title': {
                  fontSize: 14,
                  fontWeight: 600,
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
              <Box sx={{ mt: 1.5 }}>
                <Typography variant="caption" sx={{ color: mutedText }}>
                  {predictUi.loading
                    ? 'Generating overlay…'
                    : pred
                      ? 'Saliency overlay on the same study — same pixel dimensions as the input image.'
                      : 'Heatmap appears here after Analyze.'}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Container>
    </Box>
  );
}
