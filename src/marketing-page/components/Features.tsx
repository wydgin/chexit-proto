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
import { db } from '../../firebase';

// Assets served from /assets at the project root (e.g., public/assets)
const cxrIn = '/assets/cxrin.png';
const cxrOut = '/assets/cxrout.png';

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

export default function Features() {
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

  const previewSrc = latestUpload?.downloadURL ?? cxrIn;
  const previewSubheader = latestUpload
    ? formatUploadedAt(latestUpload.uploadedAt)
    : 'Uploaded • 2 min ago';

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
              <Box
                component="img"
                src={previewSrc}
                alt="Input chest X-ray"
                sx={{
                  borderRadius: 2,
                  bgcolor: 'background.default',
                  border: '1px dashed',
                  borderColor: cardBorder,
                  width: '100%',
                  aspectRatio: '3 / 4',
                  objectFit: 'cover',
                }}
              />
              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" sx={{ color: mutedText }}>
                  View: PA • Resolution: 1024×1024
                </Typography>
              </Box>
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
    {/* Top: diagnosis + score */}
    <Box>
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
          }}
        >
          TB Positive
        </Typography>
        <Chip
          label="High risk"
          size="small"
          sx={{
            mt: 1,
            alignSelf: 'flex-start',
            bgcolor: 'rgba(248,113,113,0.12)',
            border: '1px solid #f87171',
            color: '#fecaca',
            fontSize: 11,
            height: 24,
            borderRadius: 999,
            px: 1.5,
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
          70%
        </Typography>
        <Typography
          variant="caption"
          sx={{ color: mutedText, display: 'block', mt: 1 }}
        >
          Ensemble prediction averaged across all models.
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

      {[
        { label: 'MobileNet-V2', value: 60, barColor: '#6366f1' },
        { label: 'ResNet-50', value: 80, barColor: '#3b82f6' },
        { label: 'DenseNet-121', value: 75, barColor: '#22c55e' },
      ].map((item) => (
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
              <Box
                component="img"
                 src={cxrOut}
                alt="Prediction Heatmap"
                sx={{
                  borderRadius: 2,
                  width: '100%',
                  aspectRatio: '3 / 4',
                  border: '1px solid',
                  borderColor: cardBorder,
                  objectFit: 'cover',
                }}
              />
              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" sx={{ color: mutedText }}>
                  Brighter areas indicate regions with higher model attention.
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Container>
    </Box>
  );
}
