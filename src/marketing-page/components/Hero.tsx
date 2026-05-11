import * as React from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Container from '@mui/material/Container';
import LinearProgress from '@mui/material/LinearProgress';
import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import { predictImagesSequential, releaseBatchPreviewUrls, uploadImage } from '../../api/chexit';
import type { PredictUiState } from '../../api/chexit';
import { gray } from '../../../shared-theme/themePrimitives';
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_BATCH_IMAGES = 5;

type HeroProps = {
  /** Latest analyze state (shown in alerts + passed to Features via parent). */
  predictUi: PredictUiState;
  onUploadComplete?: (downloadUrl: string) => void;
  /** Temporary `blob:` URL for instant preview in the dashboard (no upload required). */
  onLocalPreviewChange?: (previewUrl: string | null) => void;
  onPredictUiChange?: (state: PredictUiState) => void;
};

export default function Hero({
  predictUi,
  onUploadComplete,
  onLocalPreviewChange,
  onPredictUiChange,
}: HeroProps) {
  const [selectedFiles, setSelectedFiles] = React.useState<File[]>([]);
  const [analyzing, setAnalyzing] = React.useState(false);
  const [uploading, setUploading] = React.useState(false);
  const [uploadError, setUploadError] = React.useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = React.useState(false);
  const [progressText, setProgressText] = React.useState<string | null>(null);
  const [progressValue, setProgressValue] = React.useState<number>(0);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const previousItemsRef = React.useRef<PredictUiState['items']>([]);

  React.useEffect(
    () => () => {
      releaseBatchPreviewUrls(previousItemsRef.current);
    },
    [],
  );

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(event.target.files ?? []);
    if (picked.length === 0) {
      return;
    }
    const files = picked.slice(0, MAX_BATCH_IMAGES);
    if (picked.length > MAX_BATCH_IMAGES) {
      setUploadError(`Only up to ${MAX_BATCH_IMAGES} images are allowed per batch.`);
      setUploadSuccess(false);
    } else {
      setUploadError(null);
    }
    const badType = files.find((file) => {
      const isImage = file.type.startsWith('image/');
      const isDicom =
        file.type === 'application/dicom' ||
        file.type === 'application/octet-stream' ||
        /\.(dcm|dicom)$/i.test(file.name);
      return !isImage && !isDicom;
    });
    if (badType) {
      setUploadError('Please upload PNG/JPG or DICOM (.dcm).');
      setUploadSuccess(false);
      return;
    }
    const tooLarge = files.find((file) => file.size > MAX_IMAGE_BYTES);
    if (tooLarge) {
      setUploadError('Max file size is 10MB per image.');
      setUploadSuccess(false);
      return;
    }
    setSelectedFiles(files);
    setUploadSuccess(false);
    setProgressText(null);
    setProgressValue(0);
    releaseBatchPreviewUrls(previousItemsRef.current);
    previousItemsRef.current = [];
    onLocalPreviewChange?.(null);
    onPredictUiChange?.({ loading: false, error: null, data: null, items: [], currentIndex: 0 });
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleAnalyze = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    e.stopPropagation();

    if (selectedFiles.length === 0) {
      onPredictUiChange?.({
        loading: false,
        error: 'Choose a chest X-ray with Browse file, then click Analyze.',
        data: null,
        items: [],
        currentIndex: 0,
      });
      return;
    }
    setAnalyzing(true);
    setProgressText(`Preparing ${selectedFiles.length} image(s)…`);
    setProgressValue(0);
    onPredictUiChange?.({ loading: true, error: null, data: null, items: [], currentIndex: 0 });
    try {
      releaseBatchPreviewUrls(previousItemsRef.current);
      const finalItems = await predictImagesSequential(selectedFiles, (nextItems, currentIndex) => {
        previousItemsRef.current = nextItems;
        const doneCount = nextItems.filter((item) => item.status === 'done' || item.status === 'error').length;
        const pct = Math.round((doneCount / Math.max(nextItems.length, 1)) * 100);
        setProgressValue(pct);
        setProgressText(`Processing image ${Math.min(currentIndex + 1, nextItems.length)} of ${nextItems.length}`);
        const selected = nextItems[currentIndex]?.result ?? null;
        onLocalPreviewChange?.(nextItems[currentIndex]?.localPreviewUrl ?? null);
        onPredictUiChange?.({
          loading: true,
          error: null,
          data: selected,
          items: nextItems,
          currentIndex,
        });
      });
      previousItemsRef.current = finalItems;
      const firstSuccessful = finalItems.findIndex((item) => item.status === 'done' && item.result);
      const finalIndex = firstSuccessful >= 0 ? firstSuccessful : 0;
      const selected = finalItems[finalIndex]?.result ?? null;
      const errors = finalItems.filter((item) => item.status === 'error').length;
      onLocalPreviewChange?.(finalItems[finalIndex]?.localPreviewUrl ?? null);
      onPredictUiChange?.({
        loading: false,
        error: errors > 0 ? `${errors} image(s) failed. Use arrows below to inspect each result.` : null,
        data: selected,
        items: finalItems,
        currentIndex: finalIndex,
      });
      setProgressValue(100);
      setProgressText(`Completed ${finalItems.length} image(s).`);
    } catch {
      onPredictUiChange?.({ loading: false, error: 'Analyze failed', data: null, items: [], currentIndex: 0 });
    } finally {
      setAnalyzing(false);
    }
  };

  const handleUpload = async () => {
    const selectedFile = selectedFiles[0];
    if (!selectedFile) return;
    setUploading(true);
    setUploadError(null);
    setUploadSuccess(false);
    try {
      const uploaded = await uploadImage(selectedFile);
      onUploadComplete?.(uploaded.downloadURL);
      setUploadSuccess(true);
      onLocalPreviewChange?.(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setUploadError(message);
      setUploadSuccess(false);
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box
      id="hero"
      sx={(theme) => ({
        width: '100%',
        backgroundRepeat: 'no-repeat',

        backgroundImage:
          'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 90%), transparent)',
        ...theme.applyStyles('dark', {
          backgroundImage:
            'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 16%), transparent)',
        }),
      })}
    >
      <Container
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          pt: { xs: 12, sm: 16 },
          pb: { xs: 4, sm: 6 },
        }}
      >
        <Stack
          spacing={2}
          useFlexGap
          sx={{ alignItems: 'center', width: { xs: '100%', sm: '70%' } }}
        >
          <Typography
            variant="h1"
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', sm: 'row' },
              alignItems: 'center',
              fontSize: 'clamp(3rem, 10vw, 3.5rem)',
            }}
          >
            Your&nbsp;latest&nbsp;
            <Typography
              component="span"
              variant="h1"
              sx={(theme) => ({
                fontSize: 'inherit',
                color: 'primary.main',
                ...theme.applyStyles('dark', {
                  color: 'primary.light',
                }),
              })}
            >
              TB assistant
            </Typography>
          </Typography>
          <Typography
            sx={{
              textAlign: 'center',
              color: 'text.secondary',
              width: { sm: '100%', md: '80%' },
            }}
          >
            Upload a chest X-ray to get AI-assisted TB risk scores and heatmaps
          </Typography>
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={1}
            useFlexGap
            flexWrap="wrap"
            justifyContent="center"
            sx={{ pt: 2, width: { xs: '100%', sm: 'auto' }, maxWidth: 520 }}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*,.dcm,.dicom,application/dicom"
              multiple
              style={{ display: 'none' }}
            />
            <Button
              type="button"
              variant="outlined"
              color="primary"
              size="small"
              onClick={handleBrowseClick}
              disabled={uploading || analyzing}
              sx={(theme) => ({
                minWidth: 'fit-content',
                '&.Mui-disabled': {
                  color: 'rgba(15,23,42,0.74)',
                  borderColor: 'rgba(100,116,139,0.4)',
                  backgroundColor: 'rgba(241,245,249,0.75)',
                  opacity: 1,
                  ...theme.applyStyles('dark', {
                    color: 'rgba(241,245,249,0.92)',
                    borderColor: 'rgba(148,163,184,0.55)',
                    backgroundColor: 'rgba(30,41,59,0.85)',
                  }),
                },
              })}
            >
              {selectedFiles.length > 0 ? `${selectedFiles.length} file(s) selected` : 'Browse files (max 5)'}
            </Button>
            <Button
              type="button"
              variant={selectedFiles.length > 0 ? 'contained' : 'outlined'}
              color={selectedFiles.length > 0 ? 'primary' : 'inherit'}
              size="small"
              onClick={handleAnalyze}
              disabled={uploading || analyzing}
              startIcon={analyzing ? <CircularProgress size={16} color="inherit" /> : undefined}
              sx={(theme) => ({
                minWidth: 'fit-content',
                ...(!(selectedFiles.length > 0) && {
                  backgroundColor: 'action.disabledBackground',
                  color: 'text.secondary',
                  borderColor: 'action.disabled',
                }),
                ...(selectedFiles.length > 0 && {
                  color: 'primary.contrastText',
                }),
                '&.Mui-disabled': analyzing
                  ? {
                      color: 'white',
                      backgroundColor: gray[900],
                      backgroundImage: `linear-gradient(to bottom, ${gray[700]}, ${gray[800]})`,
                      boxShadow: `inset 0 1px 0 ${gray[600]}, inset 0 -1px 0 1px hsl(220, 0%, 0%)`,
                      borderColor: gray[700],
                      opacity: 1,
                      ...theme.applyStyles('dark', {
                        color: 'black',
                        backgroundColor: gray[50],
                        backgroundImage: `linear-gradient(to bottom, ${gray[100]}, ${gray[50]})`,
                        boxShadow: 'inset 0 -1px 0 hsl(220, 30%, 80%)',
                        borderColor: gray[50],
                      }),
                    }
                  : undefined,
              })}
            >
              {analyzing ? 'Analyzing…' : 'Analyze'}
            </Button>
            <Button
              type="button"
              variant="outlined"
              color="primary"
              size="small"
              onClick={handleUpload}
              disabled={selectedFiles.length === 0 || uploading || analyzing}
              startIcon={uploading ? <CircularProgress size={16} color="inherit" /> : undefined}
              sx={(theme) => ({
                minWidth: 'fit-content',
                '&.Mui-disabled': {
                  color: 'rgba(15,23,42,0.55)',
                  borderColor: 'rgba(100,116,139,0.35)',
                  backgroundColor: 'rgba(241,245,249,0.6)',
                  opacity: 1,
                  ...theme.applyStyles('dark', {
                    color: 'rgba(241,245,249,0.78)',
                    borderColor: 'rgba(148,163,184,0.45)',
                    backgroundColor: 'rgba(30,41,59,0.65)',
                  }),
                },
              })}
            >
              {uploading ? 'Uploading…' : 'Upload to cloud'}
            </Button>
          </Stack>
          {uploadError && (
            <Typography variant="body2" color="error" sx={{ textAlign: 'center' }}>
              {uploadError}
            </Typography>
          )}
          {uploadSuccess && (
            <Typography variant="body2" color="success.main" sx={{ textAlign: 'center' }}>
              File uploaded. Check preview below.
            </Typography>
          )}
          {(predictUi.loading || analyzing) && (
            <Box
              sx={(theme) => ({
                maxWidth: 560,
                width: '100%',
                px: 2,
                py: 1.5,
                borderRadius: 3,
                border: '1px solid',
                borderColor: 'rgba(148, 163, 184, 0.35)',
                bgcolor: 'rgba(255, 255, 255, 0.45)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                boxShadow: '0 8px 24px rgba(15, 23, 42, 0.08)',
                ...theme.applyStyles('dark', {
                  bgcolor: 'rgba(30, 41, 59, 0.45)',
                  borderColor: 'rgba(148, 163, 184, 0.25)',
                  boxShadow: '0 8px 24px rgba(2, 6, 23, 0.45)',
                }),
              })}
            >
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                <Box
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    bgcolor: 'info.main',
                    animation: 'pulseDot 1.4s ease-in-out infinite',
                    '@keyframes pulseDot': {
                      '0%, 100%': { opacity: 0.45, transform: 'scale(0.95)' },
                      '50%': { opacity: 1, transform: 'scale(1.08)' },
                    },
                  }}
                />
                <Typography variant="body2" sx={{ textAlign: 'left', fontWeight: 500 }}>
                  {progressText ?? 'Analyzing… This may take a few minutes.'}
                </Typography>
              </Stack>
              <LinearProgress
                variant="determinate"
                value={progressValue}
                sx={{
                  height: 7,
                  borderRadius: 999,
                  bgcolor: 'rgba(148, 163, 184, 0.22)',
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 999,
                  },
                }}
              />
            </Box>
          )}
          {predictUi.items.length > 0 ? (
            <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" justifyContent="center">
              {predictUi.items.map((item, idx) => (
                <Chip
                  key={item.id}
                  size="small"
                  label={`${idx + 1}. ${item.fileName}`}
                  color={item.status === 'error' ? 'error' : item.status === 'done' ? 'success' : 'default'}
                  variant={predictUi.currentIndex === idx ? 'filled' : 'outlined'}
                />
              ))}
            </Stack>
          ) : null}
          {!predictUi.loading && !analyzing && predictUi.error ? (
            <Box
              role="alert"
              sx={(theme) => ({
                maxWidth: 560,
                width: '100%',
                px: 2,
                py: 1.5,
                borderRadius: 3,
                border: '1px solid',
                borderColor: 'rgba(245, 158, 11, 0.45)',
                bgcolor: 'rgba(255, 251, 235, 0.6)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                boxShadow: '0 8px 24px rgba(15, 23, 42, 0.08)',
                ...theme.applyStyles('dark', {
                  bgcolor: 'rgba(69, 26, 3, 0.45)',
                  borderColor: 'rgba(245, 158, 11, 0.35)',
                  boxShadow: '0 8px 24px rgba(2, 6, 23, 0.45)',
                }),
              })}
            >
              <Stack direction="row" spacing={1} alignItems="center">
                <Box
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    bgcolor: 'warning.main',
                    animation: 'pulseDot 1.4s ease-in-out infinite',
                    '@keyframes pulseDot': {
                      '0%, 100%': { opacity: 0.45, transform: 'scale(0.95)' },
                      '50%': { opacity: 1, transform: 'scale(1.08)' },
                    },
                  }}
                />
                <Typography variant="body2" sx={{ textAlign: 'left', fontWeight: 500 }}>
                  {predictUi.error}
                </Typography>
              </Stack>
            </Box>
          ) : null}
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ textAlign: 'center' }}
          >
            By using our service, you agree to our&nbsp;
            <Link href="#" color="primary">
              Terms & Conditions
            </Link>
            .
          </Typography>
        </Stack>
        {/* removed dashboard image */}
      </Container>
    </Box>
  );
}
