import * as React from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Container from '@mui/material/Container';
import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import { predictImage } from '../../api/chexit';
import type { PredictUiState } from '../../api/chexit';
import { storage, db } from '../../firebase';

const UPLOADS_COLLECTION = 'uploads';
const LATEST_DOC_ID = 'latest';
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;

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
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const selectedFileRef = React.useRef<File | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = React.useState<string | null>(null);
  const [analyzing, setAnalyzing] = React.useState(false);
  const [uploading, setUploading] = React.useState(false);
  const [uploadError, setUploadError] = React.useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = React.useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    return () => {
      if (localPreviewUrl) {
        URL.revokeObjectURL(localPreviewUrl);
      }
    };
  }, [localPreviewUrl]);

  React.useEffect(() => {
    onLocalPreviewChange?.(localPreviewUrl);
  }, [localPreviewUrl, onLocalPreviewChange]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const isImage = file.type.startsWith('image/');
    const isDicom =
      file.type === 'application/dicom' ||
      file.type === 'application/octet-stream' ||
      /\.(dcm|dicom)$/i.test(file.name);

    if (!isImage && !isDicom) {
      setUploadError('Please upload PNG/JPG or DICOM (.dcm).');
      setUploadSuccess(false);
      return;
    }

    if (file.size > MAX_IMAGE_BYTES) {
      setUploadError('Max file size is 10MB.');
      setUploadSuccess(false);
      return;
    }
    setSelectedFile(file);
    selectedFileRef.current = file;
    setUploadError(null);
    setUploadSuccess(false);
    setLocalPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return isDicom ? null : URL.createObjectURL(file);
    });
    onPredictUiChange?.({ loading: false, error: null, data: null });
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleAnalyze = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    e.stopPropagation();

    const file =
      selectedFileRef.current ?? selectedFile ?? fileInputRef.current?.files?.[0] ?? null;

    if (!file) {
      onPredictUiChange?.({
        loading: false,
        error: 'Choose a chest X-ray with Browse file, then click Analyze.',
        data: null,
      });
      return;
    }

    if (file !== selectedFile) {
      const isDicom =
        file.type === 'application/dicom' ||
        file.type === 'application/octet-stream' ||
        /\.(dcm|dicom)$/i.test(file.name);
      setSelectedFile(file);
      selectedFileRef.current = file;
      setLocalPreviewUrl((prev) => {
        if (prev) {
          URL.revokeObjectURL(prev);
        }
        return isDicom ? null : URL.createObjectURL(file);
      });
    }

    setAnalyzing(true);
    onPredictUiChange?.({ loading: true, error: null, data: null });
    try {
      const data = await predictImage(file);
      onPredictUiChange?.({ loading: false, error: null, data });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analyze failed';
      onPredictUiChange?.({ loading: false, error: message, data: null });
    } finally {
      setAnalyzing(false);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadError(null);
    setUploadSuccess(false);
    try {
      const storageRef = ref(storage, `uploads/${Date.now()}_${selectedFile.name}`);
      await uploadBytes(storageRef, selectedFile);
      const downloadURL = await getDownloadURL(storageRef);
      const latestRef = doc(db, UPLOADS_COLLECTION, LATEST_DOC_ID);
      await setDoc(latestRef, {
        downloadURL,
        fileName: selectedFile.name,
        uploadedAt: serverTimestamp(),
      });
      onUploadComplete?.(downloadURL);
      setUploadSuccess(true);
      setLocalPreviewUrl(null);
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
              style={{ display: 'none' }}
            />
            <Button
              type="button"
              variant="outlined"
              color="primary"
              size="small"
              onClick={handleBrowseClick}
              disabled={uploading || analyzing}
              sx={{ minWidth: 'fit-content' }}
            >
              {selectedFile ? selectedFile.name : 'Browse file'}
            </Button>
            <Button
              type="button"
              variant={selectedFile ? 'contained' : 'outlined'}
              color={selectedFile ? 'primary' : 'inherit'}
              size="small"
              onClick={handleAnalyze}
              disabled={uploading || analyzing}
              startIcon={analyzing ? <CircularProgress size={16} color="inherit" /> : undefined}
              sx={{
                minWidth: 'fit-content',
                ...(!selectedFile && {
                  backgroundColor: 'action.disabledBackground',
                  color: 'action.disabled',
                  borderColor: 'action.disabled',
                }),
              }}
            >
              {analyzing ? 'Analyzing…' : 'Analyze'}
            </Button>
            <Button
              type="button"
              variant="outlined"
              color="primary"
              size="small"
              onClick={handleUpload}
              disabled={!selectedFile || uploading || analyzing}
              startIcon={uploading ? <CircularProgress size={16} color="inherit" /> : undefined}
              sx={{ minWidth: 'fit-content' }}
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
            <Alert severity="info" sx={{ maxWidth: 560, width: '100%', textAlign: 'center' }}>
              Analyzing… This may take a few minutes.
            </Alert>
          )}
          {!predictUi.loading && !analyzing && predictUi.error ? (
            <Alert severity="error" sx={{ maxWidth: 560, width: '100%', textAlign: 'left' }}>
              {predictUi.error}
            </Alert>
          ) : null}
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ textAlign: 'center' }}
          >
            By clicking &quot;Upload&quot; you agree to our&nbsp;
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
