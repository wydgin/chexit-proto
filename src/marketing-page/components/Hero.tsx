import * as React from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Container from '@mui/material/Container';
import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import { storage, db } from '../../firebase';

const UPLOADS_COLLECTION = 'uploads';
const LATEST_DOC_ID = 'latest';

export default function Hero() {
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [uploading, setUploading] = React.useState(false);
  const [uploadError, setUploadError] = React.useState<string | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadError(null);
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadError(null);
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
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setUploadError(message);
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
            Our&nbsp;latest&nbsp;
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
            sx={{ pt: 2, width: { xs: '100%', sm: '350px' } }}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />
            <Button
              variant="outlined"
              color="primary"
              size="small"
              onClick={handleBrowseClick}
              fullWidth
              disabled={uploading}
              sx={{ minWidth: 'fit-content' }}
            >
              {selectedFile ? selectedFile.name : 'Browse file'}
            </Button>
            <Button
              variant={selectedFile ? "contained" : "outlined"}
              color={selectedFile ? "primary" : "inherit"}
              size="small"
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
              startIcon={uploading ? <CircularProgress size={16} color="inherit" /> : undefined}
              sx={{
                minWidth: 'fit-content',
                ...(!selectedFile && {
                  backgroundColor: 'action.disabledBackground',
                  color: 'action.disabled',
                  borderColor: 'action.disabled',
                }),
              }}
            >
              {uploading ? 'Uploading…' : 'Upload'}
            </Button>
          </Stack>
          {uploadError && (
            <Typography variant="body2" color="error" sx={{ textAlign: 'center' }}>
              {uploadError}
            </Typography>
          )}
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
