import * as React from 'react';
import CssBaseline from '@mui/material/CssBaseline';
import type { PredictUiState } from '../api/chexit';
import AppTheme from '../../shared-theme/AppTheme';
import AppAppBar from './components/AppAppBar';
import Hero from './components/Hero';
// import LogoCollection from './components/LogoCollection';
// import Highlights from './components/Highlights';
// import Pricing from './components/Pricing';
import Features from './components/Features';
// import Testimonials from './components/Testimonials';
// import FAQ from './components/FAQ';
import Footer from './components/Footer';

const initialPredict: PredictUiState = {
  loading: false,
  error: null,
  data: null,
};

export default function MarketingPage(props: { disableCustomTheme?: boolean }) {
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = React.useState<string | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = React.useState<string | null>(null);
  const [predictUi, setPredictUi] = React.useState<PredictUiState>(initialPredict);

  React.useEffect(() => {
    if (!predictUi.data || predictUi.loading || predictUi.error) {
      return;
    }
    document.getElementById('features')?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  }, [predictUi.data, predictUi.loading, predictUi.error]);

  return (
    <AppTheme {...props}>
      <CssBaseline enableColorScheme />

      <AppAppBar />
      <Hero
        onUploadComplete={setUploadedPreviewUrl}
        onLocalPreviewChange={setLocalPreviewUrl}
        onPredictUiChange={setPredictUi}
      />
      <div>
        {/* <LogoCollection /> */}
        <Features
          previewImageUrl={uploadedPreviewUrl}
          localPreviewUrl={localPreviewUrl}
          predictUi={predictUi}
        />
        {/* <Divider />
        <Testimonials />
        <Divider />
        <Highlights />
        <Divider />
        <Pricing />
        <Divider />
        <FAQ />
        <Divider /> */}
        <Footer />
      </div>
    </AppTheme>
  );
}
