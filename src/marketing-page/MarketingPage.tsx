import * as React from 'react';
import CssBaseline from '@mui/material/CssBaseline';
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

export default function MarketingPage(props: { disableCustomTheme?: boolean }) {
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = React.useState<string | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = React.useState<string | null>(null);

  return (
    <AppTheme {...props}>
      <CssBaseline enableColorScheme />

      <AppAppBar />
      <Hero
        onUploadComplete={setUploadedPreviewUrl}
        onLocalPreviewChange={setLocalPreviewUrl}
      />
      <div>
        {/* <LogoCollection /> */}
        <Features
          previewImageUrl={uploadedPreviewUrl}
          localPreviewUrl={localPreviewUrl}
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
