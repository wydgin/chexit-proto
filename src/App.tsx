import * as React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

// Import templates
import SignInSide from './sign-in-side/SignInSide';
import MarketingPage from './marketing-page/MarketingPage';
import AboutPage from './about/AboutPage';

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    if (import.meta.env.DEV) {
      console.error('Error caught by boundary:', error, errorInfo);
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box sx={{ p: 4, backgroundColor: '#fff', minHeight: '100vh' }}>
          <Typography variant="h4" color="error" gutterBottom>
            Something went wrong
          </Typography>
          <Typography variant="body1" sx={{ mt: 2 }}>
            {this.state.error?.message}
          </Typography>
        </Box>
      );
    }

    return this.props.children;
  }
}

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/dashboard" element={<MarketingPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/" element={<SignInSide />} />
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;

