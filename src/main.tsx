import './firebase';
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

console.log('main.tsx: Starting to load...');

const rootElement = document.getElementById('root');
if (!rootElement) {
  console.error('Root element not found!');
  document.body.innerHTML = '<div style="padding: 20px; color: red;"><h1>Error: Root element not found!</h1></div>';
} else {
  console.log('Root element found, attempting to render...');
  try {
    ReactDOM.createRoot(rootElement).render(
      <React.StrictMode>
        <App />
      </React.StrictMode>,
    );
    console.log('React rendered successfully!');
  } catch (error) {
    console.error('Error rendering React:', error);
    rootElement.innerHTML = `<div style="padding: 20px; color: red;">
      <h1>Error Loading React</h1>
      <pre>${error instanceof Error ? error.message : String(error)}</pre>
      <pre>${error instanceof Error ? error.stack : ''}</pre>
    </div>`;
  }
}

