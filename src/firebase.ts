import { initializeApp } from 'firebase/app';
import { getAnalytics } from 'firebase/analytics';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY ?? 'AIzaSyACuUi4mA7d6TVVJHvlHRuzox7CzmD68iA',
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN ?? 'capstonechexit.firebaseapp.com',
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID ?? 'capstonechexit',
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET ?? 'capstonechexit.firebasestorage.app',
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID ?? '989822629597',
  appId: import.meta.env.VITE_FIREBASE_APP_ID ?? '1:989822629597:web:c8a2225ff03e53730ecd34',
  measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID ?? 'G-RS2C2QL8T6',
};

const app = initializeApp(firebaseConfig);
const analytics = typeof window !== 'undefined' ? getAnalytics(app) : null;
const db = getFirestore(app);
const storage = getStorage(app);

export { app, analytics, db, storage };
