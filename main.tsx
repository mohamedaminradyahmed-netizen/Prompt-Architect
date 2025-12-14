import React from 'react';
import ReactDOM from 'react-dom/client';
import PromptEngineer from './prompt-engineer';

/**
 * نقطة الدخول الرئيسية للتطبيق
 * تقوم بربط مكون PromptEngineer مع عنصر DOM المحدد
 */
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <PromptEngineer />
  </React.StrictMode>
);


