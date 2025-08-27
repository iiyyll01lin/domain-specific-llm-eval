import React from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './app/App'
import './app/i18n'
import './styles/theme.css'

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
