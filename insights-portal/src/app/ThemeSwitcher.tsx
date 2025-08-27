import React from 'react'
import { usePortalStore } from '@/app/store/usePortalStore'

export const ThemeSwitcher: React.FC = () => {
  const theme = usePortalStore((s) => s.theme)
  const toggle = usePortalStore((s) => s.toggleTheme)
  const setTheme = usePortalStore((s) => s.setTheme)

  React.useEffect(() => {
    // Ensure DOM reflects current theme on mount
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <label htmlFor="theme-select" style={{ fontSize: 12, opacity: 0.8 }}>Theme</label>
      <select
        id="theme-select"
        value={theme}
        onChange={(e) => setTheme(e.target.value as 'dark'|'light')}
        title="Theme"
      >
        <option value="dark">Dark</option>
        <option value="light">Light</option>
      </select>
      <button type="button" onClick={toggle} title="Toggle theme" style={{ fontSize: 12 }}>
        {theme === 'dark' ? '☀️' : '🌙'}
      </button>
    </div>
  )
}
