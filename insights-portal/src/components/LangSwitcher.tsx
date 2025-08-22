import React from 'react'
import i18n from '@/app/i18n'
import { usePortalStore } from '@/app/store/usePortalStore'

export const LangSwitcher: React.FC = () => {
  const lang = usePortalStore((s) => s.locale)
  const setLocale = usePortalStore((s) => s.setLocale)
  const onChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const v = e.target.value as 'zh-TW'|'en-US'
    setLocale(v)
    i18n.changeLanguage(v)
  }
  return (
    <select value={lang} onChange={onChange}>
      <option value="zh-TW">繁體中文</option>
      <option value="en-US">English</option>
    </select>
  )
}
