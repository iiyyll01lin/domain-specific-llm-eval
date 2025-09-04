import React from 'react'
import { usePortalStore } from '@/app/store/usePortalStore'
import type { PersonaProfile } from '@/core/types'

interface LoadedPersonaMap { [id: string]: PersonaProfile }

const personasCache: LoadedPersonaMap = {}

export const PersonaManager: React.FC = () => {
  const personaId = usePortalStore(s => s.personaId)
  const setPersonaId = usePortalStore(s => s.setPersonaId)
  const setLocale = usePortalStore(s => s.setLocale)
  const [profiles, setProfiles] = React.useState<LoadedPersonaMap>({})
  const fileInputRef = React.useRef<HTMLInputElement|null>(null)

  // Load cached persona from localStorage on mount if present
  React.useEffect(() => {
    const saved = localStorage.getItem('portal.persona')
    if (saved) {
      setPersonaId(saved)
    }
  }, [setPersonaId])

  const handleImport = async () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json,application/json'
    const file: File = await new Promise((resolve, reject) => {
      input.onchange = () => {
        const f = input.files?.[0]
        if (f) resolve(f); else reject(new Error('No file selected'))
      }
      input.click()
    })
    const text = await file.text()
    let data: PersonaProfile
    try { data = JSON.parse(text) } catch { alert('Invalid JSON'); return }
    if (data.schemaVersion !== 1) { alert('Unsupported persona schemaVersion'); return }
    personasCache[data.id] = data
    setProfiles({ ...personasCache })
    setPersonaId(data.id)
    if (data.locale) setLocale(data.locale as any)
    // Optional: apply default filters or threshold template (out of scope now)
  }

  const handleExport = () => {
    const id = personaId
    if (!id || !personasCache[id]) { alert('No persona to export'); return }
    const blob = new Blob([JSON.stringify(personasCache[id], null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `persona_${id}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const handleClear = () => {
    setPersonaId(undefined)
  }

  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
      <select value={personaId || ''} onChange={(e) => setPersonaId(e.target.value || undefined)} title="Persona profile">
        <option value=''>Persona</option>
        {Object.values(profiles).map(p => (
          <option key={p.id} value={p.id}>{p.name}</option>
        ))}
      </select>
      <button onClick={handleImport} title="Import persona JSON">Imp</button>
      <button onClick={handleExport} disabled={!personaId} title="Export current persona">Exp</button>
      <button onClick={handleClear} disabled={!personaId} title="Clear persona">x</button>
      <input ref={fileInputRef} type="file" style={{ display: 'none' }} />
    </div>
  )
}

export default PersonaManager