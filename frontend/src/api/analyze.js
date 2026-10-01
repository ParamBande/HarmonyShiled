/** Thin wrapper around /api/analyze POST */
export async function analyzeAudio(file, onStageChange) {
  const formData = new FormData()
  formData.append('file', file)

  // Simulate staged progress (real work happens server-side)
  const stages = [
    'Loading audio',
    'Segmenting waveform',
    'Extracting spectral features',
    'Running Classical ML',
    'Running ResNet18 ensemble',
    'Running AST baseline',
    'Comparing model outputs',
  ]

  // Advance stages roughly proportionally while we wait
  let stageIdx = 0
  const advanceStage = () => {
    if (stageIdx < stages.length - 1) {
      stageIdx++
      onStageChange?.(stages[stageIdx], stageIdx)
    }
  }

  onStageChange?.(stages[0], 0)
  // First three stages fire quickly (local)
  const t1 = setTimeout(() => { advanceStage(); }, 300)
  const t2 = setTimeout(() => { advanceStage(); }, 700)
  const t3 = setTimeout(() => { advanceStage(); }, 1200)
  // Remaining stages stagger during server call
  const t4 = setTimeout(() => { advanceStage(); }, 2000)
  const t5 = setTimeout(() => { advanceStage(); }, 4000)
  const t6 = setTimeout(() => { advanceStage(); }, 8000)

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    })

    clearTimeout(t1); clearTimeout(t2); clearTimeout(t3)
    clearTimeout(t4); clearTimeout(t5); clearTimeout(t6)

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }))
      throw new Error(err.detail || 'Server error')
    }

    onStageChange?.(stages[stages.length - 1], stages.length - 1)
    return await response.json()
  } catch (e) {
    clearTimeout(t1); clearTimeout(t2); clearTimeout(t3)
    clearTimeout(t4); clearTimeout(t5); clearTimeout(t6)
    throw e
  }
}

export const STAGE_LABELS = [
  'Loading audio',
  'Segmenting waveform',
  'Extracting spectral features',
  'Running Classical ML',
  'Running ResNet18 ensemble',
  'Running AST baseline',
  'Comparing model outputs',
]
