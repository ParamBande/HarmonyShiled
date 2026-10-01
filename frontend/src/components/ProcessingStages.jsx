import { motion } from 'framer-motion'
import { CheckCircle, Circle, Loader } from 'lucide-react'

const STAGES = [
  'Loading audio',
  'Segmenting waveform',
  'Extracting spectral features',
  'Running Classical ML',
  'Running ResNet18 ensemble',
  'Running AST baseline',
  'Comparing model outputs',
]

export default function ProcessingStages({ currentStage }) {
  const currentIdx = STAGES.indexOf(currentStage)

  return (
    <div style={{ maxWidth: 480, margin: '0 auto' }}>
      {STAGES.map((stage, i) => {
        const done    = i < currentIdx
        const active  = i === currentIdx
        const pending = i > currentIdx

        return (
          <motion.div
            key={stage}
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.04 }}
            style={{
              display: 'flex', alignItems: 'center', gap: 12,
              padding: '10px 0',
              borderBottom: i < STAGES.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none',
            }}
          >
            <div style={{ width: 20, flexShrink: 0 }}>
              {done    && <CheckCircle size={16} color="#4ade80" />}
              {active  && (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                >
                  <Loader size={16} color="#38bdf8" />
                </motion.div>
              )}
              {pending && <Circle size={16} color="#334155" />}
            </div>
            <span style={{
              fontSize: '0.82rem',
              color: done ? '#64748b' : active ? '#e2e8f0' : '#334155',
              fontWeight: active ? 600 : 400,
              transition: 'color 0.3s',
            }}>
              {stage}
            </span>
            {done && (
              <span style={{ marginLeft: 'auto', fontSize: '0.65rem', color: '#4ade80', letterSpacing: '0.08em' }}>
                ✓
              </span>
            )}
          </motion.div>
        )
      })}
    </div>
  )
}
