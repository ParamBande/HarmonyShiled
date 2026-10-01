import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ChevronUp } from 'lucide-react'

/**
 * Probability gauge bar  
 * label: "AI" | "Human"
 * prob: 0-1
 */
function ProbBar({ label, prob, variant = 'ai' }) {
  const pct = Math.round(prob * 100)
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: '0.7rem', letterSpacing: '0.08em', color: '#94a3b8', textTransform: 'uppercase' }}>{label}</span>
        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: variant === 'ai' ? '#818cf8' : '#4ade80' }}>{pct}%</span>
      </div>
      <div className="prob-bar-track" style={{ height: 6 }}>
        <div
          className={variant === 'ai' ? 'prob-bar-fill-ai' : 'prob-bar-fill-human'}
          style={{ height: '100%', width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

/**
 * Single model card — Classical / ResNet / AST
 */
export default function ModelCard({ title, subtitle, badge, data, checkpoints }) {
  const [expanded, setExpanded] = useState(false)
  if (!data) return null

  const isAI = data.prediction === 'AI'
  const predColor = isAI ? '#818cf8' : '#4ade80'
  const predLabel = isAI ? 'AI-GENERATED' : 'HUMAN-COMPOSED'
  const hasError  = !!data.error

  return (
    <motion.div
      className="glass glow-border"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      style={{ borderRadius: 12, padding: '1.5rem', position: 'relative', overflow: 'hidden' }}
    >
      {/* Subtle top accent line */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 2,
        background: isAI
          ? 'linear-gradient(90deg, #818cf8, #38bdf8)'
          : 'linear-gradient(90deg, #34d399, #4ade80)',
      }} />

      {/* Header */}
      <div style={{ marginBottom: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <p style={{ fontSize: '0.6rem', letterSpacing: '0.15em', color: '#64748b', textTransform: 'uppercase', margin: 0 }}>
              {badge}
            </p>
            <h3 style={{ margin: '2px 0 2px', fontSize: '1rem', fontWeight: 700, letterSpacing: '0.08em', color: '#e2e8f0' }}>
              {title}
            </h3>
            <p style={{ margin: 0, fontSize: '0.75rem', color: '#64748b' }}>{subtitle}</p>
          </div>
          {!hasError && (
            <div style={{
              padding: '4px 10px',
              borderRadius: 6,
              background: isAI ? 'rgba(129,140,248,0.12)' : 'rgba(74,222,128,0.12)',
              border: `1px solid ${predColor}33`,
            }}>
              <span style={{ fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.12em', color: predColor }}>
                {predLabel}
              </span>
            </div>
          )}
        </div>
      </div>

      {hasError ? (
        <div style={{ color: '#f87171', fontSize: '0.8rem', fontStyle: 'italic' }}>
          ⚠ {data.error}
        </div>
      ) : (
        <>
          <ProbBar label="AI Probability"    prob={data.ai_probability}    variant="ai" />
          <ProbBar label="Human Probability" prob={data.human_probability} variant="human" />

          {/* ResNet checkpoint breakdown */}
          {checkpoints && checkpoints.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <button
                onClick={() => setExpanded(e => !e)}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: '#38bdf8', fontSize: '0.72rem', display: 'flex', alignItems: 'center', gap: 4,
                  letterSpacing: '0.05em', padding: 0,
                }}
              >
                {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                {expanded ? 'Hide' : 'Show'} individual checkpoint probabilities
              </button>

              <AnimatePresence>
                {expanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    style={{ overflow: 'hidden' }}
                  >
                    <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {checkpoints.map(ck => (
                        <div key={ck.name} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                          <span style={{ fontSize: '0.7rem', color: '#64748b', width: 160, flexShrink: 0 }}>
                            held-out {ck.name}
                          </span>
                          <div className="prob-bar-track" style={{ flex: 1, height: 4 }}>
                            <div
                              className="prob-bar-fill-ai"
                              style={{ height: '100%', width: `${Math.round(ck.ai_probability * 100)}%` }}
                            />
                          </div>
                          <span style={{ fontSize: '0.7rem', color: '#818cf8', width: 36, textAlign: 'right' }}>
                            {Math.round(ck.ai_probability * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </>
      )}
    </motion.div>
  )
}
