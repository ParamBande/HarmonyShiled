import { motion } from 'framer-motion'
import { AlertTriangle, CheckCircle, XCircle, Info } from 'lucide-react'

const CONFIGS = {
  STRONG_CONSENSUS_AI: {
    icon:    <XCircle size={28} color="#f87171" />,
    label:   'STRONG CONSENSUS',
    sub:     'AI-GENERATED',
    bg:      'rgba(248,113,113,0.06)',
    border:  'rgba(248,113,113,0.25)',
    color:   '#f87171',
  },
  STRONG_CONSENSUS_HUMAN: {
    icon:    <CheckCircle size={28} color="#4ade80" />,
    label:   'STRONG CONSENSUS',
    sub:     'HUMAN-COMPOSED',
    bg:      'rgba(74,222,128,0.06)',
    border:  'rgba(74,222,128,0.25)',
    color:   '#4ade80',
  },
  MIXED: {
    icon:    <Info size={28} color="#fbbf24" />,
    label:   'MIXED SIGNALS',
    sub:     'PARTIAL AGREEMENT',
    bg:      'rgba(251,191,36,0.06)',
    border:  'rgba(251,191,36,0.25)',
    color:   '#fbbf24',
  },
  INCONCLUSIVE: {
    icon:    <AlertTriangle size={28} color="#fbbf24" />,
    label:   '⚠ INCONCLUSIVE',
    sub:     'LOW CONFIDENCE',
    bg:      'rgba(251,191,36,0.06)',
    border:  'rgba(251,191,36,0.3)',
    color:   '#fbbf24',
  },
  ERROR: {
    icon:    <AlertTriangle size={28} color="#f87171" />,
    label:   'ERROR',
    sub:     'ANALYSIS FAILED',
    bg:      'rgba(248,113,113,0.06)',
    border:  'rgba(248,113,113,0.25)',
    color:   '#f87171',
  },
}

export default function ConsensusPanel({ consensus, numAI, numHuman }) {
  if (!consensus) return null
  const cfg = CONFIGS[consensus.status] ?? CONFIGS.ERROR

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.3 }}
      style={{
        borderRadius: 14,
        border: `1px solid ${cfg.border}`,
        background: cfg.bg,
        padding: '1.5rem 2rem',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '0.75rem',
        textAlign: 'center',
      }}
    >
      {cfg.icon}

      <div>
        <p style={{ margin: 0, fontSize: '0.65rem', letterSpacing: '0.15em', color: '#64748b', textTransform: 'uppercase' }}>
          FORENSIC CONSENSUS
        </p>
        <h2 style={{ margin: '4px 0 2px', fontSize: '1.1rem', fontWeight: 800, color: cfg.color, letterSpacing: '0.08em' }}>
          {cfg.label}
        </h2>
        <p style={{ margin: 0, fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', letterSpacing: '0.1em' }}>
          {cfg.sub}
        </p>
      </div>

      <p style={{ margin: 0, fontSize: '0.8rem', color: '#64748b', maxWidth: 520, lineHeight: 1.6 }}>
        {consensus.message}
      </p>

      {/* Vote tally */}
      <div style={{ display: 'flex', gap: '2rem', marginTop: 4 }}>
        <div style={{ textAlign: 'center' }}>
          <p style={{ margin: 0, fontSize: '1.4rem', fontWeight: 800, color: '#818cf8' }}>{numAI ?? consensus.num_ai_votes}</p>
          <p style={{ margin: 0, fontSize: '0.6rem', letterSpacing: '0.1em', color: '#64748b', textTransform: 'uppercase' }}>AI votes</p>
        </div>
        <div style={{ width: 1, background: 'rgba(255,255,255,0.06)' }} />
        <div style={{ textAlign: 'center' }}>
          <p style={{ margin: 0, fontSize: '1.4rem', fontWeight: 800, color: '#4ade80' }}>{numHuman ?? consensus.num_human_votes}</p>
          <p style={{ margin: 0, fontSize: '0.6rem', letterSpacing: '0.1em', color: '#64748b', textTransform: 'uppercase' }}>Human votes</p>
        </div>
      </div>

      <p style={{
        margin: 0, fontSize: '0.7rem', color: '#475569', fontStyle: 'italic',
        borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.75rem', width: '100%',
      }}>
        Agreement between methods increases consistency of the observed evidence, but does not establish ground truth.
      </p>
    </motion.div>
  )
}
