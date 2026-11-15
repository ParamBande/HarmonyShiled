import { motion } from 'framer-motion'
import { AlertTriangle, CheckCircle, XCircle, Info, Activity } from 'lucide-react'

const CONFIGS = {
  AI_GENERATED: {
    icon:    <XCircle size={28} color="#f87171" />,
    label:   'AI-GENERATED',
    sub:     'UNANIMOUS VOTE',
    bg:      'rgba(248,113,113,0.06)',
    border:  'rgba(248,113,113,0.25)',
    color:   '#f87171',
  },
  LIKELY_AI_GENERATED: {
    icon:    <AlertTriangle size={28} color="#fb923c" />,
    label:   'LIKELY AI-GENERATED',
    sub:     'MAJORITY VOTE (2-TO-1)',
    bg:      'rgba(251,146,60,0.06)',
    border:  'rgba(251,146,60,0.25)',
    color:   '#fb923c',
  },
  LIKELY_HUMAN_CREATED: {
    icon:    <CheckCircle size={28} color="#a3e635" />,
    label:   'LIKELY HUMAN-CREATED',
    sub:     'MAJORITY VOTE (2-TO-1)',
    bg:      'rgba(163,230,53,0.06)',
    border:  'rgba(163,230,53,0.25)',
    color:   '#a3e635',
  },
  HUMAN_CREATED: {
    icon:    <CheckCircle size={28} color="#4ade80" />,
    label:   'HUMAN-CREATED',
    sub:     'UNANIMOUS VOTE',
    bg:      'rgba(74,222,128,0.06)',
    border:  'rgba(74,222,128,0.25)',
    color:   '#4ade80',
  },
  UNABLE_TO_DETERMINE: {
    icon:    <AlertTriangle size={28} color="#fbbf24" />,
    label:   'UNABLE TO DETERMINE',
    sub:     'INSUFFICIENT MODEL EVIDENCE',
    bg:      'rgba(251,191,36,0.06)',
    border:  'rgba(251,191,36,0.25)',
    color:   '#fbbf24',
  },

  // Legacy fallback mappings
  STRONG_CONSENSUS_AI: {
    icon:    <XCircle size={28} color="#f87171" />,
    label:   'AI-GENERATED',
    sub:     'UNANIMOUS VOTE',
    bg:      'rgba(248,113,113,0.06)',
    border:  'rgba(248,113,113,0.25)',
    color:   '#f87171',
  },
  STRONG_CONSENSUS_HUMAN: {
    icon:    <CheckCircle size={28} color="#4ade80" />,
    label:   'HUMAN-CREATED',
    sub:     'UNANIMOUS VOTE',
    bg:      'rgba(74,222,128,0.06)',
    border:  'rgba(74,222,128,0.25)',
    color:   '#4ade80',
  },
  MIXED: {
    icon:    <Info size={28} color="#fbbf24" />,
    label:   'MAJORITY DECISION',
    sub:     'PARTIAL AGREEMENT',
    bg:      'rgba(251,191,36,0.06)',
    border:  'rgba(251,191,36,0.25)',
    color:   '#fbbf24',
  },
  INCONCLUSIVE: {
    icon:    <AlertTriangle size={28} color="#fbbf24" />,
    label:   'UNABLE TO DETERMINE',
    sub:     'INSUFFICIENT MODEL EVIDENCE',
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

  const label = consensus.label || cfg.label
  const sub = consensus.sub || cfg.sub
  const aiVotes = numAI ?? consensus.num_ai_votes ?? 0
  const humanVotes = numHuman ?? consensus.num_human_votes ?? 0
  const voteSummary = consensus.vote_summary || `${aiVotes} of ${aiVotes + humanVotes} models indicate AI-generated.`

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
        gap: '0.85rem',
        textAlign: 'center',
      }}
    >
      {cfg.icon}

      <div>
        <p style={{ margin: 0, fontSize: '0.65rem', letterSpacing: '0.15em', color: '#64748b', textTransform: 'uppercase' }}>
          FINAL FORENSIC CLASSIFICATION · MAJORITY VOTING
        </p>
        <h2 style={{ margin: '4px 0 2px', fontSize: '1.25rem', fontWeight: 800, color: cfg.color, letterSpacing: '0.08em' }}>
          {label}
        </h2>
        <p style={{ margin: 0, fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', letterSpacing: '0.1em' }}>
          {sub}
        </p>
      </div>

      {/* Vote summary pill */}
      <div style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 12px',
        borderRadius: 20,
        background: 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.08)',
        fontSize: '0.75rem',
        color: '#e2e8f0',
        fontWeight: 500,
      }}>
        <span>{voteSummary}</span>
      </div>

      <p style={{ margin: 0, fontSize: '0.8rem', color: '#64748b', maxWidth: 540, lineHeight: 1.6 }}>
        {consensus.message}
      </p>

      {/* Vote tally counters */}
      <div style={{ display: 'flex', gap: '2rem', marginTop: 2 }}>
        <div style={{ textAlign: 'center' }}>
          <p style={{ margin: 0, fontSize: '1.4rem', fontWeight: 800, color: '#818cf8' }}>{aiVotes}</p>
          <p style={{ margin: 0, fontSize: '0.6rem', letterSpacing: '0.1em', color: '#64748b', textTransform: 'uppercase' }}>AI votes</p>
        </div>
        <div style={{ width: 1, background: 'rgba(255,255,255,0.06)' }} />
        <div style={{ textAlign: 'center' }}>
          <p style={{ margin: 0, fontSize: '1.4rem', fontWeight: 800, color: '#4ade80' }}>{humanVotes}</p>
          <p style={{ margin: 0, fontSize: '0.6rem', letterSpacing: '0.1em', color: '#64748b', textTransform: 'uppercase' }}>Human votes</p>
        </div>
      </div>

      {/* Diagnostic Telemetry (Preserves Mean AI Probability, Variance, and Disagreement) */}
      {(consensus.mean_ai_probability != null || consensus.variance != null) && (
        <div style={{
          width: '100%',
          marginTop: '0.5rem',
          padding: '0.6rem 0.8rem',
          borderRadius: 8,
          background: 'rgba(13,17,23,0.5)',
          border: '1px solid rgba(255,255,255,0.05)',
          display: 'flex',
          justifyContent: 'space-around',
          flexWrap: 'wrap',
          gap: '0.5rem',
          fontSize: '0.7rem',
          color: '#94a3b8',
        }}>
          {consensus.mean_ai_probability != null && (
            <span>
              Mean AI Prob: <strong style={{ color: '#e2e8f0' }}>{(consensus.mean_ai_probability * 100).toFixed(1)}%</strong>
            </span>
          )}
          {consensus.variance != null && (
            <span>
              Model Variance: <strong style={{ color: '#e2e8f0' }}>{consensus.variance.toFixed(4)}</strong>
            </span>
          )}
          <span>
            Decision Rule: <strong style={{ color: '#38bdf8' }}>Majority (Threshold 0.5)</strong>
          </span>
          {consensus.disagreement && (
            <span style={{ color: '#fb923c', fontWeight: 600 }}>
              ⚠ Disagreement Present
            </span>
          )}
        </div>
      )}

      {/* Scientific Authenticity Disclaimer */}
      <p style={{
        margin: 0, fontSize: '0.68rem', color: '#475569', fontStyle: 'italic',
        borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.75rem', width: '100%',
        lineHeight: 1.5,
      }}>
        Majority voting is a practical decision policy, not mathematical proof of authenticity.
        The three detection models are not statistically independent, and their probabilities are not calibrated.
        Disagreement between architectures is expected across out-of-distribution audio.
      </p>
    </motion.div>
  )
}
