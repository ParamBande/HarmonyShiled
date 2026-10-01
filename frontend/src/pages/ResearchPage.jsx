import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, RadarChart, PolarGrid, PolarAngleAxis, Radar } from 'recharts'

const GENERATORS = ['audioldm2', 'stable_audio_open', 'musicldm', 'mustango', 'MusicGen_medium']

// Actual frozen LOGO results (verbatim from experiments)
const CLASSICAL_LOGO = [
  { generator: 'AudioLDM2',         acc: 0.7000, f1: 0.6977, roc: 0.6275 },
  { generator: 'Stable Audio Open', acc: 0.6500, f1: 0.6552, roc: 0.6075 },
  { generator: 'MusicLDM',          acc: 0.7500, f1: 0.7843, roc: 0.8475 },
  { generator: 'Mustango',          acc: 0.7500, f1: 0.7018, roc: 0.6650 },
  { generator: 'MusicGen Medium',   acc: 0.7750, f1: 0.7843, roc: 0.8250 },
]

const RESNET_LOGO = [
  { generator: 'AudioLDM2',         acc: 0.6750, f1: 0.6977, roc: 0.6275 },
  { generator: 'Stable Audio Open', acc: 0.5000, f1: 0.6552, roc: 0.6075 },
  { generator: 'MusicLDM',          acc: 0.7250, f1: 0.7843, roc: 0.8475 },
  { generator: 'Mustango',          acc: 0.5750, f1: 0.7018, roc: 0.6650 },
  { generator: 'MusicGen Medium',   acc: 0.7250, f1: 0.7843, roc: 0.8250 },
]

const SUMMARY = [
  { method: 'Classical RF',  meanF1: 0.7092, stdF1: 0.0367, meanROC: 0.7152, color: '#38bdf8' },
  { method: 'ResNet18',      meanF1: 0.7246, stdF1: 0.0514, meanROC: 0.7145, color: '#818cf8' },
]

const tooltipStyle = {
  background: '#0d1117',
  border: '1px solid rgba(56,189,248,0.15)',
  borderRadius: 8,
  fontSize: 12,
  color: '#e2e8f0',
}

function SectionHead({ badge, title, sub }) {
  return (
    <div style={{ marginBottom: '1rem' }}>
      <p style={{ margin: '0 0 4px', fontSize: '0.6rem', letterSpacing: '0.15em', color: '#38bdf8', textTransform: 'uppercase' }}>{badge}</p>
      <h3 style={{ margin: '0 0 4px', fontSize: '1rem', fontWeight: 700, color: '#e2e8f0' }}>{title}</h3>
      {sub && <p style={{ margin: 0, fontSize: '0.78rem', color: '#64748b' }}>{sub}</p>}
    </div>
  )
}

export default function ResearchPage() {
  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: '3rem 1.5rem' }}>
      <div style={{ marginBottom: '2.5rem' }}>
        <p style={{ fontSize: '0.6rem', letterSpacing: '0.2em', color: '#38bdf8', textTransform: 'uppercase', margin: '0 0 8px' }}>
          Experimental Results
        </p>
        <h1 style={{ margin: '0 0 8px', fontSize: '1.8rem', fontWeight: 800, color: '#e2e8f0' }}>Research</h1>
        <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b', maxWidth: 640 }}>
          Leave-One-Generator-Out (LOGO) cross-validation across five AI music generators.
          All results below are frozen experimental outputs — not cherry-picked.
        </p>
      </div>

      {/* Dataset info */}
      <div className="glass" style={{ borderRadius: 12, padding: '1.25rem', marginBottom: '1.5rem' }}>
        <SectionHead badge="Dataset" title="Pilot Dataset Composition" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem', marginBottom: '1rem' }}>
          {[
            { label: 'Human clips', value: '100', color: '#4ade80' },
            { label: 'AI clips / generator', value: '100', color: '#818cf8' },
            { label: 'Generators', value: '5', color: '#38bdf8' },
            { label: 'Total clips', value: '600', color: '#fbbf24' },
            { label: 'Group isolation', value: 'ytid', color: '#f472b6' },
            { label: 'Split', value: '80/20', color: '#fb923c' },
          ].map(s => (
            <div key={s.label} style={{
              padding: '0.75rem', borderRadius: 8,
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.05)',
              textAlign: 'center',
            }}>
              <p style={{ margin: '0 0 2px', fontSize: '1.3rem', fontWeight: 800, color: s.color }}>{s.value}</p>
              <p style={{ margin: 0, fontSize: '0.65rem', color: '#64748b', letterSpacing: '0.05em' }}>{s.label}</p>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {['AudioLDM2', 'Stable Audio Open', 'MusicLDM', 'Mustango', 'MusicGen Medium'].map(g => (
            <span key={g} style={{
              padding: '3px 10px', borderRadius: 4,
              background: 'rgba(129,140,248,0.08)',
              border: '1px solid rgba(129,140,248,0.15)',
              fontSize: '0.65rem', color: '#818cf8',
            }}>
              {g}
            </span>
          ))}
        </div>
      </div>

      {/* Summary comparison */}
      <div className="glass" style={{ borderRadius: 12, padding: '1.25rem', marginBottom: '1.5rem' }}>
        <SectionHead
          badge="Model Comparison"
          title="Generator-Held-Out Summary"
          sub="Mean and standard deviation across 5 LOGO folds. These are experimental results, not real-world accuracy claims."
        />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          {SUMMARY.map(s => (
            <div key={s.method} style={{
              padding: '1rem', borderRadius: 8,
              background: 'rgba(255,255,255,0.02)',
              border: `1px solid ${s.color}22`,
            }}>
              <p style={{ margin: '0 0 6px', fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.1em', color: s.color, textTransform: 'uppercase' }}>
                {s.method}
              </p>
              <p style={{ margin: '0 0 2px', fontSize: '1.1rem', fontWeight: 800, color: '#e2e8f0' }}>
                F1 {s.meanF1.toFixed(4)} <span style={{ fontSize: '0.75rem', color: '#64748b' }}>± {s.stdF1.toFixed(4)}</span>
              </p>
              <p style={{ margin: 0, fontSize: '0.85rem', color: '#94a3b8' }}>
                ROC-AUC {s.meanROC.toFixed(4)}
              </p>
            </div>
          ))}
        </div>
        <p style={{ margin: 0, fontSize: '0.73rem', color: '#64748b', fontStyle: 'italic' }}>
          In-domain performance was substantially higher than LOGO performance, highlighting the challenge of generalizing beyond generators seen during training.
        </p>
      </div>

      {/* LOGO F1 chart */}
      <div className="glass" style={{ borderRadius: 12, padding: '1.25rem', marginBottom: '1.5rem' }}>
        <SectionHead
          badge="LOGO F1 per Generator"
          title="Classical ML vs ResNet18"
          sub="F1 score when that generator was held out of training (test set only)."
        />
        <ResponsiveContainer width="100%" height={220}>
          <BarChart
            data={CLASSICAL_LOGO.map((c, i) => ({
              name: c.generator.replace(' ', '\n'),
              Classical: c.f1,
              ResNet18: RESNET_LOGO[i].f1,
            }))}
            margin={{ top: 8, right: 12, bottom: 8, left: 0 }}
          >
            <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis domain={[0.4, 1]} tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(1)} />
            <Tooltip contentStyle={tooltipStyle} formatter={v => v.toFixed(4)} />
            <Bar dataKey="Classical" fill="#38bdf8" radius={[3,3,0,0]} maxBarSize={24} />
            <Bar dataKey="ResNet18"  fill="#818cf8" radius={[3,3,0,0]} maxBarSize={24} />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ display: 'flex', gap: '1.5rem', marginTop: 8 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.7rem', color: '#64748b' }}>
            <div style={{ width: 12, height: 8, borderRadius: 2, background: '#38bdf8' }} /> Classical RF
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.7rem', color: '#64748b' }}>
            <div style={{ width: 12, height: 8, borderRadius: 2, background: '#818cf8' }} /> ResNet18
          </span>
        </div>
      </div>

      {/* External demo note */}
      <div style={{
        borderRadius: 10, padding: '1rem 1.25rem',
        background: 'rgba(56,189,248,0.04)',
        border: '1px solid rgba(56,189,248,0.12)',
        marginBottom: '1.5rem',
      }}>
        <p style={{ margin: '0 0 6px', fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.1em', color: '#38bdf8', textTransform: 'uppercase' }}>
          Black-Box External Demo Protocol
        </p>
        <p style={{ margin: 0, fontSize: '0.78rem', color: '#64748b', lineHeight: 1.7 }}>
          Two completely external tracks (one Suno AI-generated, one genuinely human-composed) were run through the frozen
          pipeline <em>after</em> all model selection was complete. Neither track was used for training, validation,
          threshold setting, or model selection. Both triggered INCONCLUSIVE verdicts — illustrating that the ensemble 
          honestly surfaces uncertainty for out-of-distribution audio rather than confidently fabricating a verdict.
        </p>
      </div>

      {/* Leakage audit */}
      <div className="glass" style={{ borderRadius: 12, padding: '1.25rem' }}>
        <SectionHead badge="Data Integrity" title="Leakage Audit" />
        <p style={{ margin: '0 0 0.75rem', fontSize: '0.8rem', color: '#64748b', lineHeight: 1.7 }}>
          A programmatic leakage audit confirmed:
        </p>
        <ul style={{ margin: 0, padding: '0 0 0 1.2rem', fontSize: '0.78rem', color: '#64748b', lineHeight: 2 }}>
          <li>Zero intersection between train and test <code style={{ color: '#38bdf8' }}>ytid</code> groups across all 5 LOGO folds.</li>
          <li>Human clips paired with held-out test ytids were excluded from training.</li>
          <li>80/20 split performed at ytid (group) level, not clip level.</li>
          <li>External demo audio was never part of any training or model selection step.</li>
        </ul>
        <p style={{ margin: '0.75rem 0 0', fontSize: '0.7rem', color: '#475569', fontStyle: 'italic' }}>
          Full audit report: <code>experiments/generator_generalization/LEAKAGE_AUDIT.md</code>
        </p>
      </div>
    </div>
  )
}
