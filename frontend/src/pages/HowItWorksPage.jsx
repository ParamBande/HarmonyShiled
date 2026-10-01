export default function HowItWorksPage() {
  const steps = [
    { n: '01', label: 'Audio Upload',          desc: 'Any audio file (MP3, WAV, FLAC, M4A) is uploaded to the server and stored in a temporary file for processing.' },
    { n: '02', label: '10-Second Segmentation', desc: 'The audio is loaded at 16 kHz mono and segmented into 10-second windows. Short tracks are zero-padded.' },
    { n: '03', label: 'Feature Extraction',     desc: 'Spectral features (centroid, bandwidth, rolloff, flatness, contrast, chroma) plus harmonic/temporal features are extracted from the first segment using librosa.' },
    { n: '04', label: 'Classical ML',           desc: 'A frozen Random Forest (Feature Set B, spectral only) trained on 400 pilot clips predicts AI/Human probability from the extracted feature vector.' },
    { n: '05', label: 'Mel-Spectrogram',        desc: 'A 128-band Mel-spectrogram (1024 FFT, 512 hop) is computed and converted to dB scale, then normalized to [0,1] for CNN input.' },
    { n: '06', label: 'ResNet18 Ensemble',      desc: '5 frozen ResNet18 checkpoints — each held out one AI generator during training — each independently predict AI probability. Their outputs are averaged.' },
    { n: '07', label: 'AST Baseline',           desc: 'An independent pretrained Audio Spectrogram Transformer (AI-Music-Detection/ai_music_detection_large_60s) processes the full audio as an external reference.' },
    { n: '08', label: 'Uncertainty Analysis',   desc: 'Variance and mean probability across all three methods are computed. High inter-model variance or mid-range mean probability triggers an INCONCLUSIVE verdict.' },
  ]

  const methods = [
    {
      title: 'Classical ML',
      color: '#38bdf8',
      desc: 'Engineered interpretable audio features. Fast, transparent, and explainable. Fragile to generators outside its training distribution.',
    },
    {
      title: 'ResNet18',
      color: '#818cf8',
      desc: 'Learns spatial patterns in Mel-spectrograms via a pretrained CNN. More flexible than handcrafted features, but still limited by small pilot dataset.',
    },
    {
      title: 'AST Baseline',
      color: '#4ade80',
      desc: 'An independent pretrained Audio Spectrogram Transformer — not trained by this project. Provides an external reference point for cross-validation of the research models.',
    },
  ]

  return (
    <div style={{ maxWidth: 860, margin: '0 auto', padding: '3rem 1.5rem' }}>
      <div style={{ marginBottom: '2.5rem' }}>
        <p style={{ fontSize: '0.6rem', letterSpacing: '0.2em', color: '#38bdf8', textTransform: 'uppercase', margin: '0 0 8px' }}>
          Methodology
        </p>
        <h1 style={{ margin: '0 0 8px', fontSize: '1.8rem', fontWeight: 800, color: '#e2e8f0' }}>How It Works</h1>
        <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b', maxWidth: 600 }}>
          HarmonyShield runs three detection methods independently and exposes disagreement rather than forcing a single verdict.
        </p>
      </div>

      {/* Pipeline steps */}
      <div style={{ marginBottom: '3rem' }}>
        {steps.map((s, i) => (
          <div key={i} style={{ display: 'flex', gap: '1.25rem', marginBottom: '1rem' }}>
            <div style={{
              width: 40, height: 40, flexShrink: 0,
              borderRadius: 8, background: 'rgba(56,189,248,0.08)',
              border: '1px solid rgba(56,189,248,0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.7rem', fontWeight: 700, color: '#38bdf8', letterSpacing: '0.05em',
            }}>
              {s.n}
            </div>
            <div style={{ paddingTop: 8 }}>
              <h4 style={{ margin: '0 0 4px', fontSize: '0.9rem', fontWeight: 600, color: '#e2e8f0' }}>{s.label}</h4>
              <p style={{ margin: 0, fontSize: '0.8rem', color: '#64748b', lineHeight: 1.6 }}>{s.desc}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Why multiple methods */}
      <div className="glass" style={{ borderRadius: 12, padding: '1.5rem', marginBottom: '2rem' }}>
        <h3 style={{ margin: '0 0 1rem', fontSize: '1rem', fontWeight: 700, color: '#e2e8f0' }}>
          Why Multiple Methods?
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
          {methods.map(m => (
            <div key={m.title} style={{
              padding: '1rem',
              borderRadius: 8,
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.05)',
            }}>
              <div style={{
                fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.12em',
                color: m.color, textTransform: 'uppercase', marginBottom: 6,
              }}>
                {m.title}
              </div>
              <p style={{ margin: 0, fontSize: '0.78rem', color: '#64748b', lineHeight: 1.6 }}>{m.desc}</p>
            </div>
          ))}
        </div>
        <p style={{ margin: 0, fontSize: '0.8rem', color: '#64748b', fontStyle: 'italic' }}>
          "Rather than assuming that one detector is always correct, HarmonyShield exposes disagreement between methods. 
          When methods disagree materially, the system reports INCONCLUSIVE instead of forcing a binary verdict."
        </p>
      </div>

      {/* Limitations box */}
      <div style={{
        borderRadius: 10, padding: '1rem 1.25rem',
        background: 'rgba(251,191,36,0.05)',
        border: '1px solid rgba(251,191,36,0.15)',
      }}>
        <p style={{ margin: '0 0 6px', fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.1em', color: '#fbbf24', textTransform: 'uppercase' }}>
          ⚠ Important Limitations
        </p>
        <ul style={{ margin: 0, padding: '0 0 0 1.2rem', fontSize: '0.78rem', color: '#64748b', lineHeight: 1.9 }}>
          <li>All models were trained on a 600-clip pilot dataset — far smaller than production-grade datasets.</li>
          <li>Generalization to generators not seen during training is limited and explicitly quantified.</li>
          <li>A correct prediction on any single song does not prove the system will generalize.</li>
          <li>INCONCLUSIVE is a valid and honest output — do not override it manually.</li>
          <li>This is a research prototype, not a certified forensic tool.</li>
        </ul>
      </div>
    </div>
  )
}
