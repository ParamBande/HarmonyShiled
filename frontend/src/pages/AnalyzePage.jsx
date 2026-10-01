import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Music, Clock, FileAudio, RotateCcw } from 'lucide-react'
import { analyzeAudio } from '../api/analyze'
import ProcessingStages from '../components/ProcessingStages'
import ModelCard from '../components/ModelCard'
import ConsensusPanel from '../components/ConsensusPanel'
import FeatureImportanceChart from '../components/FeatureImportanceChart'
import MelSpectrogramCanvas from '../components/MelSpectrogramCanvas'

// ─── Helpers ─────────────────────────────────────────────────────────────────
const fmtBytes = b => b < 1024 * 1024 ? `${(b/1024).toFixed(1)} KB` : `${(b/1024/1024).toFixed(2)} MB`
const fmtDur   = s => {
  if (!s) return '—'
  const m = Math.floor(s / 60), sec = Math.floor(s % 60)
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`
}
const ACCEPTED = '.mp3,.wav,.flac,.m4a,.ogg'

// ─── Waveform bars (decorative SVG) ──────────────────────────────────────────
function WaveformBars({ n = 60, height = 48 }) {
  return (
    <svg width="100%" height={height} style={{ display: 'block' }}>
      {Array.from({ length: n }, (_, i) => {
        const x    = (i / (n - 1)) * 100
        const amp  = 0.2 + 0.8 * Math.abs(Math.sin(i * 0.45) * Math.cos(i * 0.18))
        const barH = amp * (height - 4)
        return (
          <rect
            key={i}
            x={`${x}%`}
            y={(height - barH) / 2}
            width={1.5}
            height={barH}
            rx={1}
            fill={`rgba(56,189,248,${0.25 + amp * 0.45})`}
          />
        )
      })}
    </svg>
  )
}

// ─── Upload Zone ─────────────────────────────────────────────────────────────
function UploadZone({ onFile }) {
  const [drag, setDrag] = useState(false)
  const input = useRef()

  const handleDrop = useCallback(e => {
    e.preventDefault(); setDrag(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onFile(f)
  }, [onFile])

  return (
    <div>
      {/* Hero */}
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <p style={{ fontSize: '0.65rem', letterSpacing: '0.25em', color: '#38bdf8', textTransform: 'uppercase', margin: '0 0 12px' }}>
            AI Audio Forensics Research Prototype
          </p>
          <h1 style={{
            margin: '0 0 8px',
            fontSize: 'clamp(2rem, 5vw, 3.2rem)',
            fontWeight: 800,
            letterSpacing: '0.05em',
            background: 'linear-gradient(135deg, #e2e8f0 0%, #38bdf8 50%, #818cf8 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            HARMONYSHIELD
          </h1>
          <p style={{ margin: '0 0 6px', fontSize: '0.9rem', color: '#64748b', letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            AI-Generated Music Forensics
          </p>
          <p style={{ margin: '0 0 4px', fontSize: '0.8rem', color: '#475569' }}>
            Three detection approaches · Transparent uncertainty · Honest disagreement
          </p>
        </motion.div>
      </div>

      {/* Drop zone */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        onClick={() => input.current.click()}
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${drag ? '#38bdf8' : 'rgba(56,189,248,0.2)'}`,
          borderRadius: 16,
          padding: '3rem 2rem',
          textAlign: 'center',
          cursor: 'pointer',
          background: drag ? 'rgba(56,189,248,0.04)' : 'rgba(13,17,23,0.6)',
          transition: 'all 0.2s',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <input
          ref={input}
          type="file"
          accept={ACCEPTED}
          style={{ display: 'none' }}
          onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f) }}
        />

        <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, opacity: 0.4 }}>
          <WaveformBars n={80} height={36} />
        </div>

        <Upload size={36} color="#38bdf8" style={{ marginBottom: 16, opacity: 0.8 }} />
        <h3 style={{ margin: '0 0 8px', fontSize: '1.1rem', fontWeight: 600, color: '#e2e8f0' }}>
          Drop an audio file here
        </h3>
        <p style={{ margin: '0 0 16px', fontSize: '0.8rem', color: '#64748b' }}>
          or click to browse
        </p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: 8, flexWrap: 'wrap' }}>
          {['MP3', 'WAV', 'FLAC', 'M4A'].map(fmt => (
            <span key={fmt} style={{
              padding: '3px 10px',
              borderRadius: 4,
              background: 'rgba(56,189,248,0.08)',
              border: '1px solid rgba(56,189,248,0.15)',
              fontSize: '0.65rem', letterSpacing: '0.1em', color: '#38bdf8',
            }}>{fmt}</span>
          ))}
        </div>
      </motion.div>

      {/* Tagline */}
      <p style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.75rem', color: '#334155' }}>
        An experimental prototype. Results should not be treated as definitive ground truth.
      </p>
    </div>
  )
}

// ─── File Preview ─────────────────────────────────────────────────────────────
function FilePreview({ file, onAnalyze, analyzing }) {
  const url = URL.createObjectURL(file)
  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
      <div className="glass" style={{ borderRadius: 12, padding: '1.25rem', marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: '0.75rem' }}>
          <FileAudio size={20} color="#38bdf8" />
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <p style={{ margin: 0, fontWeight: 600, color: '#e2e8f0', fontSize: '0.9rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {file.name}
            </p>
            <p style={{ margin: 0, fontSize: '0.7rem', color: '#64748b' }}>{fmtBytes(file.size)}</p>
          </div>
        </div>
        <audio
          src={url}
          controls
          style={{ width: '100%', height: 36, outline: 'none' }}
        />
      </div>

      <button
        onClick={onAnalyze}
        disabled={analyzing}
        style={{
          width: '100%',
          padding: '0.85rem',
          borderRadius: 10,
          border: 'none',
          background: analyzing
            ? 'rgba(56,189,248,0.15)'
            : 'linear-gradient(135deg, #0ea5e9, #818cf8)',
          color: '#fff',
          fontWeight: 700,
          fontSize: '0.9rem',
          letterSpacing: '0.1em',
          cursor: analyzing ? 'not-allowed' : 'pointer',
          transition: 'opacity 0.2s',
        }}
      >
        {analyzing ? 'ANALYZING…' : 'ANALYZE AUDIO'}
      </button>
    </motion.div>
  )
}

// ─── Results Dashboard ────────────────────────────────────────────────────────
function ResultsDashboard({ result, onReset }) {
  const { file: fi, classical, resnet18, ast, consensus, features, mel_spectrogram } = result

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <p style={{ margin: 0, fontSize: '0.6rem', letterSpacing: '0.2em', color: '#38bdf8', textTransform: 'uppercase' }}>
            Forensic Analysis Complete
          </p>
          <h2 style={{ margin: '4px 0 4px', fontSize: '1.3rem', fontWeight: 700, color: '#e2e8f0' }}>
            {fi.name}
          </h2>
          <div style={{ display: 'flex', gap: '1.5rem' }}>
            <span style={{ fontSize: '0.75rem', color: '#64748b', display: 'flex', alignItems: 'center', gap: 4 }}>
              <Clock size={12} /> {fmtDur(fi.duration)}
            </span>
            <span style={{ fontSize: '0.75rem', color: '#64748b', display: 'flex', alignItems: 'center', gap: 4 }}>
              <Music size={12} /> {fi.num_segments} segment{fi.num_segments !== 1 ? 's' : ''} analysed
            </span>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
              {fi.sample_rate / 1000} kHz
            </span>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
              {result.inference_time_s}s inference
            </span>
          </div>
        </div>
        <button
          onClick={onReset}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '6px 14px', borderRadius: 7,
            background: 'transparent',
            border: '1px solid rgba(56,189,248,0.2)',
            color: '#38bdf8', cursor: 'pointer', fontSize: '0.75rem',
          }}
        >
          <RotateCcw size={13} /> New Analysis
        </button>
      </div>

      {/* Consensus */}
      <div style={{ marginBottom: '2rem' }}>
        <ConsensusPanel consensus={consensus} />
      </div>

      {/* Three model cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
        <ModelCard
          badge="Method 1 — Plan A"
          title="CLASSICAL ML"
          subtitle="Spectral Features + Random Forest · Feature Set B"
          data={classical}
        />
        <ModelCard
          badge="Method 2 — Plan A"
          title="RESNET18"
          subtitle={`Mel-Spectrogram CNN Ensemble · ${resnet18.ensemble_size} frozen checkpoints`}
          data={resnet18}
          checkpoints={resnet18.checkpoint_results}
        />
        <ModelCard
          badge="Method 3 — Plan B · External baseline"
          title="PRETRAINED AST"
          subtitle="AI-Music-Detection (HuggingFace) · Independent reference"
          data={ast}
        />
      </div>

      {/* Mel-spectrogram */}
      {mel_spectrogram && (
        <div className="glass" style={{ borderRadius: 12, padding: '1.25rem', marginBottom: '1.5rem' }}>
          <p style={{ margin: '0 0 4px', fontSize: '0.6rem', letterSpacing: '0.15em', color: '#64748b', textTransform: 'uppercase' }}>
            Audio Representation
          </p>
          <h4 style={{ margin: '0 0 0.75rem', fontSize: '0.85rem', fontWeight: 600, color: '#e2e8f0' }}>
            Mel-Spectrogram
          </h4>
          <MelSpectrogramCanvas data={mel_spectrogram} />
          <p style={{ margin: '0.5rem 0 0', fontSize: '0.68rem', color: '#475569', fontStyle: 'italic' }}>
            Mel-spectrogram representation used by the ResNet18 detector. Low frequencies at bottom, high at top; brighter = higher energy.
          </p>
        </div>
      )}

      {/* Feature importance */}
      {features && features.length > 0 && (
        <div className="glass" style={{ borderRadius: 12, padding: '1.25rem', marginBottom: '1.5rem' }}>
          <p style={{ margin: '0 0 4px', fontSize: '0.6rem', letterSpacing: '0.15em', color: '#64748b', textTransform: 'uppercase' }}>
            Classical ML Transparency
          </p>
          <h4 style={{ margin: '0 0 4px', fontSize: '0.85rem', fontWeight: 600, color: '#e2e8f0' }}>
            Feature Evidence
          </h4>
          <p style={{ margin: '0 0 1rem', fontSize: '0.72rem', color: '#64748b' }}>
            Top features from the trained Random Forest model. Feature importance indicates which variables influenced the model's decisions during training.
            It does not establish that a feature causes AI generation.
          </p>
          <FeatureImportanceChart features={features} />
        </div>
      )}

      {/* Segment info */}
      <div className="glass" style={{ borderRadius: 12, padding: '1rem 1.25rem', marginBottom: '1.5rem' }}>
        <p style={{ margin: '0 0 4px', fontSize: '0.6rem', letterSpacing: '0.15em', color: '#64748b', textTransform: 'uppercase' }}>
          Segment Analysis
        </p>
        <p style={{ margin: 0, fontSize: '0.85rem', color: '#94a3b8' }}>
          Audio was segmented into <strong style={{ color: '#e2e8f0' }}>{fi.num_segments}</strong> × 10-second window{fi.num_segments !== 1 ? 's' : ''}.
          Classical ML and ResNet18 used the first segment; the AST baseline processes the full file in 60-second chunks.
        </p>
      </div>
    </motion.div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function AnalyzePage() {
  const [file,      setFile]      = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [stage,     setStage]     = useState('')
  const [result,    setResult]    = useState(null)
  const [error,     setError]     = useState(null)

  const handleFile = f => {
    setFile(f)
    setResult(null)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!file) return
    setAnalyzing(true)
    setError(null)
    setResult(null)
    try {
      const data = await analyzeAudio(file, (s) => setStage(s))
      setResult(data)
    } catch (e) {
      setError(e.message || 'Analysis failed')
    } finally {
      setAnalyzing(false)
    }
  }

  const handleReset = () => {
    setFile(null); setResult(null); setError(null); setStage('')
  }

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: '3rem 1.5rem' }}>
      <AnimatePresence mode="wait">
        {/* Upload screen */}
        {!file && !result && (
          <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <UploadZone onFile={handleFile} />
          </motion.div>
        )}

        {/* File selected, not yet analysing */}
        {file && !analyzing && !result && (
          <motion.div key="preview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div style={{ maxWidth: 520, margin: '0 auto', paddingTop: '4rem' }}>
              <FilePreview file={file} onAnalyze={handleAnalyze} analyzing={false} />
              {error && (
                <motion.div
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  style={{
                    marginTop: '1rem', padding: '1rem',
                    borderRadius: 10,
                    background: 'rgba(248,113,113,0.08)',
                    border: '1px solid rgba(248,113,113,0.3)',
                    color: '#f87171', fontSize: '0.82rem',
                  }}
                >
                  ⚠ {error}
                </motion.div>
              )}
              <button
                onClick={handleReset}
                style={{
                  marginTop: '0.75rem', width: '100%',
                  background: 'none', border: 'none', color: '#64748b',
                  cursor: 'pointer', fontSize: '0.75rem',
                }}
              >
                Choose a different file
              </button>
            </div>
          </motion.div>
        )}

        {/* Analyzing */}
        {analyzing && (
          <motion.div key="analyzing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div style={{ maxWidth: 520, margin: '0 auto', paddingTop: '4rem', textAlign: 'center' }}>
              <div style={{ position: 'relative', marginBottom: '2rem', overflow: 'hidden', borderRadius: 8, height: 60 }}>
                <WaveformBars n={80} height={60} />
                <div className="scanline" />
              </div>
              <p style={{
                fontSize: '0.65rem', letterSpacing: '0.2em', color: '#38bdf8',
                textTransform: 'uppercase', marginBottom: '1.5rem',
              }}>
                Running Forensic Analysis
              </p>
              <div className="glass" style={{ borderRadius: 12, padding: '1.5rem' }}>
                <ProcessingStages currentStage={stage} />
              </div>
            </div>
          </motion.div>
        )}

        {/* Results */}
        {result && !analyzing && (
          <motion.div key="results" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <ResultsDashboard result={result} onReset={handleReset} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
