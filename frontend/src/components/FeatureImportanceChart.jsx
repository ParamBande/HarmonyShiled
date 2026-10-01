import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

// Group colours
const FAMILY_COLOUR = {
  'Spectral Contrast':    '#38bdf8',
  'Spectral Centroid':    '#818cf8',
  'Spectral Bandwidth':   '#a78bfa',
  'Spectral Rolloff':     '#60a5fa',
  'Spectral Flatness':    '#7dd3fc',
  'MFCC':                 '#34d399',
  'Chroma':               '#f472b6',
  'Energy':               '#fbbf24',
  'Zero-Crossing Rate':   '#fb923c',
  'Temporal':             '#a3e635',
  'Onset':                '#f87171',
  'Harmonic/Percussive':  '#e879f9',
  'Other':                '#94a3b8',
}

export default function FeatureImportanceChart({ features }) {
  if (!features || features.length === 0) return null

  const data = features.map(f => ({
    name: f.name.replace(/_/g, ' ').replace('spectral', 'sp.').replace('contrast', 'ctr'),
    importance: +(f.importance * 100).toFixed(2),
    family: f.family,
    fullName: f.name,
  }))

  return (
    <div>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 24, bottom: 0, left: 8 }}
        >
          <XAxis
            type="number"
            domain={[0, 'auto']}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={v => `${v.toFixed(1)}%`}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={110}
            tick={{ fill: '#94a3b8', fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.03)' }}
            contentStyle={{
              background: '#0d1117',
              border: '1px solid rgba(56,189,248,0.15)',
              borderRadius: 8,
              fontSize: 12,
              color: '#e2e8f0',
            }}
            formatter={(val, _name, props) => [
              `${val.toFixed(2)}%  (${props.payload.family})`,
              props.payload.fullName,
            ]}
          />
          <Bar dataKey="importance" radius={[0, 3, 3, 0]} maxBarSize={14}>
            {data.map((entry, i) => (
              <Cell key={i} fill={FAMILY_COLOUR[entry.family] || '#94a3b8'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 16px', marginTop: '0.75rem' }}>
        {[...new Set(data.map(d => d.family))].map(fam => (
          <div key={fam} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: FAMILY_COLOUR[fam] || '#94a3b8' }} />
            <span style={{ fontSize: '0.65rem', color: '#64748b' }}>{fam}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
