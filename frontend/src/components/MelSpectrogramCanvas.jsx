import { useRef, useEffect } from 'react'

/**
 * Renders the mel-spectrogram 2-D array as a colour-mapped canvas.
 * data: number[][] (rows = mel bins, cols = time frames), values in [0,1]
 */
export default function MelSpectrogramCanvas({ data }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!data || !canvasRef.current) return
    const rows = data.length
    const cols = data[0]?.length ?? 0
    if (!rows || !cols) return

    const canvas = canvasRef.current
    canvas.width  = cols
    canvas.height = rows
    const ctx = canvas.getContext('2d')
    const imgData = ctx.createImageData(cols, rows)

    // Colour map: dark-blue → teal → yellow (viridis-ish)
    const colourmap = (v) => {
      // v ∈ [0,1]
      const r = Math.round(Math.max(0, Math.min(255, (v > 0.6 ? (v - 0.6) * 640 : 0))))
      const g = Math.round(Math.max(0, Math.min(255, v < 0.5 ? v * 400 : 200 - (v - 0.5) * 80)))
      const b = Math.round(Math.max(0, Math.min(255, v < 0.4 ? 200 + v * 140 : 255 - (v - 0.4) * 600)))
      return [r, g, b]
    }

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const val = data[rows - 1 - row][col] ?? 0  // flip vertically so low freq at bottom
        const [r, g, b] = colourmap(val)
        const idx = (row * cols + col) * 4
        imgData.data[idx]     = r
        imgData.data[idx + 1] = g
        imgData.data[idx + 2] = b
        imgData.data[idx + 3] = 255
      }
    }
    ctx.putImageData(imgData, 0, 0)
  }, [data])

  if (!data || !data.length) return null

  return (
    <canvas
      ref={canvasRef}
      className="mel-canvas"
      style={{ width: '100%', height: 140, objectFit: 'fill' }}
    />
  )
}
