import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import AnalyzePage from './pages/AnalyzePage'
import HowItWorksPage from './pages/HowItWorksPage'
import ResearchPage from './pages/ResearchPage'

export default function App() {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar />
      <main style={{ flex: 1 }}>
        <Routes>
          <Route path="/"           element={<AnalyzePage />} />
          <Route path="/analyze"    element={<AnalyzePage />} />
          <Route path="/how-it-works" element={<HowItWorksPage />} />
          <Route path="/research"   element={<ResearchPage />} />
        </Routes>
      </main>
    </div>
  )
}
