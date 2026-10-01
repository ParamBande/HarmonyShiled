import { NavLink } from 'react-router-dom'
import { Shield } from 'lucide-react'

export default function Navbar() {
  return (
    <nav style={{
      borderBottom: '1px solid rgba(56,189,248,0.1)',
      background: 'rgba(7,11,15,0.9)',
      backdropFilter: 'blur(12px)',
      position: 'sticky', top: 0, zIndex: 100,
    }}>
      <div style={{
        maxWidth: 1200, margin: '0 auto',
        padding: '0 1.5rem',
        height: 56,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        {/* Brand */}
        <NavLink to="/" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none' }}>
          <Shield size={22} color="#38bdf8" />
          <span style={{
            fontSize: '0.95rem',
            fontWeight: 700,
            letterSpacing: '0.12em',
            color: '#e2e8f0',
            textTransform: 'uppercase',
          }}>
            HarmonyShield
          </span>
        </NavLink>

        {/* Links */}
        <div style={{ display: 'flex', gap: '2rem' }}>
          <NavLink to="/analyze"      className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>Analyze</NavLink>
          <NavLink to="/how-it-works" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>How It Works</NavLink>
          <NavLink to="/research"     className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>Research</NavLink>
        </div>
      </div>
    </nav>
  )
}
