import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import './index.css'
import Overview from './pages/Overview'
import P1Page   from './pages/P1Page'
import P2Page   from './pages/P2Page'
import P3Page   from './pages/P3Page'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/"   element={<Overview />} />
        <Route path="/p1" element={<P1Page />} />
        <Route path="/p2" element={<P2Page />} />
        <Route path="/p3" element={<P3Page />} />
        <Route path="*"   element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)
