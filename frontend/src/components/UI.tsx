import { useEffect, useRef, useState } from 'react'

// ── PARTICLE CANVAS ───────────────────────────────────────────────────────
export function ParticleCanvas() {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current!; const ctx = c.getContext('2d')!
    let raf: number; let mouse = { x:-2000, y:-2000 }
    const resize = () => { c.width = window.innerWidth; c.height = window.innerHeight }
    resize()
    window.addEventListener('resize', resize)
    window.addEventListener('mousemove', e => { mouse.x=e.clientX; mouse.y=e.clientY })
    const ps = Array.from({length:50}, () => ({
      x: Math.random()*c.width, y: Math.random()*c.height,
      vx: (Math.random()-.5)*.25, vy: (Math.random()-.5)*.25,
      r: Math.random()*1.2+.4, p: Math.random()*Math.PI*2,
    }))
    const draw = () => {
      ctx.clearRect(0,0,c.width,c.height)
      for (const p of ps) {
        p.x+=p.vx; p.y+=p.vy; p.p+=.015
        if(p.x<0||p.x>c.width)  p.vx*=-1
        if(p.y<0||p.y>c.height) p.vy*=-1
        const dx=mouse.x-p.x, dy=mouse.y-p.y, d=Math.hypot(dx,dy)
        if(d<130){ p.x+=dx*.0018; p.y+=dy*.0018 }
        ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2)
        ctx.fillStyle=`rgba(79,142,247,${.35+Math.sin(p.p)*.15})`; ctx.fill()
      }
      for (let i=0;i<ps.length;i++) for (let j=i+1;j<ps.length;j++) {
        const d=Math.hypot(ps[i].x-ps[j].x, ps[i].y-ps[j].y)
        if(d<100){ ctx.beginPath(); ctx.moveTo(ps[i].x,ps[i].y); ctx.lineTo(ps[j].x,ps[j].y)
          ctx.strokeStyle=`rgba(79,142,247,${(1-d/100)*.18})`; ctx.lineWidth=.5; ctx.stroke() }
      }
      raf=requestAnimationFrame(draw)
    }
    draw()
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  },[])
  return <canvas ref={ref} style={{ position:'fixed',top:0,left:0,width:'100%',height:'100%',pointerEvents:'none',zIndex:0 }}/>
}

// ── SPARKLINE ─────────────────────────────────────────────────────────────
export function Sparkline({ data, color='#4f8ef7', w=90, h=28 }: { data:number[], color?:string, w?:number, h?:number }) {
  if(data.length<2) return null
  const mx=Math.max(...data), mn=Math.min(...data), rng=mx-mn||1
  const pts=data.map((v,i)=>`${(i/(data.length-1))*w},${h-((v-mn)/rng)*h}`).join(' ')
  const last=pts.split(' ').pop()!.split(',')
  return (
    <svg width={w} height={h} style={{overflow:'visible',display:'block',flexShrink:0}}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      <circle cx={last[0]} cy={last[1]} r="2.5" fill={color}/>
    </svg>
  )
}

// ── ANIMATED AREA CHART ───────────────────────────────────────────────────
export function AreaChart({ series, h=140, yMin, yMax }: {
  series: {label:string, color:string, fill:string, data:number[]}[]
  h?: number, yMin?:number, yMax?:number
}) {
  const W=560, pad={t:8,b:24,l:36,r:8}
  const allV=series.flatMap(s=>s.data)
  const mn=yMin??Math.min(...allV), mx=yMax??Math.max(...allV), rng=mx-mn||1
  const iw=W-pad.l-pad.r, ih=h-pad.t-pad.b
  const n=Math.max(...series.map(s=>s.data.length))
  const xp=(i:number)=>pad.l+(i/(n-1||1))*iw
  const yp=(v:number)=>pad.t+ih-((v-mn)/rng)*ih
  return (
    <svg viewBox={`0 0 ${W} ${h}`} style={{width:'100%',height:h}} preserveAspectRatio="none">
      {[0,.25,.5,.75,1].map(f=>(
        <line key={f} x1={pad.l} x2={W-pad.r} y1={pad.t+ih*(1-f)} y2={pad.t+ih*(1-f)}
          stroke="rgba(255,255,255,0.04)" strokeWidth="1"/>
      ))}
      {[0,.5,1].map(f=>(
        <text key={f} x={pad.l-4} y={pad.t+ih*(1-f)+4} textAnchor="end"
          style={{fontSize:9,fill:'rgba(255,255,255,0.25)',fontFamily:'JetBrains Mono,monospace'}}>
          {Math.round(mn+rng*f)}
        </text>
      ))}
      {series.map(s=>{
        const pts=s.data.map((_,i)=>`${xp(i)},${yp(s.data[i])}`).join(' ')
        const areaPath=`M ${xp(0)} ${yp(s.data[0])} ` +
          s.data.map((_,i)=>`L ${xp(i)} ${yp(s.data[i])}`).join(' ') +
          ` L ${xp(n-1)} ${pad.t+ih} L ${pad.l} ${pad.t+ih} Z`
        return (
          <g key={s.label}>
            <path d={areaPath} fill={s.fill}/>
            <polyline points={pts} fill="none" stroke={s.color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          </g>
        )
      })}
    </svg>
  )
}

// ── BAR CHART ─────────────────────────────────────────────────────────────
export function BarChart({ data, color='#4f8ef7', h=56 }: { data:{label:string,value:number}[], color?:string, h?:number }) {
  const mx=Math.max(...data.map(d=>d.value),1)
  return (
    <div style={{display:'flex',alignItems:'flex-end',gap:4,height:h}}>
      {data.map((d,i)=>(
        <div key={i} style={{flex:1,display:'flex',flexDirection:'column',alignItems:'center',gap:2}}>
          <div style={{fontSize:7,color:'rgba(255,255,255,0.25)',fontFamily:'monospace'}}>{d.value}</div>
          <div style={{width:'100%',borderRadius:'2px 2px 0 0',background:color,
            height:`${(d.value/mx)*(h-18)}px`,transition:'height .7s ease',
            opacity:.5+(i/data.length)*.5}}/>
          <div style={{fontSize:7,color:'rgba(255,255,255,0.2)',fontFamily:'monospace'}}>{d.label}</div>
        </div>
      ))}
    </div>
  )
}

// ── DONUT ─────────────────────────────────────────────────────────────────
export function Donut({ value, color, size=64, thickness=8 }: { value:number, color:string, size?:number, thickness?:number }) {
  const r=(size-thickness)/2, circ=2*Math.PI*r, dash=(value/100)*circ
  return (
    <svg width={size} height={size} style={{transform:'rotate(-90deg)',flexShrink:0}}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={thickness}/>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={thickness}
        strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
        style={{transition:'stroke-dasharray .8s cubic-bezier(.4,0,.2,1)'}}/>
    </svg>
  )
}

// ── RADAR CHART ───────────────────────────────────────────────────────────
export function Radar({ data, size=130, color='#4f8ef7' }: { data:{label:string,value:number}[], size?:number, color?:string }) {
  const cx=size/2, cy=size/2, r=size/2-22, n=data.length
  const pts=data.map((d,i)=>{
    const a=(i/n)*Math.PI*2-Math.PI/2, rv=(d.value/100)*r
    return {x:cx+rv*Math.cos(a),y:cy+rv*Math.sin(a),ax:cx+r*Math.cos(a),ay:cy+r*Math.sin(a),
            lx:cx+(r+15)*Math.cos(a),ly:cy+(r+15)*Math.sin(a),label:d.label}
  })
  const poly=pts.map(p=>`${p.x},${p.y}`).join(' ')
  return (
    <svg width={size} height={size}>
      {[.25,.5,.75,1].map(f=>{
        const rp=data.map((_,i)=>{const a=(i/n)*Math.PI*2-Math.PI/2;return `${cx+r*f*Math.cos(a)},${cy+r*f*Math.sin(a)}`}).join(' ')
        return <polygon key={f} points={rp} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>
      })}
      {pts.map((p,i)=><line key={i} x1={cx} y1={cy} x2={p.ax} y2={p.ay} stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>)}
      <polygon points={poly} fill={`${color}1a`} stroke={color} strokeWidth="1.5"/>
      {pts.map((p,i)=><circle key={i} cx={p.x} cy={p.y} r="3" fill={color}/>)}
      {pts.map((p,i)=><text key={i} x={p.lx} y={p.ly} textAnchor="middle" dominantBaseline="middle"
        style={{fontSize:8,fill:'rgba(255,255,255,0.35)',fontFamily:'monospace'}}>{p.label}</text>)}
    </svg>
  )
}

// ── SIDEBAR NAV ───────────────────────────────────────────────────────────
export function Sidebar({ active }: { active: 'overview'|'p1'|'p2'|'p3' }) {
  const items = [
    { key:'overview', href:'/',   color:'#4f8ef7', label:'Overview' },
    { key:'p1',       href:'/p1', color:'#ff4757', label:'Tower Anomaly' },
    { key:'p2',       href:'/p2', color:'#9b7cf8', label:'Mobile Experience' },
    { key:'p3',       href:'/p3', color:'#00d68f', label:'Churn Risk' },
  ]
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-mark">📡</div>
        <div className="sidebar-logo-title">Telekom AI</div>
        <div className="sidebar-logo-sub">MOBILE INTELLIGENCE</div>
      </div>
      <nav style={{padding:'8px 0'}}>
        {items.map(it=>(
          <a key={it.key} href={it.href} className={`nav-item${active===it.key?' active':''}`}>
            <span className="nav-dot" style={{background:it.color,opacity:active===it.key?1:.4}}/>
            {it.label}
            {active===it.key&&<span style={{marginLeft:'auto',width:5,height:5,borderRadius:'50%',background:it.color,animation:'pulse 2s infinite'}}/>}
          </a>
        ))}
      </nav>
      <div className="sidebar-footer">
        {[
          {k:'Cell Towers',   v:'500'},
          {k:'Customers',     v:'500'},
          {k:'Readings/day',  v:'9.56M'},
        ].map(r=>(
          <div key={r.k} className="sidebar-stat"><span>{r.k}</span><span>{r.v}</span></div>
        ))}
      </div>
    </aside>
  )
}

// ── STAT CARD ─────────────────────────────────────────────────────────────
export function StatCard({ icon, label, value, sub, color, spark, trend }: {
  icon:string, label:string, value:string|number, sub?:string,
  color?:string, spark?:number[], trend?:string
}) {
  return (
    <div className="stat-card">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start'}}>
        <div>
          <div className="stat-label">{label}</div>
          <div className="stat-value" style={{color:color||'var(--text)'}}>{value}</div>
          {sub&&<div className="stat-sub">{sub}</div>}
        </div>
        <div style={{display:'flex',flexDirection:'column',alignItems:'flex-end',gap:4}}>
          <span style={{fontSize:20}}>{icon}</span>
          {trend&&<span style={{fontSize:9,fontWeight:700,color:trend.startsWith('+')?'var(--green)':'var(--red)'}}>{trend}</span>}
        </div>
      </div>
      {spark&&<div style={{marginTop:8}}><Sparkline data={spark} color={color||'#4f8ef7'} w={100}/></div>}
    </div>
  )
}

// ── SECTION DIVIDER ───────────────────────────────────────────────────────
export function Div({ label, right }: { label:string, right?:string }) {
  return (
    <div className="divider">
      <span className="divider-label">{label}</span>
      <span className="divider-line"/>
      {right&&<span className="divider-right">{right}</span>}
    </div>
  )
}

// ── LIVE BADGE ────────────────────────────────────────────────────────────
export function LiveBadge({ ok }: { ok:boolean }) {
  return (
    <span style={{display:'inline-flex',alignItems:'center',gap:5,fontSize:10,
      padding:'3px 9px',borderRadius:20,
      background:ok?'rgba(0,214,143,0.1)':'rgba(255,71,87,0.1)',
      color:ok?'var(--green)':'var(--red)'}}>
      <span style={{width:5,height:5,borderRadius:'50%',background:'currentColor',animation:'pulse 2s infinite'}}/>
      {ok?'LIVE':'OFFLINE'}
    </span>
  )
}

// ── PROGRESS BAR ──────────────────────────────────────────────────────────
export function ProgressBar({ value, color='#4f8ef7', h=4 }: { value:number, color?:string, h?:number }) {
  return (
    <div style={{height:h,background:'rgba(255,255,255,0.06)',borderRadius:h,overflow:'hidden'}}>
      <div style={{height:'100%',borderRadius:h,background:color,width:`${value}%`,transition:'width .7s ease'}}/>
    </div>
  )
}

// ── FLIP CARD ─────────────────────────────────────────────────────────────
export function FlipCard({ front, back, height=150, borderColor='var(--border)', bg='var(--s2)' }: {
  front:React.ReactNode, back:React.ReactNode, height?:number,
  borderColor?:string, bg?:string
}) {
  const [flipped, setFlipped] = useState(false)
  return (
    <div className="flip-wrap" style={{height}}>
      <div className={`flip-inner${flipped?' flipped':''}`} style={{height}}>
        <div className="flip-front" style={{background:bg,border:`1px solid ${borderColor}`,overflow:'hidden'}}>
          {front}
          <button onClick={()=>setFlipped(true)} style={{position:'absolute',bottom:8,right:10,
            background:'rgba(255,255,255,0.06)',border:'1px solid rgba(255,255,255,0.08)',
            borderRadius:7,padding:'3px 9px',cursor:'pointer',color:'rgba(255,255,255,0.35)',fontSize:9,fontFamily:'Inter,sans-serif'}}>
            details ↻
          </button>
        </div>
        <div className="flip-back" style={{background:bg,border:`1px solid ${borderColor}`,overflow:'hidden'}}>
          {back}
          <button onClick={()=>setFlipped(false)} style={{position:'absolute',bottom:8,right:10,
            background:'rgba(255,255,255,0.06)',border:'1px solid rgba(255,255,255,0.08)',
            borderRadius:7,padding:'3px 9px',cursor:'pointer',color:'rgba(255,255,255,0.35)',fontSize:9,fontFamily:'Inter,sans-serif'}}>
            ↻ back
          </button>
        </div>
      </div>
    </div>
  )
}
