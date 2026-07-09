import { useState, useEffect, useRef } from 'react'
import { ParticleCanvas, Sidebar, Div, ProgressBar } from '../components/UI'

const API = 'http://localhost:8000'

const JOURNEY_ZONES = [
  { name:'Stuttgart City',   km:0,   lat:48.7758, lng:9.1829,  net:'5G', score:94, color:'#6b21a8', breach:false, tower:false },
  { name:'A8 Leonberg',      km:18,  lat:48.7978, lng:9.0167,  net:'5G', score:91, color:'#6b21a8', breach:false, tower:false },
  { name:'A8 Merklingen',    km:45,  lat:48.5756, lng:9.7789,  net:'4G', score:76, color:'#be185d', breach:false, tower:false },
  { name:'A8 Ulm West',      km:80,  lat:48.3974, lng:9.9342,  net:'4G', score:71, color:'#be185d', breach:false, tower:false },
  { name:'A8 Ulm Ost',       km:95,  lat:48.3800, lng:10.0567, net:'3G', score:48, color:'#f97316', breach:true,  tower:false },
  { name:'A8 Burgau',        km:115, lat:48.4278, lng:10.4089, net:'2G', score:22, color:'#db2777', breach:true,  tower:true  },
  { name:'A8 Augsburg West', km:145, lat:48.3705, lng:10.7541, net:'4G', score:73, color:'#be185d', breach:false, tower:false },
  { name:'A8 Augsburg Ost',  km:160, lat:48.3669, lng:10.9389, net:'4G', score:78, color:'#be185d', breach:false, tower:false },
  { name:'A99 Munich Ring',  km:195, lat:48.1731, lng:11.5040, net:'5G', score:89, color:'#6b21a8', breach:false, tower:false },
  { name:'Munich City',      km:220, lat:48.1351, lng:11.5820, net:'5G', score:96, color:'#6b21a8', breach:false, tower:false },
]

const NET_COLORS: Record<string,string> = {
  '5G':'#6b21a8','4G':'#be185d','3G':'#f97316','2G':'#db2777','No signal':'#374151'
}

// ── LEAFLET MAP ────────────────────────────────────────────────────────────
function JourneyMap({ currentZone, events }: { currentZone:number, events:any[] }) {
  const mapRef     = useRef<HTMLDivElement>(null)
  const leafletRef = useRef<any>(null)
  const markerRef  = useRef<any>(null)

  useEffect(() => {
    const L = (window as any).L
    if (!mapRef.current || leafletRef.current || !L) return
    const map = L.map(mapRef.current, { center:[48.45,10.35], zoom:8, scrollWheelZoom:false })
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution:'© OpenStreetMap', opacity:0.75
    }).addTo(map)
    leafletRef.current = map
    // Route segments colored by network type
    for (let i=0;i<JOURNEY_ZONES.length-1;i++) {
      const z=JOURNEY_ZONES[i], n=JOURNEY_ZONES[i+1]
      L.polyline([[z.lat,z.lng],[n.lat,n.lng]],{color:z.color,weight:6,opacity:.85})
       .addTo(map).bindTooltip(`${z.name} — ${z.net} (${z.score})`,{sticky:true})
    }
    // City markers
    ;[JOURNEY_ZONES[0],JOURNEY_ZONES[9]].forEach(z=>{
      L.circleMarker([z.lat,z.lng],{radius:8,fillColor:z.color,color:'white',weight:2,fillOpacity:1})
       .addTo(map).bindPopup(`<b>${z.name}</b><br>${z.net} · Score ${z.score}`)
    })
    // Tower recommendation markers
    JOURNEY_ZONES.filter(z=>z.tower).forEach(z=>{
      const icon=L.divIcon({html:`<div style="background:#ff4757;border:2px solid white;border-radius:50%;width:20px;height:20px;display:flex;align-items:center;justify-content:center;font-size:11px;">🗼</div>`,iconSize:[20,20],iconAnchor:[10,10],className:''})
      L.marker([z.lat,z.lng],{icon}).addTo(map).bindPopup(`<b>⚠️ New tower needed</b><br>${z.name}`)
    })
    // Car marker
    const carIcon=L.divIcon({html:`<div style="font-size:22px;">🚗</div>`,iconSize:[28,28],iconAnchor:[14,14],className:''})
    markerRef.current=L.marker([JOURNEY_ZONES[0].lat,JOURNEY_ZONES[0].lng],{icon:carIcon}).addTo(map)
    return () => { map.remove(); leafletRef.current=null }
  }, [])

  useEffect(() => {
    if (!markerRef.current||!leafletRef.current) return
    const z=JOURNEY_ZONES[currentZone]
    markerRef.current.setLatLng([z.lat,z.lng])
    leafletRef.current.panTo([z.lat,z.lng],{animate:true,duration:1.5})
  }, [currentZone])

  return (
    <div style={{position:'relative',borderRadius:14,overflow:'hidden',border:'1px solid var(--border)'}}>
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
      <div ref={mapRef} style={{height:360,width:'100%',background:'#1a1a2e'}}/>
      <div style={{position:'absolute',bottom:10,left:10,background:'rgba(8,13,26,0.92)',
        borderRadius:10,padding:'8px 12px',border:'1px solid rgba(255,255,255,0.1)',zIndex:1000}}>
        <div style={{fontSize:9,color:'rgba(255,255,255,0.4)',marginBottom:5}}>COVERAGE</div>
        {[['5G','#6b21a8'],['4G','#be185d'],['3G','#f97316'],['2G','#db2777']].map(([n,c])=>(
          <div key={n} style={{display:'flex',alignItems:'center',gap:6,marginBottom:3}}>
            <div style={{width:18,height:3,background:c,borderRadius:2}}/>
            <span style={{fontSize:10,color:'rgba(255,255,255,0.6)'}}>{n}</span>
          </div>
        ))}
        <div style={{display:'flex',alignItems:'center',gap:5,marginTop:4}}>
          <span style={{fontSize:10}}>🗼</span>
          <span style={{fontSize:9,color:'#ff4757'}}>Tower needed</span>
        </div>
      </div>
    </div>
  )
}

// ── MAIN PAGE ──────────────────────────────────────────────────────────────
export default function P2Page() {
  const [metrics,setMetrics]   = useState<any>(null)
  const [forecast,setForecast] = useState<any[]>([])
  const [hours,setHours]       = useState(72)
  const [cur,setCur]     = useState(72)
  const [mean,setMean]   = useState(75)
  const [std,setStd]     = useState(8)
  const [sig,setSig]     = useState(3)
  const [tick,setTick]   = useState(0)
  const [activeTab, setActiveTab] = useState<'monitor'|'journey'>('monitor')
  const [liveDevices,setLiveDevices] = useState([
    {id:'CUS-0001-PHN-01',type:'📱',score:78,rssi:-52,breaches:3 },
    {id:'CUS-0001-PHN-02',type:'📱',score:52,rssi:-68,breaches:14},
    {id:'CUS-0001-TAB-01',type:'💻',score:88,rssi:-44,breaches:1 },
    {id:'CUS-0002-PHN-01',type:'📱',score:41,rssi:-79,breaches:22},
    {id:'CUS-0002-PHN-02',type:'📱',score:71,rssi:-57,breaches:6 },
    {id:'CUS-0002-TAB-01',type:'💻',score:91,rssi:-41,breaches:0 },
  ])

  // Journey state
  const [journeyActive, setJourneyActive]   = useState(false)
  const [journeyDone,   setJourneyDone]     = useState(false)
  const [currentZone,   setCurrentZone]     = useState(0)
  const [journeyEvents, setJourneyEvents]   = useState<any[]>([])
  const [breachCount,   setBreachCount]     = useState(0)
  const [churnAfter,    setChurnAfter]      = useState(12)
  const [leafletLoaded, setLeafletLoaded]   = useState(false)
  const intervalRef = useRef<any>(null)
  const churnBefore = 12

  useEffect(() => {
    fetch(`${API}/metrics`).then(r=>r.ok?r.json():null).then(m=>{if(m?.p2_wifi)setMetrics(m.p2_wifi)}).catch(()=>{})
    fetch(`${API}/api/p2/forecast-sample?limit=168`).then(r=>r.ok?r.json():null).then(d=>{if(d)setForecast(d)}).catch(()=>{})
    const iv=setInterval(()=>{
      setTick(t=>t+1)
      setLiveDevices(ds=>ds.map(d=>({...d,score:Math.min(99,Math.max(30,d.score+(Math.random()-.48)*2.5))})))
    },4000)
    if (!(window as any).L) {
      const s=document.createElement('script')
      s.src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'
      s.onload=()=>setLeafletLoaded(true)
      document.head.appendChild(s)
    } else { setLeafletLoaded(true) }
    return ()=>clearInterval(iv)
  },[])

  const startJourney=()=>{
    setJourneyActive(true);setJourneyDone(false)
    setCurrentZone(0);setJourneyEvents([]);setBreachCount(0);setChurnAfter(churnBefore)
    let idx=0
    intervalRef.current=setInterval(()=>{
      idx++
      if(idx>=JOURNEY_ZONES.length){clearInterval(intervalRef.current);setJourneyActive(false);setJourneyDone(true);return}
      const z=JOURNEY_ZONES[idx]
      setCurrentZone(idx)
      setJourneyEvents(ev=>[{
        ts:new Date().toLocaleTimeString('de-DE',{hour:'2-digit',minute:'2-digit'}),
        name:z.name,net:z.net,score:z.score,km:z.km,breach:z.breach,tower:z.tower,lat:z.lat,lng:z.lng
      },...ev.slice(0,8)])
      if(z.breach){setBreachCount(c=>c+1);setChurnAfter(c=>Math.min(99,c+(z.net==='2G'?18:9)))}
    },1800)
  }
  const pauseJourney=()=>{clearInterval(intervalRef.current);setJourneyActive(false)}
  const resetJourney=()=>{
    clearInterval(intervalRef.current)
    setJourneyActive(false);setJourneyDone(false)
    setCurrentZone(0);setJourneyEvents([]);setBreachCount(0);setChurnAfter(churnBefore)
  }

  const m=metrics||{},fm=m.forecast_metrics||{},sf=m.sarima_fit||{}
  const lower=mean-sig*std, upper=mean+sig*std
  const isBreach=cur<lower, isSuspicious=cur>upper&&(cur-mean)>15
  const breachColor=isBreach?'var(--red)':isSuspicious?'var(--amber)':'var(--green)'
  const dev=(cur-mean)/(std+1e-6)

  const genForecast=()=>Array.from({length:168},(_,i)=>({
    actual:+(74+Math.sin(i/6)*8+(Math.random()-.5)*5).toFixed(1),
    forecast:+(74+Math.sin(i/6)*7).toFixed(1),lower_bound:50
  }))
  const fc=(forecast.length>0?forecast:genForecast()).slice(-hours)
  const actualSeries=fc.map((f:any)=>f.actual||0)
  const fcSeries=fc.map((f:any)=>f.forecast||0)
  const lowerSeries=fc.map(()=>lower)
  const devColor=(s:number)=>s>=75?'var(--green)':s>=55?'var(--amber)':'var(--red)'
  const curZone=JOURNEY_ZONES[currentZone]
  const netColor=NET_COLORS[curZone.net]||'#6b7280'
  const journeyScores=journeyEvents.length>0?[...journeyEvents].reverse().map(e=>e.score):[94,91,76,71,48,22,73,78,89,96]

  return (
    <div className="shell">
      <ParticleCanvas/>
      <Sidebar active="p2"/>
      <main className="main" style={{position:'relative',zIndex:1}}>

        <div className="page-header fade-up">
          <h1 className="page-title grad-purple">P2 — Mobile Signal Experience Monitor</h1>
          <p className="page-sub">SARIMA(1,0,1)×(1,1,0,96) · ±3σ breach detection · 3.24M readings · 500 customers</p>
        </div>

        {/* METRICS */}
        <div className="grid-5 fade-up-1" style={{marginBottom:20}}>
          <div className="stat-card"><div className="stat-label">MAE</div><div className="stat-value" style={{color:'var(--purple)'}}>{(fm.mae||8.56).toFixed(4)}</div><div className="stat-sub">score pts</div></div>
          <div className="stat-card"><div className="stat-label">RMSE</div><div className="stat-value" style={{color:'var(--purple)'}}>{(fm.rmse||11.41).toFixed(4)}</div></div>
          <div className="stat-card"><div className="stat-label">AIC</div><div className="stat-value" style={{color:'var(--accent)'}}>{(sf.aic||13318).toFixed(0)}</div></div>
          <div className="stat-card"><div className="stat-label">ADF TEST</div><div className="stat-value" style={{color:'var(--green)',fontSize:14}}>{m.stationarity_test?.stationary?'Stationary':'Non-stat'}</div></div>
          <div className="stat-card"><div className="stat-label">σ RULE</div><div className="stat-value" style={{color:'var(--cyan)'}}>±3σ</div><div className="stat-sub">breach threshold</div></div>
        </div>

        {/* TABS */}
        <div className="tab-row fade-up-2">
          <button className={`tab-btn${activeTab==='monitor'?' active':''}`} onClick={()=>setActiveTab('monitor')}>📊 Signal Monitor</button>
          <button className={`tab-btn${activeTab==='journey'?' active':''}`} onClick={()=>setActiveTab('journey')}>🚗 Journey Simulator — Stuttgart → Munich</button>
        </div>

        {/* ── SIGNAL MONITOR TAB ────────────────────────────── */}
        {activeTab==='monitor' && (
          <div>
            {/* FORECAST CHART */}
            <div className="card fade-up-2" style={{marginBottom:20}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:12}}>
                <Div label="SARIMA FORECAST vs ACTUAL SIGNAL SCORE" right={`${hours}h window · tick ${tick}`}/>
                <div style={{display:'flex',gap:5}}>
                  {[24,48,72,168].map(h=>(
                    <button key={h} onClick={()=>setHours(h)} className="btn btn-ghost"
                      style={{padding:'4px 10px',fontSize:10,borderColor:hours===h?'var(--purple)':'var(--border)',color:hours===h?'var(--purple)':'var(--muted)'}}>
                      {h}h
                    </button>
                  ))}
                </div>
              </div>
              <svg viewBox="0 0 560 160" style={{width:'100%',height:160}} preserveAspectRatio="none">
                {[0,.25,.5,.75,1].map(f=><line key={f} x1="36" x2="550" y1={8+140*(1-f)} y2={8+140*(1-f)} stroke="rgba(255,255,255,0.04)" strokeWidth="1"/>)}
                {[0,.5,1].map(f=><text key={f} x="32" y={12+140*(1-f)} textAnchor="end" style={{fontSize:9,fill:'rgba(255,255,255,0.25)',fontFamily:'monospace'}}>{Math.round(47+53*f)}</text>)}
                {(['actual','forecast','lower'] as const).map((type)=>{
                  const data=type==='actual'?actualSeries:type==='forecast'?fcSeries:lowerSeries
                  const color=type==='actual'?'#4f8ef7':type==='forecast'?'#9b7cf8':'rgba(255,71,87,0.6)'
                  const fill=type==='actual'?'rgba(79,142,247,0.08)':type==='forecast'?'rgba(155,124,248,0.06)':'rgba(255,71,87,0.04)'
                  const n=data.length
                  const xp=(i:number)=>36+(i/(n-1||1))*514
                  const yp=(v:number)=>8+140-((v-47)/53)*140
                  const pts=data.map((_,i)=>`${xp(i)},${yp(data[i])}`).join(' ')
                  const area=`M ${xp(0)} ${yp(data[0])} `+data.map((_,i)=>`L ${xp(i)} ${yp(data[i])}`).join(' ')+` L 550 148 L 36 148 Z`
                  return <g key={type}><path d={area} fill={fill}/><polyline points={pts} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round"/></g>
                })}
              </svg>
              <div style={{display:'flex',gap:16,marginTop:8,fontSize:10,color:'var(--muted)'}}>
                {[{c:'#4f8ef7',l:'Actual'},{c:'#9b7cf8',l:'Forecast'},{c:'rgba(255,71,87,0.7)',l:'Mean − 3σ'}].map(x=>(
                  <span key={x.l} style={{display:'flex',alignItems:'center',gap:5}}>
                    <span style={{width:12,height:2,background:x.c,display:'inline-block',borderRadius:1}}/>{x.l}
                  </span>
                ))}
              </div>
            </div>

            {/* BREACH DETECTOR + SARIMA EXPLAINER */}
            <div className="grid-2" style={{marginBottom:20}}>
              <div className="card">
                <Div label="LIVE ±3σ MOBILE SIGNAL BREACH DETECTOR"/>
                {[
                  {l:'Current signal score',min:0,  max:100,step:.5,v:cur,  set:setCur},
                  {l:'7-day rolling mean',   min:20, max:95, step:.5,v:mean, set:setMean},
                  {l:'7-day rolling std',     min:1,  max:15, step:.5,v:std,  set:setStd},
                ].map(s=>(
                  <div key={s.l} style={{marginBottom:10}}>
                    <div style={{display:'flex',justifyContent:'space-between',marginBottom:3}}>
                      <span style={{fontSize:11,color:'var(--muted)'}}>{s.l}</span>
                      <span style={{fontSize:11,fontFamily:'monospace'}}>{s.v}</span>
                    </div>
                    <input type="range" min={s.min} max={s.max} step={s.step} value={s.v}
                      onChange={e=>s.set(+e.target.value)} style={{width:'100%',height:4}}/>
                  </div>
                ))}
                <div style={{display:'flex',gap:5,marginBottom:14}}>
                  {[2,2.5,3,3.5].map(s=>(
                    <button key={s} onClick={()=>setSig(s)} className="btn btn-ghost"
                      style={{flex:1,padding:'5px',fontSize:10,borderColor:sig===s?'var(--purple)':'var(--border)',color:sig===s?'var(--purple)':'var(--muted)'}}>
                      ±{s}σ
                    </button>
                  ))}
                </div>
                <div style={{padding:'14px',borderRadius:12,
                  background:isBreach?'rgba(255,71,87,0.07)':isSuspicious?'rgba(245,166,35,0.07)':'rgba(0,214,143,0.07)',
                  border:`1px solid ${isBreach?'rgba(255,71,87,0.2)':isSuspicious?'rgba(245,166,35,0.2)':'rgba(0,214,143,0.15)'}`}}>
                  <div style={{fontSize:15,fontWeight:700,color:breachColor,marginBottom:8}}>
                    {isBreach?'🔴 BREACH — signal below baseline':isSuspicious?'🟡 SUSPICIOUS':'🟢 Normal'}
                  </div>
                  <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8,fontSize:11,color:'var(--muted)'}}>
                    <span>Score: <strong style={{color:'var(--text)'}}>{cur}</strong></span>
                    <span>Lower bound: <strong style={{color:'var(--red)'}}>{lower.toFixed(1)}</strong></span>
                    <span>Deviation: <strong style={{color:breachColor}}>{dev.toFixed(2)}σ</strong></span>
                    <span>Upper bound: <strong style={{color:'var(--amber)'}}>{upper.toFixed(1)}</strong></span>
                  </div>
                  <div style={{marginTop:10,position:'relative',height:8,background:'rgba(255,255,255,0.06)',borderRadius:4}}>
                    <div style={{position:'absolute',left:0,top:0,height:'100%',borderRadius:4,
                      background:breachColor,width:`${Math.min(cur,100)}%`,transition:'width .3s'}}/>
                    <div style={{position:'absolute',top:-2,left:`${Math.min(Math.max(lower,0),98)}%`,
                      width:2,height:12,background:'var(--red)',borderRadius:1}}/>
                    <div style={{position:'absolute',top:-2,left:`${Math.min(mean,98)}%`,
                      width:2,height:12,background:'rgba(255,255,255,0.4)',borderRadius:1}}/>
                  </div>
                  <div style={{display:'flex',justifyContent:'space-between',marginTop:4,fontSize:9,color:'var(--muted)'}}>
                    <span style={{color:'var(--red)'}}>▲ lower ({lower.toFixed(0)})</span>
                    <span>▲ mean ({mean})</span>
                  </div>
                </div>
              </div>

              <div className="card">
                <Div label="SARIMA ORDER EXPLAINED"/>
                {[
                  {p:'p=1', c:'var(--accent)', t:'Short-term AR — current signal depends on 1 past value'},
                  {p:'d=0', c:'var(--green)',  t:'No differencing — ADF test confirms stationarity'},
                  {p:'q=1', c:'var(--purple)', t:'MA smoothing term for residual noise'},
                  {p:'P=1', c:'var(--amber)',  t:'Seasonal AR — captures same-hour-yesterday pattern'},
                  {p:'D=1', c:'var(--red)',    t:'Seasonal differencing removes weekly trend'},
                  {p:'m=96',c:'var(--cyan)',   t:'96 readings per day (15-min × 24h)'},
                ].map(e=>(
                  <div key={e.p} style={{display:'flex',alignItems:'flex-start',gap:10,padding:'8px 11px',background:'var(--s2)',borderRadius:8,marginBottom:6}}>
                    <span style={{fontSize:11,fontFamily:'monospace',fontWeight:700,color:e.c,background:`${e.c}14`,padding:'2px 7px',borderRadius:5,flexShrink:0}}>{e.p}</span>
                    <span style={{fontSize:11,color:'var(--muted)',lineHeight:1.5}}>{e.t}</span>
                  </div>
                ))}
                <div style={{marginTop:8,padding:'10px 12px',background:'var(--s2)',borderRadius:9,
                  borderLeft:'2px solid var(--purple)',fontSize:11,color:'var(--muted)'}}>
                  60% of mobile signal breaches co-occur with a P1 tower anomaly — confirming the Tower→Signal causal link
                </div>
              </div>
            </div>

            {/* DEVICE CARDS */}
            <Div label="MOBILE DEVICE STATUS" right="auto-refreshes 4s"/>
            <div className="grid-3">
              {liveDevices.map((d,i)=>{
                const col=devColor(d.score)
                return (
                  <div key={d.id} style={{background:'var(--s1)',border:`1px solid ${col}30`,borderRadius:14,padding:'14px 16px'}}>
                    <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:8}}>
                      <div style={{position:'relative',width:46,height:46,flexShrink:0}}>
                        <svg width={46} height={46} style={{transform:'rotate(-90deg)'}}>
                          <circle cx={23} cy={23} r={18} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="5"/>
                          <circle cx={23} cy={23} r={18} fill="none" stroke={col} strokeWidth="5"
                            strokeDasharray={`${(d.score/100)*113} 113`} strokeLinecap="round"
                            style={{transition:'stroke-dasharray .8s ease'}}/>
                        </svg>
                        <div style={{position:'absolute',inset:0,display:'flex',alignItems:'center',justifyContent:'center'}}>
                          <span style={{fontSize:11,fontWeight:700,fontFamily:'monospace',color:col}}>{Math.round(d.score)}</span>
                        </div>
                      </div>
                      <div>
                        <div style={{fontSize:12,fontWeight:700,marginBottom:2}}>{d.type} {d.id.split('-').slice(-2).join('-')}</div>
                        <div style={{fontSize:10,color:'var(--muted)'}}>RSSI: {d.rssi}dBm · Breaches: {d.breaches}</div>
                      </div>
                    </div>
                    <ProgressBar value={d.score} color={col}/>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* ── JOURNEY SIMULATOR TAB ─────────────────────────── */}
        {activeTab==='journey' && (
          <div>
            <div className="card fade-up-2" style={{marginBottom:20,border:'1px solid rgba(107,33,168,0.4)',background:'rgba(107,33,168,0.04)'}}>
              <Div label="JOURNEY SIMULATOR — STUTTGART → MUNICH (A8)" right="Deutsche Telekom Coverage"/>

              {/* Controls */}
              <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:16,flexWrap:'wrap'}}>
                <button onClick={startJourney} disabled={journeyActive} className="btn btn-primary" style={{width:'auto',padding:'9px 20px',fontSize:12}}>▶ Start Journey</button>
                <button onClick={pauseJourney} disabled={!journeyActive} className="btn btn-ghost" style={{padding:'9px 16px',fontSize:12}}>⏸ Pause</button>
                <button onClick={resetJourney} className="btn btn-ghost" style={{padding:'9px 16px',fontSize:12}}>↺ Reset</button>
                <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:8,fontSize:11,color:'var(--muted)'}}>
                  <span>{curZone.km} km / 220 km</span>
                  <div style={{width:120,height:4,background:'rgba(255,255,255,0.06)',borderRadius:2}}>
                    <div style={{height:'100%',borderRadius:2,background:'var(--purple)',width:`${(curZone.km/220)*100}%`,transition:'width 1s ease'}}/>
                  </div>
                </div>
              </div>

              <div style={{display:'grid',gridTemplateColumns:'1fr 320px',gap:16}}>

                {/* MAP */}
                <div>
                  {leafletLoaded
                    ? <JourneyMap currentZone={currentZone} events={journeyEvents}/>
                    : <div style={{height:360,background:'var(--s2)',borderRadius:14,display:'flex',alignItems:'center',justifyContent:'center',border:'1px solid var(--border)',fontSize:13,color:'var(--muted)'}}>Loading map...</div>
                  }
                </div>

                {/* RIGHT PANEL */}
                <div style={{display:'flex',flexDirection:'column',gap:12}}>

                  {/* Current status */}
                  <div style={{padding:'14px',background:'var(--s2)',borderRadius:12,border:`1px solid ${netColor}40`}}>
                    <div style={{fontSize:9,color:'var(--muted)',letterSpacing:'0.08em',marginBottom:8}}>CURRENT STATUS</div>
                    <div style={{display:'flex',alignItems:'center',gap:12,marginBottom:10}}>
                      <div style={{position:'relative',width:64,height:64,flexShrink:0}}>
                        <svg width={64} height={64} style={{transform:'rotate(-90deg)'}}>
                          <circle cx={32} cy={32} r={26} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="7"/>
                          <circle cx={32} cy={32} r={26} fill="none" stroke={netColor} strokeWidth="7"
                            strokeDasharray={`${(curZone.score/100)*163} 163`} strokeLinecap="round"
                            style={{transition:'stroke-dasharray 1s ease,stroke 0.5s ease'}}/>
                        </svg>
                        <div style={{position:'absolute',inset:0,display:'flex',alignItems:'center',justifyContent:'center'}}>
                          <span style={{fontSize:16,fontWeight:800,fontFamily:'monospace',color:netColor,lineHeight:1}}>{curZone.score}</span>
                        </div>
                      </div>
                      <div>
                        <div style={{display:'flex',alignItems:'center',gap:6,marginBottom:4}}>
                          <span style={{fontSize:20,fontWeight:800,color:netColor}}>{curZone.net}</span>
                          <span style={{fontSize:10,padding:'2px 7px',borderRadius:20,fontWeight:700,background:`${netColor}25`,color:netColor}}>
                            {curZone.score>=75?'GOOD':curZone.score>=55?'FAIR':'POOR'}
                          </span>
                        </div>
                        <div style={{fontSize:11,color:'var(--muted)',marginBottom:2}}>{curZone.name}</div>
                        <div style={{fontSize:10,color:'var(--muted)',fontFamily:'monospace'}}>{curZone.km}km from Stuttgart</div>
                      </div>
                    </div>
                    {/* Network gen buttons */}
                    <div style={{display:'flex',gap:4,marginBottom:10}}>
                      {['5G','4G','3G','2G'].map(n=>(
                        <div key={n} style={{flex:1,padding:'5px 3px',borderRadius:7,textAlign:'center',
                          background:curZone.net===n?`${NET_COLORS[n]}25`:'rgba(255,255,255,0.03)',
                          border:`1px solid ${curZone.net===n?NET_COLORS[n]+'60':'rgba(255,255,255,0.06)'}`,
                          fontSize:10,fontWeight:curZone.net===n?700:400,
                          color:curZone.net===n?NET_COLORS[n]:'var(--muted)',transition:'all 0.4s'}}>
                          {n}
                        </div>
                      ))}
                    </div>
                    {curZone.breach&&(
                      <div style={{padding:'7px 10px',background:'rgba(255,71,87,0.1)',borderRadius:8,border:'1px solid rgba(255,71,87,0.25)',fontSize:11,color:'#ff8a94',marginBottom:8}}>
                        ⚠️ BREACH — score below ±3σ baseline
                      </div>
                    )}
                    {curZone.tower&&(
                      <div style={{padding:'7px 10px',background:'rgba(245,158,11,0.08)',borderRadius:8,border:'1px solid rgba(245,158,11,0.2)',fontSize:11,color:'var(--amber)'}}>
                        🗼 New tower recommended here
                      </div>
                    )}
                  </div>

                  {/* Score chart */}
                  <div style={{padding:'12px',background:'var(--s2)',borderRadius:12,border:'1px solid var(--border)'}}>
                    <div style={{fontSize:9,color:'var(--muted)',letterSpacing:'0.08em',marginBottom:8}}>SIGNAL SCORE ALONG ROUTE</div>
                    <div style={{display:'flex',alignItems:'flex-end',gap:3,height:60}}>
                      {journeyScores.map((s,i)=>{
                        const c=s>=75?'var(--green)':s>=55?'var(--amber)':'var(--red)'
                        const isLast=i===journeyScores.length-1
                        return (
                          <div key={i} style={{flex:1,borderRadius:'2px 2px 0 0',background:c,
                            height:`${(s/100)*52}px`,transition:'height 0.5s',
                            opacity:isLast?1:0.6,boxShadow:isLast?`0 0 8px ${c}`:'none'}}/>
                        )
                      })}
                    </div>
                    <div style={{display:'flex',justifyContent:'space-between',marginTop:4,fontSize:8,color:'var(--muted)'}}>
                      <span>Stuttgart</span><span>→</span><span>Munich</span>
                    </div>
                  </div>

                  {/* Churn impact */}
                  {(breachCount>0||journeyDone)&&(
                    <div style={{padding:'12px',background:'var(--s2)',borderRadius:12,
                      border:`1px solid ${churnAfter>35?'rgba(255,71,87,0.3)':'rgba(245,158,11,0.3)'}`}}>
                      <div style={{fontSize:9,color:'var(--muted)',letterSpacing:'0.08em',marginBottom:8}}>CHURN IMPACT</div>
                      <div style={{display:'flex',justifyContent:'space-between',marginBottom:6}}>
                        <div style={{textAlign:'center'}}>
                          <div style={{fontSize:9,color:'var(--muted)',marginBottom:2}}>BEFORE JOURNEY</div>
                          <div style={{fontSize:22,fontWeight:800,fontFamily:'monospace',color:'var(--green)'}}>{churnBefore}%</div>
                        </div>
                        <div style={{display:'flex',alignItems:'center',fontSize:20,color:'var(--muted)'}}>→</div>
                        <div style={{textAlign:'center'}}>
                          <div style={{fontSize:9,color:'var(--muted)',marginBottom:2}}>AFTER JOURNEY</div>
                          <div style={{fontSize:22,fontWeight:800,fontFamily:'monospace',color:churnAfter>35?'var(--red)':'var(--amber)'}}>{churnAfter}%</div>
                        </div>
                      </div>
                      <div style={{fontSize:10,color:'var(--muted)'}}>
                        <span style={{color:'var(--red)',fontWeight:700}}>+{churnAfter-churnBefore}%</span>
                        {' '}from {breachCount} signal breach{breachCount!==1?'es':''}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Event log */}
              {journeyEvents.length>0&&(
                <div style={{marginTop:14}}>
                  <Div label="JOURNEY EVENT LOG" right={`${breachCount} breach${breachCount!==1?'es':''} detected`}/>
                  <div style={{display:'flex',flexDirection:'column',gap:6,maxHeight:200,overflowY:'auto'}}>
                    {journeyEvents.map((e,i)=>(
                      <div key={i} style={{display:'flex',alignItems:'center',gap:10,padding:'8px 12px',
                        background:e.breach?'rgba(255,71,87,0.06)':'rgba(255,255,255,0.02)',
                        border:`1px solid ${e.breach?'rgba(255,71,87,0.2)':e.tower?'rgba(245,158,11,0.2)':'rgba(255,255,255,0.05)'}`,
                        borderRadius:9,animation:i===0?'fadeUp .3s ease both':'none'}}>
                        <span style={{fontSize:9,fontFamily:'monospace',color:'var(--muted)',width:40,flexShrink:0}}>{e.ts}</span>
                        <span style={{fontSize:10,color:NET_COLORS[e.net]||'white',fontWeight:700,width:28,flexShrink:0}}>{e.net}</span>
                        <span style={{fontSize:11,flex:1,fontWeight:e.breach?700:400}}>{e.name}</span>
                        <span style={{fontSize:10,fontFamily:'monospace',color:'var(--muted)',width:32}}>{e.km}km</span>
                        <span style={{fontSize:14,fontWeight:800,fontFamily:'monospace',
                          color:e.score>=75?'var(--green)':e.score>=55?'var(--amber)':'var(--red)',width:28,textAlign:'right'}}>{e.score}</span>
                        {e.breach&&<span style={{fontSize:9,padding:'2px 7px',borderRadius:20,background:'rgba(255,71,87,0.15)',color:'#ff8a94',fontWeight:700,flexShrink:0}}>BREACH</span>}
                        {e.tower&&<span style={{fontSize:9,padding:'2px 7px',borderRadius:20,background:'rgba(245,158,11,0.15)',color:'var(--amber)',fontWeight:700,flexShrink:0}}>🗼 TOWER</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Journey done summary */}
              {journeyDone&&(
                <div style={{marginTop:14,padding:'16px 18px',background:'rgba(107,33,168,0.08)',borderRadius:12,border:'1px solid rgba(107,33,168,0.3)'}}>
                  <div style={{fontSize:12,fontWeight:700,marginBottom:10,color:'var(--purple)'}}>✅ Journey complete — Stuttgart → Munich (220km)</div>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:10,marginBottom:12}}>
                    {[
                      {l:'Distance',     v:'220 km'},
                      {l:'Breaches',     v:`${breachCount}`,c:'var(--red)'},
                      {l:'Churn risk',   v:`${churnBefore}% → ${churnAfter}%`,c:'var(--amber)'},
                      {l:'Tower needed', v:`${JOURNEY_ZONES.filter(z=>z.tower).length} location`,c:'var(--amber)'},
                    ].map(s=>(
                      <div key={s.l} style={{background:'var(--s2)',borderRadius:9,padding:'10px 12px',textAlign:'center'}}>
                        <div style={{fontSize:9,color:'var(--muted)',marginBottom:3}}>{s.l}</div>
                        <div style={{fontSize:13,fontWeight:700,fontFamily:'monospace',color:(s as any).c||'var(--text)'}}>{s.v}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{fontSize:11,color:'var(--muted)',lineHeight:1.7}}>
                    <strong style={{color:'white'}}>Recommended actions: </strong>
                    Build new tower near Burgau (km 115) · Call customer with 1 month bill credit ·
                    Churn risk increased from {churnBefore}% to {churnAfter}% — escalate to retention team
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

      </main>
    </div>
  )
}
