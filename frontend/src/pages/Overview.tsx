import { useState, useEffect, useCallback } from 'react'
import { ParticleCanvas, Sidebar, StatCard, Div, AreaChart, Radar, LiveBadge, ProgressBar } from '../components/UI'

const API = 'http://localhost:8000'

export default function Overview() {
  const [metrics, setMetrics] = useState<any>(null)
  const [health,  setHealth]  = useState<any>(null)
  const [tick,    setTick]    = useState(0)

  // Simulated real-time data that updates every 3s
  const [liveStats, setLiveStats] = useState({
    modemsFlagged: 47, wifiBreaches: 12, highRiskCustomers: 31, dataProcessed: 9.56,
  })
  const [anomalyFeed, setAnomalyFeed] = useState([
    { ts:'12:04:21', modem:'TWR-0312', type:'5G→4G Drop',    sev:'HIGH', snr:22.4 },
    { ts:'12:04:09', modem:'TWR-0087', type:'Tower Overload', sev:'MED',  snr:35.1 },
    { ts:'12:03:55', modem:'TWR-0445', type:'4G→3G Drop',    sev:'HIGH', snr:31.7 },
    { ts:'12:03:41', modem:'TWR-0201', type:'Signal Loss',    sev:'LOW',  snr:33.8 },
    { ts:'12:03:28', modem:'TWR-0158', type:'Handover Fail',  sev:'MED',  snr:29.2 },
  ])

  useEffect(() => {
    fetch(`${API}/metrics`).then(r=>r.ok?r.json():null).then(m=>{ if(m) setMetrics(m) }).catch(()=>{})
    fetch(`${API}/health`).then(r=>r.ok?r.json():null).then(h=>{ if(h) setHealth(h) }).catch(()=>{})
  }, [])

  // Live simulation — new anomaly every 4s
  useEffect(() => {
    const iv = setInterval(() => {
      setTick(t=>t+1)
      setLiveStats(s=>({
        modemsFlagged:     Math.max(30, s.modemsFlagged     + Math.round((Math.random()-.45)*3)),
        wifiBreaches:      Math.max(5,  s.wifiBreaches      + Math.round((Math.random()-.5)*2)),
        highRiskCustomers: Math.max(20, s.highRiskCustomers + Math.round((Math.random()-.5)*2)),
        dataProcessed: +(s.dataProcessed + 0.02).toFixed(2),
      }))
      // Inject new anomaly event
      const modems=['TWR-0012','TWR-0078','TWR-0134','TWR-0289','TWR-0367','TWR-0411','TWR-0489']
      const types =['5G→4G Drop','4G→3G Drop','Signal Loss','Tower Overload','Handover Fail']
      const sevs  =['HIGH','MED','LOW']
      const now = new Date()
      const ts = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`
      const sev = sevs[Math.floor(Math.random()*3)]
      const newEvent = {
        ts, modem: modems[Math.floor(Math.random()*modems.length)],
        type: types[Math.floor(Math.random()*types.length)], sev,
        snr: +(20+Math.random()*20).toFixed(1),
      }
      setAnomalyFeed(f=>[newEvent,...f.slice(0,7)])
    }, 4000)
    return ()=>clearInterval(iv)
  }, [])

  const p1=metrics?.p1_hfc_anomaly||{}, p2=metrics?.p2_wifi||{}, p3=metrics?.p3_churn||{}
  const p3e=p3.evaluation||p3

  // Trend lines
  const anomalyTrend =[11.2,11.8,12.1,11.6,11.9,12.3,11.8,12.0,11.7,12.1,11.8,+(liveStats.modemsFlagged/500*100).toFixed(1)]
  const breachTrend  =[3.6,3.8,4.0,3.9,4.1,3.9,4.0,3.9,4.1,3.8,+(liveStats.wifiBreaches/360*100).toFixed(1),3.9]
  const churnTrend   =[25,26,25,27,26,26,27,25,26,27,26,+(liveStats.highRiskCustomers/500*100+20).toFixed(1)]

  const sevColor=(s:string)=>s==='HIGH'?'var(--red)':s==='MED'?'var(--amber)':'var(--green)'

  return (
    <div className="shell">
      <ParticleCanvas/>
      <Sidebar active="overview"/>
      <main className="main" style={{position:'relative',zIndex:1}}>

        {/* HEADER */}
        <div className="page-header fade-up">
          <div style={{display:'flex',alignItems:'center',gap:12,marginBottom:6}}>
            <h1 className="page-title grad-default">Deutsche Telekom — Mobile Intelligence Platform</h1>
            <LiveBadge ok={health!=null}/>
          </div>
          <p className="page-sub">AI-powered mobile network monitoring · Cell tower anomaly → Signal degradation → Customer churn</p>
        </div>

        {/* CAUSAL CHAIN BANNER */}
        <div className="fade-up" style={{display:'flex',alignItems:'center',gap:8,
          padding:'12px 18px',background:'var(--s1)',border:'1px solid var(--border)',
          borderRadius:14,marginBottom:22,flexWrap:'wrap'}}>
          {[
            {dot:'#ff4757',label:'P1 — Tower Anomaly',      out:'tower_anomaly_count · signal_flag'},
            {dot:'#9b7cf8',label:'P2 — Mobile Experience',  out:'breach_count_30d · network_drops'},
            {dot:'#00d68f',label:'P3 — Churn Prediction',   out:'retention action'},
          ].map((p,i)=>(
            <div key={i} style={{display:'flex',alignItems:'center',gap:8}}>
              <div style={{padding:'5px 14px',borderRadius:20,fontSize:11,fontWeight:700,
                background:`${p.dot}14`,border:`1px solid ${p.dot}30`,color:p.dot}}>{p.label}</div>
              {i<2&&<>
                <span style={{fontSize:10,color:'var(--muted)',fontFamily:'monospace'}}>{p.out}</span>
                <span style={{color:'var(--muted)',fontSize:16}}>→</span>
              </>}
            </div>
          ))}
        </div>

        {/* LIVE KPI GRID */}
        <div className="grid-4 fade-up-1" style={{marginBottom:20}}>
          <StatCard icon="🗼" label="TOWERS FLAGGED" value={liveStats.modemsFlagged}
            sub={`of 500 · ${(liveStats.modemsFlagged/5).toFixed(1)}%`}
            color="var(--red)" spark={anomalyTrend.map(v=>v*5)} trend="+2.3%"/>
          <StatCard icon="📵" label="SIGNAL BREACHES" value={liveStats.wifiBreaches}
            sub="active now · ±3σ rule" color="var(--purple)"
            spark={breachTrend.map(v=>v*3)} trend="-0.5%"/>
          <StatCard icon="⚠️" label="HIGH RISK CUSTOMERS" value={liveStats.highRiskCustomers}
            sub={`${(p3.roc_auc||0.84).toFixed(3)} ROC-AUC`} color="var(--amber)"
            spark={churnTrend} trend="+1.2%"/>
          <StatCard icon="📡" label="DATA PROCESSED" value={`${liveStats.dataProcessed}M`}
            sub="rows · live pipeline" color="var(--cyan)"
            spark={[9.1,9.2,9.3,9.4,9.45,9.5,9.52,9.54,9.55,9.56,liveStats.dataProcessed]}/>
        </div>

        {/* MAIN CONTENT: CHART + LIVE FEED */}
        <div className="grid-2 fade-up-2" style={{marginBottom:20}}>

          {/* TREND CHARTS */}
          <div className="card">
            <Div label="PLATFORM HEALTH TRENDS" right={`tick ${tick}`}/>
            <div style={{marginBottom:14}}>
              <div style={{fontSize:9,color:'var(--muted)',marginBottom:6}}>HFC ANOMALY RATE (%)</div>
              <AreaChart series={[{label:'Anomaly',color:'#ff4757',fill:'rgba(255,71,87,0.08)',data:anomalyTrend}]} h={70}/>
            </div>
            <div style={{marginBottom:14}}>
              <div style={{fontSize:9,color:'var(--muted)',marginBottom:6}}>Wi-Fi BREACH RATE (%)</div>
              <AreaChart series={[{label:'Breach',color:'#9b7cf8',fill:'rgba(155,124,248,0.08)',data:breachTrend}]} h={70}/>
            </div>
            <div>
              <div style={{fontSize:9,color:'var(--muted)',marginBottom:6}}>HIGH RISK CHURN CUSTOMERS (%)</div>
              <AreaChart series={[{label:'Churn',color:'#f5a623',fill:'rgba(245,166,35,0.08)',data:churnTrend}]} h={70}/>
            </div>
          </div>

          {/* LIVE ANOMALY FEED */}
          <div className="card">
            <Div label="LIVE ANOMALY FEED" right="auto-refresh 4s"/>
            <div style={{display:'flex',flexDirection:'column',gap:7}}>
              {anomalyFeed.map((e,i)=>(
                <div key={i} style={{display:'flex',alignItems:'center',gap:10,padding:'9px 12px',
                  background:i===0?`${sevColor(e.sev)}08`:'var(--s2)',
                  border:`1px solid ${i===0?sevColor(e.sev)+'25':'var(--border)'}`,
                  borderRadius:10,transition:'all .4s',
                  animation:i===0?'fadeUp .4s ease both':'none'}}>
                  <span style={{width:7,height:7,borderRadius:'50%',background:sevColor(e.sev),
                    flexShrink:0,animation:i===0?'pulse 2s infinite':'none'}}/>
                  <span style={{fontSize:9,fontFamily:'monospace',color:'var(--muted)',width:56,flexShrink:0}}>{e.ts}</span>
                  <span style={{fontSize:11,fontWeight:600,flex:1}}>{e.modem}</span>
                  <span style={{fontSize:10,color:'var(--muted)'}}>{e.type}</span>
                  <span style={{fontSize:9,fontFamily:'monospace',color:'var(--muted)'}}>{e.snr}dB</span>
                  <span style={{fontSize:9,fontWeight:700,padding:'1px 7px',borderRadius:20,
                    background:`${sevColor(e.sev)}15`,color:sevColor(e.sev)}}>{e.sev}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* MODEL PERFORMANCE + DATASET CARDS */}
        <div className="grid-3 fade-up-3">
          {[
            { color:'#ff4757', dot:'🔴', title:'P1 — Tower Anomaly', href:'/p1',
              model:'IF + LOF Ensemble', rows:'4.32M', interval:'15 min',
              auc: p1.roc_auc||0.7585, recall: p1.recall||0.311,
              radarData:[{label:'AUC',value:76},{label:'Recall',value:31},{label:'Prec',value:44},{label:'F1',value:36},{label:'Cov',value:88}] },
            { color:'#9b7cf8', dot:'🟣', title:'P2 — Mobile Experience', href:'/p2',
              model:'SARIMA + ±3σ', rows:'3.24M', interval:'15 min',
              auc: null, recall: null,
              mae: p2.forecast_metrics?.mae||8.56, aic: p2.sarima_fit?.aic||13318,
              radarData:[{label:'MAE',value:85},{label:'AIC',value:70},{label:'Stat',value:100},{label:'Covg',value:90},{label:'Link',value:60}] },
            { color:'#00d68f', dot:'🟢', title:'P3 — Churn Prediction', href:'/p3',
              model:'MLP + SHAP', rows:'500', interval:'Per customer',
              auc: p3.roc_auc||p3e.roc_auc||0.84, recall: p3.recall||p3e.recall||0.85,
              radarData:[{label:'AUC',value:84},{label:'Recall',value:85},{label:'Prec',value:48},{label:'F1',value:63},{label:'SHAP',value:95}] },
          ].map(p=>(
            <a key={p.title} href={p.href} style={{textDecoration:'none'}}>
              <div className="card" style={{borderTop:`3px solid ${p.color}`,cursor:'pointer',
                transition:'transform .15s,box-shadow .15s'}}
                onMouseEnter={e=>{(e.currentTarget as HTMLDivElement).style.transform='translateY(-2px)';(e.currentTarget as HTMLDivElement).style.boxShadow=`0 8px 24px ${p.color}20`}}
                onMouseLeave={e=>{(e.currentTarget as HTMLDivElement).style.transform='';(e.currentTarget as HTMLDivElement).style.boxShadow=''}}>
                <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:12}}>
                  <div>
                    <div style={{fontSize:13,fontWeight:700,marginBottom:3}}>{p.dot} {p.title}</div>
                    <div style={{fontSize:10,color:'var(--muted)'}}>
                      {p.rows} rows · {p.interval} · {p.model}
                    </div>
                  </div>
                  <Radar data={p.radarData} size={80} color={p.color}/>
                </div>
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8}}>
                  {p.auc!=null&&[
                    {k:'ROC-AUC',v:p.auc.toFixed(4)},{k:'Recall',v:(p.recall!).toFixed(4)}
                  ].map(m=>(
                    <div key={m.k} style={{background:'var(--s2)',borderRadius:9,padding:'8px 10px'}}>
                      <div style={{fontSize:9,color:'var(--muted)',marginBottom:2}}>{m.k}</div>
                      <div style={{fontSize:16,fontWeight:700,fontFamily:'monospace',color:p.color}}>{m.v}</div>
                    </div>
                  ))}
                  {p.mae&&[
                    {k:'MAE',v:p.mae.toFixed(2)+'pts'},{k:'AIC',v:(p.aic!).toFixed(0)}
                  ].map(m=>(
                    <div key={m.k} style={{background:'var(--s2)',borderRadius:9,padding:'8px 10px'}}>
                      <div style={{fontSize:9,color:'var(--muted)',marginBottom:2}}>{m.k}</div>
                      <div style={{fontSize:16,fontWeight:700,fontFamily:'monospace',color:p.color}}>{m.v}</div>
                    </div>
                  ))}
                </div>
                <div style={{marginTop:10,display:'flex',alignItems:'center',gap:6}}>
                  <span style={{fontSize:10,color:p.color,fontWeight:700}}>Open dashboard →</span>
                </div>
              </div>
            </a>
          ))}
        </div>
      </main>
    </div>
  )
}
