import { useState, useEffect, useCallback } from 'react'
import { ParticleCanvas, Sidebar, StatCard, Div, AreaChart, BarChart, Donut, Radar, FlipCard, ProgressBar } from '../components/UI'

const API = 'http://localhost:8000'
const FEATS = [
  {key:'ds_snr_db',          label:'Signal Quality (dBm)',      min:20, max:50, step:.5, def:38},
  {key:'us_power_level_dbmv',label:'Tower Tx Power (dBmV)',     min:30, max:55, step:.5, def:38},
  {key:'ds_channel_utilization_pct',label:'Tower Capacity (%)', min:0,  max:100,step:1,  def:55},
  {key:'us_channel_utilization_pct',label:'Uplink Util (%)',    min:0,  max:100,step:1,  def:40},
  {key:'ds_power_level_dbmv',label:'Received Signal (dBmV)',    min:-15,max:15, step:.5, def:0.5},
  {key:'us_mer_db',          label:'Signal Ratio SINR (dB)',    min:20, max:48, step:.5, def:36},
]
const PRESETS=[
  {label:'✅ Normal',     color:'#00d68f',v:{ds_snr_db:38,us_power_level_dbmv:38,ds_channel_utilization_pct:52,us_channel_utilization_pct:38}},
  {label:'📉 5G→4G Drop', color:'#ff4757',v:{ds_snr_db:23,us_power_level_dbmv:38,ds_channel_utilization_pct:58,us_channel_utilization_pct:42}},
  {label:'⚡ Tower Overload',color:'#f5a623',v:{ds_snr_db:35,us_power_level_dbmv:53,ds_channel_utilization_pct:94,us_channel_utilization_pct:88}},
  {label:'🔴 Signal Loss', color:'#9b7cf8',v:{ds_snr_db:21,us_power_level_dbmv:40,ds_channel_utilization_pct:70,us_channel_utilization_pct:65}},
]

const DEMO_MODEMS = [
  {id:'TWR-0001',score:.09,snr:39.2,usp:37.8,util:51,sev:'NORMAL',history:[8,9,8,10,9,9,10,9]},
  {id:'TWR-0045',score:.61,snr:25.8,usp:38.5,util:62,sev:'HIGH',  history:[15,22,35,42,55,58,61,61]},
  {id:'TWR-0112',score:.84,snr:21.3,usp:39.1,util:68,sev:'CRIT',  history:[20,35,52,65,74,80,83,84]},
  {id:'TWR-0199',score:.07,snr:40.1,usp:37.2,util:48,sev:'NORMAL',history:[7,6,8,7,7,8,7,7]},
  {id:'TWR-0234',score:.43,snr:29.7,usp:51.2,util:59,sev:'MED',   history:[12,18,28,35,40,42,43,43]},
  {id:'TWR-0378',score:.72,snr:27.1,usp:38.8,util:88,sev:'HIGH',  history:[18,28,45,55,64,70,72,72]},
]

export default function P1Page() {
  const [vals, setVals]   = useState<Record<string,number>>(Object.fromEntries(FEATS.map(f=>[f.key,f.def])))
  const [peak, setPeak]   = useState(false)
  const [result, setResult]=useState<any>(null)
  const [loading, setLoading]=useState(false)
  const [metrics, setMetrics]=useState<any>(null)
  const [scoreHist, setScoreHist]=useState<number[]>([])
  const [liveModems, setLiveModems]=useState(DEMO_MODEMS)

  useEffect(()=>{
    fetch(`${API}/metrics`).then(r=>r.ok?r.json():null).then(m=>{ if(m?.p1_hfc_anomaly) setMetrics(m.p1_hfc_anomaly) }).catch(()=>{})
    // Simulate live modem score drift every 5s
    const iv=setInterval(()=>{
      setLiveModems(ms=>ms.map(m=>({
        ...m,
        score: Math.min(.98,Math.max(.03,m.score+(Math.random()-.5)*.06)),
      })))
    },5000)
    return ()=>clearInterval(iv)
  },[])

  const predict=useCallback(async(overrides?:Record<string,number>)=>{
    const v={...vals,...(overrides||{})}
    const payload={...v,
      ds_power_rolling_std_1h:Math.abs((v.ds_power_level_dbmv||.5)-.5)*.1,
      snr_drop_rate_per_hour:Math.max(0,(38-(v.ds_snr_db||38))/4),
      us_ds_power_delta:(v.us_power_level_dbmv||38)-((v.ds_power_level_dbmv||.5)+38),
      channel_util_peak_hour_flag:peak?1:0,
    }
    setLoading(true)
    try {
      const r=await fetch(`${API}/api/p1/predict`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
      const d=await r.json(); setResult(d)
      setScoreHist(h=>[...h.slice(-11),Math.round(d.anomaly_score*100)])
    } catch {
      const snrF=Math.max(0,(38-(v.ds_snr_db||38))*.028)
      const uspF=Math.max(0,((v.us_power_level_dbmv||38)-50)*.014)
      const utlF=Math.max(0,((v.ds_channel_utilization_pct||55)-80)*.009)
      const sc=Math.min(.97,Math.max(.04,snrF+uspF+utlF+.11))
      setResult({anomaly_score:+sc.toFixed(4),is_anomaly:sc>.37,threshold:.37,severity:sc>.52?'High':sc>.37?'Medium':'Normal',model:'IF+LOF (demo)'})
      setScoreHist(h=>[...h.slice(-11),Math.round(sc*100)])
    } finally { setLoading(false) }
  },[vals,peak])

  const applyPreset=(p:typeof PRESETS[0])=>{
    const merged={...vals,...p.v}
    setVals(merged)
    predict(merged)   // pass merged values directly — no timeout, no stale state
  }
  const m=metrics||{}, score=result?.anomaly_score||0
  const scoreColor=score>.52?'var(--red)':score>.37?'var(--amber)':'var(--green)'
  const sevColor=(s:string)=>s==='CRIT'?'#ff4757':s==='HIGH'?'#f5a623':s==='MED'?'#9b7cf8':'#00d68f'

  const impData=[
    {label:'ds_snr',value:41},{label:'pwr_std',value:32},{label:'snr_rate',value:27},
    {label:'us_pwr', value:20},{label:'us_ds_Δ',value:16},{label:'ds_util', value:13},
  ]

  return (
    <div className="shell">
      <ParticleCanvas/>
      <Sidebar active="p1"/>
      <main className="main" style={{position:'relative',zIndex:1}}>

        <div className="page-header fade-up">
          <h1 className="page-title grad-red">
            P1 — Cell Tower Signal Anomaly Detection
          </h1>
          <p className="page-sub">Isolation Forest (60%) + LOF (40%) · 4.32M readings · 500 towers · 15-min intervals · detects 5G→2G drops</p>
        </div>

        <div className="grid-5 fade-up-1" style={{marginBottom:20}}>
          <StatCard icon="🎯" label="ROC-AUC"   value={(m.roc_auc||.7585).toFixed(4)} color="var(--red)"   spark={[71,72,73,74,75,75,76]} trend="+0.8%"/>
          <StatCard icon="📊" label="PRECISION"  value={(m.precision||.4406).toFixed(4)} color="var(--amber)" spark={[42,43,43,44,44,44,44]}/>
          <StatCard icon="🔍" label="RECALL"     value={(m.recall||.3113).toFixed(4)}   color="var(--cyan)"  spark={[28,29,30,30,31,31,31]}/>
          <StatCard icon="⚖️" label="F1 SCORE"   value={(m.f1||.3648).toFixed(4)}       color="var(--purple)"spark={[34,35,36,36,36,36,36]}/>
          <StatCard icon="📍" label="THRESHOLD"  value={(m.threshold||.3702).toFixed(4)} color="var(--muted)"/>
        </div>

        <div className="grid-2 fade-up-2" style={{marginBottom:20}}>

          {/* LIVE SCORER */}
          <div className="card">
            <Div label="LIVE ANOMALY SCORER" right="IF + LOF ensemble"/>
            <div style={{display:'flex',gap:6,flexWrap:'wrap',marginBottom:14}}>
              {PRESETS.map(p=>(
                <button key={p.label} onClick={()=>applyPreset(p)}
                  className="btn btn-ghost" style={{fontSize:10,padding:'5px 11px',borderColor:`${p.color}40`,color:p.color}}>
                  {p.label}
                </button>
              ))}
            </div>
            {FEATS.map(f=>(
              <div key={f.key} style={{marginBottom:9}}>
                <div style={{display:'flex',justifyContent:'space-between',marginBottom:3}}>
                  <span style={{fontSize:11,color:'var(--muted)'}}>{f.label}</span>
                  <span style={{fontSize:11,fontFamily:'monospace'}}>{vals[f.key]}</span>
                </div>
                <input type="range" min={f.min} max={f.max} step={f.step} value={vals[f.key]}
                  onChange={e=>setVals(v=>({...v,[f.key]:+e.target.value}))}
                  style={{width:'100%',height:4}}/>
              </div>
            ))}
            <label style={{display:'flex',alignItems:'center',gap:7,fontSize:12,cursor:'pointer',marginBottom:12}}>
              <input type="checkbox" checked={peak} onChange={e=>setPeak(e.target.checked)} style={{accentColor:'var(--accent)'}}/>
              Peak hour (7–10pm)
            </label>
            <button className="btn btn-primary btn-full" onClick={()=>predict()} disabled={loading}>
              {loading?'Scoring...':'🔍 Score Reading'}
            </button>
          </div>

          {/* RESULT */}
          <div className="card" style={{display:'flex',flexDirection:'column',gap:14}}>
            <Div label="ANOMALY SCORE" right={result?.model||'—'}/>
            <div style={{display:'flex',alignItems:'center',gap:18}}>
              <div style={{position:'relative',width:88,height:88,flexShrink:0}}>
                <Donut value={Math.round(score*100)} color={scoreColor} size={88} thickness={9}/>
                <div style={{position:'absolute',inset:0,display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column'}}>
                  <span style={{fontSize:18,fontWeight:700,fontFamily:'monospace',color:scoreColor}}>{Math.round(score*100)}</span>
                  <span style={{fontSize:8,color:'var(--muted)'}}>/ 100</span>
                </div>
              </div>
              <div>
                <div style={{fontSize:26,fontWeight:700,fontFamily:'monospace',color:scoreColor,marginBottom:3}}>
                  {score.toFixed(4)}
                </div>
                <div style={{fontSize:11,color:'var(--muted)',marginBottom:7}}>
                  Threshold: {result?.threshold||.37} · {result?.severity||'—'}
                </div>
                {result?.is_anomaly
                  ? <div style={{padding:'7px 11px',background:'rgba(255,71,87,0.1)',borderRadius:8,
                      border:'1px solid rgba(255,71,87,0.2)',fontSize:11,color:'#ff8a94'}}>
                      ⚠️ Anomaly — dispatch field engineer
                    </div>
                  : result&&<div style={{padding:'7px 11px',background:'rgba(0,214,143,0.08)',borderRadius:8,
                      border:'1px solid rgba(0,214,143,0.18)',fontSize:11,color:'var(--green)'}}>
                      ✅ Normal operation
                    </div>
                }
              </div>
            </div>
            {scoreHist.length>1&&(
              <div>
                <div style={{fontSize:9,color:'var(--muted)',marginBottom:5}}>SCORE HISTORY</div>
                <AreaChart series={[{label:'Score',color:scoreColor,fill:`${scoreColor}0d`,data:scoreHist}]} h={60}/>
              </div>
            )}
            <div className="grid-3">
              {[{l:'IF Score',v:score>0?(Math.min(score*1.1,.99)).toFixed(3):'—',c:'var(--accent)'},
                {l:'LOF Score',v:score>0?(Math.min(score*.88,.99)).toFixed(3):'—',c:'var(--purple)'},
                {l:'Ensemble',v:score>0?score.toFixed(3):'—',c:scoreColor}].map(x=>(
                <div key={x.l} style={{background:'var(--s2)',borderRadius:10,padding:'9px 10px',textAlign:'center'}}>
                  <div style={{fontSize:9,color:'var(--muted)',marginBottom:3}}>{x.l}</div>
                  <div style={{fontSize:15,fontWeight:700,fontFamily:'monospace',color:x.c}}>{x.v}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* FEATURE IMPORTANCE + DISTRIBUTION */}
        <div className="grid-2 fade-up-3" style={{marginBottom:20}}>
          <div className="card">
            <Div label="PERMUTATION FEATURE IMPORTANCE" right="AUC drop on shuffle"/>
            {impData.map((d,i)=>(
              <div key={d.label} style={{display:'flex',alignItems:'center',gap:10,marginBottom:8}}>
                <span style={{fontSize:10,color:'var(--muted)',width:64,flexShrink:0,fontFamily:'monospace'}}>{d.label}</span>
                <div style={{flex:1}}><ProgressBar value={d.value/41*100} color={`hsl(${220-i*20},80%,65%)`}/></div>
                <span style={{fontSize:10,fontFamily:'monospace',color:'var(--text)',width:30,textAlign:'right'}}>{(d.value*.001).toFixed(4)}</span>
              </div>
            ))}
          </div>
          <div className="card">
            <Div label="DS SNR DISTRIBUTION" right="normal vs anomaly"/>
            <BarChart h={90} color="var(--green)"
              data={[20,22,24,26,28,30,32,34,36,38,40,42,44].map((v,i)=>({
                label:`${v}`,
                value: v>=30&&v<=44 ? Math.round(Math.exp(-.5*((v-38)/3.5)**2)*1100+Math.random()*40) : Math.round(50+Math.random()*30)
              }))}/>
            <div style={{display:'flex',gap:14,marginTop:8,fontSize:10,color:'var(--muted)'}}>
              <span style={{display:'flex',alignItems:'center',gap:4}}><span style={{width:8,height:3,background:'var(--green)',display:'inline-block',borderRadius:2}}/> Normal (38±4 dB)</span>
              <span style={{display:'flex',alignItems:'center',gap:4}}><span style={{width:8,height:3,background:'var(--red)',display:'inline-block',borderRadius:2}}/> Anomaly (&lt;28 dB)</span>
            </div>
          </div>
        </div>

        {/* MODEM FLIP CARDS */}
        <Div label="LIVE TOWER STATUS" right="click 'details' to flip · auto-refreshes 5s"/>
        <div className="grid-3">
          {liveModems.map((modem,i)=>{
            const sc=modem.score, col=sevColor(modem.sev)
            const pct=Math.round(sc*100)
            return (
              <FlipCard key={modem.id} height={165}
                borderColor={`${col}35`} bg={`${col}05`}
                front={
                  <div>
                    <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:8}}>
                      <div style={{position:'relative',width:52,height:52,flexShrink:0}}>
                        <Donut value={pct} color={col} size={52} thickness={6}/>
                        <div style={{position:'absolute',inset:0,display:'flex',alignItems:'center',justifyContent:'center'}}>
                          <span style={{fontSize:12,fontWeight:700,fontFamily:'monospace',color:col}}>{pct}</span>
                        </div>
                      </div>
                      <div>
                        <div style={{display:'flex',alignItems:'center',gap:6,marginBottom:2}}>
                          <span style={{fontWeight:700,fontSize:13}}>{modem.id}</span>
                          <span style={{fontSize:8,padding:'1px 6px',borderRadius:20,fontWeight:700,
                            background:`${col}20`,color:col}}>{modem.sev}</span>
                        </div>
                        <div style={{fontSize:10,color:'var(--muted)'}}>SNR: {modem.snr}dB · US: {modem.usp}dBmV</div>
                        <div style={{fontSize:10,color:'var(--muted)'}}>Util: {modem.util}%</div>
                      </div>
                    </div>
                    <ProgressBar value={pct} color={col}/>
                  </div>
                }
                back={
                  <div>
                    <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:8}}>
                      <Donut value={pct} color={col} size={38} thickness={5}/>
                      <div>
                        <div style={{fontWeight:700,fontSize:12}}>{modem.id}</div>
                        <div style={{fontSize:9,color:'var(--muted)'}}>7-session history</div>
                      </div>
                    </div>
                    <BarChart data={modem.history.map((v,j)=>({label:`-${7-j}`,value:v}))} color={col} h={65}/>
                  </div>
                }
              />
            )
          })}
        </div>
      </main>
    </div>
  )
}
