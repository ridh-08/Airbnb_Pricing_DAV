
// ── CONSTANTS ────────────────────────────────────────────────────────
const CC = {newyork:'#FF5A5F',chicago:'#00C4B4',nashville:'#FC642D',
            neworleans:'#CC44AA',austin:'#00D4E8',losangeles:'#9494B8'};
const CL = {newyork:'New York',chicago:'Chicago',nashville:'Nashville',
            neworleans:'New Orleans',austin:'Austin',losangeles:'Los Angeles'};
const CO = ['newyork','chicago','nashville','neworleans','austin','losangeles'];
const CL_LABELS = ['Budget / High Avail','Mid-Range Active','Well-Equipped','Premium / Sparse'];
const TAGLINES = {newyork:'Tourism-dominant market',chicago:'Mixed-demand market',
  nashville:'Music city · emerging',neworleans:'Festival & tourism hub',
  austin:'Tech & culture driven',losangeles:'Sprawling coastal market'};

const BASE = {
  paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{family:"'Outfit',sans-serif",color:'#e2e2f5',size:12},
  margin:{t:8,r:8,b:40,l:50},
  xaxis:{gridcolor:'#1a1a35',zeroline:false,tickfont:{size:11,color:'#6060a0'},linecolor:'#1a1a35'},
  yaxis:{gridcolor:'#1a1a35',zeroline:false,tickfont:{size:11,color:'#6060a0'},linecolor:'#1a1a35'},
};
const CFG = {responsive:true,displayModeBar:false};

let selectedCity = 'newyork';
let DATA = null;
let PRED_META = {};

// ── CURSOR ───────────────────────────────────────────────────────────
const cur = document.getElementById('cursor');
const curR = document.getElementById('cursor-ring');
let mx=0,my=0,rx=0,ry=0;
document.addEventListener('mousemove',e=>{
  mx=e.clientX; my=e.clientY;
  cur.style.left=mx+'px'; cur.style.top=my+'px';
});
function animRing(){
  rx+=(mx-rx)*.12; ry+=(my-ry)*.12;
  curR.style.left=rx+'px'; curR.style.top=ry+'px';
  requestAnimationFrame(animRing);
}
animRing();
document.querySelectorAll('a,button,.city-card,.sp-card,.sn-dot,.r2-item').forEach(el=>{
  el.addEventListener('mouseenter',()=>{cur.style.width='18px';cur.style.height='18px'});
  el.addEventListener('mouseleave',()=>{cur.style.width='10px';cur.style.height='10px'});
});

// ── TWEEN ────────────────────────────────────────────────────────────
function tween(el,start,end,dur,pre='',suf='',dec=0){
  const startT=performance.now();
  function tick(now){
    const p=Math.min((now-startT)/dur,1);
    const e=1-Math.pow(1-p,3);
    const v=start+(end-start)*e;
    el.textContent=pre+(dec>0?v.toFixed(dec):Math.round(v).toLocaleString())+suf;
    if(p<1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── SCROLL OBSERVER ──────────────────────────────────────────────────
const obs=new IntersectionObserver(entries=>{
  entries.forEach(e=>{if(e.isIntersecting) e.target.classList.add('visible')});
},{threshold:.12});
document.querySelectorAll('.reveal').forEach(el=>obs.observe(el));

// ── SECTION NAV ──────────────────────────────────────────────────────
const secs=['hero','profile','predict','progress','compare','shap','cluster','spatial','calendar','stats'];
const secObs=new IntersectionObserver(entries=>{
  entries.forEach(e=>{
    if(e.isIntersecting){
      document.querySelectorAll('.sn-dot').forEach(d=>d.classList.remove('active'));
      const dot=document.querySelector(`.sn-dot[data-sec="${e.target.id}"]`);
      if(dot) dot.classList.add('active');
    }
  });
},{rootMargin:'-40% 0px -40% 0px'});
secs.forEach(id=>{const s=document.getElementById(id);if(s) secObs.observe(s)});
document.querySelectorAll('.sn-dot').forEach(d=>{
  d.addEventListener('click',()=>{
    const s=document.getElementById(d.dataset.sec);
    if(s) s.scrollIntoView({behavior:'smooth',block:'start'});
  });
});

// ── ACCENT COLOR ─────────────────────────────────────────────────────
function setAccent(city){
  const c=CC[city];
  const r=parseInt(c.slice(1,3),16);
  const g=parseInt(c.slice(3,5),16);
  const b=parseInt(c.slice(5,7),16);
  document.documentElement.style.setProperty('--active',c);
  document.documentElement.style.setProperty('--active-20',c+'33');
  document.documentElement.style.setProperty('--active-08',c+'14');
  cur.style.background=c;
  curR.style.borderColor=c;
}

// ── MAIN DATA FETCH ──────────────────────────────────────────────────
fetch('/api/data').then(r=>r.json()).then(d=>{
  DATA=d;
  buildCityCards();
  buildPredictorSection();
  buildProgressionSection();
  buildCompareCharts();
  buildSHAPCharts();
  buildClusterCharts();
  buildSpatialSection();
  buildCalendarSection();
  buildStatsSection();
  selectCity('newyork',true);
  initMapButtons();
}).catch(err=>console.error(err));

// ── CITY SELECTION ───────────────────────────────────────────────────
function selectCity(city,init=false){
  selectedCity=city;
  setAccent(city);

  // pill state
  document.querySelectorAll('.city-card').forEach(c=>{
    c.classList.toggle('active',c.dataset.city===city);
  });
  // map buttons
  document.querySelectorAll('.map-tog').forEach(b=>{
    b.classList.toggle('active',b.dataset.city===city);
  });

  // update live profile
  fetch(`/api/city/${city}`).then(r=>r.json()).then(s=>{
    document.getElementById('profileName').textContent=CL[city]||city;
    document.getElementById('profileName').style.color=CC[city];
    document.getElementById('profileTagline').textContent=TAGLINES[city]||'';

    const prev={med:0,r2:0,av:0,li:0};
    if(!init){
      prev.med=parseFloat(document.getElementById('kpiMedian').textContent.replace(/[$,]/g,''))||0;
      prev.r2=parseFloat(document.getElementById('kpiR2').textContent)||0;
      prev.av=parseFloat(document.getElementById('kpiAvail').textContent)||0;
      prev.li=parseInt(document.getElementById('kpiListings').textContent.replace(/,/g,''))||0;
    }
    tween(document.getElementById('kpiMedian'),prev.med,s.median_price||0,700,'$','');
    tween(document.getElementById('kpiR2'),prev.r2,s.r2||0,700,'','',2);
    tween(document.getElementById('kpiAvail'),prev.av,s.availability_pct||0,700,'','%',1);
    tween(document.getElementById('kpiListings'),prev.li,s.unique_listings||0,700,'','');

    document.getElementById('profNeigh').textContent=
      (s.top_neighbourhood||'—')+' · $'+(s.top_neigh_price||'—').toLocaleString();
    document.getElementById('profMorans').textContent='Moran\'s I: '+(s.morans_i||'—');
    document.getElementById('profTopFeat').textContent='Top driver: '+(s.top_feature||'—');
  });

  // SHAP bars
  updateSHAPBars(city);

  // spotlight charts
  if(!init){
    spotlightBarChart('chartPrice', city);
    spotlightFEChart(city);
    spotlightScatter(city);
    updateR2List(city);
  }

  // maps
  loadMaps(city);

  // predictor controls
  setupPredictorForCity(city);
}

// ── CITY SELECTOR CARDS ──────────────────────────────────────────────
function buildCityCards(){
  const row=document.getElementById('csRow');
  CO.forEach(city=>{
    const cal=DATA.calendar.find(d=>d.city===city)||{};
    const r2d=DATA.r2.find(d=>d.city===city)||{};
    const r2=(r2d.r2||0);
    const card=document.createElement('button');
    card.className='city-card';
    card.dataset.city=city;
    const color=CC[city];
    card.style.setProperty('--cc',color);
    card.innerHTML=`
      <div class="cc-name">${CL[city]}</div>
      <div class="cc-price">$${cal.median_price||'—'}</div>
      <div class="cc-r2">R² ${r2.toFixed(3)} · ${((cal.availability_rate||0)*100).toFixed(0)}% avail</div>
      <div class="cc-bar"><div class="cc-bar-fill" style="width:${r2*100}%"></div></div>`;
    card.addEventListener('click',()=>selectCity(city));
    row.appendChild(card);
  });
}

// ── PREDICTOR ───────────────────────────────────────────────────────
function buildPredictorSection(){
  fetch('/api/predict/meta').then(r=>r.json()).then(meta=>{
    PRED_META=meta||{};

    ['predAccommodates','predAmenities','predAvailability','predReviewDensity'].forEach(id=>{
      const el=document.getElementById(id);
      if(el){
        el.addEventListener('input',updatePredictorLabels);
      }
    });

    const runBtn=document.getElementById('predRun');
    if(runBtn){
      runBtn.addEventListener('click',runPrediction);
    }

    const neighSel=document.getElementById('predNeighbourhood');
    if(neighSel){
      neighSel.addEventListener('change',runPrediction);
    }

    const sensSel=document.getElementById('predSensitivityFeature');
    if(sensSel){
      sensSel.addEventListener('change',()=>{
        if(PRED_META[selectedCity]){
          renderPredictionSensitivity(getPredictorPayload());
        }
      });
    }

    setupPredictorForCity(selectedCity);
  }).catch(()=>{
    document.getElementById('predPrice').textContent='$—';
    document.getElementById('predBand').textContent='Prediction unavailable (city model metadata not loaded).';
  });
}

function setupPredictorForCity(city){
  const meta=PRED_META[city];
  if(!meta) return;

  const setRange=(id,key,step='1')=>{
    const el=document.getElementById(id);
    const r=meta.ranges[key];
    if(!el||!r) return;
    el.min=Math.floor(r.min);
    el.max=Math.ceil(r.max);
    el.step=step;
    el.value=r.default;
  };

  setRange('predAccommodates','accommodates','1');
  setRange('predAmenities','amenities_count','1');
  setRange('predAvailability','availability_365','1');
  setRange('predReviewDensity','review_density','0.05');

  const sel=document.getElementById('predNeighbourhood');
  if(sel){
    const opts=(meta.top_neighbourhoods||[]).map(n=>`<option value="${n}">${n}</option>`).join('');
    sel.innerHTML=opts+'<option value="OTHER">OTHER</option>';
    sel.value=meta.default_neighbourhood||'OTHER';
  }

  const sens=document.getElementById('predSensitivityFeature');
  if(sens && !sens.value){
    sens.value='accommodates';
  }

  updatePredictorLabels();
  runPrediction();
}

function updatePredictorLabels(){
  const a=document.getElementById('predAccommodates');
  const m=document.getElementById('predAmenities');
  const av=document.getElementById('predAvailability');
  const rd=document.getElementById('predReviewDensity');
  if(!a||!m||!av||!rd) return;

  document.getElementById('predAccommodatesVal').textContent=Math.round(Number(a.value||0));
  document.getElementById('predAmenitiesVal').textContent=Math.round(Number(m.value||0));
  document.getElementById('predAvailabilityVal').textContent=Math.round(Number(av.value||0));
  document.getElementById('predReviewDensityVal').textContent=Number(rd.value||0).toFixed(2);
}

function getPredictorPayload(){
  return {
    accommodates:Number(document.getElementById('predAccommodates').value),
    amenities_count:Number(document.getElementById('predAmenities').value),
    availability_365:Number(document.getElementById('predAvailability').value),
    review_density:Number(document.getElementById('predReviewDensity').value),
    neighbourhood:document.getElementById('predNeighbourhood').value,
  };
}

function runPrediction(){
  if(!PRED_META[selectedCity]) return;

  const payload=getPredictorPayload();

  fetch(`/api/predict/${selectedCity}`,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload),
  }).then(r=>r.json()).then(res=>{
    if(res.error){
      document.getElementById('predPrice').textContent='$—';
      document.getElementById('predBand').textContent='Prediction unavailable for this city.';
      return;
    }

    document.getElementById('predPrice').textContent=`$${Math.round(res.prediction).toLocaleString()}`;
    document.getElementById('predBand').textContent=`Prediction interval: $${Math.round(res.band_low).toLocaleString()} - $${Math.round(res.band_high).toLocaleString()}`;
    document.getElementById('predMetaCity').textContent=`City model: ${CL[selectedCity]} · XGBoost`;
    document.getElementById('predMetaNeigh').textContent=`Neighbourhood: ${res.used_neighbourhood}`;
    document.getElementById('predMetaRmse').textContent=`Model uncertainty (RMSE): $${Math.round(res.rmse).toLocaleString()}`;
    document.getElementById('predMetaSample').textContent=`Training sample: ${Number(res.sample_size).toLocaleString()} listings`;
    renderPredictionSensitivity(payload);
  }).catch(()=>{
    document.getElementById('predPrice').textContent='$—';
    document.getElementById('predBand').textContent='Prediction request failed.';
    Plotly.purge('chartPredSensitivity');
  });
}

function renderPredictionSensitivity(basePayload){
  const sel=document.getElementById('predSensitivityFeature');
  const feature=(sel && sel.value) ? sel.value : 'accommodates';
  const labels={
    accommodates:'Accommodates',
    amenities_count:'Amenities Count',
    availability_365:'Availability (days/year)',
    review_density:'Review Density',
  };

  const payload={...basePayload,feature,points:14};
  fetch(`/api/predict/sweep/${selectedCity}`,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload),
  }).then(r=>r.json()).then(res=>{
    if(res.error || !res.x || !res.y){
      Plotly.purge('chartPredSensitivity');
      return;
    }

    Plotly.newPlot('chartPredSensitivity',[{
      type:'scatter',
      mode:'lines+markers',
      x:res.x,
      y:res.y,
      line:{color:CC[selectedCity],width:3},
      marker:{color:CC[selectedCity],size:6},
      hovertemplate:'%{x:.2f} → $%{y:,.0f}<extra></extra>',
    }],{
      ...BASE,
      margin:{t:26,r:8,b:42,l:56},
      title:{text:`Sensitivity: ${labels[feature]||feature}`,font:{size:11,color:'#9bb0ca',family:"'IBM Plex Mono',monospace"}},
      xaxis:{...BASE.xaxis,title:{text:labels[feature]||feature,font:{size:10,color:'#6060a0'}}},
      yaxis:{...BASE.yaxis,title:{text:'Predicted Price ($)',font:{size:10,color:'#6060a0'}}},
    },CFG);
  }).catch(()=>{
    Plotly.purge('chartPredSensitivity');
  });
}

// ── MODEL PROGRESSION ───────────────────────────────────────────────
function buildProgressionSection(){
  const rows=(DATA.xgb_progression||[]).slice();
  if(!rows.length){
    const t=document.getElementById('progressTable');
    if(t){ t.innerHTML='<tr><td style="padding:10px;color:var(--muted)">Progression data not found.</td></tr>'; }
    return;
  }

  const cityOrder=CO.filter(c=>rows.some(r=>String(r.city).toLowerCase()===c));
  const ordered=cityOrder.map(c=>rows.find(r=>String(r.city).toLowerCase()===c)).filter(Boolean);

  const stage0=ordered.map(r=>Number(r.stage0_old_r2||0));
  const stage1=ordered.map(r=>Number(r.stage1_random_r2||0));
  const stage2=ordered.map(r=>Number(r.stage2_grouped_r2||0));

  Plotly.newPlot('chartProgressionR2',[
    {type:'scatter',mode:'lines+markers',name:'Stage 0 (baseline)',x:cityOrder.map(c=>CL[c]),y:stage0,line:{width:2,color:'#8fa0b6'},marker:{size:7,color:'#8fa0b6'}},
    {type:'scatter',mode:'lines+markers',name:'Stage 1 (enhanced random)',x:cityOrder.map(c=>CL[c]),y:stage1,line:{width:2,color:'#ffb703'},marker:{size:7,color:'#ffb703'}},
    {type:'scatter',mode:'lines+markers',name:'Stage 2 (grouped split)',x:cityOrder.map(c=>CL[c]),y:stage2,line:{width:3,color:'#4cc9f0'},marker:{size:8,color:'#4cc9f0'}},
  ],{
    ...BASE,
    margin:{t:8,r:8,b:55,l:50},
    yaxis:{...BASE.yaxis,title:{text:'R2',font:{size:10,color:'#6060a0'}}},
    xaxis:{...BASE.xaxis,tickangle:-20},
    legend:{font:{size:10,color:'#6060a0'},bgcolor:'rgba(0,0,0,0)',orientation:'h',y:-0.26,x:0},
    showlegend:true,
  },CFG);

  const tbl=document.getElementById('progressTable');
  if(!tbl) return;
  tbl.innerHTML=`<thead style="font-family:var(--fm);font-size:9px;letter-spacing:.12em;color:var(--muted)">
    <tr>
      <th style="padding:8px 10px;text-align:left;border-bottom:1px solid var(--b2);text-transform:uppercase">City</th>
      <th style="padding:8px 10px;text-align:right;border-bottom:1px solid var(--b2);text-transform:uppercase">Stage 0</th>
      <th style="padding:8px 10px;text-align:right;border-bottom:1px solid var(--b2);text-transform:uppercase">Stage 1</th>
      <th style="padding:8px 10px;text-align:right;border-bottom:1px solid var(--b2);text-transform:uppercase">Stage 2</th>
      <th style="padding:8px 10px;text-align:right;border-bottom:1px solid var(--b2);text-transform:uppercase">Delta S2-S0</th>
    </tr>
  </thead>
  <tbody>
    ${ordered.map(r=>{
      const city=String(r.city).toLowerCase();
      const d20=Number(r.delta_old_to_stage2||0);
      const dColor=d20>=0 ? '#4ade80' : '#ff7a59';
      return `<tr style="border-bottom:1px solid var(--b1)">
        <td style="padding:9px 10px;color:${CC[city]||'var(--text)'};font-family:var(--fm);font-size:11px">${CL[city]||city}</td>
        <td style="padding:9px 10px;text-align:right;font-family:var(--fm);font-size:11px">${Number(r.stage0_old_r2||0).toFixed(3)}</td>
        <td style="padding:9px 10px;text-align:right;font-family:var(--fm);font-size:11px">${Number(r.stage1_random_r2||0).toFixed(3)}</td>
        <td style="padding:9px 10px;text-align:right;font-family:var(--fm);font-size:11px">${Number(r.stage2_grouped_r2||0).toFixed(3)}</td>
        <td style="padding:9px 10px;text-align:right;font-family:var(--fm);font-size:11px;color:${dColor}">${d20>=0?'+':''}${d20.toFixed(3)}</td>
      </tr>`;
    }).join('')}
  </tbody>`;
}

// ── SHAP BARS ────────────────────────────────────────────────────────
function updateSHAPBars(city){
  const el=document.getElementById('shapBarsBody');
  const rows=DATA.shap.filter(d=>d.city===city).sort((a,b)=>a.rank-b.rank);
  if(!rows.length){el.innerHTML='<div style="color:var(--muted);font-size:12px">No data</div>';return;}
  const maxVal=rows[0].mean_abs_shap||1;
  el.innerHTML=rows.map(r=>`
    <div class="shap-item">
      <div class="shap-lbl">${r.feature.replace('neigh_','').replace(/_/g,' ')}</div>
      <div class="shap-track"><div class="shap-fill" data-w="${(r.mean_abs_shap/maxVal*100).toFixed(1)}%" style="background:${CC[city]}"></div></div>
      <div class="shap-rank" style="color:${CC[city]}">#${r.rank}</div>
    </div>`).join('');
  // animate after paint
  requestAnimationFrame(()=>{
    el.querySelectorAll('.shap-fill').forEach((f,i)=>{
      setTimeout(()=>{ f.style.width=f.dataset.w; },i*80);
    });
  });
}

// ── COMPARISON CHARTS ────────────────────────────────────────────────
function buildCompareCharts(){
  buildPriceBar();
  buildR2List(selectedCity);
  buildFEChart();
  buildScatterChart();
}

function buildPriceBar(){
  const cal=DATA.calendar;
  const ord=CO.map(c=>cal.find(d=>d.city===c)).filter(Boolean);
  Plotly.newPlot('chartPrice',[{
    type:'bar',
    x:ord.map(d=>CL[d.city]),
    y:ord.map(d=>d.median_price),
    marker:{color:ord.map(d=>CC[d.city]),opacity:.85,
      line:{color:'rgba(0,0,0,0)',width:0}},
    text:ord.map(d=>'$'+d.median_price),
    textposition:'outside',
    textfont:{family:"'JetBrains Mono',monospace",size:11,color:'#e2e2f5'},
    hovertemplate:'<b>%{x}</b><br>$%{y}/night<extra></extra>',
  }],{...BASE,
    margin:{t:8,r:8,b:55,l:50},
    yaxis:{...BASE.yaxis,title:{text:'USD/night',font:{size:10,color:'#6060a0'}}},
    xaxis:{...BASE.xaxis,tickangle:-20},
  },CFG);
}

function buildR2List(activeCity){
  const el=document.getElementById('r2List');
  if(!el) return;
  const sorted=[...DATA.r2].sort((a,b)=>b.r2-a.r2);
  el.innerHTML=sorted.map(d=>`
    <div class="r2-item ${d.city===activeCity?'highlighted':''}" data-city="${d.city}"
         onclick="selectCity('${d.city}')">
      <div class="r2-city-name" style="color:${d.city===activeCity?CC[d.city]:'var(--text)'}">${CL[d.city]||d.city}</div>
      <div class="r2-track"><div class="r2-bar" data-w="${(d.r2*100).toFixed(1)}%"
        style="background:${d.city===activeCity?CC[d.city]:'var(--dim)'};width:0%"></div></div>
      <div class="r2-score" style="color:${d.city===activeCity?CC[d.city]:'var(--muted)'}">${d.r2.toFixed(3)}</div>
    </div>`).join('');
  requestAnimationFrame(()=>{
    el.querySelectorAll('.r2-bar').forEach((b,i)=>{
      setTimeout(()=>{ b.style.width=b.dataset.w; },i*80);
    });
  });
}

function updateR2List(city){buildR2List(city);}

function spotlightBarChart(divId,city){
  const d=Plotly.d3.select('#'+divId).node();
  if(!d) return;
  const cityLabels=CO.map(c=>CL[c]);
  const selectedLabel=CL[city];
  const opacities=cityLabels.map(l=>l===selectedLabel?1:.22);
  Plotly.restyle(divId,{'marker.opacity':[opacities]},{transition:{duration:350}});
}

function buildFEChart(){
  const fe=DATA.fixed_effects||[];
  const cityRows=fe.filter(d=>d.term&&d.term.includes('T.')&&d.term.includes('city'))
    .map(d=>({city:d.term.replace('C(city)[T.','').replace(']',''),coef:d.coefficient}))
    .sort((a,b)=>b.coef-a.coef);
  Plotly.newPlot('chartFE',[{
    type:'bar',orientation:'h',
    y:cityRows.map(r=>CL[r.city]||r.city),
    x:cityRows.map(r=>r.coef),
    marker:{color:cityRows.map(r=>CC[r.city]||'#666'),opacity:.85},
    text:cityRows.map(r=>(r.coef>0?'+':'')+r.coef.toFixed(3)),
    textposition:'outside',
    textfont:{family:"'JetBrains Mono',monospace",size:10,color:'#e2e2f5'},
    hovertemplate:'<b>%{y}</b><br>log-price vs Austin: %{x:.3f}<extra></extra>',
  }],{...BASE,
    margin:{t:8,r:60,b:40,l:110},
    xaxis:{...BASE.xaxis,zeroline:true,zerolinecolor:'#282845',zerolinewidth:1,
      title:{text:'log-price coeff (vs Austin)',font:{size:10,color:'#6060a0'}}},
  },CFG);
}

function spotlightFEChart(city){
  const fe=DATA.fixed_effects||[];
  const cityRows=fe.filter(d=>d.term&&d.term.includes('T.')&&d.term.includes('city'))
    .map(d=>({city:d.term.replace('C(city)[T.','').replace(']','')}))
    .sort((a,b)=>{const r=DATA.fixed_effects.find(x=>x.term.includes(b.city))?.coefficient||0;
      const l=DATA.fixed_effects.find(x=>x.term.includes(a.city))?.coefficient||0;
      return r-l;});
  const labels=cityRows.map(r=>CL[r.city]||r.city);
  const sel=CL[city];
  Plotly.restyle('chartFE',{'marker.opacity':[labels.map(l=>l===sel?1:.25)]});
}

function buildScatterChart(){
  const cal=DATA.calendar||[];
  Plotly.newPlot('chartScatter',[{
    type:'scatter',mode:'markers+text',
    x:cal.map(d=>d.availability_rate*100),
    y:cal.map(d=>d.median_price),
    text:cal.map(d=>CL[d.city]||d.city),
    textposition:'top center',
    textfont:{size:10,color:'#e2e2f5',family:"'JetBrains Mono',monospace"},
    marker:{color:cal.map(d=>CC[d.city]||'#666'),size:14,
      line:{color:'#04040d',width:2}},
    hovertemplate:'<b>%{text}</b><br>Avail: %{x:.1f}%<br>Median: $%{y}<extra></extra>',
  }],{...BASE,
    margin:{t:8,r:8,b:50,l:60},
    xaxis:{...BASE.xaxis,title:{text:'Availability rate (%)',font:{size:10,color:'#6060a0'}}},
    yaxis:{...BASE.yaxis,title:{text:'Median price ($)',font:{size:10,color:'#6060a0'}}},
  },CFG);
}

function spotlightScatter(city){
  const cal=DATA.calendar||[];
  const sel=CL[city];
  const sizes=cal.map(d=>(CL[d.city]||d.city)===sel?18:11);
  const opacities=cal.map(d=>(CL[d.city]||d.city)===sel?1:.3);
  Plotly.restyle('chartScatter',{'marker.size':[sizes],'marker.opacity':[opacities]});
}

// ── SHAP CHARTS ──────────────────────────────────────────────────────
function buildSHAPCharts(){
  buildSHAPHeatmap();
  buildNeighShareChart();
  buildTopDriverChart();
}

function buildSHAPHeatmap(){
  const univ=['accommodates','amenities_count','review_density','availability_365'];
  const all=[...new Set(DATA.shap.map(d=>d.feature))];
  const neigh=all.filter(f=>!univ.includes(f));
  const features=[...univ,...neigh];
  const z=features.map(feat=>CO.map(city=>{
    const r=DATA.shap.find(d=>d.city===city&&d.feature===feat);
    return r?r.rank:7;
  }));
  const txt=features.map(feat=>CO.map(city=>{
    const r=DATA.shap.find(d=>d.city===city&&d.feature===feat);
    return r?`#${r.rank}`:'';
  }));
  Plotly.newPlot('chartSHAP',[{
    type:'heatmap',
    z,
    x:CO.map(c=>CL[c]),
    y:features.map(f=>f.replace('neigh_','📍 ').replace(/_/g,' ')),
    text:txt,texttemplate:'%{text}',
    textfont:{family:"'JetBrains Mono',monospace",size:13,color:'#fff'},
    colorscale:[[0,'#FF5A5F'],[.2,'#FC642D'],[.4,'#CC44AA'],[.6,'#006070'],[.85,'#1a1a35'],[1,'#0e0e22']],
    showscale:true,zmin:1,zmax:7,
    colorbar:{tickvals:[1,2,3,4,5],ticktext:['#1','#2','#3','#4','#5'],
      thickness:12,len:.75,bgcolor:'rgba(0,0,0,0)',bordercolor:'#1a1a35',
      tickfont:{size:10,color:'#6060a0'}},
    hovertemplate:'<b>%{y}</b><br>%{x}: Rank %{z}<extra></extra>',
  }],{...BASE,
    margin:{t:8,r:110,b:40,l:180},
    xaxis:{...BASE.xaxis,side:'bottom',tickangle:-10},
    yaxis:{...BASE.yaxis,autorange:'reversed'},
  },CFG);
}

function buildNeighShareChart(){
  const dom=DATA.shap_dominance||[];
  const ord=CO.map(c=>dom.find(d=>d.city===c)).filter(Boolean);
  Plotly.newPlot('chartNeighShare',[
    {type:'bar',name:'Neighbourhood',
      x:ord.map(d=>CL[d.city]),y:ord.map(d=>d.neighbourhood_share_pct),
      marker:{color:'#CC44AA',opacity:.85},
      hovertemplate:'%{x}<br>Neighbourhood: %{y:.1f}%<extra></extra>'},
    {type:'bar',name:'Amenities',
      x:ord.map(d=>CL[d.city]),y:ord.map(d=>d.amenities_share_pct),
      marker:{color:'#00D4E8',opacity:.85},
      hovertemplate:'%{x}<br>Amenities: %{y:.1f}%<extra></extra>'},
  ],{...BASE,barmode:'group',
    margin:{t:8,r:8,b:55,l:50},
    yaxis:{...BASE.yaxis,title:{text:'% SHAP importance',font:{size:10,color:'#6060a0'}}},
    xaxis:{...BASE.xaxis,tickangle:-20},
    legend:{font:{size:10,color:'#6060a0'},bgcolor:'rgba(0,0,0,0)',
      orientation:'h',y:-0.22,x:0},
    showlegend:true,
  },CFG);
}

function buildTopDriverChart(){
  const dom=DATA.shap_dominance||[];
  const ord=CO.map(c=>dom.find(d=>d.city===c)).filter(Boolean);
  Plotly.newPlot('chartTopDriver',[{
    type:'bar',
    x:ord.map(d=>CL[d.city]),
    y:ord.map(d=>d.top_value),
    text:ord.map(d=>d.top_feature),
    textposition:'inside',
    textfont:{family:"'JetBrains Mono',monospace",size:9,color:'rgba(255,255,255,.8)'},
    marker:{color:ord.map(d=>CC[d.city]),opacity:.9},
    hovertemplate:'<b>%{x}</b><br>%{text}<br>SHAP: %{y:.1f}<extra></extra>',
  }],{...BASE,
    margin:{t:8,r:8,b:55,l:50},
    yaxis:{...BASE.yaxis,title:{text:'SHAP value',font:{size:10,color:'#6060a0'}}},
    xaxis:{...BASE.xaxis,tickangle:-20},
  },CFG);
}

// ── CLUSTER CHARTS ───────────────────────────────────────────────────
function buildClusterCharts(){
  const cc=DATA.cluster_comp||[];
  const clusters=[...new Set(cc.map(d=>d.cluster_pooled))].sort((a,b)=>a-b);

  // grouped bar: cities within each cluster
  const traces=CO.map(city=>({
    type:'bar',name:CL[city],
    x:clusters.map(c=>CL_LABELS[c]||'C'+c),
    y:clusters.map(c=>{
      const r=cc.find(d=>d.cluster_pooled===c&&d.city===city);
      return r?r.pct_within_cluster:0;
    }),
    marker:{color:CC[city],opacity:.85},
    hovertemplate:`<b>${CL[city]}</b><br>%{x}: %{y:.1f}%<extra></extra>`,
  }));
  Plotly.newPlot('chartCluster',traces,{...BASE,barmode:'group',
    margin:{t:8,r:8,b:80,l:50},
    yaxis:{...BASE.yaxis,title:{text:'% within cluster',font:{size:10,color:'#6060a0'}}},
    xaxis:{...BASE.xaxis,tickangle:-15},
    legend:{font:{size:10,color:'#6060a0'},bgcolor:'rgba(0,0,0,0)',orientation:'h',y:-0.32,x:0},
    showlegend:true,
  },CFG);

  // stacked absolute
  const traces2=CO.map(city=>({
    type:'bar',name:CL[city],
    x:clusters.map(c=>CL_LABELS[c]||'C'+c),
    y:clusters.map(c=>{
      const r=cc.find(d=>d.cluster_pooled===c&&d.city===city);
      return r?r.count:0;
    }),
    marker:{color:CC[city],opacity:.85},
    hovertemplate:`<b>${CL[city]}</b><br>%{x}: %{y:,}<extra></extra>`,
  }));
  Plotly.newPlot('chartClusterStack',traces2,{...BASE,barmode:'stack',
    margin:{t:8,r:8,b:80,l:60},
    yaxis:{...BASE.yaxis,title:{text:'Listings',font:{size:10,color:'#6060a0'}}},
    xaxis:{...BASE.xaxis,tickangle:-15},
    legend:{font:{size:10,color:'#6060a0'},bgcolor:'rgba(0,0,0,0)',orientation:'h',y:-0.32,x:0},
    showlegend:true,
  },CFG);

  // host archetypes
  const icons=['🛌','📈','💎'];
  const accentColors=['#00D4E8','#FC642D','#FF5A5F'];
  const el=document.getElementById('archetypeRow');
  (DATA.host_clusters||[]).forEach((d,i)=>{
    const c=accentColors[i%3];
    const div=document.createElement('div');
    div.className='arch';
    div.innerHTML=`
      <div class="arch-stripe" style="background:${c}"></div>
      <span class="arch-icon">${icons[i%3]}</span>
      <div class="arch-name" style="color:${c}">${d.cluster_name}</div>
      <div class="arch-row-item"><span class="arch-key">Hosts</span><span class="arch-v">${Number(d.host_count).toLocaleString()}</span></div>
      <div class="arch-row-item"><span class="arch-key">Avg price</span><span class="arch-v">$${Math.round(d.avg_price).toLocaleString()}</span></div>
      <div class="arch-row-item"><span class="arch-key">Avg availability</span><span class="arch-v">${Math.round(d.avg_availability)} days/yr</span></div>
      <div class="arch-row-item"><span class="arch-key">Review density</span><span class="arch-v">${Number(d.avg_review_density).toFixed(2)}</span></div>
    `;
    el.appendChild(div);
  });
}

// ── SPATIAL ──────────────────────────────────────────────────────────
function buildSpatialSection(){
  const sp=DATA.spatial||[];
  const el=document.getElementById('spatialCards');
  CO.map(c=>sp.find(d=>d.city===c)).filter(Boolean).forEach(d=>{
    const div=document.createElement('div');
    div.className='sp-card';
    div.onclick=()=>selectCity(d.city);
    div.style.borderColor=CC[d.city]+'40';
    div.innerHTML=`
      <div class="sp-city" style="color:${CC[d.city]}">${CL[d.city]}</div>
      <div class="sp-neigh">${d.top_priced_neighbourhood}</div>
      <div class="sp-price" style="color:${CC[d.city]}">$${Math.round(d.top_avg_price).toLocaleString()}</div>
      <div class="sp-detail">Moran's I: ${Number(d.morans_i).toFixed(3)} · ${d.spatial_cluster_class.replace(/_/g,' ')}</div>`;
    el.appendChild(div);
  });

  // Moran's I
  const ord=CO.map(c=>sp.find(d=>d.city===c)).filter(Boolean);
  Plotly.newPlot('chartMorans',[{
    type:'bar',
    x:ord.map(d=>CL[d.city]),y:ord.map(d=>d.morans_i),
    marker:{color:ord.map(d=>CC[d.city]),opacity:.9},
    text:ord.map(d=>Number(d.morans_i).toFixed(3)),
    textposition:'outside',
    textfont:{family:"'JetBrains Mono',monospace",size:11,color:'#e2e2f5'},
    hovertemplate:"<b>%{x}</b><br>Moran's I: %{y:.3f}<extra></extra>",
  }],{...BASE,
    margin:{t:8,r:8,b:55,l:55},
    yaxis:{...BASE.yaxis,range:[0,.6],title:{text:"Moran's I",font:{size:10,color:'#6060a0'}}},
    xaxis:{...BASE.xaxis,tickangle:-20},
  },CFG);

  // top neighbourhood prices
  Plotly.newPlot('chartTopNeigh',[{
    type:'bar',orientation:'h',
    y:ord.map(d=>CL[d.city]),
    x:ord.map(d=>d.top_avg_price),
    marker:{color:ord.map(d=>CC[d.city]),opacity:.85},
    text:ord.map(d=>'$'+Math.round(d.top_avg_price).toLocaleString()),
    textposition:'outside',
    textfont:{family:"'JetBrains Mono',monospace",size:11,color:'#e2e2f5'},
    customdata:ord.map(d=>d.top_priced_neighbourhood),
    hovertemplate:'<b>%{y}</b><br>%{customdata}<br>$%{x:,.0f} avg<extra></extra>',
  }],{...BASE,
    margin:{t:8,r:70,b:40,l:110},
    xaxis:{...BASE.xaxis,title:{text:'Avg price ($)',font:{size:10,color:'#6060a0'}}},
  },CFG);
}

// ── MAP BUTTONS ───────────────────────────────────────────────────────
function initMapButtons(){
  const el=document.getElementById('mapButtons');
  CO.forEach(city=>{
    const b=document.createElement('button');
    b.className='map-tog'+(city===selectedCity?' active':'');
    b.dataset.city=city;
    b.textContent=CL[city];
    b.addEventListener('click',()=>selectCity(city));
    el.appendChild(b);
  });
}

function loadMaps(city){
  document.getElementById('loadC').style.display='flex';
  document.getElementById('loadK').style.display='flex';
  document.getElementById('iC').src=`/map/${city}/choropleth`;
  document.getElementById('iK').src=`/map/${city}/cluster`;
  document.querySelectorAll('.map-tog').forEach(b=>{
    b.classList.toggle('active',b.dataset.city===city);
  });
}

// ── CALENDAR ─────────────────────────────────────────────────────────
function buildCalendarSection(){
  const monthly=DATA.calendar_monthly||[];
  const traces=CO.map(city=>{
    const rows=monthly.filter(d=>d.city===city).sort((a,b)=>a.month_num-b.month_num);
    return {
      type:'scatter',mode:'lines+markers',name:CL[city],
      x:rows.map(d=>d.month),y:rows.map(d=>(d.availability_rate*100).toFixed(1)),
      line:{color:CC[city],width:2.5},marker:{color:CC[city],size:7},
      hovertemplate:`<b>${CL[city]}</b><br>%{x}: %{y}%<extra></extra>`,
    };
  });
  Plotly.newPlot('chartCal',traces,{...BASE,
    margin:{t:8,r:8,b:50,l:55},
    yaxis:{...BASE.yaxis,range:[0,100],title:{text:'Availability (%)',font:{size:10,color:'#6060a0'}}},
    xaxis:{...BASE.xaxis},
    legend:{font:{size:11,color:'#6060a0'},bgcolor:'rgba(0,0,0,0)',orientation:'h',y:-0.2,x:0},
    showlegend:true,
  },CFG);

  // toggle buttons
  const tog=document.getElementById('calToggles');
  CO.forEach((city,i)=>{
    const btn=document.createElement('button');
    btn.style.cssText=`font-family:var(--fm);font-size:9px;letter-spacing:.08em;
      text-transform:uppercase;padding:5px 12px;border-radius:100px;cursor:pointer;
      border:1px solid ${CC[city]}60;background:${CC[city]}1a;color:${CC[city]};transition:.2s`;
    btn.textContent=CL[city];
    btn.onclick=()=>{
      const vis=document.getElementById('chartCal').data[i].visible;
      Plotly.restyle('chartCal',{visible:vis===true?'legendonly':true},[i]);
    };
    tog.appendChild(btn);
  });

  // summary table
  const cal=DATA.calendar||[];
  const tbl=document.getElementById('calTable');
  tbl.innerHTML=`<thead style="font-family:var(--fm);font-size:9px;letter-spacing:.12em;color:var(--muted)">
    <tr>${['City','Avg Price','Median','Avail %','Listings'].map(h=>`<th style="padding:8px 12px;text-align:left;border-bottom:1px solid var(--b2);text-transform:uppercase">${h}</th>`).join('')}</tr>
  </thead><tbody>${CO.map(city=>{
    const d=cal.find(r=>r.city===city);
    if(!d) return '';
    return `<tr style="border-bottom:1px solid var(--b1);cursor:pointer" onclick="selectCity('${city}')">
      <td style="padding:9px 12px;color:${CC[city]};font-family:var(--fm);font-size:11px">${CL[city]}</td>
      <td style="padding:9px 12px;font-family:var(--fm);font-size:11px">$${d.avg_price}</td>
      <td style="padding:9px 12px;font-family:var(--fm);font-size:11px">$${d.median_price}</td>
      <td style="padding:9px 12px;font-family:var(--fm);font-size:11px">${(d.availability_rate*100).toFixed(1)}%</td>
      <td style="padding:9px 12px;font-family:var(--fm);font-size:11px">${Number(d.unique_listings).toLocaleString()}</td>
    </tr>`;
  }).join('')}</tbody>`;

  // avg price bar
  const ord=CO.map(c=>cal.find(d=>d.city===c)).filter(Boolean);
  Plotly.newPlot('chartAvgPrice',[{
    type:'bar',
    x:ord.map(d=>CL[d.city]),y:ord.map(d=>d.avg_price),
    marker:{color:ord.map(d=>CC[d.city]),opacity:.85},
    text:ord.map(d=>'$'+d.avg_price),textposition:'outside',
    textfont:{family:"'JetBrains Mono',monospace",size:11,color:'#e2e2f5'},
    hovertemplate:'<b>%{x}</b><br>Avg: $%{y}<extra></extra>',
  }],{...BASE,
    margin:{t:8,r:8,b:55,l:50},
    yaxis:{...BASE.yaxis,title:{text:'USD/night',font:{size:10,color:'#6060a0'}}},
    xaxis:{...BASE.xaxis,tickangle:-20},
  },CFG);
}

// ── STATS SECTION ─────────────────────────────────────────────────────
function buildStatsSection(){
  buildSigMatrix();
  buildPvalsChart();
}

function buildSigMatrix(){
  const tests=DATA.cross_tests||[];
  const pairs=tests.filter(d=>d.test&&d.test.includes('Mann-Whitney')&&d.test.includes('city pair'));
  const el=document.getElementById('sigMatrix');

  // build lookup
  const lookup={};
  pairs.forEach(d=>{
    const key=`${d.group_1}_${d.group_2}`;
    lookup[key]=lookup[`${d.group_2}_${d.group_1}`]={sig:d.significant_0_05_bh,p:d.p_value_bh,stat:d.statistic};
  });

  const tt=document.getElementById('tooltip');
  const ttTitle=document.getElementById('tt-title');
  const ttBody=document.getElementById('tt-body');

  const size=38,gap=3,labelH=64;
  const n=CO.length;
  const total=(size+gap)*n+labelH;

  let html=`<div style="display:inline-grid;grid-template-columns:${labelH}px ${Array(n).fill(size+'px').join(' ')};
    grid-template-rows:${labelH}px ${Array(n).fill(size+'px').join(' ')};gap:${gap}px">`;

  // empty corner
  html+=`<div></div>`;
  // column headers
  CO.forEach(c=>{
    html+=`<div style="display:flex;align-items:flex-end;justify-content:center;padding-bottom:6px;
      font-family:var(--fm);font-size:9px;letter-spacing:.06em;color:${CC[c]};
      writing-mode:vertical-rl;text-orientation:mixed;white-space:nowrap">${CL[c]}</div>`;
  });
  // rows
  CO.forEach((r,ri)=>{
    html+=`<div style="display:flex;align-items:center;justify-content:flex-end;padding-right:8px;
      font-family:var(--fm);font-size:9px;letter-spacing:.06em;color:${CC[r]};
      white-space:nowrap">${CL[r]}</div>`;
    CO.forEach((c,ci)=>{
      if(ri===ci){
        html+=`<div style="width:${size}px;height:${size}px;background:${CC[r]}22;
          border-radius:4px;border:1px solid ${CC[r]}44"></div>`;
      } else {
        const info=lookup[`${r}_${c}`]||lookup[`${c}_${r}`];
        const sig=info?.sig;
        const logp=info?.p?-Math.log10(Math.max(info.p,1e-300)):0;
        const intensity=Math.min(logp/20,1);
        const bg=sig?`rgba(74,222,128,${.15+intensity*.4})`:`rgba(40,40,70,.5)`;
        const border=sig?`rgba(74,222,128,${.3+intensity*.4})`:`var(--b1)`;
        const label=sig?'✓':'';
        html+=`<div style="width:${size}px;height:${size}px;background:${bg};
          border-radius:4px;border:1px solid ${border};cursor:pointer;
          display:flex;align-items:center;justify-content:center;
          font-family:var(--fm);font-size:11px;color:rgba(74,222,128,.9);
          transition:transform .15s,box-shadow .15s"
          onmouseenter="showTT(event,'${CL[r]} vs ${CL[c]}',${info?info.p:1},${info?info.stat:0},${!!sig})"
          onmouseleave="hideTT()"
          onclick="selectCity('${r}')">${label}</div>`;
      }
    });
  });
  html+='</div>';
  el.innerHTML=html;
}

function showTT(e,title,p,stat,sig){
  const tt=document.getElementById('tooltip');
  document.getElementById('tt-title').textContent=title;
  document.getElementById('tt-body').innerHTML=`
    <div class="tt-row"><span>Significant</span><span class="tt-val" style="color:${sig?'#4ade80':'#6060a0'}">${sig?'Yes ✓':'No'}</span></div>
    <div class="tt-row"><span>p-value (BH)</span><span class="tt-val">${Number(p).toExponential(2)}</span></div>
    <div class="tt-row"><span>U statistic</span><span class="tt-val">${Number(stat).toLocaleString()}</span></div>`;
  tt.style.left=(e.clientX+14)+'px';
  tt.style.top=(e.clientY-10)+'px';
  tt.classList.add('show');
}
function hideTT(){document.getElementById('tooltip').classList.remove('show');}

function buildPvalsChart(){
  const tests=DATA.cross_tests||[];
  const pairs=tests.filter(d=>d.test&&d.test.includes('Mann-Whitney')&&d.test.includes('city pair'))
    .sort((a,b)=>a.p_value_bh-b.p_value_bh);
  Plotly.newPlot('chartPvals',[{
    type:'bar',orientation:'h',
    y:pairs.map(d=>`${CL[d.group_1]||d.group_1} vs ${CL[d.group_2]||d.group_2}`),
    x:pairs.map(d=>-Math.log10(Math.max(d.p_value_bh,1e-300))),
    marker:{color:pairs.map(d=>CC[d.group_1]||'#666'),opacity:.82},
    hovertemplate:'<b>%{y}</b><br>−log₁₀(p) = %{x:.1f}<extra></extra>',
  }],{...BASE,
    margin:{t:8,r:20,b:40,l:210},
    xaxis:{...BASE.xaxis,title:{text:'−log₁₀(p)',font:{size:10,color:'#6060a0'}}},
    yaxis:{...BASE.yaxis,tickfont:{size:10,color:'#6060a0'}},
    shapes:[{type:'line',x0:-Math.log10(.05),x1:-Math.log10(.05),y0:-0.5,y1:pairs.length-.5,
      line:{color:'#FF5A5F',width:1.5,dash:'dot'}}],
    annotations:[{x:-Math.log10(.05),y:pairs.length-.2,text:'α=0.05',
      font:{size:9,color:'#FF5A5F',family:"'JetBrains Mono',monospace"},
      showarrow:false,xanchor:'left',xshift:4}],
  },CFG);
}

