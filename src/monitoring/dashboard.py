"""
src/monitoring/dashboard.py
-----------------------------
FastAPI monitoring dashboard server.
Exposes metrics endpoints and serves the dashboard UI.

Run separately from the main RAG app:
    python -m src.monitoring.dashboard

Open: http://localhost:8002
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import asdict

from src.monitoring.metrics import get_collector

app = FastAPI(title="RAG Monitoring Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/metrics/summary")
def metrics_summary(window_hours: float = 24.0):
    """Get aggregated metrics summary."""
    collector = get_collector()
    summary   = collector.summary(window_hours=window_hours)
    return asdict(summary)


@app.get("/metrics/recent")
def recent_queries(n: int = 20):
    """Get recent query metrics."""
    collector = get_collector()
    queries   = collector.recent_queries(n=n)
    return [asdict(q) for q in queries]


@app.get("/metrics/regression-check")
def regression_check(
    p95_threshold: float = 5.0,
    faithfulness_threshold: float = 0.80,
    error_rate_threshold: float = 0.05,
):
    """Run regression checks against thresholds."""
    collector = get_collector()
    return collector.check_regression(
        p95_threshold=p95_threshold,
        faithfulness_threshold=faithfulness_threshold,
        error_rate_threshold=error_rate_threshold,
    )


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the monitoring dashboard UI."""
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Monitoring Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&family=DM+Mono&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--navy:#0D1B2A;--teal:#0D9488;--teal-lt:#5EEAD4;--purple:#7C3AED;--amber:#F59E0B;--coral:#F97316;--white:#FFFFFF;--g1:#F0F4F8;--g2:#CBD5E1;--g3:#64748B;--g4:#1E293B;--success:#10B981;--danger:#EF4444;--r:10px}
body{font-family:'DM Sans',sans-serif;background:var(--navy);color:var(--white);min-height:100vh}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;background:radial-gradient(ellipse 60% 40% at 85% 5%,rgba(124,58,237,.12) 0%,transparent 60%),radial-gradient(ellipse 50% 50% at 5% 90%,rgba(13,148,136,.08) 0%,transparent 60%)}
.app{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:24px}
.header{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px}
.logo{font-family:'Syne',sans-serif;font-size:24px;font-weight:800;display:flex;align-items:center;gap:10px}
.logo-dot{width:10px;height:10px;background:var(--teal);border-radius:50%;box-shadow:0 0 12px var(--teal);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.7;transform:scale(1.2)}}
.window-btns{display:flex;gap:6px}
.win-btn{padding:6px 14px;border-radius:20px;font-size:12px;font-weight:500;cursor:pointer;border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.05);color:var(--g2);transition:all .2s}
.win-btn.active{border-color:var(--teal);color:var(--teal-lt);background:rgba(13,148,136,.12)}
.metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
.metric-card{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:var(--r);padding:20px;position:relative;overflow:hidden}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.metric-card.teal::before{background:var(--teal)}
.metric-card.purple::before{background:var(--purple)}
.metric-card.amber::before{background:var(--amber)}
.metric-card.coral::before{background:var(--coral)}
.metric-card.success::before{background:var(--success)}
.metric-val{font-size:32px;font-weight:700;font-family:'DM Mono',monospace;margin:8px 0 4px}
.metric-lbl{font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--g3)}
.metric-sub{font-size:12px;color:var(--g3);margin-top:4px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}
.card{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:var(--r);padding:20px}
.card-title{font-family:'Syne',sans-serif;font-size:16px;font-weight:700;margin-bottom:16px;color:var(--g2)}
.latency-bars{display:flex;flex-direction:column;gap:10px}
.lat-row{display:flex;align-items:center;gap:12px}
.lat-lbl{font-size:12px;color:var(--g3);width:40px;flex-shrink:0;font-family:'DM Mono',monospace}
.lat-bar-bg{flex:1;height:8px;background:rgba(255,255,255,.06);border-radius:4px;overflow:hidden}
.lat-bar{height:100%;border-radius:4px;transition:width .5s ease}
.lat-val{font-size:12px;font-family:'DM Mono',monospace;color:var(--teal-lt);width:50px;text-align:right;flex-shrink:0}
.quality-row{display:flex;align-items:center;justify-content:space-between;padding:12px 0;border-bottom:1px solid rgba(255,255,255,.05)}
.quality-row:last-child{border-bottom:none}
.quality-name{font-size:13px;color:var(--g2)}
.quality-score{font-size:20px;font-weight:700;font-family:'DM Mono',monospace}
.quality-bar-bg{width:100%;height:6px;background:rgba(255,255,255,.06);border-radius:3px;margin-top:6px;overflow:hidden}
.quality-bar{height:100%;border-radius:3px;transition:width .5s}
.regression-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px}
.check-card{background:rgba(255,255,255,.04);border:1px solid;border-radius:var(--r);padding:16px}
.check-card.pass{border-color:rgba(16,185,129,.3);background:rgba(16,185,129,.05)}
.check-card.fail{border-color:rgba(239,68,68,.3);background:rgba(239,68,68,.05)}
.check-status{font-size:18px;margin-bottom:6px}
.check-name{font-size:12px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--g3);margin-bottom:8px}
.check-val{font-size:24px;font-weight:700;font-family:'DM Mono',monospace}
.check-threshold{font-size:11px;color:var(--g3);margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:8px 12px;font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--g3);border-bottom:1px solid rgba(255,255,255,.08)}
td{padding:10px 12px;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:top}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(255,255,255,.02)}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600}
.badge-ok{background:rgba(16,185,129,.15);color:var(--success)}
.badge-warn{background:rgba(245,158,11,.15);color:var(--amber)}
.badge-err{background:rgba(239,68,68,.15);color:var(--danger)}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="logo"><div class="logo-dot"></div>RAG Monitoring</div>
    <div class="window-btns">
      <button class="win-btn active" onclick="setWindow(1)">1h</button>
      <button class="win-btn" onclick="setWindow(6)">6h</button>
      <button class="win-btn" onclick="setWindow(24)">24h</button>
      <button class="win-btn" onclick="setWindow(0)">All time</button>
    </div>
  </div>

  <!-- Top metrics -->
  <div class="metrics-grid">
    <div class="metric-card teal">
      <div class="metric-lbl">Total queries</div>
      <div class="metric-val" id="totalQueries">—</div>
      <div class="metric-sub" id="errorRate">Error rate: —</div>
    </div>
    <div class="metric-card purple">
      <div class="metric-lbl">Avg latency</div>
      <div class="metric-val" id="avgLatency">—</div>
      <div class="metric-sub">seconds per query</div>
    </div>
    <div class="metric-card amber">
      <div class="metric-lbl">Total cost</div>
      <div class="metric-val" id="totalCost">—</div>
      <div class="metric-sub" id="avgCost">Avg per query: —</div>
    </div>
    <div class="metric-card success">
      <div class="metric-lbl">Avg faithfulness</div>
      <div class="metric-val" id="faithfulness">—</div>
      <div class="metric-sub" id="relevancy">Relevancy: —</div>
    </div>
  </div>

  <!-- Regression checks -->
  <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;margin-bottom:12px;color:var(--g2)">CI Regression Gate</div>
  <div class="regression-grid" id="regressionGrid">
    <div class="check-card"><div class="check-name">Loading...</div></div>
  </div>

  <!-- Latency + Quality -->
  <div class="grid2">
    <div class="card">
      <div class="card-title">Latency Percentiles</div>
      <div class="latency-bars" id="latencyBars">
        <div style="color:var(--g3);font-size:13px">No data yet</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Quality Scores</div>
      <div id="qualityScores">
        <div style="color:var(--g3);font-size:13px">No quality scores yet — run eval to populate</div>
      </div>
    </div>
  </div>

  <!-- Recent queries -->
  <div class="card" style="margin-top:16px">
    <div class="card-title">Recent Queries</div>
    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>Time</th><th>Query</th><th>Latency</th>
          <th>Tokens</th><th>Cost</th><th>Status</th>
        </tr></thead>
        <tbody id="recentTable"><tr><td colspan="6" style="color:var(--g3)">Loading...</td></tr></tbody>
      </table>
    </div>
  </div>
</div>

<script>
let currentWindow = 1;

function setWindow(h) {
  currentWindow = h;
  document.querySelectorAll('.win-btn').forEach((b,i) => {
    b.classList.toggle('active', [1,6,24,0][i] === h);
  });
  loadAll();
}

async function loadAll() {
  await Promise.all([loadSummary(), loadRegression(), loadRecent()]);
}

async function loadSummary() {
  try {
    const res  = await fetch(`/metrics/summary?window_hours=${currentWindow}`);
    const data = await res.json();

    document.getElementById('totalQueries').textContent = data.total_queries;
    document.getElementById('errorRate').textContent    = `Error rate: ${(data.error_rate*100).toFixed(1)}%`;
    document.getElementById('avgLatency').textContent   = data.avg_latency + 's';
    document.getElementById('totalCost').textContent    = '$' + data.total_cost_usd.toFixed(4);
    document.getElementById('avgCost').textContent      = `Avg per query: $${data.avg_cost_usd.toFixed(6)}`;
    document.getElementById('faithfulness').textContent = data.avg_faithfulness != null ? data.avg_faithfulness : 'N/A';
    document.getElementById('relevancy').textContent    = `Relevancy: ${data.avg_answer_relevancy != null ? data.avg_answer_relevancy : 'N/A'}`;

    // Latency bars
    const maxLat = Math.max(data.p99_latency, 1);
    document.getElementById('latencyBars').innerHTML = [
      ['p50', data.p50_latency, 'var(--success)'],
      ['p95', data.p95_latency, 'var(--amber)'],
      ['p99', data.p99_latency, 'var(--coral)'],
      ['avg', data.avg_latency, 'var(--teal)'],
    ].map(([lbl, val, color]) => `
      <div class="lat-row">
        <div class="lat-lbl">${lbl}</div>
        <div class="lat-bar-bg">
          <div class="lat-bar" style="width:${Math.min(100,(val/maxLat)*100)}%;background:${color}"></div>
        </div>
        <div class="lat-val">${val}s</div>
      </div>`).join('');

    // Quality scores
    if (data.avg_faithfulness != null) {
      document.getElementById('qualityScores').innerHTML = [
        ['Faithfulness',    data.avg_faithfulness,     'No hallucination score', 'var(--success)'],
        ['Answer Relevancy',data.avg_answer_relevancy,  'Addresses the question', 'var(--teal)'],
      ].map(([name, val, desc, color]) => `
        <div class="quality-row">
          <div>
            <div class="quality-name">${name}</div>
            <div style="font-size:11px;color:var(--g3)">${desc}</div>
            <div class="quality-bar-bg">
              <div class="quality-bar" style="width:${(val||0)*100}%;background:${color}"></div>
            </div>
          </div>
          <div class="quality-score" style="color:${color}">${val != null ? val.toFixed(2) : 'N/A'}</div>
        </div>`).join('');
    }
  } catch(e) { console.error(e); }
}

async function loadRegression() {
  try {
    const res  = await fetch('/metrics/regression-check');
    const data = await res.json();
    const checks = [
      ['p95_latency',  'P95 Latency',    's'],
      ['faithfulness', 'Faithfulness',   ''],
      ['error_rate',   'Error Rate',     ''],
    ];
    document.getElementById('regressionGrid').innerHTML = checks.map(([key, label, unit]) => {
      const c   = data[key];
      const cls = c.passed ? 'pass' : 'fail';
      const ico = c.passed ? '✅' : '❌';
      const val = c.value != null ? (typeof c.value === 'number' ? c.value.toFixed(3) : c.value) : 'N/A';
      return `<div class="check-card ${cls}">
        <div class="check-status">${ico}</div>
        <div class="check-name">${label}</div>
        <div class="check-val">${val}${unit}</div>
        <div class="check-threshold">Threshold: ${c.threshold}${unit}</div>
      </div>`;
    }).join('');
  } catch(e) { console.error(e); }
}

async function loadRecent() {
  try {
    const res  = await fetch('/metrics/recent?n=15');
    const data = await res.json();
    if (!data.length) {
      document.getElementById('recentTable').innerHTML =
        '<tr><td colspan="6" style="color:var(--g3)">No queries yet</td></tr>';
      return;
    }
    document.getElementById('recentTable').innerHTML = data.map(q => {
      const t   = new Date(q.timestamp * 1000).toLocaleTimeString();
      const lat = q.total_latency.toFixed(2) + 's';
      const tok = (q.input_tokens + q.output_tokens) || '—';
      const cost= '$' + (q.cost_usd || 0).toFixed(6);
      const cls = q.error ? 'badge-err' : q.total_latency > 5 ? 'badge-warn' : 'badge-ok';
      const status = q.error ? 'Error' : q.total_latency > 5 ? 'Slow' : 'OK';
      const qshort = q.query.length > 50 ? q.query.substring(0,50)+'...' : q.query;
      return `<tr>
        <td style="font-family:'DM Mono',monospace;color:var(--g3);white-space:nowrap">${t}</td>
        <td style="color:var(--g1)">${qshort}</td>
        <td style="font-family:'DM Mono',monospace;color:var(--amber)">${lat}</td>
        <td style="font-family:'DM Mono',monospace;color:var(--g2)">${tok}</td>
        <td style="font-family:'DM Mono',monospace;color:var(--teal-lt)">${cost}</td>
        <td><span class="badge ${cls}">${status}</span></td>
      </tr>`;
    }).join('');
  } catch(e) { console.error(e); }
}

loadAll();
setInterval(loadAll, 10000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    print("\n📊 RAG Monitoring Dashboard starting...")
    print("   Open your browser at: http://localhost:8002\n")
    uvicorn.run("src.monitoring.dashboard:app", host="0.0.0.0", port=8002, reload=True)
