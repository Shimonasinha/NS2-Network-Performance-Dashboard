# dashboard.py - Complete Streamlit Dashboard
# NS2 TCP Performance + ML Recommendation + Network Simulation

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import pickle
import os
import subprocess
import json
from sklearn.preprocessing import LabelEncoder

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="NS2 TCP Performance Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .recommend-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────
DATA_DIR  = "."
PROTOCOLS = ["reno", "tahoe", "vegas"]
METRICS   = ["throughput", "latency", "drop", "cwnd"]
COLORS    = {
    "reno":"#e74c3c","tahoe":"#3498db","vegas":"#2ecc71",
    "cubic":"#f39c12","bbr":"#9b59b6","dctcp":"#1abc9c"
}
FEATURES = [
    "throughput_mbps","mean_rtt_ms","p90_rtt_ms","p99_rtt_ms",
    "std_rtt_ms","loss_rate","retx_rate",
    "flow_count","network_load","app_type_enc"
]

# ── Helper functions ───────────────────────────────────────────────
@st.cache_data
def read_xg_file(filepath):
    times, values = [], []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2: continue
                try:
                    t, v = float(parts[0]), float(parts[1])
                    if t == 0 and v == 0: continue
                    times.append(t); values.append(v)
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return pd.DataFrame({"Time": times, "Value": values})

@st.cache_resource
def load_model():
    try:
        with open("tcp_model.pkl", "rb") as f:
            data = pickle.load(f)
        return data["model"], data["le_label"], data["le_app"]
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("tcp_dataset.csv")
    except FileNotFoundError:
        return None

# ── HEADER ─────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#2c3e50,#3498db);
            padding:25px;border-radius:15px;text-align:center;
            margin-bottom:20px;color:white;'>
    <h1 style='margin:0;font-size:2em;'>📡 NS2 TCP Performance Dashboard</h1>
    <p style='margin:5px 0 0 0;opacity:0.9;'>
        TCP Congestion Control Analysis · ML Recommendation · Network Simulation
    </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Metrics & Graphs",
    "🤖 ML TCP Recommender",
    "📈 Model Performance",
    "🖥️ Network Simulation"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Metrics & Graphs
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📊 TCP Protocol Metrics")

    col1, col2 = st.columns([1, 3])
    with col1:
        selected = st.selectbox("Select Protocol",
                                ["Reno","Tahoe","Vegas"],
                                key="proto_select")
        proto = selected.lower()

        st.markdown("### 📋 Summary")
        for m in METRICS:
            fp  = os.path.join(DATA_DIR, f"{m}-{proto}.xg")
            df  = read_xg_file(fp)
            avg = round(df["Value"].mean(), 2) if not df.empty else 0
            mx  = round(df["Value"].max(),  2) if not df.empty else 0
            st.metric(label=m.capitalize(), value=f"Avg: {avg}", delta=f"Max: {mx}")

    with col2:
        st.markdown(f"### {selected} — Individual Metrics")
        cols = st.columns(2)
        for i, m in enumerate(METRICS):
            fp = os.path.join(DATA_DIR, f"{m}-{proto}.xg")
            df = read_xg_file(fp)
            with cols[i % 2]:
                if not df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["Time"], y=df["Value"],
                        mode="lines", name=m,
                        line={"color": COLORS[proto], "width": 2},
                        fill="tozeroy",
                        fillcolor=COLORS[proto].replace(")", ",0.1)").replace("rgb","rgba")
                    ))
                    fig.update_layout(
                        title=f"{m.capitalize()} - TCP {selected}",
                        xaxis_title="Time (s)",
                        yaxis_title=m.capitalize(),
                        height=250,
                        margin=dict(l=40,r=20,t=40,b=40),
                        paper_bgcolor="white",
                        plot_bgcolor="#f8f9fa"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data for {m}-{proto}.xg")

    st.markdown("---")
    st.markdown("### 📊 Protocol Comparison (Reno vs Tahoe vs Vegas)")
    cols2 = st.columns(2)
    for i, m in enumerate(METRICS):
        with cols2[i % 2]:
            fig = go.Figure()
            for p in PROTOCOLS:
                fp = os.path.join(DATA_DIR, f"{m}-{p}.xg")
                df = read_xg_file(fp)
                if not df.empty:
                    fig.add_trace(go.Scatter(
                        x=df["Time"], y=df["Value"],
                        mode="lines", name=p.capitalize(),
                        line={"color": COLORS[p], "width": 2}
                    ))
            fig.update_layout(
                title=f"{m.capitalize()} Comparison",
                xaxis_title="Time (s)",
                yaxis_title=m.capitalize(),
                height=280,
                margin=dict(l=40,r=20,t=40,b=40),
                paper_bgcolor="white",
                plot_bgcolor="#f8f9fa",
                legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — ML TCP Recommender
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🤖 Real-Time TCP Variant Recommender")
    st.markdown("*Enter your current network conditions to get the best TCP variant recommendation*")

    model, le_label, le_app = load_model()

    if model is None:
        st.error("❌ Model not found! Run `python3 ml_recommender.py` first.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🌐 Network Conditions")
            throughput   = st.slider("Throughput (Mbps)",  0.1, 15.0, 6.5, 0.1)
            mean_rtt     = st.slider("Mean RTT (ms)",       0.1, 300.0, 50.0, 0.5)
            p90_rtt      = st.slider("P90 RTT (ms)",        0.1, 350.0, 65.0, 0.5)
            p99_rtt      = st.slider("P99 RTT (ms)",        0.1, 400.0, 75.0, 0.5)
            std_rtt      = st.slider("RTT Std Dev (ms)",    0.0, 50.0,  8.0,  0.1)
            loss_rate    = st.slider("Loss Rate",           0.0, 0.3,   0.02, 0.001,
                                     format="%.3f")
            retx_rate    = st.slider("Retransmission Rate", 0.0, 0.25,  0.016,0.001,
                                     format="%.3f")
            flow_count   = st.slider("Flow Count",          1,   50,    20)
            network_load = st.slider("Network Load",        0.1, 1.0,   0.5,  0.05)
            app_type     = st.selectbox("Application Type",
                                        ["streaming","io","sort"])

        with col2:
            st.markdown("### 🎯 ML Recommendation")

            app_enc = le_app.transform([app_type])[0]
            X = np.array([[throughput, mean_rtt, p90_rtt, p99_rtt,
                           std_rtt, loss_rate, retx_rate,
                           flow_count, network_load, app_enc]])

            pred  = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            rec   = le_label.inverse_transform([pred])[0]

            # Recommendation box
            st.markdown(f"""
            <div class='recommend-box'>
                🎯 Recommended TCP: {rec.upper()}
            </div>
            """, unsafe_allow_html=True)

            # Confidence bars
            st.markdown("### 📊 Confidence Scores")
            sorted_results = sorted(
                zip(le_label.classes_, proba),
                key=lambda x: x[1], reverse=True
            )
            for cls, p in sorted_results:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(float(p), text=f"{cls.upper()}")
                with col_b:
                    st.markdown(f"**{p*100:.1f}%**")

            # Why this TCP?
            st.markdown("### 💡 Why this recommendation?")
            reasons = {
                "bbr":   "🚀 High throughput + Low RTT → BBR excels at bandwidth utilization",
                "cubic": "📈 Good throughput + Moderate RTT → Cubic is efficient for standard networks",
                "dctcp": "🏢 Very low RTT → DCTCP is optimized for datacenter environments",
                "reno":  "⚖️ Moderate conditions → Reno provides balanced performance",
                "tahoe": "🛡️ High loss rate → Tahoe handles packet loss conservatively",
                "vegas": "⚡ Low RTT + Low loss → Vegas proactively avoids congestion",
            }
            st.info(reasons.get(rec, "Based on current network conditions"))

            # Input summary
            st.markdown("### 📋 Input Summary")
            summary_df = pd.DataFrame({
                "Parameter": ["Throughput","Mean RTT","Loss Rate",
                              "Retx Rate","Flow Count","App Type","Network Load"],
                "Value":     [f"{throughput} Mbps", f"{mean_rtt} ms",
                              f"{loss_rate:.3f}", f"{retx_rate:.3f}",
                              str(flow_count), app_type, f"{network_load:.2f}"]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📈 ML Model Performance")

    df_data    = load_dataset()
    ml_results = None
    if os.path.exists("ml_results.json"):
        with open("ml_results.json","r") as f:
            ml_results = json.load(f)

    if ml_results is None:
        st.error("❌ ml_results.json not found! Run `python3 ml_recommender.py` first.")
    else:
        # ── Top metrics ──────────────────────────────────────────
        best_model_name = max(ml_results["models"],
                              key=lambda n: ml_results["models"][n]["accuracy"])
        best_acc = ml_results["models"][best_model_name]["accuracy"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples",
                    f"{ml_results['dataset']['total_samples']:,}")
        col2.metric("TCP Variants",
                    str(ml_results["dataset"]["variants"]))
        col3.metric("Best Accuracy",
                    f"{best_acc:.2f}% ({best_model_name})")

        st.markdown("---")

        # ── Dataset charts ────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Dataset Distribution")
            label_counts = ml_results["dataset"]["label_counts"]
            counts = pd.DataFrame({
                "Protocol": list(label_counts.keys()),
                "Count":    list(label_counts.values())
            })
            fig = px.bar(counts, x="Protocol", y="Count",
                         color="Protocol",
                         color_discrete_map=COLORS,
                         title="Samples per TCP Variant")
            fig.update_layout(height=350, showlegend=False,
                              paper_bgcolor="white", plot_bgcolor="#f8f9fa")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🍕 Data Source Breakdown")
            src_counts = ml_results["dataset"]["source_counts"]
            src = pd.DataFrame({
                "Source": list(src_counts.keys()),
                "Count":  list(src_counts.values())
            })
            fig2 = px.pie(src, values="Count", names="Source",
                          title="Data Sources",
                          color_discrete_sequence=["#3498db","#e74c3c","#2ecc71"])
            fig2.update_layout(height=350, paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

        # ── Model accuracy comparison ─────────────────────────────
        st.markdown("### 📉 Model Accuracy Comparison")
        model_names = list(ml_results["models"].keys())
        model_accs  = [ml_results["models"][n]["accuracy"] for n in model_names]

        col1, col2 = st.columns(2)
        with col1:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                name="Test Accuracy",
                x=model_names, y=model_accs,
                marker_color="#2980b9",
                text=[f"{a:.1f}%" for a in model_accs],
                textposition="outside"
            ))
            fig3.update_layout(
                title="Model Accuracy Comparison",
                yaxis_title="Accuracy (%)",
                yaxis_range=[0,115],
                height=350,
                paper_bgcolor="white", plot_bgcolor="#f8f9fa"
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            st.markdown(f"### 🏆 Per-Class Accuracy ({best_model_name})")
            per_class = ml_results["models"][best_model_name]["per_class"]
            class_acc = pd.DataFrame({
                "Class":    [c.upper() for c in per_class.keys()],
                "Accuracy": list(per_class.values())
            })
            fig4 = px.bar(class_acc, x="Class", y="Accuracy",
                          color="Class",
                          color_discrete_sequence=list(COLORS.values()),
                          title=f"Per-Class Accuracy % ({best_model_name})",
                          text="Accuracy")
            fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig4.update_layout(height=350, showlegend=False,
                               paper_bgcolor="white", plot_bgcolor="#f8f9fa",
                               yaxis_range=[0,120])
            st.plotly_chart(fig4, use_container_width=True)

        # ── Scatter plot ──────────────────────────────────────────
        if df_data is not None:
            st.markdown("### 🔍 Throughput vs Latency by Protocol")
            fig5 = px.scatter(df_data, x="throughput_mbps", y="mean_rtt_ms",
                              color="best_protocol",
                              color_discrete_map=COLORS,
                              opacity=0.4,
                              title="Throughput vs RTT — All Variants",
                              labels={"throughput_mbps":"Throughput (Mbps)",
                                      "mean_rtt_ms":"Mean RTT (ms)",
                                      "best_protocol":"Protocol"})
            fig5.update_layout(height=400, paper_bgcolor="white",
                               plot_bgcolor="#f8f9fa")
            st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — Network Simulation
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🖥️ 100-Node TCP Network Simulation")
    st.markdown("*Animated visualization of 100-node NS2 network topology*")

    SIM_HTML = """
    <style>
      body{margin:0;background:#0d1117;}
      #sw{background:#0d1117;border-radius:10px;padding:12px;
          font-family:'Segoe UI',sans-serif;color:#fff;}
      #sw h3{text-align:center;color:#58a6ff;margin:0 0 8px 0;font-size:1.1em;}
      #sc{display:flex;justify-content:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;align-items:center;}
      .sb{padding:6px 16px;border:none;border-radius:6px;cursor:pointer;font-weight:bold;font-size:0.82em;}
      #s1{background:#238636;color:#fff;}#s2{background:#b08800;color:#fff;}#s3{background:#da3633;color:#fff;}
      .pb{background:#21262d;color:#8b949e;border:1px solid #30363d;
          padding:4px 12px;border-radius:20px;cursor:pointer;font-size:0.8em;}
      .pb.on{border-color:#58a6ff;color:#58a6ff;background:#0d2044;}
      #ss{display:flex;justify-content:center;gap:20px;margin-bottom:6px;flex-wrap:wrap;}
      .sv{text-align:center;}.sv .v{font-size:1.1em;font-weight:bold;color:#58a6ff;}
      .sv .l{font-size:0.7em;color:#8b949e;}
      canvas{display:block;margin:0 auto;border-radius:8px;}
      #sl{display:flex;justify-content:center;gap:14px;margin-top:6px;
          font-size:0.75em;color:#8b949e;flex-wrap:wrap;}
      .ll{display:flex;align-items:center;gap:4px;}
      .ld{width:10px;height:10px;border-radius:50%;}
    </style>
    <div id="sw">
      <h3>📡 100-Node TCP Network — NS2 Simulation Visualization</h3>
      <div id="sc">
        <button class="sb" id="s1">▶ Start</button>
        <button class="sb" id="s2">⏸ Pause</button>
        <button class="sb" id="s3">↺ Reset</button>
        <button class="pb on" data-p="reno">TCP Reno</button>
        <button class="pb" data-p="tahoe">TCP Tahoe</button>
        <button class="pb" data-p="vegas">TCP Vegas</button>
        <button class="pb" data-p="bbr">TCP BBR</button>
        <button class="pb" data-p="cubic">TCP Cubic</button>
        <span style="color:#8b949e;font-size:0.8em;">Speed</span>
        <input type="range" id="spd" min="1" max="5" value="2"
               style="width:80px;accent-color:#58a6ff;">
      </div>
      <div id="ss">
        <div class="sv"><div class="v" id="t">0.0s</div><div class="l">Time</div></div>
        <div class="sv"><div class="v" id="ps">0</div><div class="l">Sent</div></div>
        <div class="sv"><div class="v" id="pr">0</div><div class="l">Received</div></div>
        <div class="sv"><div class="v" id="pd">0</div><div class="l">Dropped</div></div>
        <div class="sv"><div class="v" id="tp">0.000</div><div class="l">Mbps</div></div>
      </div>
      <canvas id="c"></canvas>
      <div id="sl">
        <div class="ll"><div class="ld" style="background:#1f6feb"></div>Sender</div>
        <div class="ll"><div class="ld" style="background:#196c2e"></div>Receiver</div>
        <div class="ll"><div class="ld" style="background:#f0883e;border-radius:3px"></div>Router</div>
        <div class="ll"><div class="ld" style="background:#58a6ff"></div>Packet</div>
        <div class="ll"><div class="ld" style="background:#f85149"></div>Dropped</div>
      </div>
    </div>
    <script>
    (function(){
      const cv=document.getElementById('c'),ctx=cv.getContext('2d');
      cv.width=Math.min(window.innerWidth-40,1050);cv.height=430;
      const P={reno:{d:0.05,g:1.2,m:31,c:'#e74c3c'},tahoe:{d:0.08,g:1.0,m:25,c:'#3498db'},
               vegas:{d:0.02,g:1.1,m:22,c:'#2ecc71'},bbr:{d:0.01,g:1.3,m:38,c:'#9b59b6'},
               cubic:{d:0.022,g:1.25,m:35,c:'#f39c12'}};
      let proto='reno',run=false,t=0,ps=0,pr=0,pd=0,pkts=[],cw=new Array(50).fill(1);
      let aid=null,lts=null,spd=2,nd=[],R0={},R1={};
      function init(){
        nd=[];pkts=[];t=0;ps=0;pr=0;pd=0;cw=new Array(50).fill(1);
        const W=cv.width,H=cv.height;
        R0={x:W*.38,y:H/2,l:'R0'};R1={x:W*.62,y:H/2,l:'R1'};
        for(let i=0;i<50;i++){const a=(i/50)*Math.PI*2;
          nd.push({x:W*.13+Math.cos(a)*W*.09,y:H/2+Math.sin(a)*H*.38});}
        for(let i=0;i<50;i++){const a=(i/50)*Math.PI*2;
          nd.push({x:W*.87+Math.cos(a)*W*.09,y:H/2+Math.sin(a)*H*.38});}
        upd();drw();
      }
      function drw(){
        const W=cv.width,H=cv.height;
        ctx.clearRect(0,0,W,H);ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
        ctx.strokeStyle='#21262d';ctx.lineWidth=0.4;
        for(let i=0;i<50;i++){ctx.beginPath();ctx.moveTo(nd[i].x,nd[i].y);
          ctx.lineTo(R0.x,R0.y);ctx.stroke();}
        for(let i=50;i<100;i++){ctx.beginPath();ctx.moveTo(nd[i].x,nd[i].y);
          ctx.lineTo(R1.x,R1.y);ctx.stroke();}
        ctx.strokeStyle='#388bfd55';ctx.lineWidth=3;
        ctx.beginPath();ctx.moveTo(R0.x,R0.y);ctx.lineTo(R1.x,R1.y);ctx.stroke();
        ctx.fillStyle='#58a6ff66';ctx.font='10px Segoe UI';ctx.textAlign='center';
        ctx.fillText('2Mb/50ms',(R0.x+R1.x)/2,R0.y-12);
        for(let i=0;i<50;i++){ctx.beginPath();ctx.arc(nd[i].x,nd[i].y,4,0,Math.PI*2);
          ctx.fillStyle=cw[i]>1?'#1f6feb':'#21262d';ctx.fill();}
        for(let i=50;i<100;i++){ctx.beginPath();ctx.arc(nd[i].x,nd[i].y,4,0,Math.PI*2);
          ctx.fillStyle='#196c2e';ctx.fill();}
        [R0,R1].forEach(r=>{ctx.fillStyle='#f0883e';ctx.strokeStyle='#ffa657';
          ctx.lineWidth=2;ctx.beginPath();ctx.roundRect(r.x-15,r.y-15,30,30,4);
          ctx.fill();ctx.stroke();ctx.fillStyle='#fff';ctx.font='bold 9px Segoe UI';
          ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(r.l,r.x,r.y);});
        for(const p of pkts){ctx.beginPath();ctx.arc(p.x,p.y,2.5,0,Math.PI*2);
          ctx.fillStyle=p.dr?'#f85149':P[proto].c;
          ctx.globalAlpha=p.a;ctx.fill();ctx.globalAlpha=1;}
        const pg=Math.min(t/10,1);
        ctx.fillStyle='#161b22';ctx.fillRect(15,H-14,W-30,7);
        ctx.fillStyle=P[proto].c;ctx.fillRect(15,H-14,(W-30)*pg,7);
        ctx.fillStyle='#8b949e';ctx.font='10px Segoe UI';ctx.textAlign='left';
        ctx.textBaseline='alphabetic';
        ctx.fillText('t='+t.toFixed(1)+'s / 10.0s | Protocol: '+proto.toUpperCase(),15,H-18);
      }
      function spawn(){
        const p=P[proto];
        for(let i=0;i<50;i++){
          if(Math.random()>.25)continue;
          const dr=Math.random()<p.d;ps++;
          if(dr){pd++;cw[i]=Math.max(1,cw[i]/2);}
          else{pr++;cw[i]=Math.min(p.m,cw[i]*p.g);}
          pkts.push({x:nd[i].x,y:nd[i].y,
            path:[{x:R0.x,y:R0.y},{x:R1.x,y:R1.y},{x:nd[i+50].x,y:nd[i+50].y}],
            st:0,dr,a:1,sp:.018+Math.random()*.012});
        }
      }
      function upkts(dt){
        pkts=pkts.filter(p=>{
          if(p.st>=p.path.length)return false;
          const tg=p.path[p.st];
          const dx=tg.x-p.x,dy=tg.y-p.y,ds=Math.sqrt(dx*dx+dy*dy);
          const mv=p.sp*spd*dt*60*2;
          if(ds<mv+1){p.x=tg.x;p.y=tg.y;p.st++;}
          else{p.x+=dx/ds*mv;p.y+=dy/ds*mv;}
          if(p.dr&&p.st>=1)p.a-=.03;
          return p.a>0&&p.st<p.path.length+1;
        });
        if(pkts.length>500)pkts.splice(0,pkts.length-500);
      }
      function upd(){
        document.getElementById('t').textContent=t.toFixed(1)+'s';
        document.getElementById('ps').textContent=ps;
        document.getElementById('pr').textContent=pr;
        document.getElementById('pd').textContent=pd;
        document.getElementById('tp').textContent=((pr*8000)/Math.max(t,.1)/1e6).toFixed(3);
      }
      function loop(ts){
        if(!run)return;if(!lts)lts=ts;
        const dt=Math.min((ts-lts)/1000,.05);lts=ts;
        t+=dt*spd;
        if(t>=10){t=10;run=false;upd();drw();return;}
        spawn();upkts(dt);upd();drw();
        aid=requestAnimationFrame(loop);
      }
      document.getElementById('s1').onclick=()=>{
        if(t>=10)init();run=true;lts=null;aid=requestAnimationFrame(loop);};
      document.getElementById('s2').onclick=()=>{
        run=!run;if(run){lts=null;aid=requestAnimationFrame(loop);}};
      document.getElementById('s3').onclick=()=>{
        run=false;cancelAnimationFrame(aid);init();};
      document.getElementById('spd').oninput=e=>{spd=parseInt(e.target.value);};
      document.querySelectorAll('.pb').forEach(b=>{
        b.onclick=()=>{document.querySelectorAll('.pb').forEach(x=>x.classList.remove('on'));
          b.classList.add('on');proto=b.dataset.p;run=false;cancelAnimationFrame(aid);init();};
      });
      init();
    })();
    </script>
    """

    st.components.v1.html(SIM_HTML, height=600, scrolling=False)

    st.markdown("---")
    st.markdown("###  Generated NAM Files")
    col1, col2, col3 = st.columns(3)
    for col, proto in zip([col1,col2,col3], ["reno","tahoe","vegas"]):
        nam_file = f"100nodes-{proto}.nam"
        exists   = os.path.exists(nam_file)
        with col:
            if exists:
                size = os.path.getsize(nam_file)
                st.success(f" {nam_file}\n{size:,} bytes")
            else:
                st.warning(f" {nam_file} not found")

# ── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#7f8c8d;font-size:0.85em;padding:10px;'>
    NS2 TCP Performance Dashboard |
    TCP Reno · Tahoe · Vegas · Cubic · BBR · DCTCP |
    ML Recommendation Engine (XGBoost 83.83%)
</div>
""", unsafe_allow_html=True)