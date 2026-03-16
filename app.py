"""
NexusQuant — AI Stock Intelligence Platform
Main Streamlit entry point
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NexusQuant · AI Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.data_engine import (
    resolve_ticker, fetch_stock, fetch_info,
    add_features, build_models, ensemble_forecast,
    technical_indicators, investment_signal,
)
from utils.charts import (
    price_ma_chart, candlestick_chart, split_chart,
    model_compare_chart, bollinger_chart, rsi_chart,
    macd_chart, forecast_chart, backtest_chart,
    volatility_histogram, rolling_returns_chart,
    portfolio_growth_chart, metrics_bar_chart,
    portfolio_donut,
)

# ═══════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root ── */
:root {
  --bg:      #070B14;
  --bg2:     #0D1321;
  --bg3:     #131A2A;
  --border:  #1E2A3A;
  --text:    #E4EAF5;
  --text2:   #8496AE;
  --accent:  #E8C96D;
  --blue:    #6EE7F7;
  --pink:    #F76EA0;
  --green:   #5CF5A0;
  --red:     #F76E6E;
  --purple:  #B08BFF;
}

/* ── App shell ── */
.stApp { background: var(--bg) !important; }
section[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.8rem 2.2rem 3rem !important; max-width: 100% !important; }

/* ── Typography ── */
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }
p, li, span, div { font-family: 'DM Sans', sans-serif !important; }
code, .stCode { font-family: 'JetBrains Mono', monospace !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 16px 18px !important;
}
[data-testid="metric-container"] label {
  font-size: 11px !important; letter-spacing: 0.8px !important;
  text-transform: uppercase !important; color: var(--text2) !important;
  font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 1.5rem !important; font-weight: 700 !important;
  color: var(--text) !important;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg2) !important;
  border-radius: 10px !important; padding: 4px !important;
  gap: 4px !important; border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; border-radius: 7px !important;
  color: var(--text2) !important; padding: 8px 20px !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 13px !important;
  border: none !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(232,201,109,0.12) !important;
  color: var(--accent) !important;
}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stSelectbox select {
  background: var(--bg2) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(232,201,109,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--accent) !important; color: #0D1010 !important;
  border: none !important; border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
  padding: 10px 22px !important; transition: all .18s !important;
}
.stButton > button:hover {
  background: #F0A64A !important; transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(232,201,109,0.25) !important;
}
button[kind="secondary"] {
  background: var(--bg2) !important; color: var(--text2) !important;
  border: 1px solid var(--border) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--bg2) !important; border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important; color: var(--text2) !important;
  border: 1px solid var(--border) !important;
}

/* ── Sidebar radio buttons → nav items ── */
.stRadio > div { gap: 4px !important; }
.stRadio label {
  background: transparent !important; border-radius: 8px !important;
  padding: 8px 14px !important; cursor: pointer !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 13.5px !important;
  color: var(--text2) !important; transition: all .15s !important;
  display: flex !important; align-items: center !important;
  border: 1px solid transparent !important;
}
.stRadio label:hover { color: var(--text) !important; background: var(--bg3) !important; }
.stRadio [data-checked="true"] label, .stRadio input:checked + div {
  color: var(--accent) !important; background: rgba(232,201,109,0.1) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Custom card component ── */
.nq-card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 12px; padding: 18px 20px; margin-bottom: 4px;
}
.nq-card-title {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: var(--text2); text-transform: uppercase; letter-spacing: 0.8px;
  margin-bottom: 10px;
}
.nq-value {
  font-family: 'Syne', sans-serif; font-weight: 700;
  font-size: 1.4rem; color: var(--text);
}

/* ── Signal badge ── */
.sig-buy  { background:rgba(92,245,160,0.12); color:#5CF5A0; border:1px solid rgba(92,245,160,0.3);  border-radius:8px; padding:6px 18px; font-family:'Syne',sans-serif; font-weight:800; font-size:1.4rem; letter-spacing:2px; display:inline-block; }
.sig-hold { background:rgba(232,201,109,0.12);color:#E8C96D; border:1px solid rgba(232,201,109,0.3); border-radius:8px; padding:6px 18px; font-family:'Syne',sans-serif; font-weight:800; font-size:1.4rem; letter-spacing:2px; display:inline-block; }
.sig-sell { background:rgba(247,110,110,0.12);color:#F76E6E; border:1px solid rgba(247,110,110,0.3); border-radius:8px; padding:6px 18px; font-family:'Syne',sans-serif; font-weight:800; font-size:1.4rem; letter-spacing:2px; display:inline-block; }

/* ── Hero banner ── */
.nq-hero {
  background: linear-gradient(135deg, #0D1321 0%, #131A2A 60%, #0D1321 100%);
  border: 1px solid #1E2A3A; border-radius: 16px;
  padding: 40px 48px; margin-bottom: 28px; position: relative; overflow: hidden;
}
.nq-hero::before {
  content: ''; position: absolute; top: -60px; right: -60px;
  width: 240px; height: 240px; border-radius: 50%;
  background: radial-gradient(circle, rgba(232,201,109,0.08) 0%, transparent 70%);
}
.nq-hero h1 { font-size: 2.4rem !important; margin-bottom: 8px !important; }
.nq-hero p  { color: #8496AE !important; font-size: 15px !important; line-height: 1.7 !important; }

/* ── Chart container ── */
.nq-chart-wrap {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin-bottom: 16px;
}
.nq-chart-label {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: var(--text2); text-transform: uppercase; letter-spacing: 0.8px;
  margin-bottom: 8px;
}

/* ── Streamlit plot container borders ── */
[data-testid="stPlotlyChart"] { border-radius: 10px; overflow: hidden; }

/* ── Sidebar logo area ── */
.nq-sidebar-logo {
  padding: 20px 16px 16px;
  border-bottom: 1px solid #1E2A3A;
  margin-bottom: 12px;
}
.nq-logo-mark {
  background: linear-gradient(135deg, #E8C96D, #F0A64A);
  color: #000; font-family: 'Syne', sans-serif; font-weight: 800;
  font-size: 12px; width: 36px; height: 36px; border-radius: 9px;
  display: inline-flex; align-items: center; justify-content: center;
  margin-bottom: 8px;
}
.nq-logo-name {
  font-family: 'Syne', sans-serif; font-weight: 700; font-size: 16px;
  color: #E4EAF5; display: block;
}
.nq-logo-tag {
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  color: #8496AE; display: block;
}

/* ── Status dot ── */
.nq-status { display:flex; align-items:center; gap:8px; padding: 8px 16px; }
.nq-dot { width:7px; height:7px; border-radius:50%; background:#5CF5A0;
          box-shadow:0 0 8px #5CF5A0; animation: blink 2s infinite; display:inline-block; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }
.nq-status-text { font-size: 12px; color: #8496AE; font-family:'JetBrains Mono',monospace; }

/* ── Portfolio result row ── */
.pb-row-card {
  background: var(--bg3); border-radius: 10px; padding: 14px 18px;
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 8px; border: 1px solid var(--border);
}
.pb-ret-pos { color: #5CF5A0; font-family:'JetBrains Mono',monospace; font-size:12px; }
.pb-ret-neg { color: #F76E6E; font-family:'JetBrains Mono',monospace; font-size:12px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: #1E2A3A; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='nq-sidebar-logo'>
      <div class='nq-logo-mark'>NQ</div>
      <span class='nq-logo-name'>NexusQuant</span>
      <span class='nq-logo-tag'>AI Stock Intelligence</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Search**")
    company_input = st.text_input("", placeholder="Apple, Tesla, NVIDIA…",
                                  label_visibility="collapsed", key="company_search")

    st.markdown("**Investment ($)**")
    investment = st.number_input("", min_value=100, value=10000, step=500,
                                 label_visibility="collapsed")

    st.markdown("**Period**")
    period = st.selectbox("", ["1y","2y","5y","max"],
                          index=1, label_visibility="collapsed")

    analyze_btn = st.button("⚡  Analyze", use_container_width=True)

    st.divider()

    st.markdown("**Navigate**")
    page = st.radio("", [
        "🏠  Overview",
        "📊  ML Models",
        "📉  Technical",
        "🔮  Predictions",
        "🌊  Volatility",
        "💼  Portfolio",
    ], label_visibility="collapsed")

    st.divider()

    # Quick chips
    st.markdown("<p style='font-size:11px;color:#4a5a70;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px'>Quick Pick</p>", unsafe_allow_html=True)
    chips = ["Apple","Tesla","NVIDIA","Microsoft","Amazon"]
    cols_c = st.columns(2)
    for i, c in enumerate(chips):
        if cols_c[i%2].button(c, key=f"chip_{c}", use_container_width=True):
            st.session_state["company_search"] = c
            st.session_state["_auto_analyze"]  = True
            st.rerun()

    st.markdown("""
    <div class='nq-status' style='margin-top:24px'>
      <span class='nq-dot'></span>
      <span class='nq-status-text'>Markets Live</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════
if "data" not in st.session_state:
    st.session_state["data"] = None

# Handle quick-chip auto-analyze
if st.session_state.get("_auto_analyze"):
    del st.session_state["_auto_analyze"]
    analyze_btn = True
    company_input = st.session_state.get("company_search", "")


# ═══════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════
if analyze_btn and company_input.strip():
    with st.spinner("Fetching market data and training models…"):
        try:
            ticker = resolve_ticker(company_input)
            df_raw = fetch_stock(ticker, period)
            df     = add_features(df_raw)
            info   = fetch_info(ticker)
            models = build_models(df)
            tech   = technical_indicators(df)
            fc     = ensemble_forecast(models, df)
            sig    = investment_signal(df, fc["ensemble"])

            st.session_state["data"] = dict(
                ticker=ticker, df=df, df_raw=df_raw,
                info=info, models=models, tech=tech,
                fc=fc, sig=sig, investment=investment,
            )
        except Exception as e:
            st.error(f"⚠ {e}")


# ═══════════════════════════════════════════════════════════
#  NO DATA → WELCOME
# ═══════════════════════════════════════════════════════════
if st.session_state["data"] is None:
    st.markdown("""
    <div class='nq-hero'>
      <h1>AI-Powered Stock Intelligence</h1>
      <p>Search any company to unlock ML predictions, technical indicators,<br>
         risk analysis, and portfolio simulations — all in one dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1,"🤖","3 ML Models","Linear Regression, Random Forest & SVR trained on 2 years of data"),
        (c2,"📡","Technical Suite","RSI, MACD, Bollinger Bands & Moving Averages"),
        (c3,"💼","Portfolio Sim","Simulate investment growth and multi-stock allocations"),
    ]:
        col.markdown(f"""
        <div class='nq-card'>
          <div style='font-size:1.6rem;margin-bottom:10px'>{icon}</div>
          <div style='font-family:Syne,sans-serif;font-weight:700;font-size:15px;
                      color:#E4EAF5;margin-bottom:6px'>{title}</div>
          <div style='font-size:13px;color:#8496AE;line-height:1.6'>{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════
#  UNPACK SESSION DATA
# ═══════════════════════════════════════════════════════════
D      = st.session_state["data"]
ticker = D["ticker"]
df     = D["df"]
info   = D["info"]
models = D["models"]
tech   = D["tech"]
fc     = D["fc"]
sig    = D["sig"]
inv    = D["investment"]

close_now  = df["Close"].iloc[-1]
close_prev = df["Close"].iloc[-2]
day_chg    = (close_now - close_prev) / close_prev * 100


# ═══════════════════════════════════════════════════════════
#  HEADER BAR
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;
            background:#0D1321;border:1px solid #1E2A3A;border-radius:12px;
            padding:14px 22px;margin-bottom:24px'>
  <div>
    <span style='font-family:Syne,sans-serif;font-weight:800;font-size:1.4rem;
                 color:#E4EAF5'>{info.get("name", ticker)}</span>
    <span style='font-family:JetBrains Mono,monospace;font-size:12px;
                 color:#4a5a70;margin-left:12px'>{ticker}</span>
    <span style='font-size:12px;color:#8496AE;margin-left:10px'>
      {info.get("sector","N/A")} · {info.get("industry","N/A")}
    </span>
  </div>
  <div style='display:flex;align-items:center;gap:20px'>
    <span style='font-family:Syne,sans-serif;font-weight:700;font-size:1.5rem;
                 color:#E4EAF5'>${close_now:,.2f}</span>
    <span style='font-size:13px;color:{"#5CF5A0" if day_chg>=0 else "#F76E6E"};
                 font-family:JetBrains Mono,monospace'>
      {"▲" if day_chg>=0 else "▼"} {abs(day_chg):.2f}%
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  ── PAGE: OVERVIEW ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════
if "Overview" in page:

    # ── KPI row ─────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    mc = info.get("market_cap")
    mc_fmt = f"${mc/1e12:.2f}T" if mc and mc>1e12 else (f"${mc/1e9:.1f}B" if mc else "N/A")

    k1.metric("Current Price",    f"${close_now:,.2f}")
    k2.metric("Day Change",       f"{day_chg:+.2f}%",
              delta_color="normal" if day_chg>=0 else "inverse")
    k3.metric("52W High",  f"${info['52w_high']:,.2f}"  if info.get("52w_high") else "N/A")
    k4.metric("52W Low",   f"${info['52w_low']:,.2f}"   if info.get("52w_low")  else "N/A")
    k5.metric("Market Cap", mc_fmt)
    k6.metric("P/E Ratio",  f"{info['pe']:.1f}" if info.get("pe") else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Signal row ───────────────────────────────────────────
    sa, sb, sc, sd = st.columns([1,1,1,2])
    sig_cls = {"BUY":"sig-buy","HOLD":"sig-hold","SELL":"sig-sell"}[sig["signal"]]
    sa.markdown(f"<div class='{sig_cls}'>{sig['signal']}</div>", unsafe_allow_html=True)
    sb.metric("Risk Level",    sig["risk"])
    sc.metric("Annual Vol",    f"{sig['annual_vol']}%")
    sd.metric("Confidence",    f"{sig['score']}/4 signals bullish")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────
    tab_price, tab_candle = st.tabs(["📈 Price + Moving Averages", "🕯 Candlestick (90 days)"])
    with tab_price:
        st.plotly_chart(price_ma_chart(df), use_container_width=True, config={"displayModeBar":False})
    with tab_candle:
        st.plotly_chart(candlestick_chart(df), use_container_width=True, config={"displayModeBar":False})

    # ── Description ──────────────────────────────────────────
    desc = info.get("description","")
    if desc:
        with st.expander("About this company"):
            st.markdown(f"<p style='font-size:13px;color:#8496AE;line-height:1.7'>{desc[:800]}…</p>",
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  ── PAGE: ML MODELS ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════
elif "ML Models" in page:
    st.markdown("### Machine Learning Models")

    # ── Metrics ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    for col, key, label, color in zip(
        st.columns(3),
        ["lr","rf","svr"],
        ["Linear Regression","Random Forest","SVR"],
        ["#6EE7F7","#5CF5A0","#B08BFF"],
    ):
        m = models["metrics"][key]
        r2c = "#5CF5A0" if m["r2"]>=.9 else ("#E8C96D" if m["r2"]>=.7 else "#F76E6E")
        col.markdown(f"""
        <div class='nq-card'>
          <div class='nq-card-title'>{label}</div>
          <div style='display:flex;justify-content:space-between;padding:8px 0;
                      border-bottom:1px solid #1E2A3A'>
            <span style='font-size:12px;color:#8496AE'>R² Score</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:13px;
                         color:{r2c};font-weight:600'>{m["r2"]}</span>
          </div>
          <div style='display:flex;justify-content:space-between;padding:8px 0'>
            <span style='font-size:12px;color:#8496AE'>RMSE</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:13px;
                         color:#E4EAF5'>{m["rmse"]}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────
    ca, cb = st.columns(2)
    with ca:
        st.markdown("<div class='nq-chart-label'>Train / Test Split</div>", unsafe_allow_html=True)
        st.plotly_chart(split_chart(models), use_container_width=True, config={"displayModeBar":False})
    with cb:
        st.markdown("<div class='nq-chart-label'>Model Predictions vs Actual</div>", unsafe_allow_html=True)
        st.plotly_chart(model_compare_chart(models), use_container_width=True, config={"displayModeBar":False})

    st.markdown("<div class='nq-chart-label'>Performance Comparison</div>", unsafe_allow_html=True)
    st.plotly_chart(metrics_bar_chart(models["metrics"]), use_container_width=True, config={"displayModeBar":False})


# ═══════════════════════════════════════════════════════════
#  ── PAGE: TECHNICAL ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════
elif "Technical" in page:
    st.markdown("### Technical Indicators")

    st.markdown("<div class='nq-chart-label'>Bollinger Bands</div>", unsafe_allow_html=True)
    st.plotly_chart(bollinger_chart(df, tech), use_container_width=True, config={"displayModeBar":False})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='nq-chart-label'>RSI (14) — Overbought > 70, Oversold < 30</div>", unsafe_allow_html=True)
        st.plotly_chart(rsi_chart(tech), use_container_width=True, config={"displayModeBar":False})
    with c2:
        st.markdown("<div class='nq-chart-label'>MACD</div>", unsafe_allow_html=True)
        st.plotly_chart(macd_chart(tech), use_container_width=True, config={"displayModeBar":False})

    # RSI reading
    last_rsi = tech["rsi"].dropna().iloc[-1]
    rsi_color = "#F76E6E" if last_rsi>70 else ("#5CF5A0" if last_rsi<30 else "#E8C96D")
    rsi_label = "Overbought ⚠" if last_rsi>70 else ("Oversold 🟢" if last_rsi<30 else "Neutral")
    st.markdown(f"""
    <div style='display:flex;gap:20px;margin-top:8px'>
      <div class='nq-card' style='flex:1'>
        <div class='nq-card-title'>Current RSI</div>
        <span style='font-family:Syne,sans-serif;font-weight:700;font-size:1.6rem;
                     color:{rsi_color}'>{last_rsi:.1f}</span>
        <span style='font-size:12px;color:#8496AE;margin-left:10px'>{rsi_label}</span>
      </div>
      <div class='nq-card' style='flex:1'>
        <div class='nq-card-title'>MACD Signal</div>
        <span style='font-family:Syne,sans-serif;font-weight:700;font-size:1.2rem;
                     color:{"#5CF5A0" if tech["hist"].iloc[-1]>0 else "#F76E6E"}'>
          {"Bullish Cross 🟢" if tech["hist"].iloc[-1]>0 else "Bearish Cross 🔴"}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  ── PAGE: PREDICTIONS ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════
elif "Predictions" in page:
    st.markdown("### AI Predictions & Forecasting")

    # ── Ensemble cards ──────────────────────────────────────
    e1,e2,e3,e4 = st.columns(4)
    for col, key, label in [
        (e1,"lr","Linear Reg"),
        (e2,"rf","Random Forest"),
        (e3,"svr","SVR"),
        (e4,"ensemble","⚡ Ensemble"),
    ]:
        val = fc[key]
        chg = (val - close_now) / close_now * 100
        col.markdown(f"""
        <div class='nq-card' style='{"border-color:rgba(232,201,109,0.4);background:rgba(232,201,109,0.04);" if key=="ensemble" else ""}'>
          <div class='nq-card-title'>{label}</div>
          <div style='font-family:Syne,sans-serif;font-weight:700;font-size:1.3rem;
                      color:{"#E8C96D" if key=="ensemble" else "#E4EAF5"}'>${val}</div>
          <div style='font-size:11px;color:{"#5CF5A0" if chg>=0 else "#F76E6E"};
                      font-family:JetBrains Mono,monospace;margin-top:4px'>
            {chg:+.2f}% vs now
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='nq-chart-label'>30-Day Ensemble Forecast</div>", unsafe_allow_html=True)
    st.plotly_chart(forecast_chart(df, fc), use_container_width=True, config={"displayModeBar":False})

    st.markdown("<div class='nq-chart-label'>Backtesting — Predicted vs Actual</div>", unsafe_allow_html=True)
    st.plotly_chart(backtest_chart(models), use_container_width=True, config={"displayModeBar":False})

    # Backtest metrics
    ens_bt = (models["preds"]["lr"] + models["preds"]["rf"] + models["preds"]["svr"]) / 3
    from sklearn.metrics import r2_score, mean_squared_error
    bt_r2   = r2_score(models["actuals"], ens_bt)
    bt_rmse = np.sqrt(mean_squared_error(models["actuals"], ens_bt))
    b1, b2 = st.columns(2)
    b1.metric("Ensemble R² (Backtest)", f"{bt_r2:.4f}")
    b2.metric("Ensemble RMSE (Backtest)", f"${bt_rmse:.2f}")


# ═══════════════════════════════════════════════════════════
#  ── PAGE: VOLATILITY ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════
elif "Volatility" in page:
    st.markdown("### Volatility & Risk Analysis")

    v1,v2,v3,v4 = st.columns(4)
    daily_vol  = df["Daily_Return"].std() * 100
    annual_vol = daily_vol * np.sqrt(252)
    skew       = float(df["Daily_Return"].skew())
    kurt       = float(df["Daily_Return"].kurtosis())

    v1.metric("Daily Volatility",  f"{daily_vol:.2f}%")
    v2.metric("Annual Volatility", f"{annual_vol:.2f}%")
    v3.metric("Return Skewness",   f"{skew:.3f}")
    v4.metric("Excess Kurtosis",   f"{kurt:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='nq-chart-label'>Daily Returns Distribution</div>", unsafe_allow_html=True)
        st.plotly_chart(volatility_histogram(df), use_container_width=True, config={"displayModeBar":False})
    with c2:
        st.markdown("<div class='nq-chart-label'>Daily Returns (1 Year)</div>", unsafe_allow_html=True)
        st.plotly_chart(rolling_returns_chart(df), use_container_width=True, config={"displayModeBar":False})

    # Risk scoring
    rs = sig["risk_score"]
    rs_color = "#5CF5A0" if rs<30 else ("#E8C96D" if rs<60 else "#F76E6E")
    st.markdown(f"""
    <div class='nq-card' style='margin-top:8px'>
      <div class='nq-card-title'>Risk Score</div>
      <div style='display:flex;align-items:center;gap:16px'>
        <span style='font-family:Syne,sans-serif;font-weight:800;font-size:2.4rem;color:{rs_color}'>{rs}</span>
        <div>
          <div style='font-size:13px;color:#E4EAF5'>Risk Level: <strong style="color:{rs_color}">{sig["risk"]}</strong></div>
          <div style='font-size:12px;color:#8496AE;margin-top:3px'>Based on annualized volatility of {sig["annual_vol"]}%</div>
        </div>
        <div style='flex:1;height:8px;background:#1E2A3A;border-radius:99px;overflow:hidden;margin-left:16px'>
          <div style='width:{min(rs,100)}%;height:100%;background:{rs_color};border-radius:99px;
                      transition:width .6s ease'></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  ── PAGE: PORTFOLIO ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════
elif "Portfolio" in page:
    st.markdown("### Portfolio Simulator")

    # Current stock simulation
    base    = df["Close"].iloc[0]
    curr    = df["Close"].iloc[-1]
    growth  = inv * (curr / base)
    profit  = growth - inv
    ret_pct = profit / inv * 100

    p1, p2, p3 = st.columns(3)
    p1.metric("Initial Investment", f"${inv:,.2f}")
    p2.metric("Current Value",      f"${growth:,.2f}",
              delta=f"${profit:+,.2f}")
    p3.metric("Total Return",       f"{ret_pct:+.2f}%")

    st.markdown("<div class='nq-chart-label' style='margin-top:12px'>Investment Growth — {}</div>".format(
        info.get("name", ticker)), unsafe_allow_html=True)
    st.plotly_chart(portfolio_growth_chart(df, inv), use_container_width=True, config={"displayModeBar":False})

    st.divider()

    # ── Multi-stock allocator ────────────────────────────────
    st.markdown("#### Multi-Stock Allocation Simulator")
    st.caption("Add stocks below and simulate your portfolio across multiple companies.")

    if "portfolio_rows" not in st.session_state:
        st.session_state["portfolio_rows"] = [
            {"name":"Apple",     "amount":5000},
            {"name":"Microsoft", "amount":3000},
            {"name":"NVIDIA",    "amount":2000},
        ]

    rows = st.session_state["portfolio_rows"]
    updated = []
    for i, row in enumerate(rows):
        c1, c2, c3 = st.columns([3,2,1])
        n = c1.text_input(f"Company {i+1}", value=row["name"], key=f"pn_{i}",
                          label_visibility="collapsed")
        a = c2.number_input(f"Amount {i+1}", value=row["amount"], min_value=0,
                             key=f"pa_{i}", label_visibility="collapsed")
        if c3.button("✕", key=f"del_{i}"):
            rows.pop(i); st.rerun()
        updated.append({"name": n, "amount": a})
    st.session_state["portfolio_rows"] = updated

    col_add, col_run = st.columns([1,4])
    if col_add.button("＋ Add Stock"):
        st.session_state["portfolio_rows"].append({"name":"", "amount":1000})
        st.rerun()

    if col_run.button("🚀  Run Portfolio Simulation", use_container_width=True):
        results = []
        with st.spinner("Fetching multi-stock data…"):
            for row in updated:
                if not row["name"].strip(): continue
                try:
                    t2 = resolve_ticker(row["name"])
                    d2 = fetch_stock(t2, "1y")
                    ret = (d2["Close"].iloc[-1] - d2["Close"].iloc[0]) / d2["Close"].iloc[0]
                    results.append({
                        "name":      row["name"],
                        "ticker":    t2,
                        "amount":    row["amount"],
                        "projected": round(row["amount"] * (1+ret), 2),
                        "return_pct":round(ret*100, 2),
                    })
                except Exception:
                    pass

        if results:
            st.session_state["portfolio_results"] = results

    if "portfolio_results" in st.session_state:
        res = st.session_state["portfolio_results"]
        total_inv  = sum(r["amount"]    for r in res)
        total_proj = sum(r["projected"] for r in res)
        total_ret  = (total_proj - total_inv) / total_inv * 100

        t1,t2,t3 = st.columns(3)
        t1.metric("Total Invested",  f"${total_inv:,.2f}")
        t2.metric("Total Projected", f"${total_proj:,.2f}",
                  delta=f"${total_proj-total_inv:+,.2f}")
        t3.metric("Portfolio Return", f"{total_ret:+.2f}%")

        rr, dd = st.columns([2,1])
        with rr:
            st.markdown("<br>", unsafe_allow_html=True)
            for r in res:
                pos = r["return_pct"] >= 0
                st.markdown(f"""
                <div style='background:#131A2A;border:1px solid #1E2A3A;border-radius:10px;
                             padding:12px 18px;margin-bottom:8px;display:flex;
                             align-items:center;justify-content:space-between'>
                  <div>
                    <span style='font-weight:600;font-size:14px;color:#E4EAF5'>{r["name"]}</span>
                    <span style='font-size:11px;color:#4a5a70;margin-left:8px'>{r["ticker"]}</span>
                  </div>
                  <span style='font-size:12px;color:#8496AE'>Invested: ${r["amount"]:,.0f}</span>
                  <span style='font-family:JetBrains Mono,monospace;font-weight:600;
                               color:{"#5CF5A0" if pos else "#F76E6E"}'>
                    ${r["projected"]:,.2f}
                  </span>
                  <span style='font-family:JetBrains Mono,monospace;font-size:12px;
                               padding:3px 8px;border-radius:4px;
                               background:{"rgba(92,245,160,0.1)" if pos else "rgba(247,110,110,0.1)"};
                               color:{"#5CF5A0" if pos else "#F76E6E"}'>
                    {r["return_pct"]:+.2f}%
                  </span>
                </div>""", unsafe_allow_html=True)
        with dd:
            st.plotly_chart(portfolio_donut(res), use_container_width=True,
                            config={"displayModeBar":False})
