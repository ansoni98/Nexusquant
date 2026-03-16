"""
NexusQuant — AI Stock Intelligence Platform
Single-file Streamlit app (no external utils imports)
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(
    page_title="NexusQuant · AI Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# ═══════════════════════════════════════════════════════════
#  PALETTE
# ═══════════════════════════════════════════════════════════
PAL = dict(
    bg="#070B14", bg2="#0D1321", bg3="#131A2A", border="#1E2A3A",
    text="#E4EAF5", text2="#8496AE",
    accent="#E8C96D", accent2="#F0A64A",
    blue="#6EE7F7", pink="#F76EA0",
    green="#5CF5A0", red="#F76E6E", purple="#B08BFF",
)

# ═══════════════════════════════════════════════════════════
#  COMPANY MAP
# ═══════════════════════════════════════════════════════════
COMPANY_MAP = {
    "apple":"AAPL","microsoft":"MSFT","google":"GOOGL","alphabet":"GOOGL",
    "amazon":"AMZN","tesla":"TSLA","meta":"META","facebook":"META",
    "nvidia":"NVDA","netflix":"NFLX","uber":"UBER","airbnb":"ABNB",
    "spotify":"SPOT","paypal":"PYPL","adobe":"ADBE","salesforce":"CRM",
    "intel":"INTC","amd":"AMD","qualcomm":"QCOM","oracle":"ORCL",
    "ibm":"IBM","cisco":"CSCO","sony":"SONY","toyota":"TM",
    "boeing":"BA","ford":"F","general motors":"GM",
    "johnson & johnson":"JNJ","pfizer":"PFE","moderna":"MRNA",
    "berkshire hathaway":"BRK-B","jpmorgan":"JPM","goldman sachs":"GS",
    "bank of america":"BAC","visa":"V","mastercard":"MA",
    "walmart":"WMT","target":"TGT","shopify":"SHOP","zoom":"ZM",
    "palantir":"PLTR","snowflake":"SNOW","coinbase":"COIN",
    "square":"SQ","block":"SQ","tcs":"TCS.NS","infosys":"INFY",
    "wipro":"WIT","reliance":"RELIANCE.NS","hdfc":"HDFCBANK.NS",
    "hcl":"HCLTECH.NS",
}

def resolve_ticker(name: str) -> str:
    key = name.strip().lower()
    if key in COMPANY_MAP:
        return COMPANY_MAP[key]
    for k, v in COMPANY_MAP.items():
        if key in k or k in key:
            return v
    return name.upper()

# ═══════════════════════════════════════════════════════════
#  DATA ENGINE
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock(ticker: str, period: str = "2y") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for '{ticker}'. Try using the ticker symbol directly (e.g. AAPL).")
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":        info.get("longName", ticker),
            "sector":      info.get("sector", "N/A"),
            "industry":    info.get("industry", "N/A"),
            "market_cap":  info.get("marketCap"),
            "pe":          info.get("trailingPE"),
            "52w_high":    info.get("fiftyTwoWeekHigh"),
            "52w_low":     info.get("fiftyTwoWeekLow"),
            "description": info.get("longBusinessSummary", ""),
        }
    except Exception:
        return {"name": ticker}

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"]          = df["Close"].rolling(5).mean()
    df["MA20"]         = df["Close"].rolling(20).mean()
    df["MA50"]         = df["Close"].rolling(50).mean()
    df["MA200"]        = df["Close"].rolling(200).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"]   = df["Daily_Return"].rolling(20).std()
    df["Lag1"]         = df["Close"].shift(1)
    df["Lag5"]         = df["Close"].shift(5)
    df.dropna(inplace=True)
    return df

def build_models(df: pd.DataFrame) -> dict:
    FEATURES = ["Open","High","Low","Volume","MA5","MA20","Lag1","Lag5","Volatility"]
    X = df[FEATURES].values
    y = df["Close"].values
    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    scaler  = MinMaxScaler()
    Xs_tr   = scaler.fit_transform(X_tr)
    Xs_te   = scaler.transform(X_te)
    lr  = LinearRegression().fit(Xs_tr, y_tr)
    rf  = RandomForestRegressor(100, random_state=42).fit(Xs_tr, y_tr)
    svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1).fit(Xs_tr, y_tr)
    def met(yt, yp):
        return {"r2":round(float(r2_score(yt,yp)),4),
                "rmse":round(float(np.sqrt(mean_squared_error(yt,yp))),4)}
    lr_p=lr.predict(Xs_te); rf_p=rf.predict(Xs_te); sv_p=svr.predict(Xs_te)
    return dict(
        models=dict(lr=lr,rf=rf,svr=svr), scaler=scaler, features=FEATURES,
        split=split,
        preds=dict(lr=lr_p,rf=rf_p,svr=sv_p),
        actuals=y_te,
        metrics=dict(lr=met(y_te,lr_p),rf=met(y_te,rf_p),svr=met(y_te,sv_p)),
        y_train=y_tr, y_test=y_te,
        dates_train=df.index[:split], dates_test=df.index[split:],
    )

def ensemble_forecast(m: dict, df: pd.DataFrame, days=30) -> dict:
    last   = df[m["features"]].iloc[-1].values.reshape(1,-1)
    lasts  = m["scaler"].transform(last)
    lr_p   = m["models"]["lr"].predict(lasts)[0]
    rf_p   = m["models"]["rf"].predict(lasts)[0]
    sv_p   = m["models"]["svr"].predict(lasts)[0]
    ens    = (lr_p+rf_p+sv_p)/3
    current= df["Close"].iloc[-1]
    drift  = (ens-current)/days
    vol    = df["Daily_Return"].std()
    rng    = np.random.default_rng(42)
    future = []
    for _ in range(days):
        current += drift + rng.normal(0, vol*current)
        future.append(round(float(current),4))
    last_dt  = df.index[-1]
    f_dates  = pd.date_range(last_dt+pd.Timedelta(days=1), periods=days, freq="B")
    return dict(lr=round(lr_p,4),rf=round(rf_p,4),svr=round(sv_p,4),
                ensemble=round(ens,4),future=future,future_dates=f_dates)

def technical_indicators(df: pd.DataFrame) -> dict:
    c = df["Close"]
    d   = c.diff()
    rs  = d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    rsi = 100 - 100/(1+rs)
    macd   = c.ewm(span=12,adjust=False).mean() - c.ewm(span=26,adjust=False).mean()
    signal = macd.ewm(span=9,adjust=False).mean()
    sma    = c.rolling(20).mean()
    std    = c.rolling(20).std()
    return dict(rsi=rsi,macd=macd,signal=signal,hist=macd-signal,
                bb_upper=sma+2*std,bb_mid=sma,bb_lower=sma-2*std)

def investment_signal(df: pd.DataFrame, ens_price: float) -> dict:
    c    = df["Close"].iloc[-1]
    ma50 = df["MA50"].iloc[-1]
    ma200= df["MA200"].iloc[-1]
    vol  = df["Daily_Return"].std()*np.sqrt(252)
    score= sum([c>ma50,c>ma200,ens_price>c,vol<0.30])
    sig  = "BUY" if score>=3 else ("HOLD" if score==2 else "SELL")
    risk = "LOW" if vol<0.20 else ("MEDIUM" if vol<0.40 else "HIGH")
    return dict(signal=sig,score=score,risk=risk,
                risk_score=round(min(vol*100,100),1),annual_vol=round(vol*100,2))

# ═══════════════════════════════════════════════════════════
#  CHART HELPERS
# ═══════════════════════════════════════════════════════════
_AXIS = dict(showgrid=True,gridcolor=PAL["border"],gridwidth=0.5,zeroline=False,tickfont=dict(size=10))

def _layout(**kw):
    base = dict(
        paper_bgcolor=PAL["bg2"], plot_bgcolor=PAL["bg"],
        font=dict(family="monospace", color=PAL["text2"], size=11),
        margin=dict(l=48,r=20,t=36,b=36),
        legend=dict(bgcolor="rgba(0,0,0,0)",borderwidth=0,font=dict(size=10,color=PAL["text2"])),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=PAL["bg2"],bordercolor=PAL["border"],font=dict(size=11,color=PAL["text"])),
    )
    # merge axis defaults with any overrides passed in kw
    xaxis_kw = kw.pop("xaxis", {})
    yaxis_kw = kw.pop("yaxis", {})
    base["xaxis"] = {**_AXIS, **xaxis_kw}
    base["yaxis"] = {**_AXIS, **yaxis_kw}
    base.update(kw)
    return base

CFG = {"displayModeBar": False}

def price_ma_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=df["Close"],name="Price",
        line=dict(color=PAL["accent"],width=1.8),fill="tozeroy",fillcolor="rgba(232,201,109,0.06)"))
    fig.add_trace(go.Scatter(x=df.index,y=df["MA50"],name="MA 50",
        line=dict(color=PAL["blue"],width=1.3,dash="dot")))
    fig.add_trace(go.Scatter(x=df.index,y=df["MA200"],name="MA 200",
        line=dict(color=PAL["pink"],width=1.3,dash="dash")))
    fig.update_layout(**_layout(height=360))
    return fig

def candlestick_chart(df, tail=90):
    d = df.tail(tail)
    fig = go.Figure(go.Candlestick(
        x=d.index,open=d["Open"],high=d["High"],low=d["Low"],close=d["Close"],
        increasing_line_color=PAL["green"],decreasing_line_color=PAL["red"],
        increasing_fillcolor="rgba(92,245,160,0.25)",decreasing_fillcolor="rgba(247,110,110,0.25)"))
    fig.update_layout(**_layout(height=340))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def split_chart(m):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m["dates_train"],y=m["y_train"],name="Train (80%)",
        line=dict(color=PAL["blue"],width=1.6),fill="tozeroy",fillcolor="rgba(110,231,247,0.05)"))
    fig.add_trace(go.Scatter(x=m["dates_test"],y=m["y_test"],name="Test (20%)",
        line=dict(color=PAL["accent"],width=1.6),fill="tozeroy",fillcolor="rgba(232,201,109,0.07)"))
    fig.update_layout(**_layout(height=300))
    return fig

def model_compare_chart(m):
    dates=m["dates_test"]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=m["actuals"],name="Actual",line=dict(color=PAL["text"],width=2)))
    fig.add_trace(go.Scatter(x=dates,y=m["preds"]["lr"],name="Lin Reg",line=dict(color=PAL["blue"],width=1.3,dash="dot")))
    fig.add_trace(go.Scatter(x=dates,y=m["preds"]["rf"],name="Rand Forest",line=dict(color=PAL["green"],width=1.3,dash="dot")))
    fig.add_trace(go.Scatter(x=dates,y=m["preds"]["svr"],name="SVR",line=dict(color=PAL["purple"],width=1.3,dash="dot")))
    fig.update_layout(**_layout(height=300))
    return fig

def bollinger_chart(df, tech, tail=150):
    d=df.tail(tail); tu=tech["bb_upper"].tail(tail); tm=tech["bb_mid"].tail(tail); tl=tech["bb_lower"].tail(tail)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d.index,y=tu,name="Upper",line=dict(color=PAL["purple"],width=1,dash="dot")))
    fig.add_trace(go.Scatter(x=d.index,y=tl,name="Lower",fill="tonexty",fillcolor="rgba(176,139,255,0.07)",line=dict(color=PAL["purple"],width=1,dash="dot")))
    fig.add_trace(go.Scatter(x=d.index,y=tm,name="SMA20",line=dict(color=PAL["blue"],width=1.2,dash="dash")))
    fig.add_trace(go.Scatter(x=d.index,y=d["Close"],name="Price",line=dict(color=PAL["accent"],width=1.8)))
    fig.update_layout(**_layout(height=320))
    return fig

def rsi_chart(tech, tail=120):
    rsi=tech["rsi"].dropna().tail(tail)
    fig=go.Figure()
    fig.add_hrect(y0=70,y1=100,fillcolor="rgba(247,110,110,0.07)",line_width=0)
    fig.add_hrect(y0=0,y1=30,fillcolor="rgba(92,245,160,0.07)",line_width=0)
    fig.add_hline(y=70,line_dash="dot",line_color=PAL["red"],line_width=1)
    fig.add_hline(y=30,line_dash="dot",line_color=PAL["green"],line_width=1)
    fig.add_trace(go.Scatter(x=rsi.index,y=rsi.values,name="RSI",
        line=dict(color=PAL["blue"],width=1.8),fill="tozeroy",fillcolor="rgba(110,231,247,0.06)"))
    fig.update_layout(**_layout(height=260,yaxis=dict(range=[0,100],gridcolor=PAL["border"])))
    return fig

def macd_chart(tech, tail=120):
    macd=tech["macd"].dropna().tail(tail); sig=tech["signal"].dropna().tail(tail); hist=tech["hist"].dropna().tail(tail)
    colors=[PAL["green"] if v>=0 else PAL["red"] for v in hist.values]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=hist.index,y=hist.values,name="Histogram",marker_color=colors,opacity=0.6))
    fig.add_trace(go.Scatter(x=macd.index,y=macd.values,name="MACD",line=dict(color=PAL["blue"],width=1.6)))
    fig.add_trace(go.Scatter(x=sig.index,y=sig.values,name="Signal",line=dict(color=PAL["pink"],width=1.4)))
    fig.update_layout(**_layout(height=260,barmode="overlay"))
    return fig

def forecast_chart(df, fc):
    hist=df["Close"].tail(60)
    fut=pd.Series(fc["future"],index=fc["future_dates"])
    upper=[v*1.05 for v in fc["future"]]; lower=[v*0.95 for v in fc["future"]]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hist.index,y=hist.values,name="Historical",line=dict(color=PAL["text2"],width=1.6)))
    fig.add_trace(go.Scatter(
        x=list(fc["future_dates"])+list(fc["future_dates"])[::-1],y=upper+lower[::-1],
        fill="toself",fillcolor="rgba(232,201,109,0.08)",line=dict(width=0),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut.index,y=fut.values,name="AI Forecast",
        line=dict(color=PAL["accent"],width=2.2,dash="dot"),mode="lines+markers",marker=dict(size=3,color=PAL["accent"])))
    fig.update_layout(**_layout(height=340))
    return fig

def backtest_chart(m):
    dates=m["dates_test"]; ens=(m["preds"]["lr"]+m["preds"]["rf"]+m["preds"]["svr"])/3
    diff=m["actuals"]-ens
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=m["actuals"],name="Actual",line=dict(color=PAL["text"],width=2)))
    fig.add_trace(go.Scatter(x=dates,y=ens,name="Ensemble",line=dict(color=PAL["accent"],width=1.8,dash="dash")))
    fig.add_trace(go.Bar(x=dates,y=diff,name="Error",
        marker_color=[PAL["green"] if v>=0 else PAL["red"] for v in diff],opacity=0.4,yaxis="y2"))
    fig.update_layout(**_layout(height=320,
        yaxis2=dict(overlaying="y",side="right",showgrid=False,tickfont=dict(size=9))))
    return fig

def volatility_histogram(df):
    returns=df["Daily_Return"].dropna()*100
    mu,sigma=returns.mean(),returns.std()
    xs=np.linspace(returns.min(),returns.max(),200)
    ys=(np.exp(-0.5*((xs-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi)))*len(returns)*(returns.max()-returns.min())/50
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=returns,nbinsx=50,name="Daily Returns",
        marker_color=PAL["blue"],marker_line=dict(color=PAL["bg"],width=0.5),opacity=0.8))
    fig.add_trace(go.Scatter(x=xs,y=ys,name="Normal Dist",line=dict(color=PAL["accent"],width=1.8)))
    fig.update_layout(**_layout(height=300))
    return fig

def rolling_returns_chart(df, tail=252):
    r=df["Daily_Return"].dropna().tail(tail)*100
    colors=[PAL["green"] if v>=0 else PAL["red"] for v in r.values]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=r.index,y=r.values,name="Daily Return %",marker_color=colors,opacity=0.75))
    fig.update_layout(**_layout(height=260))
    return fig

def portfolio_growth_chart(df, investment):
    base=df["Close"].iloc[0]; growth=df["Close"]/base*investment
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=growth.index,y=growth.values,name="Portfolio Value",
        line=dict(color=PAL["green"],width=2),fill="tozeroy",fillcolor="rgba(92,245,160,0.07)"))
    fig.add_hline(y=investment,line_dash="dot",line_color=PAL["text2"],line_width=1,
        annotation_text=f"Initial ${investment:,.0f}",annotation_font_color=PAL["text2"])
    fig.update_layout(**_layout(height=300,yaxis=dict(tickprefix="$",gridcolor=PAL["border"])))
    return fig

def metrics_bar_chart(metrics):
    models=["Linear Reg","Random Forest","SVR"]
    r2s=[metrics["lr"]["r2"],metrics["rf"]["r2"],metrics["svr"]["r2"]]
    rmses=[metrics["lr"]["rmse"],metrics["rf"]["rmse"],metrics["svr"]["rmse"]]
    colors=[PAL["blue"],PAL["green"],PAL["purple"]]
    fig=make_subplots(rows=1,cols=2,subplot_titles=["R² Score (higher = better)","RMSE (lower = better)"])
    for i,(m,r,e) in enumerate(zip(models,r2s,rmses)):
        fig.add_trace(go.Bar(x=[m],y=[r],name=m,marker_color=colors[i],showlegend=False),row=1,col=1)
        fig.add_trace(go.Bar(x=[m],y=[e],name=m,marker_color=colors[i],showlegend=False),row=1,col=2)
    fig.update_layout(**_layout(height=280))
    fig.update_annotations(font_color=PAL["text2"],font_size=11)
    return fig

def portfolio_donut(results):
    labels=[r["name"] for r in results]; amounts=[r["projected"] for r in results]
    colors=[PAL["accent"],PAL["blue"],PAL["green"],PAL["purple"],PAL["pink"],PAL["red"]]
    fig=go.Figure(go.Pie(labels=labels,values=amounts,hole=0.55,
        marker_colors=colors[:len(labels)],textfont=dict(size=11,color=PAL["text"]),
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<extra></extra>"))
    fig.update_layout(**_layout(height=300,showlegend=True,margin=dict(l=20,r=20,t=20,b=20)))
    return fig

# ═══════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

.stApp { background: #070B14 !important; }
section[data-testid="stSidebar"] {
  background: #0D1321 !important;
  border-right: 1px solid #1E2A3A !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.8rem 2.2rem 3rem !important; max-width: 100% !important; }
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; color: #E4EAF5 !important; }

[data-testid="metric-container"] {
  background: #0D1321 !important; border: 1px solid #1E2A3A !important;
  border-radius: 12px !important; padding: 16px 18px !important;
}
[data-testid="metric-container"] label {
  font-size: 11px !important; text-transform: uppercase !important;
  color: #8496AE !important; font-family: 'JetBrains Mono', monospace !important;
  letter-spacing: 0.8px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important; font-size: 1.5rem !important;
  font-weight: 700 !important; color: #E4EAF5 !important;
}

.stTabs [data-baseweb="tab-list"] {
  background: #0D1321 !important; border-radius: 10px !important;
  padding: 4px !important; gap: 4px !important; border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; border-radius: 7px !important;
  color: #8496AE !important; padding: 8px 20px !important; border: none !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(232,201,109,0.12) !important; color: #E8C96D !important;
}

.stTextInput input, .stNumberInput input {
  background: #0D1321 !important; border: 1px solid #1E2A3A !important;
  border-radius: 8px !important; color: #E4EAF5 !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: #E8C96D !important;
  box-shadow: 0 0 0 2px rgba(232,201,109,0.15) !important;
}

.stButton > button {
  background: #E8C96D !important; color: #0D1010 !important;
  border: none !important; border-radius: 8px !important;
  font-weight: 600 !important; padding: 10px 22px !important;
}
.stButton > button:hover { background: #F0A64A !important; }

.stRadio > div { gap: 4px !important; }
.stRadio label {
  background: transparent !important; border-radius: 8px !important;
  padding: 8px 14px !important; color: #8496AE !important;
  font-size: 13.5px !important; border: 1px solid transparent !important;
}
.stRadio label:hover { color: #E4EAF5 !important; background: #131A2A !important; }

hr { border-color: #1E2A3A !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0D1321; }
::-webkit-scrollbar-thumb { background: #1E2A3A; border-radius: 4px; }

.nq-card {
  background: #0D1321; border: 1px solid #1E2A3A;
  border-radius: 12px; padding: 18px 20px; margin-bottom: 4px;
}
.nq-card-title {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: #8496AE; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px;
}
.sig-buy  { background:rgba(92,245,160,0.12);color:#5CF5A0;border:1px solid rgba(92,245,160,0.3);border-radius:8px;padding:6px 18px;font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;letter-spacing:2px;display:inline-block; }
.sig-hold { background:rgba(232,201,109,0.12);color:#E8C96D;border:1px solid rgba(232,201,109,0.3);border-radius:8px;padding:6px 18px;font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;letter-spacing:2px;display:inline-block; }
.sig-sell { background:rgba(247,110,110,0.12);color:#F76E6E;border:1px solid rgba(247,110,110,0.3);border-radius:8px;padding:6px 18px;font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;letter-spacing:2px;display:inline-block; }
.nq-chart-label { font-family:'JetBrains Mono',monospace;font-size:11px;color:#8496AE;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px; }
.nq-hero { background:linear-gradient(135deg,#0D1321 0%,#131A2A 60%,#0D1321 100%);border:1px solid #1E2A3A;border-radius:16px;padding:40px 48px;margin-bottom:28px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 16px 16px;border-bottom:1px solid #1E2A3A;margin-bottom:12px'>
      <div style='background:linear-gradient(135deg,#E8C96D,#F0A64A);color:#000;
                  font-family:Syne,sans-serif;font-weight:800;font-size:12px;
                  width:36px;height:36px;border-radius:9px;display:flex;
                  align-items:center;justify-content:center;margin-bottom:8px'>NQ</div>
      <span style='font-family:Syne,sans-serif;font-weight:700;font-size:16px;color:#E4EAF5;display:block'>NexusQuant</span>
      <span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#8496AE'>AI Stock Intelligence</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Search Company**")
    company_input = st.text_input("", placeholder="Apple, Tesla, NVIDIA…",
                                  label_visibility="collapsed", key="company_search")

    st.markdown("**Investment ($)**")
    investment = st.number_input("", min_value=100, value=10000, step=500,
                                 label_visibility="collapsed")

    st.markdown("**Period**")
    period = st.selectbox("", ["1y","2y","5y","max"], index=1, label_visibility="collapsed")

    analyze_btn = st.button("⚡  Analyze", use_container_width=True)

    st.divider()
    st.markdown("**Navigate**")
    page = st.radio("", [
        "🏠  Overview","📊  ML Models","📉  Technical",
        "🔮  Predictions","🌊  Volatility","💼  Portfolio",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("<p style='font-size:11px;color:#4a5a70;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px'>Quick Pick</p>", unsafe_allow_html=True)
    chips = ["Apple","Tesla","NVIDIA","Microsoft","Amazon"]
    cols_c = st.columns(2)
    for i, c in enumerate(chips):
        if cols_c[i%2].button(c, key=f"chip_{c}", use_container_width=True):
            st.session_state["company_search"] = c
            st.session_state["_auto"] = True
            st.rerun()

    st.markdown("""
    <div style='display:flex;align-items:center;gap:8px;margin-top:24px;padding:8px 4px'>
      <span style='width:7px;height:7px;border-radius:50%;background:#5CF5A0;
                   box-shadow:0 0 8px #5CF5A0;display:inline-block;
                   animation:blink 2s infinite'></span>
      <span style='font-size:12px;color:#8496AE;font-family:JetBrains Mono,monospace'>Markets Live</span>
    </div>
    <style>@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}</style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  SESSION STATE & AUTO ANALYZE
# ═══════════════════════════════════════════════════════════
if "data" not in st.session_state:
    st.session_state["data"] = None

if st.session_state.get("_auto"):
    del st.session_state["_auto"]
    analyze_btn   = True
    company_input = st.session_state.get("company_search","")

# ═══════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════
if analyze_btn and company_input.strip():
    with st.spinner("Fetching data & training models…"):
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
                ticker=ticker, df=df, info=info, models=models,
                tech=tech, fc=fc, sig=sig, investment=investment,
            )
        except Exception as e:
            st.error(f"⚠ {e}")

# ═══════════════════════════════════════════════════════════
#  WELCOME SCREEN
# ═══════════════════════════════════════════════════════════
if st.session_state["data"] is None:
    st.markdown("""
    <div class='nq-hero'>
      <h1 style='font-size:2.4rem;margin-bottom:8px'>AI-Powered Stock Intelligence</h1>
      <p style='color:#8496AE;font-size:15px;line-height:1.7'>
        Search any company to unlock ML predictions, technical indicators,<br>
        risk analysis, and portfolio simulations — all in one dashboard.
      </p>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,icon,title,desc in [
        (c1,"🤖","3 ML Models","Linear Regression, Random Forest & SVR trained on historical data"),
        (c2,"📡","Technical Suite","RSI, MACD, Bollinger Bands & Moving Averages"),
        (c3,"💼","Portfolio Sim","Simulate investment growth and multi-stock allocations"),
    ]:
        col.markdown(f"""
        <div class='nq-card'>
          <div style='font-size:1.6rem;margin-bottom:10px'>{icon}</div>
          <div style='font-family:Syne,sans-serif;font-weight:700;font-size:15px;color:#E4EAF5;margin-bottom:6px'>{title}</div>
          <div style='font-size:13px;color:#8496AE;line-height:1.6'>{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════
#  UNPACK
# ═══════════════════════════════════════════════════════════
D      = st.session_state["data"]
ticker = D["ticker"]; df = D["df"]; info = D["info"]
models = D["models"]; tech = D["tech"]; fc = D["fc"]
sig    = D["sig"];    inv  = D["investment"]
close_now  = df["Close"].iloc[-1]
close_prev = df["Close"].iloc[-2]
day_chg    = (close_now - close_prev) / close_prev * 100

# Header
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;
            background:#0D1321;border:1px solid #1E2A3A;border-radius:12px;
            padding:14px 22px;margin-bottom:24px'>
  <div>
    <span style='font-family:Syne,sans-serif;font-weight:800;font-size:1.4rem;color:#E4EAF5'>{info.get("name",ticker)}</span>
    <span style='font-family:JetBrains Mono,monospace;font-size:12px;color:#4a5a70;margin-left:12px'>{ticker}</span>
    <span style='font-size:12px;color:#8496AE;margin-left:10px'>{info.get("sector","N/A")} · {info.get("industry","N/A")}</span>
  </div>
  <div style='display:flex;align-items:center;gap:20px'>
    <span style='font-family:Syne,sans-serif;font-weight:700;font-size:1.5rem;color:#E4EAF5'>${close_now:,.2f}</span>
    <span style='font-size:13px;color:{"#5CF5A0" if day_chg>=0 else "#F76E6E"};font-family:JetBrains Mono,monospace'>
      {"▲" if day_chg>=0 else "▼"} {abs(day_chg):.2f}%
    </span>
  </div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  OVERVIEW
# ═══════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("### Overview")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    mc=info.get("market_cap")
    mc_fmt=f"${mc/1e12:.2f}T" if mc and mc>1e12 else (f"${mc/1e9:.1f}B" if mc else "N/A")
    k1.metric("Current Price",f"${close_now:,.2f}")
    k2.metric("Day Change",f"{day_chg:+.2f}%")
    k3.metric("52W High",f"${info['52w_high']:,.2f}" if info.get("52w_high") else "N/A")
    k4.metric("52W Low",f"${info['52w_low']:,.2f}"   if info.get("52w_low")  else "N/A")
    k5.metric("Market Cap",mc_fmt)
    k6.metric("P/E Ratio",f"{info['pe']:.1f}" if info.get("pe") else "N/A")
    st.markdown("<br>", unsafe_allow_html=True)

    sa,sb,sc,sd = st.columns([1,1,1,2])
    sig_cls={"BUY":"sig-buy","HOLD":"sig-hold","SELL":"sig-sell"}[sig["signal"]]
    sa.markdown(f"<div class='{sig_cls}'>{sig['signal']}</div>", unsafe_allow_html=True)
    sb.metric("Risk Level",sig["risk"])
    sc.metric("Annual Vol",f"{sig['annual_vol']}%")
    sd.metric("Confidence",f"{sig['score']}/4 signals bullish")
    st.markdown("<br>", unsafe_allow_html=True)

    t1,t2 = st.tabs(["📈 Price + Moving Averages","🕯 Candlestick (90 days)"])
    with t1: st.plotly_chart(price_ma_chart(df), use_container_width=True, config=CFG)
    with t2: st.plotly_chart(candlestick_chart(df), use_container_width=True, config=CFG)

    desc=info.get("description","")
    if desc:
        with st.expander("About this company"):
            st.markdown(f"<p style='font-size:13px;color:#8496AE;line-height:1.7'>{desc[:800]}…</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  ML MODELS
# ═══════════════════════════════════════════════════════════
elif "ML Models" in page:
    st.markdown("### Machine Learning Models")
    for col,key,label in zip(st.columns(3),["lr","rf","svr"],["Linear Regression","Random Forest","SVR"]):
        m=models["metrics"][key]
        r2c="#5CF5A0" if m["r2"]>=.9 else ("#E8C96D" if m["r2"]>=.7 else "#F76E6E")
        col.markdown(f"""
        <div class='nq-card'>
          <div class='nq-card-title'>{label}</div>
          <div style='display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1E2A3A'>
            <span style='font-size:12px;color:#8496AE'>R² Score</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:13px;color:{r2c};font-weight:600'>{m["r2"]}</span>
          </div>
          <div style='display:flex;justify-content:space-between;padding:8px 0'>
            <span style='font-size:12px;color:#8496AE'>RMSE</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:13px;color:#E4EAF5'>{m["rmse"]}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    ca,cb = st.columns(2)
    with ca:
        st.markdown("<div class='nq-chart-label'>Train / Test Split</div>", unsafe_allow_html=True)
        st.plotly_chart(split_chart(models), use_container_width=True, config=CFG)
    with cb:
        st.markdown("<div class='nq-chart-label'>Model Predictions vs Actual</div>", unsafe_allow_html=True)
        st.plotly_chart(model_compare_chart(models), use_container_width=True, config=CFG)
    st.markdown("<div class='nq-chart-label'>Performance Comparison</div>", unsafe_allow_html=True)
    st.plotly_chart(metrics_bar_chart(models["metrics"]), use_container_width=True, config=CFG)

# ═══════════════════════════════════════════════════════════
#  TECHNICAL
# ═══════════════════════════════════════════════════════════
elif "Technical" in page:
    st.markdown("### Technical Indicators")
    st.markdown("<div class='nq-chart-label'>Bollinger Bands</div>", unsafe_allow_html=True)
    st.plotly_chart(bollinger_chart(df, tech), use_container_width=True, config=CFG)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='nq-chart-label'>RSI (14)</div>", unsafe_allow_html=True)
        st.plotly_chart(rsi_chart(tech), use_container_width=True, config=CFG)
    with c2:
        st.markdown("<div class='nq-chart-label'>MACD</div>", unsafe_allow_html=True)
        st.plotly_chart(macd_chart(tech), use_container_width=True, config=CFG)
    last_rsi=tech["rsi"].dropna().iloc[-1]
    rsi_color="#F76E6E" if last_rsi>70 else ("#5CF5A0" if last_rsi<30 else "#E8C96D")
    rsi_label="Overbought ⚠" if last_rsi>70 else ("Oversold 🟢" if last_rsi<30 else "Neutral")
    r1,r2 = st.columns(2)
    r1.markdown(f"""<div class='nq-card'><div class='nq-card-title'>Current RSI</div>
      <span style='font-family:Syne,sans-serif;font-weight:700;font-size:1.6rem;color:{rsi_color}'>{last_rsi:.1f}</span>
      <span style='font-size:12px;color:#8496AE;margin-left:10px'>{rsi_label}</span></div>""", unsafe_allow_html=True)
    r2.markdown(f"""<div class='nq-card'><div class='nq-card-title'>MACD Signal</div>
      <span style='font-family:Syne,sans-serif;font-weight:700;font-size:1.2rem;
                   color:{"#5CF5A0" if tech["hist"].iloc[-1]>0 else "#F76E6E"}'>
        {"Bullish Cross 🟢" if tech["hist"].iloc[-1]>0 else "Bearish Cross 🔴"}
      </span></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PREDICTIONS
# ═══════════════════════════════════════════════════════════
elif "Predictions" in page:
    st.markdown("### AI Predictions & Forecasting")
    for col,key,label in zip(st.columns(4),["lr","rf","svr","ensemble"],
                              ["Linear Reg","Random Forest","SVR","⚡ Ensemble"]):
        val=fc[key]; chg=(val-close_now)/close_now*100
        col.markdown(f"""
        <div class='nq-card' style='{"border-color:rgba(232,201,109,0.4);background:rgba(232,201,109,0.04);" if key=="ensemble" else ""}'>
          <div class='nq-card-title'>{label}</div>
          <div style='font-family:Syne,sans-serif;font-weight:700;font-size:1.3rem;
                      color:{"#E8C96D" if key=="ensemble" else "#E4EAF5"}'>${val}</div>
          <div style='font-size:11px;color:{"#5CF5A0" if chg>=0 else "#F76E6E"};
                      font-family:JetBrains Mono,monospace;margin-top:4px'>{chg:+.2f}% vs now</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='nq-chart-label'>30-Day Ensemble Forecast</div>", unsafe_allow_html=True)
    st.plotly_chart(forecast_chart(df, fc), use_container_width=True, config=CFG)
    st.markdown("<div class='nq-chart-label'>Backtesting — Predicted vs Actual</div>", unsafe_allow_html=True)
    st.plotly_chart(backtest_chart(models), use_container_width=True, config=CFG)
    ens_bt=(models["preds"]["lr"]+models["preds"]["rf"]+models["preds"]["svr"])/3
    bt_r2=r2_score(models["actuals"],ens_bt)
    bt_rmse=np.sqrt(mean_squared_error(models["actuals"],ens_bt))
    b1,b2=st.columns(2)
    b1.metric("Ensemble R² (Backtest)",f"{bt_r2:.4f}")
    b2.metric("Ensemble RMSE (Backtest)",f"${bt_rmse:.2f}")

# ═══════════════════════════════════════════════════════════
#  VOLATILITY
# ═══════════════════════════════════════════════════════════
elif "Volatility" in page:
    st.markdown("### Volatility & Risk Analysis")
    daily_vol=df["Daily_Return"].std()*100; annual_vol=daily_vol*np.sqrt(252)
    skew=float(df["Daily_Return"].skew()); kurt=float(df["Daily_Return"].kurtosis())
    v1,v2,v3,v4=st.columns(4)
    v1.metric("Daily Volatility",f"{daily_vol:.2f}%")
    v2.metric("Annual Volatility",f"{annual_vol:.2f}%")
    v3.metric("Return Skewness",f"{skew:.3f}")
    v4.metric("Excess Kurtosis",f"{kurt:.3f}")
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("<div class='nq-chart-label'>Daily Returns Distribution</div>", unsafe_allow_html=True)
        st.plotly_chart(volatility_histogram(df), use_container_width=True, config=CFG)
    with c2:
        st.markdown("<div class='nq-chart-label'>Daily Returns (1 Year)</div>", unsafe_allow_html=True)
        st.plotly_chart(rolling_returns_chart(df), use_container_width=True, config=CFG)
    rs=sig["risk_score"]; rs_color="#5CF5A0" if rs<30 else ("#E8C96D" if rs<60 else "#F76E6E")
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
          <div style='width:{min(rs,100)}%;height:100%;background:{rs_color};border-radius:99px'></div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PORTFOLIO
# ═══════════════════════════════════════════════════════════
elif "Portfolio" in page:
    st.markdown("### Portfolio Simulator")
    base=df["Close"].iloc[0]; curr=df["Close"].iloc[-1]
    growth=inv*(curr/base); profit=growth-inv; ret_pct=profit/inv*100
    p1,p2,p3=st.columns(3)
    p1.metric("Initial Investment",f"${inv:,.2f}")
    p2.metric("Current Value",f"${growth:,.2f}",delta=f"${profit:+,.2f}")
    p3.metric("Total Return",f"{ret_pct:+.2f}%")
    st.markdown(f"<div class='nq-chart-label' style='margin-top:12px'>Investment Growth — {info.get('name',ticker)}</div>", unsafe_allow_html=True)
    st.plotly_chart(portfolio_growth_chart(df, inv), use_container_width=True, config=CFG)
    st.divider()

    st.markdown("#### Multi-Stock Allocation Simulator")
    st.caption("Add stocks and simulate your portfolio across multiple companies.")

    if "portfolio_rows" not in st.session_state:
        st.session_state["portfolio_rows"]=[
            {"name":"Apple","amount":5000},
            {"name":"Microsoft","amount":3000},
            {"name":"NVIDIA","amount":2000},
        ]

    rows=st.session_state["portfolio_rows"]; updated=[]
    for i,row in enumerate(rows):
        c1,c2,c3=st.columns([3,2,1])
        n=c1.text_input(f"Co{i}",value=row["name"],key=f"pn_{i}",label_visibility="collapsed")
        a=c2.number_input(f"Amt{i}",value=float(row["amount"]),min_value=0.0,key=f"pa_{i}",label_visibility="collapsed")
        if c3.button("✕",key=f"del_{i}"):
            rows.pop(i); st.rerun()
        updated.append({"name":n,"amount":a})
    st.session_state["portfolio_rows"]=updated

    col_add,col_run=st.columns([1,4])
    if col_add.button("＋ Add Stock"):
        st.session_state["portfolio_rows"].append({"name":"","amount":1000}); st.rerun()

    if col_run.button("🚀  Run Portfolio Simulation", use_container_width=True):
        results=[]
        with st.spinner("Fetching multi-stock data…"):
            for row in updated:
                if not row["name"].strip(): continue
                try:
                    t2=resolve_ticker(row["name"]); d2=fetch_stock(t2,"1y")
                    ret=(d2["Close"].iloc[-1]-d2["Close"].iloc[0])/d2["Close"].iloc[0]
                    results.append({"name":row["name"],"ticker":t2,"amount":row["amount"],
                                    "projected":round(row["amount"]*(1+ret),2),"return_pct":round(ret*100,2)})
                except Exception: pass
        if results: st.session_state["portfolio_results"]=results

    if "portfolio_results" in st.session_state:
        res=st.session_state["portfolio_results"]
        total_inv=sum(r["amount"] for r in res); total_proj=sum(r["projected"] for r in res)
        total_ret=(total_proj-total_inv)/total_inv*100
        t1,t2,t3=st.columns(3)
        t1.metric("Total Invested",f"${total_inv:,.2f}")
        t2.metric("Total Projected",f"${total_proj:,.2f}",delta=f"${total_proj-total_inv:+,.2f}")
        t3.metric("Portfolio Return",f"{total_ret:+.2f}%")
        rr,dd=st.columns([2,1])
        with rr:
            st.markdown("<br>", unsafe_allow_html=True)
            for r in res:
                pos=r["return_pct"]>=0
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
                               color:{"#5CF5A0" if pos else "#F76E6E"}'>${r["projected"]:,.2f}</span>
                  <span style='font-family:JetBrains Mono,monospace;font-size:12px;padding:3px 8px;
                               border-radius:4px;
                               background:{"rgba(92,245,160,0.1)" if pos else "rgba(247,110,110,0.1)"};
                               color:{"#5CF5A0" if pos else "#F76E6E"}'>{r["return_pct"]:+.2f}%</span>
                </div>""", unsafe_allow_html=True)
        with dd:
            st.plotly_chart(portfolio_donut(res), use_container_width=True, config=CFG)
