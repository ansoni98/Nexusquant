"""
NexusQuant India - AI Stock Intelligence | NSE/BSE | INR
"""
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
st.set_page_config(page_title="NexusQuant India",page_icon="📈",layout="wide",initial_sidebar_state="expanded")
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

PAL = dict(bg="#070B14",bg2="#0D1321",bg3="#131A2A",border="#1E2A3A",text="#E4EAF5",text2="#8496AE",
    accent="#E8C96D",accent2="#F0A64A",blue="#6EE7F7",pink="#F76EA0",green="#5CF5A0",red="#F76E6E",purple="#B08BFF")

COMPANY_MAP = {
    "tcs":"TCS.NS","tata consultancy":"TCS.NS","tata consultancy services":"TCS.NS",
    "infosys":"INFY.NS","infy":"INFY.NS",
    "wipro":"WIPRO.NS",
    "hcl":"HCLTECH.NS","hcl tech":"HCLTECH.NS","hcl technologies":"HCLTECH.NS",
    "tech mahindra":"TECHM.NS","tech m":"TECHM.NS",
    "mphasis":"MPHASIS.NS",
    "ltimindtree":"LTIM.NS","lti mindtree":"LTIM.NS","lti":"LTIM.NS",
    "coforge":"COFORGE.NS",
    "persistent":"PERSISTENT.NS","persistent systems":"PERSISTENT.NS",
    "oracle financial":"OFSS.NS","ofss":"OFSS.NS",
    "hdfc bank":"HDFCBANK.NS","hdfc":"HDFCBANK.NS","hdfcbank":"HDFCBANK.NS",
    "icici bank":"ICICIBANK.NS","icici":"ICICIBANK.NS",
    "sbi":"SBIN.NS","state bank":"SBIN.NS","state bank of india":"SBIN.NS",
    "axis bank":"AXISBANK.NS","axis":"AXISBANK.NS",
    "kotak":"KOTAKBANK.NS","kotak mahindra":"KOTAKBANK.NS","kotak bank":"KOTAKBANK.NS",
    "indusind":"INDUSINDBK.NS","indusind bank":"INDUSINDBK.NS",
    "yes bank":"YESBANK.NS",
    "federal bank":"FEDERALBNK.NS",
    "bank of baroda":"BANKBARODA.NS","bob":"BANKBARODA.NS",
    "punjab national bank":"PNB.NS","pnb":"PNB.NS",
    "canara bank":"CANARABANK.NS",
    "union bank":"UNIONBANK.NS",
    "idfc first":"IDFCFIRSTB.NS","idfc":"IDFCFIRSTB.NS",
    "bajaj finance":"BAJFINANCE.NS",
    "bajaj finserv":"BAJAJFINSV.NS",
    "hdfc life":"HDFCLIFE.NS",
    "sbi life":"SBILIFE.NS",
    "lic":"LICI.NS",
    "muthoot finance":"MUTHOOTFIN.NS",
    "shriram finance":"SHRIRAMFIN.NS",
    "reliance":"RELIANCE.NS","reliance industries":"RELIANCE.NS","ril":"RELIANCE.NS",
    "ongc":"ONGC.NS","oil and natural gas":"ONGC.NS",
    "ioc":"IOC.NS","indian oil":"IOC.NS","indian oil corporation":"IOC.NS",
    "bpcl":"BPCL.NS","bharat petroleum":"BPCL.NS",
    "hpcl":"HPCL.NS","hindustan petroleum":"HPCL.NS",
    "coal india":"COALINDIA.NS",
    "gail":"GAIL.NS",
    "petronet":"PETRONET.NS",
    "adani green":"ADANIGREEN.NS",
    "adani ports":"ADANIPORTS.NS",
    "adani enterprises":"ADANIENT.NS",
    "adani power":"ADANIPOWER.NS",
    "tata power":"TATAPOWER.NS",
    "ntpc":"NTPC.NS",
    "power grid":"POWERGRID.NS",
    "tata motors":"TATAMOTORS.NS",
    "maruti":"MARUTI.NS","maruti suzuki":"MARUTI.NS",
    "bajaj auto":"BAJAJ-AUTO.NS",
    "hero motocorp":"HEROMOTOCO.NS","hero":"HEROMOTOCO.NS",
    "mahindra":"M&M.NS","m&m":"M&M.NS","mahindra and mahindra":"M&M.NS",
    "eicher motors":"EICHERMOT.NS","royal enfield":"EICHERMOT.NS",
    "tvs motor":"TVSMOTOR.NS","tvs":"TVSMOTOR.NS",
    "ashok leyland":"ASHOKLEY.NS",
    "mrf":"MRF.NS",
    "apollo tyres":"APOLLOTYRE.NS",
    "hindustan unilever":"HINDUNILVR.NS","hul":"HINDUNILVR.NS",
    "itc":"ITC.NS",
    "nestle":"NESTLEIND.NS","nestle india":"NESTLEIND.NS",
    "britannia":"BRITANNIA.NS",
    "dabur":"DABUR.NS",
    "godrej consumer":"GODREJCP.NS",
    "marico":"MARICO.NS",
    "colgate":"COLPAL.NS","colgate palmolive":"COLPAL.NS",
    "tata consumer":"TATACONSUM.NS",
    "varun beverages":"VBL.NS",
    "sun pharma":"SUNPHARMA.NS","sun pharmaceutical":"SUNPHARMA.NS",
    "dr reddy":"DRREDDY.NS","dr reddys":"DRREDDY.NS",
    "cipla":"CIPLA.NS",
    "divis":"DIVISLAB.NS","divis lab":"DIVISLAB.NS","divi labs":"DIVISLAB.NS",
    "apollo hospitals":"APOLLOHOSP.NS","apollo":"APOLLOHOSP.NS",
    "max healthcare":"MAXHEALTH.NS",
    "lupin":"LUPIN.NS",
    "aurobindo":"AUROPHARMA.NS",
    "biocon":"BIOCON.NS",
    "torrent pharma":"TORNTPHARM.NS",
    "mankind pharma":"MANKIND.NS",
    "tata steel":"TATASTEEL.NS",
    "jsw steel":"JSWSTEEL.NS","jsw":"JSWSTEEL.NS",
    "hindalco":"HINDALCO.NS",
    "vedanta":"VEDL.NS",
    "sail":"SAIL.NS","steel authority":"SAIL.NS",
    "nmdc":"NMDC.NS",
    "hindustan zinc":"HINDZINC.NS",
    "larsen & toubro":"LT.NS","l&t":"LT.NS","larsen toubro":"LT.NS","lt":"LT.NS",
    "ultratech cement":"ULTRACEMCO.NS","ultratech":"ULTRACEMCO.NS",
    "ambuja cement":"AMBUJACEM.NS","ambuja":"AMBUJACEM.NS",
    "acc":"ACC.NS",
    "shree cement":"SHREECEM.NS",
    "airtel":"BHARTIARTL.NS","bharti airtel":"BHARTIARTL.NS",
    "vodafone idea":"IDEA.NS","vi":"IDEA.NS",
    "indus towers":"INDUSTOWER.NS",
    "avenue supermarts":"DMART.NS","dmart":"DMART.NS",
    "trent":"TRENT.NS",
    "titan":"TITAN.NS",
    "zomato":"ZOMATO.NS",
    "nykaa":"NYKAA.NS",
    "paytm":"PAYTM.NS",
    "policybazaar":"POLICYBZR.NS","policy bazaar":"POLICYBZR.NS",
    "asian paints":"ASIANPAINT.NS",
    "pidilite":"PIDILITIND.NS",
    "havells":"HAVELLS.NS",
    "voltas":"VOLTAS.NS",
    "siemens":"SIEMENS.NS",
    "abb":"ABB.NS","abb india":"ABB.NS",
    "info edge":"NAUKRI.NS","naukri":"NAUKRI.NS",
    "indiamart":"INDIAMART.NS",
    "indigo":"INDIGO.NS","interglobe aviation":"INDIGO.NS",
    "irctc":"IRCTC.NS",
    "sbi cards":"SBICARD.NS",
    "dixon tech":"DIXON.NS",
    "pi industries":"PIIND.NS",
    "upl":"UPL.NS",
    "bosch":"BOSCHLTD.NS",
    "torrent power":"TORNTPOWER.NS",
    "emami":"EMAMILTD.NS",
    "godrej properties":"GODREJPROP.NS",
    "dlf":"DLF.NS",
    "oberoi realty":"OBEROIRLTY.NS",
    "prestige estates":"PRESTIGE.NS",
    "brigade enterprises":"BRIGADE.NS",
}

def resolve_ticker(name):
    key = name.strip().lower()
    if key in COMPANY_MAP:
        return COMPANY_MAP[key]
    for k, v in COMPANY_MAP.items():
        if key in k or k in key:
            return v
    raw = name.strip().upper()
    return raw if (raw.endswith(".NS") or raw.endswith(".BO")) else raw + ".NS"

def fmt_inr(val):
    if val is None: return "N/A"
    try:
        v = float(val)
        return f"Rs {v:,.2f}"
    except: return "N/A"

def fmt_mcap(val):
    if val is None: return "N/A"
    try:
        v = float(val)
        if v >= 1e12: return f"Rs {v/1e12:.2f}L Cr"
        if v >= 1e9:  return f"Rs {v/1e7:.0f} Cr"
        return f"Rs {v/1e7:.1f} Cr"
    except: return "N/A"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock(ticker, period="2y"):
    errors = []
    for t in [ticker, ticker.replace(".NS",".BO") if ".NS" in ticker else ticker+".BO"]:
        try:
            df = yf.Ticker(t).history(period=period)
            if not df.empty and len(df) > 50:
                df = df[["Open","High","Low","Close","Volume"]].copy()
                df.dropna(inplace=True)
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            errors.append(str(e))
    raise ValueError(f"No data for {ticker}. Try: RELIANCE.NS, TCS.NS, HDFCBANK.NS")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_info(ticker, df_raw=None):
    """Fetch company info. Falls back to computing key metrics from price data."""
    result = {"name": ticker, "sector": "N/A", "industry": "N/A",
              "market_cap": None, "pe": None, "52w_high": None, "52w_low": None,
              "description": "", "exchange": "NSE", "shares": None}
    try:
        i = yf.Ticker(ticker).info
        result["name"]        = i.get("longName") or i.get("shortName") or ticker
        result["sector"]      = i.get("sector") or "N/A"
        result["industry"]    = i.get("industry") or "N/A"
        result["description"] = i.get("longBusinessSummary") or ""
        result["exchange"]    = i.get("exchange") or "NSE"
        # These often return None for Indian stocks — we compute them from df if missing
        result["market_cap"]  = i.get("marketCap")
        result["pe"]          = i.get("trailingPE") or i.get("forwardPE")
        result["52w_high"]    = i.get("fiftyTwoWeekHigh")
        result["52w_low"]     = i.get("fiftyTwoWeekLow")
        result["shares"]      = i.get("sharesOutstanding") or i.get("impliedSharesOutstanding")
    except Exception:
        pass
    # Always compute from price history — guaranteed to work
    if df_raw is not None and not df_raw.empty:
        one_year_ago = df_raw.index[-1] - pd.Timedelta(days=365)
        df_1y = df_raw[df_raw.index >= one_year_ago]
        if not df_1y.empty:
            result["52w_high"] = result["52w_high"] or round(float(df_1y["High"].max()), 2)
            result["52w_low"]  = result["52w_low"]  or round(float(df_1y["Low"].min()), 2)
        # If shares outstanding available, compute market cap
        if result["shares"] and result["market_cap"] is None:
            result["market_cap"] = float(df_raw["Close"].iloc[-1]) * float(result["shares"])
    return result

def add_features(df):
    df=df.copy()
    df["MA5"]=df["Close"].rolling(5).mean()
    df["MA20"]=df["Close"].rolling(20).mean()
    df["MA50"]=df["Close"].rolling(50).mean()
    df["MA200"]=df["Close"].rolling(200).mean()
    df["Daily_Return"]=df["Close"].pct_change()
    df["Volatility"]=df["Daily_Return"].rolling(20).std()
    df["Lag1"]=df["Close"].shift(1)
    df["Lag5"]=df["Close"].shift(5)
    df.dropna(inplace=True)
    return df

def build_models(df):
    F=["Open","High","Low","Volume","MA5","MA20","Lag1","Lag5","Volatility"]
    X=df[F].values; y=df["Close"].values
    sp=int(len(X)*0.80)
    Xtr,Xte=X[:sp],X[sp:]; ytr,yte=y[:sp],y[sp:]
    sc=MinMaxScaler(); Xstr=sc.fit_transform(Xtr); Xste=sc.transform(Xte)
    lr=LinearRegression().fit(Xstr,ytr)
    rf=RandomForestRegressor(100,random_state=42).fit(Xstr,ytr)
    svr=SVR(kernel="rbf",C=100,gamma=0.1,epsilon=0.1).fit(Xstr,ytr)
    def m(yt,yp): return {"r2":round(float(r2_score(yt,yp)),4),"rmse":round(float(np.sqrt(mean_squared_error(yt,yp))),4)}
    lp=lr.predict(Xste); rp=rf.predict(Xste); sp2=svr.predict(Xste)
    return dict(models=dict(lr=lr,rf=rf,svr=svr),scaler=sc,features=F,split=sp,
                preds=dict(lr=lp,rf=rp,svr=sp2),actuals=yte,
                metrics=dict(lr=m(yte,lp),rf=m(yte,rp),svr=m(yte,sp2)),
                y_train=ytr,y_test=yte,dates_train=df.index[:sp],dates_test=df.index[sp:])

def ensemble_forecast(m,df,days=30):
    last=df[m["features"]].iloc[-1].values.reshape(1,-1)
    ls=m["scaler"].transform(last)
    lp=m["models"]["lr"].predict(ls)[0]; rp=m["models"]["rf"].predict(ls)[0]; sp=m["models"]["svr"].predict(ls)[0]
    ens=(lp+rp+sp)/3; cur=df["Close"].iloc[-1]; drift=(ens-cur)/days; vol=df["Daily_Return"].std()
    rng=np.random.default_rng(42); fut=[]
    for _ in range(days):
        cur+=drift+rng.normal(0,vol*cur); fut.append(round(float(cur),2))
    fd=pd.date_range(df.index[-1]+pd.Timedelta(days=1),periods=days,freq="B")
    return dict(lr=round(lp,2),rf=round(rp,2),svr=round(sp,2),ensemble=round(ens,2),future=fut,future_dates=fd)

def technical_indicators(df):
    c=df["Close"]; d=c.diff()
    rs=d.clip(lower=0).rolling(14).mean()/(-d.clip(upper=0).rolling(14).mean()).replace(0,np.nan)
    rsi=100-100/(1+rs)
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    sig=macd.ewm(span=9,adjust=False).mean()
    sma=c.rolling(20).mean(); std=c.rolling(20).std()
    return dict(rsi=rsi,macd=macd,signal=sig,hist=macd-sig,bb_upper=sma+2*std,bb_mid=sma,bb_lower=sma-2*std)

def investment_signal(df,ens):
    c=df["Close"].iloc[-1]; ma50=df["MA50"].iloc[-1]; ma200=df["MA200"].iloc[-1]
    vol=df["Daily_Return"].std()*np.sqrt(252)
    score=sum([c>ma50,c>ma200,ens>c,vol<0.30])
    sig="BUY" if score>=3 else ("HOLD" if score==2 else "SELL")
    risk="LOW" if vol<0.20 else ("MEDIUM" if vol<0.40 else "HIGH")
    return dict(signal=sig,score=score,risk=risk,risk_score=round(min(vol*100,100),1),annual_vol=round(vol*100,2))

_AX=dict(showgrid=True,gridcolor="#1E2A3A",gridwidth=0.5,zeroline=False,tickfont=dict(size=10))
def _L(**kw):
    xkw=kw.pop("xaxis",{}); ykw=kw.pop("yaxis",{})
    b=dict(paper_bgcolor="#0D1321",plot_bgcolor="#070B14",
           font=dict(family="monospace",color="#8496AE",size=11),
           margin=dict(l=60,r=20,t=36,b=36),
           legend=dict(bgcolor="rgba(0,0,0,0)",borderwidth=0,font=dict(size=10,color="#8496AE")),
           hovermode="x unified",
           hoverlabel=dict(bgcolor="#0D1321",bordercolor="#1E2A3A",font=dict(size=11,color="#E4EAF5")))
    b["xaxis"]={**_AX,**xkw}; b["yaxis"]={**_AX,**ykw}; b.update(kw)
    return b
CFG={"displayModeBar":False}

def price_ma_chart(df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=df["Close"],name="Price",line=dict(color="#E8C96D",width=1.8),fill="tozeroy",fillcolor="rgba(232,201,109,0.06)"))
    fig.add_trace(go.Scatter(x=df.index,y=df["MA50"],name="MA50",line=dict(color="#6EE7F7",width=1.3,dash="dot")))
    fig.add_trace(go.Scatter(x=df.index,y=df["MA200"],name="MA200",line=dict(color="#F76EA0",width=1.3,dash="dash")))
    fig.update_layout(**_L(height=360,yaxis=dict(tickprefix="Rs ")))
    return fig

def candle_chart(df):
    d=df.tail(90)
    fig=go.Figure(go.Candlestick(x=d.index,open=d["Open"],high=d["High"],low=d["Low"],close=d["Close"],
        increasing_line_color="#5CF5A0",decreasing_line_color="#F76E6E",
        increasing_fillcolor="rgba(92,245,160,0.25)",decreasing_fillcolor="rgba(247,110,110,0.25)"))
    fig.update_layout(**_L(height=340,yaxis=dict(tickprefix="Rs ")))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def split_chart(m):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=m["dates_train"],y=m["y_train"],name="Train 80%",line=dict(color="#6EE7F7",width=1.6),fill="tozeroy",fillcolor="rgba(110,231,247,0.05)"))
    fig.add_trace(go.Scatter(x=m["dates_test"],y=m["y_test"],name="Test 20%",line=dict(color="#E8C96D",width=1.6),fill="tozeroy",fillcolor="rgba(232,201,109,0.07)"))
    fig.update_layout(**_L(height=300,yaxis=dict(tickprefix="Rs ")))
    return fig

def model_chart(m):
    d=m["dates_test"]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d,y=m["actuals"],name="Actual",line=dict(color="#E4EAF5",width=2)))
    fig.add_trace(go.Scatter(x=d,y=m["preds"]["lr"],name="Lin Reg",line=dict(color="#6EE7F7",width=1.3,dash="dot")))
    fig.add_trace(go.Scatter(x=d,y=m["preds"]["rf"],name="Rand Forest",line=dict(color="#5CF5A0",width=1.3,dash="dot")))
    fig.add_trace(go.Scatter(x=d,y=m["preds"]["svr"],name="SVR",line=dict(color="#B08BFF",width=1.3,dash="dot")))
    fig.update_layout(**_L(height=300,yaxis=dict(tickprefix="Rs ")))
    return fig

def bollinger_chart(df,tech):
    d=df.tail(150); tu=tech["bb_upper"].tail(150); tm=tech["bb_mid"].tail(150); tl=tech["bb_lower"].tail(150)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d.index,y=tu,name="Upper",line=dict(color="#B08BFF",width=1,dash="dot")))
    fig.add_trace(go.Scatter(x=d.index,y=tl,name="Lower",fill="tonexty",fillcolor="rgba(176,139,255,0.07)",line=dict(color="#B08BFF",width=1,dash="dot")))
    fig.add_trace(go.Scatter(x=d.index,y=tm,name="SMA20",line=dict(color="#6EE7F7",width=1.2,dash="dash")))
    fig.add_trace(go.Scatter(x=d.index,y=d["Close"],name="Price",line=dict(color="#E8C96D",width=1.8)))
    fig.update_layout(**_L(height=320,yaxis=dict(tickprefix="Rs ")))
    return fig

def rsi_chart(tech):
    rsi=tech["rsi"].dropna().tail(120)
    fig=go.Figure()
    fig.add_hrect(y0=70,y1=100,fillcolor="rgba(247,110,110,0.07)",line_width=0)
    fig.add_hrect(y0=0,y1=30,fillcolor="rgba(92,245,160,0.07)",line_width=0)
    fig.add_hline(y=70,line_dash="dot",line_color="#F76E6E",line_width=1)
    fig.add_hline(y=30,line_dash="dot",line_color="#5CF5A0",line_width=1)
    fig.add_trace(go.Scatter(x=rsi.index,y=rsi.values,name="RSI",line=dict(color="#6EE7F7",width=1.8),fill="tozeroy",fillcolor="rgba(110,231,247,0.06)"))
    fig.update_layout(**_L(height=260,yaxis=dict(range=[0,100],gridcolor="#1E2A3A")))
    return fig

def macd_chart(tech):
    mc=tech["macd"].dropna().tail(120); sg=tech["signal"].dropna().tail(120); hs=tech["hist"].dropna().tail(120)
    fig=go.Figure()
    fig.add_trace(go.Bar(x=hs.index,y=hs.values,name="Hist",marker_color=["#5CF5A0" if v>=0 else "#F76E6E" for v in hs.values],opacity=0.6))
    fig.add_trace(go.Scatter(x=mc.index,y=mc.values,name="MACD",line=dict(color="#6EE7F7",width=1.6)))
    fig.add_trace(go.Scatter(x=sg.index,y=sg.values,name="Signal",line=dict(color="#F76EA0",width=1.4)))
    fig.update_layout(**_L(height=260,barmode="overlay"))
    return fig

def forecast_chart(df,fc):
    h=df["Close"].tail(60); f=pd.Series(fc["future"],index=fc["future_dates"])
    u=[v*1.05 for v in fc["future"]]; lo=[v*0.95 for v in fc["future"]]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=h.index,y=h.values,name="Historical",line=dict(color="#8496AE",width=1.6)))
    fig.add_trace(go.Scatter(x=list(fc["future_dates"])+list(fc["future_dates"])[::-1],y=u+lo[::-1],fill="toself",fillcolor="rgba(232,201,109,0.08)",line=dict(width=0),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f.index,y=f.values,name="AI Forecast",line=dict(color="#E8C96D",width=2.2,dash="dot"),mode="lines+markers",marker=dict(size=3,color="#E8C96D")))
    fig.update_layout(**_L(height=340,yaxis=dict(tickprefix="Rs ")))
    return fig

def backtest_chart(m):
    d=m["dates_test"]; ens=(m["preds"]["lr"]+m["preds"]["rf"]+m["preds"]["svr"])/3; diff=m["actuals"]-ens
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d,y=m["actuals"],name="Actual",line=dict(color="#E4EAF5",width=2)))
    fig.add_trace(go.Scatter(x=d,y=ens,name="Ensemble",line=dict(color="#E8C96D",width=1.8,dash="dash")))
    fig.add_trace(go.Bar(x=d,y=diff,name="Error",marker_color=["#5CF5A0" if v>=0 else "#F76E6E" for v in diff],opacity=0.4,yaxis="y2"))
    fig.update_layout(**_L(height=320,yaxis=dict(tickprefix="Rs "),yaxis2=dict(overlaying="y",side="right",showgrid=False,tickfont=dict(size=9))))
    return fig

def vol_hist(df):
    r=df["Daily_Return"].dropna()*100; mu,sigma=r.mean(),r.std()
    xs=np.linspace(r.min(),r.max(),200)
    ys=(np.exp(-0.5*((xs-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi)))*len(r)*(r.max()-r.min())/50
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=r,nbinsx=50,name="Returns",marker_color="#6EE7F7",opacity=0.8))
    fig.add_trace(go.Scatter(x=xs,y=ys,name="Normal",line=dict(color="#E8C96D",width=1.8)))
    fig.update_layout(**_L(height=300))
    return fig

def returns_chart(df):
    r=df["Daily_Return"].dropna().tail(252)*100
    fig=go.Figure()
    fig.add_trace(go.Bar(x=r.index,y=r.values,marker_color=["#5CF5A0" if v>=0 else "#F76E6E" for v in r.values],opacity=0.75))
    fig.update_layout(**_L(height=260))
    return fig

def port_chart(df,inv):
    g=df["Close"]/df["Close"].iloc[0]*inv
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=g.index,y=g.values,name="Value",line=dict(color="#5CF5A0",width=2),fill="tozeroy",fillcolor="rgba(92,245,160,0.07)"))
    fig.add_hline(y=inv,line_dash="dot",line_color="#8496AE",line_width=1,annotation_text=f"Initial Rs {inv:,.0f}",annotation_font_color="#8496AE")
    fig.update_layout(**_L(height=300,yaxis=dict(tickprefix="Rs ",gridcolor="#1E2A3A")))
    return fig

def metrics_chart(metrics):
    ms=["Lin Reg","Rand Forest","SVR"]
    r2s=[metrics["lr"]["r2"],metrics["rf"]["r2"],metrics["svr"]["r2"]]
    rmses=[metrics["lr"]["rmse"],metrics["rf"]["rmse"],metrics["svr"]["rmse"]]
    clrs=["#6EE7F7","#5CF5A0","#B08BFF"]
    fig=make_subplots(rows=1,cols=2,subplot_titles=["R2 Score (higher better)","RMSE (lower better)"])
    for i,(m,r,e) in enumerate(zip(ms,r2s,rmses)):
        fig.add_trace(go.Bar(x=[m],y=[r],marker_color=clrs[i],showlegend=False),row=1,col=1)
        fig.add_trace(go.Bar(x=[m],y=[e],marker_color=clrs[i],showlegend=False),row=1,col=2)
    fig.update_layout(**_L(height=280))
    fig.update_annotations(font_color="#8496AE",font_size=11)
    return fig

def donut_chart(results):
    fig=go.Figure(go.Pie(labels=[r["name"] for r in results],values=[r["projected"] for r in results],hole=0.55,
        marker_colors=["#E8C96D","#6EE7F7","#5CF5A0","#B08BFF","#F76EA0","#F76E6E"][:len(results)],
        textfont=dict(size=11),hovertemplate="<b>%{label}</b><br>Rs %{value:,.2f}<extra></extra>"))
    fig.update_layout(**_L(height=300,showlegend=True,margin=dict(l=20,r=20,t=20,b=20)))
    return fig

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
.stApp{background:#070B14!important}
section[data-testid="stSidebar"]{background:#0D1321!important;border-right:1px solid #1E2A3A!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:1.8rem 2.2rem 3rem!important;max-width:100%!important}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;color:#E4EAF5!important}
[data-testid="metric-container"]{background:#0D1321!important;border:1px solid #1E2A3A!important;border-radius:12px!important;padding:16px 18px!important}
[data-testid="metric-container"] label{font-size:11px!important;text-transform:uppercase!important;color:#8496AE!important;font-family:'JetBrains Mono',monospace!important;letter-spacing:.8px!important}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Syne',sans-serif!important;font-size:1.3rem!important;font-weight:700!important;color:#E4EAF5!important}
.stTabs [data-baseweb="tab-list"]{background:#0D1321!important;border-radius:10px!important;padding:4px!important;gap:4px!important;border-bottom:none!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:7px!important;color:#8496AE!important;padding:8px 20px!important;border:none!important}
.stTabs [aria-selected="true"]{background:rgba(232,201,109,0.12)!important;color:#E8C96D!important}
.stTextInput input,.stNumberInput input{background:#0D1321!important;border:1px solid #1E2A3A!important;border-radius:8px!important;color:#E4EAF5!important}
.stTextInput input:focus,.stNumberInput input:focus{border-color:#E8C96D!important;box-shadow:0 0 0 2px rgba(232,201,109,.15)!important}
.stButton>button{background:#E8C96D!important;color:#0D1010!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:10px 22px!important}
.stButton>button:hover{background:#F0A64A!important}
.stRadio>div{gap:4px!important}
.stRadio label{background:transparent!important;border-radius:8px!important;padding:8px 14px!important;color:#8496AE!important;font-size:13px!important;border:1px solid transparent!important}
.stRadio label:hover{color:#E4EAF5!important;background:#131A2A!important}
hr{border-color:#1E2A3A!important}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:#0D1321}
::-webkit-scrollbar-thumb{background:#1E2A3A;border-radius:4px}
.nq-card{background:#0D1321;border:1px solid #1E2A3A;border-radius:12px;padding:18px 20px;margin-bottom:4px}
.nq-card-title{font-family:'JetBrains Mono',monospace;font-size:11px;color:#8496AE;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px}
.sig-buy{background:rgba(92,245,160,.12);color:#5CF5A0;border:1px solid rgba(92,245,160,.3);border-radius:8px;padding:6px 18px;font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;letter-spacing:2px;display:inline-block}
.sig-hold{background:rgba(232,201,109,.12);color:#E8C96D;border:1px solid rgba(232,201,109,.3);border-radius:8px;padding:6px 18px;font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;letter-spacing:2px;display:inline-block}
.sig-sell{background:rgba(247,110,110,.12);color:#F76E6E;border:1px solid rgba(247,110,110,.3);border-radius:8px;padding:6px 18px;font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;letter-spacing:2px;display:inline-block}
.nq-lbl{font-family:'JetBrains Mono',monospace;font-size:11px;color:#8496AE;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}
</style>
""",unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("""<div style='padding:20px 16px 16px;border-bottom:1px solid #1E2A3A;margin-bottom:12px'>
      <div style='background:linear-gradient(135deg,#E8C96D,#F0A64A);color:#000;font-family:Syne,sans-serif;
                  font-weight:800;font-size:12px;width:36px;height:36px;border-radius:9px;display:flex;
                  align-items:center;justify-content:center;margin-bottom:8px'>NQ</div>
      <span style='font-family:Syne,sans-serif;font-weight:700;font-size:16px;color:#E4EAF5;display:block'>NexusQuant India</span>
      <span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#E8C96D'>NSE &middot; BSE &middot; INR</span>
    </div>""",unsafe_allow_html=True)
    st.markdown("**Search Indian Company**")
    company_input=st.text_input("",placeholder="Reliance, TCS, HDFC Bank...",label_visibility="collapsed",key="company_search")
    st.markdown("**Investment Amount (Rs)**")
    investment=st.number_input("",min_value=1000,value=100000,step=5000,label_visibility="collapsed")
    st.markdown("**Period**")
    period=st.selectbox("",["1y","2y","5y","max"],index=1,label_visibility="collapsed")
    analyze_btn=st.button("Analyse",use_container_width=True)
    st.divider()
    st.markdown("**Navigate**")
    page=st.radio("",["Home  Overview","Chart  ML Models","Tech  Technical","AI  Predictions","Risk  Volatility","Money  Portfolio"],label_visibility="collapsed")
    st.markdown("<div style='display:flex;align-items:center;gap:8px;margin-top:16px'><span style='width:7px;height:7px;border-radius:50%;background:#5CF5A0;box-shadow:0 0 8px #5CF5A0;display:inline-block'></span><span style='font-size:11px;color:#8496AE;font-family:JetBrains Mono,monospace'>NSE/BSE Live Data</span></div>",unsafe_allow_html=True)

# SESSION STATE
if "data" not in st.session_state: st.session_state["data"]=None
# (quick pick removed)

# LOAD DATA
if analyze_btn and company_input.strip():
    with st.spinner(f"Fetching NSE/BSE data for {company_input}..."):
        try:
            ticker=resolve_ticker(company_input)
            df_raw=fetch_stock(ticker,period)
            df=add_features(df_raw)
            info=fetch_info(ticker, df_raw=df_raw)
            models=build_models(df)
            tech=technical_indicators(df)
            fc=ensemble_forecast(models,df)
            sig=investment_signal(df,fc["ensemble"])
            st.session_state["data"]=dict(ticker=ticker,df=df,info=info,models=models,tech=tech,fc=fc,sig=sig,investment=investment)
        except Exception as e:
            st.error(str(e))
            st.info("Try: Reliance, TCS, Infosys, HDFC Bank, ICICI Bank, Tata Motors, Wipro, Zomato, or use NSE ticker like RELIANCE.NS")

# WELCOME
if st.session_state["data"] is None:
    st.markdown("""<div style='background:linear-gradient(135deg,#0D1321 0%,#131A2A 60%,#0D1321 100%);border:1px solid #1E2A3A;border-radius:16px;padding:40px 48px;margin-bottom:28px'>
      <h1 style='font-size:2.4rem;margin-bottom:8px;font-family:Syne,sans-serif'>Indian Stock Intelligence</h1>
      <p style='color:#8496AE;font-size:15px;line-height:1.8'>AI analysis for <strong style='color:#E8C96D'>NSE & BSE</strong> listed companies. All prices in <strong style='color:#E8C96D'>Indian Rupees (Rs)</strong></p>
    </div>""",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    for col,icon,title,desc in [(c1,"Machine Learning","3 ML Models","Linear Regression, Random Forest & SVR"),(c2,"Charts","Technical Suite","RSI, MACD, Bollinger Bands, Moving Averages"),(c3,"Portfolio","Portfolio Sim","Simulate Rs investment growth across Indian stocks")]:
        col.markdown(f"<div class='nq-card'><div style='font-size:1.4rem;margin-bottom:10px'>{icon}</div><div style='font-family:Syne,sans-serif;font-weight:700;font-size:15px;color:#E4EAF5;margin-bottom:6px'>{title}</div><div style='font-size:13px;color:#8496AE;line-height:1.6'>{desc}</div></div>",unsafe_allow_html=True)
    st.markdown("<br>**Try searching for:**",unsafe_allow_html=True)
    ex=st.columns(5)
    for col,n in zip(ex,["Reliance","TCS","HDFC Bank","Infosys","Zomato"]): col.code(n)
    st.stop()

# UNPACK
D=st.session_state["data"]
ticker=D["ticker"]; df=D["df"]; info=D["info"]; models=D["models"]
tech=D["tech"]; fc=D["fc"]; sig=D["sig"]; inv=D["investment"]
cn=df["Close"].iloc[-1]; cp=df["Close"].iloc[-2]; dc=(cn-cp)/cp*100

# HEADER
st.markdown(f"""<div style='display:flex;align-items:center;justify-content:space-between;background:#0D1321;border:1px solid #1E2A3A;border-radius:12px;padding:14px 22px;margin-bottom:24px'>
  <div>
    <span style='font-family:Syne,sans-serif;font-weight:800;font-size:1.3rem;color:#E4EAF5'>{info.get("name",ticker)}</span>
    <span style='font-family:JetBrains Mono,monospace;font-size:11px;color:#4a5a70;margin-left:10px'>{ticker}</span>
    <span style='font-size:11px;color:#8496AE;margin-left:10px'>{info.get("sector","N/A")} &middot; {info.get("exchange","NSE")}</span>
  </div>
  <div style='text-align:right'>
    <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.6rem;color:#E4EAF5'>Rs {cn:,.2f}</div>
    <div style='font-size:13px;color:{"#5CF5A0" if dc>=0 else "#F76E6E"};font-family:JetBrains Mono,monospace'>{"+" if dc>=0 else ""}{dc:.2f}% today</div>
  </div>
</div>""",unsafe_allow_html=True)

# PAGES
if "Overview" in page:
    st.markdown("### Overview")
    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric("Price",fmt_inr(cn)); k2.metric("Day Change",f"{dc:+.2f}%")
    k3.metric("52W High",fmt_inr(info.get("52w_high"))); k4.metric("52W Low",fmt_inr(info.get("52w_low")))
    k5.metric("Market Cap",fmt_mcap(info.get("market_cap"))); k6.metric("P/E",f"{info['pe']:.1f}" if info.get("pe") else "N/A")
    st.markdown("<br>",unsafe_allow_html=True)
    sa,sb,sc,sd=st.columns([1,1,1,2])
    sa.markdown(f"<div class='sig-{sig['signal'].lower()}'>{sig['signal']}</div>",unsafe_allow_html=True)
    sb.metric("Risk",sig["risk"]); sc.metric("Annual Vol",f"{sig['annual_vol']}%"); sd.metric("Confidence",f"{sig['score']}/4 bullish signals")
    st.markdown("<br>",unsafe_allow_html=True)
    t1,t2=st.tabs(["Price + Moving Averages","Candlestick 90 days"])
    with t1: st.plotly_chart(price_ma_chart(df),use_container_width=True,config=CFG)
    with t2: st.plotly_chart(candle_chart(df),use_container_width=True,config=CFG)
    desc=info.get("description","")
    if desc:
        with st.expander("About this company"): st.markdown(f"<p style='font-size:13px;color:#8496AE;line-height:1.7'>{desc[:900]}...</p>",unsafe_allow_html=True)

elif "ML Models" in page:
    st.markdown("### Machine Learning Models")
    for col,key,label in zip(st.columns(3),["lr","rf","svr"],["Linear Regression","Random Forest","SVR"]):
        m=models["metrics"][key]; rc="#5CF5A0" if m["r2"]>=.9 else ("#E8C96D" if m["r2"]>=.7 else "#F76E6E")
        col.markdown(f"<div class='nq-card'><div class='nq-card-title'>{label}</div><div style='display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1E2A3A'><span style='font-size:12px;color:#8496AE'>R2 Score</span><span style='font-family:JetBrains Mono,monospace;font-size:13px;color:{rc};font-weight:600'>{m['r2']}</span></div><div style='display:flex;justify-content:space-between;padding:8px 0'><span style='font-size:12px;color:#8496AE'>RMSE</span><span style='font-family:JetBrains Mono,monospace;font-size:13px;color:#E4EAF5'>Rs {m['rmse']:,.2f}</span></div></div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.markdown("<div class='nq-lbl'>Train / Test Split</div>",unsafe_allow_html=True); st.plotly_chart(split_chart(models),use_container_width=True,config=CFG)
    with c2: st.markdown("<div class='nq-lbl'>Model Predictions vs Actual</div>",unsafe_allow_html=True); st.plotly_chart(model_chart(models),use_container_width=True,config=CFG)
    st.markdown("<div class='nq-lbl'>Performance Comparison</div>",unsafe_allow_html=True)
    st.plotly_chart(metrics_chart(models["metrics"]),use_container_width=True,config=CFG)

elif "Technical" in page:
    st.markdown("### Technical Indicators")
    st.markdown("<div class='nq-lbl'>Bollinger Bands</div>",unsafe_allow_html=True)
    st.plotly_chart(bollinger_chart(df,tech),use_container_width=True,config=CFG)
    c1,c2=st.columns(2)
    with c1: st.markdown("<div class='nq-lbl'>RSI (14)</div>",unsafe_allow_html=True); st.plotly_chart(rsi_chart(tech),use_container_width=True,config=CFG)
    with c2: st.markdown("<div class='nq-lbl'>MACD</div>",unsafe_allow_html=True); st.plotly_chart(macd_chart(tech),use_container_width=True,config=CFG)
    lr=float(tech["rsi"].dropna().iloc[-1]); rc="#F76E6E" if lr>70 else ("#5CF5A0" if lr<30 else "#E8C96D"); rl="Overbought" if lr>70 else ("Oversold" if lr<30 else "Neutral")
    mb=float(tech["hist"].dropna().iloc[-1])>0
    r1,r2=st.columns(2)
    r1.markdown(f"<div class='nq-card'><div class='nq-card-title'>RSI Reading</div><span style='font-family:Syne,sans-serif;font-weight:700;font-size:1.6rem;color:{rc}'>{lr:.1f}</span><span style='font-size:12px;color:#8496AE;margin-left:10px'>{rl}</span></div>",unsafe_allow_html=True)
    r2.markdown(f"<div class='nq-card'><div class='nq-card-title'>MACD Signal</div><span style='font-family:Syne,sans-serif;font-weight:700;font-size:1.2rem;color:{"#5CF5A0" if mb else "#F76E6E"}'>{"Bullish" if mb else "Bearish"}</span></div>",unsafe_allow_html=True)

elif "Predictions" in page:
    st.markdown("### AI Predictions & Forecasting")
    for col,key,label in zip(st.columns(4),["lr","rf","svr","ensemble"],["Linear Reg","Random Forest","SVR","Ensemble"]):
        val=fc[key]; chg=(val-cn)/cn*100
        col.markdown(f"<div class='nq-card' style='{"border-color:rgba(232,201,109,.4);background:rgba(232,201,109,.04);" if key=="ensemble" else ""}' ><div class='nq-card-title'>{label}</div><div style='font-family:Syne,sans-serif;font-weight:700;font-size:1.3rem;color:{"#E8C96D" if key=="ensemble" else "#E4EAF5"}'>Rs {val:,.2f}</div><div style='font-size:11px;color:{"#5CF5A0" if chg>=0 else "#F76E6E"};font-family:JetBrains Mono,monospace;margin-top:4px'>{chg:+.2f}% vs now</div></div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("<div class='nq-lbl'>30-Day Forecast</div>",unsafe_allow_html=True)
    st.plotly_chart(forecast_chart(df,fc),use_container_width=True,config=CFG)
    st.markdown("<div class='nq-lbl'>Backtesting</div>",unsafe_allow_html=True)
    st.plotly_chart(backtest_chart(models),use_container_width=True,config=CFG)
    ens=(models["preds"]["lr"]+models["preds"]["rf"]+models["preds"]["svr"])/3
    b1,b2=st.columns(2)
    b1.metric("Ensemble R2",f"{r2_score(models['actuals'],ens):.4f}")
    b2.metric("Ensemble RMSE",f"Rs {np.sqrt(mean_squared_error(models['actuals'],ens)):,.2f}")

elif "Volatility" in page:
    st.markdown("### Volatility & Risk Analysis")
    dv=df["Daily_Return"].std()*100; av=dv*np.sqrt(252)
    v1,v2,v3,v4=st.columns(4)
    v1.metric("Daily Vol",f"{dv:.2f}%"); v2.metric("Annual Vol",f"{av:.2f}%")
    v3.metric("Skewness",f"{df['Daily_Return'].skew():.3f}"); v4.metric("Kurtosis",f"{df['Daily_Return'].kurtosis():.3f}")
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.markdown("<div class='nq-lbl'>Returns Distribution</div>",unsafe_allow_html=True); st.plotly_chart(vol_hist(df),use_container_width=True,config=CFG)
    with c2: st.markdown("<div class='nq-lbl'>Daily Returns 1 Year</div>",unsafe_allow_html=True); st.plotly_chart(returns_chart(df),use_container_width=True,config=CFG)
    rs=sig["risk_score"]; rc="#5CF5A0" if rs<30 else ("#E8C96D" if rs<60 else "#F76E6E")
    risk_html = f"""<div class='nq-card' style='margin-top:8px'>
      <div class='nq-card-title'>Risk Score</div>
      <div style='display:flex;align-items:center;gap:16px'>
        <span style='font-family:Syne,sans-serif;font-weight:800;font-size:2.4rem;color:{rc}'>{rs}</span>
        <div>
          <div style='font-size:13px;color:#E4EAF5'>Level: <strong>{sig["risk"]}</strong></div>
          <div style='font-size:12px;color:#8496AE;margin-top:3px'>Annualised volatility: {sig["annual_vol"]}%</div>
        </div>
        <div style='flex:1;height:8px;background:#1E2A3A;border-radius:99px;overflow:hidden;margin-left:16px'>
          <div style='width:{min(rs,100)}%;height:100%;background:{rc};border-radius:99px'></div>
        </div>
      </div>
    </div>"""
    st.markdown(risk_html, unsafe_allow_html=True)

elif "Portfolio" in page:
    st.markdown("### Portfolio Simulator (Rs)")
    g=inv*(df["Close"].iloc[-1]/df["Close"].iloc[0]); prf=g-inv; ret=(g-inv)/inv*100
    p1,p2,p3=st.columns(3)
    p1.metric("Invested",fmt_inr(inv)); p2.metric("Current Value",fmt_inr(g),delta=f"Rs {prf:+,.2f}"); p3.metric("Return",f"{ret:+.2f}%")
    st.markdown(f"<div class='nq-lbl' style='margin-top:12px'>Investment Growth - {info.get('name',ticker)}</div>",unsafe_allow_html=True)
    st.plotly_chart(port_chart(df,inv),use_container_width=True,config=CFG)
    st.divider()
    st.markdown("#### Multi-Stock Allocation Simulator")
    if "portfolio_rows" not in st.session_state:
        st.session_state["portfolio_rows"]=[{"name":"Reliance","amount":50000},{"name":"TCS","amount":30000},{"name":"HDFC Bank","amount":20000}]
    rows=st.session_state["portfolio_rows"]; updated=[]
    for i,row in enumerate(rows):
        c1,c2,c3=st.columns([3,2,1])
        n=c1.text_input(f"Co{i}",value=row["name"],key=f"pn_{i}",label_visibility="collapsed",placeholder="Company")
        a=c2.number_input(f"Amt{i}",value=float(row["amount"]),min_value=0.0,key=f"pa_{i}",label_visibility="collapsed")
        if c3.button("x",key=f"del_{i}"): rows.pop(i); st.rerun()
        updated.append({"name":n,"amount":a})
    st.session_state["portfolio_rows"]=updated
    ca,cr=st.columns([1,4])
    if ca.button("+ Add"): st.session_state["portfolio_rows"].append({"name":"","amount":10000}); st.rerun()
    if cr.button("Run Simulation",use_container_width=True):
        res=[]
        with st.spinner("Fetching data..."):
            for row in updated:
                if not row["name"].strip(): continue
                try:
                    t2=resolve_ticker(row["name"]); d2=fetch_stock(t2,"1y")
                    ret2=(d2["Close"].iloc[-1]-d2["Close"].iloc[0])/d2["Close"].iloc[0]
                    res.append({"name":row["name"],"ticker":t2,"amount":row["amount"],"projected":round(row["amount"]*(1+ret2),2),"return_pct":round(ret2*100,2)})
                except: st.warning(f"Could not fetch {row['name']}")
        if res: st.session_state["portfolio_results"]=res
    if "portfolio_results" in st.session_state:
        res=st.session_state["portfolio_results"]
        ti=sum(r["amount"] for r in res); tp=sum(r["projected"] for r in res); tr=(tp-ti)/ti*100
        t1,t2,t3=st.columns(3)
        t1.metric("Total Invested",fmt_inr(ti)); t2.metric("Total Projected",fmt_inr(tp),delta=f"Rs {tp-ti:+,.2f}"); t3.metric("Return",f"{tr:+.2f}%")
        rr,dd=st.columns([2,1])
        with rr:
            st.markdown("<br>",unsafe_allow_html=True)
            for r in res:
                pos=r["return_pct"]>=0
                st.markdown(f"<div style='background:#131A2A;border:1px solid #1E2A3A;border-radius:10px;padding:12px 18px;margin-bottom:8px;display:flex;align-items:center;justify-content:space-between'><div><span style='font-weight:600;font-size:14px;color:#E4EAF5'>{r['name']}</span><span style='font-size:11px;color:#4a5a70;margin-left:8px'>{r['ticker']}</span></div><span style='font-size:12px;color:#8496AE'>Rs {r['amount']:,.0f}</span><span style='font-family:JetBrains Mono,monospace;font-weight:600;color:{"#5CF5A0" if pos else "#F76E6E"}'>Rs {r['projected']:,.2f}</span><span style='font-family:JetBrains Mono,monospace;font-size:12px;padding:3px 8px;border-radius:4px;background:{"rgba(92,245,160,.1)" if pos else "rgba(247,110,110,.1)"};color:{"#5CF5A0" if pos else "#F76E6E"}'>{r['return_pct']:+.2f}%</span></div>",unsafe_allow_html=True)
        with dd: st.plotly_chart(donut_chart(res),use_container_width=True,config=CFG)
