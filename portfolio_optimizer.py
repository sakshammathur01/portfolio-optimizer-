"""
Portfolio Optimizer — Nifty 500 Universe  |  Indian Markets
Equity Capital Markets & Wealth Management Project
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NiftyEdge | Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design Tokens ──────────────────────────────────────────────────────────────
NAVY    = "#0F1C3F"
NAVY2   = "#162244"
GOLD    = "#C9A84C"
GOLD2   = "#F0D080"
TEAL    = "#0EA5A0"
WHITE   = "#FFFFFF"
OFF_WHITE = "#F7F8FC"
SLATE   = "#64748B"
LIGHT   = "#E8EDF5"
GREEN   = "#10B981"
RED     = "#EF4444"
AMBER   = "#F59E0B"

# ── Professional CSS ───────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {OFF_WHITE};
    color: #1A2540;
  }}
  .stApp {{ background-color: {OFF_WHITE}; }}

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 0 2rem 3rem 2rem; max-width: 1400px; }}

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY} 0%, {NAVY2} 100%);
    border-right: 1px solid rgba(201,168,76,0.2);
  }}
  section[data-testid="stSidebar"] * {{
    color: {WHITE} !important;
  }}
  section[data-testid="stSidebar"] .stSlider > div > div > div > div {{
    background: {GOLD} !important;
  }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stNumberInput label,
  section[data-testid="stSidebar"] .stTextInput label {{
    color: {GOLD2} !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
  }}
  section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    color: {WHITE} !important;
    border-radius: 8px !important;
  }}
  section[data-testid="stSidebar"] input {{
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    color: {WHITE} !important;
    border-radius: 8px !important;
  }}
  section[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    background: linear-gradient(135deg, {GOLD} 0%, {AMBER} 100%) !important;
    color: {NAVY} !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 20px rgba(201,168,76,0.35);
    transition: all 0.2s ease;
    margin-top: 1rem;
  }}
  section[data-testid="stSidebar"] .stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(201,168,76,0.5);
  }}

  /* ── Top banner ── */
  .top-banner {{
    background: linear-gradient(135deg, {NAVY} 0%, #1a3070 60%, #0d2550 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin: 1.5rem 0 1.8rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid rgba(201,168,76,0.25);
    box-shadow: 0 8px 40px rgba(15,28,63,0.18);
    position: relative;
    overflow: hidden;
  }}
  .top-banner::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(201,168,76,0.12) 0%, transparent 70%);
    border-radius: 50%;
  }}
  .top-banner::after {{
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(14,165,160,0.1) 0%, transparent 70%);
    border-radius: 50%;
  }}
  .banner-left h1 {{
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    color: {WHITE};
    margin: 0 0 0.3rem 0;
    line-height: 1.2;
  }}
  .banner-left h1 span {{ color: {GOLD}; }}
  .banner-left p {{
    color: rgba(255,255,255,0.65);
    font-size: 0.9rem;
    margin: 0;
    font-weight: 300;
  }}
  .banner-pills {{
    display: flex; gap: 0.6rem; flex-wrap: wrap; justify-content: flex-end;
  }}
  .pill {{
    background: rgba(201,168,76,0.12);
    border: 1px solid rgba(201,168,76,0.35);
    color: {GOLD2};
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}

  /* ── Section labels ── */
  .section-label {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: {SLATE};
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 2rem 0 0.4rem 0;
  }}
  .section-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: {NAVY};
    margin: 0 0 1.2rem 0;
    line-height: 1.3;
  }}

  /* ── KPI Cards ── */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.2rem 0;
  }}
  .kpi-card {{
    background: {WHITE};
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    border: 1px solid {LIGHT};
    box-shadow: 0 2px 16px rgba(15,28,63,0.06);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }}
  .kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(15,28,63,0.11);
  }}
  .kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
  }}
  .kpi-navy::before  {{ background: {NAVY}; }}
  .kpi-gold::before  {{ background: {GOLD}; }}
  .kpi-teal::before  {{ background: {TEAL}; }}
  .kpi-green::before {{ background: {GREEN}; }}
  .kpi-label {{
    font-size: 0.72rem;
    font-weight: 600;
    color: {SLATE};
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.5rem;
  }}
  .kpi-value {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: {NAVY};
    line-height: 1;
    margin-bottom: 0.25rem;
  }}
  .kpi-sub {{
    font-size: 0.75rem;
    color: {SLATE};
    font-weight: 400;
  }}

  /* ── Profile card ── */
  .profile-card {{
    background: {WHITE};
    border-radius: 14px;
    padding: 1.6rem;
    border: 1px solid {LIGHT};
    box-shadow: 0 2px 16px rgba(15,28,63,0.06);
    margin-bottom: 1rem;
  }}
  .profile-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1.1rem;
    border-radius: 30px;
    font-weight: 700;
    font-size: 0.85rem;
    margin-bottom: 1rem;
  }}
  .badge-aggressive {{ background: #FEE2E2; color: #991B1B; }}
  .badge-moderate   {{ background: #FEF3C7; color: #92400E; }}
  .badge-conservative {{ background: #DCFCE7; color: #14532D; }}
  .profile-row {{
    display: flex;
    justify-content: space-between;
    padding: 0.45rem 0;
    border-bottom: 1px solid {LIGHT};
    font-size: 0.88rem;
  }}
  .profile-row:last-child {{ border-bottom: none; }}
  .profile-row span:first-child {{ color: {SLATE}; }}
  .profile-row span:last-child {{ font-weight: 600; color: {NAVY}; }}

  /* ── Chart wrapper ── */
  .chart-card {{
    background: {WHITE};
    border-radius: 14px;
    padding: 1.5rem;
    border: 1px solid {LIGHT};
    box-shadow: 0 2px 16px rgba(15,28,63,0.06);
    margin-bottom: 1.2rem;
  }}
  .chart-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: {NAVY};
    margin-bottom: 0.2rem;
  }}
  .chart-sub {{
    font-size: 0.78rem;
    color: {SLATE};
    margin-bottom: 1rem;
  }}

  /* ── Allocation table ── */
  .alloc-table {{ width: 100%; border-collapse: collapse; font-size: 0.87rem; }}
  .alloc-table th {{
    background: {NAVY};
    color: {GOLD2};
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.75rem 1rem;
    text-align: left;
  }}
  .alloc-table th:last-child {{ text-align: right; }}
  .alloc-table td {{
    padding: 0.7rem 1rem;
    color: #1A2540;
    border-bottom: 1px solid {LIGHT};
    font-weight: 500;
  }}
  .alloc-table tr:last-child td {{ border-bottom: none; }}
  .alloc-table tr:hover td {{ background: {OFF_WHITE}; }}
  .alloc-table td.num {{ text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.83rem; color: {NAVY}; }}
  .cat-chip {{
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
  }}

  /* ── Status banners ── */
  .status-live {{
    background: #ECFDF5; border: 1px solid #6EE7B7;
    color: #065F46; border-radius: 10px; padding: 0.75rem 1.1rem;
    font-size: 0.85rem; font-weight: 500; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
  }}
  .status-sim {{
    background: #FFFBEB; border: 1px solid #FCD34D;
    color: #78350F; border-radius: 10px; padding: 0.75rem 1.1rem;
    font-size: 0.85rem; font-weight: 500; margin-bottom: 1rem;
  }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    padding: 2rem 0 1rem;
    color: {SLATE};
    font-size: 0.78rem;
    border-top: 1px solid {LIGHT};
    margin-top: 3rem;
  }}
  .footer strong {{ color: {NAVY}; }}
</style>
""", unsafe_allow_html=True)

# ── Nifty 500 Asset Universe ───────────────────────────────────────────────────
ASSET_UNIVERSE = {
    # ── Large Cap (Nifty 50) ──
    "Reliance Industries":   ("RELIANCE.NS",   "Large Cap"),
    "HDFC Bank":             ("HDFCBANK.NS",   "Large Cap"),
    "Infosys":               ("INFY.NS",        "Large Cap"),
    "TCS":                   ("TCS.NS",         "Large Cap"),
    "ICICI Bank":            ("ICICIBANK.NS",   "Large Cap"),
    "Bharti Airtel":         ("BHARTIARTL.NS",  "Large Cap"),
    "SBI":                   ("SBIN.NS",        "Large Cap"),
    "HUL":                   ("HINDUNILVR.NS",  "Large Cap"),
    "Axis Bank":             ("AXISBANK.NS",    "Large Cap"),
    "Kotak Mahindra Bank":   ("KOTAKBANK.NS",   "Large Cap"),
    "L&T":                   ("LT.NS",          "Large Cap"),
    "Wipro":                 ("WIPRO.NS",       "Large Cap"),
    "HCL Technologies":      ("HCLTECH.NS",     "Large Cap"),
    "ITC":                   ("ITC.NS",         "Large Cap"),
    "Sun Pharma":            ("SUNPHARMA.NS",   "Large Cap"),
    "Maruti Suzuki":         ("MARUTI.NS",      "Large Cap"),
    "Titan Company":         ("TITAN.NS",       "Large Cap"),
    "UltraTech Cement":      ("ULTRACEMCO.NS",  "Large Cap"),
    "NTPC":                  ("NTPC.NS",        "Large Cap"),
    "Power Grid":            ("POWERGRID.NS",   "Large Cap"),

    # ── Mid Cap (Nifty Midcap 150) ──
    "Tata Motors":           ("TATAMOTORS.NS",  "Mid Cap"),
    "Bajaj Finance":         ("BAJFINANCE.NS",  "Mid Cap"),
    "Tata Power":            ("TATAPOWER.NS",   "Mid Cap"),
    "Voltas":                ("VOLTAS.NS",      "Mid Cap"),
    "IDFC First Bank":       ("IDFCFIRSTB.NS",  "Mid Cap"),
    "Mphasis":               ("MPHASIS.NS",     "Mid Cap"),
    "Crompton Greaves":      ("CROMPTON.NS",    "Mid Cap"),
    "Trent":                 ("TRENT.NS",       "Mid Cap"),
    "Indian Hotels":         ("INDHOTEL.NS",    "Mid Cap"),
    "Persistent Systems":    ("PERSISTENT.NS",  "Mid Cap"),
    "Coforge":               ("COFORGE.NS",     "Mid Cap"),
    "Max Healthcare":        ("MAXHEALTH.NS",   "Mid Cap"),
    "Apollo Hospitals":      ("APOLLOHOSP.NS",  "Mid Cap"),
    "Godrej Properties":     ("GODREJPROP.NS",  "Mid Cap"),
    "Oberoi Realty":         ("OBEROIRLTY.NS",  "Mid Cap"),

    # ── Small Cap (Nifty Smallcap 250) ──
    "Happiest Minds":        ("HAPPSTMNDS.NS",  "Small Cap"),
    "KPIT Technologies":     ("KPITTECH.NS",    "Small Cap"),
    "Latent View Analytics": ("LATENTVIEW.NS",  "Small Cap"),
    "Kaynes Technology":     ("KAYNES.NS",      "Small Cap"),
    "Syrma SGS":             ("SYRMA.NS",       "Small Cap"),
    "Tata Elxsi":            ("TATAELXSI.NS",   "Small Cap"),
    "Affle India":           ("AFFLE.NS",       "Small Cap"),
    "Route Mobile":          ("ROUTE.NS",       "Small Cap"),
    "Bikaji Foods":          ("BIKAJI.NS",      "Small Cap"),
    "Medplus Health":        ("MEDPLUS.NS",     "Small Cap"),

    # ── Index ETFs ──
    "Nifty 50 ETF (NIFTYBEES)":    ("NIFTYBEES.NS",  "Index ETF"),
    "Nifty Next 50 ETF":           ("JUNIORBEES.NS",  "Index ETF"),
    "Nifty Midcap 150 ETF":        ("MIDCAP150.NS",   "Index ETF"),
    "Nifty IT ETF":                ("NIFTYIT.NS",     "Sectoral ETF"),
    "Nifty Pharma ETF":            ("PHARMABEES.NS",  "Sectoral ETF"),
    "Nifty Bank ETF (BANKBEES)":   ("BANKBEES.NS",    "Sectoral ETF"),

    # ── Alternatives ──
    "Gold ETF (GOLDBEES)":         ("GOLDBEES.NS",    "Gold"),
    "Bharat Bond ETF Apr 2032":    ("EBBETF0432.NS",  "Debt ETF"),
}

# ── Category colours (for chips and charts) ──────────────────────────────────
CAT_COLOURS = {
    "Large Cap":    ("#1E3A8A", "#DBEAFE", "#1D4ED8"),
    "Mid Cap":      ("#7C2D12", "#FEF3C7", "#B45309"),
    "Small Cap":    ("#14532D", "#DCFCE7", "#16A34A"),
    "Index ETF":    ("#1E2761", "#E0E7FF", "#4338CA"),
    "Sectoral ETF": ("#6B21A8", "#F3E8FF", "#9333EA"),
    "Gold":         ("#78350F", "#FEF3C7", "#D97706"),
    "Debt ETF":     ("#0F766E", "#CCFBF1", "#0D9488"),
}

# ── Risk profile definitions ───────────────────────────────────────────────────
RISK_PROFILES = {
    "Conservative": {
        "score_range": (0, 35),
        "badge_class": "badge-conservative",
        "description": "Capital preservation with steady, low-risk returns.",
        "expected_return_range": "7–9%",
        "target_assets": [
            "Bharat Bond ETF Apr 2032", "Gold ETF (GOLDBEES)",
            "Nifty 50 ETF (NIFTYBEES)", "HDFC Bank", "ITC", "Power Grid",
        ],
    },
    "Moderate": {
        "score_range": (36, 65),
        "badge_class": "badge-moderate",
        "description": "Balanced growth with diversified risk exposure.",
        "expected_return_range": "10–13%",
        "target_assets": [
            "Nifty 50 ETF (NIFTYBEES)", "Nifty Next 50 ETF", "HDFC Bank",
            "Infosys", "Apollo Hospitals", "Gold ETF (GOLDBEES)",
            "Bharat Bond ETF Apr 2032", "Trent", "Max Healthcare",
        ],
    },
    "Aggressive": {
        "score_range": (66, 100),
        "badge_class": "badge-aggressive",
        "description": "Maximum growth across large, mid and small cap universe.",
        "expected_return_range": "14–20%",
        "target_assets": [
            "Reliance Industries", "HDFC Bank", "Infosys", "TCS",
            "Tata Motors", "Bajaj Finance", "Persistent Systems", "Coforge",
            "Tata Elxsi", "KPIT Technologies", "Happiest Minds",
            "Kaynes Technology", "Gold ETF (GOLDBEES)", "Nifty Midcap 150 ETF",
        ],
    },
}

# ── Fallback GBM data ─────────────────────────────────────────────────────────
CAT_PARAMS = {
    "Large Cap":    (0.13, 0.17),
    "Mid Cap":      (0.17, 0.24),
    "Small Cap":    (0.20, 0.30),
    "Index ETF":    (0.13, 0.15),
    "Sectoral ETF": (0.15, 0.22),
    "Gold":         (0.10, 0.13),
    "Debt ETF":     (0.07, 0.04),
}

def generate_fallback_data(asset_names, years=3):
    np.random.seed(42)
    n_days = years * 252
    dates  = pd.bdate_range(end=datetime.now(), periods=n_days)
    data   = {}
    for i, name in enumerate(asset_names):
        cat         = ASSET_UNIVERSE.get(name, (None, "Index ETF"))[1]
        mu, sigma   = CAT_PARAMS.get(cat, (0.13, 0.18))
        # add small cross-correlation via common factor
        common      = np.random.normal(0, 0.008, n_days)
        idio        = np.random.normal(mu/252, sigma/np.sqrt(252), n_days)
        shocks      = 0.55 * common + 0.45 * idio
        data[name]  = pd.Series(100 * np.cumprod(1 + shocks), index=dates)
    return pd.DataFrame(data).dropna()

# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(tickers_tuple, years=3):
    tickers = dict(tickers_tuple)
    end     = datetime.now()
    start   = end - timedelta(days=years * 365)
    data    = {}
    for name, (ticker, _) in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty or len(df) < 50:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(c).strip() for c in df.columns]
                close_cols = [c for c in df.columns if 'close' in c.lower()]
                if not close_cols: continue
                series = df[close_cols[0]]
            else:
                if 'Close' not in df.columns: continue
                series = df['Close']
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = pd.to_numeric(series, errors='coerce').dropna()
            if len(series) >= 50:
                data[name] = series
        except Exception:
            continue

    fallback = len(data) < 2
    if fallback:
        return generate_fallback_data(list(tickers.keys()), years), True
    try:
        result = pd.DataFrame(data).dropna()
        if result.shape[1] < 2:
            return generate_fallback_data(list(tickers.keys()), years), True
        return result, False
    except Exception:
        return generate_fallback_data(list(tickers.keys()), years), True

# ── Portfolio math ─────────────────────────────────────────────────────────────
def compute_returns(prices):
    return prices.pct_change().dropna()

def portfolio_perf(weights, returns):
    r  = np.sum(returns.mean() * weights) * 252
    v  = np.sqrt(weights @ (returns.cov() * 252) @ weights)
    sh = (r - 0.065) / v
    return r, v, sh

def monte_carlo(returns, n=3000):
    n_a  = len(returns.columns)
    res  = np.zeros((3, n))
    wts  = []
    for i in range(n):
        w          = np.random.dirichlet(np.ones(n_a))
        r, v, s    = portfolio_perf(w, returns)
        res[:, i]  = [r, v, s]
        wts.append(w)
    return res, wts

def optimal_portfolio(returns):
    n   = len(returns.columns)
    con = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    bnd = tuple((0.02, 0.40) for _ in range(n))
    ini = np.ones(n) / n
    res = minimize(lambda w: -portfolio_perf(w, returns)[2],
                   ini, method='SLSQP', bounds=bnd, constraints=con)
    return res.x if res.success else ini

def risk_score(age, income, horizon, existing, risk_q, deps):
    s  = 0
    s += {True: 25, False: 0}[age < 30] or ({True: 20}[age < 40] if age < 40 else ({True: 12}[age < 50] if age < 50 else ({True: 6}[age < 60] if age < 60 else 2)))
    s += 20 if horizon > 15 else (14 if horizon > 7 else (8 if horizon > 3 else 3))
    s += 20 if income > 200000 else (15 if income > 100000 else (10 if income > 50000 else 5))
    s += {"Very Low": 3, "Low": 8, "Medium": 14, "High": 20, "Very High": 25}[risk_q]
    s -= min(deps * 3, 10)
    s += {"None": 0, "FD/RD only": 3, "MF/Stocks": 8, "Diverse portfolio": 12}[existing]
    return max(0, min(100, int(s)))

def get_profile(score):
    for p, d in RISK_PROFILES.items():
        if d["score_range"][0] <= score <= d["score_range"][1]:
            return p
    return "Moderate"

def sip_projection(monthly, rate, years):
    mr = rate / 12
    m  = years * 12
    return monthly * (((1 + mr)**m - 1) / mr) * (1 + mr)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:1.4rem 0.5rem 1rem; border-bottom:1px solid rgba(201,168,76,0.2); margin-bottom:1.2rem;'>
        <div style='font-family:"DM Serif Display",serif; font-size:1.3rem; color:{GOLD};'>NiftyEdge</div>
        <div style='font-size:0.72rem; color:rgba(255,255,255,0.5); letter-spacing:0.1em; text-transform:uppercase;'>Portfolio Optimizer</div>
    </div>
    <div style='font-size:0.7rem; color:{GOLD2}; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.8rem;'>
        📋 Risk Questionnaire
    </div>
    """, unsafe_allow_html=True)

    name        = st.text_input("Your Name", "Investor", label_visibility="visible")
    age         = st.slider("Age", 18, 70, 28)
    income      = st.select_slider("Monthly Income (₹)",
                    options=[25000,50000,75000,100000,150000,200000,300000,500000],
                    value=100000,
                    format_func=lambda x: f"₹{x:,}")
    horizon     = st.slider("Investment Horizon (Years)", 1, 30, 10)
    deps        = st.slider("Dependents", 0, 5, 1)
    existing    = st.selectbox("Existing Investments",
                    ["None","FD/RD only","MF/Stocks","Diverse portfolio"])
    risk_q      = st.select_slider("Risk Comfort",
                    options=["Very Low","Low","Medium","High","Very High"],
                    value="Medium")
    monthly_sip = st.number_input("Monthly SIP (₹)", 1000, 500000, 10000, 1000)

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
    run = st.button("⚡ Generate Portfolio")

    st.markdown(f"""
    <div style='margin-top:2rem; padding:1rem; background:rgba(255,255,255,0.05);
         border-radius:10px; border:1px solid rgba(201,168,76,0.15);'>
        <div style='font-size:0.68rem; color:{GOLD2}; font-weight:700;
             letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.6rem;'>
             Universe Coverage
        </div>
        <div style='font-size:0.82rem; color:rgba(255,255,255,0.75); line-height:1.8;'>
            🔵 20 Large Cap stocks<br>
            🟠 15 Mid Cap stocks<br>
            🟢 10 Small Cap stocks<br>
            🟣 6 Index / Sectoral ETFs<br>
            🟡 1 Gold ETF · 1 Debt ETF
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Top Banner ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-banner">
    <div class="banner-left">
        <h1>Nifty<span>Edge</span> Portfolio Optimizer</h1>
        <p>Modern Portfolio Theory · Nifty 500 Universe · Markowitz Efficient Frontier · Live NSE Data</p>
    </div>
    <div class="banner-pills">
        <span class="pill">Nifty 500</span>
        <span class="pill">MPT</span>
        <span class="pill">Monte Carlo</span>
        <span class="pill">Sharpe Optimised</span>
        <span class="pill">Feb 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Landing state ──────────────────────────────────────────────────────────────
if not run:
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🧠", "Risk Profiling Engine", "6-factor questionnaire scores your risk appetite from 0–100 and maps you to Conservative, Moderate, or Aggressive profile."),
        (c2, "📐", "Markowitz Optimisation", "Runs 3,000 Monte Carlo simulations to plot the Efficient Frontier, then finds your Sharpe-optimal portfolio using SLSQP."),
        (c3, "📈", "Nifty 500 Universe", "Draws from 53 assets across large, mid & small cap stocks, index ETFs, sectoral ETFs, gold and debt — matched to your risk profile."),
    ]:
        with col:
            st.markdown(f"""
            <div class="chart-card" style="text-align:center; padding:2rem 1.5rem;">
                <div style="font-size:2.5rem; margin-bottom:0.8rem;">{icon}</div>
                <div style="font-family:'DM Serif Display',serif; font-size:1.1rem;
                     color:{NAVY}; margin-bottom:0.6rem;">{title}</div>
                <div style="font-size:0.85rem; color:{SLATE}; line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# ── Compute risk profile ───────────────────────────────────────────────────────
score   = risk_score(age, income, horizon, existing, risk_q, deps)
profile = get_profile(score)
pdata   = RISK_PROFILES[profile]
targets = {k: v for k, v in ASSET_UNIVERSE.items() if k in pdata["target_assets"]}

# ── Profile Section ────────────────────────────────────────────────────────────
st.markdown(f'<div class="section-label">Your Profile</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-title">Risk Assessment for {name}</div>', unsafe_allow_html=True)

col_profile, col_gauge, col_kpis = st.columns([1.3, 1, 2.2])

with col_profile:
    badge_map  = {"Conservative":"badge-conservative","Moderate":"badge-moderate","Aggressive":"badge-aggressive"}
    cat_colors = {"Conservative":"#065F46","Moderate":"#92400E","Aggressive":"#991B1B"}
    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-badge {badge_map[profile]}">{profile} Investor</div>
        <div style="font-size:0.82rem; color:{SLATE}; margin-bottom:1rem; line-height:1.5;">{pdata['description']}</div>
        <div class="profile-row"><span>Name</span><span>{name}</span></div>
        <div class="profile-row"><span>Age</span><span>{age} years</span></div>
        <div class="profile-row"><span>Monthly Income</span><span>₹{income:,}</span></div>
        <div class="profile-row"><span>Horizon</span><span>{horizon} years</span></div>
        <div class="profile-row"><span>Dependents</span><span>{deps}</span></div>
        <div class="profile-row"><span>Risk Comfort</span><span>{risk_q}</span></div>
        <div class="profile-row"><span>Expected Returns</span>
            <span style="color:{cat_colors[profile]};font-weight:700;">{pdata['expected_return_range']} p.a.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_gauge:
    gauge_colors = {"Conservative": GREEN, "Moderate": AMBER, "Aggressive": RED}
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 36, "family": "DM Serif Display", "color": NAVY}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0,100], "tickwidth": 0, "tickcolor": SLATE,
                     "tickvals": [0,35,65,100], "ticktext": ["0","35","65","100"],
                     "tickfont": {"size": 10}},
            "bar": {"color": gauge_colors[profile], "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,35],   "color": "#DCFCE7"},
                {"range": [35,65],  "color": "#FEF9C3"},
                {"range": [65,100], "color": "#FEE2E2"},
            ],
        }
    ))
    fig_g.update_layout(height=200, margin=dict(t=20,b=0,l=20,r=20),
                        paper_bgcolor="white", plot_bgcolor="white",
                        font={"family":"DM Sans"})
    st.markdown('<div class="chart-card" style="padding:1rem;">', unsafe_allow_html=True)
    st.plotly_chart(fig_g, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_kpis:
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card kpi-navy">
            <div class="kpi-label">Risk Score</div>
            <div class="kpi-value">{score}</div>
            <div class="kpi-sub">out of 100</div>
        </div>
        <div class="kpi-card kpi-gold">
            <div class="kpi-label">Profile</div>
            <div class="kpi-value" style="font-size:1.4rem;">{profile}</div>
            <div class="kpi-sub">investor type</div>
        </div>
        <div class="kpi-card kpi-teal">
            <div class="kpi-label">Target Return</div>
            <div class="kpi-value" style="font-size:1.4rem;">{pdata['expected_return_range']}</div>
            <div class="kpi-sub">per annum</div>
        </div>
        <div class="kpi-card kpi-green">
            <div class="kpi-label">Assets in Pool</div>
            <div class="kpi-value">{len(targets)}</div>
            <div class="kpi-sub">from Nifty 500</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Fetch live data ────────────────────────────────────────────────────────────
with st.spinner("📡 Fetching live NSE/BSE data..."):
    tickers_tuple = tuple(targets.items())
    prices, is_fallback = fetch_price_data(tickers_tuple, years=3)

if prices is None or prices.empty or prices.shape[1] < 2:
    st.error("Could not load sufficient data. Please refresh.")
    st.stop()

if is_fallback:
    st.markdown(f"""
    <div class="status-sim">
    ⚠️ <strong>Using Simulated Data</strong> — Yahoo Finance is occasionally rate-limited on cloud servers.
    Prices are generated using Geometric Brownian Motion calibrated to each asset class's historical
    return & volatility. All portfolio math (MPT, Sharpe, Monte Carlo) is fully functional.
    </div>""", unsafe_allow_html=True)
else:
    n_assets = prices.shape[1]
    n_days   = len(prices)
    st.markdown(f"""
    <div class="status-live">
    ✅ <strong>Live NSE Data</strong> — {n_assets} assets loaded · {n_days} trading days · refreshed hourly
    </div>""", unsafe_allow_html=True)

avail_assets = {k: v for k, v in targets.items() if k in prices.columns}
returns      = compute_returns(prices)

# ── Efficient Frontier ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Optimisation</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Efficient Frontier & Optimal Portfolio</div>', unsafe_allow_html=True)

with st.spinner("Running 3,000 Monte Carlo simulations…"):
    mc_res, mc_wts  = monte_carlo(returns, n=3000)
    opt_w           = optimal_portfolio(returns)
    opt_r, opt_v, opt_sh = portfolio_perf(opt_w, returns)

# Scatter colour = Sharpe ratio
fig_ef = go.Figure()
fig_ef.add_trace(go.Scatter(
    x=mc_res[1]*100, y=mc_res[0]*100,
    mode="markers",
    marker=dict(
        color=mc_res[2], colorscale=[[0,"#DBEAFE"],[0.5,"#60A5FA"],[1,"#1D4ED8"]],
        size=5, opacity=0.55,
        colorbar=dict(title="Sharpe", thickness=12, len=0.7,
                      tickfont=dict(size=10, color=SLATE),
                      titlefont=dict(size=10, color=SLATE)),
        line=dict(width=0),
    ),
    name="Simulated Portfolios",
    hovertemplate="Return: %{y:.1f}%<br>Risk: %{x:.1f}%<extra></extra>",
))
fig_ef.add_trace(go.Scatter(
    x=[opt_v*100], y=[opt_r*100],
    mode="markers+text",
    marker=dict(color=GOLD, size=22, symbol="star",
                line=dict(color=NAVY, width=2)),
    text=["  ★ Optimal"], textposition="middle right",
    textfont=dict(size=13, color=NAVY, family="DM Serif Display"),
    name="Optimal Portfolio",
    hovertemplate=f"Return: {opt_r*100:.1f}%<br>Risk: {opt_v*100:.1f}%<br>Sharpe: {opt_sh:.2f}<extra></extra>",
))
fig_ef.update_layout(
    height=420,
    plot_bgcolor=OFF_WHITE, paper_bgcolor=WHITE,
    xaxis=dict(title="Annualised Volatility (Risk) %", gridcolor=LIGHT,
               title_font=dict(size=11, color=SLATE), tickfont=dict(size=10, color=SLATE),
               showline=True, linecolor=LIGHT),
    yaxis=dict(title="Annualised Return %", gridcolor=LIGHT,
               title_font=dict(size=11, color=SLATE), tickfont=dict(size=10, color=SLATE),
               showline=True, linecolor=LIGHT),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                bordercolor=LIGHT, borderwidth=1,
                font=dict(size=11, color=NAVY)),
    margin=dict(t=20, b=40, l=50, r=20),
    font=dict(family="DM Sans"),
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="chart-title">Efficient Frontier — Nifty 500 Assets</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-sub">Each dot is a randomly-weighted portfolio. Gold star = max Sharpe Ratio (optimal risk-adjusted return).</div>', unsafe_allow_html=True)
st.plotly_chart(fig_ef, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Portfolio metrics row ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card kpi-green">
        <div class="kpi-label">Expected Annual Return</div>
        <div class="kpi-value">{opt_r*100:.1f}%</div>
        <div class="kpi-sub">annualised</div>
    </div>
    <div class="kpi-card kpi-navy">
        <div class="kpi-label">Annual Volatility (Risk)</div>
        <div class="kpi-value">{opt_v*100:.1f}%</div>
        <div class="kpi-sub">standard deviation</div>
    </div>
    <div class="kpi-card kpi-gold">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value">{opt_sh:.2f}</div>
        <div class="kpi-sub">risk-free rate 6.5%</div>
    </div>
    <div class="kpi-card kpi-teal">
        <div class="kpi-label">Assets Selected</div>
        <div class="kpi-value">{len(prices.columns)}</div>
        <div class="kpi-sub">in optimal mix</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Allocation charts ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Allocation</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Optimal Asset Allocation</div>', unsafe_allow_html=True)

alloc_df = pd.DataFrame({
    "Asset":    returns.columns.tolist(),
    "Category": [avail_assets.get(a, (None,"Other"))[1] for a in returns.columns],
    "Weight":   (opt_w * 100).round(2),
    "Ret":      (returns.mean() * 252 * 100).round(1).values,
    "Vol":      (returns.std() * np.sqrt(252) * 100).round(1).values,
}).sort_values("Weight", ascending=False)

# Build a colour list aligned to sorted df categories
def cat_chart_color(cat):
    return CAT_COLOURS.get(cat, (NAVY, LIGHT, TEAL))[2]

chart_colors = [cat_chart_color(c) for c in alloc_df["Category"]]

col_pie, col_bar = st.columns(2)

with col_pie:
    fig_pie = go.Figure(go.Pie(
        labels=alloc_df["Asset"],
        values=alloc_df["Weight"],
        hole=0.55,
        marker=dict(colors=chart_colors, line=dict(color=WHITE, width=2)),
        textinfo="percent",
        textfont=dict(size=11, color=WHITE, family="DM Sans"),
        hovertemplate="%{label}<br>Weight: %{value:.1f}%<extra></extra>",
    ))
    fig_pie.update_layout(
        height=340,
        showlegend=False,
        paper_bgcolor=WHITE,
        margin=dict(t=10, b=10, l=10, r=10),
        annotations=[dict(text=f"<b>{len(alloc_df)}</b><br>Assets",
                          x=0.5, y=0.5, font_size=16, showarrow=False,
                          font=dict(family="DM Serif Display", color=NAVY))],
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Portfolio Composition</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_bar:
    fig_bar = go.Figure(go.Bar(
        x=alloc_df["Weight"],
        y=alloc_df["Asset"],
        orientation="h",
        marker=dict(color=chart_colors, line=dict(width=0)),
        text=[f"{w:.1f}%" for w in alloc_df["Weight"]],
        textposition="outside",
        textfont=dict(size=10, color=NAVY, family="JetBrains Mono"),  # dark text on white bg
        hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>",
    ))
    fig_bar.update_layout(
        height=340,
        plot_bgcolor=OFF_WHITE,
        paper_bgcolor=WHITE,
        xaxis=dict(title="Weight (%)", gridcolor=LIGHT, showline=False,
                   tickfont=dict(size=10, color=SLATE), range=[0, alloc_df["Weight"].max()*1.18]),
        yaxis=dict(tickfont=dict(size=10, color=NAVY), autorange="reversed"),
        margin=dict(t=10, b=40, l=0, r=60),
        font=dict(family="DM Sans"),
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Weight by Asset</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Allocation table ───────────────────────────────────────────────────────────
def cat_chip_html(cat):
    c = CAT_COLOURS.get(cat, (NAVY, LIGHT, TEAL))
    return f'<span class="cat-chip" style="background:{c[1]};color:{c[0]};">{cat}</span>'

rows_html = ""
for _, row in alloc_df.iterrows():
    bar_w = min(row["Weight"] / alloc_df["Weight"].max() * 100, 100)
    bar_c = cat_chart_color(row["Category"])
    ret_color = GREEN if row["Ret"] > 0 else RED
    rows_html += f"""
    <tr>
      <td style="font-weight:600;color:{NAVY}">{row['Asset']}</td>
      <td>{cat_chip_html(row['Category'])}</td>
      <td>
        <div style="display:flex;align-items:center;gap:0.6rem;">
          <div style="flex:1;background:{LIGHT};border-radius:4px;height:7px;overflow:hidden;">
            <div style="width:{bar_w}%;background:{bar_c};height:100%;border-radius:4px;"></div>
          </div>
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:{NAVY};font-weight:600;white-space:nowrap;">{row['Weight']:.1f}%</span>
        </div>
      </td>
      <td class="num" style="color:{ret_color};">{row['Ret']:+.1f}%</td>
      <td class="num" style="color:{SLATE};">{row['Vol']:.1f}%</td>
    </tr>"""

st.markdown(f"""
<div class="chart-card">
  <div class="chart-title">Detailed Allocation Breakdown</div>
  <div class="chart-sub" style="margin-bottom:1rem;">Sorted by weight · Colour-coded by market cap segment</div>
  <table class="alloc-table">
    <thead>
      <tr>
        <th>Asset</th><th>Category</th><th>Allocation</th>
        <th style="text-align:right;">1Y Return</th><th style="text-align:right;">Volatility</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

# ── Wealth Projection ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Projection</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">SIP Wealth Projection</div>', unsafe_allow_html=True)

yrs  = list(range(1, horizon + 1))
opt_ = [sip_projection(monthly_sip, opt_r,             y) for y in yrs]
bull = [sip_projection(monthly_sip, opt_r + 0.03,      y) for y in yrs]
bear = [sip_projection(monthly_sip, max(opt_r-0.04,.05),y) for y in yrs]
inv  = [monthly_sip * 12 * y                             for y in yrs]

fig_w = go.Figure()
fig_w.add_trace(go.Scatter(x=yrs, y=[v/1e5 for v in bull], mode="lines",
    name="Bull Case", line=dict(color=GREEN, dash="dash", width=2),
    hovertemplate="Year %{x}<br>₹%{y:.1f}L<extra>Bull</extra>"))
fig_w.add_trace(go.Scatter(x=yrs, y=[v/1e5 for v in opt_], mode="lines",
    name="Base Case", line=dict(color=NAVY, width=3),
    hovertemplate="Year %{x}<br>₹%{y:.1f}L<extra>Base</extra>"))
fig_w.add_trace(go.Scatter(x=yrs, y=[v/1e5 for v in bear], mode="lines",
    name="Bear Case", line=dict(color=RED, dash="dot", width=2),
    hovertemplate="Year %{x}<br>₹%{y:.1f}L<extra>Bear</extra>"))
fig_w.add_trace(go.Scatter(x=yrs, y=[v/1e5 for v in inv],  mode="lines",
    name="Invested", line=dict(color=SLATE, dash="longdash", width=1.5),
    fill="tozeroy", fillcolor="rgba(100,116,139,0.06)",
    hovertemplate="Year %{x}<br>₹%{y:.1f}L<extra>Invested</extra>"))
fig_w.update_layout(
    height=400, plot_bgcolor=OFF_WHITE, paper_bgcolor=WHITE,
    xaxis=dict(title="Years", gridcolor=LIGHT, tickfont=dict(size=10,color=SLATE)),
    yaxis=dict(title="Value (₹ Lakhs)", gridcolor=LIGHT, tickfont=dict(size=10,color=SLATE)),
    legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                bgcolor="rgba(0,0,0,0)", font=dict(size=11,color=NAVY)),
    margin=dict(t=10, b=60, l=60, r=20),
    font=dict(family="DM Sans"),
    hovermode="x unified",
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown(f'<div class="chart-title">SIP Growth — ₹{monthly_sip:,}/month over {horizon} Years</div>', unsafe_allow_html=True)
st.plotly_chart(fig_w, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

total_inv = monthly_sip * 12 * horizon
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card kpi-navy">
        <div class="kpi-label">Total Invested</div>
        <div class="kpi-value">₹{total_inv/1e5:.1f}L</div>
        <div class="kpi-sub">over {horizon} years</div>
    </div>
    <div class="kpi-card kpi-gold">
        <div class="kpi-label">Base Case Corpus</div>
        <div class="kpi-value">₹{opt_[-1]/1e5:.1f}L</div>
        <div class="kpi-sub">at {opt_r*100:.1f}% p.a.</div>
    </div>
    <div class="kpi-card kpi-green">
        <div class="kpi-label">Wealth Multiplier</div>
        <div class="kpi-value">{opt_[-1]/total_inv:.1f}x</div>
        <div class="kpi-sub">on invested capital</div>
    </div>
    <div class="kpi-card kpi-teal">
        <div class="kpi-label">Total Gain</div>
        <div class="kpi-value">₹{(opt_[-1]-total_inv)/1e5:.1f}L</div>
        <div class="kpi-sub">absolute gain</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Historical Performance ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">Historical</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Normalised Price Performance (3 Years)</div>', unsafe_allow_html=True)

norm = prices / prices.iloc[0] * 100
fig_h = go.Figure()
palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
for i, col in enumerate(norm.columns):
    fig_h.add_trace(go.Scatter(
        x=norm.index, y=norm[col], mode="lines", name=col,
        line=dict(width=1.5, color=palette[i % len(palette)]),
        hovertemplate=f"{col}: %{{y:.1f}}<extra></extra>",
    ))
fig_h.add_hline(y=100, line_dash="dot", line_color=SLATE, line_width=1, opacity=0.5)
fig_h.update_layout(
    height=400, plot_bgcolor=OFF_WHITE, paper_bgcolor=WHITE,
    xaxis=dict(gridcolor=LIGHT, tickfont=dict(size=10,color=SLATE)),
    yaxis=dict(title="Normalised Price (Base = 100)", gridcolor=LIGHT,
               tickfont=dict(size=10,color=SLATE)),
    legend=dict(orientation="h", y=-0.25, font=dict(size=9,color=NAVY)),
    margin=dict(t=10, b=80, l=60, r=20),
    font=dict(family="DM Sans"), hovermode="x unified",
)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="chart-title">Price History (Base = 100 at Start)</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-sub">Dotted line at 100 = no gain. Assets above have outperformed since start date.</div>', unsafe_allow_html=True)
st.plotly_chart(fig_h, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Correlation Heatmap ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Correlation</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Asset Correlation Matrix</div>', unsafe_allow_html=True)

corr = returns.corr()
fig_c = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.columns,
    colorscale=[[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, NAVY]],
    zmid=0.5,
    text=corr.values.round(2),
    texttemplate="%{text}",
    textfont=dict(size=9, color=NAVY, family="JetBrains Mono"),
    hoverongaps=False,
    colorbar=dict(thickness=12, len=0.9,
                  tickfont=dict(size=9, color=SLATE)),
))
fig_c.update_layout(
    height=420, paper_bgcolor=WHITE,
    xaxis=dict(tickfont=dict(size=9, color=NAVY), side="bottom"),
    yaxis=dict(tickfont=dict(size=9, color=NAVY), autorange="reversed"),
    margin=dict(t=10, b=10, l=10, r=10),
    font=dict(family="DM Sans"),
)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="chart-title">Correlation Matrix</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-sub">Lower correlation between assets = better diversification. Dark blue = highly correlated.</div>', unsafe_allow_html=True)
st.plotly_chart(fig_c, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    <strong>NiftyEdge Portfolio Optimizer</strong> · Equity Capital Markets & Wealth Management Project<br>
    Nifty 500 universe · NSE/BSE live data via Yahoo Finance · Markowitz MPT · Sharpe Ratio Optimisation<br><br>
    <em>For academic and educational purposes only. Not financial advice.
    Past performance does not guarantee future results. Consult a SEBI-registered advisor.</em>
</div>
""", unsafe_allow_html=True)
