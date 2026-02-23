"""
Portfolio Optimizer with Risk Profiling Engine — Indian Markets (NSE/BSE)
Project 1 | Equity Capital Markets & Wealth Management
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
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer | Indian Markets",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F4F6FB; }
    .main-header {
        background: linear-gradient(135deg, #1E2761 0%, #2563EB 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white; text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .main-header p { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0 0; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07); border-left: 4px solid #C9A84C;
        margin-bottom: 1rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.8rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card h2 { margin: 0.2rem 0 0 0; font-size: 1.8rem; font-weight: 700; color: #1E2761; }
    .section-header {
        font-size: 1.2rem; font-weight: 700; color: #1E2761;
        border-bottom: 3px solid #C9A84C; padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0;
    }
    .risk-badge {
        display: inline-block; padding: 0.3rem 1rem; border-radius: 20px;
        font-weight: 700; font-size: 0.9rem;
    }
    .stSidebar { background-color: #1E2761 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Asset Universe (Indian Markets) ─────────────────────────────────────────
ASSET_UNIVERSE = {
    # Large Cap Stocks (NSE)
    "Reliance Industries":     ("RELIANCE.NS", "Large Cap Equity"),
    "HDFC Bank":               ("HDFCBANK.NS",  "Large Cap Equity"),
    "Infosys":                 ("INFY.NS",       "Large Cap Equity"),
    "TCS":                     ("TCS.NS",        "Large Cap Equity"),
    "ICICI Bank":              ("ICICIBANK.NS",  "Large Cap Equity"),
    # Mid Cap
    "Tata Motors":             ("TATAMOTORS.NS", "Mid Cap Equity"),
    "Bajaj Finance":           ("BAJFINANCE.NS", "Mid Cap Equity"),
    # Index ETFs
    "Nifty 50 ETF (NIFTYBEES)": ("NIFTYBEES.NS", "Index ETF"),
    "Nifty Next 50 ETF":       ("JUNIORBEES.NS", "Index ETF"),
    # Gold
    "Gold ETF (GOLDBEES)":     ("GOLDBEES.NS",   "Gold"),
    # Bonds / Stable
    "Bharat Bond ETF Apr 2032":("EBBETF0432.NS", "Debt ETF"),
    # International
    "Nifty IT ETF":            ("NIFTYIT.NS",    "Sectoral ETF"),
}

RISK_PROFILES = {
    "Conservative": {
        "score_range": (0, 35),
        "color": "#16A34A",
        "description": "Capital preservation with stable returns",
        "constraints": {"equity_max": 0.30, "debt_min": 0.50, "gold_min": 0.10},
        "expected_return_range": "7–9%",
        "target_assets": ["Bharat Bond ETF Apr 2032", "Gold ETF (GOLDBEES)", "Nifty 50 ETF (NIFTYBEES)", "HDFC Bank"],
    },
    "Moderate": {
        "score_range": (36, 65),
        "color": "#CA8A04",
        "description": "Balanced growth with moderate risk tolerance",
        "constraints": {"equity_max": 0.60, "debt_min": 0.25, "gold_min": 0.05},
        "expected_return_range": "10–13%",
        "target_assets": ["Nifty 50 ETF (NIFTYBEES)", "HDFC Bank", "Infosys", "Gold ETF (GOLDBEES)", "Bharat Bond ETF Apr 2032"],
    },
    "Aggressive": {
        "score_range": (66, 100),
        "color": "#DC2626",
        "description": "Maximum growth, comfortable with high volatility",
        "constraints": {"equity_max": 0.90, "debt_min": 0.05, "gold_min": 0.05},
        "expected_return_range": "14–18%",
        "target_assets": ["Reliance Industries", "HDFC Bank", "Infosys", "TCS", "Tata Motors", "Bajaj Finance", "Gold ETF (GOLDBEES)"],
    }
}

# ─── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_price_data(tickers, years=3):
    end = datetime.now()
    start = end - timedelta(days=years*365)
    data = {}
    for name, (ticker, _) in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) > 50:
                data[name] = df['Close']
        except:
            pass
    return pd.DataFrame(data).dropna()

def compute_returns(prices):
    return prices.pct_change().dropna()

def portfolio_performance(weights, returns):
    port_return = np.sum(returns.mean() * weights) * 252
    port_vol    = np.sqrt(weights @ (returns.cov() * 252) @ weights)
    sharpe      = (port_return - 0.065) / port_vol  # 6.5% risk-free (India)
    return port_return, port_vol, sharpe

def run_monte_carlo(returns, n_portfolios=3000):
    n_assets = len(returns.columns)
    results = np.zeros((4, n_portfolios))
    weights_arr = []
    for i in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        r, v, s = portfolio_performance(w, returns)
        results[:, i] = [r, v, s, i]
        weights_arr.append(w)
    return results, weights_arr

def get_optimal_portfolio(returns):
    n = len(returns.columns)
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    bounds = tuple((0.02, 0.45) for _ in range(n))
    init   = np.ones(n) / n
    # Maximize Sharpe
    res = minimize(lambda w: -portfolio_performance(w, returns)[2], init, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x if res.success else init

def calculate_risk_score(age, income, horizon, existing_investments, risk_q, dependents):
    score = 0
    # Age (younger = more aggressive)
    if age < 30: score += 25
    elif age < 40: score += 20
    elif age < 50: score += 12
    elif age < 60: score += 6
    else: score += 2
    # Horizon
    if horizon > 15: score += 20
    elif horizon > 7: score += 14
    elif horizon > 3: score += 8
    else: score += 3
    # Income (higher = can take more risk)
    if income > 200000: score += 20
    elif income > 100000: score += 15
    elif income > 50000: score += 10
    else: score += 5
    # Self-assessed risk
    score += {"Very Low": 3, "Low": 8, "Medium": 14, "High": 20, "Very High": 25}[risk_q]
    # Dependents (more = conservative)
    score -= min(dependents * 3, 10)
    # Existing investments
    score += {"None": 0, "FD/RD only": 3, "MF/Stocks": 8, "Diverse portfolio": 12}[existing_investments]
    return max(0, min(100, score))

def get_risk_profile(score):
    for profile, data in RISK_PROFILES.items():
        lo, hi = data["score_range"]
        if lo <= score <= hi:
            return profile
    return "Moderate"

def project_wealth(monthly_sip, annual_return, years):
    monthly_rate = annual_return / 12
    months = years * 12
    future_value = monthly_sip * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
    return future_value

# ─── Sidebar: Risk Questionnaire ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Risk Profiling")
    st.markdown("---")

    name   = st.text_input("Your Name", "Investor")
    age    = st.slider("Age", 18, 70, 30)
    income = st.select_slider("Monthly Income (₹)", options=[25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000], value=100000,
                              format_func=lambda x: f"₹{x:,}")
    horizon     = st.slider("Investment Horizon (Years)", 1, 30, 10)
    dependents  = st.slider("No. of Dependents", 0, 5, 1)
    existing_inv= st.selectbox("Existing Investments", ["None", "FD/RD only", "MF/Stocks", "Diverse portfolio"])
    risk_q      = st.select_slider("Risk Comfort Level", options=["Very Low", "Low", "Medium", "High", "Very High"], value="Medium")
    monthly_sip = st.number_input("Monthly SIP Amount (₹)", min_value=1000, max_value=500000, value=10000, step=1000)

    st.markdown("---")
    run_btn = st.button("🚀 Generate My Portfolio", type="primary", use_container_width=True)

# ─── Main App ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📈 Portfolio Optimizer — Indian Markets</h1>
    <p>Modern Portfolio Theory · NSE/BSE Assets · Efficient Frontier · Wealth Projection</p>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.info("👈 Fill in your risk profile in the sidebar and click **Generate My Portfolio** to begin.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🔬 What This Tool Does**
        - Scores your risk profile from 0–100
        - Selects appropriate assets from NSE/BSE
        - Runs 3,000 Monte Carlo simulations
        - Plots the Efficient Frontier
        - Finds your optimal Sharpe Ratio portfolio
        """)
    with col2:
        st.markdown("""
        **📊 Methodology**
        - Markowitz Modern Portfolio Theory
        - 3-year historical price data (NSE)
        - Annualised return & volatility
        - Sharpe Ratio = (Return − 6.5%) / Vol
        - Scipy SLSQP optimisation
        """)
    with col3:
        st.markdown("""
        **💼 Asset Universe**
        - Large Cap: Reliance, HDFC Bank, TCS, Infosys
        - Mid Cap: Tata Motors, Bajaj Finance
        - Index ETFs: Nifty50, Nifty Next 50
        - Gold: GOLDBEES ETF
        - Debt: Bharat Bond ETF
        """)
    st.stop()

# ─── Compute Risk Score ────────────────────────────────────────────────────────
risk_score  = calculate_risk_score(age, income, horizon, existing_inv, risk_q, dependents)
risk_profile= get_risk_profile(risk_score)
profile_data= RISK_PROFILES[risk_profile]
selected_assets = {k: v for k, v in ASSET_UNIVERSE.items() if k in profile_data["target_assets"]}

# ─── Profile Summary ───────────────────────────────────────────────────────────
st.markdown(f'<div class="section-header">👤 Risk Profile: {name}</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card"><h3>Risk Score</h3><h2>{risk_score}/100</h2></div>""", unsafe_allow_html=True)
with col2:
    color = profile_data["color"]
    st.markdown(f"""<div class="metric-card" style="border-left-color:{color}"><h3>Risk Profile</h3><h2 style="color:{color}">{risk_profile}</h2></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card"><h3>Expected Return Range</h3><h2>{profile_data['expected_return_range']}</h2></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card"><h3>Investment Horizon</h3><h2>{horizon} Years</h2></div>""", unsafe_allow_html=True)

st.markdown(f"**Profile Description:** {profile_data['description']}")

# Risk Score Gauge
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_score,
    title={"text": "Risk Score", "font": {"size": 16}},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": profile_data["color"]},
        "steps": [
            {"range": [0, 35],  "color": "#DCFCE7"},
            {"range": [35, 65], "color": "#FEF9C3"},
            {"range": [65, 100],"color": "#FEE2E2"},
        ],
        "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": risk_score}
    }
))
fig_gauge.update_layout(height=250, margin=dict(t=30, b=10, l=30, r=30))
st.plotly_chart(fig_gauge, use_container_width=True)

# ─── Fetch Data ────────────────────────────────────────────────────────────────
with st.spinner("📡 Fetching live price data from NSE..."):
    prices = fetch_price_data(selected_assets, years=3)

if prices.empty or len(prices.columns) < 2:
    st.error("Could not fetch sufficient price data. Please check your internet connection and try again.")
    st.stop()

available_assets = {k: v for k, v in selected_assets.items() if k in prices.columns}
returns = compute_returns(prices)

st.success(f"✅ Loaded {len(prices.columns)} assets | {len(prices)} trading days of data (3 years)")

# ─── Run Monte Carlo Simulations ──────────────────────────────────────────────
st.markdown('<div class="section-header">🎲 Efficient Frontier (3,000 Simulations)</div>', unsafe_allow_html=True)

with st.spinner("Running Monte Carlo simulations..."):
    results, weights_arr = run_monte_carlo(returns, n_portfolios=3000)
    optimal_weights      = get_optimal_portfolio(returns)
    opt_ret, opt_vol, opt_sharpe = portfolio_performance(optimal_weights, returns)

# Efficient Frontier Plot
fig_ef = go.Figure()
fig_ef.add_trace(go.Scatter(
    x=results[1, :] * 100,
    y=results[0, :] * 100,
    mode="markers",
    marker=dict(color=results[2, :], colorscale="Viridis", size=4, opacity=0.6,
                colorbar=dict(title="Sharpe Ratio")),
    text=[f"Sharpe: {results[2,i]:.2f}" for i in range(results.shape[1])],
    name="Simulated Portfolios",
))
fig_ef.add_trace(go.Scatter(
    x=[opt_vol * 100], y=[opt_ret * 100],
    mode="markers+text",
    marker=dict(color="#C9A84C", size=18, symbol="star"),
    text=["★ Optimal"], textposition="top right",
    textfont=dict(size=13, color="#1E2761"),
    name="Optimal Portfolio (Max Sharpe)",
))
fig_ef.update_layout(
    title="Efficient Frontier — NSE/BSE Assets",
    xaxis_title="Annualised Volatility (Risk) %",
    yaxis_title="Annualised Return %",
    height=450,
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(x=0.01, y=0.99),
)
st.plotly_chart(fig_ef, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1: st.metric("📈 Expected Annual Return", f"{opt_ret*100:.2f}%")
with col2: st.metric("📊 Annual Volatility (Risk)", f"{opt_vol*100:.2f}%")
with col3: st.metric("⚡ Sharpe Ratio", f"{opt_sharpe:.3f}")

# ─── Optimal Allocation ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Optimal Asset Allocation</div>', unsafe_allow_html=True)

allocation_df = pd.DataFrame({
    "Asset": returns.columns.tolist(),
    "Category": [available_assets.get(a, (None, "Other"))[1] for a in returns.columns],
    "Weight (%)": (optimal_weights * 100).round(2),
    "Annual Return (%)": (returns.mean() * 252 * 100).round(2).values,
    "Annual Volatility (%)": (returns.std() * np.sqrt(252) * 100).round(2).values,
})
allocation_df = allocation_df.sort_values("Weight (%)", ascending=False)

col1, col2 = st.columns([1, 1])
with col1:
    fig_pie = go.Figure(go.Pie(
        labels=allocation_df["Asset"],
        values=allocation_df["Weight (%)"],
        hole=0.45,
        marker=dict(colors=px.colors.qualitative.Bold),
        textinfo="label+percent",
        textfont_size=11,
    ))
    fig_pie.update_layout(
        title="Portfolio Allocation", height=400,
        showlegend=False, paper_bgcolor="white",
        annotations=[dict(text="Optimal", x=0.5, y=0.5, font_size=14, showarrow=False)]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar = go.Figure(go.Bar(
        x=allocation_df["Weight (%)"],
        y=allocation_df["Asset"],
        orientation="h",
        marker_color="#1E2761",
        text=[f"{w:.1f}%" for w in allocation_df["Weight (%)"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Weight by Asset", height=400,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Weight (%)", yaxis_title="",
        xaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.dataframe(
    allocation_df.style.background_gradient(subset=["Weight (%)"], cmap="Blues")
                       .format({"Weight (%)": "{:.2f}%", "Annual Return (%)": "{:.2f}%", "Annual Volatility (%)": "{:.2f}%"}),
    use_container_width=True, hide_index=True
)

# ─── Wealth Projection ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💰 SIP Wealth Projection</div>', unsafe_allow_html=True)

years_list   = list(range(1, horizon + 1))
optimistic   = [project_wealth(monthly_sip, opt_ret + 0.03, y) for y in years_list]
base         = [project_wealth(monthly_sip, opt_ret,         y) for y in years_list]
conservative = [project_wealth(monthly_sip, max(opt_ret - 0.04, 0.05), y) for y in years_list]
invested     = [monthly_sip * 12 * y for y in years_list]

fig_wealth = go.Figure()
fig_wealth.add_trace(go.Scatter(x=years_list, y=[v/1e5 for v in optimistic],   mode="lines", name="Optimistic Scenario",   line=dict(color="#16A34A", dash="dash", width=2)))
fig_wealth.add_trace(go.Scatter(x=years_list, y=[v/1e5 for v in base],         mode="lines", name="Base Case (Optimal)",   line=dict(color="#1E2761", width=3)))
fig_wealth.add_trace(go.Scatter(x=years_list, y=[v/1e5 for v in conservative], mode="lines", name="Conservative Scenario", line=dict(color="#DC2626", dash="dot", width=2)))
fig_wealth.add_trace(go.Scatter(x=years_list, y=[v/1e5 for v in invested],     mode="lines", name="Total Invested",        line=dict(color="#94A3B8", dash="longdash", width=1.5), fill="tozeroy", fillcolor="rgba(148,163,184,0.1)"))

fig_wealth.update_layout(
    title=f"SIP Wealth Growth — ₹{monthly_sip:,}/month over {horizon} Years",
    xaxis_title="Years", yaxis_title="Portfolio Value (₹ Lakhs)",
    height=420, plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(x=0.01, y=0.99), xaxis=dict(gridcolor="#E2E8F0"), yaxis=dict(gridcolor="#E2E8F0"),
)
st.plotly_chart(fig_wealth, use_container_width=True)

col1, col2, col3 = st.columns(3)
total_invested = monthly_sip * 12 * horizon
with col1: st.metric("💸 Total Invested", f"₹{total_invested/1e5:.2f}L")
with col2: st.metric("🎯 Base Case Corpus", f"₹{base[-1]/1e5:.2f}L")
with col3: st.metric("📈 Wealth Multiplier", f"{base[-1]/total_invested:.2f}x")

# ─── Historical Performance ────────────────────────────────────────────────────
st.markdown('<div class="section-header">📅 Historical Normalized Performance (3 Years)</div>', unsafe_allow_html=True)
norm_prices = prices / prices.iloc[0] * 100
fig_hist = go.Figure()
for col in norm_prices.columns:
    fig_hist.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[col], mode="lines", name=col))
fig_hist.update_layout(
    height=380, plot_bgcolor="white", paper_bgcolor="white",
    xaxis_title="Date", yaxis_title="Normalized Price (Base=100)",
    xaxis=dict(gridcolor="#E2E8F0"), yaxis=dict(gridcolor="#E2E8F0"),
    legend=dict(orientation="h", y=-0.2)
)
st.plotly_chart(fig_hist, use_container_width=True)

# ─── Correlation Heatmap ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔗 Asset Correlation Matrix</div>', unsafe_allow_html=True)
corr = returns.corr()
fig_corr = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.columns,
    colorscale="RdBu_r", zmid=0,
    text=corr.values.round(2), texttemplate="%{text}", textfont_size=10,
))
fig_corr.update_layout(height=380, paper_bgcolor="white")
st.plotly_chart(fig_corr, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#64748B; font-size:0.85rem'>
    Portfolio Optimizer | Equity Capital Markets & Wealth Management Project | NSE/BSE Data via Yahoo Finance<br>
    <em>For academic purposes only. Not financial advice. Past performance does not guarantee future results.</em>
</div>
""", unsafe_allow_html=True)
