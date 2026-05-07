"""
CommodityKing — Streamlit dashboard.
Run with: streamlit run dashboard.py
Requires the FastAPI server running on port 8000.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

COMMODITIES = {
    "wheat":       "Wheat",
    "corn":        "Corn",
    "soybeans":    "Soybeans",
    "sugar":       "Sugar (Raw)",
    "gold":        "Gold",
    "silver":      "Silver",
    "platinum":    "Platinum",
    "copper":      "Copper",
    "oil_brent":   "Brent Crude Oil",
    "natural_gas": "Natural Gas (Henry Hub)",
}

DIRECTION_EMOJI = {
    "BULLISH":        "🟢",
    "BEARISH":        "🔴",
    "BEARISH_SUPPLY": "🔴",
    "BULLISH_SUPPLY": "🟢",
    "NEUTRAL":        "🟡",
}


def direction_label(d: str) -> str:
    emoji = DIRECTION_EMOJI.get(d, "⚪")
    return f"{emoji} {d.replace('_', ' ')}"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CommodityKing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 CommodityKing")
    st.caption("Multi-agent commodity fundamental analysis")
    st.divider()

    selected = st.selectbox(
        "Select commodity",
        options=list(COMMODITIES.keys()),
        format_func=lambda x: COMMODITIES[x],
        index=3,  # sugar as default
    )
    run = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    st.caption("⏱ Analysis takes ~90 seconds")

    st.divider()
    st.markdown(
        "**Signal stack**\n"
        "- Kalman filter\n"
        "- HMM regime\n"
        "- Supply Shock Index\n"
        "- Bayesian surprise\n"
        "- Cointegration\n"
    )
    st.caption("Powered by Claude Haiku + Sonnet · ChromaDB · Open-Meteo · USDA NASS · EIA")

# ── Main ──────────────────────────────────────────────────────────────────────
if not run:
    st.markdown("## Select a commodity and click **Run Analysis**")
    st.markdown(
        "The system runs three parallel agents — Geopolitical, Weather, and "
        "Fundamentals — then verifies their theses against a historical knowledge "
        "base, runs a structured inter-agent debate, and synthesises a final "
        "recommendation with conviction score."
    )
    st.stop()

# ── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Analysing {COMMODITIES[selected]}...  this takes ~90 seconds"):
    try:
        resp = requests.post(
            f"{API_URL}/analyse",
            json={"symbols": [selected]},
            timeout=200,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the CommodityKing API.  "
            "Start it first:\n\n"
            "```\npython -m uvicorn main:app --port 8000\n```"
        )
        st.stop()
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()

a    = data["analyses"][0]
pc   = a["price_context"]
geo  = a["agents"]["geopolitical"]
wx   = a["agents"]["weather"]
fund = a["agents"]["fundamentals"]
q    = a["quantitative"]
ha   = a["historical_analog"]
fr   = a["final_recommendation"]
dbt  = a["debate"]
meta = a["metadata"]

# ═════════════════════════════════════════════════════════════════════════════
# 1. PRICE HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.header(f"{a['commodity_name']}")
st.caption(f"Generated {a['generated_at'][:19].replace('T', ' ')}  ·  {a['duration_seconds']:.0f}s  ·  {meta['llm_calls']} LLM calls  ·  ${meta['estimated_cost_usd']:.4f}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", f"{pc['current_price']} {pc['unit']}")
c2.metric("24h", f"{pc['change_24h_pct']:+.2f}%",
          delta=f"{pc['change_24h_pct']:+.2f}%",
          delta_color="normal" if pc["change_24h_pct"] >= 0 else "inverse")
c3.metric("7d",  f"{pc['change_7d_pct']:+.2f}%",
          delta=f"{pc['change_7d_pct']:+.2f}%",
          delta_color="normal" if pc["change_7d_pct"] >= 0 else "inverse")
c4.metric("Kalman trend", pc["kalman_trend"].upper(), f"{pc['signal_sigma']:.2f}σ")
c5.metric("Regime", pc["regime"].replace("_", " ").title())

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 2. FINAL RECOMMENDATION
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("Final Recommendation")

r1, r2, r3 = st.columns(3)
r1.metric("Direction",   direction_label(fr["direction"]))
r2.metric("Conviction",  f"{int(fr['conviction'] * 100)}%")
r3.metric("Horizon",     fr["time_horizon"])

st.markdown(fr["reasoning"])

up_col, dn_col = st.columns(2)
with up_col:
    st.markdown("**⬆ Upside risks**")
    for item in fr.get("upside_risks", []):
        st.markdown(f"- {item}")
with dn_col:
    st.markdown("**⬇ Downside risks**")
    for item in fr.get("downside_risks", []):
        st.markdown(f"- {item}")

if fr.get("watch_list"):
    st.markdown("**👁 Watch list**")
    for item in fr["watch_list"]:
        st.markdown(f"- {item}")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 3. EXECUTIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("Executive Summary")
st.markdown(a["executive_summary"])

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 4. AGENT REPORTS
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("Agent Reports")

tab_geo, tab_wx, tab_fund = st.tabs(["🌍  Geopolitical", "🌦️  Weather", "📦  Fundamentals"])

with tab_geo:
    g1, g2, g3 = st.columns(3)
    g1.metric("Direction",       direction_label(geo["direction"]))
    g2.metric("Confidence",      f"{int(geo['confidence'] * 100)}%")
    g3.metric("Surprise score",  f"{geo['surprise_score']:.3f}  ({geo.get('n_articles', 0)} articles)")

    st.markdown(f"**Headline:** {geo['headline']}")

    if geo.get("analysis_24h"):
        st.markdown("**24-hour analysis**")
        st.markdown(geo["analysis_24h"])
    if geo.get("analysis_7d"):
        st.markdown("**7-day narrative**")
        st.markdown(geo["analysis_7d"])
    if geo.get("historical_analog"):
        st.markdown("**Historical analog**")
        st.markdown(geo["historical_analog"])
    if geo.get("key_risks") and geo["key_risks"] != ["Insufficient data to assess risks"]:
        st.markdown("**Key risks**")
        for r in geo["key_risks"]:
            st.markdown(f"- {r}")

with tab_wx:
    w1, w2, w3 = st.columns(3)
    w1.metric("Direction",   direction_label(wx["direction"]))
    w2.metric("Confidence",  f"{int(wx['confidence'] * 100)}%")
    w3.metric("Max Z-score", f"{wx['max_zscore']:.2f}σ")

    st.markdown(f"**Headline:** {wx['headline']}")

    if wx.get("critical_regions"):
        st.markdown("**Critical regions**")
        for region in wx["critical_regions"]:
            st.markdown(f"- {region}")
    if wx.get("forecast_outlook"):
        st.markdown("**Forecast outlook**")
        st.markdown(wx["forecast_outlook"])
    if wx.get("supply_impact"):
        st.markdown("**Supply impact**")
        st.markdown(wx["supply_impact"])

with tab_fund:
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Direction",   direction_label(fund["direction"]))
    f2.metric("Confidence",  f"{int(fund['confidence'] * 100)}%")
    f3.metric("SSI",         f"{fund['ssi']:.3f}" if fund["ssi"] else "—")
    f4.metric("SSI level",   (fund["ssi_level"] or "—").title())

    st.markdown(f"**Headline:** {fund['headline']}")

    if fund.get("supply_analysis"):
        st.markdown("**Supply analysis**")
        st.markdown(fund["supply_analysis"])
    if fund.get("demand_analysis"):
        st.markdown("**Demand analysis**")
        st.markdown(fund["demand_analysis"])
    if fund.get("inventory_analysis"):
        st.markdown("**Inventory analysis**")
        st.markdown(fund["inventory_analysis"])

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 5. QUANTITATIVE MODELS
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("Quantitative Models")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("**Kalman Filter**")
    k = q["kalman"]
    st.metric("Trend",    k.get("trend", "—").upper())
    st.metric("Signal σ", f"{k.get('signal_sigma', 0):.3f}")
    st.caption("Threshold: 1.5σ  ·  " + ("✅ Signal" if k.get("is_signal") else "Noise level"))

with m2:
    st.markdown("**HMM Regime**")
    h = q["hmm"]
    st.metric("Regime",     h.get("regime_label", "—").replace("_", " ").title())
    st.metric("Confidence", f"{int(h.get('confidence', 0) * 100)}%")

with m3:
    st.markdown("**Supply Shock Index**")
    s = q["ssi"]
    st.metric("SSI",   f"{s.get('ssi', 0):.3f}")
    st.metric("Level", s.get("level", "—").title())
    st.caption(
        f"Weather: {s.get('weather_component', 0):.3f}  ·  "
        f"Geo: {s.get('geo_component', 0):.3f}  ·  "
        f"Inventory: {s.get('inventory_component', 0):.3f}"
    )

m4, m5 = st.columns(2)

with m4:
    st.markdown("**Bayesian Surprise**")
    b = q["bayesian_surprise"]
    st.metric("Score",          f"{b.get('surprise_score', 0):.3f}")
    st.metric("Interpretation", b.get("interpretation", "—").replace("_", " ").title())
    st.caption(f"KL divergence: {b.get('kl_divergence', 0):.4f}  ·  History: {b.get('history_source', '—')}")

with m5:
    st.markdown("**Cointegration**")
    coins = q.get("cointegration", [])
    if coins:
        for c in coins:
            if c["cointegrated"]:
                st.metric(c["pair"], c["signal"].replace("_", " ").title())
                st.caption(f"✅ cointegrated  ·  p = {c['pvalue']:.4f}  ·  spread z = {c['spread_zscore']:.3f}")
            else:
                st.markdown(f"**{c['pair']}**")
                st.caption(f"❌ not cointegrated on this 5-year window  ·  p = {c['pvalue']:.4f}")
    else:
        st.caption("No cointegrated pairs for this commodity.")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 6. HISTORICAL ANALOG
# ═════════════════════════════════════════════════════════════════════════════
if ha.get("found"):
    st.subheader("Historical Analog")
    h1, h2, h3 = st.columns(3)
    h1.metric("Similarity",       f"{ha['similarity']:.3f}")
    h2.metric("Date",             ha["date"])
    h3.metric("Price impact then", ha["price_impact_then"])
    st.markdown(f"**Event:** {ha['event']}")
    if ha.get("context"):
        st.markdown(f"**Context:** {ha['context']}")
    if ha.get("resolution"):
        st.markdown(f"**Resolution:** {ha['resolution']}")
    st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 7. DEBATE TRANSCRIPT
# ═════════════════════════════════════════════════════════════════════════════
if dbt.get("occurred"):
    st.subheader(f"Agent Debate  ·  {dbt['rounds']} round(s)")
    for i, rnd in enumerate(dbt["transcript"]):
        revised_tag = "  ✅ View revised" if rnd["revised"] else ""
        with st.expander(
            f"Round {i+1}: **{rnd['challenger'].title()}** challenges **{rnd['challenged'].title()}**{revised_tag}"
        ):
            st.markdown(f"**Challenge**\n\n{rnd['challenge_text']}")
            st.markdown(f"**Response**\n\n{rnd['response_text']}")
            b_col, a_col = st.columns(2)
            b_col.metric("Confidence before", f"{int(rnd['confidence_before'] * 100)}%")
            a_col.metric("Confidence after",  f"{int(rnd['confidence_after']  * 100)}%",
                         delta=f"{int((rnd['confidence_after'] - rnd['confidence_before']) * 100):+d}%")
    st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 8. FULL REPORT
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("📄  Full analyst report"):
    st.markdown(a["full_report_text"])
