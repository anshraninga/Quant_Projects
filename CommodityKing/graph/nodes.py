"""
LangGraph node functions for CommodityKing.

Execution order:
  validate_node
    → fan_out_geo_weather    (returns [Send("geo_node"), Send("weather_node")])
    → geo_node | weather_node  (parallel)
    → collect_and_fund_node  (unpacks theses, runs fundamentals with real SSI inputs)
    → rag_node               (verifies claims, refreshes news store)
    → analog_node            (finds historical analog)
    → debate_node            (multi-agent debate or skip)
    → synthesis_node         (sonnet LLM — full research note)
    → format_node            (assembles CommodityReport)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from langgraph.types import Send

import config
from agents.base import fetch_price_data, parse_bullet_list, parse_confidence, parse_direction, parse_section
from agents.fundamentals_agent import run_fundamentals_agent
from agents.geo_agent import run_geo_agent
from agents.weather_agent import run_weather_agent
from commodities_config import COMMODITIES
from graph.state import CommodityState
from llm_client import LLMClient
from models import (
    AgentThesis,
    BayesianSurpriseOutput,
    CointegrationOutput,
    CommodityReport,
    DebateRound,
    DebateTranscript,
    FinalRecommendation,
    HMMOutput,
    HistoricalAnalog,
    KalmanOutput,
    SSIOutput,
)
from rag.ingester import ingest_articles, refresh_for_commodity
from rag.vector_store import cleanup_old_news
from rag.verifier import find_historical_analog, summarise_rag_status, verify_claims

logger = logging.getLogger(__name__)


def _make_llm(symbol: str) -> LLMClient:
    return LLMClient(symbol=symbol)


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: validate
# ══════════════════════════════════════════════════════════════════════════════

def validate_node(state: CommodityState) -> dict:
    symbol = state["symbol"]
    if symbol not in COMMODITIES:
        return {
            "errors":           [f"Unknown commodity symbol: {symbol}"],
            "commodity_config": {},
            "start_time":       time.time(),
        }
    cfg = dict(COMMODITIES[symbol])
    cfg["symbol"] = symbol
    logger.info("[validate_node] Symbol %s is valid", symbol)
    return {
        "commodity_config": cfg,
        "start_time":       time.time(),
        "errors":           [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: fan-out (returns Send commands for parallel geo + weather)
# ══════════════════════════════════════════════════════════════════════════════

def fan_out_geo_weather(state: CommodityState) -> list[Send]:
    return [
        Send("geo_node",     state),
        Send("weather_node", state),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# NODES 3 & 4: geo and weather (parallel branches)
# ══════════════════════════════════════════════════════════════════════════════

async def geo_node(state: CommodityState) -> dict:
    llm = _make_llm(state["symbol"])
    thesis = await run_geo_agent(state["commodity_config"], llm)
    logger.info("[geo_node] %s → %s (conf=%.2f)", state["symbol"], thesis.direction, thesis.confidence)
    return {"agent_theses": [thesis], "llm_calls": llm.total_calls, "llm_cost_usd": llm.total_cost_usd}


async def weather_node(state: CommodityState) -> dict:
    llm = _make_llm(state["symbol"])
    thesis = await run_weather_agent(state["commodity_config"], llm)
    logger.info("[weather_node] %s → %s (conf=%.2f)", state["symbol"], thesis.direction, thesis.confidence)
    return {"agent_theses": [thesis], "llm_calls": llm.total_calls, "llm_cost_usd": llm.total_cost_usd}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5: collect geo + weather, run fundamentals with real SSI inputs
# ══════════════════════════════════════════════════════════════════════════════

async def collect_and_fund_node(state: CommodityState) -> dict:
    """
    Both parallel branches have completed. Extract geo/weather theses,
    build real SSI inputs, then run fundamentals.
    """
    theses = state.get("agent_theses", [])

    geo_thesis:     AgentThesis | None = None
    weather_thesis: AgentThesis | None = None

    for t in theses:
        if t.agent == "geopolitical":
            geo_thesis = t
        elif t.agent == "weather":
            weather_thesis = t

    # Precipitation z-scores feed SSI weather_component
    w_zscores: list[float] = []
    if weather_thesis and weather_thesis.weather_zscores:
        for region_zs in weather_thesis.weather_zscores.values():
            w_zscores.append(region_zs.get("precip_z", 0.0))

    geo_surprise = geo_thesis.surprise_score if geo_thesis else 0.5

    llm = _make_llm(state["symbol"])
    fund_thesis = await run_fundamentals_agent(
        state["commodity_config"],
        llm,
        weather_zscores=w_zscores,
        geo_surprise=geo_surprise,
    )
    logger.info("[collect_and_fund_node] fund → %s (conf=%.2f)",
                fund_thesis.direction, fund_thesis.confidence)

    return {
        "geo_thesis":     geo_thesis,
        "weather_thesis": weather_thesis,
        "fund_thesis":    fund_thesis,
        "agent_theses":   [fund_thesis],
        "llm_calls":      llm.total_calls,
        "llm_cost_usd":   llm.total_cost_usd,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6: RAG — ingest news + verify agent claims
# ══════════════════════════════════════════════════════════════════════════════

def rag_node(state: CommodityState) -> dict:
    symbol = state["symbol"]
    cfg    = state["commodity_config"]

    if config.RAG_REFRESH_ON_ANALYSIS:
        try:
            from agents.base import (
                build_news_query, fetch_newsapi_articles, fetch_rss_articles, get_date_str,
            )
            query     = build_news_query(cfg.get("name", symbol), cfg.get("key_drivers", []))
            keywords  = [cfg.get("name", symbol).lower()] + [
                d.split()[0].lower() for d in cfg.get("key_drivers", [])[:3]
            ]
            arts_api  = fetch_newsapi_articles(query, from_date=get_date_str(7),
                                               sort_by="relevancy", page_size=25)
            arts_rss  = fetch_rss_articles(keywords=keywords, max_age_days=7)
            refresh_for_commodity(symbol, arts_api, arts_rss)
            cleanup_old_news(days=config.NEWS_RETENTION_DAYS)
        except Exception as exc:
            logger.warning("[rag_node] Refresh failed for %s: %s", symbol, exc)

    rag_results: dict[str, str] = {}
    for thesis in [state.get("geo_thesis"), state.get("weather_thesis"), state.get("fund_thesis")]:
        if thesis is None:
            continue
        claims = _extract_claims(thesis)
        verification = verify_claims(claims, symbol)
        status = summarise_rag_status(verification)
        thesis.rag_status = status
        rag_results[thesis.agent] = status
        logger.info("[rag_node] %s/%s → RAG=%s", symbol, thesis.agent, status)

    return {"rag_status": rag_results}


def _extract_claims(thesis: AgentThesis) -> list[str]:
    candidates = [thesis.headline]
    if thesis.agent == "geopolitical" and thesis.analysis_24h:
        candidates.append(thesis.analysis_24h[:200])
    elif thesis.agent == "weather" and thesis.supply_impact:
        candidates.append(thesis.supply_impact[:200])
    elif thesis.agent == "fundamentals" and thesis.supply_analysis:
        candidates.append(thesis.supply_analysis[:200])
    return [c for c in candidates if c][:2]


# ══════════════════════════════════════════════════════════════════════════════
# NODE 7: historical analog
# ══════════════════════════════════════════════════════════════════════════════

def analog_node(state: CommodityState) -> dict:
    symbol = state["symbol"]
    parts  = []
    for t in [state.get("geo_thesis"), state.get("weather_thesis"), state.get("fund_thesis")]:
        if t:
            parts.append(t.headline)
    context_text = " ".join(parts)

    analog = find_historical_analog(symbol, context_text)
    if analog and analog.found:
        logger.info("[analog_node] %s: %s (sim=%.3f)", symbol, analog.event[:60], analog.similarity)
    else:
        analog = HistoricalAnalog(found=False)

    return {"historical_analog": analog}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 8: debate
# ══════════════════════════════════════════════════════════════════════════════

async def debate_node(state: CommodityState) -> dict:
    theses = [
        t for t in [state.get("geo_thesis"), state.get("weather_thesis"), state.get("fund_thesis")]
        if t is not None
    ]

    if len(theses) < 2:
        return {"debate_transcript": DebateTranscript(
            occurred=False, rounds=0,
            skip_reason="Fewer than 2 agents completed successfully",
        )}

    def _norm(d: str) -> str:
        return "BULLISH" if "BULLISH" in d else ("BEARISH" if "BEARISH" in d else "NEUTRAL")

    directions  = [_norm(t.direction) for t in theses]
    confidences = [t.confidence for t in theses]
    conf_spread = max(confidences) - min(confidences)
    all_agree   = len(set(directions)) == 1

    if all_agree and conf_spread < config.DEBATE_SKIP_THRESHOLD:
        logger.info("[debate_node] Skipping debate — all agree (%s), spread=%.2f",
                    directions[0], conf_spread)
        return {"debate_transcript": DebateTranscript(
            occurred=False, rounds=0,
            skip_reason=f"All agree ({directions[0]}), conf spread={conf_spread:.2f}",
            final_directions={t.agent: t.direction for t in theses},
        )}

    logger.info("[debate_node] Debating: directions=%s conf_spread=%.2f", directions, conf_spread)

    llm  = _make_llm(state["symbol"])
    cfg  = state["commodity_config"]

    sorted_theses = sorted(theses, key=lambda t: t.confidence, reverse=True)
    challenger    = sorted_theses[0]
    challenged    = sorted_theses[-1]

    challenge_text = await _run_challenge(challenger, challenged, cfg, llm)
    response_text  = await _run_response(challenged, challenger, challenge_text, cfg, llm)

    conf_before = challenged.confidence
    revised     = response_text.strip().upper().startswith("REVISED")
    if revised:
        challenged.confidence = min(1.0, challenged.confidence + 0.10)

    return {
        "debate_transcript": DebateTranscript(
            occurred=True,
            rounds=1,
            transcript=[DebateRound(
                challenger=challenger.agent,
                challenged=challenged.agent,
                challenge_text=challenge_text,
                response_text=response_text,
                confidence_before=conf_before,
                confidence_after=challenged.confidence,
                revised=revised,
            )],
            final_directions={t.agent: t.direction for t in theses},
        ),
        "llm_calls":      llm.total_calls,
        "llm_cost_usd":   llm.total_cost_usd,
    }


async def _run_challenge(challenger, challenged, cfg, llm) -> str:
    prompt = (
        f"You are the {challenger.agent} analyst for {cfg['name']}. "
        f"Your view: {challenger.direction} ({challenger.confidence:.0%} confidence).\n"
        f"Thesis: {challenger.headline}\n\n"
        f"The {challenged.agent} analyst disagrees: {challenged.direction} "
        f"({challenged.confidence:.0%} confidence).\n"
        f"Their headline: {challenged.headline}\n\n"
        "In 2-3 sentences, challenge their view with specific evidence. "
        "Be direct and quantitative."
    )
    try:
        return await llm.agent(prompt=prompt, purpose="debate_challenge", max_tokens=300)
    except Exception as exc:
        return f"[Challenge unavailable: {exc}]"


async def _run_response(challenged, challenger, challenge_text, cfg, llm) -> str:
    prompt = (
        f"You are the {challenged.agent} analyst for {cfg['name']}. "
        f"Your view: {challenged.direction} ({challenged.confidence:.0%} confidence).\n"
        f"The {challenger.agent} analyst challenges you:\n{challenge_text}\n\n"
        "Respond in 2-3 sentences. If this changes your view, start with 'REVISED:'. "
        "Otherwise start with 'MAINTAIN:'."
    )
    try:
        return await llm.agent(prompt=prompt, purpose="debate_response", max_tokens=300)
    except Exception as exc:
        return f"[Response unavailable: {exc}]"


# ══════════════════════════════════════════════════════════════════════════════
# NODE 9: synthesis (sonnet model — full research note)
# ══════════════════════════════════════════════════════════════════════════════

_SYNTHESIS_SYSTEM = (
    "You are a senior commodity research analyst at a quantitative hedge fund. "
    "You synthesise geopolitical intelligence, weather analysis, and supply/demand "
    "fundamentals into a rigorous, actionable research note. You are direct, "
    "quantitative, and intellectually honest about uncertainty."
)


async def synthesis_node(state: CommodityState) -> dict:
    cfg    = state["commodity_config"]
    symbol = state["symbol"]
    geo    = state.get("geo_thesis")
    wx     = state.get("weather_thesis")
    fund   = state.get("fund_thesis")
    analog = state.get("historical_analog")
    debate = state.get("debate_transcript")
    rag    = state.get("rag_status", {})

    if not any([geo, wx, fund]):
        return {"errors": ["All agents failed — cannot synthesise"]}

    price_data = fetch_price_data(cfg["ticker"])
    price_context_out = {
        "current_price":  float(price_data.get("current_price") or 0.0),
        "change_24h_pct": float(price_data.get("change_24h_pct") or 0.0),
        "change_7d_pct":  float(price_data.get("change_7d_pct") or 0.0),
    }

    prompt = f"""You are synthesising a commodity research note for {cfg['name']} ({symbol.upper()}).

CURRENT PRICE: {price_data['current_price']} {cfg['unit']}  |  24h: {price_data['change_24h_pct']:+.2f}%  |  7d: {price_data['change_7d_pct']:+.2f}%

{_fmt_section("GEOPOLITICAL & MACRO", _fmt_geo(geo))}
{_fmt_section("WEATHER & SUPPLY CONDITIONS", _fmt_weather(wx))}
{_fmt_section("SUPPLY/DEMAND FUNDAMENTALS", _fmt_fund(fund))}

RAG VERIFICATION:
  Geopolitical:  {rag.get('geopolitical', 'NOT_CHECKED')}
  Weather:       {rag.get('weather', 'NOT_CHECKED')}
  Fundamentals:  {rag.get('fundamentals', 'NOT_CHECKED')}

HISTORICAL ANALOG:
{_fmt_analog(analog)}

AGENT DEBATE:
{_fmt_debate(debate)}

Produce a complete research note with EXACTLY these labelled sections (no markdown headers):

DIRECTION: [BULLISH | BEARISH | NEUTRAL]
CONVICTION: [0.0-1.0]
TIME_HORIZON: [SHORT | MEDIUM]

EXECUTIVE SUMMARY:
[3-5 sentences. The single clearest statement of the market situation, the dominant driver, and the directional call.]

FULL REPORT:
[A complete research note: 400-600 words. Integrate all three agent views with the quant signals. Name specific numbers. Resolve any agent disagreements. Reference the historical analog if relevant. End with a clear directional statement.]

UPSIDE RISKS:
- [Risk 1 — specific]
- [Risk 2]

DOWNSIDE RISKS:
- [Risk 1]
- [Risk 2]

WATCH LIST:
- [Key data release or event to monitor]
- [Second item]

Be direct. If agents disagree, explain which view dominates and why.
Do not use markdown headers (##). Use the exact section labels above."""

    llm = _make_llm(symbol)
    try:
        response = await llm.synthesis(
            prompt=prompt,
            system=_SYNTHESIS_SYSTEM,
            purpose="synthesis",
            max_tokens=2000,
        )
    except Exception as exc:
        logger.error("[synthesis_node] LLM call failed: %s", exc)
        return {"errors": [f"Synthesis LLM failed: {exc}"]}

    _SYNTH_HDRS = ["EXECUTIVE SUMMARY", "FULL REPORT", "UPSIDE RISKS", "DOWNSIDE RISKS", "WATCH LIST"]

    direction  = parse_direction(response)
    conviction = _parse_conviction(response)
    time_h     = _parse_time_horizon(response)
    exec_summ  = parse_section(response, "EXECUTIVE SUMMARY",
                               ["FULL REPORT", "UPSIDE RISKS", "DOWNSIDE RISKS", "WATCH LIST"])
    full_txt   = parse_section(response, "FULL REPORT",
                               ["UPSIDE RISKS", "DOWNSIDE RISKS", "WATCH LIST"])
    up_raw     = parse_section(response, "UPSIDE RISKS",   ["DOWNSIDE RISKS", "WATCH LIST"])
    down_raw   = parse_section(response, "DOWNSIDE RISKS", ["WATCH LIST"])
    watch_raw  = parse_section(response, "WATCH LIST", ["<<<END>>>"])

    final_rec = FinalRecommendation(
        direction=direction,
        conviction=conviction,
        time_horizon=time_h,
        reasoning=exec_summ,
        upside_risks=parse_bullet_list(up_raw),
        downside_risks=parse_bullet_list(down_raw),
        watch_list=parse_bullet_list(watch_raw),
    )

    logger.info("[synthesis_node] %s → %s conviction=%.2f", symbol, direction, conviction)

    return {
        "executive_summary":    exec_summ,
        "full_report_text":     full_txt,
        "final_recommendation": final_rec,
        "price_context":        price_context_out,
        "llm_calls":            llm.total_calls,
        "llm_cost_usd":         llm.total_cost_usd,
    }


def _parse_conviction(text: str) -> float:
    import re
    m = re.search(r"CONVICTION\s*:\s*([0-9]\.[0-9]+|0|1)", text, re.IGNORECASE)
    if m:
        try:
            return round(min(max(float(m.group(1)), 0.0), 1.0), 2)
        except ValueError:
            pass
    return parse_confidence(text)  # fallback to CONFIDENCE field


def _parse_time_horizon(text: str) -> str:
    import re
    m = re.search(r"TIME_HORIZON\s*:\s*(SHORT|MEDIUM)", text, re.IGNORECASE)
    return m.group(1).upper() if m else "MEDIUM"


def _fmt_section(title: str, body: str) -> str:
    return f"{title}:\n{body}"


def _fmt_geo(t: AgentThesis | None) -> str:
    if not t:
        return "  Unavailable"
    lines = [
        f"  {t.direction} | conf={t.confidence:.0%} | RAG={t.rag_status}",
        f"  Headline: {t.headline}",
    ]
    if t.analysis_24h:
        lines.append(f"  24h: {t.analysis_24h[:200]}")
    if t.key_risks:
        lines.append(f"  Risks: {'; '.join(t.key_risks[:2])}")
    if t.kalman_result:
        k = t.kalman_result
        lines.append(f"  Kalman: trend={k.get('trend')} sigma={k.get('signal_sigma',0):.2f}")
    return "\n".join(lines)


def _fmt_weather(t: AgentThesis | None) -> str:
    if not t:
        return "  Unavailable"
    if not t.weather_relevant:
        return f"  {t.headline}"
    lines = [
        f"  {t.direction} | conf={t.confidence:.0%} | RAG={t.rag_status}",
        f"  Headline: {t.headline}",
        f"  Max z-score: {t.max_zscore:+.2f}σ",
    ]
    if t.supply_impact:
        lines.append(f"  Supply impact: {t.supply_impact[:200]}")
    return "\n".join(lines)


def _fmt_fund(t: AgentThesis | None) -> str:
    if not t:
        return "  Unavailable"
    lines = [
        f"  {t.direction} | conf={t.confidence:.0%} | RAG={t.rag_status}",
        f"  Headline: {t.headline}",
    ]
    if t.ssi_result:
        s = t.ssi_result
        lines.append(f"  SSI={s.get('ssi',0):.3f} ({s.get('level')}) direction={s.get('direction')}")
    if t.hmm_result:
        h = t.hmm_result
        lines.append(f"  HMM={h.get('regime_label')} conf={h.get('confidence',0):.0%}")
    if t.supply_analysis:
        lines.append(f"  Supply: {t.supply_analysis[:200]}")
    return "\n".join(lines)


def _fmt_analog(a: HistoricalAnalog | None) -> str:
    if not a or not a.found:
        return "  No analog found above threshold."
    return (
        f"  {a.event} ({a.date}) | similarity={a.similarity:.3f}\n"
        f"  Price impact: {a.price_impact_then}\n"
        f"  Resolution:   {a.resolution}"
    )


def _fmt_debate(d: DebateTranscript | None) -> str:
    if not d:
        return "  Not run."
    if not d.occurred:
        return f"  Skipped: {d.skip_reason}"
    lines = [f"  {d.rounds} round(s):"]
    for r in d.transcript:
        lines.append(f"  {r.challenger} vs {r.challenged}: {'REVISED' if r.revised else 'MAINTAINED'}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# NODE 10: format — assemble final CommodityReport
# ══════════════════════════════════════════════════════════════════════════════

def format_node(state: CommodityState) -> dict:
    cfg    = state["commodity_config"]
    symbol = state["symbol"]
    fund   = state.get("fund_thesis")
    geo    = state.get("geo_thesis")

    # Unpack quant model outputs from agent thesis dicts
    kalman_out: KalmanOutput | None = None
    hmm_out:    HMMOutput    | None = None
    ssi_out:    SSIOutput    | None = None
    bs_out:     BayesianSurpriseOutput | None = None
    coint_list: list[CointegrationOutput] = []

    if geo and geo.kalman_result:
        k = geo.kalman_result
        kalman_out = KalmanOutput(
            latest_signal=k.get("latest_signal", 0.0),
            is_signal=k.get("is_signal", False),
            trend=k.get("trend", "up"),
            signal_sigma=k.get("signal_sigma", 0.0),
        )
    if geo:
        bs_out = BayesianSurpriseOutput(
            surprise_score=geo.surprise_score,
            top_surprise_terms=list(geo.surprise_terms),
            interpretation=geo.surprise_interp or "insufficient_history",
            kl_divergence=geo.surprise_kl,
            history_source=geo.surprise_history_source,
        )

    if fund:
        if fund.hmm_result:
            h = fund.hmm_result
            hmm_out = HMMOutput(
                current_regime=h.get("current_regime", 0),
                regime_label=h.get("regime_label", "unknown"),
                confidence=h.get("confidence", 0.0),
                is_high_volatility="high" in h.get("regime_label", ""),
            )
        if fund.ssi_result:
            s = fund.ssi_result
            ssi_out = SSIOutput(
                ssi=s.get("ssi", 0.0),
                level=s.get("level", "normal"),
                direction=s.get("direction", "bearish"),
                weather_component=s.get("weather_component", 0.0),
                geo_component=s.get("geo_component", 0.0),
                inventory_component=s.get("inventory_component", 0.0),
            )
        for c in fund.cointegration_results:
            coint_list.append(CointegrationOutput(
                pair=c.get("pair", ""),
                cointegrated=c.get("cointegrated", False),
                pvalue=c.get("pvalue", 1.0),
                spread_zscore=c.get("spread_zscore", 0.0),
                signal=c.get("signal", ""),
                mean_reverting=c.get("mean_reverting", False),
            ))

    duration      = round(time.time() - state.get("start_time", time.time()), 2)
    price_context = state.get("price_context", {})

    report = CommodityReport(
        symbol=symbol,
        commodity_name=cfg.get("name", symbol),
        generated_at=datetime.now(timezone.utc).isoformat(),
        duration_seconds=duration,
        executive_summary=state.get("executive_summary", ""),
        full_report_text=state.get("full_report_text", ""),
        price_context=price_context,
        geo_thesis=geo,
        weather_thesis=state.get("weather_thesis"),
        fund_thesis=fund,
        kalman=kalman_out,
        hmm=hmm_out,
        ssi=ssi_out,
        bayesian_surprise=bs_out,
        cointegration=coint_list,
        debate=state.get("debate_transcript"),
        historical_analog=state.get("historical_analog"),
        final_recommendation=state.get("final_recommendation"),
        llm_calls=state.get("llm_calls", 0),
        estimated_cost_usd=round(state.get("llm_cost_usd", 0.0), 6),
        errors=state.get("errors", []),
    )

    return {"report": report}
