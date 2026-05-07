"""
Weather & Climate Agent.

For weather_relevant=False: returns a neutral thesis immediately with no API calls.

For weather_relevant=True:
  1. For each region: fetch Open-Meteo forecast + archive
  2. Compute z-scores (temp, precipitation, soil moisture)
  3. Build structured LLM prompt with z-score table
  4. Parse LLM response into AgentThesis
"""

from __future__ import annotations

import logging
from datetime import date

from agents.base import (
    compute_weather_zscores,
    fetch_weather_archive,
    fetch_weather_forecast,
    get_growing_season_context,
    parse_bullet_list,
    parse_confidence,
    parse_direction,
    parse_section,
    strip_markdown,
    _first_meaningful_line,
)
from llm_client import LLMClient
from models import AgentThesis

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an agricultural meteorologist and commodity weather analyst. "
    "You translate weather data and z-scores into supply impact assessments. "
    "You are specific about which regions are stressed, by how much, and what "
    "it means for production. You always quantify when data allows."
)

_ALL_HEADERS = [
    "DIRECTION", "CONFIDENCE", "HEADLINE",
    "CRITICAL REGIONS", "FORECAST OUTLOOK", "SUPPLY IMPACT",
]

_Z_INTERPRETATION = {
    (None, -2.5): "severely below normal",
    (-2.5, -1.5): "well below normal",
    (-1.5, -0.5): "slightly below normal",
    (-0.5,  0.5): "near normal",
    ( 0.5,  1.5): "slightly above normal",
    ( 1.5,  2.5): "well above normal",
    ( 2.5, None): "severely above normal",
}


def _interp_z(z: float) -> str:
    for (lo, hi), label in _Z_INTERPRETATION.items():
        if (lo is None or z >= lo) and (hi is None or z < hi):
            return label
    return "near normal"


async def run_weather_agent(
    commodity_config: dict,
    llm_client:       LLMClient,
) -> AgentThesis:
    """
    Run the weather & climate agent for one commodity.
    Returns neutral immediately if weather_relevant=False.
    Never raises.
    """
    name            = commodity_config["name"]
    symbol          = commodity_config.get("symbol", "")
    weather_relevant = commodity_config["weather_relevant"]
    regions         = commodity_config.get("weather_regions", [])

    if not weather_relevant:
        logger.info("[WeatherAgent] %s is not weather-sensitive — returning NEUTRAL", name)
        return AgentThesis(
            agent="weather",
            direction="NEUTRAL",
            confidence=1.0,
            headline=f"Weather is not a primary driver of {name} prices.",
            weather_relevant=False,
            supply_impact=(
                f"{name} price dynamics are driven by financial flows, "
                f"monetary policy, and industrial demand — not agricultural weather."
            ),
        )

    logger.info("[WeatherAgent] Starting for %s (%d regions)", name, len(regions))

    # ── Fetch weather data for all regions in parallel ────────────────────────
    import asyncio
    from functools import partial

    def _fetch_region(region: dict) -> tuple[dict, dict]:
        forecast = fetch_weather_forecast(region["lat"], region["lon"])
        archive  = fetch_weather_archive(region["lat"], region["lon"])
        return forecast, archive

    loop = asyncio.get_event_loop()
    fetch_tasks = [
        loop.run_in_executor(None, partial(_fetch_region, r))
        for r in regions
    ]
    fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    region_data: list[dict] = []
    all_zscores:  list[float] = []

    for region, result in zip(regions, fetch_results):
        lat   = region["lat"]
        lon   = region["lon"]
        rname = region["name"]

        if isinstance(result, Exception):
            forecast, archive = {"error": str(result)}, {"error": str(result)}
        else:
            forecast, archive = result
        zscores  = compute_weather_zscores(forecast, archive)

        if "error" not in forecast:
            # Most impactful z-score per region (temp or precip, whichever is larger)
            region_max_z = max(
                abs(zscores.get("temp_z", 0.0)),
                abs(zscores.get("precip_z", 0.0)),
            )
            all_zscores.append(region_max_z)

            # Forecast summary: next 7 days
            f_daily  = forecast.get("daily", {})
            fc_temps = f_daily.get("temperature_2m_max") or []
            fc_precip = f_daily.get("precipitation_sum") or []
            fc_temps_next  = [v for v in fc_temps[7:]  if v is not None][:7]
            fc_precip_next = [v for v in fc_precip[7:] if v is not None][:7]

            region_data.append({
                "name":        rname,
                "importance":  region["importance"],
                "reason":      region["reason"],
                "zscores":     zscores,
                "fc_temps":    fc_temps_next,
                "fc_precip":   fc_precip_next,
                "data_ok":     True,
            })
        else:
            logger.warning("[WeatherAgent] Data unavailable for region %s", rname)
            region_data.append({
                "name":       rname,
                "importance": region["importance"],
                "reason":     region["reason"],
                "zscores":    {},
                "data_ok":    False,
            })

    max_zscore  = max((abs(z) for z in all_zscores), default=0.0)
    today       = date.today()
    season_ctx  = get_growing_season_context(symbol, today.month)

    # ── Build LLM weather table ───────────────────────────────────────────────
    def _fmt_region(r: dict) -> str:
        if not r["data_ok"]:
            return f"\n{r['name']} ({r['importance'].upper()}) — data unavailable"
        z   = r["zscores"]
        tz  = z.get("temp_z", 0.0)
        pz  = z.get("precip_z", 0.0)
        mz  = z.get("moisture_z", 0.0)
        ct  = z.get("curr_temp")
        cp  = z.get("curr_precip")
        fc_t = ", ".join(f"{v:.0f}°C" for v in r["fc_temps"][:5]) or "unavailable"
        fc_p = ", ".join(f"{v:.1f}mm" for v in r["fc_precip"][:5]) or "unavailable"
        return (
            f"\n{r['name']} ({r['importance'].upper()}) — {r['reason']}\n"
            f"  Temperature:   {f'{ct:.1f}°C' if ct is not None else 'N/A'} | "
            f"z-score: {tz:+.2f} ({_interp_z(tz)})\n"
            f"  Precipitation: {f'{cp:.1f}mm' if cp is not None else 'N/A'} (7-day sum) | "
            f"z-score: {pz:+.2f} ({_interp_z(pz)})\n"
            f"  Soil moisture: z-score: {mz:+.2f} ({_interp_z(mz)})\n"
            f"  7-day temp forecast:   {fc_t}\n"
            f"  7-day precip forecast: {fc_p}"
        )

    weather_table = "\n".join(_fmt_region(r) for r in region_data)

    # ── LLM prompt ────────────────────────────────────────────────────────────
    prompt = f"""You are an agricultural meteorologist analysing weather conditions for {name}.

COMMODITY CONTEXT:
  {name} is weather-sensitive because: {regions[0]['reason'] if regions else 'multiple producing regions affected'}
  Today is {today.isoformat()}. Growing season context: {season_ctx}.
  Worst-case weather z-score across regions: {max_zscore:.2f} standard deviations from normal.

WEATHER DATA BY PRODUCING REGION (z-score = standard deviations from 5-year seasonal normal):
{weather_table}

Z-SCORE GUIDE: +2 = severely hot/wet; -2 = severely cold/dry. \
Negative precipitation z-score = drought stress (bearish supply). \
Positive temperature z-score in summer = heat stress (bearish supply). \
Positive moisture z-score = favourable conditions.

Produce your analysis with EXACTLY these labelled sections:

DIRECTION: [BULLISH_SUPPLY | BEARISH_SUPPLY | NEUTRAL]
  (BEARISH_SUPPLY = weather is stressing supply → bullish for price)
  (BULLISH_SUPPLY = weather is favourable for supply → bearish for price)
CONFIDENCE: [0.0-1.0]

HEADLINE:
[One sentence — the most important weather development right now]

CRITICAL REGIONS:
- [Region name and specific weather stress]
- [Second region if applicable]

FORECAST OUTLOOK:
[2-3 sentences on what the next 7 days look like across key regions]

SUPPLY IMPACT:
[2-3 sentences on supply implications — be quantitative where possible. \
E.g. 'Kansas accounts for X% of US winter wheat production. \
A -2σ precipitation event during grain fill reduces yield by Y%.']

Be specific about z-scores and which regions drive the headline.
If data is unavailable for a region, say so and reduce confidence."""

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        response_text = await llm_client.agent(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            purpose="weather_agent",
            max_tokens=900,
        )
    except Exception as exc:
        logger.error("[WeatherAgent] LLM call failed for %s: %s", name, exc)
        return _neutral_thesis(name, str(exc))

    # ── Parse response ────────────────────────────────────────────────────────
    direction   = parse_direction(response_text)
    confidence  = parse_confidence(response_text)
    headline    = _first_meaningful_line(parse_section(response_text, "HEADLINE", _ALL_HEADERS))
    regions_raw = parse_section(response_text, "CRITICAL REGIONS", _ALL_HEADERS)
    outlook     = parse_section(response_text, "FORECAST OUTLOOK", _ALL_HEADERS)
    supply_txt  = parse_section(response_text, "SUPPLY IMPACT", ["<<<END>>>"])
    crit_regions = parse_bullet_list(regions_raw)

    # Build weather_zscores dict for SSI consumption downstream
    weather_zscores_dict = {
        r["name"]: r["zscores"] for r in region_data if r["data_ok"]
    }

    logger.info(
        "[WeatherAgent] %s → %s (conf=%.2f, max_z=%.2f, regions=%d)",
        name, direction, confidence, max_zscore, len(region_data)
    )

    return AgentThesis(
        agent="weather",
        direction=direction,
        confidence=confidence,
        headline=headline or f"Weather analysis for {name}",
        critical_regions=crit_regions,
        forecast_outlook=outlook,
        supply_impact=supply_txt,
        weather_zscores=weather_zscores_dict,
        max_zscore=round(max_zscore, 3),
        weather_relevant=True,
    )


def _neutral_thesis(name: str, error: str) -> AgentThesis:
    return AgentThesis(
        agent="weather",
        direction="NEUTRAL",
        confidence=0.0,
        headline=f"Weather data unavailable for {name}",
        weather_relevant=True,
        error=error,
    )
