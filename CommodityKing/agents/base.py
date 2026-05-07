"""
Shared data-fetching utilities for all three agents.
No LLM calls here. Pure data in, structured data out.
All failures handled gracefully — callers never crash on missing data.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import feedparser
import numpy as np
import requests
import yfinance as yf

import config

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 15  # seconds


# ══════════════════════════════════════════════════════════════════════════════
# PRICE DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_price_data(ticker: str) -> dict:
    """
    Fetch multi-period price data via yfinance.

    Returns:
      current_price:   float
      change_24h_pct:  float
      change_7d_pct:   float
      prices_1h:       np.ndarray — hourly closes, last 2 days  (Kalman input)
      prices_1d:       np.ndarray — daily closes, last 10 days
      prices_1wk:      np.ndarray — weekly closes, last 5 years (HMM + cointegration)
      weekly_returns:  np.ndarray — pct-change of weekly closes
      error:           str | None
    """
    out = {
        "current_price":  0.0,
        "change_24h_pct": 0.0,
        "change_7d_pct":  0.0,
        "prices_1h":      np.array([]),
        "prices_1d":      np.array([]),
        "prices_1wk":     np.array([]),
        "weekly_returns": np.array([]),
        "error":          None,
    }

    try:
        t = yf.Ticker(ticker)

        h1  = t.history(period="2d",  interval="1h")
        h1d = t.history(period="10d", interval="1d")
        h5y = t.history(period="5y",  interval="1wk")

        if h1.empty or h1d.empty:
            out["error"] = f"No price data returned for {ticker}"
            return out

        closes_1h  = h1["Close"].dropna().values
        closes_1d  = h1d["Close"].dropna().values
        closes_1wk = h5y["Close"].dropna().values if not h5y.empty else np.array([])

        current = float(closes_1h[-1]) if len(closes_1h) else 0.0
        prev_1d = float(closes_1h[0])  if len(closes_1h) > 1 else current
        prev_7d = float(closes_1d[0])  if len(closes_1d) > 1 else current

        out["current_price"]  = round(current, 4)
        out["change_24h_pct"] = round((current - prev_1d) / (prev_1d + 1e-10) * 100, 2)
        out["change_7d_pct"]  = round((current - prev_7d) / (prev_7d + 1e-10) * 100, 2)
        out["prices_1h"]      = closes_1h
        out["prices_1d"]      = closes_1d
        out["prices_1wk"]     = closes_1wk

        if len(closes_1wk) > 1:
            out["weekly_returns"] = np.diff(closes_1wk) / (closes_1wk[:-1] + 1e-10)

    except Exception as exc:
        out["error"] = str(exc)
        logger.warning("[base] fetch_price_data(%s) failed: %s", ticker, exc)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# NEWS — NEWSAPI
# ══════════════════════════════════════════════════════════════════════════════

def build_news_query(commodity_name: str, key_drivers: list[str]) -> str:
    """
    Build a focused NewsAPI query from commodity name + top 3 key drivers.
    Quotes multi-word terms; joins with OR.
    """
    parts: list[str] = [commodity_name]
    for driver in key_drivers[:3]:
        words = driver.split()
        phrase = " ".join(words[:4])
        parts.append(f'"{phrase}"' if len(words) > 1 else phrase)
    return " OR ".join(parts[:5])


def fetch_newsapi_articles(
    query:     str,
    from_date: str,           # ISO date string: YYYY-MM-DD
    page_size: int = 20,
    sort_by:   str = "publishedAt",
) -> list[dict]:
    """
    Fetch articles from NewsAPI /v2/everything.
    Returns [] on 429, missing key, or network error.
    """
    if not config.NEWSAPI_KEY:
        logger.debug("[base] NEWSAPI_KEY not set — skipping NewsAPI")
        return []

    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   sort_by,
        "pageSize": page_size,
        "language": "en",
        "apiKey":   config.NEWSAPI_KEY,
    }

    try:
        resp = requests.get(config.NEWSAPI_BASE_URL, params=params,
                            timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 429:
            logger.warning("[base] NewsAPI rate limit (429) — using RSS only")
            return []
        if resp.status_code == 401:
            logger.warning("[base] NewsAPI unauthorised — check NEWSAPI_KEY")
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", [])
    except Exception as exc:
        logger.warning("[base] fetch_newsapi_articles failed: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# NEWS — RSS FEEDS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_rss_articles(
    keywords:     list[str],
    max_age_days: int = 7,
) -> list[dict]:
    """
    Fetch articles from all configured RSS feeds.
    Filters to items containing at least one keyword (case-insensitive).
    Returns [] if all feeds fail.
    """
    cutoff   = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    kw_lower = [k.lower() for k in keywords]
    articles: list[dict] = []

    for feed_url in config.RSS_FEEDS:
        try:
            # feedparser has no built-in timeout; enforce via socket
            import socket as _socket
            old_timeout = _socket.getdefaulttimeout()
            _socket.setdefaulttimeout(10)
            try:
                feed = feedparser.parse(feed_url)
            finally:
                _socket.setdefaulttimeout(old_timeout)
            for entry in feed.entries:
                title   = getattr(entry, "title", "") or ""
                summary = getattr(entry, "summary", "") or ""
                text    = f"{title} {summary}".lower()

                if not any(kw in text for kw in kw_lower):
                    continue

                published = _parse_rss_date(entry)
                if published and published < cutoff:
                    continue

                articles.append({
                    "title":       title,
                    "description": summary[:500],
                    "url":         getattr(entry, "link", ""),
                    "publishedAt": published.isoformat() if published else "",
                    "source":      feed.feed.get("title", feed_url),
                    "feed_name":   feed.feed.get("title", ""),
                })
        except Exception as exc:
            logger.debug("[base] RSS feed %s failed: %s", feed_url, exc)

    logger.debug("[base] RSS returned %d articles for keywords %s",
                 len(articles), keywords[:3])
    return articles


def _parse_rss_date(entry) -> Optional[datetime]:
    for attr in ("published_parsed", "updated_parsed"):
        val = getattr(entry, attr, None)
        if val:
            import time as _time
            try:
                ts = _time.mktime(val)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER — OPEN-METEO
# ══════════════════════════════════════════════════════════════════════════════

def fetch_weather_forecast(lat: float, lon: float) -> dict:
    """
    Fetch current conditions + 7-day forecast from Open-Meteo.
    Returns dict with daily arrays for temp_max, temp_min, precipitation,
    soil_moisture (last 7 days + next 7 days).
    Returns {"error": str} on failure.
    """
    # soil_moisture_0_to_7cm is NOT a valid daily forecast variable — archive only
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "daily":           "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "past_days":       7,
        "forecast_days":   7,
        "timezone":        "auto",
    }
    try:
        resp = requests.get(config.OPEN_METEO_URL, params=params,
                            timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("[base] fetch_weather_forecast(%.2f,%.2f) failed: %s",
                       lat, lon, exc)
        return {"error": str(exc)}


def fetch_weather_archive(lat: float, lon: float) -> dict:
    """
    Fetch 5 years of historical weather from Open-Meteo Archive API.
    Returns {"error": str} on failure.
    """
    today      = date.today()
    five_ago   = today - timedelta(days=5 * 365 + 2)

    params = {
        "latitude":    lat,
        "longitude":   lon,
        "start_date":  five_ago.isoformat(),
        "end_date":    (today - timedelta(days=1)).isoformat(),
        "daily":       "temperature_2m_max,precipitation_sum",
        "timezone":    "auto",
    }
    try:
        resp = requests.get(config.OPEN_METEO_ARCHIVE, params=params,
                            timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("[base] fetch_weather_archive(%.2f,%.2f) failed: %s",
                       lat, lon, exc)
        return {"error": str(exc)}


def compute_weather_zscores(
    forecast: dict,
    archive:  dict,
) -> dict[str, float]:
    """
    Compute z-scores for current temperature and precipitation vs 5-year normal
    for the same calendar window (±15 days around today's day-of-year).

    Returns: {"temp_z": float, "precip_z": float, "moisture_z": float}
    """
    zero = {"temp_z": 0.0, "precip_z": 0.0, "moisture_z": 0.0}

    if "error" in forecast or "error" in archive:
        return zero

    try:
        today_doy = date.today().timetuple().tm_yday

        # Current values: average of past 7 days from forecast data
        f_daily = forecast.get("daily", {})
        curr_temps   = [v for v in (f_daily.get("temperature_2m_max") or []) if v is not None]
        curr_precips = [v for v in (f_daily.get("precipitation_sum") or []) if v is not None]
        curr_moist   = [v for v in (f_daily.get("soil_moisture_0_to_7cm") or []) if v is not None]

        curr_temp   = float(np.mean(curr_temps[:7]))   if curr_temps   else None
        curr_precip = float(np.sum(curr_precips[:7]))  if curr_precips else None
        curr_moist_ = float(np.mean(curr_moist[:7]))   if curr_moist   else None

        # Historical: same ±20 day window, across all 5 years
        a_daily = archive.get("daily", {})
        a_dates  = a_daily.get("time", [])
        a_tmax   = a_daily.get("temperature_2m_max",    [None] * len(a_dates))
        a_precip = a_daily.get("precipitation_sum",     [None] * len(a_dates))
        a_moist  = a_daily.get("soil_moisture_0_to_7cm", [None] * len(a_dates))

        hist_temps:   list[float] = []
        hist_precips: list[float] = []
        hist_moists:  list[float] = []

        for i, d_str in enumerate(a_dates):
            try:
                d = date.fromisoformat(d_str)
            except Exception:
                continue
            doy = d.timetuple().tm_yday
            diff = abs(doy - today_doy)
            if diff > 180:
                diff = 365 - diff
            if diff <= 20:
                if a_tmax[i]   is not None: hist_temps.append(a_tmax[i])
                if a_precip[i] is not None: hist_precips.append(a_precip[i])
                if a_moist[i]  is not None: hist_moists.append(a_moist[i])

        def _z(current, historical):
            if current is None or not historical:
                return 0.0
            mu  = float(np.mean(historical))
            std = float(np.std(historical)) + 1e-10
            return round((current - mu) / std, 3)

        return {
            "temp_z":    _z(curr_temp,   hist_temps),
            "precip_z":  _z(curr_precip, hist_precips),
            "moisture_z": _z(curr_moist_, hist_moists),
            "curr_temp":   round(curr_temp,   1) if curr_temp   is not None else None,
            "curr_precip": round(curr_precip, 1) if curr_precip is not None else None,
            "n_hist_days": len(hist_temps),
        }
    except Exception as exc:
        logger.warning("[base] compute_weather_zscores failed: %s", exc)
        return zero


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_eia_crude_inventory() -> dict:
    """
    Fetch last 8 weeks of US crude oil inventory (excl. SPR) from EIA API.
    Returns {"values": [float, ...], "error": None} or {"values": [], "error": str}.
    """
    if not config.EIA_API_KEY:
        return {"values": [], "error": "EIA_API_KEY not set"}

    url = f"{config.EIA_BASE_URL}petroleum/stoc/wstk/data/"
    params = {
        "frequency":              "weekly",
        "data[]":                 "value",
        "facets[duoarea][]":      "NUS",
        "facets[process][]":      "SAX",
        "facets[series][]":       "WCESTUS1",
        "sort[0][column]":        "period",
        "sort[0][direction]":     "desc",
        "length":                 8,
        "api_key":                config.EIA_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("response", {}).get("data", [])
        values = []
        for r in rows:
            v = r.get("value")
            if v is not None:
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    pass
        return {"values": values, "error": None}
    except Exception as exc:
        logger.warning("[base] fetch_eia_crude_inventory failed: %s", exc)
        return {"values": [], "error": str(exc)}


def fetch_eia_natgas_storage() -> dict:
    """
    Fetch last 8 weeks of US Lower-48 natural gas working storage from EIA API.
    Returns {"values": [float, ...], "error": None} or {"values": [], "error": str}.
    """
    if not config.EIA_API_KEY:
        return {"values": [], "error": "EIA_API_KEY not set"}

    url = f"{config.EIA_BASE_URL}natural-gas/stor/wkly/data/"
    params = {
        "frequency":          "weekly",
        "data[]":             "value",
        "facets[duoarea][]":  "R48",
        "facets[process][]":  "SWO",
        "sort[0][column]":    "period",
        "sort[0][direction]": "desc",
        "length":             8,
        "api_key":            config.EIA_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("response", {}).get("data", [])
        values = []
        for r in rows:
            v = r.get("value")
            if v is not None:
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    pass
        return {"values": values, "error": None}
    except Exception as exc:
        logger.warning("[base] fetch_eia_natgas_storage failed: %s", exc)
        return {"values": [], "error": str(exc)}


def compute_inventory_deviation(values: list) -> float:
    """
    Compute inventory deviation vs recent baseline.
    Deviation = (current - mean(weeks 2-8)) / mean(weeks 2-8)
    Returns 0.0 on insufficient data.
    """
    if len(values) < 2:
        return 0.0
    try:
        floats = [float(v) for v in values]
    except (ValueError, TypeError):
        return 0.0
    current  = floats[0]
    baseline = float(np.mean(floats[1:]))
    if baseline < 1e-6:
        return 0.0
    return round((current - baseline) / baseline, 4)


def fetch_nass_supply_data(commodity: str) -> dict:
    """
    Fetch US production + ending stocks from USDA NASS QuickStats API.
    Returns 3 years of annual data for production and stocks.

    Supported commodities: wheat, corn, soybeans, sugar
    Returns {"production": [...], "stocks": [...], "error": None}
    or      {"production": [], "stocks": [], "error": str}
    """
    if not config.USDA_API_KEY:
        return {"production": [], "stocks": [], "error": "USDA_API_KEY not set"}

    # Map commodity → NASS short_desc strings (exact match)
    NASS_MAP: dict[str, dict[str, str]] = {
        "wheat":    {"production": "WHEAT - PRODUCTION, MEASURED IN BU",
                     "stocks":     "WHEAT - STOCKS, MEASURED IN BU"},
        "corn":     {"production": "CORN, GRAIN - PRODUCTION, MEASURED IN BU",
                     "stocks":     "CORN, GRAIN - STOCKS, MEASURED IN BU"},
        "soybeans": {"production": "SOYBEANS - PRODUCTION, MEASURED IN BU",
                     "stocks":     "SOYBEANS - STOCKS, MEASURED IN BU"},
        "sugar":    {"production": "SUGARBEETS - PRODUCTION, MEASURED IN TONS",
                     "stocks":     ""},   # no standard stocks series
    }

    if commodity not in NASS_MAP:
        return {"production": [], "stocks": [], "error": f"No NASS mapping for '{commodity}'"}

    mapping = NASS_MAP[commodity]
    url     = "https://quickstats.nass.usda.gov/api/api_GET/"
    min_year = date.today().year - 4

    def _fetch_series(short_desc: str) -> list[dict]:
        if not short_desc:
            return []
        params = {
            "key":           config.USDA_API_KEY,
            "source_desc":   "SURVEY",
            "short_desc":    short_desc,
            "agg_level_desc": "NATIONAL",
            "year__GE":      min_year,
            "format":        "JSON",
        }
        try:
            resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            rows = resp.json().get("data", [])
        except Exception as exc:
            logger.warning("[base] NASS fetch failed (%s): %s", short_desc, exc)
            return []

        # One row per year — take max value per year (most recent/final estimate)
        by_year: dict[int, int] = {}
        for row in rows:
            yr = int(row.get("year", 0))
            val_str = row.get("Value", "").replace(",", "").strip()
            if not val_str or not val_str.isdigit():
                continue
            val = int(val_str)
            if yr not in by_year or val > by_year[yr]:
                by_year[yr] = val

        return sorted(
            [{"year": yr, "value": val} for yr, val in by_year.items()],
            key=lambda x: x["year"],
            reverse=True,
        )

    prod  = _fetch_series(mapping["production"])
    stocks = _fetch_series(mapping["stocks"])

    if not prod and not stocks:
        return {"production": [], "stocks": [], "error": "No data returned from NASS"}

    return {"production": prod, "stocks": stocks, "error": None}


def format_nass_summary(data: dict, commodity: str, unit: str = "BU") -> str:
    """
    Format NASS production + stocks data as a readable summary for the LLM.
    Returns a multi-line string ready to embed in the fundamentals prompt.
    """
    prod   = data.get("production", [])
    stocks = data.get("stocks", [])
    lines  = [f"USDA NASS (US {commodity.title()} Supply Data):"]

    def _yoy(rows: list[dict]) -> str:
        if len(rows) < 2:
            return ""
        cur, prev = rows[0]["value"], rows[1]["value"]
        pct = (cur / prev - 1) * 100
        direction = "UP" if pct > 0 else "DOWN"
        return f"  YoY: {direction} {abs(pct):.1f}% vs {rows[1]['year']}"

    if prod:
        lines.append("  Production (US, annual):")
        for r in prod[:3]:
            lines.append(f"    {r['year']}: {r['value']/1e9:.3f}B {unit}")
        lines.append(_yoy(prod))

    if stocks:
        lines.append("  Ending Stocks (US, annual):")
        for r in stocks[:3]:
            lines.append(f"    {r['year']}: {r['value']/1e9:.3f}B {unit}")
        lines.append(_yoy(stocks))
        # Stocks-to-production ratio as rough tightness signal
        if prod and stocks and prod[0]["value"] > 0:
            stu = stocks[0]["value"] / prod[0]["value"]
            signal = (
                "TIGHT (bullish)" if stu < 0.15
                else "COMFORTABLE" if stu < 0.25
                else "AMPLE (bearish)"
            )
            lines.append(f"  Stocks-to-Production ratio: {stu:.1%}  [{signal}]")

    return "\n".join(l for l in lines if l.strip())


def fetch_baltic_dry_index() -> Optional[float]:
    """
    Scrape Baltic Dry Index from Business Insider.
    Returns None on failure — this is supplementary data only.
    """
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(config.BDI_URL, timeout=_REQUEST_TIMEOUT,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Look for a price element (structure varies; try common patterns)
        for selector in ["span.price-section__current-value",
                         "[data-value]", ".price"]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(strip=True).replace(",", "")
                val  = re.search(r"[\d.]+", text)
                if val:
                    return float(val.group())
    except Exception as exc:
        logger.debug("[base] fetch_baltic_dry_index failed: %s", exc)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LLM RESPONSE PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_direction(text: str) -> str:
    """Extract BULLISH / BEARISH / NEUTRAL / BULLISH_SUPPLY / BEARISH_SUPPLY."""
    for label in ("BULLISH_SUPPLY", "BEARISH_SUPPLY", "BULLISH", "BEARISH", "NEUTRAL"):
        if label in text.upper():
            return label
    return "NEUTRAL"


def parse_confidence(text: str) -> float:
    """Extract confidence float from LLM output. Returns 0.5 on failure."""
    match = re.search(r"CONFIDENCE[:\s]+([0-9]\.[0-9]+|0|1)", text, re.IGNORECASE)
    if match:
        try:
            return round(min(max(float(match.group(1)), 0.0), 1.0), 2)
        except ValueError:
            pass
    return 0.5


def parse_section(text: str, header: str, next_headers: list[str]) -> str:
    """
    Extract the text content of a section between `header:` and the next header.
    Case-insensitive. Returns "" if not found.
    """
    pattern = re.compile(
        rf"{re.escape(header)}\s*:?\s*\n?(.*?)(?={'|'.join(re.escape(h) for h in next_headers)}|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def strip_markdown(text: str) -> str:
    """Remove common LLM markdown formatting from a single-line string."""
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)  # bold/italic
    text = re.sub(r"`(.*?)`", r"\1", text)               # inline code
    return text.strip("# \t")


def _first_meaningful_line(text: str) -> str:
    """
    Return the first non-empty, non-markdown-only line from a section body.
    Handles LLMs that output ** on its own line before the actual content.
    """
    for line in text.splitlines():
        cleaned = strip_markdown(line.strip())
        if cleaned and cleaned not in ("**", "*", "***", "--", "—"):
            return cleaned
    return strip_markdown(text.split("\n")[0]) if text else ""


def parse_bullet_list(text: str) -> list[str]:
    """Extract bullet points from a string (lines starting with - or *)."""
    lines = text.splitlines()
    bullets = []
    for line in lines:
        line = line.strip()
        if line.startswith(("-", "*", "•")):
            item = line.lstrip("-*• ").strip()
            if item:
                bullets.append(item)
    return bullets[:5]


def get_date_str(days_ago: int = 0) -> str:
    """Return ISO date string for today minus N days."""
    return (date.today() - timedelta(days=days_ago)).isoformat()


def get_growing_season_context(commodity: str, month: int) -> str:
    """Return growing season context for a commodity based on current month."""
    contexts: dict[str, dict[int, str]] = {
        "wheat": {
            1: "dormant winter wheat period", 2: "late winter dormancy",
            3: "green-up and spring growth", 4: "jointing and stem elongation",
            5: "heading and grain fill", 6: "harvest in southern US plains",
            7: "harvest in northern plains and Canada",
            8: "post-harvest; Southern Hemisphere spring planting",
            9: "winter wheat planting in US and Europe",
            10: "primary US and European winter wheat planting",
            11: "Southern Hemisphere spring wheat grain fill",
            12: "early dormancy in Northern Hemisphere",
        },
        "corn": {
            1: "Southern Hemisphere (Brazil/Argentina) safrinha planting",
            2: "Brazil safrinha growing; US pre-planting planning",
            3: "US soil preparation; Brazil safrinha grain fill",
            4: "early US corn planting begins (southern states)",
            5: "critical US planting window",
            6: "crop establishment and early vegetative stage",
            7: "critical pollination period for US corn belt",
            8: "grain fill — most price-sensitive period",
            9: "US harvest begins; Brazil safrinha wrapping up",
            10: "US harvest peak",
            11: "Southern Hemisphere planting season",
            12: "Southern Hemisphere early vegetative stage",
        },
        "soybeans": {
            1: "Southern Hemisphere pod fill (critical quality period)",
            2: "Brazil early harvest begins; Argentina pod fill",
            3: "Brazil peak harvest",
            4: "Brazil harvest wraps up; US pre-planting",
            5: "US soybean planting window",
            6: "US crop establishment",
            7: "US vegetative stage; pod initiation",
            8: "critical pod fill period for US soybeans",
            9: "US harvest begins",
            10: "US harvest peak; Brazil pre-planting",
            11: "Brazil planting; Argentina planting",
            12: "Southern Hemisphere vegetative growth",
        },
        "sugar": {
            1: "Brazil inter-harvest (crushing ended Oct); India harvest",
            2: "India peak harvest; Thai milling season",
            3: "Northern Hemisphere off-season; India late harvest",
            4: "Brazil new crushing season begins",
            5: "Brazil crushing ramp-up; peak season through Nov",
            6: "Brazil peak crushing; India planting",
            7: "Brazil mid-season; Australia planting",
            8: "Brazil peak production month",
            9: "Brazil late season; Australian harvest begins",
            10: "Brazil crushing ends; Australian peak harvest",
            11: "Australian harvest; Brazil inter-harvest begins",
            12: "Northern Hemisphere off-season; Australian harvest",
        },
    }
    default = {i: "active production period" for i in range(1, 13)}
    return contexts.get(commodity, default).get(month, "active production period")
