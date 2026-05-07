"""
NewsAnalysis.py
---------------
Fetches and scores crypto news using FinBERT.
Sources: CryptoPanic (via CryptoPanicCache) + NewsAPI

CryptoPanic fetches go through CryptoPanicCache.py so NewsAnalysis and
SentimentAnalysis never call the API simultaneously with the same key.
"""

import os
import math
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from transformers import pipeline

from SignalNormalizer import normalize_news, SignalVector
from CryptoPanicCache  import fetch_cryptopanic_cached

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()


# ─────────────────────────────────────────────────────────
# FINBERT
# ─────────────────────────────────────────────────────────

print("⏳ Loading FinBERT model (first run may take ~30s to download)...")
finbert = pipeline(
    task="text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    top_k=None, truncation=True, max_length=512,
)
print("✅ FinBERT ready.")


def finbert_score(text: str) -> dict:
    if not text or not text.strip():
        return {"bullish_score": 0.0, "bearish_score": 0.0,
                "neutral_score": 1.0, "label": "neutral"}
    try:
        results   = finbert(text[:1000])[0]
        scores    = {r["label"].lower(): r["score"] for r in results}
        raw_label = max(scores, key=scores.get)
        label_map = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
        return {
            "bullish_score": round(scores.get("positive", 0.0), 6),
            "bearish_score": round(scores.get("negative", 0.0), 6),
            "neutral_score": round(scores.get("neutral",  0.0), 6),
            "label":         label_map.get(raw_label, "neutral"),
        }
    except Exception as e:
        print(f"  [FinBERT error] {e}")
        return {"bullish_score": 0.0, "bearish_score": 0.0,
                "neutral_score": 1.0, "label": "neutral"}


# ─────────────────────────────────────────────────────────
# RECENCY WEIGHTING
# ─────────────────────────────────────────────────────────

def _recency_weight(published_at: str) -> float:
    """
    Exponential decay by article age. Half-life = 16 h.
    A 16h-old article counts as 0.5×, a 48h-old as ~0.05×.
    Prevents stale news (already priced in) from inflating the signal.
    """
    if not published_at:
        return 1.0
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        hours_old = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
        return max(math.exp(-hours_old / 16.0), 0.05)
    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────
# 1. CRYPTOPANIC  (via shared cache)
# ─────────────────────────────────────────────────────────

def fetch_cryptopanic_news(symbol: str, limit: int = 10) -> list:
    if not os.getenv("CRYPTOPANIC_API_KEY", "").strip():
        print("  [CryptoPanic] No API key — skipping")
        return []

    posts = fetch_cryptopanic_cached(symbol=symbol, kind="news", limit=limit)
    if not posts:
        print(f"  [CryptoPanic] No news found for {symbol}")
        return []

    return [
        {
            "source":       "CryptoPanic",
            "title":        p.get("title", ""),
            "description":  "",
            "url":          p.get("url", ""),
            "published_at": p.get("published_at", ""),
        }
        for p in posts
    ]


# ─────────────────────────────────────────────────────────
# 2. NEWSAPI
# ─────────────────────────────────────────────────────────

SYMBOL_TO_NAME = {
    # Majors
    "BTC": "Bitcoin",              "ETH": "Ethereum",
    "SOL": "Solana",               "ICP": "Internet Computer",
    "NEAR": "NEAR Protocol",       "ADA": "Cardano",
    "DOT": "Polkadot",             "AVAX": "Avalanche",
    "MATIC": "Polygon",            "POL": "Polygon",
    "LINK": "Chainlink",           "UNI": "Uniswap",
    "ATOM": "Cosmos",              "LTC": "Litecoin",
    "DOGE": "Dogecoin",            "XRP": "Ripple",
    "BNB": "Binance Coin",         "FTM": "Fantom",
    "SAND": "The Sandbox",         "MANA": "Decentraland",
    "AXS": "Axie Infinity",        "VET": "VeChain",
    # Portfolio coins
    "OXT": "Orchid Protocol",      "FET": "Fetch.ai",
    "ZEN": "Horizen",              "CSPR": "Casper Network",
    "ACX": "Across Protocol",      "SAFE": "Safe crypto",
    "CELO": "Celo blockchain",     "CFX": "Conflux Network",
    "MINA": "Mina Protocol",       "AGI": "SingularityNET",
    "TNSR": "Tensor crypto",       "LINEA": "Linea blockchain",
    "PORTAL": "Portal crypto",     "NEAR": "NEAR Protocol",
}


def fetch_newsapi_news(symbol: str, limit: int = 10) -> list:
    if not NEWSAPI_KEY:
        print("  [NewsAPI] No API key — skipping")
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
    full_name = SYMBOL_TO_NAME.get(symbol.upper())
    query     = full_name if full_name else (
                f"{symbol} cryptocurrency" if len(symbol) <= 5 else symbol)

    try:
        r = requests.get("https://newsapi.org/v2/everything", params={
            "apiKey": NEWSAPI_KEY, "q": query, "language": "en",
            "sortBy": "publishedAt", "pageSize": limit, "from": from_date,
        }, timeout=10)
        if r.status_code in (401, 429):
            return []
        r.raise_for_status()
        articles = r.json().get("articles", [])[:limit]
        return [
            {
                "source":       a.get("source", {}).get("name", "NewsAPI"),
                "title":        a.get("title", "") or "",
                "description":  a.get("description", "") or "",
                "url":          a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            }
            for a in articles
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
    except Exception as e:
        print(f"  [NewsAPI ERROR] {e}")
        return []


# ─────────────────────────────────────────────────────────
# 3. SCORE + AGGREGATE
# ─────────────────────────────────────────────────────────

def score_articles(articles: list) -> list:
    for a in articles:
        text = f"{a.get('title', '')}. {a.get('description', '')}".strip(". ")
        a["finbert"] = finbert_score(text)
    return articles


def aggregate_sentiment_display(scored: list) -> dict:
    if not scored:
        return {"total_articles": 0, "bullish_pct": 0.0,
                "bearish_pct": 0.0, "neutral_pct": 0.0, "overall": "neutral"}
    n        = len(scored)
    avg_bull = sum(a["finbert"]["bullish_score"] for a in scored) / n
    avg_bear = sum(a["finbert"]["bearish_score"] for a in scored) / n
    avg_neut = sum(a["finbert"]["neutral_score"] for a in scored) / n
    total    = avg_bull + avg_bear + avg_neut + 1e-10
    bull_pct = round(avg_bull / total * 100, 1)
    bear_pct = round(avg_bear / total * 100, 1)
    neut_pct = round(avg_neut / total * 100, 1)
    overall  = "bullish" if bull_pct > bear_pct + 10 else \
               "bearish" if bear_pct > bull_pct + 10 else "neutral"
    return {"total_articles": n, "bullish_pct": bull_pct,
            "bearish_pct": bear_pct, "neutral_pct": neut_pct, "overall": overall}


# ─────────────────────────────────────────────────────────
# 4. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────

def run_news_analysis(symbol: str) -> dict:
    print(f"\n📰 News analysis for: {symbol.upper()}")

    raw_articles = fetch_cryptopanic_news(symbol) + fetch_newsapi_news(symbol)

    # Deduplicate by URL — same story from CryptoPanic + NewsAPI would
    # otherwise be double-counted and amplify a single narrative.
    seen_urls: set = set()
    all_articles = []
    for a in raw_articles:
        url = a.get("url", "")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        all_articles.append(a)

    if not all_articles:
        print("  ⚠️  No articles found — check API keys in .env")
        empty_vec = normalize_news([], base_weight=0.125, n_articles=0)
        return {"symbol": symbol.upper(), "signal_vector": empty_vec,
                "news_sentiment": aggregate_sentiment_display([]),
                "top_headlines": [], "raw_articles": []}

    print(f"  Fetched {len(all_articles)} articles — scoring with FinBERT...")
    scored = score_articles(all_articles)
    disp   = aggregate_sentiment_display(scored)
    print(f"  FinBERT raw -> Bull: {disp['bullish_pct']}% | "
          f"Bear: {disp['bearish_pct']}% | Neutral: {disp['neutral_pct']}%")

    raw_scores       = [a["finbert"]                       for a in scored if "finbert" in a]
    recency_weights  = [_recency_weight(a.get("published_at", "")) for a in scored if "finbert" in a]
    signal_vec = normalize_news(raw_scores=raw_scores, base_weight=0.125,
                                n_articles=len(raw_scores),
                                article_weights=recency_weights)

    print(f"  After calibration -> Bull {signal_vec.bullish_pct()}% | "
          f"Bear {signal_vec.bearish_pct()}% | "
          f"Neutral {round(signal_vec.neutral*100,1)}% | "
          f"Confidence {signal_vec.confidence:.2f} | EffW {signal_vec.effective_weight:.4f}")

    top_headlines = [
        {"title": a["title"], "source": a["source"],
         "published_at": a.get("published_at", ""),
         "label": a["finbert"]["label"],
         "bullish_score": a["finbert"]["bullish_score"],
         "bearish_score": a["finbert"]["bearish_score"],
         "neutral_score": a["finbert"]["neutral_score"],
         "url": a.get("url", "")}
        for a in scored[:5] if a.get("title")
    ]

    print("  Top Headlines:")
    for i, h in enumerate(top_headlines, 1):
        print(f"    {i}. [{h['label'].upper():8}] {h['title'][:75]}")

    return {
        "symbol":        symbol.upper(),
        "signal_vector": signal_vec,
        "news_sentiment": {**disp,
                           "bullish_pct": signal_vec.bullish_pct(),
                           "bearish_pct": signal_vec.bearish_pct(),
                           "neutral_pct": round(signal_vec.neutral * 100, 1)},
        "top_headlines": top_headlines,
        "raw_articles":  scored,
    }


if __name__ == "__main__":
    sym = input("Enter symbol (e.g. BTC, ETH): ").strip()
    run_news_analysis(sym)