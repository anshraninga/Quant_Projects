"""
ingester.py
-----------
Pulls fresh news from three free sources and upserts into ChromaDB:
  1. CoinDesk RSS
  2. CryptoPanic (via existing CryptoPanicCache)
  3. NewsAPI (via existing key)

Called at startup and then every RAG_REFRESH_HOURS hours.
"""

import logging
import re
import time
from datetime import datetime, timezone

import requests
import feedparser

import config
from alpha_seeker.rag.vector_store import upsert_docs, make_doc_id

logger = logging.getLogger(__name__)

# Symbols Alpha-Seeker might be asked about — used to tag docs
_KNOWN_SYMBOLS = {
    "BTC":  ["bitcoin", "btc"],
    "ETH":  ["ethereum", "eth"],
    "SOL":  ["solana", "sol"],
    "BNB":  ["bnb", "binance"],
    "XRP":  ["xrp", "ripple"],
    "ADA":  ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "AVAX": ["avalanche", "avax"],
    "LINK": ["chainlink", "link"],
    "DOT":  ["polkadot", "dot"],
    "ICP":  ["internet computer", "icp", "dfinity"],
    "SUI":  ["sui network", " sui "],
    "APT":  ["aptos", " apt "],
    "TON":  ["toncoin", " ton "],
}

_COINDESK_RSS = "https://www.coindesk.com/arc/outboundfeeds/rss/"


def _detect_symbol(text: str) -> str:
    """Return best-matching symbol from known list, or 'CRYPTO' as fallback."""
    lower = text.lower()
    for sym, keywords in _KNOWN_SYMBOLS.items():
        if any(k in lower for k in keywords):
            return sym
    return "CRYPTO"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Source 1: CoinDesk RSS ────────────────────────────────

def _fetch_coindesk() -> list[dict]:
    docs = []
    try:
        feed = feedparser.parse(_COINDESK_RSS)
        for entry in feed.entries[:50]:
            title = entry.get("title", "")
            desc  = entry.get("summary", entry.get("description", ""))
            url   = entry.get("link", "")
            if not url:
                continue
            pub = entry.get("published", _now_iso())
            text = f"{title}. {desc}".strip()
            docs.append({
                "id":           make_doc_id(url),
                "text":         text[:800],
                "source":       "CoinDesk",
                "symbol":       _detect_symbol(text),
                "published_at": pub,
                "url":          url,
            })
    except Exception as exc:
        logger.warning("CoinDesk RSS fetch failed: %s", exc)
    return docs


# ── Source 2: CryptoPanic (via existing cache) ────────────

def _fetch_cryptopanic(symbol: str) -> list[dict]:
    docs = []
    if not config.CRYPTOPANIC_API_KEY:
        return docs
    try:
        # Re-use the existing module's HTTP layer to avoid duplicate calls
        from CryptoPanicCache import fetch_cryptopanic_cached
        raw = fetch_cryptopanic_cached(symbol, config.CRYPTOPANIC_API_KEY)
        for item in (raw or [])[:20]:
            url   = item.get("url", "")
            title = item.get("title", "")
            if not url or not title:
                continue
            pub = item.get("published_at", _now_iso())
            docs.append({
                "id":           make_doc_id(url),
                "text":         title[:800],
                "source":       "CryptoPanic",
                "symbol":       symbol.upper(),
                "published_at": pub,
                "url":          url,
            })
    except Exception as exc:
        logger.warning("CryptoPanic fetch failed for %s: %s", symbol, exc)
    return docs


# ── Source 3: NewsAPI ─────────────────────────────────────

def _fetch_newsapi(symbol: str) -> list[dict]:
    docs = []
    if not config.NEWSAPI_KEY:
        return docs
    keywords = _KNOWN_SYMBOLS.get(symbol.upper(), [symbol.lower()])
    query    = " OR ".join(keywords[:2])
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        query,
                "language": "en",
                "pageSize": 20,
                "sortBy":   "publishedAt",
                "apiKey":   config.NEWSAPI_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        for art in resp.json().get("articles", []):
            url   = art.get("url", "")
            title = art.get("title", "") or ""
            desc  = art.get("description", "") or ""
            pub   = art.get("publishedAt", _now_iso())
            if not url or title == "[Removed]":
                continue
            text = f"{title}. {desc}".strip()
            docs.append({
                "id":           make_doc_id(url),
                "text":         text[:800],
                "source":       "NewsAPI",
                "symbol":       symbol.upper(),
                "published_at": pub,
                "url":          url,
            })
    except Exception as exc:
        logger.warning("NewsAPI fetch failed for %s: %s", symbol, exc)
    return docs


# ── Public entry point ────────────────────────────────────

def refresh_rag(symbols: list[str] | None = None) -> int:
    """
    Pull news from all sources for the given symbols (defaults to known list)
    and upsert into ChromaDB.  Returns total docs upserted.
    """
    if symbols is None:
        symbols = list(_KNOWN_SYMBOLS.keys())

    all_docs: list[dict] = []
    all_docs.extend(_fetch_coindesk())
    for sym in symbols:
        all_docs.extend(_fetch_cryptopanic(sym))
        all_docs.extend(_fetch_newsapi(sym))
        time.sleep(0.2)  # gentle rate limiting

    total = upsert_docs(all_docs)
    logger.info("RAG refresh complete: %d docs upserted", total)
    return total
