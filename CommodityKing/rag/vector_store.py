"""
ChromaDB vector store — low-level operations only.

Two collections:
  commodity_news    : live news articles, refreshed per analysis run
  historical_events : curated event database, loaded once at startup

All embedding handled by ChromaDB via SentenceTransformer.
No LLM calls, no API calls. Pure local operations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

import config

logger = logging.getLogger(__name__)

# ── Embedding function (shared across all collections) ────────────────────────
_EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=config.EMBEDDING_MODEL
)

# ── Singleton client ──────────────────────────────────────────────────────────
_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        Path(config.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    return _client


def get_collection(name: str) -> chromadb.Collection:
    return _get_client().get_or_create_collection(
        name=name,
        embedding_function=_EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


# ── Initialise historical events collection ───────────────────────────────────

def init_historical_events(json_path: str | None = None) -> int:
    """
    Load historical_events.json into ChromaDB on startup.
    Skips documents already present (idempotent).
    Returns number of documents in collection after loading.
    """
    path = json_path or config.HISTORICAL_EVENTS_PATH
    col  = get_collection(config.HISTORY_COLLECTION)

    try:
        events: list[dict] = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("historical_events.json not found at %s", path)
        return col.count()

    existing_ids: set[str] = set(col.get()["ids"])
    to_add = [e for e in events if e["id"] not in existing_ids]

    if not to_add:
        logger.info("[RAG] Historical events already loaded (%d docs)", col.count())
        return col.count()

    col.add(
        ids=[e["id"] for e in to_add],
        documents=[_format_event_text(e) for e in to_add],
        metadatas=[
            {
                "commodity":        e["commodity"],
                "date":             e["date"],
                "price_impact":     e.get("price_impact", ""),
                "event_short":      e["event"][:120],
            }
            for e in to_add
        ],
    )
    logger.info("[RAG] Loaded %d new historical events (total %d)", len(to_add), col.count())
    return col.count()


def _format_event_text(event: dict) -> str:
    return (
        f"Commodity: {event['commodity']}. "
        f"Date: {event['date']}. "
        f"Event: {event['event']}. "
        f"Price impact: {event.get('price_impact', 'unknown')}. "
        f"Context: {event.get('context', '')}. "
        f"Resolution: {event.get('resolution', '')}"
    )


# ── News collection operations ────────────────────────────────────────────────

def add_news_articles(
    articles: list[dict],
    commodity: str,
) -> int:
    """
    Upsert news articles into the commodity_news collection.

    Each article dict must have: {id, title, description, source, published_at, url}
    'id' should be a stable hash of the URL (or the URL itself, truncated).
    Returns count of documents added (skipping duplicates).
    """
    if not articles:
        return 0

    col          = get_collection(config.NEWS_COLLECTION)
    existing_ids = set(col.get()["ids"])

    new_articles = [a for a in articles if a.get("id") and a["id"] not in existing_ids]
    if not new_articles:
        return 0

    col.add(
        ids=[a["id"] for a in new_articles],
        documents=[_format_article_text(a) for a in new_articles],
        metadatas=[
            {
                "commodity":    commodity,
                "source":       a.get("source", ""),
                "published_at": a.get("published_at", ""),
                "url":          a.get("url", "")[:500],
            }
            for a in new_articles
        ],
    )
    return len(new_articles)


def _format_article_text(article: dict) -> str:
    title = article.get("title") or ""
    desc  = article.get("description") or ""
    return f"{title}. {desc}".strip(". ")


# ── Semantic search ───────────────────────────────────────────────────────────

def query_news(
    text:      str,
    commodity: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Search commodity_news collection for articles similar to text.
    Filters to the specified commodity.
    Returns list of {text, source, published_at, url, distance}.
    """
    col = get_collection(config.NEWS_COLLECTION)
    if col.count() == 0:
        return []

    try:
        results = col.query(
            query_texts=[text],
            n_results=min(n_results, col.count()),
            where={"commodity": commodity},
        )
        return _unpack_results(results)
    except Exception as exc:
        logger.warning("[RAG] query_news failed: %s", exc)
        return []


def query_historical(
    text:      str,
    commodity: str,
    n_results: int = 3,
) -> list[dict]:
    """
    Search historical_events collection for events similar to text.
    Filters to the specified commodity.
    Returns list of {text, commodity, date, event_short, distance}.
    """
    col = get_collection(config.HISTORY_COLLECTION)
    if col.count() == 0:
        return []

    try:
        results = col.query(
            query_texts=[text],
            n_results=min(n_results, col.count()),
            where={"commodity": commodity},
        )
        return _unpack_results(results)
    except Exception as exc:
        logger.warning("[RAG] query_historical failed: %s", exc)
        return []


def _unpack_results(results: dict) -> list[dict]:
    """Flatten ChromaDB query results into a list of dicts."""
    out = []
    docs      = results.get("documents", [[]])[0]
    metas     = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        out.append({
            "text":     doc,
            "distance": round(float(dist), 4),
            **meta,
        })
    return out


# ── Bayesian baseline retrieval ───────────────────────────────────────────────

def get_articles_for_commodity(symbol: str, days_back: int = 30) -> list[str]:
    """
    Return document texts from the commodity_news collection for `symbol`
    published within the last `days_back` days.

    ChromaDB comparison operators require numeric types, so we filter by
    commodity in the DB and apply the date cutoff in Python.
    Returns empty list when the collection has no relevant history.
    """
    col = get_collection(config.NEWS_COLLECTION)
    if col.count() == 0:
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        results = col.get(
            where={"commodity": {"$eq": symbol}},
            include=["documents", "metadatas"],
        )
        docs      = results.get("documents") or []
        metadatas = results.get("metadatas")  or []

        # Filter by date in Python — published_at stored as ISO string prefix
        filtered = [
            doc for doc, meta in zip(docs, metadatas)
            if (meta.get("published_at") or "") >= cutoff
        ]
        logger.debug(
            "[RAG] get_articles_for_commodity(%s, %dd): %d/%d docs (cutoff %s)",
            symbol, days_back, len(filtered), len(docs), cutoff,
        )
        return filtered
    except Exception as exc:
        logger.warning("[RAG] get_articles_for_commodity failed: %s", exc)
        return []


# ── Maintenance ───────────────────────────────────────────────────────────────

def cleanup_old_news(days: int = 30) -> int:
    """
    Remove news articles older than `days` days.
    ChromaDB does not support deletion by metadata filter in all versions,
    so we fetch IDs with old dates and delete them explicitly.
    Returns number of documents removed.
    """
    col = get_collection(config.NEWS_COLLECTION)
    if col.count() == 0:
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    try:
        all_items = col.get(include=["metadatas"])
        old_ids   = [
            id_
            for id_, meta in zip(all_items["ids"], all_items["metadatas"])
            if meta.get("published_at", "9999") < cutoff
        ]
        if old_ids:
            col.delete(ids=old_ids)
            logger.info("[RAG] Removed %d old news articles (>%d days)", len(old_ids), days)
        return len(old_ids)
    except Exception as exc:
        logger.warning("[RAG] cleanup_old_news failed: %s", exc)
        return 0


def collection_counts() -> dict[str, int]:
    """Return document counts for both collections — used by /health endpoint."""
    try:
        news = get_collection(config.NEWS_COLLECTION).count()
    except Exception:
        news = 0
    try:
        hist = get_collection(config.HISTORY_COLLECTION).count()
    except Exception:
        hist = 0
    return {"commodity_news": news, "historical_events": hist}
