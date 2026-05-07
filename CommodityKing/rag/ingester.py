"""
News ingester — formats and loads raw article dicts into the vector store.

Called at the start of each analysis run (when RAG_REFRESH_ON_ANALYSIS=True).
Handles deduplication, article ID generation, and retention cleanup.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from rag.vector_store import add_news_articles, cleanup_old_news
import config

logger = logging.getLogger(__name__)


def _make_article_id(url: str) -> str:
    """Stable ID from URL — truncated SHA256."""
    return hashlib.sha256(url.encode()).hexdigest()[:32]


def _normalise_article(raw: dict, commodity: str) -> dict | None:
    """
    Convert a raw article dict (from NewsAPI or RSS) into the standard shape
    expected by vector_store.add_news_articles.

    Returns None if the article lacks a title (unparseable).
    """
    title = (raw.get("title") or "").strip()
    if not title or title.lower() == "[removed]":
        return None

    url          = raw.get("url") or raw.get("link") or ""
    description  = (raw.get("description") or raw.get("summary") or "").strip()
    published_at = raw.get("publishedAt") or raw.get("published") or raw.get("updated") or ""

    # Normalise published_at to ISO format string
    if published_at and not isinstance(published_at, str):
        try:
            published_at = published_at.isoformat()
        except Exception:
            published_at = str(published_at)

    return {
        "id":           _make_article_id(url) if url else _make_article_id(title),
        "title":        title,
        "description":  description[:500],      # cap length
        "source":       _extract_source(raw),
        "published_at": published_at[:32],      # cap to ISO date portion
        "url":          url,
        "commodity":    commodity,
    }


def _extract_source(raw: dict) -> str:
    src = raw.get("source")
    if isinstance(src, dict):
        return src.get("name") or src.get("id") or ""
    if isinstance(src, str):
        return src
    return raw.get("feed_name") or ""


def ingest_articles(
    articles: list[dict],
    commodity: str,
    source_label: str = "",
) -> int:
    """
    Normalise and load a batch of raw articles for a commodity.
    Returns count of net-new documents added.
    """
    normalised = []
    for raw in articles:
        article = _normalise_article(raw, commodity)
        if article:
            normalised.append(article)

    if not normalised:
        logger.debug("[Ingester] No valid articles to ingest for %s (%s)", commodity, source_label)
        return 0

    added = add_news_articles(normalised, commodity)
    if added:
        logger.info(
            "[Ingester] %s | %s | +%d articles (%d total normalised)",
            commodity, source_label or "unknown", added, len(normalised)
        )
    return added


def refresh_for_commodity(
    commodity: str,
    articles_newsapi: list[dict],
    articles_rss:     list[dict],
) -> dict[str, int]:
    """
    Full refresh cycle for one commodity before analysis:
      1. Ingest NewsAPI articles
      2. Ingest RSS articles
      3. Clean up articles older than retention window

    Returns ingestion summary.
    """
    n_newsapi = ingest_articles(articles_newsapi, commodity, source_label="NewsAPI")
    n_rss     = ingest_articles(articles_rss,     commodity, source_label="RSS")
    n_removed = cleanup_old_news(days=config.NEWS_RETENTION_DAYS)

    return {
        "newsapi_added": n_newsapi,
        "rss_added":     n_rss,
        "old_removed":   n_removed,
    }


def get_article_texts(articles: list[dict]) -> list[str]:
    """
    Extract plain-text strings from raw article dicts for use in
    bayesian_surprise() — returns title + description concatenated.
    """
    texts = []
    for a in articles:
        title = (a.get("title") or "").strip()
        desc  = (a.get("description") or a.get("summary") or "").strip()
        text  = f"{title}. {desc}".strip(". ")
        if text:
            texts.append(text)
    return texts
