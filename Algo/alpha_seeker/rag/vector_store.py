"""
vector_store.py
---------------
Thin wrapper around a local ChromaDB collection.
Embedding model: all-MiniLM-L6-v2 (local, no cloud).
"""

import logging
import hashlib
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)

_COLLECTION = "crypto_news"
_EMBED_MODEL = "all-MiniLM-L6-v2"

_client:     chromadb.Client | None     = None
_collection: chromadb.Collection | None = None
_embedder:   SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBED_MODEL)
    return _embedder


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' loaded (%d docs)",
                    _COLLECTION, _collection.count())
    return _collection


def doc_count() -> int:
    try:
        return _get_collection().count()
    except Exception:
        return 0


def upsert_docs(docs: list[dict]) -> int:
    """
    Insert or update documents.  Each doc dict must have:
      id, text, source, symbol, published_at, url

    Returns the number of new/updated docs.
    """
    if not docs:
        return 0

    # Deduplicate by id within this batch (keep last occurrence)
    seen: dict[str, dict] = {}
    for d in docs:
        seen[d["id"]] = d
    docs = list(seen.values())

    col      = _get_collection()
    embedder = _get_embedder()
    texts    = [d["text"] for d in docs]
    vectors  = embedder.encode(texts, show_progress_bar=False).tolist()

    ids       = [d["id"]   for d in docs]
    def _to_ts(date_str: str) -> float:
        if not date_str:
            return 0.0
        try:
            return datetime.fromisoformat(date_str).timestamp()
        except Exception:
            pass
        try:
            return parsedate_to_datetime(date_str).timestamp()
        except Exception:
            return 0.0

    metadatas = [
        {
            "source":           d.get("source", ""),
            "symbol":           d.get("symbol", ""),
            "published_at":     d.get("published_at", ""),
            "published_at_ts":  _to_ts(d.get("published_at", "")),
            "url":              d.get("url", ""),
        }
        for d in docs
    ]

    col.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    logger.info("Upserted %d docs into ChromaDB", len(docs))
    return len(docs)


def query_docs(
    query:        str,
    symbol:       str,
    n_results:    int  = 5,
    max_age_hours: int = 72,
) -> list[dict]:
    """
    Embed query, filter by symbol and recency, return top-n matches.
    Returns list of {text, source, url, published_at, distance}.
    """
    col      = _get_collection()
    embedder = _get_embedder()
    vector   = embedder.encode([query], show_progress_bar=False).tolist()[0]

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).timestamp()

    try:
        res = col.query(
            query_embeddings=[vector],
            n_results=min(n_results, max(col.count(), 1)),
            where={
                "$and": [
                    {"symbol":        {"$eq": symbol.upper()}},
                    {"published_at_ts": {"$gte": cutoff_ts}},
                ]
            },
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.warning("ChromaDB query failed: %s", exc)
        return []

    results = []
    docs_list  = res.get("documents",  [[]])[0]
    metas_list = res.get("metadatas",  [[]])[0]
    dists_list = res.get("distances",  [[]])[0]

    for text, meta, dist in zip(docs_list, metas_list, dists_list):
        results.append({
            "text":         text,
            "source":       meta.get("source", ""),
            "url":          meta.get("url", ""),
            "published_at": meta.get("published_at", ""),
            "distance":     round(float(dist), 4),
        })

    return results


def make_doc_id(url: str) -> str:
    """Stable, deduplication-safe document ID from URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]
