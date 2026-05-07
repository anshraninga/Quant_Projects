"""
CryptoPanicCache.py
-------------------
Shared CryptoPanic API layer used by both NewsAnalysis.py and SentimentAnalysis.py.

Problem it solves:
  Both modules run in parallel via ThreadPoolExecutor in main.py.
  Both call CryptoPanic with the same API key. The free tier allows
  only ~5 req/s and a low hourly cap. Without coordination, one module
  hits 429 and returns empty, causing permanent 50/50 neutral sentiment.

Solution:
  This module maintains a process-level cache (thread-safe via Lock).
  The first caller fetches from CryptoPanic and stores the result.
  The second caller (arriving within CACHE_TTL seconds) gets the cached
  result instantly — no second API call, no rate limit collision.

  Cache is keyed by (symbol, kind) so news and sentiment fetches for
  the same symbol share data when their kind overlaps, but different
  kinds are fetched independently.

Usage:
  from CryptoPanicCache import fetch_cryptopanic_cached
  posts = fetch_cryptopanic_cached(symbol="BTC", kind="news", limit=10)
  posts = fetch_cryptopanic_cached(symbol="BTC", kind="all",  limit=50)
"""

import os
import json
import time
import threading
import requests
from dotenv import load_dotenv

load_dotenv()

CACHE_TTL = 60  # seconds — reuse result for 60s, enough for one full analysis run

_cache      = {}          # { cache_key: (timestamp, data) }
_in_flight  = set()       # cache keys currently being fetched
_cache_lock = threading.Lock()


def fetch_cryptopanic_cached(symbol: str, kind: str = "all", limit: int = 50) -> list:
    """
    Fetches posts from CryptoPanic with caching and retry-backoff.

    Returns list of raw post dicts from the API, or [] on failure.
    Thread-safe — safe to call from multiple parallel threads.

    Parameters:
      symbol  : coin symbol e.g. "BTC", "ETH"
      kind    : "news" | "media" | "all"
      limit   : max posts to return
    """
    api_key = os.getenv("CRYPTOPANIC_API_KEY", "").strip()
    if not api_key:
        return []

    cache_key = f"{symbol.upper()}:{kind}"

    # Check cache and claim fetch rights atomically.
    # _in_flight tracks keys currently being fetched so a second thread
    # arriving during the fetch waits instead of firing a duplicate call.
    with _cache_lock:
        if cache_key in _cache:
            ts, data = _cache[cache_key]
            if time.time() - ts < CACHE_TTL:
                return data[:limit]
        # Mark as in-flight so other threads wait
        already_fetching = cache_key in _in_flight
        _in_flight.add(cache_key)

    # If another thread is already fetching this key, wait for it
    if already_fetching:
        deadline = time.time() + 15  # max 15s wait
        while time.time() < deadline:
            time.sleep(0.2)
            with _cache_lock:
                if cache_key in _cache:
                    ts, data = _cache[cache_key]
                    if time.time() - ts < CACHE_TTL:
                        return data[:limit]
        # Timed out — return whatever is cached (may be empty)
        with _cache_lock:
            if cache_key in _cache:
                return _cache[cache_key][1][:limit]
        return []

    # Not cached and not in-flight — fetch from API
    url    = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": api_key,
        "currencies": symbol.upper(),
        "kind":       kind,
    }

    posts = []
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.encoding = "utf-8"

            if r.status_code == 429:
                wait = 2 ** attempt   # 1, 2, 4, 8 seconds
                print(f"  [CryptoPanic] Rate limited — retrying in {wait}s (attempt {attempt+1}/4)...")
                time.sleep(wait)
                continue

            if r.status_code in (401, 403):
                print("  [CryptoPanic] Invalid or unverified API key")
                break

            if r.status_code == 404:
                break

            r.raise_for_status()
            data  = json.loads(r.content.decode("utf-8", errors="replace"))
            posts = data.get("results", [])

            # If empty with currency filter, retry broader
            if not posts:
                broad_params = {"auth_token": api_key, "kind": kind}
                r2 = requests.get(url, params=broad_params, timeout=15)
                r2.encoding = "utf-8"
                if r2.status_code == 200:
                    data2 = json.loads(r2.content.decode("utf-8", errors="replace"))
                    all_posts = data2.get("results", [])
                    sym = symbol.upper()
                    posts = [
                        p for p in all_posts
                        if sym in (p.get("title", "") or "").upper()
                        or any(c.get("code", "") == sym for c in (p.get("currencies") or []))
                    ]

            break  # success (even if posts is empty)

        except requests.exceptions.ConnectionError:
            print("  [CryptoPanic] Connection error")
            break
        except Exception as e:
            print(f"  [CryptoPanic ERROR] {e}")
            break

    # Store result and release in-flight lock so waiting threads can read
    with _cache_lock:
        _cache[cache_key] = (time.time(), posts)
        _in_flight.discard(cache_key)

    return posts[:limit]