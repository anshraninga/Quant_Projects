"""
SentimentAnalysis.py
--------------------
CryptoBERT sentiment analysis using two reliable free sources:

  Primary  — Reddit public JSON API (no auth, no API key, no encoding issues)
             Subreddits: r/Bitcoin, r/CryptoCurrency, r/ethtrader, + symbol-specific
             Returns hot posts with title + selftext, scored by CryptoBERT.
             Upvote ratio used as a direct crowd-sentiment signal blended in.

  Fallback — StockTwits public stream (no API key required)
             Uses urllib (NOT requests) to avoid Windows cp1252 encoding crash.
             requests internally encodes response body with system codepage before
             we can read .content — urllib returns raw bytes, immune to this.

No paid APIs. No rate-limit collisions. No encoding issues.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
import urllib.parse
from dotenv import load_dotenv
from transformers import pipeline

from SignalNormalizer import normalize_sentiment, SignalVector

load_dotenv()


# ─────────────────────────────────────────────────────────
# WINDOWS UTF-8 FIX
# ─────────────────────────────────────────────────────────
if sys.platform == "win32" or (hasattr(sys.stdout, "encoding") and
                                sys.stdout.encoding and
                                sys.stdout.encoding.lower() not in ("utf-8", "utf8")):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                      errors="replace", line_buffering=True)


def _safe_print(*args, **kwargs):
    """Writes UTF-8 directly to stdout.buffer — immune to cp1252 console."""
    text = " ".join(str(a) for a in args) + kwargs.get("end", "\n")
    try:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
    except Exception:
        try:
            print(text.encode("ascii", errors="replace").decode("ascii"), end="")
        except Exception:
            pass


def _urllib_get(url: str, timeout: int = 10) -> bytes:
    """
    Fetches a URL using urllib and returns raw bytes.
    Never touches the response as text — immune to cp1252 encoding issues.
    """
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "CryptoAlgo/1.0 (sentiment analysis bot)")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ─────────────────────────────────────────────────────────
# CRYPTOBERT
# ─────────────────────────────────────────────────────────

print("⏳ Loading CryptoBERT model (first run may take ~30s to download)...")
cryptobert = pipeline(
    task="text-classification",
    model="ElKulako/cryptobert",
    tokenizer="ElKulako/cryptobert",
    top_k=None, truncation=True, max_length=512,
)
print("✅ CryptoBERT ready.")


def cryptobert_score(text: str) -> dict:
    if not text or not text.strip():
        return {"bullish_score": 0.0, "bearish_score": 0.0,
                "neutral_score": 1.0, "label": "neutral"}
    try:
        results = cryptobert(text[:512])[0]
        scores  = {r["label"].lower(): r["score"] for r in results}
        label   = max(scores, key=scores.get)
        return {
            "bullish_score": round(scores.get("bullish", 0.0), 6),
            "bearish_score": round(scores.get("bearish", 0.0), 6),
            "neutral_score": round(scores.get("neutral", 0.0), 6),
            "label": label,
        }
    except Exception as e:
        _safe_print(f"  [CryptoBERT error] {e}")
        return {"bullish_score": 0.0, "bearish_score": 0.0,
                "neutral_score": 1.0, "label": "neutral"}


# ─────────────────────────────────────────────────────────
# REDDIT
# ─────────────────────────────────────────────────────────

# Subreddits checked per symbol.
# General crypto subreddits always included; symbol-specific ones added where they exist.
SYMBOL_SUBREDDITS = {
    # Major coins
    "BTC":   ["Bitcoin", "CryptoCurrency"],
    "ETH":   ["ethereum", "CryptoCurrency"],
    "SOL":   ["solana", "CryptoCurrency"],
    "ADA":   ["cardano", "CryptoCurrency"],
    "XRP":   ["XRP", "CryptoCurrency"],
    "DOGE":  ["dogecoin", "CryptoCurrency"],
    "AVAX":  ["Avax", "CryptoCurrency"],
    "LINK":  ["Chainlink", "CryptoCurrency"],
    "DOT":   ["Polkadot", "CryptoCurrency"],
    "MATIC": ["0xPolygon", "CryptoCurrency"],
    "POL":   ["0xPolygon", "CryptoCurrency"],
    "NEAR":  ["nearprotocol", "CryptoCurrency"],
    "ICP":   ["dfinity", "CryptoCurrency"],
    "LTC":   ["litecoin", "CryptoCurrency"],
    "BNB":   ["binance", "CryptoCurrency"],
    "UNI":   ["Uniswap", "CryptoCurrency"],
    "ATOM":  ["cosmosnetwork", "CryptoCurrency"],
    "FTM":   ["FantomFoundation", "CryptoCurrency"],
    # From portfolio list
    "OXT":   ["orchid", "CryptoCurrency"],
    "FET":   ["fetchai", "CryptoCurrency"],
    "ZEN":   ["Horizen", "CryptoCurrency"],
    "CSPR":  ["CasperNetwork", "CryptoCurrency"],
    "ACX":   ["AcrossProtocol", "CryptoCurrency"],
    "CELO":  ["celo", "CryptoCurrency"],
    "CFX":   ["Conflux_Network", "CryptoCurrency"],
    "VET":   ["Vechain", "CryptoCurrency"],
    "MINA":  ["MinaProtocol", "CryptoCurrency"],
    "AGI":   ["SingularityNET", "CryptoCurrency"],
    # Smaller caps — fall back to CryptoCurrency only
    "SAFE":  ["CryptoCurrency"],
    "CC":    ["CryptoCurrency"],
    "AZTEC": ["CryptoCurrency"],
    "BABB":  ["CryptoCurrency"],
    "BREV":  ["CryptoCurrency"],
    "TNSR":  ["CryptoCurrency"],
    "LINEA": ["CryptoCurrency"],
    "ZKC":   ["CryptoCurrency"],
    "PORTAL":["CryptoCurrency"],
}
# Unknown symbols fall back to CryptoCurrency subreddit
DEFAULT_SUBREDDITS = ["CryptoCurrency"]


def fetch_reddit_sentiment(symbol: str, limit: int = 25) -> dict:
    """
    Fetches hot posts from crypto subreddits using Reddit's public JSON API.
    No authentication required. Returns raw bytes via urllib — no encoding issues.

    Each post contributes two blended signals:
      1. CryptoBERT score on title + selftext
      2. upvote_ratio converted to bull/bear (upvote_ratio=1.0 → fully bullish,
         upvote_ratio=0.5 → neutral, upvote_ratio=0.0 → fully bearish)

    Posts are weighted by score (upvotes) so high-engagement posts matter more.
    """
    subreddits = SYMBOL_SUBREDDITS.get(symbol.upper(), DEFAULT_SUBREDDITS)
    raw_scores   = []
    sample_posts = []
    posts_seen   = 0

    for subreddit in subreddits:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
        try:
            raw_bytes = _urllib_get(url, timeout=10)
            data      = json.loads(raw_bytes.decode("utf-8", errors="replace"))
            posts     = data.get("data", {}).get("children", [])

            for post_wrapper in posts:
                post = post_wrapper.get("data", {})

                # Skip pinned/stickied mod posts
                if post.get("stickied") or post.get("pinned"):
                    continue

                title     = post.get("title", "") or ""
                selftext  = post.get("selftext", "") or ""
                score     = int(post.get("score", 1) or 1)
                upvote_r  = float(post.get("upvote_ratio", 0.5) or 0.5)

                # Text to score: title + body (truncated)
                text = f"{title}. {selftext[:200]}".strip(". ").strip()

                # Signal 1: CryptoBERT on text
                cb_score = cryptobert_score(text)

                # Signal 2: upvote ratio → bull/bear
                # upvote_ratio 0.5 = equal up/down = neutral
                # upvote_ratio 1.0 = all upvotes = bullish
                # upvote_ratio 0.0 = all downvotes = bearish
                ur_bull = max(0.01, round((upvote_r - 0.5) * 2, 6)) if upvote_r > 0.5 else 0.05
                ur_bear = max(0.01, round((0.5 - upvote_r) * 2, 6)) if upvote_r < 0.5 else 0.05
                ur_neut = max(0.01, round(1.0 - ur_bull - ur_bear, 6))
                ur_total = ur_bull + ur_bear + ur_neut + 1e-10
                ur_score = {
                    "bullish_score": round(ur_bull / ur_total, 6),
                    "bearish_score": round(ur_bear / ur_total, 6),
                    "neutral_score": round(ur_neut / ur_total, 6),
                    "label": "bullish" if ur_bull > ur_bear + 0.1 else
                             "bearish" if ur_bear > ur_bull + 0.1 else "neutral",
                }

                # Blend: 65% CryptoBERT text + 35% upvote ratio
                blended = {
                    "bullish_score": round(cb_score["bullish_score"] * 0.65 + ur_score["bullish_score"] * 0.35, 6),
                    "bearish_score": round(cb_score["bearish_score"] * 0.65 + ur_score["bearish_score"] * 0.35, 6),
                    "neutral_score": round(cb_score["neutral_score"] * 0.65 + ur_score["neutral_score"] * 0.35, 6),
                    "label": cb_score["label"],
                }

                # Weight by post score (upvotes), capped at 10
                weight = min(max(score // 50, 1), 10)
                raw_scores.extend([blended] * weight)
                posts_seen += 1

                if len(sample_posts) < 5:
                    display = title.encode("ascii", errors="replace").decode("ascii")
                    sample_posts.append({
                        "text":         display[:90],
                        "label":        blended["label"],
                        "upvote_ratio": upvote_r,
                        "score":        score,
                        "subreddit":    subreddit,
                    })

            # Small delay between subreddit calls to respect rate limit
            if len(subreddits) > 1:
                time.sleep(0.5)

        except urllib.error.HTTPError as e:
            _safe_print(f"  [Reddit] r/{subreddit} HTTP {e.code} — skipping")
        except Exception as e:
            _safe_print(f"  [Reddit] r/{subreddit} error: {e}")

    if not raw_scores:
        return _empty_result("Reddit")

    n        = len(raw_scores)
    avg_bull = sum(s["bullish_score"] for s in raw_scores) / n
    avg_bear = sum(s["bearish_score"] for s in raw_scores) / n
    avg_neut = sum(s["neutral_score"] for s in raw_scores) / n
    total    = avg_bull + avg_bear + avg_neut + 1e-10
    bull_pct = round(avg_bull / total * 100, 1)
    bear_pct = round(avg_bear / total * 100, 1)
    neut_pct = round(avg_neut / total * 100, 1)
    overall  = "bullish" if bull_pct > bear_pct + 10 else \
               "bearish" if bear_pct > bull_pct + 10 else "neutral"

    return {
        "source":          "Reddit",
        "posts_analyzed":  n,
        "posts_raw_count": posts_seen,
        "subreddits":      subreddits,
        "bullish_pct":     bull_pct,
        "bearish_pct":     bear_pct,
        "neutral_pct":     neut_pct,
        "overall":         overall,
        "sample_posts":    sample_posts,
        "raw_scores":      raw_scores,
    }


# ─────────────────────────────────────────────────────────
# STOCKTWITS  (fallback — urllib to avoid cp1252 crash)
# ─────────────────────────────────────────────────────────

def fetch_stocktwits_sentiment(symbol: str, limit: int = 30) -> dict:
    """
    Fallback when Reddit fails. Uses urllib — never requests — to avoid
    the Windows cp1252 UnicodeEncodeError that requests triggers internally.
    """
    url   = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    token = os.getenv("STOCKTWITS_ACCESS_TOKEN", "").strip()

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    if token:
        req.add_header("Authorization", f"OAuth {token}")

    raw_scores   = []
    sample_posts = []
    user_labels  = {"bullish": 0, "bearish": 0, "none": 0}

    try:
        raw_bytes = _urllib_get(url, timeout=10)
        data      = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        messages  = data.get("messages", [])[:limit]

        if not messages:
            return _empty_result("StockTwits")

        for msg in messages:
            raw_body   = msg.get("body", "") or ""
            score      = cryptobert_score(raw_body)
            user_tag   = msg.get("entities", {}).get("sentiment", {})
            user_label = (user_tag.get("basic", "") or "none").lower()
            if user_label in user_labels:
                user_labels[user_label] += 1
            likes  = msg.get("likes", {}).get("total", 0) or 0
            weight = min(max(likes // 5, 1), 5)
            raw_scores.extend([score] * weight)
            if len(sample_posts) < 5:
                display = raw_body.encode("ascii", errors="replace").decode("ascii")
                sample_posts.append({"text": display[:100], "label": score["label"],
                                     "user_tag": user_label, "likes": likes})

    except urllib.error.HTTPError as e:
        if e.code in (404, 429):
            return _empty_result("StockTwits")
        _safe_print(f"  [StockTwits] HTTP {e.code}")
        return _empty_result("StockTwits")
    except Exception as e:
        _safe_print(f"  [StockTwits ERROR] {e}")
        return _empty_result("StockTwits")

    n = len(raw_scores)
    if n == 0:
        return _empty_result("StockTwits")

    avg_bull = sum(s["bullish_score"] for s in raw_scores) / n
    avg_bear = sum(s["bearish_score"] for s in raw_scores) / n
    avg_neut = sum(s["neutral_score"] for s in raw_scores) / n
    total    = avg_bull + avg_bear + avg_neut + 1e-10
    bull_pct = round(avg_bull / total * 100, 1)
    bear_pct = round(avg_bear / total * 100, 1)
    neut_pct = round(avg_neut / total * 100, 1)
    overall  = "bullish" if bull_pct > bear_pct + 10 else \
               "bearish" if bear_pct > bull_pct + 10 else "neutral"

    result = {
        "source": "StockTwits", "posts_analyzed": n,
        "bullish_pct": bull_pct, "bearish_pct": bear_pct, "neutral_pct": neut_pct,
        "overall": overall, "sample_posts": sample_posts,
        "raw_scores": raw_scores, "user_tagged": user_labels,
    }
    total_tagged = user_labels["bullish"] + user_labels["bearish"]
    if total_tagged > 0:
        result["user_bullish_pct"] = round(user_labels["bullish"] / total_tagged * 100, 1)
        result["user_bearish_pct"] = round(user_labels["bearish"] / total_tagged * 100, 1)
    return result


def _empty_result(source: str) -> dict:
    return {
        "source": source, "posts_analyzed": 0,
        "bullish_pct": 0.0, "bearish_pct": 0.0, "neutral_pct": 0.0,
        "overall": "neutral", "sample_posts": [], "raw_scores": [],
        "user_tagged": {"bullish": 0, "bearish": 0, "none": 0},
    }


# ─────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────

def run_sentiment_analysis(symbol: str) -> dict:
    _safe_print(f"\n💬 Sentiment analysis for: {symbol.upper()}")

    # ── Primary: Reddit ──────────────────────────────────
    reddit_result = fetch_reddit_sentiment(symbol)

    if reddit_result["posts_analyzed"] > 0:
        _safe_print(
            f"  Reddit -> Bull {reddit_result['bullish_pct']}% | "
            f"Bear {reddit_result['bearish_pct']}% | "
            f"Posts: {reddit_result['posts_raw_count']} from "
            f"r/{', r/'.join(reddit_result['subreddits'])}"
        )
        for i, p in enumerate(reddit_result.get("sample_posts", [])[:3], 1):
            icon = "↑" if p["label"] == "bullish" else "↓" if p["label"] == "bearish" else "-"
            _safe_print(f"    {i}. [{icon}] {p['text'][:75]}")

        raw_scores    = reddit_result["raw_scores"]
        source_result = reddit_result
        user_bull_pct = None
        user_bear_pct = None

    else:
        # ── Fallback: StockTwits ─────────────────────────
        _safe_print("  Reddit unavailable — trying StockTwits fallback...")
        st_result = fetch_stocktwits_sentiment(f"{symbol.upper()}.X")
        if st_result["posts_analyzed"] == 0:
            st_result = fetch_stocktwits_sentiment(symbol.upper())

        _safe_print(
            f"  StockTwits -> Bull {st_result['bullish_pct']}% | "
            f"Bear {st_result['bearish_pct']}% | Posts: {st_result['posts_analyzed']}"
        )
        raw_scores    = st_result["raw_scores"]
        source_result = st_result
        user_bull_pct = st_result.get("user_bullish_pct")
        user_bear_pct = st_result.get("user_bearish_pct")

    # ── Normalize ────────────────────────────────────────
    signal_vec = normalize_sentiment(
        raw_scores=raw_scores, base_weight=0.125,
        user_bull_pct=user_bull_pct, user_bear_pct=user_bear_pct,
        n_posts=len(raw_scores),
    )

    _safe_print(
        f"  After calibration -> "
        f"Bull {signal_vec.bullish_pct()}% | Bear {signal_vec.bearish_pct()}% | "
        f"Neutral {round(signal_vec.neutral * 100, 1)}% | "
        f"Confidence {signal_vec.confidence:.2f} | EffW {signal_vec.effective_weight:.4f}"
    )

    return {
        "symbol":               symbol.upper(),
        "signal_vector":        signal_vec,
        "combined_bullish_pct": signal_vec.bullish_pct(),
        "combined_bearish_pct": signal_vec.bearish_pct(),
        "combined_neutral_pct": round(signal_vec.neutral * 100, 1),
        "combined_overall":     source_result["overall"],
        "reddit":               reddit_result,
        "lunarcrush":           _empty_result("LunarCrush"),
        "stocktwits":           (source_result if reddit_result["posts_analyzed"] == 0
                                 else _empty_result("StockTwits")),
        "twitter":              _empty_result("Twitter"),
    }


if __name__ == "__main__":
    sym = input("Enter symbol (e.g. BTC, ETH, SOL): ").strip()
    result = run_sentiment_analysis(sym)
    print(f"\n  Final: {result['combined_overall'].upper()} — "
          f"Bull {result['combined_bullish_pct']}% | "
          f"Bear {result['combined_bearish_pct']}% | "
          f"Neut {result['combined_neutral_pct']}%")