# Alpha-Seeker

Alpha-Seeker is a production-grade multi-agent research system that analyses crypto assets using three independent specialist agents (Technical, Sentiment, Fundamental), a structured debate round, Self-RAG claim verification against a live news vector store, and long-term memory that tracks outcome accuracy over time — delivering a written, evidence-based trade thesis rather than a bare signal.

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │              Alpha-Seeker Graph              │
                         │                 (LangGraph)                  │
                         └─────────────────────────────────────────────┘
                                            │
                              ┌─────────────▼─────────────┐
                              │     check_outcomes_node    │  yfinance outcome tracking
                              └─────────────┬─────────────┘
                              ┌─────────────▼─────────────┐
                              │        memory_node         │  SQLite long-term memory
                              └──────┬──────┬──────┬──────┘
                                     │      │      │  (parallel)
                           ┌─────────▼──┐ ┌─▼────────┐ ┌──▼──────────┐
                           │ technical  │ │ sentiment │ │ fundamental │
                           │   agent    │ │   agent   │ │    agent    │
                           │TA+CS+Chart │ │News+BERT  │ │Binance Fut. │
                           └─────────┬──┘ └─┬────────┘ └──┬──────────┘
                                     └──────┼──────────────┘
                              ┌─────────────▼─────────────┐
                              │    rag_verification_node   │  ChromaDB + Self-RAG
                              │    + Corrective RAG        │  (claim-level)
                              └─────────────┬─────────────┘
                              ┌─────────────▼─────────────┐
                              │        debate_node         │  structured agent debate
                              └─────────────┬─────────────┘
                              ┌─────────────▼─────────────┐
                              │       synthesis_node       │  LLM final recommendation
                              └─────────────┬─────────────┘
                              ┌─────────────▼─────────────┐
                              │  FastAPI  /analyse  /report│
                              └─────────────────────────────┘
```

**Data sources (all free)**

| Source | Used for |
|---|---|
| Binance Spot API | OHLCV data (TechnicalAnalysis, backtest) |
| Binance Futures API | Funding rates, OI, long/short ratios |
| CryptoPanic | News headlines (RAG + NewsAnalysis) |
| NewsAPI | Broader news (RAG + NewsAnalysis) |
| CoinDesk RSS | RAG news ingestion |
| CryptoCompare | On-chain social proxy |
| Reddit / CryptoBERT | Sentiment scoring |
| yfinance | Outcome price tracking |

---

## Setup

```bash
git clone <repo>
cd alpha-seeker
cp .env.example .env          # add your ANTHROPIC_API_KEY
docker build -t alpha-seeker .
docker run -p 8000:8000 -v $(pwd)/data:/app/data alpha-seeker
```

Then open `http://localhost:8000/report/BTC` for the portfolio demo page.

---

## Example output

```
┌─────────────────────────────────────────────────────┐
│  ALPHA-SEEKER REPORT — BTC — 2026-05-05 14:30 UTC  │
├─────────────────────────────────────────────────────┤
│  TECHNICAL     BULLISH  confidence: 0.72            │
│  RSI divergence on 4h confirmed, EMA crossover      │
│  holding. Chart pattern suggests continuation       │
│  above 95k resistance.                              │
│  RAG: SUPPORTED — 2 articles confirm momentum.      │
├─────────────────────────────────────────────────────┤
│  SENTIMENT     NEUTRAL  confidence: 0.45            │
│  Reddit mixed, news volume low, no strong           │
│  directional conviction from NLP models.            │
│  RAG: INSUFFICIENT_EVIDENCE                         │
├─────────────────────────────────────────────────────┤
│  FUNDAMENTAL   BEARISH  confidence: 0.61            │
│  Funding rate 0.08%/8h — longs overleveraged.       │
│  OI up 12% in 24h. Liquidation cascade risk         │
│  elevated.                                          │
│  RAG: SUPPORTED — CoinDesk confirms funding spike.  │
├─────────────────────────────────────────────────────┤
│  DEBATE ROUND 1                                     │
│  Fundamental → Technical: "Elevated funding rates   │
│  historically precede corrections even in           │
│  uptrends. Your pattern may fail at this            │
│  leverage level."                                   │
│  Technical responds: Confidence revised 0.72→0.58.  │
├─────────────────────────────────────────────────────┤
│  MEMORY: Last 3× BTC bullish + high funding:        │
│  2/3 resulted in correction before continuation.    │
├─────────────────────────────────────────────────────┤
│  FINAL: STAY OUT  conviction: 0.52                  │
│  Technical signal real but funding risk material.   │
│  Revisit if funding normalises below 0.03%.         │
└─────────────────────────────────────────────────────┘
```

---

## API

```
POST /analyse/{symbol}?force_refresh=false   # JSON analysis
GET  /report/{symbol}                        # HTML portfolio page
GET  /memory/{symbol}                        # agent accuracy stats
GET  /health                                 # service health
```

---

## Skills demonstrated

*(Written for a hiring manager)*

**LangGraph multi-agent orchestration** — Three specialist agents run in parallel via LangGraph's fan-out pattern, with a structured fan-in to a shared RAG verification layer. The graph uses SQLite checkpointing for full resumability.

**Self-RAG and Corrective RAG** — Each agent's top-2 key signals are individually verified against a live ChromaDB news vector store (all-MiniLM-L6-v2 embeddings). If a claim is CONTRADICTED, Corrective RAG triggers an LLM revision of the thesis and reduces confidence by 15%, propagating the correction downstream.

**Long-term agent memory with outcome tracking** — Every analysis is persisted to SQLite. A background tracker checks yfinance 24h and 72h after each call, updates the outcome, and maintains per-agent accuracy tables. Memory context is injected into the synthesis prompt, closing the feedback loop.

**Production FastAPI + Docker deployment** — Full async FastAPI application with Jinja2 HTML reporting, a 15-minute result cache, and a Docker image that pre-downloads the embedding model for offline operation.

**Real quantitative signals from live trading background** — The technical layer calls a working 8-indicator trading algorithm (RSI, EMA alignment, Bollinger %B, MACD, candlestick patterns, chart patterns) that was backtested with proper walk-forward cross-validation and look-ahead-free holdout evaluation.

**Derivatives market structure analysis** — The fundamental agent queries Binance futures endpoints (no auth required) for funding rates, open interest history, long/short ratios, and mark/index basis — the same data professional crypto traders use to assess leverage risk.

---

## Cost

~$0.012 per full analysis run using claude-haiku-4-5-20251001. See [costs.md](costs.md) for full breakdown.
