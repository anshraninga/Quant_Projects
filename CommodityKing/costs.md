# CommodityKing — Cost Reference

## Per-Call Cost Breakdown

| Call type | Model | Cost |
|---|---|---|
| Geo agent, Weather agent, Fundamentals agent | Claude Haiku | ~$0.001 each |
| Debate rounds (if triggered) | Claude Haiku | ~$0.001 each |
| Synthesis (final recommendation) | Claude Sonnet | ~$0.025 |

A standard single-commodity run makes 5 Haiku calls and 1 Sonnet call.

---

## Cost Per Analysis

| Scenario | Cost |
|---|---|
| Single commodity (e.g. `wheat`) | ~$0.035 |
| 3-commodity parallel run (e.g. `wheat, gold, copper`) | ~$0.105 |
| 10-commodity full portfolio run | ~$0.35 |

Parallel runs share one server process but each commodity runs its own independent graph — costs add linearly.

---

## Monthly Cost Projections

| Usage | Monthly cost |
|---|---|
| 5 analyses/day × 1 commodity | ~$5.25 |
| 20 analyses/day × 1 commodity | ~$21.00 |
| 5 analyses/day × 10 commodities | ~$52.50 |

---

## Data Source Costs

### Free (zero cost, no registration)
- **Open-Meteo** — weather data across 25 global producing regions
- **yfinance** — real-time and historical price data (Yahoo Finance)
- **RSS feeds** — Reuters commodities, FT markets, USDA releases, EIA releases
- **ChromaDB** — local vector store (runs in-process, no cloud)
- **sentence-transformers** — local embeddings via `all-MiniLM-L6-v2` (runs on CPU)

### Free with registration
- **NewsAPI** — free tier: 1,000 requests/day. Registration at [newsapi.org](https://newsapi.org)
- **USDA NASS QuickStats** — free, registration required at [quickstats.nass.usda.gov/api](https://quickstats.nass.usda.gov/api)
- **EIA Open Data API** — free, registration required at [eia.gov/opendata](https://www.eia.gov/opendata/)

### Pay-per-use
- **Anthropic API** — ~$0.035 per single-commodity analysis

---

## Context

A Bloomberg Terminal subscription costs approximately **$2,000/month** for commodity data access, providing price data, news, and some fundamental data. CommodityKing delivers structured fundamental research — three specialised agents, five quantitative models, weather analysis across 25 producing regions, USDA supply data, EIA inventory data, historical analog matching, and a synthesised final recommendation — at approximately **$0.035 per analysis** using entirely free data sources. A full 10-commodity portfolio analysed daily for a month costs less than a single Bloomberg login.
