# CommodityKing

CommodityKing is a multi-agent commodity fundamental analysis system that takes direct inspiration from Jim Simons and Renaissance Technologies' early insight: markets are slow to price in non-price signals. Where RenTech built signal libraries from weather records, shipping manifests, and satellite imagery, CommodityKing fuses open-source equivalents — Open-Meteo weather anomalies across 25 global producing regions, USDA NASS agricultural supply data, EIA energy inventory reports, and real-time geopolitical news — into structured, evidence-based research notes. Three specialised agents (Geopolitical, Weather, Fundamentals) run in parallel via LangGraph and produce directional theses that are verified against a curated RAG knowledge base, stress-tested in a structured inter-agent debate, and synthesised into a final recommendation. Five mathematical models underpin the signal layer: a Kalman filter for price noise reduction, a Hidden Markov Model for regime classification, a composite Supply Shock Index, Bayesian surprise scoring for news novelty detection, and Engle-Granger cointegration monitoring across commodity pairs.

---

## Architecture

```
DATA SOURCES
┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌───────────┐ ┌──────────┐
│  yfinance  │ │ Open-Meteo │ │   NewsAPI    │ │ USDA NASS │ │ EIA API  │
│  (prices)  │ │ (weather)  │ │  + RSS feeds │ │(ag supply)│ │(energy)  │
└─────┬──────┘ └──────┬─────┘ └──────┬───────┘ └─────┬─────┘ └────┬─────┘
      └───────────────┴──────────────┴───────────────┴────────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  LangGraph Pipeline │
                           └─────────┬───────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │       (parallel fan-out via Send API)               │
   ┌──────▼───────┐          ┌───────▼───────┐         ┌────────▼────────┐
   │  Geo Agent   │          │ Weather Agent │         │  Fund. Agent    │
   │              │          │               │         │                 │
   │ • NewsAPI    │          │ • Open-Meteo  │         │ • USDA NASS     │
   │ • RSS feeds  │          │   25 regions  │         │ • EIA inventory │
   │ • Kalman     │          │ • Temp/precip │         │ • HMM regime    │
   │ • Bayesian   │          │   z-scores    │         │ • SSI scoring   │
   │   surprise   │          │ • yfinance    │         │ • Cointegration │
   └──────┬───────┘          └───────┬───────┘         └────────┬────────┘
          └──────────────────────────┼──────────────────────────┘
                          (fan-in reducer)
                                     │
                           ┌─────────▼──────────┐
                           │   RAG + Analog      │
                           │                     │
                           │ • ChromaDB verify   │
                           │   agent claims vs   │
                           │   news corpus       │
                           │ • Historical event  │
                           │   similarity match  │
                           │   (30 curated       │
                           │    events, 10       │
                           │    commodities)     │
                           └─────────┬───────────┘
                                     │
                           ┌─────────▼──────────┐
                           │   Debate Node       │
                           │                     │
                           │ Divergent agents    │
                           │ challenge each      │
                           │ other's theses;     │
                           │ confidence updated  │
                           │ after each round    │
                           └─────────┬───────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  Synthesis Node     │
                           │                     │
                           │ • Claude Sonnet     │
                           │ • SSI composite     │
                           │ • Cointegration     │
                           │ • Final direction   │
                           │   + conviction      │
                           └─────────┬───────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  CommodityReport    │
                           │  FastAPI JSON       │
                           └────────────────────┘
```

---

## Supported Commodities

| Symbol | Name | Ticker | Category | Weather | USDA | EIA |
|---|---|---|---|---|---|---|
| `gold` | Gold | GC=F | Precious Metals | | | |
| `silver` | Silver | SI=F | Precious Metals | | | |
| `platinum` | Platinum | PL=F | Precious Metals | | | |
| `copper` | Copper | HG=F | Industrial Metals | ✓ | | |
| `wheat` | Wheat | ZW=F | Agricultural | ✓ | ✓ | |
| `corn` | Corn | ZC=F | Agricultural | ✓ | ✓ | |
| `soybeans` | Soybeans | ZS=F | Agricultural | ✓ | ✓ | |
| `sugar` | Sugar (Raw) | SB=F | Agricultural | ✓ | ✓ | |
| `oil_brent` | Brent Crude Oil | BZ=F | Energy | ✓ | | ✓ |
| `natural_gas` | Natural Gas (Henry Hub) | NG=F | Energy | ✓ | | ✓ |

---

## Five Mathematical Models

**Kalman Filter — Price Signal Extraction**

The Kalman filter runs on hourly price data to separate genuine trend from noise. It models price as a hidden state that evolves according to a linear process, updating its estimate at each new observation using an optimal weighting between the prior and the new measurement. The output is a smoothed signal with a sigma score that quantifies how many standard deviations the current price movement is from the estimated trend. For commodity analysis this matters because intraday price swings are dominated by order flow and liquidity dynamics rather than fundamentals; the Kalman filter strips that noise and surfaces whether the underlying trend is genuinely directional. The Geopolitical agent uses the Kalman signal to modulate confidence — a high-sigma signal confirms a news-driven move, while a low-sigma drift signals noise.

**Hidden Markov Model — Regime Classification**

The HMM fits a two-state Gaussian mixture to recent returns and classifies the current price regime as either low-volatility trending or high-volatility stressed. The model learns emission parameters (mean and variance per state) and transition probabilities from data, then uses the Viterbi algorithm to decode the most likely current state. Commodity markets are fundamentally regime-dependent: a price move in a low-volatility regime signals new information, whereas the same move in a high-volatility regime is likely noise from stop cascades or liquidity gaps. The Fundamentals agent uses the HMM regime label as a prior on how much weight to give supply/demand signals — supply shocks in stressed regimes warrant higher conviction than identical signals in calm regimes.

**Supply Shock Index — Composite Stress Scoring**

The SSI aggregates the three agent confidence signals — weather z-scores, geopolitical news surprise, and inventory deviation from seasonal norms — into a single continuous score with level labels (normal, elevated, high, extreme). Each component is weighted by commodity-specific coefficients that reflect how sensitive that commodity historically is to each driver (e.g., wheat weights weather at 45%, geopolitical at 35%; gold weights geopolitical at 50%, inventory at 50%). The SSI provides the Synthesis node with a single number that encodes cross-agent agreement about stress severity, enabling consistent final recommendations across commodities without requiring the synthesis LLM to re-derive the signal from raw inputs.

**Bayesian Surprise — News Novelty Detection**

The Bayesian surprise model computes KL divergence between the unigram word distribution in today's news corpus and a 30-day historical baseline stored in ChromaDB. A high KL divergence means today's articles contain terms that rarely appeared in the baseline — genuine new information. A low divergence means the news is repeating known themes already priced in. This matters because commodity markets are efficient about expected news: a scheduled OPEC meeting announcement has low surprise score even if the headline is dramatic. Unscheduled pipeline outages, export bans, or weather emergencies score high. The surprise score directly modulates the Geopolitical agent's confidence: agents should not be bullish with high conviction when news is expected. Crucially, the model improves with usage — the ChromaDB cache grows with each analysis run, making the baseline richer and the novelty detection more precise from run 2 onwards.

**Engle-Granger Cointegration — Cross-Commodity Spread Monitoring**

The cointegration model tests whether commodity pairs share a long-run equilibrium relationship using the Engle-Granger two-step procedure: fitting the cointegrating regression, then testing the residuals for stationarity via ADF. When cointegration is confirmed (p < 0.05), the spread z-score measures how far the current ratio has deviated from the historical mean, generating mean-reversion signals. Commodity pairs are structurally linked by shared inputs, substitution in end-use, or correlated production geography: wheat and corn compete in feed markets; gold and silver track the same store-of-value demand; oil and natural gas share energy demand. The Synthesis node uses cointegration signals to flag when a directional call on one commodity should be cross-checked against anomalous divergence from its cointegrated peer.

---

## Sample Output — Wheat (Live Run, May 2026)

```
POST /analyse
{"symbols": ["wheat"]}

PRICE CONTEXT
  current_price:  611.75 cents/bushel
  change_24h:     -1.96%
  change_7d:      -1.57%
  kalman_trend:   down  (0.48σ — below signal threshold)
  regime:         high_volatility  (HMM confidence: 1.00)

AGENTS
  Geopolitical  →  NEUTRAL        (conf: 0.50, surprise: 0.944 highly_unexpected, 33 articles)
  Weather       →  BEARISH_SUPPLY (conf: 0.50, max_zscore: +6.96σ)
  Fundamentals  →  BULLISH        (conf: 0.60, SSI: 4.125 extreme)

QUANTITATIVE SIGNALS
┌─────────────────────────────────────────────────────────────────────┐
│ Kalman Filter                                                       │
│   latest_signal: -0.479    is_signal: False    trend: down         │
│   signal_sigma:   0.48σ   (threshold 1.5σ — noise level)          │
│                                                                     │
│ HMM Regime                                                          │
│   regime: high_volatility  (state 0)   confidence: 1.00           │
│   is_high_volatility: True                                         │
│                                                                     │
│ Supply Shock Index                                                  │
│   SSI:    4.125  →  extreme                                        │
│   weather_component:    3.134  (Krasnodar +6.96σ precip anomaly)  │
│   geo_component:        0.991  (Sudan war, Black Sea risk)         │
│   inventory_component:  0.000  (stocks rebuilding, neutral)       │
│                                                                     │
│ Bayesian Surprise                                                   │
│   surprise_score:  0.944  →  highly_unexpected                    │
│   kl_divergence:   1.887                                           │
│   history_source:  chroma_cache  (91 articles in baseline)        │
│                                                                     │
│ Cointegration                                                       │
│   wheat/corn:  cointegrated  (p=0.016)                            │
│   spread_zscore: 0.314  →  within_normal_range                    │
└─────────────────────────────────────────────────────────────────────┘

HISTORICAL ANALOG
  Event:       Russia announces wheat export ban after severe drought
               destroys 30% of crop (2010-08-05)
  Similarity:  0.632
  Then:        +60% over 3 months
  Resolution:  Ban lifted in 2011 after new harvest proved strong

FINAL RECOMMENDATION
  Direction:   BULLISH
  Conviction:  0.42  (moderate — three agents partially divergent)
  Horizon:     SHORT

  Reasoning:
  Wheat at 611.75 cents/bushel is caught in a genuine tug-of-war
  between a structurally bullish supply shock signal (SSI 4.125) and
  persistent near-term price weakness (-1.96% 24h, -1.57% 7d).
  The dominant driver is Krasnodar's extraordinary +6.96σ precipitation
  anomaly during grain fill, which threatens Russian export quality and
  volume — Russia supplies ~20% of global wheat exports. However,
  insufficient evidence across all three intelligence pillars and a
  rebuilding global inventory backdrop prevent a high-conviction call.
  Conviction rises to STRONG if Russian crop damage data materialises
  into confirmed export disruptions over the next 2–4 weeks.

  Upside risks:
    • Krasnodar surveys confirm Fusarium contamination or test-weight
      failures → Russian export quota reductions (2010 analogue)
    • US winter wheat dryness persists through June → domestic
      production falls materially below 1.985B bushel forecast

  Downside risks:
    • Southern Hemisphere (Australia, Argentina) crop forecasts
      improve, capping any Russian-driven rally with substitute supply
    • Demand destruction from elevated prices softens EM import buying
      (Egypt, Pakistan, Bangladesh)

  Watch list:
    • Russian ag ministry crop condition reports + export quota
      announcements for 2025–26 marketing year (next 2–3 weeks)
    • USDA June WASDE: winter wheat conditions, harvested area
      revisions, global ending stocks 2025–26

METADATA
  llm_calls:          6
  estimated_cost_usd: $0.036
  errors:             []
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/<anshraninga>/CommodityKing.git
cd CommodityKing
```

**2. Create your environment file**
```bash
cp .env.example .env
```

**3. Add your API keys to `.env`**

All keys are free:

| Key | Source |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) — pay-per-use, ~$0.035/analysis |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org) — free tier, 1,000 req/day |
| `USDA_API_KEY` | [quickstats.nass.usda.gov/api](https://quickstats.nass.usda.gov/api) — free, registration required |
| `EIA_API_KEY` | [eia.gov/opendata](https://www.eia.gov/opendata/) — free, registration required |

Open-Meteo (weather), yfinance (prices), and RSS feeds require no key.

**4. Start the service**
```bash
docker compose up
```

API available at `http://localhost:8000`. Try:
```bash
curl -s -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["wheat", "gold"]}' | python -m json.tool
```

---

## Skills Demonstrated

This project was built as a portfolio demonstration of production-grade quant systems engineering. Specific techniques:

**LangGraph multi-agent orchestration with parallel execution and fan-out/fan-in pattern.** The three commodity agents (Geopolitical, Weather, Fundamentals) are dispatched simultaneously via LangGraph's `Send` API. Results are merged via `Annotated[list, operator.add]` reducers into the shared graph state, ensuring no agent blocks another. Multiple commodities requested in one API call run as entirely separate graph instances via `asyncio.gather`, giving true O(1) latency scaling.

**Five quantitative signal models: Kalman filter for price noise reduction, HMM regime classification, Supply Shock Index composite scoring, Bayesian surprise for news novelty detection, Engle-Granger cointegration monitoring.** All five are implemented from scratch using NumPy and SciPy — no quant library black boxes. Each model produces an interpretable output (sigma score, regime label, SSI level, surprise score, p-value) that feeds directly into agent prompts and final synthesis.

**Multi-source data fusion: Open-Meteo weather API across 25 global producing regions, EIA government energy data, USDA NASS QuickStats agricultural data, NewsAPI geopolitical news, RSS feeds.** Each commodity has a hand-curated list of weather regions (e.g. Krasnodar, Odessa, Kansas, Punjab) relevant to its supply chain. Temperature and precipitation anomalies are computed as z-scores against 30-year historical baselines and fed directly to the Weather agent. USDA NASS provides 4-year production and stocks series with year-on-year change and stocks-to-production ratios. EIA provides weekly crude and natural gas inventory data. All data sources degrade gracefully — if any fetch fails, the agent continues with a reduced dataset rather than erroring.

**Self-RAG with curated historical event knowledge base (30 events across 10 commodities).** A ChromaDB collection of hand-curated commodity market events (e.g. 2010 Russian wheat export ban, 2020 OPEC price war, 2022 LNG supply shock) is embedded using a local SentenceTransformer model. After each agent produces its thesis, a RAG node retrieves semantically similar historical events and returns a `SUPPORTED / CONTRADICTED / INSUFFICIENT_EVIDENCE` verdict that feeds into the synthesis prompt.

**Bayesian surprise scoring that improves with usage — ChromaDB news cache grows with each analysis run, making surprise detection more accurate over time.** The first run on a new commodity uses an empty baseline and returns a low surprise score. From run 2 onwards, the ChromaDB news collection contains the prior run's articles, giving the KL divergence a meaningful baseline. A system with 30 days of cached articles for a commodity produces highly calibrated surprise scores — genuinely new terms stand out clearly against the established vocabulary.

**Inspired by Renaissance Technologies early commodity work — non-price signals that markets are slow to price in.** The system deliberately avoids price-only technical analysis. Every signal layer — weather z-scores, news surprise, inventory deviations, cointegration spreads — measures something other than price. This mirrors Simons' insight that markets are more efficient at pricing known price patterns than they are at pricing structural, non-price information embedded in weather records, government reports, and geopolitical event flows.

**Production FastAPI deployment with Docker, async parallel analysis, graceful degradation.** The service exposes three endpoints: `POST /analyse` (parallel multi-commodity analysis), `GET /commodities` (metadata), `GET /health` (ChromaDB doc count, config warnings). All external API calls are wrapped in try/except with structured fallbacks. Cost tracking accumulates across all LLM-using nodes via `Annotated[float, operator.add]` state reducers and is returned in every response.

---

## Known Limitations

**USDA WASDE plain-text endpoint discontinued by USDA.** The monthly World Agricultural Supply and Demand Estimates report was previously available as a machine-readable plain-text file. USDA discontinued this endpoint. The system now uses USDA NASS QuickStats, which provides US production and stocks data with year-on-year comparisons. Global supply/demand balance tables (WASDE's primary analytical value) are not available via any free API.

**Bayesian surprise accuracy improves with usage — first run on a new commodity has thin history.** On the first analysis of a commodity, the ChromaDB news baseline is empty. The system detects this (`history_source: insufficient`) and applies a conservative surprise score. From run 2 onwards the baseline grows and surprise detection becomes increasingly calibrated. After approximately 30 days of daily runs the baseline is robust.

**Open-Meteo archive endpoint rate-limits under rapid repeated calls — one weather region per run may be skipped.** When multiple commodities are analysed in a single request, the weather agent for each commodity makes multiple Open-Meteo archive calls (one per region). Under concurrent load, occasional requests time out. The weather agent handles this gracefully: skipped regions are logged and the z-score computation continues with the available subset.

**Sugar cane production not available via NASS (US-focused database).** NASS covers US domestic agricultural production. Cane sugar is produced primarily in Brazil, India, and Thailand — none of which are in NASS scope. Sugar analysis therefore relies on weather signals (São Paulo, Uttar Pradesh, Queensland) and geopolitical news for supply fundamentals, without the NASS production/stocks series that wheat, corn, and soybeans benefit from.
