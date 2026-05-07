# Alpha-Seeker — Cost Estimate per Analysis Run

## LLM pricing (May 2026)

| Model | Input | Output |
|---|---|---|
| claude-haiku-4-5-20251001 | $0.80 / 1M tokens | $4.00 / 1M tokens |
| gpt-4o-mini | $0.15 / 1M tokens | $0.60 / 1M tokens |

## Calls per full analysis run

| Step | Prompt (tokens) | Response (tokens) | Calls |
|---|---|---|---|
| Technical thesis | ~600 | ~200 | 1 |
| Sentiment thesis | ~400 | ~200 | 1 |
| Fundamental thesis | ~500 | ~200 | 1 |
| RAG verify ×3 agents ×2 signals | ~350 | ~80 | 6 |
| Corrective RAG (if triggered) | ~400 | ~150 | 0–3 |
| Debate challenge | ~250 | ~100 | 1–2 |
| Debate response | ~300 | ~100 | 1–2 |
| Synthesis | ~800 | ~300 | 1 |
| **Total (typical)** | **~5,000** | **~1,900** | **~12** |

## Cost calculation (Haiku, typical run)

```
Input:  5,000 tokens × $0.80/1M  = $0.0040
Output: 1,900 tokens × $4.00/1M  = $0.0076
                              ─────────────
Total                             = $0.0116
```

**~$0.012 per full BTC analysis — well under the $0.05 target.**

Worst case (all RAG corrections triggered, 2 debate rounds): ~$0.025

## Cost per month (daily BTC + ETH analysis)

```
$0.012 × 2 symbols × 30 days = $0.72 / month
```

## With OpenAI gpt-4o-mini

```
Input:  5,000 × $0.15/1M  = $0.00075
Output: 1,900 × $0.60/1M  = $0.00114
                         ────────────
Total                     = $0.00189
```

~$0.002 per run — ~6× cheaper than Haiku, but lower reasoning quality.

## Notes

- The 15-minute result cache (`ANALYSIS_CACHE_MINUTES=15`) means repeated
  requests for the same symbol cost $0 after the first call.
- RAG embeddings (all-MiniLM-L6-v2) run locally — zero cost.
- ChromaDB, SQLite, yfinance — all free.
- Live token counts and running session cost are logged by `llm_client.py`
  and returned in every `/analyse` response under `metadata.estimated_cost_usd`.
