from dotenv import load_dotenv
import os

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic").lower()

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str    = os.getenv("OPENAI_API_KEY", "")

# Agent nodes use cheap/fast models; synthesis uses a larger model
ANTHROPIC_AGENT_MODEL:     str = "claude-haiku-4-5-20251001"
ANTHROPIC_SYNTHESIS_MODEL: str = "claude-sonnet-4-6"
OPENAI_AGENT_MODEL:        str = "gpt-4o-mini"
OPENAI_SYNTHESIS_MODEL:    str = "gpt-4o"

# ── News ──────────────────────────────────────────────────────────────────────
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")

# ── Fundamental data APIs ─────────────────────────────────────────────────────
USDA_API_KEY: str = os.getenv("USDA_API_KEY", "")
EIA_API_KEY:  str = os.getenv("EIA_API_KEY", "")

# ── Storage ───────────────────────────────────────────────────────────────────
CHROMA_DB_PATH:          str = os.getenv("CHROMA_DB_PATH", "./data/chroma")
HISTORICAL_EVENTS_PATH:  str = os.getenv("HISTORICAL_EVENTS_PATH", "./rag/historical_events.json")

# ── Analysis settings ─────────────────────────────────────────────────────────
RAG_REFRESH_ON_ANALYSIS: bool = os.getenv("RAG_REFRESH_ON_ANALYSIS", "true").lower() == "true"
MAX_DEBATE_ROUNDS:       int  = int(os.getenv("MAX_DEBATE_ROUNDS", "2"))
SYNTHESIS_MODEL_OVERRIDE: bool = os.getenv("SYNTHESIS_MODEL_OVERRIDE", "true").lower() == "true"

# ── RSS feeds ─────────────────────────────────────────────────────────────────
RSS_FEEDS: list[str] = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.ft.com/rss/home/uk",
    "https://www.usda.gov/rss/home.xml",
    "https://www.eia.gov/rss/press_releases.xml",
]

# ── External URLs ─────────────────────────────────────────────────────────────
NEWSAPI_BASE_URL:   str = "https://newsapi.org/v2/everything"
OPEN_METEO_URL:     str = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE: str = "https://archive-api.open-meteo.com/v1/archive"
USDA_QUICKSTATS_URL: str = "https://quickstats.nass.usda.gov/api/api_GET/"
EIA_BASE_URL:       str = "https://api.eia.gov/v2/"
BDI_URL:            str = "https://markets.businessinsider.com/commodities/baltic-dry-index"

# ── Debate thresholds ─────────────────────────────────────────────────────────
DEBATE_SKIP_THRESHOLD:   float = 0.15   # skip if all agents within this confidence
DEBATE_MEDIATE_THRESHOLD: float = 0.10  # trigger round 2 if revision exceeds this

# ── RAG ───────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL:      str   = "all-MiniLM-L6-v2"
RAG_SIMILARITY_THRESHOLD: float = 0.6
NEWS_COLLECTION:      str   = "commodity_news"
HISTORY_COLLECTION:   str   = "historical_events"
NEWS_RETENTION_DAYS:  int   = 30

# ── Validation ────────────────────────────────────────────────────────────────
def validate() -> list[str]:
    """Return list of warnings for missing optional keys."""
    warnings = []
    if not ANTHROPIC_API_KEY and LLM_PROVIDER == "anthropic":
        warnings.append("ANTHROPIC_API_KEY not set")
    if not OPENAI_API_KEY and LLM_PROVIDER == "openai":
        warnings.append("OPENAI_API_KEY not set")
    if not NEWSAPI_KEY:
        warnings.append("NEWSAPI_KEY not set — will use RSS feeds only")
    if not USDA_API_KEY:
        warnings.append("USDA_API_KEY not set — USDA Quick Stats unavailable")
    if not EIA_API_KEY:
        warnings.append("EIA_API_KEY not set — EIA inventory data unavailable")
    return warnings
