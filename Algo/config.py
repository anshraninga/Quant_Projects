import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Add root to sys.path so existing modules resolve correctly
# when imported via the existing_modules re-export stubs.
_root_str = str(ROOT_DIR)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

# ── LLM ───────────────────────────────────────────────────
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_MODEL    = "claude-haiku-4-5-20251001"
OPENAI_MODEL       = "gpt-4o-mini"

# ── RAG ───────────────────────────────────────────────────
CHROMA_DB_PATH    = os.getenv("CHROMA_DB_PATH",    str(DATA_DIR / "chroma"))
RAG_REFRESH_HOURS = int(os.getenv("RAG_REFRESH_HOURS", "6"))
RAG_WINDOW_HOURS  = 72   # only docs published within this window are queried

# ── Memory / Checkpointing ────────────────────────────────
SQLITE_DB_PATH     = os.getenv("SQLITE_DB_PATH",     str(DATA_DIR / "alpha_seeker_memory.db"))
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", str(DATA_DIR / "alpha_seeker_checkpoints.db"))

# ── API ───────────────────────────────────────────────────
ANALYSIS_CACHE_MINUTES = int(os.getenv("ANALYSIS_CACHE_MINUTES", "15"))
MAX_DEBATE_ROUNDS      = int(os.getenv("MAX_DEBATE_ROUNDS", "2"))

# ── External API keys (reused from existing modules) ──────
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
NEWSAPI_KEY         = os.getenv("NEWSAPI_KEY", "")
