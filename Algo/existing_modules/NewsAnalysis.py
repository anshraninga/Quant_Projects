import sys, os as _os
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from NewsAnalysis import run_news_analysis  # noqa: F401, E402
