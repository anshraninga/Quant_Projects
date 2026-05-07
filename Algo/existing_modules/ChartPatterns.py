import sys, os as _os
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from ChartPatterns import run_chart_analysis  # noqa: F401, E402
