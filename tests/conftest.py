import sys
from pathlib import Path

# Make benchmarks/ and tools/ importable from tests (e.g. ``import metrics``).
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "benchmarks"))
sys.path.insert(0, str(_ROOT / "tools"))
