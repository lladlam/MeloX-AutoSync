import sys
from pathlib import Path

# Make benchmarks/ importable from tests (e.g. ``import metrics``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
