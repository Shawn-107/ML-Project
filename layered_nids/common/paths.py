"""Project paths, so no module hard-codes a location."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
