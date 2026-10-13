"""AI module locations, resolved from this source file rather than cwd."""
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "ai_optimization"
MODELS_ROOT = PROJECT_ROOT / "models" / "ai_optimization"
RESULTS_ROOT = PROJECT_ROOT / "results" / "ai_optimization"
LOGS_ROOT = PROJECT_ROOT / "logs" / "ai_optimization"

