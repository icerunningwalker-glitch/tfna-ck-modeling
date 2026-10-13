"""Benchmark inputs, snapshots, outputs and logs are stored separately."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "ai_optimization" / "benchmark"
MODELS = PROJECT_ROOT / "models" / "ai_optimization" / "benchmark"
RESULTS = PROJECT_ROOT / "results" / "ai_optimization" / "benchmark"
LOGS = PROJECT_ROOT / "logs" / "ai_optimization" / "benchmark"


def locations(project_root=None):
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    return {"root": root,
            "data": root / "data" / "ai_optimization" / "benchmark",
            "models": root / "models" / "ai_optimization" / "benchmark",
            "results": root / "results" / "ai_optimization" / "benchmark",
            "logs": root / "logs" / "ai_optimization" / "benchmark"}
