"""Run the independent reference tFNA transport model, preserving parameter values."""
import sys
from src.submission_runtime import dispatch

if __name__ == "__main__":
    dispatch("src/transport/run_model.py", sys.argv[1:])
