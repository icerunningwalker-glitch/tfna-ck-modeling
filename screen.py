"""Eight-box virtual screening requires user-supplied real inputs and a Vina executable."""
import sys
from src.submission_runtime import dispatch

if __name__ == "__main__":
    dispatch("src/virtual_screening/screen.py", sys.argv[1:])
