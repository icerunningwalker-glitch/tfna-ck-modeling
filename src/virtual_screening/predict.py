"""Regenerate candidate exports from an existing terminal screen without docking."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from finalize_report import main


if __name__=='__main__':main()
