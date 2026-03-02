#!/usr/bin/env python3
"""
run_gui.py — Convenience launcher for the SLOP web interface.

Usage:
    python run_gui.py

This is equivalent to:
    streamlit run gui/app.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
GUI_APP = REPO_ROOT / "gui" / "app.py"

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(GUI_APP)],
        cwd=str(REPO_ROOT),
    )
    sys.exit(result.returncode)
