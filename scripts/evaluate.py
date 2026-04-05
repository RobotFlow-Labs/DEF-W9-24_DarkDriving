#!/usr/bin/env python3
"""CLI wrapper for DarkDriving evaluation.

Usage:
    uv run python scripts/evaluate.py --config configs/paper.toml --checkpoint best.pth
    uv run python scripts/evaluate.py --config configs/detection.toml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dark_driving.evaluate import main

if __name__ == "__main__":
    main()
