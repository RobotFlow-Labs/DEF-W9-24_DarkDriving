#!/usr/bin/env python3
"""CLI wrapper for DarkDriving enhancement training.

Usage:
    uv run python scripts/train.py --config configs/paper.toml
    uv run python scripts/train.py --config configs/debug.toml --max-steps 5
    uv run python scripts/train.py --config configs/paper.toml --resume checkpoints/best.pth
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dark_driving.train import main

if __name__ == "__main__":
    main()
