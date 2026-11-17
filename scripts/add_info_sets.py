"""Add walk-forward ML predictions for extra information sets to saved NBA predictions.

    python scripts/add_info_sets.py d1
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sq.research import nba_models  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df = nba_models.add_info_sets(sys.argv[1:] or ["d1"])
    print(df.filter(like="p_ensemble").describe().T)
