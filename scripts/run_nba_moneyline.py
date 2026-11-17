"""NBA moneyline benchmark end to end.

    python scripts/run_nba_moneyline.py            # tune (cached), predict, evaluate
    python scripts/run_nba_moneyline.py --retune   # refit hyperparameters on 2004-2015
    python scripts/run_nba_moneyline.py --eval     # evaluation only (uses saved predictions)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sq.config import OUT  # noqa: E402
from sq.research import nba_models, nba_report  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retune", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(OUT / "logs" / "nba_moneyline.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    df = None if a.eval else nba_models.run(force_tune=a.retune)
    res = nba_report.run(df)
    print(json.dumps(res, indent=2, default=str))
