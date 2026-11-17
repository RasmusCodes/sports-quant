"""NFL pipeline: python scripts/run_nfl.py [--retune] [--eval]"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sq.config import OUT  # noqa: E402
from sq.research import nfl_models, nfl_report  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retune", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(OUT / "logs" / "nfl.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    df = None if a.eval else nfl_models.run(force_tune=a.retune)
    print(json.dumps(nfl_report.run(df), indent=2, default=str))
