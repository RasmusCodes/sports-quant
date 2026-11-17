"""Backtests, efficiency studies and figures (after run_nba_moneyline.py and run_nfl.py).

    python scripts/run_research.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sq.config import OUT  # noqa: E402
from sq.research import backtests, figures, nba_lines, studies  # noqa: E402

STEPS = {"backtests": lambda: {"nba": backtests.nba_backtests(), "nfl": backtests.nfl_backtests()},
         "studies": lambda: {"nba": studies.nba_studies(), "nfl": studies.nfl_studies()},
         "lines": nba_lines.run,
         "figures": figures.make_all}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(OUT / "logs" / "research.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    todo = sys.argv[1:] or list(STEPS)
    for step in todo:
        res = STEPS[step]()
        print(step, "done:", json.dumps(res, default=str)[:600])
