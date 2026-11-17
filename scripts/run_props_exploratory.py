"""Post-hoc (NOT pre-registered) diagnostics for the props replication; run after run_props_replication.py.

    python scripts/run_props_exploratory.py      # writes outputs/nba/props_rep_exploratory.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sq.research.props_exploratory import run  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=float))
