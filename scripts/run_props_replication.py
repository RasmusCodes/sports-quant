"""Pre-registered props replication (see paper/preregistration_props.md), run with the registered code.

The registered code is ``sq/research/nba_lines_registered.py``: a byte-for-byte copy of the
``sq/research/nba_lines.py`` whose sha256 the pre-registration lists. The default pipeline in
``sq/research/nba_lines.py`` has since moved to settlement-based scoring, so this script refuses to run if the
registered copy's hash has changed. It reads the registered discovery outputs (frozen coefficients and discovery
games) from, and writes the replication outputs to, ``outputs/nba/registered/`` (or ``--outdir``; that directory
must hold ``lines_props.json``, ``props_panel.parquet`` and ``predictions.parquet``, which the registered code
loads).

    python scripts/fetch.py kalshi nba player_points --no-hourly --min-volume 500 \
        --exclude-from data/tidy/kalshi_nba_player_points_markets.parquet --suffix rep
    python scripts/run_props_replication.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sq.config import OUT  # noqa: E402
from sq.research import nba_lines_registered as R  # noqa: E402

REGISTERED_SHA256 = "9b38870759d472ef804b766d28202d0e9cc2e63cf414d2d86e3a5be4858cdf30"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=str(OUT / "nba" / "registered"))
    args = ap.parse_args()
    digest = hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest()
    if digest != REGISTERED_SHA256:
        sys.exit(f"{R.__file__} no longer matches the registered sha256 ({digest})")
    R.OUTDIR = Path(args.outdir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    r = R.props_replication()
    keep = ("n_scored", "games", "players", "dates", "encompassing", "yes_at_ask", "strategy_h3_combined",
            "strategy_h3", "paired_model_minus_market", "paired_combined_minus_market", "verdict",
            "overlap_with_discovery_games")
    print(json.dumps({k: r.get(k) for k in keep}, indent=2, default=float))
