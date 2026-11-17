"""Download raw data into data/raw (cached, resumable) and write tidy market tables.

    python scripts/fetch.py nba                      # sportsdataverse NBA tables
    python scripts/fetch.py nfl                      # nflverse NFL tables
    python scripts/fetch.py kalshi nba moneyline     # Kalshi markets + candles
    python scripts/fetch.py poly nba moneyline       # Polymarket markets + price history
    python scripts/fetch.py kalshi nba player_points --sample 0.15 --no-hourly
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sq.config import OUT  # noqa: E402
from sq.io import kalshi, opendata, polymarket  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["nba", "nfl", "kalshi", "poly"])
    ap.add_argument("league", nargs="?", default="nba")
    ap.add_argument("market", nargs="?", default="moneyline")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--sample", type=float, default=None, help="fraction of events (Kalshi props)")
    ap.add_argument("--no-hourly", action="store_true")
    ap.add_argument("--top-lines", type=int, default=None, help="Polymarket: lines per game for spreads/totals")
    ap.add_argument("--min-volume", type=float, default=0.0, help="Kalshi: skip markets with less volume")
    ap.add_argument("--exclude-from", default=None, help="Kalshi: parquet whose event_ticker values are skipped")
    ap.add_argument("--suffix", default=None, help="Kalshi: output file suffix (keeps earlier files)")
    a = ap.parse_args()

    logdir = OUT / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    tag = a.source if a.source in ("nba", "nfl") else f"{a.source}_{a.league}_{a.market}"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(logdir / f"fetch_{tag}.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    if a.source == "nba":
        opendata.fetch_nba(a.only)
    elif a.source == "nfl":
        opendata.fetch_nfl(a.only)
    elif a.source == "kalshi":
        excl = None
        if a.exclude_from:
            import pandas as pd

            excl = set(pd.read_parquet(a.exclude_from, columns=["event_ticker"])["event_ticker"])
        kalshi.collect(a.league, a.market, hourly=not a.no_hourly, sample_frac=a.sample,
                       min_volume=a.min_volume, exclude_events=excl, out_suffix=a.suffix)
    elif a.source == "poly":
        polymarket.collect(a.league, a.market, hourly=not a.no_hourly, top_lines_per_game=a.top_lines)


if __name__ == "__main__":
    main()
