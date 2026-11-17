"""Import sportsbook lines the user obtains themselves (e.g. a licensed feed or a file they
downloaded by hand) into the odds database.

Expected file: data/manual/sportsbook_lines.csv with columns
  league      nba | nfl
  date        local game date, YYYY-MM-DD
  home, away  team abbreviations as used in nba_games / nfl_games (e.g. BOS, GSW / KC, LV)
  book        free text (e.g. "pinnacle", "consensus")
  snapshot    open | close | other
  ts_utc      optional ISO timestamp of the quote (UTC)
  market      moneyline | spread | total
  line        spread for the home team (negative = favourite) or total points; blank for moneyline
  home_price  American odds for home / over
  away_price  American odds for away / under
Rows that do not match exactly one scheduled game are reported and skipped, never guessed.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import MANUAL_DIR

log = logging.getLogger(__name__)
REQUIRED = ["league", "date", "home", "away", "book", "snapshot", "market", "home_price", "away_price"]


def load_manual_lines(games: pd.DataFrame) -> pd.DataFrame:
    path = MANUAL_DIR / "sportsbook_lines.csv"
    if not path.exists():
        return pd.DataFrame()
    m = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in m.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    g = games[["league", "game_id", "date", "home_abbr", "away_abbr"]].rename(columns={"home_abbr": "home",
                                                                                          "away_abbr": "away"})
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    j = m.merge(g, on=["league", "date", "home", "away"], how="left", indicator=True)
    bad = j[j["_merge"] != "both"]
    if len(bad):
        log.warning("%d manual rows matched no game (first: %s)", len(bad), bad.head(3).to_dict("records"))
    j = j[j["_merge"] == "both"].drop(columns="_merge")
    j["source"] = "manual:" + path.name
    cols = ["league", "game_id", "book", "snapshot", "market", "line", "home_price", "away_price", "source"]
    if "ts_utc" in j:
        cols.append("ts_utc")
    if "line" not in j:
        j["line"] = None
    return j[cols]
