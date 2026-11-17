"""Build tidy NFL tables from nflverse raw files (CC-BY-4.0).

Outputs (data/tidy):
  nfl_games.parquet      one row per game: kickoff in UTC, scores, closing moneyline /
                         spread / total with prices as recorded by nflverse, rest days,
                         starting QBs, roof/surface/weather, neutral flag
  nfl_injuries.parquet   weekly injury-report rows with ``date_modified`` (UTC)

Franchise keys are relocation-stable (OAK->LV, SD->LAC, STL->LA). ``spread_line`` in
nflverse is positive when the home team is favoured; we store ``home_spread`` with
the sportsbook sign convention (negative = home favoured).
"""
from __future__ import annotations

import glob
import logging

import numpy as np
import pandas as pd

from ..config import RAW_DIR, TIDY_DIR

log = logging.getLogger(__name__)
SRC = RAW_DIR / "nflverse"
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def build_games() -> pd.DataFrame:
    g = pd.read_parquet(SRC / "schedules" / "games.parquet")
    g = g.copy()
    g["home"] = g["home_team"].replace(FRANCHISE)
    g["away"] = g["away_team"].replace(FRANCHISE)
    local = pd.to_datetime(g["gameday"] + " " + g["gametime"].fillna("13:00"), errors="coerce")
    g["tip_utc"] = local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    g["date"] = pd.to_datetime(g["gameday"]).dt.normalize()
    out = pd.DataFrame({
        "game_id": g["game_id"], "season": g["season"].astype(int), "season_type": g["game_type"],
        "week": g["week"].astype(int), "playoff": g["game_type"] != "REG", "date": g["date"], "tip_utc": g["tip_utc"],
        "tip_known": g["gametime"].notna(), "home": g["home"], "away": g["away"],
        "home_abbr": g["home"], "away_abbr": g["away"], "home_abbr_hist": g["home_team"], "away_abbr_hist": g["away_team"],
        "home_pts": g["home_score"], "away_pts": g["away_score"], "neutral": g["location"].eq("Neutral"),
        "overtime": g["overtime"], "home_rest": g["home_rest"], "away_rest": g["away_rest"],
        "home_ml": g["home_moneyline"], "away_ml": g["away_moneyline"],
        "home_spread": -g["spread_line"], "home_spread_odds": g["home_spread_odds"],
        "away_spread_odds": g["away_spread_odds"], "total_line": g["total_line"], "over_odds": g["over_odds"],
        "under_odds": g["under_odds"], "div_game": g["div_game"].astype(bool), "roof": g["roof"],
        "surface": g["surface"], "temp": g["temp"], "wind": g["wind"], "home_qb_id": g["home_qb_id"],
        "away_qb_id": g["away_qb_id"], "home_qb": g["home_qb_name"], "away_qb": g["away_qb_name"],
        "home_coach": g["home_coach"], "away_coach": g["away_coach"], "stadium_id": g["stadium_id"],
    })
    out["result_available_utc"] = out["tip_utc"] + pd.Timedelta(hours=3, minutes=30)
    out = out.sort_values(["tip_utc", "game_id"]).reset_index(drop=True)
    log.info("nfl_games: %d rows, %d with closing moneyline", len(out), out["home_ml"].notna().sum())
    return out


def build_injuries() -> pd.DataFrame:
    files = sorted(glob.glob(str(SRC / "injuries" / "*.parquet")))
    inj = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    inj["team"] = inj["team"].replace(FRANCHISE)
    inj["report_utc"] = pd.to_datetime(inj["date_modified"], utc=True, errors="coerce")
    keep = ["season", "game_type", "week", "team", "gsis_id", "position", "full_name", "report_status",
            "report_primary_injury", "practice_status", "report_utc"]
    return inj[keep].reset_index(drop=True)


def build_all() -> dict[str, pd.DataFrame]:
    games = build_games()
    inj = build_injuries()
    games.to_parquet(TIDY_DIR / "nfl_games.parquet", index=False)
    inj.to_parquet(TIDY_DIR / "nfl_injuries.parquet", index=False)
    return {"games": games, "injuries": inj}


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(TIDY_DIR / f"nfl_{name}.parquet")
