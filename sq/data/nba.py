"""Build tidy NBA tables from the sportsdataverse raw files.

Outputs (data/tidy):
  nba_games.parquet         one row per NBA-vs-NBA game (regular season, NBA Cup final,
                            play-in, playoffs) with UTC tip-off time and neutral flag
  nba_team_box.parquet      team box score per game (+ possessions estimate)
  nba_player_box.parquet    player box score per game (minutes as float, DNP comment,
                            starter flag where the source marks starters)
  nba_injuries.parquet      official NBA injury report rows with publication time in UTC

Timestamps: ``tip_utc`` comes from ESPN's schedule (UTC). Box-score information
is treated as available at ``tip_utc + 3h``; an injury-report row is available
at its ``report_utc``. Seasons are labelled by the year they end (2025 = 2024-25).
"""
from __future__ import annotations

import glob
import logging
import re
import unicodedata

import numpy as np
import pandas as pd

from ..config import RAW_DIR, TIDY_DIR

log = logging.getLogger(__name__)
SRC = RAW_DIR / "sportsdataverse"
KEEP_TYPES = {"regular-season", "play-in", "playoffs", "nba-cup"}
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
BUBBLE = (pd.Timestamp("2020-07-30"), pd.Timestamp("2020-10-11"))  # Orlando restart: all games neutral

# franchise id -> current abbreviation (for display and market matching)
CURRENT_ABBR = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN", 1610612766: "CHA", 1610612741: "CHI",
    1610612739: "CLE", 1610612742: "DAL", 1610612743: "DEN", 1610612765: "DET", 1610612744: "GSW",
    1610612745: "HOU", 1610612754: "IND", 1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM",
    1610612748: "MIA", 1610612749: "MIL", 1610612750: "MIN", 1610612740: "NOP", 1610612752: "NYK",
    1610612760: "OKC", 1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX", 1610612757: "POR",
    1610612758: "SAC", 1610612759: "SAS", 1610612761: "TOR", 1610612762: "UTA", 1610612764: "WAS",
}


def _read_all(table: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(SRC / table / "*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def _minutes(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    if ":" in s:
        m, sec = s.split(":", 1)
        try:
            return float(m) + float(sec) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    if "," in s:  # injury report style "Porter Jr., Michael"
        last, first = s.split(",", 1)
        s = f"{first} {last}"
    s = re.sub(r"[^a-z ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def build_games() -> pd.DataFrame:
    sch = _read_all("nba_stats_schedules")
    sch = sch[sch["season_type"].isin(KEEP_TYPES)].copy()
    nba = set(CURRENT_ABBR)
    sch = sch[sch["home_team_id"].isin(nba) & sch["away_team_id"].isin(nba)]
    sch = sch.drop_duplicates("game_id")
    sch["date"] = pd.to_datetime(sch["game_date"]).dt.normalize()

    es = _read_all("espn_nba_schedules")
    es["home_abbr_n"] = es["home_abbreviation"].replace(ESPN_TO_NBA)
    es["away_abbr_n"] = es["away_abbreviation"].replace(ESPN_TO_NBA)
    es["tip_utc"] = pd.to_datetime(es["date"], utc=True, errors="coerce")
    es["date_et"] = es["tip_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    # ESPN team id -> NBA franchise id, learned from seasons with stable abbreviations
    rec = sch[sch["season"] >= 2015][["date", "home_team_id", "home_team_abbreviation"]]
    m = es.merge(rec, left_on=["date_et", "home_abbr_n"], right_on=["date", "home_team_abbreviation"])
    id_map = m.groupby("home_id")["home_team_id"].agg(lambda s: s.mode().iloc[0]).to_dict()
    es["home_nba"] = es["home_id"].map(id_map)
    es["away_nba"] = es["away_id"].map(id_map)
    es = es.dropna(subset=["home_nba", "away_nba", "tip_utc"])
    es = es.drop_duplicates(["date_et", "home_nba", "away_nba"])
    es_cols = es[["date_et", "home_nba", "away_nba", "tip_utc", "neutral_site", "id", "notes_headline"]].rename(
        columns={"id": "espn_id"})
    g = sch.merge(es_cols, left_on=["date", "home_team_id", "away_team_id"],
                  right_on=["date_et", "home_nba", "away_nba"], how="left")
    g["tip_known"] = g["tip_utc"].notna()
    # default tip 19:30 ET when ESPN has no record (pre-2002 seasons, rare gaps)
    default = (g["date"] + pd.Timedelta(hours=19, minutes=30)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    g["tip_utc"] = g["tip_utc"].fillna(default)
    g["neutral"] = g["neutral_site"].fillna(False).astype(bool) | g["date"].between(*BUBBLE) | (
        g["season_type"] == "nba-cup")
    g["playoff"] = g["season_type"].isin(["playoffs", "play-in"])
    out = pd.DataFrame({
        "game_id": g["game_id"], "season": g["season"].astype(int), "season_type": g["season_type"],
        "playoff": g["playoff"], "date": g["date"], "tip_utc": g["tip_utc"], "tip_known": g["tip_known"],
        "home": g["home_team_id"].astype(int), "away": g["away_team_id"].astype(int),
        "home_abbr": g["home_team_id"].map(CURRENT_ABBR), "away_abbr": g["away_team_id"].map(CURRENT_ABBR),
        "home_abbr_hist": g["home_team_abbreviation"], "away_abbr_hist": g["away_team_abbreviation"],
        "home_pts": g["home_pts"], "away_pts": g["away_pts"], "neutral": g["neutral"],
        "espn_id": g["espn_id"], "notes": g["notes_headline"],
    })
    # a score of 0-0 or missing means not played yet
    unplayed = (out["home_pts"].fillna(0) + out["away_pts"].fillna(0)) <= 0
    out.loc[unplayed, ["home_pts", "away_pts"]] = np.nan
    out["result_available_utc"] = out["tip_utc"] + pd.Timedelta(hours=3)
    out = out.sort_values(["tip_utc", "game_id"]).reset_index(drop=True)
    log.info("nba_games: %d rows, tip known %.1f%%", len(out), 100 * out["tip_known"].mean())
    return out


def build_team_box(games: pd.DataFrame) -> pd.DataFrame:
    tb = _read_all("nba_stats_team_boxscores")
    tb = tb[tb["game_id"].isin(games["game_id"])].drop_duplicates(["game_id", "team_id"])
    tb["poss"] = tb["field_goals_attempted"] - tb["rebounds_offensive"] + tb["turnovers"] + 0.44 * tb["free_throws_attempted"]
    tb["minutes_played"] = tb["minutes"].map(_minutes)
    keep = ["game_id", "season", "team_id", "side", "minutes_played", "points", "field_goals_made",
            "field_goals_attempted", "three_pointers_made", "three_pointers_attempted", "free_throws_made",
            "free_throws_attempted", "rebounds_offensive", "rebounds_defensive", "assists", "steals", "blocks",
            "turnovers", "fouls_personal", "poss"]
    return tb[keep].reset_index(drop=True)


def build_player_box(games: pd.DataFrame) -> pd.DataFrame:
    pb = _read_all("nba_stats_player_boxscores")
    pb = pb[pb["game_id"].isin(games["game_id"])].drop_duplicates(["game_id", "person_id"])
    pb["min"] = pb["minutes"].map(_minutes)
    pos_filled = pb["position"].fillna("").str.len() > 0
    per_game = pos_filled.groupby(pb["game_id"]).transform("sum")
    pb["starter"] = np.where(per_game == 10, pos_filled, np.nan)  # only where the source marks exactly 10 starters
    pb["name"] = (pb["first_name"].fillna("") + " " + pb["family_name"].fillna("")).str.strip()
    pb["name_key"] = pb["name"].map(norm_name)
    keep = {"game_id": "game_id", "season": "season", "team_id": "team_id", "person_id": "person_id",
            "name": "name", "name_key": "name_key", "starter": "starter", "comment": "comment", "min": "min",
            "points": "pts", "field_goals_made": "fgm", "field_goals_attempted": "fga",
            "three_pointers_made": "fg3m", "three_pointers_attempted": "fg3a", "free_throws_made": "ftm",
            "free_throws_attempted": "fta", "rebounds_offensive": "oreb", "rebounds_defensive": "dreb",
            "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov", "fouls_personal": "pf",
            "plus_minus_points": "plus_minus"}
    out = pb[list(keep)].rename(columns=keep)
    return out.reset_index(drop=True)


def build_injuries(player_box: pd.DataFrame) -> pd.DataFrame:
    inj = _read_all("nba_stats_injury_report")
    inj["report_utc"] = (pd.to_datetime(inj["report_datetime"]).dt.tz_localize("America/New_York", ambiguous="NaT",
                                                                                 nonexistent="shift_forward")
                         .dt.tz_convert("UTC"))
    inj["game_id"] = inj["game_id"].map(lambda x: f"{int(x):010d}" if pd.notna(x) else None)
    inj["name_key"] = inj["report_player_name"].map(norm_name)
    inj["person_id"] = inj["person_id"].astype("float")
    # fill missing person ids by name within team-season from the box scores
    pk = player_box[["season", "team_id", "name_key", "person_id"]].drop_duplicates(["season", "team_id", "name_key"])
    miss = inj["person_id"].isna()
    fill = inj.loc[miss, ["season", "team_id", "name_key"]].merge(pk, on=["season", "team_id", "name_key"], how="left")
    inj.loc[miss, "person_id"] = fill["person_id"].to_numpy()
    out = inj[["season", "report_utc", "game_id", "game_date", "team_id", "person_id", "report_player_name",
               "current_status", "previous_status", "reason"]].rename(columns={"report_player_name": "player"})
    out = out.dropna(subset=["report_utc"]).sort_values("report_utc").reset_index(drop=True)
    log.info("nba_injuries: %d rows, person_id matched %.1f%%", len(out), 100 * out["person_id"].notna().mean())
    return out


def build_all() -> dict[str, pd.DataFrame]:
    games = build_games()
    tb = build_team_box(games)
    pb = build_player_box(games)
    inj = build_injuries(pb)
    for name, df in {"nba_games": games, "nba_team_box": tb, "nba_player_box": pb, "nba_injuries": inj}.items():
        df.to_parquet(TIDY_DIR / f"{name}.parquet", index=False)
    return {"games": games, "team_box": tb, "player_box": pb, "injuries": inj}


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(TIDY_DIR / f"nba_{name}.parquet")
