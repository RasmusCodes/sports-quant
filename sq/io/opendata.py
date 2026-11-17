"""Open-licensed league data distributed as GitHub release assets.

NBA - sportsdataverse/sportsdataverse-data (MIT-licensed repository). Tables come
      from stats.nba.com (``nba_stats_*``: schedules, team/player box scores, the
      official NBA injury report parsed with its publication timestamp) and ESPN
      (``espn_nba_*``: schedule with UTC tip-off times, player box with starter /
      did-not-play flags).
NFL - nflverse/nflverse-data (CC-BY-4.0): games (results + closing moneyline,
      spread and total with prices), weekly injury reports, player stats, snap counts,
      depth charts.

Assets are listed once per day through the GitHub releases API and downloaded
from github.com/<repo>/releases/download/<tag>/<file> (not robots-disallowed).
The current season's files change daily and are re-downloaded; finished seasons
are fetched once.
"""
from __future__ import annotations

import datetime as dt
import logging

from .http import Fetcher

log = logging.getLogger(__name__)
GH_API = "https://api.github.com/repos/{repo}/releases/tags/{tag}"
GH_DL = "https://github.com/{repo}/releases/download/{tag}/{name}"

SDV = "sportsdataverse/sportsdataverse-data"
NFLV = "nflverse/nflverse-data"

# season label = year the season ends (2025 = 2024-25)
NBA_TABLES = {
    "schedule": ("nba_stats_schedules", "nba_schedule_{y}.parquet", range(1997, 2028)),
    "team_box": ("nba_stats_team_boxscores", "team_boxscores_{y}.parquet", range(1997, 2028)),
    "player_box": ("nba_stats_player_boxscores", "player_boxscores_{y}.parquet", range(1997, 2028)),
    "injury_report": ("nba_stats_injury_report", "injury_report_{y}.parquet", range(2019, 2028)),
    "espn_schedule": ("espn_nba_schedules", "nba_schedule_{y}.parquet", range(2002, 2028)),
    "espn_player_box": ("espn_nba_player_boxscores", "player_box_{y}.parquet", range(2002, 2028)),
}
NBA_CURRENT = 2027

# NFL season label = calendar year the season starts
NFL_TABLES = {
    "games": ("schedules", "games.parquet", [None]),
    "injuries": ("injuries", "injuries_{y}.parquet", range(2009, 2027)),
    "player_week": ("stats_player", "stats_player_week_{y}.parquet", range(1999, 2027)),
    "team_week": ("stats_team", "stats_team_week_{y}.parquet", range(1999, 2027)),
    "snap_counts": ("snap_counts", "snap_counts_{y}.parquet", range(2012, 2027)),
    "depth_charts": ("depth_charts", "depth_charts_{y}.parquet", range(2001, 2027)),
}
NFL_CURRENT = 2026


def release_assets(f: Fetcher, repo: str, tag: str) -> dict[str, dict]:
    stamp = dt.date.today().strftime("%Y%m%d")
    j = f.get_json(GH_API.format(repo=repo, tag=tag), name=f"gh_releases/{repo.split('/')[0]}_{tag}_{stamp}")
    return {a["name"]: {"size": a["size"], "updated_at": a.get("updated_at")} for a in j.get("assets", [])}


def fetch_tables(repo: str, tables: dict, current: int, only: list[str] | None = None) -> dict[str, list]:
    f = Fetcher(repo.split("/")[0], min_interval=0.5)
    got: dict[str, list] = {}
    for key, (tag, pattern, years) in tables.items():
        if only and key not in only:
            continue
        assets = release_assets(f, repo, tag)
        got[key] = []
        for y in years:
            name = pattern.format(y=y) if y is not None else pattern
            if name not in assets:
                log.info("missing asset %s/%s", tag, name)
                continue
            refresh = (y is None) or (y >= current)
            p = f.get_file(GH_DL.format(repo=repo, tag=tag, name=name), f"{tag}/{name}", refresh=refresh)
            got[key].append(p)
        log.info("%s/%s: %d files", repo, key, len(got[key]))
    return got


def fetch_nba(only: list[str] | None = None):
    return fetch_tables(SDV, NBA_TABLES, NBA_CURRENT, only)


def fetch_nfl(only: list[str] | None = None):
    return fetch_tables(NFLV, NFL_TABLES, NFL_CURRENT, only)
