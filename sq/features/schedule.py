"""Schedule, rest, travel and altitude features (known as soon as the schedule is).

For each team-game: days of rest before the game (0 = back-to-back), games in the
previous 4 and 7 days, great-circle km travelled since the previous game, time-zone
hours shifted (signed, east positive), and a flag for playing at altitude
(DEN, UTA). Locations: home team's arena by era; 2020 bubble games in Orlando.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (lat, lon, utc offset std, altitude flag) for NBA franchise arenas, current era
NBA_ARENA = {
    "ATL": (33.757, -84.396, -5, 0), "BOS": (42.366, -71.062, -5, 0), "BKN": (40.683, -73.976, -5, 0),
    "CHA": (35.225, -80.839, -5, 0), "CHI": (41.881, -87.674, -6, 0), "CLE": (41.497, -81.688, -5, 0),
    "DAL": (32.790, -96.810, -6, 0), "DEN": (39.749, -105.008, -7, 1), "DET": (42.341, -83.055, -5, 0),
    "GSW": (37.768, -122.388, -8, 0), "HOU": (29.751, -95.362, -6, 0), "IND": (39.764, -86.155, -5, 0),
    "LAC": (34.043, -118.267, -8, 0), "LAL": (34.043, -118.267, -8, 0), "MEM": (35.138, -90.051, -6, 0),
    "MIA": (25.781, -80.188, -5, 0), "MIL": (43.045, -87.917, -6, 0), "MIN": (44.979, -93.276, -6, 0),
    "NOP": (29.949, -90.082, -6, 0), "NYK": (40.751, -73.993, -5, 0), "OKC": (35.463, -97.515, -6, 0),
    "ORL": (28.539, -81.384, -5, 0), "PHI": (39.901, -75.172, -5, 0), "PHX": (33.446, -112.071, -7, 0),
    "POR": (45.532, -122.667, -8, 0), "SAC": (38.580, -121.500, -8, 0), "SAS": (29.427, -98.438, -6, 0),
    "TOR": (43.643, -79.379, -5, 0), "UTA": (40.768, -111.901, -7, 1), "WAS": (38.898, -77.021, -5, 0),
}
# historical home cities by historical abbreviation
NBA_ARENA_HIST = {"SEA": (47.622, -122.354, -8, 0), "NJN": (40.812, -74.074, -5, 0), "VAN": (49.278, -123.109, -8, 0),
                  "NOK": (35.463, -97.515, -6, 0)}
BUBBLE_LOC = (28.37, -81.55, -5, 0)

NFL_STADIUM = {
    "ARI": (33.528, -112.263, -7), "ATL": (33.755, -84.401, -5), "BAL": (39.278, -76.623, -5),
    "BUF": (42.774, -78.787, -5), "CAR": (35.226, -80.853, -5), "CHI": (41.862, -87.617, -6),
    "CIN": (39.095, -84.516, -5), "CLE": (41.506, -81.700, -5), "DAL": (32.748, -97.093, -6),
    "DEN": (39.744, -105.020, -7), "DET": (42.340, -83.046, -5), "GB": (44.501, -88.062, -6),
    "HOU": (29.685, -95.411, -6), "IND": (39.760, -86.164, -5), "JAX": (30.324, -81.637, -5),
    "KC": (39.049, -94.484, -6), "LA": (33.953, -118.339, -8), "LAC": (33.953, -118.339, -8),
    "LV": (36.091, -115.184, -8), "MIA": (25.958, -80.239, -5), "MIN": (44.974, -93.258, -6),
    "NE": (42.091, -71.264, -5), "NO": (29.951, -90.081, -6), "NYG": (40.814, -74.074, -5),
    "NYJ": (40.814, -74.074, -5), "PHI": (39.901, -75.168, -5), "PIT": (40.447, -80.016, -5),
    "SEA": (47.595, -122.332, -8), "SF": (37.403, -121.970, -8), "TB": (27.976, -82.503, -5),
    "TEN": (36.166, -86.771, -6), "WAS": (38.908, -76.864, -5),
}
NFL_HIST = {"OAK": (37.752, -122.201, -8), "SD": (32.783, -117.120, -8), "STL": (38.633, -90.188, -6)}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _nba_location(row):
    if row["neutral"] and pd.Timestamp("2020-07-30") <= row["date"] <= pd.Timestamp("2020-10-11"):
        return BUBBLE_LOC
    hist = row.get("home_abbr_hist")
    if hist in NBA_ARENA_HIST:
        return NBA_ARENA_HIST[hist]
    return NBA_ARENA.get(row["home_abbr"], (np.nan, np.nan, np.nan, 0))


def _nfl_location(row):
    hist = row.get("home_abbr_hist")
    if hist in NFL_HIST:
        lat, lon, tz = NFL_HIST[hist]
    else:
        lat, lon, tz = NFL_STADIUM.get(row["home_abbr"], (np.nan, np.nan, np.nan))
    return (lat, lon, tz, 1 if row["home_abbr"] == "DEN" else 0)


def team_schedule_features(games: pd.DataFrame, league: str = "nba") -> pd.DataFrame:
    """Return per-game home_/away_ rest, density, travel, tz shift and altitude features."""
    g = games.copy()
    loc = g.apply(_nba_location if league == "nba" else _nfl_location, axis=1, result_type="expand")
    g[["lat", "lon", "tz", "alt"]] = loc.to_numpy()
    long = pd.concat([
        g[["game_id", "season", "date", "lat", "lon", "tz", "alt"]].assign(team=g["home"], side="home"),
        g[["game_id", "season", "date", "lat", "lon", "tz", "alt"]].assign(team=g["away"], side="away"),
    ], ignore_index=True).sort_values(["team", "date", "game_id"])
    grp = long.groupby("team", sort=False)
    prev_date = grp["date"].shift(1)
    prev_season = grp["season"].shift(1)
    new_season = prev_season.ne(long["season"])
    rest = (long["date"] - prev_date).dt.days - 1
    long["rest"] = np.where(new_season | rest.isna(), 7, np.clip(rest, 0, 7))
    long["b2b"] = (long["rest"] == 0).astype(int)
    long["travel_km"] = np.where(new_season, 0.0, haversine_km(grp["lat"].shift(1), grp["lon"].shift(1),
                                                               long["lat"], long["lon"]))
    long["travel_km"] = np.nan_to_num(long["travel_km"].astype(float))
    long["tz_shift"] = np.where(new_season, 0.0, (long["tz"] - grp["tz"].shift(1)).fillna(0.0))
    # games played in the previous 4 / 7 days (excluding today)
    for w in (4, 7):
        cnt = []
        for _, d in long.groupby("team", sort=False):
            dates = d["date"].to_numpy()
            left = np.searchsorted(dates, dates - np.timedelta64(w, "D"), side="left")
            cnt.append(pd.Series(np.arange(len(dates)) - left, index=d.index))
        long[f"games_last{w}d"] = pd.concat(cnt)
    long["game_no"] = long.groupby(["team", "season"]).cumcount() + 1
    feats = ["rest", "b2b", "travel_km", "tz_shift", "games_last4d", "games_last7d", "game_no"]
    h = long[long["side"] == "home"].set_index("game_id")[feats].add_prefix("home_")
    a = long[long["side"] == "away"].set_index("game_id")[feats].add_prefix("away_")
    out = g[["game_id", "alt"]].set_index("game_id").join(h).join(a).reset_index()
    out = out.rename(columns={"alt": "home_altitude"})
    out["rest_diff"] = out["home_rest"].clip(0, 3) - out["away_rest"].clip(0, 3)
    return out
