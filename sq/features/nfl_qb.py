"""NFL starting-quarterback value (the dominant NFL player-availability effect).

For each QB, a strictly pre-game exponentially weighted EPA per play
(passing + rushing EPA over dropbacks + carries), shrunk toward a replacement-level
prior. Per game and team: the value of the announced starter (nflverse ``*_qb_id``,
known before kickoff), the value of the team's previous starter and the change.
"""
from __future__ import annotations

import glob
from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import RAW_DIR

SRC = RAW_DIR / "nflverse" / "stats_player"


def load_qb_weeks() -> pd.DataFrame:
    files = sorted(glob.glob(str(SRC / "stats_player_week_*.parquet")))
    cols = ["player_id", "season", "week", "game_id", "team", "attempts", "sacks_suffered", "carries",
            "passing_epa", "rushing_epa"]
    out = []
    for p in files:
        d = pd.read_parquet(p)
        d = d[d["attempts"].fillna(0) >= 5]
        out.append(d[[c for c in cols if c in d.columns]])
    q = pd.concat(out, ignore_index=True)
    q["plays"] = q["attempts"].fillna(0) + q["sacks_suffered"].fillna(0) + q["carries"].fillna(0)
    q["epa"] = q["passing_epa"].fillna(0) + q["rushing_epa"].fillna(0)
    return q


def qb_features(games: pd.DataFrame, qb: pd.DataFrame, half_life_games=10.0, prior_plays=120.0,
                prior_epa=-0.05, plays_per_game=36.0) -> pd.DataFrame:
    g = games.sort_values(["tip_utc", "game_id"]).reset_index(drop=True)
    by_game = {k: d for k, d in qb.groupby("game_id")}
    lam = 0.5 ** (1.0 / half_life_games)
    num = defaultdict(float)
    den = defaultdict(float)
    prev_starter = {}
    rows = []

    def value(pid):
        if not isinstance(pid, str):
            return np.nan
        rate = (num[pid] + prior_plays * prior_epa) / (den[pid] + prior_plays)
        return rate * plays_per_game

    for r in g.itertuples(index=False):
        rec = {"game_id": r.game_id}
        for side, team, qid in (("home", r.home, r.home_qb_id), ("away", r.away, r.away_qb_id)):
            v = value(qid)
            pv = value(prev_starter.get(team))
            rec[f"{side}_qb_val"] = v
            rec[f"{side}_qb_change"] = float(isinstance(qid, str) and prev_starter.get(team) not in (None, qid))
            rec[f"{side}_qb_delta"] = (v - pv) if np.isfinite(v) and np.isfinite(pv) else 0.0
            rec[f"{side}_qb_exp_plays"] = den[qid] if isinstance(qid, str) else 0.0
        rows.append(rec)
        if r.home_pts == r.home_pts:   # played: update after the game
            d = by_game.get(r.game_id)
            if d is not None:
                for x in d.itertuples(index=False):
                    num[x.player_id] = lam * num[x.player_id] + x.epa
                    den[x.player_id] = lam * den[x.player_id] + x.plays
            for team, qid in ((r.home, r.home_qb_id), (r.away, r.away_qb_id)):
                if isinstance(qid, str):
                    prev_starter[team] = qid
    f = pd.DataFrame(rows)
    f["qb_val_diff"] = f["home_qb_val"].fillna(0) - f["away_qb_val"].fillna(0)
    f["qb_delta_diff"] = f["home_qb_delta"] - f["away_qb_delta"]
    return f
