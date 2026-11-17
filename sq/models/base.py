"""Online (sequential) model interface and the leakage-safe runner.

A model sees games strictly in time order. For every calendar date the runner
first asks for predictions for *all* games on that date and only then feeds the
results back, so no same-day result can leak into a same-day forecast. Season
boundaries trigger ``start_season`` (regression to the mean, variance inflation).

Required columns of ``games``: game_id, season, date (local game date),
home, away, neutral (bool), home_pts, away_pts. Optional: tip_utc.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class OnlineModel:
    name = "base"

    def reset(self, teams: list[Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def start_season(self, season: int) -> None:
        pass

    def predict(self, g) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, g) -> None:  # pragma: no cover - interface
        raise NotImplementedError


def _sorted(games: pd.DataFrame) -> pd.DataFrame:
    keys = ["date", "tip_utc", "game_id"] if "tip_utc" in games else ["date", "game_id"]
    return games.sort_values(keys, kind="mergesort").reset_index(drop=True)


def run_online(model: OnlineModel, games: pd.DataFrame, predict_mask: np.ndarray | None = None,
               update_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Sequentially predict-then-update. Returns one prediction row per game.

    ``update_mask`` lets callers withhold results (e.g. unplayed games); games with
    missing scores are never used for updates.
    """
    g = _sorted(games)
    teams = sorted(set(g["home"]).union(g["away"]), key=str)
    model.reset(teams)
    has_score = g["home_pts"].notna().to_numpy() & g["away_pts"].notna().to_numpy()
    upd = has_score if update_mask is None else (has_score & np.asarray(update_mask)[g.index])
    out: list[dict] = []
    season = None
    rows = list(g.itertuples(index=False))
    dates = g["date"].to_numpy()
    i, n = 0, len(rows)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        day = rows[i:j]
        if day[0].season != season:
            season = day[0].season
            model.start_season(season)
        for r in day:
            pr = model.predict(r)
            pr["game_id"] = r.game_id
            out.append(pr)
        for k, r in enumerate(day):
            if upd[i + k]:
                model.update(r)
        i = j
    pred = pd.DataFrame(out)
    return pred


def normal_cdf(x):
    from scipy.special import ndtr

    return ndtr(x)
