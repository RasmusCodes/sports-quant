"""Elo ratings with home advantage, margin-of-victory multiplier and season regression.

Defaults follow FiveThirtyEight's published NBA/NFL Elo methodology (K=20, home
advantage 100 / 48 points, MOV multiplier with autocorrelation correction, 75% /
67% carry-over between seasons). ``mov_mode=None`` gives classic win/loss Elo.
"""
from __future__ import annotations

import math

from .base import OnlineModel, normal_cdf

PRESETS = {
    "nba": dict(k=20.0, hca=100.0, carry=0.75, mean=1505.0, mov_mode="nba538", pts_per_elo=28.0, sd_margin=12.5),
    "nfl": dict(k=20.0, hca=48.0, carry=2 / 3, mean=1505.0, mov_mode="nfl538", pts_per_elo=25.0, sd_margin=13.5),
}


class Elo(OnlineModel):
    def __init__(self, k=20.0, hca=100.0, carry=0.75, mean=1505.0, mov_mode="nba538", pts_per_elo=28.0,
                 sd_margin=12.5, mov_exp=0.8, name="elo"):
        self.k, self.hca, self.carry, self.mean = k, hca, carry, mean
        self.mov_mode, self.pts_per_elo, self.sd_margin, self.mov_exp = mov_mode, pts_per_elo, sd_margin, mov_exp
        self.name = name
        self.r: dict = {}

    @classmethod
    def preset(cls, league, **over):
        return cls(**{**PRESETS[league], **over})

    def reset(self, teams):
        self.r = {t: self.mean for t in teams}

    def start_season(self, season):
        for t in self.r:
            self.r[t] = self.carry * self.r[t] + (1 - self.carry) * self.mean

    def diff(self, g):
        return self.r[g.home] - self.r[g.away] + (0.0 if g.neutral else self.hca)

    def predict(self, g):
        d = self.diff(g)
        p = 1.0 / (1.0 + 10 ** (-d / 400.0))
        mu = d / self.pts_per_elo
        return {"p_home": p, "elo_home": self.r[g.home], "elo_away": self.r[g.away], "elo_diff": d,
                "mu_margin": mu, "sd_margin": self.sd_margin, "p_home_margin": float(normal_cdf(mu / self.sd_margin))}

    def _mult(self, mov, winner_diff):
        if self.mov_mode == "nba538":
            return ((mov + 3.0) ** self.mov_exp) / (7.5 + 0.006 * winner_diff)
        if self.mov_mode == "nfl538":
            return math.log(mov + 1.0) * 2.2 / (winner_diff * 0.001 + 2.2)
        return 1.0

    def update(self, g):
        d = self.diff(g)
        p = 1.0 / (1.0 + 10 ** (-d / 400.0))
        margin = g.home_pts - g.away_pts
        s = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        winner_diff = d if margin > 0 else -d
        shift = self.k * self._mult(abs(margin), winner_diff) * (s - p)
        self.r[g.home] += shift
        self.r[g.away] -= shift
