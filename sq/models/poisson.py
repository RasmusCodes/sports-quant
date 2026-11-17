"""Time-weighted Poisson attack/defence model (Maher 1982; Dixon & Coles 1997 weighting).

log E[home_pts] = mu + h + att_home - def_away
log E[away_pts] = mu       + att_away - def_home
Fitted by weighted, ridge-penalised Poisson maximum likelihood (IRLS) on games in
a trailing window with weights exp(-ln2 * age / half_life), refitted every game
day using only results available before that day.

Two win-probability read-outs are reported:
  p_home_skellam  - independent Poisson scores -> Skellam margin (ties split 50/50)
  p_home          - Gaussian margin with the Poisson means and an empirically
                    tracked residual SD
The comparison is itself a finding: independent Poisson scores overstate margin
variance in the NBA (scores are positively correlated through pace), so the
Skellam read-out is too timid.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import skellam

from .base import OnlineModel, normal_cdf


class PoissonTeam(OnlineModel):
    def __init__(self, half_life=60.0, window=400, ridge=2.0, sd0=12.5, sd_decay=0.002, min_games=60,
                 name="poisson"):
        self.half_life, self.window, self.ridge = half_life, window, ridge
        self.sd0, self.sd_decay, self.min_games = sd0, sd_decay, min_games
        self.name = name

    def reset(self, teams):
        self.teams = list(teams)
        self.idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        self.beta = np.zeros(2 * n + 2)
        self.beta[0] = math.log(100.0)
        self.hist = []        # (date, i, j, hpts, apts, neutral), appended in time order
        self.start = 0        # first index inside the trailing window
        self.fitted_on = None
        self.var_res = self.sd0 ** 2

    def _fit(self, date):
        n = len(self.teams)
        cutoff = date - np.timedelta64(self.window, "D")
        while self.start < len(self.hist) and self.hist[self.start][0] < cutoff:
            self.start += 1
        H = self.hist[self.start:]
        if len(H) < self.min_games:
            return
        d = np.array([r[0] for r in H])
        age = (date - d) / np.timedelta64(1, "D")
        w = np.exp(-math.log(2) * age / self.half_life)
        i = np.array([r[1] for r in H])
        j = np.array([r[2] for r in H])
        hp = np.array([r[3] for r in H], dtype=float)
        ap = np.array([r[4] for r in H], dtype=float)
        neu = np.array([r[5] for r in H], dtype=float)
        m = len(H)
        X = np.zeros((2 * m, 2 * n + 2))
        rows = np.arange(m)
        X[rows, 0] = 1.0
        X[rows, 1] = 1.0 - neu
        X[rows, 2 + i] = 1.0
        X[rows, 2 + n + j] = -1.0
        X[m + rows, 0] = 1.0
        X[m + rows, 2 + j] = 1.0
        X[m + rows, 2 + n + i] = -1.0
        y = np.concatenate([hp, ap])
        ww = np.concatenate([w, w])
        b = self.beta.copy()
        for _ in range(8):
            eta = X @ b
            lam = np.exp(eta)
            W = ww * lam
            z = eta + (y - lam) / lam
            # ridge worth ``ridge`` average observations on team effects: identifiability + mild shrinkage
            pen = np.full(2 * n + 2, self.ridge * W.mean())
            pen[:2] = 1e-9
            XW = X * W[:, None]
            bn = np.linalg.solve(XW.T @ X + np.diag(pen), XW.T @ z)
            if np.max(np.abs(bn - b)) < 1e-5:
                b = bn
                break
            b = bn
        self.beta = b
        self.fitted_on = date

    def _lams(self, g):
        n = len(self.teams)
        i, j = self.idx[g.home], self.idx[g.away]
        b = self.beta
        h = 0.0 if g.neutral else b[1]
        lh = math.exp(b[0] + h + b[2 + i] - b[2 + n + j])
        la = math.exp(b[0] + b[2 + j] - b[2 + n + i])
        return lh, la

    def predict(self, g):
        date = np.datetime64(g.date, "D")
        if self.fitted_on != date and self.hist and self.hist[-1][0] < date:
            self._fit(date)
        lh, la = self._lams(g)
        sd = math.sqrt(self.var_res)
        mu = lh - la
        p_sk = float(skellam.sf(0, lh, la) + 0.5 * skellam.pmf(0, lh, la))
        return {"p_home": float(normal_cdf(mu / sd)), "p_home_skellam": p_sk, "mu_margin": mu, "sd_margin": sd,
                "mu_total": lh + la, "lam_home": lh, "lam_away": la}

    def update(self, g):
        date = np.datetime64(g.date, "D")
        lh, la = self._lams(g)
        res = (g.home_pts - g.away_pts) - (lh - la)
        if self.fitted_on is not None:
            self.var_res = (1 - self.sd_decay) * self.var_res + self.sd_decay * res ** 2
        self.hist.append((date, self.idx[g.home], self.idx[g.away], g.home_pts, g.away_pts, bool(g.neutral)))
