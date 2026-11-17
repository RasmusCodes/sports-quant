"""Bayesian dynamic hierarchical model of team scoring (Gaussian state space).

State  x = [off_1..off_n, def_1..def_n, mu, h]
  off_i, def_i : team offensive / defensive strength (points vs league average)
  mu           : league scoring level per team-game
  h            : home advantage in points (split +h/2 home, -h/2 away)
Observation for a game (home i, away j):
  home_pts = mu + h/2 + off_i - def_j + e1
  away_pts = mu - h/2 + off_j - def_i + e2,     (e1, e2) ~ N(0, s^2 [[1, rho], [rho, 1]])
Dynamics: every state is a random walk; per elapsed day the variances grow by
q_team (teams), q_mu, q_h. Between seasons team strengths shrink toward the
league mean: x <- carry * x and P <- carry^2 P + (1 - carry^2) tau^2 I, i.e. the
hierarchical prior off_i, def_i ~ N(0, tau^2) is partially re-imposed each summer.

Exact Kalman filtering gives the posterior after every game and, through the
prediction-error decomposition, the marginal likelihood used to fit the
hyperparameters (empirical Bayes) on training seasons only. The predictive
distribution of the margin is Gaussian, so P(home win) = Phi(E[margin] / sd[margin]),
and totals / spreads come from the same joint distribution.
"""
from __future__ import annotations

import math

import numpy as np

from .base import OnlineModel, normal_cdf

PRESETS = {
    "nba": dict(s_obs=11.5, rho_obs=0.25, q_team=0.012, q_mu=0.004, q_h=0.0004, carry=0.65, tau=3.5,
                mu0=102.0, h0=3.0, var_mu0=25.0, var_h0=4.0, season_mu_var=4.0),
    "nfl": dict(s_obs=9.5, rho_obs=0.0, q_team=0.02, q_mu=0.004, q_h=0.0004, carry=0.6, tau=3.0,
                mu0=21.0, h0=2.5, var_mu0=9.0, var_h0=2.0, season_mu_var=1.0),
}


class KalmanScores(OnlineModel):
    def __init__(self, s_obs=11.5, rho_obs=0.25, q_team=0.012, q_mu=0.004, q_h=0.0004, carry=0.65, tau=3.5,
                 mu0=102.0, h0=3.0, var_mu0=25.0, var_h0=4.0, season_mu_var=4.0, name="kalman"):
        self.s_obs, self.rho_obs = s_obs, rho_obs
        self.q_team, self.q_mu, self.q_h = q_team, q_mu, q_h
        self.carry, self.tau = carry, tau
        self.mu0, self.h0, self.var_mu0, self.var_h0, self.season_mu_var = mu0, h0, var_mu0, var_h0, season_mu_var
        self.name = name
        self.loglik = 0.0
        self.n_obs = 0

    @classmethod
    def preset(cls, league, **over):
        return cls(**{**PRESETS[league], **over})

    def reset(self, teams):
        n = len(teams)
        self.n = n
        self.idx = {t: i for i, t in enumerate(teams)}
        dim = 2 * n + 2
        self.x = np.zeros(dim)
        self.x[2 * n] = self.mu0
        self.x[2 * n + 1] = self.h0
        self.P = np.diag([self.tau ** 2] * (2 * n) + [self.var_mu0, self.var_h0]).astype(float)
        s2 = self.s_obs ** 2
        self.R = np.array([[s2, self.rho_obs * s2], [self.rho_obs * s2, s2]])
        self.qdiag = np.array([self.q_team] * (2 * n) + [self.q_mu, self.q_h])
        self.last_date = None
        self.loglik, self.n_obs = 0.0, 0
        self.started = False

    def start_season(self, season):
        if not self.started:
            self.started = True
            return
        n = self.n
        a = np.ones(2 * n + 2)
        a[: 2 * n] = self.carry
        self.x[: 2 * n] *= self.carry
        self.P = self.P * np.outer(a, a)
        add = np.zeros(2 * n + 2)
        add[: 2 * n] = (1 - self.carry ** 2) * self.tau ** 2
        add[2 * n] = self.season_mu_var
        self.P[np.diag_indices_from(self.P)] += add

    def _advance(self, date):
        if self.last_date is not None:
            days = (date - self.last_date) / np.timedelta64(1, "D") if isinstance(date, np.datetime64) else (
                (date - self.last_date).days)
            days = max(float(days), 0.0)
            if days > 0:
                self.P[np.diag_indices_from(self.P)] += self.qdiag * min(days, 60.0)
        self.last_date = date

    def _design(self, g):
        n = self.n
        i, j = self.idx[g.home], self.idx[g.away]
        cols = np.array([i, j, n + i, n + j, 2 * n, 2 * n + 1])
        hh = 0.0 if g.neutral else 0.5
        # rows: home pts, away pts ; columns as in ``cols``
        Hc = np.array([[1.0, 0.0, 0.0, -1.0, 1.0, hh],
                       [0.0, 1.0, -1.0, 0.0, 1.0, -hh]])
        return cols, Hc

    def _moments(self, g):
        cols, Hc = self._design(g)
        m = Hc @ self.x[cols]
        S = Hc @ self.P[np.ix_(cols, cols)] @ Hc.T + self.R
        return cols, Hc, m, S

    def predict(self, g):
        self._advance(np.datetime64(g.date, "D"))
        _, _, m, S = self._moments(g)
        mu_margin = m[0] - m[1]
        var_margin = S[0, 0] + S[1, 1] - 2 * S[0, 1]
        mu_total = m[0] + m[1]
        var_total = S[0, 0] + S[1, 1] + 2 * S[0, 1]
        sd = math.sqrt(max(var_margin, 1e-9))
        n = self.n
        i, j = self.idx[g.home], self.idx[g.away]
        return {"p_home": float(normal_cdf(mu_margin / sd)), "mu_margin": float(mu_margin), "sd_margin": sd,
                "mu_total": float(mu_total), "sd_total": math.sqrt(max(var_total, 1e-9)),
                "mu_home_pts": float(m[0]), "mu_away_pts": float(m[1]),
                "off_home": float(self.x[i]), "def_home": float(self.x[n + i]),
                "off_away": float(self.x[j]), "def_away": float(self.x[n + j]),
                "hca": float(self.x[2 * n + 1]), "league_mu": float(self.x[2 * n]),
                "sd_strength": math.sqrt(max(var_margin - (self.R[0, 0] + self.R[1, 1] - 2 * self.R[0, 1]), 0.0))}

    def update(self, g):
        self._advance(np.datetime64(g.date, "D"))
        cols, Hc, m, S = self._moments(g)
        y = np.array([g.home_pts, g.away_pts], dtype=float)
        e = y - m
        Sinv = np.linalg.inv(S)
        PHt = self.P[:, cols] @ Hc.T                      # dim x 2
        K = PHt @ Sinv
        self.x = self.x + K @ e
        self.P = self.P - K @ PHt.T
        self.P = 0.5 * (self.P + self.P.T)
        # Gaussian log-likelihood of this observation (prediction-error decomposition)
        sign, logdet = np.linalg.slogdet(S)
        self.loglik += -0.5 * (e @ Sinv @ e + logdet + 2 * math.log(2 * math.pi))
        self.n_obs += 1


PARAM_KEYS = ["s_obs", "rho_obs", "q_team", "q_mu", "q_h", "carry", "tau"]


def _to_theta(p):
    return np.array([math.log(p["s_obs"]), math.atanh(p["rho_obs"]), math.log(p["q_team"]), math.log(p["q_mu"]),
                     math.log(p["q_h"]), math.log(p["carry"] / (1 - p["carry"])), math.log(p["tau"])])


def _from_theta(t, base):
    p = dict(base)
    p.update(s_obs=math.exp(t[0]), rho_obs=math.tanh(t[1]), q_team=math.exp(t[2]), q_mu=math.exp(t[3]),
             q_h=math.exp(t[4]), carry=1 / (1 + math.exp(-t[5])), tau=math.exp(t[6]))
    return p


def fit_hyperparameters(games, league, train_mask, burn_in_seasons=1, maxiter=250, verbose=False):
    """Empirical-Bayes fit: maximise the marginal likelihood of the training games only.

    The filter runs over every game up to the end of training (needed to carry state),
    but only training games after the burn-in seasons contribute to the objective.
    """
    from scipy.optimize import minimize

    from .base import _sorted

    g = _sorted(games[train_mask])
    first = sorted(g["season"].unique())[:burn_in_seasons]
    base = dict(PRESETS[league])
    rows = list(g.itertuples(index=False))
    scored = ~g["season"].isin(first).to_numpy()
    teams = sorted(set(g["home"]).union(g["away"]), key=str)
    dates = g["date"].to_numpy()

    def nll(t):
        p = _from_theta(t, base)
        m = KalmanScores(**p)
        m.reset(teams)
        season = None
        ll = 0.0
        for k, r in enumerate(rows):
            if r.season != season:
                season = r.season
                m.start_season(season)
            before = m.loglik
            m.update(r)
            if scored[k]:
                ll += m.loglik - before
        if verbose:
            print(round(-ll, 1), {k: round(v, 4) for k, v in p.items() if k in PARAM_KEYS})
        return -ll

    res = minimize(nll, _to_theta(base), method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 0.5})
    best = _from_theta(res.x, base)
    return best, res
