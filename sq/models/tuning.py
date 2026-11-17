"""Hyperparameter tuning for online rating models on training seasons only.

Objective: mean log loss of pre-game home-win probabilities over the scored
training games (a burn-in season is excluded so ratings can settle). Optimiser:
Nelder-Mead on transformed parameters (log for positive, logit for (0,1)).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..eval.metrics import log_loss
from .base import run_online

POS, UNIT, FREE = "pos", "unit", "free"


def _fwd(v, kind):
    return math.log(v) if kind == POS else (math.log(v / (1 - v)) if kind == UNIT else v)


def _inv(t, kind):
    return math.exp(t) if kind == POS else (1 / (1 + math.exp(-t)) if kind == UNIT else t)


def tune(make_model, games: pd.DataFrame, space: dict[str, tuple[float, str]], score_seasons, maxiter=120,
         fixed: dict | None = None, verbose=False):
    fixed = fixed or {}
    names = list(space)
    x0 = np.array([_fwd(space[n][0], space[n][1]) for n in names])
    g = games[games["season"] <= max(score_seasons)]
    y_all = (g["home_pts"] > g["away_pts"]).astype(float)
    scored_ids = set(g.loc[g["season"].isin(score_seasons) & g["home_pts"].notna(), "game_id"])

    def obj(x):
        params = {n: _inv(t, space[n][1]) for n, t in zip(names, x)}
        m = make_model(**fixed, **params)
        p = run_online(m, g)
        d = p[p["game_id"].isin(scored_ids)].merge(g[["game_id"]].assign(y=y_all.to_numpy()), on="game_id")
        ll = log_loss(d["p_home"], d["y"])
        if verbose:
            print(round(ll, 5), {k: round(v, 4) for k, v in params.items()})
        return ll

    if not names:  # nothing to optimise: just score the fixed configuration
        return dict(fixed), float(obj(np.array([])))
    res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-6})
    best = {n: _inv(t, space[n][1]) for n, t in zip(names, res.x)}
    return {**fixed, **best}, float(res.fun)
