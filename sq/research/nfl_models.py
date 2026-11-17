"""NFL pipeline: online models tuned on 2002-2012, out-of-sample from 2013.

Markets: nflverse sportsbook closing moneyline / spread / total (2006-2026) and
Kalshi / Polymarket game markets (2024-2026). The starting QB (announced before
kickoff) enters through ``sq.features.nfl_qb``.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..config import OUT
from ..data import nfl
from ..features.nfl_qb import load_qb_weeks, qb_features
from ..features.schedule import team_schedule_features
from ..models import kalman as K
from ..models.base import run_online
from ..models.elo import Elo
from ..models.glicko import Glicko2
from ..models.ml import walk_forward_gbm, walk_forward_logistic, walk_forward_margin_adjust
from ..models.poisson import PoissonTeam
from ..models.tuning import POS, UNIT, tune

log = logging.getLogger(__name__)
OUTDIR = OUT / "nfl"
TRAIN = list(range(2002, 2013))


def load_games() -> pd.DataFrame:
    g = nfl.load("games")
    return g[g["season"] >= 1999].reset_index(drop=True)


def tune_all(g, path, force=False):
    if path.exists() and not force:
        return json.loads(path.read_text())
    params = {}
    p, ll = tune(Elo, g, {"k": (20.0, POS), "hca": (48.0, POS), "carry": (0.67, UNIT)}, TRAIN,
                 fixed=dict(mean=1505.0, mov_mode="nfl538", pts_per_elo=25.0, sd_margin=13.5))
    params["elo_tuned"] = {**p, "train_logloss": ll}
    p, ll = tune(Glicko2, g, {"hfa": (50.0, POS), "carry": (0.7, UNIT), "sigma0": (0.06, POS),
                              "period_days": (7.0, POS)}, TRAIN, maxiter=80, fixed=dict(rd_season=150.0))
    params["glicko_tuned"] = {**p, "train_logloss": ll}
    mask = (g["season"] >= 2001) & (g["season"] <= 2012) & g["home_pts"].notna()
    best, res = K.fit_hyperparameters(g, "nfl", mask.to_numpy(), burn_in_seasons=1, maxiter=220)
    params["kalman_fitted"] = {**best, "neg_loglik": float(res.fun)}
    best_hl, best_ll = None, np.inf
    for hl in (60.0, 120.0, 240.0, 365.0):
        _, ll = tune(PoissonTeam, g, {}, TRAIN, fixed=dict(half_life=hl, window=800, sd0=13.5, min_games=40))
        if ll < best_ll:
            best_hl, best_ll = hl, ll
    params["poisson_tuned"] = {"half_life": best_hl, "window": 800, "sd0": 13.5, "min_games": 40,
                               "train_logloss": best_ll}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2))
    return params


def run(force_tune=False) -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    g = load_games()
    params = tune_all(g, OUTDIR / "params.json", force_tune)

    def clean(d):
        return {k: v for k, v in d.items() if k not in ("train_logloss", "neg_loglik")}

    models = {"elo538": Elo.preset("nfl"), "elo": Elo(**clean(params["elo_tuned"])),
              "glicko": Glicko2(**clean(params["glicko_tuned"])),
              "kalman": K.KalmanScores(**clean(params["kalman_fitted"])),
              "poisson": PoissonTeam(**clean(params["poisson_tuned"]))}
    pred = g[["game_id"]].copy()
    for name, m in models.items():
        p = run_online(m, g)
        pred = pred.merge(p.rename(columns={c: f"{name}_{c}" for c in p.columns if c != "game_id"}), on="game_id")
        log.info("nfl online model %s done", name)
    sf = team_schedule_features(g, "nfl")
    # nflverse already carries exact rest days; keep only travel / time-zone / density from the schedule builder
    sf = sf.drop(columns=["home_rest", "away_rest", "home_b2b", "away_b2b", "rest_diff"], errors="ignore")
    qb = qb_features(g, load_qb_weeks())
    df = g.merge(pred, on="game_id").merge(sf, on="game_id", how="left").merge(qb, on="game_id", how="left")
    df["y"] = np.where(df["home_pts"].notna(), (df["home_pts"] > df["away_pts"]).astype(float), np.nan)
    df.loc[df["home_pts"] == df["away_pts"], "y"] = np.nan          # ties excluded from win-probability scoring
    df["margin"] = df["home_pts"] - df["away_pts"]
    df["total_pts"] = df["home_pts"] + df["away_pts"]
    df["rest_diff_nfl"] = (df["home_rest"] - df["away_rest"]).clip(-7, 7)
    df["travel_diff"] = (df["home_travel_km"] - df["away_travel_km"]) / 1000.0
    df["tz_diff"] = df["home_tz_shift"].abs() - df["away_tz_shift"].abs()
    df["neutral_f"] = df["neutral"].astype(float)
    df["div_f"] = df["div_game"].astype(float)
    df["playoff_f"] = df["playoff"].astype(float)
    test = list(range(2008, 2027))
    d = df[df["season"] >= 2002]
    feats = ["elo_elo_diff", "kalman_mu_margin", "qb_val_diff", "qb_delta_diff", "rest_diff_nfl", "travel_diff",
             "tz_diff", "neutral_f", "div_f", "playoff_f"]
    lr = walk_forward_logistic(d, feats, test, name="logit", first_season=2002)
    gb = walk_forward_gbm(d, feats + ["glicko_p_home", "kalman_sd_margin", "home_qb_change", "away_qb_change"],
                          test, name="gbm", first_season=2002, params=dict(min_child_samples=80, num_leaves=7),
                          monotone={"kalman_mu_margin": 1, "elo_elo_diff": 1, "qb_val_diff": 1})
    ka = walk_forward_margin_adjust(d, "kalman_mu_margin", "kalman_sd_margin",
                                    ["qb_delta_diff", "rest_diff_nfl", "tz_diff"], test, name="kalman_plus",
                                    first_season=2002)
    for t in (lr, gb, ka):
        keep = [c for c in t.columns if c.startswith(("p_", "mu_", "sd_"))] + ["game_id"]
        df = df.merge(t[keep], on="game_id", how="left")
    lr.groupby("season").first().filter(like="coef_").to_csv(OUTDIR / "logit_coefs.csv")
    ka.groupby("season").first().filter(like="b_").to_csv(OUTDIR / "kalman_plus_coefs.csv")
    from ..eval.metrics import logit

    comps = ["p_logit", "p_gbm", "p_kalman_plus"]
    ok = df[comps].notna().all(axis=1)
    df.loc[ok, "p_ensemble"] = 1 / (1 + np.exp(-np.mean([logit(df.loc[ok, c]) for c in comps], axis=0)))
    df.to_parquet(OUTDIR / "predictions.parquet", index=False)
    return df
