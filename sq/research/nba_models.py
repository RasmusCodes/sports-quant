"""NBA moneyline model pipeline: tune on 2004-2015, predict every game out-of-sample.

Protocol
--------
* Hyperparameters of the online rating models are fitted on seasons 2004-2015 only
  (2002-2003 warm-up). Everything from 2016 on is out-of-sample.
* Walk-forward ML models (logistic, LightGBM, margin-adjusted Kalman) for season s
  are trained on seasons 2004..s-1 using pre-game features only.
* Two information sets: ``close`` (availability from the last injury report before
  tip - 5 min) and ``early`` (last report before tip - 6 h). Ratings only ever use
  results of earlier calendar days, so they are valid at both times.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import OUT, TIDY_DIR
from ..data import nba
from ..features.players import availability_features
from ..features.schedule import team_schedule_features
from ..models import kalman as K
from ..models.base import run_online
from ..models.elo import Elo
from ..models.glicko import Glicko2
from ..models.ml import walk_forward_gbm, walk_forward_logistic, walk_forward_margin_adjust
from ..models.poisson import PoissonTeam
from ..models.tuning import POS, UNIT, tune

log = logging.getLogger(__name__)
OUTDIR = OUT / "nba"
TRAIN = list(range(2004, 2016))
TEST_FROM = 2016
OFFSETS = (5, 360, 1440)


def load_games() -> pd.DataFrame:
    g = nba.load("games")
    return g[g["season"] >= 2002].reset_index(drop=True)


def tune_all(g: pd.DataFrame, path: Path, force=False) -> dict:
    if path.exists() and not force:
        return json.loads(path.read_text())
    params = {}
    log.info("tuning Elo")
    p, ll = tune(Elo, g, {"k": (20.0, POS), "hca": (100.0, POS), "carry": (0.75, UNIT), "mov_exp": (0.8, POS)},
                 TRAIN, fixed=dict(mean=1505.0, mov_mode="nba538", pts_per_elo=28.0, sd_margin=12.5))
    params["elo_tuned"] = {**p, "train_logloss": ll}
    log.info("elo tuned %s", params["elo_tuned"])
    log.info("tuning Glicko-2")
    p, ll = tune(Glicko2, g, {"hfa": (70.0, POS), "carry": (0.75, UNIT), "sigma0": (0.06, POS),
                              "period_days": (7.0, POS), "rd_season": (150.0, POS)}, TRAIN, maxiter=80)
    params["glicko_tuned"] = {**p, "train_logloss": ll}
    log.info("glicko tuned %s", params["glicko_tuned"])
    log.info("fitting Kalman hyperparameters (marginal likelihood)")
    mask = (g["season"] >= 2003) & (g["season"] <= 2015) & g["home_pts"].notna()
    best, res = K.fit_hyperparameters(g, "nba", mask.to_numpy(), burn_in_seasons=1, maxiter=220)
    params["kalman_fitted"] = {**best, "neg_loglik": float(res.fun)}
    log.info("kalman fitted %s", params["kalman_fitted"])
    log.info("Poisson half-life grid")
    best_hl, best_ll = None, np.inf
    for hl in (30.0, 45.0, 60.0, 90.0, 120.0):
        _, ll = tune(PoissonTeam, g, {}, TRAIN, fixed=dict(half_life=hl), maxiter=1)
        log.info("  half_life %s -> %.5f", hl, ll)
        if ll < best_ll:
            best_hl, best_ll = hl, ll
    params["poisson_tuned"] = {"half_life": best_hl, "train_logloss": best_ll}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2))
    return params


def online_predictions(g: pd.DataFrame, params: dict) -> pd.DataFrame:
    def clean(d):
        return {k: v for k, v in d.items() if k not in ("train_logloss", "neg_loglik")}

    models = {
        "elo538": Elo.preset("nba"),
        "elo": Elo(**clean(params["elo_tuned"])),
        "glicko": Glicko2(**clean(params["glicko_tuned"])),
        "kalman": K.KalmanScores(**clean(params["kalman_fitted"])),
        "poisson": PoissonTeam(**clean(params["poisson_tuned"])),
    }
    base = g[["game_id"]].copy()
    for name, m in models.items():
        p = run_online(m, g)
        p = p.rename(columns={c: f"{name}_{c}" for c in p.columns if c != "game_id"})
        base = base.merge(p, on="game_id", how="left")
        log.info("online model %s done", name)
    return base


def build_features(g: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    sf = team_schedule_features(g, "nba")
    pb = nba.load("player_box")
    inj = nba.load("injuries")
    av = availability_features(g[g["season"] >= 2003], pb[pb["season"] >= 2003], inj, decision_offsets_min=OFFSETS)
    df = g.merge(pred, on="game_id", how="left").merge(sf, on="game_id", how="left").merge(av, on="game_id", how="left")
    df["y"] = np.where(df["home_pts"].notna(), (df["home_pts"] > df["away_pts"]).astype(float), np.nan)
    df["margin"] = df["home_pts"] - df["away_pts"]
    df["b2b_diff"] = df["home_b2b"] - df["away_b2b"]
    df["travel_diff"] = (df["home_travel_km"] - df["away_travel_km"]) / 1000.0
    df["tz_diff"] = df["home_tz_shift"].abs() - df["away_tz_shift"].abs()
    df["dense_diff"] = df["home_games_last4d"] - df["away_games_last4d"]
    df["neutral_f"] = df["neutral"].astype(float)
    df["playoff_f"] = df["playoff"].astype(float)
    for off in OFFSETS:
        df[f"lost_delta_{off}"] = df[f"lost_delta_{off}"].fillna(0.0)
    return df


SCHED = ["rest_diff", "b2b_diff", "travel_diff", "tz_diff", "dense_diff", "home_altitude", "neutral_f", "playoff_f"]


# information sets: name -> minutes before tip at which injury reports are cut off.
# A bet placed at T - x must use an information set whose cut-off is at or before T - x.
INFO_SETS = {"close": 5, "early": 360, "d1": 1440}


def ml_predictions(df: pd.DataFrame, info_sets: dict | None = None) -> pd.DataFrame:
    test = sorted(s for s in df["season"].unique() if s >= 2008)
    d = df[df["season"] >= 2004].copy()
    out = d[["game_id"]].copy()
    for info, off in (info_sets or INFO_SETS).items():
        lost = f"lost_delta_{off}"
        feats_lr = ["elo_elo_diff", "kalman_mu_margin", "poisson_mu_margin", lost] + SCHED
        lr = walk_forward_logistic(d, feats_lr, test, name=f"logit_{info}", first_season=2004)
        feats_gb = feats_lr + ["kalman_sd_margin", "glicko_p_home", "kalman_mu_total", "home_rest", "away_rest",
                               "home_lost_base", "away_lost_base", "home_top_value", "away_top_value",
                               "home_game_no"]
        gb = walk_forward_gbm(d, feats_gb, test, name=f"gbm_{info}", first_season=2004,
                              monotone={"kalman_mu_margin": 1, "elo_elo_diff": 1, lost: -1})
        ka = walk_forward_margin_adjust(d, "kalman_mu_margin", "kalman_sd_margin", [lost] + SCHED[:5], test,
                                        name=f"kalman_plus_{info}", first_season=2004)
        for t in (lr, gb, ka):
            keep = [c for c in t.columns if c.startswith(("p_", "mu_", "sd_"))] + ["game_id"]
            out = out.merge(t[keep], on="game_id", how="left")
        # coefficient trace for the paper
        lr.groupby("season").first().filter(like="coef_").to_csv(OUTDIR / f"logit_{info}_coefs.csv")
        ka.groupby("season").first().filter(like="b_").to_csv(OUTDIR / f"kalman_plus_{info}_coefs.csv")
    return out


def run(force_tune=False) -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    g = load_games()
    params = tune_all(g, OUTDIR / "params.json", force=force_tune)
    pred = online_predictions(g, params)
    df = build_features(g, pred)
    ml = ml_predictions(df)
    df = df.merge(ml, on="game_id", how="left")
    df = add_ensembles(df, INFO_SETS)
    df.to_parquet(OUTDIR / "predictions.parquet", index=False)
    return df


def add_ensembles(df: pd.DataFrame, info_sets) -> pd.DataFrame:
    """Ensemble per information set: mean of the logits of its three walk-forward models."""
    from ..eval.metrics import logit

    for info in info_sets:
        comps = [f"p_logit_{info}", f"p_gbm_{info}", f"p_kalman_plus_{info}"]
        ok = df[comps].notna().all(axis=1)
        df.loc[ok, f"p_ensemble_{info}"] = 1 / (1 + np.exp(-np.mean([logit(df.loc[ok, c]) for c in comps], axis=0)))
    return df


def add_info_sets(names: list[str]) -> pd.DataFrame:
    """Add information sets to saved predictions without re-tuning or re-running online models."""
    df = pd.read_parquet(OUTDIR / "predictions.parquet")
    sets = {k: INFO_SETS[k] for k in names}
    drop = [c for c in df.columns if any(c.endswith(f"_{k}") and c.startswith(("p_", "mu_", "sd_")) for k in sets)]
    df = df.drop(columns=drop)
    df = df.merge(ml_predictions(df, sets), on="game_id", how="left")
    df = add_ensembles(df, sets)
    df.to_parquet(OUTDIR / "predictions.parquet", index=False)
    return df
