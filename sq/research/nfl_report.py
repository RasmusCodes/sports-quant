"""NFL evaluation: models vs sportsbook closing lines (moneyline, spread, total) and exchanges."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ..config import OUT
from ..eval import metrics as M
from ..markets import devig as DV
from ..markets.odds import american_to_prob, overround
from ..models.simulate import simulate_nfl
from . import market_eval as ME

log = logging.getLogger(__name__)
OUTDIR = OUT / "nfl"

MODELS = {"Elo (538 params)": "elo538_p_home", "Elo (tuned)": "elo_p_home", "Glicko-2": "glicko_p_home",
          "Poisson (normal margin)": "poisson_p_home", "Bayesian state-space": "kalman_p_home",
          "Bayes + QB/rest": "p_kalman_plus", "Logistic": "p_logit", "Gradient boosting": "p_gbm",
          "Ensemble": "p_ensemble"}


def add_book(df: pd.DataFrame) -> pd.DataFrame:
    qh, qa = american_to_prob(df["home_ml"]), american_to_prob(df["away_ml"])
    df["book_q_home"], df["book_q_away"] = qh, qa
    df["book_overround"] = overround(qh, qa)
    q = np.column_stack([qh, qa])
    for m in DV.METHODS:
        df[f"book_{m}"] = DV.devig(q, m)[:, 0]
    # spread and total markets (devig the two prices, multiplicative)
    sh, sa = american_to_prob(df["home_spread_odds"]), american_to_prob(df["away_spread_odds"])
    df["book_p_cover"] = DV.devig(np.column_stack([sh, sa]), "multiplicative")[:, 0]
    df["book_spread_overround"] = overround(sh, sa)
    ov, un = american_to_prob(df["over_odds"]), american_to_prob(df["under_odds"])
    df["book_p_over"] = DV.devig(np.column_stack([ov, un]), "multiplicative")[:, 0]
    df["book_total_overround"] = overround(ov, un)
    return df


def run(df: pd.DataFrame | None = None, n_sims=20000) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.read_parquet(OUTDIR / "predictions.parquet")
    df = add_book(df.copy())
    res = {}
    played = df[df["home_pts"].notna()].copy()
    oos = played[played["season"] >= 2013]

    # ---------- long run (ties excluded from moneyline scoring)
    cols = {k: v for k, v in MODELS.items() if v in oos}
    s = oos.dropna(subset=list(cols.values()) + ["y"])
    ME.score_table(s, cols).to_csv(OUTDIR / "longrun_metrics.csv", index=False)
    ME.paired_table(s, cols, "elo538_p_home").to_csv(OUTDIR / "longrun_paired_vs_elo538.csv", index=False)
    res["longrun_n"] = int(len(s))

    # ---------- sportsbook closing moneyline
    b = oos.dropna(subset=["book_q_home", "book_q_away", "y"] + list(cols.values()))
    book_cols = {f"Book close ({m})": f"book_{m}" for m in DV.METHODS}
    ME.score_table(b, {**book_cols, **cols}).to_csv(OUTDIR / "book_ml_metrics.csv", index=False)
    ME.paired_table(b, {**cols, **book_cols}, "book_shin").to_csv(OUTDIR / "book_ml_paired_vs_shin.csv", index=False)
    ME.encompassing_table(b, "book_shin", cols).to_csv(OUTDIR / "book_ml_encompassing.csv", index=False)
    ME.calibration_curves(b, {"Book close (Shin)": "book_shin", "Ensemble": "p_ensemble",
                              "Elo (538 params)": "elo538_p_home"}).to_csv(OUTDIR / "book_ml_calibration.csv", index=False)
    fl = ME.favourite_longshot(b, "book_multiplicative")
    fl["bands"].to_csv(OUTDIR / "flb_book_multiplicative.csv", index=False)
    res["flb_book"] = {k: v for k, v in fl.items() if k != "bands"}
    res["book_ml_n"] = int(len(b))
    res["book_ml_seasons"] = [int(b["season"].min()), int(b["season"].max())]
    res["book_overround_mean"] = float(b["book_overround"].mean())

    # ---------- spread market: devigged cover probability vs model (normal and drive simulation)
    sp = oos.dropna(subset=["home_spread", "book_p_cover", "kalman_mu_margin"]).copy()
    sp["cover_margin"] = sp["margin"] + sp["home_spread"]
    sp = sp[sp["cover_margin"] != 0]
    sp["cover"] = (sp["cover_margin"] > 0).astype(float)
    mu = sp["p_kalman_plus"].notna()
    mu_col = np.where(mu, sp.get("mu_kalman_plus", sp["kalman_mu_margin"]), sp["kalman_mu_margin"])
    sd_col = np.where(mu, sp.get("sd_kalman_plus", sp["kalman_sd_margin"]), sp["kalman_sd_margin"])
    sp["model_p_cover_normal"] = ndtr((mu_col + sp["home_spread"]) / sd_col)
    # drive simulation: split the model's expected margin/total into team means
    sims_cover, key3, key7 = [], [], []
    for r in sp.itertuples(index=False):
        mt = r.kalman_mu_total if np.isfinite(r.kalman_mu_total) else 44.0
        mh, ma = (mt + r.kalman_mu_margin) / 2, (mt - r.kalman_mu_margin) / 2
        sim = simulate_nfl(max(mh, 3.0), max(ma, 3.0), n_sims=n_sims, seed=hash(r.game_id) % 2 ** 32)
        m = sim["home"] - sim["away"]
        cm = m + r.home_spread
        decided = cm != 0
        sims_cover.append(float(np.mean(cm[decided] > 0)) if decided.any() else np.nan)
        key3.append(float(np.mean(np.abs(m) == 3)))
        key7.append(float(np.mean(np.abs(m) == 7)))
    sp["model_p_cover_sim"] = sims_cover
    ME.score_table(sp, {"Book spread (devig)": "book_p_cover", "Model, normal margin": "model_p_cover_normal",
                        "Model, drive simulation": "model_p_cover_sim"}, y="cover").to_csv(
        OUTDIR / "spread_metrics.csv", index=False)
    res["spread_n"] = int(len(sp))
    res["spread_cover_rate_home"] = float(sp["cover"].mean())
    res["spread_book_mean_p"] = float(sp["book_p_cover"].mean())
    fav = np.where(sp["home_spread"] < 0, sp["cover"], 1 - sp["cover"])
    res["spread_favourite_cover_rate"] = float(np.mean(fav[sp["home_spread"] != 0]))
    actual = sp["margin"].abs()
    res["key_numbers"] = {"actual_share_3": float(np.mean(actual == 3)), "actual_share_7": float(np.mean(actual == 7)),
                          "sim_share_3": float(np.mean(key3)), "sim_share_7": float(np.mean(key7))}
    zz = (sp["margin"] - sp["kalman_mu_margin"]) / sp["kalman_sd_margin"]
    res["normal_share_3"] = float(np.mean(ndtr((3.5 - sp["kalman_mu_margin"]) / sp["kalman_sd_margin"])
                                          - ndtr((2.5 - sp["kalman_mu_margin"]) / sp["kalman_sd_margin"])
                                          + ndtr((-2.5 - sp["kalman_mu_margin"]) / sp["kalman_sd_margin"])
                                          - ndtr((-3.5 - sp["kalman_mu_margin"]) / sp["kalman_sd_margin"])))
    res["margin_z_sd"] = float(zz.std())

    # ---------- totals
    tt = oos.dropna(subset=["total_line", "book_p_over", "kalman_mu_total"]).copy()
    tt = tt[tt["total_pts"] != tt["total_line"]]
    tt["over"] = (tt["total_pts"] > tt["total_line"]).astype(float)
    tt["model_p_over"] = 1 - ndtr((tt["total_line"] - tt["kalman_mu_total"]) / tt["kalman_sd_total"])
    ME.score_table(tt, {"Book total (devig)": "book_p_over", "Model (state-space)": "model_p_over"}, y="over").to_csv(
        OUTDIR / "total_metrics.csv", index=False)
    res["total_n"] = int(len(tt))
    res["total_over_rate"] = float(tt["over"].mean())
    wind = tt["wind"].fillna(0) >= 15
    res["total_over_rate_wind15"] = float(tt.loc[wind, "over"].mean()) if wind.any() else None
    res["total_book_p_over_wind15"] = float(tt.loc[wind, "book_p_over"].mean()) if wind.any() else None
    res["total_n_wind15"] = int(wind.sum())

    # ---------- exchanges vs sportsbook on the same games
    mk = ME.market_wide("nfl")
    x = played.merge(mk, on="game_id", how="inner")
    x = x.dropna(subset=["y"])
    xc = {"Book close (Shin)": "book_shin", "Kalshi close": "k_p_close", "Polymarket close": "pm_p_close",
          "Kalshi open": "k_p_open", "Ensemble": "p_ensemble", "Elo (538 params)": "elo538_p_home"}
    common = x.dropna(subset=["book_shin", "k_p_close", "pm_p_close"])
    ME.score_table(common, xc).to_csv(OUTDIR / "exchange_vs_book.csv", index=False)
    ME.paired_table(common, xc, "book_shin").to_csv(OUTDIR / "exchange_vs_book_paired.csv", index=False)
    res["exchange_common_n"] = int(len(common))
    if len(common):
        res["kalshi_minus_book_mean_abs"] = float((common["k_p_close"] - common["book_shin"]).abs().mean())
    x.to_parquet(OUTDIR / "market_panel.parquet", index=False)
    sp.to_parquet(OUTDIR / "spread_panel.parquet", index=False)
    (OUTDIR / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    return res
