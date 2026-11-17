"""NBA moneyline benchmark: long-run model comparison and model-vs-market tests.

Writes CSV/JSON tables to outputs/nba/ (figures are drawn by sq.research.figures).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..config import OUT
from . import market_eval as ME

log = logging.getLogger(__name__)
OUTDIR = OUT / "nba"

MODELS = {
    "Elo (538 params)": "elo538_p_home",
    "Elo (tuned)": "elo_p_home",
    "Glicko-2": "glicko_p_home",
    "Poisson (normal margin)": "poisson_p_home",
    "Poisson (Skellam)": "poisson_p_home_skellam",
    "Bayesian state-space": "kalman_p_home",
    "Bayes + players/rest": "p_kalman_plus_close",
    "Logistic": "p_logit_close",
    "Gradient boosting": "p_gbm_close",
    "Ensemble": "p_ensemble_close",
}
MODELS_EARLY = {
    "Bayes + players/rest (T-6h info)": "p_kalman_plus_early",
    "Logistic (T-6h info)": "p_logit_early",
    "Gradient boosting (T-6h info)": "p_gbm_early",
    "Ensemble (T-6h info)": "p_ensemble_early",
    "Ensemble (T-24h info)": "p_ensemble_d1",
}
KALSHI = {"Kalshi close": "k_p_close", "Kalshi T-15m": "k_p_t15m", "Kalshi T-1h": "k_p_t1h",
          "Kalshi T-6h": "k_p_t6h", "Kalshi T-24h": "k_p_t24h", "Kalshi open": "k_p_open"}
POLY = {"Polymarket close": "pm_p_close", "Polymarket T-15m": "pm_p_t15m", "Polymarket T-1h": "pm_p_t1h",
        "Polymarket T-6h": "pm_p_t6h", "Polymarket T-24h": "pm_p_t24h", "Polymarket open": "pm_p_open"}


def _season_metrics(df, cols):
    rows = []
    for s, d in df.groupby("season"):
        for label, c in cols.items():
            ok = d[c].notna() & d["y"].notna()
            if ok.sum() < 50:
                continue
            from ..eval.metrics import brier, log_loss

            rows.append({"season": s, "model": label, "n": int(ok.sum()), "log_loss": log_loss(d.loc[ok, c], d.loc[ok, "y"]),
                         "brier": brier(d.loc[ok, c], d.loc[ok, "y"])})
    return pd.DataFrame(rows)


def run(df: pd.DataFrame | None = None) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.read_parquet(OUTDIR / "predictions.parquet")
    df = df[df["y"].notna()].copy()
    res = {}

    # ---------------- long run, out-of-sample seasons
    lr = df[df["season"] >= 2016].copy()
    lr_cols = {k: v for k, v in MODELS.items() if v in lr}
    t = ME.score_table(lr.dropna(subset=list(lr_cols.values())), lr_cols)
    t.to_csv(OUTDIR / "longrun_metrics.csv", index=False)
    res["longrun_n"] = int(t["n"].iloc[0]) if len(t) else 0
    _season_metrics(df[df["season"] >= 2008], lr_cols).to_csv(OUTDIR / "season_metrics.csv", index=False)
    common = lr.dropna(subset=list(lr_cols.values()))
    ME.paired_table(common, lr_cols, "elo538_p_home").to_csv(OUTDIR / "longrun_paired_vs_elo538.csv", index=False)
    ME.calibration_curves(common, lr_cols).to_csv(OUTDIR / "longrun_calibration.csv", index=False)

    # ---------------- market period
    mk = ME.market_wide("nba")
    m = df.merge(mk, on="game_id", how="inner")
    m = ME.add_devig(m, "k_q_home_close", "k_q_away_close", "k_devig")
    m.to_parquet(OUTDIR / "market_panel.parquet", index=False)
    allcols = {**MODELS, **MODELS_EARLY, **KALSHI, **POLY}
    for name, req in (("kalshi", ["k_p_close"]), ("polymarket", ["pm_p_close"]),
                      ("common", ["k_p_close", "pm_p_close"])):
        s = m.dropna(subset=req + ["p_ensemble_close"])
        cols = {k: v for k, v in allcols.items() if v in s}
        ME.score_table(s, cols).to_csv(OUTDIR / f"market_{name}_metrics.csv", index=False)
        ref = req[0]
        ME.paired_table(s, cols, ref).to_csv(OUTDIR / f"market_{name}_paired_vs_close.csv", index=False)
        ME.encompassing_table(s, ref, {**MODELS, **MODELS_EARLY}).to_csv(
            OUTDIR / f"market_{name}_encompassing.csv", index=False)
        ME.calibration_curves(s, {k: cols[k] for k in cols if k in ("Kalshi close", "Polymarket close",
                                                                     "Kalshi open", "Polymarket open", "Ensemble",
                                                                     "Elo (538 params)", "Bayesian state-space")}
                              ).to_csv(OUTDIR / f"market_{name}_calibration.csv", index=False)
        res[f"market_{name}_n"] = int(len(s))
        res[f"market_{name}_seasons"] = sorted(int(x) for x in s["season"].unique())
        res[f"market_{name}_dates"] = [str(s["date"].min().date()), str(s["date"].max().date())]

    # ---------------- devig methods on Kalshi closing asks (+ mid)
    s = m.dropna(subset=["k_q_home_close", "k_q_away_close"])
    dcols = {"mid": "k_p_close", **{meth: f"k_devig_{meth}" for meth in ("multiplicative", "additive", "power",
                                                                          "shin", "odds_ratio")}}
    ME.score_table(s, dcols).to_csv(OUTDIR / "devig_kalshi_close.csv", index=False)
    res["kalshi_close_overround_mean"] = float(s["k_overround_close"].mean())
    res["kalshi_open_overround_mean"] = float(m["k_overround_open"].mean())

    # ---------------- favourite-longshot
    for c, label in (("k_p_close", "kalshi_close"), ("pm_p_close", "poly_close"), ("k_p_open", "kalshi_open")):
        fl = ME.favourite_longshot(m, c)
        fl["bands"].to_csv(OUTDIR / f"flb_{label}.csv", index=False)
        res[f"flb_{label}"] = {k: v for k, v in fl.items() if k != "bands"}

    # ---------------- cross-venue disagreement
    both = m.dropna(subset=["k_p_close", "pm_p_close"])
    d = both["k_p_close"] - both["pm_p_close"]
    res["venue_gap"] = {"n": int(len(both)), "mean": float(d.mean()), "mean_abs": float(d.abs().mean()),
                        "p90_abs": float(d.abs().quantile(0.9)), "corr": float(np.corrcoef(both["k_p_close"], both["pm_p_close"])[0, 1])}
    both = both.assign(avg=(both["k_p_close"] + both["pm_p_close"]) / 2)
    ME.score_table(both, {"Kalshi close": "k_p_close", "Polymarket close": "pm_p_close", "Venue average": "avg"}
                   ).to_csv(OUTDIR / "venue_comparison.csv", index=False)
    (OUTDIR / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    return res
