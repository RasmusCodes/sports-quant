"""Market-efficiency studies (NBA exchange prices; NFL sportsbook closes).

Each test asks whether information that is public before the decision time is
fully reflected in the price. The common form is a logistic regression with the
market's own log-odds as an offset:

    logit P(y=1) = logit(p_market) + b0 + b1 * x

Under efficiency b0 = b1 = 0. All p-values are collected and adjusted with
Benjamini-Hochberg because many hypotheses are tested on the same games.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..config import OUT
from ..eval.metrics import benjamini_hochberg, logit

log = logging.getLogger(__name__)


def offset_logit(df: pd.DataFrame, pcol: str, xcols: list[str], y="y", cluster="date", label=""):
    d = df.dropna(subset=[pcol, y] + xcols)
    X = sm.add_constant(d[xcols].astype(float)) if xcols else np.ones((len(d), 1))
    mod = sm.GLM(d[y].astype(float), X, family=sm.families.Binomial(), offset=logit(d[pcol]))
    groups = pd.factorize(d[cluster])[0]
    r = mod.fit(cov_type="cluster", cov_kwds={"groups": groups})
    names = ["const"] + xcols
    rows = []
    for i, n in enumerate(names):
        rows.append({"test": label, "market": pcol, "term": n, "coef": float(r.params.iloc[i]),
                     "se": float(r.bse.iloc[i]), "p_value": float(r.pvalues.iloc[i]), "n": int(len(d))})
    return rows


def ols(df, ycol, xcols, cluster="date", label=""):
    d = df.dropna(subset=[ycol] + xcols)
    X = sm.add_constant(d[xcols].astype(float))
    r = sm.OLS(d[ycol].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    return [{"test": label, "market": ycol, "term": n, "coef": float(r.params.iloc[i]), "se": float(r.bse.iloc[i]),
             "p_value": float(r.pvalues.iloc[i]), "n": int(len(d)), "r2": float(r.rsquared)}
            for i, n in enumerate(["const"] + xcols)]


def nba_studies() -> dict:
    outdir = OUT / "nba"
    m = pd.read_parquet(outdir / "market_panel.parquet")
    m = m[m["y"].notna()].copy()
    rows = []
    res = {}
    for venue, pc, p24, popen in (("kalshi", "k_p_close", "k_p_t24h", "k_p_open"),
                                  ("polymarket", "pm_p_close", "pm_p_t24h", "pm_p_open")):
        d = m.dropna(subset=[pc]).copy()
        d["move_24h"] = logit(d[pc]) - logit(d[p24])
        d["move_open"] = logit(d[pc]) - logit(d[popen])
        # 1. calibration-in-the-large / home bias at the close
        rows += offset_logit(d, pc, [], label=f"{venue}: home bias at close")
        # 2. does late line movement over- or under-shoot?
        rows += offset_logit(d, pc, ["move_24h"], label=f"{venue}: movement T-24h->close")
        rows += offset_logit(d, pc, ["move_open"], label=f"{venue}: movement open->close")
        # 3. schedule and fatigue information left in the price
        rows += offset_logit(d, pc, ["b2b_diff", "rest_diff", "travel_diff", "tz_diff", "home_altitude"],
                             label=f"{venue}: schedule/fatigue at close")
        # 4. availability: point-in-time injury information left in the close
        rows += offset_logit(d, pc, ["lost_delta_5"], label=f"{venue}: availability at close")
        # 5. late injury news (T-24h -> T-5m) left in the close
        d["late_news"] = d["lost_delta_5"] - d["lost_delta_1440"]
        rows += offset_logit(d, pc, ["late_news"], label=f"{venue}: late injury news vs close")
        # repricing regression: how much did the price move per unit of late injury news?
        rows += ols(d, "move_24h", ["late_news"], label=f"{venue}: price move on late injury news")
        # 6. model probability beyond the price (information test)
        d["model_edge"] = logit(d["p_ensemble_close"]) - logit(d[pc])
        rows += offset_logit(d, pc, ["model_edge"], label=f"{venue}: ensemble edge vs close")
        d["elo_edge"] = logit(d["elo538_p_home"]) - logit(d[pc])
        rows += offset_logit(d, pc, ["elo_edge"], label=f"{venue}: Elo edge vs close")
        # 7. drift: does the T-24h model-vs-price gap predict where the price goes by the close?
        #    (descriptive; uses only the T-24h information set, so it is a fair ex-ante signal)
        if "p_ensemble_d1" in d:
            d["gap_24h"] = logit(d["p_ensemble_d1"]) - logit(d[p24])
            rows += ols(d, "move_24h", ["gap_24h"], label=f"{venue}: price move toward T-24h model gap")
        res[f"{venue}_late_news_share_nonzero"] = float((d["late_news"].abs() > 0.5).mean())
    t = _bh_family(pd.DataFrame(rows))
    t.to_csv(outdir / "efficiency_tests.csv", index=False)

    # outcome-side effect sizes (training seasons, no market): what rest/injuries are worth in the data
    pred = pd.read_parquet(outdir / "predictions.parquet")
    tr = pred[(pred["season"] >= 2004) & (pred["season"] <= 2023) & pred["margin"].notna()].copy()
    tr["resid"] = tr["margin"] - tr["kalman_mu_margin"]
    eff = ols(tr.assign(date=tr["date"]), "resid", ["b2b_diff", "rest_diff", "travel_diff", "tz_diff",
                                                    "home_altitude", "lost_delta_5"], label="margin residual on info (2004-2023)")
    pd.DataFrame(eff).to_csv(outdir / "effect_sizes.csv", index=False)
    res["n_tests"] = int(len(t[t["term"] != "const"]) + len(t[(t["term"] == "const") & t["test"].str.contains("home bias")]))
    (outdir / "studies.json").write_text(json.dumps(res, indent=2))
    return res


def nfl_studies() -> dict:
    outdir = OUT / "nfl"
    from .nfl_report import add_book

    df = add_book(pd.read_parquet(outdir / "predictions.parquet"))
    d = df[(df["season"] >= 2010) & df["y"].notna() & df["book_shin"].notna()].copy()
    rows = []
    rows += offset_logit(d, "book_shin", [], label="book: home bias at close")
    rows += offset_logit(d, "book_shin", ["div_f", "playoff_f", "neutral_f"], label="book: game context")
    rows += offset_logit(d, "book_shin", ["rest_diff_nfl", "travel_diff", "tz_diff"], label="book: rest/travel")
    rows += offset_logit(d, "book_shin", ["qb_delta_diff", "home_qb_change", "away_qb_change"],
                         label="book: QB change")
    d["model_edge"] = logit(d["p_ensemble"]) - logit(d["book_shin"])
    rows += offset_logit(d, "book_shin", ["model_edge"], label="book: ensemble edge vs close")
    # spread market: cover regressions on the devigged cover probability
    sp = pd.read_parquet(outdir / "spread_panel.parquet")
    sp["fav_home"] = (sp["home_spread"] < 0).astype(float)
    rows += offset_logit(sp, "book_p_cover", ["fav_home"], y="cover", label="spread: home favourite vs dog")
    t = _bh_family(pd.DataFrame(rows))
    t.to_csv(outdir / "efficiency_tests.csv", index=False)
    return {"n": int(len(d))}


def _bh_family(t: pd.DataFrame) -> pd.DataFrame:
    """Benjamini-Hochberg over the efficiency hypotheses only: every non-constant term of an
    offset-logit test plus the constant of the home-bias tests. Descriptive regressions
    (price moves on news) and nuisance intercepts are reported unadjusted."""
    fam = ((t["term"] != "const") | t["test"].str.contains("home bias")) & ~t["test"].str.contains("price move")
    t["in_family"] = fam
    t["p_bh"] = np.nan
    t["reject_bh_5pct"] = False
    adj, rej = benjamini_hochberg(t.loc[fam, "p_value"].to_numpy())
    t.loc[fam, "p_bh"] = adj
    t.loc[fam, "reject_bh_5pct"] = rej
    return t
