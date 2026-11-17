"""Out-of-sample betting simulations on exchange prices.

The strategy is fixed before any result is looked at (pre-registered):
quarter Kelly, minimum edge 2 points after all costs, at most 2% of bankroll per
bet and 15% per day, Kalshi taker fees, executable asks only, bet sized off the
start-of-day bankroll. Model probabilities are walk-forward out-of-sample. A
sensitivity grid is reported alongside, but nothing in it is used to choose the
headline configuration.
"""
from __future__ import annotations

import itertools
import json
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from ..backtest.engine import Config, run
from ..config import OUT

log = logging.getLogger(__name__)
PRE_REGISTERED = Config(kelly_fraction=0.25, min_edge=0.02, max_bet_frac=0.02, max_day_frac=0.15,
                        fee_model="kalshi", slippage=0.0, liquidity_frac=0.02)


def nba_universe(pred: pd.DataFrame, panel: pd.DataFrame, decision: str, venue: str, model: str,
                 poly_half_spread=0.01) -> pd.DataFrame:
    d = panel.copy()
    if venue == "kalshi":
        qh, qa = (f"k_q_home_{decision}", f"k_q_away_{decision}") if decision != "close" else (
            "k_q_home_close", "k_q_away_close")
        out = pd.DataFrame({"game_id": d["game_id"], "date": d["date"], "q_home": d[qh], "q_away": d[qa],
                            "p_close": d["k_p_close"], "volume": d.get("k_volume_pre_tip")})
    else:
        pcol = f"pm_p_{decision}" if decision != "close" else "pm_p_close"
        out = pd.DataFrame({"game_id": d["game_id"], "date": d["date"], "q_home": d[pcol] + poly_half_spread,
                            "q_away": 1 - d[pcol] + poly_half_spread, "p_close": d["pm_p_close"],
                            "volume": d.get("pm_volume_total")})
    out["p_model"] = d[model].to_numpy()
    out["y"] = d["y"].to_numpy()
    out["season"] = d["season"].to_numpy()
    return out


def run_grid(universe: pd.DataFrame, base: Config, label: str) -> pd.DataFrame:
    rows = []
    grid = itertools.product([0.1, 0.25, 0.5], [0.01, 0.02, 0.03, 0.05], [0.0, 0.01])
    for kf, me, sl in grid:
        cfg = replace(base, kelly_fraction=kf, min_edge=me, slippage=sl)
        _, _, s = run(universe, cfg)
        rows.append({"label": label, "kelly": kf, "min_edge": me, "slippage": sl,
                     **{k: v for k, v in s.items() if k != "config"}})
    return pd.DataFrame(rows)


def combined_probability(panel: pd.DataFrame, model_col: str, mkt_col: str, min_train: int = 300) -> np.ndarray:
    """Walk-forward forecast combination: logit P(y) = a + b_m logit(p_mkt) + b_x logit(p_model),
    refitted every game day on all earlier games only (expanding window, at least ``min_train``).

    This is the decision rule implied by the encompassing test: if the market already contains
    the model's information, b_x ~ 0 and the combined probability collapses onto the price.
    """
    import statsmodels.api as sm

    from ..eval.metrics import logit

    d = panel.sort_values(["date", "game_id"])
    out = pd.Series(np.nan, index=d.index)
    ok_all = d[model_col].notna() & d[mkt_col].notna()
    for date in d["date"].unique():
        tr = d[(d["date"] < date) & ok_all & d["y"].notna()]
        te = d[(d["date"] == date) & ok_all]
        if len(tr) < min_train or te.empty:
            continue
        X = sm.add_constant(np.column_stack([logit(tr[mkt_col]), logit(tr[model_col])]))
        r = sm.GLM(tr["y"].astype(float), X, family=sm.families.Binomial()).fit()
        Xt = sm.add_constant(np.column_stack([logit(te[mkt_col]), logit(te[model_col])]), has_constant="add")
        out.loc[te.index] = 1 / (1 + np.exp(-(Xt @ r.params.to_numpy())))
    return out.reindex(panel.index).to_numpy()


def nba_backtests() -> dict:
    outdir = OUT / "nba"
    panel = pd.read_parquet(outdir / "market_panel.parquet")
    panel = panel[panel["y"].notna()].copy()
    # market-aware decision rule (walk-forward combination of model and T-6h mid)
    panel["p_combined_early"] = combined_probability(panel, "p_ensemble_early", "k_p_t6h")
    res, grids, ledgers, paths = {}, [], {}, {}
    # each decision time is paired with the information set available at that time:
    # T-24h -> injury reports up to T-24h (d1), T-6h -> up to T-6h (early), close -> up to T-5m (close)
    specs = [
        ("kalshi", "t6h", "p_ensemble_early", PRE_REGISTERED),
        ("kalshi", "t6h", "elo538_p_home", PRE_REGISTERED),
        ("kalshi", "t6h", "p_kalman_plus_early", PRE_REGISTERED),
        ("kalshi", "t24h", "p_ensemble_d1", PRE_REGISTERED),
        ("kalshi", "close", "p_ensemble_close", PRE_REGISTERED),
        ("kalshi", "t6h", "p_combined_early", PRE_REGISTERED),
        # deliberately leaky control: T-6h injury information used for a T-24h bet (look-ahead of up to 18h)
        ("kalshi", "t24h", "p_ensemble_early", PRE_REGISTERED),
        ("polymarket", "t6h", "p_ensemble_early", replace(PRE_REGISTERED, fee_model="polymarket", liquidity_frac=None)),
        ("polymarket", "close", "p_ensemble_close", replace(PRE_REGISTERED, fee_model="polymarket", liquidity_frac=None)),
    ]
    for venue, decision, model, cfg in specs:
        u = nba_universe(None, panel, decision, venue, model)
        label = f"{venue}|{decision}|{model}"
        led, dd, s = run(u, cfg)
        res[label] = {k: v for k, v in s.items() if k != "config"}
        res[label]["universe_games"] = int(u.dropna(subset=["q_home", "q_away", "p_model"]).shape[0])
        by = led.merge(u[["game_id", "season"]], on="game_id").groupby("season").agg(
            n=("pnl", "size"), staked=("stake", "sum"), pnl=("pnl", "sum"), clv=("clv", "mean")) if len(led) else None
        if by is not None:
            by["roi"] = by["pnl"] / by["staked"]
            res[label]["by_season"] = by.reset_index().to_dict(orient="records")
        ledgers[label] = led
        paths[label] = dd
        grids.append(run_grid(u, cfg, label))
        log.info("%s: %s", label, {k: res[label].get(k) for k in ("n_bets", "roi", "avg_clv", "p_roi_le_0")})
    pd.concat(grids).to_csv(outdir / "backtest_grid.csv", index=False)
    pd.concat([l.assign(strategy=k) for k, l in ledgers.items() if len(l)]).to_parquet(outdir / "backtest_ledger.parquet")
    pd.concat([p.assign(strategy=k) for k, p in paths.items() if len(p)]).to_parquet(outdir / "backtest_paths.parquet")
    (outdir / "backtests.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def nfl_backtests() -> dict:
    outdir = OUT / "nfl"
    x = pd.read_parquet(outdir / "market_panel.parquet")
    pred = pd.read_parquet(outdir / "predictions.parquet")
    res, grids = {}, []
    # Kalshi at T-24h and T-6h (starting QB as announced; see paper for the caveat)
    for decision in ("t24h", "t6h", "close"):
        qh = f"k_q_home_{decision}" if decision != "close" else "k_q_home_close"
        qa = f"k_q_away_{decision}" if decision != "close" else "k_q_away_close"
        u = pd.DataFrame({"game_id": x["game_id"], "date": x["date"], "q_home": x[qh], "q_away": x[qa],
                          "p_close": x["k_p_close"], "volume": x.get("k_volume_pre_tip"), "p_model": x["p_ensemble"],
                          "y": x["y"], "season": x["season"]})
        label = f"kalshi|{decision}|p_ensemble"
        led, dd, s = run(u, PRE_REGISTERED)
        res[label] = {k: v for k, v in s.items() if k != "config"}
        grids.append(run_grid(u, PRE_REGISTERED, label))
    # sportsbook closing moneyline 2013-2025: the price paid is the vigged close (CLV is zero by construction)
    from ..markets.odds import american_to_prob

    b = pred[(pred["season"] >= 2013) & pred["home_ml"].notna() & pred["y"].notna()]
    u = pd.DataFrame({"game_id": b["game_id"], "date": b["date"], "q_home": american_to_prob(b["home_ml"]),
                      "q_away": american_to_prob(b["away_ml"]), "p_model": b["p_ensemble"], "y": b["y"],
                      "season": b["season"]})
    cfg = replace(PRE_REGISTERED, fee_model="none", liquidity_frac=None)
    led, dd, s = run(u, cfg)
    res["book|close|p_ensemble"] = {k: v for k, v in s.items() if k != "config"}
    grids.append(run_grid(u, cfg, "book|close|p_ensemble"))
    dd.assign(strategy="book|close|p_ensemble").to_parquet(outdir / "backtest_paths.parquet")
    pd.concat(grids).to_csv(outdir / "backtest_grid.csv", index=False)
    (outdir / "backtests.json").write_text(json.dumps(res, indent=2, default=str))
    return res
