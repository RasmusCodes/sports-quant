"""Collect the dashboard's data into one JSON-serialisable dict (compact per-game rows)."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from ..config import OUT
from ..eval.metrics import brier, log_loss


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def _rows(df: pd.DataFrame, spec: dict[str, str]) -> dict:
    out = {"cols": list(spec)}
    data = []
    for _, r in df.iterrows():
        row = []
        for key, col in spec.items():
            v = r.get(col)
            if isinstance(v, (pd.Timestamp, dt.date)):
                row.append(str(v)[:10])
            elif isinstance(v, (float, np.floating)):
                row.append(_r(v))
            elif isinstance(v, (np.integer,)):
                row.append(int(v))
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                row.append(None)
            else:
                row.append(v if not pd.isna(v) else None)
        data.append(row)
    out["rows"] = data
    return out


def _paths(path_parquet, strategies):
    try:
        p = pd.read_parquet(path_parquet)
    except FileNotFoundError:
        return {}
    out = {}
    for s in strategies:
        d = p[p["strategy"] == s].sort_values("date")
        if len(d):
            out[s] = {"date": [str(x)[:10] for x in d["date"]], "bankroll": [round(float(x), 2) for x in d["bankroll"]]}
    return out


def _season_metrics(df, cols, first):
    rows = []
    for s, d in df[df["season"] >= first].groupby("season"):
        rec = {"season": int(s)}
        for name, c in cols.items():
            ok = d[c].notna() & d["y"].notna()
            if ok.sum() >= 50:
                rec[name] = round(log_loss(d.loc[ok, c], d.loc[ok, "y"]), 4)
        rows.append(rec)
    return rows


def collect() -> dict:
    nba_dir, nfl_dir = OUT / "nba", OUT / "nfl"
    data = {"built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

    # ------------------------------------------------------------------ NBA
    panel = pd.read_parquet(nba_dir / "market_panel.parquet")
    panel = panel[panel["y"].notna()].sort_values("date")
    try:
        led = pd.read_parquet(nba_dir / "backtest_ledger.parquet")
        led = led[led["strategy"] == "kalshi|t6h|p_ensemble_early"][["game_id", "side", "stake", "pnl", "clv"]]
        panel = panel.merge(led, on="game_id", how="left")
    except FileNotFoundError:
        pass
    spec = {"date": "date", "season": "season", "home": "home_abbr", "away": "away_abbr", "y": "y",
            "hp": "home_pts", "ap": "away_pts", "ens": "p_ensemble_close", "ens_e": "p_ensemble_early",
            "elo": "elo538_p_home", "bayes": "kalman_p_home", "k": "k_p_close", "k6": "k_p_t6h", "ko": "k_p_open",
            "pm": "pm_p_close", "pmo": "pm_p_open", "late": "lost_delta_5", "side": "side", "stake": "stake",
            "pnl": "pnl", "clv": "clv"}
    data["nba"] = {"games": _rows(panel, spec)}
    pred = pd.read_parquet(nba_dir / "predictions.parquet")
    data["nba"]["season_ll"] = _season_metrics(pred, {"Elo (538 params)": "elo538_p_home",
                                                      "Bayesian state-space": "kalman_p_home",
                                                      "Ensemble": "p_ensemble_close"}, 2010)
    try:
        bt = json.loads((nba_dir / "backtests.json").read_text())
        data["nba"]["backtests"] = {k: {kk: v.get(kk) for kk in ("n_bets", "roi", "roi_ci", "p_roi_le_0", "avg_clv",
                                                                 "clv_ci", "max_drawdown", "total_return", "daily_sharpe",
                                                                 "hit_rate", "exp_roi_at_close", "universe_games")}
                                    for k, v in bt.items()}
    except FileNotFoundError:
        data["nba"]["backtests"] = {}
    data["nba"]["paths"] = _paths(nba_dir / "backtest_paths.parquet",
                                  ["kalshi|t6h|p_ensemble_early", "kalshi|t6h|elo538_p_home",
                                   "kalshi|t6h|p_combined_early"])
    # upcoming games: model probabilities for unplayed games in the next days (+ live prices if snapshot exists)
    up = pred[pred["home_pts"].isna() & (pred["date"] >= pd.Timestamp.now().normalize() - pd.Timedelta(days=1))]
    up = up.sort_values("tip_utc").head(40)
    data["nba"]["upcoming"] = _rows(up, {"date": "date", "home": "home_abbr", "away": "away_abbr",
                                         "ens": "p_ensemble_early", "elo": "elo538_p_home", "bayes": "kalman_p_home"})

    # ------------------------------------------------------------------ NFL
    try:
        from .nfl_report import add_book

        npred = add_book(pd.read_parquet(nfl_dir / "predictions.parquet"))
        npred = npred[npred["y"].notna()]
        try:
            ex = pd.read_parquet(nfl_dir / "market_panel.parquet")[["game_id", "k_p_close", "k_p_open", "pm_p_close",
                                                                    "k_p_t6h", "pm_p_open"]]
            npred = npred.merge(ex, on="game_id", how="left")
        except FileNotFoundError:
            pass
        npred = npred[(npred["season"] >= 2013)].sort_values("date")
        nspec = {"date": "date", "season": "season", "home": "home_abbr", "away": "away_abbr", "y": "y",
                 "hp": "home_pts", "ap": "away_pts", "ens": "p_ensemble", "elo": "elo538_p_home",
                 "bayes": "kalman_p_home", "book": "book_shin", "k": "k_p_close", "ko": "k_p_open", "pm": "pm_p_close",
                 "pmo": "pm_p_open"}
        data["nfl"] = {"games": _rows(npred, nspec)}
        allp = pd.read_parquet(nfl_dir / "predictions.parquet")
        data["nfl"]["season_ll"] = _season_metrics(allp, {"Elo (538 params)": "elo538_p_home",
                                                          "Bayesian state-space": "kalman_p_home",
                                                          "Ensemble": "p_ensemble"}, 2010)
        try:
            bt = json.loads((nfl_dir / "backtests.json").read_text())
            data["nfl"]["backtests"] = {k: {kk: v.get(kk) for kk in ("n_bets", "roi", "roi_ci", "p_roi_le_0", "avg_clv",
                                                                     "clv_ci", "max_drawdown", "total_return",
                                                                     "daily_sharpe", "hit_rate")} for k, v in bt.items()}
        except FileNotFoundError:
            data["nfl"]["backtests"] = {}
        data["nfl"]["paths"] = _paths(nfl_dir / "backtest_paths.parquet", ["book|close|p_ensemble"])
    except FileNotFoundError:
        data["nfl"] = None
    return data
