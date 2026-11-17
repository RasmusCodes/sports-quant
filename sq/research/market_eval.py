"""Model-vs-market evaluation on a common set of games.

Market probabilities are margin-free before any calibration test:
  * Kalshi: mid of both books (p_mid) and, as a check, the two asks devigged with each
    method in ``sq.markets.devig``.
  * Polymarket: the single book's price (a mid).
  * Sportsbooks (NFL): both moneylines devigged (multiplicative / additive / power /
    Shin / odds-ratio).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIDY_DIR
from ..eval import metrics as M
from ..markets import devig as DV


def market_wide(league: str) -> pd.DataFrame:
    ms = pd.read_parquet(TIDY_DIR / "market_summary.parquet")
    ms = ms[ms["league"] == league]
    out = None
    for venue, tag in (("kalshi", "k"), ("polymarket", "pm")):
        v = ms[ms["venue"] == venue].drop(columns=["venue", "league"]).dropna(axis=1, how="all")
        v = v.rename(columns={c: f"{tag}_{c}" for c in v.columns if c != "game_id"})
        out = v if out is None else out.merge(v, on="game_id", how="outer")
    return out


def add_devig(df: pd.DataFrame, qh: str, qa: str, prefix: str) -> pd.DataFrame:
    q = df[[qh, qa]].to_numpy(float)
    for m in DV.METHODS:
        df[f"{prefix}_{m}"] = DV.devig(q, m)[:, 0]
    return df


def score_table(df: pd.DataFrame, cols: dict[str, str], y="y") -> pd.DataFrame:
    rows = []
    for label, c in cols.items():
        ok = df[c].notna() & df[y].notna()
        if ok.sum() < 30:
            continue
        s = M.summarize(df.loc[ok, c], df.loc[ok, y], label)
        rows.append(s)
    return pd.DataFrame(rows)


def paired_table(df: pd.DataFrame, cols: dict[str, str], ref: str, y="y", cluster="date") -> pd.DataFrame:
    rows = []
    for label, c in cols.items():
        if c == ref:
            continue
        ok = df[c].notna() & df[ref].notna() & df[y].notna()
        if ok.sum() < 30:
            continue
        t = M.paired_loss_test(df.loc[ok, c], df.loc[ok, ref], df.loc[ok, y], clusters=df.loc[ok, cluster])
        rows.append({"model": label, "vs": ref, **t})
    return pd.DataFrame(rows)


def encompassing_table(df: pd.DataFrame, market: str, cols: dict[str, str], y="y", cluster="date") -> pd.DataFrame:
    rows = []
    for label, c in cols.items():
        if c == market:
            continue
        ok = df[c].notna() & df[market].notna() & df[y].notna()
        if ok.sum() < 100:
            continue
        r = M.encompassing_test(df.loc[ok, market], df.loc[ok, c], df.loc[ok, y], clusters=df.loc[ok, cluster])
        rows.append({"model": label, "market": market, "n": int(ok.sum()), **r})
    return pd.DataFrame(rows)


def calibration_curves(df: pd.DataFrame, cols: dict[str, str], y="y", n_bins=10) -> pd.DataFrame:
    out = []
    for label, c in cols.items():
        ok = df[c].notna() & df[y].notna()
        if ok.sum() < 100:
            continue
        t = M.calibration_table(df.loc[ok, c], df.loc[ok, y], n_bins=n_bins)
        out.append(t.assign(model=label))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def favourite_longshot(df: pd.DataFrame, pcol: str, y="y") -> dict:
    """Bettor-side view: realised win rate vs fair price for favourites and longshots.

    Each game contributes both sides; the regression slope of logit(outcome) on
    logit(price) > 1 means favourites win more than priced (classic FLB)."""
    ok = df[pcol].notna() & df[y].notna()
    # regression on one side only (doubling the sample would understate the SE)
    cr = M.calibration_regression(df.loc[ok, pcol], df.loc[ok, y])
    p = np.concatenate([df.loc[ok, pcol], 1 - df.loc[ok, pcol]])
    yy = np.concatenate([df.loc[ok, y], 1 - df.loc[ok, y]])
    bands = pd.cut(p, [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
    t = pd.DataFrame({"band": bands, "p": p, "y": yy}).groupby("band", observed=True).agg(
        n=("y", "size"), mean_price=("p", "mean"), win_rate=("y", "mean")).reset_index()
    t["band"] = t["band"].astype(str)
    return {"slope": cr["slope"], "slope_se": cr["slope_se"], "intercept": cr["intercept"], "bands": t}
