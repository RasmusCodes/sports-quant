"""Event-driven backtest of binary game contracts.

Realism rules
-------------
* A bet at decision time T can only use the model probability available at T and
  the executable price at T: the ask (q_home / q_away) of the snapshot nearest
  before T. Mid prices are never traded. Games without a two-sided quote at T are
  skipped.
* Costs: venue fee model (Kalshi taker fee 0.07*P*(1-P) per contract, rounded up
  per order; Polymarket fee rate parameter), optional slippage in price points, and
  an optional liquidity cap (stake <= a fraction of the market's pre-tip volume).
* Sizing: fractional Kelly for a $1 binary contract bought at all-in cost c with
  win probability p: f* = (p - c) / (1 - c). Caps per bet and per day; all bets of a
  day are sized off the bankroll at the start of that day (no intraday compounding).
* Evaluation: besides P&L, every bet records closing-line value (CLV) - the closing
  fair probability of the side bought minus the price paid - which measures edge
  with far less noise than realised returns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..markets.odds import kalshi_fee


@dataclass
class Config:
    kelly_fraction: float = 0.25
    min_edge: float = 0.02
    max_bet_frac: float = 0.02
    max_day_frac: float = 0.15
    fee_model: str = "kalshi"         # kalshi | polymarket | none
    poly_fee_rate: float = 0.0
    slippage: float = 0.0
    liquidity_frac: float | None = 0.02
    order_contracts: float = 100.0    # order size used for Kalshi fee rounding
    start_bankroll: float = 10_000.0
    one_side_only: bool = True


def all_in_cost(q, cfg: Config):
    q = np.asarray(q, float) + cfg.slippage
    if cfg.fee_model == "kalshi":
        return q + kalshi_fee(q, cfg.order_contracts) / cfg.order_contracts
    if cfg.fee_model == "polymarket":
        return q + cfg.poly_fee_rate * q * (1 - q)
    return q


def run(bets: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """bets columns: game_id, date, p_model (home), q_home, q_away, y (1 home win),
    optional p_close (home fair at close), volume (pre-tip $ volume)."""
    b = bets.dropna(subset=["p_model", "q_home", "q_away", "y"]).sort_values(["date", "game_id"]).copy()
    b["c_home"] = all_in_cost(b["q_home"], cfg)
    b["c_away"] = all_in_cost(b["q_away"], cfg)
    b["edge_home"] = b["p_model"] - b["c_home"]
    b["edge_away"] = (1 - b["p_model"]) - b["c_away"]
    bank = cfg.start_bankroll
    ledger, days = [], []
    peak = bank
    for date, d in b.groupby("date", sort=True):
        start = bank
        picks = []
        for r in d.itertuples(index=False):
            side = "home" if r.edge_home >= r.edge_away else "away"
            edge = max(r.edge_home, r.edge_away)
            if edge < cfg.min_edge:
                continue
            p = r.p_model if side == "home" else 1 - r.p_model
            c = r.c_home if side == "home" else r.c_away
            q = r.q_home if side == "home" else r.q_away
            if not (0.01 < c < 0.99):
                continue
            f_star = (p - c) / (1 - c)
            stake = min(cfg.kelly_fraction * f_star, cfg.max_bet_frac) * start
            if cfg.liquidity_frac and "volume" in d.columns and np.isfinite(getattr(r, "volume", np.nan)):
                stake = min(stake, cfg.liquidity_frac * r.volume * c)
            if stake <= 0:
                continue
            picks.append([r, side, p, c, q, f_star, stake])
        tot = sum(x[6] for x in picks)
        scale = min(1.0, cfg.max_day_frac * start / tot) if tot > 0 else 1.0
        pnl_day = 0.0
        for r, side, p, c, q, f_star, stake in picks:
            stake *= scale
            won = (r.y == 1) if side == "home" else (r.y == 0)
            contracts = stake / c
            pnl = contracts * (1 - c) if won else -stake
            pnl_day += pnl
            pc = getattr(r, "p_close", np.nan)
            pclose_side = pc if side == "home" else 1 - pc
            ledger.append({"game_id": r.game_id, "date": date, "side": side, "p_model": p, "price": q, "cost": c,
                           "edge": p - c, "kelly_f": f_star, "stake": stake, "contracts": contracts, "won": int(won),
                           "pnl": pnl, "bankroll_before": start, "p_close_side": pclose_side,
                           "clv": pclose_side - q, "clv_after_cost": pclose_side - c,
                           "exp_pnl_at_close": contracts * (pclose_side - c)})
        bank = start + pnl_day
        peak = max(peak, bank)
        days.append({"date": date, "bankroll": bank, "pnl": pnl_day, "staked": sum(x[6] for x in picks) * scale,
                     "n_bets": len(picks), "drawdown": bank / peak - 1})
    led = pd.DataFrame(ledger)
    dd = pd.DataFrame(days)
    return led, dd, summarize(led, dd, cfg)


def summarize(led: pd.DataFrame, dd: pd.DataFrame, cfg: Config, n_boot=2000, seed=0) -> dict:
    s = {"config": asdict(cfg)}
    if led.empty:
        s.update(n_bets=0)
        return s
    staked = led["stake"].sum()
    s.update(n_bets=int(len(led)), staked=float(staked), pnl=float(led["pnl"].sum()),
             roi=float(led["pnl"].sum() / staked), final_bankroll=float(dd["bankroll"].iloc[-1]),
             total_return=float(dd["bankroll"].iloc[-1] / cfg.start_bankroll - 1),
             max_drawdown=float(dd["drawdown"].min()), hit_rate=float(led["won"].mean()),
             avg_price=float(led["price"].mean()), avg_edge_pred=float(led["edge"].mean()),
             avg_clv=float(led["clv"].mean()), avg_clv_after_cost=float(led["clv_after_cost"].mean()),
             share_positive_clv=float((led["clv"] > 0).mean()),
             exp_roi_at_close=float(led["exp_pnl_at_close"].sum() / staked))
    r = dd["pnl"] / dd["bankroll"].shift(1).fillna(cfg.start_bankroll)
    s["daily_sharpe"] = float(r.mean() / r.std()) if r.std() > 0 else np.nan
    s["log_growth_per_bet"] = float(np.log(dd["bankroll"].iloc[-1] / cfg.start_bankroll) / len(led))
    # bootstrap over betting days: ROI and CLV uncertainty
    rng = np.random.default_rng(seed)
    g = led.groupby("date").agg(pnl=("pnl", "sum"), stake=("stake", "sum"), clv=("clv", "sum"), n=("clv", "size"))
    P, S, C, N = (g[c].to_numpy() for c in ("pnl", "stake", "clv", "n"))
    k = len(g)
    roi_b, clv_b = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, k)
        roi_b[i] = P[idx].sum() / S[idx].sum()
        clv_b[i] = C[idx].sum() / N[idx].sum()
    s.update(roi_ci=[float(np.quantile(roi_b, 0.025)), float(np.quantile(roi_b, 0.975))],
             p_roi_le_0=float(np.mean(roi_b <= 0)),
             clv_ci=[float(np.quantile(clv_b, 0.025)), float(np.quantile(clv_b, 0.975))],
             p_clv_le_0=float(np.mean(clv_b <= 0)))
    return s
