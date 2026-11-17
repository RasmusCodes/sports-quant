"""Paper figures (PNG) from the saved result tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import OUT
from ..eval.metrics import logit
from . import style as S

FIG = OUT / "figures"


def _ax(title, xlabel, ylabel, size=(6.4, 4.0)):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def calibration(path_csv, series, out, title):
    t = pd.read_csv(path_csv)
    fig, ax = _ax(title, "Forecast probability (home win)", "Observed home-win frequency", (5.6, 5.0))
    ax.plot([0, 1], [0, 1], color=S.BASELINE, linewidth=1.0, zorder=1)
    for i, name in enumerate(series):
        d = t[t["model"] == name].sort_values("mean_pred")
        if d.empty:
            continue
        ax.plot(d["mean_pred"], d["obs_freq"], color=S.SERIES[i], marker="o", markeredgecolor=S.SURFACE,
                markeredgewidth=1.5, label=name, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    S.save(fig, out)


def season_lines(path_csv, series, out, title, first=2010):
    t = pd.read_csv(path_csv)
    t = t[t["season"] >= first]
    fig, ax = _ax(title, "Season (year it ends)", "Log loss (lower is better)", (7.0, 4.0))
    for i, name in enumerate(series):
        d = t[t["model"] == name].sort_values("season")
        ax.plot(d["season"], d["log_loss"], color=S.SERIES[i], marker="o", markeredgecolor=S.SURFACE,
                markeredgewidth=1.5, label=name)
    ax.legend(loc="upper left", ncol=len(series))
    ax.set_xticks(sorted(t["season"].unique())[::2])
    S.save(fig, out)


def information_arrival(path_csv, out, title):
    t = pd.read_csv(path_csv).set_index("model")
    steps = ["open", "T-24h", "T-6h", "T-1h", "T-15m", "close"]
    fig, ax = _ax(title, "Snapshot before tip-off", "Log loss of the market mid-price", (7.0, 4.0))
    for i, venue in enumerate(["Kalshi", "Polymarket"]):
        ys = [t.loc[f"{venue} {s}", "log_loss"] if f"{venue} {s}" in t.index else np.nan for s in steps]
        ax.plot(range(len(steps)), ys, color=S.SERIES[i], marker="o", markeredgecolor=S.SURFACE,
                markeredgewidth=1.5, label=venue)
    for name, color in (("Ensemble", S.SERIES[2]), ("Elo (538 params)", S.SERIES[3])):
        if name in t.index:
            v = t.loc[name, "log_loss"]
            ax.axhline(v, color=color, linewidth=1.2)
            ax.text(len(steps) - 1, v, f" {name} (close info)" if name == "Ensemble" else f" {name}",
                    va="bottom", ha="right", color=S.INK2, fontsize=8.5)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps)
    ax.legend(loc="lower left")
    S.save(fig, out)


def paired_dots(path_csv, out, title, order=None):
    t = pd.read_csv(path_csv)
    t = t[~t["model"].str.contains("Kalshi|Polymarket")]
    if order:
        t = t.set_index("model").reindex([o for o in order if o in set(t["model"])]).reset_index()
    fig, ax = _ax(title, "Log-loss difference vs market close (positive = worse than market)", "", (7.0, 4.6))
    y = np.arange(len(t))[::-1]
    ax.axvline(0, color=S.BASELINE, linewidth=1.0)
    ax.hlines(y, t["ci_lo"], t["ci_hi"], color=S.SERIES[0], linewidth=2)
    ax.plot(t["mean_diff"], y, "o", color=S.SERIES[0], markeredgecolor=S.SURFACE, markeredgewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(t["model"])
    ax.grid(axis="y", visible=False)
    S.save(fig, out)


def bankroll(paths_parquet, strategies, labels, out, title):
    p = pd.read_parquet(paths_parquet)
    fig, ax = _ax(title, "", "Bankroll (start = 10,000)", (7.0, 4.0))
    for i, (s, lab) in enumerate(zip(strategies, labels)):
        d = p[p["strategy"] == s].sort_values("date")
        if d.empty:
            continue
        ax.plot(pd.to_datetime(d["date"]), d["bankroll"], color=S.SERIES[i], label=lab, linewidth=1.8)
    ax.axhline(10_000, color=S.BASELINE, linewidth=1.0)
    ax.legend(loc="upper left")
    S.save(fig, out)


def clv_hist(ledger_parquet, strategy, out, title):
    led = pd.read_parquet(ledger_parquet)
    d = led[led["strategy"] == strategy]
    if d.empty:
        return
    fig, ax = _ax(title, "Closing-line value per bet (probability points)", "Bets", (6.4, 3.8))
    ax.hist(d["clv"] * 100, bins=np.arange(-15, 15.5, 1.0), color=S.SERIES[0], edgecolor=S.SURFACE, linewidth=1.5)
    ax.axvline(0, color=S.BASELINE, linewidth=1.0)
    m = d["clv"].mean() * 100
    ax.axvline(m, color=S.SERIES[1], linewidth=1.5)
    ax.text(m, ax.get_ylim()[1] * 0.95, f" mean {m:+.2f}", color=S.INK2, fontsize=9, va="top")
    S.save(fig, out)


def flb(paths, labels, out, title):
    fig, ax = _ax(title, "Fair price of the side (devigged)", "Realised win rate", (5.6, 5.0))
    ax.plot([0, 1], [0, 1], color=S.BASELINE, linewidth=1.0)
    for i, (p, lab) in enumerate(zip(paths, labels)):
        try:
            t = pd.read_csv(p)
        except FileNotFoundError:
            continue
        ax.plot(t["mean_price"], t["win_rate"], color=S.SERIES[i], marker="o", markeredgecolor=S.SURFACE,
                markeredgewidth=1.5, label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    S.save(fig, out)


def injury_repricing(panel_parquet, out, title):
    m = pd.read_parquet(panel_parquet)
    d = m.dropna(subset=["k_p_close", "k_p_t24h", "lost_delta_5", "lost_delta_1440"]).copy()
    d["late_news"] = d["lost_delta_5"] - d["lost_delta_1440"]
    d["move"] = (logit(d["k_p_close"]) - logit(d["k_p_t24h"]))
    d = d[d["late_news"].abs() > 1e-9]
    if len(d) < 30:
        return
    d["bin"] = pd.qcut(d["late_news"], 12, duplicates="drop")
    b = d.groupby("bin", observed=True).agg(x=("late_news", "mean"), y=("move", "mean"), n=("move", "size"))
    fig, ax = _ax(title, "Late availability news, T-24h to T-5m (home minus away, Game Score lost)",
                  "Kalshi price move, T-24h to close (log-odds)", (7.0, 4.2))
    ax.axhline(0, color=S.BASELINE, linewidth=1.0)
    ax.axvline(0, color=S.BASELINE, linewidth=1.0)
    ax.scatter(d["late_news"], d["move"], s=8, color=S.SERIES[0], alpha=0.18, linewidths=0)
    ax.plot(b["x"], b["y"], color=S.SERIES[1], marker="o", markeredgecolor=S.SURFACE, markeredgewidth=1.5,
            label="bin means (12 quantile bins)")
    ax.legend(loc="upper right")
    S.save(fig, out)


def nfl_key_numbers(spread_parquet, out, title, n_sims=3000):
    from scipy.special import ndtr

    from ..models.simulate import simulate_nfl

    sp = pd.read_parquet(spread_parquet)
    ks = np.arange(0, 22)
    actual = np.array([np.mean(sp["margin"].abs() == k) for k in ks])
    sim = np.zeros(len(ks))
    norm = np.zeros(len(ks))
    for r in sp.itertuples(index=False):
        mt = r.kalman_mu_total if np.isfinite(r.kalman_mu_total) else 44.0
        mh, ma = (mt + r.kalman_mu_margin) / 2, (mt - r.kalman_mu_margin) / 2
        s = simulate_nfl(max(mh, 3), max(ma, 3), n_sims=n_sims, seed=7)
        a = np.abs(s["home"] - s["away"])
        sim += np.array([np.mean(a == k) for k in ks])
        mu, sd = r.kalman_mu_margin, r.kalman_sd_margin
        p = ndtr((ks + 0.5 - mu) / sd) - ndtr((ks - 0.5 - mu) / sd) + ndtr((-ks + 0.5 - mu) / sd) - ndtr((-ks - 0.5 - mu) / sd)
        p[0] = ndtr((0.5 - mu) / sd) - ndtr((-0.5 - mu) / sd)
        norm += p
    sim /= len(sp)
    norm /= len(sp)
    fig, ax = _ax(title, "Final margin (absolute points)", "Share of games", (7.0, 4.0))
    ax.bar(ks, actual, width=0.7, color=S.SERIES[0], edgecolor=S.SURFACE, linewidth=1.5, label="Actual")
    ax.plot(ks, sim, color=S.SERIES[1], marker="o", markeredgecolor=S.SURFACE, markeredgewidth=1.5,
            label="Drive simulation")
    ax.plot(ks, norm, color=S.SERIES[2], label="Normal margin")
    ax.set_xticks(ks[::3])
    ax.legend(loc="upper right")
    S.save(fig, out)
    return {"ks": ks.tolist(), "actual": actual.tolist(), "sim": sim.tolist(), "normal": norm.tolist()}


def make_all() -> list[str]:
    S.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    nba, nfl = OUT / "nba", OUT / "nfl"
    made = []

    def go(fn, *a, name):
        try:
            fn(*a)
            made.append(name)
        except FileNotFoundError:
            pass

    go(calibration, nba / "market_common_calibration.csv", ["Kalshi close", "Ensemble", "Elo (538 params)"],
       FIG / "nba_calibration_market.png", "NBA: calibration on games priced by both exchanges",
       name="nba_calibration_market")
    go(season_lines, nba / "season_metrics.csv", ["Elo (538 params)", "Bayesian state-space", "Ensemble"],
       FIG / "nba_logloss_by_season.png", "NBA: out-of-sample log loss by season", name="nba_logloss_by_season")
    go(information_arrival, nba / "market_common_metrics.csv", FIG / "nba_information_arrival.png",
       "NBA: exchange prices sharpen toward tip-off", name="nba_information_arrival")
    go(paired_dots, nba / "market_kalshi_paired_vs_close.csv", FIG / "nba_models_vs_kalshi.png",
       "NBA: every model is worse than the Kalshi closing price (95% CI)", name="nba_models_vs_kalshi")
    go(bankroll, nba / "backtest_paths.parquet",
       ["kalshi|t6h|p_ensemble_early", "kalshi|t6h|elo538_p_home", "kalshi|t6h|p_combined_early"],
       ["Ensemble alone, Kalshi T-6h", "Elo (538) alone, Kalshi T-6h", "Model + market combination, Kalshi T-6h"],
       FIG / "nba_bankroll.png", "NBA: pre-registered quarter-Kelly strategies, out of sample", name="nba_bankroll")
    go(clv_hist, nba / "backtest_ledger.parquet", "kalshi|t6h|p_ensemble_early", FIG / "nba_clv.png",
       "NBA: closing-line value of the ensemble's Kalshi bets (T-6h)", name="nba_clv")
    go(flb, [nba / "flb_kalshi_close.csv", nfl / "flb_book_multiplicative.csv"],
       ["NBA, Kalshi close", "NFL, sportsbook close"], FIG / "flb.png", "Favourite-longshot check", name="flb")
    go(injury_repricing, nba / "market_panel.parquet", FIG / "nba_injury_repricing.png",
       "NBA: prices move with late availability news", name="nba_injury_repricing")
    try:
        nfl_key_numbers(nfl / "spread_panel.parquet", FIG / "nfl_key_numbers.png",
                        "NFL: final margins cluster on 3 and 7; a normal margin misses it")
        made.append("nfl_key_numbers")
    except FileNotFoundError:
        pass
    return made
