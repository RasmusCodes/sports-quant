"""Monte Carlo game simulation.

NBA - possession model. Both teams get N ~ round(Normal(pace, pace_sd)) possessions
(shared draw, which is what correlates the two scores). Each possession scores
k in {0,1,2,3,4} points from a league base distribution exponentially tilted to
hit the team's expected points per possession (P_k proportional to base_k * e^(theta*k)).
Ties go to 5-minute overtimes until decided. Player points: each scoring possession's
points are credited to a player with probability equal to his share of team scoring.

NFL - drive model. Each team gets D ~ round(Normal(drives, 1.0)) drives; a drive
ends in TD (6 + PAT 0.94 / 2-pt 0.48 if attempted), FG (3), safety for the defence
(2) or nothing, with TD/FG rates tilted to the team's expected points. Ties after
regulation get a 10-minute overtime approximated by one extra possession each; ties
remain possible. The drive model reproduces NFL key numbers (3, 7) that a normal
margin cannot.

Simulation output is the full joint distribution of scores, so moneyline, any
spread, any total and player props are priced from one model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

NBA_BASE = np.array([0.475, 0.060, 0.305, 0.150, 0.010])   # P(0,1,2,3,4 pts) per possession, ~1.13 ppp
PTS = np.arange(5)


def tilt(base: np.ndarray, target_mean: float, values=PTS) -> np.ndarray:
    def f(th):
        w = base * np.exp(th * values)
        return (w @ values) / w.sum() - target_mean

    lo, hi = -3.0, 3.0
    th = brentq(f, lo, hi)
    w = base * np.exp(th * values)
    return w / w.sum()


def simulate_nba(mu_home: float, mu_away: float, pace: float = 99.0, pace_sd: float = 4.5, n_sims: int = 20000,
                 seed: int | None = None, player_shares: dict | None = None):
    """Return dict with simulated home/away points (and player points if shares given).

    player_shares: {"home": {name: share_of_team_points}, "away": {...}}
    """
    rng = np.random.default_rng(seed)
    ph = tilt(NBA_BASE, mu_home / pace)
    pa = tilt(NBA_BASE, mu_away / pace)
    n_pos = np.maximum(np.rint(rng.normal(pace, pace_sd, n_sims)).astype(int), 60)
    # multinomial counts of 0/1/2/3/4-point possessions
    ch = np.vstack([rng.multinomial(n, ph) for n in n_pos])
    ca = np.vstack([rng.multinomial(n, pa) for n in n_pos])
    home = ch @ PTS
    away = ca @ PTS
    ot_pos = max(int(round(pace * 5 / 48)), 8)
    tied = home == away
    while tied.any():
        k = int(tied.sum())
        oh = rng.multinomial(ot_pos, ph, size=k) @ PTS
        oa = rng.multinomial(ot_pos, pa, size=k) @ PTS
        home[tied] += oh
        away[tied] += oa
        ch[tied] += 0  # counts no longer exact after OT; player props use regulation counts
        tied = home == away
    out = {"home": home, "away": away}
    if player_shares:
        for side, counts in (("home", ch), ("away", ca)):
            for name, s in player_shares.get(side, {}).items():
                pts = np.zeros(n_sims)
                for k in range(1, 5):
                    pts += k * rng.binomial(counts[:, k], s)
                out[f"{side}:{name}"] = pts
    return out


def price_from_sims(sims: dict, spread: float | None = None, total: float | None = None) -> dict:
    h, a = sims["home"], sims["away"]
    m = h - a
    res = {"p_home": float(np.mean(m > 0)), "mu_margin": float(m.mean()), "sd_margin": float(m.std()),
           "mu_total": float((h + a).mean()), "sd_total": float((h + a).std())}
    if spread is not None:  # home spread in sportsbook convention (negative = favourite)
        res["p_home_cover"] = float(np.mean(m + spread > 0))
        res["p_push"] = float(np.mean(m + spread == 0))
    if total is not None:
        res["p_over"] = float(np.mean(h + a > total))
        res["p_total_push"] = float(np.mean(h + a == total))
    return res


# ------------------------------------------------------------------ NFL
def simulate_nfl(mu_home: float, mu_away: float, drives: float = 11.0, n_sims: int = 40000, seed: int | None = None,
                 p_fg_given_score: float = 0.42, safety_rate: float = 0.004):
    rng = np.random.default_rng(seed)

    def drive_probs(mu_per_drive):
        # outcome values: TD ~ 6.95 expected incl. PAT, FG 3, safety conceded handled separately, empty 0
        def f(s):
            p_score = s
            td = p_score * (1 - p_fg_given_score)
            fg = p_score * p_fg_given_score
            return td * 6.95 + fg * 3.0 - mu_per_drive

        s = brentq(f, 1e-4, 0.95)
        return s * (1 - p_fg_given_score), s * p_fg_given_score

    def team_points(mu, n):
        td, fg = drive_probs(mu / drives)
        u = rng.random((n_sims, n.max()))
        mask = np.arange(n.max())[None, :] < n[:, None]
        is_td = (u < td) & mask
        is_fg = (u >= td) & (u < td + fg) & mask
        pat = rng.random(is_td.shape) < 0.94
        return (is_td * (6 + pat)).sum(axis=1) + 3 * is_fg.sum(axis=1)

    nh = np.maximum(np.rint(rng.normal(drives, 1.0, n_sims)).astype(int), 6)
    na = np.maximum(nh + rng.integers(-1, 2, n_sims), 6)
    home = team_points(mu_home, nh)
    away = team_points(mu_away, na)
    # safeties
    home = home + 2 * rng.binomial(na, safety_rate)
    away = away + 2 * rng.binomial(nh, safety_rate)
    tied = home == away
    if tied.any():
        k = int(tied.sum())
        ph = mu_home / (mu_home + mu_away)
        r = rng.random(k)
        decided = r < 0.92            # ~8% of NFL overtimes historically ended (or would end) tied
        winner_home = rng.random(k) < ph
        add = np.where(winner_home, 1, -1) * 3 * decided
        hh, aa = home[tied].copy(), away[tied].copy()
        hh = hh + np.where(add > 0, 3, 0)
        aa = aa + np.where(add < 0, 3, 0)
        home[tied], away[tied] = hh, aa
    return {"home": home, "away": away}
