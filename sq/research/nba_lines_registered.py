"""NBA spreads, totals (Polymarket main lines) and player points props (Kalshi sample).

Spreads / totals: the highest-volume Polymarket line per game; closing price = last price
point at or before tip-off. Model probabilities come from the Bayesian state-space model's
Gaussian margin and total (close information set).

Player props: Kalshi "Player: k+ points" markets; closing mid = last two-sided 1-minute quote
at or before tip-off. Kalshi resolves a prop "No" when the player does not play, so the
price embeds P(play). The model is P(play) x P(points >= k | play) with
  P(play)      from the last official injury report before tip - 5 min,
  points | play ~ negative binomial with mean = EW points-per-minute x EW minutes (pre-game)
                  and a dispersion fitted on 2019-2024 player games (training only).
"""
from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr

from ..config import OUT, TIDY_DIR
from ..data import nba
from ..data.nba import norm_name
from ..eval import metrics as M
from ..markets import quotes as Q
from . import market_eval as ME

log = logging.getLogger(__name__)
OUTDIR = OUT / "nba"


def _poly_event_map(games) -> dict:
    mk = pd.read_parquet(TIDY_DIR / "poly_nba_moneyline_markets.parquet")
    mk["a"] = mk["outcome0"].map(lambda s: Q._nick_to_abbr(s, "nba"))
    mk["b"] = mk["outcome1"].map(lambda s: Q._nick_to_abbr(s, "nba"))
    mk = mk.dropna(subset=["a", "b"])
    gst = pd.to_datetime(mk["game_start_time"].str.replace(r"\+00$", "+00:00", regex=True), utc=True, errors="coerce")
    ev = pd.DataFrame({"key": mk["event_slug"].astype(str),
                       "date": gst.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize(),
                       "team_a": mk["a"], "team_b": mk["b"]}).drop_duplicates("key")
    m = Q._match(ev, games)
    return dict(zip(m["key"], m["game_id"]))


def _poly_close(tag: str) -> pd.DataFrame:
    from ..io.polymarket import LINE_RE

    mk = pd.read_parquet(TIDY_DIR / f"poly_nba_{tag}_markets.parquet")
    # re-derive the line from the question text (authoritative; the API field can lose its sign)
    parsed = mk["question"].str.extract(LINE_RE)
    mk["line"] = pd.to_numeric(parsed[0].fillna(parsed[1]), errors="coerce")
    px = pd.read_parquet(TIDY_DIR / f"poly_nba_{tag}_prices.parquet")
    mk["tip"] = pd.to_datetime(mk["game_start_time"].str.replace(r"\+00$", "+00:00", regex=True), utc=True,
                               errors="coerce")
    px = px.merge(mk[["id", "tip"]], on="id")
    px = px[(px["ts_utc"] <= px["tip"]) & (px["p"] > 0.005) & (px["p"] < 0.995)]
    last = px.sort_values("ts_utc").groupby("id").tail(1)[["id", "p", "ts_utc", "tip"]]
    last["min_to_tip"] = (last["tip"] - last["ts_utc"]).dt.total_seconds() / 60
    return mk.merge(last[["id", "p", "min_to_tip"]], on="id")


def spreads_totals() -> dict:
    pred = pd.read_parquet(OUTDIR / "predictions.parquet")
    games = pred[["game_id", "date", "home_abbr", "away_abbr", "tip_utc"]].copy()
    emap = _poly_event_map(games)
    res = {}
    # ---------------- spreads
    sp = _poly_close("spreads")
    sp["game_id"] = sp["event_slug"].map(emap)
    sp = sp.dropna(subset=["game_id", "line", "p"]).merge(pred, on="game_id")
    sp = sp[sp["min_to_tip"] <= 30]
    sp["team0"] = sp["outcome0"].map(lambda s: Q._nick_to_abbr(s, "nba"))
    sp = sp[(sp["team0"] == sp["home_abbr"]) | (sp["team0"] == sp["away_abbr"])]   # outcome team must be in the game
    # integrity check against the market's own settlement: our cover flag must reproduce it
    home0 = sp["team0"] == sp["home_abbr"]
    sp["home_spread"] = np.where(home0, sp["line"], -sp["line"])
    sp["p_home_cover_mkt"] = np.where(home0, sp["p"], 1 - sp["p"])
    cm = sp["margin"] + sp["home_spread"]
    sp = sp[(cm != 0) & sp["margin"].notna()]
    sp["cover"] = ((sp["margin"] + sp["home_spread"]) > 0).astype(float)
    settled = sp["final_price0"].isin([0.0, 1.0])
    cover0 = np.where(home0.loc[sp.index], sp["cover"], 1 - sp["cover"])
    agree = float(np.mean(cover0[settled] == sp.loc[settled, "final_price0"]))
    sp = sp[~settled | (cover0 == sp["final_price0"])]          # drop rows our outcome cannot reproduce
    res["spread_settlement_agreement"] = agree
    mu = sp["mu_kalman_plus_close"].fillna(sp["kalman_mu_margin"])
    sd = sp["sd_kalman_plus_close"].fillna(sp["kalman_sd_margin"])
    sp["p_home_cover_model"] = ndtr((mu + sp["home_spread"]) / sd)
    t = ME.score_table(sp, {"Polymarket spread close": "p_home_cover_mkt", "State-space model": "p_home_cover_model"},
                       y="cover")
    t.to_csv(OUTDIR / "nba_spread_metrics.csv", index=False)
    enc = M.encompassing_test(sp["p_home_cover_mkt"], sp["p_home_cover_model"], sp["cover"], clusters=sp["date"])
    res["spread"] = {"n": int(len(sp)), "cover_rate_home": float(sp["cover"].mean()),
                     "mkt_mean_p": float(sp["p_home_cover_mkt"].mean()), "encompassing": enc,
                     "median_abs_line": float(sp["home_spread"].abs().median())}
    # ---------------- totals
    tt = _poly_close("totals")
    tt["game_id"] = tt["event_slug"].map(emap)
    tt = tt.dropna(subset=["game_id", "line", "p"]).merge(pred, on="game_id")
    tt = tt[(tt["min_to_tip"] <= 30) & tt["home_pts"].notna()]
    over0 = tt["outcome0"].str.lower().str.startswith("over")
    tt["p_over_mkt"] = np.where(over0, tt["p"], 1 - tt["p"])
    tot = tt["home_pts"] + tt["away_pts"]
    tt = tt[tot != tt["line"]]
    tt["over"] = ((tt["home_pts"] + tt["away_pts"]) > tt["line"]).astype(float)
    settled = tt["final_price0"].isin([0.0, 1.0])
    over0v = np.where(over0.loc[tt.index], tt["over"], 1 - tt["over"])
    res["total_settlement_agreement"] = float(np.mean(over0v[settled] == tt.loc[settled, "final_price0"]))
    tt = tt[~settled | (over0v == tt["final_price0"])]
    tt["p_over_model"] = 1 - ndtr((tt["line"] - tt["kalman_mu_total"]) / tt["kalman_sd_total"])
    t2 = ME.score_table(tt, {"Polymarket total close": "p_over_mkt", "State-space model": "p_over_model"}, y="over")
    t2.to_csv(OUTDIR / "nba_total_metrics.csv", index=False)
    enc2 = M.encompassing_test(tt["p_over_mkt"], tt["p_over_model"], tt["over"], clusters=tt["date"])
    res["total"] = {"n": int(len(tt)), "over_rate": float(tt["over"].mean()), "mkt_mean_p": float(tt["p_over_mkt"].mean()),
                    "encompassing": enc2, "model_bias_pts": float((tot.loc[tt.index] - tt["kalman_mu_total"]).mean())}
    return res


# ------------------------------------------------------------------ props
TITLE_RE = re.compile(r"^(.*?):\s*(\d+)\+")


def player_pregame(pb: pd.DataFrame, games: pd.DataFrame, hl_games=12.0) -> pd.DataFrame:
    """Strictly pre-game EW points-per-minute and minutes for every player-game."""
    p = pb.merge(games[["game_id", "tip_utc"]], on="game_id").sort_values(["person_id", "tip_utc"])
    p["played"] = p["min"] > 0
    alpha = 1 - 0.5 ** (1 / hl_games)
    out = []
    for pid, d in p.groupby("person_id", sort=False):
        pl = d[d["played"]]
        pts = pl["pts"].astype(float).ewm(alpha=alpha).mean().shift(1)
        mins = pl["min"].astype(float).ewm(alpha=alpha).mean().shift(1)
        n_prev = pd.Series(np.arange(len(pl)), index=pl.index)
        x = pd.DataFrame({"ew_pts": pts, "ew_min": mins, "n_prev": n_prev}).reindex(d.index).ffill()
        # a game the player missed still gets the latest pre-game estimate (shifted values carry forward)
        out.append(pd.concat([d[["game_id", "person_id", "team_id", "name_key", "pts", "min"]], x], axis=1))
    return pd.concat(out, ignore_index=True)


def nb_sf(k, mean, disp):
    """P(X >= k) for a negative binomial with mean and variance = disp * mean (disp > 1)."""
    mean = np.maximum(mean, 0.05)
    r = mean / max(disp - 1, 1e-6)
    pr = r / (r + mean)
    return stats.nbinom.sf(np.asarray(k) - 1, r, pr)


def _cluster_boot_roi(pnl, cost, games, n_boot=2000, seed=0):
    g = pd.DataFrame({"pnl": np.asarray(pnl, float), "cost": np.asarray(cost, float), "game": np.asarray(games)})
    g = g.groupby("game").sum()
    rng = np.random.default_rng(seed)
    boot = [g.iloc[rng.integers(0, len(g), len(g))].pipe(lambda d: d["pnl"].sum() / d["cost"].sum())
            for _ in range(n_boot)]
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def combined_prop_probability(s: pd.DataFrame, coefs: dict) -> np.ndarray:
    """logit p = b_const + b_market * logit(p_mkt) + b_model * logit(p_model), coefficients frozen elsewhere."""
    from ..eval.metrics import logit

    z = coefs["b_const"] + coefs["b_market"] * logit(s["p_mkt"]) + coefs["b_model"] * logit(s["p_model"])
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def prop_strategy(s: pd.DataFrame, min_edge: float = 0.02, contracts: int = 100, prob_col: str = "p_model") -> dict:
    """Flat-stake rule: at the closing quote, buy Yes at the ask or No at 1 - bid, whichever side's edge
    (probability in ``prob_col`` minus the all-in cost after the Kalshi taker fee) is larger, if that
    edge is at least ``min_edge``. One contract per bet, so the result does not depend on bankroll path."""
    from ..markets.odds import kalshi_fee

    p = s[prob_col]
    yes_cost = s["ask_close"] + kalshi_fee(s["ask_close"], contracts) / contracts
    no_price = 1 - s["bid_close"]
    no_cost = no_price + kalshi_fee(no_price, contracts) / contracts
    e_yes, e_no = p - yes_cost, (1 - p) - no_cost
    yes = e_yes >= e_no
    edge = np.where(yes, e_yes, e_no)
    bet = edge >= min_edge
    cost = np.where(yes, yes_cost, no_cost)
    win = np.where(yes, s["y"] == 1, s["y"] == 0)
    pnl = np.where(win, 1 - cost, -cost)
    if bet.sum() == 0:
        return {"n_bets": 0}
    lo, hi = _cluster_boot_roi(pnl[bet], cost[bet], s["game_id"].to_numpy()[bet])
    return {"n_bets": int(bet.sum()), "share_yes": float(np.mean(yes[bet])), "mean_pred_edge": float(edge[bet].mean()),
            "roi": float(pnl[bet].sum() / cost[bet].sum()), "roi_ci": [lo, hi], "hit_rate": float(np.mean(win[bet]))}


def props(tag: str = "kalshi_nba_player_points", out_prefix: str = "props", exclude_games=None) -> dict:
    mkf = TIDY_DIR / f"{tag}_markets.parquet"
    if not mkf.exists():
        return {"available": False}
    mk = pd.read_parquet(mkf)
    cd = pd.read_parquet(TIDY_DIR / f"{tag}_candles.parquet")
    pred = pd.read_parquet(OUTDIR / "predictions.parquet")
    games = nba.load("games")
    # event -> game via the game-winner series (same event suffix)
    gm = pd.read_parquet(TIDY_DIR / "market_quotes.parquet", columns=["game_id", "venue", "league"])
    km = pd.read_parquet(TIDY_DIR / "kalshi_nba_moneyline_markets.parquet")
    ev = km.groupby("event_ticker")["team_code"].agg(list).reset_index()
    ev = ev[ev["team_code"].map(len) == 2]
    ev = pd.DataFrame({"key": ev["event_ticker"], "date": ev["event_ticker"].map(Q._event_date),
                       "team_a": ev["team_code"].str[0], "team_b": ev["team_code"].str[1]})
    emap = dict(zip(*Q._match(ev, games)[["key", "game_id"]].T.to_numpy()))
    mk["game_event"] = mk["event_ticker"].str.replace("KXNBAPTS", "KXNBAGAME", regex=False)
    mk["game_id"] = mk["game_event"].map(emap)
    t = mk["title"].str.extract(TITLE_RE)
    mk["player"], mk["k"] = t[0], pd.to_numeric(t[1], errors="coerce")
    mk["name_key"] = mk["player"].map(norm_name)
    mk["team"] = mk["ticker"].str.split("-").str[2].str[:3]
    mk = mk.dropna(subset=["game_id", "k"]).merge(games[["game_id", "tip_utc", "date", "season"]], on="game_id")
    if exclude_games is not None:
        mk = mk[~mk["game_id"].isin(set(exclude_games))]
    # closing two-sided mid at or before tip-off
    c = cd[cd["interval"] == 1].merge(mk[["ticker", "tip_utc"]], on="ticker")
    c = c[(c["ts_utc"] <= c["tip_utc"]) & (c["bid_close"] > 0) & (c["ask_close"] < 1)]
    last = c.sort_values("ts_utc").groupby("ticker").tail(1)
    last = last.assign(p_mkt=(last["bid_close"] + last["ask_close"]) / 2, spread=last["ask_close"] - last["bid_close"],
                       min_to_tip=(last["tip_utc"] - last["ts_utc"]).dt.total_seconds() / 60)
    mk = mk.merge(last[["ticker", "p_mkt", "spread", "min_to_tip", "bid_close", "ask_close"]], on="ticker")
    mk = mk[mk["min_to_tip"] <= 30]
    # players and outcomes
    pb = nba.load("player_box")
    pre = player_pregame(pb[pb["season"] >= 2018], games)
    abbr = {v: k for k, v in nba.CURRENT_ABBR.items()}
    mk["team_id"] = mk["team"].map(abbr)
    x = mk.merge(pre, on=["game_id", "team_id", "name_key"], how="left")
    matched = x["person_id"].notna()
    x = x[matched].copy()
    x["y"] = (x["pts"] >= x["k"]).astype(float)
    # point-in-time availability from the injury report (tip - 5 min)
    inj = nba.load("injuries")
    inj = inj.dropna(subset=["game_id", "person_id"])
    inj = inj.merge(games[["game_id", "tip_utc"]], on="game_id")
    inj = inj[inj["report_utc"] <= inj["tip_utc"] - pd.Timedelta(minutes=5)]
    lastrep = inj.sort_values("report_utc").groupby(["game_id", "person_id"]).tail(1)
    x = x.merge(lastrep[["game_id", "person_id", "current_status"]], on=["game_id", "person_id"], how="left")
    from ..features.players import P_PLAY

    x["p_play"] = x["current_status"].map(P_PLAY).fillna(1.0)
    # dispersion fitted on training player-games (2019-2024 seasons, played games, >= 10 prior games)
    tr = pre.merge(games[["game_id", "season"]], on="game_id")
    tr = tr[(tr["season"] >= 2019) & (tr["season"] <= 2024) & (tr["min"] > 0) & (tr["n_prev"] >= 10)].dropna(
        subset=["ew_pts"])
    mu_tr = tr["ew_pts"].clip(lower=0.1)
    disp = float(np.mean((tr["pts"] - mu_tr) ** 2) / np.mean(mu_tr))
    x["mu_pts"] = x["ew_pts"].fillna(0.1)
    x["p_model"] = x["p_play"] * nb_sf(x["k"].to_numpy(), x["mu_pts"].to_numpy(), disp)
    x["p_model"] = x["p_model"].clip(0.005, 0.995)
    ok = x["p_mkt"].between(0.01, 0.99) & x["n_prev"].fillna(0).ge(5)
    s = x[ok].copy()
    tab = ME.score_table(s, {"Kalshi prop close (mid)": "p_mkt", "Player model": "p_model"})
    tab.to_csv(OUTDIR / f"{out_prefix}_metrics.csv", index=False)
    enc = M.encompassing_test(s["p_mkt"], s["p_model"], s["y"], clusters=s["game_id"])
    # one-sided p-value for H1 (b_model > 0)
    enc["p_model_one_sided"] = enc["p_model"] / 2 if enc["b_model"] > 0 else 1 - enc["p_model"] / 2
    cal = M.calibration_table(s["p_mkt"], s["y"], n_bins=10)
    cal.to_csv(OUTDIR / f"{out_prefix}_calibration_market.csv", index=False)
    fl = M.calibration_regression(s["p_mkt"], s["y"])
    paired = M.paired_loss_test(s["p_model"], s["p_mkt"], s["y"], clusters=s["game_id"])
    # does the yes-underpricing at the mid survive execution? buy one Yes at the closing ask + taker fee
    from ..markets.odds import kalshi_fee

    cost = s["ask_close"] + kalshi_fee(s["ask_close"], 100) / 100
    pnl = s["y"] - cost
    lo, hi = _cluster_boot_roi(pnl, cost, s["game_id"])
    yes_at_ask = {"roi": float(pnl.sum() / cost.sum()), "roi_ci": [lo, hi],
                  "mean_gap_vs_mid": float((s["y"] - s["p_mkt"]).mean()),
                  "mean_half_spread": float((s["ask_close"] - s["bid_close"]).mean() / 2)}
    s.to_parquet(OUTDIR / f"{out_prefix}_panel.parquet", index=False)
    seasons = sorted(int(v) for v in s["season"].unique())
    return {"available": True, "markets_collected": int(len(mk)), "matched_players_share": float(matched.mean()),
            "n_scored": int(len(s)), "games": int(s["game_id"].nunique()), "players": int(s["person_id"].nunique()),
            "seasons": seasons, "dates": [str(s["date"].min().date()), str(s["date"].max().date())],
            "dispersion": disp, "mean_spread": float(s["spread"].mean()), "yes_rate": float(s["y"].mean()),
            "mean_price": float(s["p_mkt"].mean()), "calibration_market": fl, "encompassing": enc,
            "paired_model_minus_market": paired, "yes_at_ask": yes_at_ask, "strategy_h3": prop_strategy(s),
            "dnp_share": float((s["min"].fillna(0) <= 0).mean())}


def props_replication() -> dict:
    """Held-out replication of the props result, as specified in paper/preregistration_props.md."""
    disc = json.loads((OUTDIR / "lines_props.json").read_text())["props"]
    frozen = {k: disc["encompassing"][k] for k in ("b_const", "b_market", "b_model")}
    disc_games = set(pd.read_parquet(OUTDIR / "props_panel.parquet", columns=["game_id"])["game_id"])
    rep = props(tag="kalshi_nba_player_points_rep", out_prefix="props_rep", exclude_games=disc_games)
    if not rep.get("available"):
        return rep
    s = pd.read_parquet(OUTDIR / "props_rep_panel.parquet")
    s["p_comb"] = combined_prop_probability(s, frozen)
    rep["frozen_discovery_coefs"] = frozen
    rep["strategy_h3_combined"] = prop_strategy(s, prob_col="p_comb")
    t = ME.score_table(s, {"Kalshi prop close (mid)": "p_mkt", "Player model": "p_model",
                           "Frozen combination": "p_comb"})
    t.to_csv(OUTDIR / "props_rep_metrics.csv", index=False)
    rep["paired_combined_minus_market"] = M.paired_loss_test(s["p_comb"], s["p_mkt"], s["y"], clusters=s["game_id"])
    enc = rep["encompassing"]
    rep["verdict"] = {
        "H1_model_adds_information": bool(enc["b_model"] > 0 and enc["p_model_one_sided"] < 0.05),
        "H2_yes_at_ask_not_profitable": bool(rep["yes_at_ask"]["roi_ci"][0] <= 0),
        "H3_combined_rule_profitable": bool(rep["strategy_h3_combined"].get("n_bets", 0) > 0
                                            and rep["strategy_h3_combined"]["roi_ci"][0] > 0),
        "H3b_model_only_rule_profitable": bool(rep["strategy_h3"].get("n_bets", 0) > 0
                                               and rep["strategy_h3"]["roi_ci"][0] > 0),
    }
    rep["overlap_with_discovery_games"] = int(len(disc_games & set(
        pd.read_parquet(OUTDIR / "props_rep_panel.parquet", columns=["game_id"])["game_id"])))
    (OUTDIR / "props_replication.json").write_text(json.dumps(rep, indent=2, default=float))
    return rep


def run() -> dict:
    res = {"spreads_totals": spreads_totals(), "props": props()}
    (OUTDIR / "lines_props.json").write_text(json.dumps(res, indent=2, default=float))
    return res
