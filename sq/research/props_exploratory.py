"""Post-hoc diagnostics for the props replication.

NOT pre-registered. Written on 2026-11-15 after the replication results had been seen; the paper reports them as
exploratory, apart from the four registered tests in ``paper/preregistration_props.md``. Everything here starts from
the registered record: the registered outputs in ``outputs/nba/registered/`` and the registered code, preserved
byte-for-byte as ``sq/research/nba_lines_registered.py`` (the default ``nba_lines.py`` now scores props from
settlement, which is what section 4 below does post hoc).

Four questions:

1. Inference. The registered test clusters by game. The same player's props recur across games, and games on
   the same date share league-wide scoring shocks, so the model weight is re-estimated with standard errors
   clustered by player, by date, and two ways (game x player, date x player).
2. Source of the information. The player model has two parts: points given that the player plays, and
   availability from the injury report. The regression is re-run separately on props for players with no
   injury designation (P(play) = 1) and with one (P(play) < 1).
3. Trading rules. Descriptive facts behind the losses: side of the bets, the Yes premium over the mid versus the
   half-spread paid, and the frozen intercept versus the one the replication sample would have produced.
4. Settlement-consistent rescoring. The registered code scores every prop from the box score
   (y = 1 if points >= line), so a player who sits scores 0. Kalshi does not settle those markets No: it settles
   them at a scalar value ("Confirmed DNP", result = "scalar", paid at ``settlement_value``). Here the payoff is
   taken from the settlement itself (1 for Yes, 0 for No, ``settlement_value`` for scalar), scalar-settled props are
   left out of the 0/1 information test, and the trading rules are re-priced with the actual payoff. Everything else
   (filters, model, frozen coefficients, costs) is unchanged.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from sq.config import OUT
from sq.eval import metrics as M

OUTDIR = OUT / "nba"
REGDIR = OUTDIR / "registered"          # registered discovery + replication outputs (frozen)


def _codes(x) -> np.ndarray:
    return pd.factorize(pd.Series(np.asarray(x)).astype(str))[0]


def encompassing_clustered(s: pd.DataFrame, groups: np.ndarray) -> dict:
    """The registered regression (metrics.encompassing_test) with one-way or two-way clustered errors.

    ``groups`` is one integer column (one-way) or two (two-way: V_a + V_b - V_ab, statsmodels).
    """
    import statsmodels.api as sm

    X = sm.add_constant(np.column_stack([M.logit(s["p_mkt"]), M.logit(s["p_model"])]))
    res = sm.GLM(s["y"].to_numpy(float), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    b, se = float(res.params[2]), float(res.bse[2])
    return {"b_model": b, "se_model": se, "z": b / se, "p_one_sided": float(norm.sf(b / se)), "n": int(len(s))}


def clustering_table(s: pd.DataFrame) -> list[dict]:
    g, p, d = _codes(s["game_id"]), _codes(s["person_id"]), _codes(s["date"])
    specs = [
        ("Game (registered)", g),
        ("Player", p),
        ("Date", d),
        ("Game and player (two-way)", np.column_stack([g, p])),
        ("Date and player (two-way)", np.column_stack([d, p])),
    ]
    clusters = {"Game (registered)": s["game_id"].nunique(), "Player": s["person_id"].nunique(),
                "Date": s["date"].nunique()}
    rows = []
    for name, groups in specs:
        r = encompassing_clustered(s, groups)
        r["clustering"] = name
        r["clusters"] = clusters.get(name)
        rows.append(r)
    return rows


def settled_payoff(s: pd.DataFrame) -> pd.Series:
    """What one Yes contract actually paid: 1 (yes), 0 (no) or the scalar settlement value."""
    pay = s["result"].map({"yes": 1.0, "no": 0.0})
    return pay.where(s["result"] != "scalar", s["settlement_value"]).astype(float)


def strategy(s: pd.DataFrame, prob_col: str, payoff_col: str, min_edge: float = 0.02, contracts: int = 100) -> dict:
    """The registered ``prop_strategy`` with the payoff taken from ``payoff_col`` (may be fractional)."""
    from sq.markets.odds import kalshi_fee
    from sq.research.nba_lines_registered import _cluster_boot_roi

    p = s[prob_col]
    yes_cost = s["ask_close"] + kalshi_fee(s["ask_close"], contracts) / contracts
    no_price = 1 - s["bid_close"]
    no_cost = no_price + kalshi_fee(no_price, contracts) / contracts
    e_yes, e_no = p - yes_cost, (1 - p) - no_cost
    yes = (e_yes >= e_no).to_numpy()
    edge = np.where(yes, e_yes, e_no)
    bet = edge >= min_edge
    cost = np.where(yes, yes_cost, no_cost)
    pay = np.where(yes, s[payoff_col], 1 - s[payoff_col])
    pnl = pay - cost
    if bet.sum() == 0:
        return {"n_bets": 0}
    lo, hi = _cluster_boot_roi(pnl[bet], cost[bet], s["game_id"].to_numpy()[bet])
    return {"n_bets": int(bet.sum()), "share_yes": float(np.mean(yes[bet])), "mean_pred_edge": float(edge[bet].mean()),
            "roi": float(pnl[bet].sum() / cost[bet].sum()), "roi_ci": [lo, hi],
            "n_scalar_bets": int((s["result"].to_numpy()[bet] == "scalar").sum())}


def yes_at_ask(s: pd.DataFrame, payoff_col: str) -> dict:
    from sq.markets.odds import kalshi_fee
    from sq.research.nba_lines_registered import _cluster_boot_roi

    cost = s["ask_close"] + kalshi_fee(s["ask_close"], 100) / 100
    pnl = s[payoff_col] - cost
    lo, hi = _cluster_boot_roi(pnl, cost, s["game_id"])
    return {"roi": float(pnl.sum() / cost.sum()), "roi_ci": [lo, hi],
            "mean_gap_vs_mid": float((s[payoff_col] - s["p_mkt"]).mean())}


def rescore(s: pd.DataFrame, frozen: dict, disp: float, check: dict | None = None) -> dict:
    """Section 4 of the module docstring, for one panel (discovery or replication)."""
    from sq.research.nba_lines_registered import combined_prop_probability, nb_sf

    s = s.copy()
    s["y_pay"] = settled_payoff(s)
    s["p_comb"] = combined_prop_probability(s, frozen)
    s["p_cond"] = np.clip(nb_sf(s["k"].to_numpy(), s["mu_pts"].to_numpy(), disp), 0.005, 0.995)
    sc = s["result"] == "scalar"
    dnp = s["min"].fillna(0) <= 0
    out = {"n": int(len(s)), "n_scalar": int(sc.sum()), "n_scalar_dnp": int((sc & dnp).sum()),
           "n_dnp_not_scalar": int((~sc & dnp).sum()),
           "n_scalar_listed_out": int((sc & (s["current_status"] == "Out")).sum()),
           "box_score_vs_settlement_agreement_binary": float((s.loc[~sc, "y"] == s.loc[~sc, "y_pay"]).mean()),
           "scalar_settlement_value_mean": float(s.loc[sc, "settlement_value"].mean()) if sc.any() else None,
           "scalar_mid_mean": float(s.loc[sc, "p_mkt"].mean()) if sc.any() else None,
           "scalar_spread_mean": float(s.loc[sc, "spread"].mean()) if sc.any() else None,
           "scalar_expiration_values": s.loc[sc, "expiration_value"].fillna("").value_counts().to_dict()}
    b = s[~sc]
    g2 = np.column_stack([_codes(b["game_id"]), _codes(b["person_id"])])
    enc = {}
    for name, col in (("registered_model", "p_model"), ("model_given_play", "p_cond")):
        bb = b.assign(p_model=b[col])
        rg, r2 = encompassing_clustered(bb, _codes(bb["game_id"])), encompassing_clustered(bb, g2)
        enc[name] = {"b_model": rg["b_model"], "se_game": rg["se_model"], "p_one_sided_game": rg["p_one_sided"],
                     "se_two_way": r2["se_model"], "p_one_sided_two_way": r2["p_one_sided"], "n": rg["n"],
                     "games": int(bb["game_id"].nunique())}
    out["encompassing_binary_settled"] = enc
    out["log_loss_binary_settled"] = {"market": M.log_loss(b["p_mkt"], b["y_pay"]),
                                      "model": M.log_loss(b["p_model"], b["y_pay"]),
                                      "frozen_combination": M.log_loss(b["p_comb"], b["y_pay"])}
    out["paired_combined_minus_market_binary_settled"] = M.paired_loss_test(b["p_comb"], b["p_mkt"], b["y_pay"],
                                                                            clusters=b["game_id"])
    out["yes_at_ask_settled"] = yes_at_ask(s, "y_pay")
    out["h3_combined_settled"] = strategy(s, "p_comb", "y_pay")
    out["h3b_model_settled"] = strategy(s, "p_model", "y_pay")
    # the same numbers with the registered box-score payoff (a player who sits = No)
    s["y_box"] = s["y"]
    e_box = M.encompassing_test(s["p_mkt"], s["p_model"], s["y"], clusters=s["game_id"])
    e_box["p_model_one_sided"] = float(norm.sf(e_box["b_model"] / e_box["se_model"]))
    out["box_score"] = {"encompassing": e_box, "yes_at_ask": yes_at_ask(s, "y_box"),
                        "h3_combined": strategy(s, "p_comb", "y_box"), "h3b_model": strategy(s, "p_model", "y_box"),
                        "paired_combined_minus_market": M.paired_loss_test(s["p_comb"], s["p_mkt"], s["y"],
                                                                           clusters=s["game_id"])}
    if check is not None:  # with the box-score payoff the re-implementation must reproduce the registered numbers
        for key, mine in (("strategy_h3_combined", "h3_combined"), ("strategy_h3", "h3b_model")):
            if key in check:
                r, c = out["box_score"][mine], check[key]
                assert r["n_bets"] == c["n_bets"] and abs(r["roi"] - c["roi"]) < 1e-12, key
        assert abs(out["box_score"]["yes_at_ask"]["roi"] - check["yes_at_ask"]["roi"]) < 1e-12
        out["reproduces_registered_numbers"] = True
    return out


def game_market_settlement_check() -> dict:
    """Do any scalar-settled Kalshi game-winner markets reach the moneyline evaluation?

    Uses the same event-to-game matching as ``quotes.kalshi_quotes``. A scalar-settled event that matches no
    played game never enters the evaluation; NFL ties are already excluded from scoring (``nfl_models``).
    """
    from sq.config import TIDY_DIR
    from sq.data import nba, nfl
    from sq.markets import quotes as Q

    out = {}
    for league, loader in (("nba", nba.load), ("nfl", nfl.load)):
        f = TIDY_DIR / f"kalshi_{league}_moneyline_markets.parquet"
        if not f.exists():
            continue
        km = pd.read_parquet(f)
        if league == "nfl":
            km["team_code"] = km["team_code"].replace(Q.KALSHI_NFL)
        sc = set(km.loc[km["result"] == "scalar", "event_ticker"])
        ev = km.groupby("event_ticker")["team_code"].agg(list).reset_index()
        ev = ev[ev["team_code"].map(len) == 2]
        ev = pd.DataFrame({"key": ev["event_ticker"], "date": ev["event_ticker"].map(Q._event_date),
                           "team_a": ev["team_code"].str[0], "team_b": ev["team_code"].str[1]})
        games = loader("games")
        hit = Q._match(ev[ev["key"].isin(sc)], games).dropna(subset=["game_id"])
        hit = hit.merge(games[["game_id", "home_pts", "away_pts"]], on="game_id")
        ties = hit["home_pts"] == hit["away_pts"]
        out[league] = {"scalar_settled_events": len(sc), "matched_to_played_games": int(len(hit)),
                       "matched_ties": int(ties.sum()), "matched_non_ties": int((~ties).sum()),
                       "matched_non_tie_games": hit.loc[~ties, "game_id"].astype(str).tolist()}
    return out


def registration_timing() -> dict:
    """Creation time of the registration file versus the props requests logged in the Kalshi fetch manifest."""
    from datetime import datetime, timezone

    from sq.config import RAW_DIR, ROOT

    st = (ROOT / "paper" / "preregistration_props.md").stat()
    t_reg = pd.Timestamp(datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_ctime), tz=timezone.utc))
    before = after = None
    with open(RAW_DIR / "kalshi" / "_manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if "KXNBAPTS" not in line:
                continue
            t = pd.Timestamp(json.loads(line)["fetched_at_utc"])
            if t < t_reg:
                before = t if before is None or t > before else before
            elif after is None or t < after:
                after = t
    fmt = "%Y-%m-%d %H:%M:%S UTC"
    return {"registration_file_created": t_reg.strftime(fmt),
            "last_props_request_before": before.strftime(fmt) if before is not None else None,
            "first_props_request_after": after.strftime(fmt) if after is not None else None,
            "seconds_after_registration": (after - t_reg).total_seconds() if after is not None else None}


def run() -> dict:
    rep_json = json.loads((REGDIR / "props_replication.json").read_text())
    disc_json = json.loads((REGDIR / "lines_props.json").read_text())["props"]
    rep = pd.read_parquet(REGDIR / "props_rep_panel.parquet")
    disc = pd.read_parquet(REGDIR / "props_panel.parquet")

    out: dict = {"registered": False,
                 "note": "Post-hoc diagnostics written after the replication results were seen.",
                 "registration_timing": registration_timing(),
                 "game_market_settlement": game_market_settlement_check()}

    # 1. inference under other clusterings, replication and (for comparison) discovery
    out["clustering_replication"] = clustering_table(rep)
    out["clustering_discovery"] = clustering_table(disc)

    # 2. where the information sits: players with and without an injury designation
    src = []
    for name, m in (("No injury designation (P(play) = 1)", rep["p_play"] >= 1.0),
                    ("Injury designation (P(play) < 1)", rep["p_play"] < 1.0)):
        sub = rep[m]
        if sub["game_id"].nunique() < 30:
            src.append({"subset": name, "n": int(len(sub)), "games": int(sub["game_id"].nunique()),
                        "skipped": "fewer than 30 games"})
            continue
        r_game = encompassing_clustered(sub, _codes(sub["game_id"]))
        r_two = encompassing_clustered(sub, np.column_stack([_codes(sub["game_id"]), _codes(sub["person_id"])]))
        src.append({"subset": name, "n": int(len(sub)), "games": int(sub["game_id"].nunique()),
                    "players": int(sub["person_id"].nunique()),
                    "b_model": r_game["b_model"], "se_game": r_game["se_model"],
                    "p_one_sided_game": r_game["p_one_sided"], "se_two_way": r_two["se_model"],
                    "p_one_sided_two_way": r_two["p_one_sided"],
                    "log_loss_market": M.log_loss(sub["p_mkt"], sub["y"]),
                    "log_loss_model": M.log_loss(sub["p_model"], sub["y"])})
    out["information_source"] = src

    # 3. facts behind the trading-rule results
    frozen = disc_json["encompassing"]
    out["trading_rules"] = {
        "h3_share_yes": rep_json["strategy_h3_combined"]["share_yes"],
        "h3_mean_predicted_edge": rep_json["strategy_h3_combined"]["mean_pred_edge"],
        "h3_hit_rate": rep_json["strategy_h3_combined"]["hit_rate"],
        "h3b_share_yes": rep_json["strategy_h3"]["share_yes"],
        "frozen_b_const": frozen["b_const"],
        "replication_b_const": rep_json["encompassing"]["b_const"],
        "frozen_b_model": frozen["b_model"],
        "replication_b_model": rep_json["encompassing"]["b_model"],
        "yes_premium_vs_mid_discovery": disc_json["yes_at_ask"]["mean_gap_vs_mid"],
        "yes_premium_vs_mid_replication": rep_json["yes_at_ask"]["mean_gap_vs_mid"],
        "half_spread_discovery": disc_json["yes_at_ask"]["mean_half_spread"],
        "half_spread_replication": rep_json["yes_at_ask"]["mean_half_spread"],
    }

    # 4. settlement-consistent rescoring of both samples
    fz = {k: frozen[k] for k in ("b_const", "b_market", "b_model")}
    out["settled_replication"] = rescore(rep, fz, rep_json["dispersion"], check=rep_json)
    out["settled_discovery"] = rescore(disc, fz, disc_json["dispersion"],
                                       check={"strategy_h3": disc_json["strategy_h3"],
                                              "yes_at_ask": disc_json["yes_at_ask"]})
    (OUTDIR / "props_rep_exploratory.json").write_text(json.dumps(out, indent=2, default=float))
    return out
