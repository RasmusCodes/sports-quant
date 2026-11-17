"""Information-leakage tests.

The central check: change every result after a cut-off date and confirm that no
prediction for a game on or before that date moves. Same for same-day results,
walk-forward ML, injury reports published after the decision time and the
backtester's decision inputs.
"""
import numpy as np
import pandas as pd
import pytest

from sq.backtest.engine import Config, run as bt_run
from sq.features.players import availability_features
from sq.models.base import run_online
from sq.models.elo import Elo
from sq.models.glicko import Glicko2
from sq.models.kalman import KalmanScores
from sq.models.ml import walk_forward_logistic
from sq.models.poisson import PoissonTeam


def synthetic_league(n_teams=8, seasons=(2020, 2021), games_per_day=4, days=60, seed=0):
    rng = np.random.default_rng(seed)
    off, dfn = rng.normal(0, 3, n_teams), rng.normal(0, 3, n_teams)
    strength = off + dfn          # net rating: contribution to the point margin
    rows, gid = [], 0
    for s in seasons:
        start = pd.Timestamp(f"{s - 1}-11-01")
        for d in range(days):
            date = start + pd.Timedelta(days=d)
            teams = rng.permutation(n_teams)
            for k in range(games_per_day):
                h, a = int(teams[2 * k]), int(teams[2 * k + 1])
                mu_h, mu_a = 105 + 1.5 + off[h] - dfn[a], 105 - 1.5 + off[a] - dfn[h]
                hp, ap = rng.normal(mu_h, 11), rng.normal(mu_a, 11)
                if round(hp) == round(ap):
                    hp += 1
                rows.append({"game_id": f"g{gid:05d}", "season": s, "date": date,
                             "tip_utc": (date + pd.Timedelta(hours=19 + k)).tz_localize("UTC"),
                             "home": h, "away": a, "neutral": False,
                             "home_pts": float(round(hp)), "away_pts": float(round(ap))})
                gid += 1
    return pd.DataFrame(rows), strength


MODELS = [lambda: Elo.preset("nba"), lambda: Glicko2(), lambda: KalmanScores.preset("nba"),
          lambda: PoissonTeam(min_games=20)]


@pytest.mark.parametrize("make", MODELS)
def test_future_results_do_not_move_past_predictions(make):
    g, _ = synthetic_league()
    cut = g["date"].iloc[len(g) // 2]
    a = run_online(make(), g).set_index("game_id")["p_home"]
    g2 = g.copy()
    future = g2["date"] > cut
    g2.loc[future, ["home_pts", "away_pts"]] = g2.loc[future, ["away_pts", "home_pts"]].to_numpy() + [[0, 7]]
    b = run_online(make(), g2).set_index("game_id")["p_home"]
    past = g.loc[g["date"] <= cut, "game_id"]
    assert np.allclose(a.loc[past], b.loc[past], atol=0, rtol=0)
    assert not np.allclose(a.loc[g.loc[future, "game_id"]], b.loc[g.loc[future, "game_id"]])


@pytest.mark.parametrize("make", MODELS)
def test_same_day_results_are_not_used(make):
    g, _ = synthetic_league()
    day = g["date"].iloc[len(g) // 2]
    same = g[g["date"] == day]
    first, others = same["game_id"].iloc[0], same["game_id"].iloc[1:]
    a = run_online(make(), g).set_index("game_id")["p_home"]
    g2 = g.copy()
    g2.loc[g2["game_id"] == first, ["home_pts", "away_pts"]] = [80.0, 140.0]
    b = run_online(make(), g2).set_index("game_id")["p_home"]
    assert np.allclose(a.loc[others], b.loc[others], atol=0, rtol=0)


def test_walk_forward_ml_ignores_test_season_outcomes():
    rng = np.random.default_rng(5)
    n = 3000
    df = pd.DataFrame({"game_id": np.arange(n), "season": np.repeat([2018, 2019, 2020, 2021, 2022, 2023], n // 6),
                       "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * df["x1"] - 0.3 * df["x2"])))).astype(float)
    a = walk_forward_logistic(df, ["x1", "x2"], [2022, 2023]).set_index("game_id")["p_logit"]
    df2 = df.copy()
    df2.loc[df2["season"] == 2023, "y"] = 1 - df2.loc[df2["season"] == 2023, "y"]
    b = walk_forward_logistic(df2, ["x1", "x2"], [2022, 2023]).set_index("game_id")["p_logit"]
    assert np.allclose(a, b)  # 2022 and 2023 predictions never see 2023 outcomes


def _availability_fixture():
    g = pd.DataFrame({"game_id": ["0001", "0002"], "season": [2024, 2024],
                      "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
                      "tip_utc": pd.to_datetime(["2024-01-02 00:00", "2024-01-04 00:00"], utc=True),
                      "home": [1, 1], "away": [2, 2], "home_pts": [100.0, np.nan], "away_pts": [90.0, np.nan]})
    pb = []
    for gid in ["0001"]:
        for team, pids in ((1, [10, 11, 12, 13, 14]), (2, [20, 21, 22, 23, 24])):
            for pid in pids:
                pb.append({"game_id": gid, "season": 2024, "team_id": team, "person_id": pid, "comment": "",
                           "min": 30.0, "pts": 20 if pid == 10 else 10, "fgm": 7, "fga": 14, "ftm": 3, "fta": 4,
                           "oreb": 1, "dreb": 4, "ast": 3, "stl": 1, "blk": 0, "tov": 2, "pf": 2})
    pb = pd.DataFrame(pb)
    inj = pd.DataFrame({"game_id": ["0002", "0002"], "team_id": [1, 1], "person_id": [10, 10],
                        "current_status": ["Questionable", "Out"],
                        "report_utc": pd.to_datetime(["2024-01-03 12:00", "2024-01-03 23:58"], utc=True)})
    return g, pb, inj


def test_injury_reports_after_decision_time_are_ignored():
    g, pb, inj = _availability_fixture()
    f = availability_features(g, pb, inj, decision_offsets_min=(5, 360)).set_index("game_id")
    # at T-6h only the 'Questionable' report exists; at T-5m only reports up to 23:55 count
    assert f.loc["0002", "home_src_360"] == "report"
    q = f.loc["0002", "home_lost_pit_360"]
    late = f.loc["0002", "home_lost_pit_5"]
    assert q > 0 and late == pytest.approx(q)   # the 23:58 'Out' report is after tip-5min
    inj2 = inj.copy()
    inj2.loc[1, "report_utc"] = pd.Timestamp("2024-01-03 23:50", tz="UTC")
    f2 = availability_features(g, pb, inj2, decision_offsets_min=(5, 360)).set_index("game_id")
    assert f2.loc["0002", "home_lost_pit_5"] > late      # now the Out report is in time
    assert f2.loc["0002", "home_lost_pit_360"] == pytest.approx(q)


def test_backtest_decisions_do_not_use_closing_prices():
    rng = np.random.default_rng(9)
    n = 400
    bets = pd.DataFrame({"game_id": np.arange(n), "date": np.repeat(pd.date_range("2025-01-01", periods=40), 10),
                         "p_model": rng.uniform(0.3, 0.7, n), "q_home": rng.uniform(0.35, 0.65, n)})
    bets["q_away"] = 1.02 - bets["q_home"]
    bets["y"] = (rng.random(n) < bets["p_model"]).astype(float)
    bets["p_close"] = rng.uniform(0.3, 0.7, n)
    cfg = Config(fee_model="none", liquidity_frac=None)
    l1, _, _ = bt_run(bets, cfg)
    bets2 = bets.assign(p_close=rng.uniform(0.3, 0.7, n))
    l2, _, _ = bt_run(bets2, cfg)
    assert l1[["game_id", "side", "stake", "pnl"]].equals(l2[["game_id", "side", "stake", "pnl"]])
