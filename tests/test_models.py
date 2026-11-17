import numpy as np
import pandas as pd
import pytest

from sq.backtest.engine import Config, run as bt_run
from sq.models.base import run_online
from sq.models.elo import Elo
from sq.models.kalman import KalmanScores
from sq.models.simulate import price_from_sims, simulate_nba, simulate_nfl, tilt, NBA_BASE
from tests.test_leakage import synthetic_league


def test_elo_is_zero_sum_and_probabilities_valid():
    g, _ = synthetic_league(seasons=(2020,))
    m = Elo.preset("nba")
    p = run_online(m, g)
    assert p["p_home"].between(0, 1).all()
    total = sum(m.r.values())
    assert total == pytest.approx(1505.0 * len(m.r), rel=1e-9)


def test_kalman_recovers_team_strengths():
    g, strength = synthetic_league(n_teams=16, games_per_day=8, seasons=(2020, 2021), days=120)
    m = KalmanScores.preset("nba", tau=5.0, carry=0.95, q_team=1e-4)
    run_online(m, g)
    est = np.array([m.x[m.idx[t]] + m.x[m.n + m.idx[t]] for t in range(16)])   # net = off + def
    assert np.corrcoef(est, strength)[0, 1] > 0.9
    assert m.x[-1] == pytest.approx(3.0, abs=0.8)   # home advantage (1.5 + 1.5 points)


def test_tilt_hits_target_mean():
    p = tilt(NBA_BASE, 1.20)
    assert p @ np.arange(5) == pytest.approx(1.20, abs=1e-9)


def test_nba_simulator_means_and_monotonicity():
    s = simulate_nba(116.0, 110.0, n_sims=8000, seed=1)
    pr = price_from_sims(s, spread=-6.0, total=226.0)
    assert pr["mu_total"] == pytest.approx(226.0, abs=2.5)
    assert 0.6 < pr["p_home"] < 0.8
    s2 = simulate_nba(120.0, 106.0, n_sims=8000, seed=1)
    assert price_from_sims(s2)["p_home"] > pr["p_home"]


def test_nfl_simulator_key_numbers():
    s = simulate_nfl(24.0, 21.0, n_sims=30000, seed=2)
    m = s["home"] - s["away"]
    share3 = np.mean(np.abs(m) == 3)
    share5 = np.mean(np.abs(m) == 5)
    assert share3 > 2 * share5          # 3 is a key number, 5 is not


def test_kelly_sizing_and_caps():
    bets = pd.DataFrame({"game_id": [1], "date": [pd.Timestamp("2025-01-01")], "p_model": [0.60],
                         "q_home": [0.50], "q_away": [0.52], "y": [1.0], "p_close": [0.55]})
    cfg = Config(kelly_fraction=0.25, max_bet_frac=0.10, min_edge=0.0, fee_model="none", liquidity_frac=None)
    led, dd, s = bt_run(bets, cfg)
    # f* = (0.6-0.5)/(1-0.5) = 0.2 ; quarter Kelly = 5% of 10,000
    assert led["stake"].iloc[0] == pytest.approx(500.0)
    assert led["pnl"].iloc[0] == pytest.approx(500.0)       # 1000 contracts * (1 - 0.5)
    assert led["clv"].iloc[0] == pytest.approx(0.05)
    cfg2 = Config(kelly_fraction=0.25, max_bet_frac=0.02, min_edge=0.0, fee_model="none", liquidity_frac=None)
    led2, _, _ = bt_run(bets, cfg2)
    assert led2["stake"].iloc[0] == pytest.approx(200.0)
