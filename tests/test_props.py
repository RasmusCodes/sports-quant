"""Props payoff handling: Kalshi settles a prop on a player who does not play at a scalar fair price, not No."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from sq.research import nba_lines
from sq.research import nba_lines_registered as registered
from sq.research.props_exploratory import settled_payoff, strategy

prop_strategy = registered.prop_strategy          # the registered (box-score) rule
REGISTERED_SHA256 = "9b38870759d472ef804b766d28202d0e9cc2e63cf414d2d86e3a5be4858cdf30"


def _panel(n=400, seed=3):
    rng = np.random.default_rng(seed)
    mid = rng.uniform(0.05, 0.95, n)
    half = rng.uniform(0.005, 0.04, n)
    y = (rng.uniform(size=n) < mid).astype(float)
    return pd.DataFrame({
        "game_id": rng.integers(0, 40, n).astype(str),
        "p_mkt": mid, "bid_close": np.clip(mid - half, 0.01, 0.99), "ask_close": np.clip(mid + half, 0.01, 0.99),
        "p_model": np.clip(mid + rng.normal(0, 0.08, n), 0.005, 0.995),
        "y": y, "result": np.where(y == 1, "yes", "no"), "settlement_value": np.nan,
    })


def test_settled_payoff_uses_scalar_value():
    s = pd.DataFrame({"result": ["yes", "no", "scalar"], "settlement_value": [np.nan, np.nan, 0.43]})
    assert settled_payoff(s).tolist() == [1.0, 0.0, 0.43]


def test_strategy_matches_registered_rule_on_binary_payoffs():
    s = _panel()
    s["y_pay"] = settled_payoff(s)
    mine, reg = strategy(s, "p_model", "y_pay"), prop_strategy(s)
    assert mine["n_bets"] == reg["n_bets"] and abs(mine["roi"] - reg["roi"]) < 1e-12


def test_scalar_settlement_changes_the_payoff_not_the_bets():
    s = _panel()
    s.loc[:9, "result"] = "scalar"
    s.loc[:9, "settlement_value"] = 0.5
    s.loc[:9, "y"] = 0.0                      # what a box score says about a player who sat
    s["y_pay"] = settled_payoff(s)
    box, settled = strategy(s.assign(y_box=s["y"]), "p_model", "y_box"), strategy(s, "p_model", "y_pay")
    assert box["n_bets"] == settled["n_bets"]
    assert settled["n_scalar_bets"] > 0 and box["roi"] != settled["roi"]


def test_registered_code_is_byte_identical_to_registration():
    path = Path(registered.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REGISTERED_SHA256


def test_default_payoff_comes_from_settlement():
    s = pd.DataFrame({"result": ["yes", "no", "scalar", "void"], "settlement_value": [np.nan, np.nan, 0.61, np.nan]})
    pay = nba_lines.settled_payoff(s)
    assert pay.iloc[:3].tolist() == [1.0, 0.0, 0.61] and np.isnan(pay.iloc[3])


def test_default_rule_reduces_to_registered_rule_on_binary_payoffs():
    s = _panel()
    new, old = nba_lines.prop_strategy(s, payoff_col="y"), prop_strategy(s)
    assert new["n_bets"] == old["n_bets"] and abs(new["roi"] - old["roi"]) < 1e-12
    assert abs(new["hit_rate"] - old["hit_rate"]) < 1e-12


def test_default_rule_pays_scalar_settlements_at_their_value():
    s = _panel()
    s.loc[:9, "result"] = "scalar"
    s.loc[:9, "settlement_value"] = 0.5
    s["y_pay"] = nba_lines.settled_payoff(s)
    a, b = nba_lines.prop_strategy(s), strategy(s, "p_model", "y_pay")
    assert a["n_bets"] == b["n_bets"] and abs(a["roi"] - b["roi"]) < 1e-12 and a["n_scalar_bets"] > 0
