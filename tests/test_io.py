import pandas as pd
import pytest

from sq.io.kalshi import norm_candle
from sq.io.polymarket import market_row
from sq.markets.quotes import _event_date, _match


def test_polymarket_line_comes_from_question_not_field():
    m = {"id": "1", "question": "Spread: Thunder (-9.5)", "line": 9.5, "outcomes": '["Thunder", "Suns"]',
         "clobTokenIds": '["11", "22"]', "outcomePrices": '["1", "0"]', "events": [{"slug": "nba-phx-okc-2025-11-01"}],
         "gameStartTime": "2025-11-02 01:00:00+00"}
    assert market_row(m, "nba", "spreads")["line"] == pytest.approx(-9.5)
    m2 = dict(m, question="Spurs vs. Knicks: O/U 216.5", line=None, outcomes='["Over", "Under"]')
    assert market_row(m2, "nba", "totals")["line"] == pytest.approx(216.5)


def test_kalshi_candle_schemas_are_normalised():
    live = {"end_period_ts": 1, "price": {"close_dollars": "0.38"}, "yes_bid": {"close_dollars": "0.36"},
            "yes_ask": {"close_dollars": "0.38"}, "volume_fp": "10.5", "open_interest_fp": "3"}
    hist = {"end_period_ts": 1, "price": {"close": "0.38"}, "yes_bid": {"close": "0.36"}, "yes_ask": {"close": "0.38"},
            "volume": "10.5", "open_interest": "3"}
    assert norm_candle(live) == norm_candle(hist)
    assert norm_candle(live)["bid_close"] == pytest.approx(0.36)


def test_event_date_and_matching_by_unordered_team_pair():
    assert _event_date("KXNBAGAME-26NOV12CHAPHX") == pd.Timestamp("2026-11-12")
    games = pd.DataFrame({"game_id": ["g1"], "date": [pd.Timestamp("2026-11-12")], "home_abbr": ["PHX"],
                          "away_abbr": ["CHA"], "tip_utc": [pd.Timestamp("2026-11-13 02:00", tz="UTC")]})
    ev = pd.DataFrame({"key": ["e1", "e2"], "date": [pd.Timestamp("2026-11-12"), pd.Timestamp("2026-11-12")],
                       "team_a": ["CHA", "CHA"], "team_b": ["PHX", "BOS"]})
    m = _match(ev, games)
    assert list(m["key"]) == ["e1"] and m["game_id"].iloc[0] == "g1"
