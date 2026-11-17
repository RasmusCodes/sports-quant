"""Snapshot of current (open) game markets for the dashboard's upcoming-games view.

Kalshi: /markets?series_ticker=KXNBAGAME&status=open (yes bid/ask per team market).
Polymarket: gamma /markets?sports_market_types=moneyline&closed=false (best bid/ask).
Each snapshot is cached under its UTC timestamp, so the dashboard states exactly
when the market prices were observed.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from ..config import TIDY_DIR
from .http import Fetcher
from .kalshi import API as KAPI, SERIES, _num
from .polymarket import GAMMA, TAG, _jl


def snapshot(league: str) -> pd.DataFrame:
    now = dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    rows = []
    fk = Fetcher("kalshi", min_interval=0.3)
    j = fk.get_json(KAPI + "/markets", {"series_ticker": SERIES[(league, "moneyline")], "status": "open",
                                        "limit": 1000}, name=f"live/{league}_{stamp}")
    for m in j.get("markets", []):
        rows.append({"venue": "kalshi", "event": m["event_ticker"], "team_code": m["ticker"].rsplit("-", 1)[-1],
                     "bid": _num(m.get("yes_bid_dollars")), "ask": _num(m.get("yes_ask_dollars")),
                     "expected_expiration_time": m.get("expected_expiration_time"), "observed_utc": now.isoformat()})
    fp = Fetcher("polymarket", min_interval=0.3)
    j = fp.get_json(GAMMA + "/markets", {"sports_market_types": "moneyline", "tag_id": TAG[league], "closed": "false",
                                         "limit": 500}, name=f"live/{league}_{stamp}")
    for m in j or []:
        outs, px = _jl(m.get("outcomes")), _jl(m.get("outcomePrices"))
        if len(outs) != 2:
            continue
        rows.append({"venue": "polymarket", "event": m.get("slug"), "outcome0": outs[0], "outcome1": outs[1],
                     "p0": float(px[0]) if px else None, "bid": _num(m.get("bestBid")), "ask": _num(m.get("bestAsk")),
                     "game_start_time": m.get("gameStartTime"), "observed_utc": now.isoformat()})
    df = pd.DataFrame(rows)
    df.to_parquet(TIDY_DIR / f"live_{league}.parquet", index=False)
    return df
