"""Kalshi public market-data API (CFTC-regulated exchange; no authentication needed).

Endpoints used (all GET, read-only):
  live API       /markets?series_ticker=..&status=settled          markets settled after the cutoff
                 /series/{series}/markets/{ticker}/candlesticks
  historical API /historical/cutoff
                 /historical/markets?series_ticker=..              markets settled before the cutoff
                 /historical/markets/{ticker}/candlesticks
Candles carry trade-price OHLC plus yes-bid / yes-ask OHLC, volume and open interest.
The two APIs spell fields differently (``close_dollars`` vs ``close``); ``norm_candle``
maps both onto one schema.

Per market we keep an hourly series over the whole listing life (opening price,
line movement) and a 1-minute series over roughly [tip - 3.5h, tip + 0.5h]
(closing price). Kalshi sets ``expected_expiration_time`` = scheduled start + 3h for
these game markets, which is used to place the minute window before the schedule
join; the exact tip-off time comes from the league schedule later.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable

import pandas as pd

from ..config import TIDY_DIR
from .http import FetchError, Fetcher

API = "https://api.elections.kalshi.com/trade-api/v2"
log = logging.getLogger(__name__)

SERIES = {
    ("nba", "moneyline"): "KXNBAGAME",
    ("nfl", "moneyline"): "KXNFLGAME",
    ("nba", "spread"): "KXNBASPREAD",
    ("nba", "total"): "KXNBATOTAL",
    ("nfl", "spread"): "KXNFLSPREAD",
    ("nfl", "total"): "KXNFLTOTAL",
    ("nba", "player_points"): "KXNBAPTS",
    ("nba", "player_rebounds"): "KXNBAREB",
    ("nba", "player_assists"): "KXNBAAST",
    ("nba", "player_threes"): "KXNBA3PT",
}


def _ts(s: str) -> int:
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _paginate(f: Fetcher, path: str, params: dict, tag: str) -> list[dict]:
    """Cursor pagination; pages cached under a date stamp so same-day reruns are free."""
    stamp = dt.date.today().strftime("%Y%m%d")
    out, cursor, page = [], None, 0
    while True:
        p = dict(params, cursor=cursor) if cursor else dict(params)
        j = f.get_json(API + path, p, name=f"lists/{tag}_{stamp}_{page:03d}")
        ms = j.get("markets", [])
        out.extend(ms)
        cursor = j.get("cursor")
        page += 1
        if not cursor or not ms:
            return out


def list_markets(f: Fetcher, series: str) -> list[dict]:
    hist = _paginate(f, "/historical/markets", {"series_ticker": series, "limit": 1000}, f"{series}_hist")
    live = _paginate(f, "/markets", {"series_ticker": series, "status": "settled", "limit": 1000}, f"{series}_live")
    for m in hist:
        m["_api"] = "historical"
    for m in live:
        m["_api"] = "live"
    seen, out = set(), []
    for m in live + hist:  # live wins if a ticker appears in both
        if m["ticker"] not in seen:
            seen.add(m["ticker"])
            out.append(m)
    return out


def norm_candle(c: dict) -> dict:
    price, bid, ask = c.get("price") or {}, c.get("yes_bid") or {}, c.get("yes_ask") or {}

    def g(d, k):
        return _num(d.get(k + "_dollars", d.get(k)))

    return {
        "end_ts": int(c["end_period_ts"]),
        "p_open": g(price, "open"), "p_high": g(price, "high"), "p_low": g(price, "low"),
        "p_close": g(price, "close"), "p_mean": g(price, "mean"), "p_prev": g(price, "previous"),
        "bid_open": g(bid, "open"), "bid_high": g(bid, "high"), "bid_low": g(bid, "low"), "bid_close": g(bid, "close"),
        "ask_open": g(ask, "open"), "ask_high": g(ask, "high"), "ask_low": g(ask, "low"), "ask_close": g(ask, "close"),
        "volume": _num(c.get("volume_fp", c.get("volume"))),
        "open_interest": _num(c.get("open_interest_fp", c.get("open_interest"))),
    }


def candles(f: Fetcher, series: str, m: dict, start: int, end: int, interval: int) -> list[dict]:
    t = m["ticker"]
    if m.get("_api") == "historical":
        path = f"/historical/markets/{t}/candlesticks"
    else:
        path = f"/series/{series}/markets/{t}/candlesticks"
    name = f"candles/{series}/{t}_{interval}_{start}_{end}"
    j = f.get_json(API + path, {"start_ts": start, "end_ts": end, "period_interval": interval},
                   name=name, allow_404=True)
    if not j:
        return []
    return [norm_candle(c) for c in j.get("candlesticks", [])]


MARKET_FIELDS = ["ticker", "event_ticker", "title", "yes_sub_title", "no_sub_title", "open_time", "close_time",
                 "expected_expiration_time", "occurrence_datetime", "settlement_ts", "result", "expiration_value",
                 "status", "market_type", "strike_type", "floor_strike", "cap_strike", "functional_strike",
                 "rules_primary", "_api"]


def market_row(m: dict, league: str, market: str, series: str) -> dict:
    row = {k: m.get(k) for k in MARKET_FIELDS}
    row.update(league=league, market=market, series=series,
               volume=_num(m.get("volume_fp", m.get("volume"))),
               open_interest=_num(m.get("open_interest_fp", m.get("open_interest"))),
               last_price=_num(m.get("last_price_dollars", m.get("last_price"))),
               settlement_value=_num(m.get("settlement_value_dollars", m.get("settlement_value"))),
               team_code=m["ticker"].rsplit("-", 1)[-1])
    return row


def collect(league: str, market: str = "moneyline", minute_window: bool = True,
            hourly: bool = True, max_markets: int | None = None, sample_frac: float | None = None,
            seed: int = 7, minute_offsets_h: tuple[float, float] | None = None,
            min_volume: float = 0.0, exclude_events: set | None = None,
            out_suffix: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect markets + candles for one (league, market) series. Resumable via the raw cache.

    ``minute_offsets_h`` = (a, b): the 1-minute window is [exp - a h, exp - b h], where exp is the
    market's expected expiration. Game markets and most player props (84% of a sampled 1,800) use
    exp = tip + 3h, so (6.5, 2.5) covers tip - 3.5h to tip + 0.5h. A minority of props use
    exp = tip + 6h; for those the window still contains the 30 minutes before tip-off.

    ``exclude_events`` drops whole events (e.g. a discovery sample, for a held-out replication);
    ``out_suffix`` writes to kalshi_<league>_<market>_<suffix>_*.parquet so earlier files are kept.
    """
    series = SERIES[(league, market)]
    if minute_offsets_h is None:
        minute_offsets_h = (6.5, 2.5)
    f = Fetcher("kalshi", min_interval=0.2)
    ms = list_markets(f, series)
    ms = [m for m in ms if m.get("result") in ("yes", "no") or m.get("status") in ("settled", "finalized")]
    if min_volume:
        ms = [m for m in ms if (_num(m.get("volume_fp", m.get("volume"))) or 0) >= min_volume]
    if sample_frac:
        # sample whole events so both sides / all strikes of a game stay together
        evs = sorted({m["event_ticker"] for m in ms})
        keep = set(pd.Series(evs).sample(frac=sample_frac, random_state=seed))
        ms = [m for m in ms if m["event_ticker"] in keep]
    if exclude_events:
        ms = [m for m in ms if m["event_ticker"] not in exclude_events]
    if max_markets:
        ms = ms[:max_markets]
    log.info("%s: %d markets", series, len(ms))
    mrows, crow = [], []
    for i, m in enumerate(ms):
        mrows.append(market_row(m, league, market, series))
        try:
            exp = _ts(m.get("expected_expiration_time") or m["close_time"])
            close = _ts(m["close_time"])
            if hourly:
                start = (_ts(m["open_time"]) // 3600) * 3600
                end = min(close, exp + 3600)
                for c in candles(f, series, m, start, end, 60):
                    crow.append({"ticker": m["ticker"], "interval": 60, **c})
            if minute_window:
                a, b = minute_offsets_h
                start, end = exp - int(a * 3600), min(close, exp - int(b * 3600))
                if end > start:
                    for c in candles(f, series, m, start, end, 1):
                        crow.append({"ticker": m["ticker"], "interval": 1, **c})
        except FetchError as exc:
            log.warning("candles failed for %s: %s", m["ticker"], exc)
        if (i + 1) % 200 == 0:
            log.info("  %d/%d markets (network %d, cache %d)", i + 1, len(ms), f.n_network, f.n_cache)
    mk = pd.DataFrame(mrows)
    cd = pd.DataFrame(crow)
    if not cd.empty:
        cd["ts_utc"] = pd.to_datetime(cd["end_ts"], unit="s", utc=True)
    tag = f"kalshi_{league}_{market}" + (f"_{out_suffix}" if out_suffix else "")
    mk.to_parquet(TIDY_DIR / f"{tag}_markets.parquet", index=False)
    cd.to_parquet(TIDY_DIR / f"{tag}_candles.parquet", index=False)
    log.info("wrote %s: %d markets, %d candles", tag, len(mk), len(cd))
    return mk, cd


def collect_many(specs: Iterable[tuple[str, str]], **kw) -> None:
    for league, market in specs:
        collect(league, market, **kw)
