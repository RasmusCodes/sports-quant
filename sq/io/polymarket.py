"""Polymarket public APIs (Gamma market metadata + CLOB price history; no authentication).

Endpoints used (GET, read-only):
  gamma  /markets?sports_market_types=moneyline|spreads|totals&tag_id=<league>&closed=true
  clob   /prices-history?market=<token id>&startTs=..&endTs=..&fidelity=<minutes>
A binary Polymarket market has one order book; the two outcome tokens are
complements, so the history of token 0 gives P(outcome 0). The history series is
Polymarket's own price point (no bid/ask), stored as ``p``.

Listing is done in calendar-month windows on ``end_date`` so offset paging stays
shallow.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re

import pandas as pd

from ..config import TIDY_DIR
from .http import FetchError, Fetcher

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
TAG = {"nba": 745, "nfl": 450}
log = logging.getLogger(__name__)


def _months(start: dt.date, end: dt.date):
    d = dt.date(start.year, start.month, 1)
    while d <= end:
        nxt = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        yield d, nxt
        d = nxt


def list_markets(f: Fetcher, league: str, smt: str, start=dt.date(2023, 8, 1), end=None) -> list[dict]:
    end = end or dt.date.today()
    today = dt.date.today()
    out = []
    for a, b in _months(start, end):
        off = 0
        # closed months never change: cache permanently; the current month is re-listed daily
        stamp = "final" if b <= today - dt.timedelta(days=3) else today.strftime("%Y%m%d")
        while True:
            params = {"sports_market_types": smt, "tag_id": TAG[league], "closed": "true", "limit": 500,
                      "offset": off, "end_date_min": a.isoformat() + "T00:00:00Z",
                      "end_date_max": b.isoformat() + "T00:00:00Z"}
            j = f.get_json(GAMMA + "/markets", params, name=f"lists/{league}_{smt}_{a:%Y%m}_{stamp}_{off:05d}")
            if not j:
                break
            out.extend(j)
            if len(j) < 500:
                break
            off += 500
    seen, uniq = set(), []
    for m in out:
        if m.get("id") not in seen:
            seen.add(m.get("id"))
            uniq.append(m)
    return uniq


def _jl(x):
    if isinstance(x, list):
        return x
    try:
        return json.loads(x) if x else []
    except (TypeError, ValueError):
        return []


LINE_RE = re.compile(r"\(([-+]?\d+(?:\.\d+)?)\)|O/U\s*([-+]?\d+(?:\.\d+)?)")


def market_row(m: dict, league: str, smt: str) -> dict:
    outs, toks, px = _jl(m.get("outcomes")), _jl(m.get("clobTokenIds")), _jl(m.get("outcomePrices"))
    ev = (m.get("events") or [{}])[0]
    # the question text is authoritative: the API's ``line`` field sometimes drops the sign
    # ("Spread: Thunder (-9.5)" with line = 9.5), which silently flips the handicap
    line = None
    mm = LINE_RE.search(m.get("question") or "")
    if mm:
        line = float(mm.group(1) or mm.group(2))
    elif m.get("line") is not None:
        line = float(m["line"])
    return {
        "id": m.get("id"), "slug": m.get("slug"), "question": m.get("question"), "league": league,
        "market": smt, "event_slug": ev.get("slug"), "event_title": ev.get("title"),
        "game_start_time": m.get("gameStartTime"), "start_date": m.get("startDate"), "end_date": m.get("endDate"),
        "closed_time": m.get("closedTime"), "volume": m.get("volumeNum"), "line": line,
        "outcome0": outs[0] if len(outs) > 0 else None, "outcome1": outs[1] if len(outs) > 1 else None,
        "token0": toks[0] if len(toks) > 0 else None, "token1": toks[1] if len(toks) > 1 else None,
        "final_price0": float(px[0]) if len(px) > 0 else None,
        "resolution_status": m.get("umaResolutionStatus"),
    }


def _ts(s):
    if not isinstance(s, str) or not s:
        return None
    s = s.replace(" ", "T")
    if s.endswith("+00"):
        s += ":00"
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


MAX_SPAN = 7 * 86400  # the CLOB rejects long startTs..endTs windows; request in 7-day chunks


def history(f: Fetcher, token: str, start: int, end: int, fidelity: int) -> list[dict]:
    out = []
    a = start
    while a < end:
        b = min(end, a + MAX_SPAN)
        name = f"prices/{token[:16]}_{token[-8:]}_{fidelity}_{a}_{b}"
        j = f.get_json(CLOB + "/prices-history", {"market": token, "startTs": a, "endTs": b,
                                                   "fidelity": fidelity}, name=name, allow_404=True)
        out.extend((j or {}).get("history", []))
        a = b
    seen, uniq = set(), []
    for h in out:
        if h["t"] not in seen:
            seen.add(h["t"])
            uniq.append(h)
    return uniq


def collect(league: str, smt: str = "moneyline", hourly: bool = True, minute_window: bool = True,
            top_lines_per_game: int | None = None, start=dt.date(2023, 8, 1)) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = Fetcher("polymarket", min_interval=0.25)
    ms = list_markets(f, league, smt, start=start)
    rows = [market_row(m, league, smt) for m in ms]
    mk = pd.DataFrame(rows)
    mk = mk[mk["token0"].notna() & mk["game_start_time"].notna()].copy()
    if top_lines_per_game and not mk.empty:
        mk = (mk.sort_values("volume", ascending=False).groupby("event_slug", dropna=False)
                .head(top_lines_per_game))
    log.info("polymarket %s %s: %d markets", league, smt, len(mk))
    prow = []
    for i, r in enumerate(mk.itertuples(index=False)):
        tip = _ts(r.game_start_time)
        if tip is None:
            continue
        try:
            if hourly:
                st = _ts(r.start_date) or tip - 7 * 86400
                st = max(st, tip - 21 * 86400)
                st = (st // 3600) * 3600
                if st < tip:
                    for h in history(f, r.token0, st, tip + 1800, 60):
                        prow.append({"id": r.id, "fidelity": 60, "t": int(h["t"]), "p": float(h["p"])})
            if minute_window:
                for h in history(f, r.token0, tip - int(3.5 * 3600), tip + 1800, 1):
                    prow.append({"id": r.id, "fidelity": 1, "t": int(h["t"]), "p": float(h["p"])})
        except FetchError as exc:
            log.warning("history failed for %s: %s", r.slug, exc)
        if (i + 1) % 200 == 0:
            log.info("  %d/%d (network %d, cache %d)", i + 1, len(mk), f.n_network, f.n_cache)
    px = pd.DataFrame(prow)
    if not px.empty:
        px["ts_utc"] = pd.to_datetime(px["t"], unit="s", utc=True)
    tag = f"poly_{league}_{smt}"
    mk.to_parquet(TIDY_DIR / f"{tag}_markets.parquet", index=False)
    px.to_parquet(TIDY_DIR / f"{tag}_prices.parquet", index=False)
    log.info("wrote %s: %d markets, %d price points", tag, len(mk), len(px))
    return mk, px
