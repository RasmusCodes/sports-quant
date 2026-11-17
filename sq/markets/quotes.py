"""Turn venue price histories into game-level market quotes.

Kalshi lists one binary market per team ("Toronto wins", "Philadelphia wins"), each
with its own book. At every candle we keep both books and derive
  q_home  = min(ask_home, 1 - bid_away)   cheapest way to back the home team
  q_away  = min(ask_away, 1 - bid_home)
  overround = q_home + q_away - 1          the exchange analogue of a bookmaker margin
  p_mid   = (mid_home + 1 - mid_away) / 2  mid-price fair probability
Polymarket has a single book per game; its history gives one price (treated as a
mid). Every quote is matched to a league game and stamped with minutes-to-tip.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from ..config import TIDY_DIR

log = logging.getLogger(__name__)
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
EV_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")

# Kalshi NFL codes that differ from nflverse
KALSHI_NFL = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS"}

NBA_NICK = {
    "hawks": "ATL", "celtics": "BOS", "nets": "BKN", "hornets": "CHA", "bulls": "CHI", "cavaliers": "CLE",
    "mavericks": "DAL", "nuggets": "DEN", "pistons": "DET", "warriors": "GSW", "rockets": "HOU", "pacers": "IND",
    "clippers": "LAC", "lakers": "LAL", "grizzlies": "MEM", "heat": "MIA", "bucks": "MIL", "timberwolves": "MIN",
    "pelicans": "NOP", "knicks": "NYK", "thunder": "OKC", "magic": "ORL", "76ers": "PHI", "sixers": "PHI",
    "suns": "PHX", "trail blazers": "POR", "blazers": "POR", "kings": "SAC", "spurs": "SAS", "raptors": "TOR",
    "jazz": "UTA", "wizards": "WAS",
}
NFL_NICK = {
    "cardinals": "ARI", "falcons": "ATL", "ravens": "BAL", "bills": "BUF", "panthers": "CAR", "bears": "CHI",
    "bengals": "CIN", "browns": "CLE", "cowboys": "DAL", "broncos": "DEN", "lions": "DET", "packers": "GB",
    "texans": "HOU", "colts": "IND", "jaguars": "JAX", "chiefs": "KC", "rams": "LA", "chargers": "LAC",
    "raiders": "LV", "dolphins": "MIA", "vikings": "MIN", "patriots": "NE", "saints": "NO", "giants": "NYG",
    "jets": "NYJ", "eagles": "PHI", "steelers": "PIT", "seahawks": "SEA", "49ers": "SF", "buccaneers": "TB",
    "titans": "TEN", "commanders": "WAS", "washington": "WAS",
}


def _event_date(ev: str):
    m = EV_RE.search(ev)
    if not m:
        return pd.NaT
    yy, mon, dd = m.groups()
    return pd.Timestamp(2000 + int(yy), MON[mon], int(dd))


def _match(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """events: key, date, team_a, team_b -> adds game_id, home, swapped flag. Tries date, then +-1 day."""
    gm = games[["game_id", "date", "home_abbr", "away_abbr", "tip_utc"]].copy()
    gm["pair"] = [frozenset(x) for x in zip(gm["home_abbr"], gm["away_abbr"])]
    ev = events.copy()
    ev["pair"] = [frozenset(x) for x in zip(ev["team_a"], ev["team_b"])]
    out = []
    remaining = ev
    for shift in (0, -1, 1):
        r = remaining.assign(d=remaining["date"] + pd.Timedelta(days=shift))
        m = r.merge(gm, left_on=["d", "pair"], right_on=["date", "pair"], how="left", suffixes=("", "_g"))
        hit = m["game_id"].notna()
        out.append(m[hit])
        remaining = remaining[~remaining["key"].isin(m.loc[hit, "key"])]
        if remaining.empty:
            break
    res = pd.concat(out, ignore_index=True).drop(columns=["d", "date_g"], errors="ignore")
    log.info("matched %d/%d events to games", res["key"].nunique(), events["key"].nunique())
    return res.drop_duplicates("key")


# ------------------------------------------------------------------ Kalshi
def kalshi_quotes(league: str, games: pd.DataFrame, market: str = "moneyline") -> pd.DataFrame:
    mk = pd.read_parquet(TIDY_DIR / f"kalshi_{league}_{market}_markets.parquet")
    cd = pd.read_parquet(TIDY_DIR / f"kalshi_{league}_{market}_candles.parquet")
    if league == "nfl":
        mk["team_code"] = mk["team_code"].replace(KALSHI_NFL)
    ev = mk.groupby("event_ticker")["team_code"].agg(list).reset_index()
    ev = ev[ev["team_code"].map(len) == 2]
    ev = pd.DataFrame({"key": ev["event_ticker"], "date": ev["event_ticker"].map(_event_date),
                       "team_a": ev["team_code"].str[0], "team_b": ev["team_code"].str[1]})
    m = _match(ev, games)
    side = mk.merge(m[["key", "game_id", "home_abbr", "away_abbr", "tip_utc"]], left_on="event_ticker", right_on="key")
    side["side"] = np.where(side["team_code"] == side["home_abbr"], "home", "away")
    cd = cd.merge(side[["ticker", "game_id", "side", "tip_utc"]], on="ticker")
    cols = ["bid_close", "ask_close", "p_close", "volume", "open_interest"]
    wide = cd.pivot_table(index=["game_id", "interval", "ts_utc"], columns="side", values=cols, aggfunc="last")
    wide.columns = [f"{s}_{c}" for c, s in wide.columns]
    wide = wide.reset_index().sort_values(["game_id", "interval", "ts_utc"])
    # carry the last known quote forward within a game/interval when one book has no candle
    qcols = [c for c in wide.columns if c.endswith(("bid_close", "ask_close", "p_close"))]
    wide[qcols] = wide.groupby(["game_id", "interval"])[qcols].ffill()
    q = pd.DataFrame({"game_id": wide["game_id"], "venue": "kalshi", "interval": wide["interval"], "ts_utc": wide["ts_utc"]})
    hb, ha = wide.get("home_bid_close"), wide.get("home_ask_close")
    ab, aa = wide.get("away_bid_close"), wide.get("away_ask_close")
    q["home_bid"], q["home_ask"], q["away_bid"], q["away_ask"] = hb, ha, ab, aa
    q["q_home"] = np.fmin(ha, 1 - ab)
    q["q_away"] = np.fmin(aa, 1 - hb)
    q["overround"] = q["q_home"] + q["q_away"] - 1
    mid_h, mid_a = (hb + ha) / 2, (ab + aa) / 2
    q["p_mid"] = np.where(mid_h.notna() & mid_a.notna(), (mid_h + 1 - mid_a) / 2, np.where(mid_h.notna(), mid_h, 1 - mid_a))
    q["p_last"] = wide.get("home_p_close")
    q["volume"] = wide.get("home_volume", 0).fillna(0) + wide.get("away_volume", 0).fillna(0)
    q["open_interest"] = wide.get("home_open_interest", 0).fillna(0) + wide.get("away_open_interest", 0).fillna(0)
    # a quote only counts when both sides are genuinely two-sided (no 0.01/1.00 placeholder books)
    q["two_sided"] = (hb > 0.0) & (ha < 1.0) & (ab > 0.0) & (aa < 1.0) & (q["overround"] < 0.25)
    q = q.merge(games[["game_id", "tip_utc"]], on="game_id")
    q["min_to_tip"] = (q["tip_utc"] - q["ts_utc"]).dt.total_seconds() / 60.0
    return q.drop(columns=["tip_utc"])


# --------------------------------------------------------------- Polymarket
def _nick_to_abbr(name: str, league: str):
    if not isinstance(name, str):
        return None
    n = name.lower().strip()
    table = NBA_NICK if league == "nba" else NFL_NICK
    if n in table:
        return table[n]
    for k, v in table.items():
        if n.endswith(k):
            return v
    return None


def poly_quotes(league: str, games: pd.DataFrame, market: str = "moneyline") -> pd.DataFrame:
    mk = pd.read_parquet(TIDY_DIR / f"poly_{league}_{market}_markets.parquet")
    px = pd.read_parquet(TIDY_DIR / f"poly_{league}_{market}_prices.parquet")
    mk["a"] = mk["outcome0"].map(lambda s: _nick_to_abbr(s, league))
    mk["b"] = mk["outcome1"].map(lambda s: _nick_to_abbr(s, league))
    mk = mk.dropna(subset=["a", "b"])
    gst = pd.to_datetime(mk["game_start_time"].str.replace(r"\+00$", "+00:00", regex=True), utc=True, errors="coerce")
    date_et = gst.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    ev = pd.DataFrame({"key": mk["id"].astype(str), "date": date_et, "team_a": mk["a"], "team_b": mk["b"]})
    m = _match(ev, games)
    mm = mk.assign(key=mk["id"].astype(str)).merge(m[["key", "game_id", "home_abbr"]], on="key")
    mm["flip"] = mm["a"] != mm["home_abbr"]
    px = px.merge(mm[["id", "game_id", "flip", "volume"]].rename(columns={"volume": "mkt_volume"}), on="id")
    q = pd.DataFrame({"game_id": px["game_id"], "venue": "polymarket", "interval": px["fidelity"], "ts_utc": px["ts_utc"],
                      "p_mid": np.where(px["flip"], 1 - px["p"], px["p"])})
    q["two_sided"] = (q["p_mid"] > 0.005) & (q["p_mid"] < 0.995)
    q["mkt_volume"] = px["mkt_volume"].to_numpy()
    q = q.merge(games[["game_id", "tip_utc"]], on="game_id")
    q["min_to_tip"] = (q["tip_utc"] - q["ts_utc"]).dt.total_seconds() / 60.0
    return q.drop(columns=["tip_utc"]).sort_values(["game_id", "interval", "ts_utc"]).reset_index(drop=True)


# ------------------------------------------------------------ game summary
SNAP_MIN = {"t24h": 24 * 60, "t6h": 6 * 60, "t1h": 60, "t15m": 15}


def summarize(q: pd.DataFrame, venue: str, close_tolerance_min: float = 30.0) -> pd.DataFrame:
    """One row per game: open / fixed-horizon / close fair probabilities and costs.

    close  = last two-sided minute quote at or before tip (within ``close_tolerance_min``)
    open   = first two-sided hourly quote of the listing
    t24h.. = last two-sided quote at least that many minutes before tip
    """
    q = q[(q["venue"] == venue) & q["two_sided"] & (q["min_to_tip"] >= 0)]
    rows = []
    has_ask = "q_home" in q.columns
    for gid, d in q.groupby("game_id", sort=False):
        d = d.sort_values("ts_utc")
        r = {"game_id": gid, "venue": venue}
        mn = d[d["interval"] == 1]
        hr = d[d["interval"] == 60]
        src_close = mn if len(mn) else hr
        c = src_close.iloc[-1]
        if c["min_to_tip"] <= close_tolerance_min:
            r.update(p_close=c["p_mid"], close_min_to_tip=c["min_to_tip"])
            if has_ask:
                r.update(q_home_close=c["q_home"], q_away_close=c["q_away"], overround_close=c["overround"],
                         home_bid_close=c["home_bid"], home_ask_close=c["home_ask"],
                         away_bid_close=c["away_bid"], away_ask_close=c["away_ask"])
        if len(hr):
            o = hr.iloc[0]
            r.update(p_open=o["p_mid"], open_min_to_tip=o["min_to_tip"])
            if has_ask:
                r.update(q_home_open=o["q_home"], q_away_open=o["q_away"], overround_open=o["overround"])
        for k, mins in SNAP_MIN.items():
            s = d[d["min_to_tip"] >= mins]
            if len(s):
                x = s.iloc[-1]
                r[f"p_{k}"] = x["p_mid"]
                if has_ask:
                    r[f"q_home_{k}"], r[f"q_away_{k}"] = x["q_home"], x["q_away"]
        if "volume" in d.columns:
            r["volume_pre_tip"] = float(d.loc[d["interval"] == 60, "volume"].sum())
        if "mkt_volume" in d.columns:
            r["volume_total"] = float(d["mkt_volume"].iloc[0])
        rows.append(r)
    return pd.DataFrame(rows)
