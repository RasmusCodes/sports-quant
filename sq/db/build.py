"""Build ``data/db/sports.sqlite`` - the historical odds database.

Tables
  sources            one row per data source: licence / terms basis, endpoints, coverage
  games              NBA + NFL games with UTC tip-off and result-availability timestamps
  market_quotes      every exchange quote (Kalshi bid/ask both books, Polymarket price)
                     with ts_utc and minutes-to-tip; hourly over the listing life and
                     1-minute around tip-off
  market_summary     per game x venue: open, T-24h/6h/1h/15m and closing fair prices,
                     closing asks / overround, pre-tip volume
  sportsbook_lines   bookmaker lines with prices (nflverse closing moneyline / spread /
                     total for the NFL; user-supplied files via sq.io.manual)
  injuries_nba       official NBA injury report rows (report_utc)
  injuries_nfl       nflverse weekly injury report rows (report_utc = date_modified)
Views
  v_close            one row per game with closing probabilities from every venue
Point-in-time rule: a query for a decision at time T may only use rows with
ts_utc / report_utc <= T and games with result_available_utc <= T.
"""
from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from ..config import DB_PATH, TIDY_DIR
from ..markets import quotes as Q

log = logging.getLogger(__name__)

SOURCES = [
    ("sportsdataverse", "NBA schedules, box scores, injury reports", "MIT-licensed GitHub release assets",
     "github.com/sportsdataverse/sportsdataverse-data/releases", "1996-97 .. 2026-27"),
    ("nflverse", "NFL games with closing lines, injuries, player stats", "CC-BY-4.0 GitHub release assets",
     "github.com/nflverse/nflverse-data/releases", "1999 .. 2026"),
    ("kalshi", "Exchange order-book candles for NBA/NFL game markets", "Public market-data API, no auth",
     "api.elections.kalshi.com/trade-api/v2 (/markets, /historical/markets, candlesticks)", "2025-04 .. 2026-11"),
    ("polymarket", "Exchange price history for NBA/NFL game markets", "Public Gamma + CLOB APIs, no auth",
     "gamma-api.polymarket.com/markets, clob.polymarket.com/prices-history", "2024-04 .. 2026-11"),
]


def _ts_to_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ" if getattr(df[c].dt, "tz", None) is not None else "%Y-%m-%d")
    return df


def games_union() -> pd.DataFrame:
    nba = pd.read_parquet(TIDY_DIR / "nba_games.parquet").assign(league="nba")
    nfl = pd.read_parquet(TIDY_DIR / "nfl_games.parquet").assign(league="nfl")
    cols = ["league", "game_id", "season", "season_type", "playoff", "date", "tip_utc", "tip_known", "home", "away",
            "home_abbr", "away_abbr", "home_pts", "away_pts", "neutral", "result_available_utc"]
    nba["home"], nba["away"] = nba["home"].astype(str), nba["away"].astype(str)
    return pd.concat([nba[cols], nfl[cols]], ignore_index=True)


def nfl_sportsbook_lines() -> pd.DataFrame:
    g = pd.read_parquet(TIDY_DIR / "nfl_games.parquet")
    base = {"league": "nfl", "book": "nflverse_close", "snapshot": "close", "source": "nflverse games"}
    ml = g[["game_id", "home_ml", "away_ml"]].dropna().rename(columns={"home_ml": "home_price", "away_ml": "away_price"})
    ml = ml.assign(market="moneyline", line=None, **base)
    sp = g[["game_id", "home_spread", "home_spread_odds", "away_spread_odds"]].dropna(subset=["home_spread"])
    sp = sp.rename(columns={"home_spread": "line", "home_spread_odds": "home_price", "away_spread_odds": "away_price"})
    sp = sp.assign(market="spread", **base)
    to = g[["game_id", "total_line", "over_odds", "under_odds"]].dropna(subset=["total_line"])
    to = to.rename(columns={"total_line": "line", "over_odds": "home_price", "under_odds": "away_price"})
    to = to.assign(market="total", **base)
    out = pd.concat([ml, sp, to], ignore_index=True)
    return out[["league", "game_id", "book", "snapshot", "market", "line", "home_price", "away_price", "source"]]


def build(leagues=("nba", "nfl")) -> dict[str, int]:
    games = games_union()
    quotes, summaries = [], []
    for lg in leagues:
        gl = games[games["league"] == lg].copy()
        gl["home_abbr"] = gl["home_abbr"].astype(str)
        for fn, venue in ((Q.kalshi_quotes, "kalshi"), (Q.poly_quotes, "polymarket")):
            try:
                q = fn(lg, gl)
            except FileNotFoundError as exc:
                log.warning("no %s data for %s: %s", venue, lg, exc)
                continue
            q["league"] = lg
            quotes.append(q)
            s = Q.summarize(q, venue)
            s["league"] = lg
            summaries.append(s)
            log.info("%s %s: %d quotes, %d games summarised", lg, venue, len(q), len(s))
    mq = pd.concat(quotes, ignore_index=True)
    ms = pd.concat(summaries, ignore_index=True)
    mq.to_parquet(TIDY_DIR / "market_quotes.parquet", index=False)
    ms.to_parquet(TIDY_DIR / "market_summary.parquet", index=False)
    sb = nfl_sportsbook_lines()
    from ..io.manual import load_manual_lines

    manual = load_manual_lines(games)
    if len(manual):
        sb = pd.concat([sb, manual], ignore_index=True)
        log.info("added %d user-supplied sportsbook lines", len(manual))
    inj_nba = pd.read_parquet(TIDY_DIR / "nba_injuries.parquet")
    inj_nfl = pd.read_parquet(TIDY_DIR / "nfl_injuries.parquet")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    pd.DataFrame(SOURCES, columns=["source", "content", "licence_basis", "endpoints", "coverage"]).to_sql(
        "sources", con, index=False)
    counts = {}
    for name, df in {"games": games, "market_quotes": mq, "market_summary": ms, "sportsbook_lines": sb,
                     "injuries_nba": inj_nba, "injuries_nfl": inj_nfl}.items():
        _ts_to_str(df).to_sql(name, con, index=False, chunksize=50_000)
        counts[name] = len(df)
    con.executescript("""
        CREATE INDEX ix_games ON games(league, game_id);
        CREATE INDEX ix_mq ON market_quotes(game_id, venue, interval, ts_utc);
        CREATE INDEX ix_ms ON market_summary(game_id, venue);
        CREATE INDEX ix_sb ON sportsbook_lines(game_id, market);
        CREATE INDEX ix_inj_nba ON injuries_nba(game_id, report_utc);
        CREATE VIEW v_close AS
          SELECT g.league, g.game_id, g.season, g.tip_utc, g.home_abbr, g.away_abbr, g.home_pts, g.away_pts,
                 k.p_close AS p_kalshi_close, k.q_home_close AS kalshi_q_home, k.q_away_close AS kalshi_q_away,
                 k.overround_close AS kalshi_overround, p.p_close AS p_poly_close,
                 s.home_price AS book_home_ml, s.away_price AS book_away_ml
          FROM games g
          LEFT JOIN market_summary k ON k.game_id = g.game_id AND k.venue = 'kalshi'
          LEFT JOIN market_summary p ON p.game_id = g.game_id AND p.venue = 'polymarket'
          LEFT JOIN sportsbook_lines s ON s.game_id = g.game_id AND s.market = 'moneyline';
    """)
    con.commit()
    con.close()
    if DB_PATH.exists():
        DB_PATH.unlink()
    tmp.rename(DB_PATH)
    log.info("database written: %s %s", DB_PATH, counts)
    return counts
