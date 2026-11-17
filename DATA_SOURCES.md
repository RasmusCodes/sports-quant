# Data sources

Ground rules applied: public endpoints only, robots.txt and terms of use respected, no logins, no
CAPTCHA or rate-limit workarounds, a generic user agent with no personal details, every
response cached with its UTC fetch time and sha256 in `data/raw/<source>/_manifest.jsonl`.
Nothing is interpolated: a missing value stays missing.

## Used

| Source | What | Access | Licence / terms basis | Coverage used |
|---|---|---|---|---|
| sportsdataverse-data (GitHub releases) | NBA schedules, team and player box scores (stats.nba.com tables), the official NBA injury report with publication timestamps, ESPN schedule (UTC tip-off, neutral-site flag) | `github.com/sportsdataverse/sportsdataverse-data/releases/download/<tag>/<file>.parquet`; asset lists via `api.github.com/repos/.../releases/tags/<tag>` | Repository licensed MIT; release downloads are not disallowed by github.com robots.txt | Seasons 1996-97 to 2026-27 (injury reports from 2018-12-17) |
| nflverse-data (GitHub releases) | NFL games with closing moneyline, spread and total plus prices, rest days, starting QBs, weather; weekly injury reports; player and team weekly stats; snap counts; depth charts | `github.com/nflverse/nflverse-data/releases/download/<tag>/<file>.parquet` | CC-BY-4.0 | 1999 to 2026 (moneylines from 2006, complete from 2010) |
| Kalshi public market-data API | NBA and NFL game-winner markets: hourly and 1-minute candles with trade price, yes-bid and yes-ask OHLC, volume, open interest | `api.elections.kalshi.com/trade-api/v2`: `/markets`, `/series/{s}/markets/{t}/candlesticks`; markets settled before the cutoff (2026-09-13) through `/historical/markets` and `/historical/markets/{t}/candlesticks` | Public, unauthenticated read-only API; the API host has no robots.txt | KXNBAGAME 2025-04-14 to 2026-11-12 (1,690 events); KXNFLGAME 2025-07-31 to 2026-11-12 (519 events, preseason unmatched) |
| Polymarket Gamma + CLOB APIs | NBA and NFL moneyline markets (metadata, outcome tokens, game start time) and price history | `gamma-api.polymarket.com/markets?sports_market_types=moneyline&tag_id=745` (NBA) / `450` (NFL); `clob.polymarket.com/prices-history` (at most 7 days per request) | Public, unauthenticated read-only APIs; no robots.txt on the API hosts; polymarket.com robots.txt has no disallow rules | NBA from 2024-04-20 (1,824 markets); NFL from 2024 preseason (841 markets) |

ESPN-derived fields (tip-off times, neutral-site flags) come only through the MIT-licensed
sportsdataverse redistribution; ESPN endpoints were not called (see below).

## Checked and not used

| Source | Why not |
|---|---|
| sportsbookreviewsonline.com scores and odds archive (NBA 2007-08 to 2022-23, NFL) | robots.txt allows it, but the site's terms say content may not be used or reproduced without express written consent. This would have been the longest NBA sportsbook history; it needs the user's decision or the site's consent. |
| aussportsbetting.com historical odds files, thepredictiontracker.com | robots.txt disallows ClaudeBot, anthropic-ai and Claude-Web (and the data pages return 403 to scripts). A person can download these files by hand and load them through the manual importer. |
| ESPN site/core APIs | www.espn.com robots.txt disallows anthropic-ai; the API hosts return 403 for robots.txt. Not called. |
| cdn.nba.com live-data JSON | Returns 403 to scripts; not worked around. |
| Kaggle odds datasets | Require an account login. |
| The Odds API historical endpoints | Require a paid key. |
| FiveThirtyEight NBA Elo forecasts | The file is gone (GitHub 404; the projects site now redirects to ABC News). The Elo benchmark reimplements 538's published method instead. |

## Adding sportsbook history

Put a CSV at `data/manual/sportsbook_lines.csv` (schema in `sq/io/manual.py`) and rerun
`scripts/build_db.py`. Rows that do not match exactly one scheduled game are logged and skipped.

## Timestamps

* `games.tip_utc`: scheduled tip-off / kickoff in UTC. NBA: ESPN schedule time (85% of all
  games since 1996-97; 100% since 2015-16, the period used for evaluation). NFL: nflverse
  `gameday` + `gametime` (Eastern) converted to UTC.
* `games.result_available_utc`: tip + 3h (NBA) / 3.5h (NFL); results are never used before this.
* `market_quotes.ts_utc`: candle end time (Kalshi) or price-point time (Polymarket).
* `injuries_nba.report_utc`: publication time of the official report, converted from ET.
* `injuries_nfl.report_utc`: nflverse `date_modified`.
