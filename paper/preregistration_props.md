# Pre-registration: NBA player-points props replication

Registered 2026-11-14 22:52 UTC, before any replication data was downloaded
(`data/tidy/kalshi_nba_player_points_rep_markets.parquet` did not exist at registration time).

## Why

In the discovery sample (a random 15% of Kalshi NBA player-points events with at least 500 contracts traded;
2,752 closing prices, 125 games, 2026-01-16 to 2026-11-11), the player model's coefficient in a
forecast-encompassing regression next to the closing prop price was 0.196 (game-clustered SE 0.091;
two-sided p = 0.032, one-sided p = 0.016). It was the only test in the study where a model carried
information the closing price lacked, and it was a single test that would not survive the paper's
multiple-testing correction. This replication asks whether it holds on games the discovery sample never saw.

## Sample (fixed in advance)

* Every settled Kalshi `KXNBAPTS` market listed on the day of the fetch (live and historical APIs) with at
  least 500 contracts traded, **excluding every event in the discovery sample** (the event tickers in
  `data/tidy/kalshi_nba_player_points_markets.parquet`) and, at analysis time, any game that appears in the
  discovery panel.
* Same matching and filters as the discovery analysis: event mapped to a schedule game through the matching
  game-winner event; player matched by normalised name within team; closing quote = last two-sided 1-minute
  quote at or before tip-off and no more than 30 minutes before it; mid price strictly between 0.01 and 0.99;
  player with at least 5 previous games.
* No stopping rule beyond the data available on the fetch date; nothing will be added or dropped after the
  results are seen.

## Model (frozen)

The discovery code, unchanged: `sq/research/nba_lines.py` `props()`. Points given that the player plays follow a
negative binomial with mean = exponentially weighted pre-game points (half-life 12 games) and dispersion fitted on
2018-19 to 2023-24 player-games (3.39). P(play) from the last official injury report published at least 5 minutes
before tip-off (Out 0, Doubtful 0.15, Questionable 0.55, Probable 0.9, otherwise 1).

Code hashes (sha256) at registration:

```
9b38870759d472ef804b766d28202d0e9cc2e63cf414d2d86e3a5be4858cdf30  sq/research/nba_lines.py
93f0eae1938ece663fb1d703fe15cb15d2208c7a5ca8a5cc9293f48beec256f0  sq/io/kalshi.py
7cb56be8ba76b6b64db36359025cb5c84766fcabaa58aeb704c88233ca3ff2b1  scripts/fetch.py
8b4adfc1626594f3cd9413ecf3ce1ba6805805c02de41c9e7d1b9b8006b2f3c5  sq/features/players.py
80d0d282652c55a2a849a666b9ee0abcfb43aa6931727ad9c3283fe122f464ec  sq/eval/metrics.py
```

## Hypotheses and decision rules

* **H1 (primary, information).** In the logistic regression of outcomes on logit(closing mid) and
  logit(player model), with game-clustered standard errors, the player-model coefficient is positive with
  one-sided p < 0.05. Replicates if yes; otherwise the discovery result is treated as a false positive.
* **H2 (secondary, pricing).** Buying every Yes at the closing ask plus the Kalshi taker fee
  (0.07 x P x (1 - P) per contract, rounded up on 100-contract orders) is not profitable: the lower end of the
  game-clustered 95% bootstrap interval of its return is at or below zero.
* **H3 (secondary, economic).** A flat one-contract rule at the closing quote using the combined probability
  logit p = 0.1269 + 0.7769 logit(mid) + 0.1963 logit(model), with the coefficients frozen from the discovery
  sample: buy Yes at the ask or No at 1 - bid, whichever edge after the taker fee is larger, if at least 2
  points. Profitable only if the lower end of the game-clustered 95% interval of its return is above zero.
* **H3b (secondary).** The same rule with the player model alone.

Discovery-sample results of the trading rules, recorded now so they cannot be reinterpreted later: H3's rule,
in sample, returned -5.1% on 312 bets (95% CI -21.4% to +11.3%); H3b's rule returned -9.7% on 1,436 bets
(95% CI -17.1% to -3.0%).

## Reporting

All four results are reported whatever their direction, together with log loss and Brier score of the market,
the model and the frozen combination, the sample size, games, players and date range. Output:
`outputs/nba/props_replication.json`, `outputs/nba/props_rep_metrics.csv`.
