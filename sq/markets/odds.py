"""Price conversions, overround and exchange fee models.

Conventions: ``q`` is a raw implied probability (1 / decimal odds, or an exchange
price in dollars per $1 contract) and still contains margin; ``p`` is a fair
(margin-free) probability.
"""
from __future__ import annotations

import numpy as np


def american_to_decimal(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / np.abs(a))


def decimal_to_american(d):
    d = np.asarray(d, dtype=float)
    return np.where(d >= 2.0, (d - 1.0) * 100.0, -100.0 / (d - 1.0))


def american_to_prob(a):
    return 1.0 / american_to_decimal(a)


def prob_to_decimal(p):
    return 1.0 / np.asarray(p, dtype=float)


def overround(*qs):
    """Sum of raw implied probabilities minus one (the book's margin)."""
    return np.sum(np.vstack([np.asarray(q, dtype=float) for q in qs]), axis=0) - 1.0


# ---------------------------------------------------------------- exchange fees
# Kalshi general fee schedule: fee = round_up_to_cent(rate * C * P * (1 - P)) per order;
# taker rate 0.07, maker rate 0.0175 on series flagged "quadratic_with_maker_fees".
KALSHI_TAKER = 0.07
KALSHI_MAKER = 0.0175


def kalshi_fee(price, contracts=1.0, rate=KALSHI_TAKER, round_up=True):
    price = np.asarray(price, dtype=float)
    raw = rate * contracts * price * (1.0 - price)
    return np.ceil(raw * 100.0 - 1e-9) / 100.0 if round_up else raw


def kalshi_all_in_price(price, rate=KALSHI_TAKER, contracts=100.0):
    """Cost per $1 contract including the (rounded) taker fee for an order of ``contracts``."""
    price = np.asarray(price, dtype=float)
    return price + kalshi_fee(price, contracts, rate) / contracts


def exchange_decimal_odds(all_in_price):
    """Decimal odds of buying a $1 contract at an all-in cost."""
    return 1.0 / np.asarray(all_in_price, dtype=float)


def polymarket_all_in_price(price, fee_rate=0.0):
    """Polymarket sports markets: price plus an optional proportional taker fee.

    The fee rate is a parameter (default 0) because Polymarket's sports fee schedule
    has changed over time; backtests report a sensitivity grid instead of one number.
    """
    price = np.asarray(price, dtype=float)
    return price + fee_rate * price * (1.0 - price)
