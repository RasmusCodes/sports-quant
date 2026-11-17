"""Remove the bookmaker margin from a set of raw implied probabilities.

All functions take ``q`` with shape (n_events, n_outcomes) (or a single row) and
return fair probabilities that sum to one per row. Rows containing NaN return NaN.

Methods
-------
multiplicative  p_i = q_i / sum(q)                       (proportional / "basic")
additive        p_i = q_i - (sum(q) - 1) / n             (equal margin per outcome)
power           p_i = q_i ** k,  k solves sum = 1        (Clarke et al. 2017)
shin            Shin (1993) insider-trading model, z solves sum = 1
odds_ratio      p_i/(1-p_i) = (q_i/(1-q_i)) / c          (Cheung 2015; logit shift)
"""
from __future__ import annotations

import numpy as np

METHODS = ("multiplicative", "additive", "power", "shin", "odds_ratio")


def _as2d(q):
    q = np.asarray(q, dtype=float)
    return (q[None, :], True) if q.ndim == 1 else (q, False)


def _bisect(fun, lo, hi, n_iter=80):
    """Vectorised bisection for a function decreasing in its argument; fun(x) -> residual per row."""
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        r = fun(mid)
        pos = r > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def multiplicative(q):
    q, one = _as2d(q)
    p = q / q.sum(axis=1, keepdims=True)
    return p[0] if one else p


def additive(q):
    q, one = _as2d(q)
    n = q.shape[1]
    p = q - (q.sum(axis=1, keepdims=True) - 1.0) / n
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p[0] if one else p


def power(q):
    q, one = _as2d(q)
    bad = ~np.isfinite(q).all(axis=1)
    qq = np.where(bad[:, None], 0.5, q)
    # sum(q^k) decreasing in k; at k=1 sum>1 (overround) -> k>1; underround -> k<1
    k = _bisect(lambda k: (qq ** k[:, None]).sum(axis=1) - 1.0, np.full(len(qq), 0.2), np.full(len(qq), 5.0))
    p = qq ** k[:, None]
    p = p / p.sum(axis=1, keepdims=True)
    p[bad] = np.nan
    return p[0] if one else p


def _shin_p(q, z):
    Q = q.sum(axis=1, keepdims=True)
    zz = z[:, None]
    return (np.sqrt(zz ** 2 + 4.0 * (1.0 - zz) * q ** 2 / Q) - zz) / (2.0 * (1.0 - zz))


def shin(q, return_z=False):
    q, one = _as2d(q)
    bad = ~np.isfinite(q).all(axis=1)
    qq = np.where(bad[:, None], 0.5, q)
    # sum of Shin probabilities is decreasing in z on [0, 1)
    z = _bisect(lambda z: _shin_p(qq, z).sum(axis=1) - 1.0, np.zeros(len(qq)), np.full(len(qq), 0.5))
    under = qq.sum(axis=1) <= 1.0
    z = np.where(under, 0.0, z)
    p = _shin_p(qq, z)
    p = p / p.sum(axis=1, keepdims=True)
    p[bad] = np.nan
    z[bad] = np.nan
    if return_z:
        return (p[0], z[0]) if one else (p, z)
    return p[0] if one else p


def odds_ratio(q):
    q, one = _as2d(q)
    bad = ~np.isfinite(q).all(axis=1)
    qq = np.clip(np.where(bad[:, None], 0.5, q), 1e-9, 1 - 1e-9)
    o = qq / (1.0 - qq)

    def fair(c):
        oc = o / c[:, None]
        return oc / (1.0 + oc)

    # sum of fair probs decreasing in c
    c = _bisect(lambda c: fair(c).sum(axis=1) - 1.0, np.full(len(qq), 0.2), np.full(len(qq), 5.0))
    p = fair(c)
    p = p / p.sum(axis=1, keepdims=True)
    p[bad] = np.nan
    return p[0] if one else p


def devig(q, method="multiplicative"):
    return {"multiplicative": multiplicative, "additive": additive, "power": power,
            "shin": shin, "odds_ratio": odds_ratio}[method](q)


def two_way(q_home, q_away, method="multiplicative"):
    """Convenience: fair home probability from two raw implied probabilities."""
    q = np.column_stack([np.asarray(q_home, float), np.asarray(q_away, float)])
    return devig(q, method)[:, 0]
