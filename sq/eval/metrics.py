"""Forecast evaluation for binary outcomes.

Everything here scores *probabilities* with proper scoring rules (log loss, Brier)
and calibration diagnostics; betting returns are deliberately kept elsewhere so
that model selection is never driven by historical ROI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def log_loss(p, y):
    p, y = _clip(p), np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def pointwise_log_loss(p, y):
    p, y = _clip(p), np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def accuracy(p, y):
    return float(np.mean((np.asarray(p) > 0.5) == (np.asarray(y) > 0.5)))


def logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def calibration_regression(p, y):
    """Fit y ~ Bernoulli(sigmoid(a + b * logit(p))).

    b = 1, a = 0 for a calibrated forecast. b < 1: forecasts too extreme
    (over-confident); b > 1: too timid. Returns a, b with standard errors.
    """
    import statsmodels.api as sm

    x = sm.add_constant(logit(p))
    res = sm.GLM(np.asarray(y, float), x, family=sm.families.Binomial()).fit()
    (a, b), (sa, sb) = res.params, res.bse
    return {"intercept": float(a), "intercept_se": float(sa), "slope": float(b), "slope_se": float(sb),
            "slope_p_vs_1": float(2 * stats.norm.sf(abs((b - 1) / sb)))}


def calibration_table(p, y, n_bins=10, strategy="quantile"):
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        k = y[m].sum()
        lo, hi = wilson(k, n)
        rows.append({"bin": b, "p_lo": edges[b], "p_hi": edges[b + 1], "n": n, "mean_pred": p[m].mean(),
                     "obs_freq": k / n, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    den = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def ece(p, y, n_bins=10, strategy="quantile"):
    t = calibration_table(p, y, n_bins, strategy)
    return float((t["n"] * (t["mean_pred"] - t["obs_freq"]).abs()).sum() / t["n"].sum())


def murphy_decomposition(p, y, n_bins=10):
    """Brier = reliability - resolution + uncertainty (binned approximation)."""
    t = calibration_table(p, y, n_bins, "quantile")
    ybar = float(np.mean(y))
    w = t["n"] / t["n"].sum()
    rel = float((w * (t["mean_pred"] - t["obs_freq"]) ** 2).sum())
    res = float((w * (t["obs_freq"] - ybar) ** 2).sum())
    unc = ybar * (1 - ybar)
    return {"reliability": rel, "resolution": res, "uncertainty": unc, "brier_binned": rel - res + unc}


def summarize(p, y, name="model"):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    cr = calibration_regression(p, y)
    md = murphy_decomposition(p, y)
    return {"model": name, "n": int(len(p)), "log_loss": log_loss(p, y), "brier": brier(p, y),
            "accuracy": accuracy(p, y), "ece": ece(p, y), "cal_intercept": cr["intercept"],
            "cal_slope": cr["slope"], "cal_slope_se": cr["slope_se"], "reliability": md["reliability"],
            "resolution": md["resolution"], "mean_pred": float(p.mean()), "base_rate": float(y.mean())}


# ------------------------------------------------------------- paired comparison
def paired_loss_test(p_a, p_b, y, clusters=None, n_boot=2000, seed=0, loss="log"):
    """Compare two forecasts of the same events.

    Returns mean loss difference (a - b; negative = a better), a cluster bootstrap
    CI (clusters = e.g. game dates, so same-night games resample together) and a
    Diebold-Mariano style z statistic with cluster-robust variance.
    """
    p_a, p_b, y = np.asarray(p_a, float), np.asarray(p_b, float), np.asarray(y, float)
    if loss == "log":
        d = pointwise_log_loss(p_a, y) - pointwise_log_loss(p_b, y)
    else:
        d = (p_a - y) ** 2 - (p_b - y) ** 2
    n = len(d)
    if clusters is None:
        clusters = np.arange(n)
    cl = pd.Series(d).groupby(pd.Series(np.asarray(clusters)).values).agg(["sum", "count"])
    sums, counts = cl["sum"].to_numpy(), cl["count"].to_numpy()
    g = len(sums)
    mean = sums.sum() / counts.sum()
    # cluster-robust variance of the mean
    resid = sums - mean * counts
    var = (resid ** 2).sum() / counts.sum() ** 2 * g / max(g - 1, 1)
    z = mean / np.sqrt(var) if var > 0 else np.nan
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, g, g)
        bs[i] = sums[k].sum() / counts[k].sum()
    return {"mean_diff": float(mean), "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975)),
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z))), "n": int(n), "clusters": int(g)}


def encompassing_test(p_market, p_model, y, clusters=None):
    """Logit regression y ~ a + b_m*logit(p_market) + b_x*logit(p_model).

    If the market price encompasses the model, b_x = 0 (the model adds nothing the
    price does not already contain). Cluster-robust SEs by ``clusters``.
    """
    import statsmodels.api as sm

    X = sm.add_constant(np.column_stack([logit(p_market), logit(p_model)]))
    mod = sm.GLM(np.asarray(y, float), X, family=sm.families.Binomial())
    if clusters is not None:
        codes = pd.factorize(pd.Series(np.asarray(clusters)))[0]
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": codes})
    else:
        res = mod.fit()
    names = ["const", "market", "model"]
    out = {f"b_{n}": float(v) for n, v in zip(names, res.params)}
    out.update({f"se_{n}": float(v) for n, v in zip(names, res.bse)})
    out.update({f"p_{n}": float(v) for n, v in zip(names, res.pvalues)})
    return out


def benjamini_hochberg(pvals, q=0.05):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out, out <= q
