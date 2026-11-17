import numpy as np
import pytest

from sq.eval import metrics as M


def test_scores_basic():
    y = np.array([1, 0, 1, 0])
    assert M.brier(np.full(4, 0.5), y) == pytest.approx(0.25)
    assert M.log_loss(np.full(4, 0.5), y) == pytest.approx(np.log(2))
    assert M.log_loss(np.array([1, 0, 1, 0]), y) < 1e-5


def test_calibration_regression_recovers_slope():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = rng.random(20000) < p
    cr = M.calibration_regression(p, y)
    assert cr["slope"] == pytest.approx(1.0, abs=0.06)
    assert cr["intercept"] == pytest.approx(0.0, abs=0.05)
    # an over-confident forecast (logit stretched x2) is detected as slope ~0.5
    lg = np.log(p / (1 - p)) * 2
    cr2 = M.calibration_regression(1 / (1 + np.exp(-lg)), y)
    assert cr2["slope"] == pytest.approx(0.5, abs=0.05)


def test_paired_test_detects_better_forecast():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.2, 0.8, 5000)
    y = (rng.random(5000) < p).astype(float)
    noisy = np.clip(p + rng.normal(0, 0.15, 5000), 0.02, 0.98)
    t = M.paired_loss_test(p, noisy, y, clusters=np.arange(5000) // 5)
    assert t["mean_diff"] < 0 and t["ci_hi"] < 0 and t["p_value"] < 0.01


def test_encompassing_true_model_adds_information():
    rng = np.random.default_rng(3)
    true = rng.uniform(0.2, 0.8, 8000)
    y = (rng.random(8000) < true).astype(float)
    market = np.clip(true + rng.normal(0, 0.08, 8000), 0.02, 0.98)   # noisy market
    r = M.encompassing_test(market, true, y)
    assert r["p_model"] < 0.01 and r["b_model"] > 0.3


def test_benjamini_hochberg():
    adj, rej = M.benjamini_hochberg([0.001, 0.02, 0.04, 0.5])
    assert rej[0] and not rej[3]
    assert np.all(adj >= np.array([0.001, 0.02, 0.04, 0.5]) - 1e-12)
