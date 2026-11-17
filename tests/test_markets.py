import numpy as np
import pytest

from sq.markets import devig as DV
from sq.markets import odds as O


def test_american_decimal_roundtrip():
    assert O.american_to_decimal(150) == pytest.approx(2.5)
    assert O.american_to_decimal(-200) == pytest.approx(1.5)
    assert O.decimal_to_american(2.5) == pytest.approx(150)
    assert O.decimal_to_american(1.5) == pytest.approx(-200)
    assert O.american_to_prob(-110) == pytest.approx(110 / 210)


def test_overround_standard_line():
    q = O.american_to_prob(np.array([-110.0]))
    assert O.overround(q, q)[0] == pytest.approx(2 * 110 / 210 - 1)


@pytest.mark.parametrize("method", DV.METHODS)
def test_devig_sums_to_one_and_symmetric(method):
    q = np.array([[0.5238, 0.5238], [0.70, 0.35], [0.20, 0.85], [0.6, 0.45]])
    p = DV.devig(q, method)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert p[0, 0] == pytest.approx(0.5, abs=1e-9)
    assert np.all((p > 0) & (p < 1))
    # fair probability keeps the ordering of the raw prices
    assert p[1, 0] > p[1, 1] and p[2, 0] < p[2, 1]


def test_devig_methods_differ_on_longshots_as_theory_says():
    q = np.array([[0.20, 0.85]])  # longshot vs favourite, 5% overround
    mult = DV.multiplicative(q)[0, 0]
    add = DV.additive(q)[0, 0]
    shin = DV.shin(q)[0, 0]
    power = DV.power(q)[0, 0]
    # multiplicative removes margin proportionally; Shin/power shift more margin onto the longshot
    assert shin < mult and power < mult
    assert add < mult  # additive takes an equal absolute slice -> longshot loses relatively more
    _, z = DV.shin(q, return_z=True)
    assert 0 < z[0] < 0.2


def test_devig_nan_rows():
    q = np.array([[np.nan, 0.5], [0.55, 0.5]])
    for m in DV.METHODS:
        p = DV.devig(q, m)
        assert np.isnan(p[0]).all() or m in ("multiplicative", "additive")
        assert np.isfinite(p[1]).all()


def test_kalshi_fee_schedule():
    # 100 contracts at 50c: 0.07 * 100 * 0.25 = $1.75
    assert O.kalshi_fee(0.5, 100) == pytest.approx(1.75)
    # rounding up to the next cent on a single contract
    assert O.kalshi_fee(0.5, 1) == pytest.approx(0.02)
    assert O.kalshi_all_in_price(0.5, contracts=100) == pytest.approx(0.5175)
