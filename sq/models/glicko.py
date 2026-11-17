"""Glicko-2 (Glickman 2012) adapted to team sports.

Each game is its own rating period for the two teams involved. Before a game a
team's deviation grows with the idle time: phi* = sqrt(phi^2 + sigma^2 * days/period_days),
so a team that has not played for a while is treated as more uncertain. Home
advantage enters as a rating offset. The forecast integrates both teams'
uncertainty: E = 1 / (1 + exp(-g(sqrt(phi_h^2 + phi_a^2)) * (mu_h + hfa - mu_a))).
Glicko uses wins/losses only (no margin).
"""
from __future__ import annotations

import math

from .base import OnlineModel

SCALE = 173.7178


def _g(phi):
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / math.pi ** 2)


class Glicko2(OnlineModel):
    def __init__(self, hfa=70.0, rd0=200.0, sigma0=0.06, tau=0.5, period_days=7.0, carry=0.75,
                 rd_season=150.0, name="glicko2"):
        self.hfa = hfa / SCALE
        self.phi0, self.sigma0, self.tau = rd0 / SCALE, sigma0, tau
        self.period_days, self.carry, self.phi_season = period_days, carry, rd_season / SCALE
        self.name = name
        self.st: dict = {}

    def reset(self, teams):
        self.st = {t: [0.0, self.phi0, self.sigma0, None] for t in teams}  # mu, phi, sigma, last date

    def start_season(self, season):
        for t, s in self.st.items():
            s[0] *= self.carry
            s[1] = max(s[1], self.phi_season)

    def _phi_star(self, s, date):
        if s[3] is None:
            return s[1]
        days = max((date - s[3]).days if hasattr(date - s[3], "days") else float(date - s[3]), 0)
        return math.sqrt(s[1] ** 2 + s[2] ** 2 * days / self.period_days)

    def predict(self, g):
        h, a = self.st[g.home], self.st[g.away]
        ph, pa = self._phi_star(h, g.date), self._phi_star(a, g.date)
        adv = 0.0 if g.neutral else self.hfa
        x = _g(math.sqrt(ph * ph + pa * pa)) * (h[0] + adv - a[0])
        return {"p_home": 1.0 / (1.0 + math.exp(-x)), "glicko_home": 1500 + SCALE * h[0],
                "glicko_away": 1500 + SCALE * a[0], "rd_home": SCALE * ph, "rd_away": SCALE * pa}

    def _volatility(self, phi, sigma, delta, v):
        a = math.log(sigma * sigma)
        tau2 = self.tau ** 2

        def f(x):
            ex = math.exp(x)
            return ex * (delta ** 2 - phi ** 2 - v - ex) / (2 * (phi ** 2 + v + ex) ** 2) - (x - a) / tau2

        A = a
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau
        fA, fB = f(A), f(B)
        for _ in range(60):
            if abs(B - A) < 1e-7:
                break
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB <= 0:
                A, fA = B, fB
            else:
                fA /= 2
            B, fB = C, fC
        return math.exp(A / 2)

    def _update_one(self, s, opp_mu, opp_phi, score, date):
        phi = self._phi_star(s, date)
        gj = _g(opp_phi)
        E = 1.0 / (1.0 + math.exp(-gj * (s[0] - opp_mu)))
        v = 1.0 / (gj * gj * E * (1 - E))
        delta = v * gj * (score - E)
        sigma = self._volatility(phi, s[2], delta, v)
        phi_star = math.sqrt(phi * phi + sigma * sigma)
        phi_new = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
        mu_new = s[0] + phi_new ** 2 * gj * (score - E)
        return [mu_new, phi_new, sigma, date]

    def update(self, g):
        h, a = self.st[g.home], self.st[g.away]
        adv = 0.0 if g.neutral else self.hfa
        ph, pa = self._phi_star(h, g.date), self._phi_star(a, g.date)
        m = g.home_pts - g.away_pts
        s = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
        new_h = self._update_one(h, a[0] - adv, pa, s, g.date)
        new_a = self._update_one(a, h[0] + adv, ph, 1 - s, g.date)
        self.st[g.home], self.st[g.away] = new_h, new_a
