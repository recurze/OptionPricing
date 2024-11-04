# Source: QF5204, 2024 sem II

import math
import numpy as np

from options import Option
from options import PriceType
from vol import LocalVol


def price_option(option: Option,
                 S0: PriceType,
                 r: float,
                 q: float | None,
                 lv: LocalVol,
                 T: float,
                 NX: int,
                 NT: int,
                 w: float,
                 num_sd: int = 5) -> PriceType:
    mu = r - (q or 0)
    X0, vol0 = math.log(S0), lv.Vol(0, S0)

    xrange = num_sd * vol0 * math.sqrt(T)
    # Should it be minX = X0 + ... instead of X0 - ... ?
    minX = X0 - (mu - 0.5*vol0*vol0)*T - xrange
    maxX = X0 + (mu - 0.5*vol0*vol0)*T + xrange

    dx = (maxX - minX) / (NX - 1)
    X = np.arange(minX, maxX + dx/2, dx)
    S = np.exp(X)

    ps = np.array([option.payoff(s) for s in S])
    for j in range(1, NT):
        dt = T / (NT - 1)

        M = np.zeros((NX, NX))
        M[0, 0] = M[-1, -1] = 1

        for i in range(1, NX - 1):
            vol = lv.Vol(j * dt, S[i])

            M[i, i] = r + (vol * vol)/(dx * dx)
            M[i, i - 1] = (mu - 0.5 * vol * vol)/(2 * dx) - (vol * vol)/(2 * dx * dx)
            M[i, i + 1] = -(mu - 0.5 * vol * vol)/(2 * dx) - (vol * vol)/(2 * dx * dx)

        D = np.eye(NX)
        D[0, 0] = D[-1, -1] = 0

        rhsM = w * (D - dt * M) + (1 - w) * np.eye(NX)
        ps = rhsM.dot(ps)

        discount_factor = math.exp(-r * j * dt)
        ps[0] = dt * discount_factor * option.payoff(S[0])
        ps[-1] = dt * discount_factor * option.payoff(S[-1])

        lhsM = w * np.eye(NX) + (1 - w) * (D + dt * M)
        ps = np.linalg.inv(lhsM).dot(ps)

    return np.interp(X0, X, ps)  # type: ignore[return-value]
