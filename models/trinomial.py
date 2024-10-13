import math

from calibrators import lam_calibrator
from options import Option
from options import PriceType
from typing import Callable


def compute_lambda(K: PriceType,
                   S: PriceType,
                   sd: float,
                   num_steps: int,
                   default_lam: float = math.sqrt(3)) -> float:
    if K == S:
        return default_lam

    # S * (u ** up_minus_down) = K
    # up_minus_down * lam = G = 1/sd * log(K/S)
    G = 1/sd * math.log(K / S)

    # Use default_lam to compute up_minus_down rounded to nearest integer that is not 0
    up_minus_down = G / default_lam
    rounded_up_minus_down = round(up_minus_down) if abs(up_minus_down) > 0.5 else (1 if up_minus_down > 0 else -1)

    # Use rounded_up_minus_down to compute lam
    return max(G / rounded_up_minus_down, 1)


def price_option(option: Option,
                 S: PriceType,
                 r: float,
                 q: float | None,
                 vol: float,
                 T: float,
                 lam: float | None,
                 anchor: float | None,
                 calibrator: Callable = lam_calibrator,
                 num_steps: int = 50) -> PriceType:
    mu = r - (q or 0)
    dt = T / num_steps
    sd = vol * math.sqrt(dt)

    lam = lam or compute_lambda(anchor or option.K, S, sd, num_steps)
    u, pd, pm, pu = calibrator(mu, vol, dt, lam)

    value = [option.payoff((u ** (num_steps - i)) * S) for i in range(2*num_steps + 1)]

    for i in range(num_steps - 1, -1, -1):
        for j in range(2*i + 1):
            tau = T - i*dt

            current_payoff = option.payoff((u ** (i - j)) * S, tau)
            expected_payoff_upon_continuation = math.exp(-r*dt) * (pu*value[j] + pm*value[j + 1] + pd*value[j + 2])

            value[j] = max(current_payoff, expected_payoff_upon_continuation)

    return value[0]
