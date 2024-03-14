import math

from options import Option
from options import PriceType


def compute_lambda(K: float,
                   S: float,
                   sd: float,
                   num_steps: int,
                   lam0: float = 3**0.5) -> float:
    if K == S:
        return lam0

    G = 1/sd * math.log(K / S)
    up_minus_down = G / lam0

    # We can't choose up_minus_down = 0 when K != S
    rounded_up_minus_down = round(up_minus_down) if abs(up_minus_down) > 0.5 else (1 if up_minus_down > 0 else -1)

    return G / rounded_up_minus_down


def calibrator(r: float, vol: float, t: float, lam: float) -> tuple[float, float, float, float]:
    sd = vol * (t ** 0.5)
    u = math.exp(lam * sd)

    dd = (sd/(4 * lam)) * (2*r/(vol*vol) - 1)
    pm = 1 - 1/(lam*lam)
    pu = (1 - pm)/2 + dd
    pd = (1 - pm)/2 - dd

    return u, pu, pm, pd


def price_option(option: Option,
                 risk_free_rate: float,
                 volatility: float,
                 time_to_maturity_in_years: float,
                 anchor: float,
                 num_steps: int = 50) -> PriceType:

    r = risk_free_rate
    t = time_to_maturity_in_years / num_steps

    S = option.underlying.spot_price

    sd = volatility * (t ** 0.5)
    lam = compute_lambda(anchor, S, sd, num_steps)

    u, pu, pm, pd = calibrator(r - option.dividend_yield, volatility, t, lam)

    value = [option.payoff((u ** (num_steps - i)) * S) for i in range(2*num_steps + 1)]

    discount_factor = math.exp(-r*t)
    for i in range(num_steps - 1, -1, -1):
        for j in range(2*i + 1):
            current_time_to_maturity = time_to_maturity_in_years - i*t

            current_payoff = option.payoff((u ** (i - j)) * S, current_time_to_maturity)
            expected_payoff_upon_continuation = discount_factor * (pu*value[j] + pm*value[j + 1] + pd*value[j + 2])

            value[j] = max(current_payoff, expected_payoff_upon_continuation)

    return round(value[0], 2)
