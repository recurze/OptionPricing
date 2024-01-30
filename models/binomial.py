import math
from typing import Callable

from options import Option
from options import PriceType


# Note: there are alternate calibrators like Jarrow-Rudd and Tian.
def crr_calibrator(r: float, vol: float, t: float) -> tuple[float, float, float]:
    b = math.exp((r + vol*vol) * t) + math.exp(-r*t)
    u = (b + math.sqrt(b*b - 4)) / 2
    p = (u*math.exp(r*t) - 1) / (u*u - 1) if u != 1 else 1/2
    return (u, 1/u, p)


def price_option(option: Option,
                 risk_free_rate: float,
                 volatility: float,
                 time_to_maturity_in_years: float,
                 calibrator: Callable = crr_calibrator,
                 num_steps: int = 1) -> PriceType:

    r = risk_free_rate
    t = time_to_maturity_in_years / num_steps

    # To handle dividends, see:
    # https://homepage.ntu.edu.tw/~jryanwang/courses/Financial%20Computation%20or%20Financial%20Engineering%20(graduate%20level)/FE_Ch04%20Binomial%20Tree%20Model.pdf
    u, d, p = calibrator(r - option.dividend_yield, volatility, t)

    S = option.underlying.spot_price
    value = [option.payoff((u ** (num_steps - i)) * (d ** i) * S) for i in range(num_steps + 1)]

    discount_factor = math.exp(-r*t)
    for i in range(num_steps - 1, -1, -1):
        for j in range(i + 1):
            current_time_to_maturity = time_to_maturity_in_years - i*t

            current_payoff = option.payoff((u ** (i - j)) * (d ** j) * S, current_time_to_maturity)
            expected_payoff_upon_continuation = discount_factor * (p*value[j] + (1 - p)*value[j + 1])

            # Note: for European options, current_payoff is 0
            value[j] = max(current_payoff, expected_payoff_upon_continuation)

    return value[0]
