import math

from calibrators import crr_calibrator
from options import Option
from options import AmericanOption
from options import EuropeanOption
from options import PriceType
from pricer import black_scholes
from typing import Callable


def price_option(option: Option,
                 S: PriceType,
                 r: float,
                 q: float | None,
                 vol: float,
                 T: float,
                 calibrator: Callable = crr_calibrator,
                 num_steps: int = 50) -> PriceType:
    mu = r - (q or 0)
    dt = T / num_steps
    u, d, p = calibrator(mu, vol, dt)

    value = [option.payoff((u ** (num_steps - i)) * (d ** i) * S) for i in range(num_steps + 1)]

    for i in range(num_steps - 1, -1, -1):
        for j in range(i + 1):
            tau = T - i*dt

            current_spot = (u ** (i - j)) * (d ** j) * S
            current_payoff = option.payoff(current_spot, tau)

            if i == num_steps - 1 and isinstance(option, AmericanOption):
                # Attempt to smoothen:
                # Typically, the final time slice is non-differentiable at strike, but Black-Scholes formula is.
                # AmericanOption that is not exercised at the penultimate step can be treated as an EuropeanOption.
                european_option = EuropeanOption(option.K, option.option_type)
                expected_payoff_upon_continuation = black_scholes.price_option(european_option,
                                                                               current_spot,
                                                                               r,
                                                                               q,
                                                                               vol,
                                                                               tau)
            else:
                expected_payoff_upon_continuation = math.exp(-r*dt) * (p*value[j] + (1 - p)*value[j + 1])

            value[j] = max(current_payoff, expected_payoff_upon_continuation)

    return value[0]
