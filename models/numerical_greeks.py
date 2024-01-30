import copy
from typing import Callable

from options import Option


def compute_numerical_greeks(option: Option, pricer: Callable, **kwargs) -> dict[str, float]:
    def compute_delta(finite_difference_percentage: float) -> float:
        finite_difference = finite_difference_percentage * option.underlying.spot_price

        up = copy.deepcopy(option)
        up.underlying.spot_price += finite_difference

        down = copy.deepcopy(option)
        down.underlying.spot_price -= finite_difference

        high = pricer(up, **kwargs)
        low = pricer(down, **kwargs)

        return (high - low) / (2 * finite_difference)

    def compute_gamma(finite_difference_percentage: float) -> float:
        finite_difference = finite_difference_percentage * option.underlying.spot_price

        up = copy.deepcopy(option)
        up.underlying.spot_price += finite_difference

        down = copy.deepcopy(option)
        down.underlying.spot_price -= finite_difference

        high = pricer(up, **kwargs)
        low = pricer(down, **kwargs)
        same = pricer(option, **kwargs)

        return (high - 2*same + low) / (2 * finite_difference * finite_difference)

    def compute_theta(finite_difference: float) -> float:
        now = pricer(option, **kwargs)

        kwargs["time_to_maturity_in_years"] -= finite_difference
        later = pricer(option, **kwargs)

        return (later - now) / finite_difference

    def compute_rho(finite_difference: float) -> float:
        kwargs["risk_free_rate"] -= finite_difference
        low = pricer(option, **kwargs)

        kwargs["risk_free_rate"] += finite_difference
        high = pricer(option, **kwargs)

        return (high - low) / (2 * finite_difference)

    def compute_vega(finite_difference_percentage: float) -> float:
        finite_difference = finite_difference_percentage * kwargs["volatility"]

        kwargs["volatility"] -= finite_difference
        low = pricer(option, **kwargs)

        kwargs["volatility"] += finite_difference
        high = pricer(option, **kwargs)

        return (high - low) / (2 * finite_difference)

    return {
        "delta": compute_delta(0.1),
        "gamma": compute_gamma(0.1),
        "theta": compute_theta(0.004),
        "vega": compute_vega(0.1),
        "rho": compute_rho(0.0001),
    }
