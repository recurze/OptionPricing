from enum import Enum
from options import Option
from options import PriceType
from typing import Callable


class GreekType(Enum):
    DELTA = "DELTA"
    GAMMA = "GAMMA"
    THETA = "THETA"
    RHO = "RHO"
    VEGA = "VEGA"


def numerical_greeks(greek_type: GreekType,
                     pricer: Callable,
                     option: Option,
                     S: PriceType,
                     r: float,
                     q: float | None,
                     vol: float,
                     T: float,
                     **kwargs) -> float:

    def compute_delta(finite_difference_ratio: float = 0.001) -> float:
        assert finite_difference_ratio > 0
        finite_difference = finite_difference_ratio * S

        high = pricer(option, S + finite_difference, r, q, vol, T, **kwargs)
        low = pricer(option, S - finite_difference, r, q, vol, T, **kwargs)

        return (high - low) / (2 * finite_difference)

    def compute_gamma(finite_difference_ratio: float = 0.001) -> float:
        assert finite_difference_ratio > 0
        finite_difference = finite_difference_ratio * S

        high = pricer(option, S + finite_difference, r, q, vol, T, **kwargs)
        same = pricer(option, S, r, q, vol, T, **kwargs)
        low = pricer(option, S - finite_difference, r, q, vol, T, **kwargs)

        return (high - 2*same + low) / (finite_difference * finite_difference)

    def compute_theta(finite_difference: float = 0.004) -> float:
        assert T >= finite_difference
        assert finite_difference != 0

        now = pricer(option, S, r, q, vol, T, **kwargs)
        later = pricer(option, S, r, q, vol, T - finite_difference, **kwargs)

        return (later - now) / finite_difference

    def compute_rho(finite_difference: float = 0.0001) -> float:
        assert finite_difference != 0

        high = pricer(option, S, r + finite_difference, q, vol, T, **kwargs)
        low = pricer(option, S, r - finite_difference, q, vol, T, **kwargs)

        return (high - low) / (2 * finite_difference)

    def compute_vega(finite_difference_ratio: float = 0.001) -> float:
        assert finite_difference_ratio > 0
        finite_difference = finite_difference_ratio * vol

        high = pricer(option, S, r, q, vol + finite_difference, T, **kwargs)
        low = pricer(option, S, r, q, vol - finite_difference, T, **kwargs)

        return (high - low) / (2 * finite_difference)

    if greek_type == GreekType.DELTA:
        return compute_delta()
    if greek_type == GreekType.GAMMA:
        return compute_gamma()
    if greek_type == GreekType.THETA:
        return compute_theta()
    if greek_type == GreekType.VEGA:
        return compute_vega() / 100
    if greek_type == GreekType.RHO:
        return compute_rho() / 100
