import math

from options import EuropeanOption
from options import PriceType
from scipy.stats import norm  # type: ignore[import-untyped]
from scipy.optimize import bisect  # type: ignore[import-untyped]
from typing import Callable


def undiscounted_pricer(option: EuropeanOption,
                        F: PriceType,
                        vol: float,
                        T: float) -> PriceType:
    sd = vol * math.sqrt(T)
    d1 = math.log(F / option.K)/sd + sd/2
    d2 = d1 - sd

    K = option.K
    return F*norm.cdf(d1) - K*norm.cdf(d2) if option.is_call() else K*norm.cdf(-d2) - F*norm.cdf(-d1)


def compute_implied_volatility(option: EuropeanOption,
                               value: PriceType,
                               F: PriceType,
                               T: float,
                               pricer: Callable = undiscounted_pricer) -> float:

    def binary_search(f: Callable,
                      min_x: float,
                      max_x: float,
                      tol: float = 1e-6,
                      maxiter: int = 100) -> float:
        return bisect(f, min_x, max_x, xtol=tol, maxiter=maxiter)

    return binary_search(lambda vol: pricer(option, F, vol, T) - value, min_x=-0.01, max_x=100)


def price_option(option: EuropeanOption,
                 S: PriceType,
                 r: float,
                 q: float | None,
                 vol: float,
                 T: float) -> PriceType:
    mu = r - (q or 0)
    fwd = S * math.exp(mu*T)
    discount = math.exp(-r*T)
    return discount * undiscounted_pricer(option, fwd, vol, T)


def price_quanto_option(option: EuropeanOption,
                        F: PriceType,
                        Sa: PriceType,
                        Sb: PriceType,
                        ra: float,
                        rb: float,
                        rd: float,
                        rho: float,
                        volA: float,
                        volB: float,
                        T: float) -> PriceType:
    mu = rd - ra + (rho * volA * volB)
    fwd = Sa * math.exp(mu*T)
    discount = math.exp(-rb*T)
    return F * Sb * discount * undiscounted_pricer(option, fwd, volA, T)
