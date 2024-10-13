import math

from options import Option
from options import PriceType
from scipy.optimize import bisect
from typing import Callable


def approximate_implied_volatility(option: Option,
                                   value: PriceType,
                                   S: PriceType,
                                   r: float,
                                   q: float | None,
                                   T: float) -> float:
    """
    Bharadia, M. A. J., N. Christofides, and G. R. Salkin.
    "Computing the Black-Scholes implied volatility: Generalization of a simple formula."
    Advances in futures and options research 8 (1995): 15-30.

    Review: https://www.sciencedirect.com/science/article/pii/S0377042717300602

    I don't know if you can use Q (dividends), but assuming SQ / KD is close to 1, it should work.
    """

    # C = SQ*N(d + x) - KD*N(d - x)
    # x = 1/2 * sigma * sqrt(T), d = (ln(F) - ln(K))/(sigma * sqrt(T)) where F = SQ/D
    #
    # N(d + x) - N(d - x) = 2xN'(d)  from Taylor's approximation
    # C = SQ[N(d + x) - N(d - x)] + (2delta)N(d - x) where delta = (SQ - KD)/2
    # C = 2xSQN'(d) + 2delta*N(d - x)
    #
    # When d is close to 0,
    # C = sigma * sqrt(T) * SQ / sqrt(2pi) + 2 * delta * (1/2 - sigma*sqrt(T)/sqrt(2pi))
    # sigma = (C - delta)/(SQ - delta) * sqrt(2pi / T)

    K = option.K
    q = q or 0

    D = math.exp(-r*T)
    Q = math.exp(-q*T)

    delta = (S*Q - K*D)/2

    if option.is_call():
        return math.sqrt(2 * math.pi / T) * (value - delta) / (S - delta)

    if option.is_put():
        return math.sqrt(2 * math.pi / T) * (value + delta) / (S - delta)

    raise NotImplementedError()


def compute_implied_volatility(option: Option,
                               value: PriceType,
                               S: PriceType,
                               r: float,
                               q: float | None,
                               T: float,
                               pricer: Callable,
                               **kwargs) -> float:
    # https://en.wikipedia.org/wiki/Newton%27s_method#Code
    def newtons(initial_guess_vol: float,
                tol: float = 1e-6,
                eps: float = 1e-6,
                maxiter: int = 100) -> float | None:
        vol = initial_guess_vol
        for i in range(maxiter):
            computed_value, greeks = pricer(option, S, r, q, T, vol, **kwargs)

            vega = greeks["vega"]
            if abs(vega) < eps:
                break

            new_vol = vol + (value - computed_value)/vega
            if abs(new_vol - vol) < tol:
                return new_vol

            vol = new_vol
        return None

    # https://github.com/opendoor-labs/pyfin/blob/master/pyfin/pyfin.py#L55C1-L55C66
    def binary_search(min_vol: float = -0.01,
                      max_vol: float = 100,
                      tol: float = 1e-6,
                      maxiter: int = 100) -> float:
        def f(vol):
            computed_value, greeks = pricer(option, S, r, q, T, vol, **kwargs)
            return value - computed_value

        return bisect(f, min_vol, max_vol, xtol=tol, maxiter=maxiter)

    initial_guess = approximate_implied_volatility(option, value, S, r, q, T)
    return newtons(initial_guess) or binary_search() or initial_guess
