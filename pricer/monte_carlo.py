import gbm
import math
import numpy as np

from options import Option
from options import PathIndependentOption
from options import PriceType
from scipy.stats import norm  # type: ignore[import-untyped]
from typing import Callable


def montecarlo(simulate_path: Callable,
               f: Callable,
               num_paths: int,
               steps: np.ndarray,
               confidence: float = 0.95):
    x = [f(simulate_path(steps)) for i in range(num_paths)]

    mu = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(num_paths)

    lb, ub = norm.interval(confidence, mu, scale=se)
    return mu, lb, ub


def price_option(option: Option,
                 S: PriceType,
                 r: float,
                 q: float | None,
                 vol: float,
                 T: float,
                 num_paths: int = 50,
                 num_steps: int = 100) -> PriceType:

    def simulate_path(steps: np.ndarray):
        return gbm.simulate1d(S, r - (q or 0), vol, steps)

    D = math.exp(-r * T)

    def payoff(path: list[PriceType]):
        if isinstance(option, PathIndependentOption):
            return D * option.payoff(path[-1])
        return D * option.payoff(path)

    dt = T/num_steps
    steps = np.arange(0, T + dt/2, dt)
    return montecarlo(simulate_path, payoff, num_paths, steps)


def price_quanto_option(option: Option,
                        F: PriceType,
                        Sa: PriceType,
                        Sb: PriceType,
                        ra: float,
                        rb: float,
                        rd: float,
                        rho: float,
                        volA: float,
                        volB: float,
                        T: float,
                        num_paths: int = 50,
                        num_steps: int = 100) -> PriceType:

    cov = np.array([[1, rho], [rho, 1]])

    def simulate_path(steps: np.ndarray) -> tuple[list[PriceType], list[PriceType]]:
        return gbm.simulate2d(np.array([Sa, Sb]),
                              np.array([rd - ra, rd - rb]),
                              np.array([volA, volB]),
                              cov,
                              steps)

    D = np.exp(-rb * T)

    def payoff(path: tuple[list[PriceType], list[PriceType]]):
        if isinstance(option, PathIndependentOption):
            return D * F * option.quanto_payoff(Sa=path[0][-1], Sb=path[1][-1])
        return D * F * option.quanto_payoff(Sa=path[0], Sb=path[1][-1])

    dt = T/num_steps
    steps = np.arange(0, T + dt/2, dt)
    return montecarlo(simulate_path, payoff, num_paths, steps)
