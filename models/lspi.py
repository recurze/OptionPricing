# References:
# LSPI: https://users.cs.duke.edu/~parr/jmlr03.pdf
# LSPI for Options: https://proceedings.mlr.press/v5/li09d/li09d.pdf

import math
import numpy as np

from options import Option
from options import PriceType


def simulate_prices(starting_price: PriceType,
                    mu: float,
                    sigma: float,
                    dt: float,
                    num_steps: int) -> list[PriceType]:

    # Geometric Brownian Motion
    # W_t+dt - W_t ~ N(0, dt)
    # S_t+dt = S_t * exp((mu - 1/2 sigma^2)dt) * exp(sigma * (W_t+dt - W_t))

    multiplicative_factor = math.exp((mu - 0.5 * (sigma ** 2)) * dt)
    noise = np.exp(sigma * np.random.normal(0, dt ** 0.5, num_steps))

    path = [starting_price]
    for t in range(num_steps):
        path.append(path[t] * multiplicative_factor * noise[t])

    return path


def feature_map(S: PriceType,
                T: float,
                t: float,
                dt: float,
                num_steps: int) -> np.ndarray:

    # dt = T/num_steps
    return np.array([
        1,
        math.exp(-S/2),
        math.exp(-S/2) * (1 - S),
        math.exp(-S/2) * (1 - 2*S + 0.5*S*S),

        math.sin(0.5 * math.pi * (1 - t/num_steps)),
        math.log(T - t*dt) if t != num_steps else math.log(dt),
        (t / num_steps)**2
    ])


def price_option(option: Option,
                 risk_free_rate: float,
                 volatility: float,
                 time_to_maturity_in_years: float,
                 # For simulations
                 num_paths: int = 500,
                 num_steps: int = 100,
                 # For policy iteration
                 tol: float = 1e-6,
                 maxiter: int = 20) -> PriceType:

    def simulate(drift: float, dt: float) -> tuple[list[PriceType], list[np.ndarray], list[PriceType]]:
        path = simulate_prices(
            starting_price=option.underlying.spot_price,
            mu=drift,
            sigma=volatility,
            dt=dt,
            num_steps=num_steps
        )
        phi = [
            feature_map(
                S=(path[t] / option.strike_price),
                T=time_to_maturity_in_years,
                t=t,
                dt=dt,
                num_steps=num_steps
            )
            for t in range(num_steps + 1)
        ]
        payoffs = [option.payoff(path[t], time_to_maturity_in_years - t*dt) for t in range(num_steps + 1)]

        return path, phi, payoffs

    def LSTDQ(k: int, w: np.ndarray, gamma: float) -> np.ndarray:
        def learn_from_path(path: list[PriceType],
                            phi: list[np.ndarray],
                            payoffs: list[PriceType],
                            A: np.ndarray,
                            b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            for t in range(num_steps + 1 - 1):
                next_payoff = payoffs[t + 1]
                next_expected_continuation = max(np.dot(w, phi[t + 1]), 0) if t != num_steps - 1 else 0

                if next_expected_continuation < next_payoff:
                    # Exercise at time t + 1
                    A += np.outer(phi[t], phi[t])
                    b += gamma * next_payoff * phi[t]

                    # Should we return A, b here?
                else:
                    A += np.outer(phi[t], phi[t] - gamma*phi[t + 1])
            return A, b

        A, b = np.zeros((k, k)), np.zeros(k)
        for j in range(num_paths):
            path, phi, payoffs = simulations[j]
            A, b = learn_from_path(path, phi, payoffs, A, b)

        return np.linalg.solve(A, b)

    def LSPI(k: int, gamma: float) -> np.ndarray:
        w = np.zeros(k)
        for i in range(maxiter):
            new_w = LSTDQ(k, w, gamma)
            if np.linalg.norm(new_w - w) < tol:
                return new_w
            w = new_w
        return w

    def greedy_exercise_payoff(path: list[PriceType],
                               phi: list[np.ndarray],
                               payoffs: list[PriceType],
                               w: np.ndarray) -> PriceType:
        for t in range(num_steps):
            current_payoff = payoffs[t]

            # gamma is already accounted for, this is Q(s, a)
            expected_payoff_upon_continuation = max(np.dot(w, phi[t]), 0)

            if current_payoff > expected_payoff_upon_continuation:
                return current_payoff
        return payoffs[-1]

    # Step 1: simulate paths
    dt = time_to_maturity_in_years / num_steps
    drift = risk_free_rate - option.dividend_yield
    simulations = [simulate(drift=drift, dt=dt) for j in range(num_paths)]

    # Step 2: Learn optimal stopping with LSPI
    k = len(simulations[0][1][0])
    gamma = math.exp(-risk_free_rate * dt)
    w = LSPI(k, gamma)

    # Step 3: Reap benefits of optimal stopping
    mean_payoff = sum(greedy_exercise_payoff(path, phi, payoffs, w) for (path, phi, payoffs) in simulations) / num_paths
    return round(mean_payoff, 2)
