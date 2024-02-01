# References:
# LSPI: https://users.cs.duke.edu/~parr/jmlr03.pdf
# LSPI for Options: https://proceedings.mlr.press/v5/li09d/li09d.pdf

import math
import numpy as np

from gbm import GeometricBrownianMotion
from options import Option
from options import PriceType


def feature_map(S: PriceType,
                t: float,
                T: float,
                dt: float,
                num_steps: int) -> np.ndarray:

    if t == num_steps:
        return np.zeros(7)

    # dt = T/num_steps
    return np.array([
        1,
        math.exp(-S/2),
        math.exp(-S/2) * (1 - S),
        math.exp(-S/2) * (1 - 2*S + 0.5*S*S),

        math.sin(0.5 * math.pi * (1 - t/num_steps)),
        math.log(T - t*dt) if t != num_steps else 0,
        (t / num_steps)**2
    ])


def price_option(option: Option,
                 risk_free_rate: float,
                 volatility: float,
                 time_to_maturity_in_years: float,
                 # For simulations
                 num_paths: int = 5000,
                 num_steps: int = 50,
                 # For policy iteration
                 tol: float = 1e-6,
                 maxiter: int = 20) -> PriceType:

    dt = time_to_maturity_in_years / num_steps
    drift = risk_free_rate - option.dividend_yield

    # Step 1: Simulation
    simulated_paths = [
        GeometricBrownianMotion.simulate(
            init=option.underlying.spot_price,
            mu=drift,
            sigma=volatility,
            dt=dt,
            num_steps=num_steps
        )
        for j in range(num_paths)
    ]
    total_num_price_points = num_paths * (num_steps + 1)

    phi = np.array([
        feature_map(
            S=S,
            t=t,
            T=time_to_maturity_in_years,
            dt=dt,
            num_steps=num_steps
        )
        for path in simulated_paths for (t, S) in enumerate(path)
    ]).T
    assert phi.shape[1] == total_num_price_points

    payoffs = np.array([
        option.payoff(S, time_to_maturity_in_years - t*dt)
        for path in simulated_paths for (t, S) in enumerate(path)
    ])
    assert payoffs.shape == (total_num_price_points,)

    k = phi.shape[0]
    gamma = math.exp(-risk_free_rate * dt)

    A = phi @ phi.T
    assert A.shape == (k, k)

    phi_dash = phi.reshape((k, num_paths, num_steps + 1))[:, :, :-1].reshape((k, -1))

    def LSTDQ(w: np.ndarray) -> np.ndarray:
        Q = np.maximum(w.dot(phi), 0)
        assert Q.shape == (total_num_price_points,)

        to_continue = (payoffs < Q)

        psi = (to_continue * phi).reshape((k, num_paths, num_steps + 1))[:, :, 1:].reshape((k, -1))

        a = phi_dash[:, None, :] * psi[None, :, :]
        assert a.shape == (k, k, num_paths * num_steps), a.shape

        g = ((~to_continue) * payoffs).reshape((num_paths, num_steps + 1))[:, 1:].reshape(-1)
        b = gamma * (phi_dash @ g)
        assert b.shape == (k,)

        return np.linalg.lstsq(A - gamma * np.sum(a, axis=2), b, rcond=None)[0]

    def LSPI() -> np.ndarray:
        w = np.zeros(k)
        for i in range(maxiter):
            new_w = LSTDQ(w)
            if np.linalg.norm(new_w - w) < tol:
                return new_w
            w = new_w
        return w

    def greedy_exercise_payoff(w: np.ndarray):
        Q = np.maximum(w.dot(phi), 0)
        to_exercise = (payoffs >= Q)

        def first_nonzero(a: np.ndarray) -> float:
            nz = np.nonzero(a)
            return a[nz][0] if len(a[nz]) > 0 else 0

        payoff_from_best_policy = np.apply_along_axis(
            first_nonzero,
            axis=1,
            arr=(to_exercise * payoffs).reshape((num_paths, num_steps + 1))
        )
        assert payoff_from_best_policy.shape == (num_paths,)

        return np.mean(payoff_from_best_policy)

    # Step 2: Learn optimal stopping policy
    w = LSPI()

    # Step 3: Compute expected payoff
    expected_payoff = greedy_exercise_payoff(w)
    return round(expected_payoff, 2)
