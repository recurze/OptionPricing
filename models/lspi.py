# References:
# LSPI: https://users.cs.duke.edu/~parr/jmlr03.pdf
# LSPI for Options: https://proceedings.mlr.press/v5/li09d/li09d.pdf

import gbm
import math
import numpy as np

from options import AmericanOption
from options import PriceType


def feature_maps(simulated_paths: list[list[PriceType]],
                 K: PriceType,
                 T: float,
                 num_paths: int,
                 num_steps: int) -> np.ndarray:

    S = np.array(simulated_paths).flatten() / K
    fs = np.vstack([
        np.ones_like(S),
        np.exp(-S/2),
        np.exp(-S/2) * (1 - S),
        np.exp(-S/2) * (1 - 2*S + 0.5*S*S)
    ])

    t = np.arange(num_steps)

    dt = T / num_steps
    ft = np.vstack([
        np.sin(0.5 * math.pi * (1 - t/num_steps)),
        np.log(T - t*dt),
        (t / num_steps) ** 2
    ])
    ft = np.hstack([ft, np.array([0, 0, 1]).reshape(3, 1)])
    ft = np.tile(ft, num_paths)

    return np.vstack([fs, ft])


def price_option(option: AmericanOption,
                 S: PriceType,
                 r: float,
                 q: float | None,
                 sigma: float,
                 T: float,
                 # For simulations
                 num_paths: int = 5000,
                 num_steps: int = 50,
                 # For policy iteration
                 tol: float = 1e-6,
                 maxiter: int = 20) -> PriceType:

    dt = T / num_steps
    mu = r - (q or 0)

    # Step 1: Simulation
    steps = np.arange(0, T + dt/2, dt)
    simulated_paths = [gbm.simulate1d(S, mu, sigma, steps) for j in range(num_paths)]
    total_num_price_points = num_paths * (num_steps + 1)

    phi = feature_maps(simulated_paths, option.K, T, num_paths, num_steps)
    assert phi.shape[1] == total_num_price_points

    payoffs = np.array([
        option.payoff(S, T - t*dt)
        for path in simulated_paths for (t, S) in enumerate(path)
    ])
    assert payoffs.shape == (total_num_price_points,)

    k = phi.shape[0]
    gamma = math.exp(-r * dt)

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
    immediate_payoff = option.payoff(S, T)

    return max(expected_payoff, immediate_payoff)
