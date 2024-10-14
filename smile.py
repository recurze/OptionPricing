import cvxpy as cp
import math
import numpy as np
import utils

from pricer import black_scholes
from options import PriceType
from options import OptionType
from options import EuropeanOption
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]


class Smile:
    # Source: QF5204, 2024 sem II, Prof. Li Hao (with some mods).
    def __init__(self, k: list[PriceType], sigma: list[float], T: float = 0):
        assert len(k) >= 2, "Need at least 2 points to extrapolate"

        # add an additional point on the right to avoid arbitrage
        _k, _s = (1.1*k[-1] - 0.1*k[-2]), (sigma[-1] + (sigma[-1] - sigma[-2])/10)
        self.k, self.sigma = k + [_k], sigma + [_s]

        # Should we use (k, sigma) or (self.k, sigma)
        self.cubic_spline = CubicSpline(k, sigma, bc_type="clamped")

    def Vol(self, K: PriceType) -> float:
        if K < self.k[0]:
            return self.sigma[0]

        if K > self.k[-1]:
            return self.sigma[-1]

        return self.cubic_spline(K)


class SmileAF(Smile):
    def __init__(self, k_hat: list[PriceType], sigma: list[float], T: float):
        M = len(k_hat)
        assert M >= 2, "Need at least 2 points to extrapolate"
        assert M % 2 == 1, "Need a unique middle element (atmvol)"

        F, atmvol = k_hat[M // 2], sigma[M // 2]
        ks = SmileAF.generate_evenly_spaced_k(F, atmvol, T)

        c_hat = np.array([
            black_scholes.undiscounted_pricer(EuropeanOption(k, OptionType.CALL), F, vol, T)
            for (k, vol) in zip(k_hat, sigma)
        ])

        cs, _ = SmileAF.solve_af_cs_ps(ks, k_hat, c_hat, F)

        self.k, self.sigma = (list(x) for x in zip(*SmileAF.get_kv_pairs(ks, cs, k_hat, F, T)))
        self.cubic_spline = CubicSpline(self.k, self.sigma, bc_type="clamped")

    @staticmethod
    def get_kv_pairs(ks: np.ndarray,
                     cs: np.ndarray,
                     k_hat: list[PriceType],
                     F: PriceType,
                     T: float) -> list[tuple[PriceType, float]]:
        # With (cs, ks), we can compute implied volatilities.
        # But, at the tails, the price-vol gradient is low and is not numerically stable. So:
        # 1. Compute volatilities for all points within 25 delta points.
        # 2. Flatten volatilities for the points that are outside 25 delta points.
        i_min, i_max = utils.subarray_in_range(ks, k_hat[0], k_hat[-1])
        kvs = [
            (ks[i], black_scholes.compute_implied_volatility(EuropeanOption(ks[i], OptionType.CALL), cs[i], F, T))
            for i in range(i_min - 1, i_max + 1)
        ]
        kvs = [(ks[0], kvs[0][1])] + kvs + [(ks[-1], kvs[-1][1])]
        return kvs

    @staticmethod
    def generate_evenly_spaced_k(F: PriceType,
                                 atmvol: float,
                                 T: float,
                                 num_sd: int = 5,
                                 num_points: int = 50) -> np.ndarray:
        sd = atmvol * math.sqrt(T)
        kmin, kmax = F * math.exp(-sd*sd/2 - num_sd*sd), F * math.exp(-sd*sd/2 + num_sd*sd)
        dk = (kmax - kmin) / (num_points - 1)
        return np.arange(kmin, kmax + dk/2, dk)

    @staticmethod
    def solve_af_cs_ps(ks: np.ndarray,
                       k_hat: list[float],
                       c_hat: np.ndarray,
                       F: float) -> tuple[np.ndarray, np.ndarray]:
        u = ks[1] - ks[0]

        n = ks.shape[0]
        p = cp.Variable(n)
        c = cp.Variable(n)

        constraints = []

        # CONSTRAINT: Probability density
        constraints.append(p >= 0)
        constraints.append(p[0] == 0)
        constraints.append(p[-1] == 0)
        constraints.append(u * cp.sum(p) == 1)

        # CONSTRAINT: c_1 and c_n
        constraints.append(c[0] == F - ks[0])
        constraints.append(c[-1] == 0)

        # CONSTRAINT: Monotonically decreasing c
        A = (np.eye(n) + np.diagflat(-np.ones(n - 1), 1))[:-1]
        constraints.append(A @ c >= 0)

        # CONSTRAINT: left and right first derivative match
        q = -1*np.eye(n) + np.diagflat(np.ones(n - 1), 1)
        r = (2*np.eye(n) + np.diagflat(np.ones(n - 1), 1)) / 6
        Q, R = (q + q.T), (u*u) * (r + r.T)
        constraints.append(Q[1: -1]@c == R[1: -1]@p)

        def get_abcd(i, x):
            assert i > 0

            a, b = (ks[i] - x) / u, (x - ks[i - 1]) / u
            assert a >= 0 and b >= 0 and utils.is_close(a + b, 1)

            c, d = (u*u)/6 * (a**3 - a), (u*u)/6 * (b**3 - b)

            ab = np.zeros(n)
            ab[i - 1], ab[i] = a, b

            cd = np.zeros(n)
            cd[i - 1], cd[i] = c, d

            return ab, cd

        # CONSTRAINT: Match input marks exactly
        i_hat = np.searchsorted(ks, k_hat, side="left")
        AB, CD = zip(*[get_abcd(i_k, k_hat[i]) for i, i_k in enumerate(i_hat)])
        constraints.append(np.vstack(AB)@c + np.vstack(CD)@p == c_hat)

        problem = cp.Problem(cp.Minimize(p.T @ R @ p), constraints)
        problem.solve(solver=cp.CVXOPT)
        assert problem.status == cp.OPTIMAL, f"The problem is {problem.status}"

        return c.value, p.value


def strike_from_delta(S: PriceType,
                      r: float,
                      q: float | None,
                      vol: float,
                      T: float,
                      delta: float,
                      option_type: OptionType) -> PriceType:
    delta = abs(delta)

    F = S * math.exp((r - (q or 0)) * T)
    sd = vol * (T ** 0.5)
    d1 = norm.ppf(delta if option_type == OptionType.CALL else 1 - delta)

    return F / math.exp((d1 - sd/2) * sd)


def smile_from_marks(S: PriceType,
                     r: float,
                     q: float | None,
                     atmvol: float,
                     bf25: float,
                     rr25: float,
                     bf10: float,
                     rr10: float,
                     T: float,
                     smile_cls: type[Smile] = Smile) -> Smile:

    call_vol10 = atmvol + bf10 + rr10/2
    CK10 = strike_from_delta(S, r, q, call_vol10, T, 0.1, OptionType.CALL)

    call_vol25 = atmvol + bf25 + rr25/2
    CK25 = strike_from_delta(S, r, q, call_vol25, T, 0.25, OptionType.CALL)

    put_vol10 = atmvol + bf10 - rr10/2
    PK10 = strike_from_delta(S, r, q, put_vol10, T, -0.10, OptionType.PUT)

    put_vol25 = atmvol + bf25 - rr25/2
    PK25 = strike_from_delta(S, r, q, put_vol25, T, -0.25, OptionType.PUT)

    K_atm = S * math.exp((r - (q or 0)) * T)

    strikes = [PK10, PK25, K_atm, CK25, CK10]
    vols = [put_vol10, put_vol25, atmvol, call_vol25, call_vol10]

    return smile_cls(strikes, vols, T)
