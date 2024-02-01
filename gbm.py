import numpy as np


class GeometricBrownianMotion:
    @staticmethod
    def simulate(init: float,
                 mu: float,
                 sigma: float,
                 dt: float,
                 num_steps: int) -> list[float]:

        # W_t+dt - W_t ~ N(0, dt)
        # S_t+dt = S_t * exp((mu - 1/2 sigma^2)dt) * exp(sigma * (W_t+dt - W_t))

        multiplicative_factor = np.exp((mu - 0.5 * (sigma ** 2)) * dt)
        noise = np.exp(sigma * np.random.normal(0, dt ** 0.5, num_steps))

        path = [init]
        for t in range(num_steps):
            path.append(path[t] * multiplicative_factor * noise[t])

        return path

    @staticmethod
    def estimate_parameters(series: np.ndarray, dt: float) -> tuple[float, float]:
        # Ref:
        # https://allenfrostline.com/blog/estimation-brownian-motion/
        # https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/08.03-Binomial-Model-for-Pricing-Options.html

        log_series_diff = np.diff(np.log(series))

        observed_mean = np.mean(log_series_diff)
        observed_var = np.var(log_series_diff)

        estimated_mu = (2*observed_mean + observed_var) / (2 * dt)
        estimated_sigma = observed_var / dt

        return estimated_mu, estimated_sigma ** 0.5
