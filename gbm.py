import numpy as np


def simulate1d(init: float,
               mu: float,
               sigma: float,
               steps: np.ndarray) -> list[float]:
    dt = steps[1:] - steps[:-1]
    num_steps = len(steps) - 1

    Z = np.random.standard_normal(num_steps)

    mf = np.exp(dt * (mu - 0.5 * (sigma ** 2)))
    noise = np.exp(np.sqrt(dt) * sigma * Z)

    path = [init]
    for t in range(num_steps):
        path.append(path[t] * mf[t] * noise[t])

    return path


def simulate2d(init: np.ndarray,
               mu: np.ndarray,
               sigma: np.ndarray,
               cov: np.ndarray,
               steps: np.ndarray) -> tuple[list[float], list[float]]:
    dt = steps[1:] - steps[:-1]
    num_steps = len(steps) - 1

    # CZ ~ N(0, CC^T) where Z ~ N(0, I)
    C = np.linalg.cholesky(cov)
    Z = np.random.standard_normal((num_steps, init.shape[0]))
    W = Z.dot(C.T)

    mf = np.exp(dt[:, np.newaxis] * (mu - 0.5 * (sigma ** 2)))
    noise = np.exp(np.sqrt(dt[:, np.newaxis]) * sigma * W)

    path = [init]
    for t in range(num_steps):
        path.append(path[t] * mf[t] * noise[t])

    return ([s[0] for s in path], [s[1] for s in path])
