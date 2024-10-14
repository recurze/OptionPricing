import numpy as np
from scipy.optimize import bisect  # type: ignore[import-untyped]


def is_close(a: float, b: float, tol=1e-6) -> bool:
    return abs(a - b) < tol


def subarray_in_range(x: np.ndarray, lx: float, rx: float) -> tuple[int, int]:
    i_min, i_max = bisect.bisect_left(x, lx), bisect.bisect_right(x, rx)
    assert lx <= x[i_min] and x[i_max - 1] <= rx
    return i_min, i_max
