import math
from scipy.stats import norm

from options import EuropeanOption
from options import PriceType


def price_option(option: EuropeanOption,
                 risk_free_rate: float,
                 volatility: float,
                 time_to_maturity_in_years: float) -> tuple[PriceType, dict[str, float]]:
    # Using standard notation (also more concise)
    S = option.underlying.spot_price
    K = option.strike_price
    r = risk_free_rate
    q = option.dividend_yield
    sigma = volatility

    # Time to maturity
    tau = time_to_maturity_in_years

    # Compute d +/-
    sqrt_tau = math.sqrt(tau)
    sd = sigma * sqrt_tau

    d1 = (math.log(S / K) + (r - q) * tau)/sd + sd/2
    d2 = d1 - sd

    d1_cdf = norm.cdf(d1)
    d1_pdf = norm.pdf(d1)
    d2_cdf = norm.cdf(d2)

    # Compute value and greeks
    # * Delta: partial derivative of V w.r.t S
    # * Gamma: second partial derivative of V w.r.t S
    # * Theta: partial derivative of V w.r.t t
    # * Vega (nu) : partial derivative of V w.r.t $\sigma$
    # * Rho : partial derivative of V w.r.t r

    # Discount factor and dividends
    D = math.exp(-r*tau)
    Q = math.exp(-q*tau)

    SQ, KD = S*Q, K*D
    value = SQ*d1_cdf - KD*d2_cdf
    greeks = {
        "delta": Q*d1_cdf,
        "gamma": (Q*d1_pdf) / (S*sd),
        "theta": - (SQ*d1_pdf*sigma) / (2*sqrt_tau) - r*KD*d2_cdf + q*SQ*d1_cdf,
        "vega": SQ*d1_pdf*sqrt_tau,
        "rho":  K*tau*D*d2_cdf,
    }

    if option.is_put():
        # P = C - SQ + KD
        value = value - SQ + KD
        greeks["delta"] = greeks["delta"] - Q
        greeks["theta"] = greeks["theta"] + r*KD - q*SQ
        greeks["rho"] = greeks["rho"] - tau*KD

    return value, greeks
