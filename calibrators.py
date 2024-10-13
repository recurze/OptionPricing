import math


def crr_calibrator(r: float, vol: float, t: float) -> tuple[float, float, float]:
    b = math.exp((r + vol*vol) * t) + math.exp(-r*t)
    u = (b + math.sqrt(b*b - 4)) / 2
    p = (u*math.exp(r*t) - 1) / (u*u - 1) if u != 1 else 1/2
    return (u, 1/u, p)


def jrrn_calibrator(r: float, vol: float, t: float) -> tuple[float, float, float]:
    sd = vol * math.sqrt(t)
    u = math.exp((r - vol*vol/2)*t + sd)
    d = math.exp((r - vol*vol/2)*t - sd)
    p = (math.exp(r*t) - d) / (u - d)
    return (u, d, p)


def jreq_calibrator(r: float, vol: float, t: float) -> tuple[float, float, float]:
    sd = vol * math.sqrt(t)
    u = math.exp((r - vol*vol/2)*t + sd)
    d = math.exp((r - vol*vol/2)*t - sd)
    return (u, d, 1/2)


def tian_calibrator(r: float, vol: float, t: float) -> tuple[float, float, float]:
    v = math.exp(vol*vol*t)
    u = 0.5 * math.exp(r*t) * v * (v + 1 + math.sqrt(v*v + 2*v - 3))
    d = 0.5 * math.exp(r*t) * v * (v + 1 - math.sqrt(v*v + 2*v - 3))
    p = (math.exp(r*t) - d) / (u - d)
    return (u, d, p)


def lam_calibrator(r: float, vol: float, t: float, lam: float) -> tuple[float, float, float, float]:
    sd = vol * math.sqrt(t)
    u = math.exp(lam * sd)

    dd = (sd/(4 * lam)) * (2*r/(vol*vol) - 1)
    pm = 1 - 1/(lam*lam)
    pu = (1 - pm)/2 + dd
    pd = (1 - pm)/2 - dd

    return u, pu, pm, pd
