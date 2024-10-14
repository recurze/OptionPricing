# Source: QF5204, 2024 sem II, Prof. Li Hao (with some mods).
# Implementation of formula presented in class.
import bisect
import math

from options import PriceType
from smile import Smile
from utils import is_close


class ImpliedVol:
    FINITE_DIFFERENCE_K = 0.001
    FINITE_DIFFERENCE_T = 0.005

    def __init__(self, pillars: list[float], smiles: list[Smile]):
        self.pillars = pillars
        self.smiles = smiles

    def get_smile_index(self, T: float) -> int:
        if T < self.pillars[0]:
            return 0

        if T > self.pillars[-1]:
            return -1

        return bisect.bisect_left(self.pillars, T)

    def Vol(self, T: float, K: PriceType) -> float:
        i_smile = self.get_smile_index(T)
        if i_smile == 0 or i_smile == -1:
            return self.smiles[i_smile].Vol(K)

        nextVol, nextT = self.smiles[i_smile].Vol(K), self.pillars[i_smile]
        nextVar = nextVol * nextVol * nextT

        prevVol, prevT = self.smiles[i_smile - 1].Vol(K), self.pillars[i_smile - 1]
        prevVar = prevVol * prevVol * prevT

        w = (nextT - T) / (nextT - prevT)
        return math.sqrt((w * prevVar + (1 - w) * nextVar) / T)

    def dVoldK(self, T: float, K: PriceType) -> float:
        dk = self.FINITE_DIFFERENCE_K
        return (self.Vol(T, K + dk) - self.Vol(T, K - dk)) / (2 * dk)

    def dVoldT(self, T: float, K: PriceType) -> float:
        dt = self.FINITE_DIFFERENCE_T
        return (self.Vol(T + dt, K) - self.Vol(T, K)) / dt

    def dVol2dK2(self, T: float, K: PriceType) -> float:
        dk = self.FINITE_DIFFERENCE_K
        return (self.Vol(T, K + dk) + self.Vol(T, K - dk) - 2*self.Vol(T, K)) / (dk*dk)


class LocalVol:
    MIN_VOL = 1e-8
    MAX_VOL = 1

    def __init__(self, iv: ImpliedVol, S0: PriceType, rd: float, rf: float):
        self.iv = iv
        self.S0 = S0
        self.r = rd - rf

    def Vol(self, T: float, S: PriceType) -> float:
        imp = self.iv.Vol(T, S)
        if is_close(T, 0):
            return imp

        sqrt_T = math.sqrt(T)
        sd = imp * sqrt_T
        d1 = (math.log(self.S0 / S) + self.r*T)/sd + sd/2

        dvdk, dvdt, d2vdk2 = self.iv.dVoldK(T, S), self.iv.dVoldT(T, S), self.iv.dVol2dK2(T, S)
        den = (1 + S*d1*sqrt_T*dvdk)**2 + S*S*T*imp*(d2vdk2 - d1*sqrt_T*dvdk*dvdk)
        if den <= 0:
            return math.sqrt(self.MAX_VOL)

        num = imp * (imp + 2*T*(dvdt + self.r*S*dvdk))

        lv = max(self.MIN_VOL, min(num/den, self.MAX_VOL))
        return math.sqrt(lv)
