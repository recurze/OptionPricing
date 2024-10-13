from abc import abstractmethod
from enum import Enum
from utils import is_close


PriceType = float


class OptionType(Enum):
    CALL = "CALL"
    PUT = "PUT"


class BaseOption:
    def __init__(self, K: PriceType, option_type: OptionType):
        self.K = K
        self.option_type = option_type

    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    def is_put(self) -> bool:
        return self.option_type == OptionType.PUT

    def simple_payoff(self, S: PriceType) -> PriceType:
        if self.is_call():
            return max(S - self.K, 0)
        if self.is_put():
            return max(self.K - S, 0)
        raise NotImplementedError()


class PathIndependentOption(BaseOption):
    @abstractmethod
    def payoff(self, S: PriceType, T: float = 0) -> PriceType:
        pass

    @abstractmethod
    def quanto_payoff(self, Sa: PriceType, Sb: PriceType, T: float = 0) -> PriceType:
        pass


class PathDependentOption(BaseOption):
    @abstractmethod
    def payoff(self, S: list[PriceType], T: float = 0) -> PriceType:
        pass

    @abstractmethod
    def quanto_payoff(self, Sa: list[PriceType], Sb: PriceType, T: float = 0) -> PriceType:
        pass


Option = PathDependentOption | PathIndependentOption


class EuropeanOption(PathIndependentOption):
    def payoff(self, S: PriceType, T: float = 0) -> PriceType:
        return 0 if not is_close(T, 0) else super().simple_payoff(S)

    def quanto_payoff(self, Sa: PriceType, Sb: PriceType, T: float = 0) -> PriceType:
        return Sb * self.payoff(Sa, T)


class AmericanOption(PathIndependentOption):
    def payoff(self, S: PriceType, T: float = 0) -> PriceType:
        return super().simple_payoff(S)


class EuropeanBinaryOption(PathIndependentOption):
    def __init__(self, fixed_payoff: PriceType = 1, **kwargs):
        self.fixed_payoff = fixed_payoff
        super().__init__(**kwargs)

    def payoff(self, S: PriceType, T: float = 0) -> PriceType:
        return 0 if not is_close(T, 0) or super().simple_payoff(S) == 0 else self.fixed_payoff


class AsianOption(PathDependentOption):
    def payoff(self, S: list[PriceType], T: float = 0) -> PriceType:
        return super().simple_payoff(sum(S) / len(S))

    def quanto_payoff(self, Sa: list[PriceType], Sb: PriceType, T: float = 0) -> PriceType:
        return Sb * self.payoff(Sa, T)
