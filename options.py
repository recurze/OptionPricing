from datetime import date
from enum import Enum
from typing import Optional


QtyType = int
PriceType = float  # please don't kill me
TickerType = str


class OptionType(Enum):
    CALL = 1
    PUT = 2


class Stock:
    def __init__(self, ticker: TickerType, spot_price: PriceType):
        self.ticker = ticker
        self.spot_price = spot_price


class Option:
    def __init__(self,
                 option_type: OptionType,
                 underlying: Stock,
                 dividend_yield: Optional[PriceType],
                 qty: QtyType,
                 strike_price: PriceType,
                 expiration_date: date):
        self.option_type = option_type

        self.underlying = underlying
        self.dividend_yield = dividend_yield or 0

        self.qty = qty
        self.strike_price = strike_price
        self.expiration_date = expiration_date

    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    def is_put(self) -> bool:
        return not self.is_call()

    def simple_payoff(self, spot_price: PriceType) -> PriceType:
        if self.is_call():
            return max(spot_price - self.strike_price, 0)
        return max(self.strike_price - spot_price, 0)

    def payoff(self, spot_price: PriceType, time_to_maturity_in_years: float = 0) -> PriceType:
        raise NotImplementedError()


class EuropeanOption(Option):
    def payoff(self, spot_price: PriceType, time_to_maturity_in_years: float = 0) -> PriceType:
        return 0 if time_to_maturity_in_years != 0 else super().simple_payoff(spot_price)


class AmericanOption(Option):
    def payoff(self, spot_price: PriceType, time_to_maturity_in_years: float = 0) -> PriceType:
        return super().simple_payoff(spot_price)
