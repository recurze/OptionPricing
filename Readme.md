## Options

The right (or option)
1. to buy (call) or sell (put)
2. a specific quantity
3. of underlying asset or instrument
4. at a specified strike price
5. on (European style) or before a specified date

Call option is (normally) exercised only  when the strike price is below the
market price. Otherwise, you could simply buy it at the market price.
Similarly, put option is (normally) exercised only when the strike price is
above the market price.

The option could be sold at a fixed strike price or the spot price. If the
former, it could be at a discount (does this happen?) or at a premium (price).
Premium is income to the issuer regardless of whether the option is exercised.


## Valuation of Options

The price can be split into 2 components:

1. intrinsic value (for the holder):
    a. put : max(strike - spot, 0)
    b. call: max(spot - strike, 0)
2. extrinsic value (time value): premium - intrinsic

The extrinsic value is the risk that the writer takes on.

Factors affecting premium:
1. price of underlying
    a. payment of dividends decreases price too
2. how far strike price is from spot price
3. volatility of underlying

## Black-Scholes (BS) model

### Concepts:
* [Weiner process](https://en.wikipedia.org/wiki/Wiener_process#Characterisations_of_the_Wiener_process)
* [Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion#Technical_definition:_the_SDE) (GBM): log follows [Weiner process with drift](https://en.wikipedia.org/wiki/Wiener_process#Related_processes) $\mu$
* [Ito's drift-diffusion processes](https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma#It%C3%B4_drift-diffusion_processes_(due_to:_Kunita%E2%80%93Watanabe))

### Assumptions

* Exists at least one risky and one riskless asset
* Stock prices follow geometric Brownian motion with constant drift and volatility
* European options
* No dividends
* No arbitrage opportunities
* No fees
* Can buy/sell any amount, even fractional

### Notation

* t: time in years
* r: annualized risk-free rate

* S(t): price of underlying asset S at time t
* $\mu$: drift rate of S
* $\sigma$: s.d. of stock's return

* V(t, S): price of option
* T: Expiration time
* $\tau$: time till expiration
* K: Strike price

### Derivation (in words) of BS equation

1. Stock prices follow GBM.
2. Use Ito's lemma to get $dV$.
3. Consider a delta neutral portfolio with one short option and $\partial V \over \partial S$ long shares. Compute the value of the holdings $\Pi$ as sum of these 2 assets and the total profit/loss as changes in the value.
4. Plug in 1 and 2 into 3 and watch the $W_t$ corresponding to the Weiner process vanish. There's no more uncertainty.
5. Our portfolio is riskless, so has risk-free rate of return.
6. Use 3, 4 and 5 to eliminate $\Pi$ and receive the BS equation.

### Solution

$C(S_t, t) = \phi(d_+)S_t - \phi(d_-)Ke^(-r\tau)$
$sd = \sigma \tau^0.5$
$d_+/- = (log(S_t/K) + r\tau)/sd +/- 0.5sd$

$P(S_t, t) = Ke^(-r\tau) - S_t + C(S_t, t)$


Alternatively, $D = e^(-r\tau)$ and $S = DF$ and $C - P = D(F - K)$.

### Interpretation

See [pdf](https://www.ltnielsen.com/wp-content/uploads/Understanding.pdf).

## Greeks

* Delta: partial derivative of V w.r.t S
* Gamma: second partial derivative of V w.r.t S
* Theta: partial derivative of V w.r.t t
* Vega (nu) : partial derivative of V w.r.t $\sigma$
* Rho : partial derivative of V w.r.t r

Common for financial institutions to set risk limits on for each of the greeks. Delta and Gamma are the main ones for trading.
