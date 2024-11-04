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

$C(S_t, t) = \phi(d_+)S_t - \phi(d_-)Ke^{-r\tau}$

$sd = \sigma \sqrt{\tau}$

$d_{\pm} = \frac{log(\frac{S_t}{K}) + r\tau}{sd} \pm \frac{sd}{2}$

$P(S_t, t) = Ke^{-r\tau} - S_t + C(S_t, t)$


Alternatively, $D = e^{-r\tau}$ and $S = DF$ and $C - P = D(F - K)$.

### Interpretation

See [pdf](https://www.ltnielsen.com/wp-content/uploads/Understanding.pdf).

## Greeks

* Delta: partial derivative of V w.r.t S
* Gamma: second partial derivative of V w.r.t S
* Theta: partial derivative of V w.r.t t
* Vega (nu) : partial derivative of V w.r.t $\sigma$
* Rho : partial derivative of V w.r.t r

Common for financial institutions to set risk limits on for each of the greeks. Delta and Gamma are the main ones for trading.


## Binomial model

Consider a one-step binomial tree where with probability p, the underlying stock price goes up to S_u and with probability 1 - p, it goes down to S_d. We might be tempted to price this by the getting the expected payoff: p(S_u - S_0), but that would be incorrect.

Instead we price by replication. Consider a delta neutral portfolio with one short option and $\delta$ long shares. The value of the holding would be $V_0 = V(S_0, 0) - \delta S_0$. Now, the value of my portfolio at T would be $\delta S_u + \frac{V_0}{D} = V_u$ where D is the discount factor ($V_0$ is the forward price), or $\delta S_d + \frac{V_0}{D} = V_d$. Solving this, we have $V_0$ and $\delta$ and we can find $V(S_0, 0)$ using these.

Binomial model can be viewed as discrete version of Black-Scholes model. Consider n-step binomial tree. Now match the first two moments of $S_t$ with that of the Black-Scholes model.

Binomial model can easily handle American option.


## Reinforcement Learning (RL) - Least-Square Policy Iteration (LSPI)

See papers [Least-Square Policy Iteration](https://users.cs.duke.edu/~parr/jmlr03.pdf) and [Learning Exercise Policies for American Options](https://proceedings.mlr.press/v5/li09d/li09d.pdf).

American option can be formulated as a finite-horizon optimal stopping problem (best time to exercise). LSPI solves this through [policy iteration](https://en.wikipedia.org/wiki/Markov_decision_process#Policy_iteration). It uses Least-Square [Temporal Differencing](https://en.wikipedia.org/wiki/Temporal_difference_learning) [Q-learning](https://en.wikipedia.org/wiki/Q-learning) to improve the value which in turn is used to improve the policy, and this goes on until there's no change in policy. As for "least-square" part of LSPI, instead of using Q directly, we approximate Q as dot(w, phi(s, a)) where phi is the feature map. Writing the bellman equation in matrix form and plugging in approximate Q gives us a nice system of linear equations in the form Ax = b that we solve.

Once we have the policy, we can go ahead and calculate the expected payoff by simulating paths the spot price might take. For this we use GBM. Alternatives are [GARCH](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity) and Heston model. The expected payoff is nothing but the fair value of the option.

To summarize the algorithm:
1. Simulate price paths
2. Solve optimal stopping problem using LSPI
3. Calculate expected payoff using the optimal policy across numerous simulations.

```
LSPI:
    pi <- initial policy
    do:
        pi' <- pi
        pi <- LSTDQ(..., pi')
    until pi' = pi
```

## Monte Carlo methods

[MC](https://en.wikipedia.org/wiki/Monte_Carlo_method) is great for pricing path dependent or exotic options. I did not implement any [variance reduction](https://en.wikipedia.org/wiki/Variance_reduction) methods like control variates, mode matching or cross entropy.

## PDE methods

Black-scholes is the analytical solution of the PDE. Why not solve it numerically instead?
We may discretize S, but discretizing $log(S)$ is better as we want more points near the center.
`pricer/pde.py` is from QF5204 course I took. It uses local volatility.
I wrote my own pde-based pricer in `notebooks/4.ipynb` to price double no touch option using crank-nicolson (CN) scheme. Other options are Douglas schema (instead of 0.5 like in CN, we use `w`) and Euler implicit/explict.

## Future/Todo

* Add [regression tests](https://en.wikipedia.org/wiki/Regression_testing)
* Improve LSPI variance: maybe average out multiple runs of LSPI?
* Use real world data to compare payoffs: [Kaggle $SPY](https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022), [Kaggle $AAPL](https://www.kaggle.com/datasets/kylegraupe/aapl-options-data-2016-2020)
* Implement [Longstaff-Schwartz least-squares method](https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf).
* Implement RNNs like LSTM or GRU.
