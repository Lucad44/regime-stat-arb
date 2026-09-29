# Methodology

Notation: `y`, `x` are log prices of the two legs, `s` is the spread, `t` indexes trading days.

## 1. Pair screening

1. Standardise daily log returns, remove the leading `k` principal components (default 3), and rank
   pairs by the absolute correlation of what is left. Market and sector factors dominate raw
   correlation and say nothing about a pair-specific long-run link.
2. Run Engle-Granger on the top `K` candidates in both directions, keep the smaller p-value and
   multiply it by 2 to pay for choosing the direction.
3. Apply Benjamini-Hochberg over the `K` tests and keep `q <= 0.20`. Keep pairs whose OLS half-life
   lies in `[2, 60]` days, and cap how many pairs any single asset may appear in.

## 2. OU state-space model

$$y_t = \beta_t (x_t - \bar x) + s_t + v_t,\qquad v_t \sim N(0, r)$$
$$\beta_t = \beta_{t-1} + w^{(1)}_t,\qquad w^{(1)}_t \sim N(0, q_b)$$
$$s_t = \mu(1-\phi) + \phi\, s_{t-1} + w^{(2)}_t,\qquad w^{(2)}_t \sim N(0, q_s)$$

State `(beta_t, s_t)`, observation row `(x_t - x_bar, 1)`. The parameters `(q_b, q_s, phi, mu, r)` are
estimated by maximising the Gaussian innovation likelihood on the formation window (L-BFGS-B,
two starting points), then frozen.

The standard alternative puts `[beta, alpha]` in the state and reads the one-step innovation as the
spread. That cannot distinguish a drifting hedge ratio from a mean-reverting deviation, because both
end up in the same residual. Here they are separated by their dynamics: `beta` is a random walk with
small variance, `s` is stationary with a half-life of days.

The spread series used downstream is the filtered `s_{t|t}`. Standardising uses the stationary mean
and standard deviation of the *calm* regime (section 3).

## 3. Markov-switching AR(1)

$$s_t = c_k + \phi_k s_{t-1} + \sigma_k \varepsilon_t,\qquad k = S_t,\quad P(S_t=j \mid S_{t-1}=i)=P_{ij}$$

Fitted by EM. The E-step uses the scaled forward-backward recursions; the M-step is closed-form
weighted least squares per state and re-estimates `P` from the expected transition counts. Four
starts (one heuristic, three random); the best log-likelihood is kept. States are sorted by the
stationary standard deviation `sigma_k / sqrt(1 - phi_k^2)` so state 0 is the calm one.

**Using it.** The online filtered probability `P(S_t = 0 | s_{1:t})` is the only regime quantity that
enters a signal. The two-sided smoother uses future data and is for diagnostics.

**When to trust the split.** BIC of the two-state model is compared with a single AR(1). If the gain is
below 2, or the stressed state is not at least 1.25x wider than the calm one, the split is not
learned and a fixed prior (`phi_stress = 0.99`, stress volatility 1.5x calm, persistence 0.99 / 0.93)
is used instead.

## 4. Optimal entry level

For `dX = -theta X dt + sigma dW`, a round trip that buys at `a`, sells at `m > a` and pays cost `c`
takes on average

$$E[T_{a\to m\to a}] = \frac{\pi}{\theta}\left[\operatorname{erfi}\!\left(\frac{m\sqrt\theta}{\sigma}\right) - \operatorname{erfi}\!\left(\frac{a\sqrt\theta}{\sigma}\right)\right]$$

and earns `m - a - c`, so the profit rate is `(m - a - c) / E[T]` (Bertram 2010). Measuring levels in
stationary standard deviations `sigma_eq = sigma / sqrt(2 theta)` gives `sqrt(theta)/sigma = 1/(sqrt(2) sigma_eq)`,
so the argument of `erfi` no longer involves `theta` and the optimal `a` depends only on
`c / sigma_eq`. Entry is `k* = argmax_k rate(-k, 0)`, clipped to `[1, 3]`. On the simulated data the unclipped
`k*` has a median of 0.70 and is below 1 in 94% of pair-windows, so the floor is the binding
constraint most of the time; using `k*` unclipped (floor 0.5) lowered the net Sharpe.

The cost `c` is the round trip in spread units, `2 * cost_bps / 1e4 * (1 + |beta|)`.

## 5. Trading rule

Long the spread (long `y`, short `beta x`) when `z <= -k*`, short when `z >= k*`. A position is closed
at `|z| <= z_exit`, on a stop at `|z| >= z_stop`, when `P(calm) < gate_out`, or after
`max_hold_hl` half-lives. New positions require `P(calm) >= gate_in`, and size is `min(1, P(calm))`
when probability sizing is on. Decisions made at the close of `t` become positions at
`t + 1 + execution_lag`.

## 6. Portfolio and costs

Each pair occupies one of `max_pairs` equal slots. Per pair, weights are `w_y = pos / (1 + |beta|)`
and `w_x = -pos * beta / (1 + |beta|)`, rebalanced daily. Costs are `cost_bps` on absolute weight
change plus `borrow_bps / 252` per day on short notional. All positions are closed at the end of
each window. Optional volatility targeting scales the book by target / EWMA volatility, lagged one day.

## 7. Validation

**PSR** (Bailey & Lopez de Prado 2014). For per-period Sharpe `SR` from `n` returns with skewness
`g3` and raw kurtosis `g4`:

$$\text{PSR}(SR^*) = \Phi\!\left(\frac{(SR - SR^*)\sqrt{n-1}}{\sqrt{1 - g_3 SR + \frac{g_4-1}{4}SR^2}}\right)$$

**DSR.** PSR evaluated at `SR* = E[max SR]` over `N` trials with cross-trial Sharpe variance `V`:

$$SR_0 = \sqrt{V}\left[(1-\gamma)\Phi^{-1}\!\left(1-\tfrac1N\right) + \gamma\,\Phi^{-1}\!\left(1-\tfrac1{Ne}\right)\right]$$

with `gamma` the Euler-Mascheroni constant. All 180 configurations count as trials.

**CSCV / PBO.** Cut the time axis into 16 equal blocks. For each of the C(16, 8) = 12,870 choices of 8
in-sample blocks, pick the configuration with the best in-sample Sharpe and record its relative rank
`w` among all configurations out of sample. The logit is `ln(w / (1 - w))`, and
`PBO = P(logit <= 0)`. Block sums and sums of squares are precomputed, so each split is a small
matrix product.
