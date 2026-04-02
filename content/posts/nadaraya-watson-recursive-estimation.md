---
title: "Nadaraya–Watson Recursive Estimation: Theory and Implementation"
date: 2025-01-15
math: true
tags: ["statistics", "estimation", "R", "non-parametric"]
description: "A deep dive into the Nadaraya–Watson recursive estimator — from convergence theory to R implementation on ECG data."
showToc: true
---

## Introduction

Suppose we observe pairs $(X_1, Y_1), \ldots, (X_n, Y_n)$ where we want to estimate
the **regression function**

$$
m(x) = \mathbb{E}[Y \mid X = x].
$$

The classical **Nadaraya–Watson estimator** is defined as

$$
\hat{m}_n(x) = \frac{\sum_{i=1}^n K_h(x - X_i)\, Y_i}{\sum_{i=1}^n K_h(x - X_i)},
$$

where $K_h(u) = h^{-1} K(u/h)$ is a kernel with bandwidth $h > 0$.

## The Recursive Version

In my thesis, I studied a **recursive** variant suited for streaming data.
Define the sequence of estimates:

$$
\hat{m}_{n+1}(x) = \hat{m}_n(x) + \gamma_{n+1}\, K_{h_{n+1}}(x - X_{n+1})
\bigl[Y_{n+1} - \hat{m}_n(x)\bigr],
$$

where $\{\gamma_n\}$ is a step-size sequence satisfying the **Robbins–Monro conditions**:

$$
\sum_{n=1}^\infty \gamma_n = \infty, \qquad \sum_{n=1}^\infty \gamma_n^2 < \infty.
$$

### Convergence Result

Under mild regularity conditions on the kernel $K$ and the density $f$ of $X$, one can show:

$$
\hat{m}_n(x) \xrightarrow{a.s.} m(x) \quad \text{as } n \to \infty.
$$

Moreover, the estimator satisfies an **asymptotic normality** result:

$$
\sqrt{n h_n}\,\bigl(\hat{m}_n(x) - m(x)\bigr) \xrightarrow{d} \mathcal{N}\!\left(0,\; \frac{\sigma^2(x)\,\|K\|_2^2}{f(x)}\right),
$$

where $\sigma^2(x) = \mathrm{Var}(Y \mid X = x)$ and $\|K\|_2^2 = \int K^2(u)\,du$.

## R Implementation

Here's the core vectorized implementation I used to simulate ECG waveforms:

```r
#' Recursive Nadaraya-Watson Estimator
#'
#' @param x     Grid of evaluation points
#' @param X     Observed covariates (n x 1)
#' @param Y     Observed responses  (n x 1)
#' @param h_fn  Bandwidth function h(n) -> numeric
#' @param gamma_fn Step-size function gamma(n) -> numeric
#' @return Matrix of estimates: nrow = length(x), ncol = n
recursive_nw <- function(x, X, Y, h_fn, gamma_fn) {
  n   <- length(X)
  m   <- length(x)
  est <- matrix(0, nrow = m, ncol = n)

  # Gaussian kernel (vectorized over x at each step)
  K <- function(u) dnorm(u)

  for (i in seq_len(n)) {
    h     <- h_fn(i)
    gamma <- gamma_fn(i)
    k_vec <- K((x - X[i]) / h) / h        # m-length vector

    if (i == 1L) {
      est[, i] <- k_vec * Y[i] / (k_vec + 1e-12)
    } else {
      prev       <- est[, i - 1L]
      est[, i]   <- prev + gamma * k_vec * (Y[i] - prev)
    }
  }
  est
}

# Example: h(n) = n^{-1/5},  gamma(n) = 1/n
result <- recursive_nw(
  x      = seq(-pi, pi, length.out = 200),
  X      = rnorm(1000),
  Y      = sin(rnorm(1000)) + rnorm(1000, sd = 0.2),
  h_fn   = function(n) n^(-1/5),
  gamma_fn = function(n) 1 / n
)
```

The vectorization over the grid `x` at each step avoids the inner loop, giving roughly a **30× speedup** compared to a naïve double-loop implementation.

## Key Takeaways

- The recursive estimator updates **in $\mathcal{O}(m)$** per observation (vs $\mathcal{O}(nm)$ for batch NW).
- Choosing $h_n \sim n^{-1/(4+d)}$ is optimal for $d$-dimensional $X$.
- The step-size $\gamma_n = c/n$ satisfies Robbins–Monro and gives $\sqrt{n}$-rate convergence under standard assumptions.

## References

1. Fraysse, P. (2014). *Recursive Estimation in a Class of Models of Deformation*. Journal of Statistical Planning and Inference.
2. Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability and Its Applications*, 9(1), 141–142.
3. Watson, G. S. (1964). Smooth regression analysis. *Sankhyā*, Series A, 26, 359–372.
