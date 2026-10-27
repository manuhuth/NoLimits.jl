# Mini-batching

For data sets with many individuals, each iteration of an estimator can use a random subset
of the individuals instead of all of them. In NoLimits this is a wrapper around the
estimation method:

```julia
using NoLimits
using Random

res = fit_model(dm, MiniBatch(Laplace(); schedule = 8); rng = Xoshiro(1))
```

`MiniBatch` works with `MLE`, `MAP`, `Laplace`, `FOCEI`, `GHQuadrature`, `VEM`, `SAEM` and
`MCEM`. The result is the `FitResult` of the wrapped method, so every accessor, plot and
uncertainty method applies unchanged.

## Batches and Schedules

The unit of selection is a random-effect batch: the individuals that share a random-effect
level (for a single random effect on the id column, one batch per individual). For `MLE`
and `MAP` every individual is one batch. The `schedule` keyword selects the batches of each
iteration:

- an `Int` `m`: a random subset of `min(m, nbatches)` batches, drawn without replacement
  in every iteration;
- a callable `(nbatches::Int, iter::Int, rng) -> indices` returning any iterable of `Int`
  (duplicates are ignored), for example a deterministic cycle:

  ```julia
  cycle(nbatches, iter, rng) = [mod1(iter, nbatches)]
  method = MiniBatch(Laplace(); schedule = cycle)
  ```

  A stateful callable is shared by every fit that uses the same wrapper object (including
  all `Multistart` starts), so pass a fresh instance per fit;
- `:all`: every batch in every iteration. The wrapped method then gives exactly the
  unwrapped fit.

All subsets are drawn from the `rng` passed to `fit_model` (`Multistart` gives each start
its own forked `rng`), so seeded fits are reproducible, and threaded and serial runs agree.

## Gradient-Based Methods and VEM

For `MLE`, `MAP`, `Laplace`, `FOCEI`, `GHQuadrature` and `VEM` one subset `B` is drawn per
optimizer iteration, and the objective and gradient of that iteration both use it. The
contribution of `B` is multiplied by `nbatches / |B|`, so the objective and its gradient
are unbiased estimates of the full-data objective and gradient. Priors, `penalty` and
`extra_objective` are global terms and are added once, unscaled. After the fit, the
reported objective is evaluated on all batches at the fitted parameters.

The objective is stochastic, so line-search optimizers are inappropriate. Unless the wrapped
method was given its own optimizer, `MiniBatch` replaces the default LBFGS by
`Optimisers.Adam(0.01)` with `optim_kwargs = (; maxiters = 1000)`; `save_best = false` is
added for Optimisers.jl rules (the best single-minibatch objective is not a meaningful
criterion). The optimizer and its keywords can also be set on the wrapper:

```julia
using Optimisers
method = MiniBatch(
    FOCEI(); schedule = 16, optimizer = Optimisers.Adam(0.005),
    optim_kwargs = (; maxiters = 3000)
)
```

A non-Optimisers.jl optimizer warns, and it must support Optimization.jl callbacks (the
callback advances the schedule). With Optimisers.jl rules, finite bounds (model bounds or
`lb`/`ub` of the wrapped method) are enforced by projecting each update onto the box.
`AcceptedWarmStarts` requires `schedule = :all`.

For `VEM` (stochastic variational EM) the objective is `-(K/|B|) Σ_{i∈B} w_i ELBO_i(θ, κ_i)`
over the `K` batches with random effects (`w_i` the likelihood weights, 1 without a weight
column), plus the unscaled `penalty` and `extra_objective`. Its gradient is the unbiased estimate of the
gradient of the full negative ELBO with respect to the fixed effects and all variational
parameters: the variational parameters `κ_i` of an unselected batch do not enter it and get
a zero gradient. The full objective is separable in the `κ_i`, so this stochastic gradient
targets the same optimum as the full-batch fit. With `refine = :auto` the
importance-sampling refinement (a full-data pass per iteration) is skipped under
mini-batching; `refine = true` still runs it. With an Optimisers.jl rule the curvature
preconditioning of the fixed effects is off, so the rule's steps act on the transformed
fixed-effect scale.

## SAEM and MCEM

For `SAEM` and `MCEM` the schedule selects the batches whose random effects are sampled in
the E-step of each iteration. The `em_variant` keyword chooses the M-step:

- `:incremental` (default; incremental EM, Neal and Hinton 1998): the batches that were not
  selected keep their draws from the iteration that last visited them, and the M-step uses
  all batches. This trades a small bias from draws made under an older `θ` for a cheaper
  E-step.
- `:scaled`: the M-step uses only the selected batches, with their likelihood weights
  multiplied by `nbatches / |B|`, an unbiased estimate of the full-data Q-function. The
  M-step cost then also scales with `|B|`. For `SAEM` it requires
  `mstep_sa_on_params = true` (the default) and no custom `suffstats`. The reported
  objective is the scaled Q of the last minibatch.

```julia
res = fit_model(dm, MiniBatch(SAEM(); schedule = 30); rng = Xoshiro(1))
res = fit_model(dm, MiniBatch(MCEM(); schedule = 30, em_variant = :scaled); rng = Xoshiro(1))
```

### Choice of the Default

The default `em_variant = :incremental` was chosen from a screening comparison (5 seeds per
configuration, default `SAEM()` and `MCEM(maxiters = 60)`, a quarter of the batches per
iteration, 4 threads) on four simulated models: a linear Gaussian model with a correlated
random intercept and slope (120 individuals x 8 observations), an exponential decay with
log-normal amplitude and rate (100 x 6), a Poisson model with a random intercept (150 x 5),
and a Michaelis-Menten ODE with a random maximal rate (60 x 6). The error is the mean over
parameters of `|θ - θ_full| / max(|θ_full|, 0.1)` on the transformed scale, where `θ_full`
is the mean of the five full-data fits of the same method; the seed standard deviation is
averaged over parameters.

| Model | Method | Variant | Batches | Rel. error | Seed sd | Iterations | Wall time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Gaussian | SAEM | full | 120 | 0.0151 | 0.0120 | 294 | 2.42 |
| Linear Gaussian | SAEM | incremental | 30/120 | 0.0194 | 0.0123 | 275 | 2.03 |
| Linear Gaussian | SAEM | scaled | 30/120 | 0.0474 | 0.0304 | 255 | 1.24 |
| Linear Gaussian | MCEM | full | 120 | 0.0071 | 0.0051 | 30 | 2.42 |
| Linear Gaussian | MCEM | incremental | 30/120 | 0.0085 | 0.0044 | 52 | 3.51 |
| Linear Gaussian | MCEM | scaled | 30/120 | 0.1535 | 0.0980 | 28 | 0.89 |
| Exponential decay | SAEM | full | 100 | 0.0033 | 0.0071 | 209 | 0.85 |
| Exponential decay | SAEM | incremental | 25/100 | 0.0041 | 0.0079 | 232 | 0.85 |
| Exponential decay | SAEM | scaled | 25/100 | 0.0734 | 0.0473 | 209 | 0.43 |
| Exponential decay | MCEM | full | 100 | 0.0016 | 0.0027 | 23 | 1.07 |
| Exponential decay | MCEM | incremental | 25/100 | 0.0073 | 0.0080 | 60 | 2.47 |
| Exponential decay | MCEM | scaled | 25/100 | 0.1021 | 0.0783 | 35 | 0.64 |
| Poisson | SAEM | full | 150 | 0.0343 | 0.0268 | 223 | 0.95 |
| Poisson | SAEM | incremental | 38/150 | 0.0294 | 0.0166 | 297 | 1.11 |
| Poisson | SAEM | scaled | 38/150 | 0.0507 | 0.0368 | 222 | 0.79 |
| Poisson | MCEM | full | 150 | 0.0053 | 0.0044 | 47 | 1.33 |
| Poisson | MCEM | incremental | 38/150 | 0.0176 | 0.0048 | 60 | 1.53 |
| Poisson | MCEM | scaled | 38/150 | 0.0642 | 0.0464 | 37 | 0.49 |
| Michaelis-Menten ODE | SAEM | full | 60 | 0.0388 | 0.0200 | 285 | 3.76 |
| Michaelis-Menten ODE | SAEM | incremental | 15/60 | 0.3998 | 0.1056 | 300 | 3.55 |
| Michaelis-Menten ODE | SAEM | scaled | 15/60 | 0.1505 | 0.0830 | 232 | 1.45 |
| Michaelis-Menten ODE | MCEM | full | 60 | 0.0165 | 0.0086 | 60 | 8.74 |
| Michaelis-Menten ODE | MCEM | incremental | 15/60 | 0.1536 | 0.0056 | 60 | 6.70 |
| Michaelis-Menten ODE | MCEM | scaled | 15/60 | 0.1500 | 0.0790 | 60 | 3.28 |

The incremental variant has the smaller error in 6 of 8 model/method pairs (MCEM on the
Michaelis-Menten model is a tie, 0.154 against 0.150) and the smaller seed standard
deviation in 7 of 8. The exception is SAEM on the Michaelis-Menten model, where the
incremental fit had not converged in the weakly identified `Km` after 300 iterations; with
`maxiters = 1200` it reached the full-data value. The scaled variant has 1.4 to 3.9 times
lower wall time, because its M-step also uses only the selected batches, but its final
iterate is noisier; MCEM, which has no stochastic-approximation averaging, ends with errors
of 0.06 to 0.15 and its convergence test fires early on the noise. With the iteration
budget multiplied by `nbatches / |B|` (the same number of E-step draws per individual as
the full fit), the incremental variant had the smaller error in all 8 pairs. The
incremental variant evaluates the M-step on all batches, so its time per iteration was 77
to 90 percent of the full fit for these models; it saves more when the E-step dominates
the cost of an iteration.

## API

```@docs
MiniBatch
```
