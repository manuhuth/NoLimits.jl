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

For `VEM` (stochastic variational EM) the objective is `-(K/|B|) Σ_{i∈B} ELBO_i(θ, κ_i)`
over the `K` batches with random effects. Its gradient is the unbiased estimate of the
gradient of the full negative ELBO with respect to the fixed effects and all variational
parameters: the variational parameters `κ_i` of an unselected batch do not enter it and get
a zero gradient. The full objective is separable in the `κ_i`, so this stochastic gradient
targets the same optimum as the full-batch fit. With `refine = :auto` the
importance-sampling refinement (a full-data pass per iteration) is skipped under
mini-batching; `refine = true` still runs it. With an Optimisers.jl rule the curvature preconditioning of the fixed effects is off, so
the rule's steps act on the transformed fixed-effect scale.

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

MINIBATCH_BENCHMARK_PLACEHOLDER

## API

```@docs
MiniBatch
```
