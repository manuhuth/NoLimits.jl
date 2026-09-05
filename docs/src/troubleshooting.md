# Troubleshooting

What to do when a fit fails, stalls, or returns something implausible. Each entry names the
symptom first, so you can scan for yours.

## `get_converged(res)` is `false`

`get_converged` reports what the optimizer said about its own stopping criterion. It is not a
statement about the quality of the fit: a fit can stop at `maxiters` with perfectly usable
estimates, and a fit can report success at a poor local optimum. Treat it as one signal among
several, never as a gate.

Check, in this order:

1. `get_objective(res)` - is it finite and better than at the starting values?
2. `NoLimits.summarize(res)` - do the estimates sit on a bound, or at an implausible order of magnitude?
3. `plot_fits(res)` - do the predictions track the data? A picture settles most of these questions faster than any number.

```julia
get_objective(res)              # finite, and better than at the starting values?
NoLimits.summarize(res)         # estimates on a bound, or implausible in magnitude?
plot_fits(res)                  # do the predictions track the data?
```

If the optimizer simply ran out of budget, raise it:

```julia
res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 5000)))
```

## The fit stops at a poor optimum

Nonlinear mixed-effects objectives are routinely multimodal. Two remedies, cheapest first:

- **Warm-start from a pooled fit** - a fast naive-pooled fit hands its estimates to the real
  estimator. See [Pooled / PooledMap](estimation/pooled.md).

  ```julia
  res = fit_model(dm, NoLimits.Laplace(); pooled_init = true)
  ```

- **Search from several starting points** - see [Multistart](estimation/multistart.md).

  ```julia
  ms = NoLimits.Multistart(; dists = (; A0 = LogNormal(log(10.0), 0.5)), n_draws_requested = 12, n_draws_used = 6)
  res = fit_model(ms, dm, NoLimits.Laplace())
  ```

Poor scaling is the other common cause. Parameters that differ by many orders of magnitude make
the objective badly conditioned; declare them with `scale = :log` in `@fixedEffects` so the
optimizer works on a comparable scale.

## The objective is `NaN` or `-Inf`

Usually the model was evaluated somewhere it is not defined: a negative variance, a `log` of a
non-positive prediction, or an ODE solve that failed. Fixes:

- Constrain the parameter with `scale = :log` (or model bounds) instead of letting the optimizer
  step into the invalid region.
- Guard the formula itself, for example with a softplus on a quantity that must stay positive.
- The optimization-based methods take `nan_recovery = :backtrack` (the default), which retries a
  shorter step instead of aborting:

  ```julia
  res = fit_model(dm, NoLimits.Laplace(; nan_recovery = :backtrack))
  ```

## Laplace or FOCEI reports a non-positive-definite Hessian

The inner empirical-Bayes problem needs a positive-definite curvature matrix. When it is not,
the method adds a small jitter to the diagonal and retries. The relevant `Laplace` / `FOCEI`
options are `jitter`, `max_tries`, `jitter_growth`, `adaptive_jitter`, and `jitter_scale`
(see [Laplace](estimation/laplace.md) for the shipped defaults):

```julia
res = fit_model(dm, NoLimits.Laplace(; jitter = 1.0e-5, max_tries = 10, jitter_growth = 20.0))
```

Persistent non-positive-definiteness is usually a modeling signal rather than a numerical one:
the model is close to unidentified in the random effects at those parameter values. Reducing the
number of random effects, or fixing one of them with `constants_re`, is often the real fix.

## The ODE solver fails or the fit is very slow

- **Stiffness.** The default solver is a good general choice, but a stiff system needs a stiff
  solver, as in the [Neural Differential Equations](tutorials/mixed-effects-nn-saem.md) tutorial:

  ```julia
  model = set_solver_config(model; alg = AutoTsit5(Rosenbrock23()), abstol = 1.0e-6, reltol = 1.0e-6)
  ```
- **Tolerances.** Tight `abstol`/`reltol` on an ODE model dominates the runtime. Loosen them
  through `set_solver_config` while exploring, then tighten for the final fit.
- **Parallelism.** Evaluate individuals in parallel, and start Julia with `-t auto`:

  ```julia
  res = fit_model(dm, NoLimits.Laplace(); serialization = SciMLBase.EnsembleThreads())
  ```
- **Method cost.** `MCEM` is the most expensive of the mixed-effects methods; `SAEM` reaches a
  comparable answer for far less work on most models. See
  [Choosing a method](estimation/index.md#Choosing-a-Method).

## An error names a package I have not installed

Plotting, Turing-based estimation, JLD2 serialization, profile-likelihood UQ, and several other
features live in package extensions. Calling one without its package raises an error that names
exactly what to install. Add it once with `Pkg.add`, then `using` it. The full list is in
[Optional Dependencies](installation.md#Optional-Dependencies).

## Between-subject variance collapses towards zero

An `omega` estimate at the lower edge of its range means the data do not support the random
effect at that level: too few individuals, too few observations per individual, or a random
effect that is not identifiable given the structural model. Refit without that random effect and
compare objectives before concluding that the variance is genuinely small.

## The same fit gives slightly different numbers

Per-individual quantities are bitwise reproducible; the population objective is reproducible only
up to floating-point summation order, and the fitted parameters only up to the conditioning of the
problem. [Reproducibility and Individual Order](estimation/reproducibility.md) gives the measured
magnitudes and the tolerances to use when comparing fits.

## Still stuck

Open an issue at [github.com/manuhuth/NoLimits.jl](https://github.com/manuhuth/NoLimits.jl/issues)
with the model definition, a minimal data frame that reproduces the problem, and the output of:

```julia
NoLimits.summarize(dm)
```
