# VEM

Variational expectation maximization (VEM, Tarek and Afonso 2026, arXiv:2604.26160)
replaces the marginal likelihood of a mixed-effects model by the evidence lower bound

```math
\sum_i \mathrm{ELBO}_i(\theta, \kappa_i), \qquad
\mathrm{ELBO}_i = \mathbb{E}_{q(\eta;\kappa_i)}\big[\log p(y_i, \eta \mid \theta)\big] + H\big(q(\cdot;\kappa_i)\big),
```

with one Gaussian variational distribution ``q(\eta; \kappa_i)`` per random-effect batch.
The fixed effects ``\theta`` and all variational parameters ``\kappa`` are optimized
jointly. The expectation uses a fixed rule per fit: a sparse Gauss-Hermite rule for batches
with at most three random effects, otherwise a fixed set of standard-normal draws. The
objective is therefore deterministic and admits quasi-Newton steps and line searches.

VEM needs no inner mode finding or Hessian per objective evaluation, unlike
[Laplace](laplace.md) and [FOCEI](focei.md), so it scales to models with many fixed
effects (e.g. neural networks) and many random effects.

## Applicability

- Requires a model with random effects and at least one free fixed effect.
- `penalty`, `extra_objective` and likelihood weights of the `DataModel` are supported (see
  [Penalties and Weights](#Penalties-and-Weights));
  fixed-effect priors are ignored.
- Any continuous random-effect distribution is supported with the default `adtype`
  (the same set as [GHQuadrature](ghquadrature.md)); discrete random effects are rejected.
  A non-ForwardDiff `adtype` (reverse mode) requires Gaussian (`Normal`/`MvNormal`)
  random effects.

## Basic Usage

```julia
using NoLimits
using DataFrames
using Distributions
using Random

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.0)
        b = RealNumber(0.0)
        σ = RealNumber(0.5; scale = :log)
        Ω = RealPSDMatrix([0.3 0.0; 0.0 0.3]; scale = :cholesky)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(MvNormal([0.0, 0.0], Ω); column = :ID)
    end
    @formulas begin
        y ~ Normal((a + η[1]) + (b + η[2]) * t, σ)
    end
end

# df: a DataFrame with columns ID, t and y
dm = DataModel(model, df; primary_id = :ID, time_col = :t)
res = fit_model(dm, NoLimits.VEM(); rng = Random.Xoshiro(1))
get_params(res; scale = :untransformed)
```

## Optimizer Interface

VEM takes the same optimizer keywords as [`Laplace`](@ref) and [`MLE`](@ref):

- `optimizer`: any Optimization.jl optimizer. The default is
  `LBFGS(m = 20, linesearch = BackTracking(maxstep = 30.0))` from OptimizationOptimJL.
- `optim_kwargs`: keyword arguments forwarded to `Optimization.solve`. With the default
  (`nothing`) VEM uses `(; maxiters = 1000)`, plus `g_abstol = 1e-5`, `reltol = 1e-8` and
  `allow_f_increases = false` for an Optim.jl optimizer. A NamedTuple you pass replaces
  these defaults.
- `adtype`: the default `AutoForwardDiff()` differentiates batch by batch in forward mode.
  Reverse mode through Enzyme differentiates each batch's ELBO over the fixed effects and
  that batch's variational parameters in one pass (batches threaded like the default) and pays
  off for roughly 100 or more fixed-effect coordinates.
- `lb`, `ub`, `ignore_model_bounds`: bounds on the transformed fixed-effect scale, resolved
  as for the other methods (model-declared bounds by default). The variational parameters
  are unbounded, so optimizers that need finite bounds on every coordinate
  (BlackBoxOptim methods, `ParticleSwarm`) are rejected with an error.

The optimizer receives the objective value and its analytic gradient in one call, on a
diagonally rescaled offset of the fixed effects. The optional refinement step (`refine`)
uses the same optimizer, `optim_kwargs` and bounds.

```julia
using OptimizationOptimJL

res = fit_model(
    dm,
    NoLimits.VEM(; optimizer = OptimizationOptimJL.BFGS(), optim_kwargs = (; maxiters = 300));
    rng = Random.Xoshiro(1),
)
get_iterations(res)
```

Reverse mode with Enzyme (load Enzyme first):

```julia
using Enzyme

ad = NoLimits.Optimization.AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Duplicated,
)
res = fit_model(dm, NoLimits.VEM(; adtype = ad); rng = Random.Xoshiro(1))
```

## Penalties and Weights

The `penalty` and `extra_objective` keywords of [`fit_model`](@ref) work as for
[Laplace](laplace.md). Both are functions of the fixed effects on the natural scale; they
are added to the negative ELBO of the joint fit, to the objective of the `refine` step and to
the reported objective, and their gradients (ForwardDiff) enter the optimizer's gradient.
Under [`MiniBatch`](@ref) they are added once per iteration and are not multiplied by
`nbatches / |B|`. The ELBO in the notes of `get_result(res)` excludes them.

```julia
res = fit_model(
    dm, NoLimits.VEM();
    penalty = (; a = 10.0),                       # adds 10 a²
    extra_objective = θ -> 5.0 * (θ.b + 0.5)^2,   # any differentiable function of θ
    rng = Random.Xoshiro(1),
)
```

Likelihood weights of the `DataModel` (`DataModel(...; weights = :w)`) multiply the ELBO
term of each random-effect batch, and its marginal-likelihood term in the `refine` step and
in the reported objective, so a weight of 2 is equivalent to a duplicated individual. The
optimal variational distribution of a batch does not depend on its weight. Random-effect
levels fixed through `constants_re` enter as for Laplace: with the prior density of the fixed
values, counted once per fixed level (not once per batch that contains it), and a batch without
free levels (and hence without variational distribution) with its log-likelihood at the fixed
values. Batches of weight zero do not enter the objective; their reported variational
distribution is the Laplace approximation at ``\hat\theta``.

## Variational Options

- `family = :dense`: full lower-triangular scale; `:diagonal` gives a mean-field family,
  which cannot represent posterior correlations between random effects.
- `expectation = :auto`, `n_samples = :auto`: quadrature for batches with at most three
  random effects, otherwise `n_samples` fixed draws (`:auto` uses max(16, 4 n_b)).
- `init = :laplace`: start each q at the Laplace approximation of the batch posterior at
  the starting values; `:prior` starts from the random-effect prior.
- `refine = :auto`, `refine_nodes = 64`: after the joint fit, maximize an adaptive
  quadrature (or importance-sampled) marginal likelihood over the fixed effects, with the
  rule of each batch re-centered at the mode of its integrand at every θ. This removes the
  bias of the Gaussian family. `:auto` refines for at most 50 free fixed-effect
  coordinates and at most three random effects per batch.
- `n_is = 1000`, `is_df = 5.0`: importance-sampling draws and t degrees of freedom of the
  reported objective, an estimate of ``-\log p(y \mid \hat\theta)``, for batches with more
  than three random effects. Batches with one to three random effects use adaptive
  Gauss-Hermite quadrature around q (21, 9² or 7³ nodes, the `refine` rule; after a
  refinement centered at the mode at the estimate) instead.
  `n_is = 0` reports the negative ELBO instead.
- `ebe_*`: options of the empirical Bayes modes computed after the fit.

## Results and Reproducibility

`get_objective` returns the quadrature or importance-sampling estimate of the negative marginal
log-likelihood (plus `penalty` and `extra_objective`), `get_iterations` the iterations of the
joint fit and `get_converged` whether the joint fit (and an accepted refinement) ended with a
successful return code. The notes of `get_result(res)` hold the final ELBO, the number of
refinement iterations and the fitted variational parameters in random-effect batch order.

[`get_variational_distributions`](@ref) returns the fitted variational distributions per
individual, in the order of the individuals of the `DataModel`: the batch, mean and
Cholesky scale factor of ``q``, whether ``q`` is a distribution of the random effects
themselves (`:natural`, Gaussian random effects) or of the standardized variables of the
prior transport (`:standardized`), and the random effect, level and component of each
coordinate. Individuals linked by a shared random-effect level share one joint ``q``.

```julia
vd = get_variational_distributions(res)
vd.mean[1]          # q mean of the first individual
vd.coordinates[1]   # [(re = :η, level = 1, component = 1), (re = :η, level = 1, component = 2)]
```

All randomness (Monte Carlo draws, refinement nodes, importance-sampling draws, start
repair and EBE multistart) comes from the `rng` keyword of `fit_model`, as for
[SAEM](saem.md) and [MCEM](mcem.md). The default is a fresh `Xoshiro(42)` per call, so
fits are reproducible by default; pass another seeded generator for different draws.

The `serialization` keyword of `fit_model` (default `EnsembleThreads()`) parallelizes the
per-batch work (ELBO and gradient, initialization, refinement, importance sampling and
EBEs) over Julia threads; `EnsembleSerial()` runs it on one thread. Random draws are made
before the threaded loops and batch results are summed in batch order, so a threaded fit is
bitwise equal to a serial one with the same `rng`, for every `adtype`.

Stochastic VEM (a random subset of batches per optimizer iteration) is a wrapper around the method: `fit_model(dm, MiniBatch(VEM(); schedule = 50))`. See [Mini-batching](minibatch.md).

## Downstream Analysis

- Wald uncertainty quantification (`compute_uq(res; method = :wald)`) uses the Laplace
  approximation of the marginal likelihood at the VEM estimate, as for SAEM and MCEM.
- [`sample_random_effects`](@ref) and the `:conditional` mode of [`fit_cv`](@ref) draw
  from the fitted variational distribution of each batch, ``b = m_i + L_i \xi`` with
  ``\xi \sim N(0, I)`` (through the prior transport for non-Gaussian random effects). For a
  `PlanarFlowFamily` the draws come from its moment-matched Gaussian; batches of weight zero
  are drawn from the Laplace approximation at ``\hat\theta``. Fits restored with `load_fit`
  sample the same way.
- [`fit_cv`](@ref) passes `constants_re` to the VEM training fits.

## API

```@docs
VEM
PlanarFlowFamily
get_variational_distributions
```

## Where to go next

- [Choosing a method](index.md#Choosing-a-Method) - how this compares with the alternatives.
- [Laplace](laplace.md) - the deterministic approximation VEM is often compared with.
- [Uncertainty Quantification](../uncertainty-quantification/index.md) - standard errors and intervals for a fit.
