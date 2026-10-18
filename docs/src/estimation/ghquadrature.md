# GH Quadrature

Gauss-Hermite Quadrature (`GHQuadrature`) is a deterministic numerical integration method for approximating the marginal likelihood in nonlinear mixed-effects models. Like [`Laplace`](laplace.md), it targets the integral

$$\log L(\theta) = \sum_\text{batch} \log \int p(y_\text{batch} \mid b, \theta)\, p(b \mid \theta)\, db$$

but instead of approximating the integrand by a Gaussian, it evaluates it at a set of carefully chosen quadrature nodes and sums the weighted values. More nodes at higher accuracy levels lead to a more faithful approximation of the true marginal likelihood.

## What It Is

The method applies a change of variables that maps the random-effects vector $b$ to a standard-normal reference variable $z$, then uses a **Smolyak sparse-grid** construction of Gauss-Hermite (GH) nodes to approximate the resulting integral:

$$\log L(\theta) \approx \text{signed-logsumexp}_r \bigl[\log |w_r| + \textstyle\sum_i \ell_i(b_r, \theta) + \log c(b_r, z_r)\bigr]$$

where $\{(z_r, w_r)\}$ are the Smolyak-GH nodes and weights, $b_r = T(z_r)$ is the node in natural parameter space via a transport map $T$, and $\log c$ is a log-correction factor that accounts for the change of measure.

The **Smolyak construction** controls the curse of dimensionality: rather than forming the full tensor product of one-dimensional GH rules (which grows exponentially in the number of random effects), Smolyak grids include only a judiciously chosen subset of tensor-product points. This keeps the number of nodes manageable even for moderately high-dimensional random-effects vectors while preserving the accuracy of the approximation.

The `level` parameter controls the trade-off between accuracy and cost. At level 1, a single node is used per dimension; each additional level adds more nodes at the cost of more likelihood evaluations.

### Transport Maps for Non-Gaussian Random Effects

When random effects follow non-Gaussian distributions, `GHQuadrature` applies a distribution-specific transport map $T: \mathbb{R} \to \text{support}$ before placing nodes:

| RE distribution | Transport | Notes |
| --- | --- | --- |
| `Normal`, `MvNormal` | Identity ($b = \mu + Lz$) | Prior absorbed into GH weights exactly; log-correction = 0 |
| `LogNormal` | Exponential ($b = e^{\mu + \sigma z}$) | Push-forward is LogNormal; log-correction = 0 |
| `Beta` | Scaled logistic | Log-correction is non-zero |
| `Gamma`, `Exponential`, `Weibull` | Exponential ($b = e^z$) | Log-correction is non-zero |
| `Pareto`, any lower bound $a$ | Shifted exponential ($b = a + e^z$) | Log-correction is non-zero |
| Support $(-\infty, b_{\max})$ | Mirrored exponential ($b = b_{\max} - e^z$) | Log-correction is non-zero |
| `TDist` | Identity | Log-correction accounts for heavier tails |
| `NormalizingPlanarFlow` | Bijector flow | Jacobian cancels; log-correction = 0 |
| Any `ContinuousUnivariateDistribution` | By support shape | Generic fallback; log-correction is non-zero |

Every interval support is covered, with either bound finite or infinite.

Discrete distributions (`Poisson`, `Bernoulli`, etc.) are not supported and raise an error at validation time. Continuous univariate distributions whose `logpdf` calls the Rmath C library (`NoncentralT`, `NoncentralF`, `NoncentralChisq`, `NoncentralBeta`, `StudentizedRange`) are also rejected: that `logpdf` is not ForwardDiff-compatible, so no gradient-based marginal estimator (`GHQuadrature`, `Laplace`, `FOCEI`) can use them.

## Adaptive and Prior-Centered Rules

On every objective evaluation, `GHQuadrature` first solves the empirical-Bayes (EB) mode of each batch with the inner optimizer (warm-started from the previous evaluation). How the nodes are then placed depends on the support of the batch's random effects.

**Random effects supported on the whole real line: mode-centered AGHQ.** For batches whose random effects are all `Normal`/`MvNormal`, `TDist`, normalizing flows, or a composite of marginals supported on the whole real line, the nodes are centered and whitened at the EB mode, as `Laplace` does. At level 1 this rule's value equals the Laplace approximation, but its gradient does not; use [`Laplace`](laplace.md) for that, and `level ≥ 3` here.

**Bounded-support random effects: adaptive in the reference space.** For a batch with at least one bounded-support random effect (`LogNormal`, `Gamma`, `Beta`, `Weibull`, ...), centering in `b` would place nodes outside the support. The rule is instead made adaptive in the standard-normal reference variable `z` of the transport map: the mode of the integrand in `z` (found by a damped Newton search warm-started from the EB mode) becomes the center, and the curvature there the scale. The change of variables is exact, so only the node placement changes. This rule needs a higher `level` than the mode-centered one for the same accuracy, which is why `level = :auto` starts at 5 for such models when every batch has at most 3 random-effect dimensions.

In both cases the center and scale are held fixed with respect to the outer gradient; at a converged rule the dropped terms are of the order of the quadrature error. If the mode or its curvature is not available (the integrand is not finite at the start of the search, or the negative Hessian is not positive definite), the batch keeps the prior-centered rule: nodes are placed around the prior through the transport map above. That rule can miss the region of highest likelihood when the data are informative, and the signed sum can then turn negative; see [Numerical Stability](#Numerical-Stability).

## When to Use It

`GHQuadrature` is a good choice when:

- You want a **more accurate marginal likelihood than Laplace** for Gaussian random effects. The adaptive rule refines the Laplace approximation with extra nodes around the EB mode.
- For bounded-support random effects, you want a more faithful marginal likelihood than Laplace at a controlled cost; the reference-space adaptive rule needs a higher `level` than for Gaussian random effects.
- You want to **cross-check Laplace results**. At the same optimum, a consistent GHQuadrature estimate (especially at level ≥ 2) provides evidence that the Laplace approximation is adequate.
- You are fitting models with **non-Gaussian random effects** and want a quadrature-based alternative to Laplace without the approximation error of a Gaussian envelope.

Consider [`Laplace`](laplace.md) or [`MCEM`](mcem.md) / [`SAEM`](saem.md) if the posterior is far from the prior (common with very informative data) or if the model is high-dimensional in random effects.

## Applicability

- Requires a model with random effects.
- Requires at least one free fixed effect.
- Supports every continuous univariate RE distribution with an interval support, plus `MvNormal` and `NormalizingPlanarFlow`.
- Does not support discrete RE distributions, nor the Rmath-backed noncentral families (`NoncentralT`, `NoncentralF`, `NoncentralChisq`, `NoncentralBeta`, `StudentizedRange`), whose `logpdf` is not ForwardDiff-compatible.
- Supports nonlinear models, including ODE-based models.

## Basic Usage

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        b = RealNumber(0.1)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end

    @formulas begin
        mu = a + b * t + eta
        y ~ Normal(mu, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C],
    t  = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y  = [1.0, 1.3, 0.9, 1.2, 1.1, 1.5],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

res = fit_model(dm, NoLimits.GHQuadrature())
```

## Constructor Options

```julia
NoLimits.GHQuadrature(;
    level = :auto,
    optimizer = OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking(maxstep=1.0)),
    optim_kwargs = nothing,   # reltol = 1e-12 for an Optim.jl optimizer
    adtype = Optimization.AutoForwardDiff(),
    inner_optimizer = NoLimits.NewtonInner(x_abstol=1e-10),   # EB modes
    inner_kwargs = NamedTuple(),
    inner_adtype = Optimization.AutoForwardDiff(),
    inner_grad_tol = :auto,
    multistart_n = 50,
    multistart_k = 10,
    multistart_grad_tol = :auto,
    multistart_max_rounds = 1,
    multistart_sampling = :lhs,
    lb = nothing,
    ub = nothing,
    ignore_model_bounds = false,
    precondition = true,
    update_schedule = :all,
)
```

`update_schedule=:all`: which random-effect batches enter the outer objective and gradient per optimizer iteration (mini-batching).

- `:all` (default): use all batches every iteration; the objective is deterministic and identical to previous versions.
- integer `m`: a random minibatch of `min(m, nbatches)` batches, sampled without replacement each iteration.
- callable `(nbatches::Int, iter::Int, rng) -> Vector{Int}`: returns the batch indices to use in optimizer iteration `iter`; a plain function or a callable struct (for stateful schedules).

The selected batches' contribution is scaled by `nbatches / length(selected)`, so the objective remains an unbiased estimate of the full-data objective; priors, `penalty`, and `extra_objective` are never scaled. One minibatch is drawn per optimizer iteration from the fit `rng`, and the objective and gradient of that iteration share it. When mini-batching is active and no `optimizer` is given, the default optimizer becomes `Optimisers.Adam(0.01)` (from Optimisers.jl, with `maxiters=1000` and `save_best=false` unless set in `optim_kwargs`) instead of LBFGS; passing a non-Optimisers optimizer emits a warning because line-search optimizers do not behave well on a stochastic objective. After the fit, the reported objective is re-evaluated once on all batches at the fitted parameters. With Optimisers.jl rules, finite bounds (model bounds or `lb`/`ub`) are enforced by projecting each update onto the box.

### `level`

Controls the accuracy of the Smolyak-GH quadrature. Can be:

- **`:auto`** (default): the fit starts at level 5 when some random effect has bounded support (for example `LogNormal`, `Beta`, `Gamma`) and every batch has at most 3 random-effect dimensions, and at level 3 otherwise. When every random-effect batch is scalar, the marginal log-likelihood at the estimate is then recomputed with the next finer rule (`2L + 1`); while the two values differ by more than 0.5 (whole data set), the fit is refined from the estimate at the finer level. At most 4 refinements are made, and refinement stops before a rule would need more than 2000 nodes per batch. Batches of higher dimension keep the starting level.
- **`Int`** (isotropic): the same level is applied to all RE groups. Levels 1-3 are numerically stable; higher levels can exhibit cancellation in the signed logsumexp for non-Gaussian posteriors.
- **`NamedTuple`** (anisotropic): a different level per RE group, e.g. `level = (eta_id = 3, eta_site = 2)`. RE groups not mentioned default to level 1. Useful when different random effects are expected to need different quadrature accuracy.
- **`Vector{Int}`** (progressive refinement): runs optimization sequentially through the listed levels, warm-starting each from the previous result. For example, `level = [1, 2]` first optimizes at level 1 and then refines at level 2 starting from the level-1 estimates. This can improve stability when starting values are uncertain.

```julia
# Automatic level (default)
res = fit_model(dm, NoLimits.GHQuadrature())

# Isotropic level 3
res = fit_model(dm, NoLimits.GHQuadrature(level=3))

# Anisotropic: higher accuracy for one RE group
res = fit_model(dm, NoLimits.GHQuadrature(level=(eta_id=3, eta_site=2)))

# Progressive: start at level 1 for stability, refine at level 2
res = fit_model(dm, NoLimits.GHQuadrature(level=[1, 2]))
```

### Option Groups

| Group | Keywords | What they control |
| --- | --- | --- |
| Outer optimization | `optimizer`, `optim_kwargs`, `adtype` | Optimization over fixed effects. |
| Inner EB | `inner_optimizer`, `inner_kwargs`, `inner_adtype`, `inner_grad_tol`, `multistart_*` | Inner optimizer for the EB modes. It runs on every objective evaluation (the modes center the adaptive rule, or warm-start the reference-space mode search for bounded-support random effects) and once more, cold-started, at the optimum for `get_random_effects`. |
| Bounds | `lb`, `ub`, `ignore_model_bounds` | Box bounds on transformed fixed-effect parameters. |
| Mini-batching | `update_schedule` | Which batches enter the outer objective/gradient per optimizer iteration. |

The default outer optimizer is `OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking(maxstep = 1.0))`; when `update_schedule != :all` and no optimizer is passed, the default is `Optimisers.Adam(0.01)` instead. The default inner optimizer is [`NewtonInner(x_abstol = 1e-10)`](@ref NewtonInner), as for `Laplace`; see the [Laplace](laplace.md) page for the inner options.

!!! note "Inner optimization during fitting"
    Like `Laplace`, `GHQuadrature` solves the EB modes on every objective evaluation; they center the adaptive rule for random effects supported on the whole real line, and warm-start the reference-space mode search for bounded-support random effects. The modes are solved once more, cold-started, at the optimum for `get_random_effects`.

## Accessing Results

```julia
theta_u    = get_params(res; scale=:untransformed)
theta_t    = get_params(res; scale=:transformed)
obj        = get_objective(res)
converged  = get_converged(res)
re         = get_random_effects(res)   # empirical Bayes mode estimates
ll         = get_loglikelihood(res)   # sparse-grid marginal log-likelihood
```

## Numerical Stability

The Smolyak weights alternate in sign at higher levels (inclusion-exclusion construction). When the integrand is not well approximated by the Gaussian reference of the rule, the positive and negative contributions can nearly cancel, producing a numerically small or negative result from the signed logsumexp. When the signed sum of a batch is not positive and finite, that batch falls back to the single-node level-1 rule (weight 1, so it cannot be negative; under the adaptive rule its value is the Laplace approximation), and a warning is emitted once. This applies to adaptive and prior-centered batches alike. The batch log-likelihood is `-Inf` only if the level-1 value is not finite either.

Practical guidance:

- Levels 1-3 are numerically stable for most NLME models with moderate data.
- If instability appears at level 2, try `level = [1, 2]` (progressive refinement) to reach a better starting region first.
- If instability persists, fall back to `Laplace`, which avoids this issue by design.
- For heavily non-Gaussian posteriors or very informative data, `MCEM` or `SAEM` may be more appropriate.

## Serialization

```julia
using SciMLBase

res_threads = fit_model(dm, NoLimits.GHQuadrature(level=2); serialization=EnsembleThreads())
```

## Where to go next

- [Choosing a method](index.md#Choosing-a-Method) - how this compares with the alternatives.
- [MCEM](mcem.md) - the next method.
- [Uncertainty Quantification](../uncertainty-quantification/index.md) - standard errors and intervals for a fit.
- [Troubleshooting](../troubleshooting.md) - when a fit fails or does not converge.
