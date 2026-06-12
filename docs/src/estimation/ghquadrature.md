# GH Quadrature

Gauss-Hermite Quadrature (`GHQuadrature`) is a deterministic numerical integration method for approximating the marginal likelihood in nonlinear mixed-effects models. Like [`Laplace`](laplace.md), it targets the integral

$$\log L(\theta) = \sum_\text{batch} \log \int p(y_\text{batch} \mid b, \theta)\, p(b \mid \theta)\, db$$

but instead of approximating the integrand by a Gaussian, it evaluates it at a set of carefully chosen quadrature nodes and sums the weighted values. More nodes at higher accuracy levels lead to a more faithful approximation of the true marginal likelihood.

## What It Is

The method applies a change of variables that maps the random-effects vector $b$ to a standard-normal reference variable $z$, then uses a **Smolyak sparse-grid** construction of Gauss-Hermite (GH) nodes to approximate the resulting integral:

$$\log L(\theta) \approx \text{signed-logsumexp}_r \bigl[\log |w_r| + \textstyle\sum_i \ell_i(b_r, \theta) + \log c(b_r, z_r)\bigr]$$

where $\{(z_r, w_r)\}$ are the Smolyak–GH nodes and weights, $b_r = T(z_r)$ is the node in natural parameter space via a transport map $T$, and $\log c$ is a log-correction factor that accounts for the change of measure.

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
| `TDist` | Identity | Log-correction accounts for heavier tails |
| `NormalizingPlanarFlow` | Bijector flow | Jacobian cancels; log-correction = 0 |
| Any `ContinuousUnivariateDistribution` | By support shape | Generic fallback; log-correction is non-zero |

Discrete distributions (`Poisson`, `Bernoulli`, etc.) are not supported and raise an error at validation time.

## What It Is Not

**`GHQuadrature` is currently prior-centered, not posterior-centered.** The quadrature nodes are placed around the prior mean of each random-effects distribution, not around the posterior mode. This is the standard (non-adaptive) Gauss-Hermite approach.

As a consequence:

- When the data are highly informative and the posterior mode lies far from the prior mean, the nodes may miss the region of highest likelihood. In that case, the signed logsumexp can become numerically negative (especially at level ≥ 2), signalling poor quadrature accuracy.
- The method works best when the posterior is roughly centered on the prior - for example, with moderate-information data, well-calibrated priors, or Gaussian random effects where the prior is a reasonable envelope for the posterior.
- Level 1 in the current (prior-centered) form is **not** equivalent to the Laplace approximation. That equivalence holds only for the adaptive (posterior-centered) variant, which is not yet implemented.

An adaptive version (AGHQ), which re-centers nodes at the posterior mode per outer iteration, is planned and will address these limitations. For now, if you observe numerical instability at higher levels, consider using [`Laplace`](laplace.md) instead or staying at level 1.

## When to Use It

`GHQuadrature` is a good choice when:

- You have **no reliable starting values** and want a robust objective surface to explore. Because the objective is fully differentiable with respect to the fixed effects (no inner optimization during the forward pass), gradient-based optimizers can traverse the parameter space more freely than under Laplace.
- You **know the posterior is close to the prior** - for instance, with weak data or well-specified priors - and want a more faithful marginal likelihood than Laplace at a controlled cost.
- You want to **cross-check Laplace results**. At the same optimum, a consistent GHQuadrature estimate (especially at level ≥ 2) provides evidence that the Laplace approximation is adequate.
- You are fitting models with **non-Gaussian random effects** and want a quadrature-based alternative to Laplace without the approximation error of a Gaussian envelope.

Consider [`Laplace`](laplace.md) or [`MCEM`](mcem.md) / [`SAEM`](saem.md) if the posterior is far from the prior (common with very informative data) or if the model is high-dimensional in random effects.

## Applicability

- Requires a model with random effects.
- Requires at least one free fixed effect.
- Supports all continuous univariate RE distributions, `MvNormal`, and `NormalizingPlanarFlow`.
- Does not support discrete RE distributions.
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
    level = 3,
    optimizer = OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs = NamedTuple(),
    adtype = Optimization.AutoForwardDiff(),
    inner_optimizer = ...,   # used post-hoc for get_random_effects only
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
)
```

### `level`

Controls the accuracy of the Smolyak–GH quadrature. Can be:

- **`Int`** (isotropic): the same level is applied to all RE groups. Levels 1–3 are numerically stable; higher levels can exhibit cancellation in the signed logsumexp for non-Gaussian posteriors.
- **`NamedTuple`** (anisotropic): a different level per RE group, e.g. `level = (eta_id = 3, eta_site = 2)`. RE groups not mentioned default to level 1. Useful when different random effects are expected to need different quadrature accuracy.
- **`Vector{Int}`** (progressive refinement): runs optimization sequentially through the listed levels, warm-starting each from the previous result. For example, `level = [1, 2]` first optimizes at level 1 and then refines at level 2 starting from the level-1 estimates. This can improve stability when starting values are uncertain.

```julia
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
| Inner EB (post-hoc) | `inner_optimizer`, `inner_kwargs`, `inner_adtype`, `inner_grad_tol`, `multistart_*` | Inner optimizer used **only** after fitting to compute EB modes for `get_random_effects`. Not used during the forward pass. |
| Bounds | `lb`, `ub`, `ignore_model_bounds` | Box bounds on transformed fixed-effect parameters. |

!!! note "No inner optimization during fitting"
    Unlike `Laplace`, `GHQuadrature` does **not** run an inner optimization during the forward pass. The objective is a direct sum over quadrature nodes and is fully differentiable by ForwardDiff. The inner optimizer is used only after convergence to compute empirical Bayes mode estimates for `get_random_effects`.

## Accessing Results

```julia
theta_u    = get_params(res; scale=:untransformed)
theta_t    = get_params(res; scale=:transformed)
obj        = get_objective(res)
converged  = get_converged(res)
re         = get_random_effects(res)   # empirical Bayes mode estimates
ll         = get_loglikelihood(res)
```

## Numerical Stability

The Smolyak weights alternate in sign at higher levels (inclusion-exclusion construction). When the integrand is not well approximated by the prior-centered Gaussian, the positive and negative contributions can nearly cancel, producing a numerically small or negative result from the signed logsumexp. When this happens, the batch marginal likelihood is returned as `-Inf` and a warning is emitted.

Practical guidance:

- Levels 1–3 are numerically stable for most NLME models with moderate data.
- If instability appears at level 2, try `level = [1, 2]` (progressive refinement) to reach a better starting region first.
- If instability persists, fall back to `Laplace`, which avoids this issue by design.
- For heavily non-Gaussian posteriors or very informative data, `MCEM` or `SAEM` may be more appropriate.

## Serialization

```julia
using SciMLBase

res_threads = fit_model(dm, NoLimits.GHQuadrature(level=2); serialization=EnsembleThreads())
```
