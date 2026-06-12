# FOCEI

First-Order Conditional Estimation with Interaction (FOCEI) is a workhorse algorithm for
nonlinear mixed-effects models, introduced by [lindstrom1990nonlinear](@citet) and analyzed
in the pharmacometric setting by [wang2007derivation](@citet). Like the
[Laplace](laplace.md) approximation, FOCEI integrates out the random effects through a
second-order expansion of the log-joint density around each individual's empirical Bayes
(EB) mode. The two methods differ only in how the inner curvature (the negative Hessian of
the log-joint) is formed.

## How FOCEI Differs from Laplace

The Laplace marginal-likelihood approximation requires the negative Hessian of the
individual log-joint with respect to the random effects, evaluated at the EB mode. FOCEI
replaces this exact Hessian with the **expected-information (Gauss–Newton) form**

```math
H_i(\theta) \;=\; \sum_{j} J_{ij}^{\top}\, \mathcal{I}(\phi_{ij})\, J_{ij} \;-\; \nabla_{\eta}^2 \log p_\eta(\hat\eta_i \mid \theta),
```

where ``J_{ij} = \partial \phi_{ij}/\partial \eta`` is the first-order Jacobian of the
outcome-distribution parameters ``\phi_{ij}`` with respect to the random effects, and
``\mathcal{I}(\phi_{ij})`` is the closed-form Fisher information of the outcome family.

This has two consequences:

- The per-subject curvature drops from **second-order to first-order** automatic
  differentiation, which is typically cheaper and more numerically stable.
- The curvature is **positive-definite by construction**, avoiding the Hessian
  stabilization that the exact Laplace path may require.

The `interaction` flag controls whether the dependence of dispersion-type parameters
(e.g. a residual-error standard deviation) on the random effects is retained:

- `interaction=true` (default) - full FOCEI; the interaction between random effects and
  the residual model is kept.
- `interaction=false` - FOCE; dispersion-type parameters are frozen at the random-effects
  prior mean and their dependence on the random effects is ignored.

## Supported Outcome Families

FOCEI requires a registered closed-form Fisher information for each outcome distribution.
The supported families are:

`Normal`, `LogNormal`, `Laplace`, `Cauchy`, `Exponential`, `Poisson`, `Bernoulli`,
`Binomial`, `Geometric`, `Gamma`, `Beta`, and `MvNormal`.

!!! warning "Unsupported outcome models"
    Hidden Markov / Markov outcome models and any distribution without a registered Fisher
    information are **not** supported by FOCEI. Use the [Laplace](laplace.md) approximation
    for those models - it makes no closed-form-information assumption.

## Applicability

- Requires a model with random effects.
- Requires at least one free fixed effect.
- The outcome distribution must be one of the supported families above.

## Basic Usage

The following example fits a nonlinear mixed-effects model with a subject-level random
effect and a Normal residual model. Because the residual standard deviation `sigma` is a
dispersion parameter and the conditional mean depends on `eta`, this model exercises the
interaction term.

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @fixedEffects begin
        a     = RealNumber(1.0)
        b     = RealNumber(0.2, scale=:log)
        sigma = RealNumber(0.3, scale=:log)
        omega = RealNumber(0.4, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, omega); column=:ID)
    end

    @formulas begin
        mu = a * exp(-b * t) + eta   # nonlinear in t, conditional on eta
        y ~ Normal(mu, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :A, :B, :B, :B, :C, :C, :C],
    t  = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    y  = [1.9, 1.2, 0.8, 2.1, 1.5, 1.0, 1.7, 1.1, 0.7],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

res = fit_model(dm, NoLimits.FOCEI())
```

To estimate without the interaction term (FOCE), pass `interaction=false`:

```julia
res_foce = fit_model(dm, NoLimits.FOCEI(; interaction=false))
```

## Constructor Options

All keyword arguments of `FOCEI` **mirror those of [`Laplace`](laplace.md)** - the outer
fixed-effects optimizer, the inner EB optimization, the EB multistart policy, the Hessian
jitter controls, caching, and bounds all behave identically. The only addition is
`interaction::Bool=true`. See the [Laplace](laplace.md) page for the full description of
the shared option groups, and the [`FOCEI`](@ref) API entry for the complete keyword list.

```julia
using NoLimits
using Optimization
using OptimizationOptimJL
using LineSearches

method = NoLimits.FOCEI(;
    interaction=true,
    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs=(maxiters=100,),
    adtype=Optimization.AutoForwardDiff(),
    inner_grad_tol=:auto,
    multistart_n=50,
    multistart_k=10,
)
```

## Accessing Results

FOCEI fits expose the standard accessor interface, including empirical Bayes estimates of
the random effects and the FOCEI-approximated marginal log-likelihood.

```julia
theta_u = get_params(res; scale=:untransformed)
obj     = get_objective(res)
ok      = get_converged(res)

re_df   = get_random_effects(res)
ll      = get_loglikelihood(res)
```

Conditional weighted residuals (CWRES), the diagnostic most closely associated with the
FOCE family [hooker2007conditional](@cite), and the other residual diagnostics are
available through [`get_residuals`](@ref) and the residual-plot functions; see
[Plotting](../plotting/index.md).

## Uncertainty Quantification

Wald confidence intervals from the inverse observed-information matrix are available for
FOCEI fits through [`compute_uq`](@ref); see
[Uncertainty Quantification](../uncertainty-quantification/index.md).
