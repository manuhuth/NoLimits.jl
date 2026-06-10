
# Pooled / PooledMap {#Pooled-/-PooledMap}

Pooled estimation is the simplest way to fit a model that contains random effects: instead of integrating the random effects out, each individual's random-effect vector is replaced by a single **plug-in value** drawn from its random-effect distribution, and the data log-likelihood alone is optimized over the free fixed effects. It is fast, requires no inner optimization or sampling, and is most useful as a quick exploratory fit, a sanity check, or a warm start for a full mixed-effects method. It is the naive-pooled member of the nonlinear mixed-effects toolbox surveyed by [Davidian and Giltinan [2]](/references#davidian2003nonlinear).

## The Plug-In Objective {#The-Plug-In-Objective}

For each individual $i$ the random effects are fixed at the plug-in value of their distribution $p_\eta(\cdot \mid \theta)$:

$$\tilde\eta_i(\theta) = \mathbb{E}[\eta_i \mid \theta],$$

the distributional **mean** (falling back to the **median** when the mean is undefined, and to a fixed-draw **Monte-Carlo mean** for normalizing-flow random effects). The population parameters are then estimated by maximizing the data likelihood with the plug-in substituted:

$$\hat\theta = \arg\max_{\theta} \sum_{i=1}^{N} \log p_{y}\!\left(y_i \mid \tilde\eta_i(\theta), \theta\right).$$

The plug-in $\tilde\eta_i(\theta)$ is a function of the fixed effects and is **recomputed at every objective evaluation**. Consequently, any parameter that shifts the plug-in — for example a population mean located _inside_ the random-effect distribution — is estimated normally. The random-effect prior itself is never evaluated.

## Automatic Freezing of Non-Contributing Parameters {#Automatic-Freezing-of-Non-Contributing-Parameters}

Because the random effects are fixed at their mean, parameters that only control the _spread_ of the random-effect distribution carry no signal in the pooled objective. NoLimits.jl detects and holds constant exactly those fixed effects with **no detectable likelihood contribution**:
- **Dispersion-only** parameters whose plug-in sensitivity (the mean-Jacobian) is zero at the start and at jittered probe points, cross-checked against a spread measure (variance / IQR) and an end-to-end objective-invariance test.
  
- **Collinear** parameters whose plug-in effect is redundant given the remaining free parameters at every probe point (for example, only the ratio of a `Beta(α, β)` is identified from its mean).
  

The freeze classification — which parameters were held constant and why — is reported in [`get_notes`](/api#NoLimits.get_notes):

```julia
res = fit_model(dm, NoLimits.Pooled())
get_notes(res)   # explains the auto-freeze decisions
```


::: warning Pooled estimation ignores between-subject variability

The dispersion/variance parameters of the random-effect distribution are not identifiable from the pooled objective and are held at their initial values. To estimate variance components, use a full mixed-effects method such as [Laplace](laplace.md), [FOCEI](focei.md), [SAEM](saem.md), or [MCEM](mcem.md).

:::

## Basic Usage {#Basic-Usage}

In the model below, `base` is the population mean _inside_ the random-effect distribution, so it shifts the plug-in and is estimated; `omega` is the random-effect standard deviation and is automatically frozen; `slope` and `sigma` enter the data likelihood directly and are estimated.

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @fixedEffects begin
        base  = RealNumber(1.0)               # mean of the RE distribution -> estimated
        slope = RealNumber(0.2)
        sigma = RealNumber(0.3, scale=:log)
        omega = RealNumber(0.4, scale=:log)   # RE spread -> auto-frozen
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(base, omega); column=:ID)
    end

    @formulas begin
        mu = eta + slope * t
        y ~ Normal(mu, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :A, :B, :B, :B, :C, :C, :C],
    t  = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    y  = [1.0, 1.3, 1.5, 0.8, 1.0, 1.2, 1.2, 1.4, 1.7],
)

dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
res = fit_model(dm, NoLimits.Pooled())

theta_u = get_params(res; scale=:untransformed)
get_notes(res)
```


## Constructor Options {#Constructor-Options}

```julia
using NoLimits
using Optimization
using OptimizationOptimJL
using LineSearches

method = NoLimits.Pooled(;
    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs=NamedTuple(),
    adtype=Optimization.AutoForwardDiff(),
    force_free=Symbol[],        # parameters to exempt from auto-freezing
    refreeze_check=:warn,       # :warn records violations; :refit unfreezes and continues
    identifiable_only=true,     # freeze plug-in-collinear parameters
    n_probes=3,                 # probe points for the sensitivity analysis
    mc_draws=256,               # base draws for the flow plug-in mean
    lb=nothing,
    ub=nothing,
    ignore_model_bounds=false,
)
```


|                               Keyword |                            Default |                                                                                                                                                               Description |
| -------------------------------------:| ----------------------------------:| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `optimizer`, `optim_kwargs`, `adtype` | `LBFGS`, `()`, `AutoForwardDiff()` |                                                               Optimization over the free fixed effects via [Optimization.jl](https://docs.sciml.ai/Optimization/stable/). |
|                          `force_free` |                         `Symbol[]` |                                                                           Parameter names exempted from auto-freezing (kept free even if classified as non-contributing). |
|                      `refreeze_check` |                            `:warn` | Post-fit sensitivity re-check at the optimum. `:warn` records violations in the notes; `:refit` unfreezes violators and continues, warm-started from the current optimum. |
|                   `identifiable_only` |                             `true` |                                                   Freeze plug-in-collinear parameters via pivoted redundancy elimination; `false` keeps all contributing parameters free. |
|                            `n_probes` |                                `3` |                                                                                 Number of probe points (start + jittered) for the transformed-scale sensitivity analysis. |
|                            `mc_draws` |                              `256` |                                                                                     Fixed base draws for the Monte-Carlo plug-in mean of normalizing-flow random effects. |
|     `lb`, `ub`, `ignore_model_bounds` |      `nothing`, `nothing`, `false` |                                                           Optional transformed-scale bounds for free fixed effects; `ignore_model_bounds` disables model-declared bounds. |


## MAP Regularization: `PooledMap` {#MAP-Regularization:-PooledMap}

[`PooledMap`](/api#NoLimits.PooledMap) is identical to `Pooled` but adds the log-prior of the fixed effects to the objective (a MAP estimate on the data likelihood with the random effects plugged in at their distributional means). It requires a prior on at least one fixed effect. Auto-frozen parameters are held constant, and their priors contribute a constant offset to the reported objective.

```julia
model_map = @Model begin
    @fixedEffects begin
        base  = RealNumber(1.0,             prior=Normal(1.0, 0.5))
        slope = RealNumber(0.2,             prior=Normal(0.0, 1.0))
        sigma = RealNumber(0.3, scale=:log, prior=LogNormal(log(0.3), 0.5))
        omega = RealNumber(0.4, scale=:log, prior=LogNormal(log(0.4), 0.5))
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(base, omega); column=:ID)
    end

    @formulas begin
        mu = eta + slope * t
        y ~ Normal(mu, sigma)
    end
end

dm_map  = DataModel(model_map, df; primary_id=:ID, time_col=:t)
res_map = fit_model(dm_map, NoLimits.PooledMap())
```


## Use as a Warm Start {#Use-as-a-Warm-Start}

Because it is cheap, a quick pooled pre-fit is an effective way to initialize a full mixed-effects estimation. Pass `pooled_init=true` to [`fit_model`](/api#NoLimits.fit_model) to run a short `Pooled` (or `PooledMap`) fit first and start the requested method from its estimate:

```julia
res = fit_model(dm, NoLimits.Laplace(); pooled_init=true)
```


Inside [`Multistart`](multistart.md) the pre-fit runs once per starting point. Pass a custom `Pooled`/`PooledMap` instance instead of `true` for full control of the pre-fit, and use `fit_options_pooled_init` to override its keywords. The warm start requires random effects and is not available when the fitted method is itself `Pooled`/`PooledMap`; a failing pre-fit warns and falls back to the unmodified start.

## Uncertainty Quantification {#Uncertainty-Quantification}

Wald confidence intervals for the free (non-frozen) parameters are available through [`compute_uq`](/api#NoLimits.compute_uq); see [Uncertainty Quantification](../uncertainty-quantification/index.md).
