# MLE

Maximum likelihood estimation (MLE) is the standard frequentist approach to parameter estimation. It finds the fixed-effect values that make the observed data most probable under the assumed model. In NoLimits.jl, MLE supports nonlinear models, ODE-based dynamics, and non-Gaussian outcome distributions, provided that the model contains only fixed effects (no random effects).

## Applicability

- Applicable only to models without random effects.
- The model must declare at least one fixed effect.
- At least one fixed effect must remain free (i.e., not all parameters may be held constant via `constants`).

If the model includes random effects, `MLE` will raise an error. Use a mixed-effects method such as `Laplace`, `SAEM`, or `MCEM` instead.

## Basic Usage

The following example defines a simple nonlinear model with an exponential mean function and fits it to a small dataset.

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @covariates begin
        t = Covariate()
        z = Covariate()
    end

    @fixedEffects begin
        a = RealNumber(0.1)
        b = RealNumber(0.2)
        sigma = RealNumber(0.5, scale=:log)
    end

    @formulas begin
        mu = exp(a + b * z)
        y ~ Exponential(mu * sigma)
    end
end

df = DataFrame(
    ID = [1, 1, 2, 2],
    t = [0.0, 1.0, 0.0, 1.0],
    z = [0.0, 0.5, 1.0, 1.5],
    y = [1.0, 1.1, 1.3, 1.7],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
res = fit_model(dm, NoLimits.MLE())
```

## Constructor Options

The `MLE` constructor exposes options for the optimizer, automatic differentiation backend, and optional parameter bounds.

```julia
using NoLimits
using Optimization
using OptimizationOptimJL
using LineSearches

method = NoLimits.MLE(;
    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs=NamedTuple(),
    adtype=Optimization.AutoForwardDiff(),
    lb=nothing,
    ub=nothing,
)
```

## Option Groups

| Group | Keywords | Description |
| --- | --- | --- |
| Optimization | `optimizer`, `optim_kwargs`, `adtype` | Controls fixed-effect objective optimization via [Optimization.jl](https://docs.sciml.ai/Optimization/stable/). |
| Bounds | `lb`, `ub` | Optional transformed-scale bounds for free fixed effects. |

## Optimization.jl Interface

`MLE` delegates numerical optimization to the SciML [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) ecosystem. Any optimizer compatible with that interface can, in principle, be used. The following have been tested:

- `OptimizationOptimJL.LBFGS` (default)
- `Optim.BFGS`
- `Optim.NelderMead`
- `OptimizationOptimisers.Adam`
- `OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited()`

## Bounds and BlackBoxOptim

User-supplied bounds (`lb`, `ub`) operate on the transformed scale of the free parameters. Bound entries for parameters held fixed via `constants` are ignored automatically.

Derivative-free optimizers from BlackBoxOptim require finite lower and upper bounds for every free parameter. The convenience function `default_bounds_from_start(dm; margin=1.0)` constructs default box bounds on the transformed scale, centered on the initial values.

## Objective and Penalty

The `MLE` objective is the negative log-likelihood of the data given the model parameters. An optional L2-style penalty can be added through the `penalty` keyword of `fit_model`. The penalty is applied parameter-wise and supports both scalar and vector parameter blocks.

## Fit Keywords

The following keyword arguments are accepted by `fit_model(dm, NoLimits.MLE(...); ...)`:

- `constants` -- named tuple of fixed-effect values to hold constant during optimization.
- `penalty` -- named tuple of per-parameter penalty weights (L2, on the natural scale).
- `ode_args`, `ode_kwargs` -- additional positional and keyword arguments forwarded to the ODE solver.
- `serialization` -- ensemble algorithm controlling parallelism (e.g., `EnsembleThreads()`).
- `rng` -- random number generator, used where stochastic initialization is needed.
- `theta_0_untransformed` -- custom starting values on the natural (untransformed) scale.
- `store_data_model` -- whether to store the `DataModel` in the result (default: `true`).

The keyword `constants_re` does not apply to `MLE`, as this method does not involve random effects.

## ODE Example

Models that include ordinary differential equations are handled transparently -- no special configuration is needed beyond specifying the ODE system. The example below fits a scalar ODE with a quadratic nonlinearity.

```julia
using NoLimits
using DataFrames
using Distributions

model_ode = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @DifferentialEquation begin
        D(x1) ~ -a * x1^2
    end

    @initialDE begin
        x1 = 1.0
    end

    @formulas begin
        y ~ Erlang(3, log1p(x1(t)^2) + sigma)
    end
end

df_ode = DataFrame(
    ID = [1, 1],
    t = [0.0, 1.0],
    y = [1.0, 1.05],
)

model_saveat = set_solver_config(model_ode; saveat_mode=:saveat)
dm_ode = DataModel(model_saveat, df_ode; primary_id=:ID, time_col=:t)
res_ode = fit_model(dm_ode, NoLimits.MLE())
```

## Non-Gaussian Outcome Example

NoLimits.jl supports any outcome distribution available in `Distributions.jl`. The following example uses a Poisson likelihood for count data.

```julia
using NoLimits
using DataFrames
using Distributions

model_pois = @Model begin
    @covariates begin
        t = Covariate()
        z = Covariate()
    end

    @fixedEffects begin
        a = RealNumber(0.1)
        b = RealNumber(0.2)
    end

    @formulas begin
        lambda = exp(a + b * z)
        y ~ Poisson(lambda)
    end
end

df_pois = DataFrame(
    ID = [1, 1, 2, 2],
    t = [0.0, 1.0, 0.0, 1.0],
    z = [0.0, 0.5, 1.0, 1.5],
    y = [1, 1, 2, 3],
)

dm_pois = DataModel(model_pois, df_pois; primary_id=:ID, time_col=:t)
res_pois = fit_model(dm_pois, NoLimits.MLE())
```

## HMM Example (Fixed-Effects-Only)

Hidden Markov models (HMMs) with discrete states and discrete time steps can also be estimated via `MLE`. In the example below, emission probabilities are parameterized through constrained logistic transforms to ensure they remain in the valid range.

```julia
using NoLimits
using DataFrames
using Distributions

model_hmm = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        p1_r = RealNumber(0.0)
        p2_r = RealNumber(0.0)
    end

    @formulas begin
        p1 = 0.8 / (1 + exp(-p1_r)) + 0.1
        p2 = 0.8 / (1 + exp(-p2_r)) + 0.1
        P = [0.9 0.1; 0.2 0.8]
        y ~ DiscreteTimeDiscreteStatesHMM(
            P,
            (Bernoulli(p1), Bernoulli(p2)),
            Categorical([0.6, 0.4]),
        )
    end
end

df_hmm = DataFrame(
    ID = [1, 1, 1, 2, 2, 2],
    t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    y = [0, 1, 1, 1, 0, 1],
)

dm_hmm = DataModel(model_hmm, df_hmm; primary_id=:ID, time_col=:t)
res_hmm = fit_model(dm_hmm, NoLimits.MLE(optim_kwargs=(iterations=5,)))
```

## Accessing Results

After fitting, use the standard accessor functions to retrieve parameter estimates, the final objective value, convergence status, and diagnostics.

```julia
theta_u = get_params(res; scale=:untransformed)
theta_t = get_params(res; scale=:transformed)
obj = get_objective(res)
ok = get_converged(res)
diag = get_diagnostics(res)
```
