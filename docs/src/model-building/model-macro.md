# `@Model`

`@Model` is the top-level macro that assembles a complete NoLimits model from one or more block macros. It validates the block composition, wires cross-block dependencies, and returns a `Model` struct ready for data binding and estimation.

## Syntax

A minimal model requires only `@formulas` and at least one parameter (fixed or random):

```julia
model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @formulas begin
        y ~ Normal(a, sigma)
    end
end
```

Only macro blocks are allowed inside `@Model` -- no bare Julia statements.

## Available Blocks

Each block can appear at most once within `@Model`:

| Block | Required? | Purpose |
| --- | --- | --- |
| `@helpers` | No | Reusable helper functions |
| `@fixedEffects` | No* | Population-level parameters |
| `@covariates` | No | Covariate declarations |
| `@randomEffects` | No* | Random-effect definitions |
| `@preDifferentialEquation` | No | Time-constant derived quantities |
| `@DifferentialEquation` | No | ODE dynamics |
| `@initialDE` | No | ODE initial conditions |
| `@formulas` | **Yes** | Observation model |

*At least one of `@fixedEffects` or `@randomEffects` must be present.

## Validation Rules

`@Model` enforces the following constraints at construction time:

- `@formulas` is always required.
- At least one fixed effect or one random effect must be defined.
- `@DifferentialEquation` requires `@initialDE`, and vice versa.
- Duplicate blocks are rejected.
- Unknown blocks are rejected.
- Non-macro statements inside `@Model` are rejected.

When a `@DifferentialEquation` block is present, additional covariate usage rules are enforced:

- Non-dynamic varying covariates cannot appear in DE expressions.
- Dynamic covariates must be called with a time argument (e.g., `w(t)`), not used as bare symbols.
- Constant covariates cannot be called as functions in DEs.

## Default Behavior for Omitted Blocks

Blocks that are omitted default to empty or inactive states:

- `@helpers` defaults to an empty `NamedTuple`.
- `@fixedEffects`, `@covariates`, and `@randomEffects` default to empty blocks.
- `@preDifferentialEquation`, `@DifferentialEquation`, and `@initialDE` default to `nothing`.

## Example: Nonlinear Mixed-Effects Model (No ODE)

```julia
using NoLimits
using Distributions

model = @Model begin
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end

    @fixedEffects begin
        a = RealNumber(0.4)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
        x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
    end

    @randomEffects begin
        eta = RandomEffect(TDist(6.0); column=:ID)
    end

    @formulas begin
        mu = sat(a + eta^2 + 0.01 * x.Age^2 + log1p(x.BMI^2))
        y ~ Normal(mu, sigma)
    end
end
```

## Example: Full ODE Model

This example demonstrates the full composition of blocks, including a dynamic covariate, a pre-differential-equation transformation, ODE dynamics with a derived signal, and a nonlinear observation model:

```julia
using NoLimits
using Distributions

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.25)
        b = RealNumber(0.1)
        sigma = RealNumber(0.2, scale=:log)
    end

    @covariates begin
        t = Covariate()
        w = DynamicCovariate()
    end

    @randomEffects begin
        eta = RandomEffect(SkewNormal(0.0, 1.0, 0.8); column=:ID)
    end

    @preDifferentialEquation begin
        pre = exp(a + eta^2)
    end

    @DifferentialEquation begin
        s(t) = tanh(w(t) + pre)
        D(x1) ~ -(b + tanh(eta)) * x1 + s(t)^2
    end

    @initialDE begin
        x1 = pre
    end

    @formulas begin
        y ~ Gamma(log1p(abs(x1(t)) + eta^2) + 1e-6, sigma)
    end
end
```

## Model Summary

After construction, use `NoLimits.summarize(model)` to inspect the declared structure:

```julia
model_summary = NoLimits.summarize(model)
model_summary
```

## Runtime Evaluation Helpers

`@Model` wires internal evaluation functions used during fitting and simulation:

- `calculate_prede(...)` -- evaluate pre-DE expressions
- `calculate_initial_state(...)` -- compute ODE initial conditions
- `calculate_formulas_all(...)` -- evaluate all formula nodes
- `calculate_formulas_obs(...)` -- evaluate observation-node formulas only

When formulas reference ODE states or signals, formula evaluation requires DE solution accessors from `get_de_accessors_builder(...)`.
