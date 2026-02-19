# `@preDifferentialEquation`

In many longitudinal models, certain quantities depend on fixed effects, random effects, and covariates but do not vary with time. Computing these derived quantities once before ODE integration -- rather than redundantly at each solver step -- improves both clarity and performance.

The `@preDifferentialEquation` block defines such time-constant derived quantities. Values declared here are available in `@DifferentialEquation`, `@initialDE`, and `@formulas`.

## Core Syntax

The block accepts only assignment statements. Each left-hand side must be a plain symbol, and expressions may reference any in-scope model quantities except time (`t` and `\xi` are forbidden).

```julia
prede = @preDifferentialEquation begin
    ka = exp(tka + eta[1])
    cl = exp(tcl + eta[2])
    v = exp(tv + eta[3])
end
```

## Parsing and Validation Rules

The following constraints are enforced at parse time:

- The block must be wrapped in `begin ... end`.
- Only assignments are allowed; other statement forms are rejected.
- Left-hand side must be a symbol.
- The time symbols `t` and `\xi` are forbidden in expressions, since preDE values must be time-invariant.
- Mutating patterns trigger a warning, as they may break automatic differentiation.

## Symbol Resolution

`@preDifferentialEquation` expressions can reference the following model components:

- Fixed effects
- Random effects
- Constant covariates (`constant_features_i`)
- Helper functions declared in `@helpers`
- Model functions produced by learned parameter blocks (`NNParameters`, `SoftTreeParameters`, `SplineParameters`, `NPFParameter`)

## Example: Nonlinear Transform of Multivariate Random Effects

A common pattern is to define individual-level parameters as log-linear functions of population means and random effects. Here, three rate and volume parameters are derived from fixed effects and a multivariate random effect vector before being passed into the ODE system.

```julia
using NoLimits
using Distributions
using LinearAlgebra

model = @Model begin
    @fixedEffects begin
        tka = RealNumber(0.45)
        tcl = RealNumber(1.0)
        tv = RealNumber(3.45)
        omega1 = RealNumber(1.0, scale=:log)
        omega2 = RealNumber(1.0, scale=:log)
        omega3 = RealNumber(1.0, scale=:log)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(MvNormal([0.0, 0.0, 0.0], Diagonal([omega1, omega2, omega3])); column=:ID)
    end

    @preDifferentialEquation begin
        ka = exp(tka + eta[1])
        cl = exp(tcl + eta[2])
        v = exp(tv + eta[3])
    end

    @DifferentialEquation begin
        D(depot) ~ -ka * depot
        D(center) ~ ka * depot - cl / v * center
    end

    @initialDE begin
        depot = 0.0
        center = 0.0
    end

    @formulas begin
        y ~ Erlang(3, log1p((center(t) / v)^2) + sigma)
    end
end
```

## Example: Using Helpers and Learned Model Functions

PreDE expressions can also incorporate neural networks, soft decision trees, and spline functions. This is useful when individual-level baseline quantities are modeled as flexible, learned functions of covariates.

```julia
using NoLimits
using Lux
using Distributions

chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
knots = collect(range(0.0, 1.0; length=6))

model = @Model begin
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end

    @fixedEffects begin
        ζ = NNParameters(chain; function_name=:NNB, calculate_se=false)
        Γ = SoftTreeParameters(2, 2; function_name=:STB, calculate_se=false)
        sp = SplineParameters(knots; function_name=:SPB, calculate_se=false)
        sy = RealNumber(0.4, scale=:log)
    end

    @covariates begin
        t = Covariate()
        x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
    end

    @preDifferentialEquation begin
        nn = NNB([x.Age, x.BMI], ζ)[1]
        st = STB([x.Age, x.BMI], Γ)[1]
        spv = SPB(0.4, sp)
        drive = sat(nn + st + spv)
    end

    @formulas begin
        y ~ Gamma(log1p(drive^2) + 1e-6, sy)
    end
end
```

## Accessors

The following functions provide programmatic access to preDE internals:

- `get_prede_names(prede)` -- returns the declared preDE variable names.
- `get_prede_syms(prede)` -- returns the parsed symbol dependencies for each variable.
- `get_prede_builder(prede)` -- returns the evaluation function that computes preDE values from model inputs.

## Data-Model Constraint for Random Effects in preDE

Because preDE values are computed once per individual per model evaluation, any random effects referenced in this block must be grouped by `primary_id` in the `DataModel`. Random effects grouped by a different column (e.g., a site-level grouping) are not permitted in preDE expressions, since their values would not be uniquely determined at the individual level.
