# `@initialDE`

Every ODE system requires a complete specification of initial conditions. The `@initialDE` block defines the initial value of each state declared in `@DifferentialEquation`, and it is mandatory whenever that block is present.

Initial values may be constants, expressions involving fixed effects and random effects, or outputs of learned model functions -- enabling individual-specific starting conditions in mixed-effects models.

## Core Syntax

The block accepts only assignment statements. Each left-hand side must match a state name from `@DifferentialEquation`.

```julia
init = @initialDE begin
    x1 = 1.0
    x2 = 0.0
end
```

## Parsing and Validation Rules

The following constraints are enforced at parse time:

- The block must be wrapped in `begin ... end`.
- Only assignments are allowed; other statement forms are rejected.
- Left-hand side must be a symbol.
- Duplicate initial-state names are rejected.
- The time symbols `t` and `\xi` are forbidden, since initial conditions are evaluated before integration begins.

At build time, additional checks ensure consistency with `@DifferentialEquation`:

- Every DE state must have a corresponding initial value.
- No extra names beyond those declared as DE states are permitted.

## Symbol and Function Resolution

Initial-value expressions can reference the following model components:

- Fixed effects
- Random effects
- Constant covariates
- PreDE outputs (`@preDifferentialEquation`)
- Helper functions (`@helpers`)
- Model functions from learned parameter blocks (e.g., neural network, soft tree, or spline functions)

This resolution order mirrors that of `@preDifferentialEquation`, ensuring a consistent namespace across the pre-integration model components.

## Example: Basic Initial State Builder

The `@initialDE` macro can be used independently for testing. Here, the builder function is constructed and called directly with explicit model inputs.

```julia
using NoLimits
using ComponentArrays

init = @initialDE begin
    x1 = 1.0
    x2 = a + b
end

builder = get_initialde_builder(init, [:x1, :x2])

theta = ComponentArray(a = 2.0, b = 3.0)
eta = ComponentArray()
const_covariates = NamedTuple()
model_funs = NamedTuple()
helpers = NamedTuple()
prede = NamedTuple()

u0 = builder(theta, eta, const_covariates, model_funs, helpers, prede)
```

## Example: Using Helpers, preDE, and Model Functions

Initial conditions can incorporate helper functions, preDE outputs, and learned model functions. This is useful when the starting state depends on a nonlinear transformation of covariates or parameters.

```julia
using NoLimits
using ComponentArrays

init = @initialDE begin
    x1 = helper(a) + preA
    x2 = NN1([c1], z)[1]
end

helpers = @helpers begin
    helper(u) = u + 1.0
end

builder = get_initialde_builder(init, [:x1, :x2])

theta = ComponentArray(a = 2.0, z = 3.0)
eta = ComponentArray()
const_covariates = (c1 = 4.0,)
model_funs = (NN1 = (x, z) -> [x[1] + z],)
prede = (preA = 5.0,)

u0 = builder(theta, eta, const_covariates, model_funs, helpers, prede)
```

## Example: Nonlinear Mixed-Effects ODE Initialization

In a complete model, initial conditions often depend on individual-level derived quantities. Here, the initial state is set to a preDE value that combines a fixed effect, a covariate, and a random effect -- so each individual begins integration from a distinct starting point.

```julia
using NoLimits
using Distributions

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.3)
        b = RealNumber(0.1)
        sigma = RealNumber(0.2, scale=:log)
    end

    @covariates begin
        t = Covariate()
        age = ConstantCovariate(; constant_on=:ID)
    end

    @randomEffects begin
        eta = RandomEffect(Gamma(2.0, 1.0); column=:ID)
    end

    @preDifferentialEquation begin
        pre = exp(a + 0.01 * age + eta^2)
    end

    @DifferentialEquation begin
        D(x1) ~ -b * x1 + pre
    end

    @initialDE begin
        x1 = pre
    end

    @formulas begin
        y ~ Gamma(log1p(x1(t)^2 + eta^2) + 1e-6, sigma)
    end
end
```

## Static Initial-State Variant

For performance-sensitive workflows using static arrays, the builder can return an `SVector` instead of a standard mutable vector. This is enabled by passing `static=true`.

```julia
builder_static = get_initialde_builder(init, [:x1, :x2]; static=true)
```
