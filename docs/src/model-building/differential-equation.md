# `@DifferentialEquation`

Many longitudinal processes are naturally described by systems of ordinary differential equations (ODEs). The `@DifferentialEquation` block specifies the dynamics of such systems, defining how model states evolve over time and optionally declaring derived signals that depend on time and the current state.

Within `@Model`, this block is optional. When present, the companion `@initialDE` block is required to supply initial conditions.

## Core Syntax

Statements inside `@DifferentialEquation` take one of two forms:

- **State dynamics:** `D(state) ~ rhs_expression` -- defines the time derivative of a state variable.
- **Derived signals:** `signal(t) = expression` (or `signal(\xi) = expression`) -- defines an auxiliary quantity computed within the ODE, useful for intermediate calculations that appear in multiple state equations.

```julia
de = @DifferentialEquation begin
    s(t) = sin(t)
    D(x1) ~ -a * x1 + s(t)
    D(x2) ~ a * x1 - b * x2
end
```

The ordering of `D(state)` declarations determines the state indexing in the solver's `u` and `du` vectors.

## Parsing and Validation Rules

The following constraints are enforced at parse time:

- The block must be wrapped in `begin ... end`.
- Only `D(state) ~ expr` and `signal(t) = expr` / `signal(\xi) = expr` forms are permitted.
- State names and signal names must be unique across the block.
- `D(...)` must contain exactly one symbolic state name.
- Derived signals must always be referenced as function calls (`s(t)` or `s(\xi)`), never as bare symbols (`s`).
- Unresolved symbols or functions in DE expressions raise an error during compilation.

## Symbol and Function Resolution

NoLimits resolves symbols appearing in DE expressions from the following sources:

**Variable-like symbols** are drawn from:
- Fixed effects
- Random effects
- Constant covariates
- PreDE outputs (`@preDifferentialEquation`)

**Function-like calls** are drawn from:
- Dynamic covariates (e.g., `w(t)`)
- Helper functions (`@helpers`)
- Model functions from learned parameter blocks (e.g., neural network, soft tree, or spline functions)

Note that time-varying (non-dynamic) covariates are not available inside the ODE, since their values are defined only at observation times. Use dynamic covariates with an appropriate interpolation method for continuous-time access.

## Example: Standalone DE Compilation and Evaluation

The `@DifferentialEquation` macro can be used outside of `@Model` for testing or prototyping. The following demonstrates direct compilation and evaluation of a single-state ODE.

```julia
using NoLimits
using ComponentArrays

de = @DifferentialEquation begin
    s(t) = sin(t)
    D(x1) ~ -a * x1^2 + s(t) + w(t) + pre
end

ctx = (
    fixed_effects = ComponentArray(a = 0.3),
    random_effects = ComponentArray(),
    constant_covariates = NamedTuple(),
    varying_covariates = (w = t -> 0.2 * t,),
    helpers = NamedTuple(),
    model_funs = NamedTuple(),
    preDE = (pre = 0.5,),
)

pc = get_de_compiler(de)(ctx)
u = [1.0]
du = similar(u)
get_de_f!(de)(du, u, pc, 0.4)
```

## Example: Nonlinear Mixed-Effects ODE Model

This example illustrates a complete model in which random effects enter both the ODE dynamics and the observation distribution. The derived signal `s(t)` encapsulates a nonlinear transformation of a dynamic covariate and a preDE quantity, keeping the state equation concise.

```julia
using NoLimits
using Distributions

model = @Model begin
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end

    @fixedEffects begin
        a = RealNumber(0.4)
        b = RealNumber(0.2)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
        w = DynamicCovariate()
    end

    @randomEffects begin
        eta = RandomEffect(TDist(6.0); column=:ID)
    end

    @preDifferentialEquation begin
        pre = exp(a + eta^2)
    end

    @DifferentialEquation begin
        s(t) = sat(w(t) + pre)
        D(x1) ~ -(b + tanh(eta)) * x1 + s(t)^2
    end

    @initialDE begin
        x1 = pre
    end

    @formulas begin
        y ~ Normal(log1p(abs(x1(t)) + eta^2), sigma)
    end
end
```

## Example: Learned Right-Hand Side with Neural Networks and Soft Trees

When the functional form of the ODE right-hand side is unknown or difficult to specify mechanistically, learned components such as neural networks and soft decision trees can be embedded directly into the dynamics. In this example, the transfer and elimination rates of a two-state system are parameterized by neural networks and soft trees, with individual-level random effects on the learned parameters.

```julia
using NoLimits
using Distributions
using Lux
using LinearAlgebra

chain_A1 = Chain(Dense(1, 4, tanh), Dense(4, 1))
chain_A2 = Chain(Dense(1, 4, tanh), Dense(4, 1))
depth_st = 2

model = @Model begin
    @helpers begin
        softplus(u) = u > 20 ? u : log1p(exp(u))
    end

    @covariates begin
        t = Covariate()
        d = ConstantCovariate(; constant_on=:ID)
    end

    @fixedEffects begin
        sigma = RealNumber(0.3, scale=:log)

        zA1 = NNParameters(chain_A1; function_name=:NNA1, calculate_se=false)
        zA2 = NNParameters(chain_A2; function_name=:NNA2, calculate_se=false)
        gC1 = SoftTreeParameters(1, depth_st; function_name=:STC1, calculate_se=false)
        gC2 = SoftTreeParameters(1, depth_st; function_name=:STC2, calculate_se=false)
    end

    @randomEffects begin
        etaA1 = RandomEffect(MvNormal(zA1, Diagonal(ones(length(zA1)))); column=:ID)
        etaA2 = RandomEffect(MvNormal(zA2, Diagonal(ones(length(zA2)))); column=:ID)
        etaC1 = RandomEffect(MvNormal(gC1, Diagonal(ones(length(gC1)))); column=:ID)
        etaC2 = RandomEffect(MvNormal(gC2, Diagonal(ones(length(gC2)))); column=:ID)
    end

    @DifferentialEquation begin
        a_A(t) = softplus(depot)
        x_C(t) = softplus(center)

        fA1(t) = softplus(NNA1([t / 24], etaA1)[1])
        fA2(t) = softplus(NNA2([a_A(t)], etaA2)[1])
        fC1(t) = -softplus(STC1([x_C(t)], etaC1)[1])
        fC2(t) = softplus(STC2([t / 24], etaC2)[1])

        D(depot) ~ -d * fA1(t) - fA2(t)
        D(center) ~ d * fA1(t) + fA2(t) + fC1(t) + d * fC2(t)
    end

    @initialDE begin
        depot = d
        center = 0.0
    end

    @formulas begin
        y ~ LogNormal(center(t), sigma)
    end
end
```

## DE Accessors for States and Signals

After ODE integration, accessor functions provide callable trajectories for each state and derived signal. These are the same accessors that formula evaluation uses internally when outcomes reference `x1(t)` or `s(t)`.

```julia
if @isdefined(sol) && @isdefined(compiled_de_context)
    accessors = get_de_accessors_builder(model.de.de)(sol, compiled_de_context)

    x1_at_12 = accessors.x1(1.2)
    s_at_12 = accessors.s(1.2)
end
```

## Related APIs

The following functions provide lower-level access to the DE subsystem:

- `get_de_states(de)` and `get_de_signals(de)` -- enumerate declared states and signals.
- `get_de_compiler(de)`, `get_de_f!(de)`, `get_de_f(de)` -- access the compiled ODE function and its context builder.
- `get_de_accessors_builder(de)` -- returns a function that constructs solution accessors from an ODE solution.
- `build_de_params(de, \theta; ...)` -- assembles ODE parameters with tunable modes (`:theta`, `:eta`, `:both`).
