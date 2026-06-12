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

Only macro blocks are allowed inside `@Model` - no bare Julia statements.

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

## Extending an Existing Model

`@Model` has a second form that builds a new model from an existing one, overriding only
the blocks you supply:

```julia
new_model = @Model base_model begin
    # only the blocks you want to change
end
```

Every model built by `@Model` stores the source expression of each of its blocks (see
[`get_source`](@ref)). The extension form reuses those stored blocks for anything you do
not mention, so you only restate what actually changes between the two models.

### Merge semantics

Within an overridden block, entries are merged **by name** - the left-hand side of each
declaration:

- An entry whose name matches one already in the base block **replaces** it, keeping its
  original position.
- A new name is **appended** to the end of the block.
- Assigning a name to `nothing` (e.g. `sigma_k = nothing`) **removes** it from the block.

A block you do not mention at all is inherited verbatim. The merged result is rebuilt
through the full single-argument `@Model` pipeline, so **all normal validation applies**
(required `@formulas`, at least one parameter, DE/initial-DE pairing, covariate-usage
rules, etc.).

### Example: Gaussian → normalizing-flow random effect

Starting from a Gaussian baseline:

```julia
model_gauss = @Model begin
    @covariates begin
        age = Covariate()
    end
    @fixedEffects begin
        L_inf_pop = RealNumber(705.0, scale = :log)
        k_pop     = RealNumber(0.104, scale = :log)
        sigma_L   = RealNumber(0.10,  scale = :log)
        sigma_k   = RealNumber(0.40,  scale = :log)
        sigma_y   = RealNumber(15.0,  scale = :log)
    end
    @randomEffects begin
        eta_k = RandomEffect(Normal(0.0, sigma_k); column = :fishid)
        eta_L = RandomEffect(Normal(0.0, sigma_L); column = :fishid)
    end
    @formulas begin
        length ~ Normal(
            exp(log(L_inf_pop) + eta_L) *
                (1 - exp(-exp(log(k_pop) + eta_k) * age)),
            sigma_y)
    end
end
```

the flow variant changes only three blocks - `@covariates`, `@fixedEffects` keeps all
parameters except `sigma_k` (removed) and adds `psi`, `eta_k` becomes a flow while `eta_L`
is inherited, and the `length` formula switches `eta_k` to `eta_k[1]`:

```julia
model_flow = @Model model_gauss begin
    @fixedEffects begin
        sigma_k = nothing                                      # drop the Gaussian scale
        psi     = NPFParameter(1, 4; seed = 42, calculate_se = false)
    end
    @randomEffects begin
        eta_k = RandomEffect(NormalizingPlanarFlow(psi); column = :fishid)
        # eta_L is inherited unchanged
    end
    @formulas begin
        length ~ Normal(
            exp(log(L_inf_pop) + eta_L) *
                (1 - exp(-exp(log(k_pop) + eta_k[1]) * age)),
            sigma_y)
    end
end
```

`model_flow` ends up with fixed effects `L_inf_pop, k_pop, sigma_L, sigma_y, psi` and
random effects `eta_k` (flow) and `eta_L` (inherited Gaussian). Extension models can
themselves be extended - the new model again stores its own source.

!!! warning "Blocks must be self-contained"
    Inherited and overriding blocks are re-evaluated in the module where the extension
    call occurs. Any variable they reference (for example a `Lux.Chain`, a `knots` vector,
    or a precomputed matrix) must be available as a global in that module. Local variables
    captured at the base model's original call site are **not** carried over - declare
    such values as globals, or re-declare them before the extension call.

## Model Summary

After construction, use `NoLimits.summarize(model)` to inspect the declared structure. It returns a `ModelSummary` describing the model's blocks - counts and lists of fixed effects, random effects, covariates, deterministic formulas, outcome distributions, and differential-equation states/signals - which is pretty-printed via `Base.show`.

```julia
model_summary = NoLimits.summarize(model)
model_summary
```

## Runtime Evaluation Helpers

`@Model` wires internal evaluation functions used during fitting and simulation - `calculate_prede` (pre-DE expressions), `calculate_initial_state` (ODE initial conditions), `calculate_formulas_all` (all formula nodes), and `calculate_formulas_obs` (observation-node formulas only). Full signatures are in the [API reference](../api.md).

When formulas reference ODE states or signals, formula evaluation requires DE solution accessors from `get_de_accessors_builder(...)`.
