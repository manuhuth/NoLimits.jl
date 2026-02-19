# `@helpers`

The `@helpers` block defines reusable helper functions within the `@Model` DSL. These are useful when the same nonlinear transformation appears in multiple places -- for example, a saturation function used in both the structural model and the observation model.

Helper functions are parsed at macro-expansion time and returned as a `NamedTuple` of callables.

## Syntax

Both short-form and long-form function definitions are supported:

```julia
helpers = @helpers begin
    sat(u) = u / (1 + abs(u))

    function softplus(u::Float64)
        return log1p(exp(u))
    end
end
```

## Validation Rules

- Only function definitions are allowed inside `@helpers`.
- Helper arguments must be simple symbols, optionally typed (e.g., `x` or `x::Float64`).
- Duplicate helper names within the same block are rejected.
- An empty block is valid and yields `NamedTuple()`.
- Mutating helper patterns (e.g., in-place array operations) trigger a warning, since reverse-mode automatic differentiation backends such as Zygote require non-mutating code.

## Example: Standalone Helper Block

```julia
using NoLimits
using LinearAlgebra

helpers = @helpers begin
    clamp01(u) = max(0.0, min(1.0, u))
    softplus(u) = log1p(exp(u))
    dotp(a, b) = dot(a, b)
end

helpers.clamp01(-1.0)   # 0.0
helpers.softplus(0.0)   # log(2)
helpers.dotp([1.0, 2.0], [3.0, 4.0])  # 11.0
```

## Example: Helper in Formulas

```julia
using NoLimits
using Distributions

model = @Model begin
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end

    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.5, scale=:log)
    end

    @covariates begin
        t = Covariate()
        x = ConstantCovariateVector([:Age])
    end

    @randomEffects begin
        η = RandomEffect(TDist(6.0); column=:id)
    end

    @formulas begin
        μ = sat(a + η^2 + x.Age)
        y ~ Laplace(μ, σ)
    end
end
```

## Example: Helper in Random-Effect Distribution

Helpers can also appear inside random-effect distribution expressions, enabling nonlinear transformations of fixed effects before they enter the distribution:

```julia
using NoLimits
using Distributions

model = @Model begin
    @helpers begin
        softplus(u) = log1p(exp(u))
    end

    @fixedEffects begin
        β = RealNumber(0.2)
        ση = RealNumber(0.7, scale=:log)
        σy = RealNumber(0.5, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        η = RandomEffect(LogNormal(softplus(β), ση); column=:ID)
    end

    @formulas begin
        y ~ Gamma(log1p(η^2) + 1e-6, σy)
    end
end
```

## Note on Automatic Differentiation

Helpers that mutate arrays in place are detected and produce a warning. When reverse-mode AD compatibility is needed (e.g., with Zygote), prefer non-mutating implementations.
