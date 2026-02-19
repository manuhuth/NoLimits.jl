# `@covariates`

Covariates encode subject-level attributes, time-varying measurements, and externally observed signals that inform model dynamics and observation distributions. The `@covariates` block declares all covariates used by formulas, random-effect distributions, and differential equation components.

NoLimits distinguishes three covariate classes, each with distinct semantics for how values are resolved at evaluation time:

- **Varying covariates** (`Covariate`, `CovariateVector`) -- values that change across observations and are accessed row-by-row from the data.
- **Constant covariates** (`ConstantCovariate`, `ConstantCovariateVector`) -- values that remain fixed within a grouping level (e.g., per subject or per site) and are extracted once per group.
- **Dynamic covariates** (`DynamicCovariate`, `DynamicCovariateVector`) -- time-series values that are converted into continuous interpolating functions, enabling evaluation at arbitrary time points within differential equations and formulas.

## Core Syntax

Inside `@covariates`, each line is an assignment that maps a name to a covariate constructor. No other statement forms are permitted.

```julia
using DataInterpolations

cov = @covariates begin
    t = Covariate()
    x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
    w = DynamicCovariate(; interpolation=LinearInterpolation)
end
```

## Constructor Reference

The following constructor forms are available within `@covariates`:

- `name = Covariate()`
- `name = CovariateVector([:col1, :col2, ...])`
- `name = ConstantCovariate(; constant_on=...)`
- `name = ConstantCovariateVector([:col1, :col2, ...]; constant_on=...)`
- `name = DynamicCovariate(; interpolation=...)`
- `name = DynamicCovariateVector([:col1, :col2, ...]; interpolations=[...])`

Note the following macro-level conventions:

- For scalar constructors (`Covariate`, `ConstantCovariate`, `DynamicCovariate`), the left-hand side name determines the DataFrame column that will be read. Passing an explicit column name to these constructors is not supported.
- For vector constructors, the column names are specified in the first positional argument.
- `Covariate` and `CovariateVector` accept no keyword arguments.

## Supported Interpolation Types

Dynamic covariates construct interpolating functions from observed time--value pairs using [DataInterpolations.jl](https://docs.sciml.ai/DataInterpolations/stable/). The following interpolation methods are currently supported:

- `ConstantInterpolation`
- `SmoothedConstantInterpolation`
- `LinearInterpolation`
- `QuadraticInterpolation`
- `LagrangeInterpolation`
- `QuadraticSpline`
- `CubicSpline`
- `AkimaInterpolation`

When no interpolation is specified, `DynamicCovariate` defaults to `LinearInterpolation`. To use a non-default method, ensure that `DataInterpolations` is loaded in the model-definition environment (e.g., `using DataInterpolations`).

## Example: Mixed Covariate Specification

The following example demonstrates all three covariate classes used together, including scalar and vector variants.

```julia
using NoLimits
using DataInterpolations

cov = @covariates begin
    t = Covariate()
    z = CovariateVector([:z1, :z2, :z3])

    x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
    center = ConstantCovariate(; constant_on=:SITE)

    w1 = DynamicCovariate(; interpolation=CubicSpline)
    w2 = DynamicCovariateVector(
        [:u1, :u2];
        interpolations=[LinearInterpolation, AkimaInterpolation],
    )
end
```

## Example: Covariates in a Nonlinear Model

Covariates can appear in random-effect distributions and observation formulas, enabling covariate-dependent distributional parameters. In the model below, subject-level attributes modulate both the random-effect distribution and the outcome probability.

```julia
using NoLimits
using Distributions
using DataInterpolations

model = @Model begin
    @fixedEffects begin
        b0 = RealNumber(0.2)
        sη = RealNumber(0.5, scale=:log)
    end

    @covariates begin
        t = Covariate()
        x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
        w = DynamicCovariate(; interpolation=QuadraticSpline)
    end

    @randomEffects begin
        η = RandomEffect(LogNormal(b0 + 0.01 * x.Age^2 + log1p(abs(x.BMI)), sη); column=:ID)
    end

    @formulas begin
        p = 1 / (1 + exp(-(0.3 + tanh(η) + 0.02 * w(t)^2 + 0.001 * x.BMI^2)))
        y ~ Bernoulli(p)
    end
end
```

## `constant_on` and Random-Effect Grouping

Constant covariates that appear inside random-effect distributions must be invariant within the corresponding grouping level. The following rules apply:

- If there is exactly one random-effect grouping column, a missing `constant_on` keyword is automatically set to that column.
- If there are multiple random-effect grouping columns, `constant_on` must be specified explicitly.
- A constant covariate used in a random-effect distribution must declare `constant_on` for (at least) that random effect's grouping column.

## Covariate Rules in Differential Equations

Covariates used within `@DifferentialEquation` are subject to restrictions that reflect the continuous-time nature of ODE integration:

- **Constant covariates** may appear as ordinary variables (they do not depend on time).
- **Dynamic covariates** must be called as functions of time, e.g., `w(t)`. Referencing a dynamic covariate without `(t)` is rejected.
- **Varying covariates** (`Covariate`, `CovariateVector`) are not permitted in differential equations, as they lack a continuous-time representation. Use `DynamicCovariate` with an appropriate interpolation instead.
- Calling a constant covariate as a function (e.g., `x(t)`) is rejected.

## Data-Model Validation

When a `DataModel` is constructed, the covariate declarations are validated against the supplied DataFrame:

- The configured `time_col` must be declared as `Covariate()` or `DynamicCovariate()` in the `@covariates` block.
- All declared covariate columns must be present in the DataFrame and free of missing values.
- Constant covariates are checked for consistency: values must be identical within each level of the `primary_id` column and within each declared `constant_on` group.
