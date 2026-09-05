# Model Building API

Macros, parameter types, covariates, random effects, and the `Model` struct itself.

!!! note "Some entries need an optional dependency"
    Parts of this API are implemented in package extensions and become available only once the
    corresponding package is loaded. Calling one without its package raises an error naming what
    to install. See [Optional Dependencies](../installation.md#Optional-Dependencies).

## Macros

```@docs
@Model
@helpers
@fixedEffects
@covariates
@randomEffects
@preDifferentialEquation
@DifferentialEquation
@initialDE
@formulas
```

## Parameter Types

```@docs
RealNumber
RealVector
RealPSDMatrix
RealLiePSDMatrix
RealDiagonalMatrix
ProbabilityVector
DiscreteTransitionMatrix
ContinuousTransitionMatrix
NNParameters
FFNNParameters
NPFParameter
SoftTreeParameters
SplineParameters
Priorless
```

## Covariate Types

```@docs
Covariate
CovariateVector
ConstantCovariate
ConstantCovariateVector
DynamicCovariate
DynamicCovariateVector
```

## Random Effects

```@docs
RandomEffect
```

## Model Struct and Solver Configuration

```@docs
Model
ODESolverConfig
ClosedFormPlan
set_solver_config
get_model_funs
get_helper_funs
get_solver_config
get_source
```

## Model Component Structs

These structs hold the parsed, compiled form of each model block. They are constructed
automatically by the block macros and stored inside `Model`.

```@docs
FixedEffects
Covariates
finalize_covariates
RandomEffects
PreDifferentialEquation
DifferentialEquation
InitialDE
get_initialde_builder
Formulas
get_formulas_builders
```

## Model Display

```@docs
show_equations
```

## Where to go next

- [Model Building guide](../model-building/index.md) - the prose behind these names.
- [Data Binding](data.md) - the next API page.
- [API overview](../api.md) - all seven reference pages.
