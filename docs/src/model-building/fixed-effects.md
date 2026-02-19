# `@fixedEffects`

Fixed effects are the parameters shared across all individuals in a dataset. The `@fixedEffects` block declares these parameters together with their initial values, scales, bounds, priors, and standard-error flags. The resulting objects are referenced throughout the model: in formulas, differential equations, pre-ODE computations, and random-effect distributions.

In nonlinear mixed-effects (NLME) models, fixed effects represent population-level quantities. In fixed-effects-only models, they constitute the full set of estimands.

## Syntax

Each statement in `@fixedEffects` is an assignment whose left-hand side is a symbol and whose right-hand side is a parameter constructor call. The parameter name is injected automatically from the left-hand side unless an explicit `name=` keyword is provided.

```julia
fe = @fixedEffects begin
    ka = RealNumber(1.0, scale=:log)
    beta = RealVector([0.2, -0.1], scale=[:identity, :identity])
end
```

## Block Rules

The following rules are enforced at macro-expansion time:

- The block must use `begin ... end` syntax.
- Only assignment statements are permitted.
- The left-hand side of each assignment must be a single symbol.
- The right-hand side must be one of the supported parameter constructor calls listed below.
- An empty block is valid and produces an empty fixed-effects object.

## Supported Parameter Types

NoLimits provides parameter types for scalars, vectors, structured matrices, and learned function approximators. Each type controls how values are stored, transformed during optimization, and optionally regularized via priors.

| Constructor | Purpose |
|---|---|
| `RealNumber(value; scale, lower, upper, prior, calculate_se)` | Scalar parameter (`:identity` or `:log` scale) |
| `RealVector(value; scale, lower, upper, prior, calculate_se)` | Vector parameter with per-element scale |
| `RealPSDMatrix(value; scale, prior, calculate_se)` | Symmetric positive semi-definite matrix (`:cholesky` or `:expm`) |
| `RealDiagonalMatrix(value; scale, prior, calculate_se)` | Diagonal matrix (`:log` scale on diagonal entries) |
| `NNParameters(chain; function_name, seed, prior, calculate_se)` | Lux neural network weights |
| `SoftTreeParameters(input_dim, depth; function_name, n_output, seed, prior, calculate_se)` | Soft decision tree parameters |
| `SplineParameters(knots; function_name, degree, prior, calculate_se)` | B-spline coefficients |
| `NPFParameter(n_input, n_layers; seed, init, prior, calculate_se)` | Normalizing planar flow parameters |

## Example: Classical Parameter Blocks

The most common use case combines scalar, vector, and matrix parameters to define a population model. Priors can be assigned to any parameter for Bayesian estimation or regularization.

```julia
using NoLimits
using Distributions
using LinearAlgebra

fe = @fixedEffects begin
    ka = RealNumber(1.0, scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
    beta = RealVector(
        [0.2, -0.1],
        scale=[:identity, :identity],
        lower=[-Inf, -Inf],
        upper=[Inf, Inf],
        prior=MvNormal(zeros(2), Matrix(I, 2, 2)),
        calculate_se=true,
    )
    Omega = RealPSDMatrix(Matrix(I, 2, 2), scale=:cholesky, prior=Wishart(3, Matrix(I, 2, 2)))
    D = RealDiagonalMatrix([0.3, 0.2], calculate_se=true)
end
```

## Transforms, Bounds, and Priors

Internally, `@fixedEffects` constructs both transformed and untransformed parameter representations, along with associated bounds, priors, and standard-error masks. Accessor functions provide a uniform interface to these components.

```julia
theta_u = get_θ0_untransformed(fe)
theta_t = get_θ0_transformed(fe)
theta_u_rt = get_inverse_transform(fe)(theta_t)

names = get_names(fe)
flat_names = get_flat_names(fe)
se_names = get_se_names(fe)
priors = get_priors(fe)
lp = logprior(fe, theta_u)
```

The behaviour of each scale option is summarized below:

- **`:log` scale** applies an elementwise log transform and is supported for `RealNumber`, `RealVector`, and `RealDiagonalMatrix`. For inherently positive quantities such as standard deviations, `:log` enforces positivity in transformed space; an explicit `lower` bound is therefore optional.
- **`:cholesky`** (for `RealPSDMatrix`) parameterizes the matrix via its Cholesky factor with log-transformed diagonal entries.
- **`:expm`** (for `RealPSDMatrix`) parameterizes the matrix via matrix logarithm/exponential, storing only the upper-triangular elements.

## Example: Learned Function Approximators

Neural networks, soft decision trees, and B-splines can be declared as fixed effects and are automatically exposed as callable model functions. This enables flexible, data-driven components within an otherwise parametric model specification.

```julia
using NoLimits
using Lux

chain = Chain(Dense(2, 4, tanh), Dense(4, 1))
knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]

fe_learned = @fixedEffects begin
    z_nn = NNParameters(chain; function_name=:NN1, calculate_se=false)
    z_st = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
    z_sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
end

model_funs = get_model_funs(fe_learned)
params = get_params(fe_learned)

y_nn = model_funs.NN1([0.4, 0.6], params.z_nn.value)[1]
y_st = model_funs.ST1([0.4, 0.6], params.z_st.value)[1]
y_sp = model_funs.SP1(0.5, params.z_sp.value)
```

## Example: Normalizing Flows for Flexible Random-Effect Distributions

Normalizing planar flows (`NPFParameter`) allow the random-effect distribution to depart from standard parametric families. The flow parameters are declared as fixed effects and referenced inside `@randomEffects` via `NormalizingPlanarFlow`. The corresponding model function is registered automatically.

```julia
using NoLimits
using Distributions

model = @Model begin
    @fixedEffects begin
        psi = NPFParameter(1, 3, seed=1, calculate_se=false)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(NormalizingPlanarFlow(psi); column=:ID)
    end

    @formulas begin
        y ~ Normal(log1p(eta^2), sigma)
    end
end
```
