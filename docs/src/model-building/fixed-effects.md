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

For standard-error eligibility, `RealNumber` and `RealVector` default to `calculate_se=true`. All other fixed-effect block types default to `calculate_se=false`.

## Supported Parameter Types

NoLimits provides parameter types for scalars, vectors, structured matrices, and learned function approximators. Each type controls how values are stored, transformed during optimization, and optionally regularized via priors. The overview below lists the purpose and value kind of each type; see the [Parameter Types](../api.md#Parameter-Types) section of the API reference for full constructor signatures, keyword arguments, and defaults.

| Type | Purpose | Value kind |
|---|---|---|
| `RealNumber` | Scalar parameter | scalar `Real` |
| `RealVector` | Vector parameter with per-element scale | `Vector{<:Real}` |
| `RealPSDMatrix` | Symmetric positive semi-definite matrix | square `Matrix` |
| `RealLiePSDMatrix` | Symmetric positive-definite matrix with eigenvalue bounds and optional block structure | square `Matrix` |
| `RealDiagonalMatrix` | Diagonal matrix | diagonal entries `Vector` |
| `ProbabilityVector` | Probability simplex of length k≥2 | `Vector` summing to 1 |
| `DiscreteTransitionMatrix` | n×n row-stochastic matrix (n≥2) | row-stochastic `Matrix` |
| `ContinuousTransitionMatrix` | n×n rate matrix / Q-matrix (n≥2) | rate `Matrix` |
| `NNParameters` | Lux `Chain` or SimpleChains `SimpleChain` weights | flattened network params |
| `SoftTreeParameters` | Soft decision tree parameters | flattened tree params |
| `SplineParameters` | B-spline coefficients | coefficient `Vector` |
| `NPFParameter` | Normalizing planar flow parameters | flattened flow params |

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

The behavior of each scale option is summarized below:

- **`:log` scale** applies an elementwise log transform and is supported for `RealNumber`, `RealVector`, and `RealDiagonalMatrix`. For inherently positive quantities such as standard deviations, `:log` enforces positivity in transformed space; an explicit `lower` bound is therefore optional.
- **`:logit` scale** applies the logit transform (`log(x/(1-x))`, clamped to `[-20, 20]`) and is supported for `RealNumber` and `RealVector`. Use this for parameters that must lie in `(0, 1)`, such as probabilities. The inverse is the sigmoid function. The initial value must be strictly between 0 and 1; the constructor errors otherwise. Bounds are enforced implicitly via clamping - no explicit `lower`/`upper` are needed.
- **`:cholesky`** (for `RealPSDMatrix`) parameterizes the matrix via its Cholesky factor with log-transformed diagonal entries.
- **`:expm`** (for `RealPSDMatrix`) parameterizes the matrix via matrix logarithm/exponential, storing only the upper-triangular elements.
- **`:lie`** (for `RealLiePSDMatrix`) parameterizes the matrix via its eigendecomposition Σ = exp(A) diag(exp λ) exp(A)ᵀ, with log-eigenvalues λ and an antisymmetric A built from rotation coefficients. Because the eigenvalues are exposed directly, box bounds on the spectrum are supported (`eigenvalue_lower`/`eigenvalue_upper`). Setting `blocks` fixes cross-block rotation coefficients to zero, enforcing a block-diagonal covariance without constrained optimization; `fix_eigenvalues` pins individual eigenvalues. Fixed coefficients are dropped from the optimizer.
- **`:stickbreak`** (for `ProbabilityVector`) maps a k-probability simplex to k-1 unconstrained reals via the logistic stick-breaking transform. Each element νᵢ = pᵢ/(1-Σⱼ<ᵢ pⱼ) is passed through logit. The last probability is determined and not stored. Silently normalizes the initial value if the sum is within 1e-6 of 1.
- **`:stickbreakrows`** (for `DiscreteTransitionMatrix`) applies the stick-breaking transform independently to each row of an n×n row-stochastic matrix, yielding n*(n-1) unconstrained parameters.
- **`:lograterows`** (for `ContinuousTransitionMatrix`) maps each off-diagonal entry of a rate matrix to its logarithm, yielding n*(n-1) unconstrained reals. The diagonal is always recomputed as minus the row sum and is never stored as a free parameter. Initial off-diagonal values must be non-negative.

For `RealVector`, scales can be mixed per element by passing a `Vector{Symbol}`, e.g. `scale=[:logit, :log, :identity]`. Mixed vectors use an elementwise dispatch that applies each element's transform independently.

## Choosing a Covariance Parameterization

`RealPSDMatrix` (`:cholesky`, `:expm`) and `RealLiePSDMatrix` (`:lie`) all represent a full covariance, but they differ in cost, conditioning, and what structure they can enforce. The guidance below reflects both benchmarking and the Lie parameterization's design intent.

- **Default: `RealPSDMatrix(:cholesky)`.** For an ordinary random-effect covariance with no special requirements, log-Cholesky is the cheapest map and the best-conditioned under optimization. `:expm` is a reasonable alternative with no clear advantage.
- **`RealLiePSDMatrix` without bounds or blocks is not recommended for a full matrix.** It costs more per gradient than Cholesky and conditions worse as the dimension grows, because a full rotation's coefficients are not uniquely identified.
- **Use `RealLiePSDMatrix` with eigenvalue bounds when the covariance drives an ODE.** When random effects enter ODE initial conditions or rates, an unconstrained Cholesky/matrix-log start can produce an ill-conditioned Σ that makes the solver fail at the first optimization step. Bounding the spectrum with `eigenvalue_lower`/`eigenvalue_upper` keeps every candidate matrix well-conditioned. This is the parameterization's primary motivation (Stapor, 2020).
- **Use `blocks` for known block structure.** When some random effects are known not to correlate, `blocks` enforces the zeros exactly and removes the cross-block parameters from the optimizer, which reduces the parameter count and speeds up convergence versus fitting a full matrix.
- **Use `RealDiagonalMatrix` for a diagonal covariance.** It is simpler and cheaper than any full parameterization.

```julia
# Full covariance driving an ODE: bound the eigenvalues for solver robustness.
Omega = RealLiePSDMatrix(Matrix(I, 3, 3), eigenvalue_lower=1e-3, eigenvalue_upper=1e3)

# Block-diagonal covariance: random effects 1-2 correlate, 3 is independent.
Omega_block = RealLiePSDMatrix(Matrix(I, 3, 3), blocks=[1, 1, 2])
```

## Example: Learned Function Approximators

Neural networks, soft decision trees, and B-splines can be declared as fixed effects and are automatically exposed as callable model functions. This enables flexible, data-driven components within an otherwise parametric model specification. `NNParameters` accepts either a Lux `Chain` or a SimpleChains `SimpleChain` - see [Function Approximators](universal-function-approximators.md) for the trade-offs (SimpleChains is faster and lower-allocation under ForwardDiff).

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

## Example: Constrained Stochastic Matrices

Three dedicated parameter types handle the structural constraints that arise in Markov models (both hidden and observed-state) and other latent-variable models - `ProbabilityVector` (`:stickbreak`), `DiscreteTransitionMatrix` (`:stickbreakrows`), and `ContinuousTransitionMatrix` (`:lograterows`). Their constraints, transforms, and free-parameter counts are described in the scale list above.

All three types are AD-compatible and can be used anywhere in a model formula where the corresponding matrix or vector is expected. The values they provide in formulas are plain Julia arrays (`Vector` or `Matrix`), enabling direct indexing and arithmetic.

The first example uses `ProbabilityVector` for the initial state distribution and `DiscreteTransitionMatrix` for the row-stochastic transition matrix of a discrete-time two-state HMM:

```julia
using NoLimits
using Distributions

model_disc = @Model begin
    @fixedEffects begin
        pi0   = ProbabilityVector([0.6, 0.4])
        P     = DiscreteTransitionMatrix([0.9 0.1; 0.2 0.8])
        mu1   = RealNumber(0.0)
        mu2   = RealNumber(2.0)
        sigma = RealNumber(0.5, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @formulas begin
        y ~ DiscreteTimeDiscreteStatesHMM(
            P,
            (Normal(mu1, sigma), Normal(mu2, sigma)),
            Categorical(pi0),
        )
    end
end
```

For continuous-time transitions, `ContinuousTransitionMatrix` declares the full rate matrix. Off-diagonal entries must be non-negative; the diagonal is always derived as minus the row sum and never appears among the free parameters:

```julia
using NoLimits
using Distributions

model_cont = @Model begin
    @fixedEffects begin
        Q     = ContinuousTransitionMatrix([-0.2 0.2; 0.3 -0.3])
        mu1   = RealNumber(0.0)
        mu2   = RealNumber(2.0)
        sigma = RealNumber(0.5, scale=:log)
    end

    @covariates begin
        t       = Covariate()
        delta_t = Covariate()
    end

    @formulas begin
        y ~ ContinuousTimeDiscreteStatesHMM(
            Q,
            (Normal(mu1, sigma), Normal(mu2, sigma)),
            Categorical([0.5, 0.5]),
            delta_t,
        )
    end
end
```

## Example: Normalizing Flows for Flexible Random-Effect Distributions

Normalizing planar flows (`NPFParameter`) allow the random-effect distribution to depart from standard parametric families. The flow parameters are declared as fixed effects and referenced inside `@randomEffects` via `NormalizingPlanarFlow`. The corresponding model function is registered automatically.

By default the flow uses a standard `MvNormal` base distribution. A custom base distribution can be passed via the `base_dist` keyword - any continuous multivariate distribution from Distributions.jl is supported (e.g. `MvTDist` for heavier tails, or a `MvNormal` with a non-zero mean or non-identity covariance).

```julia
using NoLimits
using Distributions
using LinearAlgebra

# Default base distribution: MvNormal(zeros(1), I)
model = @Model begin
    @fixedEffects begin
        psi = NPFParameter(1, 3; seed=1, calculate_se=false)
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

# Custom base distribution: MvTDist for heavier tails
model_t = @Model begin
    @fixedEffects begin
        psi = NPFParameter(1, 3; seed=1, calculate_se=false,
                           base_dist=MvTDist(5, zeros(1), ones(1, 1)))
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
