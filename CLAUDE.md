# CLAUDE.md - NoLimits.jl

This file provides guidance for Claude Code when working on the NoLimits.jl package.

## Project Overview

**NoLimits.jl** is a Julia package for random effects analysis of longitudinal data using nonlinear mixed-effects models. It provides a comprehensive framework for modeling, estimating, and analyzing longitudinal data with support for:

- Fixed and random effects (univariate and multivariate)
- Nonlinear models with and without ordinary differential equations (ODEs)
- Multiple estimation methods (MLE, MAP, MCMC, Laplace, FOCEI, SAEM, MCEM)
- Complex covariate structures (constant, varying, and dynamic/interpolated)
- Visualization and diagnostic tools

**Status:** This package is being bootstrapped. APIs may change.

## Directory Structure

```
src/
├── NoLimits.jl          # Module definition and includes
├── Constants.jl                 # Global constants (scales, epsilon)
├── model/                       # Core model components
│   ├── Model.jl                 # @Model macro and bundles
│   ├── Parameters.jl            # Parameter blocks (RealNumber, RealVector, etc.)
│   ├── FixedEffects.jl          # @fixedEffects macro and structures
│   ├── RandomEffects.jl         # @randomEffects macro and distributions
│   ├── Covariates.jl            # @covariates macro (constant/varying/dynamic)
│   ├── Helpers.jl               # @helpers macro for user functions
│   ├── PreDE.jl                 # @preDifferentialEquation macro
│   ├── DifferentialEquation.jl  # @DifferentialEquation macro and ODE handling
│   ├── InitialDE.jl             # @initialDE macro for initial conditions
│   └── Formulas.jl              # @formulas macro for outcomes
├── data_model/
│   └── DataModel.jl             # DataModel construction and validation
├── data_simulation/
│   └── data_simulation.jl       # simulate_data and simulate_data_model
├── estimation/                  # Parameter estimation methods
│   ├── common.jl                # Shared structures and accessors
│   ├── mle.jl                   # Maximum Likelihood Estimation
│   ├── map.jl                   # Maximum A Posteriori estimation
│   ├── mcmc_turing.jl           # MCMC via Turing.jl
│   ├── laplace.jl               # Laplace approximation
│   ├── FOCEI.jl                 # FOCEI/FOCEIMAP estimation
│   ├── mcem.jl                  # Monte Carlo EM
│   ├── saem.jl                  # Stochastic Approximation EM
│   └── Multistart.jl            # Multistart optimization
├── plotting/                    # Visualization
│   ├── plotting.jl              # Core plotting utilities
│   ├── plots.jl                 # PlotStyle and plot_fits/plot_data
│   ├── plotting_vpc.jl          # Visual Predictive Checks
│   ├── plotting_uq.jl           # UQ distribution plots
│   ├── plotting_observation_distributions.jl
│   └── plotting_random_effects.jl
├── distributions/               # Custom distributions
│   ├── outcomes/
│   │   ├── ContinuousTimeHMM.jl
│   │   └── DiscreteTimeHMM.jl
│   └── random_effects/
│       └── NormalizingPlanarFlows.jl
├── soft_trees/
│   └── SoftTrees.jl             # Soft decision trees
└── utils/
    ├── GeneralUtils.jl
    ├── ParameterTranformations.jl
    └── Splines.jl
test/                            # Comprehensive test suite (~54 test files)
benchmarks/                      # Performance benchmarks
examples/                        # Usage examples
docs/                            # Documentation
```

## Key Coding Conventions

### ComponentArray Construction
Always use `ComponentArray(values, getaxes(existing_array))` - do not splat axes and do not call `axes` directly.

### Parameter Blocks (src/model/Parameters.jl)
Types: `RealNumber`, `RealVector`, `RealPSDMatrix`, `RealDiagonalMatrix`, `NNParameters`, `NPFParameter`, `SoftTreeParameters`, `SplineParameters`

Scales:
- `REAL_SCALES = (:identity, :log)`
- `PSD_SCALES = (:cholesky, :expm)`
- `DIAGONAL_SCALES = (:log,)`

### Struct Organization
- Nested structures with ≤5 fields each
- Accessor functions for immutable access (use `get_X` pattern, not dot access)

### AD Support
- Zygote-friendly code: non-mutating implementations
- Multiple AD backends: ForwardDiff, ReverseDiff, Zygote
- SciMLStructures for adjoint AD support

### NamedTuple Iteration
Use `Base.pairs(nt)` - do not call `pairs` unqualified.

## Model Construction

The `@Model` macro composes the following blocks into a `Model` struct:
- `@helpers` - User-defined helper functions (returns `NamedTuple` of functions)
- `@fixedEffects` - Fixed parameters with bounds, transforms, and optional priors
- `@covariates` - Constant, varying, and dynamic covariates
- `@randomEffects` - Random effects and grouping columns
- `@preDifferentialEquation` - Time-constant derived quantities for DEs
- `@DifferentialEquation` - ODE system (requires `@initialDE`)
- `@initialDE` - Initial conditions for DE states
- `@formulas` - Deterministic nodes and observation distributions (REQUIRED)

### Model Struct Bundles
The `Model` struct contains nested bundle types:
- `FixedBundle{F, T, IT}` - fixed effects, transform, inverse_transform
- `RandomBundle{R}` - random effects
- `CovariatesBundle{C}` - covariates
- `DEBundle{D, I, P, S, B}` - de, initial, prede, solver config, initial_builder
- `HelpersBundle{F}` - helper functions
- `FormulasBundle{F, A, O}` - formulas, all builder, obs builder, required_states, required_signals

### Model Validation Rules
- A model must have at least one fixed effect OR random effect
- `@DifferentialEquation` requires `@initialDE` and vice versa
- `@formulas` is always required
- DE covariate validation: varying covariates not allowed in DE, dynamic must use `w(t)`, constants must not be called as functions

### Parameter Block Types (src/model/Parameters.jl)

All parameter blocks inherit from `AbstractParameterBlock` and have common fields:
- `name::Symbol` - Parameter name (auto-injected by macro)
- `value` - Initial value
- `prior` - `Priorless()` or `Distribution`
- `calculate_se::Bool` - Include in standard error calculation

| Type | Value | Scale Options | Extra Fields |
|------|-------|---------------|--------------|
| `RealNumber` | `T<:Real` | `:identity`, `:log` | `lower`, `upper` |
| `RealVector` | `Vector{T}` | per-element `:identity`/`:log` | `lower`, `upper` |
| `RealPSDMatrix` | `Matrix{T}` (symmetric PSD) | `:cholesky`, `:expm` | - |
| `RealDiagonalMatrix` | `Vector{T}` (diagonal entries) | `:log` | - |
| `NNParameters` | flattened Lux chain params | - | `chain`, `function_name`, `reconstructor` |
| `SoftTreeParameters` | flattened tree params | - | `input_dim`, `depth`, `n_output`, `function_name`, `reconstructor` |
| `SplineParameters` | B-spline coefficients | - | `knots`, `degree`, `function_name` |
| `NPFParameter` | flattened planar flow params | - | `n_input`, `n_layers`, `reconstructor` |

**Prior validation for NN/SoftTree/Spline/NPF**: Must be `Priorless()`, a `Vector{Distribution}` of matching length, or a multivariate `Distribution` with matching `length()`.

### Covariate Types (src/model/Covariates.jl)

| Type | Usage | Key Fields |
|------|-------|------------|
| `Covariate()` | Time-varying scalar | `column` (inferred from LHS) |
| `CovariateVector([:a, :b])` | Time-varying vector | `columns` |
| `ConstantCovariate(; constant_on=:ID)` | Constant within group | `column`, `constant_on` |
| `ConstantCovariateVector([:a, :b]; constant_on=:ID)` | Constant vector | `columns`, `constant_on` |
| `DynamicCovariate(; interpolation=LinearInterpolation)` | Interpolated function of time | `column`, `interpolation` |
| `DynamicCovariateVector([:a, :b]; interpolations=[...])` | Interpolated vector | `columns`, `interpolations` |

**Important**: The LHS name in `@covariates` becomes the column name for scalar covariates. Do NOT pass explicit column names.

**`constant_on` behavior**:
- If only one RE grouping column exists, `constant_on` defaults to it
- If multiple RE groups exist, `constant_on` MUST be specified
- Covariates used in RE distributions must be `constant_on` for that RE's grouping column

### Allowed Dynamic Interpolations (DataInterpolations.jl)
`ConstantInterpolation`, `SmoothedConstantInterpolation`, `LinearInterpolation`, `QuadraticInterpolation`, `LagrangeInterpolation`, `QuadraticSpline`, `CubicSpline`, `AkimaInterpolation`

### DE Covariate Rules
- Varying covariates (`Covariate`, `CovariateVector`) are NOT allowed in DEs
- Dynamic covariates must be called as `w(t)`
- Constant covariates must NOT be called like functions

## DataModel

```julia
DataModel(model, df;
    primary_id::Union{Nothing, Symbol}=nothing,  # Required if multiple RE groups
    time_col::Symbol=:TIME,
    evid_col::Union{Nothing, Symbol}=nothing,    # Enables PKPD events
    amt_col::Symbol=:AMT,
    rate_col::Symbol=:RATE,
    cmt_col::Symbol=:CMT,
    serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial()
)
```

### DataModel Struct Components
```julia
struct DataModel{M, D, I, P, C, K, G, R}
    model::M              # The Model
    df::D                 # Original DataFrame
    individuals::I        # Vector{Individual}
    pairing::P            # PairingInfo (batches for RE groups)
    config::C             # DataModelConfig
    id_index::K           # Dict mapping id values to indices
    row_groups::G         # RowGroups (rows, obs_rows per individual)
    re_group_info::R      # REGroupInfo (values, index_by_row)
end
```

### Individual Struct
Each individual contains:
```julia
struct Individual{S, C, CB, TS, RG, SA}
    series::S         # IndividualSeries(obs, vary, dyn)
    const_cov::C      # NamedTuple of constant covariates
    callbacks::CB     # EventCallbacks or nothing
    tspan::TS         # (t_min, t_max) tuple
    re_groups::RG     # NamedTuple of RE group values
    saveat::SA        # Vector of save times or nothing
end
```

### Validation Checks (in order)
1. **Schema validation**: Required columns present, no missing values
2. **Time column**: Must be declared as `Covariate()` or `DynamicCovariate()` in `@covariates`
3. **RE group constants**: Covariates used in RE distributions must be `constant_on` for that group
4. **Constant covariates**: Must be constant within `primary_id` AND within all `constant_on` groups
5. **RE groups within primary**: RE grouping columns cannot vary within `primary_id`
6. **RE identifiability**: Grouping columns cannot have unique values per observation
7. **PreDE validation**: Random effects in preDE must be grouped by `primary_id`
8. **Dynamic covariate validation**: Time must be sorted; minimum observations per interpolation type

### Batching (PairingInfo)
Individuals are grouped into batches using **transitive union-find** across all RE grouping columns. Individuals sharing any RE level are in the same batch.

### Event Handling (PKPD)
When `evid_col` is provided:
- EVID=0: observation rows
- EVID=1: dose/infusion events (bolus if RATE=0, infusion if RATE>0)
- EVID=2: reset events (set compartment to AMT value)
- CMT can be integer index or Symbol/String matching state name

### Saveat Modes
- `:dense` - Full dense ODE solution
- `:saveat` - Save at observation + event times + formula time offsets
- `:auto` - Resolves to `:saveat` unless non-constant time offsets in formulas require `:dense`

### DataModel Accessors
```julia
get_model(dm)          # The Model
get_df(dm)             # Original DataFrame
get_individuals(dm)    # Vector{Individual}
get_individual(dm, id) # Individual by id value
get_batches(dm)        # Vector{Vector{Int}} - batches of individual indices
get_batch_ids(dm)      # Vector{Int} - batch id for each individual
get_primary_id(dm)     # Symbol - primary id column
get_row_groups(dm)     # RowGroups (rows, obs_rows per individual)
get_re_group_info(dm)  # REGroupInfo (values, index_by_row)
get_re_indices(dm, idx; obs_only=true)  # RE indices for individual
```

### Dynamic Covariate Minimum Observations
| Interpolation | Min Obs |
|---------------|---------|
| `ConstantInterpolation` | 1 |
| `SmoothedConstantInterpolation` | 2 |
| `LinearInterpolation` | 2 |
| `QuadraticInterpolation` | 3 |
| `LagrangeInterpolation` | 2 |
| `QuadraticSpline` | 3 |
| `CubicSpline` | 3 |
| `AkimaInterpolation` | 2 |

## Estimation Methods

### Unified API
```julia
fit_model(dm::DataModel, method::FittingMethod; constants, constants_re, penalty,
          ode_args, ode_kwargs, serialization, rng, theta_0_untransformed)
```

### Available Methods
- **MLE** - Maximum likelihood (fixed effects only)
- **MAP** - Maximum a posteriori (requires priors on fixed effects)
- **MCMC** - Turing-based sampling (fixed + optional RE, requires priors)
- **Laplace/LaplaceMAP** - Laplace approximation with Empirical Bayes (random effects)
- **FOCEI/FOCEIMAP** - First-order conditional estimation with interaction (random effects)
- **SAEM** - Stochastic Approximation EM (random effects)
- **MCEM** - Monte Carlo EM (random effects)
- **Multistart** - Runs any optimization-based method with multiple starting points

### Constants
- `constants::NamedTuple` fixes free fixed effects on the **transformed** scale; removed from optimizer state
- `constants_re::NamedTuple` fixes specific RE levels: `(; η=(; A=0.0, B=0.3))`

### Penalties
- `penalty::NamedTuple` applies per-parameter penalties on **natural scale**
- MCMC does NOT accept penalties; use MAP instead

## Fit Result Accessors

Always use accessor functions instead of dot access. Fit results store the DataModel by default (`store_data_model=false` to disable). Accessors throw an informative error for unsupported method combinations (e.g. `get_chain` on MLE).

```julia
get_method(res)           # -> FittingMethod
get_result(res)           # -> MethodResult
get_summary(res)          # -> FitSummary
get_diagnostics(res)      # -> FitDiagnostics
get_params(res; scale=:transformed|:untransformed|:both)
get_objective(res)
get_converged(res)
get_data_model(res)       # -> DataModel (stored by default)

# Optimization-based methods (MLE/MAP/Laplace/LaplaceMAP/FOCEI/FOCEIMAP/MCEM/SAEM)
get_iterations(res)
get_raw(res)
get_notes(res)

# MCMC-specific
get_chain(res)
get_observed(res)
get_sampler(res)
get_n_samples(res)

# Random effects (Laplace/LaplaceMAP/FOCEI/FOCEIMAP/MCEM/SAEM) — EB point estimates
get_random_effects(dm, res; constants_re=NamedTuple(), flatten=true, include_constants=true)
get_random_effects(res; ...)              # uses stored dm

# Log-likelihood (MLE/MAP/Laplace/LaplaceMAP/FOCEI/FOCEIMAP/MCEM/SAEM)
get_loglikelihood(dm, res; constants_re=NamedTuple(), ode_args=(), ode_kwargs=NamedTuple(), serialization=EnsembleSerial())
get_loglikelihood(res; ...)               # uses stored dm

# Multistart
get_multistart_results(res)
get_multistart_errors(res)
get_multistart_starts(res)
get_multistart_failed_results(res)
get_multistart_failed_starts(res)
get_multistart_best_index(res)
get_multistart_best(res)
```

## Transforms

`ForwardTransform` / `InverseTransform` operate on name-keyed `ComponentArray`:
- `:log` for scalars/vectors (mask supports mixed log/identity)
- `:cholesky` uses log-diag Cholesky, returns vec
- `:expm` uses matrix log/exp; stores only upper-tri vector

All transform operations are non-mutating to stay Zygote-friendly.

## Random Effects (src/model/RandomEffects.jl)

The `@randomEffects` macro builds `RandomEffects` with nested structs:
- `RandomEffectsMeta` - `re_names`, `re_groups`, `re_types`, `re_syms`
- `RandomEffectsBuilders` - `create_random_effect_distribution`, `logpdf`

### RandomEffect Declaration
```julia
@randomEffects begin
    η = RandomEffect(Normal(0.0, σ_η); column=:ID)
end
```
- `column` keyword is REQUIRED
- Distribution can use fixed effects, constant covariates, helpers, and model_funs
- Forbidden symbols: `t`, `ξ` (time variables)

### Runtime Distribution Builder
```julia
dists_builder = get_create_random_effect_distribution(dm.model.random.random)
# Signature: (θ::ComponentArray, const_cov::NamedTuple, model_funs::NamedTuple, helpers::NamedTuple) -> NamedTuple
dists = dists_builder(θ, const_cov_i, model_funs, helpers)
```

### NormalizingPlanarFlow in Random Effects
When using `NormalizingPlanarFlow(ψ)` in random effects, it's automatically rewritten to call `model_funs.NPF_ψ(ψ)`:
```julia
@fixedEffects begin
    ψ = NPFParameter(1, 3; seed=1)  # Creates NPF_ψ in model_funs
end
@randomEffects begin
    η = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
end
```

### Accessors
- `get_re_names(re)` - Vector of RE names
- `get_re_groups(re)` - NamedTuple mapping RE name -> grouping column
- `get_re_types(re)` - NamedTuple mapping RE name -> distribution type symbol
- `get_re_syms(re)` - NamedTuple mapping RE name -> Vector of symbols used in distribution
- `get_create_random_effect_distribution(re)` - Distribution builder function
- `get_re_logpdf(re)` - Logpdf function `(dists, re_values) -> Float64`

## Formulas (src/model/Formulas.jl)

The `@formulas` macro defines the observation model with two types of statements:

### Deterministic Nodes
```julia
lin = a + b * x.Age + η  # Assignment with =
```

### Observation Nodes
```julia
y ~ Normal(lin, σ)  # Distribution with ~
```

### Symbol Resolution
Formulas automatically resolve symbols from (in order):
1. **Fixed effects** - accessed as `fixed_effects.name`
2. **Random effects** - accessed as `random_effects.name`
3. **PreDE variables** - accessed as `prede.name`
4. **Constant covariates** - accessed as `constant_covariates_i.name`
5. **Varying covariates** - accessed as `varying_covariates.name`
6. **Helpers** - rewritten to `helpers.func(...)`
7. **Model functions** - rewritten to `model_funs.func(...)`
8. **DE states/signals** - rewritten to `sol_accessors.state(t)`

### DE State/Signal Access
States and signals MUST be called with time argument:
```julia
y ~ Normal(x1(t), σ)       # Current time
y ~ Normal(x1(t - 0.5), σ) # Time offset (constant offset allowed)
```

### Time Offsets
- Constant offsets like `x1(t - 0.5)` are collected for `saveat`
- Non-constant offsets require `saveat_mode=:dense`

### Implicit Varying Covariate Evaluation
Varying covariates that are callable (dynamic) are automatically evaluated at `t`:
```julia
# If w is a DynamicCovariate, both work:
y ~ Normal(w(t), σ)  # Explicit
y ~ Normal(w, σ)     # Implicit - evaluated at t
```

## Model Helper Functions

### Exported Functions from Model
```julia
get_model_funs(model)        # NamedTuple of model functions (NN, SoftTree, Spline, NPF)
get_helper_funs(model)       # NamedTuple of helper functions
get_solver_config(model)     # ODESolverConfig
set_solver_config(model; ...) # Returns new model with updated solver config

# Calculation functions
calculate_prede(model, θ, η, const_cov)
calculate_initial_state(model, θ, η, const_cov; static=false)
calculate_formulas_all(model, θ, η, const_cov, vary_cov, sol_accessors)
calculate_formulas_obs(model, θ, η, const_cov, vary_cov, sol_accessors)
```

### ODESolverConfig
```julia
struct ODESolverConfig{A, K, T}
    alg::A           # ODE algorithm (e.g., Tsit5())
    kwargs::K        # NamedTuple of solver kwargs
    args::T          # Tuple of solver args
    saveat_mode::Symbol  # :dense, :saveat, or :auto
end
```

## Plotting

All plotting functions accept:
- `ncols::Int = 3`
- `shared_x_axis::Bool = true`
- `shared_y_axis::Bool = true`
- `style::PlotStyle = PlotStyle()`
- `kwargs_subplot = NamedTuple()`
- `kwargs_layout = NamedTuple()`
- `save_path::Union{Nothing, String} = nothing`

Key plotting functions:
- `plot_data` - Raw observed data
- `plot_fits` - Model predictions vs observations
- `plot_fits_comparison` - Overlay multiple fits
- `plot_vpc` - Visual Predictive Checks
- `plot_observation_distributions` - Per-observation predicted distributions
- `get_residuals` - Returns DataFrame with `:pit`, `:quantile`, `:raw`, `:pearson`, `:logscore`
- `plot_residuals`, `plot_residual_distribution`, `plot_residual_qq`, `plot_residual_pit`, `plot_residual_acf`
- `plot_random_effect_distributions` - Per-level marginal RE distributions
- `plot_random_effect_pit`, `plot_random_effect_standardized`, `plot_random_effect_standardized_scatter`
- `plot_random_effects_pdf`, `plot_random_effects_scatter`, `plot_random_effect_pairplot`
- `plot_uq_distributions` - UQ parameter distributions
- `plot_multistart_waterfall` - Sorted objective values across starts
- `plot_multistart_fixed_effect_variability` - Parameter variation across starts
- `build_plot_cache` - Precompute ODE solutions / distributions for reuse

## Testing

Tests are organized by component:
- `estimation_*_tests.jl` - Tests for each estimation method
- `data_model_*_tests.jl` - DataModel validation tests
- `ad_*.jl` - Automatic differentiation tests
- `covariates_tests.jl`, `random_effects_tests.jl`, etc.

AD coverage includes random-effects logpdf gradients/Hessians w.r.t. fixed-effects on transformed scale.

## Key Dependencies

- **ODE Solving:** OrdinaryDiffEq.jl, DiffEqBase.jl, SciMLBase.jl
- **Data:** DataFrames.jl, CSV.jl, DataInterpolations.jl
- **Distributions:** Distributions.jl
- **AD:** ForwardDiff.jl, ReverseDiff.jl, Zygote.jl, DifferentiationInterface.jl
- **Optimization:** Optimization.jl, OptimizationOptimJL.jl
- **Bayesian/MCMC:** Turing.jl, MCMCChains.jl, DynamicPPL.jl
- **Neural Networks:** Lux.jl
- **Parameters:** ComponentArrays.jl
- **Visualization:** Plots.jl, StatsPlots.jl

## Useful Commands

```bash
# Run tests
julia --project -e 'using Pkg; Pkg.test()'

# Run specific test file
julia --project test/estimation_mle_tests.jl

# Run benchmarks
julia --project benchmarks/estimation_mle_benchmarks.jl
```

## Documentation Files

- `CAPABILITIES.md` - Consolidated capability reference (model building, estimation, UQ, plotting, simulation)

---

## API Examples (from test files)

### Complete Model with All Blocks

```julia
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using OrdinaryDiffEq
using DataInterpolations

model = @Model begin
    @helpers begin
        add1(x) = x + 1.0
        softplus(u) = log1p(exp(u))
        sat(u) = u / (1 + abs(u))
    end

    @fixedEffects begin
        a = RealNumber(1.0)
        b = RealNumber(0.2)
        σ = RealNumber(0.5)
    end

    @covariates begin
        t = Covariate()
        x = ConstantCovariateVector([:Age])
        z = Covariate()
        w1 = DynamicCovariate(; interpolation=LinearInterpolation)
    end

    @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end

    @preDifferentialEquation begin
        pre = add1(a) + x.Age + η
    end

    @DifferentialEquation begin
        D(x1) ~ -b * x1 + w1(t) + pre
    end

    @initialDE begin
        x1 = pre
    end

    @formulas begin
        lin = x1(t) + z
        obs ~ Normal(lin, σ)
    end
end
```

### Advanced Parameter Types

#### Neural Network Parameters (Lux.jl)
```julia
using Lux

chain = Chain(Dense(2, 4, relu), Dense(4, 1))

# Basic NN
nn = NNParameters(chain; name=:nn, function_name=:NN1, seed=0)

# With prior (vector of distributions)
n = length(nn.value)
nn_prior = NNParameters(chain; name=:nn, function_name=:NN1, seed=0,
                        prior=fill(Normal(), n))

# With multivariate normal prior
nn_mvn = NNParameters(chain; name=:nn, function_name=:NN1, seed=0,
                      prior=MvNormal(zeros(n), I))
```

#### SoftTree Parameters
```julia
# SoftTreeParameters(input_dim, depth; function_name, n_output=1, seed, calculate_se)
st = SoftTreeParameters(2, 3; name=:Γ, function_name=:ST, seed=0)

# Usage in model
model = @Model begin
    @fixedEffects begin
        Γ = SoftTreeParameters(2, 10; function_name=:ST, calculate_se=false)
    end
    @covariates begin
        x = ConstantCovariateVector([:Age, :BMI])
    end
    @formulas begin
        y ~ Normal(ST([x.Age, x.BMI], Γ)[1], 1.0)
    end
end
```

#### Spline Parameters
```julia
knots = collect(range(0.0, 1.0; length=6))
sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)

# Usage in model
model = @Model begin
    @fixedEffects begin
        sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
    end
    @formulas begin
        y ~ Normal(SP1(0.5, sp), 1.0)
    end
end
```

#### NPFParameter (Normalizing Planar Flows)
```julia
npf = NPFParameter(2, 3; name=:ψ, seed=123, calculate_se=false)

# Use in random effects for flexible distributions
model = @Model begin
    @fixedEffects begin
        σ = RealNumber(0.4)
        ψ = NPFParameter(1, 3; seed=1, calculate_se=false)
    end
    @randomEffects begin
        η = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
    end
    @formulas begin
        y ~ Normal(sat(η), σ)
    end
end
```

#### Complex Fixed Effects Block
```julia
using LinearAlgebra

chain1 = Chain(Dense(2, 8, relu), Dense(8, 1))
knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]

fe = @fixedEffects begin
    # Vectors with mixed scales
    β = RealVector(rand(2),
                   scale=[:identity, :log],
                   lower=[-Inf, 1e-12], upper=[Inf, Inf],
                   prior=MvNormal(zeros(2), I),
                   calculate_se=true)

    # Scalar with log scale
    λ = RealNumber(0.05, scale=:log, lower=1e-12, upper=Inf,
                   prior=Normal(0.0, 1.0))

    # PSD matrix with Cholesky parameterization
    Ω = RealPSDMatrix([1.0 0.0; 0.0 1.0], scale=:cholesky,
                      prior=Wishart(4, Matrix(I, 2, 2)))

    # Diagonal matrix
    D = RealDiagonalMatrix([1.0, 1.0], scale=:log)

    # Advanced parameters
    ζ = NNParameters(chain1; function_name=:NN1, calculate_se=false)
    Γ = SoftTreeParameters(2, 5; function_name=:ST, calculate_se=false)
    sp = SplineParameters(knots; function_name=:SP1, degree=2)
    ψ = NPFParameter(1, 5; seed=123, calculate_se=false)
end
```

### Estimation Workflows

#### MLE (Maximum Likelihood)
```julia
model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.3)
        σ = RealNumber(0.5, scale=:log)
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(softplus(a), σ)
    end
end

df = DataFrame(ID=[1, 1], t=[0.0, 1.0], y=[1.0, 1.1])
dm = DataModel(model, df; primary_id=:ID, time_col=:t)

# Basic fit
res = fit_model(dm, NoLimits.MLE())

# With optimizer options
res = fit_model(dm, NoLimits.MLE(
    optimizer=BFGS(),
    optim_kwargs=(; iterations=100)
))

# With constants (fix parameters)
res = fit_model(dm, NoLimits.MLE(); constants=(a=0.2,))

# With penalties
res = fit_model(dm, NoLimits.MLE(); penalty=(a=100.0,))

# Access results
obj = NoLimits.get_objective(res)
params = NoLimits.get_params(res; scale=:untransformed)
```

#### MAP (Maximum A Posteriori)
```julia
model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.1; prior=Normal(0.0, 1.0))
        σ = RealNumber(0.5; prior=LogNormal(0.0, 0.5))
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(exp(a), σ)
    end
end

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
res = fit_model(dm, NoLimits.MAP())
```

#### MCMC (Bayesian Sampling)
```julia
using Turing

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2, prior=Normal(0.0, 1.0))
        σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(softplus(a), σ)
    end
end

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

res = fit_model(dm, NoLimits.MCMC(;
    sampler=NUTS(5, 0.3),
    turing_kwargs=(n_samples=100, n_adapt=50, progress=true)
))

chain = NoLimits.get_chain(res)
```

#### MCMC with Random Effects
```julia
model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2, prior=Normal(0.0, 1.0))
        σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(exp(a + η), σ)
    end
end

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
res = fit_model(dm, NoLimits.MCMC(;
    turing_kwargs=(n_samples=100, n_adapt=50, progress=false)
))
```

#### Laplace Approximation (Random Effects)
```julia
model = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.5)
    end
    @randomEffects begin
        η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
        η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(a + η_id + η_site, σ)
    end
end

df = DataFrame(
    ID=[1, 1, 2, 2, 3, 3, 4, 4],
    SITE=[:A, :A, :A, :A, :B, :B, :B, :B],
    t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y=[1.0, 1.1, 0.9, 1.0, 1.2, 1.1, 1.0, 0.95]
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
res = fit_model(dm, NoLimits.Laplace())

# Get random effects estimates
re = NoLimits.get_random_effects(dm, res)
```

#### SAEM (Stochastic Approximation EM)
```julia
res = fit_model(dm, NoLimits.SAEM(;
    sampler=MH(),
    turing_kwargs=(n_samples=10, n_adapt=5, progress=false),
    max_store=10,
    maxiters=50
))

converged = NoLimits.get_converged(res)
```

#### MCEM (Monte Carlo EM)
```julia
res = fit_model(dm, NoLimits.MCEM(;
    sampler=MH(),
    turing_kwargs=(n_samples=20, n_adapt=5, progress=false),
    maxiters=20
))
```

### DataModel Construction

#### Basic DataModel
```julia
df = DataFrame(
    ID=[1, 1, 2, 2],
    t=[0.0, 1.0, 0.0, 1.0],
    Age=[30.0, 30.0, 40.0, 40.0],
    z=[1.0, 1.2, 0.8, 0.9],
    y=[1.0, 1.1, 0.9, 1.0]
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

# Access components
individuals = get_individuals(dm)
ind1 = get_individual(dm, 1)
batches = get_batches(dm)
```

#### DataModel with Events (PKPD)
```julia
df = DataFrame(
    ID=[1, 1, 1, 2, 2],
    t=[0.0, 0.5, 1.0, 0.0, 1.0],
    EVID=[1, 0, 0, 1, 0],
    AMT=[100.0, 0.0, 0.0, 50.0, 0.0],
    RATE=[0.0, 0.0, 0.0, 0.0, 0.0],
    CMT=[1, 1, 1, 1, 1],
    y=[missing, 1.1, 1.2, missing, 0.9]
)

dm = DataModel(model, df;
    primary_id=:ID,
    time_col=:t,
    evid_col=:EVID,
    amt_col=:AMT,
    rate_col=:RATE,
    cmt_col=:CMT
)
```

#### DataModel with Parallelization
```julia
dm = DataModel(model, df;
    primary_id=:ID,
    time_col=:t,
    serialization=EnsembleThreads()
)
```

### Covariate Types

```julia
@covariates begin
    # Time-varying scalar
    t = Covariate()

    # Time-varying vector
    z = CovariateVector([:z1, :z2])

    # Constant scalar (within group)
    w = ConstantCovariate(; constant_on=:ID)

    # Constant vector (within group)
    x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)

    # Dynamic covariate (interpolated)
    dose = DynamicCovariate(; interpolation=LinearInterpolation)

    # Dynamic covariate vector
    inputs = DynamicCovariateVector([:i1, :i2];
        interpolations=[LinearInterpolation, CubicSpline])
end
```

### Random Effects with Various Distributions

```julia
@randomEffects begin
    # Normal
    η_normal = RandomEffect(Normal(0.0, σ_η); column=:ID)

    # Beta
    η_beta = RandomEffect(Beta(α1, α2); column=:ID)

    # Multivariate Normal
    η_mv = RandomEffect(MvNormal(μ, Ω); column=:SITE)

    # Normalizing Planar Flow
    η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:SITE)

    # With NN-parameterized distribution
    η_nn = RandomEffect(LogNormal(NN1([x.Age], ζ)[1], σ); column=:ID)

    # With SoftTree-parameterized distribution
    η_st = RandomEffect(Gumbel(ST([x.Age, x.BMI], Γ)[1], σ); column=:ID)

    # With Spline-parameterized distribution
    η_sp = RandomEffect(Normal(SP1(0.5, sp), σ); column=:SITE)

    # With helper functions
    η_help = RandomEffect(Normal(sat(β), σ); column=:ID)
end
```

### Helpers Block

```julia
@helpers begin
    clamp01(u) = max(0.0, min(1.0, u))
    softplus(u) = log1p(exp(u))
    sat(u) = u / (1 + abs(u))
    hill(u, n) = abs(u)^n / (1 + abs(u)^n)
    sigmoid(u) = 1 / (1 + exp(-u))
    logit(u) = log(u / (1 - u))
end
```

### Differential Equations

```julia
@DifferentialEquation begin
    # Derived signal (not a state, computed from states)
    s(t) = sin(t) + x1^2

    # State equations
    D(x1) ~ -k * x1 + w(t) + pre
    D(x2) ~ k * x1 - ke * x2
end

@initialDE begin
    x1 = dose0 + η_ic
    x2 = 0.0
end
```

### PreDifferentialEquation Block

```julia
@preDifferentialEquation begin
    # Time-constant derived quantities
    clearance = β_cl + η_cl
    volume = exp(β_v + η_v)
    nn_out = NN1([x.Age, x.BMI], ζ)[1]
    st_out = ST([x.Age, x.BMI], Γ)[1]
    sp_out = SP1(x.Age / 100, sp)
end
```

### Formulas Block

```julia
@formulas begin
    # Deterministic nodes
    lin = a + b * x.Age + η
    nonlin = softplus(lin) + NN1([x.Age], ζ)[1]

    # With ODE solution accessors
    conc = x1(t) / volume

    # With time offset
    conc_delayed = x1(t - 0.5) / volume

    # Observation models
    y ~ Normal(conc, σ)
    binary ~ Bernoulli(sigmoid(lin))
    count ~ Poisson(exp(lin))
end
```

### Accessing Model Components

```julia
# Fixed effects initial values
θ = get_θ0_untransformed(model.fixed.fixed)
θ_t = get_θ0_transformed(model.fixed.fixed)

# Transforms
transform = get_transform(model.fixed.fixed)
inverse_transform = get_inverse_transform(model.fixed.fixed)
θ_transformed = transform(θ)
θ_back = inverse_transform(θ_transformed)

# Helpers and model functions
helpers = get_helper_funs(model)
model_funs = get_model_funs(model)

# Names
names = get_names(model.fixed.fixed)
flat_names = get_flat_names(model.fixed.fixed)

# SE mask
se_mask = get_se_mask(model.fixed.fixed)

# Priors
priors = get_priors(model.fixed.fixed)
```

### Data Simulation

```julia
# Simulate from a DataModel
sim = simulate_data(dm; rng=MersenneTwister(1))

# Build a new DataModel from simulated data
dm_sim = simulate_data_model(dm; rng=MersenneTwister(4))
```

### Spline and SoftTree Direct Usage

```julia
# B-spline evaluation
knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
degree = 2
coeffs = [1.0, 2.0, 3.0, 4.0]

basis = bspline_basis(0.25, knots, degree)
y = bspline_eval(0.25, coeffs, knots, degree)

# SoftTree direct usage
tree = SoftTree(3, 2, 2)  # input_dim, depth, n_output
params = init_params(tree)
params_rand = init_params(tree, Xoshiro(0))

x = [0.1, -0.2, 0.3]
y = tree(x, params)
y_fast = tree(x, params, Val(:fast))        # Mutating, faster
y_inplace = tree(x, params, Val(:inplace))  # Zygote.Buffer
```

### Model with NN/SoftTree/Spline in Formulas

```julia
chain = Chain(Dense(2, 4, tanh), Dense(4, 1))
knots = collect(range(0.0, 1.0; length=6))

model = @Model begin
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end

    @fixedEffects begin
        σ = RealNumber(0.4)
        ζ = NNParameters(chain; function_name=:NN1, calculate_se=false)
        Γ = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
        sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
    end

    @covariates begin
        x = ConstantCovariateVector([:Age, :BMI])
    end

    @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end

    @formulas begin
        μ = sat(NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1]) + SP1(0.4, sp)^2 + η
        y ~ Normal(log1p(μ^2), σ)
    end
end
```
