# NoLimits.jl Capabilities

This file is the consolidated capability reference for `NoLimits.jl`. It covers model building, estimation, uncertainty quantification, plotting, and simulation.

- Package status: active and broad in functionality, with API evolution still possible.
- Primary domain: nonlinear longitudinal data models with optional random effects, ODE dynamics, and HMM outcomes.
- Central interfaces:
  - Model definition: `@Model ...`
  - Data binding and validation: `DataModel(...)`
  - Estimation: `fit_model(dm, method; kwargs...)`
  - Uncertainty quantification: `compute_uq(res; kwargs...)`
  - Diagnostics and visualization: plotting API in `src/plotting`.

---

## 1. Model Building

### 1.1 Model DSL and Required Structure

`@Model` composes these blocks:
- `@helpers`
- `@fixedEffects`
- `@covariates`
- `@randomEffects`
- `@preDifferentialEquation`
- `@DifferentialEquation`
- `@initialDE`
- `@formulas`

Validation rules:
- `@formulas` is required.
- `@DifferentialEquation` requires `@initialDE`.
- At least one fixed effect or random effect is required.

### 1.2 Fixed Effects (`@fixedEffects`)

Declares fixed parameters with bounds, transforms, and optional priors. Supported parameter blocks:
- `RealNumber`, `RealVector`
- `RealPSDMatrix`, `RealDiagonalMatrix`
- `ProbabilityVector`, `DiscreteTransitionMatrix`
- `NNParameters`, `SoftTreeParameters`, `SplineParameters`, `NPFParameter`

Transform-aware optimization support:
- Typical scales include `:identity`, `:log`, `:logit`, matrix transforms (`:cholesky`, `:expm`), and simplex transforms (`:stickbreak`, `:stickbreakrows`).

Model functions from `NNParameters`, `SoftTreeParameters`, `SplineParameters` are registered in `model_funs` and callable in formulas/DE/preDE.

### 1.3 Helper Functions and Learned Function Blocks

- `@helpers` functions are available in formulas/DE/preDE.
- Functions induced by `NNParameters`, `SoftTreeParameters`, and `SplineParameters` are registered in `model_funs` and callable in formulas/DE/preDE.

### 1.4 Random Effects (`@randomEffects`)

Declares random effects and their grouping columns. Each `RandomEffect(dist; column=:GROUP)` defines a group-level random effect.

Supports scalar and multivariate random effects. Multiple RE groups are supported simultaneously.

Flexible RE distribution families:
- Gaussian and multivariate Gaussian (`Normal`, `MvNormal`)
- Skewed/heavy-tail/positive families (e.g. `LogNormal`, `Gamma`, `Weibull`, `InverseGaussian`, `Laplace`, `TDist`, `SkewNormal`, `Gumbel`)
- `NormalizingPlanarFlow` (flow-based RE distributions)

RE distributions can be parameterized by model expressions involving:
- fixed effects, helper functions, neural-network outputs, spline outputs, soft-tree outputs, constant covariates.

RE distributions parameterized by constant covariates are explicitly supported in model construction, simulation, and estimation paths.

Identifiability check: if the grouping column is unique per observation (no repeated levels), DataModel emits a warning about weak identifiability (possible confounding with measurement noise) and recommends validating with `identifiability_report(...)`.

### 1.5 Covariates (`@covariates`)

Covariate types:
- `Covariate`, `CovariateVector`: varying covariates (value per observation).
- `ConstantCovariate`, `ConstantCovariateVector`: constant within groups.
- `DynamicCovariate`, `DynamicCovariateVector`: interpolated functions of time.

Constant covariates:
- Must be constant within `primary_id` and within all `constant_on` groups.
- `constant_on` defaults to the only RE group when a single grouping column exists.

Dynamic covariates must declare an interpolation type from DataInterpolations.jl:
- `ConstantInterpolation`, `SmoothedConstantInterpolation`, `LinearInterpolation`, `QuadraticInterpolation`, `LagrangeInterpolation`, `QuadraticSpline`, `CubicSpline`, `AkimaInterpolation`

Per-individual time must be sorted when dynamic covariates exist. Minimum observation count:

| Interpolation | Min Obs |
|---|---|
| `ConstantInterpolation` | 1 |
| `SmoothedConstantInterpolation` | 2 |
| `LinearInterpolation` | 2 |
| `QuadraticInterpolation` | 3 |
| `LagrangeInterpolation` | 2 |
| `QuadraticSpline` | 3 |
| `CubicSpline` | 3 |
| `AkimaInterpolation` | 2 |

### 1.6 Outcome Formulas (`@formulas`)

- Deterministic nodes: `lhs = expr`
- Observation nodes: `lhs ~ Distribution(...)`

Supports: fixed effects, random effects, preDE vars, constant covariates, varying covariates, dynamic covariates via `w(t)`, DE states/signals via `x(t)`, helpers and model functions.

Specialized likelihood kernels exist for `Normal`, `Bernoulli`, `Poisson`, `LogNormal`. Non-Gaussian outcomes are supported in estimation workflows (method-specific details below).

HMM outcome distributions are supported:
- `DiscreteTimeDiscreteStatesHMM`
- `ContinuousTimeDiscreteStatesHMM`

### 1.7 PreDifferentialEquation and ODE

`@preDifferentialEquation`:
- Computes time-constant derived quantities for DEs and formulas.
- Inputs: fixed effects, random effects, constant covariates, helpers, model functions.
- Does not allow time-varying quantities.
- DataModel rejects preDE random effects not grouped by `primary_id`.

`@DifferentialEquation`:
- Defines ODE system: `D(x) ~ expr`, plus optional derived signals `s(t)=expr`.
- Varying covariates (`Covariate`, `CovariateVector`) are not allowed in DEs.
- Dynamic covariates must be called as `w(t)`.
- Constant covariates must not be called as functions.
- DE states and derived signals can be accessed as `x1(t)` in formulas.

`@initialDE`:
- Defines initial conditions for DE states.
- Inputs: fixed effects, random effects, constant covariates, helpers, model functions, preDE.

### 1.8 Solver Configuration

```julia
set_solver_config(model; saveat_mode=:dense|:saveat|:auto, alg, kwargs, args)
```

- `:auto` resolves to `:saveat` unless non-constant offsets in formulas force dense.
- Save-at behavior integrates with formula time offsets and ODE solve strategy.

### 1.9 DataModel

```julia
DataModel(model, df;
    primary_id, time_col, evid_col, amt_col, rate_col, cmt_col, obs_cols, serialization)
```

Validation checks:
- Required columns present; no missing in required columns.
- `primary_id` is required when multiple RE grouping columns exist; inferred otherwise.
- Constant covariates are constant within `primary_id` and within `constant_on` groups.
- RE grouping columns must be non-missing.
- RE grouping columns must be constant within `primary_id` (no time-varying random effects).
- preDE only uses random effects grouped by `primary_id`.
- Dynamic covariate time sorted per individual; min obs per interpolation enforced.
- Random effect identifiability: grouping columns unique per observation trigger a warning (weak identifiability risk), not a hard error.

Events:
- If `evid_col` provided, observations are `evid == 0`, and events (nonzero) are captured as callbacks.
- EVID=1: dose/infusion events; EVID=2: reset events.
- CMT can be integer index or Symbol/String matching state name.

Batching:
- Individuals are grouped into batches by transitive overlap of RE grouping columns.

Saveat:
- `saveat_mode=:saveat` uses observation times plus any constant time offsets from formulas.
- Non-constant offsets require `:dense`.

### 1.10 Examples

#### 1) Basic nonlinear mixed model
```julia
model = @Model begin
    @helpers begin
        softplus(u) = log1p(exp(u))
    end
    @fixedEffects begin
        a = RealNumber(1.0)
        b = RealNumber(0.2)
        σ = RealNumber(0.5)
    end
    @covariates begin
        x = ConstantCovariateVector([:Age]; constant_on=:ID)
        z = Covariate()
    end
    @randomEffects begin
        η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end
    @formulas begin
        μ = a + b * x.Age^2 + softplus(z) + η_id
        y ~ Normal(μ, σ)
    end
end
```

#### 2) ODE with dynamic covariate and preDE
```julia
model = @Model begin
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    @fixedEffects begin
        a = RealNumber(0.2)
        b = RealNumber(0.1)
        σ = RealNumber(0.3)
    end
    @covariates begin
        x = ConstantCovariateVector([:Age]; constant_on=:ID)
        w = DynamicCovariate(; interpolation=LinearInterpolation)
    end
    @randomEffects begin
        η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end
    @preDifferentialEquation begin
        pre = sat(a + x.Age + η_id)
    end
    @DifferentialEquation begin
        D(x1) ~ -b * x1^2 + pre + w(t)^2
    end
    @initialDE begin
        x1 = 1.0
    end
    @formulas begin
        y ~ Normal(log1p(x1(t)^2), σ)
    end
end
```

#### 3) NN + SoftTree + Spline in preDE with multiple RE groups
```julia
chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
knots = collect(range(0.0, 1.0; length=6))

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        σ = RealNumber(0.3)
        ζ = NNParameters(chain; function_name=:NN1, calculate_se=false)
        Γ = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
        sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
    end
    @covariates begin
        x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
        w = DynamicCovariate(; interpolation=LinearInterpolation)
    end
    @randomEffects begin
        η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
        η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
    end
    @preDifferentialEquation begin
        pre = NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1] + SP1(x.Age / 100, sp) + η_id
    end
    @DifferentialEquation begin
        D(x1) ~ -a * x1 + pre + w(t) + η_site
    end
    @initialDE begin
        x1 = 1.0
    end
    @formulas begin
        y ~ Normal(x1(t), σ)
    end
end
```

#### 4) DataModel construction
```julia
df = DataFrame(
    ID = [1, 1, 2, 2],
    SITE = [:A, :A, :B, :B],
    t = [0.0, 1.0, 0.0, 1.0],
    Age = [30.0, 30.0, 40.0, 40.0],
    BMI = [20.0, 20.0, 25.0, 25.0],
    w = [0.2, 0.4, 0.1, 0.3],
    y = [1.0, 0.9, 1.1, 1.0]
)
dm = DataModel(model, df; primary_id=:ID, time_col=:t)
```

#### 5) Fixed effects only (no random effects)
```julia
model = @Model begin
    @helpers begin
        softplus(u) = log1p(exp(u))
    end
    @fixedEffects begin
        a = RealNumber(1.0)
        b = RealNumber(0.3)
        σ = RealNumber(0.5)
    end
    @covariates begin
        x = ConstantCovariateVector([:Age])
        z = Covariate()
    end
    @formulas begin
        μ = a + b * x.Age^2 + softplus(z)
        y ~ Normal(μ, σ)
    end
end
```

#### 6) NN + SoftTree + Spline in formulas
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

#### 7) Planar flow random effects
```julia
model = @Model begin
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    @fixedEffects begin
        σ = RealNumber(0.4)
        ψ = NPFParameter(1, 3, seed=1, calculate_se=false)
    end
    @randomEffects begin
        η = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
    end
    @formulas begin
        y ~ Normal(sat(η), σ)
    end
end
```

#### 8) Multivariate random effects
```julia
model = @Model begin
    @fixedEffects begin
        σ1 = RealNumber(0.3)
        σ2 = RealNumber(0.5)
    end
    @randomEffects begin
        η = RandomEffect(MvNormal(zeros(2), I); column=:ID)
    end
    @formulas begin
        y1 ~ Normal(exp(η[1]), σ1)
        y2 ~ Normal(log1p(η[2]^2), σ2)
    end
end
```

#### 9) Multiple RE grouping columns
```julia
model = @Model begin
    @fixedEffects begin
        σ = RealNumber(0.5)
    end
    @randomEffects begin
        η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
        η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        η_year = RandomEffect(Normal(0.0, 1.0); column=:YEAR)
    end
    @formulas begin
        y ~ Normal(exp(η_id + η_site) + η_year^2, σ)
    end
end
```

#### 10) Constant covariates parameterize RE distributions
```julia
model = @Model begin
    @helpers begin
        softplus(u) = log1p(exp(u))
    end
    @fixedEffects begin
        σ = RealNumber(0.5)
    end
    @covariates begin
        c_id = ConstantCovariate(; constant_on=:ID)
        c_site = ConstantCovariate(; constant_on=:SITE)
    end
    @randomEffects begin
        η_id = RandomEffect(Normal(c_id, 1.0); column=:ID)
        η_site = RandomEffect(Normal(c_site, 1.0); column=:SITE)
    end
    @formulas begin
        y ~ Normal(softplus(η_id + η_site), σ)
    end
end
```

#### 11) Rich covariate palette (constant, varying, dynamic vectors)
```julia
model = @Model begin
    @fixedEffects begin
        σ = RealNumber(0.4)
    end
    @covariates begin
        x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
        z = CovariateVector([:CRP, :ALT])
        w = DynamicCovariateVector([:w1, :w2]; interpolations=[LinearInterpolation, QuadraticSpline])
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end
    @formulas begin
        μ = x.Age^2 + log1p(x.BMI^2) + tanh(z.CRP + z.ALT) + w.w1(t)^2 + w.w2(t) + η
        y ~ Normal(μ, σ)
    end
end
```

#### 12) ODE with multiple outcomes
```julia
model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        σ1 = RealNumber(0.3)
        σ2 = RealNumber(0.4)
    end
    @DifferentialEquation begin
        D(x1) ~ -a * x1^2
    end
    @initialDE begin
        x1 = 1.0
    end
    @formulas begin
        y1 ~ Normal(log1p(x1(t)^2), σ1)
        y2 ~ Normal(exp(-x1(t)) + 0.5, σ2)
    end
end
```

#### 13) Hidden Markov Model outcome
```julia
model = @Model begin
    @helpers begin
        sigmoid(u) = 1 / (1 + exp(-u))
    end
    @fixedEffects begin
        β = RealNumber(0.7)
    end
    @formulas begin
        p11 = sigmoid(β)
        p22 = sigmoid(β^2)
        trans = [p11 1 - p11; 1 - p22 p22]
        em1 = Normal(-1.0 + β, 0.4)
        em2 = Normal(1.0 + β^2, 0.6)
        y ~ DiscreteTimeDiscreteStatesHMM(trans, (em1, em2), Categorical([0.5, 0.5]))
    end
end
```

#### 14) ODE + preDE + NN/SoftTree/Spline + dynamic covariates
```julia
chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
knots = collect(range(0.0, 1.0; length=6))

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        σ = RealNumber(0.3)
        ζ = NNParameters(chain; function_name=:NNX, calculate_se=false)
        Γ = SoftTreeParameters(2, 2; function_name=:STX, calculate_se=false)
        sp = SplineParameters(knots; function_name=:SPX, degree=2, calculate_se=false)
    end
    @covariates begin
        x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
        w = DynamicCovariate(; interpolation=AkimaInterpolation)
    end
    @randomEffects begin
        η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end
    @preDifferentialEquation begin
        pre = NNX([x.Age, x.BMI], ζ)[1] + STX([x.Age, x.BMI], Γ)[1] + SPX(x.Age / 100, sp) + η_id
    end
    @DifferentialEquation begin
        D(x1) ~ -a * x1^2 + pre + w(t)
    end
    @initialDE begin
        x1 = 1.0
    end
    @formulas begin
        y ~ Normal(log1p(x1(t)^2), σ)
    end
end
```

#### 15) ODE with events (EVID/AMT/RATE/CMT)
```julia
model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.1)
        σ = RealNumber(0.4)
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end
    @DifferentialEquation begin
        D(x1) ~ -a * x1^2 + η
    end
    @initialDE begin
        x1 = 1.0
    end
    @formulas begin
        y ~ Normal(log1p(x1(t)^2), σ)
    end
end

df = DataFrame(
    ID = [1, 1, 1, 2, 2],
    t = [0.0, 0.5, 1.0, 0.0, 1.0],
    EVID = [1, 0, 0, 1, 0],
    AMT = [100.0, 0.0, 0.0, 50.0, 0.0],
    RATE = [0.0, 0.0, 0.0, 0.0, 0.0],
    CMT = [1, 1, 1, 1, 1],
    y = [missing, 1.1, 1.2, missing, 0.9]
)

model_saveat = set_solver_config(model; saveat_mode=:saveat)
dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t,
               evid_col=:EVID, amt_col=:AMT, rate_col=:RATE, cmt_col=:CMT)
```

#### 16) DE derived signals
```julia
model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        σ = RealNumber(0.3)
    end
    @DifferentialEquation begin
        s(t) = a + x1^2
        D(x1) ~ -a * x1^2
    end
    @initialDE begin
        x1 = 1.0
    end
    @formulas begin
        y ~ Normal(log1p(s(t)^2), σ)
    end
end
```

#### 17) Pairing/batching across RE grouping columns
```julia
model = @Model begin
    @fixedEffects begin
        σ = RealNumber(0.5)
    end
    @randomEffects begin
        η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
        η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
    end
    @formulas begin
        y ~ Normal(exp(η_id + η_site), σ)
    end
end

df = DataFrame(
    ID = [1, 1, 2, 2, 3, 3, 4, 4],
    SITE = [:A, :A, :A, :A, :B, :B, :B, :B],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y = [1.0, 1.1, 0.9, 1.0, 1.2, 1.1, 1.0, 0.95]
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
batches = get_batches(dm)  # transitive pairing across RE groups
```

---

## 2. Estimation

### 2.1 Unified API

```julia
fit_model(dm::DataModel, method::FittingMethod;
          constants, constants_re, penalty,
          ode_args, ode_kwargs, serialization, rng, theta_0_untransformed)
```

Shared controls:
- `constants::NamedTuple` — fixed-effect constants on the **transformed** scale; removed from optimizer state.
- `constants_re::NamedTuple` — fixes specific RE levels: `(; η=(; A=0.0, B=0.3))`.
- `penalty::NamedTuple` — per-parameter penalties on the **natural** scale. Not supported by MCMC.
- `ode_args`, `ode_kwargs` — passed through to ODE solving.
- `serialization` — `EnsembleSerial()` or `EnsembleThreads()`.
- `rng` — random number generator.
- `theta_0_untransformed` — optional custom starting values on the **natural** scale.

### 2.2 Fixed-Effects-Only Methods

Applies to models **without random effects**. Supports nonlinear models with and without ODEs.

#### MLE (Maximum Likelihood)
```julia
fit_model(dm, MLE(; optimizer, optim_kwargs, adtype, lb, ub))
```
- Optimizes **negative loglikelihood**.
- Requires at least one free fixed effect.
- Bounds are defined on the **transformed** scale.
- If all bounds are infinite, no bounds are passed to Optimization.jl.
- BlackBoxOptim methods require finite bounds; use `default_bounds_from_start(dm; margin=...)`.

#### MAP (Maximum A Posteriori)
```julia
fit_model(dm, MAP(; optimizer, optim_kwargs, adtype, lb, ub))
```
- Optimizes **negative logposterior** (loglikelihood + logprior).
- Hard error if no fixed-effect prior is provided.

#### MCMC (Turing)
```julia
fit_model(dm, MCMC(; sampler, turing_kwargs, adtype))
```
- Builds a Turing model and samples fixed effects.
- Hard error if any free fixed effect is `Priorless`.
- Ignores parameter scale (`:log`, `:cholesky`); sampling is on natural scale.
- Penalties are not supported.

### 2.3 Random-Effects Methods

#### Laplace (approximate ML)

```julia
fit_model(dm, Laplace(; optimizer, optim_kwargs, adtype,
                        inner_optimizer, inner_kwargs, inner_adtype, inner_grad_tol,
                        multistart_n, multistart_k, multistart_grad_tol,
                        jitter, max_tries, growth, adaptive, scale_factor,
                        use_trace_logdet_grad, use_hutchinson, hutchinson_n,
                        theta_tol, lb, ub))
```

Core idea: approximates the marginal likelihood by a local Gaussian expansion around batch-wise posterior modes (EBEs). For each batch, optimize `b` (random effects) and compute Hessian in `b`.

Capabilities:
- Multiple RE groups and multivariate REs.
- ODE models, `constants_re`, threaded evaluation (`EnsembleThreads`).
- Hessian stabilization with adaptive jitter.
- Multistart fallback when post-solve gradient is still large.

Outputs: `LaplaceResult` includes EB modes per batch.

Tests: `test/estimation_laplace_tests.jl`, `test/estimation_laplace_fit_tests.jl`

#### LaplaceMAP (approximate MAP)

Same as Laplace but includes fixed-effect priors in the objective. Requires priors for all fixed effects; errors if any are priorless.

Tests: `test/estimation_laplace_map_tests.jl`

#### FOCEI (first-order conditional estimation with interaction)

```julia
fit_model(dm, FOCEI(; ..., info_mode, info_jitter, info_max_tries,
                       fallback_to_laplace, mode_sensitivity, info_custom, ...))
```

Core idea: uses Laplace-style batch modes (EBEs) but an information-matrix based logdet term.

Information mode options:
- `info_mode=:fisher_common` (default) — analytic expected information for supported outcomes/priors.
- `info_mode=:custom` — user-provided information-matrix approximation with signature `(dm, batch_info, θ, b, const_cache, ll_cache) -> I::AbstractMatrix`.
- Built-in helper: `info_mode=:custom, info_custom=focei_information_opg`.

`:fisher_common` supports:
- Outcomes: `Normal`, `LogNormal`, `Bernoulli`, `Poisson`, `Exponential`, `Geometric`, `Binomial`
- RE priors: `Normal`, `LogNormal`, `Exponential`, `MvNormal`

Mode sensitivity options: `:exact_hessian` (default) or `:focei_info`.

Capabilities: multiple RE groups, multivariate REs, ODE and non-ODE models, constants/constants_re, threaded batches.

Outputs: `FOCEIResult` includes EB modes and fallback counters in notes.

Tests: `test/estimation_focei_tests.jl`

#### FOCEIMAP

Same approximation path as FOCEI but adds fixed-effect priors to the objective. Requires priors for all fixed effects.

Tests: `test/estimation_focei_map_tests.jl`

#### MCEM (Monte Carlo EM)

```julia
fit_model(dm, MCEM(; sampler, turing_kwargs, optimizer, optim_kwargs, adtype,
                     maxiters, rtol_theta, atol_theta, rtol_Q, atol_Q, ...))
```

Core idea: E-step samples random effects per batch with MCMC (Turing); M-step maximizes MC approximation of Q(θ) via Optimization.jl. EBEs computed via Laplace-style after fitting.

Capabilities: multiple RE groups, multivariate REs, ODE models, constants/constants_re, threaded E-step, optimizer choices (LBFGS, Adam, BlackBoxOptim).

Notes: priors on fixed effects are ignored (info message emitted). BlackBoxOptim requires finite bounds.

Tests: `test/estimation_mcem_tests.jl`

#### SAEM (Stochastic Approximation EM)

```julia
fit_model(dm, SAEM(; sampler, turing_kwargs, optimizer, optim_kwargs, adtype,
                     maxiters, t0, kappa, max_store, update_schedule,
                     rtol_theta, atol_theta, rtol_Q, atol_Q, consecutive_params,
                     builtin_stats, builtin_mean, ...))
```

Core idea: E-step batchwise MCMC; Q update via Robbins–Monro recursion with step size γ_t; M-step via Optimization.jl or closed-form. EBEs computed via Laplace-style after fitting.

Default options: `update_schedule=:all`, `warm_start=true`, `max_store=50`, `t0=20`, `kappa=0.65`, `maxiters=300`, `consecutive_params=4`.

Built-in sufficient statistics:
- `builtin_stats=:gaussian_re` — updates Gaussian RE variance (scalar and multivariate) and optional Normal outcome residual variance.
- `builtin_mean=:glm` — updates mean parameters for GLM-style outcomes: Normal, Bernoulli, Poisson, Exponential.

Capabilities: multiple RE groups, multivariate REs, ODE models, constants/constants_re, threaded E-step and Q evaluation.

Tests: `test/estimation_saem_tests.jl`, `test/estimation_saem_suffstats_tests.jl`

#### MCMC with Random Effects

Full Bayesian path sampling both fixed effects and RE levels via Turing. Requires priors on all free fixed effects.

Capabilities: multiple RE groups, multivariate REs, non-Gaussian REs, flow REs, ODE models, NN/SoftTree/Spline model components, HMM outcomes, threaded likelihood.

Notes: parameter scale settings (log/cholesky) are ignored; priors are on natural scale.

Tests: `test/estimation_mcmc_re_tests.jl`

### 2.4 Multistart Estimation

```julia
fit_model(dm, Multistart(base_method; n_starts, sampling, ...))
```

- `Multistart` wrapper supports multiple start generation and refits.
- Sampling modes: `:random` and `:lhs` (Latin Hypercube Sampling).
- Priors or explicit sampling distributions can drive starts.
- Accessors: `get_multistart_results`, `get_multistart_best`, `get_multistart_errors`, `get_multistart_starts`, etc.

### 2.5 Constants and Bounds Summary

- `constants` — fixed effects on transformed scale.
- `constants_re` — fixed RE levels (NamedTuple / Dict / Pair supported).
- Bounds (lb/ub) — used in Laplace/LaplaceMAP/FOCEI/FOCEIMAP/MCEM/SAEM; BlackBoxOptim requires finite bounds.

### 2.6 Threading Summary

- Laplace: objective/grad across batches uses threads.
- FOCEI/FOCEIMAP: objective/grad across batches uses threads.
- MCMC: loglikelihood can use `EnsembleThreads`.
- MCEM/SAEM: batch sampling and Q evaluation can use threads.

### 2.7 Identifiability Diagnostics

```julia
identifiability_report(dm::DataModel; ...)
identifiability_report(res::FitResult; ...)
```

Reports Hessian/SVD rank, null directions, condition metrics, and RE information diagnostics. Supported for MLE/MAP/Laplace/FOCEI families.

### 2.8 Fit Result Accessors

By default fit results store the `DataModel` (disable with `fit_model(...; store_data_model=false)`). When stored, accessors that normally require `dm` as the first argument also accept a single-argument form that reads it from the result.

Accessors throw an informative error when the method does not define the requested field (e.g. `get_chain` on MLE, or `get_loglikelihood` on MCMC).

```julia
# Core (FitResult)
get_params(res; scale=:transformed|:untransformed|:both)
get_objective(res)
get_converged(res)
get_diagnostics(res)
get_summary(res)
get_method(res)
get_result(res)
get_data_model(res)

# Optimization-based methods (MLE/MAP/Laplace/LaplaceMAP/FOCEI/FOCEIMAP/MCEM/SAEM)
get_iterations(res)
get_raw(res)
get_notes(res)

# MCMC-specific
get_chain(res)
get_observed(res)
get_sampler(res)
get_n_samples(res)

# Random effects — Laplace/LaplaceMAP/FOCEI/FOCEIMAP/MCEM/SAEM (EB point estimates)
get_random_effects(dm, res; constants_re=NamedTuple(), flatten=true, include_constants=true)
get_random_effects(res; constants_re=..., flatten=..., include_constants=...)  # uses stored dm

# Log-likelihood — MLE/MAP/Laplace/LaplaceMAP/FOCEI/FOCEIMAP/MCEM/SAEM (EB modes used for RE methods)
get_loglikelihood(dm, res; constants_re=NamedTuple(), ode_args=(), ode_kwargs=NamedTuple(), serialization=EnsembleSerial())
get_loglikelihood(res; constants_re=..., ode_args=..., ode_kwargs=..., serialization=...)  # uses stored dm
```

Multistart results route all core accessors to the best run automatically. See section 2.4 for multistart-specific accessors.

---

## 3. Uncertainty Quantification

### 3.1 Unified API

```julia
uq = compute_uq(res::FitResult; method=:auto, level=0.95,
                constants, constants_re, penalty,
                ode_args, ode_kwargs, serialization, rng, ...)
```

Active parameters: fixed-effect coordinates that are both free (not in `constants`) and marked `calculate_se=true`.

Backend selection (`method`):
- `:auto` → `:chain` for MCMC fits, `:wald` otherwise; `:profile` if `interval=:profile`.
- Explicit: `:wald`, `:chain`, `:profile`, `:mcmc_refit`.

### 3.2 Backend Capabilities

#### Wald (`method=:wald`)

Supported source methods: `MLE`, `MAP`, `Laplace`, `LaplaceMAP`, `FOCEI`, `FOCEIMAP`, `MCEM`, `SAEM`.

Core behavior:
- Builds objective around fitted point and computes transformed-scale Hessian.
- Covariance options: `:hessian` (inverse Hessian) or `:sandwich`.
- Samples draws on transformed scale; maps through inverse transform for natural scale.
- Intervals: quantile-based from draws.

Key kwargs: `vcov=:hessian|:sandwich`, `hessian_backend=:auto|:forwarddiff|:fd_gradient`, `pseudo_inverse=false`, `n_draws=2000`.

For `MCEM`/`SAEM`: `re_approx=:auto|:laplace|:focei`, `re_approx_method`.

#### Chain (`method=:chain`)

Supported source method: `MCMC` only.

Pulls post-warmup chain samples for active coordinates. Computes estimates, intervals, and covariance directly from draws. Intervals are equal-tail quantile intervals.

Key kwargs: `mcmc_warmup`, `mcmc_draws`.

#### Profile (`method=:profile`)

Supported source methods: `MLE`, `MAP`, `Laplace`, `LaplaceMAP`, `FOCEI`, `FOCEIMAP`.

Uses `LikelihoodProfiler.jl` per active coordinate. Profiles objective with all other coordinates re-optimized. No covariance matrix output.

Key kwargs: `profile_scan_width`, `profile_scan_tol`, `profile_loss_tol`, `profile_max_iter`.

#### MCMC Refit (`method=:mcmc_refit`)

Supported source methods: non-MCMC fits.

Constructs/uses an MCMC method and refits from the fitted point. Automatically fixes non-UQ fixed effects. Then applies chain-based UQ extraction.

Key kwargs: `mcmc_method`, `mcmc_sampler`, `mcmc_turing_kwargs`, `mcmc_adtype`, `mcmc_warmup`, `mcmc_draws`.

### 3.3 Scale and Transform Semantics

- Transformed scale: optimizer parameterization (`log`, `cholesky`, `expm`, etc.).
- Natural scale: model parameterization used in distributions and ODE formulas.
- Wald Gaussian approximation is defined on transformed scale; natural-scale induced through nonlinear transforms and represented through draws.

### 3.4 UQ Accessors

```julia
get_uq_backend(uq)
get_uq_source_method(uq)
get_uq_parameter_names(uq)
get_uq_estimates(uq; scale=:natural|:transformed, as_component=true)
get_uq_intervals(uq; scale=:natural|:transformed, as_component=true)
get_uq_vcov(uq; scale=:natural|:transformed)
get_uq_draws(uq; scale=:natural|:transformed)
get_uq_diagnostics(uq)
```

### 3.5 Minimal Usage Examples

```julia
# Wald UQ (auto for non-MCMC)
uq = compute_uq(res; method=:wald, vcov=:hessian, n_draws=2000)

# Chain UQ from an MCMC fit
uq_chain = compute_uq(res_mcmc; method=:chain, mcmc_draws=1000)

# Profile-likelihood intervals
uq_prof = compute_uq(res; method=:profile, profile_scan_width=2.0)

# MCMC refit UQ from an optimization fit
uq_refit = compute_uq(res; method=:mcmc_refit,
                      mcmc_turing_kwargs=(n_samples=1000, n_adapt=500, progress=false))
```

---

## 4. Plotting and Diagnostics

### 4.1 Core Types and Cache

#### PlotStyle

```julia
Base.@kwdef struct PlotStyle
    color_primary::String    = "#0173B2"    # Blue  - main data/lines
    color_secondary::String  = "#029E73"    # Green - fitted values/predictions
    color_accent::String     = "#DE8F05"    # Orange - highlights/density
    color_dark::String       = "#2C3E50"    # Dark blue-gray - text/references
    font_family::String          = "Helvetica"
    font_size_title::Int         = 11
    font_size_label::Int         = 10
    font_size_tick::Int          = 9
    font_size_legend::Int        = 8
    line_width_primary::Float64    = 2.0
    line_width_secondary::Float64  = 1.5
    marker_size::Int               = 5
    marker_alpha::Float64          = 0.7
    base_subplot_width::Int  = 350
    base_subplot_height::Int = 280
end
```

All plotting functions accept `style::PlotStyle`. Common kwargs accepted by most functions:

| Kwarg | Default | Purpose |
|---|---|---|
| `ncols` | `3` | Grid columns |
| `shared_x_axis` | `true` | Uniform x-axis limits |
| `shared_y_axis` | `true` | Uniform y-axis limits |
| `style` | `PlotStyle()` | Full style customization |
| `kwargs_subplot` | `NamedTuple()` | Override individual subplot options |
| `kwargs_layout` | `NamedTuple()` | Override layout (size, dpi, etc.) |
| `save_path` | `nothing` | File path to save (`.png`, `.pdf`, `.svg`, `.eps`) |

#### PlotCache and build_plot_cache

```julia
build_plot_cache(res::FitResult; dm=nothing, params=NamedTuple(),
                 constants_re=NamedTuple(), cache_obs_dists=false,
                 ode_args=(), ode_kwargs=NamedTuple(), mcmc_draws=1000,
                 mcmc_warmup=nothing, rng=Random.default_rng())
build_plot_cache(res::MultistartFitResult; kwargs...)
build_plot_cache(dm::DataModel; params=NamedTuple(), constants_re=NamedTuple(),
                 cache_obs_dists=false, ...)
```

Caches ODE solutions, observation distributions, chain summaries, and RE summaries for reuse across multiple plot calls.

### 4.2 Data and Fit Plots

#### plot_data
```julia
plot_data(res::FitResult; dm=nothing, x_axis_feature=nothing,
          individuals_idx=nothing, shared_x_axis=true, shared_y_axis=true,
          ncols=3, style=PlotStyle(), kwargs_subplot=NamedTuple(),
          kwargs_layout=NamedTuple(), save_path=nothing)
plot_data(dm::DataModel; ...)
```
Scatter plot of observed data, one panel per individual. X-axis defaults to time or `x_axis_feature`.

#### plot_fits
```julia
plot_fits(res::FitResult; dm=nothing, plot_density=false, plot_func=mean,
          plot_data_points=true, observable=nothing, individuals_idx=nothing,
          x_axis_feature=nothing, shared_x_axis=true, shared_y_axis=true, ncols=3,
          style=PlotStyle(), kwargs_subplot=NamedTuple(), kwargs_layout=NamedTuple(),
          save_path=nothing, cache=nothing, params=NamedTuple(),
          constants_re=NamedTuple(), cache_obs_dists=false,
          plot_mcmc_quantiles=false, mcmc_quantiles=[5,95], mcmc_quantiles_alpha=0.8,
          mcmc_draws=1000, mcmc_warmup=nothing, rng=Random.default_rng())
plot_fits(dm::DataModel; ...)
```
Overlay observed points and fitted summary curve. `plot_func` (default `mean`) is applied to each predicted observation distribution. MCMC mode supports posterior bands via `plot_mcmc_quantiles=true`. If multiple observables exist and `observable=nothing`, first is used (with warning).

#### plot_fits_comparison

Overlays trajectories from multiple fits on the same panels.

### 4.3 Observation Distribution Plots

```julia
plot_observation_distributions(res::FitResult; dm=nothing, individuals_idx=nothing,
                                obs_rows=nothing, observables=nothing,
                                x_axis_feature=nothing, shared_x_axis=true,
                                shared_y_axis=true, ncols=3, style=PlotStyle(),
                                cache=nothing, cache_obs_dists=false,
                                constants_re=NamedTuple(), mcmc_quantiles=[5,95],
                                mcmc_quantiles_alpha=0.8, mcmc_draws=1000,
                                mcmc_warmup=nothing, rng=Random.default_rng(),
                                save_path=nothing, kwargs_subplot=NamedTuple(),
                                kwargs_layout=NamedTuple())
plot_observation_distributions(dm::DataModel; kwargs...)
```

One panel per selected observation row and observable. Shows predicted observation distribution (density for continuous, PMF bars for discrete) with observed value marker. MCMC: posterior mean curve with quantile envelopes.

### 4.4 Residual Diagnostics

#### get_residuals
```julia
get_residuals(res::FitResult; dm=nothing, cache=nothing, observables=nothing,
              individuals_idx=nothing, obs_rows=nothing, x_axis_feature=nothing,
              params=NamedTuple(), constants_re=NamedTuple(),
              residuals=[:quantile,:pit,:raw,:pearson,:logscore], fitted_stat=mean,
              randomize_discrete=true, cdf_fallback_mc=0, ode_args=(),
              ode_kwargs=NamedTuple(), mcmc_draws=1000, mcmc_warmup=nothing,
              mcmc_quantiles=[5,95], rng=Random.default_rng(), return_draw_level=false)
```
Returns a `DataFrame` with residual metrics: `:pit`, `:quantile`, `:raw`, `:pearson`, `:logscore`.

#### plot_residuals
Scatter of selected residual metric vs x-axis, grouped by individual and observable.

#### plot_residual_distribution
Histogram per observable for selected residual metric.

#### plot_residual_qq
QQ plots per observable. For `:pit`, compares to Uniform quantiles; for non-`:pit`, compares to Normal quantiles.

#### plot_residual_pit
PIT diagnostics per observable. Options: `show_hist=true`, `show_kde=false`, `show_qq=false`.

#### plot_residual_acf
Residual autocorrelation by observable (average across individuals), up to `max_lag`.

### 4.5 Random-Effects Diagnostics

Supported fit types: `Laplace`, `LaplaceMAP`, `FOCEI`, `FOCEIMAP`, `MCEM`, `SAEM`, `MCMC`.

#### plot_random_effect_distributions
Per-level marginal RE distribution with EBE marker (optimization methods) or posterior mean marker (MCMC). For `NormalizingPlanarFlow`, uses sampling and KDE approximations.

#### plot_random_effect_pit
PIT diagnostics of RE values under RE distributions. Flow RE PIT uses empirical CDF from samples.

#### plot_random_effect_standardized
Standardized EBE diagnostics (z-scale), histogram and/or KDE.

#### plot_random_effect_standardized_scatter
Scatter of standardized RE values vs level/index or constant covariate.

#### plot_random_effects_pdf
Marginal RE PDFs (only distributions without covariate dependence). Shows PDFs without per-level EBE markers.

#### plot_random_effects_scatter
Scatter of EBE/posterior mean vs level/index or covariate (only distributions without covariate dependence).

#### plot_random_effect_pairplot
Pair plots grouped by RE grouping column. Diagonal: histogram. Off-diagonal: EBE/posterior mean scatter.

### 4.6 Visual Predictive Check

```julia
plot_vpc(res::FitResult; dm=nothing, n_simulations=100, percentiles=[5,50,95],
         show_obs_points=true, show_obs_percentiles=true, n_bins=nothing,
         seed=12345, observables=nothing, x_axis_feature=nothing, ncols=3,
         obs_percentiles_mode=:pooled, obs_percentiles_method=:quantile,
         bandwidth=nothing, constants_re=NamedTuple(), mcmc_draws=1000,
         mcmc_warmup=nothing, style=PlotStyle())
```

Supports continuous and discrete outcomes. For MCMC, simulations are driven by posterior draws. `obs_percentiles_method`: `:quantile` or `:kernel`. `obs_percentiles_mode`: `:pooled` or `:per_individual`.

### 4.7 Uncertainty Quantification Plots

```julia
plot_uq_distributions(uq::UQResult;
                      scale=:natural, parameters=nothing,
                      plot_type=:density,  # :density or :histogram
                      show_estimate=true, show_interval=true,
                      kde_bandwidth=nothing, ncols=3, style=PlotStyle(), ...)
```

For `:chain`/`:mcmc_refit` backends: KDE from draws. For `:wald`: closed-form Gaussian/LogNormal where available, otherwise sampling + KDE fallback.

### 4.8 Multistart Plots

```julia
plot_multistart_waterfall(res::MultistartFitResult; ncols=3, style=PlotStyle(), ...)
plot_multistart_fixed_effect_variability(res::MultistartFitResult; ...)
```

Waterfall: sorted objective values across starts. Variability: parameter variation across starts.

---

## 5. Simulation

```julia
simulate_data(dm::DataModel; rng=Random.default_rng(),
              replace_missings=true, serialization=EnsembleSerial()) -> DataFrame

simulate_data_model(dm::DataModel; rng=Random.default_rng(), ...) -> DataModel
```

Supports simulation with RE draws, ODE solves, event handling, and flexible distributions in formulas.

---

## 6. Cross-Cutting Notes

- `ComponentArray` is used end-to-end in estimation paths.
- ForwardDiff buffers/configs are cached in key gradient paths for allocation/performance improvements.
- Threaded execution is supported in major random-effects estimation workflows via serialization controls.
- Distribution flexibility is broad at the modeling layer for both outcomes and random effects. Some approximation methods have narrower analytic assumptions:
  - FOCEI `info_mode=:fisher_common` supports a defined subset of outcomes/RE priors.
  - FOCEI `info_mode=:custom` expands to arbitrary distributions through user-supplied information approximation.
- Bayesian and simulation paths are generally the most distribution-flexible, subject to valid `logpdf`/`rand` behavior and stable autodiff.

---

## 7. Key Test Files

| Area | Test Files |
|---|---|
| Model / DataModel | `test/model_tests.jl`, `test/data_model_tests.jl`, `test/data_model_ode_tests.jl` |
| RE distribution / AD | `test/random_effects_tests.jl`, `test/ad_random_effects.jl` |
| Laplace | `test/estimation_laplace_tests.jl`, `test/estimation_laplace_fit_tests.jl` |
| LaplaceMAP | `test/estimation_laplace_map_tests.jl` |
| FOCEI | `test/estimation_focei_tests.jl` |
| FOCEIMAP | `test/estimation_focei_map_tests.jl` |
| MCEM | `test/estimation_mcem_tests.jl` |
| SAEM | `test/estimation_saem_tests.jl`, `test/estimation_saem_suffstats_tests.jl` |
| MCMC RE | `test/estimation_mcmc_re_tests.jl` |
| MLE / MAP | `test/estimation_mle_tests.jl`, `test/estimation_map_tests.jl` |
| Multistart | `test/estimation_multistart_tests.jl` |
| UQ | `test/uq_tests.jl`, `test/uq_edge_cases_tests.jl`, `test/uq_plotting_tests.jl` |
| Plotting | `test/plotting_functions_tests.jl`, `test/plot_random_effects_tests.jl`, `test/vpc_tests.jl` |
| Simulation | `test/data_simulation_tests.jl` |
