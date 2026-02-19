# Model Capabilities

This file documents model and DataModel capabilities in NoLimits.jl. It is intended as a structured, detailed reference for generating future documentation.
All examples below are intentionally nonlinear (avoid purely linear models).

## Model Overview
- `@Model` composes:
  - `@helpers`
  - `@fixedEffects`
  - `@covariates`
  - `@randomEffects`
  - `@preDifferentialEquation`
  - `@DifferentialEquation`
  - `@initialDE`
  - `@formulas`
- `@DifferentialEquation` requires `@initialDE`.
- A model must define at least one fixed effect or random effect.

## Fixed Effects (`@fixedEffects`)
- Declares fixed parameters with bounds, transforms, and optional priors.
- Supported parameter blocks:
  - `RealNumber`, `RealVector`
  - `RealPSDMatrix`, `RealDiagonalMatrix`
  - `NNParameters`, `SoftTreeParameters`, `SplineParameters`, `NPFParameter`
- Model functions from `NNParameters`, `SoftTreeParameters`, `SplineParameters` are registered in `model_funs` and callable in formulas/DE/PreDE.

## Random Effects (`@randomEffects`)
- Declares random effects and their grouping columns.
- Each `RandomEffect(dist; column=:GROUP)` defines a group-level random effect.
- Supported distributions include common continuous families and `NormalizingPlanarFlow`.
- Identifiability check (DataModel): if the grouping column is unique per observation (no repeated levels), DataModel emits a warning about weak identifiability (possible confounding with measurement noise) and recommends validating with `identifiability_report(...)`.

## Covariates (`@covariates`)
Covariate types:
- `Covariate`, `CovariateVector`: varying covariates (value per observation).
- `ConstantCovariate`, `ConstantCovariateVector`: constant within groups.
- `DynamicCovariate`, `DynamicCovariateVector`: interpolated functions of time.

Constant covariates:
- Must be constant within `primary_id` and within all `constant_on` groups.
- `constant_on` defaults to the only RE group when a single grouping column exists.

Dynamic covariates:
- Must declare an interpolation type from DataInterpolations.jl:
  - `ConstantInterpolation`
  - `SmoothedConstantInterpolation`
  - `LinearInterpolation`
  - `QuadraticInterpolation`
  - `LagrangeInterpolation`
  - `QuadraticSpline`
  - `CubicSpline`
  - `AkimaInterpolation`
- Per-individual time must be sorted when dynamic covariates exist.
- Minimum observation count per individual:
  - `ConstantInterpolation`: 1
  - `SmoothedConstantInterpolation`: 2
  - `LinearInterpolation`: 2
  - `QuadraticInterpolation`: 3
  - `LagrangeInterpolation`: 2
  - `QuadraticSpline`: 3
  - `CubicSpline`: 3
  - `AkimaInterpolation`: 2

## PreDifferentialEquation (`@preDifferentialEquation`)
- Computes time-constant derived quantities for DEs and formulas.
- Inputs: fixed effects, random effects, constant covariates, helpers, model functions.
- Does not allow time-varying quantities.
- DataModel rejects preDE random effects not grouped by `primary_id`.

## DifferentialEquation (`@DifferentialEquation`)
- Defines ODE system: `D(x) ~ expr`, plus optional derived signals `s(t)=expr`.
- Validation rules at model construction:
  - Varying covariates (`Covariate`, `CovariateVector`) are not allowed in DEs.
  - Dynamic covariates must be called as `w(t)`.
  - Constant covariates must not be called as functions.
- DE states and derived signals can be accessed as `x1(t)` in formulas.

## InitialDE (`@initialDE`)
- Defines initial conditions for DE states.
- Inputs: fixed effects, random effects, constant covariates, helpers, model functions, preDE.

## Formulas (`@formulas`)
- Deterministic nodes: `lhs = expr`
- Observation nodes: `lhs ~ Distribution(...)`
- Supports:
  - fixed effects, random effects, preDE vars
  - constant covariates
  - varying covariates
  - dynamic covariates via `w(t)`
  - DE states/signals via `x(t)`
  - helpers and model functions

## Solver Configuration
- `set_solver_config(model; saveat_mode=:saveat|:dense|:auto, ...)`
- `:auto` resolves to `:saveat` unless non-constant offsets in formulas force dense.

## DataModel Capabilities
`DataModel(model, df; primary_id, time_col, evid_col, amt_col, rate_col, cmt_col, obs_cols, serialization)`

Validation checks:
- Required columns present; no missing in required columns.
- `primary_id` is required when multiple RE grouping columns exist; inferred otherwise.
- Constant covariates are constant within `primary_id` and within `constant_on` groups.
- RE grouping columns must be non-missing; missing values raise an informative error with guidance to drop rows or encode an explicit level.
- RE grouping columns must be constant within `primary_id` (no time-varying random effects).
- preDE only uses random effects grouped by `primary_id`.
- Dynamic covariate time sorted per individual; min obs per interpolation enforced.
- Random effect identifiability: grouping columns unique per observation trigger a warning (weak identifiability risk), not a hard error.

Events:
- If `evid_col` provided, observations are `evid == 0`, and events (nonzero) are captured as callbacks.

Batching:
- Individuals are grouped into batches by transitive overlap of RE grouping columns.

Saveat:
- `saveat_mode=:saveat` uses observation times plus any constant time offsets from formulas.
- Non-constant offsets require `:dense`.

## Examples

### 1) Basic nonlinear mixed model
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

### 2) ODE with dynamic covariate and preDE (nonlinear)
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

### 3) NN + SoftTree + Spline in preDE with multiple RE groups
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

### 4) DataModel construction
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

dm = DataModel(model, df; primary_id=:ID, time_col=:t, obs_cols=[:y])
```

### 5) No random effects (fixed effects only, nonlinear)
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

### 6) NN + SoftTree + Spline in formulas (nonlinear)
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

### 7) Planar flow random effects (nonlinear)
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

### 8) Multivariate random effects (vector, nonlinear)
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

### 9) Multiple RE grouping columns (nonlinear)
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

### 10) Constant covariates parameterize RE distributions (nonlinear)
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

### 11) Rich covariate palette (constant, varying, dynamic vectors, nonlinear)
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

### 12) ODE with multiple outcomes (nonlinear)
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

### 13) Hidden Markov Model outcome (nonlinear)
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

### 14) ODE + preDE + NN/SoftTree/Spline + dynamic covariates (nonlinear)
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

### 15) ODE with events (EVID/AMT/RATE/CMT) and saveat (nonlinear)
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
               evid_col=:EVID, amt_col=:AMT, rate_col=:RATE, cmt_col=:CMT, obs_cols=[:y])
```

### 16) DE derived signals and use in formulas (nonlinear)
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

### 17) Pairing/batching across RE grouping columns (nonlinear)
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

dm = DataModel(model, df; primary_id=:ID, time_col=:t, obs_cols=[:y])
batches = get_batches(dm)  # transitive pairing across RE groups
```
