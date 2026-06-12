# Mixed-Effects Tutorial 2: ODE Model with Input Events (MCEM)

Many longitudinal studies involve systems whose dynamics are governed by ordinary differential equations (ODEs) with discrete input events --- a bolus injection, a nutrient pulse, or a stimulus onset. When individuals differ in their dynamic parameters, nonlinear mixed-effects models provide a principled way to separate population-level trends from subject-level variability. In this tutorial, you will build such a model from scratch and fit it using the Monte Carlo Expectation-Maximization (MCEM) algorithm, a method well-suited to problems where random effects enter the model nonlinearly.

Starting from a publicly available concentration-time dataset (Theophylline), you will learn how to prepare event-aware data, define a two-compartment ODE model with subject-level random effects, run the full estimation workflow, and generate publication-quality diagnostics. The structural model describes a two-compartment transfer system (`depot`, `center`) in which material moves from a depot compartment into a central compartment and is subsequently eliminated. Subject-level random effects on the transfer rate (`ka`), elimination rate (`cl`), and distribution volume (`v`) capture between-individual variability. Although this tutorial uses concentration-time data, the workflow generalizes to any domain where compartmental ODE dynamics with discrete input events arise --- tracer kinetics in imaging, nutrient uptake in ecology, or substrate processing in bioprocess engineering.

## What You Will Learn

By the end of this tutorial, you will be able to:

- **Prepare event-aware data** --- convert a raw longitudinal table into the event format NoLimits.jl expects for ODE models with discrete inputs (bolus injections, perturbations, or similar events).
- **Define a nonlinear mixed-effects ODE model** --- use the `@preDifferentialEquation` and `@DifferentialEquation` blocks to specify a model whose random effects enter nonlinearly, a setting where MCEM is especially useful.
- **Fit the model with MCEM** --- run the estimation and inspect core diagnostics including the objective value and estimated parameters.
- **Visualize and assess results** --- generate fitted trajectory plots, observation-distribution diagnostics, and Wald-based uncertainty quantification summaries.

## Step 1: Prepare the Event-Table

In this step, you will convert raw longitudinal data into the event-table format that NoLimits.jl uses for ODE models with discrete inputs. An event table interleaves input events (such as a bolus entering a compartment) with observation rows, so that the ODE solver knows when and where to apply external perturbations. The key columns are:

- `EVID` --- an integer flag distinguishing input events (`1`) from observation rows (`0`).
- `AMT` --- the magnitude of the input event (e.g., total amount delivered).
- `CMT` --- the name of the ODE compartment that receives the input.
- `RATE` --- the infusion rate; `0.0` indicates an instantaneous bolus.

The code below loads the Theophylline dataset --- a widely used concentration-time dataset recording twelve individuals over time --- and reshapes it into this event-table format. Each individual receives a single bolus input at time zero, followed by a series of concentration observations.

```julia
using NoLimits
using CSV
using DataFrames
using Distributions
using Downloads
using Random
using LinearAlgebra
using OrdinaryDiffEq
using SciMLBase

include(joinpath(@__DIR__, "_data_loaders.jl"))

Random.seed!(123)

theoph_df = load_theoph()

function build_theoph_event_df(tbl::DataFrame)
    df = DataFrame(
        id=Int[],
        t=Float64[],
        AMT=Float64[],
        EVID=Int[],
        CMT=Union{String, Missing}[],
        RATE=Float64[],
        y1=Union{Float64, Missing}[],
        _event_order=Int[],
    )

    for g in groupby(tbl, :Subject)
        id = Int(first(g.Subject))
        amt = Float64(first(g.Wt)) * Float64(first(g.Dose))

        push!(df, (id, 0.0, amt, 1, "depot", 0.0, missing, 0))

        g_sorted = sort(DataFrame(g), :Time)
        for row in eachrow(g_sorted)
            push!(df, (id, Float64(row.Time), 0.0, 0, missing, 0.0, Float64(row.conc), 1))
        end
    end

    sort!(df, [:id, :t, :_event_order])
    select!(df, Not(:_event_order))
    return df
end

df = build_theoph_event_df(theoph_df)
first(df, 12)
```

<!-- injected:t2-dfhead -->
```text
12×7 DataFrame
 Row │ id     t        AMT      EVID   CMT      RATE     y1
     │ Int64  Float64  Float64  Int64  String?  Float64  Float64?
─────┼──────────────────────────────────────────────────────────────
   1 │     1     0.0   319.992      1  depot        0.0  missing
   2 │     1     0.0     0.0        0  missing      0.0        0.74
   3 │     1     0.25    0.0        0  missing      0.0        2.84
   4 │     1     0.57    0.0        0  missing      0.0        6.57
   5 │     1     1.12    0.0        0  missing      0.0       10.5
   6 │     1     2.02    0.0        0  missing      0.0        9.66
   7 │     1     3.82    0.0        0  missing      0.0        8.58
   8 │     1     5.1     0.0        0  missing      0.0        8.36
   9 │     1     7.03    0.0        0  missing      0.0        7.47
  10 │     1     9.05    0.0        0  missing      0.0        6.89
  11 │     1    12.12    0.0        0  missing      0.0        5.94
  12 │     1    24.37    0.0        0  missing      0.0        3.28
```

## Step 2: Define the ODE Mixed-Effects Model

In this step, you will specify the mechanistic model. The goal is twofold: describe the ODE dynamics of the two-compartment system, and account for inter-individual variability through random effects.

The model has three system parameters --- a transfer rate `ka`, an elimination rate `cl`, and a distribution volume `v` --- each of which varies across individuals. To ensure positivity, each parameter is expressed as an exponential transform of a population-level fixed effect plus a subject-specific random deviation:

- `ka = exp(tka + eta[1])`
- `cl = exp(tcl + eta[2])`
- `v = exp(tv + eta[3])`

Because the parameters enter the ODE nonlinearly (through the exponential transform of the random effects), standard linear mixed-effects approximations do not apply. This is precisely the setting where Monte Carlo EM methods such as MCEM offer a robust estimation strategy.

The random effects vector `eta` follows a multivariate normal distribution with a diagonal covariance matrix whose entries (`omega1`, `omega2`, `omega3`) are estimated alongside the other fixed effects. Observations are modeled as normally distributed around the predicted concentration in the central compartment, with residual standard deviation `sigma_eps`.

After defining the model, you will configure the ODE solver. `Tsit5()` is an efficient explicit Runge--Kutta method well-suited to non-stiff systems, and the tolerances below balance numerical accuracy with computational cost.

```julia
using NoLimits
using Distributions
using LinearAlgebra
using OrdinaryDiffEq

model_raw = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        tka = RealNumber(0.45, prior=Uniform(0.1, 5.0), calculate_se=true)
        tcl = RealNumber(1.0, prior=Uniform(0.1, 5.0), calculate_se=true)
        tv = RealNumber(3.45, prior=Uniform(0.1, 5.0), calculate_se=true)

        omega1 = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)
        omega2 = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)
        omega3 = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)

        sigma_eps = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)
    end

    @randomEffects begin
        eta = RandomEffect(
            MvNormal([0.0, 0.0, 0.0], Diagonal([omega1, omega2, omega3]));
            column=:id,
        )
    end

    @preDifferentialEquation begin
        ka = exp(tka + eta[1])
        cl = exp(tcl + eta[2])
        v = exp(tv + eta[3])
    end

    @DifferentialEquation begin
        D(depot) ~ -ka * depot
        D(center) ~ ka * depot - cl / v * center
    end

    @initialDE begin
        depot = 0.0
        center = 0.0
    end

    @formulas begin
        y1 ~ Normal(center(t) / v, sigma_eps)
    end
end

model = set_solver_config(
    model_raw;
    saveat_mode=:saveat,
    alg=Tsit5(),
    kwargs=(abstol=1e-6, reltol=1e-6),
)
```

### Model Summary

Before moving on, it is good practice to inspect the model structure with `NoLimits.summarize`. This confirms that all blocks --- fixed effects, random effects, covariates, ODE states, and formulas --- have been parsed correctly.

```julia
model_summary = NoLimits.summarize(model)
model_summary
```

<!-- injected:t2-model -->
```text
ModelSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  model type                          : ODE
  fixed-effect blocks                 : 7
  fixed-effect scalar values          : 7
  random effects                      : 1
  random-effect grouping columns      : 1
  covariates (declared)               : 1
  formulas (deterministic / outcomes) : 0 / 1
  requires DE accessors               : true

Structure blocks
  helpers              : false
  fixed effects        : true
  random effects       : true
  covariates           : true
  preDE                : true
  DifferentialEquation : true
  initialDE            : true

Covariate classes
  varying  : 1
  constant : 0
  dynamic  : 0

Fixed-effects declarations
  name       type        size  se  prior    scale     bounds                              details
  ------------------------------------------------------------------------------------------------------------
  tka        RealNumber     1  yes  Uniform  identity  finite lower 0/1, finite upper 0/1  -
  tcl        RealNumber     1  yes  Uniform  identity  finite lower 0/1, finite upper 0/1  -
  tv         RealNumber     1  yes  Uniform  identity  finite lower 0/1, finite upper 0/1  -
  omega1     RealNumber     1  yes  Uniform  log       finite lower 1/1, finite upper 0/1  -
  omega2     RealNumber     1  yes  Uniform  log       finite lower 1/1, finite upper 0/1  -
  omega3     RealNumber     1  yes  Uniform  log       finite lower 1/1, finite upper 0/1  -
  sigma_eps  RealNumber     1  yes  Uniform  log       finite lower 1/1, finite upper 0/1  -

Random-effects declarations
  name  group  dist    
  -----------------------
  eta   id     MvNormal

Covariate declarations
  name  kind       columns                   constant_on           interpolation
  ---------------------------------------------------------------------------------------
  t     Covariate  t                         -                     -

Formulas
  deterministic names : (none)
  outcome names       : y1
  required DE states  : center
  required DE signals : (none)
  declared DE states  : depot, center
  declared DE signals : (none)
Outcome distribution types
  y1 => Normal

Helper functions
  names : (none)
```

## Step 3: Build the DataModel

In this step, you will connect the model to the data by constructing a `DataModel`. This constructor validates the dataset against the model's requirements --- checking that all declared covariates are present, that constant covariates are indeed constant within each group, and that event columns are well-formed. You must explicitly specify the event-related column names so that NoLimits.jl can correctly distinguish input events from observation rows during ODE integration.

```julia
dm = DataModel(
    model,
    df;
    primary_id=:id,
    time_col=:t,
    evid_col=:EVID,
    amt_col=:AMT,
    rate_col=:RATE,
    cmt_col=:CMT,
)

```

### DataModel Summary

Summarizing the `DataModel` gives you an overview of the number of individuals, batches (groups of individuals linked by shared random-effect levels), observation counts, and event structure. This is a useful checkpoint before launching a potentially expensive estimation run.

```julia
dm_summary = NoLimits.summarize(dm)
dm_summary
```

<!-- injected:t2-dm -->
```text
DataModelSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  model type                 : ODE
  event-aware                : true
  individuals                : 12
  rows (total / obs / event) : 144 / 132 / 12
  fixed effects (top-level)  : 7
  outcomes                   : 1
  covariates (declared)      : 1
  random effects             : 1

Covariate classes
  varying  : 1
  constant : 0
  dynamic  : 0

Outcome distribution types
  y1 => Normal

Random-effect distribution types
  eta => MvNormal

Individual design diagnostics
  individuals with one observation              : 0
  global observed time range                    : 0.0 to 24.65
  unique observed time points                   : 78
  duplicate (ID, time) observation rows         : 0
  monotonic-time violations (observation order) : 0

Observations per individual
  metric       n          mean            sd           min           q25        median           q75           max
  ----------------------------------------------------------------------------------------------------------------
  count       12          11.0           0.0          11.0          11.0          11.0          11.0          11.0

Time span per individual
  metric       n          mean            sd           min           q25        median           q75           max
  ----------------------------------------------------------------------------------------------------------------
  span        12       24.1992        0.2439          23.7         24.11        24.195        24.355         24.65

Median sampling interval per individual
  metric          n          mean            sd           min           q25        median           q75           max
  -------------------------------------------------------------------------------------------------------------------
  median_dt      12        1.5092        0.0277         1.445        1.4975        1.5075        1.5312          1.55

Outcome descriptive statistics (observation rows)
  Variable       n          mean            sd           min           q25        median           q75           max
  ------------------------------------------------------------------------------------------------------------------
  y1           132        4.9605        2.8564           0.0        2.8775         5.275          7.14          11.4

Declared covariates
  name  kind       columns
  -------------------------------------
  t     Covariate  t

Covariate descriptive statistics (observation rows)
  Variable       n          mean            sd           min           q25        median           q75           max
  ------------------------------------------------------------------------------------------------------------------
  t.t          132        5.8946        6.8997           0.0         0.595          3.53           9.0         24.65

Per-random-effect summary
  random effect  group  dist        levels  rows/level min        median           max
  ----------------------------------------------------------------------------------
  eta            id     MvNormal        12            11.0          11.0          11.0
```

## Step 4: Configure MCEM

In this step, you will set up the MCEM algorithm. MCEM alternates between two stages: (1) an E-step that samples random effects from their conditional posterior given the current fixed-effect estimates, and (2) an M-step that maximizes the expected complete-data log-likelihood with respect to the fixed effects. The number of Monte Carlo samples in the E-step can grow across iterations, improving the approximation as the algorithm converges.

The configuration below is tuned for a tutorial setting --- moderate iteration counts and sample sizes that keep runtime reasonable. For production analyses, you would typically increase `maxiters`, raise the ceiling of the sample schedule, and draw more MCMC samples per E-step to reduce Monte Carlo noise in the gradient estimates.

A few notes on the specific settings:

- `sample_schedule` controls how the number of Monte Carlo samples grows with each EM iteration, starting at 60 and increasing by 20 per iteration up to a maximum of 260.
- `progress=false` suppresses progress bars to keep documentation output clean.
- `EnsembleThreads()` enables multithreaded ODE solving across individuals for faster execution.

```julia
mcem_method = NoLimits.MCEM(;
    maxiters=12,
    sample_schedule=i -> min(60 + 20 * (i - 1), 260),
    turing_kwargs=(n_samples=60, n_adapt=20, progress=false),
    optim_kwargs=(maxiters=220,),
    progress=false,
)

serialization = SciMLBase.EnsembleThreads()
```

## Step 5: Fit the Model and Inspect Core Outputs

With everything in place, you can now run the fit. The call to `fit_model` performs the full MCEM optimization loop and returns a result object containing parameter estimates, diagnostics, and metadata.

After fitting, you will extract the final objective value --- the marginal log-likelihood approximation at convergence. This number is useful for comparing models or monitoring optimization progress, but model adequacy is best judged through the predictive diagnostics covered in the next steps.

```julia
res_mcem = fit_model(
    dm,
    mcem_method;
    serialization=serialization,
    rng=Random.Xoshiro(33),
)

fit_summary = (
    objective=NoLimits.get_objective(res_mcem),
)

fit_summary
```

<!-- injected:t2-obj -->
```text
(objective = -145.18060999648057,)
```

### FitResult Summary

The structured summary provides a convenient overview of convergence status, iteration counts, and method-specific diagnostics.

```julia
fit_result_summary = NoLimits.summarize(res_mcem)
fit_result_summary
```

<!-- injected:t2-fitres -->
```text
FitResultSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  method                              : mcem
  inference                           : frequentist
  scale                               : natural
  objective                           : -145.1806
  iterations                          : 12
  parameters shown (reported / total) : 7 / 7

Parameter estimates
  parameter      Estimate
  -----------------------
  tka              0.4527
  tcl              1.0076
  tv               3.4566
  omega1           0.4089
  omega2           0.0708
  omega3            0.019
  sigma_eps        0.6943

Outcome data coverage
  outcome       n_obs   n_missing
  -------------------------------
  y1              132          12
  TOTAL           132          12

Empirical Bayes random effects summary (across RE levels)
  random effect  component       n          mean            sd           q25        median           q75
  --------------------------------------------------------------------------------------------------
  eta            eta_1          12       -0.0011        0.5975       -0.4122       -0.0957        0.2514
  eta            eta_2          12        0.0078        0.2431       -0.1275         0.041        0.1565
  eta            eta_3          12       -0.0048        0.1182       -0.1011        0.0051        0.0658
```

You can also extract the estimated fixed-effect parameters on their natural (untransformed) scale. The population-level log-scale parameters (`tka`, `tcl`, `tv`) can be exponentiated to recover typical-individual values, and `sigma_eps` represents the residual observation noise.

```julia
params = NoLimits.get_params(res_mcem; scale=:untransformed)
(
    tka=params.tka,
    tcl=params.tcl,
    tv=params.tv,
    sigma_eps=params.sigma_eps,
)
```

<!-- injected:t2-params -->
```text
(tka = 0.452678124367957, tcl = 1.0076137236747467, tv = 3.456574166252174, sigma_eps = 0.694302232373099)
```

## Step 6: Visualize Fitted Trajectories

A natural first diagnostic is to overlay the model's predicted trajectories on the observed data. In this step, the `plot_fits` function computes individual-level predictions (using empirical Bayes estimates of the random effects) and plots them alongside the raw observations. Showing the first two individuals provides a quick visual check that the model captures the characteristic rise-and-fall dynamics of the two-compartment system.

```julia
p_fit_mcem = plot_fits(
    res_mcem;
    observable=:y1,
    individuals_idx=[1, 2],
    ncols=2,
    shared_x_axis=true,
    shared_y_axis=true,
)

p_fit_mcem
```

<!-- injected:t2-pfit -->
![Fitted two-compartment trajectories for the first two individuals (MCEM).](figures/t2/p_fit_mcem.png)

## Step 7: Assess the Observation Distribution

Beyond trajectory-level checks, it is valuable to examine how well the model's predicted observation distribution matches the data at individual time points. In this step, `plot_observation_distributions` visualizes the predictive distribution for a given individual and observation row, letting you assess whether the assumed error model (here, a normal distribution with standard deviation `sigma_eps`) is well-calibrated.

```julia
p_obs_mcem = plot_observation_distributions(
    res_mcem;
    observables=:y1,
    individuals_idx=1,
    obs_rows=1,
)

p_obs_mcem
```

<!-- injected:t2-pobs -->
![Predicted observation distribution at the first observation of the first individual (MCEM).](figures/t2/p_obs_mcem.png)

## Step 8: Quantify Parameter Uncertainty

Point estimates alone are insufficient for scientific conclusions --- you also need to quantify how precisely each parameter has been determined. In this step, you will use the Wald approach, which constructs approximate confidence intervals from the curvature of the log-likelihood at the optimum (the observed information matrix). For MCEM fits, computing this curvature requires an auxiliary random-effects integration step; the Laplace approximation (`re_approx=:laplace`) serves this purpose here.

The resulting uncertainty estimates are visualized as density plots on the natural parameter scale, providing an intuitive picture of estimation precision given the available data.

```julia
uq_mcem = compute_uq(
    res_mcem;
    method=:wald,
    vcov=:hessian,
    re_approx=:laplace,
    pseudo_inverse=true,
    serialization=serialization,
    n_draws=400,
    rng=Random.Xoshiro(44),
)

p_uq_mcem = plot_uq_distributions(
    uq_mcem;
    scale=:natural,
    plot_type=:density,
    show_legend=false,
)

p_uq_mcem
```

<!-- injected:t2-puq -->
![Wald approximate parameter distributions on the natural scale (MCEM).](figures/t2/p_uq_mcem.png)

### UQ Summaries

Finally, the numerical summaries consolidate point estimates and confidence intervals into a single table --- a format suitable for reporting results in publications or for systematic comparison across candidate models.

```julia
uq_summary_mcem = NoLimits.summarize(uq_mcem)
fit_uq_summary_mcem = NoLimits.summarize(res_mcem, uq_mcem)

fit_uq_summary_mcem
```

<!-- injected:t2-fituq -->
```text
UQResultSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  backend                             : wald
  source_method                       : mcem
  inference                           : frequentist
  scale                               : natural
  objective                           : -145.1806
  interval level                      : 0.95
  parameters shown (reported / total) : 7 / 7

Parameter uncertainty summary
  parameter      Estimate    Std. Error      CI Lower      CI Upper
  ---------------------------------------------------
  tka              0.4527        0.2018        0.0776        0.8783
  tcl              1.0076        0.0922        0.8316         1.188
  tv               3.4566        0.0489        3.3578        3.5499
  omega1           0.4089         0.224        0.1818        1.0375
  omega2           0.0708         0.042        0.0287        0.1952
  omega3            0.019        0.0137        0.0069        0.0559
  sigma_eps        0.6943        0.0485        0.6099        0.7916

Outcome data coverage
  outcome       n_obs   n_missing
  -------------------------------
  y1              132          12
  TOTAL           132          12

Empirical Bayes random effects summary (across RE levels)
  random effect  component       n          mean            sd           q25        median           q75
  --------------------------------------------------------------------------------------------------
  eta            eta_1          12       -0.0011        0.5975       -0.4122       -0.0957        0.2514
  eta            eta_2          12        0.0078        0.2431       -0.1275         0.041        0.1565
  eta            eta_3          12       -0.0048        0.1182       -0.1011        0.0051        0.0658
```

## Practical Guidance

- **Objective values are optimization diagnostics, not model quality scores.** A lower objective indicates better optimization convergence, but model adequacy should be judged through predictive checks --- trajectory plots and observation-distribution diagnostics provide complementary views of fit quality.
- **Inspect trajectory and distributional diagnostics together.** `plot_fits` reveals whether the model captures the overall shape of the response, while `plot_observation_distributions` tests whether the local predictive distribution is well-calibrated. Discrepancies in either diagnostic suggest different kinds of model misspecification.
- **Scale up before changing model structure.** If the fit appears unstable or the objective has not plateaued, first try increasing `maxiters` and the sample schedule ceiling. MCEM convergence can be slow when Monte Carlo noise dominates the gradient signal, and increasing sample sizes is often more productive than modifying the structural model.
