# Fixed-Effects Tutorial 1: Nonlinear Longitudinal Model (MLE + MAP)

Many biological processes — organ growth, tumour progression, enzyme saturation — follow sigmoidal trajectories that linear regression cannot capture. This tutorial builds a logistic growth model for tree-trunk circumference data and estimates its parameters by maximum likelihood (MLE) and maximum a posteriori (MAP), showing how prior information shapes the fit. Random effects are omitted here; a [later tutorial](mixed-effects-multiple-methods.md) adds between-subject variability.

## What You Will Learn

- Translate a growth equation into a NoLimits model specification.
- Pair it with data in a `DataModel`.
- Estimate parameters by MLE and MAP, and read the effect of priors.
- Diagnose the fit with trajectory and observation-distribution plots.

## Step 1: Load the Data

The classic Orange tree growth study (Draper and Smith; R's `datasets::Orange`) records trunk circumference for five trees at seven time points each — a compact, representative example of nonlinear longitudinal data.

```julia
using NoLimits
using CairoMakie
using CSV
using DataFrames
using Distributions
using Downloads
using Random
using SciMLBase

include(joinpath(@__DIR__, "_data_loaders.jl"))

Random.seed!(202)

df = load_orange()

first(df, 8)
```

<!- injected:fe1-dfhead ->
```text
8×4 DataFrame
 Row │ rownames  Tree   age    circumference
     │ Int64     Int64  Int64  Int64
─────┼───────────────────────────────────────
   1 │        1      1    118             30
   2 │        2      1    484             58
   3 │        3      1    664             87
   4 │        4      1   1004            115
   5 │        5      1   1231            120
   6 │        6      1   1372            142
   7 │        7      1   1582            145
   8 │        8      2    118             33
```

Each row is one measurement: `Tree` identifies the individual, `age` is days since planting, and `circumference` is the trunk circumference in millimeters. The design is balanced (every tree shares the same ages), though NoLimits.jl handles unbalanced data equally well.

## Step 2: Define a Nonlinear Fixed-Effects Model

The structural model is a three-parameter logistic growth curve:

$$\mu(t) = \frac{a}{1 + \exp\!\bigl(-k\,(t - t_0)\bigr)}$$

`a` is the upper asymptote, `k` the growth-phase steepness, and `t0` the inflection point (the age of fastest growth); observations are Normal around this curve with standard deviation `σ`. Each parameter also carries a weakly-informative prior — ignored by MLE, but used by MAP in Step 4.

```julia
using NoLimits
using Distributions

model = @Model begin
    @covariates begin
        age = Covariate()
    end

    @fixedEffects begin
        a = RealNumber(200.0, prior=Normal(200.0, 50.0), calculate_se=true)
        k = RealNumber(0.005, prior=Normal(0.005, 0.005), calculate_se=true)
        t0 = RealNumber(700.0, prior=Normal(700.0, 200.0), calculate_se=true)
        σ = RealNumber(25.0, scale=:log, prior=LogNormal(log(25.0), 0.5), calculate_se=true)
    end

    @formulas begin
        μ = a / (1 + exp(-k * (age - t0)))
        circumference ~ Normal(μ, σ)
    end
end
```

`@covariates` declares `age` as a time-varying covariate (read from the column of the same name). `σ` is parameterized on the log scale (`scale=:log`) to stay positive without explicit bounds. `@formulas` computes the deterministic mean `μ`, then ties the observed `circumference` to it through a Normal.

### Model Summary

`summarize` confirms the declared fixed effects, expected covariates, and observation model:

```julia
model_summary = NoLimits.summarize(model)
model_summary
```

<!- injected:fe1-model ->
```text
ModelSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  model type                          : non-ODE
  fixed-effect blocks                 : 4
  fixed-effect scalar values          : 4
  random effects                      : 0
  random-effect grouping columns      : 0
  covariates (declared)               : 1
  formulas (deterministic / outcomes) : 1 / 1
  requires DE accessors               : false

Structure blocks
  helpers              : false
  fixed effects        : true
  random effects       : false
  covariates           : true
  preDE                : false
  DifferentialEquation : false
  initialDE            : false

Covariate classes
  varying  : 1
  constant : 0
  dynamic  : 0

Fixed-effects declarations
  name  type        size  se  prior      scale     bounds                              details
  ---------------------------------------------------------------------------------------------------------
  a     RealNumber     1  yes  Normal     identity  finite lower 0/1, finite upper 0/1  -
  k     RealNumber     1  yes  Normal     identity  finite lower 0/1, finite upper 0/1  -
  t0    RealNumber     1  yes  Normal     identity  finite lower 0/1, finite upper 0/1  -
  σ     RealNumber     1  yes  LogNormal  log       finite lower 1/1, finite upper 0/1  -

Random-effects declarations
  (none)

Covariate declarations
  name  kind       columns                   constant_on           interpolation
  ---------------------------------------------------------------------------------------
  age   Covariate  age                       -                     -

Formulas
  deterministic names : μ
  outcome names       : circumference
  required DE states  : (none)
  required DE signals : (none)
  declared DE states  : (none)
  declared DE signals : (none)
Outcome distribution types
  circumference => Normal

Helper functions
  names : (none)
```

## Step 3: Build the DataModel and Configure Estimation

A `DataModel` validates the data against the model and prepares the internal structures for estimation. We also configure the two estimators (500 iterations each).

```julia
dm = DataModel(model, df; primary_id=:Tree, time_col=:age)

mle_method = NoLimits.MLE(; optim_kwargs=(maxiters=500,))
map_method = NoLimits.MAP(; optim_kwargs=(maxiters=500,))

serialization = SciMLBase.EnsembleThreads()
```

`primary_id=:Tree` identifies individuals and `time_col=:age` sets the time axis; `EnsembleThreads()` evaluates the per-individual likelihood in parallel — minor here, but valuable on larger datasets.

### DataModel Summary

The summary confirms five individuals, seven observations each, no missing values, and no events.

```julia
dm_summary = NoLimits.summarize(dm)
dm_summary
```

<!- injected:fe1-dm ->
```text
DataModelSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  model type                 : non-ODE
  event-aware                : false
  individuals                : 5
  rows (total / obs / event) : 35 / 35 / 0
  fixed effects (top-level)  : 4
  outcomes                   : 1
  covariates (declared)      : 1
  random effects             : 0

Covariate classes
  varying  : 1
  constant : 0
  dynamic  : 0

Outcome distribution types
  circumference => Normal

Random-effect distribution types
  (none)

Individual design diagnostics
  individuals with one observation              : 0
  global observed time range                    : 118.0 to 1582.0
  unique observed time points                   : 7
  duplicate (ID, time) observation rows         : 0
  monotonic-time violations (observation order) : 0

Observations per individual
  metric       n          mean            sd           min           q25        median           q75           max
  ----------------------------------------------------------------------------------------------------------------
  count        5           7.0           0.0           7.0           7.0           7.0           7.0           7.0

Time span per individual
  metric       n          mean            sd           min           q25        median           q75           max
  ----------------------------------------------------------------------------------------------------------------
  span         5        1464.0           0.0        1464.0        1464.0        1464.0        1464.0        1464.0

Median sampling interval per individual
  metric          n          mean            sd           min           q25        median           q75           max
  -------------------------------------------------------------------------------------------------------------------
  median_dt       5         218.5           0.0         218.5         218.5         218.5         218.5         218.5

Outcome descriptive statistics (observation rows)
  Variable            n          mean            sd           min           q25        median           q75           max
  -----------------------------------------------------------------------------------------------------------------------
  circumference      35      115.8571        56.661          30.0          65.5         115.0         161.5         214.0

Declared covariates
  name  kind       columns
  -------------------------------------
  age   Covariate  age

Covariate descriptive statistics (observation rows)
  Variable       n          mean            sd           min           q25        median           q75           max
  ------------------------------------------------------------------------------------------------------------------
  age.age       35      922.1429       484.787         118.0         484.0        1004.0        1372.0        1582.0
```

## Step 4: Fit with MLE and MAP

Each `fit_model` call runs the default L-BFGS optimizer from the declared initial values. (L-BFGS is the default numerical optimizer for `MLE`, `MAP`, and the EM/quadrature methods; the marginal-likelihood random-effects methods [`Laplace`](../estimation/laplace.md) and [`FOCEI`](../estimation/focei.md) instead default their *outer* fixed-effects optimization to the derivative-free `NLopt.LN_BOBYQA()`.) MLE and MAP differ only in the objective: MLE uses the log-likelihood, MAP adds the log-prior, which can stabilize weakly identified parameters (see [MLE / MAP](../estimation/mle.md)).

```julia
res_mle = fit_model(dm, mle_method; serialization=serialization, rng=Random.Xoshiro(41))
res_map = fit_model(dm, map_method; serialization=serialization, rng=Random.Xoshiro(42))

(
    objective_mle=NoLimits.get_objective(res_mle),
    objective_map=NoLimits.get_objective(res_map),
)
```

<!- injected:fe1-obj ->
```text
(objective_mle = 158.39871272191726, objective_map = 168.53669404545343)
```

These objectives are negative log-likelihood (MLE) and negative log-posterior (MAP); the extra prior terms make the MAP value not directly comparable to MLE's.

The fit summary reports convergence, the final objective, and standard errors for every parameter with `calculate_se=true`:

```julia
fit_summary_mle = NoLimits.summarize(res_mle)
fit_summary_map = NoLimits.summarize(res_map)

fit_summary_mle
```

<!- injected:fe1-fitmle ->
```text
FitResultSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  method                              : mle
  inference                           : frequentist
  scale                               : natural
  objective                           : 158.3987
  iterations                          : 65
  parameters shown (reported / total) : 4 / 4

Parameter estimates
  parameter      Estimate
  -----------------------
  a              192.6876
  k                0.0028
  t0             728.7564
  σ                22.348

Outcome data coverage
  outcome             n_obs   n_missing
  -------------------------------------
  circumference          35           0
  TOTAL                  35           0
```

## Step 5: Compare Parameter Estimates

Extract the estimates on the natural (untransformed) scale and compare them side by side:

```julia
θ_mle = NoLimits.get_params(res_mle; scale=:untransformed)
θ_map = NoLimits.get_params(res_map; scale=:untransformed)

(
    mle=θ_mle,
    map=θ_map,
)
```

<!- injected:fe1-params ->
```text
(mle = (a = 192.68759193577552, k = 0.002828584934064404, t0 = 728.7563832627657, σ = 22.34804785876128), map = (a = 191.8757304534822, k = 0.0028588792012859006, t0 = 724.0263172113838, σ = 22.184112443700283))
```

With only 35 observations the priors pull the MAP estimates noticeably, most so for weakly-constrained parameters like the asymptote `a` (few trees have neared their maximum size): MLE leaves it at the pure likelihood optimum, MAP shrinks it toward the prior mean. The gap closes as data accumulate, so a large MLE–MAP discrepancy flags a parameter the data alone cannot pin down.

## Step 6: Fitted Trajectories

Overlaying predictions on the raw data is the most direct goodness-of-fit check ([plotting reference](../plotting/index.md)). Here are the fitted curves for the first two trees under each method.

```julia
inds = [1, 2]

p_fit_mle = plot_fits(
    res_mle;
    observable=:circumference,
    individuals_idx=inds,
    ncols=2,
    shared_x_axis=true,
    shared_y_axis=true,
)

p_fit_map = plot_fits(
    res_map;
    observable=:circumference,
    individuals_idx=inds,
    ncols=2,
    shared_x_axis=true,
    shared_y_axis=true,
)
```

MLE fit plot:

```julia
p_fit_mle
```

<!- injected:fe1-pfitmle ->
![Fitted population logistic trajectories under MLE for the first two trees.](figures/fe1/p_fit_mle.png)

With no random effects, every tree shares the same population curve, so deviations mix measurement noise with uncaptured between-tree variability. Systematic patterns — one tree consistently above the curve, another below — signal that random effects are needed (see the [mixed-effects tutorials](mixed-effects-multiple-methods.md)).

MAP fit plot:

```julia
p_fit_map
```

<!- injected:fe1-pfitmap ->
![Fitted population logistic trajectories under MAP for the first two trees.](figures/fe1/p_fit_map.png)

How far the MAP trajectories shift from the MLE ones is a quick visual read on prior influence.

## Step 7: Observation Distribution Diagnostics

It also helps to inspect the predicted observation distribution at a single time point — here the first observation of the first tree, with the observed value marked. A well-calibrated model places observations near the center of the distribution, not in the tails.

```julia
p_obs_mle = plot_observation_distributions(
    res_mle;
    observables=:circumference,
    individuals_idx=1,
    obs_rows=1,
)

p_obs_map = plot_observation_distributions(
    res_map;
    observables=:circumference,
    individuals_idx=1,
    obs_rows=1,
)
```

MLE observation-distribution plot:

```julia
p_obs_mle
```

<!- injected:fe1-pobsmle ->
![Predicted observation distribution at the first observation of the first tree (MLE).](figures/fe1/p_obs_mle.png)

MAP observation-distribution plot:

```julia
p_obs_map
```

<!- injected:fe1-pobsmap ->
![Predicted observation distribution at the first observation of the first tree (MAP).](figures/fe1/p_obs_map.png)

A value far in the tail points to misspecification — a residual distribution that is too narrow, or a structural model that systematically over- or under-predicts.

## Summary and Next Steps

- **MLE vs MAP.** MLE is purely data-driven; MAP adds log-prior terms that stabilize weakly identified parameters — common in nonlinear models with asymptotic behavior. Comparing them shows which parameters the data actually constrain.
- **No random effects.** Every individual shares one curve; systematic per-tree deviations call for random effects — see the [mixed-effects tutorials](mixed-effects-multiple-methods.md).
- **Diagnostics are method-agnostic.** `plot_fits` and `plot_observation_distributions` work identically across every estimator.
