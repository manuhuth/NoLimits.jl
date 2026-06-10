
# Mixed-Effects Tutorial 4: SoftTree Differential-Equation Components (SAEM) {#Mixed-Effects-Tutorial-4:-SoftTree-Differential-Equation-Components-SAEM}

When building mechanistic models of longitudinal data, you often know the broad structure of the system – compartments, conservation laws, transfer pathways – but not the precise functional forms that govern how material moves between states. Neural networks are one way to learn those unknown rate functions from data, as shown in Tutorial 3. Soft decision trees offer an appealing alternative. They can approximate arbitrary nonlinear mappings, yet their branching structure provides built-in feature selection and piecewise-smooth approximation that is often easier to interpret. For the low-dimensional inputs typical of scientific rate functions (a single state variable, or time itself), soft trees can match neural network flexibility with substantially fewer parameters.

In this tutorial, you will build a mixed-effects ODE model in which soft decision trees parameterize the ODE right-hand side, then estimate the model with the Stochastic Approximation Expectation-Maximization (SAEM) algorithm. The model is structurally parallel to Tutorial 3, so you can directly compare the two function-approximation strategies on the same data and compartmental structure.

## Learning Goals {#Learning-Goals}

By the end of this tutorial, you will be able to:
- Declare `SoftTreeParameters` blocks that create differentiable decision trees whose flattened parameters join the fixed-effects vector, exposing callable functions (e.g., `STA1`) for use inside `@DifferentialEquation`.
  
- Wire multiple soft trees into a two-compartment transfer ODE, letting the trees learn unknown rate functions from data.
  
- Couple each tree's parameter vector to subject-level random effects via `MvNormal` distributions, giving every individual a personalized version of the dynamics.
  
- Fit the model with SAEM using its default settings, which remain stable even when the random-effect vectors are high-dimensional.
  
- Visualize individual-level trajectories and observation distributions to assess model adequacy.
  

## Step 1: Data Setup {#Step-1:-Data-Setup}

In this step, you will load the Theophylline dataset used throughout these tutorials. The dataset records time-series measurements for 12 subjects and provides a clean example of two-compartment transfer dynamics: a substance enters a depot (input) compartment and moves to a central (observed) compartment, where it is measured and gradually cleared. You will reshape the data so that the initial amount `d` appears as a constant covariate for each subject.

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
using Turing

include(joinpath(@__DIR__, "_data_loaders.jl"))

Random.seed!(654)

theoph_df = load_theoph()

function build_theoph_non_event_df(tbl::DataFrame)
    df = DataFrame(
        ID=Int.(tbl.Subject),
        t=Float64.(tbl.Time),
        y=Float64.(tbl.conc),
        d=Float64.(tbl.Wt .* tbl.Dose),
    )
    sort!(df, [:ID, :t])
    return df
end

df = build_theoph_non_event_df(theoph_df)
first(df, 10)
```


&lt;!– injected:t4-dfhead –&gt;

```text
10×4 DataFrame
 Row │ ID     t        y        d
     │ Int64  Float64  Float64  Float64
─────┼──────────────────────────────────
   1 │     1     0.0      0.74  319.992
   2 │     1     0.25     2.84  319.992
   3 │     1     0.57     6.57  319.992
   4 │     1     1.12    10.5   319.992
   5 │     1     2.02     9.66  319.992
   6 │     1     3.82     8.58  319.992
   7 │     1     5.1      8.36  319.992
   8 │     1     7.03     7.47  319.992
   9 │     1     9.05     6.89  319.992
  10 │     1    12.12     5.94  319.992
```


## Step 2: Define SoftTree-Driven ODE Mixed-Effects Model {#Step-2:-Define-SoftTree-Driven-ODE-Mixed-Effects-Model}

In this step, you will construct the full mixed-effects model. The guiding idea is the same as in the neural ODE tutorial: rather than specifying closed-form rate laws, you let data-driven function approximators learn the rate functions directly from observations. The difference is the choice of approximator. Each `SoftTreeParameters` block declares a soft decision tree with a specified input dimension and depth. The `depth_st` parameter controls expressiveness – a tree of depth `d` has `2^d` leaf nodes, each contributing a smooth local approximation. The block's flattened parameters become part of the fixed-effects vector, and the associated callable function (e.g., `STA1`) evaluates the tree at any input.

The ODE system wires four soft trees into a two-compartment transfer model:
- `fA1(t)` and `fA2(t)` govern the dynamics of the depot (input) compartment.
  
- `fC1(t)` and `fC2(t)` govern the dynamics of the central (observed) compartment.
  

To capture between-subject variability, each tree's parameter vector is paired with a subject-level random-effect vector drawn from an `MvNormal` distribution centered on the population parameters. This gives every individual a personalized version of the transfer dynamics while sharing structure across the population.

```julia
using NoLimits
using Distributions
using LinearAlgebra
using OrdinaryDiffEq

depth_st = 2

model_raw = @Model begin
    @helpers begin
        softplus(u) = u > 20 ? u : log1p(exp(u))
    end

    @covariates begin
        t = Covariate()
        d = ConstantCovariate(constant_on=:ID)
    end

    @fixedEffects begin
        sigma = RealNumber(1.0, scale=:log, prior=LogNormal(log(1.0), 0.5), calculate_se=true)

        gA1 = SoftTreeParameters(1, depth_st; function_name=:STA1, calculate_se=false)
        gA2 = SoftTreeParameters(1, depth_st; function_name=:STA2, calculate_se=false)
        gC1 = SoftTreeParameters(1, depth_st; function_name=:STC1, calculate_se=false)
        gC2 = SoftTreeParameters(1, depth_st; function_name=:STC2, calculate_se=false)
    end

    @randomEffects begin
        etaA1 = RandomEffect(MvNormal(gA1, Diagonal(ones(length(gA1)))); column=:ID)
        etaA2 = RandomEffect(MvNormal(gA2, Diagonal(ones(length(gA2)))); column=:ID)
        etaC1 = RandomEffect(MvNormal(gC1, Diagonal(ones(length(gC1)))); column=:ID)
        etaC2 = RandomEffect(MvNormal(gC2, Diagonal(ones(length(gC2)))); column=:ID)
    end

    @DifferentialEquation begin
        a_A(t) = softplus(depot)
        x_C(t) = softplus(center)

        fA1(t) = softplus(STA1([t / 24], etaA1)[1])
        fA2(t) = softplus(STA2([a_A(t)], etaA2)[1])
        fC1(t) = -softplus(STC1([x_C(t)], etaC1)[1])
        fC2(t) = softplus(STC2([t / 24], etaC2)[1])

        D(depot) ~ -d * fA1(t) - fA2(t)
        D(center) ~ d * fA1(t) + fA2(t) + fC1(t) + d * fC2(t)
    end

    @initialDE begin
        depot = d
        center = 0.0
    end

    @formulas begin
        y ~ Normal(center(t), sigma)
    end
end

model = set_solver_config(
    model_raw;
    saveat_mode=:saveat,
    alg=AutoTsit5(Rosenbrock23()),
    kwargs=(abstol=1e-2, reltol=1e-2),
)
```


Before moving on, inspect the assembled model to verify that all blocks – covariates, fixed effects, random effects, ODE, and formulas – are correctly wired together.

### Model Summary {#Model-Summary}

```julia
model_summary = NoLimits.summarize(model)
model_summary
```


&lt;!– injected:t4-model –&gt;

```text
ModelSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  model type                          : ODE
  fixed-effect blocks                 : 5
  fixed-effect scalar values          : 41
  random effects                      : 4
  random-effect grouping columns      : 1
  covariates (declared)               : 2
  formulas (deterministic / outcomes) : 0 / 1
  requires DE accessors               : true

Structure blocks
  helpers              : true
  fixed effects        : true
  random effects       : true
  covariates           : true
  preDE                : false
  DifferentialEquation : true
  initialDE            : true

Covariate classes
  varying  : 1
  constant : 1
  dynamic  : 0

Fixed-effects declarations
  name   type                size  se  prior      scale  bounds                                details
  -----------------------------------------------------------------------------------------------------------------
  sigma  RealNumber             1  yes  LogNormal  log    finite lower 1/1, finite upper 0/1    -
  gA1    SoftTreeParameters    10  no  Priorless  n/a    finite lower 0/10, finite upper 0/10  function=STA1, input_dim=1, depth=2, outputs=1, weights=10
  gA2    SoftTreeParameters    10  no  Priorless  n/a    finite lower 0/10, finite upper 0/10  function=STA2, input_dim=1, depth=2, outputs=1, weights=10
  gC1    SoftTreeParameters    10  no  Priorless  n/a    finite lower 0/10, finite upper 0/10  function=STC1, input_dim=1, depth=2, outputs=1, weights=10
  gC2    SoftTreeParameters    10  no  Priorless  n/a    finite lower 0/10, finite upper 0/10  function=STC2, input_dim=1, depth=2, outputs=1, weights=10

Random-effects declarations
  name   group  dist    
  ------------------------
  etaA1  ID     MvNormal
  etaA2  ID     MvNormal
  etaC1  ID     MvNormal
  etaC2  ID     MvNormal

Covariate declarations
  name  kind               columns                   constant_on           interpolation
  -----------------------------------------------------------------------------------------------
  t     Covariate          t                         -                     -
  d     ConstantCovariate  d                         ID                    -

Formulas
  deterministic names : (none)
  outcome names       : y
  required DE states  : center
  required DE signals : (none)
  declared DE states  : depot, center
  declared DE signals : a_A, x_C, fA1, fA2, fC1, fC2
Outcome distribution types
  y => Normal

Helper functions
  names : softplus
```


## Step 3: Build `DataModel` and Configure SAEM {#Step-3:-Build-DataModel-and-Configure-SAEM}

In this step, you will pair the model with the observed data by constructing a `DataModel`, then configure the SAEM fitting algorithm. SAEM alternates between two phases: an E-step that samples subject-level random effects conditional on the current population parameters, and an M-step that updates those population parameters using stochastic sufficient statistics. Here we use the default configuration, `NoLimits.SAEM()`. With its defaults, SAEM draws the random effects with the adaptive Metropolis sampler (`SaemixMH`) in the E-step and updates the population parameters with a stochastic-approximation (Robbins-Monro) M-step. No special configuration is required even though each tree's full parameter vector is individualized, so the random-effect dimension is high. Running with defaults keeps the example simple and parallel to the neural-ODE tutorial, making the two function-approximation strategies directly comparable.

```julia
dm = DataModel(model, df; primary_id=:ID, time_col=:t)

saem_method = NoLimits.SAEM()

serialization = SciMLBase.EnsembleThreads()
```


Before fitting, review the data model summary to confirm that individuals, covariates, and grouping structures were parsed as expected.

### DataModel Summary {#DataModel-Summary}

```julia
dm_summary = NoLimits.summarize(dm)
dm_summary
```


&lt;!– injected:t4-dm –&gt;

```text
DataModelSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  model type                 : ODE
  event-aware                : false
  individuals                : 12
  rows (total / obs / event) : 132 / 132 / 0
  fixed effects (top-level)  : 5
  outcomes                   : 1
  covariates (declared)      : 2
  random effects             : 4

Covariate classes
  varying  : 1
  constant : 1
  dynamic  : 0

Outcome distribution types
  y => Normal

Random-effect distribution types
  etaA1 => MvNormal
  etaA2 => MvNormal
  etaC1 => MvNormal
  etaC2 => MvNormal

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
  y            132        4.9605        2.8564           0.0        2.8775         5.275          7.14          11.4

Declared covariates
  name  kind               columns
  ---------------------------------------------
  t     Covariate          t
  d     ConstantCovariate  d

Covariate descriptive statistics (observation rows)
  Variable       n          mean            sd           min           q25        median           q75           max
  ------------------------------------------------------------------------------------------------------------------
  t.t          132        5.8946        6.8997           0.0         0.595          3.53           9.0         24.65
  d.d          132      315.4398       14.3601        267.84       319.365        319.84       319.994        320.65

Per-random-effect summary
  random effect  group  dist        levels  rows/level min        median           max
  ----------------------------------------------------------------------------------
  etaA1          ID     MvNormal        12            11.0          11.0          11.0
  etaA2          ID     MvNormal        12            11.0          11.0          11.0
  etaC1          ID     MvNormal        12            11.0          11.0          11.0
  etaC2          ID     MvNormal        12            11.0          11.0          11.0
```


## Step 4: Fit and Inspect Core Result Summary {#Step-4:-Fit-and-Inspect-Core-Result-Summary}

In this step, you will run the SAEM algorithm and examine the results. With the default settings the algorithm iterates up to 300 times, drawing the random effects with the adaptive Metropolis sampler within each E-step. After fitting completes, you will extract the final objective value and parameter count as a quick sanity check before looking at more detailed diagnostics.

```julia
res_saem = fit_model(
    dm,
    saem_method;
    serialization=serialization,
    rng=Random.Xoshiro(31),
)

(
    objective=NoLimits.get_objective(res_saem),
    n_params=length(NoLimits.get_params(res_saem; scale=:untransformed)),
)
```


&lt;!– injected:t4-obj –&gt;

```text
(objective = -670.3364053549503, n_params = 41)
```


For a more detailed view – including parameter estimates and convergence diagnostics – call the `summarize` function on the fit result.

```julia
fit_summary_saem = NoLimits.summarize(res_saem)
fit_summary_saem
```


&lt;!– injected:t4-fit –&gt;

```text
FitResultSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  method                              : saem
  inference                           : frequentist
  scale                               : natural
  objective                           : -670.3364
  iterations                          : 300
  parameters shown (reported / total) : 1 / 41

Parameter estimates
  parameter      Estimate
  -----------------------
  sigma            0.8226

Outcome data coverage
  outcome       n_obs   n_missing
  -------------------------------
  y               132           0
  TOTAL           132           0

Empirical Bayes random effects summary (across RE levels)
  random effect  component       n          mean            sd           q25        median           q75
  --------------------------------------------------------------------------------------------------
  etaA1          etaA1_1        12       -6.4989         0.031       -6.4945       -6.4895       -6.4849
  etaA1          etaA1_2        12        2.2535        0.0104        2.2487        2.2495        2.2523
  etaA1          etaA1_3        12       -5.1222        0.0244       -5.1245       -5.1223       -5.1139
  etaA1          etaA1_4        12        0.2649        0.3076        0.0787        0.2103        0.3756
  etaA1          etaA1_5        12        0.7722        0.1468        0.7773        0.8013        0.8588
  etaA1          etaA1_6        12        0.8669        0.1964        0.7125        0.8742        1.0254
  etaA1          etaA1_7        12       -3.2423        0.1912       -3.3637       -3.2728       -3.2404
  etaA1          etaA1_8        12       -4.3963        0.0902       -4.4564       -4.4098       -4.3537
  etaA1          etaA1_9        12       -1.3254        0.1411       -1.4019       -1.3551       -1.3411
  etaA1          etaA1_10       12      -11.8404        0.0489      -11.8746       -11.834      -11.8028
  etaA2          etaA2_1        12        3.1228     0.0007118        3.1226        3.1226        3.1229
  etaA2          etaA2_2        12        3.5268     0.0008013        3.5268        3.5271        3.5271
  etaA2          etaA2_3        12        0.1855         0.002        0.1852        0.1852        0.1852
  etaA2          etaA2_4        12       -0.9161        0.0044       -0.9155       -0.9153       -0.9153
  etaA2          etaA2_5        12       -1.0653         0.001       -1.0656       -1.0656       -1.0656
  etaA2          etaA2_6        12        0.3283        0.0024        0.3283        0.3283        0.3286
  etaA2          etaA2_7        12       -5.8533        0.0083       -5.8585       -5.8568       -5.8531
  etaA2          etaA2_8        12       -3.1316        0.0022        -3.133        -3.133       -3.1311
  etaA2          etaA2_9        12       -1.1163        0.0035       -1.1172       -1.1172       -1.1172
  etaA2          etaA2_10       12       -0.5955        0.0013       -0.5955       -0.5955       -0.5951
  etaC1          etaC1_1        12        0.9385        0.2031        0.8076        0.9194        1.0491
  etaC1          etaC1_2        12       -3.2011        0.0011       -3.2018       -3.2018       -3.1998
  etaC1          etaC1_3        12        3.8715         0.025         3.849        3.8823        3.8885
  etaC1          etaC1_4        12       -8.1665        0.1306       -8.1669       -8.1151         -8.08
  etaC1          etaC1_5        12       -3.3704        0.0019         -3.37       -3.3698       -3.3698
  etaC1          etaC1_6        12       -5.7214        0.0173       -5.7374       -5.7167       -5.7117
  etaC1          etaC1_7        12        0.6949        0.0024        0.6954        0.6962        0.6962
  etaC1          etaC1_8        12       -1.0566        0.2274       -1.1082       -1.0163       -0.9412
  etaC1          etaC1_9        12       18.3039        0.0916       18.2755       18.2762       18.2778
  etaC1          etaC1_10       12       -3.3852        0.0172       -3.3912       -3.3772       -3.3738
  etaC2          etaC2_1        12       -8.7491        0.0282       -8.7471       -8.7441       -8.7371
  etaC2          etaC2_2        12       -2.8921        0.0024       -2.8923       -2.8919       -2.8914
  etaC2          etaC2_3        12       -8.0751        0.0315       -8.0864       -8.0753       -8.0622
  etaC2          etaC2_4        12        0.5952        0.1881        0.4962        0.6475        0.7392
  etaC2          etaC2_5        12        2.6882        0.0134        2.6767        2.6903        2.6952
  etaC2          etaC2_6        12        1.7983        0.2357         1.698        1.8737        1.9561
  etaC2          etaC2_7        12       -3.2972        0.3495       -3.5426        -3.296        -3.166
  etaC2          etaC2_8        12       -3.7092        0.1323       -3.8004       -3.6984       -3.5889
  etaC2          etaC2_9        12        -2.294        0.0234       -2.3095       -2.2998       -2.2799
  etaC2          etaC2_10       12      -11.1025        0.0503      -11.1147      -11.0957      -11.0812
```


## Step 5: Fitted Trajectories (First 2 Individuals) {#Step-5:-Fitted-Trajectories-First-2-Individuals}

In this step, you will overlay the model's predicted trajectories on the raw observations for the first two subjects. Plotting fitted curves against data provides an immediate visual assessment of model adequacy: do the predicted dynamics capture the timing and magnitude of the observed response?

```julia
p_fit_saem = plot_fits(
    res_saem;
    observable=:y,
    individuals_idx=[1, 2],
    ncols=2,
    shared_x_axis=true,
    shared_y_axis=true,
)

p_fit_saem
```


&lt;!– injected:t4-pfit –&gt; 
![](figures/t4/p_fit_saem.png)


## Step 6: Observation Distribution Diagnostic {#Step-6:-Observation-Distribution-Diagnostic}

As a final check, you will examine the implied observation distribution at a single data point for the first individual. Rather than showing only a point prediction, this plot displays the full predictive distribution, letting you assess whether the residual variance is well-calibrated and the model's uncertainty envelope is reasonable.

```julia
p_obs_saem = plot_observation_distributions(
    res_saem;
    observables=:y,
    individuals_idx=1,
    obs_rows=1,
)

p_obs_saem
```


&lt;!– injected:t4-pobs –&gt; 
![](figures/t4/p_obs_saem.png)


## Interpretation Notes {#Interpretation-Notes}
- This modeling pattern combines mechanistic compartmental structure with soft decision tree function approximators inside a single mixed-effects ODE. The compartments encode known domain knowledge (mass conservation, transfer pathways), while the trees learn the unknown rate functions from data. This separation means you retain interpretable system structure without needing to specify rate-law functional forms in advance.
  
- Compared to neural networks, soft trees can be more parameter-efficient for the low-dimensional inputs common in scientific rate functions, and their piecewise-smooth approximation may be easier to inspect post hoc. The choice between the two is problem-dependent; both can be embedded in the same NoLimits framework with minimal code changes (compare this tutorial with Tutorial 3).
  
- Default SAEM (`NoLimits.SAEM()`) is sufficient here: the adaptive Metropolis E-step and stochastic-approximation M-step remain stable even when the individualized parameter vectors are high-dimensional. When defaults are not enough, SAEM also exposes closed-form Gaussian block updates through `builtin_stats=:closed_form` together with the `re_mean_params` mapping.
  
- The structural settings in this tutorial (tree depth, ODE solver tolerances) are intentionally modest to keep runtime short. For production analyses, consider deeper trees, increasing `maxiters` and the number of MCMC samples to ensure thorough convergence, and tightening the ODE solver tolerances.
  
