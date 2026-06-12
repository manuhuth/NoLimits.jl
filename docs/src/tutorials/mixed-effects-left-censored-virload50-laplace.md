# Mixed-Effects Tutorial 6: Left-Censored Nonlinear Model (Laplace)

In many biomedical assays, measurements below a detection threshold cannot be reliably quantified. This situation, known as *left-censoring*, arises whenever an instrument's lower limit of quantification (LLOQ) truncates the observable range. HIV viral load monitoring is a canonical example: modern RT-PCR assays report "below detectable limit" for viral RNA concentrations under approximately 50 copies/mL -- or equivalently, below about 1.7 on the log10 scale. In patients responding well to antiretroviral therapy, censored observations can account for 30--40% of all measurements, making proper statistical treatment essential.

How should these below-limit values be handled? The answer matters. Dropping censored rows discards information and inflates uncertainty. Substituting the detection limit as though it were a true measurement concentrates probability mass at that value and biases parameter estimates downward. The principled approach is a *censored likelihood*: uncensored observations contribute their usual probability density, while censored observations contribute the cumulative probability of falling at or below the detection threshold. This formulation correctly encodes what we actually know -- that the true value lies somewhere below the limit, without committing to a specific magnitude.

In this tutorial, you will fit a nonlinear mixed-effects model to the `virload50` dataset from the `npde` R package, which contains longitudinal log10 viral load measurements from 50 HIV-positive patients. The structural model is a bi-exponential decay function that captures two distinct phases of viral dynamics: rapid initial suppression and slower long-term decline. Subject-specific parameters enter through `LogNormal` random effects, making the model nonlinear in the random effects. You will handle left-censoring directly in the observation model using NoLimits' `censored(...)` syntax and estimate the model with the Laplace approximation.

If your outcome is a discrete observed-state Markov model with ambiguous/set-valued state
labels, use `coarsed(...)` instead of `censored(...)`. See the formulas documentation:
[`Example: Coarsed Observed-State Markov Model`](../model-building/formulas.md#example-coarsed-observed-state-markov-model).

## Learning Goals

By the end of this tutorial, you will know how to:

- **Prepare censored longitudinal data.** Structure a dataset where left-censored observations are flagged by an indicator variable and pinned at the detection limit value.
- **Specify a nonlinear mixed-effects model.** Define subject-specific `LogNormal` random effects that parameterize a bi-exponential mean function, ensuring predicted viral loads remain positive.
- **Encode left-censoring in the likelihood.** Use `censored(Normal(mu, sigma), lower=1.7, upper=Inf)` so that the likelihood automatically distinguishes between density contributions (for observed values) and cumulative probability contributions (for censored values).
- **Estimate with the Laplace approximation.** Integrate over the random effects using a second-order expansion of the log-posterior around the empirical Bayes estimates.
- **Diagnose and quantify uncertainty.** Inspect fitted trajectories, predictive observation distributions, and Wald-based confidence intervals for the fixed-effects parameters.

## Step 1: Data Setup

In this step, you will load the `virload50` dataset and prepare it for modeling. The dataset contains four columns: a subject identifier (`ID`), observation time (`Time`), log10 viral load (`Log_VL`), and a censoring indicator (`cens`, where 1 flags values at or below the detection limit of 1.7). After selecting these columns, you will enforce the correct types and sort by subject and time -- a requirement for the internal data structures used by NoLimits.

The summary statistics printed at the end provide a quick overview of the dataset dimensions and the fraction of censored observations.

```julia
using NoLimits
using CSV
using DataFrames
using Distributions
using Downloads
using Random
using SciMLBase

include(joinpath(@__DIR__, "_data_loaders.jl"))

Random.seed!(2026)

df = load_virload50()
select!(df, [:ID, :Time, :Log_VL, :cens])
df.ID = string.(df.ID)
df.Time = Float64.(df.Time)
df.Log_VL = Float64.(df.Log_VL)
df.cens = Int.(df.cens)
sort!(df, [:ID, :Time])

(
    n_rows = nrow(df),
    n_subjects = length(unique(df.ID)),
    n_censored = count(==(1), df.cens),
)
```

<!-- injected:t6-data -->
```text
(n_rows = 300, n_subjects = 50, n_censored = 131)
```

## Step 2: Define the Nonlinear Left-Censored Mixed-Effects Model

In this step, you will define the structural model for viral load dynamics. The bi-exponential decay function captures two biologically distinct phases: the rapid initial clearance of free virus from the plasma, and the slower decline driven by the loss of productively infected cells. On the original scale, the model for subject $i$ at time $t$ is:

```math
V_i(t) = A_i e^{-k_{1,i} t} + B_i e^{-k_{2,i} t},
\qquad \mu_{it} = \log_{10}(V_i(t)).
```

Each subject-specific parameter ($A_i$, $B_i$, $k_{1,i}$, $k_{2,i}$) is drawn from a `LogNormal` distribution, which guarantees positivity -- a necessary constraint since both amplitudes and rate constants must be strictly positive for the model to be biologically meaningful. The fixed effects (`beta_A`, `beta_B`, `beta_k1`, `beta_k2`) represent population-level medians on the log scale, while the `omega` parameters govern the magnitude of between-subject variability for each parameter.

The observation model is where the censoring logic enters. By writing `censored(Normal(mu, sigma), lower=1.7, upper=Inf)`, you specify a Normal distribution that is left-censored at the log10 detection limit of 1.7. NoLimits handles this as follows: when the recorded `Log_VL` value exceeds 1.7, the observation contributes the standard Normal density; when `Log_VL` equals 1.7 (the pinned value for censored rows in this dataset), the observation instead contributes the cumulative probability $\Phi\bigl((1.7 - \mu) / \sigma\bigr)$, representing the probability that the true value falls below the detection limit. This censored-likelihood approach is statistically exact and avoids the biases introduced by ad hoc imputation or deletion strategies.

```julia
model = @Model begin
    @covariates begin
        Time = Covariate()
    end

    @fixedEffects begin
        beta_A = RealNumber(9.9, calculate_se=true)
        beta_B = RealNumber(5.0, calculate_se=true)
        beta_k1 = RealNumber(-1.9, calculate_se=true)
        beta_k2 = RealNumber(-5.3, calculate_se=true)

        omega_A = RealNumber(0.40, scale=:log, calculate_se=true)
        omega_B = RealNumber(0.40, scale=:log, calculate_se=true)
        omega_k1 = RealNumber(0.40, scale=:log, calculate_se=true)
        omega_k2 = RealNumber(0.40, scale=:log, calculate_se=true)

        sigma = RealNumber(0.25, scale=:log, calculate_se=true)
    end

    @randomEffects begin
        A_i = RandomEffect(LogNormal(beta_A, omega_A); column=:ID)
        B_i = RandomEffect(LogNormal(beta_B, omega_B); column=:ID)
        k1_i = RandomEffect(LogNormal(beta_k1, omega_k1); column=:ID)
        k2_i = RandomEffect(LogNormal(beta_k2, omega_k2); column=:ID)
    end

    @formulas begin
        V_i = A_i * exp(-k1_i * Time) + B_i * exp(-k2_i * Time)
        mu = log10(V_i)

        Log_VL ~ censored(Normal(mu, sigma), lower=1.7, upper=Inf)
    end
end

NoLimits.summarize(model)
```

<!-- injected:t6-model -->
```text
ModelSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  model type                          : non-ODE
  fixed-effect blocks                 : 9
  fixed-effect scalar values          : 9
  random effects                      : 4
  random-effect grouping columns      : 1
  covariates (declared)               : 1
  formulas (deterministic / outcomes) : 2 / 1
  requires DE accessors               : false

Structure blocks
  helpers              : false
  fixed effects        : true
  random effects       : true
  covariates           : true
  preDE                : false
  DifferentialEquation : false
  initialDE            : false

Covariate classes
  varying  : 1
  constant : 0
  dynamic  : 0

Fixed-effects declarations
  name      type        size  se  prior      scale     bounds                              details
  -------------------------------------------------------------------------------------------------------------
  beta_A    RealNumber     1  yes  Priorless  identity  finite lower 0/1, finite upper 0/1  -
  beta_B    RealNumber     1  yes  Priorless  identity  finite lower 0/1, finite upper 0/1  -
  beta_k1   RealNumber     1  yes  Priorless  identity  finite lower 0/1, finite upper 0/1  -
  beta_k2   RealNumber     1  yes  Priorless  identity  finite lower 0/1, finite upper 0/1  -
  omega_A   RealNumber     1  yes  Priorless  log       finite lower 1/1, finite upper 0/1  -
  omega_B   RealNumber     1  yes  Priorless  log       finite lower 1/1, finite upper 0/1  -
  omega_k1  RealNumber     1  yes  Priorless  log       finite lower 1/1, finite upper 0/1  -
  omega_k2  RealNumber     1  yes  Priorless  log       finite lower 1/1, finite upper 0/1  -
  sigma     RealNumber     1  yes  Priorless  log       finite lower 1/1, finite upper 0/1  -

Random-effects declarations
  name  group  dist     
  ------------------------
  A_i   ID     LogNormal
  B_i   ID     LogNormal
  k1_i  ID     LogNormal
  k2_i  ID     LogNormal

Covariate declarations
  name  kind       columns                   constant_on           interpolation
  ---------------------------------------------------------------------------------------
  Time  Covariate  Time                      -                     -

Formulas
  deterministic names : V_i, mu
  outcome names       : Log_VL
  required DE states  : (none)
  required DE signals : (none)
  declared DE states  : (none)
  declared DE signals : (none)
Outcome distribution types
  Log_VL => censored

Helper functions
  names : (none)
```

## Step 3: Build `DataModel` and Configure `Laplace`

In this step, you will bind the model to the dataset by constructing a `DataModel`. This validates that all required columns are present and correctly typed, groups observations by subject, and assembles the internal data structures needed for estimation.

Next, you will configure the Laplace approximation. This method approximates the marginal likelihood by finding the mode of each subject's conditional random-effects distribution (the empirical Bayes estimates) and using a second-order Taylor expansion around those modes to integrate out the random effects analytically. Two sets of optimization controls are available: `inner_kwargs` governs the per-subject random-effects optimization, while `optim_kwargs` governs the outer fixed-effects optimization. Multistart is disabled here for speed, but enabling it can help avoid local optima in models with more complex likelihood surfaces.

```julia
dm = DataModel(model, df; primary_id=:ID, time_col=:Time)

laplace_method = NoLimits.Laplace(;
    optim_kwargs=(maxiters=400,),
    inner_kwargs=(maxiters=150,),
    multistart_n=0,
    multistart_k=0,
)

serialization = SciMLBase.EnsembleThreads()

NoLimits.summarize(dm)
```

<!-- injected:t6-dm -->
```text
DataModelSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  model type                 : non-ODE
  event-aware                : false
  individuals                : 50
  rows (total / obs / event) : 300 / 300 / 0
  fixed effects (top-level)  : 9
  outcomes                   : 1
  covariates (declared)      : 1
  random effects             : 4

Covariate classes
  varying  : 1
  constant : 0
  dynamic  : 0

Outcome distribution types
  Log_VL => censored

Random-effect distribution types
  A_i  => LogNormal
  B_i  => LogNormal
  k1_i => LogNormal
  k2_i => LogNormal

Individual design diagnostics
  individuals with one observation              : 0
  global observed time range                    : 0.0 to 168.0
  unique observed time points                   : 6
  duplicate (ID, time) observation rows         : 0
  monotonic-time violations (observation order) : 0

Observations per individual
  metric       n          mean            sd           min           q25        median           q75           max
  ----------------------------------------------------------------------------------------------------------------
  count       50           6.0           0.0           6.0           6.0           6.0           6.0           6.0

Time span per individual
  metric       n          mean            sd           min           q25        median           q75           max
  ----------------------------------------------------------------------------------------------------------------
  span        50         168.0           0.0         168.0         168.0         168.0         168.0         168.0

Median sampling interval per individual
  metric          n          mean            sd           min           q25        median           q75           max
  -------------------------------------------------------------------------------------------------------------------
  median_dt      50          28.0           0.0          28.0          28.0          28.0          28.0          28.0

Outcome descriptive statistics (observation rows)
  Variable       n          mean            sd           min           q25        median           q75           max
  ------------------------------------------------------------------------------------------------------------------
  Log_VL       300         2.378        1.0243           1.7           1.7         1.885        2.6525          6.28

Declared covariates
  name  kind       columns
  -------------------------------------
  Time  Covariate  Time

Covariate descriptive statistics (observation rows)
  Variable        n          mean            sd           min           q25        median           q75           max
  -------------------------------------------------------------------------------------------------------------------
  Time.Time     300       74.6667       55.2167           0.0          28.0          70.0         112.0         168.0

Per-random-effect summary
  random effect  group  dist         levels  rows/level min        median           max
  -----------------------------------------------------------------------------------
  A_i            ID     LogNormal        50             6.0           6.0           6.0
  B_i            ID     LogNormal        50             6.0           6.0           6.0
  k1_i           ID     LogNormal        50             6.0           6.0           6.0
  k2_i           ID     LogNormal        50             6.0           6.0           6.0
```

## Step 4: Fit and Inspect Core Summary

With the data model and estimation method in place, you can now run the fit. The Laplace algorithm alternates between two stages: an inner loop that updates the empirical Bayes estimates of the random effects for each subject, and an outer loop that optimizes the fixed-effects parameters with respect to the Laplace-approximated marginal likelihood. After convergence, the summary reports the estimated parameter values alongside the final objective function value.

```julia
res = fit_model(
    dm,
    laplace_method;
    serialization=serialization,
    rng=Random.Xoshiro(7003),
)

NoLimits.summarize(res)
```

<!-- injected:t6-res -->
```text
FitResultSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  method                              : laplace
  inference                           : frequentist
  scale                               : natural
  objective                           : 261.9148
  iterations                          : 49
  parameters shown (reported / total) : 9 / 9

Parameter estimates
  parameter      Estimate
  -----------------------
  beta_A           8.9221
  beta_B          -6.6133
  beta_k1         -2.5989
  beta_k2          -5.934
  omega_A        5.928e-8
  omega_B          0.0057
  omega_k1         0.3952
  omega_k2         0.0458
  sigma            0.6915

Outcome data coverage
  outcome       n_obs   n_missing
  -------------------------------
  Log_VL          300           0
  TOTAL           300           0

Empirical Bayes random effects summary (across RE levels)
  random effect       n          mean            sd           q25        median           q75
  ---------------------------------------------------------------------------
  A_i                50     7496.0306     1.171e-10     7496.0306     7496.0306     7496.0306
  B_i                50        0.0013     9.694e-13        0.0013        0.0013        0.0013
  k1_i               50        0.0793        0.0283        0.0541        0.0721        0.1071
  k2_i               50        0.0026     3.523e-11        0.0026        0.0026        0.0026
```

## Step 5: Fitted Trajectories (First 2 Individuals)

In this step, you will overlay the model's fitted trajectories on the observed data for a visual check of model adequacy. For subjects with censored observations, pay attention to the predicted curve near the detection limit: a well-fitting model should produce predicted values at or near 1.7 at censored time points, reflecting the fact that the true viral load lies somewhere below this threshold rather than at a precisely known value.

```julia
p_fit = plot_fits(
    res;
    observable=:Log_VL,
    individuals_idx=[1, 2],
    ncols=2,
    shared_x_axis=true,
    shared_y_axis=true,
)

p_fit
```

<!-- injected:t6-pfit -->
![Fitted bi-exponential viral-load trajectories for the first two subjects (Laplace).](figures/t6/p_fit.png)

## Step 6: Observation Distribution Diagnostic (First Individual)

This diagnostic reveals the full predictive distribution at selected time points for a single subject. For uncensored observations, you will see a Normal density centered on the predicted log10 viral load. For censored observations, the distribution is truncated at the detection limit, with the probability mass below 1.7 collapsed into a point mass. Examining these distributions is a useful way to verify that the censored likelihood is behaving as intended and that the model assigns appropriate probability to the below-limit region.

```julia
p_obs = plot_observation_distributions(
    res;
    observables=:Log_VL,
    individuals_idx=1,
    obs_rows=[1, 2],
)

p_obs
```

<!-- injected:t6-pobs -->
![Predicted observation distributions at the first two observations of the first subject, including the left-censored region.](figures/t6/p_obs.png)

## Step 7: Wald Uncertainty Quantification

In this final step, you will assess the precision of the estimated fixed effects by computing Wald-based confidence intervals. The Wald method constructs approximate 95% intervals from the observed Fisher information matrix -- that is, the curvature of the log-likelihood at the optimum. This is computationally inexpensive and provides a practical first assessment of parameter identifiability: wide intervals may signal that the data contain insufficient information to pin down a particular parameter.

```julia
uq = compute_uq(
    res;
    method=:wald,
    n_draws=800,
    level=0.95,
    rng=Random.Xoshiro(153),
)

NoLimits.summarize(uq)
```

<!-- injected:t6-uq -->
```text
UQResultSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  backend                             : wald
  source_method                       : laplace
  inference                           : frequentist
  scale                               : natural
  objective                           : -
  interval level                      : 0.95
  parameters shown (reported / total) : 9 / 9

Parameter uncertainty summary
  parameter      Estimate    Std. Error      CI Lower      CI Upper
  ---------------------------------------------------
  beta_A           8.9221        0.0047        8.9132        8.9316
  beta_B          -6.6133        0.0391       -6.6868       -6.5402
  beta_k1         -2.5989        0.0755       -2.7422       -2.4495
  beta_k2          -5.934        0.0228       -5.9783       -5.8889
  omega_A        5.928e-8      3.234e-5     9.395e-11       3.66e-5
  omega_B          0.0057        0.0013        0.0037        0.0088
  omega_k1         0.3952        0.0707         0.282        0.5554
  omega_k2         0.0458     0.0006977        0.0444        0.0472
  sigma            0.6915        0.0077        0.6754        0.7063
```

For a consolidated report combining point estimates and their uncertainty, pass both the fit result and the uncertainty object to `summarize`. This tabular format is convenient for inclusion in manuscripts and supplementary materials.

```julia
NoLimits.summarize(res, uq)
```

<!-- injected:t6-resuq -->
```text
UQResultSummary
════════════════════════════════════════════════════════════════════════════════════════════════
Overview
  backend                             : wald
  source_method                       : laplace
  inference                           : frequentist
  scale                               : natural
  objective                           : 261.9148
  interval level                      : 0.95
  parameters shown (reported / total) : 9 / 9

Parameter uncertainty summary
  parameter      Estimate    Std. Error      CI Lower      CI Upper
  ---------------------------------------------------
  beta_A           8.9221        0.0047        8.9132        8.9316
  beta_B          -6.6133        0.0391       -6.6868       -6.5402
  beta_k1         -2.5989        0.0755       -2.7422       -2.4495
  beta_k2          -5.934        0.0228       -5.9783       -5.8889
  omega_A        5.928e-8      3.234e-5     9.395e-11       3.66e-5
  omega_B          0.0057        0.0013        0.0037        0.0088
  omega_k1         0.3952        0.0707         0.282        0.5554
  omega_k2         0.0458     0.0006977        0.0444        0.0472
  sigma            0.6915        0.0077        0.6754        0.7063

Outcome data coverage
  outcome       n_obs   n_missing
  -------------------------------
  Log_VL          300           0
  TOTAL           300           0

Empirical Bayes random effects summary (across RE levels)
  random effect       n          mean            sd           q25        median           q75
  ---------------------------------------------------------------------------
  A_i                50     7496.0306     1.171e-10     7496.0306     7496.0306     7496.0306
  B_i                50        0.0013     9.694e-13        0.0013        0.0013        0.0013
  k1_i               50        0.0793        0.0283        0.0541        0.0721        0.1071
  k2_i               50        0.0026     3.523e-11        0.0026        0.0026        0.0026
```

Finally, you can visualize the implied sampling distributions of the fixed-effects parameters on the natural (untransformed) scale. Parameters estimated on the log scale (the `omega` and `sigma` parameters) are back-transformed before plotting, so the density plots reflect the scale on which these quantities are scientifically interpretable.

```julia
plot_uq_distributions(uq; scale=:natural, plot_type=:density, show_legend=false)
```

<!-- injected:t6-puq -->
![Wald approximate parameter distributions on the natural scale.](figures/t6/p_uq.png)

## Interpretation Notes

- **Nonlinearity and the Laplace approximation.** The model is nonlinear in its random effects because subject-level `LogNormal` parameters (`A_i`, `B_i`, `k1_i`, `k2_i`) appear inside a bi-exponential trajectory. This nonlinearity is precisely what necessitates the Laplace approximation rather than a simpler linear mixed-effects approach.
- **Why the censored likelihood matters.** Left-censored rows contribute through the cumulative Normal probability of falling below 1.7, not through a standard density evaluated at the pinned recorded value. This distinction is critical for unbiased estimation whenever detection limits are present.
- **The slow phase is only weakly identified here.** With a large fraction of observations censored at the detection limit, the data carry little information about the slow second exponential, so its amplitude estimate collapses toward zero and the fitted trajectory is effectively dominated by the fast phase. This is the data speaking rather than a defect: the observable dynamics are a single decline to the detection limit. Datasets with longer follow-up or a higher detection limit would identify both phases more sharply.
- **A reusable template.** The workflow demonstrated here -- model definition, data binding, Laplace estimation, diagnostics, and uncertainty quantification -- serves as a baseline template for censored nonlinear mixed-effects analyses in NoLimits. For datasets with higher censoring fractions or more complex censoring patterns (e.g., interval censoring), the same `censored(...)` syntax generalizes naturally.
