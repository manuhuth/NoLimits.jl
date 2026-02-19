# Mixed-Effects Tutorial 6: Left-Censored Nonlinear Model (Laplace)

In many biomedical assays, measurements below a detection threshold cannot be reliably quantified. This situation, known as *left-censoring*, arises whenever an instrument's lower limit of quantification (LLOQ) truncates the observable range. HIV viral load monitoring is a canonical example: modern RT-PCR assays report "below detectable limit" for viral RNA concentrations under approximately 50 copies/mL -- or equivalently, below about 1.7 on the log10 scale. In patients responding well to antiretroviral therapy, censored observations can account for 30--40% of all measurements, making proper statistical treatment essential.

How should these below-limit values be handled? The answer matters. Dropping censored rows discards information and inflates uncertainty. Substituting the detection limit as though it were a true measurement concentrates probability mass at that value and biases parameter estimates downward. The principled approach is a *censored likelihood*: uncensored observations contribute their usual probability density, while censored observations contribute the cumulative probability of falling at or below the detection threshold. This formulation correctly encodes what we actually know -- that the true value lies somewhere below the limit, without committing to a specific magnitude.

In this tutorial, you will fit a nonlinear mixed-effects model to the `virload50` dataset from the `npde` R package, which contains longitudinal log10 viral load measurements from 50 HIV-positive patients. The structural model is a bi-exponential decay function that captures two distinct phases of viral dynamics: rapid initial suppression and slower long-term decline. Subject-specific parameters enter through `LogNormal` random effects, making the model nonlinear in the random effects. You will handle left-censoring directly in the observation model using NoLimits' `censored(...)` syntax and estimate the model with the Laplace approximation.

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
        beta_A = RealNumber(5.5, calculate_se=true)
        beta_B = RealNumber(4.2, calculate_se=true)
        beta_k1 = RealNumber(-1.6, calculate_se=true)
        beta_k2 = RealNumber(-3.5, calculate_se=true)

        omega_A = RealNumber(0.20, scale=:log, calculate_se=true)
        omega_B = RealNumber(0.20, scale=:log, calculate_se=true)
        omega_k1 = RealNumber(0.20, scale=:log, calculate_se=true)
        omega_k2 = RealNumber(0.20, scale=:log, calculate_se=true)

        sigma = RealNumber(0.18, scale=:log, calculate_se=true)
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

## Step 3: Build `DataModel` and Configure `Laplace`

In this step, you will bind the model to the dataset by constructing a `DataModel`. This validates that all required columns are present and correctly typed, groups observations by subject, and assembles the internal data structures needed for estimation.

Next, you will configure the Laplace approximation. This method approximates the marginal likelihood by finding the mode of each subject's conditional random-effects distribution (the empirical Bayes estimates) and using a second-order Taylor expansion around those modes to integrate out the random effects analytically. Two sets of optimization controls are available: `inner_kwargs` governs the per-subject random-effects optimization, while `optim_kwargs` governs the outer fixed-effects optimization. Multistart is disabled here for speed, but enabling it can help avoid local optima in models with more complex likelihood surfaces.

```julia
dm = DataModel(model, df; primary_id=:ID, time_col=:Time)

laplace_method = NoLimits.Laplace(;
    optim_kwargs=(maxiters=250,),
    inner_kwargs=(maxiters=100,),
    multistart_n=0,
    multistart_k=0,
)

serialization = SciMLBase.EnsembleThreads()

NoLimits.summarize(dm)
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

## Step 7: Wald Uncertainty Quantification

In this final step, you will assess the precision of the estimated fixed effects by computing Wald-based confidence intervals. The Wald method constructs approximate 95% intervals from the observed Fisher information matrix -- that is, the curvature of the log-likelihood at the optimum. This is computationally inexpensive and provides a practical first assessment of parameter identifiability: wide intervals may signal that the data contain insufficient information to pin down a particular parameter.

```julia
uq = compute_uq(
    res;
    method=:wald,
    n_draws=800,
    level=0.95,
)

NoLimits.summarize(uq)
```

For a consolidated report combining point estimates and their uncertainty, pass both the fit result and the uncertainty object to `summarize`. This tabular format is convenient for inclusion in manuscripts and supplementary materials.

```julia
NoLimits.summarize(res, uq)
```

Finally, you can visualize the implied sampling distributions of the fixed-effects parameters on the natural (untransformed) scale. Parameters estimated on the log scale (the `omega` and `sigma` parameters) are back-transformed before plotting, so the density plots reflect the scale on which these quantities are scientifically interpretable.

```julia
plot_uq_distributions(uq; scale=:natural, plot_type=:density, show_legend=false)
```

## Interpretation Notes

- **Nonlinearity and the Laplace approximation.** The model is nonlinear in its random effects because subject-level `LogNormal` parameters (`A_i`, `B_i`, `k1_i`, `k2_i`) appear inside a bi-exponential trajectory. This nonlinearity is precisely what necessitates the Laplace approximation rather than a simpler linear mixed-effects approach.
- **Why the censored likelihood matters.** Left-censored rows contribute through the cumulative Normal probability of falling below 1.7, not through a standard density evaluated at the pinned recorded value. This distinction is critical for unbiased estimation whenever detection limits are present.
- **A reusable template.** The workflow demonstrated here -- model definition, data binding, Laplace estimation, diagnostics, and uncertainty quantification -- serves as a baseline template for censored nonlinear mixed-effects analyses in NoLimits. For datasets with higher censoring fractions or more complex censoring patterns (e.g., interval censoring), the same `censored(...)` syntax generalizes naturally.
