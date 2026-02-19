# Fixed-Effects Tutorial 1: Nonlinear Longitudinal Model (MLE + MAP)

Many biological processes -- organ growth, tumor progression, enzyme saturation -- follow sigmoidal trajectories that cannot be captured by linear regression. Fitting such curves to repeated measurements collected over time is a core task in longitudinal data analysis. In this tutorial, you will build a logistic growth model for tree-trunk circumference data, estimate its parameters with two complementary methods (maximum likelihood and maximum a posteriori), and examine how prior information shapes the result. No random effects are introduced here; a later tutorial extends the same model to account for inter-subject variability.

## What You Will Learn

This tutorial walks through the complete modelling cycle for a fixed-effects nonlinear model in NoLimits.jl. Along the way you will learn how to:

- **Translate a scientific growth equation** into a NoLimits model specification.
- **Connect that specification to observed data** by constructing a `DataModel`.
- **Estimate parameters under two philosophies.** Maximum likelihood estimation (MLE) finds the parameter values that make the observed data most probable, treating the likelihood as the sole objective. Maximum a posteriori estimation (MAP) augments the likelihood with prior distributions on the parameters, which can improve numerical stability when some parameters are weakly identified -- a common situation with sigmoidal models whose asymptote or inflection point lies outside the observed time window.
- **Compare and diagnose** the resulting estimates by visualising fitted trajectories and inspecting observation-level distributions.

By the end, you will have a working template that you can adapt to your own nonlinear longitudinal datasets.

## Step 1: Load the Data

In this step, you will load the dataset used throughout the tutorial: the classic Orange tree growth study, originally published by Draper and Smith and available in R as `datasets::Orange`. It records the trunk circumference of five orange trees measured at seven time points each, making it a compact but representative example of nonlinear longitudinal data.

```julia
using NoLimits
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

Each row records one measurement occasion. The `Tree` column identifies the individual, `age` gives the tree's age in days since planting, and `circumference` is the trunk circumference in millimetres. Because every tree is measured at the same set of ages, this is a balanced design -- convenient for illustration, though NoLimits.jl handles unbalanced data equally well.

## Step 2: Define a Nonlinear Fixed-Effects Model

In this step, you will express the scientific model as a NoLimits model specification. The structural model is a three-parameter logistic growth curve, one of the simplest sigmoidal functions used in biology:

$$\mu(t) = \frac{a}{1 + \exp\!\bigl(-k\,(t - t_0)\bigr)}$$

Here, `a` sets the upper asymptote (the maximum circumference the model can predict), `k` controls the steepness of the growth phase, and `t0` locates the inflection point -- the age at which growth is fastest. Observations are assumed normally distributed around this deterministic trajectory with standard deviation `sigma`.

Notice that we also attach prior distributions to each parameter. MLE will ignore these priors entirely, but they become essential for MAP estimation in Step 4. The priors encode only weak domain knowledge: circumferences are positive and on the order of tens to hundreds of millimetres, growth rates are small and positive, and the inflection point falls somewhere within the range of observed ages.

```julia
using NoLimits
using Distributions

model = @Model begin
    @covariates begin
        age = Covariate()
    end

    @fixedEffects begin
        a = RealNumber(100.0, prior=Normal(100.0, 30.0), calculate_se=true)
        k = RealNumber(0.01, prior=Normal(0.01, 0.01), calculate_se=true)
        t0 = RealNumber(500.0, prior=Normal(500.0, 200.0), calculate_se=true)
        σ = RealNumber(5.0, scale=:log, prior=LogNormal(1.0, 0.5), calculate_se=true)
    end

    @formulas begin
        μ = a / (1 + exp(-k * (age - t0)))
        circumference ~ Normal(μ, σ)
    end
end
```

A few points worth noting about this specification. The `@covariates` block declares `age` as a time-varying covariate, which tells NoLimits to look up a column called `age` in the data frame. The residual standard deviation `sigma` is parameterised on the log scale (`scale=:log`), which constrains it to be strictly positive during optimisation without requiring explicit bounds. Finally, the `@formulas` block first computes the deterministic prediction `mu` and then declares that the observed column `circumference` follows a normal distribution centred on that prediction.

### Model Summary

Before fitting, it is good practice to inspect the model structure programmatically. The summary confirms which parameters are declared as fixed effects, which covariates are expected, and what form the observation model takes.

```julia
model_summary = NoLimits.summarize(model)
model_summary
```

## Step 3: Build the DataModel and Configure Estimation

With the model and data in hand, you will now pair them into a `DataModel` -- the central object that validates the data against the model specification and prepares the internal structures needed for estimation. In the same step, you will configure the two estimation methods (MLE and MAP), specifying a maximum of 150 optimiser iterations for each.

```julia
dm = DataModel(model, df; primary_id=:Tree, time_col=:age)

mle_method = NoLimits.MLE(; optim_kwargs=(maxiters=150,))
map_method = NoLimits.MAP(; optim_kwargs=(maxiters=150,))

serialization = SciMLBase.EnsembleThreads()
```

The `primary_id=:Tree` argument tells NoLimits which column identifies individuals, and `time_col=:age` specifies the time axis. Setting `serialization=EnsembleThreads()` enables multi-threaded likelihood evaluation across individuals -- a minor convenience here with only five trees, but essential for larger datasets where per-individual computations dominate runtime.

### DataModel Summary

It is worth verifying that the data were parsed correctly. The summary should confirm five individuals, seven observations each, no missing values, and no events.

```julia
dm_summary = NoLimits.summarize(dm)
dm_summary
```

## Step 4: Fit with MLE and MAP

You are now ready to estimate the model parameters under both methods. Each call to `fit_model` runs a gradient-based optimiser (L-BFGS by default), starting from the initial values declared in the `@fixedEffects` block. The key difference lies in the objective function: MLE maximises the log-likelihood alone, while MAP maximises the log-likelihood plus the log-prior.

```julia
res_mle = fit_model(dm, mle_method; serialization=serialization, rng=Random.Xoshiro(41))
res_map = fit_model(dm, map_method; serialization=serialization, rng=Random.Xoshiro(42))

(
    objective_mle=NoLimits.get_objective(res_mle),
    objective_map=NoLimits.get_objective(res_map),
)
```

The returned objective values are negative log-likelihoods (for MLE) or negative log-posteriors (for MAP), so lower values indicate a better fit. Because the MAP objective includes additional prior penalty terms, its numerical value is not directly comparable to the MLE objective; what matters at this stage is that each method converged successfully.

To verify convergence and examine standard errors, you can request a detailed fit summary. This reports convergence status, the final objective value, and standard errors for all parameters whose `calculate_se` flag was set to `true`.

```julia
fit_summary_mle = NoLimits.summarize(res_mle)
fit_summary_map = NoLimits.summarize(res_map)

fit_summary_mle
```

## Step 5: Compare Parameter Estimates

With both fits complete, you can now extract the estimated parameters on the original (untransformed) scale and compare them side by side.

```julia
θ_mle = NoLimits.get_params(res_mle; scale=:untransformed)
θ_map = NoLimits.get_params(res_map; scale=:untransformed)

(
    mle=θ_mle,
    map=θ_map,
)
```

Because this dataset is relatively small (35 observations total), the priors exert a noticeable pull on the MAP estimates. In particular, parameters whose likelihood surface is flat -- such as the asymptote `a`, which is only weakly constrained when few trees have approached their maximum size -- will be drawn toward the prior mean under MAP but left at the pure likelihood optimum under MLE. As sample size grows, the likelihood dominates and the two methods converge to the same values. This behaviour serves as a useful diagnostic: large MLE--MAP discrepancies flag parameters that the data alone cannot pin down precisely.

## Step 6: Fitted Trajectories

Plotting model predictions against raw observations is the most direct way to assess goodness of fit. In this step, you will overlay the estimated logistic curves on the data for the first two trees under each estimation method.

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

Because there are no random effects in this model, every tree is predicted by the same population-level curve. Deviations between the curve and individual data points therefore reflect both measurement noise and genuine between-tree variability that the current model does not capture. If these residual patterns appear systematic -- for example, if one tree consistently lies above the curve while another lies below -- that is a strong signal that random effects should be introduced (see the mixed-effects tutorials).

MAP fit plot:

```julia
p_fit_map
```

The MAP trajectories may appear nearly identical to the MLE trajectories or show subtle shifts, depending on how informative the priors are relative to the data. Comparing the two plots is a quick visual assessment of prior influence.

## Step 7: Observation Distribution Diagnostics

Beyond trajectory plots, it is informative to inspect the implied observation distribution at individual time points. In this step, you will plot the normal distribution predicted by the model at the first observation of the first tree, with the actual observed value marked. A well-calibrated model places observed values near the centre of the predicted distribution rather than in the tails.

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

MAP observation-distribution plot:

```julia
p_obs_map
```

If an observed value sits far in the tail of the predicted distribution, this may indicate model misspecification -- for instance, a residual error distribution that is too narrow, or a structural model that systematically over- or under-predicts at certain ages.

## Summary and Next Steps

This tutorial demonstrated the end-to-end workflow for fixed-effects nonlinear modelling in NoLimits.jl: specifying a scientifically motivated structural model, pairing it with data, estimating parameters via MLE and MAP, and diagnosing the results. Several practical points are worth keeping in mind:

- **MLE is purely data-driven.** It finds the parameter values that maximise the likelihood of the observed data. Declared priors have no effect on the MLE objective.
- **MAP incorporates prior knowledge.** By adding log-prior terms to the objective, MAP can stabilise estimation when the data provide limited information about certain parameters -- a common situation in nonlinear models with saturation or asymptotic behaviour.
- **Fixed-effects models assume no between-subject variability.** Every individual is predicted by the same curve. When individual trajectories diverge systematically, random effects are needed to account for that heterogeneity. The mixed-effects tutorials show how to extend this model accordingly.
- **Diagnostic tools are method-agnostic.** Whether you use MLE, MAP, or any other estimation method in NoLimits.jl, functions such as `plot_fits` and `plot_observation_distributions` work identically, making it straightforward to compare results across methods.
