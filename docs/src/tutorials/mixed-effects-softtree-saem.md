# Mixed-Effects Tutorial 4: SoftTree Differential-Equation Components (SAEM)

When building mechanistic models of longitudinal data, you often know the broad structure of the system -- compartments, conservation laws, transfer pathways -- but not the precise functional forms that govern how material moves between states. Neural networks are one way to learn those unknown rate functions from data, as shown in Tutorial 3. Soft decision trees offer an appealing alternative. They can approximate arbitrary nonlinear mappings, yet their branching structure provides built-in feature selection and piecewise-smooth approximation that is often easier to interpret. For the low-dimensional inputs typical of scientific rate functions (a single state variable, or time itself), soft trees can match neural network flexibility with substantially fewer parameters.

In this tutorial, you will build a mixed-effects ODE model in which soft decision trees parameterize the ODE right-hand side, then estimate the model with the Stochastic Approximation Expectation-Maximization (SAEM) algorithm. The model is structurally parallel to Tutorial 3, so you can directly compare the two function-approximation strategies on the same data and compartmental structure.

## Learning Goals

By the end of this tutorial, you will be able to:

- Declare `SoftTreeParameters` blocks that create differentiable decision trees whose flattened parameters join the fixed-effects vector, exposing callable functions (e.g., `STA1`) for use inside `@DifferentialEquation`.
- Wire multiple soft trees into a two-compartment transfer ODE, letting the trees learn unknown rate functions from data.
- Couple each tree's parameter vector to subject-level random effects via `MvNormal` distributions, giving every individual a personalized version of the dynamics.
- Fit the model with SAEM using closed-form Gaussian updates for the random-effect means.
- Visualize individual-level trajectories and observation distributions to assess model adequacy.

## Step 1: Data Setup

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

## Step 2: Define SoftTree-Driven ODE Mixed-Effects Model

In this step, you will construct the full mixed-effects model. The guiding idea is the same as in the neural ODE tutorial: rather than specifying closed-form rate laws, you let data-driven function approximators learn the rate functions directly from observations. The difference is the choice of approximator. Each `SoftTreeParameters` block declares a soft decision tree with a specified input dimension and depth. The `depth_st` parameter controls expressiveness -- a tree of depth `d` has `2^d` leaf nodes, each contributing a smooth local approximation. The block's flattened parameters become part of the fixed-effects vector, and the associated callable function (e.g., `STA1`) evaluates the tree at any input.

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

Before moving on, inspect the assembled model to verify that all blocks -- covariates, fixed effects, random effects, ODE, and formulas -- are correctly wired together.

### Model Summary

```julia
model_summary = NoLimits.summarize(model)
model_summary
```

## Step 3: Build `DataModel` and Configure SAEM

In this step, you will pair the model with the observed data by constructing a `DataModel`, then configure the SAEM fitting algorithm. SAEM alternates between two phases: an E-step that samples subject-level random effects conditional on the current population parameters, and an M-step that updates those population parameters using stochastic sufficient statistics. Setting `builtin_stats=:closed_form` enables built-in closed-form block updates. This avoids gradient-based optimization for those mapped parameters and can substantially improve convergence stability, particularly when the random-effect vectors are high-dimensional (as they are here, since each tree's full parameter vector is individualized).

Several configuration details are worth noting. The `re_mean_params` mapping tells SAEM which fixed-effect parameter serves as the population mean for each random-effect distribution. The `ebe_multistart_n` and `ebe_multistart_k` settings control multistart initialization of the empirical Bayes estimates, reducing the risk of poor local optima during early iterations. Finally, `progress=true` displays SAEM iteration progress at the outer level, while `progress=false` inside `turing_kwargs` suppresses verbose output from the inner sampler.

```julia
dm = DataModel(model, df; primary_id=:ID, time_col=:t)

saem_method = NoLimits.SAEM(;
    sampler=MH(),
    builtin_stats=:closed_form,
    re_mean_params=(; etaA1=:gA1, etaA2=:gA2, etaC1=:gC1, etaC2=:gC2),
    re_cov_params=NamedTuple(),
    resid_var_param=:sigma,
    maxiters=1000,
    mcmc_steps=80,
    t0=30,
    turing_kwargs=(n_samples=80, n_adapt=0, progress=false),
    optim_kwargs=(maxiters=300,),
    verbose=false,
    progress=true,
    ebe_multistart_n=300,
    ebe_multistart_k=3,
    ebe_rescue_on_high_grad=false
)

serialization = SciMLBase.EnsembleThreads()
```

Before fitting, review the data model summary to confirm that individuals, covariates, and grouping structures were parsed as expected.

### DataModel Summary

```julia
dm_summary = NoLimits.summarize(dm)
dm_summary
```

## Step 4: Fit and Inspect Core Result Summary

In this step, you will run the SAEM algorithm and examine the results. The algorithm iterates up to 1000 times, using Metropolis-Hastings sampling for the random effects within each E-step. After fitting completes, you will extract the final objective value and parameter count as a quick sanity check before looking at more detailed diagnostics.

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

For a more detailed view -- including parameter estimates and convergence diagnostics -- call the `summarize` function on the fit result.

```julia
fit_summary_saem = NoLimits.summarize(res_saem)
fit_summary_saem
```


## Step 5: Fitted Trajectories (First 2 Individuals)

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

## Step 6: Observation Distribution Diagnostic

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

## Interpretation Notes

- This modeling pattern combines mechanistic compartmental structure with soft decision tree function approximators inside a single mixed-effects ODE. The compartments encode known domain knowledge (mass conservation, transfer pathways), while the trees learn the unknown rate functions from data. This separation means you retain interpretable system structure without needing to specify rate-law functional forms in advance.
- Compared to neural networks, soft trees can be more parameter-efficient for the low-dimensional inputs common in scientific rate functions, and their piecewise-smooth approximation may be easier to inspect post hoc. The choice between the two is problem-dependent; both can be embedded in the same NoLimits framework with minimal code changes (compare this tutorial with Tutorial 3).
- The built-in closed-form updates (`builtin_stats=:closed_form`) materially improve SAEM convergence stability when the individualized parameter vectors are high-dimensional, as they are here.
- The settings in this tutorial are intentionally modest to keep runtime short. For production analyses, consider increasing `maxiters`, `mcmc_steps`, and the number of MCMC samples to ensure thorough convergence.
