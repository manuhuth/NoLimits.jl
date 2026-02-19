# Mixed-Effects Tutorial 3: Neural Differential-Equation Components (SAEM)

In many scientific domains -- from systems biology to chemical engineering to ecology -- we understand the broad structure of a dynamical system (e.g., compartments connected by transfer rates) but lack precise knowledge of every rate law or interaction term. Neural ordinary differential equations (Neural ODEs) offer a principled way to address this gap: they embed small neural networks directly inside an ODE system, allowing the data to shape the functional forms that mechanistic reasoning alone cannot specify. Crucially, this approach preserves the interpretable compartmental structure while letting learned components capture the unknown nonlinearities.

In this tutorial, you will build a mixed-effects ODE model in which multiple neural-network components parameterize the ODE right-hand side, and fit it with the Stochastic Approximation Expectation-Maximization (SAEM) algorithm. By the end, you will have a working example of a hybrid mechanistic-neural model that accounts for between-subject variability through random effects on the network weights themselves.

## Learning Goals

By the end of this tutorial, you will be able to:

- **Declare neural-network parameter blocks** (`NNParameters`) and wire them into an ODE system using the `@DifferentialEquation` macro.
- **Couple network weights to subject-level random effects** via multivariate normal distributions, so that every individual in the dataset receives a personalized version of the dynamics.
- **Fit the model using SAEM** with Gaussian-block closed-form updates, a strategy that improves stability when random-effect dimensions are high.
- **Visualize and diagnose** the fitted trajectories and observation distributions.

## Step 1: Data Setup

In this step, you will load and prepare the data. We use the classic Theophylline dataset, which records concentration-time profiles for 12 subjects after oral administration. Although this dataset originates from pharmacology, the underlying dynamics -- a substance entering a depot compartment and transferring to a central compartment where it is observed and cleared -- are a standard example of a two-compartment transfer system that arises across many fields, from tracer kinetics to nutrient cycling. You will reshape the data into a flat format where the initial amount `d` enters as a constant covariate.

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
using Lux
using Turing

include(joinpath(@__DIR__, "_data_loaders.jl"))

Random.seed!(321)

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

## Step 2: Define the Neural ODE Mixed-Effects Model

In this step, you will define the core model. The key idea is that instead of specifying closed-form rate functions (such as first-order kinetics), you let neural networks learn these functions directly from data. Each `NNParameters` block declares a small feedforward network whose flattened weights become part of the fixed-effects parameter vector. At evaluation time, a callable function (e.g., `NNA1`) reconstructs the network from its weight vector and evaluates it -- so you can use it inside `@DifferentialEquation` just like any other function.

The ODE right-hand side uses four neural components arranged in a two-compartment transfer system:

- `fA1(t)` and `fA2(t)` govern the dynamics of the depot (input) compartment.
- `fC1(t)` and `fC2(t)` govern the dynamics of the central (observed) compartment.

To capture between-subject variability, each network's weight vector is paired with a subject-level random-effect vector (`etaA1`, `etaA2`, `etaC1`, `etaC2`) drawn from a `MvNormal` distribution centered on the population weights. This means every individual effectively receives their own personalized network: the population learns shared structure, while the random effects allow individual departures.

```julia
using NoLimits
using Distributions
using LinearAlgebra
using OrdinaryDiffEq
using Lux

width_nn = 2
chain_A1 = Lux.Chain(Lux.Dense(1, width_nn, tanh), Lux.Dense(width_nn, 1))
chain_A2 = Lux.Chain(Lux.Dense(1, width_nn, tanh), Lux.Dense(width_nn, 1))
chain_C1 = Lux.Chain(Lux.Dense(1, width_nn, tanh), Lux.Dense(width_nn, 1))
chain_C2 = Lux.Chain(Lux.Dense(1, width_nn, tanh), Lux.Dense(width_nn, 1))

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

        zA1 = NNParameters(chain_A1; function_name=:NNA1, calculate_se=false)
        zA2 = NNParameters(chain_A2; function_name=:NNA2, calculate_se=false)
        zC1 = NNParameters(chain_C1; function_name=:NNC1, calculate_se=false)
        zC2 = NNParameters(chain_C2; function_name=:NNC2, calculate_se=false)
    end

    @randomEffects begin
        etaA1 = RandomEffect(MvNormal(zA1, Diagonal(ones(length(zA1)))); column=:ID)
        etaA2 = RandomEffect(MvNormal(zA2, Diagonal(ones(length(zA2)))); column=:ID)
        etaC1 = RandomEffect(MvNormal(zC1, Diagonal(ones(length(zC1)))); column=:ID)
        etaC2 = RandomEffect(MvNormal(zC2, Diagonal(ones(length(zC2)))); column=:ID)
    end

    @DifferentialEquation begin
        a_A(t) = softplus(depot)
        x_C(t) = softplus(center)

        fA1(t) = softplus(NNA1([t / 24], etaA1)[1])
        fA2(t) = softplus(NNA2([a_A(t)], etaA2)[1])
        fC1(t) = -softplus(NNC1([x_C(t)], etaC1)[1])
        fC2(t) = softplus(NNC2([t / 24], etaC2)[1])

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

Before moving on, inspect the model structure to verify that all blocks were assembled correctly.

### Model Summary

```julia
model_summary = NoLimits.summarize(model)
model_summary
```

## Step 3: Build the `DataModel` and Configure SAEM

In this step, you will pair the model with the data by constructing a `DataModel`, then configure the SAEM algorithm. SAEM alternates between sampling random effects conditional on the current parameter estimates (the E-step) and updating population parameters (the M-step). A key practical detail is the `builtin_stats=:closed_form` option, which enables built-in closed-form block updates rather than gradient-based optimization for those mapped parameters. This can substantially improve stability when the random-effect dimension is high, as it is here with four separate network weight vectors per subject.

Two configuration choices deserve attention. First, the `re_mean_params` mapping tells SAEM which fixed-effect parameter serves as the mean of each random-effect distribution, enabling the closed-form update. Second, the `ebe_multistart_n` and `ebe_multistart_k` settings control multistart initialization of the empirical Bayes estimates, helping avoid poor local optima in the early iterations.

```julia
dm = DataModel(model, df; primary_id=:ID, time_col=:t)

saem_method = NoLimits.SAEM(;
    sampler=MH(),
    builtin_stats=:closed_form,
    re_mean_params=(; etaA1=:zA1, etaA2=:zA2, etaC1=:zC1, etaC2=:zC2),
    re_cov_params=NamedTuple(),
    resid_var_param=:sigma,
    maxiters=1000,
    mcmc_steps=80,
    t0=30,
    turing_kwargs=(n_samples=80, n_adapt=0, progress=false),
    optim_kwargs=(maxiters=300,),
    progress=true,
    ebe_multistart_n=300,
    ebe_multistart_k=3,
    ebe_rescue_on_high_grad=false,
    verbose=false,
)

serialization = SciMLBase.EnsembleThreads()
```

Before fitting, review the data model summary to confirm that individuals, covariates, and grouping structures were parsed as expected.

### DataModel Summary

```julia
dm_summary = NoLimits.summarize(dm)
dm_summary
```

## Step 4: Fit the Model and Inspect Results

You are now ready to run SAEM. The algorithm will iterate up to 1000 times, using Metropolis-Hastings sampling for the random effects within each E-step. After fitting, you will extract the final objective value and the number of estimated parameters as an initial sanity check.

```julia
res_saem = fit_model(
    dm,
    saem_method;
    serialization=serialization,
    rng=Random.Xoshiro(21),
)

(
    objective=NoLimits.get_objective(res_saem),
    n_params=length(NoLimits.get_params(res_saem; scale=:untransformed)),
)
```

For a more detailed summary of the fit -- including parameter estimates and convergence diagnostics -- call the `summarize` function.

```julia
fit_summary_saem = NoLimits.summarize(res_saem)
fit_summary_saem
```

## Step 5: Visualize Fitted Trajectories

In this step, you will overlay the model predictions on the raw observations for the first two subjects. Plotting fitted trajectories against observed data provides an immediate visual assessment of model adequacy -- you should see the neural ODE tracking the characteristic rise-and-decay pattern of the two-compartment transfer system, with subject-specific variation captured by the random effects.

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

## Step 6: Inspect the Observation Distribution

As a final diagnostic, you will examine the implied observation distribution at a single data point for the first individual. This plot shows the full predictive distribution (not just the point estimate), which helps you assess whether the residual variance is well-calibrated and whether the model's uncertainty is reasonable.

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

- This modeling pattern combines mechanistic compartmental states with learned nonlinear rate functions inside a single mixed-effects ODE. The compartmental structure encodes known domain knowledge (e.g., mass conservation, transfer between compartments), while the neural networks fill in the unknown functional forms. This hybrid strategy is broadly applicable to any system where the topology is known but the governing laws are not fully specified.
- The built-in closed-form updates (`builtin_stats=:closed_form`) can materially improve SAEM stability when the random-effect dimension is large, as is typical with neural-network weight vectors.
- The hyperparameter settings used here (network width, iteration counts, tolerance levels) are intentionally modest to keep the tutorial fast. For production analyses, consider increasing `maxiters`, `mcmc_steps`, and the number of MCMC samples to ensure thorough convergence, and tightening the ODE solver tolerances.
