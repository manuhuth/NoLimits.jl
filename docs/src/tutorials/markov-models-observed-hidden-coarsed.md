# Markov Models in NoLimits.jl: Observed, Hidden, Coarsed, and Pathsum

Multi-state models are a natural framework for data in which an individual, system, or process moves between a finite number of discrete states over time. Rather than treating repeated observations as isolated outcomes, these models focus on the transitions between states: remaining where one is, moving forward, recovering, relapsing, or entering an absorbing state. This makes them useful across many settings, including disease progression, reliability analysis, behavioral processes, event history data, and longitudinal categorical outcomes.

A common simplifying assumption is the Markov property: conditional on the current state, the future evolution of the process does not depend on the full past history. In practice, this assumption is often a useful modeling compromise. It captures the local dependence that is most directly observed in longitudinal state data, while keeping the transition structure interpretable and statistically tractable. Covariates, time effects, and subject-level random effects can then be used to explain differences in transition behavior across individuals or experimental units.

In this tutorial, you will use NoLimits.jl to specify and estimate several Markov-type models for longitudinal state data. The examples illustrate how the same general idea — modeling movement through a discrete state space — can be adapted to different observation schemes and assumptions. We will consider four related models: 

1. An observed Markov model, where the state sequence is directly measured; 
2. A hidden Markov model, where the observed data are noisy proxies for an underlying latent state process; 
3. A Markov model for set-valued observations, where each observation may correspond to multiple possible states; 
4. An acyclic continuous-time Markov model, where the absence of cycles allows transition probabilities to be computed efficiently using the matrix exponential.

Together, these examples show how Markov models can be used in NoLimits.jl to represent discrete dynamic systems, accommodate imperfect or partial state information, and move between discrete-time and continuous-time formulations.

## What You Will Learn

By the end of this tutorial, you will be able to:

- **Build** a Markov model in discrete and continuous time.
- **Configure** the emission setup based on your modelling assumptions.
- **Fit** DT-HMM and CT-HMM models in NoLimits to externally generated data.

## Step 0: Setup and Data generation

```julia
using Random
using DataFrames
using Distributions
using LinearAlgebra
using Statistics
using NoLimits

Random.seed!(2026)
```

### Define Relevant Model Components

We use a 3-state setup as in the example scripts. To keep interpretation clear, we separate the components:

- `P_true`: discrete-time transition matrix for one-step transitions.
- `Q_true`: continuous-time generator matrix for rate-based transitions across arbitrary `dt`.
- `pi_true`: initial-state distribution at the first observation.
- `B_true`: emission matrix (`state -> observed label probabilities`) for HMM outcomes.

```julia
P_true = [
    0.82 0.13 0.05
    0.10 0.78 0.12
    0.06 0.14 0.80
]

Q_true = [
    -0.35 0.25 0.10
     0.12 -0.32 0.20
     0.08 0.16 -0.24
]

pi_true = [0.60, 0.30, 0.10]

B_true = [
    0.85 0.10 0.05
    0.10 0.80 0.10
    0.05 0.20 0.75
]

@assert all(abs.(sum(P_true, dims=2) .- 1.0) .< 1e-12)
@assert all(abs.(sum(B_true, dims=2) .- 1.0) .< 1e-12)
@assert abs(sum(pi_true) - 1.0) < 1e-12
```

### Generate Synthetic Data

In this tutorial, data simulation is done manually with the following stpes:

- initial hidden-state draw from `pi_true`,
- hidden-state transition draw from `P_true` (or `exp(Q_true*dt)`),
- emission draw from state-specific categorical probabilities in `B_true`.

We then fit NoLimits models to this externally generated data.

```julia
function simulate_dt_hmm_df(; n_id::Int, n_time::Int, P::AbstractMatrix, pi::AbstractVector, B::AbstractMatrix, rng=Random.default_rng())
    rows = NamedTuple[]
    K = size(P, 1)
    y_labels = 1:size(B, 2)

    for id in 1:n_id
        z = rand(rng, Categorical(pi))
        for t_idx in 1:n_time
            y = rand(rng, Categorical(vec(B[z, :])))
            push!(rows, (ID=id, t=Float64(t_idx - 1), y=y, z_true=z))
            z = rand(rng, Categorical(vec(P[z, :])))
        end
    end

    return DataFrame(rows)
end

function simulate_ct_hmm_df(; n_id::Int, n_time::Int, Q::AbstractMatrix, pi::AbstractVector, B::AbstractMatrix, dt_choices=[0.25, 0.5, 1.0], rng=Random.default_rng())
    rows = NamedTuple[]
    K = size(Q, 1)

    for id in 1:n_id
        z = rand(rng, Categorical(pi))
        t_now = 0.0
        for t_idx in 1:n_time
            dt = rand(rng, dt_choices)
            t_now += dt

            y = rand(rng, Categorical(vec(B[z, :])))
            push!(rows, (ID=id, t=t_now, dt=dt, y=y, z_true=z))

            Pt = exp(Q * dt)
            z = rand(rng, Categorical(vec(Pt[z, :])))
        end
    end

    return DataFrame(rows)
end
```

```julia
df_dt = simulate_dt_hmm_df(
    n_id=100,
    n_time=20,
    P=P_true,
    pi=pi_true,
    B=B_true,
    rng=Random.Xoshiro(1101),
)

df_ct = simulate_ct_hmm_df(
    n_id=100,
    n_time=20,
    Q=Q_true,
    pi=pi_true,
    B=B_true,
    dt_choices=[0.25, 0.5, 1.0],
    rng=Random.Xoshiro(1102),
)

first(df_dt, 6), first(df_ct, 6)
```

## Step 1: Fit a Discrete-Time MM with observed state space

First we fit the manually generated `df_dt` with `DiscreteTimeObservedStatesMarkovModel`.

Here the observed outcome is the state label itself (`z_true`). There is no separate emission layer.

### Parameterization notes

- `DiscreteTransitionMatrix` enforces row-stochasticity (each row sums to one) through an unconstrained internal representation.
- `ProbabilityVector` ensures the initial-state probabilities stay on the simplex.

This means optimization can proceed on unconstrained parameters while NoLimits guarantees valid probabilities at every iteration.

```julia
model_dt = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        P = DiscreteTransitionMatrix([
            0.70 0.20 0.10
            0.20 0.65 0.15
            0.10 0.25 0.65
        ])
        pi0 = ProbabilityVector([0.50, 0.30, 0.20])
    end

    @formulas begin
        z_true ~ DiscreteTimeObservedStatesMarkovModel(
            P,
            Categorical(pi0),
        )
    end
end
```

```julia
dm_dt = DataModel(model_dt, select(df_dt, Not(:y)); primary_id=:ID, time_col=:t)
res_dt = fit_model(dm_dt, MLE(optim_kwargs=(maxiters=200,)); rng=Random.Xoshiro(1201))
```

```julia
uq_par = uq = NoLimits.compute_uq(
    res_dt;
    method=:wald,
    vcov=:hessian,
    pseudo_inverse=false,
)
NoLimits.summarize(uq_par)
```

**Note:** The parameters are named by the macro `@fixedEffects` based on the given parametrization and name. In this example the stick-breaking transform is applied toe ach row of the transition matrix, yielding a direct fit of the first two parameters in that row and the third is indirectly defined by the constrained, that row-sums are equal to one.

## Step 2: Fit a Continuous-Time HMM in NoLimits

The previous model assumed that the Markov state is directly observed. In many applications this is unrealistic:
the true state is latent and one observes noisy proxies generated from that state.

This is exactly the Hidden Markov Model (HMM) setting. We therefore combine:

- continuous-time latent-state dynamics (`Q`, `dt`), and
- state-specific emission distributions (`e1`, `e2`, `e3`).

### Parameterization notes

- `ContinuousTransitionMatrix` parameterizes a valid generator matrix: off-diagonal entries are non-negative rates and diagonals are derived as minus row sums.
- `dt` enters as a covariate and determines transition probabilities through `exp(Q * dt)` internally.
- Emission rows are declared as `ProbabilityVector`s, then used as `Categorical` distributions.

```julia
model_ct = @Model begin
    @covariates begin
        t = Covariate()
        dt = Covariate()
    end

    @fixedEffects begin
        Q = ContinuousTransitionMatrix([
            -0.30 0.20 0.10
             0.10 -0.25 0.15
             0.08 0.12 -0.20
        ])
        pi0 = ProbabilityVector([0.50, 0.30, 0.20])

        e1 = ProbabilityVector([0.70, 0.20, 0.10]) # emission from state 1 to observables
        e2 = ProbabilityVector([0.20, 0.60, 0.20]) # emission from state 2 to observables
        e3 = ProbabilityVector([0.10, 0.20, 0.70]) # emission from state 3 to observables
    end

    @formulas begin
        y ~ ContinuousTimeDiscreteStatesHMM(
            Q,
            (Categorical(e1), Categorical(e2), Categorical(e3)),
            Categorical(pi0),
            dt;
        )
    end
end

dm_ct = DataModel(model_ct, select(df_ct, Not(:z_true)); primary_id=:ID, time_col=:t)
res_ct = fit_model(dm_ct, MLE(optim_kwargs=(maxiters=220,)); rng=Random.Xoshiro(1202))
NoLimits.summarize(res_ct)
```

## Step 3: Coarsed Observations

In some studies, the observed state is ambiguous. Instead of observing a single label, one may only know that
the process is in one of several possible states, e.g. `{2,3}`.

This is a coarsed/set-valued observation problem. In NoLimits, this is modeled explicitly with `coarsed(...)`:

- data column contains vectors such as `[1]`, `[2,3]`, `[3]`,
- formula uses `coarsed(DiscreteTimeObservedStatesMarkovModel(...))`.

Likelihood contributions are then sums over the compatible states, rather than point-probabilities for a single label.

```julia
# Build a coarsed version of the observed-state data.
# We map exact labels to sets:
#   1 -> [1], 2 -> [2,3], 3 -> [3]
# so state 2 becomes ambiguous between states 2 and 3.

coarsed_label(z::Int) = z == 1 ? [1] : (z == 2 ? [2, 3] : [3])

df_coarsed = select(df_dt, [:ID, :t, :z_true])
rename!(df_coarsed, :z_true => :z_coarse)
df_coarsed.z_coarse = [coarsed_label(Int(v)) for v in df_coarsed.z_coarse]

first(df_coarsed, 8)
```

```julia
model_coarsed = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        Q = ContinuousTransitionMatrix([
            -0.30 0.20 0.10
             0.10 -0.25 0.15
             0.08 0.12 -0.20
        ])
        pi0 = ProbabilityVector([0.50, 0.30, 0.20])
    end

    @formulas begin
        dist = ContinuousTimeObservedStatesMarkovModel(Q, Categorical(pi0), [1, 2, 3])
        z_coarse ~ coarsed(dist)
    end
end

dm_coarsed = DataModel(model_coarsed, df_coarsed; primary_id=:ID, time_col=:t)
res_coarsed = fit_model(dm_coarsed, MLE(optim_kwargs=(maxiters=200,)); rng=Random.Xoshiro(1203))

NoLimits.summarize(res_coarsed)
```

## Step 4: Optional Pathsum for an Acyclic CT Model

For continuous-time models, NoLimits offers different state-propagation modes for the transition step between observations:

- `:expv`: general-purpose propagation based on matrix exponential logic. Works for any generator matrix (cyclic or acyclic).
- `:pathsum`: path-sum propagation specialized for **acyclic** transition graphs (DAG structure).


In CT-HMMs, propagation is executed at every observation interval. For long sequences or many individuals, this can dominate runtime.

- If the transition graph is acyclic, `:pathsum` can be substantially faster because it exploits graph structure and avoids generic dense operations.
- If the graph has cycles, `:pathsum` is not applicable and `:expv` (or `:auto` fallback to `:expv`) should be used.

We first verify numerical agreement, then compare runtime.

```julia
Q_acyclic = [
    -0.40 0.25 0.15
     0.00 -0.30 0.30
     0.00 0.00 0.00
]

h_expv = ContinuousTimeDiscreteStatesHMM(
    Q_acyclic,
    (Categorical([0.8,0.15,0.05]), Categorical([0.15,0.75,0.1]), Categorical([0.05,0.2,0.75])),
    Categorical([1.0, 0.0, 0.0]),
    1.0;
    propagation_mode=:expv,
)

h_path = ContinuousTimeDiscreteStatesHMM(
    Q_acyclic,
    (Categorical([0.8,0.15,0.05]), Categorical([0.15,0.75,0.1]), Categorical([0.05,0.2,0.75])),
    Categorical([1.0, 0.0, 0.0]),
    1.0;
    propagation_mode=:pathsum,
)

maximum(abs.(probabilities_hidden_states(h_expv) .- probabilities_hidden_states(h_path)))
```

```julia
# Runtime comparison: expv vs pathsum for repeated hidden-state propagation
n_warmup = 200
n_eval = 20_000
n_repeats = 5

# Warmup
for _ in 1:n_warmup
    probabilities_hidden_states(h_expv)
    probabilities_hidden_states(h_path)
end

t_expv = Float64[]
t_path = Float64[]

for _ in 1:n_repeats
    push!(t_expv, @elapsed begin
        for _ in 1:n_eval
            probabilities_hidden_states(h_expv)
        end
    end)

    push!(t_path, @elapsed begin
        for _ in 1:n_eval
            probabilities_hidden_states(h_path)
        end
    end)
end

runtime_summary = (
    expv_median = median(t_expv),
    pathsum_median = median(t_path),
    expv_mean = mean(t_expv),
    pathsum_mean = mean(t_path),
    speedup_expv_over_pathsum = median(t_expv) / median(t_path),
)

runtime_summary
```

## Step 5: Random Effects and Other Flexible Parameterizations

This notebook focuses on fixed-effects HMMs for clarity, but the same NoLimits building blocks extend to richer models:

- random effects on transition parameters,
- custom parametrization of transition rate matrices,
- covariate-dependent transition rates/probabilities,
- multivariate emissions
- etc.

Below is an example using covariate dependent transition rates on log-scale and emission with logistic parametrization.

Using random effects also requires a different estimation method such as Laplace, SAEM or MCEM.
```julia
model_custom = @Model begin
    @covariates begin
        t = Covariate()
        dt = Covariate()
        x_level_1 = Covariate()
    end

    @fixedEffects begin
        q1_2 = RealNumber(-2.047942874620465; scale=:identity)
        q2_3 = RealNumber(-1.8904754421672125; scale=:identity)
        q1_2_l1 = RealNumber(-2.0; scale=:identity)
        q2_3_l1 = RealNumber(-2.0; scale=:identity)

        sigma_q1_2 = RealNumber(0.05; scale=:log)
        sigma_q2_3 = RealNumber(0.05; scale=:log)

        e2_2 = RealNumber(0.5; scale=:identity)

        pi0 = ProbabilityVector([0.50, 0.30, 0.20])
    end

    @randomEffects begin
        eta1_2 = RandomEffect(Normal(0.0, sigma_q1_2); column=:ID)
        eta2_3 = RandomEffect(Normal(0.0, sigma_q2_3); column=:ID)
    end

    @formulas begin
        Q = [
            -(exp(q1_2 + q1_2_l1 * x_level_1 + eta1_2))  exp(q1_2 + q1_2_l1 * x_level_1 + eta1_2)  0.0;
            0.0  -(exp(q2_3 + q2_3_l1 * x_level_1 + eta2_3))  exp(q2_3 + q2_3_l1 * x_level_1 + eta2_3);
            0.0 0.0 0.0;
        ]
        
        w2_2 = logistic(e2_2)
        w2_3 = 1 - w2_2

        E = [
            1.0  0.0  0.0;
            0.0  w2_2 w2_3;
            0.0  0.0  1.0;
        ]

        y ~ ContinuousTimeObservedStatesMarkovModel(
            Q,
            Categorical(pi0),
            dt;
            propagation_mode=:auto,
        )
    end
end
```

We need to insert a covariate column into the data.

```julia
# Build a custom dataset for model_custom from the CT-HMM synthetic data.

# Reuse t/dt from df_ct and create x_level_1 (constant per individual here).
id_levels = unique(df_ct.ID)
id_to_x = Dict(id => rand(Random.Xoshiro(1301 + Int(id)), Bernoulli(0.5)) for id in id_levels)

df_custom = select(df_ct, [:ID, :t, :dt, :y])
df_custom.x_level_1 = [id_to_x[id] for id in df_custom.ID]

# Define logistic helper if not already available in this session.
if !isdefined(Main, :logistic)
    logistic(x) = inv(1 + exp(-x))
end
```

```julia
# Simulate a fresh dataset from model_custom using df_custom as structural template.
dm_custom_template = DataModel(model_custom, df_custom; primary_id=:ID, time_col=:t)
df_custom_sim = simulate_data(dm_custom_template; rng=Random.Xoshiro(1401), replace_missings=true)

first(df_custom_sim, 8)
```

```julia
dm_custom = DataModel(model_custom, df_custom_sim; primary_id=:ID, time_col=:t)
res_custom = fit_model(dm_custom, SAEM(); rng=Random.Xoshiro(1203))

NoLimits.summarize(res_custom)
```

