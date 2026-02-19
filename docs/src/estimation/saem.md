# SAEM

The Stochastic Approximation Expectation-Maximization (SAEM) algorithm is a widely used method for parameter estimation in nonlinear mixed-effects models. Unlike standard EM, SAEM replaces the intractable E-step expectation with a stochastic approximation that is updated incrementally across iterations. Each iteration consists of three steps:

- **E-step:** MCMC sampling of random effects conditional on the current fixed-effect estimates.
- **SA-step:** stochastic smoothing of sufficient statistics (or stored latent snapshots) using a decreasing gain sequence.
- **M-step:** fixed-effect update, performed either through numerical optimization or through user-supplied closed-form expressions.

SAEM is particularly well suited to models with complex nonlinearities, including ODE-based dynamics and function-approximator components such as neural networks or soft decision trees, because its convergence properties do not require closed-form integration over the random effects.

## Applicability

SAEM is designed for models that include both fixed and random effects:

- The model must declare at least one random effect and at least one free fixed effect.
- Multiple random-effect grouping columns and multivariate random effects are fully supported.

If fixed-effect priors are defined in the model, SAEM ignores them in its objective. To incorporate priors, use `LaplaceMAP` or `MCMC` instead.

## Basic Usage

The following example demonstrates a minimal SAEM workflow with a nonlinear mixed-effects model.

```julia
using NoLimits
using DataFrames
using Distributions
using Turing

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        b = RealNumber(0.1)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, 0.4); column=:ID)
    end

    @formulas begin
        mu = exp(a + b * t + eta)   # nonlinear in random effects
        y ~ LogNormal(log(mu), sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y = [1.0, 1.25, 0.95, 1.18, 1.05, 1.42],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

method = NoLimits.SAEM(;
    sampler=MH(),
    turing_kwargs=(n_samples=20, n_adapt=0, progress=false),
    mcmc_steps=20,
    maxiters=40,
)

res = fit_model(dm, method)
```

## Constructor Options

The full set of constructor arguments is shown below. All arguments have defaults and are keyword-only.

```julia
using Optimization
using OptimizationOptimJL
using LineSearches
using Turing

method = NoLimits.SAEM(;
    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs=NamedTuple(),
    adtype=Optimization.AutoForwardDiff(),
    sampler=Turing.NUTS(0.75),
    turing_kwargs=NamedTuple(),
    update_schedule=:all,
    warm_start=true,
    verbose=false,
    progress=true,
    mcmc_steps=80,
    max_store=50,
    t0=20,
    kappa=0.65,
    maxiters=300,
    rtol_theta=5e-5,
    atol_theta=5e-7,
    rtol_Q=5e-5,
    atol_Q=5e-7,
    consecutive_params=4,
    suffstats=nothing,
    q_from_stats=nothing,
    mstep_closed_form=nothing,
    builtin_stats=:auto,
    builtin_mean=:none,
    resid_var_param=:σ,
    re_cov_params=NamedTuple(),
    re_mean_params=NamedTuple(),
    ebe_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    ebe_optim_kwargs=NamedTuple(),
    ebe_adtype=Optimization.AutoForwardDiff(),
    ebe_grad_tol=:auto,
    ebe_multistart_n=50,
    ebe_multistart_k=10,
    ebe_multistart_max_rounds=5,
    ebe_multistart_sampling=:lhs,
    ebe_rescue_on_high_grad=true,
    ebe_rescue_multistart_n=128,
    ebe_rescue_multistart_k=32,
    ebe_rescue_max_rounds=0,
    ebe_rescue_grad_tol=:auto,
    ebe_rescue_multistart_sampling=:lhs,
    lb=nothing,
    ub=nothing,
)
```

## Option Groups

The constructor arguments are organized into the following functional groups.

| Group | Keywords | What they control |
| --- | --- | --- |
| M-step optimizer | `optimizer`, `optim_kwargs`, `adtype` | Fixed-effect update in each SAEM iteration via [Optimization.jl](https://docs.sciml.ai/Optimization/stable/). |
| E-step sampler | `sampler`, `turing_kwargs`, `mcmc_steps`, `update_schedule`, `warm_start` | Random-effect sampling and batch update selection. |
| SA schedule and stopping | `t0`, `kappa`, `maxiters`, `rtol_theta`, `atol_theta`, `rtol_Q`, `atol_Q`, `consecutive_params` | Stochastic approximation gain schedule and convergence criteria. |
| Custom statistics hooks | `suffstats`, `q_from_stats`, `mstep_closed_form`, `max_store` | User-defined sufficient statistics and optional closed-form M-step. |
| Built-in statistics hooks | `builtin_stats`, `builtin_mean`, `resid_var_param`, `re_cov_params`, `re_mean_params` | Automatic closed-form parameter updates for supported distribution structures. |
| Final EB modes | `ebe_*`, `ebe_rescue_*` | Post-fit empirical Bayes mode optimization used by random-effects accessors. |
| Bounds | `lb`, `ub` | Optional transformed-scale bounds on free fixed effects. |

## Constructor Input Reference

### E-step Sampling Inputs

These arguments configure the MCMC sampling of random effects at each SAEM iteration.

- `sampler`
  - Turing sampler used for the random-effect E-step.
  - Common choices include `NUTS(...)` and `MH()`.
- `turing_kwargs`
  - Additional keyword arguments passed to Turing sampling calls.
- `mcmc_steps`
  - Number of MCMC samples drawn per iteration.
  - If `mcmc_steps <= 0`, SAEM falls back to `turing_kwargs[:n_samples]` (or `1`).
- `update_schedule`
  - Controls which batches of individuals are updated at each iteration, enabling minibatch variants of SAEM.
  - Supported values:
    - `:all` updates all batches.
    - integer `m` updates a random minibatch of size `min(m, nbatches)`.
    - function `(nbatches, iter, rng) -> Vector{Int}` returns the batch indices to update.
- `warm_start`
  - When `true`, reuses latent-state sampler state between iterations where available.

The E-step sampling interface is built on [Turing.jl](https://turinglang.org/).

### M-step Optimization Inputs

When the M-step is performed numerically (i.e., no closed-form update is provided), these arguments control the fixed-effect optimization.

- `optimizer`
  - Optimizer for the M-step fixed-effect update.
  - Default: `OptimizationOptimJL.LBFGS(...)`.
- `optim_kwargs`
  - Keyword arguments forwarded to `Optimization.solve`.
- `adtype`
  - Automatic differentiation backend used to construct the `OptimizationFunction`.

SAEM uses the SciML [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) interface for numerical M-step updates.

### SA Schedule and Convergence Inputs

The stochastic approximation gain sequence controls the rate at which running statistics are updated toward new samples. A two-phase schedule is used: an initial averaging phase followed by a decay phase.

- `t0`, `kappa`
  - Define the SA gain schedule `gamma_t`:
    - `gamma_t = 1` for `t <= t0` (averaging phase)
    - `gamma_t = (t - t0)^(-kappa)` for `t > t0` (decay phase)
- `maxiters`
  - Maximum number of SAEM iterations.
- `rtol_theta`, `atol_theta`
  - Relative and absolute stabilization tolerances for fixed effects.
- `rtol_Q`, `atol_Q`
  - Relative and absolute stabilization tolerances for the SAEM Q criterion.
- `consecutive_params`
  - Number of consecutive iterations that must simultaneously satisfy both parameter and Q criteria before convergence is declared.

### Custom Statistics Inputs

SAEM supports a fully user-defined sufficient-statistics pathway, allowing closed-form M-step updates for models where the sufficient statistics are known analytically.

- `suffstats`
  - Callback for user-defined sufficient statistics:
    - `suffstats(dm, batch_infos, b_current, theta_u, fixed_maps) -> s_new`
- `q_from_stats`
  - Callback for Q evaluation from smoothed statistics:
    - `q_from_stats(s, theta_u, dm) -> Real`
- `mstep_closed_form`
  - Callback for user-defined closed-form M-step:
    - `mstep_closed_form(s, dm) -> ComponentArray`
  - The closed-form M-step is activated only when both `suffstats` and `mstep_closed_form` are provided.
- `max_store`
  - Number of latent snapshot iterations retained for numerical Q evaluation.
  - Used in the numerical Q path (i.e., when `suffstats` is not active).

### Built-in Update Inputs

For common distribution structures, SAEM can automatically derive closed-form updates for selected parameter blocks without requiring user-supplied callbacks.

- `builtin_stats`
  - `:auto`, `:closed_form`, or `:none`.
  - `:auto` attempts to infer compatible closed-form mappings from the model structure.
  - `:gaussian_re` is accepted as a backward-compatible alias for `:closed_form`.
- `builtin_mean`
  - `:glm` or `:none`.
- `resid_var_param`, `re_cov_params`, `re_mean_params`
  - Specify the target parameters for built-in updates when enabled.

When `suffstats` is provided, `builtin_mean=:glm` is skipped by design to avoid conflicting updates.

### Final EB Mode Inputs

After convergence, SAEM computes empirical Bayes modal estimates of the random effects for use by downstream accessors and diagnostics.

- `ebe_optimizer`, `ebe_optim_kwargs`, `ebe_adtype`, `ebe_grad_tol`
  - Configuration for the final EB mode optimization.
- `ebe_multistart_n`, `ebe_multistart_k`, `ebe_multistart_max_rounds`, `ebe_multistart_sampling`
  - Multistart configuration for EB mode optimization.
- `ebe_rescue_on_high_grad` and remaining `ebe_rescue_*`
  - Rescue strategy activated if the final EB gradient norm remains above threshold.

### Bound Inputs

- `lb`, `ub`
  - Optional transformed-scale bounds for free fixed effects.
  - When a closed-form M-step is used, SAEM projects closed-form updates into these bounds on the transformed scale.

## Which Models Have Closed-Form M-step Updates?

SAEM provides two closed-form pathways that can substantially accelerate convergence by avoiding numerical optimization for selected parameter blocks.

1. **Full user-defined closed-form M-step:**
   Activated only when both `suffstats` and `mstep_closed_form` are provided.
2. **Built-in blockwise closed-form updates** (`builtin_stats=:closed_form` or `:auto`):
   Selected distribution-parameter blocks are updated in closed form, while remaining free parameters are updated through numerical optimization.

Built-in blockwise closed-form updates are available for:

- Random-effect distribution parameters in `Normal`, `MvNormal`, `LogNormal`, and `Exponential` blocks (through `re_mean_params` and `re_cov_params`).
- Observation distribution parameters in `Normal`, `LogNormal`, `Exponential`, `Bernoulli`, and `Poisson` blocks (through `resid_var_param`, including named outcome-specific mappings).

These updates are compatible with arbitrarily nonlinear model structure, including ODE-based dynamics and function-approximator components, provided that the updated parameters appear in the supported distribution blocks.

### Example 1: Neural-Network-Based Nonlinear ODE Model with Closed-Form RE-Mean and Outcome-Scale Blocks

The following example illustrates a mixed-effects ODE model in which neural network parameter vectors serve as random-effect distribution means. Despite the highly nonlinear dynamics, the random-effect mean parameters and observation scale parameter admit closed-form SAEM updates.

```julia
using NoLimits
using LinearAlgebra
using Lux

chain_A1 = Chain(Dense(1, 4, tanh), Dense(4, 1))
chain_A2 = Chain(Dense(1, 4, tanh), Dense(4, 1))
chain_C1 = Chain(Dense(1, 4, tanh), Dense(4, 1))
chain_C2 = Chain(Dense(1, 4, tanh), Dense(4, 1))

model = @Model begin
    @helpers begin
        softplus(u) = u > 20 ? u : log1p(exp(u))
    end

    @fixedEffects begin
        sigma = RealNumber(1.0, scale=:log)
        zA1 = NNParameters(chain_A1; function_name=:NNA1, calculate_se=false)
        zA2 = NNParameters(chain_A2; function_name=:NNA2, calculate_se=false)
        zC1 = NNParameters(chain_C1; function_name=:NNC1, calculate_se=false)
        zC2 = NNParameters(chain_C2; function_name=:NNC2, calculate_se=false)
    end

    @covariates begin
        t = Covariate()
        d = ConstantCovariate(; constant_on=:ID)
    end

    @randomEffects begin
        etaA1 = RandomEffect(MvNormal(zA1, Diagonal(ones(length(zA1)))); column=:ID)
        etaA2 = RandomEffect(MvNormal(zA2, Diagonal(ones(length(zA2)))); column=:ID)
        etaC1 = RandomEffect(MvNormal(zC1, Diagonal(ones(length(zC1)))); column=:ID)
        etaC2 = RandomEffect(MvNormal(zC2, Diagonal(ones(length(zC2)))); column=:ID)
    end

    @DifferentialEquation begin
        fA1(t) = softplus(NNA1([t / 24], etaA1)[1])
        fA2(t) = softplus(NNA2([softplus(depot)], etaA2)[1])
        fC1(t) = -softplus(NNC1([softplus(center)], etaC1)[1])
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

saem_method = NoLimits.SAEM(;
    builtin_stats=:closed_form,
    re_mean_params=(; etaA1=:zA1, etaA2=:zA2, etaC1=:zC1, etaC2=:zC2),
    re_cov_params=NamedTuple(),
    resid_var_param=:sigma,
)
```

The closed-form blocks arise from the following model structure:

- Each random-effect block is `MvNormal(mean_parameter, fixed_covariance)` (e.g., `etaA1 ~ MvNormal(zA1, I)`). With `re_mean_params`, SAEM updates the mean vectors (`zA1`, `zA2`, `zC1`, `zC2`) using smoothed conditional means of the sampled random effects -- a closed-form Gaussian mean update.
- The observation model is `y ~ Normal(center(t), sigma)`. With `resid_var_param=:sigma`, SAEM updates `sigma` from smoothed residual second moments -- a closed-form Normal scale update.
- Setting `re_cov_params=NamedTuple()` leaves the random-effect covariance fixed, so only mean and outcome-scale closed-form blocks are applied.

The ODE dynamics and neural network transformations introduce substantial nonlinearity, but this does not affect the availability of closed-form updates for the distribution-parameter blocks.

### Example 2: Soft-Decision-Tree-Based Nonlinear ODE Model with Closed-Form RE-Mean and Outcome-Scale Blocks

This example follows the same structural pattern as Example 1, replacing neural network components with soft decision trees.

```julia
using NoLimits
using LinearAlgebra

model = @Model begin
    @helpers begin
        softplus(u) = u > 20 ? u : log1p(exp(u))
    end

    @fixedEffects begin
        sigma = RealNumber(1.0, scale=:log)
        gA1 = SoftTreeParameters(1, 2; function_name=:STA1, calculate_se=false)
        gA2 = SoftTreeParameters(1, 2; function_name=:STA2, calculate_se=false)
        gC1 = SoftTreeParameters(1, 2; function_name=:STC1, calculate_se=false)
        gC2 = SoftTreeParameters(1, 2; function_name=:STC2, calculate_se=false)
    end

    @covariates begin
        t = Covariate()
        d = ConstantCovariate(; constant_on=:ID)
    end

    @randomEffects begin
        etaA1 = RandomEffect(MvNormal(gA1, Diagonal(ones(length(gA1)))); column=:ID)
        etaA2 = RandomEffect(MvNormal(gA2, Diagonal(ones(length(gA2)))); column=:ID)
        etaC1 = RandomEffect(MvNormal(gC1, Diagonal(ones(length(gC1)))); column=:ID)
        etaC2 = RandomEffect(MvNormal(gC2, Diagonal(ones(length(gC2)))); column=:ID)
    end

    @DifferentialEquation begin
        fA1(t) = softplus(STA1([t / 24], etaA1)[1])
        fA2(t) = softplus(STA2([softplus(depot)], etaA2)[1])
        fC1(t) = -softplus(STC1([softplus(center)], etaC1)[1])
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

saem_method = NoLimits.SAEM(;
    builtin_stats=:closed_form,
    re_mean_params=(; etaA1=:gA1, etaA2=:gA2, etaC1=:gC1, etaC2=:gC2),
    re_cov_params=NamedTuple(),
    resid_var_param=:sigma,
)
```

The reasoning is analogous to the neural network case:

- Each random-effect block is `MvNormal(mean_parameter, fixed_covariance)` with soft-tree parameter vectors as means. The `re_mean_params` mapping enables closed-form Gaussian mean updates for `gA1`, `gA2`, `gC1`, and `gC2`.
- The observation model is `Normal(..., sigma)`, so `resid_var_param=:sigma` yields a closed-form scale update.
- Random-effect covariance is fixed by construction (`re_cov_params=NamedTuple()`), so no covariance update is performed.

### Example 3: Mechanistic ODE with Auto-Detected Closed-Form Blocks

When the model uses standard distribution parameterizations, SAEM can automatically detect compatible closed-form update targets via `builtin_stats=:auto`. The following example illustrates this with a mechanistic two-compartment ODE model.

```julia
using NoLimits
using LinearAlgebra

model_saem = @Model begin
    @fixedEffects begin
        tka = RealNumber(0.45)
        tcl = RealNumber(1.0)
        tv = RealNumber(3.45)
        omega1 = RealNumber(1.0, scale=:log)
        omega2 = RealNumber(1.0, scale=:log)
        omega3 = RealNumber(1.0, scale=:log)
        sigma_eps = RealNumber(1.0, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(MvNormal([tka, tcl, tv], Diagonal([omega1, omega2, omega3])); column=:id)
    end

    @preDifferentialEquation begin
        ka = exp(eta[1])
        cl = exp(eta[2])
        v = exp(eta[3])
    end

    @DifferentialEquation begin
        D(depot) ~ -ka * depot
        D(center) ~ ka * depot - cl / v * center
    end

    @initialDE begin
        depot = 1.0
        center = 0.0
    end

    @formulas begin
        y1 ~ Normal(center(t) / v, sigma_eps)
    end
end

saem_method = NoLimits.SAEM(; builtin_stats=:auto)
```

With `builtin_stats=:auto`, SAEM inspects the model structure and identifies the following closed-form update targets:

- The random-effect distribution is `MvNormal([tka, tcl, tv], Diagonal([omega1, omega2, omega3]))`. The mean parameters (`tka`, `tcl`, `tv`) admit closed-form Gaussian mean updates, and the diagonal covariance parameters (`omega1`, `omega2`, `omega3`) admit closed-form variance updates.
- The observation model is `Normal(center(t) / v, sigma_eps)`, so `sigma_eps` admits a closed-form Normal scale update.

For `MvNormal` diagonal targets, the built-in update operates on the diagonal covariance entries (variances) for the mapped parameters.

## Custom Sufficient Statistics and Closed-Form M-step

For models where the sufficient statistics are known analytically, SAEM supports a fully custom statistics pathway. The per-iteration procedure is as follows:

1. SAEM samples random effects for the updated batches.
2. The user-defined callback computes new statistics: `s_new = suffstats(dm, batch_infos, b_current, theta_u, fixed_maps)`.
3. SA smoothing is applied: `s <- s + gamma_t * (s_new - s)`.
4. The M-step uses either the custom closed-form update (if both `suffstats` and `mstep_closed_form` are set) or falls back to numerical optimization via Optimization.jl.
5. Q evaluation for convergence monitoring uses `q_from_stats(s, theta_u, dm)` when both `suffstats` and `q_from_stats` are set; otherwise, a numerical Q is computed from stored latent snapshots.

### Callback Contracts

- `suffstats(dm, batch_infos, b_current, theta_u, fixed_maps) -> s_new`
  - The return value `s_new` can be a scalar, array, or `NamedTuple`.
  - Keys and shapes must remain stable across iterations.
  - `fixed_maps` is the normalized random-effect constant map derived from `constants_re`.
- `q_from_stats(s, theta_u, dm) -> Real`
  - A Q-like criterion computed from the smoothed statistics `s`.
- `mstep_closed_form(s, dm) -> ComponentArray`
  - Must return the full untransformed fixed-effect parameter container.
  - The closed-form M-step is activated only when `suffstats` and `mstep_closed_form` are both provided.

When using custom sufficient statistics, it is recommended to also provide `q_from_stats` so that convergence monitoring remains consistent with the statistic design.

```julia
using NoLimits
using DataFrames
using Distributions
using Turing
using ComponentArrays

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        b = RealNumber(0.1)
        sigma = RealNumber(0.3, scale=:log)
        tau = RealNumber(0.4, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, tau); column=:ID)
    end

    @formulas begin
        mu = exp(a + b * t + eta)   # nonlinear in random effects
        y ~ Exponential(mu * sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B],
    t = [0.0, 1.0, 0.0, 1.0],
    y = [1.0, 1.08, 0.96, 1.14],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

function suffstats(dm, batch_infos, b_current, theta_u, fixed_maps)
    s_sum = 0.0
    s_sq = 0.0
    n = 0
    for b in b_current
        s_sum += sum(b)
        s_sq += sum(abs2, b)
        n += length(b)
    end
    return (; s_sum, s_sq, n=max(n, 1))
end

q_from_stats = (s, theta_u, dm) -> -0.5 * (s.s_sq - (s.s_sum^2) / s.n)

theta_template = ComponentArray(a=0.2, b=0.1, sigma=0.3, tau=0.4)
function mstep_closed_form(s, dm)
    theta_u = deepcopy(theta_template)
    theta_u.a = 0.2 + 0.01 * s.s_sum
    theta_u.b = 0.1 + 0.001 * s.s_sq
    sigma_hat = sqrt(max(s.s_sq / s.n, 1e-8))
    theta_u.sigma = sigma_hat
    theta_u.tau = max(0.2, 0.5 * sigma_hat)
    return theta_u
end

method = NoLimits.SAEM(;
    sampler=MH(),
    turing_kwargs=(n_samples=12, n_adapt=0, progress=false),
    maxiters=20,
    suffstats=suffstats,
    q_from_stats=q_from_stats,
    mstep_closed_form=mstep_closed_form,
)

res = fit_model(dm, method)
```

The `mstep_closed_form` expressions above are illustrative only; they should be replaced with model-specific closed-form derivations in practice.

## Optimization.jl Interface Example

When the M-step is performed numerically, any optimizer supported by [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) can be used.

```julia
using OptimizationOptimJL
using OptimizationOptimisers
using LineSearches

method_lbfgs = NoLimits.SAEM(;
    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs=(maxiters=120,),
)

method_adam = NoLimits.SAEM(;
    optimizer=OptimizationOptimisers.Adam(0.05),
    optim_kwargs=(maxiters=150,),
)
```

## Accessing Results

After fitting, results are accessed through the standard accessor interface. Like MCEM, SAEM returns point estimates rather than a posterior chain.

```julia
theta_u = NoLimits.get_params(res; scale=:untransformed)
obj = get_objective(res)
ok = get_converged(res)

re_df = get_random_effects(res)
```
