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

If fixed-effect priors are defined in the model, SAEM ignores them in its objective. To incorporate priors, use `MCMC` instead.

## Basic Usage

The following example demonstrates a minimal SAEM workflow with a nonlinear mixed-effects model.

```julia
using NoLimits
using DataFrames
using Distributions

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

method = NoLimits.SAEM(;
    # M-step optimizer
    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs=(; iterations=10, g_abstol=1e-4, f_reltol=1e-6),
    adtype=Optimization.AutoForwardDiff(),

    # E-step sampler
    sampler=SaemixMH(),
    turing_kwargs=NamedTuple(),
    update_schedule=:all,
    warm_start=true,
    mcmc_steps=nothing,   # resolves to 1 for SaemixMH (the default sampler), 80 otherwise

    # SA schedule
    sa_schedule=:robbins_monro,
    sa_burnin_iters=6,
    maxiters=300,
    t0=maxiters ÷ 2,
    kappa=0.65,
    sa_phase1_iters=200,
    sa_phase2_kappa=-1.0,
    sa_schedule_fn=nothing,

    # Multi-chain E-step
    n_chains=1,
    auto_small_n_chains=true,
    small_n_chain_target=50,

    # SA variance annealing
    sa_anneal_targets=NamedTuple(),
    sa_anneal_schedule=:exponential,
    sa_anneal_iters=nothing,   # resolves to t0
    sa_anneal_alpha=0.95,
    sa_anneal_fn=nothing,

    # Variance lower bound
    auto_var_lb=true,
    var_lb_value=1e-5,

    # Convergence and stopping
    rtol_theta=1e-3,
    atol_theta=1e-4,
    rtol_Q=1e-3,
    atol_Q=1e-4,
    consecutive_params=4,
    convergence_window=50,

    # Adaptive Q memory policy
    q_store_max=50,
    q_store_epsilon=1e-10,
    q_store_min=0,

    # Custom statistics hooks
    suffstats=nothing,
    q_from_stats=nothing,
    mstep_closed_form=nothing,

    # Built-in statistics hooks
    builtin_stats=:auto,
    builtin_mean=:none,
    resid_var_param=:σ,
    re_cov_params=NamedTuple(),
    re_mean_params=NamedTuple(),

    # M-step variant
    mstep_sa_on_params=true,

    # E-step retry (used when mstep_sa_on_params=true)
    max_estep_retries=3,
    retry_mcmc_steps=1,

    # Verbose / progress
    verbose=false,
    progress=true,

    # Final EB modes
    ebe_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    ebe_optim_kwargs=NamedTuple(),
    ebe_adtype=Optimization.AutoForwardDiff(),
    ebe_grad_tol=:auto,
    ebe_multistart_n=50,
    ebe_multistart_k=1,
    ebe_multistart_max_rounds=5,
    ebe_multistart_sampling=:lhs,
    ebe_rescue_on_high_grad=false,
    ebe_rescue_multistart_n=128,
    ebe_rescue_multistart_k=32,
    ebe_rescue_max_rounds=8,
    ebe_rescue_grad_tol=:auto,
    ebe_rescue_multistart_sampling=:lhs,

    # Bounds
    lb=nothing,
    ub=nothing,

    # RE annealing (collapse REs toward fixed effects)
    anneal_to_fixed=(),
    anneal_schedule=:exponential,
    anneal_min_sd=1e-5,
)
```

## Option Groups

The constructor arguments are organized into the following functional groups.

| Group | Keywords | What they control |
| --- | --- | --- |
| M-step optimizer | `optimizer`, `optim_kwargs`, `adtype` | Fixed-effect update in each SAEM iteration via [Optimization.jl](https://docs.sciml.ai/Optimization/stable/). |
| E-step sampler | `sampler`, `turing_kwargs`, `mcmc_steps`, `update_schedule`, `warm_start` | Random-effect sampling and batch update selection. |
| SA schedule | `sa_schedule`, `sa_burnin_iters`, `t0`, `kappa`, `sa_phase1_iters`, `sa_phase2_kappa`, `sa_schedule_fn` | Gain sequence shape and phases. |
| Multi-chain | `n_chains`, `auto_small_n_chains`, `small_n_chain_target` | Number of parallel MCMC chains per batch. |
| SA variance annealing | `sa_anneal_targets`, `sa_anneal_schedule`, `sa_anneal_iters`, `sa_anneal_alpha`, `sa_anneal_fn` | Post-M-step variance floor that decays over iterations to prevent early collapse. |
| Variance lower bound | `auto_var_lb`, `var_lb_value` | Hard permanent floor on variance / SD parameters. |
| Convergence and stopping | `maxiters`, `rtol_theta`, `atol_theta`, `rtol_Q`, `atol_Q`, `consecutive_params`, `convergence_window` | Windowed drift test for early stopping (see [Convergence and Early Stopping](#convergence-and-early-stopping)). |
| Adaptive Q memory | `q_store_max`, `q_store_epsilon`, `q_store_min` | Ring buffer capacity and adaptive pruning policy for the numerical Q path. |
| Custom statistics hooks | `suffstats`, `q_from_stats`, `mstep_closed_form` | User-defined sufficient statistics and optional closed-form M-step. |
| Built-in statistics hooks | `builtin_stats`, `builtin_mean`, `resid_var_param`, `re_cov_params`, `re_mean_params` | Automatic closed-form parameter updates for supported distribution structures. |
| M-step variant | `mstep_sa_on_params` | Use current-iteration samples (not ring buffer) with Robbins-Monro parameter update (default). |
| E-step retry | `max_estep_retries`, `retry_mcmc_steps` | Re-sample batches that yield a non-finite objective when `mstep_sa_on_params=true`. |
| Final EB modes | `ebe_*`, `ebe_rescue_*` | Post-fit empirical Bayes mode optimization used by random-effects accessors. |
| Bounds | `lb`, `ub` | Optional transformed-scale bounds on free fixed effects. |
| RE annealing | `anneal_to_fixed`, `anneal_schedule`, `anneal_min_sd` | Progressive shrinkage of selected RE prior SDs toward zero, collapsing those effects to fixed by the final iteration. |

## Constructor Input Reference

The constructor signature block above lists every keyword with its default. See the [`SAEM`](@ref) entry in the API reference for the full list. This section explains the behavior of the option groups whose effect is not obvious from their names.

### E-step Sampling Inputs

These arguments configure the MCMC sampling of random effects at each SAEM iteration.

- `sampler`
  - Sampler used for the random-effect E-step.
  - Default: `SaemixMH()`. See [Samplers](#samplers) for all available options.
- `turing_kwargs`
  - Additional keyword arguments passed to Turing sampling calls (ignored for `SaemixMH` and `AdaptiveNoLimitsMH`).
- `mcmc_steps`
  - Number of MCMC samples drawn per iteration.
  - Default: `nothing`, which resolves to `1` when the sampler is `SaemixMH` (the default) and `80` for any other sampler.
  - If `mcmc_steps <= 0`, SAEM falls back to `turing_kwargs[:n_samples]` (or `1`).
- `update_schedule`
  - Controls which batches of individuals are updated at each iteration, enabling minibatch variants of SAEM.
  - Supported values:
    - `:all` updates all batches.
    - integer `m` updates a random minibatch of size `min(m, nbatches)`.
    - function `(nbatches, iter, rng) -> Vector{Int}` returns the batch indices to update.
- `warm_start`
  - When `true`, reuses latent-state sampler state between iterations where available.

### M-step Optimization Inputs

When the M-step is performed numerically (i.e., no closed-form update is provided), these arguments control the fixed-effect optimization.

- `optim_kwargs`
  - Keyword arguments forwarded to `Optimization.solve`.
  - Default: `(; iterations=10, g_abstol=1e-4, f_reltol=1e-6)`. The inner LBFGS is capped at 10 iterations per SAEM step; the outer SA loop provides the global convergence trajectory.
- `mstep_sa_on_params`
  - When `true` (default), the M-step optimizes only the current iteration's samples and applies a Robbins-Monro parameter update: `θ_new = θ_old + γ*(θ̂ - θ_old)`. The ring buffer is capped to capacity 1, eliminating storage and reweighting overhead from previous snapshots. This is significantly more memory-efficient than the ring-buffer path and is the recommended mode.
  - When `false`, the M-step minimizes the full ring-buffer Q-function and sets `θ_new = θ̂` directly. Useful as a diagnostic or when you want the classical ring-buffer SAEM behavior.

SAEM uses the SciML [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) interface for numerical M-step updates.

### E-step Retry Inputs

When `mstep_sa_on_params=true`, batches that produce a non-finite objective after an E-step update are automatically re-sampled before the M-step. This makes the mode safe to use with any sampler, including Turing-based samplers (`MH()`, `NUTS`) that can occasionally draw from low-density regions early in training.

- `max_estep_retries`
  - Maximum number of re-sampling attempts for non-finite batches per iteration.
  - Must be `≥ 0`. Setting to `0` disables the retry mechanism entirely.
  - Default: `3`.
- `retry_mcmc_steps`
  - Number of MCMC steps used in each retry attempt (per bad batch).
  - Must be `≥ 1`.
  - Default: `1`.

Only batches with a non-finite log-joint are retried; finite batches are not touched. If a batch remains non-finite after all retries, it is skipped for this iteration and the previous sample is retained.

### SA Schedule Inputs

The SA gain sequence `γ_t ∈ [0, 1]` controls how aggressively new samples update the running statistics at each iteration. SAEM supports three schedule modes selected by `sa_schedule`.

- `sa_schedule`
  - `:robbins_monro` (default): classic Robbins-Monro two-phase schedule built from `t0` and `kappa`.
  - `:two_phase`: explicit two-phase schedule built from `sa_phase1_iters` and `sa_phase2_kappa`.
  - `:custom`: user-supplied function `sa_schedule_fn(iter, opts) -> Float64`.

#### `:robbins_monro` schedule

| Phase | Condition | γ |
| --- | --- | --- |
| Burn-in | `iter ≤ sa_burnin_iters` | 0 (no SA update) |
| Stabilization | `sa_burnin_iters < iter ≤ sa_burnin_iters + t0` | 1 |
| Decay | otherwise | `((phase3_total - k3) / phase3_total)^kappa` |

where `phase3_total = maxiters - sa_burnin_iters - t0` and `k3 = iter - sa_burnin_iters - t0`.

- `sa_burnin_iters::Int = 6`: iterations before SA updates begin. During burn-in no SA smoothing is performed and no samples are stored.
- `t0::Int = maxiters ÷ 2`: length of the stabilization phase (γ = 1). Defaults to `nothing`, which resolves to `maxiters ÷ 2` (i.e. `150` at the default `maxiters=300`).
- `kappa::Float64 = 0.65`: decay exponent controlling how quickly γ falls off after stabilization.

#### `:two_phase` schedule

| Phase | Condition | γ |
| --- | --- | --- |
| Burn-in | `iter ≤ sa_burnin_iters` | 0 |
| Phase 1 | `sa_burnin_iters < iter ≤ sa_burnin_iters + sa_phase1_iters` | 1 |
| Phase 2 | otherwise | `k2^sa_phase2_kappa` |

where `k2 = iter - sa_burnin_iters - sa_phase1_iters`.

- `sa_phase1_iters::Int = 200`: length of the full-weight phase.
- `sa_phase2_kappa::Float64 = -1.0`: exponent for phase-2 decay. Negative values produce increasing γ (rarely useful); set to a small negative number close to 0 for a slow decay.

#### `:custom` schedule

- `sa_schedule_fn`: a callable with signature `(iter::Int, opts::SAEMOptions) -> Float64` returning `γ ∈ [0, 1]`.

### Multi-Chain E-step

Running multiple independent MCMC chains per batch and averaging their samples before the SA update reduces variance in the E-step at the cost of proportionally more likelihood evaluations.

- `n_chains::Int = 1`: number of MCMC chains run per batch per iteration.
- `auto_small_n_chains::Bool = true`: automatically increases `n_chains` for small datasets so that the total number of E-step samples (`n_batches × n_chains`) reaches `small_n_chain_target`. Useful when the dataset has few individuals and few batches.
- `small_n_chain_target::Int = 50`: target total sample count used by `auto_small_n_chains`.

### SA Variance Annealing

After each M-step, scalar variance and SD parameters for RE distributions can be clamped to a decaying lower floor. This prevents variance parameters from collapsing to near-zero too early in the run (when the E-step is still mixing poorly), while allowing them to reach their optimal value once the chain has warmed up.

The floor starts at `alpha × initial_value` and decays to zero over `sa_anneal_iters` iterations, which by default matches the SA stabilization length `t0`.

- `sa_anneal_targets::NamedTuple = NamedTuple()`: explicit mapping of fixed-effect name to `alpha` value, e.g., `(; τ = 0.9)`. When empty, targets are auto-detected from `re_cov_params` for the Gaussian RE families (Normal, LogNormal, and their multivariate diagonal-covariance forms MvNormal, MvLogNormal, MvLogitNormal). For multivariate covariances the floor is applied per diagonal entry.
- `sa_anneal_schedule::Symbol = :exponential`: shape of the floor decay.
  - `:exponential`: `floor = alpha × init × exp(-3 × frac)`.
  - `:linear`: `floor = alpha × init × (1 - frac)`.
- `sa_anneal_iters = nothing`: number of iterations over which the floor is active. Defaults to `nothing`, which resolves to `t0` (the SA stabilization length). Pass an explicit `0` to fall back to `0.3 × maxiters`.
- `sa_anneal_alpha::Float64 = 0.95`: fraction of the initial parameter value used as the starting floor (auto-detection mode only; explicit `sa_anneal_targets` carry their own alpha per entry).
- `sa_anneal_fn`: reserved for future use (not active).

SA variance annealing is distinct from `anneal_to_fixed`. The latter collapses an RE entirely into a fixed effect by shrinking its prior SD to zero; SA variance annealing only prevents its estimated variance from hitting zero prematurely during optimization.

### Variance Lower Bound

A hard, permanent lower bound is applied to scalar RE covariance and residual SD parameters after every M-step update. Unlike SA variance annealing, this floor does not decay - it is enforced for the entire run.

- `auto_var_lb::Bool = true`: when `true`, automatically applies the lower bound to all scalar RE cov params (Normal, LogNormal, MvNormal scalar covariance) and the residual variance parameter.
- `var_lb_value::Float64 = 1e-5`: minimum value enforced for the targeted parameters on the natural (untransformed) scale.

### Convergence and Early Stopping

SAEM stops before `maxiters` when both the fixed effects and the Q-function have been stationary for several consecutive iterations. Because single-iteration changes are dominated by Monte Carlo noise, stationarity is measured on a sliding window of recent iterates: the window is split in half and the drift between the two half-window means must stay within tolerance.

At each iteration past the SA stabilization phase (and outside any active SA variance annealing floor), the current transformed fixed-effect vector and the Q value are appended to a window of length `convergence_window`. Once the window is full, two tests run every iteration:

- θ test: every coordinate's drift between the half-window means (transformed scale) must satisfy `drift ≤ max(atol_theta, rtol_theta * scale_θ, 2 * mc_se)`, where `scale_θ = max(1, ‖older-half mean‖∞)` and `mc_se` is the Monte-Carlo standard error of the half-mean difference estimated from the within-half variance of the window.
- Q test: the same rule applied to the Q history. Any non-finite Q value in the window fails the test.

The `2 * mc_se` term is what makes the test fire in practice: both the θ trajectory and the per-iteration Q value fluctuate at Monte-Carlo noise scale that never fully decays, so a drift that is statistically indistinguishable from that noise counts as stationary, while a genuine trend (drift well above the noise level) blocks the stop. The absolute and relative tolerances act as deterministic floors on top of the statistical criterion.

Each passing test increments its streak counter; a failing test resets it to zero. The fit stops with `converged = true` once both streaks reach `consecutive_params`. Iterations whose M-step was skipped (non-finite optimizer result) fail both tests, so a frozen parameter vector cannot register as converged.

- `convergence_window::Int = 50`: window length (must be ≥ 4; the compared halves have length `convergence_window ÷ 2`). The earliest possible stop is `convergence_window + consecutive_params` iterations after stabilization ends.
- `rtol_theta = 1e-3`, `atol_theta = 1e-4`: tolerances for the θ drift test.
- `rtol_Q = 1e-3`, `atol_Q = 1e-4`: tolerances for the Q drift test.
- `consecutive_params::Int = 4`: consecutive passing iterations required of both tests.

Set `rtol_theta = atol_theta = 0` (or the Q pair) to disable early stopping and always run the full `maxiters` iterations. The per-iteration drift values are recorded in the fit diagnostics as `drift_θ` and `drift_Q` (`NaN` until the window first fills) and are shown in the `verbose` per-iteration log.

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
### Adaptive Q Memory Policy Inputs

These arguments control the ring buffer used for numerical Q evaluation (the path taken when `suffstats` is not provided).

- `q_store_max` (default `50`)
  - Ring buffer capacity: the maximum number of snapshots retained at any time.
- `q_store_epsilon` (default `1e-10`)
  - Weight pruning threshold. After each push, snapshots whose SA weight falls below
    this value are removed from the oldest end of the buffer (subject to `q_store_min`).
    During the γ=1 stabilization phase all previous snapshots are immediately pruned,
    keeping only the current iteration's sample in the buffer.
  - The retained weights are renormalized to sum to 1 before evaluating Q, so the
    objective is scale-invariant to pruning.
  - Has no effect when `suffstats` is provided; a warning is emitted if set to a
    non-default value alongside `suffstats`.
- `q_store_min` (default `0`)
  - Guaranteed minimum number of retained snapshots. When epsilon pruning would reduce
    the active count below this floor, the most-recent snapshots are kept
    unconditionally regardless of their weight.

### Built-in Update Inputs

For common distribution structures, SAEM can automatically derive closed-form updates for selected parameter blocks without requiring user-supplied callbacks.

- `builtin_stats`
  - `:auto`, `:closed_form`, or `:none`.
  - `:auto` attempts to infer compatible closed-form mappings from the model structure.
  - `:auto` skips any parameter that is also referenced outside the block whose sufficient statistics would update it, for example a random-effect scale that a deterministic formula also feeds into the outcome distribution, because the closed-form update would discard that information; such parameters fall back to the numeric M-step and are named in the startup info message.
  - `:gaussian_re` is accepted as a backward-compatible alias for `:closed_form`.
- `builtin_mean`
  - `:glm` or `:none`.
- `resid_var_param`, `re_cov_params`, `re_mean_params`
  - Specify the target parameters for built-in updates when enabled.

When `suffstats` is provided, `builtin_mean=:glm` is skipped by design to avoid conflicting updates.

### Final EB Mode Inputs

After convergence, SAEM computes empirical Bayes modal estimates of the random effects for use by downstream accessors and diagnostics, configured through the `ebe_*` keywords. When `ebe_rescue_on_high_grad=true` (default `false`), a rescue strategy governed by the `ebe_rescue_*` keywords is activated if the final EB gradient norm remains above threshold. See the [`SAEM`](@ref) API entry for the full keyword list and defaults.

### Bound Inputs

`lb`, `ub` are optional transformed-scale bounds for free fixed effects. When a closed-form M-step is used, SAEM projects the closed-form updates into these bounds on the transformed scale.

### RE Annealing Inputs

- `anneal_to_fixed`
  - A `Tuple` of RE name `Symbol`s to progressively collapse toward fixed effects.
  - Each named RE must satisfy two eligibility conditions:
    1. Its distribution must be `Normal(μ, σ)`.
    2. The SD `σ` must be a plain numeric literal in the `@randomEffects` block (e.g. `Normal(a, 2.0)`). Using a fixed-effect parameter or covariate as SD (e.g. `Normal(0.0, τ)`) raises an informative error at startup.
  - Default: `()` (no annealing).
- `anneal_schedule`
  - Controls the shape of the SD decay curve. Supported values:
    - `:exponential` (default) - exponential decay from the initial SD to `anneal_min_sd`.
    - `:linear` - linear interpolation from the initial SD to `anneal_min_sd`.
    - `:gamma` - decay tied to the SA gain sequence, using the same `t0` and `kappa` as the main schedule.
- `anneal_min_sd`
  - Target SD reached at the final iteration.
  - Default: `1e-5`.

## Advanced usage

Samplers, random-effect annealing, and closed-form M-step updates (including custom sufficient
statistics) have their own page: [SAEM: Samplers, Annealing, and Closed-Form Updates](saem-advanced.md).

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
used_closed_form = NoLimits.get_closed_form_mstep_used(res)
notes = NoLimits.get_notes(res)  # includes closed_form_mstep_mode/sources and builtin_stats_closed_form_eligibility

re_df = get_random_effects(res)
```

## Where to go next

- [SAEM: Samplers, Annealing, and Closed-Form Updates](saem-advanced.md) - the advanced options.
- [MCEM](mcem.md) - the other stochastic-EM method, and when to prefer it.
- [Uncertainty Quantification](../uncertainty-quantification/index.md) - standard errors for a SAEM fit.
- [Troubleshooting](../troubleshooting.md) - when a fit stalls or does not converge.
