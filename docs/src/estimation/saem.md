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
    optim_kwargs=(; iterations=50),
    adtype=Optimization.AutoForwardDiff(),

    # E-step sampler
    sampler=MH(),
    turing_kwargs=NamedTuple(),
    update_schedule=:all,
    warm_start=true,
    mcmc_steps=80,

    # SA schedule
    sa_schedule=:robbins_monro,
    sa_burnin_iters=0,
    t0=150,
    kappa=0.65,
    sa_phase1_iters=200,
    sa_phase2_kappa=-1.0,
    sa_schedule_fn=nothing,

    # Multi-chain E-step
    n_chains=1,
    auto_small_n_chains=false,
    small_n_chain_target=50,

    # SA variance annealing
    sa_anneal_targets=NamedTuple(),
    sa_anneal_schedule=:exponential,
    sa_anneal_iters=0,
    sa_anneal_alpha=0.9,
    sa_anneal_fn=nothing,

    # Variance lower bound
    auto_var_lb=true,
    var_lb_value=1e-5,

    # Convergence and stopping
    maxiters=300,
    rtol_theta=5e-5,
    atol_theta=5e-7,
    rtol_Q=5e-5,
    atol_Q=5e-7,
    consecutive_params=4,

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
    ebe_multistart_k=10,
    ebe_multistart_max_rounds=5,
    ebe_multistart_sampling=:lhs,
    ebe_rescue_on_high_grad=true,
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
| Convergence and stopping | `maxiters`, `rtol_theta`, `atol_theta`, `rtol_Q`, `atol_Q`, `consecutive_params` | Stopping criteria. |
| Adaptive Q memory | `q_store_max`, `q_store_epsilon`, `q_store_min` | Ring buffer capacity and adaptive pruning policy for the numerical Q path. |
| Custom statistics hooks | `suffstats`, `q_from_stats`, `mstep_closed_form` | User-defined sufficient statistics and optional closed-form M-step. |
| Built-in statistics hooks | `builtin_stats`, `builtin_mean`, `resid_var_param`, `re_cov_params`, `re_mean_params` | Automatic closed-form parameter updates for supported distribution structures. |
| M-step variant | `mstep_sa_on_params` | Use current-iteration samples (not ring buffer) with Robbins-Monro parameter update (default). |
| E-step retry | `max_estep_retries`, `retry_mcmc_steps` | Re-sample batches that yield a non-finite objective when `mstep_sa_on_params=true`. |
| Final EB modes | `ebe_*`, `ebe_rescue_*` | Post-fit empirical Bayes mode optimization used by random-effects accessors. |
| Bounds | `lb`, `ub` | Optional transformed-scale bounds on free fixed effects. |
| RE annealing | `anneal_to_fixed`, `anneal_schedule`, `anneal_min_sd` | Progressive shrinkage of selected RE prior SDs toward zero, collapsing those effects to fixed by the final iteration. |

## Constructor Input Reference

### E-step Sampling Inputs

These arguments configure the MCMC sampling of random effects at each SAEM iteration.

- `sampler`
  - Sampler used for the random-effect E-step.
  - Default: `MH()`. See [Samplers](#samplers) for all available options.
- `turing_kwargs`
  - Additional keyword arguments passed to Turing sampling calls (ignored for `SaemixMH` and `AdaptiveNoLimitsMH`).
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

### M-step Optimization Inputs

When the M-step is performed numerically (i.e., no closed-form update is provided), these arguments control the fixed-effect optimization.

- `optimizer`
  - Optimizer for the M-step fixed-effect update.
  - Default: `OptimizationOptimJL.LBFGS(...)`.
- `optim_kwargs`
  - Keyword arguments forwarded to `Optimization.solve`.
  - Default: `(; iterations=50)`. The inner LBFGS is capped at 50 iterations per SAEM step; the outer SA loop provides the global convergence trajectory.
- `adtype`
  - Automatic differentiation backend used to construct the `OptimizationFunction`.
- `mstep_sa_on_params`
  - When `true` (default), the M-step optimizes only the current iteration's samples and applies a Robbins-Monro parameter update: `θ_new = θ_old + γ*(θ̂ - θ_old)`. The ring buffer is capped to capacity 1, eliminating storage and reweighting overhead from previous snapshots. This is significantly more memory-efficient than the ring-buffer path and is the recommended mode.
  - When `false`, the M-step minimizes the full ring-buffer Q-function and sets `θ_new = θ̂` directly. Useful as a diagnostic or when you want the classical ring-buffer SAEM behaviour.

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

- `sa_burnin_iters::Int = 0`: iterations before SA updates begin. During burn-in no SA smoothing is performed and no samples are stored.
- `t0::Int = 150`: length of the stabilization phase (γ = 1).
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
- `auto_small_n_chains::Bool = false`: when `true`, automatically increases `n_chains` for small datasets so that the total number of E-step samples (`n_batches × n_chains`) reaches `small_n_chain_target`. Useful when the dataset has few individuals and few batches.
- `small_n_chain_target::Int = 50`: target total sample count used by `auto_small_n_chains`.

### SA Variance Annealing

After each M-step, scalar variance and SD parameters for RE distributions can be clamped to a decaying lower floor. This prevents variance parameters from collapsing to near-zero too early in the run (when the E-step is still mixing poorly), while allowing them to reach their optimal value once the chain has warmed up.

The floor starts at `alpha × initial_value` and decays to zero over `sa_anneal_iters` iterations.

- `sa_anneal_targets::NamedTuple = NamedTuple()`: explicit mapping of fixed-effect name to `alpha` value, e.g., `(; τ = 0.9)`. When empty, targets are auto-detected from `re_cov_params` for Normal and LogNormal RE families.
- `sa_anneal_schedule::Symbol = :exponential`: shape of the floor decay.
  - `:exponential`: `floor = alpha × init × exp(-3 × frac)`.
  - `:linear`: `floor = alpha × init × (1 - frac)`.
- `sa_anneal_iters::Int = 0`: number of iterations over which the floor is active. If zero, defaults to `0.3 × maxiters`.
- `sa_anneal_alpha::Float64 = 0.9`: fraction of the initial parameter value used as the starting floor (auto-detection mode only; explicit `sa_anneal_targets` carry their own alpha per entry).
- `sa_anneal_fn`: reserved for future use (not active).

SA variance annealing is distinct from `anneal_to_fixed`. The latter collapses an RE entirely into a fixed effect by shrinking its prior SD to zero; SA variance annealing only prevents its estimated variance from hitting zero prematurely during optimization.

### Variance Lower Bound

A hard, permanent lower bound is applied to scalar RE covariance and residual SD parameters after every M-step update. Unlike SA variance annealing, this floor does not decay — it is enforced for the entire run.

- `auto_var_lb::Bool = true`: when `true`, automatically applies the lower bound to all scalar RE cov params (Normal, LogNormal, MvNormal scalar covariance) and the residual variance parameter.
- `var_lb_value::Float64 = 1e-5`: minimum value enforced for the targeted parameters on the natural (untransformed) scale.

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
    During the γ=1 stabilisation phase all previous snapshots are immediately pruned,
    keeping only the current iteration's sample in the buffer.
  - The retained weights are renormalised to sum to 1 before evaluating Q, so the
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

### RE Annealing Inputs

- `anneal_to_fixed`
  - A `Tuple` of RE name `Symbol`s to progressively collapse toward fixed effects.
  - Each named RE must satisfy two eligibility conditions:
    1. Its distribution must be `Normal(μ, σ)`.
    2. The SD `σ` must be a plain numeric literal in the `@randomEffects` block (e.g. `Normal(a, 2.0)`). Using a fixed-effect parameter or covariate as SD (e.g. `Normal(0.0, τ)`) raises an informative error at startup.
  - Default: `()` (no annealing).
- `anneal_schedule`
  - Controls the shape of the SD decay curve. Supported values:
    - `:exponential` (default) — exponential decay from the initial SD to `anneal_min_sd`.
    - `:linear` — linear interpolation from the initial SD to `anneal_min_sd`.
    - `:gamma` — decay tied to the SA gain sequence, using the same `t0` and `kappa` as the main schedule.
- `anneal_min_sd`
  - Target SD reached at the final iteration.
  - Default: `1e-5`.

## Samplers

SAEM accepts three types of E-step sampler.

### `MH()` (default)

Turing's built-in random-walk Metropolis-Hastings. Uses a fixed standard-Normal proposal in the linked (unconstrained) space. Fast per-step but requires careful tuning of `mcmc_steps` to achieve adequate mixing.

```julia
using Turing
res = fit_model(dm, SAEM(sampler=MH()))
```

### `SaemixMH`

A lightweight Turing-free MH sampler that directly operates on the flat random-effects vector. Implements the first three kernels of the saemix R package:

- **Kernel 1** (`n_kern1` steps): independent proposal from the current RE prior `p(η|θ)`. Acceptance uses only the likelihood ratio. Efficient when the posterior is close to the prior.
- **Kernel 2** (`n_kern2` steps): per-level coordinate-wise random walk in the natural parameter space. Uses the full log-joint ratio and saemix's `domega2[:, 1]` adaptation rule.
- **Kernel 3** (`n_kern3` steps): block random walk using the same iteration-dependent block-size schedule as saemix. Uses the full log-joint ratio and saemix's `domega2[:, nrs2]` adaptation rule.

Because `SaemixMH` bypasses Turing entirely it avoids interpreter and compilation overhead, making it significantly faster per iteration for large models.

```julia
res = fit_model(dm, SAEM(
    sampler    = SaemixMH(),
    mcmc_steps = 1,
    maxiters   = 300,
))
```

Constructor keywords:
- `n_kern1::Int = 2`: prior-proposal steps per E-step call.
- `n_kern2::Int = 2`: coordinate random-walk steps per E-step call.
- `n_kern3::Int = 2`: block random-walk steps per E-step call, matching saemix default.
- `proba_mcmc::Float64 = 0.4`: saemix acceptance-rate target.
- `stepsize_rw::Float64 = 0.4`: saemix multiplicative adaptation step size.
- `rw_init::Float64 = 0.5`: saemix initial random-walk scale multiplier.

Backward-compatible aliases:
- `target_accept` maps to `proba_mcmc`.
- `adapt_rate` maps to `stepsize_rw`.

`SaemixMH` pairs naturally with the default `mstep_sa_on_params=true` because kernel-1's prior-proposal draws always produce a finite log-joint, so E-step retries are rarely triggered. For Turing-based samplers (`MH()`, `NUTS`) the E-step retry mechanism (`max_estep_retries=3`) handles the occasional non-finite objective that can arise early in training.

### `AdaptiveNoLimitsMH`

An adaptive MH sampler implementing the Haario et al. (2001) algorithm. Maintains a per-RE-name running covariance in the natural proposal space and pools samples across all active levels of the same RE for faster covariance adaptation.

The proposal space is adjusted per distribution family:

| Distribution | Proposal space | Bijection |
| --- | --- | --- |
| `Normal` | η ∈ ℝ | identity |
| `MvNormal` | η ∈ ℝ^d | identity |
| `LogNormal` | z = log(η) | log / exp |
| `Exponential` | z = log(η) | log / exp |
| `Beta` | z = logit(η) | logit / sigmoid |
| `NormalizingPlanarFlow` | η ∈ ℝ^d | identity |

Adaptation state persists across SAEM iterations via the warm-start mechanism.

```julia
res = fit_model(dm, SAEM(sampler=AdaptiveNoLimitsMH()))
```

Constructor keywords:
- `adapt_start::Int = 50`: pooled sample count before Haario updates activate.
- `init_scale::Float64 = 1.0`: multiplier on the prior-based initial proposal covariance.
- `eps_reg::Float64 = 1e-6`: regularisation added to the diagonal to ensure positive-definiteness.

`AdaptiveNoLimitsMH` is most useful when the RE posterior covariance differs substantially from the prior, when REs are correlated (`MvNormal` with `d ≥ 2`), or when the prior is weakly informative.

### Turing-Based Samplers (`NUTS`, etc.)

Any Turing-compatible sampler can be used:

```julia
using Turing
res = fit_model(dm, SAEM(
    sampler      = NUTS(0.75),
    turing_kwargs = (n_samples=10, n_adapt=5, progress=false),
    mcmc_steps   = 10,
))
```

Note: Turing-based samplers re-compile the model at each SAEM iteration and are significantly slower per step than `SaemixMH` or `AdaptiveNoLimitsMH` for most models.

## RE Annealing

The `anneal_to_fixed` option progressively shrinks the prior standard deviation of selected Normal random effects from their initial value toward `anneal_min_sd` over the course of SAEM iterations. By the final iteration the prior SD is negligibly small, which effectively collapses the annealed RE into a fixed shift — the sampler can no longer move it away from its mean, so it behaves as a fixed effect without requiring a model change.

Both the E-step sampler and the M-step Q function see the shrunken SD at each iteration, so the annealing is consistent across the entire algorithm.

### When to Use

Annealing is useful when:

- A random effect is suspected to be negligible and you want to assess the impact of removing it without refitting from scratch.
- You want to run an early exploration phase with tight RE priors, then let the priors relax (by using a second fit without annealing).
- A model has identifiability issues in early iterations and annealing an RE stabilizes the trajectory before the final convergence phase.

### Eligibility

A random effect is eligible for annealing if and only if:

1. Its declared distribution is `Normal(μ, σ)`.
2. The SD argument `σ` is a plain numeric literal — not a fixed-effect parameter, covariate, or helper expression.

Valid examples:
```julia
eta = RandomEffect(Normal(0.0, 2.0); column=:ID)   # literal SD 2.0 ✓
eta = RandomEffect(Normal(a, 0.5);   column=:ID)   # literal SD 0.5, mean is fixed effect ✓
```

Invalid examples (raise a clear error at startup):
```julia
eta = RandomEffect(Normal(0.0, tau); column=:ID)   # SD is fixed-effect param tau ✗
eta = RandomEffect(Normal(mu, tau);  column=:ID)   # both mu and tau are params ✗
eta = RandomEffect(MvNormal(...);    column=:ID)   # not Normal ✗
```

### Schedule Options

The three built-in schedules all start from the initial literal SD (`sd0`) and finish at `anneal_min_sd` by the last iteration.

| Schedule | Shape | Notes |
| --- | --- | --- |
| `:exponential` | exponential decay | default; reaches `anneal_min_sd` smoothly and quickly |
| `:linear` | straight-line decay | simple; slower initial shrinkage than exponential |
| `:gamma` | SA-gain-coupled decay | ties annealing speed to the main SA schedule (`t0`, `kappa`) |

### Interaction with Built-in Statistics

When `builtin_stats=:closed_form` (or `:auto`) and an annealed RE also appears in `re_cov_params`, annealing always takes precedence: the built-in closed-form covariance update for that RE is suppressed for the entire run. A one-time info message is printed at startup to make this visible.

### Example

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @fixedEffects begin
        a    = RealNumber(0.5)
        b    = RealNumber(0.2)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        # SD is a plain literal — eligible for annealing
        eta_id   = RandomEffect(Normal(0.0, 1.2); column=:ID)
        # This RE will be annealed: its SD decays from 0.8 to 1e-5
        eta_site = RandomEffect(Normal(0.0, 0.8); column=:SITE)
    end

    @formulas begin
        mu = a + b * t + eta_id + eta_site
        y ~ Normal(mu, sigma)
    end
end

# Collapse eta_site toward a fixed effect over the run
method = NoLimits.SAEM(;
    sampler=MH(),
    turing_kwargs=(n_samples=20, n_adapt=0, progress=false),
    maxiters=100,
    anneal_to_fixed=(:eta_site,),
    anneal_schedule=:exponential,   # default
    anneal_min_sd=1e-5,             # default
)

res = fit_model(dm, method)
```

To compare schedules, pass the same `anneal_to_fixed` with a different `anneal_schedule`:

```julia
method_linear = NoLimits.SAEM(;
    sampler=MH(),
    turing_kwargs=(n_samples=20, n_adapt=0, progress=false),
    maxiters=100,
    anneal_to_fixed=(:eta_site,),
    anneal_schedule=:linear,
)

method_gamma = NoLimits.SAEM(;
    sampler=MH(),
    turing_kwargs=(n_samples=20, n_adapt=0, progress=false),
    maxiters=100,
    anneal_to_fixed=(:eta_site,),
    anneal_schedule=:gamma,
    t0=150,
    kappa=0.65,
)
```

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

For HMM outcomes (`DiscreteTimeDiscreteStatesHMM`, `ContinuousTimeDiscreteStatesHMM`, and multivariate variants), built-in closed-form updates are currently limited to eligible random-effect distribution blocks. Transition/emission parameter blocks are marked ineligible in built-in mode because latent-state sufficient statistics are not constructed by this pathway.

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
used_closed_form = NoLimits.get_closed_form_mstep_used(res)
notes = NoLimits.get_notes(res)  # includes closed_form_mstep_mode/sources and builtin_stats_closed_form_eligibility

re_df = get_random_effects(res)
```
