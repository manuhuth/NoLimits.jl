# MCEM

The Monte Carlo Expectation-Maximization (MCEM) algorithm is a likelihood-based approach for fitting nonlinear mixed-effects models when the marginal likelihood cannot be computed in closed form. It alternates between two steps:

- an **E-step** that draws samples from (or forms a weighted approximation of) the conditional distribution of random effects given the current fixed-effect estimates, and
- an **optimization-based M-step** that updates the fixed effects by maximizing the Monte Carlo approximation of the expected complete-data log-likelihood.

This formulation accommodates arbitrary nonlinear observation models, including those defined through ordinary differential equations, without requiring analytically tractable integrals over the random effects.

## E-Step Strategies

MCEM supports two E-step implementations, controlled by the `e_step` argument.

### MCMC E-step (`MCEM_MCMC`)

The default strategy. Draws samples from the exact conditional distribution `p(b | y, θ)` using [Turing.jl](https://turinglang.org/).

```julia
NoLimits.MCEM_MCMC(;
    sampler        = Turing.NUTS(0.75),
    turing_kwargs  = NamedTuple(),
    sample_schedule = 250,
    warm_start     = true,
)
```

- `sampler` - Turing sampler. Common choices: `MH()`, `NUTS(...)`.
- `turing_kwargs` - forwarded to Turing; the keys `n_samples` and `n_adapt` are interpreted explicitly.
- `sample_schedule` - number of MCMC samples per iteration; accepts an integer, a vector (iteration-indexed), or a function `iter -> n_samples`.
- `warm_start` - when `true`, reuses previous latent-state values as chain initialization.

### Importance Sampling E-step (`MCEM_IS`)

An alternative E-step based on self-normalized importance sampling (IS). Samples are drawn from a tractable proposal `q(b)` and reweighted by `log p(y, b | θ) − log q(b)`. The weighted Q-function replaces the uniform average used by the MCMC E-step.

IS is often substantially faster per iteration than MCMC because it requires only independent forward passes through the log-likelihood rather than a sequential Markov chain.

```julia
NoLimits.MCEM_IS(;
    n_samples             = 500,
    proposal              = :gaussian,
    adapt                 = true,
    warm_start_mcmc_iters = 0,
    mcmc_warmup           = nothing,
)
```

- `n_samples` - number of IS draws per E-step iteration.
- `proposal` - proposal distribution (see [Proposal Modes](@ref) below).
- `adapt` - when `true`, updates the Gaussian proposal blocks after each IS iteration using IS-weighted statistics of the current samples.
- `warm_start_mcmc_iters` - number of initial iterations to run with the MCMC E-step before switching to IS. Use this to pre-warm the Gaussian proposal.
- `mcmc_warmup` - an `MCEM_MCMC` configuration used during the warm-up phase (required when `warm_start_mcmc_iters > 0`).

#### Proposal Modes

| `proposal` | Behavior |
|---|---|
| `:prior` | Each RE level is sampled independently from its prior `p(η \| θ)`. No adaptation. |
| `:gaussian` | Adaptive block-diagonal Gaussian in the bijected (unconstrained) space. Falls back to the prior in the first iteration. The blocks are updated after each iteration using IS-weighted posterior statistics. |
| `Function` | User-supplied function with signature `(θ, batch_info, re_dists, rng, n_samples) → (samples, log_qs)`. |

The `:gaussian` proposal approximates the per-individual posterior `p(η | yᵢ, θ)` in the bijected space. Because all samples are independent, the effective sample size (ESS) is typically close to `n_samples`, and the proposal improves with each iteration.

#### MCMC Warm-up then IS

When the Gaussian proposal needs a cold start, a short MCMC warm-up can be used to initialize the blocks before switching to IS:

```julia
es = NoLimits.MCEM_IS(
    n_samples             = 200,
    proposal              = :gaussian,
    adapt                 = true,
    warm_start_mcmc_iters = 5,
    mcmc_warmup           = NoLimits.MCEM_MCMC(
        sampler       = MH(),
        turing_kwargs = (n_samples=50, n_adapt=0, progress=false),
    ),
)
```

Iterations 1-5 use the MCMC E-step; the Gaussian proposal is initialized from those MCMC samples (with uniform weights). From iteration 6 onward the IS E-step is used.

#### User-Supplied Proposal

```julia
function my_proposal(θ, batch_info, re_dists, rng, n_samples)
    nb = batch_info.n_b
    samples = randn(rng, nb, n_samples) .* 2.0
    log_qs  = vec(sum(logpdf.(Normal(0.0, 2.0), samples); dims=1))
    return samples, log_qs
end

es = NoLimits.MCEM_IS(n_samples=100, proposal=my_proposal)
```

The function must return:
- `samples` - matrix of shape `(n_b, n_samples)` where `n_b` is the total dimension of all RE levels in the batch.
- `log_qs` - vector of length `n_samples` containing `log q(bₘ)` for each draw.

## Applicability

MCEM is designed for models that include both fixed and random effects:

- The model must declare at least one random effect and at least one free fixed effect.
- Multiple random-effect grouping columns and multivariate random effects are fully supported.

If fixed-effect priors are defined in the model, MCEM ignores them in its objective. To incorporate priors, use `MCMC` instead.

## Basic Usage

```julia
using NoLimits
using DataFrames
using Distributions
using Turing

model = @Model begin
    @fixedEffects begin
        a     = RealNumber(0.2)
        b     = RealNumber(0.1)
        sigma = RealNumber(0.3, scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end

    @formulas begin
        mu = a + b * t + exp(eta)
        y  ~ Normal(mu, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C],
    t  = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y  = [1.0, 1.3, 0.9, 1.2, 1.1, 1.5],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
```

**MCMC E-step (classic):**

```julia
res = fit_model(dm, NoLimits.MCEM(
    e_step  = NoLimits.MCEM_MCMC(
        sampler       = MH(),
        turing_kwargs = (n_samples=50, n_adapt=0, progress=false),
    ),
    maxiters = 30,
))
```

**IS E-step with prior proposal:**

```julia
res = fit_model(dm, NoLimits.MCEM(
    e_step   = NoLimits.MCEM_IS(n_samples=200, proposal=:prior),
    maxiters = 30,
))
```

**IS E-step with adaptive Gaussian proposal:**

```julia
res = fit_model(dm, NoLimits.MCEM(
    e_step   = NoLimits.MCEM_IS(n_samples=200, proposal=:gaussian, adapt=true),
    maxiters = 30,
))
```

## Constructor Options

The full set of constructor arguments is shown below. All arguments have defaults and are keyword-only.

```julia
using Optimization
using OptimizationOptimJL
using LineSearches
using Turing

method = NoLimits.MCEM(;
    # E-step (new unified interface)
    e_step=NoLimits.MCEM_MCMC(),          # or MCEM_IS(...)

    # Backward-compatible MCMC shorthand (passed to MCEM_MCMC internally)
    # sampler=Turing.NUTS(0.75),
    # turing_kwargs=NamedTuple(),
    # sample_schedule=250,
    # warm_start=true,

    # M-step
    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs=(; iterations=50, g_abstol=1e-4, f_reltol=1e-6),
    adtype=Optimization.AutoForwardDiff(),

    # EM convergence
    verbose=false,
    progress=true,
    maxiters=100,
    rtol_theta=1e-4,
    atol_theta=1e-6,
    rtol_Q=1e-4,
    atol_Q=1e-6,
    consecutive_params=3,

    # Final EB estimation
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
)
```

## Option Groups

| Group | Keywords | What they control |
| --- | --- | --- |
| E-step | `e_step` | Sampling or IS strategy for the random effects. |
| M-step optimizer | `optimizer`, `optim_kwargs`, `adtype` | Optimization of fixed effects using [Optimization.jl](https://docs.sciml.ai/Optimization/stable/). |
| EM stopping | `maxiters`, `rtol_theta`, `atol_theta`, `rtol_Q`, `atol_Q`, `consecutive_params` | Iteration limit and convergence checks. |
| Logging | `verbose`, `progress` | Diagnostic output and progress bar. |
| Final EB estimation | `ebe_*` and `ebe_rescue_*` options | Post-fit empirical Bayes mode computation used by random-effects accessors and diagnostics. |
| Bounds | `lb`, `ub` | Optional transformed-scale bounds for free fixed effects in M-step optimization. |

## Behavioral Notes

The constructor signature block above lists every keyword with its default. See the [`MCEM`](@ref) entry in the API reference for the full list. The notes below explain the behavior that is not obvious from the names.

- **EM convergence.** MCEM monitors both fixed-effect parameter stability (`rtol_theta`, `atol_theta`) and Q-function stability (`rtol_Q`, `atol_Q`). `consecutive_params` is the number of consecutive iterations that must simultaneously satisfy both criteria before convergence is declared. `verbose` enables iteration-level logging of `Q`, `dtheta`, `dQ`, and sample count.
- **Final EB modes.** After the EM iterations complete, MCEM computes empirical Bayes (EB) modal estimates of the random effects used by downstream accessors, configured through the `ebe_*` keywords (`ebe_grad_tol=:auto` selects a data-adaptive tolerance). When `ebe_rescue_on_high_grad=true` (default `false`), a rescue multistart governed by the `ebe_rescue_*` keywords is triggered if the final EB gradient norm remains above threshold.
- **Bounds.** `lb`, `ub` are optional transformed-scale bounds for free fixed effects in the M-step optimization. Parameters held constant via the `constants` fit keyword are excluded automatically.

## Diagnostics

The `notes` field of the fit result contains iteration-level diagnostics:

```julia
diag = get_notes(res).diagnostics
diag.Q_hist    # Vector{Float64} - Q-function value per iteration
diag.ess_hist  # Vector{Float64} - effective sample size per iteration
               #   IS iterations: ESS = 1/Σwₘ² ∈ [1, n_samples]
               #   MCMC iterations: NaN
```

The ESS is a per-iteration summary of proposal quality. Values close to `n_samples` indicate that the proposal closely approximates the posterior; low values indicate weight degeneracy (proposal too wide or misaligned).

```julia
using CairoMakie
plot(diag.Q_hist,   label="Q", xlabel="iteration")
plot(diag.ess_hist, label="ESS", xlabel="iteration")
```

## Backward Compatibility

The legacy MCMC-only keyword interface is fully preserved. Existing code that does not use `e_step` continues to work:

```julia
# Old API - still works
method = NoLimits.MCEM(
    sampler       = MH(),
    turing_kwargs = (n_samples=50, n_adapt=0, progress=false),
    maxiters      = 20,
)
```

When `sampler` or `turing_kwargs` are passed without an explicit `e_step`, they are forwarded to an internally constructed `MCEM_MCMC`.

## Fit Keywords

```julia
res = fit_model(
    dm,
    method;
    constants=(a=0.2,),
    constants_re=(; eta=(; A=0.0,)),
    penalty=NamedTuple(),
    ode_args=(),
    ode_kwargs=NamedTuple(),
    serialization=EnsembleSerial(),
    rng=Random.default_rng(),
    theta_0_untransformed=nothing,
    store_eb_modes=true,
)
```

The `constants_re` argument allows specific random-effect levels to be fixed at known values while the remaining levels are estimated.

## Optimization.jl Interface (M-step and EB)

MCEM uses [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) in two places:

1. The M-step optimization over fixed effects at each iteration.
2. The EB mode optimization (final or recomputed EB modes) through the `ebe_*` configuration.

Tested M-step optimizers include:

- `OptimizationOptimJL.LBFGS(...)` (default)
- `OptimizationOptimisers.Adam(...)`
- `OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited()`

When using derivative-free optimizers such as those from `OptimizationBBO`, finite bounds must be supplied:

```julia
using OptimizationBBO

lb, ub = default_bounds_from_start(dm; margin=1.0)

method_bbo = NoLimits.MCEM(;
    optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
    optim_kwargs=(iterations=20,),
    e_step=NoLimits.MCEM_IS(n_samples=100, proposal=:gaussian),
    maxiters=5,
    lb=lb,
    ub=ub,
)
```

## Accessing Results

After fitting, results are accessed through the standard accessor interface.

```julia
theta_u = NoLimits.get_params(res; scale=:untransformed)
obj     = get_objective(res)
ok      = get_converged(res)

re_df   = get_random_effects(res)
```
