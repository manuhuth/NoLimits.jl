# SAEM: Samplers, Annealing, and Closed-Form Updates

The advanced half of the [SAEM](saem.md) page: how the E-step samples, how random-effect
annealing works, which models get closed-form M-step updates, and how to supply your own
sufficient statistics. Read [SAEM](saem.md) first for the objective, the basic usage, and the
constructor reference.

## Samplers

SAEM accepts three types of E-step sampler.

### `MH()`

Turing's built-in random-walk Metropolis-Hastings. Uses a fixed standard-Normal proposal in the linked (unconstrained) space. Fast per-step but requires careful tuning of `mcmc_steps` to achieve adequate mixing.

```julia
using Turing
res = fit_model(dm, SAEM(sampler=MH()))
```

### `SaemixMH` (default)

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
- `eps_reg::Float64 = 1e-6`: regularization added to the diagonal to ensure positive-definiteness.

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

The `anneal_to_fixed` option progressively shrinks the prior standard deviation of selected Normal random effects from their initial value toward `anneal_min_sd` over the course of SAEM iterations. By the final iteration the prior SD is negligibly small, which effectively collapses the annealed RE into a fixed shift - the sampler can no longer move it away from its mean, so it behaves as a fixed effect without requiring a model change.

Both the E-step sampler and the M-step Q function see the shrunken SD at each iteration, so the annealing is consistent across the entire algorithm.

### When to Use

Annealing is useful when:

- A random effect is suspected to be negligible and you want to assess the impact of removing it without refitting from scratch.
- You want to run an early exploration phase with tight RE priors, then let the priors relax (by using a second fit without annealing).
- A model has identifiability issues in early iterations and annealing an RE stabilizes the trajectory before the final convergence phase.

### Eligibility

A random effect is eligible for annealing if and only if:

1. Its declared distribution is `Normal(μ, σ)`.
2. The SD argument `σ` is a plain numeric literal - not a fixed-effect parameter, covariate, or helper expression.

Valid examples:
```julia
eta = RandomEffect(Normal(0.0, 2.0); column=:ID)   # literal SD 2.0           (eligible)
eta = RandomEffect(Normal(a, 0.5);   column=:ID)   # literal SD 0.5, mean is FE (eligible)
```

Invalid examples (raise a clear error at startup):
```julia
eta = RandomEffect(Normal(0.0, tau); column=:ID)   # SD is fixed-effect param tau (not eligible)
eta = RandomEffect(Normal(mu, tau);  column=:ID)   # both mu and tau are params   (not eligible)
eta = RandomEffect(MvNormal(...);    column=:ID)   # not Normal                   (not eligible)
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
        # SD is a plain literal - eligible for annealing
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
    t0=50,
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

!!! tip "The same networks without Lux"
    Each `Chain(Dense(1, 4, tanh), Dense(4, 1))` below can be replaced by a dependency-free
    `FFNNParameters` block with identical call sites:

    ```julia
    zA1 = FFNNParameters((1, 4, 1); activation=:tanh, output_activation=:identity,
        function_name=:NNA1, calculate_se=false)
    ```

    See [function approximators](../model-building/universal-function-approximators.md).

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

- Each random-effect block is `MvNormal(mean_parameter, fixed_covariance)` (e.g., `etaA1 ~ MvNormal(zA1, I)`). With `re_mean_params`, SAEM updates the mean vectors (`zA1`, `zA2`, `zC1`, `zC2`) using smoothed conditional means of the sampled random effects - a closed-form Gaussian mean update.
- The observation model is `y ~ Normal(center(t), sigma)`. With `resid_var_param=:sigma`, SAEM updates `sigma` from smoothed residual second moments - a closed-form Normal scale update.
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

## Where to go next

- [SAEM](saem.md) - the objective, basic usage, and full constructor reference.
- [Neural Differential Equations (SAEM)](../tutorials/mixed-effects-nn-saem.md) - these options in a complete analysis.
- [Troubleshooting](../troubleshooting.md) - when a fit stalls or does not converge.
