# MCMC-Based Uncertainty

Posterior-draw uncertainty quantification in NoLimits.jl supports both exact posterior chains (`MCMC`) and approximate posterior draws (`VI`). This family of methods is useful when Gaussian approximations (Wald-style) are too restrictive, for example in correlated, skewed, or otherwise non-quadratic posterior geometries.

NoLimits.jl provides two posterior-draw UQ workflows:

- **Chain UQ** (`method=:chain`): extracts uncertainty directly from an existing `MCMC` or `VI` fit.
- **MCMC refit UQ** (`method=:mcmc_refit`): launches a new MCMC sampling run from a non-`MCMC` source fit.

Both workflows build on the package's MCMC interface, which integrates with [Turing.jl](https://turinglang.org/).

## 1) Chain UQ (`method=:chain`)

Chain UQ operates on posterior draws already available from a fitted `MCMC` or `VI` result:

- For `MCMC`, it uses retained post-warmup chain samples.
- For `VI`, it samples draws from the fitted variational posterior.

```julia
using NoLimits
using Random

if @isdefined(res_mcmc) && res_mcmc !== nothing
    uq_chain = compute_uq(
        res_mcmc;
        method=:chain,
        level=0.95,
        mcmc_warmup=200,
        mcmc_draws=1000,
        rng=Random.Xoshiro(1),
    )
end

if @isdefined(res_vi) && res_vi !== nothing
    uq_chain_vi = compute_uq(
        res_vi;
        method=:chain,
        level=0.95,
        mcmc_draws=1000,
        rng=Random.Xoshiro(2),
    )
end
```

### Requirements

- The source fit must originate from `NoLimits.MCMC` or `NoLimits.VI`.
- The fit must include a stored `DataModel` (`store_data_model=true`).

### Key Controls

- `mcmc_warmup`: number of initial iterations discarded per chain as burn-in for MCMC. If omitted, NoLimits uses `n_adapt` from the fit diagnostics when available. Ignored for VI.
- `mcmc_draws`: number of posterior draws used for interval construction (subsampled from MCMC chains, or sampled from the VI posterior).
- `constants`: optional fixed-effect constants that exclude specified coordinates from UQ.
- `rng`: controls random subsampling when `mcmc_draws` is smaller than the total number of available draws.

### Interval Mode

For the chain backend, the `interval` argument accepts `:auto`, `:equaltail`, or `:chain`. Equal-tail intervals report symmetric quantiles of the posterior distribution at the requested confidence level.

### Inclusion Rules

Only free fixed-effect coordinates with `calculate_se=true` are included in the UQ output.

A coordinate is excluded when:

- it is fixed by `constants`, or
- its parent parameter block has `calculate_se=false`.

## 2) MCMC Refit UQ (`method=:mcmc_refit`)

MCMC refit UQ bridges optimization-based and Bayesian workflows. It is designed for situations where the original fit was obtained by an optimization-based method (e.g., MLE, Laplace), but fully Bayesian uncertainty estimates are desired. It launches a new MCMC sampling run initialized at the fitted parameter values and returns chain-based uncertainty from the resulting posterior.

```julia
using NoLimits
using Random

uq_refit = compute_uq(
    res_non_mcmc;
    method=:mcmc_refit,
    level=0.95,
    mcmc_turing_kwargs=(n_samples=400, n_adapt=100, progress=false),
    mcmc_draws=300,
    rng=Random.Xoshiro(2),
)
```

### Requirements

- The source fit must not be from `MCMC` (use `method=:chain` for existing MCMC results).
- All sampled fixed effects must have priors specified in the model definition.
- At least one parameter must remain free (i.e., not held constant) in the refit.

If no sampled fixed effects remain and no random effects are present, the refit will raise an error.

### How Parameters Are Selected for Refit

The refit determines which parameters to sample as follows:

- User-specified `constants` are respected and held fixed.
- Fixed effects with `calculate_se=false` are automatically held constant at their fitted values.
- All remaining free fixed effects become sampled parameters in the MCMC run.
- `constants_re` can be passed to hold specific random-effect levels fixed during the refit.

### Refit Method Configuration

The MCMC refit can be configured in two ways. Either pass a fully configured method object:

- `mcmc_method::NoLimits.MCMC` -- an explicit MCMC method instance.

Or configure the sampler through individual keyword arguments:

- `mcmc_sampler` -- the sampling algorithm (defaults to Turing's NUTS sampler).
- `mcmc_turing_kwargs` -- keyword arguments forwarded to Turing's sampling call.
- `mcmc_adtype` -- automatic differentiation backend for the sampler.
- `mcmc_fit_kwargs` -- additional keyword arguments for the underlying `fit_model` call.

When no explicit configuration is provided, defaults from `NoLimits.MCMC` are used with a NUTS sampler.

## Returned Quantities

Both backends return interval estimates, covariance estimates computed from retained draws, and the draws themselves.

```julia
if @isdefined(uq_chain)
    backend = get_uq_backend(uq_chain)     # :chain or :mcmc_refit
    names = get_uq_parameter_names(uq_chain)

    est_nat = get_uq_estimates(uq_chain; scale=:natural)
    ints_nat = get_uq_intervals(uq_chain; scale=:natural)
    V_nat = get_uq_vcov(uq_chain; scale=:natural)
    draws_nat = get_uq_draws(uq_chain; scale=:natural)

    diag = get_uq_diagnostics(uq_chain)
end
```

## Diagnostics

The diagnostics returned by `get_uq_diagnostics` provide information about the draw source and the draws used for UQ.

Common fields include:

- `warmup`: number of discarded warmup iterations (`0` for VI chain UQ).
- `requested_draws`, `available_draws`, `used_draws`: draw accounting information.
- `n_iter`, `n_chains`, `n_samples`: draw-source metadata.
- `source`: draw source tag (for example `:mcmc_chain` or `:vi_posterior`).
- `n_active_parameters`: number of active fixed-effect coordinates used in UQ.

For `method=:mcmc_refit`, the diagnostics additionally include:

- `refit_source_method`: the estimation method of the original fit.
- `refit_sampler`: the sampler used in the refit.
- `refit_turing_kwargs`: the Turing keyword arguments used.
- `sampled_fixed_names`: names of fixed effects that were sampled.
- `constants_used`: the constants applied during the refit.
