# Uncertainty Quantification

Point estimates alone are insufficient for reliable scientific conclusions. Uncertainty quantification (UQ) provides confidence intervals and variance-covariance information that characterize the precision of estimated parameters, enabling principled model comparison and decision-making. Reporting parameter uncertainty is standard practice in the biological sciences and is typically required for publication.

NoLimits.jl offers a unified UQ interface through `compute_uq`, which accepts a fitted model result (`FitResult`) and returns a `UQResult`. The same interface supports multiple uncertainty backends -- Wald-based approximations, profile-likelihood intervals, posterior-chain intervals, and MCMC refit workflows -- so that the choice of UQ method can be varied independently of the estimation method.

## Quick Start

```julia
using NoLimits

uq = compute_uq(res)  # method=:auto
```

When `method=:auto` is used, the backend is selected automatically based on the source fit:

- If `res` comes from `MCMC`, the backend is `:chain`.
- If `interval=:profile` is explicitly requested, the backend is `:profile`.
- Otherwise, the backend defaults to `:wald`.

## Backends

The table below summarizes the available UQ backends, their associated estimation methods, and the type of output each produces.

| Backend | `method` | Typical source fit | Output style |
| --- | --- | --- | --- |
| Wald | `:wald` | `MLE`, `MAP`, `Laplace`, `LaplaceMAP`, `MCEM`, `SAEM` | covariance + Gaussian-draw intervals |
| Chain | `:chain` | `MCMC` | posterior-draw intervals |
| Profile likelihood | `:profile` | `MLE`, `MAP`, `Laplace`, `LaplaceMAP` | profile intervals |
| MCMC refit | `:mcmc_refit` | non-`MCMC` fits | posterior-draw intervals from refit |

### Wald (`method=:wald`)

The Wald backend computes an approximate variance-covariance matrix for selected fixed-effect coordinates, then draws from a Gaussian approximation to construct confidence intervals. This is the most computationally efficient approach and works well for well-identified models whose log-likelihood surface is approximately quadratic near the optimum.

```julia
uq_wald = compute_uq(
    res;
    method=:wald,
    vcov=:hessian,
    level=0.95,
    n_draws=2000,
)
```

Key options:

- `vcov`: `:hessian` or `:sandwich`
- `pseudo_inverse`: use pseudo-inverse if direct inversion is unstable
- `hessian_backend`: `:auto`, `:forwarddiff`, or `:fd_gradient`
- finite-difference controls: `fd_abs_step`, `fd_rel_step`, `fd_max_tries`

For `MCEM` and `SAEM` source fits, the Wald backend also accepts random-effects approximation controls via `re_approx` and `re_approx_method`.

### Chain (`method=:chain`)

The chain backend extracts posterior samples from an existing `MCMC` fit and computes equal-tail credible intervals directly from the retained draws. No additional computation is required beyond the original sampling run.

```julia
if @isdefined(res_mcmc) && res_mcmc !== nothing
    uq_chain = compute_uq(
        res_mcmc;
        method=:chain,
        level=0.95,
        mcmc_warmup=200,
        mcmc_draws=1000,
    )
end
```

Key options:

- `mcmc_warmup`: number of warmup/adaptation iterations to discard
- `mcmc_draws`: number of retained draws used for interval construction

### Profile (`method=:profile`)

The profile-likelihood backend computes likelihood-based confidence intervals by profiling one parameter at a time around the fitted optimum. Unlike the Wald approach, profile intervals do not assume a Gaussian posterior shape and can capture asymmetric uncertainty -- a common situation for variance parameters or parameters near constraints.

```julia
uq_profile = compute_uq(
    res;
    method=:profile,
    level=0.95,
    profile_method=:LIN_EXTRAPOL,
    profile_scan_width=3.0,
    profile_scan_tol=1e-3,
    profile_loss_tol=1e-3,
)
```

Key options:

- `profile_method`
- `profile_scan_width`, `profile_scan_tol`
- `profile_loss_tol`
- `profile_local_alg`, `profile_max_iter`, `profile_ftol_abs`
- `profile_kwargs`

### MCMC Refit (`method=:mcmc_refit`)

The MCMC refit backend is designed for cases where the original fit was obtained by an optimization-based method (such as MLE or Laplace), but fully Bayesian uncertainty quantification is desired. It launches a new MCMC sampling run initialized from the fitted parameter values and reports chain-based intervals from the resulting posterior.

```julia
uq_refit = compute_uq(
    res;
    method=:mcmc_refit,
    level=0.95,
    mcmc_draws=1000,
)
```

Important behavior:

- `method=:mcmc_refit` is intended for non-`MCMC` source fits.
- All sampled fixed effects must have priors specified.
- `constants` and `constants_re` can be used to hold selected parameters fixed during the refit.

## Returned Object and Accessors

`compute_uq` returns a `UQResult` object. The following accessor functions provide a consistent interface across all backends.

```julia
backend = get_uq_backend(uq)
source_method = get_uq_source_method(uq)
param_names = get_uq_parameter_names(uq)

est_nat = get_uq_estimates(uq; scale=:natural)
est_tr = get_uq_estimates(uq; scale=:transformed)

ints_nat = get_uq_intervals(uq; scale=:natural)
ints_tr = get_uq_intervals(uq; scale=:transformed)

V_nat = get_uq_vcov(uq; scale=:natural)
V_tr = get_uq_vcov(uq; scale=:transformed)

draws_nat = get_uq_draws(uq; scale=:natural)
draws_tr = get_uq_draws(uq; scale=:transformed)

diag = get_uq_diagnostics(uq)
```

Note that not all quantities are available from every backend:

- `:wald` and `:chain` provide covariance matrices and draws.
- `:profile` provides intervals but does not return covariance or draw matrices.
- `:mcmc_refit` returns chain-based quantities from the refit.

## UQ Summaries

For convenient inspection, `NoLimits.summarize` produces formatted summaries. A standalone UQ summary is obtained as follows.

```julia
uq_summary = NoLimits.summarize(uq)
uq_summary
```

To combine fit results and uncertainty information into a single summary, pass both objects.

```julia
fit_uq_summary = NoLimits.summarize(res, uq)
fit_uq_summary
```

## Parameter Inclusion Rules

UQ is computed on the subset of free fixed-effect coordinates that are eligible for uncertainty calculation. Eligibility is controlled at parameter definition time through the `calculate_se` argument in `@fixedEffects` constructors (e.g., `RealNumber`, `RealVector`, `RealPSDMatrix`, `NNParameters`, `SoftTreeParameters`).

By default, `calculate_se` is `false` for `NNParameters`, `SoftTreeParameters`, and `SplineParameters`, because standard errors for high-dimensional parameter vectors are rarely scientifically informative. Classical fixed-effect types such as `RealNumber` and `RealVector` are included by default unless explicitly excluded with `calculate_se=false`.

```julia
model = @Model begin
    @fixedEffects begin
        ka = RealNumber(1.0, scale=:log, calculate_se=true)   # included in UQ
        ke = RealNumber(0.5, scale=:log, calculate_se=true)   # included in UQ
        sigma = RealNumber(0.2, scale=:log, calculate_se=false) # excluded from UQ
    end

    @formulas begin
        y ~ Distributions.Normal(ka + ke, sigma)
    end
end
```

A coordinate is excluded from UQ if:

- its parent fixed effect is held constant via the `constants` argument, or
- its parameter block has `calculate_se=false`.

For mixed-effects fits, `constants_re` can be passed to hold selected random-effect levels fixed during UQ computations that involve random-effect approximations.

## Practical Notes

- `compute_uq` requires access to the original `DataModel`. This is available when the fit was run with `store_data_model=true` (the default).
- The `level` argument must satisfy `0 < level < 1`.
- For methods that rely on random draws (`:wald`, `:chain`, `:mcmc_refit`), pass an `rng` argument to ensure reproducibility.
