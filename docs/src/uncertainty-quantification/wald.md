# Wald

The Wald method is the most widely used approach to uncertainty quantification for maximum likelihood and related estimators. It approximates the log-likelihood surface as locally quadratic at the optimum, yielding a Gaussian approximation to the sampling distribution of the parameter estimates. This approximation is computationally inexpensive and accurate when the model is well-identified and the sample size is adequate.

In NoLimits.jl, Wald UQ is accessed via `compute_uq(...; method=:wald)`. It constructs an approximate variance-covariance matrix for eligible fixed-effect coordinates, then draws from the implied Gaussian distribution to form confidence intervals.

For visualization of Wald-based densities, closed-form curves are used when available:

- **Transformed scale:** Normal distribution (closed form).
- **Natural scale:** Normal for identity-scale coordinates; LogNormal for log-scale coordinates.
- **Other cases:** draw-based kernel density estimation (KDE) is used as a fallback.

## Applicability

Wald UQ is supported for results from the following estimation methods:

- Fixed-effects fits: `MLE` and `MAP`
- Mixed-effects fits: `Laplace`, `LaplaceMAP`, `MCEM`, and `SAEM`

## Minimal Usage

```julia
using NoLimits

uq = compute_uq(
    res;
    method=:wald,
    level=0.95,
    vcov=:hessian,
    n_draws=2000,
)
```

## Covariance Choice

The `vcov` argument selects the variance-covariance estimator.

- `vcov=:hessian`: computes the covariance as the inverse of the observed information (Hessian) matrix. This is the standard choice when the model is correctly specified.
- `vcov=:sandwich`: computes the sandwich (robust) covariance estimator, which provides consistent standard errors even under mild model misspecification.

```julia
uq_hessian = compute_uq(res; method=:wald, vcov=:hessian, n_draws=1000)
uq_sandwich = compute_uq(res; method=:wald, vcov=:sandwich, n_draws=1000)
```

## Numerical Robustness Controls

When the Hessian is poorly conditioned -- for example, because the log-likelihood surface is locally flat along certain parameter directions -- the following controls can improve numerical stability:

- `pseudo_inverse`: use the Moore-Penrose pseudo-inverse when direct matrix inversion is numerically unstable.
- `hessian_backend`: choose the differentiation strategy -- `:auto`, `:forwarddiff`, or `:fd_gradient` (finite-difference based).
- `fd_abs_step`: absolute step size for finite-difference computation.
- `fd_rel_step`: relative step size for finite-difference computation.
- `fd_max_tries`: maximum number of retry attempts for finite-difference gradient and Hessian evaluations.

```julia
uq_robust = compute_uq(
    res;
    method=:wald,
    pseudo_inverse=true,
    hessian_backend=:fd_gradient,
    fd_abs_step=1e-4,
    fd_rel_step=1e-3,
    fd_max_tries=8,
    n_draws=1500,
)
```

When the covariance matrix requires projection to restore positive semi-definiteness, this is reported in the diagnostics (via `vcov_projected` and eigenvalue-related fields).

## MCEM/SAEM Approximation Controls

For fits obtained with `MCEM` or `SAEM`, the marginal likelihood is not available in closed form, so the Hessian cannot be computed directly from the EM objective. In these cases, the Wald backend uses a random-effects approximation method (typically Laplace) during the Hessian evaluation.

- `re_approx=:auto` (the default) selects a Laplace-style approximation automatically.
- `re_approx_method` allows passing an explicit approximation method instance for finer control.

```julia
if @isdefined(res_saem) && res_saem !== nothing
    uq_saem = compute_uq(
        res_saem;
        method=:wald,
        re_approx=:laplace,
        n_draws=800,
    )
end

if @isdefined(res_mcem) && res_mcem !== nothing
    uq_mcem_custom = compute_uq(
        res_mcem;
        method=:wald,
        re_approx_method=NoLimits.Laplace(; multistart_n=0, multistart_k=0),
        n_draws=800,
    )
end
```

### Choosing `n_draws`

The `n_draws` parameter controls the number of Monte Carlo samples drawn from the Wald Gaussian approximation. These draws are used to compute draw-based summaries (accessible via `get_uq_draws`), interval estimates, and the natural-scale covariance matrix.

Importantly, `n_draws` does not affect the transformed-scale covariance matrix itself (`get_uq_vcov(...; scale=:transformed)`), which is derived directly from the Hessian or sandwich calculation.

Guidelines for setting `n_draws`:

- **Lower values** (e.g., a few hundred): faster computation, but with greater Monte Carlo variability in the resulting intervals.
- **Higher values** (e.g., several thousand): more stable interval estimates at the cost of additional computation.

The value `n_draws=800` used in examples above represents a practical middle ground for interactive exploration. For final reporting, it is advisable to increase `n_draws` and verify stability by rerunning with a different `rng` seed.

For density visualization via `plot_uq_distributions`, the distinction between closed-form and draw-based densities becomes relevant:

- Identity-scale and log-scale coordinates use closed-form Normal or LogNormal density curves, independent of `n_draws`.
- Coordinates requiring nontrivial inverse transforms (e.g., PSD matrix elements parameterized via `:cholesky` or `:expm`) rely on draw-based KDE, making the draw count important for plot quality.

## Parameter Inclusion Rules

Wald UQ is computed only on the subset of free fixed-effect coordinates that are eligible for uncertainty calculation.

A coordinate is excluded when:

- its fixed effect is held constant via `constants`, or
- its parameter block has `calculate_se=false`.

If no eligible coordinates remain after exclusion, Wald UQ raises an error.

```julia
fe = @fixedEffects begin
    a = RealNumber(0.2, calculate_se=true)                 # included
    b = RealNumber(0.1, calculate_se=false)                # excluded
    sigma = RealNumber(0.3, scale=:log, calculate_se=true) # included
end

uq = compute_uq(res; method=:wald, constants=(; a=0.2))
```

In this example, `a` is also excluded because it is fixed by `constants`, leaving only `sigma` in the UQ computation.

## Fit-Kwarg Forwarding

`compute_uq(...; method=:wald)` accepts overrides for fit-related settings that may differ from those used during the original estimation:

- `constants`
- `constants_re`
- `penalty`
- `ode_args`, `ode_kwargs`
- `serialization`
- `rng`

When these are omitted, the values stored from the original fit are used.

## Returned Quantities

Wald UQ returns a `UQResult` with backend `:wald`. The full set of accessor functions is shown below.

```julia
backend = get_uq_backend(uq)                 # :wald
source = get_uq_source_method(uq)            # original fit method symbol
names = get_uq_parameter_names(uq)

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
