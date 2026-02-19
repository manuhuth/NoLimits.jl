# Profile Likelihood

Profile-likelihood confidence intervals are constructed by examining how the objective function changes as each parameter is varied away from its optimum, while all other parameters are re-optimized at each step. Unlike Wald intervals, which assume a locally quadratic log-likelihood, profile intervals can capture asymmetric uncertainty and are better suited to parameters near boundary constraints or in models with nonlinear reparameterizations. They are considered the gold standard for frequentist confidence intervals in nonlinear models.

In NoLimits.jl, profile-likelihood UQ is accessed through:

```julia
compute_uq(res; method=:profile, ...)
```

The underlying interval computation is performed via [LikelihoodProfiler.jl](https://insysbio.github.io/LikelihoodProfiler.jl/v0.3/).

## Applicability

Profile UQ is available for fitted results from the following estimation methods:

- `MLE`
- `MAP`
- `Laplace`
- `LaplaceMAP`

## Minimal Usage

```julia
using NoLimits
using Random

uq_profile = compute_uq(
    res;
    method=:profile,
    level=0.95,
    profile_method=:LIN_EXTRAPOL,
    profile_scan_width=1.0,
    profile_scan_tol=1e-2,
    profile_loss_tol=1e-2,
    profile_max_iter=300,
    rng=Random.Xoshiro(1),
)
```

## Core Controls

The following parameters govern the profile-likelihood algorithm and are exposed through `compute_uq`:

- `profile_method` (default `:LIN_EXTRAPOL`): the profiling algorithm used by LikelihoodProfiler.jl.
- `profile_scan_width` (must be positive): search window around the point estimate, specified in transformed-coordinate units and subject to parameter bounds.
- `profile_scan_tol`: tolerance for the scanning phase of the profiler.
- `profile_loss_tol`: tolerance on the objective function difference used to determine the interval boundary.
- `profile_local_alg` (default `:LN_NELDERMEAD`): local optimization algorithm applied near interval endpoints.
- `profile_max_iter`: maximum number of iterations for the local optimizer.
- `profile_ftol_abs`: absolute function tolerance for the local optimizer.
- `profile_kwargs`: additional keyword arguments forwarded directly to the underlying profiler call.

In practice, `profile_scan_width` determines how far from the estimate the profiler searches. If intervals appear truncated, increasing this value or raising `profile_max_iter` may help the profiler locate the true boundary.

## Fit-Context Overrides

The profile backend accepts the same fit-context overrides available in other UQ backends:

- `constants`
- `constants_re`
- `penalty`
- `ode_args`, `ode_kwargs`
- `serialization`
- `rng`

When not provided, stored values from the original fit are reused.

## Parameter Inclusion Rules

Profile UQ is evaluated only on free fixed-effect coordinates that are eligible for uncertainty calculation.

A coordinate is excluded when:

- its fixed-effect block is held constant via `constants`, or
- its block has `calculate_se=false`.

If no eligible coordinates remain, profile UQ raises an error.

```julia
fe = @fixedEffects begin
    a = RealNumber(0.2, calculate_se=true)     # included
    b = RealNumber(0.1, calculate_se=false)    # excluded
end
```

## Returned Quantities

The result is a `UQResult` with backend `:profile`. The available accessors are shown below.

```julia
backend = get_uq_backend(uq_profile)                # :profile
source = get_uq_source_method(uq_profile)
names = get_uq_parameter_names(uq_profile)

est_nat = get_uq_estimates(uq_profile; scale=:natural)
est_tr = get_uq_estimates(uq_profile; scale=:transformed)

ints_nat = get_uq_intervals(uq_profile; scale=:natural)
ints_tr = get_uq_intervals(uq_profile; scale=:transformed)

V_nat = get_uq_vcov(uq_profile; scale=:natural)     # nothing
draws_nat = get_uq_draws(uq_profile; scale=:natural) # nothing

diag = get_uq_diagnostics(uq_profile)
```

Because profile likelihood characterizes uncertainty by tracing the objective function surface rather than by sampling, covariance matrices and draw matrices are not available for this backend. Only interval estimates are returned.

## Diagnostics and Boundary Behavior

`get_uq_diagnostics(uq_profile)` returns profiler metadata that is useful for assessing the quality of the computed intervals:

- **Algorithm settings:** `profile_method`, tolerances, and local algorithm used.
- **Objective values:** `loss_at_estimate` and `loss_critical` (the threshold corresponding to the requested confidence level).
- **Per-parameter endpoint status:** `left_status` and `right_status` indicate whether each boundary was successfully located.
- **Per-parameter endpoint counters:** `left_counter` and `right_counter` report the number of profiler evaluations at each boundary.
- **Endpoint availability:** `endpoint_found` flags whether both interval endpoints were determined.
- **Per-parameter errors:** `errors` captures any profiler-level issues encountered during computation.

These diagnostics are essential for identifying incomplete or numerically unstable intervals. If an interval endpoint was not found, common remedies include widening `profile_scan_width`, increasing `profile_max_iter`, or relaxing `profile_loss_tol`.
