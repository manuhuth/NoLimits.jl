# Profile Likelihood

Profile-likelihood confidence intervals are constructed by examining how the objective function changes as each parameter is varied away from its optimum, while all other parameters are re-optimized at each step. Unlike Wald intervals, which assume a locally quadratic log-likelihood, profile intervals can capture asymmetric uncertainty and are better suited to parameters near boundary constraints or in models with nonlinear reparameterizations. They are considered the gold standard for frequentist confidence intervals in nonlinear models.

In NoLimits.jl, profile-likelihood UQ is accessed through:

```julia
using NoLimits
using LikelihoodProfiler, OptimizationNLopt   # optional dependencies, see below

compute_uq(res; method=:profile, ...)
```

!!! note "Two optional dependencies"
    The profile backend lives in a package extension and needs both LikelihoodProfiler (the
    scan) and OptimizationNLopt (the local optimizer named by `profile_local_alg`, which
    defaults to `:LN_NELDERMEAD`). Neither is installed with NoLimits: run
    `using Pkg; Pkg.add(["LikelihoodProfiler", "OptimizationNLopt"])` once, then `using` both.
    Requesting `method=:profile` without them raises an error saying exactly this.

The underlying interval computation is performed via [LikelihoodProfiler.jl](https://insysbio.github.io/LikelihoodProfiler.jl/v0.3/).

## Applicability

Profile UQ is available for fitted results from the following estimation methods:

- `MLE`
- `MAP`
- `Laplace`

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
    profile_max_iter=300,
    rng=Random.Xoshiro(1),
)
```

## Core Controls

The following parameters govern the profile-likelihood algorithm and are exposed through `compute_uq`:

- `profile_method` (default `:LIN_EXTRAPOL`): how the profiler proposes the next profile point. `:LIN_EXTRAPOL` extrapolates all parameters from the last two points, `:SINGLE_AXIS` extrapolates only the profiled parameter, and `:FIXED_STEP` disables adaptive stepping. The LikelihoodProfiler 0.x values `:CICO_ONE_PASS` and `:QUADR_EXTRAPOL` no longer exist and are rejected.
- `profile_scan_width` (must be positive): search window around the point estimate, specified in transformed-coordinate units and subject to parameter bounds.
- `profile_local_alg` (default `:LN_NELDERMEAD`): NLopt algorithm used to re-optimize the remaining parameters at each profile point. Derivative-free algorithms are required, since the profiled objective is evaluated without automatic differentiation.
- `profile_max_iter`: maximum number of iterations for that inner optimizer.
- `profile_ftol_abs`: absolute function tolerance for that inner optimizer.
- `profile_kwargs`: additional keyword arguments forwarded to `LikelihoodProfiler.solve` (for example `maxiters` or `verbose`).

`profile_scan_tol` and `profile_loss_tol` are deprecated and ignored. They configured the CICO scan of the LikelihoodProfiler 0.x backend, which has no counterpart in 1.x; passing either warns and nothing is substituted for it.

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
- **Per-parameter endpoint status:** `left_status` and `right_status` indicate whether each boundary was successfully located (`:Identifiable`, `:NonIdentifiable`, `:MaxIters`, `:Failure`, or `:ERROR` if the profiler itself threw).
- **Per-parameter endpoint counters:** `left_counter` and `right_counter` report the number of profiler evaluations at each boundary, or `-1` when the profiler does not report them.
- **Endpoint availability:** `endpoint_found` flags whether both interval endpoints were determined.
- **Per-parameter errors:** `errors` captures any profiler-level issues encountered during computation.

These diagnostics are essential for identifying incomplete or numerically unstable intervals. If an interval endpoint was not found, common remedies include widening `profile_scan_width`, increasing `profile_max_iter`, or relaxing `profile_ftol_abs`.
