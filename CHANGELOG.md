# Changelog

## Unreleased

### Bug fixes

- `compute_uq(res; method = :profile)` returned all-`NaN` intervals under every
  resolvable LikelihoodProfiler version. The backend called the 0.x
  `LikelihoodProfiler.get_interval`, which 1.x removed, and the per-parameter
  `try`/`catch` turned the resulting `UndefVarError` into a silent `NaN` plus an entry
  in the `errors` diagnostic. Profile UQ now uses the 1.x
  `ProfileLikelihoodProblem` / `solve` / `endpoints` interface, and a coordinate whose
  interval cannot be computed warns instead of failing silently. `profile_method`
  selects the 1.x stepper (`:LIN_EXTRAPOL`, `:SINGLE_AXIS`, `:FIXED_STEP`); the 0.x
  values `:CICO_ONE_PASS` and `:QUADR_EXTRAPOL` are rejected with an explanatory error.
  `compute_uq`'s `profile_scan_tol` and `profile_loss_tol` are deprecated and ignored -
  they were CICO scan tolerances of the 0.x backend with no 1.x counterpart, so passing
  either warns rather than being remapped onto a different control, and both are gone
  from the profile diagnostics.
  `[compat]` now states `LikelihoodProfiler = "1.5"`; the previous `"0.3.3, 1"` range
  advertised a 0.x path that has not been resolvable since the `OptimizationNLopt`
  floor moved to NLopt 1.x.
- SAEM with more than one E-step chain collapsed every random-effect variance
  geometrically to the `1e-5` floor, independent of the data, the sampler, the
  starting values, and the M-step mode. The chains' η draws were AVERAGED into a
  single pseudo-sample (`b_current`) before the sufficient statistics and Q
  objectives were formed; the second moment of an average of `C` posterior draws is
  `Ω²·(1 − B(1 − 1/C))` (`B` = shrinkage fraction) instead of `Ω²`, whose only fixed
  point is `Ω = 0`. Because `auto_small_n_chains = true` silently raises the chain
  count whenever there are fewer than `small_n_chain_target` (50) batches, every
  small-dataset SAEM fit was affected. Chains are now consumed as separate draws
  everywhere: the closed-form sufficient statistics accumulate over all chains, the
  ring buffer stores one entry per chain with weight `γ/n_chains`,
  `Q_current`/`Q2_current` average the log-densities over chains, custom `suffstats`
  are evaluated per chain and averaged, and the E-step retry check flags a batch if
  any chain is non-finite. Single-chain fits are unchanged.
- The builtin closed-form M-step silently overwrote user-supplied `constants`
  (e.g. `constants = (; Omega = 0.6)` still updated `Omega` every iteration).
  Constants now always win over closed-form updates.
- The analytic outer gradient of the `Laplace` and `FOCEI` marginal objectives was wrong,
  which made a gradient-based outer optimizer perform far worse than the derivative-free
  default instead of better. Two independent causes, both invisible to `LN_BOBYQA`:
  - A positive-definite `-H` was factorized with the Cholesky `jitter` added
    unconditionally rather than only as a rescue. With the default
    `adaptive_jitter = true, jitter_scale = 1e-6` that jitter is proportional to
    `mean|diag(-H)|`, so the objective was `logdet(-H + δ(θ,b)·I)` while the analytic
    gradient differentiated it as if `δ` were constant, and the same regularized factor was
    reused for the implicit `db*/dθ` solve. On a badly scaled start `δ` reached 9% of
    `λmin(-H)`. `-H` is now factorized untouched whenever it is definite, so the objective
    is the actual Laplace marginal and its gradient is consistent with it. The jitter
    keywords are retained and still rescue an indefinite `-H`. This also makes the AGHQ
    quadrature scale, the conditional-covariance draws behind VPC/CV, and the inner Newton
    step exact rather than regularized.
  - `db*/dθ` was obtained by solving with whatever curvature the log-det term used. Under
    `FOCEI`/FOCE that is the Fisher-information surrogate, but `b*` is the mode of
    `log f`, so the implicit function theorem requires the exact inner Hessian. The
    surrogate is within ~1% of it, yet the correction it feeds is a difference of large
    nearly-cancelling terms, so the outer gradient came out wrong by up to 240%. FOCEI/FOCE
    now solve that system with the exact Hessian, falling back to the surrogate if the
    exact `-H` is indefinite. `Laplace` is unaffected and pays nothing; the FOCEI objective
    stays first-order and only its gradient costs ~1.4% more.
- Laplace- and FOCEI-based marginal likelihoods no longer report values obtained from a
  degenerate empirical-Bayes Hessian. When `-H` at the EB mode was positive definite only
  because the Cholesky `jitter` had been added, the `-½·logdet(-H)` term was set by the
  regularisation rather than by posterior curvature and inflated the marginal by
  `(n_b/2)·log(1/jitter)` per batch. Reported log-likelihoods could exceed the exact
  ceiling `-n/2·log(2πσ²)` that a marginal must satisfy, silently and with
  `converged = true`, which corrupted model comparison, AIC and BIC. Such a batch is now
  rejected: the fitting objectives return `-Inf` so the outer optimizer backtracks out of
  the degenerate region instead of being rewarded for finding it, `laplace_marginal`
  warns and returns `-Inf`, and `get_marginal_likelihood` falls back to MC integration.
  This complements the existing `nan_recovery` machinery, which only fires on non-finite
  values and thrown exceptions. Well-conditioned fits are bit-unchanged.
  The rejection threshold is the jitter the protected Cholesky actually adds, so with the
  default `adaptive_jitter = true` it is the curvature of `-H` relative to the problem's
  own diagonal scale. Previously it was compared against the bare `jitter`, which made
  admissibility depend on the units the data was recorded in: the same degenerate
  posterior was rejected in one unit system and accepted in another.
- The transformed `:cholesky` block occupied `n²` slots instead of `n(n+1)/2`, so the
  strict upper triangle of the log-Cholesky factor — which has no effect on the
  reconstructed matrix — was handed to the optimizer as `n(n-1)/2` exactly-flat
  directions. It now stores the lower triangle only, matching `:expm` and `:lie`. This
  changes the length and layout of the transformed vector, of `get_flat_names`, and of
  Wald UQ coordinates for `RealPSDMatrix(scale = :cholesky)` parameters.

- Wald UQ no longer returns `NaN` natural-scale summaries when a transformed-scale draw
  overflows. `wald_uq` draws from a Gaussian on the transformed scale and pushes each draw
  through the inverse transform; for a covariance parameter that transform is exponential, so
  a wide-but-legitimate transformed covariance (`max(diag) ~ 1e12` for `:cholesky` and
  `:lie`, versus `~1e4` for `:expm`) sends draws to `Inf`, and a single one poisoned every
  natural-scale interval, standard error and correlation. Non-finite draws are now excluded
  from the natural-scale summaries with a warning naming how many were dropped, recorded in
  the new `n_draws_nonfinite_natural` diagnostic, and an error is raised only if every draw
  is non-finite. `get_uq_draws` still returns all requested draws untouched. The
  transformed-scale summaries were never affected.
- `wald_uq` now errors with an actionable message instead of producing a silently meaningless
  covariance when the objective Hessian at the estimate contains non-finite entries, which
  `pinv` otherwise propagates into every reported quantity.

- `GHQuadrature` no longer throws `MethodError: no method matching Float64(::ForwardDiff.Dual)`
  when its outer optimizer uses automatic differentiation. `batch_loglik_ghq` took its
  accumulator element type from the random-effects measure alone, on the assumption that a
  Dual-tagged `θ` always yields a Dual-valued measure. That fails for a random effect declared
  with fixed hyperparameters (e.g. `RandomEffect(Normal(0.0, 1.0))`), whose measure carries no
  `θ` and stays `Float64` while the conditional log-likelihoods being summed into it are Dual.
  The accumulator is now promoted against `θ` as well. The bug predates this release but was
  unreachable while the default outer optimizer was derivative-free; it would have surfaced as
  soon as the default became gradient-based. Non-AD fits and models whose measure does depend
  on `θ` are bit-unchanged, since `promote_type` is the identity in both cases.

### Other changes

- All nine optimization-based methods (`MLE`, `MAP`, `Laplace`, `FOCEI`, `GHQuadrature`,
  `SAEM`, `MCEM`, `Pooled`, `PooledMap`) now precondition the outer problem by default,
  via a new `precondition::Bool = true` keyword. The optimizer works in a scaled offset
  `θ_transformed = θ0 + s .* z` and therefore always starts from `z = 0`, with `s = 1` for
  coordinates already in log/logit space and `s = max(abs(θ0), 1)` for genuinely
  natural-scale `:identity` coordinates. This removes a failure mode in which a coordinate
  whose starting value was near zero could not move at all, because several optimizers size
  their initial trial step relative to `abs(x0)`. Set `precondition = false` to optimize the
  transformed vector directly, which reproduces earlier results bit-for-bit. With
  preconditioning on, the optimizer object returned by `get_raw` works in `z`; `get_params`
  is unaffected. `SAEM` and `MCEM` re-anchor `θ0` at the current iterate each M-step.
- Optimization-based methods that can return a non-finite objective now hand the optimizer a
  large finite value derived from the best objective seen so far, rather than `Inf`. A line
  search cannot read a slope from `Inf`, so an overflowing trial step used to abort the fit at
  its starting value; it now backtracks out of the infeasible region.

- `Laplace` and `FOCEI` now document that a gradient-based outer optimizer must cap its
  line-search step. With the outer gradient corrected (see above) it is usable, and on the widest
  benchmark model it beats the derivative-free default by ~536 `-2LL` units - but the gradient's
  coordinates can span four orders of magnitude at a poorly scaled start, so an uncapped unit
  first step overflows into the region where the marginal is not finite. `BackTracking(maxstep =
  1.0)`, the convention the inner optimizer has always used, takes `pheno_sd` from `-2LL` 6038 to
  973.44 (the BOBYQA optimum) and converges in 91 s instead of exhausting `maxiters` in 623 s.
  Finite `lb`/`ub` are an alternative - they route through `Fminbox`, whose barrier keeps
  iterates interior - but are ~300x slower for the same answer. Do not combine a step cap with a
  shrunken `alphaguess`; the two starve each other.
- `FOCEI` accepts BlackBoxOptim optimizers, on the same terms as `Laplace`: finite bounds
  on every free parameter are required and the start is clamped into the box.
- `Optimization` is capped below 5.7. Optimization 5.7.0 stopped exporting the
  `SciMLBase` module name, and `LikelihoodProfiler` (up to its current 1.5.3) relies on
  that export: it imports only individual names via
  `@reexport import SciMLBase: ...` yet defines `SciMLBase.remake`, so it fails to
  precompile with `UndefVarError: SciMLBase not defined in LikelihoodProfiler`. That broke
  every fresh dependency resolve, including CI, the docs build and Aqua's
  persistent-tasks probe. The cap can be lifted once LikelihoodProfiler imports
  `SciMLBase` itself.

### Breaking changes

- Plotting migrated from Plots.jl to Makie/CairoMakie. All plotting functions now live in
  the `NoLimitsMakieExt` package extension, loaded by `using CairoMakie` (or another Makie
  backend). See the migration guide in the documentation. (#64, #65)
- Result accessors renamed to inference-neutral names. The seven optimization result
  structs are collapsed into a single `StandardOptimizationResult`. (#80, #83)
- `joint_loglikelihood` is replaced by `complete_data_loglikelihood`. (#60, #83)
- `posterior_moments` is split into `empirical_bayes` and `empirical_bayes_covariance`. (#83)
- Laplace and FOCEI now default the outer optimizer to `NLopt.LN_BOBYQA()` instead of a
  gradient-based method. Pass `optimizer=` to restore the previous behaviour. (#31)

### New features

- Public method-developer API for writing custom NLME estimators: likelihood, posterior
  and empirical-Bayes primitives, the `FittingMethod` framework, a `FitContext`
  convenience tier, a Bayesian build path, and a documentation reference. (#80, #82, #83)
- Closed-form fast path for linear ODE systems (diagonal, general, with events, and
  hybrid closed-form/numeric), detected automatically. (#72)
- `predict` gained a `re_mode` keyword with `:population`, `:ebe`, `:reestimate` and
  `:marginal` prediction modes. (#73)
- `RealLiePSDMatrix`, a Lie-algebraic covariance parameterization with per-eigenvalue box
  bounds, block-diagonal structure and eigenvalue fixing. (#50)
- `crossing_rootval` for root-finding on model signals, and a `t0` option for
  `DataModel`. (#53)
- Likelihood integration over multiple datasets, and a default `maxiters` for ODE
  solves. (#51)
- `RealNumber` and `RealVector` with finite bounds and no explicit prior now default to a
  uniform prior over the bounds. (#44)
- `logabsdetjac` implemented for all structured parameter scales. (#81)
- Exact second-order ForwardDiff derivatives for `:expm` and `:lie` covariance
  parameterizations. (#78)
- `complete_data_loglikelihood_per_individual` is part of the public API and rendered in
  the API reference. (#83)

### Estimation improvements

- SAEM: retuned defaults and a windowed, Monte-Carlo-noise-aware early-stopping criterion
  replacing the previous convergence test, which in practice never fired. (#68)
- SAEM: the `sa_anneal` floor now also applies on the closed-form Gaussian variance
  update path, where it was previously inert. (#70)
- MCEM: adopts the same windowed early-stopping criterion. (#69)

### Bug fixes

- `extra_objective` dropped the random-effects variance term in SAEM and MCEM and had a
  sign error in MCMC and VI. (#52)
- Fixed five latent bugs found in a code audit: Laplace penalty gradient, `constants_re`
  handling in cross-validation, HMM filtering in MCMC residuals, a SAEM skip path, and an
  inert NPF seed. (#48)
- `fit_cv` no longer throws a type error with `fold_serialization=EnsembleSerial()`. (#40)
- `summarize` uses fixed 4-decimal numeric formatting. (#62)
- Fixed a plotting issue for crossing computations. (#53)

### Performance

- Type-stability and allocation fixes on the estimation hot path, including the
  closed-form ODE path. (#75, #76)
- Laplace shares the level-to-index map across random-effect batches, reducing setup from
  O(N^2) to O(N). (#57)
- Removed dead code and duplicate helpers across estimation and plotting. (#48, #55, #59,
  #63, #74)

### Dependencies

- Lifted the OrdinaryDiffEq v7 and Roots v3 compat caps. (#66)
- Allow DataInterpolations v9 and DiffEqBase v7.6. (#71)
- Dropped the Parameters.jl dependency. (#49)

## v0.1.0

Initial release.
