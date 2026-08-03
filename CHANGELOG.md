# Changelog

## Unreleased

### Bug fixes

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
- The transformed `:cholesky` block occupied `n²` slots instead of `n(n+1)/2`, so the
  strict upper triangle of the log-Cholesky factor — which has no effect on the
  reconstructed matrix — was handed to the optimizer as `n(n-1)/2` exactly-flat
  directions. It now stores the lower triangle only, matching `:expm` and `:lie`. This
  changes the length and layout of the transformed vector, of `get_flat_names`, and of
  Wald UQ coordinates for `RealPSDMatrix(scale = :cholesky)` parameters.

### Other changes

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
