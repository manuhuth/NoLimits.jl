# Changelog

## v0.2.0

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
