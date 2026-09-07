# Changelog

## Unreleased

## v0.2.10

### Bug fixes

- The warning raised when a Wald covariance is projected to the nearest PSD matrix
  recommended `method = :profile / :mcmc`. Neither name was right for every fit: the
  keyword is `:mcmc_refit`, and profile UQ is restricted to MLE/MAP/Laplace/GHQuadrature
  results, so an MCEM or SAEM user following the advice hit
  "Profile UQ is currently supported for ...". The message now names only the backends that
  exist for the fit at hand.

- A normalizing-flow random effect could kill a whole `SAEM` fit instead of scoring the bad
  proposal `-Inf`. When the Q2 M-step drives a planar layer's weight to exactly `w = 0`,
  `Bijectors.get_u_hat` returns `u_hat = NaN` while `w'u_hat` stays finite, the `NaN`
  propagates into the next layer's `find_alpha`, and Roots rejects the resulting `[NaN, NaN]`
  bracket with `ArgumentError("The interval [a,b] is not a bracketing interval...")`. The
  random-effect prior already wraps that call intending to score it `-Inf`, but
  `_is_numeric_error` classified `ArgumentError` by message only and this message matched
  none of the alternatives, so the guard rethrew. Roots' bracketing message is now
  classified as a numeric error, which closes the same hole in `_re_logpdf_batch` and
  `_laplace_logf_batch` at once, so every estimator gets it. Only paths that previously
  threw are affected: no successful fit changes. This is not a Bijectors 0.16 regression;
  Bijectors 0.15.24 with `Roots.ITP` fails identically, and the compat bounds are unchanged.

### New features

- `MCEM` gained the closed-form M-step that `SAEM` already had, through four new keywords
  with SAEM's names, meanings and defaults: `builtin_stats` (default `:auto`),
  `resid_var_param`, `re_cov_params` and `re_mean_params`. Parameters whose Monte Carlo
  Q-maximizer is available in closed form (Gaussian/log-normal random-effect means and
  covariances, `Exponential` random-effect scales, the residual scale, supported HMM
  emissions) are updated from the sufficient statistics of the current E-step draws, and
  the remaining free parameters are optimized numerically as before. When nothing is left
  the numerical M-step is skipped entirely. Unlike SAEM there is no stochastic-approximation
  smoothing: the statistics are a plain Monte Carlo average over this iteration's draws.
  The path removes the optimizer's BLAS calls from the M-step, which is what made the same
  fit differ in its last bits between AVX-512 and AVX2/Zen machines. It disables itself for
  an `MCEM_IS` E-step (weighted draws, unweighted statistics) and when `extra_objective` is
  passed (the M-step is then not separable), each with an info message. The routing is
  logged at startup and recorded in `notes` as `closed_form_targets`, `numeric_targets`,
  `closed_form_mstep_mode` and `builtin_stats_closed_form_eligibility`, with
  `get_closed_form_mstep_used(get_result(res))` as the one-line check.
- A random-effect mean written as `β + offset`, where `β` is a fixed effect and the offset
  carries no fixed effect (for example the allometric `LogNormal(CL_mean + 0.75 * log(wt /
  70), sigma_CL)`), is now a closed-form target for both `SAEM` and `MCEM`. The per-level
  offset is subtracted before the moments are formed, so `β` is updated in closed form and
  the covariance update centers on the structured mean instead of the pooled empirical one.
  A mean with two free fixed effects in it (`μ0 + β * x`) remains ineligible.

### Behavior changes

- **`MCEM` results change for models with an exponential-family block**, because
  `builtin_stats` defaults to `:auto`. The closed-form update is the exact maximizer of the
  same Monte Carlo Q-function over those parameters, so the estimator's target is unchanged,
  but it carries none of the optimizer's tolerance or jitter: the sequence of iterates
  differs, the windowed drift test typically fires earlier, and the reported iteration count
  and objective move. Pass `MCEM(builtin_stats = :none)` to restore the previous, fully
  numerical M-step exactly.
- **`SAEM` covariance estimates change for random effects with a structured mean** of the
  `β + offset` form described above, because the covariance is now centered on that mean.
  The previous update centered on the pooled empirical mean, which absorbed the
  covariate-driven between-subject spread into the random-effect variance.
- **`SAEM` and `MCEM` closed-form variance estimates change for literal-mean random
  effects**, i.e. the common `Normal(0.0, ω)` / `LogNormal(0.0, ω)` form. The mean is known
  there, so the exact conditional maximizer is the second moment *about that mean*,
  `sqrt(Σ(η - μ)² / n)`. The previous update computed `second - mean * mean'`, subtracting
  the pooled empirical mean of the draws, which is the maximizer only when the mean is a
  free parameter estimated from those same moments. The old value was biased low by
  `E[η - μ]²`, and that bias is not always small: on the `fx_re_dm` test fixture the
  closed-form ω moves from 0.162 to 0.228 at identical draws. Only the closed-form path
  changes, so `builtin_stats = :none` reproduces the previous numbers.
- Method-level `lb`/`ub` never constrained closed-form updates (they are transformed-scale
  bounds for the optimizer); closed-form values are clamped to each fixed effect's own
  declared natural-scale bounds. The SAEM documentation claimed otherwise and has been
  corrected.

### Documentation

- The `SAEM` docstring listed `builtin_stats = :on`/`:off` and
  `builtin_mean = :additive`/`:all`, none of which exist. The accepted values are
  `:auto`/`:closed_form`/`:gaussian_re`/`:none` and `:none`/`:glm`.
- Enabling the Turing extension is now documented as `import Turing` rather than
  `using Turing`, because `using Turing` collides with the NoLimits exports `Laplace`,
  `MAP`, `MLE`, `loglikelihood`, `logprior` and `predict`.
- Backfilled the missing `## v0.2.0` heading below (its release notes were folded into the
  `v0.2.1` section when the changelog was first written).

## v0.2.9

### Bug fixes

- `extra_objective` was dropped from the fit metadata, so Wald covariances, profile
  likelihoods and the `mcmc_refit` UQ path inverted a different Hessian than the one that
  was minimized. Every estimator now records it and all three UQ paths re-add it (#331).
- Wald with a single usable natural-scale draw reported zero variance. The exact scalar
  closed form is now used directly, and only the coordinates that genuinely need sampled
  moments warn on fewer than two usable draws (#338).
- Profile UQ reconstructed the declared model box instead of resolving the estimator's
  effective `lb`/`ub`/`ignore_model_bounds`, so scan and nuisance optimization ran on the
  wrong domain (#340).
- A bound-pinned coordinate aborted every other profile interval, and the inward epsilon
  could cross the estimate, making a strictly interior estimate near a bound unprofilable
  (#341).
- `fit_cv` Monte Carlo modes dropped failed draws from the denominator instead of scoring
  them as zero probability, and the failed-draw count is now reported (#332).
- The cross-validation loss column was copied from the first draw rather than averaged,
  because `:loss in names(df)` is always false for a `String`-keyed `names` (#333).
- `fit_cv` now derives independent training and scoring RNG streams per fold and passes the
  training stream to `fit_model` (#334).
- Boundary simplex declarations are rejected with a parameter-specific message, and
  `stickbreak_forward` throws a `DomainError` on an exhausted stick instead of producing
  `0/0` NaN coordinates (#336).
- The stick-breaking pullback and the logit, elementwise and stick-breaking log-Jacobians
  now use the clamped derivative, so they agree with the clamped inverse they
  differentiate (#337).
- Turing's adaptive samplers take `nadapts`, not `adapt`, so the old keyword was absorbed
  by kwargs and never controlled adaptation. Turing also discards adaptation before
  returning, so the stored warmup is now the retained row count, the request is kept as
  `n_adapt_requested`, and posterior draws are no longer trimmed twice (#335).
- `predict(re_mode = :population)` on an MCMC or VI fit plugged in the posterior mean of
  the fixed effects instead of integrating over the posterior draws (#339).
- `GHQuadrature` laid its nodes out on the natural simplex dimension `d+1` while the
  `MvLogitNormal` transport needs `d` normal coordinates, throwing a `DimensionMismatch`
  for every such batch (#342).
- The population-moment helpers validate series shapes before their unchecked loops instead
  of silently truncating or reading out of bounds, and reject an empty observation series
  (#343).
- Two byte-identical `@Model` blocks produced different `typeof(model)` depending on the
  module they expanded in, so pkgimage-cached specializations were never reused. The
  `@randomEffects` and `@DifferentialEquation` RGFs are now tagged with `NoLimits` itself
  and their emitted closures replaced by top-level callable structs (#327).
- The Laplace empirical-Bayes cache validity flags raced under `EnsembleThreads`, and the
  numeric-error classifier masked genuine `ArgumentError`s and mis-probed MLE AD support
  (#326).

### Documentation

- Restructured navigation with new NONMEM migration, troubleshooting, reproducibility and
  advanced SAEM pages, and a split API reference.
- Added a guide to precompiling models with a `PrecompileTools` workload.

## v0.2.8

### Features

- Mini-batching (`update_schedule = :all | Int | (nbatches, iter, rng) -> Vector{Int}`) for
  `Laplace`, `FOCEI`, `MLE`, `MAP` and `GHQuadrature`, sharing the SAEM/MCEM selector
  (#281). One mini-batch per outer iteration, drawn lazily so objective and gradient of an
  iteration always share the selection; unselected batches do no EBE, quadrature or
  likelihood work. The optimizer default becomes `Optimisers.Adam(0.01)` under
  mini-batching, with bounds applied through a projected rule.
- `Laplace` and `FOCEI` now supply `fg` so the outer optimizer uses the analytic gradient.
- `summarize` shows the convergence flag, which is now honest for custom estimators, and
  the Laplace family reports an EBE gradient diagnostic (#311).

### Bug fixes

- Wald intervals and natural-scale standard errors are now closed-form where the math gives
  one, transformed-scale chain UQ is genuine rather than a relabelled natural scale, and
  unidentified directions, non-finite fits and degenerate sandwiches no longer report false
  precision (#306).
- Infusion rate state is per solve instead of a mutable buffer shared across threads, and
  closed-form ODE detection is thread-safe (#308).
- Crossing times at a coincident dose, `AMT=0` with a nonzero rate, dosing-only individuals
  and duplicate timestamps are handled or rejected explicitly (#308).
- Covariate validation gaps, the dynamic-covariate `tspan` guard and knot `tstops` (#309).
- `fit_cv` Monte Carlo modes score the per-subject joint marginal (#290).
- Model building now runs the DE symbol, crossing, solver-config and formula-order checks at
  build time, validates covariates by declaration, rejects swapped `DataModel` arguments,
  keeps the real error message in numeric warnings and reports line numbers from the block
  parsers (#312, #313, #314, #315, #316).
- Every EM M-step honours `adtype` and user bounds, bound-hungry optimizers are guarded, and
  starts are clamped into the box (#311).
- FOCEI prior-mean failure, the MvNormal FOCE freeze and the PooledMap prior gate (#284);
  three EM sampler and SAEM bugs (#283).
- Spline right-boundary knot, cross-validation `t0` anchoring and the multi-output soft-tree
  guard (#304); a warning when a fit never reaches a finite objective (#310).
- Mode-centered AGHQ for all real-supported random effects, random-effect post-processing at
  non-primary levels, `plot_hidden_states` for univariate HMM outcomes, and the
  `NormalizingPlanarFlow` rewrite in `@formulas`.
- Shared parameters are excluded from the SAEM closed-form M-step, and
  `MultistartFitResult` forwards the whole post-processing surface.

## v0.2.7

### Features

- Method-developer API for MCEM: `mcem_e_step` (state-threaded) plus the M-step Q
  primitives `mcem_q_objective_and_gradient` and `mcem_q_partition`.
- Method-developer API for SAEM: sufficient-statistics and eligibility primitives, and the
  stateful `saem_closed_form_mstep`.

## v0.2.6

### Features

- Method-dispatched `objective_and_gradient` protocol (#271) and `FitContext` forms of the
  gradient-bearing dev-API primitives (#273).
- The analytic Laplace marginal theta-gradient is now public (#269).

## v0.2.5

### Features

- `FFNNParameters`, a dependency-free feed-forward network parameter block (#261).
- Public APIs accept any Tables.jl table where a `DataFrame` is expected (#259), any
  `AbstractDict` where a `NamedTuple` is expected (#257), and strings wherever a `Symbol` is
  expected (#255).

### Bug fixes

- `predict(re_mode = :ebe)` ignored `constants_re` (#251).
- Plotting rejected a bare filename as `save_path`.

## v0.2.4

### Features

- `MCEM` gains `update_schedule` for E-step mini-batching (#233).
- Julia 1.10 (LTS) is supported again, with a 1.10 CI canary (#37).

### Bug fixes

- A large validation round across model construction, fit options, parameter constructors,
  solver config, covariate columns, crossing horizons, cross-validation, accessors,
  plotting, simulation overrides and `compute_uq`: invalid input is now rejected at the
  point of declaration with a specific message rather than surfacing as a numeric failure
  much later (#205 through #223).
- Estimation and cross-validation audit findings (#226, #229) and two further parallel
  bug-fix rounds (#235 through #239, #243 through #250), including vector-valued residual
  and prediction schemas for multivariate outcomes, the HMM value-support and state-dim
  contract, multivariate predictive density bands in `plot_fits`, and
  `NormalizingPlanarFlow` moments.
- Per-component missingness is preserved in multivariate residuals (#248).

### Documentation

- Tutorials for copula random effects and censored outcomes (#203).

### Internal

- Runic replaces JuliaFormatter as the formatting gate.

## v0.2.3

### Features

- Turing.jl is now a weak dependency (#36). `using NoLimits` no longer loads it, cutting
  load time and dependency count; `MCMC`, `VI`, the chain-based UQ refit and the Turing
  E-step samplers require an explicit `import Turing`. `MCEM`'s default E-step is now the
  Turing-free `MCEM_MCMC(sampler = SaemixMH(), sample_schedule = 100)` - pass
  `NUTS(0.75)` / `250` to restore the previous default.
- Random effects can be given a copula distribution from Copulas.jl (#177), loaded via
  the `NoLimitsCopulasExt` extension, with adaptive Gauss-Hermite quadrature support.

### Bug fixes

- `get_marginal_likelihood` omitted the prior density of random-effect levels pinned via
  `constants_re`, so likelihoods were not comparable across models pinning different
  levels (#171).
- `Laplace` (and the FOCEI/AGHQ paths sharing its marginal) stopped ~1900 nats short of
  the optimum on badly scaled data, reporting fixed effects far from the truth while
  `GHQuadrature` on the same data and start landed at it (#157). The admissibility test
  guarding the Laplace expansion rejected any empirical-Bayes Hessian with condition
  number `>= 1e8`, which a single-observation individual at a large time reaches
  legitimately (`-H = xxᵀ/σ² + Ω⁻¹` is well posed but conditioned like `t²`). Since the
  rejection is a hard `-Inf`, the outer optimizer climbed until the worst-conditioned
  individual sat exactly on the threshold and its line search had nowhere left to step.
  The test is now the un-jittered Cholesky itself - a log-det that measures data rather
  than the rescue jitter is kept, whatever its conditioning - which is the criterion the
  adaptive quadrature rule already used.

## v0.2.2

### Bug fixes

- `predict(res, dm_new; re_mode = :reestimate, reestimate_kwargs = (individuals = [...],))`
  leaked the training fit's empirical-Bayes estimates into every individual of `dm_new`
  that was not named in `individuals`, matched purely by batch position (#146). The
  stored-mode merge now applies only when the passed DataModel is the fit's own stored
  one; on any other DataModel, unrequested individuals get the population value
  (random-effect prior mean), so their predictions match `re_mode = :population`.
- `predict(res, dm_new::DataModel)` silently integrated ODEs from `t0 = 0.0` when
  `dm_new` was built without repeating the fit's `t0` (#148). The integration start is
  baked into each individual's time span at DataModel construction, so `predict` now
  raises an informative error on a `t0` mismatch instead of returning wrong numbers;
  the DataFrame path already reused the fit's `t0` (#130) and is unchanged.
- `predict(res, newdata; re_mode = :marginal)` threw an uninformative `BoundsError` on
  models with crossed random-effect groups whenever a grouping column varies within an
  individual (#152). Monte-Carlo draws are now made per free random-effect level and
  assembled per individual with the same shape logic as `re_mode = :population`, which
  also makes individuals sharing a level share its draw within a draw and keeps
  `constants_re`-fixed levels at their fixed values.

## v0.2.1

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
