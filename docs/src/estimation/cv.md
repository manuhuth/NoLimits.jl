# Cross-Validation

Cross-validation (CV) is the standard way to estimate out-of-sample predictive performance. For mixed-effects models the key design choice is how to handle random effects for test individuals: a subject who appeared in training has a posterior `p(b | y_train, θ̂)` to draw from, while a completely held-out subject must fall back on the prior.

NoLimits.jl provides a two-step CV workflow:

1. **Build a split** with [`cross_validate`](@ref), which returns a [`CVSpec`](@ref) storing row indices into the original DataFrame (not copies of the data).
2. **Fit and evaluate** with [`fit_cv`](@ref), which trains the model on each fold's training set, predicts on the held-out test set, and aggregates per-observation log-likelihoods.

## Split Kinds

Two splitting strategies are available via the `kind` keyword:

| `kind` | Description |
|--------|-------------|
| `:id` | Whole individuals are assigned to folds. Test individuals are entirely absent from training - the unseen-individual prediction strategy applies to all of them. |
| `:observation` | Observations from each individual are distributed across folds using a floor/ceiling round-robin. Event rows (dosing, resets) are always included in both train and test, so the ODE can be integrated correctly on either subset. |

`:id`-wise CV measures generalisation to new subjects; `:observation`-wise CV measures how well the model interpolates within a subject's time series given a subset of their observations.

## Random-Effects Prediction Modes

For models with random effects, two separate modes control prediction for seen and unseen test individuals:

**`seen_re_mode`** - individuals who appear in the training set:

- `:ebe` (default) - plug in the empirical Bayes estimate (EBE, MAP of the conditional posterior) obtained from the training fit. Fast and the standard approach in pharmacometrics.
- `:conditional` - draw `n_mc_samples` samples from the training conditional posterior `p(b | y_train, θ̂)` using the Laplace approximation (for `Laplace`) or MCMC sweeps (for `MCEM`/`SAEM`), then average per-observation log-likelihoods via `logsumexp` and predicted means arithmetically.

**`unseen_re_mode`** - individuals absent from training (only possible with `kind=:id`):

- `:mean` (default) - set the random effect to the prior mean (zero for zero-mean priors). This is the marginal population prediction.
- `:montecarlo` - draw `n_mc_samples` samples from the RE prior `p(b | θ̂)` and average in the same way as `:conditional`.

## Usage

### Step 1 - Build the Split

```julia
using NoLimits
using Random

cv = cross_validate(dm, 5; kind=:observation, rng=MersenneTwister(1))
```

### Step 2 - Fit and Evaluate

```julia
res_cv = fit_cv(cv, NoLimits.Laplace())
```

All keyword arguments accepted by [`fit_model`](@ref) are forwarded to the per-fold fits. CV-specific options are passed as additional keywords:

```julia
res_cv = fit_cv(cv, NoLimits.Laplace();
    seen_re_mode   = :ebe,       # or :conditional
    unseen_re_mode = :mean,      # or :montecarlo
    n_mc_samples   = 100,        # draws used when either mode is MC-based
    store_results  = false,      # set true to keep per-fold FitResult objects
    loss           = nothing,    # optional (dist, y) -> scalar loss
    fold_serialization = EnsembleSerial(),  # or EnsembleThreads()
    rng            = Random.default_rng(),
)
```

### Step 3 - Inspect Results

```julia
# Aggregate statistics
res_cv.mean_test_loglikelihood
res_cv.std_test_loglikelihood

# Per-observation scores (one row per held-out observation)
os = get_obs_scores(res_cv)
# Columns: :fold, :individual, :time, :outcome, :obs, :loglikelihood, :predicted_mean

# Per-fold breakdown
for fr in get_fold_results(res_cv)
    println("Fold $(fr.fold): LL = $(fr.test_loglikelihood)")
end
```

## Result Types

```@docs
CVSpec
CVFoldResult
CVResult
```

## Functions

```@docs
cross_validate
fit_cv
```

## Accessors

```@docs
get_fold_results
get_obs_scores
get_spec
```

## Example: Fixed-Effects MLE

```julia
using NoLimits
using DataFrames
using Distributions
using Random

model = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.5, scale=:log)
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(a, σ)
    end
end

df = DataFrame(
    ID = repeat(1:6, inner=3),
    t  = repeat([0.0, 1.0, 2.0], 6),
    y  = 1.0 .+ 0.1 .* randn(18),
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

# 3-fold observation-wise CV
cv     = cross_validate(dm, 3; kind=:observation, rng=MersenneTwister(1))
res_cv = fit_cv(cv, NoLimits.MLE())

println("Mean test LL: ", res_cv.mean_test_loglikelihood)
println("Std  test LL: ", res_cv.std_test_loglikelihood)
```

## Example: Mixed-Effects Laplace with EBE Prediction

This example runs 3-fold observation-wise CV. Because all subjects appear in
every training fold, the EBE from the training fit is used for every test
individual.

```julia
using NoLimits
using DataFrames
using Distributions
using Random

model = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.5, scale=:log)
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(a + η, σ)
    end
end

df = DataFrame(
    ID = repeat(1:6, inner=3),
    t  = repeat([0.0, 1.0, 2.0], 6),
    y  = 1.0 .+ 0.1 .* randn(18),
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

cv     = cross_validate(dm, 3; kind=:observation, rng=MersenneTwister(1))
res_cv = fit_cv(cv, NoLimits.Laplace();
    seen_re_mode = :ebe,
    rng          = MersenneTwister(2),
)

# Inspect per-observation predictive log-likelihoods
os = get_obs_scores(res_cv)
println(first(os, 5))
```

## Example: ID-Wise CV with Conditional MC for Seen Subjects

When `kind=:id`, some subjects are entirely absent from training. The defaults
use the prior mean for unseen subjects. To instead marginalise over 50 draws
from the conditional posterior for seen subjects:

```julia
cv = cross_validate(dm, 3; kind=:id, rng=MersenneTwister(1))

res_cv = fit_cv(cv, NoLimits.Laplace();
    seen_re_mode   = :conditional,
    unseen_re_mode = :mean,
    n_mc_samples   = 50,
    rng            = MersenneTwister(2),
)
```

## Example: Comparing Models with a User-Supplied Loss

A custom loss function (here, squared error) is applied to every held-out
observation and stored in the `:loss` column of `obs_scores`:

```julia
rmse_loss(dist, y) = (y - mean(dist))^2

cv = cross_validate(dm, 3; kind=:observation, rng=MersenneTwister(1))

res_cv = fit_cv(cv, NoLimits.Laplace();
    seen_re_mode = :ebe,
    loss         = rmse_loss,
)

os = get_obs_scores(res_cv)
println("RMSE: ", sqrt(mean(os.loss)))
```

## Notes on Computation Time

`fit_cv` runs one full `fit_model` call per fold. With three folds and SAEM, the
total computation is approximately three times the cost of a single fit. Use
`fold_serialization=EnsembleThreads()` to run folds concurrently if memory allows:

```julia
using SciMLBase: EnsembleThreads

res_cv = fit_cv(cv, NoLimits.SAEM();
    fold_serialization = EnsembleThreads(),
)
```
