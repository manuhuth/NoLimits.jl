
# Cross-Validation {#Cross-Validation}

Cross-validation (CV) is the standard way to estimate out-of-sample predictive performance. For mixed-effects models the key design choice is how to handle random effects for test individuals: a subject who appeared in training has a posterior `p(b | y_train, θ̂)` to draw from, while a completely held-out subject must fall back on the prior.

NoLimits.jl provides a two-step CV workflow:
1. **Build a split** with [`cross_validate`](/estimation/cv#NoLimits.cross_validate), which returns a [`CVSpec`](/estimation/cv#NoLimits.CVSpec) storing row indices into the original DataFrame (not copies of the data).
  
2. **Fit and evaluate** with [`fit_cv`](/estimation/cv#NoLimits.fit_cv), which trains the model on each fold's training set, predicts on the held-out test set, and aggregates per-observation log-likelihoods.
  

## Split Kinds {#Split-Kinds}

Two splitting strategies are available via the `kind` keyword:

|         `kind` |                                                                                                                                                                                                                        Description |
| --------------:| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|          `:id` |                                                                    Whole individuals are assigned to folds. Test individuals are entirely absent from training — the unseen-individual prediction strategy applies to all of them. |
| `:observation` | Observations from each individual are distributed across folds using a floor/ceiling round-robin. Event rows (dosing, resets) are always included in both train and test, so the ODE can be integrated correctly on either subset. |


`:id`-wise CV measures generalisation to new subjects; `:observation`-wise CV measures how well the model interpolates within a subject's time series given a subset of their observations.

## Random-Effects Prediction Modes {#Random-Effects-Prediction-Modes}

For models with random effects, two separate modes control prediction for seen and unseen test individuals:

**`seen_re_mode`** — individuals who appear in the training set:
- `:ebe` (default) — plug in the empirical Bayes estimate (EBE, MAP of the conditional posterior) obtained from the training fit. Fast and the standard approach in pharmacometrics.
  
- `:conditional` — draw `n_mc_samples` samples from the training conditional posterior `p(b | y_train, θ̂)` using the Laplace approximation (for `Laplace`/`LaplaceMAP`) or MCMC sweeps (for `MCEM`/`SAEM`), then average per-observation log-likelihoods via `logsumexp` and predicted means arithmetically.
  

**`unseen_re_mode`** — individuals absent from training (only possible with `kind=:id`):
- `:mean` (default) — set the random effect to the prior mean (zero for zero-mean priors). This is the marginal population prediction.
  
- `:montecarlo` — draw `n_mc_samples` samples from the RE prior `p(b | θ̂)` and average in the same way as `:conditional`.
  

## Usage {#Usage}

### Step 1 — Build the Split {#Step-1-—-Build-the-Split}

```julia
using NoLimits
using Random

cv = cross_validate(dm, 5; kind=:observation, rng=MersenneTwister(1))
```


### Step 2 — Fit and Evaluate {#Step-2-—-Fit-and-Evaluate}

```julia
res_cv = fit_cv(cv, NoLimits.Laplace())
```


All keyword arguments accepted by [`fit_model`](/api#NoLimits.fit_model) are forwarded to the per-fold fits. CV-specific options are passed as additional keywords:

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


### Step 3 — Inspect Results {#Step-3-—-Inspect-Results}

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


## Result Types {#Result-Types}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.CVSpec' href='#NoLimits.CVSpec'><span class="jlbinding">NoLimits.CVSpec</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CVSpec
```


Stores the fold split configuration for cross-validation. Row indices into the original `DataModel`'s DataFrame are stored rather than full `DataModel` copies to keep memory use low.

Created by [`cross_validate`](/estimation/cv#NoLimits.cross_validate).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/cv.jl#L7-L15" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.CVFoldResult' href='#NoLimits.CVFoldResult'><span class="jlbinding">NoLimits.CVFoldResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CVFoldResult{R}
```


Results for a single cross-validation fold, including per-observation scores on the held-out set and optionally the fitted `FitResult`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/cv.jl#L24-L29" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.CVResult' href='#NoLimits.CVResult'><span class="jlbinding">NoLimits.CVResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CVResult
```


Aggregate cross-validation results. `obs_scores` combines all folds with a `:fold` column. `mean_test_loglikelihood` and `std_test_loglikelihood` summarise predictive performance across folds.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/cv.jl#L37-L43" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Functions {#Functions}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.cross_validate' href='#NoLimits.cross_validate'><span class="jlbinding">NoLimits.cross_validate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
cross_validate(dm::DataModel, n_folds::Int; kind=:id, rng=Random.default_rng())
```


Partition `dm` into `n_folds` train/test splits for cross-validation.
- `kind=:id` — whole individuals are assigned to folds; test individuals are entirely absent from training.
  
- `kind=:observation` — observations from each individual are distributed across folds (floor/ceiling split); training includes all event rows for individuals with any training observations.
  

Returns a [`CVSpec`](/estimation/cv#NoLimits.CVSpec) storing row indices only (not full `DataModel`s).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/cv.jl#L115-L127" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.fit_cv' href='#NoLimits.fit_cv'><span class="jlbinding">NoLimits.fit_cv</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
fit_cv(cv_spec, method, args...;
       seen_re_mode=:ebe, unseen_re_mode=:mean,
       n_mc_samples=100, store_results=false, loss=nothing,
       fold_serialization=EnsembleSerial(), rng=Random.default_rng(),
       constants_re=NamedTuple(), ode_args=(), ode_kwargs=NamedTuple(),
       kwargs...)
```


Fit `method` on each training fold defined by `cv_spec` and evaluate predictive performance on the held-out test set. All `kwargs` are forwarded to [`fit_model`](/api#NoLimits.fit_model).

**Keyword Arguments**
- `seen_re_mode`: prediction strategy for individuals present in the training set.  `:ebe` uses the empirical Bayes estimate (MAP of posterior); `:conditional` integrates over `n_mc_samples` draws from `p(b|y_train, θ̂)`.
  
- `unseen_re_mode`: prediction strategy for individuals absent from training. `:mean` plugs in the RE prior mean (zero for zero-mean priors); `:montecarlo` integrates over `n_mc_samples` draws from the RE prior `p(b|θ̂)`.
  
- `n_mc_samples`: number of MC draws when either mode is `:conditional` or `:montecarlo`.
  
- `store_results`: if `true`, each [`CVFoldResult`](/estimation/cv#NoLimits.CVFoldResult) stores the full `FitResult` from that fold.
  
- `loss`: optional `(dist, y) -> scalar` function. When provided, a `:loss` column is added to `obs_scores`.
  
- `fold_serialization`: controls fold-level parallelism. Use `EnsembleThreads()` to evaluate folds concurrently.
  
- `constants_re`: fix specific RE levels on the natural scale.
  

[`Pooled`](/nlme-methodology#Pooled)/[`PooledMap`](/api#NoLimits.PooledMap) fits evaluate every test individual — seen or unseen — at the deterministic plug-in η computed from that individual's covariates with the strategies resolved by the training fit; `seen_re_mode`/`unseen_re_mode` do not apply and must be left at their defaults.

Returns a [`CVResult`](/estimation/cv#NoLimits.CVResult).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/cv.jl#L575-L610" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Accessors {#Accessors}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_fold_results' href='#NoLimits.get_fold_results'><span class="jlbinding">NoLimits.get_fold_results</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_fold_results(cv_res::CVResult) -> Vector{CVFoldResult}
```


Return the per-fold results stored in `cv_res`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/cv.jl#L54-L58" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_obs_scores' href='#NoLimits.get_obs_scores'><span class="jlbinding">NoLimits.get_obs_scores</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_obs_scores(cv_res::CVResult) -> DataFrame
```


Return the combined per-observation score table from all folds. Contains columns `:fold`, `:individual`, `:time`, `:outcome`, `:obs`, `:loglikelihood`, `:predicted_mean`, and optionally `:loss` when a loss function was supplied.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/cv.jl#L61-L67" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_spec' href='#NoLimits.get_spec'><span class="jlbinding">NoLimits.get_spec</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_spec(cv_res::CVResult) -> CVSpec
```


Return the [`CVSpec`](/estimation/cv#NoLimits.CVSpec) that describes the fold split used to produce `cv_res`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/cv.jl#L70-L74" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Example: Fixed-Effects MLE {#Example:-Fixed-Effects-MLE}

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


## Example: Mixed-Effects Laplace with EBE Prediction {#Example:-Mixed-Effects-Laplace-with-EBE-Prediction}

This example runs 3-fold observation-wise CV. Because all subjects appear in every training fold, the EBE from the training fit is used for every test individual.

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


## Example: ID-Wise CV with Conditional MC for Seen Subjects {#Example:-ID-Wise-CV-with-Conditional-MC-for-Seen-Subjects}

When `kind=:id`, some subjects are entirely absent from training. The defaults use the prior mean for unseen subjects. To instead marginalise over 50 draws from the conditional posterior for seen subjects:

```julia
cv = cross_validate(dm, 3; kind=:id, rng=MersenneTwister(1))

res_cv = fit_cv(cv, NoLimits.Laplace();
    seen_re_mode   = :conditional,
    unseen_re_mode = :mean,
    n_mc_samples   = 50,
    rng            = MersenneTwister(2),
)
```


## Example: Comparing Models with a User-Supplied Loss {#Example:-Comparing-Models-with-a-User-Supplied-Loss}

A custom loss function (here, squared error) is applied to every held-out observation and stored in the `:loss` column of `obs_scores`:

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


## Notes on Computation Time {#Notes-on-Computation-Time}

`fit_cv` runs one full `fit_model` call per fold. With three folds and SAEM, the total computation is approximately three times the cost of a single fit. Use `fold_serialization=EnsembleThreads()` to run folds concurrently if memory allows:

```julia
using SciMLBase: EnsembleThreads

res_cv = fit_cv(cv, NoLimits.SAEM();
    fold_serialization = EnsembleThreads(),
)
```

