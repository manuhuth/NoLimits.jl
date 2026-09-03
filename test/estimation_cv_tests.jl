using Test
using DataFrames
using NoLimits
using Distributions
using Random

# ── Shared fixtures ────────────────────────────────────────────────────────────

function _make_mle_model()
    return @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end
end

function _make_re_model()
    return @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
end

# 6 individuals, 3 obs each — enough for 3-fold CV
function _make_df()
    ids = repeat(1:6, inner = 3)
    ts = repeat([0.0, 1.0, 2.0], 6)
    ys = 1.0 .+ 0.1 .* randn(MersenneTwister(42), 18)
    return DataFrame(ID = ids, t = ts, y = ys)
end

# ── Tests ──────────────────────────────────────────────────────────────────────

@testset "cross_validate id-wise: structure checks" begin
    model = _make_mle_model()
    df = _make_df()
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    cv = cross_validate(dm, 3; kind = :id, rng = MersenneTwister(1))

    @test cv isa CVSpec
    @test cv.n_folds == 3
    @test cv.kind == :id
    @test length(cv.train_rows) == 3
    @test length(cv.test_rows) == 3

    n_rows = nrow(df)
    for f in 1:3
        # train + test covers all rows
        combined = sort(union(cv.train_rows[f], cv.test_rows[f]))
        @test combined == sort(unique(vcat(cv.train_rows[f], cv.test_rows[f])))
        # train and test are disjoint
        @test isempty(intersect(cv.train_rows[f], cv.test_rows[f]))
        # test rows are a subset of all rows
        @test all(r -> 1 <= r <= n_rows, cv.test_rows[f])
        @test all(r -> 1 <= r <= n_rows, cv.train_rows[f])
    end
    # Every individual appears in test exactly once across folds
    all_test = vcat(cv.test_rows...)
    @test length(all_test) == n_rows
    @test sort(all_test) == 1:n_rows

    # kind also accepts the string spelling (#255).
    cv_str = cross_validate(dm, 3; kind = "id", rng = MersenneTwister(1))
    @test cv_str.kind === cv.kind
    @test cv_str.test_rows == cv.test_rows
    @test_throws ErrorException cross_validate(dm, 3; kind = "identifier")
end

@testset "cross_validate observation-wise: structure checks" begin
    model = _make_mle_model()
    df = _make_df()
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    cv = cross_validate(dm, 3; kind = :observation, rng = MersenneTwister(2))

    @test cv.kind == :observation
    for f in 1:3
        @test !isempty(cv.train_rows[f])
        @test !isempty(cv.test_rows[f])
        # Rows are sorted
        @test cv.train_rows[f] == sort(cv.train_rows[f])
        @test cv.test_rows[f] == sort(cv.test_rows[f])
    end
    # Every observation row appears in at least one test fold
    all_test_obs = sort(unique(vcat(cv.test_rows...)))
    @test all_test_obs == 1:nrow(df)
end

@testset "fit_cv id-wise + MLE: basic shape and finite LL" begin
    model = _make_mle_model()
    df = _make_df()
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    cv = cross_validate(dm, 3; kind = :id, rng = MersenneTwister(10))
    res = fit_cv(cv, NoLimits.MLE(); rng = MersenneTwister(10))

    @test res isa CVResult
    @test isfinite(res.mean_test_loglikelihood)
    @test length(res.fold_results) == 3
    @test all(fr -> isfinite(fr.test_loglikelihood), res.fold_results)

    os = res.obs_scores
    @test os isa DataFrame
    expected_cols = [
        :fold, :individual, :time, :outcome, :obs, :loglikelihood, :predicted_mean,
    ]
    @test all(c -> c ∈ names(os), string.(expected_cols))
    @test nrow(os) == nrow(df)   # one row per observation
    @test all(isfinite, skipmissing(os[!, :loglikelihood]))
    @test all(isfinite, skipmissing(os[!, :predicted_mean]))

    # seen_re_mode / unseen_re_mode also accept strings (#255).
    res_str = fit_cv(
        cv, NoLimits.MLE(); seen_re_mode = "ebe", unseen_re_mode = "mean",
        rng = MersenneTwister(10)
    )
    @test res_str.mean_test_loglikelihood == res.mean_test_loglikelihood
    @test_throws ErrorException fit_cv(cv, NoLimits.MLE(); seen_re_mode = "ebes")
    @test_throws ErrorException fit_cv(cv, NoLimits.MLE(); unseen_re_mode = "means")
end

@testset "fit_cv id-wise + Laplace seen_re_mode=:ebe: runs without error" begin
    model = _make_re_model()
    df = _make_df()
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    cv = cross_validate(dm, 3; kind = :id, rng = MersenneTwister(20))
    res = fit_cv(
        cv, NoLimits.Laplace(); seen_re_mode = :ebe, unseen_re_mode = :mean,
        rng = MersenneTwister(20)
    )

    @test res isa CVResult
    @test isfinite(res.mean_test_loglikelihood)
    os = res.obs_scores
    @test nrow(os) == nrow(df)
    @test all(f -> f ∈ 1:3, os[!, :fold])
end

# Observation folds split each individual's observations, so the individual count is not
# the binding constraint — only observations per individual are (#229).
@testset "cross_validate observation folds are limited by observations, not individuals" begin
    model = _make_mle_model()
    df = DataFrame(ID = repeat(1:2; inner = 5), t = repeat(1.0:5.0, 2), y = randn(MersenneTwister(3), 10))
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    cv = cross_validate(dm, 4; kind = :observation, rng = MersenneTwister(4))
    @test length(cv.test_rows) == 4
    @test all(!isempty, cv.test_rows)
    # More folds than any individual has observations would leave folds empty.
    @test_throws ErrorException cross_validate(dm, 6; kind = :observation)
    # id folds keep the individual-count requirement.
    @test_throws ErrorException cross_validate(dm, 3; kind = :id)
end

@testset "fit_cv observation-wise + MLE: all individuals in both train and test" begin
    model = _make_mle_model()
    df = _make_df()
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    cv = cross_validate(dm, 3; kind = :observation, rng = MersenneTwister(30))
    res = fit_cv(cv, NoLimits.MLE(); rng = MersenneTwister(30))

    @test res isa CVResult
    @test isfinite(res.mean_test_loglikelihood)

    # Every individual should appear in obs_scores (each has 3 obs, so each
    # contributes to at least one test fold)
    os = res.obs_scores
    ids_in_scores = unique(os[!, :individual])
    @test length(ids_in_scores) == 6
end

@testset "fit_cv store_results=true: fold result holds FitResult" begin
    model = _make_mle_model()
    df = _make_df()
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    cv = cross_validate(dm, 2; kind = :id, rng = MersenneTwister(40))
    res = fit_cv(cv, NoLimits.MLE(); store_results = true, rng = MersenneTwister(40))

    @test res isa CVResult
    for fr in res.fold_results
        @test fr.fit_result isa FitResult
    end
end

@testset "fit_cv store_results=false: fold result has nothing" begin
    model = _make_mle_model()
    df = _make_df()
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    cv = cross_validate(dm, 2; kind = :id, rng = MersenneTwister(50))
    res = fit_cv(cv, NoLimits.MLE(); store_results = false, rng = MersenneTwister(50))

    for fr in res.fold_results
        @test fr.fit_result === nothing
    end
end

@testset "fit_cv user-supplied loss: :loss column appears in obs_scores" begin
    model = _make_mle_model()
    df = _make_df()
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    my_loss(dist, y) = (y - mean(dist))^2   # squared error

    cv = cross_validate(dm, 2; kind = :id, rng = MersenneTwister(60))
    res = fit_cv(cv, NoLimits.MLE(); loss = my_loss, rng = MersenneTwister(60))

    os = res.obs_scores
    @test "loss" ∈ names(os)
    @test eltype(os[!, :loss]) <: Real
    @test all(l -> !isnan(l), os[!, :loss])
end

@testset "fit_cv Laplace seen_re_mode=:conditional: finite LL, per-obs sums match total" begin
    model = _make_re_model()
    df = _make_df()
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    cv = cross_validate(dm, 2; kind = :id, rng = MersenneTwister(70))
    res = fit_cv(
        cv, NoLimits.Laplace();
        seen_re_mode = :conditional, n_mc_samples = 20,
        rng = MersenneTwister(70)
    )

    @test res isa CVResult
    @test isfinite(res.mean_test_loglikelihood)

    os = res.obs_scores
    @test all(isfinite, os[!, :loglikelihood])

    # Per-fold: sum of per-obs log-likelihoods should match stored test_loglikelihood
    for fr in res.fold_results
        fold_os = fr.obs_scores
        @test isapprox(sum(fold_os[!, :loglikelihood]), fr.test_loglikelihood; atol = 1.0e-8)
    end
end

@testset "fit_cv accessors" begin
    model = _make_mle_model()
    df = _make_df()
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    cv = cross_validate(dm, 2; kind = :id, rng = MersenneTwister(80))
    res = fit_cv(cv, NoLimits.MLE(); rng = MersenneTwister(80))

    @test get_spec(res) === res.spec
    @test get_fold_results(res) === res.fold_results
    @test get_obs_scores(res) === res.obs_scores
    # Per-observation summary is the aggregate over scored rows, not a mean of fold
    # totals (which weights a small fold like a large one, #226).
    @test get_n_scored_obs(res) == nrow(res.obs_scores)
    @test isapprox(
        get_mean_obs_loglikelihood(res),
        sum(res.obs_scores[!, :loglikelihood]) / nrow(res.obs_scores); atol = 1.0e-10
    )
end

# The observation-fold test scores must carry the HMM filter through the training
# observations that precede them; scoring them as sequence starts is wrong (#226).
@testset "fit_cv observation-wise + HMM: held-out rows keep their filter history" begin
    model = @Model begin
        @fixedEffects begin
            dummy = RealNumber(0.0)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ DiscreteTimeDiscreteStatesHMM(
                [0.9 0.1; 0.1 0.9],
                (Categorical([1.0, 0.0]), Categorical([0.0, 1.0])),
                Categorical([0.5, 0.5])
            )
        end
    end
    # Observations reveal the hidden state exactly, so the predictive density of each
    # row given its history is known in closed form: 0.5, then 0.9 (stay), then 0.1 (move).
    df = DataFrame(
        ID = repeat(1:3, inner = 3),
        t = repeat([0.0, 1.0, 2.0], 3),
        y = repeat([1, 1, 2], 3)
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    cv = cross_validate(dm, 3; kind = :observation, rng = MersenneTwister(90))
    res = fit_cv(cv, NoLimits.MLE(; optim_kwargs = (; iterations = 1)))

    # Each row is held out exactly once, and the likelihood does not depend on `dummy`,
    # so the scores must add up to the full-data loglikelihood.
    expected = 3 * (log(0.5) + log(0.9) + log(0.1))
    @test nrow(res.obs_scores) == nrow(df)
    @test isapprox(sum(res.obs_scores[!, :loglikelihood]), expected; atol = 1.0e-8)
end

@testset "constants_re method gate covers all RE-aware optimizers" begin
    @test NoLimits._cv_method_accepts_constants_re(NoLimits.Laplace())
    @test NoLimits._cv_method_accepts_constants_re(NoLimits.FOCEI())
    @test NoLimits._cv_method_accepts_constants_re(NoLimits.GHQuadrature())
    @test NoLimits._cv_method_accepts_constants_re(NoLimits.SAEM())
    @test !NoLimits._cv_method_accepts_constants_re(NoLimits.MLE())
end

@testset "fit_cv scores impossible held-out observations as -Inf (#249)" begin
    # A model that rules an observation out must lose, not win: the non-HMM scorer used
    # to map -Inf to NaN, which aggregation dropped, making the fit look perfect.
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            p = 0.0 * a
            y ~ Bernoulli(p)
        end
    end
    df = DataFrame(
        ID = repeat(1:4, inner = 2), t = repeat([0.0, 1.0], 4),
        y = [0, 0, 0, 0, 0, 1, 0, 0]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    cv = cross_validate(dm, 2; kind = :id, rng = MersenneTwister(1))
    res = fit_cv(cv, NoLimits.MLE(; optim_kwargs = (; maxiters = 1)); rng = MersenneTwister(2))

    lls = res.obs_scores[!, :loglikelihood]
    @test !any(isnan, lls)
    @test count(==(-Inf), lls) == 1
    @test length(lls) == 8
    @test res.n_scored_obs == 8
    @test res.mean_obs_loglikelihood == -Inf
    @test any(==(-Inf), [fr.test_loglikelihood for fr in res.fold_results])
end

# t0 = nothing lets every fold re-derive the integration start from its own rows, so an
# observation split would fit and score with different initial-condition times (#304).
@testset "cross_validate observation-wise: ODE with t0 = nothing is refused" begin
    dm = DataModel(
        fx_ode_model(), fx_ode_df(); primary_id = :ID, time_col = :t, t0 = nothing
    )
    @test_throws ErrorException cross_validate(dm, 2; kind = :observation)
    @test cross_validate(dm, 2; kind = :id) isa CVSpec
end
