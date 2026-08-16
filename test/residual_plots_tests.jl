using Test
using NoLimits
using CairoMakie
using DataFrames
using Distributions
using Random
import Turing   # MCMC/VI need the Turing extension loaded (#36)

# Note: "residual plots basic API (FitResult + DataModel + cache)"
# has been moved to integration_plotting.jl (shared fixtures).

@testset "residual plots support multiple observables" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.3)
            c = RealNumber(-0.2)
            σ = RealNumber(0.2, scale = :log)
        end
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @formulas begin
            y_cont ~ Normal(a + b * z, σ)
            p = logistic(c + z)
            y_bin ~ Bernoulli(p)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], z = [0.2, -0.1, 0.3, 0.0],
        y_cont = [0.1, 0.0, 0.2, 0.1], y_bin = [1, 0, 1, 0]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))

    rdf = get_residuals(res; residuals = [:pit, :raw], randomize_discrete = false)
    @test nrow(rdf) == 2 * nrow(df)
    @test Set(rdf.observable) == Set([:y_cont, :y_bin])
    @test plot_residual_distribution(res; residual = :pit) !== nothing
    @test plot_residual_pit(res; show_hist = false, show_kde = true, show_qq = false) !==
        nothing
end

@testset "residuals with constants_re inherited from fit result" begin
    res = fx_constre_laplace()

    @test nrow(get_residuals(res)) == nrow(fx_constre_df())
    @test plot_residuals(res) !== nothing
end

# Regression for #106: string-valued RE levels (Symbol keys in constants_re) broke
# every downstream consumer of a constants_re fit; new data lacking the level must
# ignore that constant rather than error.
@testset "constants_re with string levels: residuals and predict" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.5); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η + 0.2 * t, σ)
        end
    end
    df = DataFrame(
        ID = repeat(["id_001", "id_002", "id_003"], inner = 2),
        t = repeat([0.0, 1.0], outer = 3),
        y = [0.1, 0.3, 0.0, 0.25, 0.15, 0.35]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(
        dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
        constants_re = (; η = (; id_001 = 0.6)),
        serialization = NoLimits.EnsembleSerial()
    )

    η_df = get_random_effects(res).η
    @test η_df[η_df.ID .== "id_001", :η_1][1] == 0.6
    @test nrow(get_residuals(res)) == nrow(df)

    df_wo = df[df.ID .!= "id_001", :]
    for mode in (:population, :ebe, :reestimate, :marginal)
        @test nrow(NoLimits.predict(res, df; re_mode = mode)) == nrow(df)
        # The pinned level is absent here — it must be ignored, not fatal.
        @test nrow(NoLimits.predict(res, df_wo; re_mode = mode)) == nrow(df_wo)
    end

    # Regression for #147: :population must apply the fit's own constants_re, not just
    # :ebe. Before the fix it fell back to the RE's prior mean (0.0) for id_001 instead
    # of the pinned 0.6, diverging from :ebe by exactly that offset.
    pop_df = NoLimits.predict(res, df; re_mode = :population)
    ebe_df = NoLimits.predict(res, df; re_mode = :ebe)
    @test isapprox(
        pop_df.prediction[pop_df.id .== "id_001"],
        ebe_df.prediction[ebe_df.id .== "id_001"]; atol = 1.0e-8
    )
end

@testset "residuals MCMC summary and draw-level outputs" begin
    res = fx_mcmc()                       # shared no-RE MCMC fit
    df = fx_nore_df()

    rdf = get_residuals(res; mcmc_draws = 5, mcmc_quantiles = [10, 90])
    @test nrow(rdf) == nrow(df)
    @test all(rdf.n_draws .== 5)
    @test all(ismissing.(rdf.draw))
    @test all(.!ismissing.(rdf.pit_qlo))
    @test all(.!ismissing.(rdf.pit_qhi))

    rdf_draw = get_residuals(
        res; mcmc_draws = 3, return_draw_level = true, residuals = [:pit]
    )
    @test nrow(rdf_draw) == 3 * nrow(df)
    @test all(rdf_draw.n_draws .== 3)
    @test all(.!ismissing.(rdf_draw.draw))
    @test plot_residual_qq(res; mcmc_draws = 3) !== nothing
end

@testset "residuals VI summary and draw-level outputs" begin
    res = fx_vi()                         # shared no-RE VI fit
    df = fx_nore_df()

    rdf = get_residuals(res; mcmc_draws = 5, mcmc_quantiles = [10, 90])
    @test nrow(rdf) == nrow(df)
    @test all(rdf.n_draws .== 5)
    @test all(ismissing.(rdf.draw))
    @test all(.!ismissing.(rdf.pit_qlo))
    @test all(.!ismissing.(rdf.pit_qhi))

    rdf_draw = get_residuals(
        res; mcmc_draws = 5, return_draw_level = true, residuals = [:pit]
    )
    @test nrow(rdf_draw) == 5 * nrow(df)
    @test all(rdf_draw.n_draws .== 5)
    @test all(.!ismissing.(rdf_draw.draw))
    @test plot_residual_qq(res; mcmc_draws = 5) !== nothing
end

@testset "residual API validation errors" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.2, scale = :log)
        end
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @formulas begin
            y ~ Normal(a + z, σ)
        end
    end
    df = DataFrame(ID = [1, 1], t = [0.0, 1.0], z = [0.1, 0.2], y = [0.1, 0.2])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test_throws ErrorException get_residuals(res; residuals = [:not_a_metric])
    @test_throws ErrorException plot_residuals(res; residual = :not_a_metric)
    @test_throws ErrorException get_residuals(res; x_axis_feature = :missing_feature)
    @test_throws ErrorException plot_residual_acf(res; max_lag = 0)
    @test_throws ErrorException get_residuals(res; mcmc_quantiles = [-5, 95])
end

@testset "residual plots Poisson outcome" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0)
            b = RealNumber(0.3)
        end
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @formulas begin
            λ = exp(a + b * z)
            y ~ Poisson(λ)
        end
    end
    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3], t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.0, 0.4, 0.2, 0.6, 0.8, 1.0], y = [1, 2, 1, 2, 3, 4]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))

    rdf = get_residuals(res; residuals = [:pit, :raw, :pearson], randomize_discrete = true)
    @test nrow(rdf) == nrow(df)
    @test all(.!ismissing.(rdf.res_raw))
    @test all(.!ismissing.(rdf.res_pearson))
    @test plot_residual_distribution(res; residual = :pit) !== nothing
    @test plot_residuals(res; residual = :pearson) !== nothing
    @test plot_residual_pit(res; show_hist = true, show_kde = false, show_qq = true) !==
        nothing
end

@testset "residuals use row-specific random effects for varying non-ODE groups" begin
    dm = fx_varyre_dm()
    cache = build_plot_cache(
        dm; constants_re = fx_varyre_constants_re(), cache_obs_dists = true
    )
    rdf = get_residuals(dm; cache = cache, cache_obs_dists = true, residuals = [:raw])
    sort!(rdf, :row)
    @test Float64.(rdf.fitted) ≈ [0.1, 0.4, 0.4, 0.1, 0.3]
    @test maximum(abs.(Float64.(rdf.res_raw))) < 1.0e-6
end

@testset "MCMC residuals apply HMM forward filtering" begin
    hmm_model = @Model begin
        @fixedEffects begin
            μ2 = RealNumber(3.0, prior = Normal(3.0, 0.5))
            σh = RealNumber(0.5, scale = :log, prior = LogNormal(-0.7, 0.3))
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ DiscreteTimeDiscreteStatesHMM(
                [0.8 0.2; 0.3 0.7],
                (Normal(0.0, σh), Normal(μ2, σh)), Categorical([0.6, 0.4])
            )
        end
    end
    df = DataFrame(ID = [1, 1, 1], t = [0.0, 1.0, 2.0], y = [3.1, 0.05, 2.9])
    dm = DataModel(hmm_model, df; primary_id = :ID, time_col = :t)
    res = fit_model(
        dm,
        NoLimits.MCMC(;
            turing_kwargs = (n_samples = 30, n_adapt = 15, progress = false)
        )
    )

    dfres = get_residuals(
        res; residuals = [:logscore], mcmc_draws = 1,
        rng = Xoshiro(11), return_draw_level = true
    )

    # Replicate the single posterior draw the residual path selects with the
    # same rng, then forward-filter by hand.
    res_use = NoLimits._with_posterior_warmup(res, nothing)
    θd, ηd, _ = NoLimits._posterior_drawn_params(
        res_use, dm, NamedTuple(), NamedTuple(), 1, Xoshiro(11)
    )
    θ = θd[1]
    η_i = ηd[1][1]
    ind = get_individuals(dm)[1]
    rows = NoLimits.get_row_groups(dm).obs_rows[1]
    dists = [
        calculate_formulas_obs(
                hmm_model, θ, η_i, ind.const_cov,
                NoLimits._varying_at(dm, ind, j, rows[j])
            ).y for j in 1:3
    ]
    y = df.y
    post1 = posterior_hidden_states(dists[1], y[1])
    d2f = NoLimits._hmm_with_prior(dists[2], post1)
    post2 = posterior_hidden_states(d2f, y[2])
    d3f = NoLimits._hmm_with_prior(dists[3], post2)

    ls = sort(dfres, :obs_index).logscore
    @test ls[1] ≈ -logpdf(dists[1], y[1])
    @test ls[2] ≈ -logpdf(d2f, y[2])
    @test ls[3] ≈ -logpdf(d3f, y[3])
    # Guard that filtering actually matters for this data (keeps the test sharp).
    @test !(ls[2] ≈ -logpdf(dists[2], y[2]))
end

@testset "HMM logscore sum matches the conditional loglik (missing rows)" begin
    model = @Model begin
        @fixedEffects begin
            P = DiscreteTransitionMatrix([0.8 0.2; 0.3 0.7])
            π0 = ProbabilityVector([0.5, 0.5])
            μ = RealVector([-0.5, 1.5])
            σk = RealVector([0.7, 0.7]; scale = [:log, :log])
            ω = RealNumber(0.5; scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @formulas begin
            y ~ DiscreteTimeDiscreteStatesHMM(
                P,
                (Normal(μ[1] + η, σk[1]), Normal(μ[2] + η, σk[2])),
                Categorical(π0)
            )
        end
    end

    rng = Xoshiro(20260812)
    P_true = [0.9 0.1; 0.2 0.8]
    rows = NamedTuple[]
    for id in 1:4
        η = 0.3 * randn(rng)
        miss = shuffle(rng, collect(2:8))[1:2]     # missing mid-sequence, never first
        s = rand(rng, Distributions.Categorical([0.7, 0.3]))
        for k in 1:8
            s = rand(rng, Distributions.Categorical(P_true[s, :]))
            y = k in miss ? missing : [-1.0, 2.0][s] + η + [0.5, 0.6][s] * randn(rng)
            push!(rows, (; ID = "id_$id", t = Float64(k), y = y))
        end
    end
    df = DataFrame(rows)
    @test count(ismissing, df.y) == 8

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)))

    rdf = get_residuals(res; residuals = [:logscore])
    ls = sum(skipmissing(rdf.logscore))
    # logscore is the NEGATIVE filtered log predictive density, so its sum is the
    # conditional loglik at the EB modes with the sign flipped.
    @test isapprox(-ls, get_loglikelihood(res); atol = 1.0e-6)
    # Sharpness: unfiltered scoring would be a different number entirely.
    cache = build_plot_cache(res)
    θ = cache.params
    η_ind = cache.random_effects[1]
    ind = get_individuals(dm)[1]
    obs_rows = NoLimits.get_row_groups(dm).obs_rows[1]
    y1 = get_obs(get_series(ind)).y
    unfiltered = sum(
        logpdf(
                calculate_formulas_obs(
                    model, θ, η_ind, ind.const_cov,
                    NoLimits._varying_at(dm, ind, j, obs_rows[j])
                ).y, y1[j]
            )
            for j in eachindex(obs_rows) if y1[j] !== missing
    )
    ls1 = sum(skipmissing(rdf[rdf.individual_idx .== 1, :logscore]))
    @test !isapprox(-ls1, unfiltered; atol = 1.0e-3)
end

@testset "GOF and diagnostic plots (Laplace RE fit)" begin
    # Moved from coverage_gap_tests.jl (path coverage for GOF/diagnostic plots).
    dm = fx_fixre_dm()
    res = fx_fixre_laplace()

    @test plot_dv_pred(res) !== nothing
    @test plot_dv_ipred(res) !== nothing
    @test plot_wres_pred(res) !== nothing
    @test plot_shrinkage(res) !== nothing
    @test plot_observed_profiles(res) !== nothing
    @test plot_observed_profiles(dm) !== nothing

    # compute_shrinkage is the data path behind plot_shrinkage
    shrink = NoLimits.compute_shrinkage(res)
    @test haskey(shrink, :η)
    @test isfinite(shrink.η.shrinkage)
end

@testset "compute_shrinkage skips planar-flow REs instead of crashing (#109)" begin
    shrink = @test_logs (:warn, r"no analytic mean") NoLimits.compute_shrinkage(fx_npf_laplace())
    @test isempty(shrink)
    @test_throws ErrorException plot_shrinkage(fx_npf_laplace())
end

@testset "predict re_mode (population/ebe/reestimate/marginal)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.4, scale = :log)
            ω = RealNumber(0.7, scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    # Three subjects with clearly separated levels so the EBEs are non-zero.
    df = DataFrame(
        ID = repeat([1, 2, 3]; inner = 3),
        t = repeat([0.0, 1.0, 2.0]; outer = 3),
        y = [2.9, 3.1, 3.0, 0.9, 1.1, 1.0, -1.1, -0.9, -1.0]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace())

    # :population — random effect at the prior mean → one prediction level for all rows.
    pop = NoLimits.predict(res, df)
    @test nrow(pop) == nrow(df)
    @test length(unique(round.(pop.prediction; digits = 6))) == 1

    # :ebe — reuse the training EBE → IPRED tracks each subject's level.
    ebe = NoLimits.predict(res, df; re_mode = :ebe)
    @test nrow(ebe) == nrow(df)
    ebe_by_id = combine(groupby(ebe, :id), :prediction => mean => :m)
    @test ebe_by_id.m[1] > ebe_by_id.m[2] > ebe_by_id.m[3]
    @test !isapprox(ebe.prediction[1], pop.prediction[1]; atol = 1.0e-2)

    # :reestimate on the training data reproduces the stored EBEs.
    reest = NoLimits.predict(res, df; re_mode = :reestimate)
    @test isapprox(collect(reest.prediction), collect(ebe.prediction); atol = 5.0e-2)

    # :marginal integrates the RE prior, so on a linear mean-zero-RE model it matches
    # :population up to Monte-Carlo error that shrinks with marginal_draws (issue #103).
    marg = NoLimits.predict(
        res, df; re_mode = :marginal, marginal_draws = 800,
        rng = MersenneTwister(1)
    )
    marg_few = NoLimits.predict(
        res, df; re_mode = :marginal, marginal_draws = 25,
        rng = MersenneTwister(1)
    )
    dev = maximum(abs.(collect(marg.prediction) .- collect(pop.prediction)))
    dev_few = maximum(abs.(collect(marg_few.prediction) .- collect(pop.prediction)))
    @test nrow(marg) == nrow(df)
    @test dev < 0.3
    @test dev < dev_few
    @test !isapprox(collect(marg.prediction), collect(ebe.prediction); atol = 0.5)

    # Unseen subject with only missing outcomes: rows are kept, and :ebe/:marginal
    # fall back to the population value (prior mean / prior draws).
    df_new = DataFrame(ID = [99, 99], t = [0.0, 1.0], y = [missing, missing])
    pop_new = NoLimits.predict(res, df_new)
    ebe_new = NoLimits.predict(res, df_new; re_mode = :ebe)
    marg_new = NoLimits.predict(
        res, df_new; re_mode = :marginal, marginal_draws = 100,
        rng = MersenneTwister(2)
    )
    @test nrow(ebe_new) == 2
    @test isapprox(collect(ebe_new.prediction), collect(pop_new.prediction); atol = 1.0e-8)
    @test isapprox(collect(marg_new.prediction), collect(pop_new.prediction); atol = 0.3)

    # RE distributions without an analytic mean used to fall back to a hard zero (#175).
    @test NoLimits._mc_mean(Normal(3.0, 1.0), 1) ≈ 3.0 atol = 0.1
    @test NoLimits._mc_mean(MvNormal([2.0, -1.0], [1.0 0.0; 0.0 1.0]), 2) ≈ [2.0, -1.0] atol = 0.1

    # Unsupported combinations error clearly.
    model_fo = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.4, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end
    dm_fo = DataModel(
        model_fo, DataFrame(ID = [1, 1], t = [0.0, 1.0], y = [1.0, 1.1]);
        primary_id = :ID, time_col = :t
    )
    res_fo = fit_model(dm_fo, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test_throws ErrorException NoLimits.predict(res_fo, get_df(dm_fo); re_mode = :ebe)
    @test_throws ErrorException NoLimits.predict(res, df; re_mode = :nonsense)
end

@testset "predict on new data inherits the DataModel t0" begin
    model = @Model begin
        @fixedEffects begin
            A = RealNumber(2.0)
            k = RealNumber(0.2, scale = :log)
            σ = RealNumber(0.1, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @DifferentialEquation begin
            D(x1) ~ -k * x1
        end
        @initialDE begin
            x1 = A
        end
        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    # First observation well after 0 so t0 shifts the integration start.
    df = DataFrame(
        ID = repeat([1, 2]; inner = 3),
        t = repeat([5.0, 6.0, 7.0]; outer = 2),
        y = [0.75, 0.62, 0.5, 0.7, 0.58, 0.47]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t, t0 = nothing)
    @test get_t0(dm) === nothing
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 5,)))

    in_sample = NoLimits.predict(res, dm)
    on_newdata = NoLimits.predict(res, df)
    @test isapprox(
        collect(on_newdata.prediction), collect(in_sample.prediction);
        atol = 1.0e-8
    )

    # #148: a manually built DataModel must carry the fit's t0 — the silent default
    # t0 = 0.0 would integrate from the wrong start.
    dm5 = DataModel(model, df; primary_id = :ID, time_col = :t, t0 = 5.0)
    res5 = fit_model(dm5, NoLimits.MLE(; optim_kwargs = (maxiters = 5,)))
    dm_wrong = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test_throws "t0 = 5.0" NoLimits.predict(res5, dm_wrong)
    dm_match = DataModel(model, df; primary_id = :ID, time_col = :t, t0 = 5.0)
    pred_dm = NoLimits.predict(res5, dm_match)
    pred_df = NoLimits.predict(res5, df)
    @test isapprox(collect(pred_dm.prediction), collect(pred_df.prediction); atol = 1.0e-8)
end

@testset "reestimate on a new DataModel never leaks training EBEs (#146)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.4, scale = :log)
            ω = RealNumber(0.7, scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(
        ID = repeat([1, 2, 3]; inner = 3),
        t = repeat([0.0, 1.0, 2.0]; outer = 3),
        y = [2.9, 3.1, 3.0, 0.9, 1.1, 1.0, -1.1, -0.9, -1.0]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace())

    # New individuals, same count as training, so the old positional merge would
    # have copied training EBEs into IDs 12/13 by batch slot.
    df_new = DataFrame(
        ID = repeat([11, 12, 13]; inner = 3),
        t = repeat([0.0, 1.0, 2.0]; outer = 3),
        y = [2.0, 2.2, 2.1, missing, missing, missing, missing, missing, missing]
    )
    dm_new = DataModel(model, df_new; primary_id = :ID, time_col = :t)
    reest = NoLimits.predict(
        res, dm_new; re_mode = :reestimate,
        reestimate_kwargs = (individuals = [11],)
    )
    pop = NoLimits.predict(res, dm_new)
    for id in (12, 13)
        @test isapprox(
            collect(reest[reest.id .== id, :prediction]),
            collect(pop[pop.id .== id, :prediction]); atol = 1.0e-8
        )
    end
    @test !isapprox(
        collect(reest[reest.id .== 11, :prediction]),
        collect(pop[pop.id .== 11, :prediction]); atol = 1.0e-3
    )

    # Same-dm partial reestimate still keeps the stored modes for unrequested batches.
    res2 = NoLimits.reestimate_ebes(res; individuals = [1])
    ebe_before = NoLimits.get_random_effects(res)
    ebe_after = NoLimits.get_random_effects(res2)
    @test isapprox(
        Matrix(ebe_before.η[2:3, 2:end]), Matrix(ebe_after.η[2:3, 2:end]);
        atol = 1.0e-12
    )
end

@testset "predict marginal on crossed RE groups (#152)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            b = RealNumber(0.5)
            ω_id = RealNumber(0.5, scale = :log)
            ω_r = RealNumber(0.5, scale = :log)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, ω_id); column = :ID)
            η_rater = RandomEffect(Normal(0.0, ω_r); column = :RATER)
        end
        @formulas begin
            y ~ Normal(a + b * t + η_id + η_rater, σ)
        end
    end
    # The rater rotates within each subject, so RATER varies within ID and each
    # subject's η_rater needs one entry per level it sees.
    raters = [:R1, :R2, :R3]
    mkdf(ids) = DataFrame(
        ID = repeat(ids, inner = 3),
        RATER = [raters[mod(i + j, 3) + 1] for i in eachindex(ids) for j in 1:3],
        t = repeat([0.0, 1.0, 2.0], length(ids)),
        y = [1.0 + 0.5 * j + 0.1 * i for i in eachindex(ids) for j in 1:3]
    )
    df = mkdf(["id_$k" for k in 1:6])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(optim_kwargs = (maxiters = 10,)))

    holdout = mkdf(["new_$k" for k in 1:3])
    holdout.y = fill(missing, nrow(holdout))
    pred = NoLimits.predict(
        res, holdout; re_mode = :marginal, marginal_draws = 10,
        rng = MersenneTwister(3)
    )
    @test nrow(pred) == nrow(holdout)
    @test !any(ismissing, pred.prediction)
    @test all(isfinite, collect(pred.prediction))
    # Mean-zero REs enter the mean linearly, so marginal ≈ population up to MC error.
    pop = NoLimits.predict(res, holdout)
    @test isapprox(collect(pred.prediction), collect(pop.prediction); atol = 1.0)
end

@testset "residual expansion rejects mismatched component counts (#250.5)" begin
    _ext = Base.get_extension(NoLimits, :NoLimitsMakieExt)
    ok = DataFrame(
        observable = [:y], individual_idx = [1], x = [0.0],
        y = [[1.0, 2.0, 3.0]], fitted = [[1.0, 2.0, 3.0]], raw = [[0.0, 0.0, 0.0]]
    )
    @test nrow(_ext._expand_residual_components(ok, :raw)) == 3
    bad = copy(ok)
    bad.fitted = [[1.0, 2.0]]
    err = try
        _ext._expand_residual_components(bad, :raw)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("fitted", err.msg) && occursin("must agree in length", err.msg)
end

# Issue #250 finding 1: a zero-probability observation is real information.
@testset "logscore keeps impossible observations as Inf" begin
    rng = Random.default_rng()
    uni = NoLimits._compute_residual_metrics(
        Uniform(0.0, 1.0), 5.0, [:logscore], mean, true, 0, rng
    )
    @test uni.logscore == Inf
    mv = NoLimits._compute_residual_metrics(
        product_distribution([Uniform(0.0, 1.0), Uniform(0.0, 1.0)]),
        [5.0, 0.5], [:logscore], mean, true, 0, rng
    )
    @test mv.logscore == Inf
end
