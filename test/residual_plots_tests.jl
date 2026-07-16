using Test
using NoLimits
using CairoMakie
using DataFrames
using Distributions
using Random

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

    df = DataFrame(ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], z = [0.2, -0.1, 0.3, 0.0],
        y_cont = [0.1, 0.0, 0.2, 0.1], y_bin = [1, 0, 1, 0])
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
        res; mcmc_draws = 3, return_draw_level = true, residuals = [:pit])
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
        res; mcmc_draws = 5, return_draw_level = true, residuals = [:pit])
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
    df = DataFrame(ID = [1, 1, 2, 2, 3, 3], t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.0, 0.4, 0.2, 0.6, 0.8, 1.0], y = [1, 2, 1, 2, 3, 4])
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
        dm; constants_re = fx_varyre_constants_re(), cache_obs_dists = true)
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
            y ~ DiscreteTimeDiscreteStatesHMM([0.8 0.2; 0.3 0.7],
                (Normal(0.0, σh), Normal(μ2, σh)), Categorical([0.6, 0.4]))
        end
    end
    df = DataFrame(ID = [1, 1, 1], t = [0.0, 1.0, 2.0], y = [3.1, 0.05, 2.9])
    dm = DataModel(hmm_model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm,
        NoLimits.MCMC(;
            turing_kwargs = (n_samples = 30, n_adapt = 15, progress = false)))

    dfres = get_residuals(res; residuals = [:logscore], mcmc_draws = 1,
        rng = Xoshiro(11), return_draw_level = true)

    # Replicate the single posterior draw the residual path selects with the
    # same rng, then forward-filter by hand.
    res_use = NoLimits._with_posterior_warmup(res, nothing)
    θd, ηd, _ = NoLimits._posterior_drawn_params(
        res_use, dm, NamedTuple(), NamedTuple(), 1, Xoshiro(11))
    θ = θd[1]
    η_i = ηd[1][1]
    ind = get_individuals(dm)[1]
    rows = NoLimits.get_row_groups(dm).obs_rows[1]
    dists = [calculate_formulas_obs(hmm_model, θ, η_i, ind.const_cov,
                 NoLimits._varying_at(dm, ind, j, rows[j])).y for j in 1:3]
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
    df = DataFrame(ID = repeat([1, 2, 3]; inner = 3),
        t = repeat([0.0, 1.0, 2.0]; outer = 3),
        y = [2.9, 3.1, 3.0, 0.9, 1.1, 1.0, -1.1, -0.9, -1.0])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace())

    # :population — random effect at the prior mean → one prediction level for all rows.
    pop = predict(res, df)
    @test nrow(pop) == nrow(df)
    @test length(unique(round.(pop.prediction; digits = 6))) == 1

    # :ebe — reuse the training EBE → IPRED tracks each subject's level.
    ebe = predict(res, df; re_mode = :ebe)
    @test nrow(ebe) == nrow(df)
    ebe_by_id = combine(groupby(ebe, :id), :prediction => mean => :m)
    @test ebe_by_id.m[1] > ebe_by_id.m[2] > ebe_by_id.m[3]
    @test !isapprox(ebe.prediction[1], pop.prediction[1]; atol = 1e-2)

    # :reestimate on the training data reproduces the stored EBEs.
    reest = predict(res, df; re_mode = :reestimate)
    @test isapprox(collect(reest.prediction), collect(ebe.prediction); atol = 5e-2)

    # :marginal integrates the conditional posterior for seen subjects → tracks :ebe.
    marg = predict(res, df; re_mode = :marginal, marginal_draws = 100,
        rng = MersenneTwister(1))
    @test nrow(marg) == nrow(df)
    @test isapprox(collect(marg.prediction), collect(ebe.prediction); atol = 0.1)

    # Unseen subject with only missing outcomes: rows are kept, and :ebe/:marginal
    # fall back to the population value (prior mean / prior draws).
    df_new = DataFrame(ID = [99, 99], t = [0.0, 1.0], y = [missing, missing])
    pop_new = predict(res, df_new)
    ebe_new = predict(res, df_new; re_mode = :ebe)
    marg_new = predict(res, df_new; re_mode = :marginal, marginal_draws = 100,
        rng = MersenneTwister(2))
    @test nrow(ebe_new) == 2
    @test isapprox(collect(ebe_new.prediction), collect(pop_new.prediction); atol = 1e-8)
    @test isapprox(collect(marg_new.prediction), collect(pop_new.prediction); atol = 0.3)

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
    dm_fo = DataModel(model_fo, DataFrame(ID = [1, 1], t = [0.0, 1.0], y = [1.0, 1.1]);
        primary_id = :ID, time_col = :t)
    res_fo = fit_model(dm_fo, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test_throws ErrorException predict(res_fo, get_df(dm_fo); re_mode = :ebe)
    @test_throws ErrorException predict(res, df; re_mode = :nonsense)
end
