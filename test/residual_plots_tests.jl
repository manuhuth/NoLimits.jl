using Test
using NoLimits
using DataFrames
using Distributions

@testset "residual plots basic API (FitResult + DataModel + cache)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(-0.2)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            y ~ Normal(a + b * z, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2],
        y = [0.15, 0.18, 0.14, 0.19]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    rdf = get_residuals(res)
    cols = Set(Symbol.(names(rdf)))
    @test nrow(rdf) == nrow(df)
    @test :pit in cols
    @test :res_quantile in cols
    @test :res_raw in cols
    @test :res_pearson in cols
    @test :logscore in cols
    @test :observable in cols
    @test :n_draws in cols
    @test all(rdf.n_draws .== 1)

    p1 = plot_residuals(res; residual=:quantile)
    @test p1 !== nothing

    p2 = plot_residual_distribution(res; residual=:pit)
    @test p2 !== nothing

    p3 = plot_residual_qq(res; residual=:quantile)
    @test p3 !== nothing

    p4 = plot_residual_pit(res; show_hist=true, show_qq=false)
    @test p4 !== nothing

    p5 = plot_residual_acf(res; residual=:raw, max_lag=1)
    @test p5 !== nothing

    cache = build_plot_cache(dm; cache_obs_dists=false)
    rdf_dm = get_residuals(dm; cache=cache, cache_obs_dists=true)
    @test nrow(rdf_dm) == nrow(df)

    p_dm = plot_residuals(dm; cache=cache, cache_obs_dists=true)
    @test p_dm !== nothing

    mktempdir() do tmp
        p1_path = joinpath(tmp, "plot_residuals.png")
        p2_path = joinpath(tmp, "plot_residual_distribution.png")
        p3_path = joinpath(tmp, "plot_residual_qq.png")
        p4_path = joinpath(tmp, "plot_residual_pit.png")
        p5_path = joinpath(tmp, "plot_residual_acf.png")
        plot_residuals(res; residual=:quantile, plot_path=p1_path)
        plot_residual_distribution(res; residual=:pit, plot_path=p2_path)
        plot_residual_qq(res; residual=:quantile, plot_path=p3_path)
        plot_residual_pit(res; show_hist=true, show_qq=false, plot_path=p4_path)
        plot_residual_acf(res; residual=:raw, max_lag=1, plot_path=p5_path)
        @test isfile(p1_path)
        @test isfile(p2_path)
        @test isfile(p3_path)
        @test isfile(p4_path)
        @test isfile(p5_path)
    end
end

@testset "residual plots support multiple observables" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.3)
            c = RealNumber(-0.2)
            σ = RealNumber(0.2, scale=:log)
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
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        z = [0.2, -0.1, 0.3, 0.0],
        y_cont = [0.1, 0.0, 0.2, 0.1],
        y_bin = [1, 0, 1, 0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    rdf = get_residuals(res; residuals=[:pit, :raw], randomize_discrete=false)
    @test nrow(rdf) == 2 * nrow(df)
    @test Set(rdf.observable) == Set([:y_cont, :y_bin])

    p_dist = plot_residual_distribution(res; residual=:pit)
    @test p_dist !== nothing

    p_pit = plot_residual_pit(res; show_hist=false, show_kde=true, show_qq=false)
    @test p_pit !== nothing
end

@testset "residuals with constants_re inherited from fit result" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.5); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    constants_re = (; η=(; B=0.0))
    res = fit_model(dm, NoLimits.Laplace(); constants_re=constants_re)

    rdf = get_residuals(res)
    @test nrow(rdf) == nrow(df)

    p = plot_residuals(res)
    @test p !== nothing
end

@testset "residuals MCMC summary and draw-level outputs" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a * t, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=10, n_adapt=2, progress=false)))

    rdf = get_residuals(res; mcmc_draws=5, mcmc_quantiles=[10, 90])
    @test nrow(rdf) == nrow(df)
    @test all(rdf.n_draws .== 5)
    @test all(ismissing.(rdf.draw))
    @test all(.!ismissing.(rdf.pit_qlo))
    @test all(.!ismissing.(rdf.pit_qhi))

    rdf_draw = get_residuals(res; mcmc_draws=3, return_draw_level=true, residuals=[:pit])
    @test nrow(rdf_draw) == 3 * nrow(df)
    @test all(rdf_draw.n_draws .== 3)
    @test all(.!ismissing.(rdf_draw.draw))

    p = plot_residual_qq(res; mcmc_draws=3)
    @test p !== nothing
end

@testset "residual API validation errors" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.2, scale=:log)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            y ~ Normal(a + z, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        z = [0.1, 0.2],
        y = [0.1, 0.2]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    @test_throws ErrorException get_residuals(res; residuals=[:not_a_metric])
    @test_throws ErrorException plot_residuals(res; residual=:not_a_metric)
    @test_throws ErrorException get_residuals(res; x_axis_feature=:missing_feature)
    @test_throws ErrorException plot_residual_acf(res; max_lag=0)
    @test_throws ErrorException get_residuals(res; mcmc_quantiles=[-5, 95])
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
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.0, 0.4, 0.2, 0.6, 0.8, 1.0],
        y = [1, 2, 1, 2, 3, 4]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    rdf = get_residuals(res; residuals=[:pit, :raw, :pearson], randomize_discrete=true)
    @test nrow(rdf) == nrow(df)
    @test all(.!ismissing.(rdf.res_raw))
    @test all(.!ismissing.(rdf.res_pearson))

    p_dist = plot_residual_distribution(res; residual=:pit)
    @test p_dist !== nothing

    p_scatter = plot_residuals(res; residual=:pearson)
    @test p_scatter !== nothing

    p_pit = plot_residual_pit(res; show_hist=true, show_kde=false, show_qq=true)
    @test p_pit !== nothing
end
