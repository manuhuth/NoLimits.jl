using Test
using NoLimits
using DataFrames
using Distributions

# Note: "residual plots basic API (FitResult + DataModel + cache)"
# has been moved to integration_plotting.jl (shared fixtures).

@testset "residual plots support multiple observables" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1); b = RealNumber(0.3); c = RealNumber(-0.2)
            σ = RealNumber(0.2, scale=:log)
        end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @formulas begin
            y_cont ~ Normal(a + b * z, σ)
            p = logistic(c + z)
            y_bin ~ Bernoulli(p)
        end
    end

    df = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], z=[0.2,-0.1,0.3,0.0],
                   y_cont=[0.1,0.0,0.2,0.1], y_bin=[1,0,1,0])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))

    rdf = get_residuals(res; residuals=[:pit, :raw], randomize_discrete=false)
    @test nrow(rdf) == 2 * nrow(df)
    @test Set(rdf.observable) == Set([:y_cont, :y_bin])
    @test plot_residual_distribution(res; residual=:pit) !== nothing
    @test plot_residual_pit(res; show_hist=false, show_kde=true, show_qq=false) !== nothing
end

@testset "residuals with constants_re inherited from fit result" begin
    model = @Model begin
        @fixedEffects begin; a = RealNumber(0.1); σ = RealNumber(0.3, scale=:log); end
        @covariates begin; t = Covariate(); end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 0.5); column=:ID); end
        @formulas begin; y ~ Normal(a + η, σ); end
    end

    df = DataFrame(ID=[:A,:A,:B,:B,:C,:C], t=[0.0,1.0,0.0,1.0,0.0,1.0],
                   y=[0.1,0.2,0.0,0.1,0.15,0.25])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    constants_re = (; η=(; B=0.0))
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,), multistart_n=2, multistart_k=2);
                    constants_re=constants_re)

    @test nrow(get_residuals(res)) == nrow(df)
    @test plot_residuals(res) !== nothing
end

@testset "residuals MCMC summary and draw-level outputs" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end
        @covariates begin; t = Covariate(); end
        @formulas begin; y ~ Normal(a * t, σ); end
    end

    df = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=2, n_adapt=2, progress=false)))

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
    @test plot_residual_qq(res; mcmc_draws=3) !== nothing
end

@testset "residuals VI summary and draw-level outputs" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end
        @covariates begin; t = Covariate(); end
        @formulas begin; y ~ Normal(a * t, σ); end
    end

    df = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=15, progress=false)))

    rdf = get_residuals(res; mcmc_draws=5, mcmc_quantiles=[10, 90])
    @test nrow(rdf) == nrow(df)
    @test all(rdf.n_draws .== 5)
    @test all(ismissing.(rdf.draw))
    @test all(.!ismissing.(rdf.pit_qlo))
    @test all(.!ismissing.(rdf.pit_qhi))

    rdf_draw = get_residuals(res; mcmc_draws=5, return_draw_level=true, residuals=[:pit])
    @test nrow(rdf_draw) == 5 * nrow(df)
    @test all(rdf_draw.n_draws .== 5)
    @test all(.!ismissing.(rdf_draw.draw))
    @test plot_residual_qq(res; mcmc_draws=5) !== nothing
end

@testset "residual API validation errors" begin
    model = @Model begin
        @fixedEffects begin; a = RealNumber(0.1); σ = RealNumber(0.2, scale=:log); end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @formulas begin; y ~ Normal(a + z, σ); end
    end
    df = DataFrame(ID=[1,1], t=[0.0,1.0], z=[0.1,0.2], y=[0.1,0.2])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))
    @test_throws ErrorException get_residuals(res; residuals=[:not_a_metric])
    @test_throws ErrorException plot_residuals(res; residual=:not_a_metric)
    @test_throws ErrorException get_residuals(res; x_axis_feature=:missing_feature)
    @test_throws ErrorException plot_residual_acf(res; max_lag=0)
    @test_throws ErrorException get_residuals(res; mcmc_quantiles=[-5, 95])
end

@testset "residual plots Poisson outcome" begin
    model = @Model begin
        @fixedEffects begin; a = RealNumber(0.0); b = RealNumber(0.3); end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @formulas begin; λ = exp(a + b * z); y ~ Poisson(λ); end
    end
    df = DataFrame(ID=[1,1,2,2,3,3], t=[0.0,1.0,0.0,1.0,0.0,1.0],
                   z=[0.0,0.4,0.2,0.6,0.8,1.0], y=[1,2,1,2,3,4])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))

    rdf = get_residuals(res; residuals=[:pit, :raw, :pearson], randomize_discrete=true)
    @test nrow(rdf) == nrow(df)
    @test all(.!ismissing.(rdf.res_raw))
    @test all(.!ismissing.(rdf.res_pearson))
    @test plot_residual_distribution(res; residual=:pit) !== nothing
    @test plot_residuals(res; residual=:pearson) !== nothing
    @test plot_residual_pit(res; show_hist=true, show_kde=false, show_qq=true) !== nothing
end

@testset "residuals use row-specific random effects for varying non-ODE groups" begin
    model = @Model begin
        @fixedEffects begin; σ = RealNumber(1.0e-6, scale=:log); end
        @covariates begin; t = Covariate(); end
        @randomEffects begin; η_year = RandomEffect(Normal(0.0, 1.0); column=:YEAR); end
        @formulas begin; y ~ Normal(η_year, σ); end
    end
    df = DataFrame(ID=[1,1,1,2,2], YEAR=[:A,:B,:B,:A,:C],
                   t=[0.0,1.0,2.0,0.0,1.0], y=[0.1,0.4,0.4,0.1,0.3])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    constants_re = (; η_year=(; A=0.1, B=0.4, C=0.3))
    cache = build_plot_cache(dm; constants_re=constants_re, cache_obs_dists=true)
    rdf = get_residuals(dm; cache=cache, cache_obs_dists=true, residuals=[:raw])
    sort!(rdf, :row)
    @test Float64.(rdf.fitted) ≈ [0.1, 0.4, 0.4, 0.1, 0.3]
    @test maximum(abs.(Float64.(rdf.res_raw))) < 1.0e-6
end
