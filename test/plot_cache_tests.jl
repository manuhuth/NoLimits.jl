using Test
using NoLimits
using DataFrames
using Distributions
using Turing

# Note: "Plot cache (non-ODE)" and "Plot cache (RE non-ODE, Laplace)"
# have been moved to integration_plotting.jl (shared fixtures).

@testset "Plot cache (RE ODE, Laplace)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.01, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + η
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], y=[1.0,0.9,1.1,1.0])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,), multistart_n=2, multistart_k=2))

    cache = build_plot_cache(res; cache_obs_dists=false)
    @test cache isa PlotCache
    @test length(cache.sols) == length(get_individuals(dm))
end

@testset "Plot cache kwargs" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1); b = RealNumber(0.2); σ = RealNumber(0.3, scale=:log)
        end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @formulas begin; y ~ Normal(a + b * z, σ); end
    end

    df = DataFrame(ID=[1,1], t=[0.0,1.0], z=[0.1,0.2], y=[0.15,0.18])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))

    cache1 = build_plot_cache(res; cache_obs_dists=false)
    cache2 = build_plot_cache(res; cache_obs_dists=true)
    @test cache1.signature != cache2.signature

    cache3 = build_plot_cache(res; params=(a=1.5,))
    @test getproperty(cache3.params, :a) == 1.5
end

@testset "Plot cache kwargs (MCMC warmup override)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end
        @covariates begin; t = Covariate(); end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:ID); end
        @formulas begin; y ~ Normal(a + η, σ); end
    end

    df = DataFrame(ID=[1,1], t=[0.0,1.0], y=[0.1,0.2])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=2, n_adapt=2, progress=false)))

    cache = build_plot_cache(res; mcmc_warmup=1, mcmc_draws=5)
    @test cache isa PlotCache
    @test cache.chain !== nothing
end

@testset "Plot cache inherits constants_re from fit result" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1); σ = RealNumber(0.3, scale=:log)
        end
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

    cache = build_plot_cache(res)
    @test cache isa PlotCache
    @test getproperty(cache.random_effects[dm.id_index[:B]], :η) ≈ 0.0
end

@testset "Plot cache (MCMC)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end
        @covariates begin; t = Covariate(); end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:ID); end
        @formulas begin; y ~ Normal(a + η, σ); end
    end

    df = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=2, n_adapt=2, progress=false)))

    cache = build_plot_cache(res; cache_obs_dists=false, mcmc_draws=5)
    @test cache isa PlotCache
    @test cache.chain !== nothing
end

@testset "Plot cache (VI)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end
        @covariates begin; t = Covariate(); end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:ID); end
        @formulas begin; y ~ Normal(a + η, σ); end
    end

    df = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=10, progress=false)))

    cache = build_plot_cache(res; cache_obs_dists=false, mcmc_draws=5)
    @test cache isa PlotCache
    @test cache.chain === nothing
end

@testset "Plot cache (ODE)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1); σ = RealNumber(0.01, scale=:log)
        end
        @covariates begin; t = Covariate(); end
        @DifferentialEquation begin; D(x1) ~ -a * x1; end
        @initialDE begin; x1 = 1.0; end
        @formulas begin; y ~ Normal(x1(t), σ); end
    end

    df = DataFrame(ID=[1,1,1], t=[0.0,0.5,1.0], y=[1.0,0.95,0.9])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))

    cache = build_plot_cache(res; cache_obs_dists=true)
    @test cache isa PlotCache
    @test length(cache.sols) == length(get_individuals(dm))
    @test cache.sols[1] !== nothing
    @test length(cache.obs_dists[1]) == length(get_row_groups(dm).obs_rows[1])
end

@testset "Plot cache uses row-specific random effects for varying non-ODE groups" begin
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

    @test cache.random_effects[1].η_year ≈ [0.1, 0.4]
    @test cache.random_effects[2].η_year ≈ [0.1, 0.3]
    means = [
        Distributions.mean(getproperty(cache.obs_dists[1][1], :y)),
        Distributions.mean(getproperty(cache.obs_dists[1][2], :y)),
        Distributions.mean(getproperty(cache.obs_dists[1][3], :y)),
        Distributions.mean(getproperty(cache.obs_dists[2][1], :y)),
        Distributions.mean(getproperty(cache.obs_dists[2][2], :y)),
    ]
    @test means ≈ [0.1, 0.4, 0.4, 0.1, 0.3]
end
