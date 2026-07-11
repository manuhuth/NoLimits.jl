using Test
using NoLimits
using DataFrames
using Distributions
using Turing

# Note: "Plot cache (non-ODE)" and "Plot cache (RE non-ODE, Laplace)"
# have been moved to integration_plotting.jl (shared fixtures). The testsets
# below consume the shared fx_* fixture models/fits.

@testset "Plot cache (RE ODE, Laplace)" begin
    dm = fx_ode_dm()
    cache = build_plot_cache(fx_ode_laplace(); cache_obs_dists = false)
    @test cache isa PlotCache
    @test length(cache.sols) == length(get_individuals(dm))
end

@testset "Plot cache kwargs" begin
    res = fx_mle()

    cache1 = build_plot_cache(res; cache_obs_dists = false)
    cache2 = build_plot_cache(res; cache_obs_dists = true)
    @test cache1.signature != cache2.signature

    cache3 = build_plot_cache(res; params = (a = 1.5,))
    @test getproperty(cache3.params, :a) == 1.5
end

@testset "Plot cache kwargs (MCMC warmup override)" begin
    cache = build_plot_cache(fx_mcmc_re(); mcmc_warmup = 1, mcmc_draws = 5)
    @test cache isa PlotCache
    @test cache.chain !== nothing
end

@testset "Plot cache inherits constants_re from fit result" begin
    dm = fx_recov_dm()
    constants_re = (; η = (; B = 0.0))
    res = fit_model(dm,
        NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
        constants_re = constants_re)

    cache = build_plot_cache(res)
    @test cache isa PlotCache
    @test getproperty(cache.random_effects[dm.id_index[:B]], :η) ≈ 0.0
end

@testset "Plot cache (MCMC)" begin
    cache = build_plot_cache(fx_mcmc_re(); cache_obs_dists = false, mcmc_draws = 5)
    @test cache isa PlotCache
    @test cache.chain !== nothing
end

@testset "Plot cache (VI, fixed-effects only)" begin
    cache = build_plot_cache(fx_vi(); cache_obs_dists = false, mcmc_draws = 5)
    @test cache isa PlotCache
    @test cache.chain === nothing
end

@testset "Plot cache (ODE)" begin
    dm = fx_ode_dm()
    cache = build_plot_cache(fx_ode_laplace(); cache_obs_dists = true)
    @test cache isa PlotCache
    @test length(cache.sols) == length(get_individuals(dm))
    @test cache.sols[1] !== nothing
    @test length(cache.obs_dists[1]) == length(get_row_groups(dm).obs_rows[1])
end

@testset "Plot cache uses row-specific random effects for varying non-ODE groups" begin
    dm = fx_varyre_dm()
    cache = build_plot_cache(
        dm; constants_re = fx_varyre_constants_re(), cache_obs_dists = true)

    @test cache.random_effects[1].η_year ≈ [0.1, 0.4]
    @test cache.random_effects[2].η_year ≈ [0.1, 0.3]
    means = [
        Distributions.mean(getproperty(cache.obs_dists[1][1], :y)),
        Distributions.mean(getproperty(cache.obs_dists[1][2], :y)),
        Distributions.mean(getproperty(cache.obs_dists[1][3], :y)),
        Distributions.mean(getproperty(cache.obs_dists[2][1], :y)),
        Distributions.mean(getproperty(cache.obs_dists[2][2], :y))
    ]
    @test means ≈ [0.1, 0.4, 0.4, 0.1, 0.3]
end
