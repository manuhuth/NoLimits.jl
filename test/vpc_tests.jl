using Test
using NoLimits
using DataFrames
using Distributions
using Random
using Turing

# Note: "plot_vpc basic" and "plot_vpc validation errors"
# have been moved to integration_plotting.jl (shared fixtures).

@testset "plot_vpc ode" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0, lower=0.001, upper=1.1)
            σ = RealNumber(0.01, scale=:log)
        end
        @covariates begin; t = Covariate(); end
        @DifferentialEquation begin; D(x1) ~ -a * x1; end
        @initialDE begin; x1 = 1.0; end
        @formulas begin; y ~ Normal(x1(t), σ); end
    end

    df = DataFrame(ID=[1,1,1], t=[0.0,0.5,1.0], y=[1.0,0.95,0.9])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))
    @test plot_vpc(res; n_simulations=5, n_bins=3) !== nothing
end

@testset "plot_vpc skips missing observed outcomes (regression)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1); b = RealNumber(-0.2); σ = RealNumber(0.3, scale=:log)
        end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @formulas begin; y ~ Normal(a + b * z, σ); end
    end

    df = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], z=[0.1,0.2,0.1,0.2],
                   y=Union{Missing,Float64}[0.15, missing, 0.14, missing])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))
    @test plot_vpc(res; n_simulations=5, n_bins=2) !== nothing
    @test plot_vpc(res; x_axis_feature=:z, n_simulations=5, n_bins=2) !== nothing
end

@testset "plot_vpc discrete mcmc random effects" begin
    model_bern = @Model begin
        @fixedEffects begin; a = RealNumber(0.3, prior=Normal(1.0, 10.0)); end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 2.0); column=:ID); end
        @formulas begin; p = logistic(a*t + z + η); y ~ Bernoulli(p); end
    end
    df_bern = DataFrame(ID=[1,1,2,2,3,3], t=[0.0,1.0,0.0,1.0,0.0,1.0],
                        z=[0.1,0.2,0.1,0.2,0.1,0.2], y=[0,1,0,1,0,1])
    dm_bern = DataModel(model_bern, df_bern; primary_id=:ID, time_col=:t)
    res_bern = fit_model(dm_bern, MCMC(; sampler=MH(), turing_kwargs=(n_samples=2, n_adapt=2, progress=false)))
    @test plot_vpc(res_bern; n_simulations=5, n_bins=3, mcmc_draws=5) !== nothing

    model_pois = @Model begin
        @fixedEffects begin; a = RealNumber(0.1, prior=Normal(1.0, 10.0)); end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 0.3); column=:ID); end
        @formulas begin; λ = exp(a * z + η); y ~ Poisson(λ); end
    end
    df_pois = DataFrame(ID=[1,1,2,2,3,3], t=[0.0,1.0,0.0,1.0,0.0,1.0],
                        z=[0.1,0.2,0.1,0.2,0.1,0.2], y=[1,2,1,2,1,3])
    dm_pois = DataModel(model_pois, df_pois; primary_id=:ID, time_col=:t)
    res_pois = fit_model(dm_pois, MCMC(; sampler=MH(), turing_kwargs=(n_samples=2, n_adapt=2, progress=false)))
    @test plot_vpc(res_pois; n_simulations=5, n_bins=3, mcmc_draws=5) !== nothing
end

@testset "plot_vpc VI random effects" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end
        @covariates begin; t = Covariate(); end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 0.5); column=:ID); end
        @formulas begin; y ~ Normal(a * t + η, σ); end
    end
    df = DataFrame(ID=[1,1,2,2,3,3], t=[0.0,1.0,0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1,0.05,0.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, VI(; turing_kwargs=(max_iter=15, progress=false)))
    @test plot_vpc(res; n_simulations=5, n_bins=3, mcmc_draws=5) !== nothing
end

@testset "plot_vpc constants_re and unsupported serialization" begin
    model = @Model begin
        @fixedEffects begin; a = RealNumber(0.1); σ = RealNumber(0.3, scale=:log); end
        @covariates begin; t = Covariate(); end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 0.5); column=:ID); end
        @formulas begin; y ~ Normal(a + η, σ); end
    end
    df = DataFrame(ID=[:A,:A,:B,:B,:C,:C], t=[0.0,1.0,0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,0.1,0.15,0.25])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    constants_re = (; η=(; B=0.0))
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,), multistart_n=2, multistart_k=2);
                    constants_re=constants_re)
    @test plot_vpc(res; n_simulations=5, n_bins=2) !== nothing
    @test_throws ArgumentError plot_vpc(res; serialization=:unsupported)
end

@testset "plot_vpc internals use row-specific random effects for varying non-ODE groups" begin
    model = @Model begin
        @fixedEffects begin; σ = RealNumber(1.0e-6, scale=:log); end
        @covariates begin; t = Covariate(); end
        @randomEffects begin; η_year = RandomEffect(Normal(0.0, 1.0); column=:YEAR); end
        @formulas begin; y ~ Normal(η_year, σ); end
    end
    df = DataFrame(ID=[1,1,1,2,2], YEAR=[:A,:B,:B,:A,:C],
                   t=[0.0,1.0,2.0,0.0,1.0], y=[0.1,0.4,0.4,0.1,0.3])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = NoLimits.get_θ0_untransformed(dm.model.fixed.fixed)
    level_vals = Dict{Symbol,Dict{Any,Any}}(:η_year => Dict{Any,Any}(:A=>0.1,:B=>0.4,:C=>0.3))
    η_vec = NoLimits._eta_vec_from_levels(dm, level_vals)
    sim_x, sim_vals = NoLimits._simulate_obs(dm, θ, η_vec, :y, MersenneTwister(1), nothing)
    @test sim_x[1] == [0.0, 1.0, 2.0]
    @test sim_x[2] == [0.0, 1.0]
    @test sim_vals[1] ≈ [0.1, 0.4, 0.4] atol=1.0e-3
    @test sim_vals[2] ≈ [0.1, 0.3] atol=1.0e-3
end
