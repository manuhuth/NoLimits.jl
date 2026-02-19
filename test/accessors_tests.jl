using Test
using NoLimits
using Distributions
using MCMCChains
using DataFrames
using ComponentArrays
using Turing

const LD = NoLimits

@testset "Accessors (fixed effects)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    res_mle = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))
    @test NoLimits.get_params(res_mle; scale=:untransformed) isa ComponentArray
    @test isfinite(NoLimits.get_objective(res_mle))
    @test NoLimits.get_converged(res_mle) isa Bool
    @test isfinite(NoLimits.get_loglikelihood(res_mle))
    @test_throws ErrorException NoLimits.get_chain(res_mle)
    res_mle_nostore = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)); store_data_model=false)
    @test_throws ErrorException NoLimits.get_loglikelihood(res_mle_nostore)

    model_map = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end
    dm_map = DataModel(model_map, df; primary_id=:ID, time_col=:t)
    res_map = fit_model(dm_map, NoLimits.MAP(; optim_kwargs=(maxiters=2,)))
    @test isfinite(NoLimits.get_objective(res_map))
    @test isfinite(NoLimits.get_loglikelihood(res_map))
    @test_throws ErrorException NoLimits.get_chain(res_map)
end

@testset "Accessors (MCMC)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=5, n_adapt=2, progress=false)))
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
    @test NoLimits.get_observed(res).y == df.y
    @test NoLimits.get_sampler(res) isa Any
    @test NoLimits.get_n_samples(res) == 5
    @test_throws ErrorException NoLimits.get_loglikelihood(res)
end

@testset "Accessors (Laplace/LaplaceMAP)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,)))
    re = NoLimits.get_random_effects(res)
    @test !isempty(re)
    @test isfinite(NoLimits.get_loglikelihood(res))

    model_map = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    dm_map = DataModel(model_map, df; primary_id=:ID, time_col=:t)
    res_map = fit_model(dm_map, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=2,)))
    re_map = NoLimits.get_random_effects(res_map)
    @test !isempty(re_map)
    @test isfinite(NoLimits.get_loglikelihood(res_map))
end

@testset "Accessors (MCEM/SAEM)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    res_mcem = fit_model(dm, NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=6, n_adapt=0, progress=false),
                                  maxiters=2))
    re_mcem = NoLimits.get_random_effects(res_mcem)
    @test !isempty(re_mcem)
    @test res_mcem.result.eb_modes !== nothing


    res_saem = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                                  max_store=4,
                                  maxiters=2))
    re_saem = NoLimits.get_random_effects(res_saem)
    @test !isempty(re_saem)
    @test res_saem.result.eb_modes !== nothing

end
