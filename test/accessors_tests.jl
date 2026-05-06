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
    @test NoLimits.get_converged(res_mle) isa Bool
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
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=2, n_adapt=2, progress=false)))
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

    re_samples = NoLimits.sample_random_effects(res; n_samples=5)
    @test !isempty(re_samples)
    @test :sample in propertynames(re_samples.η)
    @test nrow(re_samples.η) == 5 * nrow(re.η)
    @test sort(unique(re_samples.η.sample)) == collect(1:5)

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

    re_map_samples = NoLimits.sample_random_effects(res_map; n_samples=3)
    @test !isempty(re_map_samples)
    @test nrow(re_map_samples.η) == 3 * nrow(re_map.η)
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

    res_mcem = fit_model(dm, NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
                                  maxiters=2))
    re_mcem = NoLimits.get_random_effects(res_mcem)
    @test !isempty(re_mcem)
    @test res_mcem.result.eb_modes !== nothing

    re_mcem_samples = NoLimits.sample_random_effects(res_mcem; n_samples=4, n_adapt=2)
    @test !isempty(re_mcem_samples)
    @test :sample in propertynames(re_mcem_samples.η)
    @test nrow(re_mcem_samples.η) == 4 * nrow(re_mcem.η)
    @test sort(unique(re_mcem_samples.η.sample)) == collect(1:4)


    res_saem = fit_model(dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
                                  q_store_max=2,
                                  maxiters=2))
    re_saem = NoLimits.get_random_effects(res_saem)
    @test !isempty(re_saem)
    @test res_saem.result.eb_modes !== nothing

    re_saem_samples = NoLimits.sample_random_effects(res_saem; n_samples=3, n_adapt=2)
    @test !isempty(re_saem_samples)
    @test :sample in propertynames(re_saem_samples.η)
    @test nrow(re_saem_samples.η) == 3 * nrow(re_saem.η)

    # SAEM with the default SaemixMH sampler
    res_saem_smh = fit_model(dm, NoLimits.SAEM(; q_store_max=2, maxiters=2))
    re_saem_smh_samples = NoLimits.sample_random_effects(res_saem_smh; n_samples=3)
    @test !isempty(re_saem_smh_samples)
    @test nrow(re_saem_smh_samples.η) == 3 * nrow(NoLimits.get_random_effects(res_saem_smh).η)

end
