using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using Turing
using MCMCChains
using SciMLBase

const LD = NoLimits

@testset "Multistart basic (MLE)" begin
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
    ms = NoLimits.Multistart(dists=(; a=Normal(1.0, 0.2)), n_draws_requested=4, n_draws_used=3)
    res = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))
    @test res isa NoLimits.MultistartFitResult
    @test length(NoLimits.get_multistart_results(res)) == 3
    @test NoLimits.get_multistart_best_index(res) in 1:3
    @test NoLimits.get_params(res; scale=:untransformed) isa ComponentArray
    @test isfinite(NoLimits.get_objective(res))
end

@testset "Multistart LHS + fixed params" begin
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
    ms = NoLimits.Multistart(dists=(; a=Normal(0.0, 1.0)), n_draws_requested=4, n_draws_used=3, sampling=:lhs)
    res = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))
    starts = NoLimits.get_multistart_starts(res)
    @test length(starts) == 3
    @test all(s -> s.σ == starts[1].σ, starts) # σ not sampled, stays fixed
end

@testset "Multistart bounds violation" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2; lower=-1.0, upper=1.0, scale=:identity)
        end

        @formulas begin
            y ~ Normal(a, 1.0)
        end
    end
    df = DataFrame(
        ID = [:A, :A],
        t = [0.0, 1.0],
        y = [0.1, 0.2]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    ms = NoLimits.Multistart(dists=(; a=Normal(10.0, 0.1)), n_draws_requested=2, n_draws_used=2)
    @test_throws ErrorException fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=1,)))
end

@testset "Multistart MAP" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.2))
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
    ms = NoLimits.Multistart(n_draws_requested=4, n_draws_used=3)
    res = fit_model(ms, dm, NoLimits.MAP(; optim_kwargs=(maxiters=2,)))
    @test res isa NoLimits.MultistartFitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "Multistart Laplace / LaplaceMAP" begin
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
    ms = NoLimits.Multistart(dists=(; a=Normal(0.0, 1.0)), n_draws_requested=4, n_draws_used=3)
    res = fit_model(ms, dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,)))
    @test !isempty(NoLimits.get_random_effects(res))

    model_map = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.2))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    dm_map = DataModel(model_map, df; primary_id=:ID, time_col=:t)
    res_map = fit_model(ms, dm_map, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=2,)))
    @test !isempty(NoLimits.get_random_effects(res_map))
end

@testset "Multistart MCMC" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.2))
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
    ms = NoLimits.Multistart(n_draws_requested=3, n_draws_used=2)
    res = fit_model(ms, dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false)))
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "Multistart MCEM / SAEM" begin
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
    ms = NoLimits.Multistart(dists=(; a=Normal(0.0, 1.0)), n_draws_requested=3, n_draws_used=2)
    res_mcem = fit_model(ms, dm, NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                                        maxiters=2))
    @test !isempty(NoLimits.get_random_effects(res_mcem))
    res_saem = fit_model(ms, dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                                        max_store=3,
                                        maxiters=2))
    @test !isempty(NoLimits.get_random_effects(res_saem))
end

@testset "Multistart SAEM suffstats" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
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
    ms = NoLimits.Multistart(dists=(; a=Normal(0.0, 1.0)), n_draws_requested=3, n_draws_used=2)
    res = fit_model(ms, dm, NoLimits.SAEM(; sampler=MH(), turing_kwargs=(n_samples=5, n_adapt=0, progress=false),
                                     max_store=3,
                                     builtin_stats=:closed_form,
                                     resid_var_param=:σ,
                                     re_cov_params=(; η=:τ),
                                     maxiters=2))
    @test !isempty(NoLimits.get_random_effects(res))
end

@testset "Multistart store_data_model false" begin
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
        ID = [:A, :A],
        t = [0.0, 1.0],
        y = [0.1, 0.2]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    ms = NoLimits.Multistart(dists=(; a=Normal(0.0, 1.0)), n_draws_requested=2, n_draws_used=2)
    res = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)); store_data_model=false)
    @test_throws ErrorException NoLimits.get_loglikelihood(res)
end

@testset "Multistart threading" begin
    Threads.nthreads() > 1 || return
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
        ID = [:A, :A],
        t = [0.0, 1.0],
        y = [0.1, 0.2]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    ms = NoLimits.Multistart(dists=(; a=Normal(0.0, 1.0)), n_draws_requested=4, n_draws_used=3, serialization=EnsembleThreads())
    res = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))
    @test res isa NoLimits.MultistartFitResult
end

@testset "Multistart LHS extensive (univariate + priors override)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.2))
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
    ms = NoLimits.Multistart(dists=(; a=Normal(5.0, 0.1)), n_draws_requested=6, n_draws_used=4, sampling=:lhs)
    res = fit_model(ms, dm, NoLimits.MAP(; optim_kwargs=(maxiters=2,)))
    starts = NoLimits.get_multistart_starts(res)
    @test length(starts) == 4
    @test any(s -> abs(s.a - 5.0) < 1.0, starts)
end

@testset "Multistart LHS vector parameter" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            v = RealVector([0.1, -0.2, 0.3])
            σ = RealNumber(0.5, scale=:log)
        end

        @formulas begin
            y ~ Normal(v[1], σ)
        end
    end
    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    dists = (;
        v = fill(Normal(0.0, 1.0), 3)
    )
    ms = NoLimits.Multistart(dists=dists, n_draws_requested=6, n_draws_used=4, sampling=:lhs)
    res = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))
    starts = NoLimits.get_multistart_starts(res)
    @test length(starts) == 4
    @test any(s -> s.v != starts[1].v, starts[2:end])
end

@testset "Multistart Wishart VCV (LHS fallback)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
            Ω = RealPSDMatrix(Matrix(I, 2, 2), scale=:cholesky)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η[1], σ)
        end
    end
    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    dists = (;
        Ω = Wishart(3, Matrix(I, 2, 2))
    )
    ms = Multistart(dists=dists, n_draws_requested=6, n_draws_used=4, sampling=:lhs)
    res = fit_model(ms, dm, NoLimits.Laplace(; optim_kwargs=(maxiters=3,)));
    starts = NoLimits.get_multistart_starts(res)
    @test length(starts) == 4
    @test any(s -> s.Ω != starts[1].Ω, starts[2:end])
    @test !isempty(NoLimits.get_random_effects(res))
end
