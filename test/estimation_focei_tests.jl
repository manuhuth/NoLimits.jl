using Test
using DataFrames
using NoLimits
using Distributions
using FiniteDifferences
using Optimization
using OptimizationOptimJL
using LineSearches
using ComponentArrays
using Random
using LinearAlgebra

@testset "FOCEI requires random effects" begin
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
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.FOCEI(; optim_kwargs=(maxiters=2,)))
end

@testset "FOCEI default multistart settings" begin
    method = NoLimits.FOCEI()
    @test method.multistart.n == 50
    @test method.multistart.k == 10
    @test method.multistart.sampling == :lhs
    method_map = NoLimits.FOCEIMAP()
    @test method_map.multistart.n == 50
    @test method_map.multistart.k == 10
    @test method_map.multistart.sampling == :lhs
end

@testset "FOCEI fit (non-ODE) runs and returns EB modes" begin
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
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.FOCEI(; optim_kwargs=(maxiters=3,),
     inner_kwargs=(maxiters=20,), multistart_n=0, multistart_k=0))
    @test res.result.eb_modes !== nothing
    @test length(res.result.eb_modes) == length(get_batches(dm))
    re = NoLimits.get_random_effects(res)
    @test !isempty(re)
    @test isfinite(NoLimits.get_loglikelihood(res))
end

@testset "FOCEI fit serial vs threaded is reproducible" begin
    Threads.nthreads() < 2 && return

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
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    method = NoLimits.FOCEI(; optim_kwargs=(maxiters=3,),
                                    inner_kwargs=(maxiters=20,),
                                    multistart_n=0,
                                    multistart_k=0)
    res_serial = fit_model(dm, method; serialization=EnsembleSerial(), rng=MersenneTwister(123))
    res_threads = fit_model(dm, method; serialization=EnsembleThreads(), rng=MersenneTwister(123))
    @test res_serial.summary.objective == res_threads.summary.objective
    @test collect(NoLimits.get_params(res_serial, scale=:untransformed)) ==
          collect(NoLimits.get_params(res_threads, scale=:untransformed))
end

@testset "FOCEI objective gradient matches finite differences (OPG)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:GROUP)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        GROUP = [:A, :A, :A, :A],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θ0 = get_θ0_untransformed(model.fixed.fixed)
    Tθ = eltype(θ0)
    n_batches = length(batch_infos)
    bstar_cache = NoLimits._LaplaceBStarCache([Vector{Tθ}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = NoLimits._LaplaceGradCache([Vector{Tθ}() for _ in 1:n_batches],
                                                    fill(Tθ(NaN), n_batches),
                                                    [Vector{Tθ}() for _ in 1:n_batches],
                                                    falses(n_batches))
    ad_cache = NoLimits._init_laplace_ad_cache(n_batches)
    hess_cache = NoLimits._init_laplace_hess_cache(Tθ, n_batches)
    ebe_cache = NoLimits._LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)
    method = NoLimits.FOCEI(; optim_kwargs=(maxiters=2,),
                                    inner_kwargs=(maxiters=25,),
                                    multistart_n=0,
                                    multistart_k=0,
                                    fallback_to_laplace=false,
                                    info_mode=:custom,
                                    info_custom=NoLimits.focei_information_opg,
                                    info_jitter=1e-8)
    rng = Random.Xoshiro(42)

    function focei_obj(θ_vec)
        θ = ComponentArray(θ_vec, getaxes(θ0))
        obj, _, _ = NoLimits._focei_objective_and_grad(dm, batch_infos, θ, const_cache, ll_cache, ebe_cache;
                                                                inner=method.inner,
                                                                info_opts=method.info,
                                                                fallback_hessian=method.fallback_hessian,
                                                                cache_opts=method.cache,
                                                                multistart=method.multistart,
                                                                rng=rng)
        return obj
    end

    function focei_grad(θ_vec)
        θ = ComponentArray(θ_vec, getaxes(θ0))
        _, g, _ = NoLimits._focei_objective_and_grad(dm, batch_infos, θ, const_cache, ll_cache, ebe_cache;
                                                              inner=method.inner,
                                                              info_opts=method.info,
                                                              fallback_hessian=method.fallback_hessian,
                                                              cache_opts=method.cache,
                                                              multistart=method.multistart,
                                                              rng=rng)
        return collect(g)
    end

    fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), focei_obj, collect(θ0))[1]
    g = focei_grad(collect(θ0))
    @test isapprox(g, fd; rtol=2e-2, atol=2e-2)
end

@testset "FOCEI supports constants_re in multi-group models" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @formulas begin
            y ~ Normal(a + η_id + η_site, σ)
        end
    end

    df = DataFrame(
        ID = [:id1, :id1, :id2, :id2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    constants_re = NamedTuple{(:η_id, :η_site)}((
        NamedTuple{(:id1,)}((0.3,)),
        NamedTuple{(:A,)}((-0.2,))
    ))
    res = fit_model(dm, NoLimits.FOCEI(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=20,), multistart_n=0, multistart_k=0);
                    constants_re=constants_re)
    @test res.summary.converged isa Bool
    re = NoLimits.get_random_effects(res; include_constants=true)
    @test hasproperty(re, :η_id)
    @test hasproperty(re, :η_site)
end

@testset "FOCEI supports non-Gaussian outcomes" begin
    model_bern = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            p = 1 / (1 + exp(-(a + η)))
            y ~ Bernoulli(p)
        end
    end
    df_bern = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0, 1, 1, 0]
    )
    dm_bern = DataModel(model_bern, df_bern; primary_id=:ID, time_col=:t)
    res_bern = fit_model(dm_bern, NoLimits.FOCEI(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=20,), multistart_n=0, multistart_k=0))
    @test res_bern isa FitResult

    model_pois = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            λ = exp(a + η)
            y ~ Poisson(λ)
        end
    end
    df_pois = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1, 2, 0, 1]
    )
    dm_pois = DataModel(model_pois, df_pois; primary_id=:ID, time_col=:t)
    res_pois = fit_model(dm_pois, NoLimits.FOCEI(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=20,), multistart_n=0, multistart_k=0))
    @test res_pois isa FitResult

    model_multi = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            τ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            p = 1 / (1 + exp(-(a + η)))
            y1 ~ Bernoulli(p)
            λ = exp(b + η)
            y2 ~ Poisson(λ)
        end
    end
    df_multi = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0, 1, 1, 0],
        y2 = [1, 2, 0, 1]
    )
    dm_multi = DataModel(model_multi, df_multi; primary_id=:ID, time_col=:t)
    res_multi = fit_model(dm_multi, NoLimits.FOCEI(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=20,), multistart_n=0, multistart_k=0))
    @test res_multi isa FitResult
end

@testset "FOCEI supports multivariate random-effect distributions" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], LinearAlgebra.I(2)); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η[1] + 0.2 * η[2], σ)
        end
    end
    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.FOCEI(; optim_kwargs=(maxiters=2,),
                                               inner_kwargs=(maxiters=20,),
                                               multistart_n=0,
                                               multistart_k=0,
                                               info_mode=:custom,
                                               info_custom=NoLimits.focei_information_opg))
    @test res isa FitResult
    re = NoLimits.get_random_effects(res)
    @test !isempty(re)
end

@testset "FOCEI supports NormalizingPlanarFlow random effects" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale=:log)
            ψ = NPFParameter(1, 2, seed=1, calculate_se=false)
        end

        @randomEffects begin
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η_flow[1], σ)
        end
    end
    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.FOCEI(; optim_kwargs=(maxiters=2,),
                                               inner_kwargs=(maxiters=20,),
                                               info_mode=:custom,
                                               info_custom=NoLimits.focei_information_opg,
                                               multistart_n=0,
                                               multistart_k=0))
    @test res isa FitResult
    @test res.result.eb_modes !== nothing
    @test length(res.result.eb_modes) == length(get_batches(dm))
end

@testset "FOCEI reports Laplace fallback usage" begin
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
    res = fit_model(dm, FOCEI(; optim_kwargs=(maxiters=1,),
                                               inner_kwargs=(maxiters=10,),
                                               multistart_n=0,
                                               multistart_k=0,
                                               info_max_tries=0,
                                               fallback_to_laplace=true));
    notes = NoLimits.get_notes(res)
    @test hasproperty(notes, :focei_fallback_total)
    @test hasproperty(notes, :focei_fallback_info_logdet)
    @test hasproperty(notes, :focei_fallback_mode_hessian)
    @test hasproperty(notes, :focei_fallback_at_solution)
    @test notes.focei_fallback_total > 0
    @test notes.focei_fallback_info_logdet > 0
    @test notes.focei_fallback_at_solution
end

@testset "FOCEI fisher_common gives exact scalar Gaussian information" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
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
        ID = [:A, :A],
        t = [0.0, 1.0],
        y = [0.2, 0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θ0 = get_θ0_untransformed(model.fixed.fixed)
    info = NoLimits.FOCEIInformationOptions(:fisher_common, 1e-8, 6, 10.0, true, 1e-8, true, :exact_hessian)
    b = zeros(eltype(θ0), batch_infos[1].n_b)
    I = NoLimits._focei_information_matrix(dm, batch_infos[1], θ0, b, const_cache, ll_cache, info)
    expected = 2 / (θ0.σ^2) + 1.0
    @test size(I) == (1, 1)
    @test isapprox(I[1, 1], expected; rtol=1e-8, atol=1e-8)
end

@testset "FOCEI fisher_common fit path runs" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
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
        y = [0.2, 0.1, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.FOCEI(; optim_kwargs=(maxiters=2,),
                                                inner_kwargs=(maxiters=20,),
                                                multistart_n=0,
                                                multistart_k=0,
                                                info_mode=:fisher_common))
    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))

    model_mix = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
            b = RealNumber(0.2)
            τ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            p = 1 / (1 + exp(-(a + η)))
            y1 ~ Bernoulli(p)
            λ = exp(b + η)
            y2 ~ Poisson(λ)
        end
    end
    df_mix = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0, 1, 1, 0],
        y2 = [1, 2, 0, 1]
    )
    dm_mix = DataModel(model_mix, df_mix; primary_id=:ID, time_col=:t)
    res_mix = fit_model(dm_mix, NoLimits.FOCEI(; optim_kwargs=(maxiters=1,),
                                                        inner_kwargs=(maxiters=20,),
                                                        multistart_n=0,
                                                        multistart_k=0,
                                                        info_mode=:fisher_common))
    @test res_mix isa FitResult
end

@testset "FOCEI fisher_common supports Exponential, Geometric, and Binomial outcomes" begin
    function fisher_info_diag11(model, df)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
        ll_cache = build_ll_cache(dm)
        θ0 = get_θ0_untransformed(model.fixed.fixed)
        info = NoLimits.FOCEIInformationOptions(:fisher_common, 1e-8, 6, 10.0, true, 1e-8, true, :exact_hessian)
        b = zeros(eltype(θ0), batch_infos[1].n_b)
        I = NoLimits._focei_information_matrix(dm, batch_infos[1], θ0, b, const_cache, ll_cache, info)
        @test size(I) == (1, 1)
        return I[1, 1]
    end

    model_exp = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            θ = exp(a + η)
            y ~ Exponential(θ)
        end
    end
    df_exp = DataFrame(ID=[:A, :A], t=[0.0, 1.0], y=[1.0, 2.0])
    # Two observations, each contributes 1; prior contributes 1.
    @test isapprox(fisher_info_diag11(model_exp, df_exp), 3.0; rtol=1e-8, atol=1e-8)

    model_geo = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            p = 1 / (1 + exp(-(a + η)))
            y ~ Geometric(p)
        end
    end
    df_geo = DataFrame(ID=[:A, :A], t=[0.0, 1.0], y=[0, 1])
    # At a+η=0, each observation contributes 0.5; prior contributes 1.
    @test isapprox(fisher_info_diag11(model_geo, df_geo), 2.0; rtol=1e-8, atol=1e-8)

    model_bin = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            p = 1 / (1 + exp(-(a + η)))
            y ~ Binomial(10, p)
        end
    end
    df_bin = DataFrame(ID=[:A, :A], t=[0.0, 1.0], y=[2, 4])
    # At a+η=0, each observation contributes 2.5; prior contributes 1.
    @test isapprox(fisher_info_diag11(model_bin, df_bin), 6.0; rtol=1e-8, atol=1e-8)
end

@testset "FOCEI fisher_common supports LogNormal and Exponential RE priors" begin
    function fisher_re_info_diag11(model, df)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
        ll_cache = build_ll_cache(dm)
        θ0 = get_θ0_untransformed(model.fixed.fixed)
        info = NoLimits.FOCEIInformationOptions(:fisher_common, 1e-8, 6, 10.0, true, 1e-8, true, :exact_hessian)
        b = zeros(eltype(θ0), batch_infos[1].n_b)
        I = NoLimits._focei_information_matrix(dm, batch_infos[1], θ0, b, const_cache, ll_cache, info)
        @test size(I) == (1, 1)
        return I[1, 1]
    end

    μ_re = 0.2
    σ_re = 0.5
    model_logn = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
            σ = RealNumber(1.0, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(LogNormal(0.2, 0.5); column=:ID)
        end

        @formulas begin
            # Keep outcome independent of η to isolate RE-prior information.
            y ~ Normal(a, σ)
        end
    end
    df_logn = DataFrame(ID=[:A, :A], t=[0.0, 1.0], y=[0.1, 0.2])
    expected_logn = exp(-2 * μ_re + 2 * (σ_re^2)) * (1 + inv(σ_re^2))
    @test isapprox(fisher_re_info_diag11(model_logn, df_logn), expected_logn; rtol=1e-8, atol=1e-8)

    θ_exp = 2.0
    model_exp = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
            σ = RealNumber(1.0, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Exponential(2.0); column=:ID)
        end

        @formulas begin
            # Keep outcome independent of η to isolate RE-prior information.
            y ~ Normal(a, σ)
        end
    end
    df_exp = DataFrame(ID=[:A, :A], t=[0.0, 1.0], y=[0.1, 0.2])
    expected_exp = inv(θ_exp^2)
    @test isapprox(fisher_re_info_diag11(model_exp, df_exp), expected_exp; rtol=1e-8, atol=1e-8)

    # Fit-path checks: ensure FOCEI no longer errors for these RE priors in fisher_common.
    model_logn_fit = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(LogNormal(0.0, 0.6); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df_logn_fit = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[1.0, 1.1, 0.9, 1.2])
    dm_logn_fit = DataModel(model_logn_fit, df_logn_fit; primary_id=:ID, time_col=:t)
    res_logn = fit_model(dm_logn_fit, NoLimits.FOCEI(; optim_kwargs=(maxiters=1,),
                                                             inner_kwargs=(maxiters=20,),
                                                             multistart_n=0,
                                                             multistart_k=0,
                                                             info_mode=:fisher_common))
    @test res_logn isa FitResult

    model_exp_fit = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Exponential(1.5); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df_exp_fit = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[1.0, 1.1, 0.9, 1.2])
    dm_exp_fit = DataModel(model_exp_fit, df_exp_fit; primary_id=:ID, time_col=:t)
    res_exp = fit_model(dm_exp_fit, NoLimits.FOCEI(; optim_kwargs=(maxiters=1,),
                                                           inner_kwargs=(maxiters=20,),
                                                           multistart_n=0,
                                                           multistart_k=0,
                                                           info_mode=:fisher_common))
    @test res_exp isa FitResult
end

@testset "FOCEI fisher_common errors for unsupported outcomes" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            γ = RealNumber(1.0, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Cauchy(a + η, γ)
        end
    end

    df = DataFrame(
        ID = [:A, :A],
        t = [0.0, 1.0],
        y = [0.1, -0.2]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θ0 = get_θ0_untransformed(model.fixed.fixed)
    info = NoLimits.FOCEIInformationOptions(:fisher_common, 1e-8, 6, 10.0, true, 1e-8, true, :exact_hessian)
    b = zeros(eltype(θ0), batch_infos[1].n_b)
    @test_throws "Automatic OPG fallback has been removed" NoLimits._focei_information_matrix(
        dm, batch_infos[1], θ0, b, const_cache, ll_cache, info
    )
end

@testset "FOCEI custom information callback is supported" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
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
        ID = [:A, :A],
        t = [0.0, 1.0],
        y = [0.2, 0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θ0 = get_θ0_untransformed(model.fixed.fixed)
    b = zeros(eltype(θ0), batch_infos[1].n_b)

    info_custom = NoLimits.FOCEIInformationOptions(:custom, 1e-8, 6, 10.0, true, 1e-8, true, :exact_hessian,
        (dm, batch_info, θ, b, const_cache, ll_cache) -> begin
            T = promote_type(eltype(θ), eltype(b))
            I = zeros(T, batch_info.n_b, batch_info.n_b)
            I[1, 1] = T(2.5)
            I
        end
    )
    I = NoLimits._focei_information_matrix(dm, batch_infos[1], θ0, b, const_cache, ll_cache, info_custom)
    @test size(I) == (1, 1)
    @test isapprox(I[1, 1], 2.5; rtol=1e-12, atol=1e-12)

    info_bad = NoLimits.FOCEIInformationOptions(:custom, 1e-8, 6, 10.0, true, 1e-8, true, :exact_hessian,
        (dm, batch_info, θ, b, const_cache, ll_cache) -> zeros(eltype(θ), batch_info.n_b + 1, batch_info.n_b + 1)
    )
    @test_throws "info_custom" NoLimits._focei_information_matrix(
        dm, batch_infos[1], θ0, b, const_cache, ll_cache, info_bad
    )
end

@testset "FOCEI mode_sensitivity=:focei_info matches exact on Gaussian model" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
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
        y = [0.2, 0.1, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θ0 = get_θ0_untransformed(model.fixed.fixed)
    ad_cache = NoLimits._init_laplace_ad_cache(length(batch_infos))
    b = zeros(eltype(θ0), batch_infos[1].n_b)
    hess = NoLimits.LaplaceHessianOptions(1e-6, 6, 10.0, true, 1e-6, true, false, 8)

    info_exact = NoLimits.FOCEIInformationOptions(:fisher_common, 1e-8, 6, 10.0, true, 1e-8, true, :exact_hessian)
    info_fast = NoLimits.FOCEIInformationOptions(:fisher_common, 1e-8, 6, 10.0, true, 1e-8, true, :focei_info)

    res_exact = NoLimits._focei_grad_batch(dm, batch_infos[1], θ0, b, const_cache, ll_cache, ad_cache, 1;
                                                   info_opts=info_exact,
                                                   fallback_hessian=hess)
    res_fast = NoLimits._focei_grad_batch(dm, batch_infos[1], θ0, b, const_cache, ll_cache, ad_cache, 1;
                                                  info_opts=info_fast,
                                                  fallback_hessian=hess)
    @test isfinite(res_exact.logdet)
    @test isfinite(res_fast.logdet)
    @test isapprox(res_fast.logdet, res_exact.logdet; rtol=1e-10, atol=1e-10)
    @test isapprox(collect(res_fast.grad), collect(res_exact.grad); rtol=1e-6, atol=1e-6)

    res_fit = fit_model(dm, NoLimits.FOCEI(; optim_kwargs=(maxiters=1,),
                                                   inner_kwargs=(maxiters=10,),
                                                   multistart_n=0,
                                                   multistart_k=0,
                                                   info_mode=:fisher_common,
                                                   mode_sensitivity=:focei_info))
    @test res_fit isa FitResult
end
