using Test
using DataFrames
using NoLimits
using FiniteDifferences
using LineSearches
using OptimizationBBO
using Optimization
using OptimizationOptimJL
using Distributions
using ComponentArrays
using LinearAlgebra
using Random

@testset "Laplace fit (non-ODE) runs and returns EB modes" begin
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
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=3,)))
    @test res.result.eb_modes !== nothing
    @test length(res.result.eb_modes) == length(get_batches(dm))
end

@testset "Laplace fit (ODE) runs" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.4, scale=:log)
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

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [0.9, 0.7]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,)))
    @test res.summary.converged isa Bool
end

@testset "Laplace objective gradient matches finite differences" begin
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
    pairing, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    info = batch_infos[1]
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

    function laplace_obj(θ_vec)
        θ = ComponentArray(θ_vec, getaxes(θ0))
        obj, _, _ = NoLimits._laplace_objective_and_grad(dm, batch_infos, θ, const_cache, ll_cache, ebe_cache;
                                                                 inner=NoLimits.LaplaceInnerOptions(
                                                                     OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
                                                                 (maxiters=20,),
                                                                 Optimization.AutoForwardDiff(),
                                                                 1e-6),
                                                                 hessian=NoLimits.LaplaceHessianOptions(1e-6, 6, 10.0, false, 0.0, true, false, 0),
                                                                 cache_opts=NoLimits.LaplaceCacheOptions(0.0),
                                                                 multistart=NoLimits.LaplaceMultistartOptions(0, 0, 1e-6, 5, :lhs),
                                                                 rng=Random.default_rng())
        return obj
    end

    function laplace_grad(θ_vec)
        θ = ComponentArray(θ_vec, getaxes(θ0))
        _, g, _ = NoLimits._laplace_objective_and_grad(dm, batch_infos, θ, const_cache, ll_cache, ebe_cache;
                                                              inner=NoLimits.LaplaceInnerOptions(
                                                                  OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
                                                                  (maxiters=20,),
                                                                  Optimization.AutoForwardDiff(),
                                                                  1e-6),
                                                              hessian=NoLimits.LaplaceHessianOptions(1e-6, 6, 10.0, false, 0.0, true, false, 0),
                                                              cache_opts=NoLimits.LaplaceCacheOptions(0.0),
                                                              multistart=NoLimits.LaplaceMultistartOptions(0, 0, 1e-6, 5, :lhs),
                                                              rng=Random.default_rng())
        return collect(g)
    end

    fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), laplace_obj, collect(θ0))[1]
    g = laplace_grad(collect(θ0))
    @test isapprox(g, fd; rtol=1e-3, atol=1e-3)
end

@testset "Laplace objective gradient (multivariate + multiple groups + covariates)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
            μ = RealVector([0.0, 0.0])
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @randomEffects begin
            η_id = RandomEffect(MvNormal([0.0, 0.0], LinearAlgebra.I(2)); column=:ID)
            η_site = RandomEffect(MvNormal(μ, LinearAlgebra.I(2)); column=:SITE)
        end

        @formulas begin
            y ~ Normal(a + 0.5 * z + η_id[1] + η_site[2], σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        SITE = [:A, :A, :A, :A, :B, :B, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.1, 0.0],
        y = [0.0, 0.1, -0.1, 0.0, 0.2, -0.1, 0.1, 0.0]
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

    function laplace_obj(θ_vec)
        θ = ComponentArray(θ_vec, getaxes(θ0))
        obj, _, _ = NoLimits._laplace_objective_and_grad(dm, batch_infos, θ, const_cache, ll_cache, ebe_cache;
                                                                 inner=NoLimits.LaplaceInnerOptions(
                                                                     OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
                                                                 (maxiters=30,),
                                                                 Optimization.AutoForwardDiff(),
                                                                 1e-6),
                                                                 hessian=NoLimits.LaplaceHessianOptions(1e-6, 6, 10.0, false, 0.0, true, false, 0),
                                                                 cache_opts=NoLimits.LaplaceCacheOptions(0.0),
                                                                 multistart=NoLimits.LaplaceMultistartOptions(0, 0, 1e-6, 5, :lhs),
                                                                 rng=Random.default_rng())
        return obj
    end

    function laplace_grad(θ_vec)
        θ = ComponentArray(θ_vec, getaxes(θ0))
        _, g, _ = NoLimits._laplace_objective_and_grad(dm, batch_infos, θ, const_cache, ll_cache, ebe_cache;
                                                              inner=NoLimits.LaplaceInnerOptions(
                                                                  OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
                                                                  (maxiters=30,),
                                                                  Optimization.AutoForwardDiff(),
                                                                  1e-6),
                                                              hessian=NoLimits.LaplaceHessianOptions(1e-6, 6, 10.0, false, 0.0, true, false, 0),
                                                              cache_opts=NoLimits.LaplaceCacheOptions(0.0),
                                                              multistart=NoLimits.LaplaceMultistartOptions(0, 0, 1e-6, 5, :lhs),
                                                              rng=Random.default_rng())
        return collect(g)
    end

    fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), laplace_obj, collect(θ0))[1]
    g = laplace_grad(collect(θ0))
    @test isapprox(g, fd; rtol=2e-3, atol=2e-3)
end

@testset "Laplace with constants_re fixes all REs for one individual" begin
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
    pairing, batch_infos, _ = NoLimits._build_laplace_batch_infos(dm, constants_re)
    @test length(pairing.batches) == 2
    @test sort(length.(pairing.batches)) == [1, 1]
    nbs = sort([info.n_b for info in batch_infos])
    @test nbs == [0, 2]

    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,)), constants_re=constants_re)
    @test res.summary.converged isa Bool

    re_dfs = get_laplace_random_effects(dm, res; constants_re=constants_re, flatten=true, include_constants=true)
    @test hasproperty(re_dfs, :η_id)
    @test hasproperty(re_dfs, :η_site)
    df_id = getproperty(re_dfs, :η_id)
    df_site = getproperty(re_dfs, :η_site)
    @test :ID in propertynames(df_id)
    @test :SITE in propertynames(df_site)
    @test length(df_id.ID) == 2
    @test length(df_site.SITE) == 2
end

@testset "Laplace fit single-thread vs multithread (if available)" begin
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
    method = NoLimits.Laplace(; optim_kwargs=(maxiters=3,))
    res_serial = fit_model(dm, method; serialization=EnsembleSerial(), rng=MersenneTwister(123))
    res_threads = fit_model(dm, method; serialization=EnsembleThreads(), rng=MersenneTwister(123))
    @test res_serial.summary.objective == res_threads.summary.objective
    @test collect(NoLimits.get_params(res_serial, scale=:untransformed)) ==
          collect(NoLimits.get_params(res_threads, scale=:untransformed))
end

@testset "Laplace fit with BlackBoxOptim requires bounds" begin
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
    @test_throws ErrorException fit_model(dm, NoLimits.Laplace(;
        optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
        optim_kwargs=(maxiters=5,)
    ))

    lb, ub = default_bounds_from_start(dm; margin=1.0)
    res = fit_model(dm, NoLimits.Laplace(;
        optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
        optim_kwargs=(maxiters=5,),
        lb=lb,
        ub=ub
    ))
    @test res.summary.converged isa Bool
end

@testset "Laplace multistart options" begin
    lap = NoLimits.Laplace()
    @test lap.multistart.n == 50
    @test lap.multistart.k == 10
    @test lap.multistart.sampling == :lhs
    lap_map = NoLimits.LaplaceMAP()
    @test lap_map.multistart.n == 50
    @test lap_map.multistart.k == 10
    @test lap_map.multistart.sampling == :lhs

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
    rng = MersenneTwister(1)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,),
                                                 multistart_n=2,
                                                 multistart_k=1,
                                                 multistart_grad_tol=0.0),
                    rng=rng)
    @test res isa FitResult
end

@testset "Laplace objective cache only reuses valid gradients" begin
    θ = ComponentArray((a=0.1, σ=0.2))
    axs = getaxes(θ)
    cache = NoLimits._LaplaceObjCache{Float64, ComponentArray}(nothing,
                                                                        Inf,
                                                                        ComponentArray(zeros(Float64, length(θ)), axs),
                                                                        false)

    NoLimits._laplace_obj_cache_set_obj!(cache, θ, 1.0)
    @test NoLimits._laplace_obj_cache_lookup(cache, θ, 1e9) === nothing

    grad = ComponentArray([3.0, 4.0], axs)
    NoLimits._laplace_obj_cache_set_obj_grad!(cache, θ, 2.0, grad)
    hit = NoLimits._laplace_obj_cache_lookup(cache, θ, 0.0)
    @test hit !== nothing
    @test hit[1] == 2.0
    @test collect(hit[2]) == collect(grad)
end

@testset "Laplace threaded cache fallback preserves ODE options" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.4, scale=:log)
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

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [0.9, 0.7]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    ll_cache = build_ll_cache(dm; ode_kwargs=(abstol=1e-8, reltol=1e-7))
    threaded = NoLimits._laplace_thread_caches(dm, ll_cache, 2)

    @test length(threaded) == 2
    @test all(c -> c.ode_args == ll_cache.ode_args, threaded)
    @test all(c -> c.ode_kwargs == ll_cache.ode_kwargs, threaded)
end

@testset "Laplace fit non-normal Poisson outcome" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.7); column=:ID)
        end

        @formulas begin
            λ = exp(a + b * z + η)
            y ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.0, 0.2, 0.4, 0.5, 0.8, 1.0],
        y = [1, 1, 2, 2, 3, 4]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,)))

    @test res isa FitResult
    @test res.summary.converged isa Bool
    @test res.result.eb_modes !== nothing
    @test length(res.result.eb_modes) == length(get_batches(dm))
end
