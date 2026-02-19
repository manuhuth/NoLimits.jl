using Test
using DataFrames
using NoLimits
using ComponentArrays
using Random

@testset "Hutchinson logdet gradient approximates trace" begin
    Random.seed!(1234)
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
    pairing, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θu = get_θ0_untransformed(model.fixed.fixed)

    n_batches = length(batch_infos)
    T = eltype(θu)
    bstar_cache = NoLimits._LaplaceBStarCache([Vector{T}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = NoLimits._LaplaceGradCache([Vector{T}() for _ in 1:n_batches],
                                                    fill(T(NaN), n_batches),
                                                    [Vector{T}() for _ in 1:n_batches],
                                                    falses(n_batches))
    ad_cache = NoLimits._init_laplace_ad_cache(n_batches)
    hess_cache = NoLimits._init_laplace_hess_cache(T, n_batches)
    ebe_cache = NoLimits._LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)

    info = batch_infos[1]
    b = NoLimits._laplace_default_b0(dm, info, θu, const_cache, ll_cache)

    res_exact = NoLimits._laplace_grad_batch(dm, info, θu, b, const_cache, ll_cache, ebe_cache.ad_cache, 1;
                                                     use_trace_logdet_grad=true,
                                                     use_hutchinson=false)
    res_hutch = NoLimits._laplace_grad_batch(dm, info, θu, b, const_cache, ll_cache, ebe_cache.ad_cache, 1;
                                                     use_trace_logdet_grad=true,
                                                     use_hutchinson=true,
                                                     hutchinson_n=16)

    denom = max(norm(res_exact.grad), eps())
    rel_err = norm(res_hutch.grad - res_exact.grad) / denom
    @test rel_err < 0.6
end

@testset "Hutchinson gradients are driven by passed rng" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η1 = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η2 = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η1 + η2, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.3, 0.2, 0.05, -0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θu = get_θ0_untransformed(model.fixed.fixed)
    method = NoLimits.Laplace(; use_hutchinson=true, hutchinson_n=1)

    function make_ebe_cache()
        n_batches = length(batch_infos)
        T = eltype(θu)
        bstar_cache = NoLimits._LaplaceBStarCache([Vector{T}() for _ in 1:n_batches], falses(n_batches))
        grad_cache = NoLimits._LaplaceGradCache([Vector{T}() for _ in 1:n_batches],
                                                        fill(T(NaN), n_batches),
                                                        [Vector{T}() for _ in 1:n_batches],
                                                        falses(n_batches))
        ad_cache = NoLimits._init_laplace_ad_cache(n_batches)
        hess_cache = NoLimits._init_laplace_hess_cache(T, n_batches)
        return NoLimits._LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)
    end

    function eval_grad(global_seed::Int, rng_seed::Int)
        Random.seed!(global_seed)
        ebe_cache = make_ebe_cache()
        _, g, _ = NoLimits._laplace_objective_and_grad(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                                                inner=method.inner,
                                                                hessian=method.hessian,
                                                                cache_opts=method.cache,
                                                                multistart=method.multistart,
                                                                rng=MersenneTwister(rng_seed))
        return collect(g)
    end

    g1 = eval_grad(1, 123)
    g2 = eval_grad(2, 123)
    g3 = eval_grad(1, 999)

    @test isapprox(g1, g2; atol=1e-12, rtol=1e-12)
    @test maximum(abs.(g1 .- g3)) > 1e-6
end
