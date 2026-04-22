using Test
using DataFrames
using NoLimits
using Distributions
using Random
using ComponentArrays
using Optimization
using OptimizationOptimJL
using LineSearches

function _lfp_make_df(kind::Symbol; n_id::Int=12, n_obs::Int=5, seed::Int=123)
    rng = MersenneTwister(seed)
    n = n_id * n_obs
    ID = repeat(1:n_id, inner=n_obs)
    t = repeat(collect(0.0:(n_obs - 1)), n_id)
    z = randn(rng, n)
    η = 0.6 .* randn(rng, n_id)

    if kind == :gaussian
        y = Vector{Float64}(undef, n)
        for i in eachindex(y)
            lin = 0.2 + 0.7 * z[i] + η[ID[i]]
            y[i] = lin + 0.3 * randn(rng)
        end
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :lognormal
        y = Vector{Float64}(undef, n)
        for i in eachindex(y)
            lin = -0.1 + 0.5 * z[i] + η[ID[i]]
            y[i] = exp(lin + 0.25 * randn(rng))
        end
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :bernoulli
        y = Vector{Int}(undef, n)
        for i in eachindex(y)
            lin = 0.1 + 0.6 * z[i] + η[ID[i]]
            p = inv(1 + exp(-lin))
            y[i] = rand(rng) < p ? 1 : 0
        end
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :poisson
        y = Vector{Int}(undef, n)
        for i in eachindex(y)
            lin = -0.2 + 0.4 * z[i] + 0.7 * η[ID[i]]
            y[i] = rand(rng, Poisson(exp(lin)))
        end
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :ode_offset
        y = Vector{Float64}(undef, n)
        for i in eachindex(y)
            # x1(t) for D(x1)=-k*x1 with k≈0.3 and x1(0)=1 plus additive RE
            x1t = exp(-0.3 * t[i])
            y[i] = x1t + η[ID[i]] + 0.2 * randn(rng)
        end
        return DataFrame(ID=ID, t=t, y=y)
    elseif kind == :ode_eta
        y = Vector{Float64}(undef, n)
        for i in eachindex(y)
            # surrogate generation only; model uses η in ODE
            x1t = exp(-(0.3 + 0.2 * η[ID[i]]) * t[i])
            y[i] = x1t + 0.2 * randn(rng)
        end
        return DataFrame(ID=ID, t=t, y=y)
    else
        error("Unknown fixture kind: $(kind)")
    end
end

function _lfp_model(kind::Symbol)
    if kind == :gaussian
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                σ = RealNumber(0.5, scale=:log)
                τ = RealNumber(0.7, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + b * z + η, σ)
            end
        end
    elseif kind == :lognormal
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                σ = RealNumber(0.4, scale=:log)
                τ = RealNumber(0.7, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                y ~ LogNormal(a + b * z + η, σ)
            end
        end
    elseif kind == :bernoulli
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                τ = RealNumber(0.7, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                p = logistic(a + b * z + η)
                y ~ Bernoulli(p)
            end
        end
    elseif kind == :poisson
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                τ = RealNumber(0.7, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                λ = exp(a + b * z + η)
                y ~ Poisson(λ)
            end
        end
    elseif kind == :ode_offset
        return @Model begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                k = RealNumber(0.3, scale=:log)
                σ = RealNumber(0.3, scale=:log)
                τ = RealNumber(0.6, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @DifferentialEquation begin
                D(x1) ~ -k * x1
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t) + η, σ)
            end
        end
    elseif kind == :ode_eta
        return @Model begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                k = RealNumber(0.3, scale=:log)
                σ = RealNumber(0.3, scale=:log)
                τ = RealNumber(0.6, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @DifferentialEquation begin
                D(x1) ~ -(k + η) * x1
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t), σ)
            end
        end
    else
        error("Unknown model kind: $(kind)")
    end
end

function _lfp_setup_objgrad(dm::DataModel)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    n_batches = length(batch_infos)
    Tθ = eltype(θ0)
    bstar_cache = NoLimits._LaplaceBStarCache([Vector{Tθ}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = NoLimits._LaplaceGradCache([Vector{Tθ}() for _ in 1:n_batches],
                                            fill(Tθ(NaN), n_batches),
                                            [Vector{Tθ}() for _ in 1:n_batches],
                                            falses(n_batches))
    ad_cache = NoLimits._init_laplace_ad_cache(n_batches)
    hess_cache = NoLimits._init_laplace_hess_cache(Tθ, n_batches)
    ebe_cache = NoLimits._LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)
    return (; batch_infos, const_cache, ll_cache, θ0, ebe_cache)
end

function _lfp_eval_objgrad(dm::DataModel; seed::Int=44)
    s = _lfp_setup_objgrad(dm)
    inner = NoLimits.LaplaceInnerOptions(
        OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking(maxstep=1.0)),
        (maxiters=2,),
        Optimization.AutoForwardDiff(),
        1e-6
    )
    hess = NoLimits.LaplaceHessianOptions(1e-6, 6, 10.0, true, 1e-6, true, false, 8)
    cache_opts = NoLimits.LaplaceCacheOptions(0.0)
    ms = NoLimits.LaplaceMultistartOptions(0, 0, 1e-6, 1, :lhs)
    rng = MersenneTwister(seed)
    obj, grad, _ = NoLimits._laplace_objective_and_grad(dm,
                                                         s.batch_infos,
                                                         s.θ0,
                                                         s.const_cache,
                                                         s.ll_cache,
                                                         s.ebe_cache;
                                                         inner=inner,
                                                         hessian=hess,
                                                         cache_opts=cache_opts,
                                                         multistart=ms,
                                                         rng=rng,
                                                         serialization=EnsembleSerial())
    return obj, collect(grad)
end

@testset "Laplace fastpath baseline fixtures are deterministic" begin
    for kind in (:gaussian, :lognormal, :bernoulli, :poisson, :ode_offset, :ode_eta)
        df1 = _lfp_make_df(kind; seed=123)
        df2 = _lfp_make_df(kind; seed=123)
        @test names(df1) == names(df2)
        for col in names(df1)
            @test all(df1[!, col] .== df2[!, col])
        end
    end
end

@testset "Laplace objective+gradient baseline reproducibility" begin
    for kind in (:gaussian, :lognormal, :bernoulli, :poisson)
        model = _lfp_model(kind)
        df = _lfp_make_df(kind; seed=321)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        obj1, g1 = _lfp_eval_objgrad(dm; seed=999)
        obj2, g2 = _lfp_eval_objgrad(dm; seed=999)
        @test isapprox(obj1, obj2; atol=1e-10, rtol=1e-10)
        @test length(g1) == length(g2)
        @test isapprox(g1, g2; atol=1e-8, rtol=1e-8)
    end
end

@testset "ODE fixture baseline (offset only vs RE in ODE) both run" begin
    for kind in (:ode_offset, :ode_eta)
        model = _lfp_model(kind)
        model = set_solver_config(model; saveat_mode=:saveat)
        df = _lfp_make_df(kind; seed=222)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=2,), multistart_n=2, multistart_k=2))
        @test res.summary.converged isa Bool
    end
end
