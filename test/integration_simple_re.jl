using Test
using DataFrames
using NoLimits
using Distributions
using ComponentArrays
using LinearAlgebra
using Random
using Turing
using FiniteDifferences
using LineSearches
using OptimizationBBO
using Optimization
using OptimizationOptimJL

# ─────────────────────────────────────────────────────────────────────────────
# Integration file: tests for models WITH a simple scalar random effect.
# Absorbed from: estimation_laplace_fit_tests.jl, estimation_laplace_map_tests.jl,
#   estimation_vi_tests.jl (RE testset), accessors_tests.jl (RE parts),
#   serialization_tests.jl (Laplace parts).
#
# Shared fixtures build the DataModel and run Laplace / LaplaceMAP fits ONCE;
# downstream testsets reuse the results.
# ─────────────────────────────────────────────────────────────────────────────

# ── shared fixtures ───────────────────────────────────────────────────────────

# No-prior RE model: y ~ Normal(a + η, σ)  —  used for Laplace + MCEM/SAEM accessors
const _RE_DF = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])

const _RE_DM = DataModel(
    @Model(begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin; y ~ Normal(a + η, σ); end
    end),
    _RE_DF; primary_id=:ID, time_col=:t)

# With-priors RE model — used for LaplaceMAP, VI, and serialization.
const _RE_DM_P = DataModel(
    @Model(begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2; prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log; prior=LogNormal(0.0, 0.5))
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin; y ~ Normal(a + η, σ); end
    end),
    _RE_DF; primary_id=:ID, time_col=:t)

# Pre-fit once; shared by basic Laplace/LaplaceMAP testsets.
const _RE_RES_LAP    = fit_model(_RE_DM,   NoLimits.Laplace(;    optim_kwargs=(maxiters=8,), multistart_n=0, multistart_k=0))
const _RE_RES_LAPMAP = fit_model(_RE_DM_P, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=8,), multistart_n=0, multistart_k=0))

# ══════════════════════════════════════════════════════════════════════════════
# Laplace — basic fitting
# ══════════════════════════════════════════════════════════════════════════════

@testset "Laplace fit (non-ODE) runs and returns EB modes" begin
    @test _RE_RES_LAP.result.eb_modes !== nothing
    @test length(_RE_RES_LAP.result.eb_modes) == length(get_batches(_RE_DM))
end

@testset "Laplace fit non-normal Poisson outcome" begin
    model = @Model begin
        @covariates begin; t = Covariate(); z = Covariate(); end
        @fixedEffects begin; a = RealNumber(0.1); b = RealNumber(0.2); end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 0.7); column=:ID); end
        @formulas begin; λ = exp(a + b * z + η); y ~ Poisson(λ); end
    end
    df = DataFrame(ID=[:A,:A,:B,:B,:C,:C], t=[0.0,1.0,0.0,1.0,0.0,1.0],
                   z=[0.0,0.2,0.4,0.5,0.8,1.0], y=[1,1,2,2,3,4])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,)))
    @test res isa FitResult
    @test res.summary.converged isa Bool
    @test res.result.eb_modes !== nothing
    @test length(res.result.eb_modes) == length(get_batches(dm))
end

# ══════════════════════════════════════════════════════════════════════════════
# Laplace — internal API & gradient tests
# ══════════════════════════════════════════════════════════════════════════════

@testset "Laplace objective gradient matches finite differences" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin; a = RealNumber(0.2); σ = RealNumber(0.5, scale=:log); end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:GROUP); end
        @formulas begin; y ~ Normal(a + η, σ); end
    end
    df = DataFrame(ID=[1,1,2,2], GROUP=[:A,:A,:A,:A],
                   t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θ0 = get_θ0_untransformed(model.fixed.fixed)
    Tθ = eltype(θ0)
    n = length(batch_infos)
    bstar = NoLimits._LaplaceBStarCache([Vector{Tθ}() for _ in 1:n], falses(n))
    grad  = NoLimits._LaplaceGradCache([Vector{Tθ}() for _ in 1:n], fill(Tθ(NaN), n),
                                       [Vector{Tθ}() for _ in 1:n], falses(n))
    ebe   = NoLimits._LaplaceCache(nothing, bstar, grad,
                                   NoLimits._init_laplace_ad_cache(n),
                                   NoLimits._init_laplace_hess_cache(Tθ, n))
    inner = NoLimits.LaplaceInnerOptions(
        OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
        (maxiters=10,), Optimization.AutoForwardDiff(), 1e-6)
    hess  = NoLimits.LaplaceHessianOptions(1e-6, 6, 10.0, false, 0.0, true, false, 0)
    co    = NoLimits.LaplaceCacheOptions(0.0)
    ms    = NoLimits.LaplaceMultistartOptions(0, 0, 1e-6, 5, :lhs)

    function obj(v)
        θ = ComponentArray(v, getaxes(θ0))
        o, _, _ = NoLimits._laplace_objective_and_grad(dm, batch_infos, θ, const_cache, ll_cache, ebe;
                     inner=inner, hessian=hess, cache_opts=co, multistart=ms, rng=Random.default_rng())
        return o
    end
    function grd(v)
        θ = ComponentArray(v, getaxes(θ0))
        _, g, _ = NoLimits._laplace_objective_and_grad(dm, batch_infos, θ, const_cache, ll_cache, ebe;
                     inner=inner, hessian=hess, cache_opts=co, multistart=ms, rng=Random.default_rng())
        return collect(g)
    end
    fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), obj, collect(θ0))[1]
    g  = grd(collect(θ0))
    @test isapprox(g, fd; rtol=1e-3, atol=1e-3)
end

@testset "Laplace objective gradient (multivariate + multiple groups + covariates)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1); σ = RealNumber(0.4, scale=:log); μ = RealVector([0.0, 0.0])
        end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @randomEffects begin
            η_id   = RandomEffect(MvNormal([0.0, 0.0], LinearAlgebra.I(2)); column=:ID)
            η_site = RandomEffect(MvNormal(μ,           LinearAlgebra.I(2)); column=:SITE)
        end
        @formulas begin; y ~ Normal(a + 0.5 * z + η_id[1] + η_site[2], σ); end
    end
    df = DataFrame(ID=[1,1,2,2,3,3,4,4], SITE=[:A,:A,:A,:A,:B,:B,:B,:B],
                   t=[0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0],
                   z=[0.1,0.2,-0.1,0.0,0.3,-0.2,0.1,0.0],
                   y=[0.0,0.1,-0.1,0.0,0.2,-0.1,0.1,0.0])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θ0 = get_θ0_untransformed(model.fixed.fixed)
    Tθ = eltype(θ0); n = length(batch_infos)
    bstar = NoLimits._LaplaceBStarCache([Vector{Tθ}() for _ in 1:n], falses(n))
    grad  = NoLimits._LaplaceGradCache([Vector{Tθ}() for _ in 1:n], fill(Tθ(NaN), n),
                                       [Vector{Tθ}() for _ in 1:n], falses(n))
    ebe   = NoLimits._LaplaceCache(nothing, bstar, grad,
                                   NoLimits._init_laplace_ad_cache(n),
                                   NoLimits._init_laplace_hess_cache(Tθ, n))
    inner = NoLimits.LaplaceInnerOptions(
        OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
        (maxiters=15,), Optimization.AutoForwardDiff(), 1e-6)
    hess  = NoLimits.LaplaceHessianOptions(1e-6, 6, 10.0, false, 0.0, true, false, 0)
    co    = NoLimits.LaplaceCacheOptions(0.0)
    ms    = NoLimits.LaplaceMultistartOptions(0, 0, 1e-6, 5, :lhs)

    function obj(v)
        θ = ComponentArray(v, getaxes(θ0))
        o, _, _ = NoLimits._laplace_objective_and_grad(dm, batch_infos, θ, const_cache, ll_cache, ebe;
                     inner=inner, hessian=hess, cache_opts=co, multistart=ms, rng=Random.default_rng())
        return o
    end
    function grd(v)
        θ = ComponentArray(v, getaxes(θ0))
        _, g, _ = NoLimits._laplace_objective_and_grad(dm, batch_infos, θ, const_cache, ll_cache, ebe;
                     inner=inner, hessian=hess, cache_opts=co, multistart=ms, rng=Random.default_rng())
        return collect(g)
    end
    fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), obj, collect(θ0))[1]
    g  = grd(collect(θ0))
    @test isapprox(g, fd; rtol=2e-3, atol=2e-3)
end

@testset "Laplace with constants_re fixes all REs for one individual" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin; a = RealNumber(0.1); σ = RealNumber(0.4, scale=:log); end
        @randomEffects begin
            η_id   = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end
        @formulas begin; y ~ Normal(a + η_id + η_site, σ); end
    end
    df = DataFrame(ID=[:id1,:id1,:id2,:id2], SITE=[:A,:A,:B,:B],
                   t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])
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
    @test hasproperty(re_dfs, :η_id) && hasproperty(re_dfs, :η_site)
    @test :ID   in propertynames(getproperty(re_dfs, :η_id))
    @test :SITE in propertynames(getproperty(re_dfs, :η_site))
    @test length(getproperty(re_dfs, :η_id).ID) == 2
end

@testset "Laplace fit single-thread vs multithread (if available)" begin
    Threads.nthreads() < 2 && return
    df = DataFrame(ID=[1,1,2,2,3,3,4,4], t=[0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0],
                   y=[0.1,0.2,0.0,-0.1,0.05,0.0,-0.05,0.1])
    dm = DataModel(_RE_DM.model, df; primary_id=:ID, time_col=:t)
    method = NoLimits.Laplace(; optim_kwargs=(maxiters=3,))
    res_s = fit_model(dm, method; serialization=EnsembleSerial(),  rng=MersenneTwister(123))
    res_t = fit_model(dm, method; serialization=EnsembleThreads(), rng=MersenneTwister(123))
    @test res_s.summary.objective == res_t.summary.objective
    @test collect(NoLimits.get_params(res_s, scale=:untransformed)) ==
          collect(NoLimits.get_params(res_t, scale=:untransformed))
end

@testset "Laplace fit with BlackBoxOptim requires bounds" begin
    @test_throws ErrorException fit_model(_RE_DM, NoLimits.Laplace(;
        optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
        optim_kwargs=(maxiters=5,)))
    lb, ub = default_bounds_from_start(_RE_DM; margin=1.0)
    res = fit_model(_RE_DM, NoLimits.Laplace(;
        optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
        optim_kwargs=(maxiters=5,), lb=lb, ub=ub))
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

    res = fit_model(_RE_DM, NoLimits.Laplace(; optim_kwargs=(maxiters=2,),
                                               multistart_n=2, multistart_k=1,
                                               multistart_grad_tol=0.0),
                    rng=MersenneTwister(1))
    @test res isa FitResult
end

@testset "Laplace objective cache only reuses valid gradients" begin
    θ = ComponentArray((a=0.1, σ=0.2))
    axs = getaxes(θ)
    cache = NoLimits._LaplaceObjCache{Float64, ComponentArray}(
        nothing, Inf, ComponentArray(zeros(Float64, length(θ)), axs), false)
    NoLimits._laplace_obj_cache_set_obj!(cache, θ, 1.0)
    @test NoLimits._laplace_obj_cache_lookup(cache, θ, 1e9) === nothing
    grad = ComponentArray([3.0, 4.0], axs)
    NoLimits._laplace_obj_cache_set_obj_grad!(cache, θ, 2.0, grad)
    hit = NoLimits._laplace_obj_cache_lookup(cache, θ, 0.0)
    @test hit !== nothing
    @test hit[1] == 2.0
    @test collect(hit[2]) == collect(grad)
end

# ══════════════════════════════════════════════════════════════════════════════
# Laplace — ODE model
# ══════════════════════════════════════════════════════════════════════════════

# shared ODE fixture — both ODE testsets use the same model type
const _RE_ODE_DM = DataModel(
    set_solver_config(
        @Model(begin
            @covariates begin; t = Covariate(); end
            @fixedEffects begin; a = RealNumber(0.3); σ = RealNumber(0.4, scale=:log); end
            @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:ID); end
            @DifferentialEquation begin; D(x1) ~ -a * x1 + η; end
            @initialDE begin; x1 = 1.0; end
            @formulas begin; y ~ Normal(x1(t), σ); end
        end); saveat_mode=:saveat),
    DataFrame(ID=[1,1], t=[0.0,1.0], y=[0.9,0.7]); primary_id=:ID, time_col=:t)

@testset "Laplace fit (ODE) runs" begin
    res = fit_model(_RE_ODE_DM, NoLimits.Laplace(; optim_kwargs=(maxiters=2,)))
    @test res.summary.converged isa Bool
end

@testset "Laplace threaded cache fallback preserves ODE options" begin
    dm = _RE_ODE_DM
    ll_cache = build_ll_cache(dm; ode_kwargs=(abstol=1e-8, reltol=1e-7))
    threaded = NoLimits._laplace_thread_caches(dm, ll_cache, 2)
    @test length(threaded) == 2
    @test all(c -> c.ode_args == ll_cache.ode_args, threaded)
    @test all(c -> c.ode_kwargs == ll_cache.ode_kwargs, threaded)
end

# ══════════════════════════════════════════════════════════════════════════════
# LaplaceMAP
# ══════════════════════════════════════════════════════════════════════════════

@testset "LaplaceMAP requires priors on all fixed effects" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log)  # missing prior → error
        end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:ID); end
        @formulas begin; y ~ Normal(a + η, σ); end
    end
    df = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=2,)))
end

@testset "LaplaceMAP runs with priors and penalties" begin
    # Reuse shared LaplaceMAP result to check basic success.
    @test _RE_RES_LAPMAP.summary.converged isa Bool
    # Also verify penalty option path works.
    res = fit_model(_RE_DM_P, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=3,));
                   penalty=(; a=0.1))
    @test res.summary.converged isa Bool
end

@testset "LaplaceMAP with multivariate REs and multiple groups" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.5))
            μ = RealVector([0.0, 0.0], prior=MvNormal([0.0, 0.0], LinearAlgebra.I(2)))
        end
        @randomEffects begin
            η_id   = RandomEffect(MvNormal([0.0, 0.0], LinearAlgebra.I(2)); column=:ID)
            η_site = RandomEffect(MvNormal(μ,           LinearAlgebra.I(2)); column=:SITE)
        end
        @formulas begin; y ~ Normal(a + η_id[1] + η_site[2], σ); end
    end
    df = DataFrame(ID=[1,1,2,2,3,3,4,4], SITE=[:A,:A,:A,:A,:B,:B,:B,:B],
                   t=[0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0],
                   y=[0.1,0.2,0.0,-0.1,0.05,0.0,-0.05,0.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=3,)))
    @test res.summary.converged isa Bool
end

@testset "LaplaceMAP with ODE model" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.3, prior=Normal(0.3, 0.3))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:ID); end
        @DifferentialEquation begin; D(x1) ~ -a * x1 + η; end
        @initialDE begin; x1 = 1.0; end
        @formulas begin; y ~ Normal(x1(t), σ); end
    end
    df = DataFrame(ID=[1,1], t=[0.0,1.0], y=[0.9,0.7])
    dm = DataModel(set_solver_config(model; saveat_mode=:saveat), df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=2,)))
    @test res.summary.converged isa Bool
end

@testset "LaplaceMAP normal prior equals penalty" begin
    model_prior = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.0, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:ID); end
        @formulas begin; y ~ Normal(a + η, σ); end
    end
    model_pen = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin; a = RealNumber(0.0); σ = RealNumber(0.5, scale=:log); end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:ID); end
        @formulas begin; y ~ Normal(a + η, σ); end
    end
    dm_prior = DataModel(model_prior, _RE_DF; primary_id=:ID, time_col=:t)
    dm_pen   = DataModel(model_pen,   _RE_DF; primary_id=:ID, time_col=:t)
    res_prior = fit_model(dm_prior, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=3,)), constants=(; σ=0.5))
    res_pen   = fit_model(dm_pen,   NoLimits.Laplace(;    optim_kwargs=(maxiters=3,)), penalty=(; a=0.5), constants=(; σ=0.5))
    θ_prior = NoLimits.get_params(res_prior; scale=:untransformed)
    θ_pen   = NoLimits.get_params(res_pen;   scale=:untransformed)
    @test isapprox(θ_prior.a, θ_pen.a; rtol=1e-3, atol=1e-3)
end

@testset "LaplaceMAP non-normal Bernoulli outcome" begin
    model = @Model begin
        @covariates begin; t = Covariate(); z = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            b = RealNumber(0.2, prior=Normal(0.0, 1.0))
        end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 0.8); column=:ID); end
        @formulas begin; p = logistic(a + b * z + η); y ~ Bernoulli(p); end
    end
    df = DataFrame(ID=[1,1,2,2,3,3], t=[0.0,1.0,0.0,1.0,0.0,1.0],
                   z=[0.1,0.2,-0.2,0.1,0.4,0.7], y=[0,1,0,0,1,1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=2,)))
    @test res isa FitResult
    @test res.summary.converged isa Bool
    @test isfinite(NoLimits.get_objective(res))
end

# ══════════════════════════════════════════════════════════════════════════════
# VI with random effects
# ══════════════════════════════════════════════════════════════════════════════

@testset "VI supports random effects + RE-only sampling + constants_re validation" begin
    dm = DataModel(
        @Model(begin
            @covariates begin; t = Covariate(); end
            @fixedEffects begin
                a = RealNumber(0.2, prior=Normal(0.0, 1.0))
                σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
            end
            @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:ID); end
            @formulas begin; y ~ Normal(a + η, σ); end
        end),
        DataFrame(ID=[:A,:A,:B,:B], t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1]);
        primary_id=:ID, time_col=:t)

    res = fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=15, progress=false));
                    rng=Random.Xoshiro(10))
    @test res isa FitResult
    @test res.result isa NoLimits.VIResult
    @test isfinite(NoLimits.get_objective(res))
    draws = NoLimits.sample_posterior(res; n_draws=5, rng=Random.Xoshiro(11), return_names=true)
    @test any(startswith(string(n), "η_vals[") for n in draws.names)

    res_re = fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=10, progress=false));
                       constants=(a=0.2, σ=0.5), rng=Random.Xoshiro(12))
    @test res_re isa FitResult
    draws_re = NoLimits.sample_posterior(res_re; n_draws=4, rng=Random.Xoshiro(13), return_names=true)
    @test all(startswith(string(n), "η_vals[") for n in draws_re.names)

    err = try
        fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=5, progress=false));
                  constants_re=(; η=(; A=[0.0],)))
        nothing
    catch e; e end
    @test err isa ErrorException
    @test occursin("scalar number", sprint(showerror, err))
end

# ══════════════════════════════════════════════════════════════════════════════
# Accessors (RE methods)
# ══════════════════════════════════════════════════════════════════════════════

@testset "Accessors (Laplace/LaplaceMAP)" begin
    res = _RE_RES_LAP
    re  = NoLimits.get_random_effects(res)
    @test !isempty(re)
    @test isfinite(NoLimits.get_loglikelihood(res))

    res_map = _RE_RES_LAPMAP
    re_map  = NoLimits.get_random_effects(res_map)
    @test !isempty(re_map)
    @test isfinite(NoLimits.get_loglikelihood(res_map))
end

@testset "Accessors (MCEM/SAEM)" begin
    res_mcem = fit_model(_RE_DM, NoLimits.MCEM(;
        sampler=MH(), turing_kwargs=(n_samples=6, n_adapt=0, progress=false), maxiters=2))
    @test !isempty(NoLimits.get_random_effects(res_mcem))
    @test res_mcem.result.eb_modes !== nothing

    res_saem = fit_model(_RE_DM, NoLimits.SAEM(;
        sampler=MH(), turing_kwargs=(n_samples=3, n_adapt=0, progress=false),
        q_store_max=4, maxiters=2))
    @test !isempty(NoLimits.get_random_effects(res_saem))
    @test res_saem.result.eb_modes !== nothing
end

# ══════════════════════════════════════════════════════════════════════════════
# Serialization (RE / Laplace)
# ══════════════════════════════════════════════════════════════════════════════

@testset "Serialization Laplace" begin
    res  = _RE_RES_LAP
    path = tempname() * ".jld2"
    save_fit(path, res; include_data=true)
    res2 = load_fit(path; model=_RE_DM.model)
    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_iterations(res) == get_iterations(res2)
    re1 = get_random_effects(_RE_DM, res)
    re2 = get_random_effects(res2)
    @test re1.η == re2.η
    ll1 = get_loglikelihood(_RE_DM, res)
    ll2 = get_loglikelihood(res2)
    @test ll1 ≈ ll2
end

@testset "Cross-session Laplace: params + RE + LL" begin
    res  = _RE_RES_LAP
    path = tempname() * ".jld2"
    save_fit(path, res; include_data=true)
    expected_obj    = get_objective(res)
    expected_ll     = get_loglikelihood(_RE_DM, res)
    expected_n_ids  = nrow(get_random_effects(_RE_DM, res).η)

    script = """
    using NoLimits, Distributions, DataFrames
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin t = Covariate() end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin y ~ Normal(a + η, σ) end
    end
    res = load_fit($(repr(path)); model=model)
    println(get_objective(res))
    println(get_method(res).kind)
    println(get_loglikelihood(res))
    re = get_random_effects(res)
    println(nrow(re.η))
    """
    script_path = tempname() * ".jl"
    write(script_path, script)
    out   = readchomp(`$(Base.julia_cmd()) --project=$(pkgdir(NoLimits)) $(script_path)`)
    lines = split(strip(out), '\n'; keepempty=false)
    @test length(lines) >= 4
    @test parse(Float64, strip(lines[end-3])) ≈ expected_obj    atol=1e-10
    @test strip(lines[end-2]) == "laplace"
    @test parse(Float64, strip(lines[end-1])) ≈ expected_ll     atol=1e-8
    @test parse(Int,     strip(lines[end]))   == expected_n_ids
end
