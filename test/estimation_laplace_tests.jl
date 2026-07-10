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
using SciMLBase
using OrdinaryDiffEq

# Consolidated Laplace tests (merges the former estimation_laplace,
# estimation_laplace_fit, estimation_hutchinson and estimation_newton_inner
# files). Standard structures
# reuse the shared fixtures (fit/model built once); bespoke @Models are kept only
# where a test specifically exercises that structure or an error path.

# Fresh Laplace EBE cache scaffold (shared by the FD-gradient and Hutchinson tests).
function _make_laplace_ebe_cache(T::Type, n_batches::Int)
    bstar_cache = NoLimits._LaplaceBStarCache(
        [Vector{T}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = NoLimits._LaplaceGradCache([Vector{T}() for _ in 1:n_batches],
        fill(T(NaN), n_batches),
        [Vector{T}() for _ in 1:n_batches],
        falses(n_batches))
    ad_cache = NoLimits._init_laplace_ad_cache(n_batches)
    hess_cache = NoLimits._init_laplace_hess_cache(T, n_batches)
    return NoLimits._LaplaceCache(
        nothing, bstar_cache, grad_cache, ad_cache, hess_cache)
end

# Shared Laplace objective-gradient-vs-finite-differences check (generic in the
# model, so it runs against shared archetype DataModels).
function _laplace_grad_matches_fd(dm; rtol, atol)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θ0 = get_θ0_untransformed(NoLimits.get_model(dm).fixed.fixed)
    ebe_cache = _make_laplace_ebe_cache(eltype(θ0), length(batch_infos))
    inner = NoLimits.LaplaceInnerOptions(
        OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()),
        (maxiters = 50,), Optimization.AutoForwardDiff(), 1e-6)
    hessian = NoLimits.LaplaceHessianOptions(1e-6, 6, 10.0, false, 0.0, true, false, 0)
    cache_opts = NoLimits.LaplaceCacheOptions(0.0)
    multistart = NoLimits.LaplaceMultistartOptions(0, 0, 1e-6, 5, :lhs)
    obj_of = θ_vec -> begin
        θ = ComponentArray(θ_vec, getaxes(θ0))
        o, _, _ = NoLimits._laplace_objective_and_grad(
            dm, batch_infos, θ, const_cache, ll_cache, ebe_cache;
            inner = inner, hessian = hessian, cache_opts = cache_opts,
            multistart = multistart, rng = Random.default_rng())
        o
    end
    grad_of = θ_vec -> begin
        θ = ComponentArray(θ_vec, getaxes(θ0))
        _, g, _ = NoLimits._laplace_objective_and_grad(
            dm, batch_infos, θ, const_cache, ll_cache, ebe_cache;
            inner = inner, hessian = hessian, cache_opts = cache_opts,
            multistart = multistart, rng = Random.default_rng())
        collect(g)
    end
    fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), obj_of, collect(θ0))[1]
    @test isapprox(grad_of(collect(θ0)), fd; rtol = rtol, atol = atol)
end

@testset "Laplace fit (non-ODE) returns EB modes" begin
    res = fx_laplace()
    @test res.result.eb_modes !== nothing
    @test length(res.result.eb_modes) == length(get_batches(fx_re_dm()))
end

@testset "Laplace fit (ODE) runs" begin
    @test fx_ode_laplace().summary.converged isa Bool
end

@testset "Laplace fit non-normal Poisson outcome" begin
    res = fx_pois_laplace()
    @test res isa FitResult
    @test res.result.eb_modes !== nothing
    @test length(res.result.eb_modes) == length(get_batches(fx_pois_dm()))
end

@testset "Laplace objective gradient matches FD (scalar RE)" begin
    _laplace_grad_matches_fd(fx_re_dm(); rtol = 1e-3, atol = 1e-3)
end
@testset "Laplace objective gradient matches FD (multivariate + multiple groups)" begin
    _laplace_grad_matches_fd(fx_mvn_dm(); rtol = 2e-3, atol = 2e-3)
end
@testset "Laplace objective gradient matches FD (ODE)" begin
    _laplace_grad_matches_fd(fx_ode_dm(); rtol = 2e-3, atol = 2e-3)
end
@testset "Laplace objective gradient matches FD (multiple RE groups)" begin
    _laplace_grad_matches_fd(fx_mg_dm(); rtol = 2e-3, atol = 2e-3)
end

@testset "Laplace batching with constant RE levels" begin
    dm = fx_mg_dm()
    @test length(get_batches(dm)) == 2
    @test all(length.(get_batches(dm)) .== 2)
    laplace_pairing, _, _ = NoLimits._build_laplace_batch_infos(
        dm, (; η_site = (; A = 0.2)))
    @test sort(length.(laplace_pairing.batches)) == [1, 1, 2]
    @test_throws ErrorException NoLimits._build_laplace_batch_infos(
        dm, (; η_site = (; Z = 1.0)))
end

@testset "Laplace batch info with multiple groups and multivariate REs" begin
    dm = fx_mvn_dm()
    pairing, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(
        dm, NamedTuple())
    @test length(pairing.batches) == 2
    @test all(info -> info.n_b == 6, batch_infos)
    info = batch_infos[1]
    cache = build_ll_cache(dm)
    θ = get_θ0_untransformed(fx_mvn_model().fixed.fixed)
    ll = 0.0
    for i in info.inds
        ll += NoLimits._loglikelihood_individual(dm, i, θ,
            NoLimits._build_eta_ind(dm, i, info, zeros(info.n_b), const_cache, θ), cache)
    end
    @test isfinite(ll)
end

@testset "Laplace builds local eta vectors for individuals spanning RE levels" begin
    dm = fx_mg_dm()
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    θ = get_θ0_untransformed(fx_mg_model().fixed.fixed)
    info = batch_infos[1]
    b = collect(range(0.1, 0.2; length = info.n_b))
    for i in info.inds
        η_i = NoLimits._build_eta_ind(dm, i, info, b, const_cache, θ)
        @test haskey(η_i, :η_id) && haskey(η_i, :η_site)
    end
end

@testset "Laplace uses level-specific constant covariates in RE priors" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0)
            σ = RealNumber(1.0)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age]; constant_on = :SITE)
        end
        @randomEffects begin
            η_site = RandomEffect(Normal(x.Age, 1.0); column = :SITE)
        end
        @formulas begin
            y ~ Normal(a + η_site, σ)
        end
    end
    df = DataFrame(ID = [1, 1, 2, 2], SITE = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0],
        Age = [10.0, 10.0, 20.0, 20.0], y = zeros(4))
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    pairing, batch_infos, _ = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    @test length(pairing.batches) == 2
    info = batch_infos[1]
    @test info.n_b == 1
    cache = build_ll_cache(dm)
    θ = get_θ0_untransformed(model.fixed.fixed)
    ll = 0.0
    for i in info.inds
        ll += NoLimits._loglikelihood_individual(
            dm, i, θ, ComponentArray((; η_site = 0.0)), cache)
    end
    dists_builder = get_create_random_effect_distribution(model.random.random)
    model_funs = get_model_funs(model)
    helpers = get_helper_funs(model)
    prior_sum = 0.0
    re_cache = dm.re_group_info.laplace_cache
    re_info = info.re_info[findfirst(==(:η_site), re_cache.re_names)]
    for li in eachindex(re_info.map.levels)
        dist = getproperty(
            dists_builder(
                θ, dm.individuals[re_info.reps[li]].const_cov, model_funs, helpers),
            :η_site)
        prior_sum += logpdf(dist, 0.0)
    end
    const_cache = NoLimits._build_constants_cache(dm, NamedTuple())
    @test isapprox(
        NoLimits._laplace_logf_batch(dm, info, θ, zeros(info.n_b), const_cache, cache),
        ll + prior_sum; atol = 1e-8, rtol = 1e-8)
end

@testset "Laplace with constants_re fixes all REs for one individual" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale = :log)
        end
        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column = :ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column = :SITE)
        end
        @formulas begin
            y ~ Normal(a + η_id + η_site, σ)
        end
    end
    df = DataFrame(ID = [:id1, :id1, :id2, :id2], SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.0, -0.1])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    constants_re = NamedTuple{(:η_id, :η_site)}((
        NamedTuple{(:id1,)}((0.3,)), NamedTuple{(:A,)}((-0.2,))))
    pairing, batch_infos, _ = NoLimits._build_laplace_batch_infos(dm, constants_re)
    @test length(pairing.batches) == 2
    @test sort(length.(pairing.batches)) == [1, 1]
    @test sort([info.n_b for info in batch_infos]) == [0, 2]
    res = fit_model(
        dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)), constants_re = constants_re)
    @test res.summary.converged isa Bool
    re_dfs = get_laplace_random_effects(
        dm, res; constants_re = constants_re, flatten = true, include_constants = true)
    @test hasproperty(re_dfs, :η_id) && hasproperty(re_dfs, :η_site)
    @test length(re_dfs.η_id.ID) == 2 && length(re_dfs.η_site.SITE) == 2
end

@testset "Laplace fit single-thread vs multithread (if available)" begin
    Threads.nthreads() < 2 && return
    dm = fx_re_dm()
    method = NoLimits.Laplace(; optim_kwargs = (maxiters = 2,))
    rs = fit_model(dm, method; serialization = EnsembleSerial(), rng = MersenneTwister(123))
    rt = fit_model(
        dm, method; serialization = EnsembleThreads(), rng = MersenneTwister(123))
    @test rs.summary.objective == rt.summary.objective
    @test collect(NoLimits.get_params(rs, scale = :untransformed)) ==
          collect(NoLimits.get_params(rt, scale = :untransformed))
end

@testset "Laplace fit with BlackBoxOptim requires bounds" begin
    # Bespoke: needs free params with no finite bounds so BBO errors without lb/ub.
    bbo_model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    dm = DataModel(bbo_model,
        DataFrame(ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.0, -0.1]);
        primary_id = :ID, time_col = :t)
    @test_throws ErrorException fit_model(dm,
        NoLimits.Laplace(;
            optimizer = OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(), optim_kwargs = (maxiters = 2,)))
    lb, ub = default_bounds_from_start(dm; margin = 1.0)
    res = fit_model(dm,
        NoLimits.Laplace(;
            optimizer = OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(), optim_kwargs = (maxiters = 2,), lb = lb, ub = ub))
    @test res.summary.converged isa Bool
end

@testset "Laplace multistart options" begin
    lap = NoLimits.Laplace()
    @test lap.multistart.n == 50 && lap.multistart.k == 10 &&
          lap.multistart.sampling == :lhs
    @test fit_model(fx_re_dm(),
        NoLimits.Laplace(; optim_kwargs = (maxiters = 2,),
            multistart_n = 2, multistart_k = 2, multistart_grad_tol = 0.0),
        rng = MersenneTwister(1)) isa FitResult
end

@testset "Laplace objective cache only reuses valid gradients" begin
    θ = ComponentArray((a = 0.1, σ = 0.2))
    axs = getaxes(θ)
    cache = NoLimits._LaplaceObjCache{Float64, ComponentArray}(nothing, Inf,
        ComponentArray(zeros(Float64, length(θ)), axs), false)
    NoLimits._laplace_obj_cache_set_obj!(cache, θ, 1.0)
    @test NoLimits._laplace_obj_cache_lookup(cache, θ, 1e9) === nothing
    grad = ComponentArray([3.0, 4.0], axs)
    NoLimits._laplace_obj_cache_set_obj_grad!(cache, θ, 2.0, grad)
    hit = NoLimits._laplace_obj_cache_lookup(cache, θ, 0.0)
    @test hit !== nothing && hit[1] == 2.0 && collect(hit[2]) == collect(grad)
end

@testset "Laplace threaded cache fallback preserves ODE options" begin
    dm = fx_ode_dm()
    ll_cache = build_ll_cache(dm; ode_kwargs = (abstol = 1e-8, reltol = 1e-7))
    threaded = NoLimits._laplace_thread_caches(dm, ll_cache, 2)
    @test length(threaded) == 2
    @test all(c -> c.ode_args == ll_cache.ode_args, threaded)
    @test all(c -> c.ode_kwargs == ll_cache.ode_kwargs, threaded)
end

@testset "reestimate_ebes" begin
    dm = fx_re_dm()
    n = length(get_individuals(dm))
    res_new = reestimate_ebes(fx_laplace())
    re = get_random_effects(res_new)
    @test re isa NamedTuple && haskey(re, :η) && nrow(re.η) == n
    res_nostore = fit_model(
        dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)); store_data_model = false)
    @test nrow(get_random_effects(dm, reestimate_ebes(dm, res_nostore)).η) == n
    @test nrow(get_random_effects(reestimate_ebes(fx_laplace(); individuals = [1, 2])).η) ==
          n
    path = tempname() * ".jld2"
    save_fit(path, fx_laplace())
    @test nrow(get_random_effects(dm, reestimate_ebes(dm, load_fit(path; dm = dm))).η) == n
    re_saem = get_random_effects(reestimate_ebes(fx_saem()))
    @test re_saem isa NamedTuple && haskey(re_saem, :η) && nrow(re_saem.η) == n
end

@testset "reestimate_ebes mcmc sampling" begin
    res_new = reestimate_ebes(fx_laplace(); ebe_multistart_sampling = :mcmc,
        ebe_multistart_n = 5, ebe_mcmc_n_adapt = 2)
    re = get_random_effects(res_new)
    @test re isa NamedTuple && haskey(re, :η) &&
          nrow(re.η) == length(get_individuals(fx_re_dm()))
end

@testset "Laplace with NormalizingPlanarFlow custom base_dist" begin
    df = DataFrame(ID = [:A, :A, :B, :B, :C, :C], t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.3, 0.25])
    function make_npf_model(base)
        @Model begin
            @fixedEffects begin
                a = RealNumber(0.1)
                σ = RealNumber(0.3, scale = :log)
                ψ = NPFParameter(1, 2; seed = 1, calculate_se = false, base_dist = base)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(NormalizingPlanarFlow(ψ); column = :ID)
            end
            @formulas begin
                y ~ Normal(a + η[1], σ)
            end
        end
    end
    res_default = fit_model(
        DataModel(make_npf_model(nothing), df; primary_id = :ID, time_col = :t),
        NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))
    @test res_default isa FitResult
    res_mvn = fit_model(
        DataModel(
            make_npf_model(MvNormal([0.5], [2.0;;])), df; primary_id = :ID, time_col = :t),
        NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))
    @test res_mvn isa FitResult
    res_tdist = fit_model(
        DataModel(make_npf_model(MvTDist(5, zeros(1), ones(1, 1))),
            df; primary_id = :ID, time_col = :t),
        NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))
    @test res_tdist isa FitResult
    @test NoLimits.get_objective(res_default) != NoLimits.get_objective(res_tdist)
end

@testset "Laplace penalty enters objective AND gradient" begin
    dm = fx_re_dm()
    res_unpen = fit_model(dm, NoLimits.Laplace())
    res_pen = fit_model(dm, NoLimits.Laplace(); penalty = (a = 1.0e6,))
    a_unpen = NoLimits.get_params(res_unpen; scale = :untransformed).a
    a_pen = NoLimits.get_params(res_pen; scale = :untransformed).a
    # The ridge penalty w·a² with a huge weight must pull â to ≈ 0. With the
    # historical bug the reported gradient lacked the penalty term, so the
    # optimizer stalled at (or walked toward) the unpenalized optimum.
    @test abs(a_pen) < 0.05
    @test abs(a_pen) < abs(a_unpen)
end

# ── Hutchinson logdet-gradient option (folded from estimation_hutchinson) ────

@testset "Hutchinson logdet gradient approximates trace" begin
    Random.seed!(1234)
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
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

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    pairing, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(
        dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θu = get_θ0_untransformed(model.fixed.fixed)
    ebe_cache = _make_laplace_ebe_cache(eltype(θu), length(batch_infos))

    info = batch_infos[1]
    b = NoLimits._laplace_default_b0(dm, info, θu, const_cache, ll_cache)

    res_exact = NoLimits._laplace_grad_batch(
        dm, info, θu, b, const_cache, ll_cache, ebe_cache.ad_cache, 1;
        use_trace_logdet_grad = true,
        use_hutchinson = false)
    res_hutch = NoLimits._laplace_grad_batch(
        dm, info, θu, b, const_cache, ll_cache, ebe_cache.ad_cache, 1;
        use_trace_logdet_grad = true,
        use_hutchinson = true,
        hutchinson_n = 16)

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
            σ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η1 = RandomEffect(Normal(0.0, 1.0); column = :ID)
            η2 = RandomEffect(Normal(0.0, 1.0); column = :ID)
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

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θu = get_θ0_untransformed(model.fixed.fixed)
    method = NoLimits.Laplace(; use_hutchinson = true, hutchinson_n = 1)

    function eval_grad(global_seed::Int, rng_seed::Int)
        Random.seed!(global_seed)
        ebe_cache = _make_laplace_ebe_cache(eltype(θu), length(batch_infos))
        _, g, _ = NoLimits._laplace_objective_and_grad(
            dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
            inner = method.inner,
            hessian = method.hessian,
            cache_opts = method.cache,
            multistart = method.multistart,
            rng = MersenneTwister(rng_seed))
        return collect(g)
    end

    g1 = eval_grad(1, 123)
    g2 = eval_grad(2, 123)
    g3 = eval_grad(1, 999)

    @test isapprox(g1, g2; atol = 1e-12, rtol = 1e-12)
    @test maximum(abs.(g1 .- g3)) > 1e-6
end

# ── NewtonInner inner optimizer (folded from estimation_newton_inner) ────────
# NewtonInner is an OPT-IN inner-EBE optimizer: the default (LBFGS via
# Optimization.jl) is unchanged. These tests check that (1) the option produces
# the same fits as the default within inner-solver tolerance, (2) the
# `max_dim` fallback reproduces the default path exactly, and (3) the inner
# solutions themselves agree at matching gradient tolerances.

function _newton_test_dm()
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.8)
            b = RealNumber(0.3)
            ω = RealNumber(0.5, scale = :log)
            σ = RealNumber(0.4, scale = :log)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @formulas begin
            lin = a + b * x.Age + η
            y ~ Normal(lin, σ)
        end
    end
    rng = Xoshiro(11)
    rows = NamedTuple[]
    for i in 1:40
        age = 0.5 + 0.05 * (i % 20)
        ηi = 0.4 * randn(rng)
        for j in 1:6
            t = (j - 1) * 0.5
            y = 0.8 + 0.3 * age + ηi + 0.4 * randn(rng)
            push!(rows, (ID = i, t = t, Age = age, y = y))
        end
    end
    return DataModel(model, DataFrame(rows); primary_id = :ID, time_col = :t)
end

@testset "NewtonInner inner solver (opt-in)" begin
    dm = _newton_test_dm()

    @testset "inner solve agrees with the default optimizer" begin
        llc = NoLimits.build_ll_cache(dm; force_saveat = true)
        _, binfos, ccache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
        θ = NoLimits.get_θ0_untransformed(get_model(dm).fixed.fixed)
        adc = NoLimits._init_laplace_ad_cache(length(binfos))
        for bi in (1, 5, 17)
            info = binfos[bi]
            b0 = zeros(info.n_b)
            sol_def = NoLimits._laplace_solve_batch!(dm, info, θ, ccache, llc, adc, bi, b0)
            sol_new = NoLimits._laplace_solve_batch!(dm, info, θ, ccache, llc, adc, bi, b0;
                optimizer = NewtonInner())
            @test sol_new isa NoLimits._NewtonSol
            @test sol_new.converged
            @test NoLimits._laplace_sol_grad_norm(sol_new) <= 1e-8
            @test collect(sol_new.u)≈collect(sol_def.u) atol=1e-6
            @test NoLimits._laplace_sol_logf(sol_new)≈NoLimits._laplace_sol_logf(sol_def) atol=1e-8
        end
    end

    @testset "Laplace fit matches default within tolerance" begin
        res_def = fit_model(dm, NoLimits.Laplace(optim_kwargs = (maxiters = 40,));
            serialization = EnsembleSerial(), rng = Xoshiro(3))
        res_new = fit_model(dm,
            NoLimits.Laplace(optim_kwargs = (maxiters = 40,),
                inner_optimizer = NewtonInner());
            serialization = EnsembleSerial(), rng = Xoshiro(3))
        @test isfinite(get_objective(res_new))
        @test get_objective(res_new)≈get_objective(res_def) rtol=1e-6 atol=1e-6
        # Qualified: MCMCChains (loaded by earlier files in the same batch
        # subprocess) also exports a `get_params`, making the bare name ambiguous.
        @test collect(NoLimits.get_params(res_new;
            scale = :transformed))≈
        collect(NoLimits.get_params(res_def; scale = :transformed)) rtol=1e-3 atol=1e-3
        eta_def = get_random_effects(dm, res_def, :η)
        eta_new = get_random_effects(dm, res_new, :η)
        @test eta_new≈eta_def atol=1e-4
    end

    @testset "FOCEI fit matches default within tolerance" begin
        res_def = fit_model(dm, NoLimits.FOCEI(optim_kwargs = (maxiters = 40,));
            serialization = EnsembleSerial(), rng = Xoshiro(3))
        res_new = fit_model(dm,
            NoLimits.FOCEI(optim_kwargs = (maxiters = 40,),
                inner_optimizer = NewtonInner());
            serialization = EnsembleSerial(), rng = Xoshiro(3))
        @test isfinite(get_objective(res_new))
        @test get_objective(res_new)≈get_objective(res_def) rtol=1e-6 atol=1e-6
    end

    @testset "max_dim fallback reproduces the default path" begin
        res_def = fit_model(dm, NoLimits.Laplace(optim_kwargs = (maxiters = 15,));
            serialization = EnsembleSerial(), rng = Xoshiro(3))
        res_fb = fit_model(dm,
            NoLimits.Laplace(optim_kwargs = (maxiters = 15,),
                inner_optimizer = NewtonInner(max_dim = 0));
            serialization = EnsembleSerial(), rng = Xoshiro(3))
        @test get_objective(res_fb)≈get_objective(res_def) rtol=1e-10 atol=1e-10
    end
end
