using Test
using DataFrames
using DataInterpolations
using NoLimits
using JLD2
using OptimizationNLopt
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

# Consolidated Laplace tests, part 1 of 2 (part 2: estimation_laplace2_tests.jl,
# split for CI shard balance — the halves run in different shards).
# Standard structures
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
    _, batch_infos, const_cache = NoLimits._build_re_batch_infos(dm, NamedTuple())
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

# The outer gradient the OPTIMIZER receives: on the transformed free-parameter scale, at the
# package defaults, for a given curvature. Distinct from `_laplace_grad_matches_fd` above in
# three ways that each hid a defect:
#   * transformed scale, so a `RealPSDMatrix` is covered (perturbing Ω[1,2] on the natural
#     scale breaks symmetry and the Cholesky fails, so the natural-scale check cannot);
#   * `method.hessian` verbatim, i.e. the default ADAPTIVE jitter rather than a fixed one;
#   * parameterised by curvature, so FOCEI/FOCE's Fisher information is checked too.
function _outer_grad_matches_fd_t(dm, method, hmode; rtol, h = 3e-4)
    fe = NoLimits.get_fixed(NoLimits.get_model(dm))
    layout = NoLimits.free_parameter_layout(fe)
    inner = NoLimits._resolve_inner_options(method.inner, dm)
    ms = NoLimits._resolve_multistart_options(method.multistart, inner)
    _, binfos, const_cache = NoLimits._build_re_batch_infos(dm, NamedTuple())
    llc = build_ll_cache(dm; force_saveat = true)
    kw = (; inner = inner, hessian = method.hessian, cache_opts = method.cache,
        multistart = ms, serialization = SciMLBase.EnsembleSerial(), hmode = hmode)
    fresh() = NoLimits._init_laplace_eval_cache(length(binfos), Float64)
    θu_of = θt -> layout.inv_transform(NoLimits._merge_free_into_full(
        layout.θ_const_t_vec, layout.free_idx,
        ComponentArray(collect(θt), layout.axs), layout.axs_full))
    obj = θt -> NoLimits._laplace_objective_only(dm, binfos, θu_of(θt), const_cache,
        llc, fresh(); rng = Random.Xoshiro(0), kw...)

    θt0 = collect(layout.θ0_free_t)
    θt_full = NoLimits._merge_free_into_full(layout.θ_const_t_vec, layout.free_idx,
        layout.θ0_free_t, layout.axs_full)
    _, g_u, _ = NoLimits._laplace_objective_and_grad(dm, binfos, θu_of(θt0), const_cache,
        llc, fresh(); rng = Random.Xoshiro(0), kw...)
    g_t = NoLimits.apply_inv_jacobian_T(layout.inv_transform, θt_full, g_u)
    g_free = similar(layout.θ0_free_t)
    for name in layout.free_names
        setproperty!(g_free, name, getproperty(g_t, name))
    end
    fd = map(eachindex(θt0)) do i
        θp = copy(θt0)
        θp[i] += h
        θm = copy(θt0)
        θm[i] -= h
        (obj(θp) - obj(θm)) / (2h)
    end
    @test all(isfinite, collect(g_free))
    @test isapprox(collect(g_free), fd; rtol = rtol)
end

@testset "outer gradient matches FD at the defaults (nonlinear in η)" begin
    # Nonlinear in η, so the Fisher-information curvature genuinely differs from the exact
    # Hessian (~4-6% here). Every pre-existing gradient check uses a model LINEAR in η with a
    # Gaussian outcome, where the two coincide exactly and a surrogate curvature is right by
    # accident -- which is why FOCEI's db*/dθ term could use the wrong one unnoticed.
    df = DataFrame(ID = repeat(1:5, inner = 4), t = repeat(0.0:3.0, 5),
        y = [1.35, 1.10, 0.92, 0.81, 1.62, 1.28, 1.05, 0.90,
            1.18, 0.99, 0.85, 0.74, 1.90, 1.47, 1.19, 1.02,
            1.05, 0.88, 0.77, 0.68])
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3)
            b = RealNumber(0.25, scale = :log)
            σ = RealNumber(0.08, scale = :log)
            Ω = RealPSDMatrix([1.0 0.95; 0.95 1.0], scale = :cholesky)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(MvNormal(zeros(2), Ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(exp(a + η[1]) * exp(-exp(b + η[2]) * t), σ)
        end
    end
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    _outer_grad_matches_fd_t(dm, NoLimits.Laplace(), NoLimits._ExactHess(); rtol = 2e-3)
    _outer_grad_matches_fd_t(
        dm, NoLimits.FOCEI(), NoLimits._FOCEIHess(true); rtol = 2e-3)
    _outer_grad_matches_fd_t(dm, NoLimits.FOCEI(interaction = false),
        NoLimits._FOCEIHess(false); rtol = 2e-3)
end

@testset "Laplace batching with constant RE levels" begin
    dm = fx_mg_dm()
    @test length(get_batches(dm)) == 2
    @test all(length.(get_batches(dm)) .== 2)
    laplace_pairing, _, _ = NoLimits._build_re_batch_infos(
        dm, (; η_site = (; A = 0.2)))
    @test sort(length.(laplace_pairing.batches)) == [1, 1, 2]
    @test_throws ErrorException NoLimits._build_re_batch_infos(
        dm, (; η_site = (; Z = 1.0)))
end

@testset "Laplace batch info with multiple groups and multivariate REs" begin
    dm = fx_mvn_dm()
    pairing, batch_infos, const_cache = NoLimits._build_re_batch_infos(
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
    _, batch_infos, const_cache = NoLimits._build_re_batch_infos(dm, NamedTuple())
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
    pairing, batch_infos, _ = NoLimits._build_re_batch_infos(dm, NamedTuple())
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
    dists_builder = create_random_effect_distribution(model.random.random)
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
    pairing, batch_infos, _ = NoLimits._build_re_batch_infos(dm, constants_re)
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
