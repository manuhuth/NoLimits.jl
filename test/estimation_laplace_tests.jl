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

# Consolidated Laplace tests (merges the former estimation_laplace and
# estimation_laplace_fit files). Standard structures
# reuse the shared fixtures (fit/model built once); bespoke @Models are kept only
# where a test specifically exercises that structure or an error path.

# Shared Laplace objective-gradient-vs-finite-differences check (generic in the
# model, so it runs against shared archetype DataModels).
function _laplace_grad_matches_fd(dm; rtol, atol)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θ0 = get_θ0_untransformed(NoLimits.get_model(dm).fixed.fixed)
    Tθ = eltype(θ0)
    n_batches = length(batch_infos)
    bstar_cache = NoLimits._LaplaceBStarCache(
        [Vector{Tθ}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = NoLimits._LaplaceGradCache([Vector{Tθ}() for _ in 1:n_batches],
        fill(Tθ(NaN), n_batches),
        [Vector{Tθ}() for _ in 1:n_batches],
        falses(n_batches))
    ad_cache = NoLimits._init_laplace_ad_cache(n_batches)
    hess_cache = NoLimits._init_laplace_hess_cache(Tθ, n_batches)
    ebe_cache = NoLimits._LaplaceCache(
        nothing, bstar_cache, grad_cache, ad_cache, hess_cache)
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
