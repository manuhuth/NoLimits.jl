export Multistart
export MultistartFitResult
export get_multistart_results
export get_multistart_errors
export get_multistart_starts
export get_multistart_failed_results
export get_multistart_failed_starts
export get_multistart_best_index
export get_multistart_best

using ComponentArrays
using Distributions
using LinearAlgebra
using ProgressMeter
using Random
using SciMLBase

"""
    Multistart(; dists, n_draws_requested, n_draws_used, sampling, serialization, rng,
               progress, screening, ebe_maxiters)

Multistart wrapper that runs any optimization-based fitting method from multiple initial
parameter vectors and returns the best result.

Starting points are drawn either from the fixed-effect priors or from user-supplied
`dists`; the top-`n_draws_used` candidates (by a cheap objective evaluation) are then
fully optimized.

# Keyword Arguments
- `dists::NamedTuple = NamedTuple()`: per-parameter sampling distributions, keyed by
  fixed-effect name. Parameters without an entry use their prior, if available.
- `n_draws_requested::Int = 100`: number of candidate starting points to sample.
- `n_draws_used::Int = 50`: number of candidates to fully optimize after screening.
- `sampling::Symbol = :random`: sampling strategy for starting points: `:random` or `:lhs`
  (Latin hypercube sampling).
- `serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads()`: parallelisation
  strategy for running multiple starts.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
- `progress::Bool = true`: whether to display progress bars for the screening and fitting phases.
- `screening::Symbol = :prior_mean`: objective used for pre-selecting starting points.
  - `:prior_mean` — evaluates the observation log-likelihood with random effects fixed at
    their prior means under each candidate θ. Fast but ignores RE adaptability.
  - `:ebe` — for each candidate θ, first computes per-individual Empirical Bayes Estimates
    (EBEs) by maximizing the joint log-density (observation ll + RE prior), then uses the
    resulting joint log-density as the screening score. More accurate but slower.
- `ebe_maxiters::Int = 30`: maximum inner-optimization iterations per individual when
  `screening = :ebe`. Lower values trade accuracy for speed.
"""
struct Multistart{D, S, R}
    dists::D
    n_draws_requested::Int
    n_draws_used::Int
    sampling::Symbol
    serialization::S
    rng::R
    progress::Bool
    screening::Symbol
    ebe_maxiters::Int
end

"""
    MultistartFitResult{M, R, RE, S, E, B}

Result from a [`Multistart`](@ref) run. Stores all successful and failed per-start
results together with the index of the best (lowest objective) successful start.

Use the accessor functions to retrieve individual components:
[`get_multistart_results`](@ref), [`get_multistart_errors`](@ref),
[`get_multistart_starts`](@ref), [`get_multistart_failed_results`](@ref),
[`get_multistart_failed_starts`](@ref), [`get_multistart_best_index`](@ref),
[`get_multistart_best`](@ref).
"""
struct MultistartFitResult{M, R, RE, S, E, B}
    method::M
    results_ok::R
    results_err::RE
    starts_ok::S
    starts_err::S
    errors_err::E
    best_idx::Int
    scores_ok::B
end

function Multistart(; dists = NamedTuple(),
        n_draws_requested::Int = 100,
        n_draws_used::Int = 50,
        sampling::Symbol = :random,
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Random.default_rng(),
        progress::Bool = true,
        screening::Symbol = :prior_mean,
        ebe_maxiters::Int = 30)
    n_draws_requested < 0 && error("n_draws_requested must be ≥ 0.")
    n_draws_used < 1 && error("n_draws_used must be ≥ 1.")
    (sampling == :random || sampling == :lhs) || error("sampling must be :random or :lhs.")
    (screening == :prior_mean || screening == :ebe) ||
        error("screening must be :prior_mean or :ebe.")
    ebe_maxiters < 1 && error("ebe_maxiters must be ≥ 1.")
    return Multistart(dists, n_draws_requested, n_draws_used, sampling, serialization, rng,
        progress, screening, ebe_maxiters)
end

function _lhs_unit(n::Int, rng::AbstractRNG)
    u = ((0:(n - 1)) .+ rand(rng, n)) ./ n
    return u[Random.randperm(rng, n)]
end

function _lhs_draws_univariate(dist, n::Int, rng::AbstractRNG)
    applicable(Distributions.quantile, dist, 0.5) || return nothing
    u = _lhs_unit(n, rng)
    return [Distributions.quantile(dist, ui) for ui in u]
end

function _lhs_draws_array(dist, n::Int, rng::AbstractRNG, size_hint)
    total = length(size_hint)
    m1 = _laplace_marginal_mvnormal(dist, 1)
    m1 === nothing && return nothing
    draws = [Vector{Any}(undef, n) for _ in 1:total]
    for j in 1:total
        mj = _laplace_marginal_mvnormal(dist, j)
        mj === nothing && return nothing
        dj = _lhs_draws_univariate(mj, n, rng)
        dj === nothing && return nothing
        draws[j] = dj
    end
    out = Vector{Any}(undef, n)
    for i in 1:n
        v = similar(size_hint)
        for j in 1:total
            v[j] = draws[j][i]
        end
        out[i] = reshape(v, size(size_hint))
    end
    return out
end

function _check_bounds(name::Symbol, v, lower, upper)
    if v isa Number
        if !(v >= lower && v <= upper)
            error("Multistart sampling for $(name) violates bounds. Use a truncated distribution for sampling.")
        end
        return
    end
    bad = findall((v .< lower) .| (v .> upper))
    isempty(bad) ||
        error("Multistart sampling for $(name) violates bounds at indices $(bad). Use a truncated distribution for sampling.")
end

function _sample_param(
        name::Symbol, value, dist, n::Int, sampling::Symbol, rng::AbstractRNG)
    function _fix_matrix(v)
        if v isa AbstractMatrix && size(v, 1) == size(v, 2)
            v = 0.5 .* (v .+ transpose(v))
            for k in 0:6
                try
                    cholesky(Symmetric(v))
                    return v
                catch
                    v = v + (1e-6 * (10.0^k)) .* I(size(v, 1))
                end
            end
        end
        return v
    end

    if dist isa AbstractArray
        samples = [similar(value) for _ in 1:n]
        for idx in eachindex(dist)
            d = dist[idx]
            d === nothing && continue
            if sampling == :lhs
                lhs = _lhs_draws_univariate(d, n, rng)
                lhs === nothing && (lhs = [rand(rng, d) for _ in 1:n])
                for i in 1:n
                    samples[i][idx] = lhs[i]
                end
            else
                for i in 1:n
                    samples[i][idx] = rand(rng, d)
                end
            end
        end
        for i in 1:n
            samples[i] = _fix_matrix(samples[i])
        end
        return samples
    end

    if sampling == :lhs
        if value isa AbstractArray
            lhs = _lhs_draws_array(dist, n, rng, value)
            if lhs === nothing
                lhs = Vector{Any}(undef, n)
                for i in 1:n
                    s = rand(rng, dist)
                    if s isa Number
                        v = similar(value)
                        for idx in eachindex(v)
                            v[idx] = rand(rng, dist)
                        end
                        lhs[i] = _fix_matrix(v)
                    else
                        lhs[i] = _fix_matrix(s)
                    end
                end
            end
            return lhs
        else
            lhs = _lhs_draws_univariate(dist, n, rng)
            lhs === nothing && (lhs = [rand(rng, dist) for _ in 1:n])
            return lhs
        end
    end
    if value isa AbstractArray
        out = Vector{Any}(undef, n)
        for i in 1:n
            s = rand(rng, dist)
            if s isa Number
                v = similar(value)
                for idx in eachindex(v)
                    v[idx] = rand(rng, dist)
                end
                out[i] = _fix_matrix(v)
            else
                out[i] = _fix_matrix(s)
            end
        end
        return out
    end
    return [_fix_matrix(rand(rng, dist)) for _ in 1:n]
end

function _collect_param_dists(dm::DataModel, ms::Multistart)
    fe = dm.model.fixed.fixed
    priors = get_priors(fe)
    names = get_names(fe)
    pairs = Pair{Symbol, Any}[]
    for name in names
        if haskey(ms.dists, name)
            push!(pairs, name => getfield(ms.dists, name))
        else
            p = hasproperty(priors, name) ? getfield(priors, name) : Priorless()
            if !(p isa Priorless)
                push!(pairs, name => p)
            end
        end
    end
    return NamedTuple(pairs)
end

function _multistart_initials(dm::DataModel, ms::Multistart)
    fe = dm.model.fixed.fixed
    θ0_u = get_θ0_untransformed(fe)
    dists = _collect_param_dists(dm, ms)
    bounds = get_bounds_untransformed(fe)
    lower = bounds[1]
    upper = bounds[2]

    n_req = ms.n_draws_requested
    n_used = ms.n_draws_used
    if n_used > n_req
        @warn "n_draws_used > n_draws_requested; increasing requested draws to match." n_draws_used=n_used n_draws_requested=n_req
        n_req = n_used
    end
    n_req = max(n_req, 1)   # always at least θ0_u itself

    n_sampled = n_req - 1   # θ0_u occupies slot 1
    samples_by_param = Dict{Symbol, Vector{Any}}()
    for name in get_names(fe)
        haskey(dists, name) || continue
        dist = getfield(dists, name)
        value = getproperty(θ0_u, name)
        samples = _sample_param(name, value, dist, n_sampled, ms.sampling, ms.rng)
        lb = getproperty(lower, name)
        ub = getproperty(upper, name)
        for s in samples
            _check_bounds(name, s, lb, ub)
        end
        samples_by_param[name] = samples
    end

    # Materialise ALL n_req starts
    all_starts = Vector{ComponentArray}(undef, n_req)
    all_starts[1] = θ0_u
    for i in 2:n_req
        θi = deepcopy(θ0_u)
        for (name, samples) in samples_by_param
            setproperty!(θi, name, samples[i - 1])
        end
        all_starts[i] = θi
    end
    return all_starts
end

# Prior-mean plug-in value for one RE; 0/zeros fallback when `mean` is unavailable.
function _re_mean_or_zero(dist, dim::Int, is_scalar::Bool)
    if is_scalar || dim == 1
        v = 0.0
        try
            v = Float64(mean(dist))
        catch
        end
        return v
    else
        v = zeros(dim)
        try
            m = mean(dist)
            if m isa AbstractVector && length(m) == dim
                v = Float64.(m)
            end
        catch
        end
        return v
    end
end

function _build_mean_eta(dm::DataModel, θu::ComponentArray)
    re_cache = dm.re_group_info.laplace_cache
    (re_cache === nothing || isempty(re_cache.re_names)) && return ComponentArray()
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)
    n = length(dm.individuals)
    etas = Vector{ComponentArray}(undef, n)
    for i in 1:n
        const_cov = dm.individuals[i].const_cov
        dists = dists_builder(θu, const_cov, model_funs, helpers)
        pairs = Pair{Symbol, Any}[]
        for (ri, re) in enumerate(re_cache.re_names)
            dim = re_cache.dims[ri]
            is_scalar = re_cache.is_scalar[ri]
            dist = getproperty(dists, re)
            push!(pairs, re => _re_mean_or_zero(dist, dim, is_scalar))
        end
        etas[i] = ComponentArray(NamedTuple(pairs))
    end
    return etas
end

# Compute per-individual EBEs by maximizing the joint log-density (obs ll + RE prior)
# for each candidate θu.  Returns (etas, joint_score) where joint_score is the total
# joint log-density at the EBEs (used as the screening score).
function _build_ebe_eta(dm::DataModel, θu::ComponentArray, ll_cache; maxiters::Int = 30)
    re_cache = dm.re_group_info.laplace_cache
    (re_cache === nothing || isempty(re_cache.re_names)) && return (ComponentArray(), 0.0)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    re_logpdf_fn = get_re_logpdf(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)
    re_names = re_cache.re_names
    re_names_tuple = Tuple(re_names)
    optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0))
    n = length(dm.individuals)
    etas = Vector{ComponentArray}(undef, n)
    joint_score = 0.0
    for i in 1:n
        const_cov = dm.individuals[i].const_cov
        dists = dists_builder(θu, const_cov, model_funs, helpers)
        # Build prior mean as starting point
        pairs = Pair{Symbol, Any}[]
        for (ri, re) in enumerate(re_names)
            dim = re_cache.dims[ri]
            is_scalar = re_cache.is_scalar[ri]
            dist = getproperty(dists, re)
            push!(pairs, re => _re_mean_or_zero(dist, dim, is_scalar))
        end
        η0_i = ComponentArray(NamedTuple(pairs))
        axs = getaxes(η0_i)
        η0_flat = Vector(η0_i)
        if isempty(η0_flat)
            etas[i] = η0_i
            continue
        end
        # Closure: negative joint log-density to minimize
        neg_logf = let dm = dm, i = i, θu = θu, ll_cache = ll_cache,
            axs = axs, dists = dists, re_logpdf_fn = re_logpdf_fn,
            re_names = re_names, re_names_tuple = re_names_tuple

            (η_flat, _) -> begin
                η_i = ComponentArray(η_flat, axs)
                ll = _loglikelihood_individual(dm, i, θu, η_i, ll_cache)
                !isfinite(ll) && return eltype(η_flat)(Inf)
                re_v = NamedTuple{re_names_tuple}(
                    Tuple([getproperty(η_i, re) for re in re_names]))
                lp = re_logpdf_fn(dists, re_v)
                !isfinite(lp) && return eltype(η_flat)(Inf)
                return -(ll + lp)
            end
        end
        try
            prob = OptimizationProblem(
                OptimizationFunction(neg_logf, Optimization.AutoForwardDiff()),
                η0_flat)
            sol = solve(prob, optimizer; maxiters = maxiters)
            etas[i] = ComponentArray(sol.u, axs)
            f_sol = sol.objective  # = -(joint log-density at EBE)
            joint_score += isfinite(f_sol) ? -f_sol : -Inf
        catch
            etas[i] = η0_i
            f0 = neg_logf(η0_flat, nothing)
            joint_score += isfinite(f0) ? -f0 : 0.0
        end
        !isfinite(joint_score) && break
    end
    return etas, joint_score
end

function _multistart_screen(dm::DataModel,
        candidates::Vector{ComponentArray},
        n_used::Int,
        ode_args::Tuple,
        ode_kwargs::NamedTuple,
        serialization::SciMLBase.EnsembleAlgorithm,
        screening::Symbol,
        ebe_maxiters::Int;
        progress::Bool = true)
    # EBE inner optimization is serial per individual; use EnsembleSerial for that path.
    cache_serialization = screening == :ebe ? EnsembleSerial() : serialization
    cache = build_ll_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = cache_serialization, force_saveat = true)
    scores = Vector{Float64}(undef, length(candidates))
    screen_p = progress ?
               Progress(
        length(candidates); desc = "Multistart screening: ", showspeed = true) : nothing
    for (i, θu) in enumerate(candidates)
        if screening == :ebe
            etas, joint_score = _build_ebe_eta(dm, θu, cache; maxiters = ebe_maxiters)
            scores[i] = isfinite(joint_score) ? -joint_score : Inf
        else
            η0 = _build_mean_eta(dm, θu)
            ll = loglikelihood(dm, θu, η0; cache = cache, serialization = serialization)
            scores[i] = isfinite(ll) ? -ll : Inf
        end
        progress && next!(screen_p)
    end
    idxs = partialsortperm(scores, 1:min(n_used, length(candidates)))
    # Return selected candidates and their scores (sign-flipped back from minimization scores)
    selected_scores = [-scores[j] for j in idxs]
    return candidates[idxs], selected_scores
end

function _multistart_score(res::FitResult)
    obj = try
        get_objective(res)
    catch
        NaN
    end
    if isfinite(obj)
        return obj
    end
    ll = try
        get_loglikelihood(res)
    catch
        NaN
    end
    return isfinite(ll) ? -ll : Inf
end

"""
    get_multistart_results(res::MultistartFitResult) -> Vector{FitResult}

Return the `FitResult` objects for all successful multistart runs.
"""
get_multistart_results(res::MultistartFitResult) = res.results_ok

"""
    get_multistart_errors(res::MultistartFitResult) -> Vector

Return the error objects thrown by failed multistart runs.
"""
get_multistart_errors(res::MultistartFitResult) = res.errors_err

"""
    get_multistart_starts(res::MultistartFitResult) -> Vector

Return the starting parameter vectors (on the untransformed scale) for all
successful multistart runs.
"""
get_multistart_starts(res::MultistartFitResult) = res.starts_ok

"""
    get_multistart_failed_results(res::MultistartFitResult) -> Vector

Return any partially-computed `FitResult` objects from failed multistart runs.
"""
get_multistart_failed_results(res::MultistartFitResult) = res.results_err

"""
    get_multistart_failed_starts(res::MultistartFitResult) -> Vector

Return the starting parameter vectors for all failed multistart runs.
"""
get_multistart_failed_starts(res::MultistartFitResult) = res.starts_err

"""
    get_multistart_best_index(res::MultistartFitResult) -> Int

Return the index (into `get_multistart_results`) of the run with the lowest
objective value.
"""
get_multistart_best_index(res::MultistartFitResult) = res.best_idx

"""
    get_multistart_best(res::MultistartFitResult) -> FitResult

Return the `FitResult` with the lowest objective value across all successful
multistart runs.
"""
get_multistart_best(res::MultistartFitResult) = res.results_ok[res.best_idx]

function fit_model(ms::Multistart, dm::DataModel, method::FittingMethod, args...; kwargs...)
    kw_nt = NamedTuple(kwargs)
    if haskey(kw_nt, :theta_0_untransformed)
        @warn "theta_0_untransformed ignored in multistart; use Multistart sampling to control starts."
        kw_nt = Base.structdiff(kw_nt, (theta_0_untransformed = nothing,))
    end

    progress = ms.progress
    ode_args = get(kw_nt, :ode_args, ())
    ode_kwargs_inner = get(kw_nt, :ode_kwargs, NamedTuple())
    serialization = get(kw_nt, :serialization, EnsembleThreads())

    all_starts = _multistart_initials(dm, ms)
    n_req = length(all_starts)
    n_used = min(ms.n_draws_used, n_req)
    varied = collect(keys(_collect_param_dists(dm, ms)))
    varied_str = isempty(varied) ? "none" : join(string.(varied), ", ")

    if n_req > n_used
        starts, screen_scores = _multistart_screen(dm, all_starts, n_used, ode_args,
            ode_kwargs_inner, serialization,
            ms.screening, ms.ebe_maxiters;
            progress = progress)
        finite_scores = filter(isfinite, screen_scores)
        if isempty(finite_scores)
            if ms.screening == :ebe
                @info "Multistart" candidates=n_req selected=n_used varying=varied_str screening=ms.screening best_screening_joint_ll="all -Inf"
            else
                @info "Multistart" candidates=n_req selected=n_used varying=varied_str screening=ms.screening best_screening_ll="all -Inf"
            end
        elseif ms.screening == :ebe
            @info "Multistart" candidates=n_req selected=n_used varying=varied_str screening=ms.screening best_screening_joint_ll=maximum(finite_scores) worst_screening_joint_ll=minimum(finite_scores)
        else
            @info "Multistart" candidates=n_req selected=n_used varying=varied_str screening=ms.screening best_screening_ll=maximum(finite_scores) worst_screening_ll=minimum(finite_scores)
        end
    else
        starts = all_starts
        @info "Multistart" candidates=n_req selected=n_used varying=varied_str
    end
    n_starts = length(starts)
    results = Vector{Union{FitResult, Nothing}}(undef, n_starts)
    errors = Vector{Any}(undef, n_starts)
    rngs = [Random.Xoshiro(rand(ms.rng, UInt)) for _ in 1:n_starts]
    fit_p = progress ?
            Progress(n_starts; desc = "Multistart fitting:  ", showspeed = true) : nothing

    function run_one(i)
        try
            local_kwargs = kw_nt
            if !haskey(kw_nt, :rng)
                local_kwargs = merge(kw_nt, (rng = rngs[i],))
            end
            results[i] = fit_model(
                dm, method, args...; theta_0_untransformed = starts[i], local_kwargs...)
        catch err
            errors[i] = err
            results[i] = nothing
        end
        progress && next!(fit_p)
    end

    if ms.serialization isa EnsembleThreads
        Threads.@threads for i in 1:n_starts
            run_one(i)
        end
    else
        for i in 1:n_starts
            run_one(i)
        end
    end

    results_ok = FitResult[]
    results_err = Any[]
    starts_ok = ComponentArray[]
    starts_err = ComponentArray[]
    errors_err = Any[]
    scores_ok = Float64[]
    for i in 1:n_starts
        res = results[i]
        if res === nothing
            push!(starts_err, starts[i])
            push!(results_err, nothing)
            push!(errors_err, errors[i])
        else
            push!(results_ok, res)
            push!(starts_ok, starts[i])
            push!(scores_ok, _multistart_score(res))
        end
    end
    isempty(results_ok) &&
        error("All multistart runs failed. First error: $(errors_err[1])")
    best_idx = findmin(scores_ok)[2]
    return MultistartFitResult(method, results_ok, results_err, starts_ok,
        starts_err, errors_err, best_idx, scores_ok)
end

get_summary(res::MultistartFitResult) = get_summary(get_multistart_best(res))
get_diagnostics(res::MultistartFitResult) = get_diagnostics(get_multistart_best(res))
get_result(res::MultistartFitResult) = get_result(get_multistart_best(res))
get_method(res::MultistartFitResult) = res.method
get_objective(res::MultistartFitResult) = get_objective(get_multistart_best(res))
get_converged(res::MultistartFitResult) = get_converged(get_multistart_best(res))
get_data_model(res::MultistartFitResult) = get_data_model(get_multistart_best(res))

function get_params(res::MultistartFitResult; kwargs...)
    return get_params(get_multistart_best(res); kwargs...)
end

get_chain(res::MultistartFitResult) = get_chain(get_multistart_best(res))
get_iterations(res::MultistartFitResult) = get_iterations(get_multistart_best(res))
get_raw(res::MultistartFitResult) = get_raw(get_multistart_best(res))
get_notes(res::MultistartFitResult) = get_notes(get_multistart_best(res))
get_observed(res::MultistartFitResult) = get_observed(get_multistart_best(res))
get_sampler(res::MultistartFitResult) = get_sampler(get_multistart_best(res))
get_n_samples(res::MultistartFitResult) = get_n_samples(get_multistart_best(res))

function get_random_effects(res::MultistartFitResult; kwargs...)
    return get_random_effects(get_multistart_best(res); kwargs...)
end

function get_laplace_random_effects(res::MultistartFitResult; kwargs...)
    return get_laplace_random_effects(get_multistart_best(res); kwargs...)
end

function get_loglikelihood(res::MultistartFitResult; kwargs...)
    return get_loglikelihood(get_multistart_best(res); kwargs...)
end
