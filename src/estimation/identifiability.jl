export NullDirection
export RandomEffectInformation
export IdentifiabilityReport
export identifiability_report

using ForwardDiff
using LinearAlgebra
using Random
using SciMLBase
using ComponentArrays

"""
    NullDirection{L}

A direction in the fixed-effect parameter space along which the objective function
is (numerically) flat, indicating a potential non-identifiability.

Fields:
- `singular_value::Float64`: the singular value associated with this direction.
- `vector::Vector{Float64}`: the null-space direction on the transformed scale.
- `loadings::L`: per-parameter loadings showing which parameters contribute to the
  direction.
"""
struct NullDirection{L}
    singular_value::Float64
    vector::Vector{Float64}
    loadings::L
end

"""
    RandomEffectInformation

Fisher information analysis of the random-effects contribution for a single batch of
individuals, as computed within [`identifiability_report`](@ref).

Fields:
- `batch::Int`: batch index.
- `n_latent::Int`: number of latent (random-effect) dimensions in the batch.
- `labels::Vector{String}`: human-readable labels for each latent dimension.
- `singular_values::Vector{Float64}`: singular values of the RE information matrix.
- `eigenvalues::Vector{Float64}`: eigenvalues of the RE information matrix.
- `rank::Int`: numerical rank.
- `nullity::Int`: dimension of the null space.
- `tolerance::Float64`: tolerance used for rank determination.
- `condition_number::Float64`: ratio of the largest to smallest eigenvalue.
- `positive_definite::Bool`: whether the information matrix is numerically PD.
"""
struct RandomEffectInformation
    batch::Int
    n_latent::Int
    labels::Vector{String}
    singular_values::Vector{Float64}
    eigenvalues::Vector{Float64}
    rank::Int
    nullity::Int
    tolerance::Float64
    condition_number::Float64
    positive_definite::Bool
end

"""
    IdentifiabilityReport{U, T, S}

Result of [`identifiability_report`](@ref). Contains the Hessian of the objective at the
evaluation point, its spectral decomposition, and a local identifiability verdict.

Fields:
- `method::Symbol`: estimation method used (e.g. `:mle`, `:laplace`).
- `objective::Symbol`: objective evaluated (`:nll`, `:map`, or `:laplace_nll`).
- `at::Symbol`: where the Hessian was evaluated (`:start` or `:fit`).
- `point_untransformed`: parameter values on the natural scale.
- `point_transformed`: parameter values on the transformed scale.
- `free_parameters::Vector{Symbol}`: names of the free (non-constant) parameters.
- `hessian::Matrix{Float64}`: Hessian of the objective on the transformed scale.
- `singular_values`, `eigenvalues`: spectral decomposition of the Hessian.
- `rank::Int`, `nullity::Int`: numerical rank and null-space dimension.
- `tolerance::Float64`: tolerance used for rank determination.
- `condition_number::Float64`: ratio of the largest to smallest singular value.
- `locally_identifiable::Bool`: `true` if the Hessian has full rank (nullity = 0).
- `null_directions::Vector{NullDirection}`: directions of non-identifiability.
- `random_effect_information::Vector{RandomEffectInformation}`: per-batch RE information.
- `settings::S`: NamedTuple of the tolerances and backend settings used.
"""
struct IdentifiabilityReport{U, T, S}
    method::Symbol
    objective::Symbol
    at::Symbol
    point_untransformed::U
    point_transformed::T
    free_parameters::Vector{Symbol}
    hessian::Matrix{Float64}
    singular_values::Vector{Float64}
    eigenvalues::Vector{Float64}
    rank::Int
    nullity::Int
    tolerance::Float64
    condition_number::Float64
    locally_identifiable::Bool
    null_directions::Vector{NullDirection}
    random_effect_information::Vector{RandomEffectInformation}
    settings::S
end

@inline _has_random_effects(dm::DataModel) = !isempty(get_re_names(dm.model.random.random))

function _ident_method_symbol(method)
    method isa MLE && return :mle
    method isa MAP && return :map
    method isa Laplace && return :laplace
    method isa LaplaceMAP && return :laplace_map
    method isa FOCEI && return :focei
    method isa FOCEIMAP && return :focei_map
    error("Unsupported method $(typeof(method)) for identifiability_report.")
end

function _resolve_ident_method(dm::DataModel, method_spec)
    if method_spec isa Symbol
        method_spec === :auto && return _has_random_effects(dm) ? Laplace() : MLE()
        method_spec === :mle && return MLE()
        method_spec === :map && return MAP()
        method_spec === :laplace && return Laplace()
        method_spec === :laplace_map && return LaplaceMAP()
        method_spec === :focei && return FOCEI()
        method_spec === :focei_map && return FOCEIMAP()
        error("Unknown identifiability method $(method_spec). Use :auto, :mle, :map, :laplace, :laplace_map, :focei, or :focei_map.")
    end
    if method_spec isa MLE || method_spec isa MAP || method_spec isa Laplace || method_spec isa LaplaceMAP || method_spec isa FOCEI || method_spec isa FOCEIMAP
        return method_spec
    end
    error("Unsupported method specification $(typeof(method_spec)) for identifiability_report.")
end

function _validate_ident_method(dm::DataModel, method)
    has_re = _has_random_effects(dm)
    if method isa MLE || method isa MAP
        has_re && error("MLE/MAP identifiability diagnostics require models without random effects. Use method=:laplace, :laplace_map, :focei, or :focei_map for random-effects models.")
    elseif method isa Laplace || method isa LaplaceMAP || method isa FOCEI || method isa FOCEIMAP
        has_re || error("Laplace/LaplaceMAP/FOCEI/FOCEIMAP identifiability diagnostics require random effects. Use method=:mle or :map for fixed-effects models.")
    else
        error("Unsupported method $(typeof(method)) for identifiability diagnostics.")
    end

    fe = dm.model.fixed.fixed
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("identifiability_report requires at least one fixed effect.")

    if method isa MAP
        priors = get_priors(fe)
        has_prior = !isempty(keys(priors)) && any(!(getfield(priors, k) isa Priorless) for k in keys(priors))
        has_prior || error("MAP identifiability diagnostics require at least one prior on fixed effects.")
    end

    if method isa LaplaceMAP || method isa FOCEIMAP
        priors = get_priors(fe)
        for name in fixed_names
            hasproperty(priors, name) || error("LaplaceMAP/FOCEIMAP identifiability diagnostics require priors on all fixed effects. Missing prior for $(name).")
            getfield(priors, name) isa Priorless &&
                error("LaplaceMAP/FOCEIMAP identifiability diagnostics require priors on all fixed effects. Priorless for $(name).")
        end
    end
    return nothing
end

function _select_point(fe::FixedEffects, at, default_sym::Symbol, fit_point::Union{Nothing, ComponentArray}=nothing)
    if at isa ComponentArray
        return at, :custom
    elseif at === :start
        return get_θ0_untransformed(fe), :start
    elseif at === :fit
        fit_point === nothing && error("at=:fit requires a fitted parameter vector.")
        return fit_point, :fit
    elseif at isa Symbol
        error("Unsupported at=$(at). Use :start, :fit, or a ComponentArray.")
    else
        error("Unsupported at argument of type $(typeof(at)). Use :start, :fit, or a ComponentArray.")
    end
end

function _prepare_ident_point(fe::FixedEffects, θ_at_u::ComponentArray, constants::NamedTuple)
    fixed_names = get_names(fe)
    for n in fixed_names
        hasproperty(θ_at_u, n) || error("Parameter vector at-point is missing fixed effect $(n).")
    end
    fixed_set = Set(fixed_names)
    for n in keys(constants)
        n in fixed_set || error("Unknown constant parameter $(n).")
    end

    free_names = [n for n in fixed_names if !(n in keys(constants))]
    isempty(free_names) && error("identifiability_report requires at least one free fixed effect. Remove constants or keep one parameter free.")

    transform = get_transform(fe)
    inv_transform = get_inverse_transform(fe)

    θ_const_u = deepcopy(θ_at_u)
    _apply_constants!(θ_const_u, constants)
    θ_const_t = transform(θ_const_u)
    θ_free_t = θ_const_t[free_names]

    ranges = Vector{UnitRange{Int}}(undef, length(free_names))
    offset = 1
    for (i, n) in enumerate(free_names)
        v = getproperty(θ_free_t, n)
        len = v isa Number ? 1 : length(vec(v))
        ranges[i] = offset:offset + len - 1
        offset += len
    end
    offset - 1 == length(θ_free_t) || error("Internal error while building free-parameter index map.")

    return (; free_names=free_names,
             ranges=ranges,
             transform=transform,
             inv_transform=inv_transform,
             θ_const_u=θ_const_u,
             θ_const_t=θ_const_t,
             θ_free_t=θ_free_t,
             axs_free=getaxes(θ_free_t),
             axs_full=getaxes(θ_const_t))
end

function _merge_full_theta(θ_const_t, axs_full, θt_free, free_names)
    T = eltype(θt_free)
    θt_full = ComponentArray(T.(θ_const_t), axs_full)
    for n in free_names
        setproperty!(θt_full, n, getproperty(θt_free, n))
    end
    return θt_full
end

function _build_ll_cache_ident(dm::DataModel;
                               ode_args::Tuple=(),
                               ode_kwargs::NamedTuple=NamedTuple(),
                               serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial())
    if serialization isa SciMLBase.EnsembleThreads
        return build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=Threads.maxthreadid(), force_saveat=true)
    end
    return build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)
end

function _laplace_seed(rng::AbstractRNG, rng_seed::Union{Nothing, UInt64})
    return rng_seed === nothing ? rand(rng, UInt64) : rng_seed
end

function _init_laplace_eval_cache(n_batches::Int, T::Type{<:Real})
    bstar_cache = _LaplaceBStarCache([Vector{T}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = _LaplaceGradCache([Vector{T}() for _ in 1:n_batches],
                                   fill(T(NaN), n_batches),
                                   [Vector{T}() for _ in 1:n_batches],
                                   falses(n_batches))
    ad_cache = _init_laplace_ad_cache(n_batches)
    hess_cache = _init_laplace_hess_cache(T, n_batches)
    return _LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)
end

function _svd_tol(svals::AbstractVector, atol::Real, rtol::Real)
    isempty(svals) && return float(atol)
    return max(float(atol), float(rtol) * maximum(svals))
end

function _condition_number_from_svals(svals::AbstractVector, tol::Real)
    isempty(svals) && return 1.0
    n_nonzero = count(>(tol), svals)
    n_nonzero == 0 && return Inf
    n_nonzero < length(svals) && return Inf
    return svals[1] / svals[end]
end

function _null_loadings(v::AbstractVector, free_names::Vector{Symbol}, ranges::Vector{UnitRange{Int}})
    mags = Vector{Float64}(undef, length(free_names))
    total = 0.0
    for i in eachindex(free_names)
        m = norm(view(v, ranges[i]))
        mags[i] = m
        total += m
    end
    if total > 0
        @inbounds for i in eachindex(mags)
            mags[i] /= total
        end
    end
    return NamedTuple{Tuple(free_names)}(Tuple(mags))
end

function _build_null_directions(H::AbstractMatrix,
                                free_names::Vector{Symbol},
                                ranges::Vector{UnitRange{Int}},
                                tol::Real)
    size(H, 1) == 0 && return NullDirection[]
    F = svd(H)
    out = NullDirection[]
    for i in eachindex(F.S)
        F.S[i] > tol && continue
        v = Vector{Float64}(F.V[:, i])
        load = _null_loadings(v, free_names, ranges)
        push!(out, NullDirection(Float64(F.S[i]), v, load))
    end
    return out
end

function _hessian_fd_from_grad(grad_fun::Function,
                               x0::Vector{Float64};
                               abs_step::Real,
                               rel_step::Real,
                               max_tries::Int)
    n = length(x0)
    H = zeros(Float64, n, n)
    for j in 1:n
        h = max(float(abs_step), float(rel_step) * max(abs(x0[j]), 1.0))
        ok = false
        for _ in 1:max_tries
            xp = copy(x0)
            xm = copy(x0)
            xp[j] += h
            xm[j] -= h
            gp = grad_fun(xp)
            gm = grad_fun(xm)
            if all(isfinite, gp) && all(isfinite, gm)
                @inbounds for i in 1:n
                    H[i, j] = (gp[i] - gm[i]) / (2h)
                end
                ok = true
                break
            end
            h *= 0.5
        end
        ok || error("Failed to compute finite-difference Hessian column $(j). Try larger fd_abs_step/fd_rel_step or use better starting values.")
    end
    return 0.5 .* (H .+ H')
end

function _gradient_fd_from_obj(obj_fun::Function,
                               x0::Vector{Float64};
                               abs_step::Real=1e-6,
                               rel_step::Real=1e-6,
                               max_tries::Int=8)
    n = length(x0)
    g = zeros(Float64, n)
    for j in 1:n
        h = max(float(abs_step), float(rel_step) * max(abs(x0[j]), 1.0))
        ok = false
        for _ in 1:max_tries
            xp = copy(x0)
            xm = copy(x0)
            xp[j] += h
            xm[j] -= h
            fp = obj_fun(xp)
            fm = obj_fun(xm)
            if isfinite(fp) && isfinite(fm)
                g[j] = (fp - fm) / (2h)
                ok = true
                break
            end
            h *= 0.5
        end
        ok || error("Failed to compute finite-difference gradient entry $(j).")
    end
    return g
end

function _laplace_batch_labels(dm::DataModel, info::_LaplaceBatchInfo)
    cache = dm.re_group_info.laplace_cache
    cache === nothing && return String[]
    labels = String[]
    for (ri, re) in enumerate(cache.re_names)
        re_info = info.re_info[ri]
        levels_all = cache.re_index[ri].levels
        for (li, level_id) in enumerate(re_info.map.levels)
            level = levels_all[level_id]
            r = re_info.ranges[li]
            if re_info.is_scalar || re_info.dim == 1
                push!(labels, string(re, "[", level, "]"))
            else
                for k in 1:length(r)
                    push!(labels, string(re, "[", level, "][", k, "]"))
                end
            end
        end
    end
    return labels
end

function _build_re_information(dm::DataModel,
                               method::Union{Laplace, LaplaceMAP, FOCEI, FOCEIMAP},
                               θu::ComponentArray,
                               batch_infos::Vector{_LaplaceBatchInfo},
                               const_cache::LaplaceConstantsCache,
                               ll_cache,
                               ebe_cache::_LaplaceCache,
                               seed::UInt64;
                               atol::Real,
                               rtol::Real)
    isempty(batch_infos) && return RandomEffectInformation[]
    ebe_serialization = ll_cache isa Vector ? SciMLBase.EnsembleThreads() : SciMLBase.EnsembleSerial()
    inner_opts = _resolve_inner_options(method.inner, dm)
    multistart_opts = _resolve_multistart_options(method.multistart, inner_opts)
    bstars = _laplace_get_bstar!(ebe_cache, dm, batch_infos, θu, const_cache, ll_cache;
                                 optimizer=inner_opts.optimizer,
                                 optim_kwargs=inner_opts.kwargs,
                                 adtype=inner_opts.adtype,
                                 grad_tol=inner_opts.grad_tol,
                                 theta_tol=0.0,
                                 multistart=multistart_opts,
                                 rng=Random.Xoshiro(seed),
                                 serialization=ebe_serialization)
    out = RandomEffectInformation[]
    hess_opts = if method isa Laplace || method isa LaplaceMAP
        method.hessian
    else
        method.fallback_hessian
    end
    for (bi, info) in enumerate(batch_infos)
        nb = info.n_b
        labels = _laplace_batch_labels(dm, info)
        nb == 0 && begin
            push!(out, RandomEffectInformation(bi, 0, labels, Float64[], Float64[], 0, 0, float(atol), 1.0, true))
            continue
        end
        b = bstars[bi]
        cache_for_h = ll_cache isa Vector ? ll_cache[1] : ll_cache
        _, H, _ = _laplace_logdet_negH(dm, info, θu, b, const_cache, cache_for_h, ebe_cache.ad_cache, bi;
                                       jitter=hess_opts.jitter,
                                       max_tries=hess_opts.max_tries,
                                       growth=hess_opts.growth,
                                       adaptive=hess_opts.adaptive,
                                       scale_factor=hess_opts.scale_factor,
                                       ctx="identifiability_re",
                                       hess_cache=nothing,
                                       use_cache=false)
        A = Symmetric(-H)
        svals = svdvals(Matrix(A))
        eigs = eigvals(A)
        tol = _svd_tol(svals, atol, rtol)
        rank = count(>(tol), svals)
        nullity = length(svals) - rank
        cond = _condition_number_from_svals(svals, tol)
        posdef = isempty(eigs) || minimum(eigs) > tol
        push!(out, RandomEffectInformation(bi, nb, labels, Float64.(svals), Float64.(eigs), rank, nullity, tol, cond, posdef))
    end
    return out
end

function _build_no_re_objective(dm::DataModel,
                                method::Union{MLE, MAP},
                                prep,
                                fe::FixedEffects;
                                penalty::NamedTuple=NamedTuple(),
                                ode_args::Tuple=(),
                                ode_kwargs::NamedTuple=NamedTuple(),
                                serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial())
    ll_cache = _build_ll_cache_ident(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization)
    has_penalty = !isempty(keys(penalty))
    has_prior = method isa MAP

    function obj_only(x_vec)
        θt_free = x_vec isa ComponentArray ? x_vec : ComponentArray(x_vec, prep.axs_free)
        θt_full = _merge_full_theta(prep.θ_const_t, prep.axs_full, θt_free, prep.free_names)
        θu = prep.inv_transform(θt_full)
        ll = loglikelihood(dm, θu, ComponentArray(); cache=ll_cache, serialization=serialization)
        ll == -Inf && return Inf
        obj = -ll
        if has_prior
            lp = logprior(fe, θu)
            lp == -Inf && return Inf
            obj += -lp
        end
        if has_penalty
            obj += _penalty_value(θu, penalty)
        end
        return obj
    end

    grad_fun = function (x)
        xv = Float64.(collect(x))
        try
            return Float64.(ForwardDiff.gradient(obj_only, xv))
        catch
            return _gradient_fd_from_obj(obj_only, xv)
        end
    end
    objective_symbol = method isa MAP ? :posterior : :likelihood
    return obj_only, grad_fun, objective_symbol, RandomEffectInformation[]
end

function _build_laplace_objective(dm::DataModel,
                                  method::Union{Laplace, LaplaceMAP, FOCEI, FOCEIMAP},
                                  prep,
                                  fe::FixedEffects;
                                  constants_re::NamedTuple=NamedTuple(),
                                  penalty::NamedTuple=NamedTuple(),
                                  ode_args::Tuple=(),
                                  ode_kwargs::NamedTuple=NamedTuple(),
                                  serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                                  rng::AbstractRNG=Random.default_rng(),
                                  rng_seed::Union{Nothing, UInt64}=nothing,
                                  atol::Real=1e-8,
                                  rtol::Real=sqrt(eps(Float64)))
    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)
    ll_cache = _build_ll_cache_ident(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, serialization=serialization)
    seed = _laplace_seed(rng, rng_seed)
    ebe_cache = _init_laplace_eval_cache(length(batch_infos), Float64)
    cache_opts = LaplaceCacheOptions(0.0)
    inner_opts = _resolve_inner_options(method.inner, dm)
    multistart_opts = _resolve_multistart_options(method.multistart, inner_opts)
    has_penalty = !isempty(keys(penalty))
    has_prior = method isa LaplaceMAP || method isa FOCEIMAP

    function eval_obj_grad(x_vec::Vector{Float64})
        θt_free = ComponentArray(x_vec, prep.axs_free)
        θt_full = _merge_full_theta(prep.θ_const_t, prep.axs_full, θt_free, prep.free_names)
        θu = prep.inv_transform(θt_full)

        obj, grad_u, _ = if method isa Laplace || method isa LaplaceMAP
            _laplace_objective_and_grad(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                        inner=inner_opts,
                                        hessian=method.hessian,
                                        cache_opts=cache_opts,
                                        multistart=multistart_opts,
                                        rng=Random.Xoshiro(seed),
                                        serialization=serialization)
        else
            _focei_objective_and_grad(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                      inner=inner_opts,
                                      info_opts=method.info,
                                      fallback_hessian=method.fallback_hessian,
                                      cache_opts=cache_opts,
                                      multistart=multistart_opts,
                                      rng=Random.Xoshiro(seed),
                                      serialization=serialization)
        end
        obj == Inf && return (Inf, fill(NaN, length(x_vec)), θu)

        if has_prior
            lp = logprior(fe, θu)
            lp == -Inf && return (Inf, fill(NaN, length(x_vec)), θu)
            obj += -lp
            lp_grad = ForwardDiff.gradient(v -> logprior(fe, v), θu)
            grad_u = grad_u .- lp_grad
        end

        if has_penalty
            obj += _penalty_value(θu, penalty)
            pen_grad = ForwardDiff.gradient(v -> _penalty_value(v, penalty), θu)
            grad_u = grad_u .+ pen_grad
        end

        grad_t = apply_inv_jacobian_T(prep.inv_transform, θt_full, grad_u)
        grad_free = similar(θt_free)
        for n in prep.free_names
            setproperty!(grad_free, n, getproperty(grad_t, n))
        end
        return (Float64(obj), Float64.(collect(grad_free)), θu)
    end

    obj_only = x -> eval_obj_grad(x)[1]
    grad_fun = x -> eval_obj_grad(x)[2]
    objective_symbol = if method isa LaplaceMAP
        :laplace_posterior
    elseif method isa Laplace
        :laplace_likelihood
    elseif method isa FOCEIMAP
        :focei_posterior
    else
        :focei_likelihood
    end
    re_info_fun = x -> begin
        θt_free = ComponentArray(x, prep.axs_free)
        θt_full = _merge_full_theta(prep.θ_const_t, prep.axs_full, θt_free, prep.free_names)
        θu = prep.inv_transform(θt_full)
        _build_re_information(dm, method, θu, batch_infos, const_cache, ll_cache, ebe_cache, seed; atol=atol, rtol=rtol)
    end
    return obj_only, grad_fun, objective_symbol, re_info_fun
end

function _compute_hessian(obj_only::Function,
                          grad_fun::Function,
                          x0::Vector{Float64};
                          backend::Symbol,
                          abs_step::Real,
                          rel_step::Real,
                          max_tries::Int)
    if backend == :forwarddiff
        H = ForwardDiff.hessian(obj_only, x0)
        return 0.5 .* (Float64.(H) .+ Float64.(H'))
    elseif backend == :fd_gradient
        return _hessian_fd_from_grad(grad_fun, x0; abs_step=abs_step, rel_step=rel_step, max_tries=max_tries)
    end
    error("Unknown hessian_backend $(backend). Use :auto, :forwarddiff, or :fd_gradient.")
end

function _identifiability_report(dm::DataModel,
                                 method,
                                 at,
                                 fit_point::Union{Nothing, ComponentArray};
                                 constants::NamedTuple=NamedTuple(),
                                 constants_re::NamedTuple=NamedTuple(),
                                 penalty::NamedTuple=NamedTuple(),
                                 ode_args::Tuple=(),
                                 ode_kwargs::NamedTuple=NamedTuple(),
                                 serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                                 rng::AbstractRNG=Random.default_rng(),
                                 rng_seed::Union{Nothing, UInt64}=nothing,
                                 atol::Real=1e-8,
                                 rtol::Real=sqrt(eps(Float64)),
                                 hessian_backend::Symbol=:auto,
                                 fd_abs_step::Real=1e-4,
                                 fd_rel_step::Real=1e-3,
                                 fd_max_tries::Int=8)
    fe = dm.model.fixed.fixed
    θ_at_u, at_sym = _select_point(fe, at, :start, fit_point)
    prep = _prepare_ident_point(fe, θ_at_u, constants)

    obj_only = nothing
    grad_fun = nothing
    objective_symbol = :likelihood
    re_info = RandomEffectInformation[]
    re_info_fun = x -> RandomEffectInformation[]
    default_backend = :forwarddiff

    if method isa MLE || method isa MAP
        obj_only, grad_fun, objective_symbol, re_info = _build_no_re_objective(dm, method, prep, fe;
                                                                                penalty=penalty,
                                                                                ode_args=ode_args,
                                                                                ode_kwargs=ode_kwargs,
                                                                                serialization=serialization)
        default_backend = :forwarddiff
    else
        obj_only, grad_fun, objective_symbol, re_info_fun = _build_laplace_objective(dm, method, prep, fe;
                                                                                      constants_re=constants_re,
                                                                                      penalty=penalty,
                                                                                      ode_args=ode_args,
                                                                                      ode_kwargs=ode_kwargs,
                                                                                      serialization=serialization,
                                                                                      rng=rng,
                                                                                      rng_seed=rng_seed,
                                                                                      atol=atol,
                                                                                      rtol=rtol)
        default_backend = :fd_gradient
    end

    backend = hessian_backend == :auto ? default_backend : hessian_backend
    x0 = Float64.(collect(prep.θ_free_t))
    backend_used = backend
    H = try
        _compute_hessian(obj_only, grad_fun, x0;
                         backend=backend,
                         abs_step=fd_abs_step,
                         rel_step=fd_rel_step,
                         max_tries=fd_max_tries)
    catch err
        if backend == :forwarddiff
            @warn "ForwardDiff Hessian failed in identifiability_report; falling back to finite-difference Hessian from gradients." error=sprint(showerror, err)
            backend_used = :fd_gradient
            _compute_hessian(obj_only, grad_fun, x0;
                             backend=:fd_gradient,
                             abs_step=fd_abs_step,
                             rel_step=fd_rel_step,
                             max_tries=fd_max_tries)
        else
            rethrow(err)
        end
    end

    Hsym = 0.5 .* (H .+ H')
    svals = svdvals(Hsym)
    eigs = eigvals(Symmetric(Hsym))
    tol = _svd_tol(svals, atol, rtol)
    rank = count(>(tol), svals)
    nullity = length(svals) - rank
    cond = _condition_number_from_svals(svals, tol)
    null_dirs = _build_null_directions(Hsym, prep.free_names, prep.ranges, tol)
    locally_identifiable = nullity == 0
    if method isa Laplace || method isa LaplaceMAP || method isa FOCEI || method isa FOCEIMAP
        re_info = re_info_fun(x0)
    end

    settings = (;
        atol=float(atol),
        rtol=float(rtol),
        hessian_backend=backend_used,
        fd_abs_step=float(fd_abs_step),
        fd_rel_step=float(fd_rel_step),
        fd_max_tries=fd_max_tries,
        serialization=serialization
    )

    return IdentifiabilityReport(_ident_method_symbol(method),
                                 objective_symbol,
                                 at_sym,
                                 prep.θ_const_u,
                                 prep.θ_const_t,
                                 prep.free_names,
                                 Matrix{Float64}(Hsym),
                                 Float64.(svals),
                                 Float64.(eigs),
                                 rank,
                                 nullity,
                                 tol,
                                 cond,
                                 locally_identifiable,
                                 null_dirs,
                                 re_info,
                                 settings)
end

"""
    identifiability_report(dm::DataModel; method, at, constants, constants_re, penalty,
                           ode_args, ode_kwargs, serialization, rng, rng_seed, atol, rtol,
                           hessian_backend, fd_abs_step, fd_rel_step, fd_max_tries)
                           -> IdentifiabilityReport

    identifiability_report(res::FitResult; method, at, constants, constants_re, penalty,
                           ode_args, ode_kwargs, serialization, rng, rng_seed, atol, rtol,
                           hessian_backend, fd_abs_step, fd_rel_step, fd_max_tries)
                           -> IdentifiabilityReport

Compute a local identifiability report by evaluating the Hessian of the chosen objective
at a specified parameter point and checking its rank.

When called with a `DataModel`, the starting values from the model definition are used by
default (`at=:start`). When called with a `FitResult`, the fitted parameter estimates are
used by default (`at=:fit`).

# Keyword Arguments
- `method::Union{Symbol, FittingMethod} = :auto`: estimation method whose objective is
  used. `:auto` selects `MLE` for models without random effects and `Laplace` otherwise.
  Supported symbols: `:mle`, `:map`, `:laplace`, `:laplace_map`, `:focei`, `:focei_map`.
- `at::Union{Symbol, ComponentArray} = :start`: evaluation point. `:start` uses the
  model initial values, `:fit` uses the fitted estimates (only for the `FitResult`
  method), or a `ComponentArray` of untransformed parameter values.
- `constants`, `constants_re`, `penalty`, `ode_args`, `ode_kwargs`, `serialization`, `rng`:
  forwarded to the objective; see [`fit_model`](@ref) for descriptions.
- `rng_seed::Union{Nothing, UInt64} = nothing`: optional fixed seed for reproducibility.
- `atol::Real = 1e-8`: absolute tolerance for Hessian rank determination.
- `rtol::Real = sqrt(eps(Float64))`: relative tolerance for Hessian rank determination.
- `hessian_backend::Symbol = :auto`: Hessian computation backend. `:auto` tries
  ForwardDiff then finite differences.
- `fd_abs_step::Real = 1e-4`: absolute finite-difference step size.
- `fd_rel_step::Real = 1e-3`: relative finite-difference step size.
- `fd_max_tries::Int = 8`: maximum step-size retry attempts for finite differences.

# Returns
An [`IdentifiabilityReport`](@ref) with the Hessian, its spectral decomposition, a
local identifiability verdict, and any null directions.
"""
function identifiability_report(dm::DataModel;
                                method::Union{Symbol, FittingMethod}=:auto,
                                at::Union{Symbol, ComponentArray}=:start,
                                constants::NamedTuple=NamedTuple(),
                                constants_re::NamedTuple=NamedTuple(),
                                penalty::NamedTuple=NamedTuple(),
                                ode_args::Tuple=(),
                                ode_kwargs::NamedTuple=NamedTuple(),
                                serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                                rng::AbstractRNG=Random.default_rng(),
                                rng_seed::Union{Nothing, UInt64}=nothing,
                                atol::Real=1e-8,
                                rtol::Real=sqrt(eps(Float64)),
                                hessian_backend::Symbol=:auto,
                                fd_abs_step::Real=1e-4,
                                fd_rel_step::Real=1e-3,
                                fd_max_tries::Int=8)
    method_use = _resolve_ident_method(dm, method)
    _validate_ident_method(dm, method_use)
    return _identifiability_report(dm,
                                   method_use,
                                   at,
                                   nothing;
                                   constants=constants,
                                   constants_re=constants_re,
                                   penalty=penalty,
                                   ode_args=ode_args,
                                   ode_kwargs=ode_kwargs,
                                   serialization=serialization,
                                   rng=rng,
                                   rng_seed=rng_seed,
                                   atol=atol,
                                   rtol=rtol,
                                   hessian_backend=hessian_backend,
                                   fd_abs_step=fd_abs_step,
                                   fd_rel_step=fd_rel_step,
                                   fd_max_tries=fd_max_tries)
end

function identifiability_report(res::FitResult;
                                method::Union{Symbol, FittingMethod}=:fit,
                                at::Union{Symbol, ComponentArray}=:fit,
                                constants::Union{Nothing, NamedTuple}=nothing,
                                constants_re::Union{Nothing, NamedTuple}=nothing,
                                penalty::Union{Nothing, NamedTuple}=nothing,
                                ode_args::Union{Nothing, Tuple}=nothing,
                                ode_kwargs::Union{Nothing, NamedTuple}=nothing,
                                serialization::Union{Nothing, SciMLBase.EnsembleAlgorithm}=nothing,
                                rng::Union{Nothing, AbstractRNG}=nothing,
                                rng_seed::Union{Nothing, UInt64}=nothing,
                                atol::Real=1e-8,
                                rtol::Real=sqrt(eps(Float64)),
                                hessian_backend::Symbol=:auto,
                                fd_abs_step::Real=1e-4,
                                fd_rel_step::Real=1e-3,
                                fd_max_tries::Int=8)
    dm = res.data_model
    dm === nothing && error("This fit result does not store a DataModel; call identifiability_report(dm, ...) instead.")

    method_use = if method === :fit
        if res.method isa MLE || res.method isa MAP || res.method isa Laplace || res.method isa LaplaceMAP || res.method isa FOCEI || res.method isa FOCEIMAP
            res.method
        else
            @warn "identifiability_report does not have a direct objective for $(typeof(res.method)); using method=:auto."
            :auto
        end
    else
        method
    end
    method_use = _resolve_ident_method(dm, method_use)
    _validate_ident_method(dm, method_use)

    fitkw = res.fit_kwargs
    constants_use = constants === nothing ? (haskey(fitkw, :constants) ? getfield(fitkw, :constants) : NamedTuple()) : constants
    constants_re_use = constants_re === nothing ? (haskey(fitkw, :constants_re) ? getfield(fitkw, :constants_re) : NamedTuple()) : constants_re
    penalty_use = penalty === nothing ? (haskey(fitkw, :penalty) ? getfield(fitkw, :penalty) : NamedTuple()) : penalty
    ode_args_use = ode_args === nothing ? (haskey(fitkw, :ode_args) ? getfield(fitkw, :ode_args) : ()) : ode_args
    ode_kwargs_use = ode_kwargs === nothing ? (haskey(fitkw, :ode_kwargs) ? getfield(fitkw, :ode_kwargs) : NamedTuple()) : ode_kwargs
    serialization_use = serialization === nothing ? (haskey(fitkw, :serialization) ? getfield(fitkw, :serialization) : EnsembleSerial()) : serialization
    rng_use = rng === nothing ? (haskey(fitkw, :rng) ? getfield(fitkw, :rng) : Random.default_rng()) : rng

    fit_point = get_params(res; scale=:untransformed)

    return _identifiability_report(dm,
                                   method_use,
                                   at,
                                   fit_point;
                                   constants=constants_use,
                                   constants_re=constants_re_use,
                                   penalty=penalty_use,
                                   ode_args=ode_args_use,
                                   ode_kwargs=ode_kwargs_use,
                                   serialization=serialization_use,
                                   rng=rng_use,
                                   rng_seed=rng_seed,
                                   atol=atol,
                                   rtol=rtol,
                                   hessian_backend=hessian_backend,
                                   fd_abs_step=fd_abs_step,
                                   fd_rel_step=fd_rel_step,
                                   fd_max_tries=fd_max_tries)
end
