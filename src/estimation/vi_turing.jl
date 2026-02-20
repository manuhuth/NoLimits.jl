export VI
export VIResult

using Turing
using AdvancedVI
using DynamicPPL
using SciMLBase
using ComponentArrays
using Distributions
using Random

"""
    VI(; turing_kwargs=NamedTuple()) <: FittingMethod

Variational inference via Turing/AdvancedVI for models with or without random effects.
All free fixed effects and random effects must have prior distributions.

`turing_kwargs` controls VI behavior and is forwarded to `Turing.vi` after removing
NoLimits-managed keys:
- `max_iter::Int` (default: `1000`)
- `family::Symbol` (`:meanfield` or `:fullrank`, default: `:meanfield`)
- `q_init` (optional custom variational family)
- `adtype` (default: `Turing.AutoForwardDiff()`)
- `progress` / `show_progress` (default: `false`)
- `convergence_window`, `convergence_rtol`, `convergence_atol` (NoLimits convergence rule)
"""
struct VI{K} <: FittingMethod
    turing_kwargs::K
end

VI(; turing_kwargs=NamedTuple()) = VI(turing_kwargs)

"""
    VIResult{Q, T, S, N, O, V, C} <: MethodResult

Method-specific result from a [`VI`](@ref) fit. Stores the variational posterior,
optimization trace/state, ELBO summary, and observed data.
"""
struct VIResult{Q, T, S, N, O, V, C} <: MethodResult
    posterior::Q
    trace::T
    state::S
    n_iter::Int
    max_iter::Int
    final_elbo::Float64
    converged::Bool
    notes::N
    observed::O
    varinfo::V
    coord_names::C
end

get_variational_posterior(res::VIResult) = res.posterior
get_vi_trace(res::VIResult) = res.trace
get_vi_state(res::VIResult) = res.state

@inline _as_namedtuple(x::NamedTuple) = x
@inline _as_namedtuple(x::Base.Iterators.Pairs) = NamedTuple(x)
@inline _as_namedtuple(x) = x isa NamedTuple ? x : NamedTuple(x)

function _vi_unpack_output(out)
    out isa Tuple || error("Unexpected VI output: expected tuple from Turing.vi.")
    length(out) == 3 || error("Unexpected VI output tuple length $(length(out)); expected 3.")
    q, b, c = out
    b_is_info = b isa AbstractVector && (isempty(b) || first(b) isa NamedTuple)
    c_is_info = c isa AbstractVector && (isempty(c) || first(c) isa NamedTuple)
    b_is_state = b isa NamedTuple
    c_is_state = c isa NamedTuple
    if b_is_info && c_is_state
        return q, b, c
    elseif c_is_info && b_is_state
        return q, c, b
    elseif b_is_info
        return q, b, c
    elseif c_is_info
        return q, c, b
    end
    error("Could not infer VI output ordering from Turing.vi return values.")
end

function _vi_info_elbos(info)
    isempty(info) && return Float64[]
    out = Vector{Float64}(undef, length(info))
    for i in eachindex(info)
        row = _as_namedtuple(info[i])
        haskey(row, :elbo) || error("VI trace entry $(i) is missing :elbo.")
        out[i] = Float64(getfield(row, :elbo))
    end
    return out
end

function _vi_converged(info, max_iter::Int; window::Int=20, rtol::Float64=1e-3, atol::Float64=1e-6)
    isempty(info) && return false
    if length(info) < max_iter
        return true
    end
    elbos = _vi_info_elbos(info)
    all(isfinite, elbos) || return false
    length(elbos) >= 2 || return false
    w = min(window, length(elbos) - 1)
    w >= 1 || return false
    tail = @view elbos[(end - w):end]
    deltas = abs.(diff(tail))
    scale = max(1.0, abs(elbos[end]))
    return maximum(deltas) <= atol + rtol * scale
end

function _vi_coord_names(varinfo)
    names = Symbol[]
    syms = DynamicPPL.syms(varinfo)
    for s in syms
        md = getfield(varinfo.metadata, s)
        vns = md.vns
        vals_len = length(md.vals)
        vns_len = length(vns)
        if vns_len == 0
            for i in 1:vals_len
                push!(names, Symbol(string(s), "[", i, "]"))
            end
            continue
        end
        if vns_len == vals_len
            append!(names, Symbol.(string.(vns)))
            continue
        end
        if vns_len == 1
            base = string(vns[1])
            for i in 1:vals_len
                push!(names, Symbol(base, "[", i, "]"))
            end
            continue
        end
        if vals_len % vns_len == 0
            per = div(vals_len, vns_len)
            for vn in vns
                base = string(vn)
                if per == 1
                    push!(names, Symbol(base))
                else
                    for j in 1:per
                        push!(names, Symbol(base, "[", j, "]"))
                    end
                end
            end
            continue
        end
        base = string(s)
        for i in 1:vals_len
            push!(names, Symbol(base, "[", i, "]"))
        end
    end
    return names
end

function sample_posterior(res::VIResult; n_draws::Int=1000, rng::AbstractRNG=Random.default_rng(), return_names::Bool=false)
    n_draws >= 1 || error("n_draws must be >= 1.")
    raw = rand(rng, res.posterior, n_draws)
    mat = raw isa AbstractVector ? reshape(raw, :, 1) : Matrix(raw)
    draws = Matrix(permutedims(mat))
    if return_names
        return (draws=draws, names=res.coord_names)
    end
    return draws
end

function _fit_model(dm::DataModel, method::VI, args...;
                    constants::NamedTuple=NamedTuple(),
                    constants_re::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                    rng::AbstractRNG=Random.default_rng(),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_data_model::Bool=true)
    fit_kwargs = (constants=constants,
                  constants_re=constants_re,
                  penalty=penalty,
                  ode_args=ode_args,
                  ode_kwargs=ode_kwargs,
                  serialization=serialization,
                  rng=rng,
                  theta_0_untransformed=theta_0_untransformed,
                  store_data_model=store_data_model)
    re_names = get_re_names(dm.model.random.random)
    isempty(keys(penalty)) || error("VI does not support penalty terms. Use priors and MAP instead.")

    fe = dm.model.fixed.fixed
    _warn_if_scaled_params(fe; method_name="VI")
    priors = get_priors(fe)
    fixed_names = get_names(fe)
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    free_names = [n for n in fixed_names if !(n in keys(constants))]
    if isempty(free_names) && isempty(re_names)
        error("VI requires at least one sampled parameter. For fixed-effects-only models, leave at least one fixed effect free.")
    end
    for name in free_names
        haskey(priors, name) || error("VI requires priors on all free fixed effects. Missing prior for $(name).")
        getfield(priors, name) isa Priorless && error("VI requires priors on all free fixed effects. Priorless for $(name).")
    end
    if theta_0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta_0_untransformed, n) || error("theta_0_untransformed is missing parameter $(n).")
        end
        @debug "theta_0_untransformed is currently not used by VI unless turing_kwargs provides q_init."
    end

    cache = serialization isa SciMLBase.EnsembleThreads ?
            build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=Threads.maxthreadid(), force_saveat=true) :
            build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)

    free_names_t = Tuple(free_names)
    priors_nt = NamedTuple{free_names_t}(Tuple(getfield(priors, n) for n in free_names))
    model = nothing
    if isempty(re_names)
        fname = _build_turing_model(fixed_names, free_names)
        model_fn = Base.invokelatest(getfield, @__MODULE__, fname)
        model = Base.invokelatest(model_fn, dm, cache, serialization, priors_nt, constants)
    else
        fixed_maps = _normalize_constants_re(dm, constants_re)
        _build_constants_cache(dm, fixed_maps)
        const_covs = [ind.const_cov for ind in dm.individuals]
        dists_builder = get_create_random_effect_distribution(dm.model.random.random)
        model_funs = get_model_funs(dm.model)
        helpers = get_helper_funs(dm.model)
        re_pairs = Pair{Symbol, Any}[]
        for re in re_names
            reps_map = Dict{Any, Int}()
            for (i, ind) in enumerate(dm.individuals)
                g = getfield(ind.re_groups, re)
                if g isa AbstractVector
                    for gv in g
                        haskey(reps_map, gv) || (reps_map[gv] = i)
                    end
                else
                    haskey(reps_map, g) || (reps_map[g] = i)
                end
            end
            levels_all = getfield(dm.re_group_info.values, re)
            fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
            levels_free = Any[]
            reps = Int[]
            for lvl in levels_all
                haskey(fixed, lvl) && continue
                push!(levels_free, lvl)
                push!(reps, reps_map[lvl])
            end
            level_to_index = Dict{Any, Int}()
            for (i, lvl) in enumerate(levels_free)
                level_to_index[lvl] = i
            end
            dim = 0
            is_scalar = true
            if !isempty(levels_free)
                rep = reps[1]
                θ0_re = _symmetrize_psd_params(get_θ0_untransformed(fe), fe)
                dists = dists_builder(θ0_re, const_covs[rep], model_funs, helpers)
                dist = getproperty(dists, re)
                is_scalar = dist isa Distributions.UnivariateDistribution
                dim = is_scalar ? 1 : length(dist)
                dim == 0 && error("Random effect $(re) has zero dimension.")
            end
            push!(re_pairs, re => _MCMCReMeta(levels_free, level_to_index, reps, dim, is_scalar))
        end
        re_meta = NamedTuple(re_pairs)
        fname = _build_turing_model_re(fixed_names, free_names, re_names)
        model_fn = Base.invokelatest(getfield, @__MODULE__, fname)
        model = Base.invokelatest(model_fn, dm, cache, serialization, priors_nt, constants, re_names, re_meta, fixed_maps, const_covs)
    end
    f_old = model.f
    f_wrap = (fargs...)->Base.invokelatest(f_old, fargs...)
    miss = typeof(model).parameters[4]
    threaded = typeof(model).parameters[8]
    model = DynamicPPL.Model{threaded, miss}(f_wrap, model.args, model.defaults, model.context)

    max_iter = Int(get(method.turing_kwargs, :max_iter, 1000))
    max_iter >= 1 || error("VI requires max_iter >= 1.")
    family = get(method.turing_kwargs, :family, :meanfield)
    family in (:meanfield, :fullrank) || error("VI family must be :meanfield or :fullrank.")
    q_init = get(method.turing_kwargs, :q_init, nothing)
    adtype = get(method.turing_kwargs, :adtype, Turing.AutoForwardDiff())
    show_progress = Bool(get(method.turing_kwargs, :show_progress, get(method.turing_kwargs, :progress, false)))
    algorithm = get(method.turing_kwargs, :algorithm, nothing)
    conv_window = Int(get(method.turing_kwargs, :convergence_window, 20))
    conv_rtol = Float64(get(method.turing_kwargs, :convergence_rtol, 1e-3))
    conv_atol = Float64(get(method.turing_kwargs, :convergence_atol, 1e-6))
    conv_window >= 1 || error("VI convergence_window must be >= 1.")
    conv_rtol >= 0 || error("VI convergence_rtol must be >= 0.")
    conv_atol >= 0 || error("VI convergence_atol must be >= 0.")

    vi_kwargs = Base.structdiff(method.turing_kwargs,
                                (max_iter=0,
                                 family=:meanfield,
                                 q_init=nothing,
                                 adtype=nothing,
                                 progress=false,
                                 show_progress=false,
                                 algorithm=nothing,
                                 convergence_window=0,
                                 convergence_rtol=0.0,
                                 convergence_atol=0.0))
    if q_init === nothing
        if family == :meanfield
            q_init = Turing.q_meanfield_gaussian(rng, model)
        else
            q_init = Turing.q_fullrank_gaussian(rng, model)
        end
    end

    _set_turing_adbackend!(adtype)
    out = if algorithm === nothing
        Turing.vi(rng, model, q_init, max_iter; adtype=adtype, show_progress=show_progress, vi_kwargs...)
    else
        Turing.vi(rng, model, q_init, max_iter; adtype=adtype, algorithm=algorithm, show_progress=show_progress, vi_kwargs...)
    end
    posterior, trace, state = _vi_unpack_output(out)
    n_iter = length(trace)
    elbos = _vi_info_elbos(trace)
    final_elbo = isempty(elbos) ? NaN : elbos[end]
    converged = _vi_converged(trace, max_iter; window=conv_window, rtol=conv_rtol, atol=conv_atol)

    varinfo = DynamicPPL.VarInfo(model)
    coord_names = _vi_coord_names(varinfo)
    obs = dm.df[:, dm.config.obs_cols]
    summary = FitSummary(final_elbo, converged,
                         FitParameters(ComponentArray(), ComponentArray()),
                         NamedTuple())
    diagnostics = FitDiagnostics((;),
                                 (family=family, algorithm=algorithm === nothing ? :default : algorithm, adtype=adtype),
                                 (n_iter=n_iter, max_iter=max_iter),
                                 (final_elbo=final_elbo,
                                  convergence_window=conv_window,
                                  convergence_rtol=conv_rtol,
                                  convergence_atol=conv_atol))
    result = VIResult(posterior, trace, state, n_iter, max_iter, final_elbo, converged,
                      NamedTuple(), obs, varinfo, coord_names)
    return FitResult(method, result, summary, diagnostics, store_data_model ? dm : nothing, args, fit_kwargs)
end
