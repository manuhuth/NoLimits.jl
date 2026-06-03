export Pooled
export PooledMap
export PooledResult

using Optimization
using OptimizationOptimJL
using SciMLBase
using ComponentArrays
using Random
using LineSearches
using OptimizationBBO

"""
    Pooled(; optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds) <: FittingMethod

Pooled estimation for models with random effects. Each individual's random effects are
set to the **mean of their RE distribution** (subject-wise, using each individual's
covariates), then the data log-likelihood alone is optimised over the free fixed effects.

RE distribution parameters (those appearing in `@randomEffects`) are automatically held
constant at their initial values and are **not estimated**. The RE prior is never evaluated.

This is equivalent to plugging in the prior mean for every individual and then running MLE.

# Keyword Arguments
- `optimizer`: Optimization.jl-compatible optimiser. Defaults to `LBFGS` with backtracking.
- `optim_kwargs::NamedTuple = NamedTuple()`: forwarded to `Optimization.solve`.
- `adtype`: AD backend. Defaults to `AutoForwardDiff()`.
- `lb`/`ub`: bounds on the transformed scale, or `nothing` to use model-declared bounds.
- `ignore_model_bounds::Bool = false`: ignore bounds declared in `@fixedEffects`.
"""
struct Pooled{O, K, A, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
    ignore_model_bounds::Bool
end

Pooled(; optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
       optim_kwargs=NamedTuple(),
       adtype=Optimization.AutoForwardDiff(),
       lb=nothing,
       ub=nothing,
       ignore_model_bounds=false) = Pooled(optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds)

"""
    PooledMap(; optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds) <: FittingMethod

Like [`Pooled`](@ref), but adds the log-prior of the free fixed effects to the objective
(MAP on the data likelihood with RE fixed to their distributional means). Requires priors
on at least one free fixed effect.

RE distribution parameters are held constant and their priors are not updated during
optimisation (they contribute a constant offset to the reported objective).
"""
struct PooledMap{O, K, A, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
    ignore_model_bounds::Bool
end

PooledMap(; optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
          optim_kwargs=NamedTuple(),
          adtype=Optimization.AutoForwardDiff(),
          lb=nothing,
          ub=nothing,
          ignore_model_bounds=false) = PooledMap(optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds)

"""
    PooledResult{S, O, I, R, N, E} <: MethodResult

Method-specific result from a [`Pooled`](@ref) or [`PooledMap`](@ref) fit. Stores the
optimisation solution plus the precomputed per-individual random effects.
"""
struct PooledResult{S, O, I, R, N, E} <: MethodResult
    solution::S
    objective::O
    iterations::I
    raw::R
    notes::N
    eta_vec::E   # Vector{ComponentArray} — precomputed per-individual η
end

# ─── helper: detect which fixed-effect names appear in RE distribution expressions ──

function _detect_re_dist_params(dm::DataModel)
    model       = get_model(dm)
    re_syms_nt  = get_re_syms(model.random.random)
    all_re_syms = reduce((s, v) -> union(s, v), values(re_syms_nt); init=Set{Symbol}())
    fe_names    = Set(get_names(model.fixed.fixed))
    return intersect(all_re_syms, fe_names)
end

# ─── precompute per-individual η from distributional mean/median ─────────────────

function _compute_pooled_etas(dm::DataModel, θ::ComponentArray)
    model     = get_model(dm)
    lp_cache  = dm.re_group_info.laplace_cache
    lp_cache === nothing && error("Pooled() requires a model with random effects.")
    re_names      = lp_cache.re_names
    isempty(re_names) && error("Pooled() requires at least one random effect.")
    dists_builder = get_create_random_effect_distribution(model.random.random)
    model_funs    = get_model_funs(model)
    helpers       = get_helper_funs(model)
    individuals   = get_individuals(dm)
    n             = length(individuals)
    η_vec = Vector{ComponentArray}(undef, n)
    for i in 1:n
        ind   = individuals[i]
        dists = dists_builder(θ, ind.const_cov, model_funs, helpers)
        nt_pairs = Pair{Symbol, Any}[]
        for (ri, re) in enumerate(re_names)
            dist = getproperty(dists, re)
            dim  = lp_cache.dims[ri]
            val  = _re_start_value(dist, dim, Float64)
            push!(nt_pairs, re => val)
        end
        η_vec[i] = ComponentArray(NamedTuple(nt_pairs))
    end
    return η_vec
end

# ─── convert per-individual η to per-level DataFrames ────────────────────────────

function _pooled_re_dataframes(dm::DataModel, η_vec::Vector{<:ComponentArray};
                                flatten::Bool=true)
    lp_cache = dm.re_group_info.laplace_cache
    lp_cache === nothing && return NamedTuple()
    re_names = lp_cache.re_names
    isempty(re_names) && return NamedTuple()
    re_groups = get_re_groups(dm.model.random.random)

    # (ri, lvl_id) → first individual index with that level
    level_to_ind = Dict{Tuple{Int, Int}, Int}()
    for i in 1:length(η_vec)
        for ri in 1:length(re_names)
            ids = lp_cache.ind_level_ids[i][ri]
            isempty(ids) && continue
            key = (ri, ids[1])
            haskey(level_to_ind, key) || (level_to_ind[key] = i)
        end
    end

    out_pairs = Pair{Symbol, Any}[]
    for (ri, re) in enumerate(re_names)
        col        = getfield(re_groups, re)
        levels_all = lp_cache.re_index[ri].levels
        dim        = lp_cache.dims[ri]

        rows      = Any[]
        vals_flat = Vector{Vector{Any}}()
        for (lvl_id, lvl_val) in enumerate(levels_all)
            i = get(level_to_ind, (ri, lvl_id), 0)
            i == 0 && continue
            val = getproperty(η_vec[i], re)
            push!(rows, lvl_val)
            if flatten
                push!(vals_flat, val isa Number ? [val] : collect(vec(val)))
            else
                push!(vals_flat, [val])
            end
        end

        if flatten
            names = flatten_re_names(re, zeros(dim))
            df = DataFrame(col => rows)
            for j in 1:length(names)
                df[!, names[j]] = [vals_flat[k][j] for k in 1:length(vals_flat)]
            end
            push!(out_pairs, re => df)
        else
            push!(out_pairs, re => DataFrame(col => rows, :value => [v[1] for v in vals_flat]))
        end
    end
    return NamedTuple(out_pairs)
end

# ─── shared inner optimizer loop ─────────────────────────────────────────────────

function _fit_pooled(dm::DataModel, method;
                     η_vec::Vector,
                     constants::NamedTuple,
                     penalty::NamedTuple,
                     ode_args::Tuple,
                     ode_kwargs::NamedTuple,
                     serialization::SciMLBase.EnsembleAlgorithm,
                     add_term,
                     theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                     store_data_model::Bool=true,
                     fit_args::Tuple=(),
                     fit_kwargs::NamedTuple=NamedTuple())
    fe          = dm.model.fixed.fixed
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("This method requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("Pooled() has no free fixed effects to optimise (all are RE distribution " *
              "parameters). Add at least one fixed effect in @formulas or @DifferentialEquation.")
    free_names = [n for n in fixed_names if !(n in keys(constants))]

    θ0_u = get_θ0_untransformed(fe)
    if theta_0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta_0_untransformed, n) ||
                error("theta_0_untransformed is missing parameter $(n).")
        end
        θ0_u = theta_0_untransformed
    end

    transform     = get_transform(fe)
    inv_transform = get_inverse_transform(fe)
    θ0_t          = transform(θ0_u)
    θ_const_u     = deepcopy(θ0_u)
    _apply_constants!(θ_const_u, constants)
    θ_const_t = transform(θ_const_u)
    cache = serialization isa SciMLBase.EnsembleThreads ?
            build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs,
                           nthreads=Threads.maxthreadid(), force_saveat=true) :
            build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)

    θ0_free_t = ComponentArray(NamedTuple{Tuple(free_names)}(
                    Tuple(getproperty(θ0_t, n) for n in free_names)))
    axs = getaxes(θ0_free_t)

    function obj(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs)
        T       = eltype(θt_free)
        infT    = convert(T, Inf)
        θt_full = ComponentArray(T.(θ_const_t), getaxes(θ_const_t))
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu  = inv_transform(θt_full)
        add = add_term(θu)
        add == Inf && return infT
        ll  = loglikelihood(dm, θu, η_vec; cache=cache, serialization=serialization)
        ll == -Inf && return infT
        return -ll + _penalty_value(θu, penalty) + add
    end

    optf = OptimizationFunction(obj, method.adtype)
    lower_t, upper_t = get_bounds_transformed(fe)
    lower_t_free = ComponentArray(NamedTuple{Tuple(free_names)}(
                       Tuple(getproperty(lower_t, n) for n in free_names)))
    upper_t_free = ComponentArray(NamedTuple{Tuple(free_names)}(
                       Tuple(getproperty(upper_t, n) for n in free_names)))
    lower_t_free_vec = collect(lower_t_free)
    upper_t_free_vec = collect(upper_t_free)
    use_bounds = !method.ignore_model_bounds &&
                 !(all(isinf, lower_t_free_vec) && all(isinf, upper_t_free_vec))
    normalize_bound = function(bound, fallback)
        bound === nothing && return fallback
        bound isa Number  && (length(fallback) == 1 ||
            error("Scalar bounds are only valid when there is one free parameter."); return [bound])
        if bound isa ComponentArray || bound isa NamedTuple
            b = bound isa ComponentArray ? bound : ComponentArray(bound)
            b = ComponentArray(NamedTuple{Tuple(free_names)}(
                    Tuple(getproperty(b, n) for n in free_names)))
            return collect(b)
        end
        return collect(bound)
    end
    user_bounds = method.lb !== nothing || method.ub !== nothing
    if user_bounds && !isempty(keys(constants))
        @info "Bounds for constant parameters are ignored." constants=collect(keys(constants))
    end
    lb = user_bounds ? normalize_bound(method.lb, lower_t_free_vec) : lower_t_free_vec
    ub = user_bounds ? normalize_bound(method.ub, upper_t_free_vec) : upper_t_free_vec
    use_bounds = use_bounds || user_bounds
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO && !use_bounds
        error("BlackBoxOptim requires finite bounds. Pass them via Pooled(lb=..., ub=...) " *
              "or use default_bounds_from_start(dm; margin=...).")
    end
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO &&
       !(all(isfinite, lb) && all(isfinite, ub))
        error("BlackBoxOptim requires finite lower and upper bounds for all free parameters.")
    end
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO
        lb = map((u, m) -> isfinite(m) ? max(u, m) : u, collect(lb), lower_t_free_vec)
        ub = map((u, m) -> isfinite(m) ? min(u, m) : u, collect(ub), upper_t_free_vec)
        θ0_init = clamp.(collect(θ0_free_t), lb, ub)
    else
        θ0_init = θ0_free_t
    end
    prob = use_bounds ? OptimizationProblem(optf, θ0_init; lb=lb, ub=ub) :
                        OptimizationProblem(optf, θ0_init)
    sol  = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)

    θ_hat_t_raw  = sol.u
    θ_hat_t_free = θ_hat_t_raw isa ComponentArray ?
                   θ_hat_t_raw : ComponentArray(θ_hat_t_raw, axs)
    T = eltype(θ_hat_t_free)
    θ_hat_t = ComponentArray(T.(θ_const_t), getaxes(θ_const_t))
    for name in free_names
        setproperty!(θ_hat_t, name, getproperty(θ_hat_t_free, name))
    end
    θ_hat_u  = inv_transform(θ_hat_t)
    summary  = FitSummary(sol.objective, sol.retcode == SciMLBase.ReturnCode.Success,
                          FitParameters(θ_hat_t, θ_hat_u), NamedTuple())
    diag     = FitDiagnostics((;), (optimizer=method.optimizer,), (retcode=sol.retcode,), NamedTuple())
    niter    = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ?
               sol.stats.iterations : missing
    raw      = hasproperty(sol, :original) ? sol.original : sol
    result   = PooledResult(sol, sol.objective, niter, raw, NamedTuple(), η_vec)
    return FitResult(method, result, summary, diag,
                     store_data_model ? dm : nothing, fit_args, fit_kwargs)
end

# ─── fit_model dispatches ────────────────────────────────────────────────────────

function _fit_model(dm::DataModel, method::Pooled, args...;
                    constants::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleThreads(),
                    rng::AbstractRNG=Xoshiro(0),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_data_model::Bool=true)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && error("Pooled() requires a model with random effects. " *
                               "Use MLE() for fixed-effects-only models.")

    re_dist_params = _detect_re_dist_params(dm)
    θ_0_u = theta_0_untransformed !== nothing ?
            theta_0_untransformed : get_θ0_untransformed(dm.model.fixed.fixed)
    auto_constants   = NamedTuple(k => getproperty(θ_0_u, k) for k in re_dist_params)
    merged_constants = merge(auto_constants, constants)   # user constants override auto

    η_vec = _compute_pooled_etas(dm, θ_0_u)

    fit_kwargs = (constants=merged_constants,
                  penalty=penalty,
                  ode_args=ode_args,
                  ode_kwargs=ode_kwargs,
                  serialization=serialization,
                  rng=rng,
                  theta_0_untransformed=theta_0_untransformed,
                  store_data_model=store_data_model)
    return _fit_pooled(dm, method;
                       η_vec=η_vec,
                       constants=merged_constants,
                       penalty=penalty,
                       ode_args=ode_args,
                       ode_kwargs=ode_kwargs,
                       serialization=serialization,
                       add_term=_NoOpTerm(),
                       theta_0_untransformed=theta_0_untransformed,
                       store_data_model=store_data_model,
                       fit_args=args,
                       fit_kwargs=fit_kwargs)
end

function _fit_model(dm::DataModel, method::PooledMap, args...;
                    constants::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleThreads(),
                    rng::AbstractRNG=Xoshiro(0),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_data_model::Bool=true)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && error("PooledMap() requires a model with random effects. " *
                               "Use MAP() for fixed-effects-only models.")

    fe = dm.model.fixed.fixed
    priors    = get_priors(fe)
    has_prior = !isempty(keys(priors)) &&
                any(!(getfield(priors, k) isa Priorless) for k in keys(priors))
    has_prior || error("PooledMap() requires priors on fixed effects. Define priors in " *
                       "@fixedEffects (e.g. RealNumber(...; prior=Normal(...))) or use Pooled() instead.")

    re_dist_params = _detect_re_dist_params(dm)
    θ_0_u = theta_0_untransformed !== nothing ?
            theta_0_untransformed : get_θ0_untransformed(fe)
    auto_constants   = NamedTuple(k => getproperty(θ_0_u, k) for k in re_dist_params)
    merged_constants = merge(auto_constants, constants)

    η_vec = _compute_pooled_etas(dm, θ_0_u)

    fit_kwargs = (constants=merged_constants,
                  penalty=penalty,
                  ode_args=ode_args,
                  ode_kwargs=ode_kwargs,
                  serialization=serialization,
                  rng=rng,
                  theta_0_untransformed=theta_0_untransformed,
                  store_data_model=store_data_model)
    return _fit_pooled(dm, method;
                       η_vec=η_vec,
                       constants=merged_constants,
                       penalty=penalty,
                       ode_args=ode_args,
                       ode_kwargs=ode_kwargs,
                       serialization=serialization,
                       add_term=_MAPTerm(fe),
                       theta_0_untransformed=theta_0_untransformed,
                       store_data_model=store_data_model,
                       fit_args=args,
                       fit_kwargs=fit_kwargs)
end
