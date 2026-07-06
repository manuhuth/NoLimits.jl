# ghquadrature.jl
# GHQuadrature <: FittingMethod: Smolyak sparse-grid quadrature for NLME.

export GHQuadrature
export GHQuadratureResult

using Optimization
using OptimizationOptimJL
using OptimizationBBO
using SciMLBase
using ComponentArrays
using Random
using LineSearches

# ---------------------------------------------------------------------------
# Fitting method struct
# ---------------------------------------------------------------------------

"""
    GHQuadrature(; level, optimizer, optim_kwargs, adtype,
                 inner_options, inner_optimizer, inner_kwargs, inner_adtype,
                 inner_grad_tol, multistart_options, multistart_n, multistart_k,
                 multistart_grad_tol, multistart_max_rounds, multistart_sampling,
                 lb, ub, ignore_model_bounds) <: FittingMethod

Sparse-grid (Smolyak) quadrature for NLME marginal likelihood estimation.

Approximates the batch marginal likelihood via

    log L_batch ≈ signed_logsumexp_r [ log|W_r| + Σᵢ ℓᵢ(μ + Lzᵣ, θ) ]

where `{(zᵣ, Wᵣ)}` are Smolyak–Gauss-Hermite quadrature nodes/weights at the
requested `level`.  Unlike `Laplace`, there is no inner optimization during the
forward pass: the objective is fully differentiable by `AutoForwardDiff`.

# Keyword Arguments
- `level = 3`: Smolyak accuracy level.  May be:
  - `Int` (isotropic): same level for all RE groups.
  - `NamedTuple` (anisotropic): per-RE-group level, e.g.
    `level = (η_id = 3, η_site = 2)`.  RE groups not mentioned default to
    level 1.  The batch grid is the tensor product of per-group Smolyak grids.
  Levels 1–3 are numerically stable; higher levels may exhibit cancellation
  in signed logsumexp.
- `optimizer`: outer Optimization.jl-compatible optimizer.  Defaults to LBFGS
  with backtracking line search.
- `optim_kwargs::NamedTuple = NamedTuple()`: forwarded to `Optimization.solve`
  (e.g. `maxiters`, `reltol`).
- `adtype`: AD backend for the outer gradient.  Defaults to
  `AutoForwardDiff()`.
- `inner_options / inner_optimizer / inner_kwargs / inner_adtype / inner_grad_tol`:
  configure the Laplace-style inner optimizer used **only post-hoc** to compute
  empirical-Bayes mode estimates for `get_random_effects`.
- `multistart_options / multistart_n / multistart_k / multistart_grad_tol /
  multistart_max_rounds / multistart_sampling`: multistart settings for the
  post-hoc EB mode finder.
- `lb`, `ub`: box bounds on the transformed fixed-effect scale.  `nothing`
  falls back to model-declared bounds.
- `ignore_model_bounds::Bool = false`: if `true`, model-declared parameter
  bounds are ignored (user-supplied `lb`/`ub` still apply).
"""
struct GHQuadrature{LV, O, K, A, IO, MS, L, U} <: FittingMethod
    level::LV   # Int (isotropic) or NamedTuple (anisotropic per-RE-group)
    optimizer::O
    optim_kwargs::K
    adtype::A
    inner::IO
    multistart::MS
    lb::L
    ub::U
    ignore_model_bounds::Bool
end

function GHQuadrature(;
        level = 3,  # Int or NamedTuple for anisotropic levels
        optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()),
        optim_kwargs = NamedTuple(),
        adtype = Optimization.AutoForwardDiff(),
        inner_options = nothing,
        inner_optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0)),
        inner_kwargs = NamedTuple(),
        inner_adtype = Optimization.AutoForwardDiff(),
        inner_grad_tol = :auto,
        multistart_options = nothing,
        multistart_n = 50,
        multistart_k = 10,
        multistart_grad_tol = inner_grad_tol,
        multistart_max_rounds = 1,
        multistart_sampling = :lhs,
        lb = nothing,
        ub = nothing,
        ignore_model_bounds = false
)
    inner = inner_options === nothing ?
            LaplaceInnerOptions(
        inner_optimizer, inner_kwargs, inner_adtype, inner_grad_tol) :
            inner_options
    ms = multistart_options === nothing ?
         LaplaceMultistartOptions(multistart_n, multistart_k, multistart_grad_tol,
        multistart_max_rounds, multistart_sampling) :
         multistart_options
    GHQuadrature(
        level, optimizer, optim_kwargs, adtype, inner, ms, lb, ub, ignore_model_bounds)
end

# ---------------------------------------------------------------------------
# Result struct
# ---------------------------------------------------------------------------

"""
    GHQuadratureResult{S, O, I, R, N, B} <: MethodResult

Method-specific result from a [`GHQuadrature`](@ref) fit.  Stores the solution,
objective value, iteration count, raw solver result, optional notes, and
empirical-Bayes mode estimates for each batch (used by `get_random_effects`).
"""
struct GHQuadratureResult{S, O, I, R, N, B} <: MethodResult
    solution::S
    objective::O
    iterations::I
    raw::R
    notes::N
    eb_modes::B
end

# ---------------------------------------------------------------------------
# Internal: evaluate sparse-grid marginal log-likelihood for one batch
# ---------------------------------------------------------------------------

"""
    _ghq_batch_ll(dm, info, θu_re, const_cache, ll_cache, level) -> T

Evaluate the batch marginal log-likelihood using the sparse grid at `level`.

- `level::Int`: isotropic — same Smolyak level for all RE dimensions.
- `level::NamedTuple`: anisotropic — maps RE name to level; RE groups not
  mentioned default to level 1.  The batch grid is the tensor product of
  per-RE-group Smolyak grids.

For batches with `n_b == 0` (all RE are constant), returns the sum of
individual conditional log-likelihoods directly.
"""
function _ghq_batch_ll(dm::DataModel,
        info::_LaplaceBatchInfo,
        θu_re::ComponentArray,
        const_cache::LaplaceConstantsCache,
        ll_cache::_LLCache,
        level)   # Int or NamedTuple
    T = eltype(θu_re)
    if info.n_b == 0
        # All RE are constant — no integration needed.
        total = zero(T)
        empty_b = T[]
        for i in info.inds
            η_i = _build_eta_ind(dm, i, info, empty_b, const_cache, θu_re)
            lli = _loglikelihood_individual(dm, i, θu_re, η_i, ll_cache)
            !isfinite(lli) && return T(-Inf)
            total += T(lli)
        end
        const_ll = _const_re_prior_logf(dm, info, θu_re, const_cache, ll_cache)
        !isfinite(const_ll) && return T(-Inf)
        return total + T(const_ll)
    end

    # Select grid: isotropic (Int) or anisotropic (NamedTuple)
    sgrid = if level isa Int
        get_sparse_grid(info.n_b, level)
    else
        _build_anisotropic_batch_grid(dm, info, level)
    end

    # build_re_measure_from_batch may throw DomainError when distribution
    # parameters hit numerical limits (e.g. Beta with α→0 due to underflow).
    # Treat these as invalid parameter regions and return -Inf.
    re_measure = try
        build_re_measure_from_batch(info, θu_re, const_cache, dm, ll_cache)
    catch e
        e isa DomainError && return T(-Inf)
        rethrow(e)
    end
    ghq_ll = batch_loglik_ghq(dm, info, θu_re, re_measure, sgrid, const_cache, ll_cache)
    const_ll = _const_re_prior_logf(dm, info, θu_re, const_cache, ll_cache)
    (!isfinite(ghq_ll) || !isfinite(const_ll)) && return T(-Inf)
    return ghq_ll + T(const_ll)
end

"""
    _build_anisotropic_batch_grid(dm, info, level::NamedTuple) -> GHQuadratureNodes

Build (or retrieve from cache) the tensor-product anisotropic grid for this
batch.  `level` is a NamedTuple mapping RE name → Int level.  RE groups not
present in `level` default to level 1.

Returns the concatenated tensor-product grid over all RE groups that have
free levels (non-zero dimension) in this batch.
"""
function _build_anisotropic_batch_grid(
        dm::DataModel, info::_LaplaceBatchInfo, level::NamedTuple)
    re_names = dm.re_group_info.laplace_cache.re_names
    dims = Int[]
    levels = Int[]
    for (ri, re_name) in enumerate(re_names)
        re_info = info.re_info[ri]
        # Total free RE dimension for this RE group in this batch
        total_dim = sum(length(r) for r in re_info.ranges; init = 0)
        total_dim == 0 && continue
        l = haskey(level, re_name) ? getproperty(level, re_name) : 1
        push!(dims, total_dim)
        push!(levels, l)
    end
    isempty(dims) && error("_build_anisotropic_batch_grid: no free RE dimensions found")
    return get_anisotropic_grid(dims, levels)
end

# ---------------------------------------------------------------------------
# Positional free/constants merge (mirrors mle.jl): the flat positions of the
# free parameters inside the full transformed vector are precomputed once, so
# the per-evaluation merge is plain positional indexing. The old per-name
# `setproperty!` loop dispatches on runtime Symbols, which Enzyme's runtime
# rules reject and which costs a fresh ComponentArray + dynamic writes per call.
# ---------------------------------------------------------------------------
function _ghq_free_idx(θ_const_t::ComponentArray, θ0_free_t::ComponentArray)
    lab_full = ComponentArrays.labels(θ_const_t)
    lab_free = ComponentArrays.labels(θ0_free_t)
    pos_full = Dict{String, Int}(lab_full[i] => i for i in eachindex(lab_full))
    return Int[pos_full[l] for l in lab_free]
end

function _ghq_merge_full(θ_const_t_vec::Vector, free_idx::Vector{Int}, v_free, axs_full)
    T = eltype(v_free)
    full = Vector{T}(undef, length(θ_const_t_vec))
    @inbounds for i in eachindex(full)
        full[i] = θ_const_t_vec[i]
    end
    @inbounds for k in eachindex(free_idx)
        full[free_idx[k]] = v_free[k]
    end
    return ComponentArray(full, axs_full)
end

# ---------------------------------------------------------------------------
# Cache pre-population helpers
# ---------------------------------------------------------------------------

# Pre-build all grids needed for this fit so concurrent use is thread-safe.
function _prepopulate_ghq_cache(dm::DataModel, batch_infos, level)
    if level isa Int
        for d in unique(info.n_b for info in batch_infos)
            d > 0 && get_sparse_grid(d, level)
        end
    else
        # Anisotropic: build the per-batch tensor-product grids
        for info in batch_infos
            info.n_b > 0 && _build_anisotropic_batch_grid(dm, info, level)
        end
    end
end

# Return true if any batch grid exceeds `threshold` points.
function _any_batch_too_large(dm::DataModel, batch_infos, level, threshold::Int)
    for info in batch_infos
        info.n_b == 0 && continue
        npts = if level isa Int
            n_ghq_points(info.n_b, level)
        else
            size(_build_anisotropic_batch_grid(dm, info, level).nodes, 2)
        end
        npts > threshold && return true
    end
    return false
end

# ---------------------------------------------------------------------------
# _fit_model dispatch
# ---------------------------------------------------------------------------

function _fit_model_scalar(dm::DataModel, method::GHQuadrature, args...;
        constants::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        penalty::NamedTuple = NamedTuple(),
        extra_objective = nothing,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Random.default_rng(),
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_data_model::Bool = true)
    fit_kwargs = (constants = constants,
        constants_re = constants_re,
        penalty = penalty,
        ode_args = ode_args,
        ode_kwargs = ode_kwargs,
        serialization = serialization,
        rng = rng,
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model)

    # ── Validate ────────────────────────────────────────────────────────────
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) &&
        error("GHQuadrature requires random effects. Use MLE/MAP for fixed-effects-only models.")

    _ghq_validate_re_distributions(dm)

    fe = dm.model.fixed.fixed
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("GHQuadrature requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("GHQuadrature requires at least one free fixed effect.")

    free_names = [n for n in fixed_names if !(n in keys(constants))]

    # ── Starting point ───────────────────────────────────────────────────────
    θ0_u = get_θ0_untransformed(fe)
    if theta_0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta_0_untransformed, n) ||
                error("theta_0_untransformed is missing parameter $(n).")
        end
        θ0_u = theta_0_untransformed
    end

    transform = get_transform(fe)
    inv_transform = get_inverse_transform(fe)
    θ0_t = transform(θ0_u)
    θ_const_u = deepcopy(θ0_u)
    _apply_constants!(θ_const_u, constants)
    θ_const_t = transform(θ_const_u)

    inner_opts = _resolve_inner_options(method.inner, dm)
    multistart_opts = _resolve_multistart_options(method.multistart, inner_opts)

    # ── Infrastructure ───────────────────────────────────────────────────────
    pairing, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re)

    ll_cache = build_ll_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization, force_saveat = true)

    # Pre-populate sparse-grid cache for all unique free-RE dimensions.
    _prepopulate_ghq_cache(dm, batch_infos, method.level)
    if _any_batch_too_large(dm, batch_infos, method.level, 10_000)
        @warn "GHQuadrature: one or more batches have > 10,000 quadrature nodes. " *
              "Consider reducing `level` or checking your RE batch structure."
    end

    # EB-mode cache (used post-hoc for get_random_effects).
    n_batches = length(batch_infos)
    Tθ = eltype(θ0_t)
    ebe_cache = _init_laplace_eval_cache(n_batches, Tθ)

    # ── Objective ────────────────────────────────────────────────────────────
    θ0_free_t = θ0_t[free_names]
    axs_free = getaxes(θ0_free_t)
    axs_full = getaxes(θ_const_t)
    free_idx = _ghq_free_idx(θ_const_t, θ0_free_t)
    θ_const_t_vec = collect(θ_const_t)

    function obj(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
        T = eltype(θt_free)
        infT = convert(T, Inf)

        θt_full = _ghq_merge_full(
            θ_const_t_vec, free_idx, ComponentArrays.getdata(θt_free), axs_full)
        θu = inv_transform(θt_full)
        θu_re = _symmetrize_psd_params(θu, dm.model.fixed.fixed)

        total = if ll_cache isa AbstractVector
            results = Vector{T}(undef, length(batch_infos))
            bad = Threads.Atomic{Bool}(false)
            # Chunk-indexed cache assignment — `Threads.threadid()` indexing is
            # unsafe under task migration (two tasks could share one cache slot).
            n_chunks = length(ll_cache)
            Threads.@threads for c in 1:n_chunks
                cache_c = ll_cache[c]
                for bi in c:n_chunks:length(batch_infos)
                    if bad[]
                        results[bi] = zero(T)
                        continue
                    end
                    bll = _ghq_batch_ll(dm, batch_infos[bi], θu_re, const_cache,
                        cache_c, method.level)
                    if bll == -Inf
                        Threads.atomic_or!(bad, true)
                        results[bi] = zero(T)
                    else
                        results[bi] = T(bll)
                    end
                end
            end
            bad[] && return infT
            sum(results)
        else
            s = zero(T)
            for info in batch_infos
                bll = _ghq_batch_ll(dm, info, θu_re, const_cache, ll_cache, method.level)
                bll == -Inf && return infT
                s += bll
            end
            s
        end
        return -total + convert(T, _penalty_value(θu, penalty)) +
               (extra_objective === nothing ? zero(T) : convert(T, extra_objective(θu)))
    end

    # ── Bounds ───────────────────────────────────────────────────────────────
    optf = OptimizationFunction(obj, method.adtype)
    lower_t, upper_t = get_bounds_transformed(fe)
    lower_t_free = lower_t[free_names]
    upper_t_free = upper_t[free_names]
    lower_t_free_vec = collect(lower_t_free)
    upper_t_free_vec = collect(upper_t_free)

    use_bounds = !method.ignore_model_bounds &&
                 !(all(isinf, lower_t_free_vec) && all(isinf, upper_t_free_vec))
    user_bounds = method.lb !== nothing || method.ub !== nothing
    if user_bounds && !isempty(keys(constants))
        @info "Bounds for constant parameters are ignored." constants=collect(keys(constants))
    end
    if user_bounds
        lb = method.lb
        ub = method.ub
        lb isa ComponentArray && (lb = lb[free_names])
        ub isa ComponentArray && (ub = ub[free_names])
    else
        lb = lower_t_free_vec
        ub = upper_t_free_vec
    end
    use_bounds = use_bounds || user_bounds

    if parentmodule(typeof(method.optimizer)) === OptimizationBBO && !use_bounds
        error("BlackBoxOptim methods require finite bounds. Add lower/upper bounds " *
              "in @fixedEffects (on transformed scale) or pass them via " *
              "GHQuadrature(lb=..., ub=...). A quick helper is " *
              "default_bounds_from_start(dm; margin=...).")
    end
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO
        model_lb_v = lower_t_free_vec
        model_ub_v = upper_t_free_vec
        lb = map((u, m) -> isfinite(m) ? max(u, m) : u, collect(lb), model_lb_v)
        ub = map((u, m) -> isfinite(m) ? min(u, m) : u, collect(ub), model_ub_v)
        θ0_init = clamp.(collect(θ0_free_t), lb, ub)
    else
        θ0_init = θ0_free_t
    end

    prob = use_bounds ? OptimizationProblem(optf, θ0_init; lb = lb, ub = ub) :
           OptimizationProblem(optf, θ0_init)
    sol = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)

    # ── Extract solution ─────────────────────────────────────────────────────
    θ_hat_t_raw = sol.u
    θ_hat_t_free = θ_hat_t_raw isa ComponentArray ?
                   θ_hat_t_raw : ComponentArray(θ_hat_t_raw, axs_free)
    θ_hat_t = _ghq_merge_full(
        θ_const_t_vec, free_idx, ComponentArrays.getdata(θ_hat_t_free), axs_full)
    θ_hat_u = inv_transform(θ_hat_t)

    # ── Post-hoc EB mode finding (for get_random_effects) ────────────────────
    _laplace_get_bstar!(ebe_cache, dm, batch_infos, θ_hat_u, const_cache, ll_cache;
        optimizer = inner_opts.optimizer,
        optim_kwargs = inner_opts.kwargs,
        adtype = inner_opts.adtype,
        grad_tol = inner_opts.grad_tol,
        multistart = multistart_opts,
        rng = rng,
        serialization = serialization)

    # ── Build result ─────────────────────────────────────────────────────────
    summary = FitSummary(sol.objective,
        sol.retcode == SciMLBase.ReturnCode.Success,
        FitParameters(θ_hat_t, θ_hat_u),
        NamedTuple())
    diagnostics = FitDiagnostics((;), (optimizer = method.optimizer,),
        (retcode = sol.retcode,), NamedTuple())
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ?
            sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = GHQuadratureResult(sol, sol.objective, niter, raw, NamedTuple(),
        ebe_cache.bstar_cache.b_star)
    return FitResult(method, result, summary, diagnostics,
        store_data_model ? dm : nothing, args, fit_kwargs)
end

# ---------------------------------------------------------------------------
# Progressive refinement interceptor (level::Vector{Int})
# ---------------------------------------------------------------------------
# The struct is now defined, so this method can reference it safely.

function _fit_model(dm::DataModel, method::GHQuadrature, args...;
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        kwargs...)
    level = method.level
    level isa Vector{Int} || return _fit_model_scalar(dm, method, args...;
        theta_0_untransformed = theta_0_untransformed,
        kwargs...)
    isempty(level) && error("GHQuadrature: `level` vector must not be empty.")
    all(>(0), level) ||
        error("GHQuadrature: all entries in `level` must be positive integers.")

    θ0 = theta_0_untransformed
    local res
    for lv in level
        inner = GHQuadrature(lv,
            method.optimizer, method.optim_kwargs, method.adtype,
            method.inner, method.multistart, method.lb, method.ub,
            method.ignore_model_bounds)
        res = _fit_model_scalar(dm, inner, args...; theta_0_untransformed = θ0, kwargs...)
        θ0 = get_params(res; scale = :untransformed)
    end
    return res
end
