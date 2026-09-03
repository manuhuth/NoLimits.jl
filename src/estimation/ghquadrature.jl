# ghquadrature.jl
# GHQuadrature <: FittingMethod: Smolyak sparse-grid quadrature for NLME.

export GHQuadrature
export GHQuadratureResult

using Optimization
using OptimizationOptimJL
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
                 lb, ub, ignore_model_bounds, precondition,
                 update_schedule) <: FittingMethod

Sparse-grid (Smolyak) quadrature for NLME marginal likelihood estimation.

Approximates the batch marginal likelihood via

    log L_batch ≈ signed_logsumexp_r [ log|W_r| + Σᵢ ℓᵢ(μ + Lzᵣ, θ) ]

where `{(zᵣ, Wᵣ)}` are Smolyak–Gauss-Hermite quadrature nodes/weights at the
requested `level`.  The rule is *adaptive* for Gaussian random effects: like
`Laplace` it solves for the empirical-Bayes mode of each batch, then centers and
whitens the nodes there.  Non-Gaussian random effects keep the prior-centered
transport-map rule, which needs a higher `level` for the same accuracy.
`level = 1` is a single-node rule whose value is the Laplace approximation but
whose gradient is not; use `Laplace()` for that, and `level ≥ 3` here.

# Keyword Arguments
- `level = 3`: Smolyak accuracy level.  May be:
  - `Int` (isotropic): same level for all RE groups.
  - `NamedTuple` (anisotropic): per-RE-group level, e.g.
    `level = (η_id = 3, η_site = 2)`.  RE groups not mentioned default to
    level 1.  The batch grid is the tensor product of per-group Smolyak grids.
  Levels 1–3 are numerically stable; higher levels may exhibit cancellation
  in signed logsumexp.
- `optimizer`: outer Optimization.jl-compatible optimizer.  Defaults to LBFGS
  with backtracking line search, or to `Optimisers.Adam(0.01)` when
  `update_schedule != :all`.
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
- `precondition::Bool = true`: optimize the scaled offset `z` with
  `θ_transformed = θ0 + s .* z`, so every fit starts at `z = 0` and no coordinate can be
  frozen by an unlucky starting value. `s` is 1 for any coordinate already in log/logit
  space and `max(abs(θ0), 1)` for a genuinely natural-scale `:identity` coordinate. Set
  `false` to optimize the transformed vector directly, which reproduces pre-0.2 results
  bit-for-bit. Note that with preconditioning on, the optimizer object behind
  [`get_raw`](@ref) works in `z`; [`get_params`](@ref) always returns the usual scales.
$(_UPDATE_SCHEDULE_DOC)
"""
struct GHQuadrature{LV, O, K, A, IO, MS, L, U, US} <: FittingMethod
    level::LV   # Int (isotropic) or NamedTuple (anisotropic per-RE-group)
    optimizer::O
    optim_kwargs::K
    adtype::A
    inner::IO
    multistart::MS
    lb::L
    ub::U
    ignore_model_bounds::Bool
    precondition::Bool
    update_schedule::US
end

# `level` is an Int, a Vector{Int} (level continuation), or a NamedTuple of per-RE levels.
# Returns the normalized level: any `Integer` (Int8, UInt, ...) becomes `Int`, so the
# `level isa Int` dispatch below cannot mistake an isotropic level for an anisotropic one.
function _check_ghq_level(level)
    bad(l) = !(l isa Integer) || l < 1
    if level isa NamedTuple
        isempty(level) && throw(ArgumentError("GHQuadrature: anisotropic `level` must not be empty."))
        for (k, l) in Base.pairs(level)
            bad(l) &&
                throw(ArgumentError("GHQuadrature: level for `$k` must be a positive integer. Got: $(repr(l))"))
        end
        return NamedTuple{keys(level)}(map(Int, Tuple(level)))
    elseif level isa AbstractVector
        isempty(level) && throw(ArgumentError("GHQuadrature: `level` vector must not be empty."))
        all(!bad, level) ||
            throw(ArgumentError("GHQuadrature: all entries in `level` must be positive integers. Got: $level"))
        return Int[Int(l) for l in level]
    end
    bad(level) &&
        throw(ArgumentError("GHQuadrature: `level` must be a positive integer. Got: $(repr(level))"))
    return Int(level)
end

_update_schedule(m::GHQuadrature) = m.update_schedule

function GHQuadrature(;
        level = 3,  # Int or NamedTuple for anisotropic levels
        optimizer = nothing,
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
        ignore_model_bounds = false,
        precondition = true,
        update_schedule = :all
    )
    update_schedule = _check_update_schedule(_as_symbol(update_schedule), "GHQuadrature")
    optimizer = _resolve_outer_optimizer(optimizer, update_schedule, "GHQuadrature")
    level = _check_ghq_level(_as_namedtuple(level))
    inner = inner_options === nothing ?
        LaplaceInnerOptions(
            inner_optimizer, _as_namedtuple(inner_kwargs), inner_adtype, inner_grad_tol
        ) :
        inner_options
    ms = multistart_options === nothing ?
        LaplaceMultistartOptions(
            multistart_n, multistart_k, multistart_grad_tol,
            multistart_max_rounds, _as_symbol(multistart_sampling)
        ) :
        multistart_options
    return GHQuadrature(
        level, optimizer, _as_namedtuple(optim_kwargs), adtype, inner, ms, lb, ub,
        ignore_model_bounds, precondition, update_schedule
    )
end

# ---------------------------------------------------------------------------
# Result struct
# ---------------------------------------------------------------------------

# GHQuadratureResult is a StandardOptimizationResult{:ghquadrature} alias (see common.jl).

# ---------------------------------------------------------------------------
# Internal: evaluate sparse-grid marginal log-likelihood for one batch
# ---------------------------------------------------------------------------

"""
    _ghq_batch_ll(dm, info, θu_re, const_cache, ll_cache, level, b_star = nothing) -> T

Evaluate the batch marginal log-likelihood using the sparse grid at `level`.

- `level::Int`: isotropic — same Smolyak level for all RE dimensions.
- `level::NamedTuple`: anisotropic — maps RE name to level; RE groups not
  mentioned default to level 1.  The batch grid is the tensor product of
  per-RE-group Smolyak grids.
- `b_star`: precomputed empirical-Bayes mode for this batch (adaptive
  quadrature).  `nothing` solves for it here; pass a warm-started mode when the
  caller already runs an EBE pass over all batches.

For batches with `n_b == 0` (all RE are constant), returns the sum of
individual conditional log-likelihoods directly.
"""
function _ghq_batch_ll(
        dm::DataModel,
        info::REBatchInfo,
        θu_re::ComponentArray,
        const_cache::REConstantsCache,
        ll_cache::_LLCache,
        level,   # Int or NamedTuple
        b_star = nothing
    )
    T = eltype(θu_re)
    if get_n_b(info) == 0
        # All RE are constant — no integration needed.
        total = zero(T)
        empty_b = T[]
        for i in get_inds(info)
            η_i = _build_eta_ind(dm, i, info, empty_b, const_cache, θu_re)
            lli = _loglikelihood_individual(dm, i, θu_re, η_i, ll_cache)
            !isfinite(lli) && return T(-Inf)
            total += convert(T, lli)
        end
        const_ll = _const_re_prior_logf(dm, info, θu_re, const_cache, ll_cache)
        !isfinite(const_ll) && return T(-Inf)
        return total + T(const_ll)
    end

    # Select grid: isotropic (Int) or anisotropic (NamedTuple)
    sgrid = if level isa Int
        get_sparse_grid(get_n_b(info), level)
    else
        _build_anisotropic_batch_grid(dm, info, level)
    end

    # build_re_measure_from_batch may throw when distribution parameters hit numerical
    # limits (e.g. Beta with α→0 due to underflow, or a non-PD covariance). Treat these
    # as invalid parameter regions and return -Inf, as the Laplace path does.
    adaptive = true
    re_measure = try
        m = build_re_measure_from_batch(info, θu_re, const_cache, dm, ll_cache)
        mc = _ghq_adaptive_measure(dm, info, θu_re, const_cache, ll_cache, m, b_star)
        adaptive = mc !== nothing
        adaptive ? mc : m
    catch e
        _is_numeric_error(e) && return T(-Inf)
        rethrow(e)
    end
    ghq_ll = batch_loglik_ghq(dm, info, θu_re, re_measure, sgrid, const_cache, ll_cache)
    # Adaptive centering is skipped for batches that are not purely Gaussian, and the
    # prior-centered rule's signed Smolyak weights can turn the batch marginal negative at
    # level >= 2 (issue #98), which used to surface as an `Inf` objective for the whole fit
    # (issue #176). The level-1 rule has only positive weights, so fall back to it rather
    # than declaring the batch impossible.
    if !isfinite(ghq_ll) && !adaptive && get_n_b(info) > 0
        grid1 = get_sparse_grid(get_n_b(info), 1)
        if grid1 !== sgrid
            @warn "GHQuadrature: the prior-centered rule was numerically unstable for a " *
                "non-Gaussian random-effect batch; falling back to the level-1 rule for " *
                "this batch. Pass a lower `level`, or use SAEM/MCEM for a more accurate " *
                "marginal." maxlog = 1
            ghq_ll = batch_loglik_ghq(
                dm, info, θu_re, re_measure, grid1, const_cache, ll_cache
            )
        end
    end
    const_ll = _const_re_prior_logf(dm, info, θu_re, const_cache, ll_cache)
    (!isfinite(ghq_ll) || !isfinite(const_ll)) && return T(-Inf)
    return ghq_ll + T(const_ll)
end

# Empirical-Bayes mode of one batch, standalone (own single-slot EBE cache, cold start).
function _ghq_bstar_batch(
        dm::DataModel, info::REBatchInfo, θ_val::ComponentArray,
        const_cache::REConstantsCache, ll_cache::_LLCache
    )
    cache = _init_laplace_eval_cache(1, Float64)
    _laplace_compute_bstar_batch!(cache, 1, dm, info, θ_val, const_cache, ll_cache)
    b = cache.bstar_cache.b_star[1]
    return isempty(b) ? nothing : b
end

"""
    _ghq_adaptive_measure(dm, info, θu_re, const_cache, ll_cache, prior_measure, b_star)

Upgrade a prior-centered batch measure to the adaptive (AGHQ) one: nodes centered
at the empirical-Bayes mode and whitened by the posterior curvature there.

This is what makes the rule usable. Centered on the prior, the integrand is a
likelihood bump far narrower than the prior, and the signed Smolyak weights make
the estimate oscillate and drift *away* from the true integral as the level rises,
regularly turning the batch marginal negative (issue #98). Centered on the mode it
converges at level 1-3.

Returns `nothing` (keep the prior-centered measure) when some RE in the batch has
bounded support — `CenteredREMeasure` places nodes anywhere in ℝ^n_b, which only
stays inside the random-effect support when every RE in the batch is supported on
all of ℝ — or when the mode/curvature is unavailable.
"""
function _ghq_adaptive_measure(
        dm::DataModel, info::REBatchInfo,
        θu_re::ComponentArray, const_cache::REConstantsCache, ll_cache::_LLCache,
        prior_measure::AbstractREMeasure, b_star
    )
    prior_measure isa GaussianRE ||
        (prior_measure isa CompositeRE && prior_measure.unbounded) || return nothing
    # b*, H and S are frozen w.r.t. the outer AD: at a converged rule the value no
    # longer depends on where the nodes sit, so the dropped ∂/∂b*, ∂/∂S terms are of
    # the order of the quadrature error. `θ_prior` keeps the RE-prior term of the
    # correction differentiable in θ.
    θ_val = _laplace_floatize(θu_re)
    b = b_star === nothing ?
        _ghq_bstar_batch(dm, info, θ_val, const_cache, ll_cache) : b_star
    (b === nothing || length(b) != get_n_b(info)) && return nothing
    return build_centered_re_measure(
        b, info, 1, θ_val, const_cache, dm, ll_cache;
        θ_prior = θu_re
    )
end

"""
    _build_anisotropic_batch_grid(dm, info, level::NamedTuple) -> GHQuadratureNodes

Build (or retrieve from cache) the tensor-product anisotropic grid for this
batch.  `level` is a NamedTuple mapping RE name → Int level.  RE groups not
present in `level` default to level 1.

Returns the concatenated tensor-product grid over all RE groups that have
free levels (non-zero dimension) in this batch.
"""
function _anisotropic_dims_levels(dm::DataModel, info::REBatchInfo, level::NamedTuple)
    re_names = get_re_names(get_laplace_cache(get_re_group_info(dm)))
    unknown = setdiff(collect(keys(level)), re_names)
    isempty(unknown) ||
        error("GHQuadrature: unknown random effect(s) in `level`: $(join(unknown, ", ")). Declared random effects: $(join(re_names, ", ")).")
    dims = Int[]
    levels = Int[]
    for (ri, re_name) in enumerate(re_names)
        re_info = get_re_info(info)[ri]
        # Total free RE dimension for this RE group in this batch
        total_dim = sum(length(r) for r in get_ranges(re_info); init = 0)
        total_dim == 0 && continue
        l = haskey(level, re_name) ? getproperty(level, re_name) : 1
        push!(dims, total_dim)
        push!(levels, l)
    end
    return dims, levels
end

function _build_anisotropic_batch_grid(
        dm::DataModel, info::REBatchInfo, level::NamedTuple
    )
    dims, levels = _anisotropic_dims_levels(dm, info, level)
    isempty(dims) && error("_build_anisotropic_batch_grid: no free RE dimensions found")
    return get_anisotropic_grid(dims, levels)
end

# ---------------------------------------------------------------------------
# Cache pre-population helpers
# ---------------------------------------------------------------------------

# Pre-build all grids needed for this fit so concurrent use is thread-safe.
function _prepopulate_ghq_cache(dm::DataModel, batch_infos, level)
    return if level isa Int
        for d in unique(get_n_b(info) for info in batch_infos)
            d > 0 && get_sparse_grid(d, level)
        end
    else
        # Anisotropic: build the per-batch tensor-product grids
        for info in batch_infos
            get_n_b(info) > 0 && _build_anisotropic_batch_grid(dm, info, level)
        end
    end
end

# Node-count upper bound for a batch grid, without building it.
function _batch_grid_bound(dm::DataModel, info::REBatchInfo, level)
    level isa Int && return ghq_points_bound(get_n_b(info), level)
    dims, levels = _anisotropic_dims_levels(dm, info, level)
    return prod(ghq_points_bound(d, l) for (d, l) in zip(dims, levels); init = 1.0)
end

# Hard ceiling: above this the grid cannot be built (it is one likelihood
# evaluation per node, and the grid itself runs into tens of GB).
const GHQ_MAX_NODES = 1_000_000

# Warn above `threshold` nodes, refuse above `GHQ_MAX_NODES`. Must run BEFORE any
# grid is built — building the grid is what exhausts memory on oversized batches.
function _check_batch_grid_sizes(
        dm::DataModel, batch_infos, level, threshold::Int,
        ctx::AbstractString
    )
    for (bi, info) in enumerate(batch_infos)
        get_n_b(info) == 0 && continue
        npts = _batch_grid_bound(dm, info, level)
        if npts > GHQ_MAX_NODES
            error(
                "$ctx: RE batch $bi has joint random-effect dimension " *
                    "$(get_n_b(info)) and needs ~$(round(npts, sigdigits = 3)) " *
                    "quadrature nodes at level $level (limit $GHQ_MAX_NODES). " *
                    "Reduce `level`, split the batch, or use Laplace/FOCEI/SAEM."
            )
        end
        npts > threshold &&
            @warn "$ctx: RE batch $bi needs ~$(round(Int, npts)) quadrature nodes. " *
            "Consider reducing `level` or checking your RE batch structure."
    end
    return nothing
end

# ---------------------------------------------------------------------------
# _fit_model dispatch
# ---------------------------------------------------------------------------

function _fit_model_scalar(
        dm::DataModel, method::GHQuadrature, args...;
        constants::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        penalty::NamedTuple = NamedTuple(),
        extra_objective = nothing,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Random.default_rng(),
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_data_model::Bool = true
    )
    fit_kwargs = (
        constants = constants,
        constants_re = constants_re,
        penalty = penalty,
        ode_args = ode_args,
        ode_kwargs = ode_kwargs,
        serialization = serialization,
        rng = rng,
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model,
    )

    # ── Validate ────────────────────────────────────────────────────────────
    re_names = get_re_names(get_random(get_model(dm)))
    isempty(re_names) &&
        error("GHQuadrature requires random effects. Use MLE/MAP for fixed-effects-only models.")

    _ghq_validate_re_distributions(dm)

    fe = get_fixed(get_model(dm))
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("GHQuadrature requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    _validate_constant_names(fixed_set, constants)
    all(name in keys(constants) for name in fixed_names) &&
        error("GHQuadrature requires at least one free fixed effect.")

    layout = free_parameter_layout(
        fe; constants = constants,
        theta0_untransformed = theta_0_untransformed
    )
    free_names = layout.free_names
    inv_transform = layout.inv_transform

    inner_opts = _resolve_inner_options(method.inner, dm)
    multistart_opts = _resolve_multistart_options(method.multistart, inner_opts)

    # ── Infrastructure ───────────────────────────────────────────────────────
    pairing, batch_infos, const_cache = _build_re_batch_infos(dm, constants_re)

    ll_cache = build_ll_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization, force_saveat = true
    )

    # Pre-populate sparse-grid cache for all unique free-RE dimensions.
    _check_batch_grid_sizes(dm, batch_infos, method.level, 10_000, "GHQuadrature")
    _prepopulate_ghq_cache(dm, batch_infos, method.level)

    # EB-mode cache (used post-hoc for get_random_effects).
    n_batches = length(batch_infos)
    Tθ = eltype(layout.θ0_free_t)
    ebe_cache = _init_laplace_eval_cache(n_batches, Tθ)
    mb = _minibatch_state(_update_schedule(method), n_batches, rng)
    # The ref lets the post-solve full-data evaluation switch mini-batching off.
    mb_ref = Ref{Union{Nothing, typeof(mb)}}(mb)

    # ── Objective ────────────────────────────────────────────────────────────
    θ0_free_t = layout.θ0_free_t
    axs_free = layout.axs
    axs_full = layout.axs_full
    free_idx = layout.free_idx
    θ_const_t_vec = layout.θ_const_t_vec

    # The optimizer works on the preconditioned offset z; everything below stays on θt.
    θ0_pc, s_pc, _θt_from_z, _z_from_θt = _precondition_maps(
        get_model(dm), free_names, θ0_free_t, axs_free, _precondition_on(method)
    )

    function obj(z, p)
        st = _minibatch_current!(mb_ref[])
        active = _minibatch_active(st)
        θt_free = _θt_from_z(z)
        T = eltype(θt_free)
        infT = convert(T, Inf)

        θt_full = _merge_free_into_full(
            θ_const_t_vec, free_idx, ComponentArrays.getdata(θt_free), axs_full
        )
        θu = inv_transform(θt_full)
        θu_re = _symmetrize_psd_params(θu, get_fixed(get_model(dm)))

        # One warm-started EBE pass per θ feeds the adaptive quadrature centers; the
        # per-batch fallback inside `_ghq_batch_ll` would cold-start every solve.
        bstars = _laplace_get_bstar!(
            ebe_cache, dm, batch_infos,
            _laplace_floatize(θu_re), const_cache, ll_cache;
            optimizer = inner_opts.optimizer,
            optim_kwargs = inner_opts.kwargs,
            adtype = inner_opts.adtype,
            grad_tol = inner_opts.grad_tol,
            multistart = multistart_opts,
            rng = rng,
            serialization = serialization,
            active_batches = active
        )

        # Both branches fill the same per-batch vector and reduce it with the same
        # `sum`, so the objective is bit-identical under `EnsembleSerial` and
        # `EnsembleThreads` instead of differing by the accumulation order (#151).
        results = Vector{T}(undef, length(batch_infos))
        if ll_cache isa AbstractVector
            bad = Threads.Atomic{Bool}(false)
            # Chunk-indexed cache assignment — `Threads.threadid()` indexing is
            # unsafe under task migration (two tasks could share one cache slot).
            n_chunks = length(ll_cache)
            Threads.@threads for c in 1:n_chunks
                cache_c = ll_cache[c]
                for bi in c:n_chunks:length(batch_infos)
                    active !== nothing && !(bi in active) && continue
                    if bad[]
                        results[bi] = zero(T)
                        continue
                    end
                    bll = _ghq_batch_ll(
                        dm, batch_infos[bi], θu_re, const_cache,
                        cache_c, method.level, bstars[bi]
                    )
                    if bll == -Inf
                        Threads.atomic_or!(bad, true)
                        results[bi] = zero(T)
                    else
                        # `convert`, not `T(bll)`: when `T` is a Dual and `bll` is the same
                        # Dual type, the constructor tries `Float64(::Dual)` and throws.
                        results[bi] = convert(T, bll)
                    end
                end
            end
            bad[] && return infT
        else
            for (bi, info) in enumerate(batch_infos)
                active !== nothing && !(bi in active) && continue
                bll = _ghq_batch_ll(
                    dm, info, θu_re, const_cache, ll_cache,
                    method.level, bstars[bi]
                )
                bll == -Inf && return infT
                results[bi] = convert(T, bll)
            end
        end
        total = st === nothing ? sum(results) : st.scale * sum(results[bi] for bi in st.selected)
        return -total + convert(T, _penalty_value(θu, penalty)) +
            (extra_objective === nothing ? zero(T) : convert(T, extra_objective(θu)))
    end

    # ── Bounds ───────────────────────────────────────────────────────────────
    optf = OptimizationFunction(obj, method.adtype)
    lb, ub, use_bounds, θ0_init = _resolve_optim_bounds(
        fe, free_names, θ0_free_t, method.optimizer, method.lb, method.ub, constants;
        ignore_model_bounds = method.ignore_model_bounds, method_label = "GHQuadrature"
    )

    z0 = _z_from_θt(θ0_init)
    lb_z = _z_from_θt(lb)
    ub_z = _z_from_θt(ub)
    opt_use, use_bounds = _bounded_optimizer(method.optimizer, use_bounds, lb_z, ub_z)
    prob = use_bounds ? OptimizationProblem(optf, z0; lb = lb_z, ub = ub_z) :
        OptimizationProblem(optf, z0)
    invalidate! = _LaplaceMinibatchInvalidate(nothing, ebe_cache)
    solve_kwargs = _minibatch_solve_kwargs(method.optim_kwargs, opt_use, mb, invalidate!)
    sol = Optimization.solve(prob, opt_use; solve_kwargs...)

    # Mini-batched objectives are per-minibatch; re-evaluate on all batches at the fitted θ.
    final_obj = sol.objective
    if mb !== nothing
        mb_ref[] = nothing
        invalidate!()
        final_obj = obj(sol.u, nothing)
    end

    # ── Extract solution ─────────────────────────────────────────────────────
    # Mapped back here so both consumers below (the post-hoc EB modes and `FitParameters`) see θt.
    θ_hat_t_raw = _θt_from_z(sol.u)
    θ_hat_t_free = θ_hat_t_raw isa ComponentArray ?
        θ_hat_t_raw : ComponentArray(θ_hat_t_raw, axs_free)
    θ_hat_t = _merge_free_into_full(
        θ_const_t_vec, free_idx, ComponentArrays.getdata(θ_hat_t_free), axs_full
    )
    θ_hat_u = inv_transform(θ_hat_t)

    # ── Post-hoc EB mode finding (for get_random_effects) ────────────────────
    # Cold-solve the modes at θ̂: a warm start returns the stale residual of whichever θ
    # the optimizer tried last (same warm-start issue as the Laplace family).
    invalidate!()
    fill!(ebe_cache.bstar_cache.has_bstar, false)
    _laplace_get_bstar!(
        ebe_cache, dm, batch_infos, θ_hat_u, const_cache, ll_cache;
        optimizer = inner_opts.optimizer,
        optim_kwargs = inner_opts.kwargs,
        adtype = inner_opts.adtype,
        grad_tol = inner_opts.grad_tol,
        multistart = multistart_opts,
        rng = rng,
        serialization = serialization
    )

    # Post-hoc EBE diagnostic (#311), same check as the Laplace family.
    _warn_ebe_grad_tol(
        ebe_cache, dm, batch_infos, θ_hat_u, const_cache, ll_cache,
        inner_opts.grad_tol; label = string(nameof(typeof(method)))
    )

    # ── Build result ─────────────────────────────────────────────────────────
    isfinite(final_obj) ||
        _warn_nonfinite_fit(dm, θ_hat_u, string(nameof(typeof(method))))
    summary = FitSummary(
        final_obj,
        sol.retcode == SciMLBase.ReturnCode.Success,
        FitParameters(θ_hat_t, θ_hat_u),
        NamedTuple()
    )
    diagnostics = FitDiagnostics(
        (;), (optimizer = method.optimizer,),
        (retcode = sol.retcode,), NamedTuple()
    )
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ?
        sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = GHQuadratureResult(
        sol, final_obj, niter, raw, NamedTuple(),
        ebe_cache.bstar_cache.b_star
    )
    return FitResult(
        method, result, summary, diagnostics,
        store_data_model ? dm : nothing, args, fit_kwargs
    )
end

# ---------------------------------------------------------------------------
# Progressive refinement interceptor (level::Vector{Int})
# ---------------------------------------------------------------------------
# The struct is now defined, so this method can reference it safely.

function _fit_model(
        dm::DataModel, method::GHQuadrature, args...;
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        kwargs...
    )
    level = method.level
    level isa Vector{Int} || return _fit_model_scalar(
        dm, method, args...;
        theta_0_untransformed = theta_0_untransformed,
        kwargs...
    )
    isempty(level) && error("GHQuadrature: `level` vector must not be empty.")
    all(>(0), level) ||
        error("GHQuadrature: all entries in `level` must be positive integers.")

    θ0 = theta_0_untransformed
    local res
    for lv in level
        inner = GHQuadrature(
            lv,
            method.optimizer, method.optim_kwargs, method.adtype,
            method.inner, method.multistart, method.lb, method.ub,
            method.ignore_model_bounds, method.precondition, method.update_schedule
        )
        res = _fit_model_scalar(dm, inner, args...; theta_0_untransformed = θ0, kwargs...)
        θ0 = get_params(res; scale = :untransformed)
    end
    return res
end
