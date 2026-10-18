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
                 lb, ub, ignore_model_bounds, precondition) <: FittingMethod

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
- `level = :auto`: Smolyak accuracy level, error-controlled. The fit starts at 5 when some
  random effect has bounded support and every batch has at most 3 RE dimensions (the
  transport-space rule for bounded REs needs the higher level), else 3; afterwards the
  marginal log-likelihood at the estimate is recomputed with the next finer rule
  (`2L + 1` nodes) and, while the two differ by more than 0.5 (whole data set), the fit is
  refined from the estimate at the finer level (at most 4 refinements). Refinement applies
  to models whose random-effect batches are scalar (one random effect per batch). May also be:
  - `Int` (isotropic): same level for all RE groups.
  - `NamedTuple` (anisotropic): per-RE-group level, e.g.
    `level = (η_id = 3, η_site = 2)`.  RE groups not mentioned default to
    level 1.  The batch grid is the tensor product of per-group Smolyak grids.
  The Smolyak weights are signed and the weight at the mode grows with the
  number of random effects `d` (−(d−1) at level 2, e.g. −7 for d = 8). A batch
  whose integrand is far from Gaussian can therefore get a negative signed sum;
  that batch then falls back to the level-1 (Laplace) value with a warning.
- `optimizer`: outer Optimization.jl-compatible optimizer.  Defaults to LBFGS
  with backtracking line search. For mini-batching wrap the method in
  [`MiniBatch`](@ref).
- `optim_kwargs = nothing`: forwarded to `Optimization.solve`; `nothing` = `NamedTuple()`, plus `reltol = 1e-12` for an Optim.jl optimizer (as [`Laplace`](@ref))
  (e.g. `maxiters`, `reltol`).
- `adtype`: AD backend for the outer gradient.  Defaults to
  `AutoForwardDiff()`.
- `inner_options / inner_optimizer / inner_kwargs / inner_adtype / inner_grad_tol`:
  configure the Laplace-style inner optimizer for the empirical-Bayes modes. It runs
  on every objective evaluation (the modes center the adaptive rule for Gaussian
  random effects) and again at the optimum for `get_random_effects`.
- `multistart_options / multistart_n / multistart_k / multistart_grad_tol /
  multistart_max_rounds / multistart_sampling`: multistart settings for the
  EB mode finder.
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
    level === :auto && return :auto
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

# `level = :auto` (default): 5 when some random effect has bounded support (LogNormal, Beta,
# Gamma, ...) and every batch has at most 3 RE dimensions, else 3. The transport-space
# adaptive rule for bounded REs needs a higher level for the same accuracy (M05, two bounded
# REs: per-batch error ~3e-2 at level 3 vs 1e-3..1e-2 at level 5, against a 60x60 tensor
# rule; summed over 150 subjects that biased the estimates); Gaussian REs use the
# mode-centered rule, which is accurate at level 3. Resolved once per fit.
const _GHQ_UNBOUNDED_RE_TYPES = (:Normal, :MvNormal, :TDist, :NormalizingPlanarFlow)
_ghq_has_bounded_re(dm::DataModel) =
    any(t -> !(t in _GHQ_UNBOUNDED_RE_TYPES), values(get_re_types(get_random(get_model(dm)))))
# Per-batch resolution, for code paths that evaluate the objective outside `fit_model`
# (developer API, grid checks): same rule with the batch's own dimension.
_ghq_auto_level(dm::DataModel, info::REBatchInfo) =
    _ghq_has_bounded_re(dm) && get_n_b(info) <= 3 ? 5 : 3
function _ghq_resolve_auto_level(method::GHQuadrature, dm::DataModel, batch_infos)
    method.level === :auto || return method
    maxdim = maximum((get_n_b(info) for info in batch_infos); init = 0)
    lvl = _ghq_has_bounded_re(dm) && maxdim <= 3 ? 5 : 3
    return GHQuadrature(
        lvl, method.optimizer, method.optim_kwargs, method.adtype, method.inner,
        method.multistart, method.lb, method.ub, method.ignore_model_bounds,
        method.precondition, method.update_schedule
    )
end

function GHQuadrature(;
        level = :auto,  # :auto, Int, Vector{Int} or NamedTuple for anisotropic levels
        optimizer = nothing,
        optim_kwargs = nothing,
        adtype = Optimization.AutoForwardDiff(),
        inner_options = nothing,
        inner_optimizer = NewtonInner(x_abstol = 1.0e-10),
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
        precondition = true
    )
    optimizer = _resolve_outer_optimizer(optimizer)
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
    optim_kwargs === nothing &&
        (optim_kwargs = _outer_default_optim_kwargs(optimizer, NamedTuple()))
    return GHQuadrature(
        level, optimizer, _as_namedtuple(optim_kwargs), adtype, inner, ms, lb, ub,
        ignore_model_bounds, precondition, :all
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
        level,   # Int, NamedTuple or :auto
        b_star = nothing,
        H_star = nothing   # exact ∇²_b log f at b_star (Float64 θ), when known
    )
    level === :auto && (level = _ghq_auto_level(dm, info))
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
    re_measure = try
        m = build_re_measure_from_batch(info, θu_re, const_cache, dm, ll_cache)
        mc = _ghq_adaptive_measure(dm, info, θu_re, const_cache, ll_cache, m, b_star, H_star)
        mc === nothing ? m : mc
    catch e
        _is_numeric_error(e) && return T(-Inf)
        rethrow(e)
    end
    # A negative signed Smolyak sum falls back to the level-1 rule inside batch_loglik_ghq.
    ghq_ll = batch_loglik_ghq(dm, info, θu_re, re_measure, sgrid, const_cache, ll_cache)
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

When some RE in the batch has bounded support, b-space centering is invalid
(`CenteredREMeasure` places nodes anywhere in ℝ^n_b), so the rule is made adaptive in
the transport reference space instead ([`AdaptiveTransportRE`](@ref)): same integral,
nodes at the u-space mode. Returns `nothing` (keep the prior-centered measure) when the
mode/curvature is unavailable.
"""
function _ghq_adaptive_measure(
        dm::DataModel, info::REBatchInfo,
        θu_re::ComponentArray, const_cache::REConstantsCache, ll_cache::_LLCache,
        prior_measure::AbstractREMeasure, b_star, H_star = nothing
    )
    if prior_measure isa CompositeRE && !prior_measure.unbounded
        return build_adaptive_transport_measure(
            prior_measure, dm, info, _laplace_floatize(θu_re), const_cache, ll_cache;
            b_star = b_star
        )
    end
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
        θ_prior = θu_re, H_star = b_star === nothing ? nothing : H_star
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
    if level === :auto
        for info in batch_infos
            get_n_b(info) > 0 && get_sparse_grid(get_n_b(info), _ghq_auto_level(dm, info))
        end
        return nothing
    end
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
    level === :auto && (level = _ghq_auto_level(dm, info))
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
        extra_objective = extra_objective,
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
    method = _ghq_resolve_auto_level(method, dm, batch_infos)

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
            # Each task owns cache slot `c` — `Threads.threadid()` indexing is
            # unsafe under task migration (two tasks could share one cache slot).
            _greedy_foreach(length(batch_infos), length(ll_cache)) do bi, c
                active !== nothing && !(bi in active) && return
                if bad[]
                    results[bi] = zero(T)
                    return
                end
                bll = _ghq_batch_ll(
                    dm, batch_infos[bi], θu_re, const_cache,
                    ll_cache[c], method.level, bstars[bi],
                    _laplace_mode_hess(ebe_cache.hess_cache, bi, bstars[bi])
                )
                if bll == -Inf
                    Threads.atomic_or!(bad, true)
                    results[bi] = zero(T)
                else
                    # `convert`, not `T(bll)`: when `T` is a Dual and `bll` is the same
                    # Dual type, the constructor tries `Float64(::Dual)` and throws.
                    results[bi] = _batch_weight(dm, get_inds(batch_infos[bi])) * convert(T, bll)
                end
            end
            bad[] && return infT
        else
            for (bi, info) in enumerate(batch_infos)
                active !== nothing && !(bi in active) && continue
                bll = _ghq_batch_ll(
                    dm, info, θu_re, const_cache, ll_cache,
                    method.level, bstars[bi],
                    _laplace_mode_hess(ebe_cache.hess_cache, bi, bstars[bi])
                )
                bll == -Inf && return infT
                results[bi] = _batch_weight(dm, get_inds(info)) * convert(T, bll)
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
    # Start repair (as in Laplace/SAEM): a start with a non-finite objective (e.g. an RE
    # scale that is invalid at θ0) ends the fit at once; move it along the transformed
    # segment toward the declared initial values to the first feasible point.
    let f0 = obj(z0, nothing)
        if !isfinite(f0)
            θd = free_parameter_layout(fe; constants = constants).θ0_free_t
            for λ in (0.5 .^ (1:8)..., 0.0)
                zλ = _z_from_θt(λ .* θ0_init .+ (1 - λ) .* θd)
                if isfinite(obj(zλ, nothing))
                    @warn "GHQuadrature: the objective is not finite at the given start; moved the start toward the model's declared initial values (λ = $(λ))."
                    z0 = zλ
                    break
                end
            end
        end
    end
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
# `level = :auto`: error-controlled level
# ---------------------------------------------------------------------------
# Fit at the dimension-based base level (`_ghq_resolve_auto_level`), then compare the
# marginal log-likelihood at θ̂ under that rule and the next finer one; while they differ
# by more than _GHQ_AUTO_TOL nats (whole data set), refit from θ̂ at the finer level.
# 0.5 nats: on toenail/H01 the estimate-moving rule errors are several nats (level 3 vs 7:
# 10 / 5 nats), converged levels differ by <= 0.2; 0.05 refined models whose estimates
# were already exact (default-GHQ challenger 350 -> 942 s).
# A fixed level cannot serve every model: binary outcomes with a large RE variance
# (toenail data, RE sd ~4) need ~15 nodes per dimension — level 3 there moves the
# estimates by many standard errors — while most models are exact at level 3.
const _GHQ_AUTO_TOL = 0.5
const _GHQ_AUTO_MAX_REFINE = 4
const _GHQ_AUTO_MAX_NODES = 2_000
_ghq_next_level(level::Int, maxdim::Int) = maxdim <= 1 ? 2level + 1 : level + 2

function _ghq_fit_auto(
        dm::DataModel, method::GHQuadrature, args...;
        theta_0_untransformed = nothing, kwargs...
    )
    constants_re = _normalize_constants_re(dm, get(kwargs, :constants_re, NamedTuple()))
    _, infos, _ = _build_re_batch_infos(dm, constants_re)
    m = _ghq_resolve_auto_level(method, dm, infos)
    res = _fit_model_scalar(dm, m, args...; theta_0_untransformed = theta_0_untransformed, kwargs...)
    # Refinement only for scalar random effects (1-D batches): there the finer rules are
    # nearly free and the base level is most often too coarse (binary data with a large RE
    # variance). Multi-dimensional/bounded batches keep the base rule: on M05 (two bounded
    # REs) the level-5 error hovers around the tolerance, so starts refined differently
    # (objectives at different levels) and the fit cost rose 4x without better estimates.
    maxdim = maximum((get_n_b(info) for info in infos); init = 0)
    maxdim == 1 || return res
    ode = (ode_args = get(kwargs, :ode_args, ()), ode_kwargs = get(kwargs, :ode_kwargs, NamedTuple()))
    for _ in 1:_GHQ_AUTO_MAX_REFINE
        L, L2 = m.level, _ghq_next_level(m.level, maxdim)
        maximum(info -> get_n_b(info) == 0 ? 0 : _batch_grid_bound(dm, info, L2), infos) >
            _GHQ_AUTO_MAX_NODES && break
        θ̂ = get_params(res; scale = :untransformed)
        ll1 = ghq_marginal(dm, θ̂; level = L, constants_re = constants_re, ode...)
        ll2 = ghq_marginal(dm, θ̂; level = L2, constants_re = constants_re, ode...)
        (isfinite(ll1) && isfinite(ll2)) || break
        abs(ll2 - ll1) <= _GHQ_AUTO_TOL && break
        m = GHQuadrature(
            L2, method.optimizer, method.optim_kwargs, method.adtype, method.inner,
            method.multistart, method.lb, method.ub, method.ignore_model_bounds,
            method.precondition, method.update_schedule
        )
        res = _fit_model_scalar(dm, m, args...; theta_0_untransformed = θ̂, kwargs...)
    end
    return res
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
    level === :auto && return _ghq_fit_auto(
        dm, method, args...; theta_0_untransformed = theta_0_untransformed, kwargs...
    )
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
