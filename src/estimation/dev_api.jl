# Public method-developer API: composable primitives for building new NLME
# estimators. Every function here is a thin cover over an internal kernel and
# obeys the two boundary contracts: θ and η are natural-scale (PSD blocks are
# symmetrized once here), and the batch RE argument `b` is a flat natural-scale
# vector of length `get_batch_re_dim(batch)`. The private per-batch/per-individual
# kernels keep the pre-symmetrized `θ_re` contract, so the hot fitting paths never
# route through these covers.
#
# Stability: the names exported below are the semver-stable method-developer API.
# Underscore-prefixed symbols (and the `_oldname = newname` migration aliases) are
# internal and may change at any release.

import Bijectors: logabsdetjac

# ── Public exports (method-developer API) ─────────────────────────────────────
# Scaffolding / contracts / transforms
export symmetrize_psd_parameters, apply_constants!, penalty_value, validate_constant_names,
    resolve_optimizer_bounds, free_parameter_indices, merge_free_parameters, logabsdetjac
# RE-batching currency
export build_re_batch_infos, REBatchInfo, REConstantsCache, RELevelInfo,
    get_batch_individuals, get_batch_re_info, get_batch_re_dim, get_re_levels,
    get_re_ranges,
    get_re_reps, get_re_dim, get_re_is_scalar, build_eta_individual, random_effect_value,
    eta_from_modes, LikelihoodCache, build_likelihood_cache, BatchThetaContext,
    build_batch_theta_context
# Evaluation primitives
export solve_individual, obs_distributions, hmm_filter_step!, conditional_loglikelihood,
    complete_data_loglikelihood, re_logprior, complete_data_loglikelihood_gradient,
    complete_data_loglikelihood_hessian
# Posterior / empirical Bayes / marginal / sampling
export empirical_bayes, empirical_bayes_covariance, laplace_marginal,
    laplace_marginal_gradient, ghq_marginal,
    sample_random_effect_draws,
    RandomEffectPosteriorSample, get_draws, get_log_weights, get_ess, EBEOptions
# Fisher-information registry
export expected_information, outcome_parameters, dispersion_indices,
    has_expected_information
# Quadrature nodes
export GHQuadratureNodes, build_sparse_grid, get_sparse_grid, build_tensor_product_grid,
    get_anisotropic_grid, n_ghq_points, get_nodes, get_logweights, get_signs,
    get_dimension,
    get_level
# Curvature seam
export AbstractCurvature, ExactHessianCurvature, FisherInformationCurvature,
    inner_curvature,
    CurvatureWorkspace
# Fitting-method protocol drivers
export fit_method, fit_fixed_effects, fit_laplace_family, objective_and_gradient
# MCEM M-step Q primitives + state-threaded E-step
export mcem_q_partition, mcem_q_objective_and_gradient, mcem_e_step
# SAEM closed-form M-step primitives (federation)
export saem_closed_form_eligibility, saem_sufficient_statistics, saem_closed_form_mstep

# Resolve a single-thread evaluation cache for the per-item primitives.
@inline _dev_ll_cache(::DataModel, cache::LikelihoodCache) = cache
@inline _dev_ll_cache(::DataModel, cache::AbstractVector) = first(cache)
_dev_ll_cache(dm::DataModel, ::Nothing) = build_likelihood_cache(dm; force_saveat = true)

"""
    solve_individual(dm, idx, θ, η; cache=nothing, dense=false) -> Union{Nothing, NamedTuple}

Solve individual `idx`'s differential equation at natural-scale `(θ, η)` and return the
solution accessors (state/signal getters callable at a time). Returns `NamedTuple()` for
algebraic (non-DE) models and `nothing` when the solve fails. `dense=true` returns a dense
(interpolating) solution; the default reuses the fit `saveat` grid.
"""
function solve_individual(
        dm::DataModel, idx::Integer, θ::ComponentArray, η;
        cache = nothing, dense::Bool = false
    )
    model = get_model(dm)
    get_de(model) === nothing && return NamedTuple()
    θ_re = symmetrize_psd_parameters(θ, get_fixed(model))
    η_ca = η isa NamedTuple ? ComponentArray(η) : η
    dense && return _simulate_sol_accessors(dm, Int(idx), θ_re, η_ca)
    c = _dev_ll_cache(dm, cache)
    const_cov = get_const_cov(get_individuals(dm)[Int(idx)])
    pre = calculate_prede(model, θ_re, η_ca, const_cov)
    return _ll_solve_de(dm, Int(idx), θ_re, η_ca, c, pre)
end

"""
    conditional_loglikelihood(dm, θ, η; kwargs...) -> Real                 # population
    conditional_loglikelihood(dm, idx::Integer, θ, η; cache=nothing)       # one individual
    conditional_loglikelihood(dm, batch::REBatchInfo, θ, b; const_cache, cache=nothing)  # one batch

Observation log-likelihood `log p(y | θ, η)` (no random-effect prior). The population form
sums over individuals (this is `loglikelihood`); the batch form sums over the batch's
individuals with η built from the flat vector `b`. θ is natural-scale and symmetrized here.
Returns `-Inf` on solve failure or non-finite density.
"""
conditional_loglikelihood(dm::DataModel, θ::ComponentArray, η; kwargs...) = loglikelihood(
    dm, θ, η; kwargs...
)

function conditional_loglikelihood(
        dm::DataModel, idx::Integer, θ::ComponentArray, η;
        cache = nothing
    )
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    η_ca = η isa NamedTuple ? ComponentArray(η) : η
    return _loglikelihood_individual(dm, Int(idx), θ_re, η_ca, _dev_ll_cache(dm, cache))
end

function conditional_loglikelihood(
        dm::DataModel, batch::REBatchInfo, θ::ComponentArray, b;
        const_cache::REConstantsCache, cache = nothing
    )
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    c = _dev_ll_cache(dm, cache)
    T = promote_type(eltype(θ), eltype(b))
    ll = zero(T)
    for i in get_batch_individuals(batch)
        η_ind = build_eta_individual(dm, i, batch, b, const_cache, θ_re)
        lli = _loglikelihood_individual(dm, i, θ_re, η_ind, c)
        isfinite(lli) || return convert(T, -Inf)::T
        ll += lli
    end
    return convert(T, ll)::T
end

"""
    re_logprior(dm, idx::Integer, θ, η; cache=nothing)                                  # one individual
    re_logprior(dm, batch::REBatchInfo, θ, b; const_cache, cache=nothing, anneal_sds=NamedTuple())  # one batch

Random-effect prior log-density `log p(η | θ)` summed over the (free and constant) grouping
levels, deduplicated per level. No ODE. θ is natural-scale and symmetrized here.
`complete_data_loglikelihood == conditional_loglikelihood + re_logprior` at batch scale.
"""
function re_logprior(
        dm::DataModel, batch::REBatchInfo, θ::ComponentArray, b;
        const_cache::REConstantsCache, cache = nothing,
        anneal_sds::Union{NamedTuple, AbstractDict} = NamedTuple()
    )
    anneal_sds = _as_namedtuple(anneal_sds)
    return _re_logpdf_batch(
        dm, batch, θ, b, const_cache, _dev_ll_cache(dm, cache); anneal_sds = anneal_sds
    )
end

function re_logprior(dm::DataModel, idx::Integer, θ::ComponentArray, η; cache = nothing)
    model = get_model(dm)
    θ_re = symmetrize_psd_parameters(θ, get_fixed(model))
    η_ca = η isa NamedTuple ? ComponentArray(η) : η
    const_cov = get_const_cov(get_individuals(dm)[Int(idx)])
    dists = build_re_dists(model, θ_re, const_cov)
    return get_re_logpdf(get_random(model))(dists, η_ca)
end

"""
    complete_data_loglikelihood(dm, idx::Integer, θ, η; cache=nothing)            # one individual
    complete_data_loglikelihood(dm, batch::REBatchInfo, θ, b; const_cache, cache=nothing, anneal_sds=NamedTuple(), tctx=nothing)  # one batch

Complete-data log-joint `log p(y, η | θ) = log p(y | θ, η) + log p(η | θ)` at one individual or
one batch. The batch form is canonical (the object an empirical-Bayes solver maximizes); the
per-individual form double-counts a shared grouping level's prior in crossed designs, so prefer
the batch form for fitting. θ is symmetrized here. The population form (summing over all
individuals, with η supplied or resolved from a fit) is documented above.
"""
function complete_data_loglikelihood(
        dm::DataModel, batch::REBatchInfo, θ::ComponentArray, b;
        const_cache::REConstantsCache, cache = nothing,
        anneal_sds::Union{NamedTuple, AbstractDict} = NamedTuple(), tctx = nothing
    )
    anneal_sds = _as_namedtuple(anneal_sds)
    return _laplace_logf_batch(
        dm, batch, θ, b, const_cache, _dev_ll_cache(dm, cache);
        anneal_sds = anneal_sds, tctx = tctx
    )
end

function complete_data_loglikelihood(
        dm::DataModel, idx::Integer, θ::ComponentArray, η; cache = nothing
    )
    c = _dev_ll_cache(dm, cache)
    return conditional_loglikelihood(dm, idx, θ, η; cache = c) +
        re_logprior(dm, idx, θ, η; cache = c)
end

"""
    complete_data_loglikelihood_gradient(dm, batch::REBatchInfo, θ, b; const_cache, cache=nothing) -> Vector

`∇_b log p(y, η | θ)` at the natural-scale batch RE vector `b`, via ForwardDiff. For a single
independent subject, build a singleton batch with `build_re_batch_infos`.
"""
function complete_data_loglikelihood_gradient(
        dm::DataModel, batch::REBatchInfo, θ::ComponentArray, b;
        const_cache::REConstantsCache, cache = nothing
    )
    f = _LaplaceLogfBatch(dm, batch, θ, const_cache, _dev_ll_cache(dm, cache))
    return ForwardDiff.gradient(f, b)
end

"""
    complete_data_loglikelihood_hessian(dm, batch::REBatchInfo, θ, b; const_cache, cache=nothing, curvature=ExactHessianCurvature()) -> Matrix

Hessian `H = ∇²_b log p(y, η | θ)` (negative-definite near a mode; the posterior precision is
`-H`). The caller owns `-H`/Cholesky/logdet. `curvature` selects the approximation:
`ExactHessianCurvature()` (default, full second-order AD) or `FisherInformationCurvature(interaction)`
(FOCEI/FOCE Gauss-Newton). Implement `inner_curvature(::YourCurvature, …)` to add your own.
"""
function complete_data_loglikelihood_hessian(
        dm::DataModel, batch::REBatchInfo, θ::ComponentArray, b;
        const_cache::REConstantsCache, cache = nothing,
        curvature::AbstractCurvature = ExactHessianCurvature()
    )
    return inner_curvature(
        curvature, dm, batch, θ, b, const_cache,
        _dev_ll_cache(dm, cache), CurvatureWorkspace()
    )
end

"""
    hmm_filter_step!(hmm_priors::Dict{Symbol,Any}, outcome::Symbol, dist, y) -> Distribution

One HMM forward-filter step: condition `dist` on the running `hmm_priors[outcome]`, then
update that entry with the posterior after observing `y` (or the predicted hidden-state
distribution when `y === missing`). Non-HMM distributions pass through unchanged and leave
`hmm_priors` untouched.
"""
hmm_filter_step!(hmm_priors::Dict{Symbol, Any}, outcome::Symbol, dist, y) = _apply_hmm_filter!(
    hmm_priors, outcome, dist, y
)

"""
    obs_distributions(dm, idx, θ, η; cache=nothing, sol_accessors=nothing, hmm_filter=true) -> Vector{<:NamedTuple}

Per-observation-row predicted distributions for individual `idx` at natural-scale `(θ, η)`:
one `NamedTuple{outcome => Distribution}` per row of `get_obs_rows(dm)[idx]`. Solves the ODE
once (or reuses a passed `sol_accessors`). With `hmm_filter=true` (default) HMM outcomes are
forward-filtered in sequence, matching `plot_fits`/`build_plot_cache`. Returns an empty vector
on solve failure.
"""
function obs_distributions(
        dm::DataModel, idx::Integer, θ::ComponentArray, η;
        cache = nothing, sol_accessors = nothing, hmm_filter::Bool = true
    )
    model = get_model(dm)
    θ_re = symmetrize_psd_parameters(θ, get_fixed(model))
    η_ca = η isa NamedTuple ? ComponentArray(η) : η
    ind = get_individuals(dm)[Int(idx)]
    obs_rows = get_obs_rows(get_row_groups(dm))[Int(idx)]
    const_cov = get_const_cov(ind)
    obs_series = get_obs(get_series(ind))
    obs_cols = get_obs_cols(dm)
    has_de = get_de(model) !== nothing
    if has_de && sol_accessors === nothing
        c = _dev_ll_cache(dm, cache)
        pre = calculate_prede(model, θ_re, η_ca, const_cov)
        sol_accessors = _ll_solve_de(dm, Int(idx), θ_re, η_ca, c, pre)
        sol_accessors === nothing && return NamedTuple[]
    end
    rowwise_re = _needs_rowwise_random_effects(dm, Int(idx); obs_only = true)
    time_vec = _get_col(get_df(dm), get_time_col(dm))[obs_rows]
    hmm_priors = hmm_filter ? Dict{Symbol, Any}() : nothing
    out = Vector{NamedTuple}(undef, length(obs_rows))
    for i in eachindex(obs_rows)
        vary = _varying_at(dm, ind, i, time_vec)
        η_row = _row_random_effects_at(dm, Int(idx), i, η_ca, rowwise_re; obs_only = true)
        obs = has_de ?
            calculate_formulas_obs(model, θ_re, η_row, const_cov, vary, sol_accessors) :
            calculate_formulas_obs(model, θ_re, η_row, const_cov, vary)
        prs = Pair{Symbol, Any}[]
        for col in obs_cols
            dist = getproperty(obs, col)
            hmm_filter && (
                dist = _apply_hmm_filter!(
                    hmm_priors, col, dist,
                    getfield(obs_series, col)[i]
                )
            )
            push!(prs, col => dist)
        end
        out[i] = NamedTuple(prs)
    end
    return out
end

# ── Posterior / empirical Bayes / marginal ───────────────────────────────────

# Modes + aligned batch structure at natural-scale θ. Batch order matches a fresh
# build_re_batch_infos(dm, constants_re), so bstars[bi] pairs with infos[bi]/cc.
function _empirical_bayes_batches(
        dm::DataModel, θ::ComponentArray;
        constants_re::NamedTuple = NamedTuple(), ebe_options::EBEOptions = EBEOptions(),
        rescue = nothing, ode_args::Tuple = (), ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Random.default_rng()
    )
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    cache = build_likelihood_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization
    )
    _, infos, cc = build_re_batch_infos(dm, constants_re)
    bstars, _ = _compute_bstars(
        dm, θ_re, constants_re, cache, ebe_options, rng;
        rescue = rescue
    )
    return bstars, infos, cc, θ_re, cache
end

"""
    empirical_bayes(dm, θ; constants_re=NamedTuple(), ebe_options=EBEOptions(), rescue=nothing, ode_args=(), ode_kwargs=NamedTuple(), serialization=EnsembleThreads(), rng=Random.default_rng()) -> Vector{Vector{Float64}}
    empirical_bayes(dm, θ, idx::Integer; kwargs...) -> ComponentArray

Empirical-Bayes (posterior-mode) random effects at an arbitrary natural-scale `θ`. The
population form returns the per-batch mode vectors `b*` in `build_re_batch_infos(dm, constants_re)`
order; the per-individual form returns that subject's η as a `ComponentArray`. NOT
differentiable in `θ` (the inner mode solver floatizes `θ`); for a θ-gradient of the marginal
use the Laplace fit's analytic gradient.
"""
empirical_bayes(dm::DataModel, θ::ComponentArray; kwargs...) = _empirical_bayes_batches(
    dm, θ; kwargs...
)[1]

function empirical_bayes(dm::DataModel, θ::ComponentArray, idx::Integer; kwargs...)
    bstars, infos, cc, θ_re, _ = _empirical_bayes_batches(dm, θ; kwargs...)
    return eta_from_modes(dm, infos, bstars, cc, θ_re)[Int(idx)]
end

"""
    empirical_bayes_covariance(dm, θ, batch, b_star; const_cache, cache=nothing, curvature=ExactHessianCurvature(), jitter=1e-6, max_tries=6, adaptive=false, scale_factor=0.0) -> Union{Matrix, Nothing}
    empirical_bayes_covariance(dm, θ, bstars::AbstractVector; constants_re=NamedTuple(), kwargs...) -> Vector

Curvature-based covariance `Σ = (−H)⁻¹` (natural b-space) of the random effects at an
empirical-Bayes mode, with `H = ∇²_b log p(y, b | θ)`. Together with the mode from
[`empirical_bayes`](@ref) this defines the Laplace (Gaussian) approximation `N(b*, Σ)` to the
random-effect posterior `p(b | y, θ)` - exact when the model is linear in `b` with Gaussian
noise. The batch form takes one mode `b_star` and returns its `Σ`; the vector form takes the
per-batch modes (aligned with `build_re_batch_infos` order) and returns one `Σ` per batch.
Call [`empirical_bayes`](@ref) alone when only the modes are needed - the Hessian is never
computed there. `Σ` is `nothing` when `−H` is not positive definite after jitter.
"""
function empirical_bayes_covariance(
        dm::DataModel, θ::ComponentArray, batch::REBatchInfo, b_star;
        const_cache::REConstantsCache, cache = nothing,
        curvature::AbstractCurvature = ExactHessianCurvature(), jitter = 1.0e-6,
        max_tries::Int = 6, adaptive::Bool = false, scale_factor = 0.0
    )
    c = _dev_ll_cache(dm, cache)
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    _, _, chol = _laplace_logdet_negH(
        dm, batch, θ_re, b_star, const_cache, c, nothing, 1;
        jitter = jitter, max_tries = max_tries, adaptive = adaptive,
        scale_factor = scale_factor, hmode = curvature
    )
    (chol === nothing || chol.info != 0) && return nothing
    return Matrix(inv(chol))
end

function empirical_bayes_covariance(
        dm::DataModel, θ::ComponentArray, bstars::AbstractVector;
        constants_re::Union{NamedTuple, AbstractDict} = NamedTuple(), ode_args::Tuple = (),
        ode_kwargs::Union{NamedTuple, AbstractDict} = NamedTuple(), kwargs...
    )
    constants_re = _as_namedtuple(constants_re)
    ode_kwargs = _as_namedtuple(ode_kwargs)
    cache = build_likelihood_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        force_saveat = true
    )
    _, infos, cc = build_re_batch_infos(dm, constants_re)
    length(bstars) == length(infos) ||
        error("empirical_bayes_covariance: got $(length(bstars)) modes for $(length(infos)) batches.")
    return [
        empirical_bayes_covariance(
                dm, θ, infos[bi], bstars[bi]; const_cache = cc,
                cache = cache, kwargs...
            ) for bi in eachindex(infos)
    ]
end

"""
    laplace_marginal(dm, θ, batch, b_star; const_cache, cache=nothing, jitter=1e-6, max_tries=6, adaptive=false, scale_factor=0.0) -> Real
    laplace_marginal(dm, θ; kwargs...) -> Float64

Laplace-approximate marginal log-likelihood
`log p(y | θ) ≈ log f(b*) + ½·n_b·log(2π) − ½·log det(−H)`. The batch form uses a supplied mode
`b_star`; the population form finds the modes and sums over batches. For a θ-gradient of the
marginal use [`laplace_marginal_gradient`](@ref) (the Laplace fit's analytic envelope +
trace-estimator gradient); naive AD through the recomputed mode is not supported.
"""
function laplace_marginal(
        dm::DataModel, θ::ComponentArray, batch::REBatchInfo, b_star;
        const_cache::REConstantsCache, cache = nothing,
        curvature::AbstractCurvature = ExactHessianCurvature(), jitter = 1.0e-6,
        max_tries::Int = 6, adaptive::Bool = false, scale_factor = 0.0
    )
    c = _dev_ll_cache(dm, cache)
    logf = _laplace_logf_batch(dm, batch, θ, b_star, const_cache, c)
    logdet_negH, _, _ = _laplace_logdet_negH(
        dm, batch, θ, b_star, const_cache, c, nothing, 1;
        jitter = jitter, max_tries = max_tries, adaptive = adaptive,
        scale_factor = scale_factor, hmode = curvature
    )
    # `_laplace_logdet_negH` already returns Inf for a degenerate or non-factorizable -H,
    # which makes the result -Inf; warn here (a reporting path) rather than in the hot loop.
    if isinf(logdet_negH)
        @warn "laplace_marginal: the Laplace expansion is invalid at b* (-H degenerate or " *
            "not factorizable; b* may not be a true mode). Returning -Inf."
        return convert(typeof(logf), -Inf)
    end
    n_b = get_batch_re_dim(batch)
    return logf + (n_b / 2) * log(2 * pi) - logdet_negH / 2
end

function laplace_marginal(
        dm::DataModel, θ::ComponentArray;
        constants_re::Union{NamedTuple, AbstractDict} = NamedTuple(),
        curvature::AbstractCurvature = ExactHessianCurvature(), jitter = 1.0e-6,
        max_tries::Int = 6, adaptive::Bool = false, scale_factor = 0.0, kwargs...
    )
    constants_re = _as_namedtuple(constants_re)
    bstars, infos, cc, θ_re, cache = _empirical_bayes_batches(
        dm, θ; constants_re = constants_re, kwargs...
    )
    isempty(infos) && return zero(eltype(θ_re))
    return sum(
        laplace_marginal(
                dm, θ_re, infos[bi], bstars[bi]; const_cache = cc,
                cache = cache, curvature = curvature, jitter = jitter,
                max_tries = max_tries, adaptive = adaptive, scale_factor = scale_factor
            )
            for bi in eachindex(infos)
    )
end

# Move a natural-scale θ-gradient onto the transformed scale with the same Jacobian
# the Laplace fit applies to its analytic gradient (laplace.jl `obj_grad`).
function _dev_gradient_scale(dm::DataModel, θ::ComponentArray, grad::ComponentArray, scale::Symbol)
    scale === :untransformed && return grad
    scale === :transformed ||
        error("gradient scale must be :untransformed or :transformed, got :$scale.")
    fe = get_fixed(get_model(dm))
    θ_re = symmetrize_psd_parameters(θ, fe)
    return apply_inv_jacobian_T(get_inverse_transform(fe), get_transform(fe)(θ_re), grad)
end

"""
    laplace_marginal_gradient(dm, θ, batch, b_star; const_cache, cache=nothing, scale=:untransformed, curvature=ExactHessianCurvature(), jitter=1e-6, max_tries=6, growth=10.0, adaptive=false, scale_factor=0.0, use_trace_logdet_grad=true, use_hutchinson=false, hutchinson_n=8, rng=Random.default_rng()) -> (value, gradient)
    laplace_marginal_gradient(dm, θ; constants_re=NamedTuple(), scale=:untransformed, kwargs...) -> (value, gradient)
    laplace_marginal_gradient(ctx::FitContext, θ; scale=:untransformed, kwargs...) -> (value, gradient)

Value AND analytic θ-gradient of the Laplace-approximate marginal log-likelihood
[`laplace_marginal`](@ref) - the exact gradient the `Laplace` fit optimizes, including the
implicit `db*/dθ` envelope correction, so naive AD through the re-optimized mode is not
needed. `value` is identical to `laplace_marginal` at the same arguments. The batch form
takes one supplied mode `b_star`; the population form finds the modes and sums both value and
gradient over batches (subjects are independent, so per-site sums are exact - this is the
federation property). The `FitContext` form is the same population computation with the
context's caches reused instead of rebuilt per call; kwargs frozen by the context
(`constants_re`, `ode_args`, `ode_kwargs`, the caches) error rather than silently diverge.

Scale contract: `θ` is natural-scale (as everywhere in this API) and `scale` selects the
scale of the RETURNED gradient. `:untransformed` (default) is `∂/∂θ` on the natural scale;
`:transformed` applies the fixed-effects transform Jacobian and returns `∂/∂θ_t` on the
optimizer's transformed scale (`:log`, `:logit`, Cholesky, ... - see
`get_transform`). The gradient is a `ComponentArray` on `θ`'s axes for
`:untransformed` and on the transformed-θ axes for `:transformed`.
"""
function laplace_marginal_gradient(
        dm::DataModel, θ::ComponentArray, batch::REBatchInfo, b_star;
        const_cache::REConstantsCache, cache = nothing, scale::Symbol = :untransformed,
        curvature::AbstractCurvature = ExactHessianCurvature(), jitter = 1.0e-6,
        max_tries::Int = 6, growth = 10.0, adaptive::Bool = false, scale_factor = 0.0,
        use_trace_logdet_grad::Bool = true, use_hutchinson::Bool = false,
        hutchinson_n::Int = 8, rng::AbstractRNG = Random.default_rng()
    )
    c = _dev_ll_cache(dm, cache)
    res = _laplace_grad_batch(
        dm, batch, θ, b_star, const_cache, c, nothing, 1;
        jitter = jitter, max_tries = max_tries, growth = growth, adaptive = adaptive,
        scale_factor = scale_factor, use_trace_logdet_grad = use_trace_logdet_grad,
        use_hutchinson = use_hutchinson, hutchinson_n = hutchinson_n, rng = rng,
        hmode = curvature
    )
    if isinf(res.logdet)
        @warn "laplace_marginal_gradient: the Laplace expansion is invalid at b* (-H " *
            "degenerate or not factorizable; b* may not be a true mode). Returning -Inf."
    end
    n_b = get_batch_re_dim(batch)
    value = res.logf + (n_b / 2) * log(2 * pi) - res.logdet / 2
    return (value, _dev_gradient_scale(dm, θ, res.grad, scale))
end

function laplace_marginal_gradient(
        dm::DataModel, θ::ComponentArray;
        constants_re::Union{NamedTuple, AbstractDict} = NamedTuple(),
        scale::Symbol = :untransformed,
        curvature::AbstractCurvature = ExactHessianCurvature(), jitter = 1.0e-6,
        max_tries::Int = 6, growth = 10.0, adaptive::Bool = false, scale_factor = 0.0,
        use_trace_logdet_grad::Bool = true, use_hutchinson::Bool = false,
        hutchinson_n::Int = 8, rng::AbstractRNG = Random.default_rng(), kwargs...
    )
    constants_re = _as_namedtuple(constants_re)
    bstars, infos, cc, θ_re, cache = _empirical_bayes_batches(
        dm, θ; constants_re = constants_re, rng = rng, kwargs...
    )
    value = zero(eltype(θ_re))
    grad = ComponentArray(zeros(eltype(θ_re), length(θ_re)), getaxes(θ_re))
    for bi in eachindex(infos)
        v, g = laplace_marginal_gradient(
            dm, θ_re, infos[bi], bstars[bi]; const_cache = cc, cache = cache,
            scale = :untransformed, curvature = curvature, jitter = jitter,
            max_tries = max_tries, growth = growth, adaptive = adaptive,
            scale_factor = scale_factor, use_trace_logdet_grad = use_trace_logdet_grad,
            use_hutchinson = use_hutchinson, hutchinson_n = hutchinson_n, rng = rng
        )
        value += v
        grad .+= g
    end
    return (value, _dev_gradient_scale(dm, θ_re, grad, scale))
end

"""
    ghq_marginal(dm, θ, batch::REBatchInfo; level=3, const_cache, cache=nothing) -> Real
    ghq_marginal(dm, θ; level=3, constants_re=NamedTuple(), ode_args=(), ode_kwargs=NamedTuple()) -> Float64

Gauss-Hermite (Smolyak sparse-grid) marginal log-likelihood `log p(y | θ)`, integrating the
free random effects against their prior-centered Gaussian measure - the deterministic
integrator the `GHQuadrature` estimator uses (no mode-finding). `level` is an `Int` (isotropic)
or a `NamedTuple` mapping RE name → level (anisotropic). This is distinct from the adaptive
`get_marginal_likelihood` (AGHQ, centered at the posterior mode).
"""
function ghq_marginal(
        dm::DataModel, θ::ComponentArray, batch::REBatchInfo;
        level = 3, const_cache::REConstantsCache, cache = nothing
    )
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    return _ghq_batch_ll(dm, batch, θ_re, const_cache, _dev_ll_cache(dm, cache), level)
end

function ghq_marginal(
        dm::DataModel, θ::ComponentArray;
        level = 3, constants_re::Union{NamedTuple, AbstractDict} = NamedTuple(),
        ode_args::Tuple = (), ode_kwargs::Union{NamedTuple, AbstractDict} = NamedTuple()
    )
    level = _as_namedtuple(level)
    constants_re = _as_namedtuple(constants_re)
    ode_kwargs = _as_namedtuple(ode_kwargs)
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    c = build_likelihood_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = EnsembleSerial(), force_saveat = true
    )
    _, infos, cc = build_re_batch_infos(dm, constants_re)
    isempty(infos) && return zero(eltype(θ_re))
    return sum(_ghq_batch_ll(dm, infos[bi], θ_re, cc, c, level) for bi in eachindex(infos))
end

"""
    RandomEffectPosteriorSample{D, W, E}

Posterior draws of a batch's random effects. `draws` is an `n_b × n_samples` matrix (columns
are natural-scale `b` draws); `log_weights` are importance log-weights (`nothing` for
unweighted draws); `ess` is the effective sample size. Access with `get_draws`,
`get_log_weights`, `get_ess`.
"""
struct RandomEffectPosteriorSample{D, W, E}
    draws::D
    log_weights::W
    ess::E
    method::Symbol
end
@inline get_draws(s::RandomEffectPosteriorSample) = s.draws
@inline get_log_weights(s::RandomEffectPosteriorSample) = s.log_weights
@inline get_ess(s::RandomEffectPosteriorSample) = s.ess

"""
    sample_random_effect_draws(dm, θ, batch::REBatchInfo, b_star; method=:importance, sampler=nothing, n_samples=100, n_adapt=50, const_cache, cache=nothing, rng=Random.default_rng()) -> RandomEffectPosteriorSample
    sample_random_effect_draws(dm, θ; method=:importance, sampler=nothing, n_samples=100, constants_re=NamedTuple(), rng=Random.default_rng(), ...) -> Vector{RandomEffectPosteriorSample}

Draw from the random-effect posterior `p(η | y, θ)`.

- `method=:importance` (default, Turing-free): Laplace-Gaussian importance sampling - a Gaussian
  proposal centered at the mode `b_star` with covariance `(−H)⁻¹`
  ([`empirical_bayes_covariance`](@ref)),
  reweighted by `log p(y, η | θ) − log q(η)` (exact/uniform weights for linear-Gaussian models).
  Populates `log_weights`/`ess`.
- `method=:mcmc`: draws directly from the exact posterior with a Turing `sampler` (required, e.g.
  `MH()`, `NUTS()`) via the same batch model the MCEM E-step uses; `log_weights`/`ess` are
  `nothing`. `b_star` is unused for `:mcmc`.

The population form finds the modes (for `:importance`) and returns one sample per batch.
"""
function sample_random_effect_draws(
        dm::DataModel, θ::ComponentArray, batch::REBatchInfo, b_star;
        method::Symbol = :importance, sampler = nothing, n_samples::Int = 100,
        n_adapt::Int = 50, const_cache::REConstantsCache, cache = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    c = _dev_ll_cache(dm, cache)
    n_b = get_batch_re_dim(batch)
    if method === :mcmc
        sampler === nothing &&
            error("sample_random_effect_draws(method=:mcmc) requires a Turing `sampler`, e.g. MH() or NUTS().")
        n_b == 0 &&
            return RandomEffectPosteriorSample(
            zeros(eltype(θ_re), 0, 0), nothing, nothing, :mcmc
        )
        re_names = get_re_names(get_random(get_model(dm)))
        tkw = (n_samples = n_samples, n_adapt = n_adapt, progress = false)
        samples, _, _ = _mcem_sample_batch(
            dm, batch, θ_re, const_cache, c, sampler, tkw, rng, re_names, false, nothing
        )
        return RandomEffectPosteriorSample(samples, nothing, nothing, :mcmc)
    elseif method === :importance
        n_b == 0 &&
            return RandomEffectPosteriorSample(
            zeros(0, n_samples), zeros(n_samples),
            Float64(n_samples), :importance
        )
        Σ = empirical_bayes_covariance(
            dm, θ_re, batch, b_star; const_cache = const_cache, cache = c
        )
        Σ === nothing &&
            return RandomEffectPosteriorSample(zeros(n_b, 0), Float64[], 0.0, :importance)
        q = MvNormal(collect(float.(b_star)), Symmetric(Matrix(Σ)))
        draws = Matrix{Float64}(undef, n_b, n_samples)
        logw = Vector{Float64}(undef, n_samples)
        for r in 1:n_samples
            b_r = rand(rng, q)
            @inbounds draws[:, r] = b_r
            logp = complete_data_loglikelihood(
                dm, batch, θ_re, b_r; const_cache = const_cache, cache = c
            )
            @inbounds logw[r] = logp - logpdf(q, b_r)
        end
        w = exp.(logw .- maximum(logw))
        sw = sum(w)
        ess = sw > 0 ? sw^2 / sum(abs2, w) : 0.0
        return RandomEffectPosteriorSample(draws, logw, ess, :importance)
    end
    error("Unknown sample_random_effect_draws method $(method); use :importance or :mcmc.")
end

function sample_random_effect_draws(
        dm::DataModel, θ::ComponentArray;
        method::Symbol = :importance, sampler = nothing, n_samples::Int = 100,
        n_adapt::Int = 50, constants_re::NamedTuple = NamedTuple(),
        ode_args::Tuple = (), ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        ebe_options::EBEOptions = EBEOptions(), rescue = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    if method === :mcmc
        θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
        c = build_likelihood_cache(
            dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
            serialization = EnsembleSerial(), force_saveat = true
        )
        _, infos, cc = build_re_batch_infos(dm, constants_re)
        return [
            sample_random_effect_draws(
                    dm, θ_re, infos[bi], eltype(θ_re)[]; method = :mcmc,
                    sampler = sampler, n_samples = n_samples, n_adapt = n_adapt,
                    const_cache = cc, cache = c, rng = rng
                ) for bi in eachindex(infos)
        ]
    end
    bstars, infos, cc, θ_re, cache = _empirical_bayes_batches(
        dm, θ;
        constants_re = constants_re, ebe_options = ebe_options, rescue = rescue,
        ode_args = ode_args, ode_kwargs = ode_kwargs, serialization = serialization,
        rng = rng
    )
    return [
        sample_random_effect_draws(
                dm, θ_re, infos[bi], bstars[bi]; method = :importance,
                n_samples = n_samples, const_cache = cc, cache = cache, rng = rng
            )
            for bi in eachindex(infos)
    ]
end

# ── Fitting-method protocol: the drivers a new estimator plugs into ───────────

"""
    fit_method(dm, method, args...; kwargs...) -> FitResult

The single method a new `FittingMethod` implements. `fit_model` (which layers on pooled-init
and multistart) dispatches here, so defining `fit_method(dm, ::MyMethod, …)` makes
`fit_model(dm, MyMethod(...))` and `Multistart` work automatically. Implementations usually
delegate to [`fit_fixed_effects`](@ref) (no random effects) or [`fit_laplace_family`](@ref)
(marginal random-effects).
"""
const fit_method = _fit_model

"""
    fit_fixed_effects(dm, method; objective_term=θu->0.0, constants, penalty, ode_args, ode_kwargs, serialization, kwargs...) -> FitResult

Shared driver for fixed-effects-only methods: minimizes `−loglikelihood + penalty +
objective_term(θu)` over the free parameters, with constants/bounds/transform/result-packaging
handled. `objective_term` is a natural-scale add-on (e.g. a ridge penalty or a log-prior; MAP
passes its prior term here). Requires the method to carry
`optimizer`/`optim_kwargs`/`adtype`/`lb`/`ub`/`ignore_model_bounds`.
"""
fit_fixed_effects(dm::DataModel, method; objective_term = _NoOpTerm(), kwargs...) = _fit_no_re(
    dm, method; add_term = objective_term, kwargs...
)

"""
    fit_laplace_family(dm, method, curvature::AbstractCurvature, args, fit_kwargs, validate_post_transform; kwargs...) -> FitResult

Shared driver for marginal random-effects methods (Laplace/FOCEI and any custom curvature):
finds the EB modes, assembles the Laplace marginal with the supplied `curvature`, and optimizes
it with the analytic (envelope + trace-estimator) θ-gradient. Swap only the `curvature`
([`AbstractCurvature`](@ref)/[`inner_curvature`](@ref)) to define a new marginal method.
"""
const fit_laplace_family = _fit_laplace_family

# ── Joint objective value + θ-gradient (method-dispatched) ────────────────────
# One entry point per estimator family, each reusing the exact route its own fit
# differentiates, so the gradient a caller gets is the gradient the optimizer consumed.

const _DiffResults = ForwardDiff.DiffResults

# Normalizes `scale` (Symbol or string, per the #256 convention) and validates it.
@inline function _dev_check_scale(scale::Union{Symbol, AbstractString})
    s = _as_symbol(scale)
    s in (:untransformed, :transformed) ||
        error("gradient scale must be :untransformed or :transformed, got :$s.")
    return s
end

# The penalty enters the log-objective negatively (the fits minimize `-ll + penalty`).
@inline function _dev_penalty_term(θ::ComponentArray, include_penalty::Bool, penalty::NamedTuple)
    (!include_penalty || isempty(keys(penalty))) && return zero(eltype(θ))
    return -penalty_value(θ, penalty)
end

# ∂/∂θ of `-penalty_value(θ, penalty)`, added to an analytic gradient in place.
function _dev_penalty_gradient!(g::ComponentArray, θ::ComponentArray, penalty::NamedTuple)
    for name in keys(penalty)
        w = getfield(penalty, name)
        setproperty!(g, name, getproperty(g, name) .- 2 .* w .* getproperty(θ, name))
    end
    return g
end

# Objective wrappers for the AD sweep: the differentiated argument is the flat coordinate
# vector of the requested scale; `:transformed` differentiates through the inverse
# transform, which is what every fit's optimizer-scale objective does.
struct _DevObjNatural{F, A}
    f::F
    axs::A
end
@inline (o::_DevObjNatural)(x) = o.f(ComponentArray(x, o.axs))

struct _DevObjTransformed{F, I, E, A}
    f::F
    inv_transform::I
    fe::E
    axs::A
end
@inline function (o::_DevObjTransformed)(x)
    return o.f(symmetrize_psd_parameters(o.inv_transform(ComponentArray(x, o.axs)), o.fe))
end

# Single ForwardDiff/DiffResults sweep of a natural-scale objective `f(θ::ComponentArray)`:
# value and gradient (and the Hessian when asked) come out of the one sweep.
function _dev_sweep(f, dm::DataModel, θ::ComponentArray, scale::Symbol, hessian::Bool)
    _dev_check_scale(scale)
    fe = get_fixed(get_model(dm))
    θ_re = symmetrize_psd_parameters(θ, fe)
    if scale === :untransformed
        axs = getaxes(θ_re)
        x0 = collect(ComponentArrays.getdata(θ_re))
        g = _DevObjNatural(f, axs)
    else
        θt = get_transform(fe)(θ_re)
        axs = getaxes(θt)
        x0 = collect(ComponentArrays.getdata(θt))
        g = _DevObjTransformed(f, get_inverse_transform(fe), fe, axs)
    end
    if hessian
        res = ForwardDiff.hessian!(_DiffResults.HessianResult(x0), g, x0)
        H = _DiffResults.hessian(res)
        return (
            _DiffResults.value(res),
            ComponentArray(_DiffResults.gradient(res), axs),
            (H .+ H') ./ 2,
        )
    end
    res = ForwardDiff.gradient!(_DiffResults.GradientResult(x0), g, x0)
    return (_DiffResults.value(res), ComponentArray(_DiffResults.gradient(res), axs))
end

# Central FD Jacobian of an analytic gradient, symmetrized: the Laplace/FOCEI Hessian.
function _dev_fd_hessian(g_of_x, x0::Vector{Float64}, step::Real)
    p = length(x0)
    H = Matrix{Float64}(undef, p, p)
    for j in 1:p
        h = step * max(1.0, abs(x0[j]))
        xp = copy(x0)
        xm = copy(x0)
        xp[j] += h
        xm[j] -= h
        H[:, j] .= (g_of_x(xp) .- g_of_x(xm)) ./ (2h)
    end
    return (H .+ H') ./ 2
end

"""
    objective_and_gradient(method, dm, θ; scale=:untransformed, hessian=false, include_penalty=false, penalty=NamedTuple(), kwargs...)
        -> (value, gradient) | (value, gradient, hessian)
    objective_and_gradient(method, ctx::FitContext, θ; same kwargs) -> same

Value AND θ-gradient of `method`'s log-objective at natural-scale `θ`, computed together the
way the corresponding `fit_model` computes them - the shared analytic kernel for
`Laplace`/`FOCEI`, a single ForwardDiff/DiffResults sweep elsewhere. `value` is the method's
core LOG-objective (higher is better, matching the scalar covers `laplace_marginal`,
`ghq_marginal`, `loglikelihood`); the optimizer-side negation, preconditioning and
free/constant merging are not part of the primitive. The gradient is a `ComponentArray` on
`θ`'s axes (`scale = :untransformed`) or on the transformed axes (`:transformed`).

`hessian = true` additionally returns the symmetric Hessian as a plain `Matrix` in the flat
coordinate order of the returned gradient. Per family: `Laplace`/`FOCEI` use central finite
differences over the ANALYTIC gradient (2p gradient evaluations - second-order AD through the
implicit empirical-Bayes modes is not available), everything else a second-order ForwardDiff
sweep. Expect ~5-6 digits from the FD-based Hessian and full precision from the AD one.

`penalty` is excluded by default; pass `include_penalty = true` together with the fit's
`penalty` named tuple to reproduce a penalized fit's objective. `MAP`'s log-priors are not a
penalty and are always part of the `MAP` objective.

Shipped dispatches, and the finer-grained forms:

  - `Laplace`, `FOCEI`: forwards to [`laplace_marginal_gradient`](@ref) with the method's own
    curvature (exact inner Hessian / Fisher information) and Hessian options. Per-batch form
    `objective_and_gradient(method, dm, θ, batch, b_star; const_cache, …)`; batch pairs sum to
    the population pair (the federation property).
  - `GHQuadrature`: sweeps [`ghq_marginal`](@ref), whose adaptive centers are built from the
    Dual-stripped `θ` and are therefore CONSTANT with respect to the gradient - the fit's own
    convention, so the returned gradient is the one `fit_model(dm, GHQuadrature())` optimizes.
    Per-batch form `objective_and_gradient(method, dm, θ, batch; const_cache, …)`.
  - `Pooled`: sweeps the plug-in objective `loglikelihood(dm, θ, η(θ))` with AD flowing through
    `η(θ)`, as the pooled fit does. The plug-in `strategies` are derived from the method and `θ`
    unless supplied (`strategies = get_result(res).strategies` reproduces a fit exactly), and
    `eta = …` freezes η instead. Population form only: the plug-in strategies are calibrated on
    the whole data set.
  - `MLE`, `MAP`: sweeps `loglikelihood` (plus `logprior` for `MAP`). `MLE` also has the
    per-individual form `objective_and_gradient(MLE(), dm, θ, idx)`. Like `fit_model`, these
    require a model WITHOUT random effects; use `Laplace`, `SAEM` or `MCMC` otherwise.

The `FitContext` form (`Laplace`, `FOCEI`, `GHQuadrature`, `MLE`, `MAP`, `Pooled`) returns
identical results and reuses the context's batch infos, constant-RE cache and likelihood cache
instead of rebuilding them per call - the form to use inside an iterating (e.g. federated)
loop. `Pooled`/`MLE`/`MAP` reuse only the likelihood cache (they have no free random effects to
batch, and `Pooled` recalibrates its plug-in `strategies` at every θ by construction). Options
frozen at `build_fit_context` time (`constants_re`, `ode_args`, `ode_kwargs`, and the caches
themselves) are not accepted per call and error if passed.

A user-defined `FittingMethod` plugs into downstream tooling (federated aggregation,
uncertainty workflows) by adding its own `objective_and_gradient` method.
"""
function objective_and_gradient(method::FittingMethod, dm::DataModel, θ::ComponentArray; kwargs...)
    return error(
        "objective_and_gradient is not defined for $(typeof(method)). It ships for Laplace, " *
            "FOCEI, GHQuadrature, Pooled, MLE and MAP; sampling-based methods (SAEM/MCEM/MCMC/VI) " *
            "have no deterministic θ-objective. Add a method for your own FittingMethod."
    )
end

# ── Laplace / FOCEI: the shared analytic kernel ───────────────────────────────

_dev_curvature(::Laplace) = ExactHessianCurvature()
_dev_curvature(method::FOCEI) = FisherInformationCurvature(method.interaction)

# The method's own Laplace-Hessian block (jitter, trace estimator, ...), so the primitive is
# configured like the fit; explicit kwargs win.
function _dev_laplace_kwargs(method, kwargs)
    h = method.hessian
    opts = NamedTuple(n => getfield(h, n) for n in fieldnames(typeof(h)))
    return merge((; curvature = _dev_curvature(method)), opts, NamedTuple(kwargs))
end

# `target` is a `DataModel` or a `FitContext` (defined below); dispatch picks the route.
_dev_dm(dm::DataModel) = dm

function _dev_laplace_pair(target, θ::ComponentArray, call_kwargs, include_penalty, penalty)
    v, g = laplace_marginal_gradient(target, θ; scale = :untransformed, call_kwargs...)
    if include_penalty && !isempty(keys(penalty))
        v += _dev_penalty_term(θ, true, penalty)
        g = _dev_penalty_gradient!(g, θ, penalty)
    end
    return (v, g)
end

function objective_and_gradient(
        method::Union{Laplace, FOCEI}, dm::DataModel, θ::ComponentArray; kwargs...
    )
    return _dev_laplace_og(method, dm, θ; kwargs...)
end

# Shared body for the `dm` and `FitContext` forms (`target` is either).
function _dev_laplace_og(
        method::Union{Laplace, FOCEI}, target, θ::ComponentArray;
        scale::Union{Symbol, AbstractString} = :untransformed, hessian::Bool = false,
        include_penalty::Bool = false,
        penalty::Union{NamedTuple, AbstractDict} = NamedTuple(),
        fd_step::Real = 1.0e-5, kwargs...
    )
    scale = _dev_check_scale(scale)
    penalty = _as_namedtuple(penalty)
    call_kwargs = _dev_laplace_kwargs(method, kwargs)
    dm = _dev_dm(target)
    fe = get_fixed(get_model(dm))
    θ_re = symmetrize_psd_parameters(θ, fe)
    value, grad = _dev_laplace_pair(target, θ_re, call_kwargs, include_penalty, penalty)
    scaled = _dev_gradient_scale(dm, θ_re, grad, scale)
    hessian || return (value, scaled)
    # FD over the analytic gradient, in the coordinates of the requested scale.
    if scale === :untransformed
        axs = getaxes(θ_re)
        x0 = collect(ComponentArrays.getdata(θ_re))
        g_of_x = function (xv)
            θx = ComponentArray(xv, axs)
            _, gx = _dev_laplace_pair(target, θx, call_kwargs, include_penalty, penalty)
            return collect(gx)
        end
    else
        θt = get_transform(fe)(θ_re)
        axs = getaxes(θt)
        x0 = collect(ComponentArrays.getdata(θt))
        it = get_inverse_transform(fe)
        g_of_x = function (xv)
            θx = symmetrize_psd_parameters(it(ComponentArray(xv, axs)), fe)
            _, gx = _dev_laplace_pair(target, θx, call_kwargs, include_penalty, penalty)
            return collect(_dev_gradient_scale(dm, θx, gx, :transformed))
        end
    end
    return (value, scaled, _dev_fd_hessian(g_of_x, x0, fd_step))
end

function objective_and_gradient(
        method::Union{Laplace, FOCEI}, dm::DataModel, θ::ComponentArray,
        batch::REBatchInfo, b_star; scale::Union{Symbol, AbstractString} = :untransformed, kwargs...
    )
    scale = _dev_check_scale(scale)
    return laplace_marginal_gradient(
        dm, θ, batch, b_star; scale = scale, _dev_laplace_kwargs(method, kwargs)...
    )
end

# ── GHQuadrature: the fit's fixed-center quadrature sum ───────────────────────

struct _DevGHQObjective{D, B, C, K, L, P}
    dm::D
    infos::B
    const_cache::C
    cache::K
    level::L
    penalty::P
    include_penalty::Bool
end

@inline function (o::_DevGHQObjective)(θ::ComponentArray)
    total = _dev_penalty_term(θ, o.include_penalty, o.penalty)
    for bi in eachindex(o.infos)
        total += ghq_marginal(
            o.dm, θ, o.infos[bi]; level = o.level,
            const_cache = o.const_cache, cache = o.cache
        )
    end
    return total
end

function objective_and_gradient(
        method::GHQuadrature, dm::DataModel, θ::ComponentArray;
        scale::Union{Symbol, AbstractString} = :untransformed, hessian::Bool = false,
        include_penalty::Bool = false,
        penalty::Union{NamedTuple, AbstractDict} = NamedTuple(),
        level = method.level, constants_re::Union{NamedTuple, AbstractDict} = NamedTuple(),
        ode_args::Tuple = (), ode_kwargs::Union{NamedTuple, AbstractDict} = NamedTuple(),
        cache = nothing
    )
    scale = _dev_check_scale(scale)
    _, infos, cc = build_re_batch_infos(dm, _as_namedtuple(constants_re))
    c = cache === nothing ?
        build_likelihood_cache(
            dm; ode_args = ode_args, ode_kwargs = _as_namedtuple(ode_kwargs),
            serialization = EnsembleSerial(), force_saveat = true
        ) : cache
    f = _DevGHQObjective(
        dm, infos, cc, c, level, _as_namedtuple(penalty), include_penalty
    )
    return _dev_sweep(f, dm, θ, scale, hessian)
end

function objective_and_gradient(
        method::GHQuadrature, dm::DataModel, θ::ComponentArray, batch::REBatchInfo;
        const_cache::REConstantsCache, cache = nothing, scale::Union{Symbol, AbstractString} = :untransformed,
        hessian::Bool = false, level = method.level
    )
    scale = _dev_check_scale(scale)
    f = _DevGHQObjective(
        dm, [batch], const_cache, _dev_ll_cache(dm, cache), level, NamedTuple(), false
    )
    return _dev_sweep(f, dm, θ, scale, hessian)
end

# ── Pooled: AD through the plug-in η(θ) ──────────────────────────────────────

struct _DevPooledObjective{D, K, S, T, E, P}
    dm::D
    cache::K
    serialization::S
    strategies::T
    eta::E
    penalty::P
    include_penalty::Bool
end

@inline function (o::_DevPooledObjective)(θ::ComponentArray)
    η = o.eta === nothing ? _compute_pooled_etas(o.dm, θ, o.strategies) : o.eta
    ll = loglikelihood(
        o.dm, θ, η; cache = o.cache, serialization = o.serialization
    )
    return ll + _dev_penalty_term(θ, o.include_penalty, o.penalty)
end

# The plug-in strategies the pooled fit derives before optimizing (`_fit_pooled`).
function _dev_pooled_strategies(dm::DataModel, θ::ComponentArray, method, rng::AbstractRNG)
    fe = get_fixed(get_model(dm))
    strategies = _pooled_plugin_strategies(dm, θ; mc_draws = method.mc_draws, rng = rng)
    return _pooled_dual_safe_strategies(
        dm, θ, get_transform(fe)(θ), get_inverse_transform(fe),
        strategies, method.mc_draws, rng
    )
end

function objective_and_gradient(
        method::Pooled, dm::DataModel, θ::ComponentArray;
        scale::Union{Symbol, AbstractString} = :untransformed, hessian::Bool = false,
        include_penalty::Bool = false,
        penalty::Union{NamedTuple, AbstractDict} = NamedTuple(),
        strategies = nothing, eta = nothing,
        ode_args::Tuple = (), ode_kwargs::Union{NamedTuple, AbstractDict} = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleSerial(), cache = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    scale = _dev_check_scale(scale)
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    strat = strategies === nothing ?
        _dev_pooled_strategies(dm, θ_re, method, rng) : strategies
    c = cache === nothing ?
        build_likelihood_cache(
            dm; ode_args = ode_args, ode_kwargs = _as_namedtuple(ode_kwargs),
            serialization = serialization, force_saveat = true
        ) : cache
    f = _DevPooledObjective(
        dm, c, serialization, strat, eta, _as_namedtuple(penalty), include_penalty
    )
    return _dev_sweep(f, dm, θ, scale, hessian)
end

# ── MLE / MAP: the fixed-effects log-likelihood (plus MAP's log-priors) ───────

struct _DevNoREObjective{D, K, S, P, F}
    dm::D
    cache::K
    serialization::S
    penalty::P
    include_penalty::Bool
    fe::F     # `nothing` for MLE; the fixed-effects block (log-priors) for MAP
end

@inline function (o::_DevNoREObjective)(θ::ComponentArray)
    ll = loglikelihood(
        o.dm, θ, ComponentArray(); cache = o.cache, serialization = o.serialization
    )
    lp = o.fe === nothing ? zero(ll) : logprior(o.fe, θ)
    return ll + lp + _dev_penalty_term(θ, o.include_penalty, o.penalty)
end

function objective_and_gradient(
        method::Union{MLE, MAP}, dm::DataModel, θ::ComponentArray;
        scale::Union{Symbol, AbstractString} = :untransformed, hessian::Bool = false,
        include_penalty::Bool = false,
        penalty::Union{NamedTuple, AbstractDict} = NamedTuple(),
        ode_args::Tuple = (), ode_kwargs::Union{NamedTuple, AbstractDict} = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleSerial(), cache = nothing
    )
    scale = _dev_check_scale(scale)
    _require_no_random_effects(dm)
    fe = get_fixed(get_model(dm))
    c = cache === nothing ?
        build_likelihood_cache(
            dm; ode_args = ode_args, ode_kwargs = _as_namedtuple(ode_kwargs),
            serialization = serialization, force_saveat = true
        ) : cache
    f = _DevNoREObjective(
        dm, c, serialization, _as_namedtuple(penalty), include_penalty,
        method isa MAP ? fe : nothing
    )
    return _dev_sweep(f, dm, θ, scale, hessian)
end

struct _DevIndividualObjective{D, K}
    dm::D
    idx::Int
    cache::K
end

@inline function (o::_DevIndividualObjective)(θ::ComponentArray)
    return conditional_loglikelihood(o.dm, o.idx, θ, ComponentArray(); cache = o.cache)
end

function objective_and_gradient(
        ::MLE, dm::DataModel, θ::ComponentArray, idx::Integer;
        scale::Union{Symbol, AbstractString} = :untransformed, hessian::Bool = false, cache = nothing
    )
    scale = _dev_check_scale(scale)
    _require_no_random_effects(dm)
    f = _DevIndividualObjective(dm, Int(idx), _dev_ll_cache(dm, cache))
    return _dev_sweep(f, dm, θ, scale, hessian)
end

# ── MCEM M-step Q primitives ──────────────────────────────────────────────────
# The Monte-Carlo Q(θ) = Σ_b (1/M) Σ_m log f(y_b, η_b^m | θ) at FIXED posterior draws,
# split (as the MCEM fit is) into a Q1 part (full complete-data loglik, needs the ODE)
# and a Q2 part (RE-prior only). Both reuse the exact fit kernels `_mcem_Q`/`_mcem_Q2`,
# so the returned value is bit-identical to the fit's M-step Q at the same arguments.

"""
    mcem_q_partition(dm; constants=NamedTuple()) -> (q1, q2)

Partition the free (non-`constants`) fixed-effect names into the MCEM M-step's two
independent sub-problems: `q1` (names appearing in observation-side blocks, whose Q term
needs the ODE/likelihood) and `q2` (names appearing only in `@randomEffects` distribution
expressions, no ODE). Public wrapper over the partition the MCEM fit uses internally.
"""
function mcem_q_partition(
        dm::DataModel; constants::Union{NamedTuple, AbstractDict} = NamedTuple()
    )
    constants = _as_namedtuple(constants)
    fe = get_fixed(get_model(dm))
    free = [n for n in get_names(fe) if !(n in keys(constants))]
    return _partition_q1_q2_names(get_model(dm), free)
end

# Default free set for a part: partition ALL fixed-effect names (callers freeze a
# complement by passing an explicit `free_names` subset).
function _mcem_resolve_free_names(dm::DataModel, part::Symbol, free_names)
    free_names !== nothing && return collect(Symbol, free_names)
    p = _partition_q1_q2_names(get_model(dm), collect(Symbol, get_names(get_fixed(get_model(dm)))))
    return part === :q1 ? p.q1 : p.q2
end

# Per-batch weights: `nothing` (uniform/MCMC) when every batch is unweighted, else
# per-batch self-normalized weights (matching `_mcem_Q_core`'s sum-to-one convention).
function _mcem_weights_from_draws(draws)
    all(get_log_weights(d) === nothing for d in draws) && return nothing
    return map(draws) do d
        n = size(get_draws(d), 2)
        lw = get_log_weights(d)
        lw === nothing && return fill(1.0 / max(n, 1), n)
        w = exp.(lw .- maximum(lw))
        return w ./ sum(w)
    end
end

# Build the fit's exact caches + per-batch sample/weight matrices from a draw vector.
function _mcem_q_setup(
        dm::DataModel, draws, constants_re, serialization::SciMLBase.EnsembleAlgorithm, cache
    )
    constants_re = _as_namedtuple(constants_re)
    _, batch_infos, const_cache = build_re_batch_infos(dm, constants_re)
    length(draws) == length(batch_infos) ||
        error("mcem_q_objective_and_gradient: got $(length(draws)) draw batches but the model has $(length(batch_infos)); draws must come from the same `dm`/`constants_re`.")
    ll_cache = cache === nothing ?
        build_ll_cache(dm; serialization = serialization, force_saveat = true) : cache
    samples_by_batch = [get_draws(d) for d in draws]
    weights_by_batch = _mcem_weights_from_draws(draws)
    return batch_infos, const_cache, ll_cache, samples_by_batch, weights_by_batch
end

# θ_u (natural) -> scalar Q, reusing the fit's Q1/Q2 kernels verbatim.
struct _MCEMQObjective{D, B, C, L, S, W, E}
    dm::D
    batch_infos::B
    const_cache::C
    ll_cache::L
    samples_by_batch::S
    weights_by_batch::W
    serialization::E
    part::Symbol
end
@inline function (o::_MCEMQObjective)(θu::ComponentArray)
    if o.part === :q1
        return _mcem_Q(
            o.dm, o.batch_infos, θu, o.const_cache, o.ll_cache,
            o.samples_by_batch, o.weights_by_batch; serialization = o.serialization
        )
    end
    return _mcem_Q2(
        o.dm, o.batch_infos, θu, o.const_cache, o.ll_cache,
        o.samples_by_batch, o.weights_by_batch; serialization = o.serialization
    )
end

# Free-subset reconstruction: differentiate only `free_names`, freezing the
# complement at the template — the M-step's per-part reparametrization.
struct _MCEMFreeObjNatural{F, T, A, AF}
    f::F
    tmpl::T
    axs_full::A
    free_names::Vector{Symbol}
    axs_free::AF
end
@inline function (o::_MCEMFreeObjNatural)(x)
    θ_free = ComponentArray(x, o.axs_free)
    θ_full = ComponentArray(eltype(x).(o.tmpl), o.axs_full)
    for name in o.free_names
        setproperty!(θ_full, name, getproperty(θ_free, name))
    end
    return o.f(θ_full)
end

struct _MCEMFreeObjTransformed{F, I, T, A, AF}
    f::F
    inv_transform::I
    tmpl::T
    axs_full::A
    free_names::Vector{Symbol}
    axs_free::AF
end
@inline function (o::_MCEMFreeObjTransformed)(x)
    θt_free = ComponentArray(x, o.axs_free)
    θt_full = ComponentArray(eltype(x).(o.tmpl), o.axs_full)
    for name in o.free_names
        setproperty!(θt_full, name, getproperty(θt_free, name))
    end
    return o.f(o.inv_transform(θt_full))
end

# Single DiffResults sweep of `qf(θ_u)` over the `free_names` subset (mirror of
# `_dev_sweep`, but freezing the complement at `θ`). Returns (Q, gradient on free axes).
function _mcem_dev_sweep(qf, dm::DataModel, θ::ComponentArray, free_names::Vector{Symbol}, scale::Symbol)
    scale = _dev_check_scale(scale)
    fe = get_fixed(get_model(dm))
    θ_re = symmetrize_psd_parameters(θ, fe)
    if scale === :untransformed
        tmpl = collect(ComponentArrays.getdata(θ_re))
        axs_full = getaxes(θ_re)
        θ_free = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(θ_re, n) for n in free_names)))
        isempty(free_names) && return (qf(θ_re), θ_free)
        axs_free = getaxes(θ_free)
        x0 = collect(ComponentArrays.getdata(θ_free))
        g = _MCEMFreeObjNatural(qf, tmpl, axs_full, free_names, axs_free)
    else
        θt_full = get_transform(fe)(θ_re)
        tmpl = collect(ComponentArrays.getdata(θt_full))
        axs_full = getaxes(θt_full)
        θt_free = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(θt_full, n) for n in free_names)))
        isempty(free_names) && return (qf(θ_re), θt_free)
        axs_free = getaxes(θt_free)
        x0 = collect(ComponentArrays.getdata(θt_free))
        g = _MCEMFreeObjTransformed(qf, get_inverse_transform(fe), tmpl, axs_full, free_names, axs_free)
    end
    res = ForwardDiff.gradient!(_DiffResults.GradientResult(x0), g, x0)
    return (_DiffResults.value(res), ComponentArray(_DiffResults.gradient(res), axs_free))
end

"""
    mcem_q_objective_and_gradient(dm, θ, draws; part=:q1, free_names=nothing, scale=:transformed, constants_re=NamedTuple(), serialization=EnsembleThreads(), cache=nothing) -> (Q, gradient)
    mcem_q_objective_and_gradient(dm, θ, draws, idx; ...) -> (Q, gradient)

Value and θ-gradient of one MCEM M-step Q-function at FIXED posterior `draws`
(`Vector{RandomEffectPosteriorSample}`, one per batch, e.g. from
[`sample_random_effect_draws`](@ref) or [`mcem_e_step`](@ref)).

`part=:q1` evaluates the full complete-data Q (reuses `_mcem_Q`); `part=:q2` evaluates the
RE-prior-only Q (reuses `_mcem_Q2`), so the returned value is bit-identical to the fit's
M-step Q at the same arguments. `free_names` selects the differentiated subset (defaults to
the `part`'s partition over all fixed effects; the complement is frozen at `θ`); `gradient`
is a `ComponentArray` on those free axes at `scale` (`:transformed` or `:untransformed`).

The per-batch form (`idx`) returns batch `idx`'s Q contribution; summing over `idx` equals
the population form (the per-subject seam for DP clipping / federation).
"""
function mcem_q_objective_and_gradient(
        dm::DataModel, θ::ComponentArray,
        draws::AbstractVector{<:RandomEffectPosteriorSample};
        part::Symbol = :q1,
        free_names::Union{Nothing, AbstractVector} = nothing,
        scale::Union{Symbol, AbstractString} = :transformed,
        constants_re::Union{NamedTuple, AbstractDict} = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        cache = nothing
    )
    part in (:q1, :q2) || error("mcem_q_objective_and_gradient: `part` must be :q1 or :q2, got :$part.")
    fnames = _mcem_resolve_free_names(dm, part, free_names)
    bi, cc, llc, sbb, wbb = _mcem_q_setup(dm, draws, constants_re, serialization, cache)
    qf = _MCEMQObjective(dm, bi, cc, llc, sbb, wbb, serialization, part)
    return _mcem_dev_sweep(qf, dm, θ, fnames, _as_symbol(scale))
end

function mcem_q_objective_and_gradient(
        dm::DataModel, θ::ComponentArray,
        draws::AbstractVector{<:RandomEffectPosteriorSample}, idx::Integer;
        part::Symbol = :q1,
        free_names::Union{Nothing, AbstractVector} = nothing,
        scale::Union{Symbol, AbstractString} = :transformed,
        constants_re::Union{NamedTuple, AbstractDict} = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleSerial(),
        cache = nothing
    )
    part in (:q1, :q2) || error("mcem_q_objective_and_gradient: `part` must be :q1 or :q2, got :$part.")
    fnames = _mcem_resolve_free_names(dm, part, free_names)
    bi, cc, llc, sbb, wbb = _mcem_q_setup(dm, draws, constants_re, serialization, cache)
    (1 <= idx <= length(bi)) ||
        error("mcem_q_objective_and_gradient: batch idx $idx out of range 1:$(length(bi)).")
    wbb_i = wbb === nothing ? nothing : wbb[idx:idx]
    qf = _MCEMQObjective(dm, bi[idx:idx], cc, llc, sbb[idx:idx], wbb_i, serialization, part)
    return _mcem_dev_sweep(qf, dm, θ, fnames, _as_symbol(scale))
end

# ── MCEM E-step primitive (state-threaded) ────────────────────────────────────
# One outer-iteration E-step, faithful to the MCEM fit loop's sampling: correct
# sampler dispatch (SaemixMH/Turing MCMC or importance sampling), `sample_schedule`,
# warm-start, and prior-mean first-iteration seeding. The per-batch draws are
# deterministic in the per-batch RNGs (independent of thread scheduling), so this
# serial replication is bit-identical to the fit's E-step at the same `θ`/`rng`.

const _MCEMLastParams = Union{Nothing, NamedTuple, AbstractVector, _AdaptiveMHState, _SaemixMHState}

# Fresh state (iter 1): build caches, spawn per-batch RNGs, seed warm-start from the
# prior mean exactly as the fit's pre-loop does (`_em_seed_batch_b`/`_b_to_last_params`).
function _mcem_estep_init(dm::DataModel, θ::ComponentArray, method::MCEM, rng::AbstractRNG, constants_re)
    re_names = get_re_names(get_random(get_model(dm)))
    isempty(re_names) && error("mcem_e_step requires random effects; MLE/MAP are for fixed-effects models.")
    re_types = get_re_types(get_random(get_model(dm)))
    constants_re = _normalize_constants_re(dm, _as_namedtuple(constants_re))
    const_cache = _build_constants_cache(dm, constants_re)
    _, batch_infos, _ = _build_re_batch_infos(dm, constants_re)
    ll_cache = build_ll_cache(dm; serialization = EnsembleSerial(), force_saveat = true)
    cache1 = ll_cache isa Vector ? ll_cache[1] : ll_cache
    nb = length(batch_infos)
    batch_rngs = _mcem_thread_rngs(rng, nb)
    last_params = Vector{_MCEMLastParams}(undef, nb)
    fill!(last_params, nothing)
    for bi in 1:nb
        info = batch_infos[bi]
        get_n_b(info) == 0 && continue
        b_init = _em_seed_batch_b(dm, info, θ, const_cache, cache1, rng, re_names, bi, "MCEM")
        last_params[bi] = _b_to_last_params(b_init, info, re_names)
    end
    proposal_blocks = method.e_step isa MCEM_IS ?
        [_is_init_proposal_blocks(dm, batch_infos[bi], θ, cache1, re_names, re_types) for bi in 1:nb] :
        nothing
    return (
        iter = 1, prev_use_mcmc = nothing, last_params = last_params, batch_rngs = batch_rngs,
        proposal_blocks = proposal_blocks, const_cache = const_cache, ll_cache = ll_cache,
        cache1 = cache1, batch_infos = batch_infos, re_names = re_names, re_types = re_types,
        samples_by_batch = Vector{Matrix{Float64}}(undef, nb),
        weights_store = Vector{Union{Nothing, Vector{Float64}}}(nothing, nb),
        ess_store = fill(NaN, nb), batches_buf = Int[],
    )
end

"""
    mcem_e_step(dm, θ, method::MCEM, state; rng=Random.default_rng(), constants_re=NamedTuple())
        -> (draws::Vector{RandomEffectPosteriorSample}, new_state)

Run ONE MCEM E-step at `θ`, exactly as one iteration of the MCEM fit loop does: it draws
`p(η_b | y_b, θ)` per batch with the method's sampler (`method.e_step`), honoring
`sample_schedule`, `update_schedule`, and warm-start. Pass `state === nothing` on the first
call (reproduces the fit's prior-mean seeding); thread the returned `new_state` into the
next call so warm-start / IS-proposal adaptation / per-batch RNGs persist across iterations.

Returns the per-batch posterior draws (feed directly to [`mcem_q_objective_and_gradient`](@ref))
and the updated state. Draws are deterministic in the per-batch RNGs, so a fixed `rng` on the
first call reproduces the fit's E-step draws.
"""
function mcem_e_step(
        dm::DataModel, θ::ComponentArray, method::MCEM, state;
        rng::AbstractRNG = Random.default_rng(),
        constants_re::Union{NamedTuple, AbstractDict} = NamedTuple()
    )
    st = state === nothing ? _mcem_estep_init(dm, θ, method, rng, constants_re) : state
    batch_infos = st.batch_infos
    nb = length(batch_infos)
    iter = st.iter
    use_mcmc = _use_mcmc_this_iter(iter, method.e_step)
    # Force a full refresh on iter 1 and on the MCMC-warmup -> IS switch (matches the fit).
    sched = st.prev_use_mcmc === use_mcmc ? method.update_schedule : :all
    updated = copy(_schedule_batches!(st.batches_buf, sched, nb, iter, rng))

    if use_mcmc
        mcmc_es = _mcmc_e_step(method.e_step)
        S = _mcem_schedule(mcmc_es.sample_schedule, iter)
        mcmc_es.sample_schedule === nothing && (S = get(mcmc_es.turing_kwargs, :n_samples, 100))
        S >= 1 || error("mcem_e_step: sample_schedule returned $S at iteration $iter; it must be ≥ 1.")
        tkwargs = merge(mcmc_es.turing_kwargs, (n_samples = S,))
        for bi in updated
            info = batch_infos[bi]
            samples, lastp, _ = _mcem_sample_batch(
                dm, info, θ, st.const_cache, st.cache1, mcmc_es.sampler, tkwargs,
                st.batch_rngs[bi], st.re_names, mcmc_es.warm_start, st.last_params[bi];
                outer_iter = iter
            )
            st.samples_by_batch[bi] = samples
            st.last_params[bi] = lastp
            st.weights_store[bi] = nothing
        end
        # Seed IS proposal blocks at the end of the MCMC warm-up (MCEM_IS.adapt).
        if method.e_step isa MCEM_IS && iter == method.e_step.warm_start_mcmc_iters && method.e_step.adapt
            for bi in 1:nb
                _is_update_blocks!(
                    st.proposal_blocks[bi], st.samples_by_batch[bi], batch_infos[bi],
                    st.re_names, st.re_types, 2, 1.0e-6
                )
            end
        end
    else
        is_es = method.e_step
        for bi in updated
            info = batch_infos[bi]
            samps, log_ws, ess = _is_sample_batch(
                dm, info, θ, st.const_cache, st.cache1, st.batch_rngs[bi],
                st.re_names, st.re_types, is_es, st.proposal_blocks[bi]
            )
            st.samples_by_batch[bi] = samps
            st.weights_store[bi] = log_ws
            st.ess_store[bi] = ess
            is_es.adapt && _is_update_blocks!(
                st.proposal_blocks[bi], samps, info, st.re_names, st.re_types, 2, 1.0e-6; log_ws = log_ws
            )
        end
    end

    draws = Vector{RandomEffectPosteriorSample}(undef, nb)
    for bi in 1:nb
        S_bi = st.samples_by_batch[bi]
        if use_mcmc
            draws[bi] = RandomEffectPosteriorSample(S_bi, nothing, nothing, :mcmc)
        else
            draws[bi] = RandomEffectPosteriorSample(S_bi, st.weights_store[bi], st.ess_store[bi], :importance)
        end
    end
    new_state = merge(st, (iter = iter + 1, prev_use_mcmc = use_mcmc))
    return draws, new_state
end

# ── SAEM closed-form M-step primitives (federation) ──────────────────────────
# The SAEM hybrid M-step splits free fixed effects into a closed-form-eligible block
# (Gaussian RE covariances/means, residual σ, supported HMM emission) updated from
# per-subject-additive sufficient statistics, and a numerical block optimized with the
# same Q kernels as MCEM (`mcem_q_objective_and_gradient`). These primitives expose the
# closed-form half, reusing the fit's `_saem_resolve_closed_form_config` /
# `_saem_builtin_collect_current_stats` / `_saem_builtin_smooth_stats` /
# `_saem_builtin_updates_from_smoothed_stats` verbatim so the routing and updates are
# bit-identical to `fit_model(dm, SAEM())`.

"""
    saem_closed_form_eligibility(dm; constants=NamedTuple(), method=SAEM()) -> (closed_form, numerical)

Partition the FREE (non-`constants`) fixed-effect names into the SAEM hybrid M-step's two
routes: `closed_form` (RE covariance/mean parameters, the residual σ, and supported HMM
emission parameters — resolved exactly as `fit_model(dm, SAEM())` does) and `numerical`
(everything else, e.g. structural/mean parameters entering the observation model, optimized
via [`mcem_q_objective_and_gradient`]). The federated driver routes each parameter accordingly.
"""
function saem_closed_form_eligibility(
        dm::DataModel;
        constants::Union{NamedTuple, AbstractDict} = NamedTuple(),
        method::SAEM = SAEM()
    )
    constants = _as_namedtuple(constants)
    fe = get_fixed(get_model(dm))
    free = [n for n in get_names(fe) if !(n in keys(constants))]
    cfg = _saem_resolve_closed_form_config(dm, method.saem)
    cf_syms = Symbol[]
    for v in values(cfg.re_cov_params)
        _saem_collect_target_symbols!(cf_syms, v)
    end
    for v in values(cfg.re_mean_params)
        _saem_collect_target_symbols!(cf_syms, v)
    end
    _saem_collect_target_symbols!(cf_syms, cfg.resid_var_param)
    for col in keys(cfg.hmm_emission_params)
        info = getfield(cfg.hmm_emission_params, col)
        hasproperty(info, :target) &&
            _saem_collect_target_symbols!(cf_syms, getproperty(info, :target))
    end
    cf_set = cfg.builtin_stats_mode == :none ? Set{Symbol}() : Set(cf_syms)
    closed_form = [n for n in free if n in cf_set]
    numerical = [n for n in free if !(n in cf_set)]
    return (closed_form = closed_form, numerical = numerical)
end

# draws (Matrix{Float64} per batch, n_b × S) -> the b_chains[bi][c] / n_chains shape the
# fit's `_saem_builtin_collect_current_stats` expects. A shared sample schedule gives every
# batch the same S; guard against a short batch just in case.
function _saem_draws_to_chains(draws::AbstractVector{<:RandomEffectPosteriorSample})
    nb = length(draws)
    S = 0
    for d in draws
        c = size(get_draws(d), 2)
        c > 0 && (S = S == 0 ? c : min(S, c))
    end
    S == 0 && error("saem_sufficient_statistics: draws contain no samples.")
    b_chains = Vector{Vector{Vector{Float64}}}(undef, nb)
    for bi in 1:nb
        m = get_draws(draws[bi])
        b_chains[bi] = [Float64.(@view m[:, c]) for c in 1:min(S, size(m, 2))]
        for _ in (size(m, 2) + 1):S
            push!(b_chains[bi], Float64[])
        end
    end
    return b_chains, S
end

function _saem_stats_setup(dm::DataModel, draws, constants_re, cache)
    constants_re = _as_namedtuple(constants_re)
    _, batch_infos, const_cache = build_re_batch_infos(dm, constants_re)
    length(draws) == length(batch_infos) ||
        error("saem_sufficient_statistics: got $(length(draws)) draw batches but the model has $(length(batch_infos)); draws must come from the same dm/constants_re.")
    ll_cache = cache === nothing ?
        build_ll_cache(dm; serialization = EnsembleSerial(), force_saveat = true) : cache
    llc = ll_cache isa Vector ? ll_cache[1] : ll_cache
    return batch_infos, const_cache, llc
end

function _saem_stats_collect(dm::DataModel, batch_infos, b_chains, n_chains, θ, const_cache, llc, method::SAEM, rng)
    cfg = _saem_resolve_closed_form_config(dm, method.saem)
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    return _saem_builtin_collect_current_stats(
        dm, batch_infos, b_chains, n_chains, θ_re, const_cache,
        cfg.resid_var_param, cfg.hmm_emission_params,
        cfg.re_cov_params, cfg.re_mean_params, cfg.re_family_map, llc, rng
    )
end

"""
    saem_sufficient_statistics(dm, θ, draws; constants_re=NamedTuple(), method=SAEM(), cache=nothing, rng=Random.default_rng()) -> stats
    saem_sufficient_statistics(dm, θ, draws, idx; ...) -> stats

Per-subject-additive SAEM sufficient statistics at natural-scale `θ` and FIXED posterior
`draws` (`Vector{RandomEffectPosteriorSample}`, one per batch, e.g. from [`mcem_e_step`]).
Reuses the fit's `_saem_builtin_collect_current_stats`, so the population form (all batches)
is bit-identical to the SAEM fit's per-iteration current statistics and plugs straight into
[`saem_closed_form_mstep`].

`stats` is a NamedTuple `(re, outcome, hmm)`:
- `re[name] = (family, mean, second, n)`: RE moments, `mean = Σx/n`, `second = Σxx'/n` over
  the `n` draw contributions (`x` is η; `log η` for lognormal families; the ALR transform
  for `MvLogitNormal`).
- `outcome[col] = (family, s1, s2, ss, n)`: additive residual sufficient statistics.
- `hmm[col] = (family, target, sum_w, sum_wy)`: additive HMM emission statistics.

Federated aggregation: `outcome`/`hmm` fields are plain sums (add directly across sites);
`re` moments are additive after de-normalizing (`Σx = mean*n`, `Σxx' = second*n`), summing,
then re-dividing by the pooled `n`. The `idx` form returns batch `idx`'s contribution; summed
this way over batches it equals the population form (the per-subject seam for DP / federation).
Draws are treated as equally-weighted MCMC draws, matching the SAEM MH sampler.
"""
function saem_sufficient_statistics(
        dm::DataModel, θ::ComponentArray,
        draws::AbstractVector{<:RandomEffectPosteriorSample};
        constants_re::Union{NamedTuple, AbstractDict} = NamedTuple(),
        method::SAEM = SAEM(), cache = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    batch_infos, const_cache, llc = _saem_stats_setup(dm, draws, constants_re, cache)
    b_chains, n_chains = _saem_draws_to_chains(draws)
    return _saem_stats_collect(dm, batch_infos, b_chains, n_chains, θ, const_cache, llc, method, rng)
end

function saem_sufficient_statistics(
        dm::DataModel, θ::ComponentArray,
        draws::AbstractVector{<:RandomEffectPosteriorSample}, idx::Integer;
        constants_re::Union{NamedTuple, AbstractDict} = NamedTuple(),
        method::SAEM = SAEM(), cache = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    batch_infos, const_cache, llc = _saem_stats_setup(dm, draws, constants_re, cache)
    (1 <= idx <= length(batch_infos)) ||
        error("saem_sufficient_statistics: batch idx $idx out of range 1:$(length(batch_infos)).")
    b_chains, n_chains = _saem_draws_to_chains(draws)
    return _saem_stats_collect(
        dm, batch_infos[idx:idx], b_chains[idx:idx], n_chains, θ, const_cache, llc, method, rng
    )
end

"""
    saem_closed_form_mstep(dm, aggregated_stats, smoothed_state, θ, γ; constants=NamedTuple(), method=SAEM()) -> (θ_updates, new_smoothed_state)

One coordinator-side SAEM closed-form M-step, bit-identical to the SAEM fit's closed-form
update. Stochastic-approximation-smooths `aggregated_stats` (the summed per-site population
sufficient statistics from [`saem_sufficient_statistics`]) against the carried
`smoothed_state` with step size `γ`, then closed-form updates the eligible parameters.

Pass `smoothed_state === nothing` on the first iteration (the SA state initializes to the
current stats); thread the returned `new_smoothed_state` into the next call. `θ_updates` is a
NamedTuple of natural-scale values for the closed-form-eligible parameters (keys in
`constants` are dropped — user constants win, as in the fit). Reuses `_saem_builtin_smooth_stats`
and `_saem_builtin_updates_from_smoothed_stats` verbatim; annealing/SA floors are the
federated driver's responsibility (control them through `γ`).
"""
function saem_closed_form_mstep(
        dm::DataModel, aggregated_stats, smoothed_state,
        θ::ComponentArray, γ::Real;
        constants::Union{NamedTuple, AbstractDict} = NamedTuple(),
        method::SAEM = SAEM()
    )
    constants = _as_namedtuple(constants)
    cfg = _saem_resolve_closed_form_config(dm, method.saem)
    new_state = _saem_builtin_smooth_stats(smoothed_state, aggregated_stats, γ)
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    updates = _saem_builtin_updates_from_smoothed_stats(
        dm, θ_re, new_state, cfg.resid_var_param, cfg.hmm_emission_params,
        cfg.re_cov_params, cfg.re_mean_params
    )
    for k in keys(constants)
        haskey(updates, k) &&
            (updates = Base.structdiff(updates, NamedTuple{(k,)}((nothing,))))
    end
    return (θ_updates = updates, new_smoothed_state = new_state)
end

# ── Objective factory: shared fit setup/teardown ─────────────────────────────

"""
    NLFreeLayout

Per-fit parameter bookkeeping shared by the optimization drivers: the free (non-constant)
fixed-effect names, the transform/inverse-transform, the constants-applied transformed vector,
and the free↔full index map. Build with [`free_parameter_layout`](@ref); consume with
[`resolve_fitted_parameters`](@ref) and the transformed-scale objective helpers.
"""
struct NLFreeLayout{FN, TR, IT, TC, V, AF, F0, AX}
    free_names::FN
    transform::TR
    inv_transform::IT
    θ_const_t::TC
    θ_const_t_vec::V
    axs_full::AF
    θ0_free_t::F0
    free_idx::Vector{Int}
    axs::AX
end

"""
    free_parameter_layout(fe::FixedEffects; constants=NamedTuple(), theta0_untransformed=nothing) -> NLFreeLayout

Resolve the free fixed effects (those not in `constants`), the transform pair, the
constants-applied transformed vector, the free parameters' initial transformed values, and the
free→full index map. `theta0_untransformed` overrides the model's initial natural-scale values.
"""
function free_parameter_layout(
        fe::FixedEffects; constants::Union{NamedTuple, AbstractDict} = NamedTuple(),
        theta0_untransformed = nothing
    )
    constants = _as_namedtuple(constants)
    fixed_names = get_names(fe)
    free_names = [n for n in fixed_names if !(n in keys(constants))]
    θ0_u = get_θ0_untransformed(fe)
    if theta0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta0_untransformed, n) ||
                error("theta_0_untransformed is missing parameter $(n).")
        end
        θ0_u = theta0_untransformed
    end
    transform = get_transform(fe)
    inv_transform = get_inverse_transform(fe)
    θ0_t = transform(θ0_u)
    θ_const_u = deepcopy(θ0_u)
    apply_constants!(θ_const_u, constants)
    θ_const_t = transform(θ_const_u)
    θ0_free_t = ComponentArray(
        NamedTuple{Tuple(free_names)}(
            Tuple(getproperty(θ0_t, n) for n in free_names)
        )
    )
    return NLFreeLayout(
        free_names, transform, inv_transform, θ_const_t,
        collect(θ_const_t), getaxes(θ_const_t), θ0_free_t,
        free_parameter_indices(θ_const_t, θ0_free_t), getaxes(θ0_free_t)
    )
end

"""
    resolve_fitted_parameters(layout::NLFreeLayout, θ_hat_free_t) -> FitParameters

Overlay the optimizer's free-parameter solution onto the constants and return the fitted
`FitParameters` (transformed + natural scale).
"""
function resolve_fitted_parameters(layout::NLFreeLayout, θ_hat_free_t)
    θ_hat_t_free = θ_hat_free_t isa ComponentArray ? θ_hat_free_t :
        ComponentArray(θ_hat_free_t, layout.axs)
    T = eltype(θ_hat_t_free)
    θ_hat_t = ComponentArray(T.(layout.θ_const_t), layout.axs_full)
    for name in layout.free_names
        setproperty!(θ_hat_t, name, getproperty(θ_hat_t_free, name))
    end
    return FitParameters(θ_hat_t, layout.inv_transform(θ_hat_t))
end

export NLFreeLayout, free_parameter_layout, resolve_fitted_parameters, build_fit_result

# ── Change-of-variables (unconstrained ↔ natural) ────────────────────────────

"""
    logabsdetjac(it::InverseTransform, θt) -> Real

Log absolute determinant of the Jacobian of the inverse parameter transform (unconstrained →
natural) at the transformed point `θt` - the change-of-variables correction for a density
placed on the unconstrained optimizer scale. Summed block-by-block over fixed effects: scalar
scales (`:identity`/`:log`/`:logit`/`:elementwise`) use their closed form, and the structured
matrix/simplex scales (`:cholesky`/`:expm`/`:stickbreak`/`:stickbreakrows`/`:lograterows`/`:lie`)
differentiate their minimal square inverse map. ForwardDiff-safe, including nested AD.

For `:lie` with fixed eigenvalues the correction is chart-dependent (the fixed-eigenvalue
submanifold is not axis-aligned); it is self-consistent for use as a change-of-variables term
in a single optimization, but is not a chart-invariant number.
"""
function logabsdetjac(it::InverseTransform, θt::ComponentArray)
    v = ComponentArrays.getdata(θt)
    isempty(v) && return zero(eltype(v))
    total = zero(eltype(v))
    for spec in it.specs
        total += _block_logabsdetjac(spec, getproperty(θt, spec.name))
    end
    return total
end

# ── FitContext: the convenience tier over the explicit primitives ────────────────
# Thin covers only - every context method forwards to the cache-explicit primitive
# above with the context's stored caches, so results are identical and the explicit
# API remains the full-control path.

export FitContext, build_fit_context, initial_parameters, get_batch_infos,
    optimize_parameters

"""
    FitContext

Reusable workspace for writing a custom estimator without threading caches by hand. Holds the
`DataModel`, the random-effect batch descriptors, the constant-RE cache, and one likelihood
evaluation cache. Build it once per fit with [`build_fit_context`](@ref); every context method
forwards to the corresponding cache-explicit primitive with these stored objects, so results
are identical to the explicit calls.
"""
struct FitContext{D, B, C, K}
    dm::D
    batch_infos::B
    const_cache::C
    cache::K
    constants_re::NamedTuple
end

"""
    build_fit_context(dm; constants_re=NamedTuple(), ode_args=(), ode_kwargs=NamedTuple())
        -> FitContext

Build the workspace a custom estimator iterates on. This performs, once, the setup every
hand-written fitting loop needs:

  - `build_re_batch_infos(dm, constants_re)` - the random-effect batch descriptors and the
    cache of levels fixed through `constants_re`;
  - `build_likelihood_cache(dm; force_saveat=true)` - the solver/template cache the density
    primitives reuse instead of rebuilding state on every call.

The context is θ-independent: build it once per fit and reuse it across all iterations (the
population primitives called with a bare `dm`, e.g. `empirical_bayes(dm, θ)`, rebuild these
caches on every call - inside a loop, prefer the context forms). Rebuild the context only when
`dm` or `constants_re` change. It does not store parameters; θ flows through every call.

With a context, the primitives lose their cache arguments and address batches by index:

    ctx = build_fit_context(dm)
    θ   = initial_parameters(ctx)
    complete_data_loglikelihood(ctx, bi, θ, b)          # == complete_data_loglikelihood(dm, batches[bi], θ, b;
                                                #      const_cache=cc, cache=cache)
    empirical_bayes(ctx, θ)                     # per-batch modes b* (no Hessian computed)
    empirical_bayes_covariance(ctx, θ, modes)   # per-batch Σ = (−H)⁻¹ at the modes
    laplace_marginal(ctx, θ)                    # marginal log-likelihood at θ
    laplace_marginal_gradient(ctx, θ)           # value + analytic θ-gradient of the marginal
    objective_and_gradient(Laplace(), ctx, θ)   # method-dispatched value + gradient
    sample_random_effect_draws(ctx, θ)          # posterior draws per batch
    θ̂, sol = optimize_parameters(f, ctx)        # natural-scale objective, handled transforms
    build_fit_result(ctx, method, θ̂; kind=:frequentist_re, objective=...)  # eb_modes=:auto

The evaluation cache is single-threaded (`ponytail:` serial cache; pass the explicit primitives
your own per-thread caches when parallelising a custom loop).
"""
function build_fit_context(
        dm::DataModel;
        constants_re::Union{NamedTuple, AbstractDict} = NamedTuple(),
        ode_args::Tuple = (), ode_kwargs::Union{NamedTuple, AbstractDict} = NamedTuple()
    )
    constants_re = _as_namedtuple(constants_re)
    ode_kwargs = _as_namedtuple(ode_kwargs)
    _, infos, cc = build_re_batch_infos(dm, constants_re)
    cache = build_likelihood_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        force_saveat = true
    )
    return FitContext(dm, infos, cc, cache, constants_re)
end

"""
    get_batch_infos(ctx::FitContext) -> Vector{REBatchInfo}

The context's random-effect batch descriptors; batch indices `bi` passed to the context
primitives index into this vector.
"""
@inline get_batch_infos(ctx::FitContext) = ctx.batch_infos

@inline get_data_model(ctx::FitContext) = ctx.dm
@inline _dev_dm(ctx::FitContext) = ctx.dm

# Everything the context froze at build time. Accepting these per call would silently diverge
# from the cached state, so the context methods reject them.
const _CTX_FIXED_KWARGS = (
    :constants_re, :ode_args, :ode_kwargs, :const_cache, :cache, :serialization,
)

function _ctx_check_kwargs(fname::Symbol, kwargs)
    for k in keys(kwargs)
        k in _CTX_FIXED_KWARGS && error(
            "$fname(ctx, ...): `$k` is fixed by the FitContext and cannot be passed per call. " *
                "Set constants_re/ode_args/ode_kwargs in build_fit_context, or use the explicit " *
                "`dm` form for per-call control."
        )
    end
    return nothing
end

"""
    initial_parameters(ctx::FitContext) -> ComponentArray

A fresh copy of the model's natural-scale initial fixed effects - the conventional starting
point of a fitting loop (replace it with `theta_0_untransformed` when the caller supplies one).
"""
initial_parameters(ctx::FitContext) = copy(get_θ0_untransformed(get_fixed(get_model(ctx.dm))))

# Batch-index covers over the density primitives.
for f in (
        :conditional_loglikelihood, :re_logprior, :complete_data_loglikelihood,
        :complete_data_loglikelihood_gradient, :complete_data_loglikelihood_hessian,
    )
    @eval @inline function $f(ctx::FitContext, bi::Integer, θ::ComponentArray, b; kwargs...)
        return $f(
            ctx.dm, ctx.batch_infos[bi], θ, b;
            const_cache = ctx.const_cache, cache = ctx.cache, kwargs...
        )
    end
end

@inline function ghq_marginal(ctx::FitContext, bi::Integer, θ::ComponentArray; level = 3)
    return ghq_marginal(
        ctx.dm, θ, ctx.batch_infos[bi]; level = level,
        const_cache = ctx.const_cache, cache = ctx.cache
    )
end

function ghq_marginal(ctx::FitContext, θ::ComponentArray; level = 3)
    isempty(ctx.batch_infos) && return zero(eltype(θ))
    return sum(
        ghq_marginal(ctx, bi, θ; level = level)
            for bi in eachindex(ctx.batch_infos)
    )
end

# Population forms reusing the context caches (the bare-`dm` forms rebuild them per call).
function empirical_bayes(
        ctx::FitContext, θ::ComponentArray;
        ebe_options::EBEOptions = EBEOptions(), rescue = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(ctx.dm)))
    bstars, _ = _compute_bstars(
        ctx.dm, θ_re, ctx.constants_re, ctx.cache, ebe_options, rng;
        rescue = rescue
    )
    return bstars
end

@inline function empirical_bayes_covariance(
        ctx::FitContext, bi::Integer, θ::ComponentArray, b_star; kwargs...
    )
    return empirical_bayes_covariance(
        ctx.dm, θ, ctx.batch_infos[bi], b_star;
        const_cache = ctx.const_cache, cache = ctx.cache, kwargs...
    )
end

function empirical_bayes_covariance(
        ctx::FitContext, θ::ComponentArray, bstars::AbstractVector; kwargs...
    )
    length(bstars) == length(ctx.batch_infos) ||
        error("empirical_bayes_covariance: got $(length(bstars)) modes for $(length(ctx.batch_infos)) batches.")
    return [
        empirical_bayes_covariance(ctx, bi, θ, bstars[bi]; kwargs...)
            for bi in eachindex(ctx.batch_infos)
    ]
end

function laplace_marginal(
        ctx::FitContext, θ::ComponentArray;
        curvature::AbstractCurvature = ExactHessianCurvature(),
        ebe_options::EBEOptions = EBEOptions(), rescue = nothing,
        rng::AbstractRNG = Random.default_rng(), kwargs...
    )
    isempty(ctx.batch_infos) && return zero(eltype(θ))
    bstars = empirical_bayes(ctx, θ; ebe_options = ebe_options, rescue = rescue, rng = rng)
    return sum(
        laplace_marginal(
                ctx.dm, θ, ctx.batch_infos[bi], bstars[bi];
                const_cache = ctx.const_cache, cache = ctx.cache,
                curvature = curvature, kwargs...
            )
            for bi in eachindex(ctx.batch_infos)
    )
end

# Batch-index cover, mirroring `ghq_marginal(ctx, bi, θ)`.
@inline function laplace_marginal(
        ctx::FitContext, bi::Integer, θ::ComponentArray, b_star; kwargs...
    )
    return laplace_marginal(
        ctx.dm, θ, ctx.batch_infos[bi], b_star;
        const_cache = ctx.const_cache, cache = ctx.cache, kwargs...
    )
end

function laplace_marginal_gradient(
        ctx::FitContext, θ::ComponentArray;
        scale::Union{Symbol, AbstractString} = :untransformed,
        curvature::AbstractCurvature = ExactHessianCurvature(),
        ebe_options::EBEOptions = EBEOptions(), rescue = nothing,
        rng::AbstractRNG = Random.default_rng(), kwargs...
    )
    _ctx_check_kwargs(:laplace_marginal_gradient, kwargs)
    scale = _dev_check_scale(scale)
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(ctx.dm)))
    value = zero(eltype(θ_re))
    grad = ComponentArray(zeros(eltype(θ_re), length(θ_re)), getaxes(θ_re))
    isempty(ctx.batch_infos) && return (value, _dev_gradient_scale(ctx.dm, θ_re, grad, scale))
    bstars = empirical_bayes(ctx, θ_re; ebe_options = ebe_options, rescue = rescue, rng = rng)
    for bi in eachindex(ctx.batch_infos)
        v, g = laplace_marginal_gradient(
            ctx.dm, θ_re, ctx.batch_infos[bi], bstars[bi];
            const_cache = ctx.const_cache, cache = ctx.cache, scale = :untransformed,
            curvature = curvature, rng = rng, kwargs...
        )
        value += v
        grad .+= g
    end
    return (value, _dev_gradient_scale(ctx.dm, θ_re, grad, scale))
end

# Context forms of the joint value+gradient primitive: same results, caches reused.
function objective_and_gradient(
        method::Union{Laplace, FOCEI}, ctx::FitContext, θ::ComponentArray; kwargs...
    )
    _ctx_check_kwargs(:objective_and_gradient, kwargs)
    return _dev_laplace_og(method, ctx, θ; kwargs...)
end

function objective_and_gradient(
        method::GHQuadrature, ctx::FitContext, θ::ComponentArray;
        scale::Union{Symbol, AbstractString} = :untransformed, hessian::Bool = false,
        include_penalty::Bool = false,
        penalty::Union{NamedTuple, AbstractDict} = NamedTuple(),
        level = method.level, kwargs...
    )
    _ctx_check_kwargs(:objective_and_gradient, kwargs)
    scale = _dev_check_scale(scale)
    f = _DevGHQObjective(
        ctx.dm, ctx.batch_infos, ctx.const_cache, ctx.cache, level,
        _as_namedtuple(penalty), include_penalty
    )
    return _dev_sweep(f, ctx.dm, θ, scale, hessian)
end

function objective_and_gradient(
        method::Union{MLE, MAP}, ctx::FitContext, θ::ComponentArray;
        scale::Union{Symbol, AbstractString} = :untransformed, hessian::Bool = false,
        include_penalty::Bool = false,
        penalty::Union{NamedTuple, AbstractDict} = NamedTuple(), kwargs...
    )
    _ctx_check_kwargs(:objective_and_gradient, kwargs)
    scale = _dev_check_scale(scale)
    _require_no_random_effects(ctx.dm)
    f = _DevNoREObjective(
        ctx.dm, ctx.cache, EnsembleSerial(), _as_namedtuple(penalty), include_penalty,
        method isa MAP ? get_fixed(get_model(ctx.dm)) : nothing
    )
    return _dev_sweep(f, ctx.dm, θ, scale, hessian)
end

function objective_and_gradient(
        ::MLE, ctx::FitContext, θ::ComponentArray, idx::Integer;
        scale::Union{Symbol, AbstractString} = :untransformed, hessian::Bool = false
    )
    scale = _dev_check_scale(scale)
    _require_no_random_effects(ctx.dm)
    f = _DevIndividualObjective(ctx.dm, Int(idx), ctx.cache)
    return _dev_sweep(f, ctx.dm, θ, scale, hessian)
end

function objective_and_gradient(
        method::Pooled, ctx::FitContext, θ::ComponentArray;
        scale::Union{Symbol, AbstractString} = :untransformed, hessian::Bool = false,
        include_penalty::Bool = false,
        penalty::Union{NamedTuple, AbstractDict} = NamedTuple(),
        strategies = nothing, eta = nothing, rng::AbstractRNG = Random.default_rng(),
        kwargs...
    )
    _ctx_check_kwargs(:objective_and_gradient, kwargs)
    scale = _dev_check_scale(scale)
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(ctx.dm)))
    strat = strategies === nothing ?
        _dev_pooled_strategies(ctx.dm, θ_re, method, rng) : strategies
    f = _DevPooledObjective(
        ctx.dm, ctx.cache, EnsembleSerial(), strat, eta, _as_namedtuple(penalty),
        include_penalty
    )
    return _dev_sweep(f, ctx.dm, θ, scale, hessian)
end

function sample_random_effect_draws(
        ctx::FitContext, θ::ComponentArray;
        method::Symbol = :importance, sampler = nothing, n_samples::Int = 100,
        n_adapt::Int = 50, ebe_options::EBEOptions = EBEOptions(), rescue = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    if method === :mcmc
        return [
            sample_random_effect_draws(
                    ctx.dm, θ, ctx.batch_infos[bi], eltype(θ)[]; method = :mcmc,
                    sampler = sampler, n_samples = n_samples, n_adapt = n_adapt,
                    const_cache = ctx.const_cache, cache = ctx.cache, rng = rng
                )
                for bi in eachindex(ctx.batch_infos)
        ]
    end
    bstars = empirical_bayes(ctx, θ; ebe_options = ebe_options, rescue = rescue, rng = rng)
    return [
        sample_random_effect_draws(
                ctx.dm, θ, ctx.batch_infos[bi], bstars[bi]; method = method,
                n_samples = n_samples, const_cache = ctx.const_cache, cache = ctx.cache,
                rng = rng
            ) for bi in eachindex(ctx.batch_infos)
    ]
end

"""
    optimize_parameters(f_natural, ctx::FitContext;
                        θ_start=initial_parameters(ctx),
                        optimizer=LBFGS(linesearch=BackTracking(maxstep=1.0)),
                        adtype=AutoForwardDiff(), optim_kwargs=NamedTuple())
        -> (θ̂::ComponentArray, sol)

Minimise an objective written purely in **natural-scale** parameters. `f_natural(θ)` receives a
symmetrised natural-scale `ComponentArray` and returns the value to minimise (a negative
log-likelihood, negative Q-function, ...). The unconstrained-scale round trip - transform,
`ComponentArray` reassembly, PSD symmetrisation, and the back-transform of the optimum - is
handled here, so bounded parameters (`scale=:log`, `:logit`, matrix scales) need no attention
in `f_natural`. Do-block friendly:

    θ̂, sol = optimize_parameters(ctx; θ_start=θ) do θn
        -sum(complete_data_loglikelihood(ctx, bi, θn, modes[bi]) for bi in eachindex(get_batch_infos(ctx)))
    end

The returned `sol` carries the optimizer's verdict: pass
`converged = SciMLBase.successful_retcode(sol)` to [`build_fit_result`](@ref) instead of
letting it default to `missing`.

`ponytail:` optimizes all fixed effects; apply `constants`/bounds via the explicit
`free_parameter_layout`/`resolve_optimizer_bounds` path when needed.
"""
function optimize_parameters(
        f_natural, ctx::FitContext;
        θ_start::ComponentArray = initial_parameters(ctx),
        optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0)),
        adtype = Optimization.AutoForwardDiff(),
        optim_kwargs::Union{NamedTuple, AbstractDict} = NamedTuple()
    )
    optim_kwargs = _as_namedtuple(optim_kwargs)
    dm = ctx.dm
    fe = get_fixed(get_model(dm))
    inv_transform = get_inverse_transform(fe)
    θt0 = get_transform(fe)(θ_start)
    axs = getaxes(θt0)
    obj = (θt_vec, _) -> f_natural(
        symmetrize_psd_parameters(
            dm, inv_transform(ComponentArray(θt_vec, axs))
        )
    )
    prob = OptimizationProblem(OptimizationFunction(obj, adtype), collect(θt0))
    sol = Optimization.solve(prob, optimizer; optim_kwargs...)
    θ̂ = symmetrize_psd_parameters(dm, inv_transform(ComponentArray(sol.u, axs)))
    return θ̂, sol
end

"""
    build_fit_result(ctx::FitContext, method, θ; kind=:frequentist, objective,
                     eb_modes=:auto, kwargs...) -> FitResult

Context form of [`build_fit_result`](@ref). `eb_modes = :auto` computes the per-batch
empirical-Bayes modes via `empirical_bayes(ctx, θ)` for random-effect kinds (and stores
`nothing` for fixed-effects kinds), so the common case needs no extra call.
"""
function build_fit_result(
        ctx::FitContext, method::FittingMethod, θ::ComponentArray;
        kind::Symbol = :frequentist, eb_modes = :auto, kwargs...
    )
    modes = eb_modes === :auto ?
        (
            kind in (:frequentist_re, :ghquadrature, :saem, :mcem) ?
            empirical_bayes(ctx, θ) : nothing
        ) : eb_modes
    return build_fit_result(ctx.dm, method, θ; kind = kind, eb_modes = modes, kwargs...)
end
