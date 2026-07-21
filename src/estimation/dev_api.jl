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
export empirical_bayes, posterior_moments, laplace_marginal, ghq_marginal,
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
export fit_method, fit_fixed_effects, fit_laplace_family

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
function solve_individual(dm::DataModel, idx::Integer, θ::ComponentArray, η;
        cache = nothing, dense::Bool = false)
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
    dm, θ, η; kwargs...)

function conditional_loglikelihood(dm::DataModel, idx::Integer, θ::ComponentArray, η;
        cache = nothing)
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    η_ca = η isa NamedTuple ? ComponentArray(η) : η
    return _loglikelihood_individual(dm, Int(idx), θ_re, η_ca, _dev_ll_cache(dm, cache))
end

function conditional_loglikelihood(dm::DataModel, batch::REBatchInfo, θ::ComponentArray, b;
        const_cache::REConstantsCache, cache = nothing)
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
function re_logprior(dm::DataModel, batch::REBatchInfo, θ::ComponentArray, b;
        const_cache::REConstantsCache, cache = nothing,
        anneal_sds::NamedTuple = NamedTuple())
    return _re_logpdf_batch(
        dm, batch, θ, b, const_cache, _dev_ll_cache(dm, cache); anneal_sds = anneal_sds)
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
        anneal_sds::NamedTuple = NamedTuple(), tctx = nothing)
    return _laplace_logf_batch(dm, batch, θ, b, const_cache, _dev_ll_cache(dm, cache);
        anneal_sds = anneal_sds, tctx = tctx)
end

function complete_data_loglikelihood(
        dm::DataModel, idx::Integer, θ::ComponentArray, η; cache = nothing)
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
        const_cache::REConstantsCache, cache = nothing)
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
        curvature::AbstractCurvature = ExactHessianCurvature())
    return inner_curvature(curvature, dm, batch, θ, b, const_cache,
        _dev_ll_cache(dm, cache), CurvatureWorkspace())
end

"""
    hmm_filter_step!(hmm_priors::Dict{Symbol,Any}, outcome::Symbol, dist, y) -> Distribution

One HMM forward-filter step: condition `dist` on the running `hmm_priors[outcome]`, then
update that entry with the posterior after observing `y` (or the predicted hidden-state
distribution when `y === missing`). Non-HMM distributions pass through unchanged and leave
`hmm_priors` untouched.
"""
hmm_filter_step!(hmm_priors::Dict{Symbol, Any}, outcome::Symbol, dist, y) = _apply_hmm_filter!(
    hmm_priors, outcome, dist, y)

"""
    obs_distributions(dm, idx, θ, η; cache=nothing, sol_accessors=nothing, hmm_filter=true) -> Vector{<:NamedTuple}

Per-observation-row predicted distributions for individual `idx` at natural-scale `(θ, η)`:
one `NamedTuple{outcome => Distribution}` per row of `get_obs_rows(dm)[idx]`. Solves the ODE
once (or reuses a passed `sol_accessors`). With `hmm_filter=true` (default) HMM outcomes are
forward-filtered in sequence, matching `plot_fits`/`build_plot_cache`. Returns an empty vector
on solve failure.
"""
function obs_distributions(dm::DataModel, idx::Integer, θ::ComponentArray, η;
        cache = nothing, sol_accessors = nothing, hmm_filter::Bool = true)
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
            hmm_filter && (dist = _apply_hmm_filter!(hmm_priors, col, dist,
                getfield(obs_series, col)[i]))
            push!(prs, col => dist)
        end
        out[i] = NamedTuple(prs)
    end
    return out
end

# ── Posterior / empirical Bayes / marginal ───────────────────────────────────

# Modes + aligned batch structure at natural-scale θ. Batch order matches a fresh
# build_re_batch_infos(dm, constants_re), so bstars[bi] pairs with infos[bi]/cc.
function _empirical_bayes_batches(dm::DataModel, θ::ComponentArray;
        constants_re::NamedTuple = NamedTuple(), ebe_options::EBEOptions = EBEOptions(),
        rescue = nothing, ode_args::Tuple = (), ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Random.default_rng())
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    cache = build_likelihood_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization)
    _, infos, cc = build_re_batch_infos(dm, constants_re)
    bstars, _ = _compute_bstars(dm, θ_re, constants_re, cache, ebe_options, rng;
        rescue = rescue)
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
    dm, θ; kwargs...)[1]

function empirical_bayes(dm::DataModel, θ::ComponentArray, idx::Integer; kwargs...)
    bstars, infos, cc, θ_re, _ = _empirical_bayes_batches(dm, θ; kwargs...)
    return eta_from_modes(dm, infos, bstars, cc, θ_re)[Int(idx)]
end

"""
    posterior_moments(dm, θ, batch, b_star; const_cache, cache=nothing, jitter=1e-6, max_tries=6, adaptive=false, scale_factor=0.0) -> (b_star, Σ)
    posterior_moments(dm, θ; kwargs...) -> Vector

Posterior mode and Laplace covariance `Σ = (−H)⁻¹` (natural b-space) of the random effects.
The batch form takes a mode `b_star` (e.g. from [`empirical_bayes`](@ref)); the population form
finds the modes first and returns one `(b_star, Σ)` per batch. `Σ` is `nothing` when `−H` is not
positive definite after jitter.
"""
function posterior_moments(dm::DataModel, θ::ComponentArray, batch::REBatchInfo, b_star;
        const_cache::REConstantsCache, cache = nothing,
        curvature::AbstractCurvature = ExactHessianCurvature(), jitter = 1e-6,
        max_tries::Int = 6, adaptive::Bool = false, scale_factor = 0.0)
    c = _dev_ll_cache(dm, cache)
    _, _, chol = _laplace_logdet_negH(dm, batch, θ, b_star, const_cache, c, nothing, 1;
        jitter = jitter, max_tries = max_tries, adaptive = adaptive,
        scale_factor = scale_factor, hmode = curvature)
    (chol === nothing || chol.info != 0) && return (b_star, nothing)
    return (b_star, Matrix(inv(chol)))
end

function posterior_moments(dm::DataModel, θ::ComponentArray;
        constants_re::NamedTuple = NamedTuple(),
        curvature::AbstractCurvature = ExactHessianCurvature(), jitter = 1e-6,
        max_tries::Int = 6, adaptive::Bool = false, scale_factor = 0.0, kwargs...)
    bstars, infos, cc, θ_re, cache = _empirical_bayes_batches(
        dm, θ; constants_re = constants_re, kwargs...)
    return [posterior_moments(dm, θ_re, infos[bi], bstars[bi]; const_cache = cc,
                cache = cache, curvature = curvature, jitter = jitter,
                max_tries = max_tries, adaptive = adaptive, scale_factor = scale_factor)
            for bi in eachindex(infos)]
end

"""
    laplace_marginal(dm, θ, batch, b_star; const_cache, cache=nothing, jitter=1e-6, max_tries=6, adaptive=false, scale_factor=0.0) -> Real
    laplace_marginal(dm, θ; kwargs...) -> Float64

Laplace-approximate marginal log-likelihood
`log p(y | θ) ≈ log f(b*) + ½·n_b·log(2π) − ½·log det(−H)`. The batch form uses a supplied mode
`b_star`; the population form finds the modes and sums over batches. For a θ-gradient of the
marginal use the Laplace fit's analytic (envelope + trace-estimator) gradient; naive AD through
the recomputed mode is not supported.
"""
function laplace_marginal(dm::DataModel, θ::ComponentArray, batch::REBatchInfo, b_star;
        const_cache::REConstantsCache, cache = nothing,
        curvature::AbstractCurvature = ExactHessianCurvature(), jitter = 1e-6,
        max_tries::Int = 6, adaptive::Bool = false, scale_factor = 0.0)
    c = _dev_ll_cache(dm, cache)
    logf = _laplace_logf_batch(dm, batch, θ, b_star, const_cache, c)
    logdet_negH, _, _ = _laplace_logdet_negH(
        dm, batch, θ, b_star, const_cache, c, nothing, 1;
        jitter = jitter, max_tries = max_tries, adaptive = adaptive,
        scale_factor = scale_factor, hmode = curvature)
    n_b = get_batch_re_dim(batch)
    return logf + (n_b / 2) * log(2 * pi) - logdet_negH / 2
end

function laplace_marginal(dm::DataModel, θ::ComponentArray;
        constants_re::NamedTuple = NamedTuple(),
        curvature::AbstractCurvature = ExactHessianCurvature(), jitter = 1e-6,
        max_tries::Int = 6, adaptive::Bool = false, scale_factor = 0.0, kwargs...)
    bstars, infos, cc, θ_re, cache = _empirical_bayes_batches(
        dm, θ; constants_re = constants_re, kwargs...)
    isempty(infos) && return zero(eltype(θ_re))
    return sum(laplace_marginal(dm, θ_re, infos[bi], bstars[bi]; const_cache = cc,
                   cache = cache, curvature = curvature, jitter = jitter,
                   max_tries = max_tries, adaptive = adaptive, scale_factor = scale_factor)
    for bi in eachindex(infos))
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
function ghq_marginal(dm::DataModel, θ::ComponentArray, batch::REBatchInfo;
        level = 3, const_cache::REConstantsCache, cache = nothing)
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    return _ghq_batch_ll(dm, batch, θ_re, const_cache, _dev_ll_cache(dm, cache), level)
end

function ghq_marginal(dm::DataModel, θ::ComponentArray;
        level = 3, constants_re::NamedTuple = NamedTuple(),
        ode_args::Tuple = (), ode_kwargs::NamedTuple = NamedTuple())
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    c = build_likelihood_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = EnsembleSerial(), force_saveat = true)
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
  proposal centered at the mode `b_star` with covariance `(−H)⁻¹` ([`posterior_moments`](@ref)),
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
        rng::AbstractRNG = Random.default_rng())
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
    c = _dev_ll_cache(dm, cache)
    n_b = get_batch_re_dim(batch)
    if method === :mcmc
        sampler === nothing &&
            error("sample_random_effect_draws(method=:mcmc) requires a Turing `sampler`, e.g. MH() or NUTS().")
        n_b == 0 &&
            return RandomEffectPosteriorSample(
                zeros(eltype(θ_re), 0, 0), nothing, nothing, :mcmc)
        re_names = get_re_names(get_random(get_model(dm)))
        tkw = (n_samples = n_samples, n_adapt = n_adapt, progress = false)
        samples, _, _ = _mcem_sample_batch(
            dm, batch, θ_re, const_cache, c, sampler, tkw, rng, re_names, false, nothing)
        return RandomEffectPosteriorSample(samples, nothing, nothing, :mcmc)
    elseif method === :importance
        n_b == 0 &&
            return RandomEffectPosteriorSample(zeros(0, n_samples), zeros(n_samples),
                Float64(n_samples), :importance)
        _, Σ = posterior_moments(
            dm, θ_re, batch, b_star; const_cache = const_cache, cache = c)
        Σ === nothing &&
            return RandomEffectPosteriorSample(zeros(n_b, 0), Float64[], 0.0, :importance)
        q = MvNormal(collect(float.(b_star)), Symmetric(Matrix(Σ)))
        draws = Matrix{Float64}(undef, n_b, n_samples)
        logw = Vector{Float64}(undef, n_samples)
        for r in 1:n_samples
            b_r = rand(rng, q)
            @inbounds draws[:, r] = b_r
            logp = complete_data_loglikelihood(
                dm, batch, θ_re, b_r; const_cache = const_cache, cache = c)
            @inbounds logw[r] = logp - logpdf(q, b_r)
        end
        w = exp.(logw .- maximum(logw))
        sw = sum(w)
        ess = sw > 0 ? sw^2 / sum(abs2, w) : 0.0
        return RandomEffectPosteriorSample(draws, logw, ess, :importance)
    end
    error("Unknown sample_random_effect_draws method $(method); use :importance or :mcmc.")
end

function sample_random_effect_draws(dm::DataModel, θ::ComponentArray;
        method::Symbol = :importance, sampler = nothing, n_samples::Int = 100,
        n_adapt::Int = 50, constants_re::NamedTuple = NamedTuple(),
        ode_args::Tuple = (), ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        ebe_options::EBEOptions = EBEOptions(), rescue = nothing,
        rng::AbstractRNG = Random.default_rng())
    if method === :mcmc
        θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(dm)))
        c = build_likelihood_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
            serialization = EnsembleSerial(), force_saveat = true)
        _, infos, cc = build_re_batch_infos(dm, constants_re)
        return [sample_random_effect_draws(
                    dm, θ_re, infos[bi], eltype(θ_re)[]; method = :mcmc,
                    sampler = sampler, n_samples = n_samples, n_adapt = n_adapt,
                    const_cache = cc, cache = c, rng = rng) for bi in eachindex(infos)]
    end
    bstars, infos, cc, θ_re, cache = _empirical_bayes_batches(dm, θ;
        constants_re = constants_re, ebe_options = ebe_options, rescue = rescue,
        ode_args = ode_args, ode_kwargs = ode_kwargs, serialization = serialization,
        rng = rng)
    return [sample_random_effect_draws(
                dm, θ_re, infos[bi], bstars[bi]; method = :importance,
                n_samples = n_samples, const_cache = cc, cache = cache, rng = rng)
            for bi in eachindex(infos)]
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
    dm, method; add_term = objective_term, kwargs...)

"""
    fit_laplace_family(dm, method, curvature::AbstractCurvature, args, fit_kwargs, validate_post_transform; kwargs...) -> FitResult

Shared driver for marginal random-effects methods (Laplace/FOCEI and any custom curvature):
finds the EB modes, assembles the Laplace marginal with the supplied `curvature`, and optimizes
it with the analytic (envelope + trace-estimator) θ-gradient. Swap only the `curvature`
([`AbstractCurvature`](@ref)/[`inner_curvature`](@ref)) to define a new marginal method.
"""
const fit_laplace_family = _fit_laplace_family

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
function free_parameter_layout(fe::FixedEffects; constants::NamedTuple = NamedTuple(),
        theta0_untransformed = nothing)
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
    θ0_free_t = ComponentArray(NamedTuple{Tuple(free_names)}(
        Tuple(getproperty(θ0_t, n) for n in free_names)))
    return NLFreeLayout(free_names, transform, inv_transform, θ_const_t,
        collect(θ_const_t), getaxes(θ_const_t), θ0_free_t,
        free_parameter_indices(θ_const_t, θ0_free_t), getaxes(θ0_free_t))
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
population primitives called with a bare `dm`, e.g. `posterior_moments(dm, θ)`, rebuild these
caches on every call - inside a loop, prefer the context forms). Rebuild the context only when
`dm` or `constants_re` change. It does not store parameters; θ flows through every call.

With a context, the primitives lose their cache arguments and address batches by index:

    ctx = build_fit_context(dm)
    θ   = initial_parameters(ctx)
    complete_data_loglikelihood(ctx, bi, θ, b)          # == complete_data_loglikelihood(dm, batches[bi], θ, b;
                                                #      const_cache=cc, cache=cache)
    posterior_moments(ctx, θ)                   # one (b*, Σ) per batch, reusing ctx caches
    empirical_bayes(ctx, θ)                     # per-batch modes b*
    laplace_marginal(ctx, θ)                    # marginal log-likelihood at θ
    sample_random_effect_draws(ctx, θ)          # posterior draws per batch
    θ̂, sol = optimize_parameters(f, ctx)        # natural-scale objective, handled transforms
    build_fit_result(ctx, method, θ̂; kind=:frequentist_re, objective=...)  # eb_modes=:auto

The evaluation cache is single-threaded (`ponytail:` serial cache; pass the explicit primitives
your own per-thread caches when parallelising a custom loop).
"""
function build_fit_context(dm::DataModel;
        constants_re::NamedTuple = NamedTuple(),
        ode_args::Tuple = (), ode_kwargs::NamedTuple = NamedTuple())
    _, infos, cc = build_re_batch_infos(dm, constants_re)
    cache = build_likelihood_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        force_saveat = true)
    return FitContext(dm, infos, cc, cache, constants_re)
end

"""
    get_batch_infos(ctx::FitContext) -> Vector{REBatchInfo}

The context's random-effect batch descriptors; batch indices `bi` passed to the context
primitives index into this vector.
"""
@inline get_batch_infos(ctx::FitContext) = ctx.batch_infos

@inline get_data_model(ctx::FitContext) = ctx.dm

"""
    initial_parameters(ctx::FitContext) -> ComponentArray

A fresh copy of the model's natural-scale initial fixed effects - the conventional starting
point of a fitting loop (replace it with `theta_0_untransformed` when the caller supplies one).
"""
initial_parameters(ctx::FitContext) = copy(get_θ0_untransformed(get_fixed(get_model(ctx.dm))))

# Batch-index covers over the density primitives.
for f in (:conditional_loglikelihood, :re_logprior, :complete_data_loglikelihood,
    :complete_data_loglikelihood_gradient, :complete_data_loglikelihood_hessian)
    @eval @inline function $f(ctx::FitContext, bi::Integer, θ::ComponentArray, b; kwargs...)
        return $f(ctx.dm, ctx.batch_infos[bi], θ, b;
            const_cache = ctx.const_cache, cache = ctx.cache, kwargs...)
    end
end

@inline function ghq_marginal(ctx::FitContext, bi::Integer, θ::ComponentArray; level = 3)
    return ghq_marginal(ctx.dm, θ, ctx.batch_infos[bi]; level = level,
        const_cache = ctx.const_cache, cache = ctx.cache)
end

function ghq_marginal(ctx::FitContext, θ::ComponentArray; level = 3)
    isempty(ctx.batch_infos) && return zero(eltype(θ))
    return sum(ghq_marginal(ctx, bi, θ; level = level)
    for bi in eachindex(ctx.batch_infos))
end

# Population forms reusing the context caches (the bare-`dm` forms rebuild them per call).
function empirical_bayes(ctx::FitContext, θ::ComponentArray;
        ebe_options::EBEOptions = EBEOptions(), rescue = nothing,
        rng::AbstractRNG = Random.default_rng())
    θ_re = symmetrize_psd_parameters(θ, get_fixed(get_model(ctx.dm)))
    bstars, _ = _compute_bstars(
        ctx.dm, θ_re, ctx.constants_re, ctx.cache, ebe_options, rng;
        rescue = rescue)
    return bstars
end

function posterior_moments(ctx::FitContext, θ::ComponentArray;
        curvature::AbstractCurvature = ExactHessianCurvature(),
        ebe_options::EBEOptions = EBEOptions(), rescue = nothing,
        rng::AbstractRNG = Random.default_rng(), kwargs...)
    bstars = empirical_bayes(ctx, θ; ebe_options = ebe_options, rescue = rescue, rng = rng)
    return [posterior_moments(ctx.dm, θ, ctx.batch_infos[bi], bstars[bi];
                const_cache = ctx.const_cache, cache = ctx.cache,
                curvature = curvature, kwargs...)
            for bi in eachindex(ctx.batch_infos)]
end

function laplace_marginal(ctx::FitContext, θ::ComponentArray;
        curvature::AbstractCurvature = ExactHessianCurvature(),
        ebe_options::EBEOptions = EBEOptions(), rescue = nothing,
        rng::AbstractRNG = Random.default_rng(), kwargs...)
    isempty(ctx.batch_infos) && return zero(eltype(θ))
    bstars = empirical_bayes(ctx, θ; ebe_options = ebe_options, rescue = rescue, rng = rng)
    return sum(laplace_marginal(ctx.dm, θ, ctx.batch_infos[bi], bstars[bi];
                   const_cache = ctx.const_cache, cache = ctx.cache,
                   curvature = curvature, kwargs...)
    for bi in eachindex(ctx.batch_infos))
end

function sample_random_effect_draws(ctx::FitContext, θ::ComponentArray;
        method::Symbol = :importance, sampler = nothing, n_samples::Int = 100,
        n_adapt::Int = 50, ebe_options::EBEOptions = EBEOptions(), rescue = nothing,
        rng::AbstractRNG = Random.default_rng())
    if method === :mcmc
        return [sample_random_effect_draws(
                    ctx.dm, θ, ctx.batch_infos[bi], eltype(θ)[]; method = :mcmc,
                    sampler = sampler, n_samples = n_samples, n_adapt = n_adapt,
                    const_cache = ctx.const_cache, cache = ctx.cache, rng = rng)
                for bi in eachindex(ctx.batch_infos)]
    end
    bstars = empirical_bayes(ctx, θ; ebe_options = ebe_options, rescue = rescue, rng = rng)
    return [sample_random_effect_draws(
                ctx.dm, θ, ctx.batch_infos[bi], bstars[bi]; method = method,
                n_samples = n_samples, const_cache = ctx.const_cache, cache = ctx.cache,
                rng = rng) for bi in eachindex(ctx.batch_infos)]
end

"""
    optimize_parameters(f_natural, ctx::FitContext;
                        θ_start=initial_parameters(ctx),
                        optimizer=LBFGS(linesearch=BackTracking()),
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

`ponytail:` optimizes all fixed effects; apply `constants`/bounds via the explicit
`free_parameter_layout`/`resolve_optimizer_bounds` path when needed.
"""
function optimize_parameters(f_natural, ctx::FitContext;
        θ_start::ComponentArray = initial_parameters(ctx),
        optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()),
        adtype = Optimization.AutoForwardDiff(),
        optim_kwargs::NamedTuple = NamedTuple())
    dm = ctx.dm
    fe = get_fixed(get_model(dm))
    inv_transform = get_inverse_transform(fe)
    θt0 = get_transform(fe)(θ_start)
    axs = getaxes(θt0)
    obj = (θt_vec, _) -> f_natural(symmetrize_psd_parameters(
        dm, inv_transform(ComponentArray(θt_vec, axs))))
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
function build_fit_result(ctx::FitContext, method::FittingMethod, θ::ComponentArray;
        kind::Symbol = :frequentist, eb_modes = :auto, kwargs...)
    modes = eb_modes === :auto ?
            (kind in (:frequentist_re, :ghquadrature, :saem, :mcem) ?
             empirical_bayes(ctx, θ) : nothing) : eb_modes
    return build_fit_result(ctx.dm, method, θ; kind = kind, eb_modes = modes, kwargs...)
end
