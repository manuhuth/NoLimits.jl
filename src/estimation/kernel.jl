# kernel.jl
# Batch-level log-likelihood computation via Smolyak sparse-grid quadrature.
# Implements signed logsumexp and the subject/batch integration kernel.

# ---------------------------------------------------------------------------
# Signed LogSumExp
# ---------------------------------------------------------------------------

"""
    signed_logsumexp(logvals, signs) -> (log_abs_result, result_sign::Int8)

Numerically stable computation of log|Σ_r s_r exp(a_r)| where s_r ∈ {+1, -1}.

Returns `(log|result|, sign(result))` as `(Float64, Int8)`.

The sum is split into positive and negative contributions, each stabilised by
the maximum over all `logvals`:

    pos = Σ_{s_r=+1} exp(a_r - amax)
    neg = Σ_{s_r=-1} exp(a_r - amax)
    result = exp(amax) * (pos - neg)

If |result| ≈ 0 (catastrophic cancellation), returns `(-Inf, +1)`.

This situation can occur at high Smolyak levels where the inclusion-exclusion
coefficients are large in magnitude and nearly cancel. Levels 1–3 are
numerically stable for typical NLME models.
"""
function signed_logsumexp(logvals::AbstractVector{T}, signs::AbstractVector{Int8}) where {T<:Number}
    isempty(logvals) && return (T(-Inf), Int8(1))

    amax = maximum(logvals)
    (isinf(amax) && amax < 0) && return (T(-Inf), Int8(1))

    pos = zero(T)
    neg = zero(T)
    @inbounds for i in eachindex(logvals, signs)
        shifted = exp(logvals[i] - amax)
        if signs[i] > 0
            pos += shifted
        else
            neg += shifted
        end
    end

    diff = pos - neg
    if diff > zero(T)
        return (amax + log(diff), Int8(1))
    elseif diff < zero(T)
        return (amax + log(-diff), Int8(-1))
    else
        return (T(-Inf), Int8(1))
    end
end

# ---------------------------------------------------------------------------
# Batch log-likelihood via sparse-grid quadrature
# ---------------------------------------------------------------------------

"""
    batch_loglik_ghq(dm, batch_info, θ, re_measure, sgrid, const_cache, ll_cache)
    -> Float64 (or Dual)

Estimate the batch marginal log-likelihood

    log ∫ p(y_batch | b, θ) p(b | θ) db

using the Smolyak sparse-grid quadrature rule `sgrid` and the RE measure
`re_measure`.

The integral is computed as:

    log L_batch ≈ signed_logsumexp_r [ log|W_r| + Σᵢ ℓᵢ(T(z_r), θ) + log c(z_r) ]

where `T(z_r) = transform(re_measure, z_r)` and `log c(z_r) = logcorrection(re_measure, z_r)`.

**For `GaussianRE`**: `logcorrection = 0`, and the Gauss-Hermite weights already
encode the N(z; 0, I) measure that integrates against the prior exactly via the
change of variables b = μ + Lz. The prior logpdf term is NOT added separately.

**ForwardDiff compatibility**: When `θ` carries Dual tags, the gradient flows
through `μ(θ)` and `L(θ)` in `re_measure`, and through `_loglikelihood_individual`.
The nodes `z_r` are precomputed Float64 constants.

Returns `-Inf` (promoting to the accumulator type) if:
- Any individual likelihood evaluates to `-Inf`
- The signed logsumexp result is negative (numerical instability warning)
- The result is non-finite
"""
function batch_loglik_ghq(
    dm::DataModel,
    batch_info::_LaplaceBatchInfo,
    θ::ComponentArray,
    re_measure::AbstractREMeasure,
    sgrid::GHQuadratureNodes{Float64},
    const_cache::LaplaceConstantsCache,
    ll_cache::_LLCache,
)
    R  = size(sgrid.nodes, 2)
    θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)

    # Determine accumulator element type from the RE measure.
    # When θ carries ForwardDiff.Dual tags, re_measure.μ has Dual elements,
    # so T = Dual. The Array{T} allocation handles type promotion seamlessly.
    T = eltype(re_measure)

    a_vals = Vector{T}(undef, R)

    @inbounds for r in 1:R
        z_r = sgrid.nodes[:, r]                       # Float64 column, no alloc
        b_r = transform(re_measure, z_r)              # T-valued: μ + L * z_r

        # Sum conditional log-likelihoods over all individuals in batch
        cond  = zero(T)
        valid = true
        for i in batch_info.inds
            η_i = _build_eta_ind(dm, i, batch_info, b_r, const_cache, θ_re)
            lli  = _loglikelihood_individual(dm, i, θ_re, η_i, ll_cache)
            if !isfinite(lli)
                valid = false
                break
            end
            cond += T(lli)
        end

        a_vals[r] = if valid
            T(sgrid.logweights[r]) + cond + logcorrection(re_measure, z_r)
        else
            T(-Inf)
        end
    end

    log_val, result_sign = signed_logsumexp(a_vals, sgrid.signs)

    if result_sign < 0
        @warn "GHQuadrature: batch marginal likelihood estimate is negative " *
              "(signed logsumexp returned negative result). " *
              "This indicates numerical instability — consider reducing `level` " *
              "or checking your model specification."
        return T(-Inf)
    end

    isfinite(log_val) || return T(-Inf)
    return log_val
end

# ---------------------------------------------------------------------------
# Batch log-likelihood via Monte Carlo integration
# ---------------------------------------------------------------------------

"""
    batch_loglik_mc_prior(dm, batch_info, θ, const_cache, ll_cache, n_samples, rng)

Estimate the batch marginal log-likelihood

    log ∫ p(y_batch | b, θ) p(b | θ) db

via prior importance sampling: draws `b_r ~ p(b | θ)` and returns

    logsumexp_r [ log p(y_batch | b_r, θ) ] - log(n_samples)

This is the standard Monte Carlo estimator with the prior as proposal. It is exact
in expectation but can have high variance when the posterior is narrow relative to
the prior. Returns `-Inf` if all samples have non-finite likelihoods.

Internally reuses `_is_prior_sample_batch` (from mcem.jl) for prior sampling.
"""
function batch_loglik_mc_prior(
    dm         :: DataModel,
    batch_info :: _LaplaceBatchInfo,
    θ          :: ComponentArray,
    const_cache:: LaplaceConstantsCache,
    ll_cache   :: _LLCache,
    n_samples  :: Int,
    rng        :: AbstractRNG,
)
    batch_info.n_b == 0 && error("batch_loglik_mc_prior: called with n_b == 0; no RE to integrate.")
    re_names = get_re_names(dm.model.random.random)

    # Draw b_r ~ p(b|θ) and get log_qs[r] = log p(b_r|θ)
    samples, log_qs = _is_prior_sample_batch(dm, batch_info, θ, const_cache, ll_cache, rng, n_samples, re_names)

    # log p(y|b_r,θ) = _laplace_logf_batch(b_r) - log_qs[r]
    # since _laplace_logf_batch = log p(y|b,θ) + log p(b|θ)
    log_cond = Vector{Float64}(undef, n_samples)
    for r in 1:n_samples
        b_r  = view(samples, :, r)
        logf = _laplace_logf_batch(dm, batch_info, θ, b_r, const_cache, ll_cache)
        log_cond[r] = isfinite(logf) ? Float64(logf) - log_qs[r] : -Inf
    end

    all(isequal(-Inf), log_cond) && return -Inf
    amax = maximum(lc for lc in log_cond if isfinite(lc))
    return amax + log(sum(exp(lc - amax) for lc in log_cond if isfinite(lc))) - log(Float64(n_samples))
end

"""
    batch_loglik_mc_turing(dm, batch_info, θ, const_cache, ll_cache, n_samples, sampler, n_warmup, rng)

Estimate the batch marginal log-likelihood via MCMC-guided Gaussian importance sampling.

**Algorithm:**
1. Run Turing MCMC with `n_warmup` warmup steps followed by `n_warmup` post-warmup steps
   to draw approximate posterior samples `b_r ~ p(b | y, θ)`.
2. Fit a Gaussian proposal `q = N(μ, Σ)` to those posterior samples (mean + regularised
   covariance). This captures the location and scale of the posterior without assuming
   anything about its shape at the mode.
3. Draw `n_samples` **fresh** IID samples from `q`. These are the IS points.
4. Compute importance weights and return

       logsumexp_r [ log p(y_batch, b_r | θ) - log q(b_r) ] - log(n_samples)

Using fresh samples from `q` (step 3) rather than the MCMC samples themselves makes the
IS estimator unbiased: samples and proposal match by construction. The MCMC chain is only
used to fit the proposal shape.

Returns `-Inf` if all IS weights are non-finite.

The default sampler (when `sampler === nothing`) is `MH()`.
"""
function batch_loglik_mc_turing(
    dm         :: DataModel,
    batch_info :: _LaplaceBatchInfo,
    θ          :: ComponentArray,
    const_cache:: LaplaceConstantsCache,
    ll_cache   :: _LLCache,
    n_samples  :: Int,
    sampler,
    n_warmup   :: Int,
    rng        :: AbstractRNG,
)
    batch_info.n_b == 0 && error("batch_loglik_mc_turing: called with n_b == 0; no RE to integrate.")
    n_b      = batch_info.n_b
    re_names = get_re_names(dm.model.random.random)

    # Step 1: Run MCMC to collect posterior samples for fitting the proposal.
    # n_warmup steps are discarded; then n_warmup more are kept for proposal fitting.
    effective_sampler = sampler === nothing ? Turing.MH() : sampler
    n_mcmc = max(n_warmup, 50)
    tkwargs = (n_samples=n_mcmc, n_adapt=n_warmup, progress=false, verbose=false)
    mcmc_samples, _, _ = _mcem_sample_batch(dm, batch_info, θ, const_cache, ll_cache,
                                             effective_sampler, tkwargs, rng, re_names,
                                             false, NamedTuple())
    n_mcmc_valid = size(mcmc_samples, 2)
    n_mcmc_valid == 0 && return -Inf

    # Step 2: Fit Gaussian proposal q to the posterior samples.
    # B has shape (n_b, n_mcmc_valid); transpose so rows = observations for cov.
    BT = mcmc_samples'  # (n_mcmc_valid, n_b)
    μ_q = vec(Statistics.mean(BT; dims=1))

    # Detect chain non-mixing: if variance is near-zero the chain is stuck.
    # Fall back to prior IS, which is always valid.
    max_var = n_b == 1 ? Statistics.var(view(mcmc_samples, 1, :)) :
                         maximum(Statistics.var(view(mcmc_samples, d, :)) for d in 1:n_b)
    if max_var < 1e-10
        @warn "batch_loglik_mc_turing: MCMC chain did not mix (max marginal variance ≈ 0). " *
              "Falling back to prior IS for this batch. Consider a better sampler or more warmup steps."
        return batch_loglik_mc_prior(dm, batch_info, θ, const_cache, ll_cache, n_samples, rng)
    end

    # Step 3: Draw FRESH IID samples from q for IS (samples and proposal now match).
    if n_b == 1
        σ_q  = sqrt(Statistics.var(view(mcmc_samples, 1, :)) + 1e-8)
        q1d  = Distributions.Normal(μ_q[1], σ_q)
        B_is = reshape(rand(rng, q1d, n_samples), 1, n_samples)
        q_logpdf = b -> Distributions.logpdf(q1d, Float64(b[1]))
    else
        Σ_raw = Statistics.cov(BT) + 1e-8 * LinearAlgebra.I(n_b)
        q_mv  = Distributions.MvNormal(μ_q, Symmetric(Σ_raw))
        B_is  = rand(rng, q_mv, n_samples)   # (n_b, n_samples)
        q_logpdf = b -> Distributions.logpdf(q_mv, Vector{Float64}(b))
    end

    # Step 4: Compute IS weights log w_r = log p(y,b_r|θ) - log q(b_r).
    log_ws = Vector{Float64}(undef, n_samples)
    for r in 1:n_samples
        b_r  = view(B_is, :, r)
        logf = _laplace_logf_batch(dm, batch_info, θ, b_r, const_cache, ll_cache)
        logq = q_logpdf(b_r)
        log_ws[r] = isfinite(logf) ? Float64(logf) - logq : -Inf
    end

    all(isequal(-Inf), log_ws) && return -Inf
    amax = maximum(lw for lw in log_ws if isfinite(lw))
    return amax + log(sum(exp(lw - amax) for lw in log_ws if isfinite(lw))) - log(Float64(n_samples))
end

"""
    _batch_loglik_from_mc(dm, batch_info, θ, const_cache, ll_cache, mc, fallback_rng)

Dispatch to the appropriate MC batch log-likelihood estimator based on `mc.mode`.
`fallback_rng` is used when `mc.rng === nothing` (the default, meaning "inherit from caller").
"""
function _batch_loglik_from_mc(
    dm          :: DataModel,
    batch_info  :: _LaplaceBatchInfo,
    θ           :: ComponentArray,
    const_cache :: LaplaceConstantsCache,
    ll_cache    :: _LLCache,
    mc          :: MCIntegrator,
    fallback_rng:: AbstractRNG,
)
    rng = mc.rng === nothing ? fallback_rng : mc.rng
    if mc.mode === :prior
        return batch_loglik_mc_prior(dm, batch_info, θ, const_cache, ll_cache, mc.n_samples, rng)
    elseif mc.mode === :turing
        return batch_loglik_mc_turing(dm, batch_info, θ, const_cache, ll_cache,
                                      mc.n_samples, mc.sampler, mc.n_warmup, rng)
    else
        error("_batch_loglik_from_mc: unknown MCIntegrator mode :$(mc.mode). Use :prior or :turing.")
    end
end
