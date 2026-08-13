using LinearAlgebra
using Random

# Pseudo-inverse of the objective Hessian for the Wald "bread" matrix.
#
# `pinv`'s default rank tolerance maps a near-zero singular value to 0 rather than to a
# large reciprocal, so a weakly identified direction came out with near-zero variance -
# falsely precise, the exact opposite of what a standard error must convey (issue #173).
# Keep every nonzero singular value so such a direction yields a LARGE variance, and only
# drop the exactly singular ones (which is what makes this usable where `inv` throws).
function _wald_pinv(H::AbstractMatrix, active_names)
    F = svd(H)
    n = length(F.S)
    smax = n == 0 ? 0.0 : maximum(F.S)
    tol = eps(Float64) * smax * maximum(size(H))
    weak = findall(<(tol), F.S)
    if !isempty(weak)
        # Name the parameters loading most on the (near-)null directions - those are the
        # ones whose reported SE is now large because they are not identified by the data.
        loads = length(active_names) == n ?
                [active_names[argmax(abs.(view(F.V, :, j)))] for j in weak] : active_names
        @warn "Wald covariance: the objective Hessian is (near-)singular in "*
              "$(length(weak)) direction(s); the standard errors for these parameters are "*
              "large because they are only weakly identified. Consider reparameterizing or "*
              "fixing them via `constants=`." parameters=unique(loads) smallest_singular_value=minimum(F.S)
    end
    Sinv = [s > 0 ? inv(s) : zero(s) for s in F.S]
    return F.V * Diagonal(Sinv) * F.U'
end

function _wald_bread(H::AbstractMatrix, pseudo_inverse::Bool, active_names)
    bread = try
        pseudo_inverse ? _wald_pinv(H, active_names) : inv(H)
    catch err
        pseudo_inverse ||
            error("Failed to invert Hessian for Wald covariance. Consider pseudo_inverse=true. Original error: $(sprint(showerror, err))")
        @warn "Falling back to pseudo-inverse for Hessian inversion in UQ." error=sprint(
            showerror, err)
        _wald_pinv(H, active_names)
    end
    return Matrix{Float64}(0.5 .* (bread .+ bread'))
end

# Shared Wald finalize tail: project the raw covariance to PSD, draw from the Gaussian
# approximation, map draws back to the natural scale, extend any stickbreak coordinates,
# and assemble the UQResult. `extra_diag` carries method-specific diagnostics (e.g.
# `approximation_method`) inserted between `vcov` and `pseudo_inverse`.
function _finalize_wald_uqresult(fe, θ_hat_t, θ_hat_u, free_names, active_idx,
        active_names, active_kinds, θu_from_active, Vt_raw, backend_used, vcov,
        pseudo_inverse, n_draws, level, rng, method_sym, extra_diag)
    Vt, vcov_diag = _project_psd_covariance(Vt_raw)

    θ_coords_t = _coords_on_transformed_layout(fe, θ_hat_t, free_names; natural = false)
    θ_coords_u = _coords_on_transformed_layout(fe, θ_hat_u, free_names; natural = true)
    est_t = θ_coords_t[active_idx]
    est_n = θ_coords_u[active_idx]

    draws_t = _sample_gaussian_draws(rng, est_t, Vt, n_draws)
    draws_n = Matrix{Float64}(undef, size(draws_t, 1), size(draws_t, 2))
    for i in 1:size(draws_t, 1)
        θu_i = θu_from_active(@view(draws_t[i, :]))
        coords_u_i = _coords_on_transformed_layout(fe, θu_i, free_names; natural = true)
        draws_n[i, :] .= coords_u_i[active_idx]
    end

    # A Wald draw is Gaussian on the TRANSFORMED scale, so pushing it through the inverse
    # transform can overflow when a coordinate is weakly identified: `exp` of a wide
    # log-Cholesky draw is `Inf`, and `Inf - Inf` is `NaN`. Those rows used to flow straight into
    # the natural-scale covariance and quantiles, so `get_uq_vcov()` returned an all-NaN matrix
    # (its default is `scale = :natural`) and the intervals threw on quantiles of NaN. The
    # transformed-scale covariance is unaffected and stays exact; only the natural-scale
    # summaries drop the offending rows, and the count is reported so it cannot pass unnoticed.
    # `_wald_usable_draw_row` rejects rows that are merely too large to square, not just the
    # non-finite ones - see its definition for why an `isfinite` test alone was not enough.
    finite_rows = [_wald_usable_draw_row(@view(draws_n[i, :]), n_draws)
                   for i in 1:size(draws_n, 1)]
    n_nonfinite = count(!, finite_rows)
    n_nonfinite == length(finite_rows) &&
        error("Wald natural-scale summaries unavailable: all $(n_nonfinite) draws overflowed " *
              "when mapped to the natural scale, which means the estimate is only weakly " *
              "identified in at least one coordinate. The transformed-scale covariance is " *
              "still available via get_uq_vcov(uq; scale = :transformed).")
    n_nonfinite > 0 &&
        @warn "Excluding non-finite Wald draws from the natural-scale summaries." n_nonfinite n_total=length(finite_rows)
    # `draws_n` itself is kept whole: `get_uq_draws` should report what was drawn, and a user
    # inspecting the draws should see the overflow rather than find rows silently missing.
    draws_n_fin = n_nonfinite > 0 ? draws_n[finite_rows, :] : draws_n

    intervals_t = _intervals_from_draws(draws_t, level)
    intervals_n = _intervals_from_draws(draws_n_fin, level)

    ext = _extend_natural_stickbreak(fe, free_names, active_names, active_kinds,
        est_n, draws_n, intervals_n)
    names_n = ext !== nothing ? ext[1] : nothing
    est_n_use = ext !== nothing ? ext[2] : est_n
    draws_n_use = ext !== nothing ? ext[3] : draws_n
    intervals_n_use = ext !== nothing ? ext[4] : intervals_n
    # Covariance from the finite rows only, of whatever the stickbreak extension produced.
    Vn_src = draws_n_use !== nothing ? draws_n_use : draws_n
    Vn_rows = [_wald_usable_draw_row(@view(Vn_src[i, :]), n_draws)
               for i in 1:size(Vn_src, 1)]
    Vn_use = _cov_from_draws(all(Vn_rows) ? Vn_src : Vn_src[Vn_rows, :])

    diag = merge(
        (;
            hessian_backend = backend_used,
            hessian_reduced = true,
            inactive_fixed_effects_held_constant = true,
            vcov = vcov
        ),
        extra_diag,
        (;
            pseudo_inverse = pseudo_inverse,
            n_draws = n_draws,
            n_draws_nonfinite_natural = n_nonfinite,
            n_active_parameters = length(active_idx),
            coordinate_transforms = active_kinds
        ),
        vcov_diag)

    return UQResult(
        :wald,
        method_sym,
        active_names,
        names_n,
        est_t,
        est_n_use,
        intervals_t,
        intervals_n_use,
        Vt,
        Vn_use,
        draws_t,
        draws_n_use,
        diag
    )
end

@inline function _is_re_laplace_family(method::FittingMethod)
    return method isa Laplace
end

function _resolve_wald_re_approx_method(source_method::FittingMethod;
        re_approx::Symbol,
        re_approx_method::Union{Nothing, FittingMethod})
    if _is_re_laplace_family(source_method) || source_method isa GHQuadrature ||
       source_method isa FOCEI
        re_approx == :auto ||
            error("re_approx is only used for MCEM/SAEM Wald UQ results.")
        re_approx_method === nothing ||
            error("re_approx_method is only used for MCEM/SAEM Wald UQ results.")
        return source_method
    end

    if !(source_method isa MCEM || source_method isa SAEM ||
         uq_family(source_method) == :wald_re)
        error("Wald UQ for random-effects models currently supports Laplace, FOCEI, MCEM, " *
              "SAEM, GHQuadrature, or a method with uq_family == :wald_re.")
    end

    if re_approx_method !== nothing
        _is_re_laplace_family(re_approx_method) ||
            error("re_approx_method must be a Laplace method instance.")
        return re_approx_method
    end

    approx = re_approx == :auto ? :laplace : re_approx
    if approx == :laplace
        return Laplace()
    end
    error("For MCEM/SAEM Wald UQ, re_approx must be :auto or :laplace.")
end

function _compute_uq_wald_no_re(res::FitResult;
        level::Float64,
        vcov::Symbol,
        pseudo_inverse::Bool,
        hessian_backend::Symbol,
        fd_abs_step::Real,
        fd_rel_step::Real,
        fd_max_tries::Int,
        n_draws::Int,
        constants::Union{Nothing, NamedTuple},
        penalty::Union{Nothing, NamedTuple},
        ode_args::Union{Nothing, Tuple},
        ode_kwargs::Union{Nothing, NamedTuple},
        serialization::Union{Nothing, SciMLBase.EnsembleAlgorithm},
        rng::AbstractRNG)
    dm = get_data_model(res)
    dm === nothing &&
        error("This fit result does not store a DataModel; pass store_data_model=true when fitting.")
    method = get_method(res)
    is_pooled = method isa Pooled || method isa PooledMap
    (method isa MLE || method isa MAP || is_pooled || uq_family(method) == :wald_no_re) ||
        error("This Wald path supports MLE, MAP, Pooled, PooledMap, or a method with " *
              "uq_family == :wald_no_re.")
    if is_pooled
        wk = get_notes(res).weakly_identified
        isempty(wk) ||
            @warn "Pooled Wald UQ: weakly identified parameter(s) $(join(wk, ", ")) may " *
                  "produce a near-singular Hessian. Consider pseudo_inverse=true or " *
                  "fixing them via constants."
    end

    constants_use = _resolve_fit_kw(res, constants, :constants, NamedTuple())
    penalty_use = _resolve_fit_kw(res, penalty, :penalty, NamedTuple())
    ode_args_use = _resolve_fit_kw(res, ode_args, :ode_args, ())
    ode_kwargs_use = _resolve_fit_kw(res, ode_kwargs, :ode_kwargs, NamedTuple())
    serialization_use = _resolve_fit_kw(
        res, serialization, :serialization, EnsembleSerial())

    fe = get_fixed(get_model(dm))
    free_names = _free_fixed_names(fe, constants_use)
    isempty(free_names) &&
        error("No free fixed effects are available for UQ after applying constants.")

    active_mask = _active_mask_for_free(fe, free_names)
    active_idx = findall(identity, active_mask)
    isempty(active_idx) &&
        error("No UQ-eligible fixed-effect coordinates found. Mark parameters with calculate_se=true and ensure they are not fixed via constants.")

    free_flat_names = _flat_names_for_free(fe, free_names)
    active_names = free_flat_names[active_idx]
    free_flat_kinds = _flat_transform_kinds_for_free(fe, free_names)
    active_kinds = free_flat_kinds[active_idx]

    θ_hat_u = get_params(res; scale = :untransformed)
    transform = get_transform(fe)
    inv_transform = get_inverse_transform(fe)
    θ_hat_t = transform(θ_hat_u)

    θ_const_u = deepcopy(θ_hat_u)
    _apply_constants!(θ_const_u, constants_use)
    θ_const_t = transform(θ_const_u)

    θ_hat_free_t = θ_hat_t[free_names]
    axs_free = getaxes(θ_hat_free_t)
    axs_full = getaxes(θ_const_t)

    xhat_full = Float64.(collect(θ_hat_free_t))
    xhat_active = xhat_full[active_idx]
    ll_cache = _build_ll_cache_uq(dm, ode_args_use, ode_kwargs_use, serialization_use)
    use_penalty = !isempty(keys(penalty_use))
    use_prior = method isa MAP || method isa PooledMap
    # Pooled: η is the plug-in value of the RE distributions — a function of θ that
    # must be recomputed inside the objective so the Hessian carries the chain rule.
    pooled_strategies = is_pooled ? get_result(res).strategies : nothing
    eta_for = function (θu)
        is_pooled || return ComponentArray()
        return _compute_pooled_etas(dm, θu, pooled_strategies)
    end

    function _θu_from_active(x_active::AbstractVector)
        T = eltype(x_active)
        x_full = T.(xhat_full)
        x_full[active_idx] .= x_active
        return _theta_u_from_free_t(
            x_full, axs_free, θ_const_t, axs_full, free_names, inv_transform)
    end

    function obj_active(x_active::AbstractVector)
        θu = _θu_from_active(x_active)
        η = try
            eta_for(θu)
        catch
            return Inf
        end
        ll = loglikelihood(dm, θu, η; cache = ll_cache, serialization = serialization_use)
        ll == -Inf && return Inf

        obj = -ll
        if use_prior
            lp = logprior(fe, θu)
            lp == -Inf && return Inf
            obj += -lp
        end
        use_penalty && (obj += _penalty_value(θu, penalty_use))
        return obj
    end

    H_active, backend_used = _hessian_from_objective(obj_active, xhat_active;
        backend = hessian_backend,
        fd_abs_step = fd_abs_step,
        fd_rel_step = fd_rel_step,
        fd_max_tries = fd_max_tries)
    H_active = 0.5 .* (H_active .+ H_active')
    # `pinv` of a non-finite matrix returns all-NaN without throwing, so without this the caller
    # gets NaN standard errors and no explanation. Same principle as rejecting a jitter-only
    # definite Hessian: report that the covariance is unavailable rather than fabricate one.
    all(isfinite, H_active) ||
        error("Wald covariance unavailable: the objective Hessian at the estimate is not " *
              "finite (backend $(backend_used)). The fit is at a point where the marginal " *
              "is not differentiable - typically a degenerate random-effect covariance or an " *
              "unconverged fit. Check the fit converged, or use method = :profile / :mcmc.")

    bread = _wald_bread(H_active, pseudo_inverse, active_names)

    Vt_raw = if vcov == :hessian
        copy(bread)
    elseif vcov == :sandwich
        ll_cache_local = ll_cache isa Vector ? ll_cache[1] : ll_cache
        B = zeros(Float64, length(active_idx), length(active_idx))
        for i in eachindex(get_individuals(dm))
            obj_i = function (x_active::AbstractVector)
                θu = _θu_from_active(x_active)
                η = try
                    eta_for(θu)
                catch
                    return Inf
                end
                # Non-pooled `eta_for` returns a single (empty) ComponentArray for the
                # whole model; pooled returns one eta per individual. Only index in the
                # pooled case — an empty ComponentArray is itself an AbstractVector, so a
                # plain `isa AbstractVector` test would wrongly try to index it.
                η_i = is_pooled ? η[i] : η
                ll_i = _loglikelihood_individual(dm, i, θu, η_i, ll_cache_local)
                ll_i == -Inf && return Inf
                return Float64(-ll_i)
            end
            g = _gradient_from_objective(obj_i, xhat_active;
                fd_abs_step = fd_abs_step,
                fd_rel_step = fd_rel_step,
                fd_max_tries = fd_max_tries)
            B .+= g * g'
        end
        B = 0.5 .* (B .+ B')
        M = bread * B * bread'
        Matrix{Float64}(0.5 .* (M .+ M'))
    else
        error("Unsupported vcov=$(vcov). Use :hessian or :sandwich.")
    end

    return _finalize_wald_uqresult(fe, θ_hat_t, θ_hat_u, free_names, active_idx,
        active_names, active_kinds, _θu_from_active, Vt_raw, backend_used, vcov,
        pseudo_inverse, n_draws, level, rng, _method_symbol(method), NamedTuple())
end

function _compute_uq_wald_re(res::FitResult;
        level::Float64,
        vcov::Symbol,
        re_approx::Symbol,
        re_approx_method::Union{Nothing, FittingMethod},
        pseudo_inverse::Bool,
        hessian_backend::Symbol,
        fd_abs_step::Real,
        fd_rel_step::Real,
        fd_max_tries::Int,
        n_draws::Int,
        constants::Union{Nothing, NamedTuple},
        constants_re::Union{Nothing, NamedTuple},
        penalty::Union{Nothing, NamedTuple},
        ode_args::Union{Nothing, Tuple},
        ode_kwargs::Union{Nothing, NamedTuple},
        serialization::Union{Nothing, SciMLBase.EnsembleAlgorithm},
        rng::AbstractRNG)
    dm = get_data_model(res)
    dm === nothing &&
        error("This fit result does not store a DataModel; pass store_data_model=true when fitting.")
    source_method = get_method(res)
    approx_method = _resolve_wald_re_approx_method(source_method;
        re_approx = re_approx,
        re_approx_method = re_approx_method)

    constants_use = _resolve_fit_kw(res, constants, :constants, NamedTuple())
    constants_re_use = _resolve_fit_kw(res, constants_re, :constants_re, NamedTuple())
    penalty_use = _resolve_fit_kw(res, penalty, :penalty, NamedTuple())
    ode_args_use = _resolve_fit_kw(res, ode_args, :ode_args, ())
    ode_kwargs_use = _resolve_fit_kw(res, ode_kwargs, :ode_kwargs, NamedTuple())
    # Force SERIAL evaluation of the random-effects Laplace objective (EB solve + inner
    # logdet/Hessian) used to build the covariance. The threaded per-batch inner Hessian
    # is non-deterministic run-to-run (it produces a varying Wald covariance) — the
    # optimizer-driven fit masks this, but UQ reports the covariance directly. Wald UQ is
    # a one-shot post-fit step, so serial is an acceptable cost for a reproducible, correct
    # covariance. (The MLE/MAP/Pooled Wald path is unaffected — it has no inner Laplace
    # Hessian — so it keeps the caller's serialization.)
    serialization_use = SciMLBase.EnsembleSerial()

    fe = get_fixed(get_model(dm))
    free_names = _free_fixed_names(fe, constants_use)
    isempty(free_names) &&
        error("No free fixed effects are available for UQ after applying constants.")

    active_mask = _active_mask_for_free(fe, free_names)
    active_idx = findall(identity, active_mask)
    isempty(active_idx) &&
        error("No UQ-eligible fixed-effect coordinates found. Mark parameters with calculate_se=true and ensure they are not fixed via constants.")

    free_flat_names = _flat_names_for_free(fe, free_names)
    active_names = free_flat_names[active_idx]
    free_flat_kinds = _flat_transform_kinds_for_free(fe, free_names)
    active_kinds = free_flat_kinds[active_idx]

    θ_hat_u = get_params(res; scale = :untransformed)
    transform = get_transform(fe)
    inv_transform = get_inverse_transform(fe)
    θ_hat_t = transform(θ_hat_u)

    θ_const_u = deepcopy(θ_hat_u)
    _apply_constants!(θ_const_u, constants_use)
    θ_const_t = transform(θ_const_u)

    θ_hat_free_t = θ_hat_t[free_names]
    axs_free = getaxes(θ_hat_free_t)
    axs_full = getaxes(θ_const_t)
    xhat_full = Float64.(collect(θ_hat_free_t))
    xhat_active = xhat_full[active_idx]

    ll_cache = _build_ll_cache_uq(dm, ode_args_use, ode_kwargs_use, serialization_use)
    _, batch_infos, const_cache = _build_re_batch_infos(dm, constants_re_use)
    ebe_cache = _init_laplace_eval_cache(length(batch_infos), Float64)
    cache_opts = LaplaceCacheOptions(0.0)
    use_penalty = !isempty(keys(penalty_use))
    # FOCEI: differentiate the same Fisher-information Laplace objective the
    # optimizer minimized (NONMEM-style FOCEI vcov); otherwise exact inner Hessian.
    hmode_use = approx_method isa FOCEI ? _FOCEIHess(approx_method.interaction) :
                _ExactHess()
    seed = rand(rng, UInt64)

    function _θu_from_active(x_active::AbstractVector)
        T = eltype(x_active)
        x_full = T.(xhat_full)
        x_full[active_idx] .= x_active
        return _theta_u_from_free_t(
            x_full, axs_free, θ_const_t, axs_full, free_names, inv_transform)
    end

    function obj_active(x_active::AbstractVector)
        θu = _θu_from_active(x_active)

        obj = if approx_method isa GHQuadrature
            ll_cache_local = ll_cache isa AbstractVector ? ll_cache[1] : ll_cache
            total = 0.0
            for info in batch_infos
                bll = _ghq_batch_ll(dm, info,
                    _symmetrize_psd_params(θu, fe),
                    const_cache, ll_cache_local, approx_method.level)
                bll == -Inf && return Inf
                total += bll
            end
            -total
        else
            _laplace_objective_only(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                inner = approx_method.inner,
                hessian = approx_method.hessian,
                cache_opts = cache_opts,
                multistart = approx_method.multistart,
                rng = Random.Xoshiro(seed),
                serialization = serialization_use,
                hmode = hmode_use)
        end
        obj == Inf && return Inf

        use_penalty && (obj += _penalty_value(θu, penalty_use))
        return obj
    end

    hess_backend_use = if hessian_backend != :auto
        hessian_backend
    elseif approx_method isa GHQuadrature
        :forwarddiff
    else
        :fd_gradient
    end
    H_active, backend_used = _hessian_from_objective(obj_active, xhat_active;
        backend = hess_backend_use,
        fd_abs_step = fd_abs_step,
        fd_rel_step = fd_rel_step,
        fd_max_tries = fd_max_tries)
    H_active = 0.5 .* (H_active .+ H_active')

    bread = _wald_bread(H_active, pseudo_inverse, active_names)

    Vt_raw = if vcov == :hessian
        copy(bread)
    elseif vcov == :sandwich
        B = zeros(Float64, length(active_idx), length(active_idx))
        for (bi, info) in enumerate(batch_infos)
            info_single = REBatchInfo[info]
            ebe_cache_i = _init_laplace_eval_cache(1, Float64)
            seed_i = seed + UInt64(bi)
            obj_b = function (x_active::AbstractVector)
                θu = _θu_from_active(x_active)
                obj_bi = if approx_method isa GHQuadrature
                    ll_cache_local = ll_cache isa AbstractVector ? ll_cache[1] : ll_cache
                    bll = _ghq_batch_ll(dm, info_single[1],
                        _symmetrize_psd_params(θu, fe),
                        const_cache, ll_cache_local, approx_method.level)
                    bll == -Inf ? Inf : -bll
                else
                    _laplace_objective_only(
                        dm, info_single, θu, const_cache, ll_cache, ebe_cache_i;
                        inner = approx_method.inner,
                        hessian = approx_method.hessian,
                        cache_opts = cache_opts,
                        multistart = approx_method.multistart,
                        rng = Random.Xoshiro(seed_i),
                        serialization = serialization_use,
                        hmode = hmode_use)
                end
                return obj_bi == Inf ? Inf : Float64(obj_bi)
            end
            g = _gradient_fd_from_obj(obj_b, xhat_active;
                abs_step = fd_abs_step,
                rel_step = fd_rel_step,
                max_tries = fd_max_tries)
            B .+= g * g'
        end
        B = 0.5 .* (B .+ B')
        M = bread * B * bread'
        Matrix{Float64}(0.5 .* (M .+ M'))
    else
        error("Unsupported vcov=$(vcov). Use :hessian or :sandwich.")
    end

    return _finalize_wald_uqresult(fe, θ_hat_t, θ_hat_u, free_names, active_idx,
        active_names, active_kinds, _θu_from_active, Vt_raw, backend_used, vcov,
        pseudo_inverse, n_draws, level, rng, _method_symbol(source_method),
        (; approximation_method = _method_symbol(approx_method)))
end
