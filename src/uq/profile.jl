using LikelihoodProfiler
using OptimizationNLopt
using Distributions
using Random

# LikelihoodProfiler 1.x replaced the 0.x algorithm symbols by stepper objects; the
# historical `profile_method` names are kept as the user-facing selector.
function _profile_stepper(profile_method::Symbol)
    profile_method === :FIXED_STEP && return LikelihoodProfiler.FixedStep()
    profile_method === :LIN_EXTRAPOL &&
        return LikelihoodProfiler.AdaptiveStep(;
            predictor = LikelihoodProfiler.LinearPredictor())
    profile_method === :SINGLE_AXIS &&
        return LikelihoodProfiler.AdaptiveStep(;
            predictor = LikelihoodProfiler.SingleAxisPredictor())
    error("Unsupported profile_method $(profile_method). Supported values are :LIN_EXTRAPOL, :SINGLE_AXIS and :FIXED_STEP; the LikelihoodProfiler 0.x values :CICO_ONE_PASS and :QUADR_EXTRAPOL no longer exist.")
end

_warn_removed_profile_kw(::Symbol, ::Nothing) = nothing
function _warn_removed_profile_kw(name::Symbol, value)
    @warn "`$(name) = $(value)` is ignored and deprecated. It was a CICO scan tolerance of the LikelihoodProfiler 0.x backend and has no equivalent in the 1.x profiler; nothing is substituted for it. Drop the keyword; use `profile_scan_width`, `profile_max_iter` and `profile_ftol_abs` to control the profile search."
    return nothing
end

function _profile_optimizer(profile_local_alg::Symbol)
    isdefined(NLopt, profile_local_alg) ||
        error("Unknown profile_local_alg $(profile_local_alg); expected an NLopt algorithm such as :LN_NELDERMEAD.")
    return getfield(NLopt, profile_local_alg)()
end

# LikelihoodProfiler 1.x does not populate per-branch solver stats, so report -1 (unknown).
_profile_fevals(::Nothing) = -1
_profile_fevals(s) = s.fevals > 0 ? s.fevals : -1

@inline function _profile_scan_bounds(x0::Float64, lb::Float64, ub::Float64, width::Float64)
    width > 0 || error("profile_scan_width must be positive.")
    left = isfinite(lb) ? lb : x0 - width
    right = isfinite(ub) ? ub : x0 + width
    left = max(left, x0 - width)
    right = min(right, x0 + width)

    # Keep bounds strictly enclosing x0 for LikelihoodProfiler checks.
    ϵ = max(1e-8, abs(x0) * 1e-8)
    if !(left < x0)
        left = x0 - 10 * ϵ
    end
    if !(x0 < right)
        right = x0 + 10 * ϵ
    end

    if isfinite(lb)
        left = max(left, lb + ϵ)
    end
    if isfinite(ub)
        right = min(right, ub - ϵ)
    end
    left < x0 < right ||
        error("Unable to construct valid profile scan bounds around parameter estimate $(x0). Try larger profile_scan_width or relaxed bounds.")
    return (left, right)
end

function _build_uq_obj_no_re(res::FitResult,
        constants_use::NamedTuple,
        penalty_use::NamedTuple,
        ode_args_use::Tuple,
        ode_kwargs_use::NamedTuple,
        serialization_use::SciMLBase.EnsembleAlgorithm)
    dm = get_data_model(res)
    method = get_method(res)
    fe = get_fixed(get_model(dm))
    free_names = _free_fixed_names(fe, constants_use)
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

    ll_cache = _build_ll_cache_uq(dm, ode_args_use, ode_kwargs_use, serialization_use)
    use_penalty = !isempty(keys(penalty_use))
    use_prior = method isa MAP

    function obj_full(x::AbstractVector)
        θu = _theta_u_from_free_t(
            x, axs_free, θ_const_t, axs_full, free_names, inv_transform)
        ll = loglikelihood(
            dm, θu, ComponentArray(); cache = ll_cache, serialization = serialization_use)
        ll == -Inf && return Inf

        obj = -ll
        if use_prior
            lp = logprior(fe, θu)
            lp == -Inf && return Inf
            obj += -lp
        end
        use_penalty && (obj += _penalty_value(θu, penalty_use))
        return Float64(obj)
    end

    return (;
        dm = dm,
        fe = fe,
        free_names = free_names,
        θ_hat_u = θ_hat_u,
        θ_hat_t = θ_hat_t,
        inv_transform = inv_transform,
        axs_free = axs_free,
        axs_full = axs_full,
        θ_const_t = θ_const_t,
        xhat_full = xhat_full,
        obj_full = obj_full
    )
end

function _build_uq_obj_re(res::FitResult,
        constants_use::NamedTuple,
        constants_re_use::NamedTuple,
        penalty_use::NamedTuple,
        ode_args_use::Tuple,
        ode_kwargs_use::NamedTuple,
        serialization_use::SciMLBase.EnsembleAlgorithm,
        rng::AbstractRNG)
    dm = get_data_model(res)
    method = get_method(res)
    fe = get_fixed(get_model(dm))
    free_names = _free_fixed_names(fe, constants_use)
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

    ll_cache = _build_ll_cache_uq(dm, ode_args_use, ode_kwargs_use, serialization_use)
    _, batch_infos, const_cache = _build_re_batch_infos(dm, constants_re_use)
    ebe_cache = _init_laplace_eval_cache(length(batch_infos), Float64)
    cache_opts = LaplaceCacheOptions(0.0)
    use_penalty = !isempty(keys(penalty_use))
    seed = rand(rng, UInt64)

    function obj_full(x::AbstractVector)
        θu = _theta_u_from_free_t(
            x, axs_free, θ_const_t, axs_full, free_names, inv_transform)

        obj = if method isa GHQuadrature
            ll_cache_local = ll_cache isa AbstractVector ? ll_cache[1] : ll_cache
            total = 0.0
            for info in batch_infos
                bll = _ghq_batch_ll(dm, info,
                    _symmetrize_psd_params(θu, fe),
                    const_cache, ll_cache_local, method.level)
                bll == -Inf && return Inf
                total += bll
            end
            -total
        else
            _laplace_objective_only(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                inner = method.inner,
                hessian = method.hessian,
                cache_opts = cache_opts,
                multistart = method.multistart,
                rng = Random.Xoshiro(seed),
                serialization = serialization_use)
        end
        obj == Inf && return Inf

        use_penalty && (obj += _penalty_value(θu, penalty_use))
        return Float64(obj)
    end

    return (;
        dm = dm,
        fe = fe,
        free_names = free_names,
        θ_hat_u = θ_hat_u,
        θ_hat_t = θ_hat_t,
        inv_transform = inv_transform,
        axs_free = axs_free,
        axs_full = axs_full,
        θ_const_t = θ_const_t,
        xhat_full = xhat_full,
        obj_full = obj_full
    )
end

function _compute_uq_profile(res::FitResult;
        level::Float64,
        constants::Union{Nothing, NamedTuple},
        constants_re::Union{Nothing, NamedTuple},
        penalty::Union{Nothing, NamedTuple},
        ode_args::Union{Nothing, Tuple},
        ode_kwargs::Union{Nothing, NamedTuple},
        serialization::Union{Nothing, SciMLBase.EnsembleAlgorithm},
        profile_method::Symbol,
        profile_scan_width::Real,
        profile_scan_tol::Union{Nothing, Real},
        profile_loss_tol::Union{Nothing, Real},
        profile_local_alg::Symbol,
        profile_max_iter::Int,
        profile_ftol_abs::Real,
        profile_kwargs::NamedTuple,
        rng::AbstractRNG)
    dm = get_data_model(res)
    dm === nothing &&
        error("This fit result does not store a DataModel; pass store_data_model=true when fitting.")
    method = get_method(res)
    if !(method isa MLE || method isa MAP || method isa Laplace ||
         method isa GHQuadrature)
        error("Profile UQ is currently supported for MLE, MAP, Laplace, and GHQuadrature fit results.")
    end

    _warn_removed_profile_kw(:profile_scan_tol, profile_scan_tol)
    _warn_removed_profile_kw(:profile_loss_tol, profile_loss_tol)

    constants_use = _resolve_fit_kw(res, constants, :constants, NamedTuple())
    constants_re_use = _resolve_fit_kw(res, constants_re, :constants_re, NamedTuple())
    penalty_use = _resolve_fit_kw(res, penalty, :penalty, NamedTuple())
    ode_args_use = _resolve_fit_kw(res, ode_args, :ode_args, ())
    ode_kwargs_use = _resolve_fit_kw(res, ode_kwargs, :ode_kwargs, NamedTuple())
    serialization_use = _resolve_fit_kw(
        res, serialization, :serialization, EnsembleSerial())

    ctx = if method isa MLE || method isa MAP
        _build_uq_obj_no_re(res, constants_use, penalty_use, ode_args_use,
            ode_kwargs_use, serialization_use)
    else  # Laplace, GHQuadrature
        _build_uq_obj_re(res, constants_use, constants_re_use, penalty_use,
            ode_args_use, ode_kwargs_use, serialization_use, rng)
    end

    fe = ctx.fe
    free_names = ctx.free_names
    θ_hat_u = ctx.θ_hat_u
    θ_hat_t = ctx.θ_hat_t
    inv_transform = ctx.inv_transform
    axs_free = ctx.axs_free
    axs_full = ctx.axs_full
    θ_const_t = ctx.θ_const_t
    xhat_full = ctx.xhat_full
    obj_full = ctx.obj_full

    active_mask = _active_mask_for_free(fe, free_names)
    active_idx = findall(identity, active_mask)
    isempty(active_idx) &&
        error("No UQ-eligible fixed-effect coordinates found. Mark parameters with calculate_se=true and ensure they are not fixed via constants.")
    free_flat_names = _flat_names_for_free(fe, free_names)
    active_names = free_flat_names[active_idx]

    xhat_active = xhat_full[active_idx]
    obj_active = function (x_active::Vector{Float64})
        x_full = copy(xhat_full)
        x_full[active_idx] .= x_active
        return obj_full(x_full)
    end

    obj0 = obj_active(xhat_active)
    isfinite(obj0) ||
        error("Objective at fitted parameters is not finite; profile UQ cannot proceed.")
    # The objective is -loglik, so the profile threshold is half the χ² quantile.
    threshold = 0.5 * quantile(Chisq(1), level)
    loss_crit = obj0 + threshold

    lower_t, upper_t = get_bounds_transformed(fe)
    lb_coords = _coords_on_transformed_layout(fe, lower_t, free_names; natural = false)[active_idx]
    ub_coords = _coords_on_transformed_layout(fe, upper_t, free_names; natural = false)[active_idx]

    p = length(xhat_active)
    lower_prof_t = fill(NaN, p)
    upper_prof_t = fill(NaN, p)
    left_status = Vector{Symbol}(undef, p)
    right_status = Vector{Symbol}(undef, p)
    left_counter = fill(-1, p)
    right_counter = fill(-1, p)
    endpoint_found = falses(p)
    errors = Vector{Union{Nothing, String}}(undef, p)

    optprob = OptimizationProblem(
        OptimizationFunction((x, _p) -> obj_active(collect(x))), copy(xhat_active);
        lb = collect(lb_coords), ub = collect(ub_coords))
    profiler = LikelihoodProfiler.OptimizationProfiler(;
        stepper = _profile_stepper(profile_method),
        optimizer = _profile_optimizer(profile_local_alg),
        optimizer_opts = (;
            maxiters = profile_max_iter, abstol = Float64(profile_ftol_abs)))

    for j in 1:p
        errors[j] = nothing
        scan_lo, scan_hi = _profile_scan_bounds(
            xhat_active[j], lb_coords[j], ub_coords[j], Float64(profile_scan_width))
        sol = try
            plprob = LikelihoodProfiler.ProfileLikelihoodProblem(
                optprob, copy(xhat_active); idxs = j,
                profile_lower = scan_lo, profile_upper = scan_hi, threshold = threshold)
            solve(plprob, profiler; profile_kwargs...)
        catch err
            errors[j] = sprint(showerror, err)
            left_status[j] = :ERROR
            right_status[j] = :ERROR
            @warn "Profile UQ failed for $(active_names[j]); its interval is NaN." error=errors[j]
            continue
        end

        curve = sol[1]
        rc = LikelihoodProfiler.retcodes(curve)
        ep = LikelihoodProfiler.endpoints(curve)
        st = LikelihoodProfiler.stats(curve)
        left_status[j] = rc.left
        right_status[j] = rc.right
        left_counter[j] = _profile_fevals(st.left)
        right_counter[j] = _profile_fevals(st.right)

        ep.left === nothing || (lower_prof_t[j] = Float64(ep.left))
        ep.right === nothing || (upper_prof_t[j] = Float64(ep.right))
        endpoint_found[j] = isfinite(lower_prof_t[j]) && isfinite(upper_prof_t[j])
        endpoint_found[j] ||
            @warn "Profile UQ did not locate both endpoints for $(active_names[j]); try a larger profile_scan_width or profile_max_iter." left_status=rc.left right_status=rc.right
    end

    θ_coords_t = _coords_on_transformed_layout(fe, θ_hat_t, free_names; natural = false)
    θ_coords_u = _coords_on_transformed_layout(fe, θ_hat_u, free_names; natural = true)
    est_t = θ_coords_t[active_idx]
    est_n = θ_coords_u[active_idx]

    lower_prof_n = fill(NaN, p)
    upper_prof_n = fill(NaN, p)
    x_work = copy(xhat_full)
    for j in 1:p
        if isfinite(lower_prof_t[j])
            x_work[active_idx] .= xhat_active
            x_work[active_idx[j]] = lower_prof_t[j]
            θu_j = _theta_u_from_free_t(
                x_work, axs_free, θ_const_t, axs_full, free_names, inv_transform)
            coords_u_j = _coords_on_transformed_layout(fe, θu_j, free_names; natural = true)
            lower_prof_n[j] = coords_u_j[active_idx[j]]
        end
        if isfinite(upper_prof_t[j])
            x_work[active_idx] .= xhat_active
            x_work[active_idx[j]] = upper_prof_t[j]
            θu_j = _theta_u_from_free_t(
                x_work, axs_free, θ_const_t, axs_full, free_names, inv_transform)
            coords_u_j = _coords_on_transformed_layout(fe, θu_j, free_names; natural = true)
            upper_prof_n[j] = coords_u_j[active_idx[j]]
        end
    end

    intervals_t = UQIntervals(level, lower_prof_t, upper_prof_t)
    intervals_n = UQIntervals(level, lower_prof_n, upper_prof_n)
    diag = (;
        profile_method = profile_method,
        profile_scan_width = Float64(profile_scan_width),
        profile_local_alg = profile_local_alg,
        profile_max_iter = profile_max_iter,
        profile_ftol_abs = Float64(profile_ftol_abs),
        loss_at_estimate = obj0,
        loss_critical = loss_crit,
        left_status = left_status,
        right_status = right_status,
        left_counter = left_counter,
        right_counter = right_counter,
        endpoint_found = endpoint_found,
        errors = errors
    )

    return UQResult(
        :profile,
        _method_symbol(method),
        active_names,
        nothing,
        est_t,
        est_n,
        intervals_t,
        intervals_n,
        nothing,
        nothing,
        nothing,
        nothing,
        diag
    )
end
