using LikelihoodProfiler
using Distributions
using Random

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
    left < x0 < right || error("Unable to construct valid profile scan bounds around parameter estimate $(x0). Try larger profile_scan_width or relaxed bounds.")
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
    fe = dm.model.fixed.fixed
    free_names = _free_fixed_names(fe, constants_use)
    θ_hat_u = get_params(res; scale=:untransformed)
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
        θt_free = ComponentArray(x, axs_free)
        T = eltype(θt_free)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu = _as_component_array(inv_transform(θt_full))
        ll = loglikelihood(dm, θu, ComponentArray(); cache=ll_cache, serialization=serialization_use)
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
        dm=dm,
        fe=fe,
        free_names=free_names,
        θ_hat_u=θ_hat_u,
        θ_hat_t=θ_hat_t,
        inv_transform=inv_transform,
        axs_free=axs_free,
        axs_full=axs_full,
        θ_const_t=θ_const_t,
        xhat_full=xhat_full,
        obj_full=obj_full,
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
    fe = dm.model.fixed.fixed
    free_names = _free_fixed_names(fe, constants_use)
    θ_hat_u = get_params(res; scale=:untransformed)
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
    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re_use)
    ebe_cache = _init_laplace_eval_cache(length(batch_infos), Float64)
    cache_opts = LaplaceCacheOptions(0.0)
    use_penalty = !isempty(keys(penalty_use))
    use_prior = method isa LaplaceMAP || method isa FOCEIMAP
    fallback_tracker = method isa FOCEI || method isa FOCEIMAP ? _FOCEIFallbackTracker() : nothing
    seed = rand(rng, UInt64)

    function obj_full(x::AbstractVector)
        θt_free = ComponentArray(x, axs_free)
        T = eltype(θt_free)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu = _as_component_array(inv_transform(θt_full))

        obj = if method isa Laplace || method isa LaplaceMAP
            _laplace_objective_only(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                    inner=method.inner,
                                    hessian=method.hessian,
                                    cache_opts=cache_opts,
                                    multistart=method.multistart,
                                    rng=Random.Xoshiro(seed),
                                    serialization=serialization_use)
        else
            _focei_objective_only(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                  inner=method.inner,
                                  info_opts=method.info,
                                  fallback_hessian=method.fallback_hessian,
                                  cache_opts=cache_opts,
                                  multistart=method.multistart,
                                  rng=Random.Xoshiro(seed),
                                  fallback_tracker=fallback_tracker,
                                  serialization=serialization_use)
        end
        obj == Inf && return Inf

        if use_prior
            lp = logprior(fe, θu)
            lp == -Inf && return Inf
            obj += -lp
        end
        use_penalty && (obj += _penalty_value(θu, penalty_use))
        return Float64(obj)
    end

    return (;
        dm=dm,
        fe=fe,
        free_names=free_names,
        θ_hat_u=θ_hat_u,
        θ_hat_t=θ_hat_t,
        inv_transform=inv_transform,
        axs_free=axs_free,
        axs_full=axs_full,
        θ_const_t=θ_const_t,
        xhat_full=xhat_full,
        obj_full=obj_full,
        fallback_tracker=fallback_tracker,
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
                             profile_scan_tol::Real,
                             profile_loss_tol::Real,
                             profile_local_alg::Symbol,
                             profile_max_iter::Int,
                             profile_ftol_abs::Real,
                             profile_kwargs::NamedTuple,
                             rng::AbstractRNG)
    dm = get_data_model(res)
    dm === nothing && error("This fit result does not store a DataModel; pass store_data_model=true when fitting.")
    method = get_method(res)
    if !(method isa MLE || method isa MAP || method isa Laplace || method isa LaplaceMAP || method isa FOCEI || method isa FOCEIMAP)
        error("Profile UQ is currently supported for MLE, MAP, Laplace, LaplaceMAP, FOCEI, and FOCEIMAP fit results.")
    end

    constants_use = constants === nothing ? _fit_kw(res, :constants, NamedTuple()) : constants
    constants_re_use = constants_re === nothing ? _fit_kw(res, :constants_re, NamedTuple()) : constants_re
    penalty_use = penalty === nothing ? _fit_kw(res, :penalty, NamedTuple()) : penalty
    ode_args_use = ode_args === nothing ? _fit_kw(res, :ode_args, ()) : ode_args
    ode_kwargs_use = ode_kwargs === nothing ? _fit_kw(res, :ode_kwargs, NamedTuple()) : ode_kwargs
    serialization_use = serialization === nothing ? _fit_kw(res, :serialization, EnsembleSerial()) : serialization

    ctx = if method isa MLE || method isa MAP
        _build_uq_obj_no_re(res, constants_use, penalty_use, ode_args_use, ode_kwargs_use, serialization_use)
    else
        _build_uq_obj_re(res, constants_use, constants_re_use, penalty_use, ode_args_use, ode_kwargs_use, serialization_use, rng)
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
    isempty(active_idx) && error("No UQ-eligible fixed-effect coordinates found. Mark parameters with calculate_se=true and ensure they are not fixed via constants.")
    free_flat_names = _flat_names_for_free(fe, free_names)
    active_names = free_flat_names[active_idx]

    xhat_active = xhat_full[active_idx]
    obj_active = function (x_active::Vector{Float64})
        x_full = copy(xhat_full)
        x_full[active_idx] .= x_active
        return obj_full(x_full)
    end

    obj0 = obj_active(xhat_active)
    isfinite(obj0) || error("Objective at fitted parameters is not finite; profile UQ cannot proceed.")
    loss_crit = obj0 + 0.5 * quantile(Chisq(1), level)

    lower_t, upper_t = get_bounds_transformed(fe)
    lb_coords = _coords_on_transformed_layout(fe, lower_t, free_names; natural=false)[active_idx]
    ub_coords = _coords_on_transformed_layout(fe, upper_t, free_names; natural=false)[active_idx]
    theta_bounds = [(lb_coords[j], ub_coords[j]) for j in eachindex(lb_coords)]

    p = length(xhat_active)
    lower_prof_t = fill(NaN, p)
    upper_prof_t = fill(NaN, p)
    left_status = Vector{Symbol}(undef, p)
    right_status = Vector{Symbol}(undef, p)
    left_counter = fill(-1, p)
    right_counter = fill(-1, p)
    endpoint_found = falses(p)
    errors = Vector{Union{Nothing, String}}(undef, p)

    for j in 1:p
        errors[j] = nothing
        bounds_j = theta_bounds[j]
        scan_bounds_j = _profile_scan_bounds(xhat_active[j], bounds_j[1], bounds_j[2], Float64(profile_scan_width))
        interval = try
            LikelihoodProfiler.get_interval(
                copy(xhat_active),
                j,
                obj_active,
                profile_method;
                loss_crit=loss_crit,
                scale=fill(:direct, p),
                theta_bounds=theta_bounds,
                scan_bounds=scan_bounds_j,
                scan_tol=Float64(profile_scan_tol),
                loss_tol=Float64(profile_loss_tol),
                local_alg=profile_local_alg,
                max_iter=profile_max_iter,
                ftol_abs=Float64(profile_ftol_abs),
                profile_kwargs...,
            )
        catch err
            errors[j] = sprint(showerror, err)
            left_status[j] = :ERROR
            right_status[j] = :ERROR
            continue
        end

        left = interval.result[1]
        right = interval.result[2]
        left_status[j] = left.status
        right_status[j] = right.status
        left_counter[j] = left.counter
        right_counter[j] = right.counter

        if left.value !== nothing
            lower_prof_t[j] = Float64(left.value)
        end
        if right.value !== nothing
            upper_prof_t[j] = Float64(right.value)
        end
        endpoint_found[j] = isfinite(lower_prof_t[j]) && isfinite(upper_prof_t[j])
    end

    θ_coords_t = _coords_on_transformed_layout(fe, θ_hat_t, free_names; natural=false)
    θ_coords_u = _coords_on_transformed_layout(fe, θ_hat_u, free_names; natural=true)
    est_t = θ_coords_t[active_idx]
    est_n = θ_coords_u[active_idx]

    lower_prof_n = fill(NaN, p)
    upper_prof_n = fill(NaN, p)
    x_work = copy(xhat_full)
    for j in 1:p
        if isfinite(lower_prof_t[j])
            x_work[active_idx] .= xhat_active
            x_work[active_idx[j]] = lower_prof_t[j]
            θt_free_j = ComponentArray(x_work, axs_free)
            θt_full_j = ComponentArray(eltype(θt_free_j).(θ_const_t), axs_full)
            for name in free_names
                setproperty!(θt_full_j, name, getproperty(θt_free_j, name))
            end
            θu_j = _as_component_array(inv_transform(θt_full_j))
            coords_u_j = _coords_on_transformed_layout(fe, θu_j, free_names; natural=true)
            lower_prof_n[j] = coords_u_j[active_idx[j]]
        end
        if isfinite(upper_prof_t[j])
            x_work[active_idx] .= xhat_active
            x_work[active_idx[j]] = upper_prof_t[j]
            θt_free_j = ComponentArray(x_work, axs_free)
            θt_full_j = ComponentArray(eltype(θt_free_j).(θ_const_t), axs_full)
            for name in free_names
                setproperty!(θt_full_j, name, getproperty(θt_free_j, name))
            end
            θu_j = _as_component_array(inv_transform(θt_full_j))
            coords_u_j = _coords_on_transformed_layout(fe, θu_j, free_names; natural=true)
            upper_prof_n[j] = coords_u_j[active_idx[j]]
        end
    end

    intervals_t = UQIntervals(level, lower_prof_t, upper_prof_t)
    intervals_n = UQIntervals(level, lower_prof_n, upper_prof_n)
    diag = (;
        profile_method=profile_method,
        profile_scan_width=Float64(profile_scan_width),
        profile_scan_tol=Float64(profile_scan_tol),
        profile_loss_tol=Float64(profile_loss_tol),
        profile_local_alg=profile_local_alg,
        profile_max_iter=profile_max_iter,
        profile_ftol_abs=Float64(profile_ftol_abs),
        loss_at_estimate=obj0,
        loss_critical=loss_crit,
        left_status=left_status,
        right_status=right_status,
        left_counter=left_counter,
        right_counter=right_counter,
        endpoint_found=endpoint_found,
        errors=errors,
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
