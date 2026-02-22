using LinearAlgebra
using Random

@inline function _is_re_laplace_family(method::FittingMethod)
    return method isa Laplace || method isa LaplaceMAP || method isa FOCEI || method isa FOCEIMAP
end

function _resolve_wald_re_approx_method(source_method::FittingMethod;
                                        re_approx::Symbol,
                                        re_approx_method::Union{Nothing, FittingMethod})
    if _is_re_laplace_family(source_method)
        re_approx == :auto ||
            error("re_approx is only used for MCEM/SAEM Wald UQ results.")
        re_approx_method === nothing ||
            error("re_approx_method is only used for MCEM/SAEM Wald UQ results.")
        return source_method
    end

    if !(source_method isa MCEM || source_method isa SAEM)
        error("Wald UQ for random-effects models currently supports Laplace, LaplaceMAP, FOCEI, FOCEIMAP, MCEM, and SAEM.")
    end

    if re_approx_method !== nothing
        _is_re_laplace_family(re_approx_method) ||
            error("re_approx_method must be a Laplace/LaplaceMAP/FOCEI/FOCEIMAP method instance.")
        return re_approx_method
    end

    approx = re_approx == :auto ? :laplace : re_approx
    if approx == :laplace
        return Laplace()
    elseif approx == :focei
        return FOCEI()
    end
    error("For MCEM/SAEM Wald UQ, re_approx must be :auto, :laplace, or :focei.")
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
    dm === nothing && error("This fit result does not store a DataModel; pass store_data_model=true when fitting.")
    method = get_method(res)
    (method isa MLE || method isa MAP) || error("Wald UQ is currently supported for MLE and MAP fits in this rollout.")

    constants_use = constants === nothing ? _fit_kw(res, :constants, NamedTuple()) : constants
    penalty_use = penalty === nothing ? _fit_kw(res, :penalty, NamedTuple()) : penalty
    ode_args_use = ode_args === nothing ? _fit_kw(res, :ode_args, ()) : ode_args
    ode_kwargs_use = ode_kwargs === nothing ? _fit_kw(res, :ode_kwargs, NamedTuple()) : ode_kwargs
    serialization_use = serialization === nothing ? _fit_kw(res, :serialization, EnsembleSerial()) : serialization

    fe = dm.model.fixed.fixed
    free_names = _free_fixed_names(fe, constants_use)
    isempty(free_names) && error("No free fixed effects are available for UQ after applying constants.")

    active_mask = _active_mask_for_free(fe, free_names)
    active_idx = findall(identity, active_mask)
    isempty(active_idx) && error("No UQ-eligible fixed-effect coordinates found. Mark parameters with calculate_se=true and ensure they are not fixed via constants.")

    free_flat_names = _flat_names_for_free(fe, free_names)
    active_names = free_flat_names[active_idx]
    free_flat_kinds = _flat_transform_kinds_for_free(fe, free_names)
    active_kinds = free_flat_kinds[active_idx]

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
    xhat_active = xhat_full[active_idx]
    ll_cache = _build_ll_cache_uq(dm, ode_args_use, ode_kwargs_use, serialization_use)
    use_penalty = !isempty(keys(penalty_use))
    use_prior = method isa MAP

    function _θu_from_active(x_active::AbstractVector)
        T = eltype(x_active)
        x_full = T.(xhat_full)
        x_full[active_idx] .= x_active
        θt_free = ComponentArray(x_full, axs_free)
        T = eltype(θt_free)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        return _as_component_array(inv_transform(θt_full))
    end

    function obj_active(x_active::AbstractVector)
        θu = _θu_from_active(x_active)
        ll = loglikelihood(dm, θu, ComponentArray(); cache=ll_cache, serialization=serialization_use)
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
                                                     backend=hessian_backend,
                                                     fd_abs_step=fd_abs_step,
                                                     fd_rel_step=fd_rel_step,
                                                     fd_max_tries=fd_max_tries)
    H_active = 0.5 .* (H_active .+ H_active')

    bread = try
        pseudo_inverse ? pinv(H_active) : inv(H_active)
    catch err
        pseudo_inverse || error("Failed to invert Hessian for Wald covariance. Consider pseudo_inverse=true. Original error: $(sprint(showerror, err))")
        @warn "Falling back to pseudo-inverse for Hessian inversion in UQ." error=sprint(showerror, err)
        pinv(H_active)
    end
    bread = Matrix{Float64}(0.5 .* (bread .+ bread'))

    Vt_raw = if vcov == :hessian
        copy(bread)
    elseif vcov == :sandwich
        ll_cache_local = ll_cache isa Vector ? ll_cache[1] : ll_cache
        B = zeros(Float64, length(active_idx), length(active_idx))
        for i in eachindex(dm.individuals)
            obj_i = function (x_active::AbstractVector)
                θu = _θu_from_active(x_active)
                ll_i = _loglikelihood_individual(dm, i, θu, ComponentArray(), ll_cache_local)
                ll_i == -Inf && return Inf
                return Float64(-ll_i)
            end
            g = _gradient_from_objective(obj_i, xhat_active;
                                         fd_abs_step=fd_abs_step,
                                         fd_rel_step=fd_rel_step,
                                         fd_max_tries=fd_max_tries)
            B .+= g * g'
        end
        B = 0.5 .* (B .+ B')
        Matrix{Float64}(0.5 .* ((bread * B * bread') .+ (bread * B * bread')'))
    else
        error("Unsupported vcov=$(vcov). Use :hessian or :sandwich.")
    end
    Vt, vcov_diag = _project_psd_covariance(Vt_raw)

    θ_coords_t = _coords_on_transformed_layout(fe, θ_hat_t, free_names; natural=false)
    θ_coords_u = _coords_on_transformed_layout(fe, θ_hat_u, free_names; natural=true)
    est_t = θ_coords_t[active_idx]
    est_n = θ_coords_u[active_idx]

    draws_t = _sample_gaussian_draws(rng, est_t, Vt, n_draws)
    draws_n = Matrix{Float64}(undef, size(draws_t, 1), size(draws_t, 2))
    for i in 1:size(draws_t, 1)
        θu_i = _θu_from_active(@view(draws_t[i, :]))
        coords_u_i = _coords_on_transformed_layout(fe, θu_i, free_names; natural=true)
        draws_n[i, :] .= coords_u_i[active_idx]
    end

    intervals_t = _intervals_from_draws(draws_t, level)
    intervals_n = _intervals_from_draws(draws_n, level)
    Vn = _cov_from_draws(draws_n)

    ext = _extend_natural_stickbreak(fe, free_names, active_names, active_kinds,
                                     est_n, draws_n, intervals_n)
    names_n    = ext !== nothing ? ext[1] : nothing
    est_n_use  = ext !== nothing ? ext[2] : est_n
    draws_n_use = ext !== nothing ? ext[3] : draws_n
    intervals_n_use = ext !== nothing ? ext[4] : intervals_n
    Vn_use = draws_n_use !== nothing ? _cov_from_draws(draws_n_use) : Vn

    diag = merge((;
        hessian_backend=backend_used,
        hessian_reduced=true,
        inactive_fixed_effects_held_constant=true,
        vcov=vcov,
        pseudo_inverse=pseudo_inverse,
        n_draws=n_draws,
        n_active_parameters=length(active_idx),
        coordinate_transforms=active_kinds,
    ), vcov_diag)

    return UQResult(
        :wald,
        _method_symbol(method),
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
    dm === nothing && error("This fit result does not store a DataModel; pass store_data_model=true when fitting.")
    source_method = get_method(res)
    approx_method = _resolve_wald_re_approx_method(source_method;
                                                   re_approx=re_approx,
                                                   re_approx_method=re_approx_method)

    constants_use = constants === nothing ? _fit_kw(res, :constants, NamedTuple()) : constants
    constants_re_use = constants_re === nothing ? _fit_kw(res, :constants_re, NamedTuple()) : constants_re
    penalty_use = penalty === nothing ? _fit_kw(res, :penalty, NamedTuple()) : penalty
    ode_args_use = ode_args === nothing ? _fit_kw(res, :ode_args, ()) : ode_args
    ode_kwargs_use = ode_kwargs === nothing ? _fit_kw(res, :ode_kwargs, NamedTuple()) : ode_kwargs
    serialization_use = serialization === nothing ? _fit_kw(res, :serialization, EnsembleSerial()) : serialization

    fe = dm.model.fixed.fixed
    free_names = _free_fixed_names(fe, constants_use)
    isempty(free_names) && error("No free fixed effects are available for UQ after applying constants.")

    active_mask = _active_mask_for_free(fe, free_names)
    active_idx = findall(identity, active_mask)
    isempty(active_idx) && error("No UQ-eligible fixed-effect coordinates found. Mark parameters with calculate_se=true and ensure they are not fixed via constants.")

    free_flat_names = _flat_names_for_free(fe, free_names)
    active_names = free_flat_names[active_idx]
    free_flat_kinds = _flat_transform_kinds_for_free(fe, free_names)
    active_kinds = free_flat_kinds[active_idx]

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
    xhat_active = xhat_full[active_idx]

    ll_cache = _build_ll_cache_uq(dm, ode_args_use, ode_kwargs_use, serialization_use)
    _, batch_infos, const_cache = _build_laplace_batch_infos(dm, constants_re_use)
    ebe_cache = _init_laplace_eval_cache(length(batch_infos), Float64)
    cache_opts = LaplaceCacheOptions(0.0)
    use_penalty = !isempty(keys(penalty_use))
    use_prior = approx_method isa LaplaceMAP || approx_method isa FOCEIMAP
    fallback_tracker = approx_method isa FOCEI || approx_method isa FOCEIMAP ? _FOCEIFallbackTracker() : nothing
    seed = rand(rng, UInt64)

    function _θu_from_active(x_active::AbstractVector)
        T = eltype(x_active)
        x_full = T.(xhat_full)
        x_full[active_idx] .= x_active
        θt_free = ComponentArray(x_full, axs_free)
        T = eltype(θt_free)
        θt_full = ComponentArray(T.(θ_const_t), axs_full)
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        return _as_component_array(inv_transform(θt_full))
    end

    function obj_active(x_active::AbstractVector)
        θu = _θu_from_active(x_active)

        obj = if approx_method isa Laplace || approx_method isa LaplaceMAP
            _laplace_objective_only(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                    inner=approx_method.inner,
                                    hessian=approx_method.hessian,
                                    cache_opts=cache_opts,
                                    multistart=approx_method.multistart,
                                    rng=Random.Xoshiro(seed),
                                    serialization=serialization_use)
        else
            _focei_objective_only(dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
                                  inner=approx_method.inner,
                                  info_opts=approx_method.info,
                                  fallback_hessian=approx_method.fallback_hessian,
                                  cache_opts=cache_opts,
                                  multistart=approx_method.multistart,
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
        return obj
    end

    hess_backend_use = hessian_backend == :auto ? :fd_gradient : hessian_backend
    H_active, backend_used = _hessian_from_objective(obj_active, xhat_active;
                                                     backend=hess_backend_use,
                                                     fd_abs_step=fd_abs_step,
                                                     fd_rel_step=fd_rel_step,
                                                     fd_max_tries=fd_max_tries)
    H_active = 0.5 .* (H_active .+ H_active')

    bread = try
        pseudo_inverse ? pinv(H_active) : inv(H_active)
    catch err
        pseudo_inverse || error("Failed to invert Hessian for Wald covariance. Consider pseudo_inverse=true. Original error: $(sprint(showerror, err))")
        @warn "Falling back to pseudo-inverse for Hessian inversion in UQ." error=sprint(showerror, err)
        pinv(H_active)
    end
    bread = Matrix{Float64}(0.5 .* (bread .+ bread'))

    Vt_raw = if vcov == :hessian
        copy(bread)
    elseif vcov == :sandwich
        B = zeros(Float64, length(active_idx), length(active_idx))
        for (bi, info) in enumerate(batch_infos)
            info_single = _LaplaceBatchInfo[info]
            ebe_cache_i = _init_laplace_eval_cache(1, Float64)
            seed_i = seed + UInt64(bi)
            obj_b = function (x_active::AbstractVector)
                θu = _θu_from_active(x_active)
                obj_bi = if approx_method isa Laplace || approx_method isa LaplaceMAP
                    _laplace_objective_only(dm, info_single, θu, const_cache, ll_cache, ebe_cache_i;
                                            inner=approx_method.inner,
                                            hessian=approx_method.hessian,
                                            cache_opts=cache_opts,
                                            multistart=approx_method.multistart,
                                            rng=Random.Xoshiro(seed_i),
                                            serialization=serialization_use)
                else
                    _focei_objective_only(dm, info_single, θu, const_cache, ll_cache, ebe_cache_i;
                                          inner=approx_method.inner,
                                          info_opts=approx_method.info,
                                          fallback_hessian=approx_method.fallback_hessian,
                                          cache_opts=cache_opts,
                                          multistart=approx_method.multistart,
                                          rng=Random.Xoshiro(seed_i),
                                          fallback_tracker=nothing,
                                          serialization=serialization_use)
                end
                return obj_bi == Inf ? Inf : Float64(obj_bi)
            end
            g = _gradient_fd_from_obj(obj_b, xhat_active;
                                      abs_step=fd_abs_step,
                                      rel_step=fd_rel_step,
                                      max_tries=fd_max_tries)
            B .+= g * g'
        end
        B = 0.5 .* (B .+ B')
        Matrix{Float64}(0.5 .* ((bread * B * bread') .+ (bread * B * bread')'))
    else
        error("Unsupported vcov=$(vcov). Use :hessian or :sandwich.")
    end
    Vt, vcov_diag = _project_psd_covariance(Vt_raw)

    θ_coords_t = _coords_on_transformed_layout(fe, θ_hat_t, free_names; natural=false)
    θ_coords_u = _coords_on_transformed_layout(fe, θ_hat_u, free_names; natural=true)
    est_t = θ_coords_t[active_idx]
    est_n = θ_coords_u[active_idx]

    draws_t = _sample_gaussian_draws(rng, est_t, Vt, n_draws)
    draws_n = Matrix{Float64}(undef, size(draws_t, 1), size(draws_t, 2))
    for i in 1:size(draws_t, 1)
        θu_i = _θu_from_active(@view(draws_t[i, :]))
        coords_u_i = _coords_on_transformed_layout(fe, θu_i, free_names; natural=true)
        draws_n[i, :] .= coords_u_i[active_idx]
    end

    intervals_t = _intervals_from_draws(draws_t, level)
    intervals_n = _intervals_from_draws(draws_n, level)
    Vn = _cov_from_draws(draws_n)

    ext = _extend_natural_stickbreak(fe, free_names, active_names, active_kinds,
                                     est_n, draws_n, intervals_n)
    names_n    = ext !== nothing ? ext[1] : nothing
    est_n_use  = ext !== nothing ? ext[2] : est_n
    draws_n_use = ext !== nothing ? ext[3] : draws_n
    intervals_n_use = ext !== nothing ? ext[4] : intervals_n
    Vn_use = draws_n_use !== nothing ? _cov_from_draws(draws_n_use) : Vn

    diag = if fallback_tracker === nothing
        merge((;
            hessian_backend=backend_used,
            hessian_reduced=true,
            inactive_fixed_effects_held_constant=true,
            vcov=vcov,
            approximation_method=_method_symbol(approx_method),
            pseudo_inverse=pseudo_inverse,
            n_draws=n_draws,
            n_active_parameters=length(active_idx),
            coordinate_transforms=active_kinds,
        ), vcov_diag)
    else
        fb = _focei_fallback_notes(fallback_tracker)
        merge((;
            hessian_backend=backend_used,
            hessian_reduced=true,
            inactive_fixed_effects_held_constant=true,
            vcov=vcov,
            approximation_method=_method_symbol(approx_method),
            pseudo_inverse=pseudo_inverse,
            n_draws=n_draws,
            n_active_parameters=length(active_idx),
            coordinate_transforms=active_kinds,
            focei_fallback_total=fb.focei_fallback_total,
            focei_fallback_info_logdet=fb.focei_fallback_info_logdet,
            focei_fallback_mode_hessian=fb.focei_fallback_mode_hessian,
        ), vcov_diag)
    end

    return UQResult(
        :wald,
        _method_symbol(source_method),
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
