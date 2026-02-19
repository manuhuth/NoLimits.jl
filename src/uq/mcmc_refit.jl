using Turing
using Random

function _mcmc_refit_method(method::Union{Nothing, MCMC},
                            sampler,
                            turing_kwargs::NamedTuple,
                            adtype)
    method !== nothing && return method
    return MCMC(; sampler=sampler, turing_kwargs=turing_kwargs, adtype=adtype)
end

function _mcmc_refit_constants(fe::FixedEffects,
                               θ_hat_u::ComponentArray,
                               constants_user::NamedTuple,
                               free_names::Vector{Symbol})
    params = get_params(fe)
    pairs = Pair{Symbol, Any}[]
    for name in keys(constants_user)
        push!(pairs, name => getfield(constants_user, name))
    end
    user_set = Set(keys(constants_user))
    for name in free_names
        getfield(params, name).calculate_se && continue
        name in user_set && continue
        push!(pairs, name => getproperty(θ_hat_u, name))
    end
    return NamedTuple(pairs)
end

function _validate_mcmc_refit_priors(fe::FixedEffects, sampled_fixed_names::Vector{Symbol})
    isempty(sampled_fixed_names) && return
    priors = get_priors(fe)
    for name in sampled_fixed_names
        hasproperty(priors, name) || error("mcmc_refit requires priors on sampled fixed effects. Missing prior for $(name).")
        getfield(priors, name) isa Priorless &&
            error("mcmc_refit requires priors on sampled fixed effects. Priorless for $(name). Add priors or set calculate_se=false/constant for this parameter.")
    end
end

function _compute_uq_mcmc_refit(res::FitResult;
                                level::Float64,
                                constants::Union{Nothing, NamedTuple},
                                constants_re::Union{Nothing, NamedTuple},
                                ode_args::Union{Nothing, Tuple},
                                ode_kwargs::Union{Nothing, NamedTuple},
                                serialization::Union{Nothing, SciMLBase.EnsembleAlgorithm},
                                mcmc_warmup::Union{Nothing, Int},
                                mcmc_draws::Union{Nothing, Int},
                                mcmc_method::Union{Nothing, MCMC},
                                mcmc_sampler,
                                mcmc_turing_kwargs::NamedTuple,
                                mcmc_adtype,
                                mcmc_fit_kwargs::NamedTuple,
                                rng::AbstractRNG)
    dm = get_data_model(res)
    dm === nothing && error("This fit result does not store a DataModel; pass store_data_model=true when fitting.")
    src_method = get_method(res)
    src_method isa MCMC && error("mcmc_refit is intended for non-MCMC fits. Use method=:chain for MCMC results.")

    constants_user = constants === nothing ? _fit_kw(res, :constants, NamedTuple()) : constants
    constants_re_use = constants_re === nothing ? _fit_kw(res, :constants_re, NamedTuple()) : constants_re
    ode_args_use = ode_args === nothing ? _fit_kw(res, :ode_args, ()) : ode_args
    ode_kwargs_use = ode_kwargs === nothing ? _fit_kw(res, :ode_kwargs, NamedTuple()) : ode_kwargs
    serialization_use = serialization === nothing ? _fit_kw(res, :serialization, EnsembleSerial()) : serialization

    fe = dm.model.fixed.fixed
    free_names = _free_fixed_names(fe, constants_user)
    θ_hat_u = get_params(res; scale=:untransformed)
    constants_all = _mcmc_refit_constants(fe, θ_hat_u, constants_user, free_names)
    sampled_fixed_names = [n for n in free_names if !(n in keys(constants_all))]
    _validate_mcmc_refit_priors(fe, sampled_fixed_names)

    if isempty(sampled_fixed_names) && isempty(get_re_names(dm.model.random.random))
        error("mcmc_refit requires at least one sampled parameter. With current constants/calculate_se settings there are no sampled fixed or random effects.")
    end

    method_refit = _mcmc_refit_method(mcmc_method, mcmc_sampler, mcmc_turing_kwargs, mcmc_adtype)
    fitkw = merge((
        constants=constants_all,
        constants_re=constants_re_use,
        ode_args=ode_args_use,
        ode_kwargs=ode_kwargs_use,
        serialization=serialization_use,
        rng=rng,
        theta_0_untransformed=θ_hat_u,
        store_data_model=true,
    ), mcmc_fit_kwargs)
    refit_res = fit_model(dm, method_refit; fitkw...)

    uq_chain = _compute_uq_chain(refit_res;
                                 level=level,
                                 constants=constants_all,
                                 mcmc_warmup=mcmc_warmup,
                                 mcmc_draws=mcmc_draws,
                                 rng=rng)

    diag = merge(uq_chain.diagnostics, (;
        refit_source_method=_method_symbol(src_method),
        refit_sampler=method_refit.sampler,
        refit_turing_kwargs=method_refit.turing_kwargs,
        sampled_fixed_names=sampled_fixed_names,
        constants_used=collect(keys(constants_all)),
    ))

    return UQResult(
        :mcmc_refit,
        _method_symbol(src_method),
        uq_chain.parameter_names,
        uq_chain.estimates_transformed,
        uq_chain.estimates_natural,
        uq_chain.intervals_transformed,
        uq_chain.intervals_natural,
        uq_chain.vcov_transformed,
        uq_chain.vcov_natural,
        uq_chain.draws_transformed,
        uq_chain.draws_natural,
        diag
    )
end
