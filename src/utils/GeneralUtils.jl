export dir
export hessian_fwd_over_zygote
export build_ode_params
export flatten_re_names
export flatten_re_values

using ForwardDiff
using Zygote

function dir(x)
    return fieldnames(typeof(x))
end

function hessian_fwd_over_zygote(f, x)
    g(xv) = Zygote.gradient(f, xv)[1]
    return ForwardDiff.jacobian(g, x)
end

function build_ode_params(de, θ;
                          random_effects=ComponentArray(NamedTuple()),
                          constant_covariates=NamedTuple(),
                          varying_covariates=NamedTuple(),
                          helpers=NamedTuple(),
                          model_funs=NamedTuple(),
                          prede_builder=(fe, re, consts, model_funs, helpers) -> NamedTuple(),
                          inverse_transform=identity)
    return build_de_params(de, θ;
                           random_effects=random_effects,
                           constant_covariates=constant_covariates,
                           varying_covariates=varying_covariates,
                           helpers=helpers,
                           model_funs=model_funs,
                           prede_builder=prede_builder,
                           inverse_transform=inverse_transform)
end

function flatten_re_names(name::Symbol, val)
    if val isa Number
        return Symbol[name]
    end
    vals = vec(collect(val))
    return [Symbol(name, "_", i) for i in 1:length(vals)]
end

function flatten_re_values(val)
    if val isa Number
        return [val]
    end
    return collect(vec(val))
end

@inline function _with_infusion(f!, infusion_rates)
    infusion_rates === nothing && return f!
    return function (du, u, p, t)
        f!(du, u, p, t)
        @inbounds for i in eachindex(infusion_rates)
            du[i] += infusion_rates[i]
        end
        return nothing
    end
end

@inline function _ode_solve_kwargs(base::NamedTuple,
                                   extra::NamedTuple=NamedTuple(),
                                   overrides::NamedTuple=NamedTuple())
    return merge((verbose=false,), base, extra, overrides)
end
