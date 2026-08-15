module NoLimitsLuxExt

# Lux backend for NNParameters: building the parameter block and its forward pass. The
# NNParameters struct itself is generic in the chain type and stays in NoLimits, so only
# users who actually put a network in a model pay for Lux.

using Lux: Lux
using ComponentArrays
using Functors
using Optimisers
using Random: Xoshiro
import NoLimits: NNParameters, _nn_model_fun, _nn_axes_template, _to_type, _value_type,
    _check_nn_prior, Priorless

function NNParameters(
        chain::Lux.Chain; name::Symbol = :unnamed, function_name::Symbol,
        seed::Integer = 0, prior = Priorless(), calculate_se::Bool = false
    )
    rng = Xoshiro(seed)
    init_params = Lux.initialparameters(rng, chain)
    flat, reconstructor = Optimisers.destructure(init_params)
    T = eltype(flat) <: AbstractFloat ? eltype(flat) : Float64
    v = T.(flat)
    l = fill(T(-Inf), length(v))
    u = fill(T(Inf), length(v))
    _check_nn_prior(prior, name, length(v))
    return NNParameters{T, typeof(v), typeof(chain), typeof(reconstructor)}(
        name, function_name, chain, v, reconstructor, l, u, prior, calculate_se
    )
end

function _nn_model_fun(chain::Lux.Chain, p::NNParameters, ::Type{T}) where {T}
    st = Lux.initialstates(Xoshiro(0), chain)
    stT = Functors.fmap(y -> _to_type(T, y), st)
    ps_axes = _nn_axes_template(p, T)
    return (x, θ) -> begin
        TT = promote_type(eltype(θ), _value_type(x))
        if TT === T
            xT = _to_type(T, x)
            psT = ComponentArray(_to_type(T, θ), ps_axes)
            return first(Lux.apply(chain, xT, psT, stT))
        end
        xTT = _to_type(TT, x)
        psTT = ComponentArray(_to_type(TT, θ), ps_axes)
        stTT = Functors.fmap(y -> _to_type(TT, y), stT)
        return first(Lux.apply(chain, xTT, psTT, stTT))
    end
end

end
