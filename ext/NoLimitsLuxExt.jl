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
    # Lux initialises in Float32; Float32 weights with Float64 inputs send conv layers to
    # NNlib's slow generic fallback (#405).
    init_params = Lux.f64(Lux.initialparameters(rng, chain))
    flat, reconstructor = Optimisers.destructure(init_params)
    T = Float64
    v = T.(flat)
    l = fill(T(-Inf), length(v))
    u = fill(T(Inf), length(v))
    _check_nn_prior(prior, name, length(v))
    _warn_stateful_layers(chain, name)
    return NNParameters{T, typeof(v), typeof(chain), typeof(reconstructor)}(
        name, function_name, chain, v, reconstructor, l, u, prior, calculate_se
    )
end

# A likelihood needs a deterministic mean function, so Dropout, BatchNorm and other layers
# with a `training` state are evaluated in test mode (#403).
_nn_states(chain) = Lux.testmode(Lux.initialstates(Xoshiro(0), chain))

function _stateful_layers!(out, layer, st::NamedTuple, path)
    if haskey(st, :training)
        push!(out, "$(nameof(typeof(layer))) ($(path))")
        return out
    end
    for k in keys(st)
        sub = if hasproperty(layer, :layers) && layer.layers isa NamedTuple && haskey(layer.layers, k)
            layer.layers[k]
        else
            hasproperty(layer, k) ? getproperty(layer, k) : nothing
        end
        _stateful_layers!(out, sub, st[k], isempty(path) ? string(k) : "$(path).$(k)")
    end
    return out
end
_stateful_layers!(out, layer, st, path) = out

function _warn_stateful_layers(chain, name)
    layers = _stateful_layers!(String[], chain, Lux.initialstates(Xoshiro(0), chain), "")
    isempty(layers) && return nothing
    @warn "NN parameter $(name): stateful layer(s) $(join(layers, ", ")) are frozen in test mode (Dropout acts as the identity, normalisation layers use their fixed running statistics), because the likelihood needs a deterministic mean function."
    return nothing
end

# Positional rebuild plan of the per-layer parameter arrays: leaves hold the flat range and
# the array shape, so `_nn_take` infers for any chain (the recursive ComponentArray walk it
# replaces widened to `Any` for nested containers, #402).
struct _NNLeaf{N}
    range::UnitRange{Int}
    dims::NTuple{N, Int}
end

_nn_plan(ax::ComponentArrays.Axis, off::Int) = map(v -> _nn_plan(v, off), ComponentArrays.indexmap(ax))
_nn_plan(i::Int, off::Int) = i + off
function _nn_plan(v::ComponentArrays.ViewAxis, off::Int)
    r = ComponentArrays.viewindex(v)
    v.ax isa ComponentArrays.Axis && return _nn_plan(v.ax, off + first(r) - 1)
    return _NNLeaf(r .+ off, applicable(size, v.ax) ? size(v.ax) : (length(r),))
end

# Layers get fresh arrays, not reshaped views of the flat vector or the input: Enzyme reverse
# mode through LuxLib's fused_dense! rule with aliased arrays corrupts the GC heap once the
# gradient runs on more than one thread. Same copy count as converting the flat vector.
_nn_flat(θ::ComponentArray) = getdata(θ)
_nn_flat(θ) = θ
_nn_take(::Type{T}, θ, l::_NNLeaf) where {T} = reshape(T.(view(θ, l.range)), l.dims)
_nn_take(::Type{T}, θ, i::Int) where {T} = convert(T, θ[i])
_nn_take(::Type{T}, θ, plan::NamedTuple) where {T} = map(l -> _nn_take(T, θ, l), plan)

# Vector inputs go in as one column: Lux's reshape(x, :, 1) would alias them, and BatchNorm
# needs a batch dimension.
_nn_input(::Type{T}, x::AbstractVector) where {T} = copyto!(Matrix{T}(undef, length(x), 1), x)
_nn_input(::Type{T}, x) where {T} = _to_type(T, x)
_nn_output(::AbstractVector, y::AbstractMatrix) = y[:, 1]
_nn_output(x, y) = y

struct _LuxModelFun{T, C, S, A, P}
    chain::C
    st::S
    axes::A
    plan::P
end

function _nn_model_fun(chain::Lux.Chain, p::NNParameters, ::Type{T}) where {T}
    st = Functors.fmap(y -> _to_type(T, y), _nn_states(chain))
    ps_axes = _nn_axes_template(p, T)
    plan = _nn_plan(only(ps_axes), 0)
    return _LuxModelFun{T, typeof(chain), typeof(st), typeof(ps_axes), typeof(plan)}(chain, st, ps_axes, plan)
end

(f::_LuxModelFun)(x, θ) = _lux_forward(f, x, θ, promote_type(eltype(θ), _value_type(x)))

function _lux_forward(f::_LuxModelFun{T}, x, θ, ::Type{T}) where {T}
    ps = _nn_take(T, _nn_flat(θ), f.plan)
    return _nn_output(x, first(Lux.apply(f.chain, _nn_input(T, x), ps, f.st)))
end

# Mixed element types (Float64 weights with dual inputs, e.g. the inner random-effect
# solves of Laplace, or the reverse): Lux layers promote internally, so the weights
# are NOT copied into the promoted type - converting all of them on every call (one
# call per observation row) was ~64% of the network time of a Lux latent-variable model
# (hard suite N03). Only the (small) input is converted.
function _lux_forward(f::_LuxModelFun, x, θ, ::Type{TT}) where {TT}
    return _nn_output(x, first(Lux.apply(f.chain, _nn_input(TT, x), ComponentArray(θ, f.axes), f.st)))
end

end
