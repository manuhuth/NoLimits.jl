module NoLimitsSimpleChainsExt

# SimpleChains backend for NNParameters. SimpleChains pulls in LoopVectorization,
# Polyester and SLEEFPirates, so it only loads for users who build a SimpleChain.

using SimpleChains: SimpleChains, SimpleChain
using ComponentArrays
using Random: Xoshiro
import ForwardDiff
import StaticArrays
import NoLimits: NNParameters, _nn_model_fun, _to_type, _value_type, _check_nn_prior,
    Priorless

function NNParameters(
        chain::SimpleChain; name::Symbol = :unnamed,
        function_name::Symbol, seed::Integer = 0,
        prior = Priorless(), calculate_se::Bool = false
    )
    T = Float64
    v = Vector{T}(SimpleChains.init_params(chain, T; rng = Xoshiro(seed)))
    l = fill(T(-Inf), length(v))
    u = fill(T(Inf), length(v))
    _check_nn_prior(prior, name, length(v))
    # SimpleChains parameters are already flat; the `reconstructor` field is unused for this
    # backend (`_nn_model_fun` calls `chain(x, θ)` directly), so it is set to `identity`.
    return NNParameters{T, typeof(v), typeof(chain), typeof(identity)}(
        name, function_name, chain, v, identity, l, u, prior, calculate_se
    )
end

# SimpleChains backend (NNParameters built from a `SimpleChains.SimpleChain`): parameters are
# natively a flat vector, so the forward pass is `chain(x, θ)` directly — no Lux states and no
# ComponentArray-axes rebuild. `_to_type` materialises the input and the ComponentArray view of θ
# into dense `Vector{TT}` (what SimpleChains expects), promoting so both share an eltype. This
# path is NOT Enzyme-differentiable (LoopVectorization `@turbo`) — use a Lux chain for AutoEnzyme.
# Output is indexable exactly like the Lux `Vector` output (`NN1(x, θ)[1]`). The backend is chosen
# by branching on `p.chain isa SimpleChain` (resolved at compile time for each concrete `p`),
# because parametric dispatch on the NNParameters chain type does not reliably out-specialise the
# unconstrained Lux methods when a free `Type{T}` argument is present.
#
# Two guards keep SimpleChains inside the regime it actually implements:
#
# 1. It returns its output as a value (an `SArray`) only while the forward scratch fits in its
#    `MAXSTACK`; past that it hands back a view over a per-task, per-chain-type buffer that the
#    next call reallocates (`memory.jl` `get_heap_memory`), so a caller still holding it reads
#    freed memory. That happens with no AD at all once the output has ≥64 elements or a hidden
#    layer is wide, so `_sc_value` copies whatever is not already a value.
# 2. Its `Dual` matmul is hand-written for at most two nested layers and untested above that
#    (`forwarddiff_matmul.jl` `dualeval!`), while a Laplace outer gradient under an implicit
#    solver reaches four and a Wald-UQ Hessian five. Measured on a 1-6-1 chain: exact to four
#    layers, `NaN` at five, wrong at six, then SIGBUS. Four or more layers therefore take
#    `_sc_apply`, a plain matmul over the same flat parameter vector, which has no ceiling.
_sc_value(y::StaticArrays.SArray) = y
_sc_value(y) = copy(y)

const SC_MAX_DUAL_DEPTH = 3

_sc_dual_depth(::Type{<:Base.IEEEFloat}) = 0
function _sc_dual_depth(::Type{ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    return 1 + _sc_dual_depth(V)
end
_sc_dual_depth(::Type) = SC_MAX_DUAL_DEPTH + 1   # unrecognised eltype: never hand it to `@turbo`

# `TurboDense` stores each layer's weights and bias contiguously as `[vec(W); b]`; `Activation`
# carries none.
function _sc_layer_plan(layer::SimpleChains.TurboDense{B}, nin, offset) where {B}
    nout = Int(layer.outputdim)
    nw = nin * nout
    w = (offset + 1):(offset + nw)
    b = B ? ((offset + nw + 1):(offset + nw + nout)) : (1:0)
    return (f = layer.f, nin = nin, nout = nout, w = w, b = b), nout,
        offset + nw + length(b)
end
function _sc_layer_plan(layer::SimpleChains.Activation, nin, offset)
    return (f = layer.f, nin = nin, nout = nin, w = 1:0, b = 1:0), nin, offset
end

_sc_apply(y, θ, ::Tuple{}) = y
function _sc_apply(y, θ, plan::Tuple)
    l = first(plan)
    z = isempty(l.w) ? y :
        reshape(view(θ, l.w), l.nout, l.nin) * y
    z = isempty(l.b) ? z : z .+ view(θ, l.b)
    return _sc_apply(l.f.(z), θ, Base.tail(plan))
end

# `nothing` when the architecture is outside what the fallback mirrors — such a chain keeps
# working on the shallow path it already worked on, and only deep AD is refused (below).
function _sc_plan(chain, name::Symbol)
    all(l -> l isa Union{SimpleChains.TurboDense, SimpleChains.Activation}, chain.layers) ||
        return nothing
    nin = Int(only(SimpleChains.chain_input_dims(chain)))
    plan, offset = (), 0
    for layer in chain.layers
        entry, nin, offset = _sc_layer_plan(layer, nin, offset)
        plan = (plan..., entry)
    end
    offset == Int(SimpleChains.numparam(chain)) ||
        error("NN parameter $(name): flat layout ($(offset) parameters) does not match the chain ($(Int(SimpleChains.numparam(chain)))).")
    return plan
end

@noinline function _sc_deep_ad_unsupported(name::Symbol, chain)
    layers = join(unique(string(typeof(l).name.name) for l in chain.layers), ", ")
    return error("NN parameter $(name): this fit differentiates the network more deeply than SimpleChains supports, and its layers ($(layers)) are outside the plain fallback. Rebuild this network as a `Lux.Chain`, which has no such limit.")
end

# Same convention as `_check_nn_flat_layout`: verify the layout once at model-build time, at
# non-zero probe parameters, so a future SimpleChains layout change is a build error rather than
# silently wrong deep-AD gradients.
function _check_sc_plan(chain, plan, name::Symbol)
    θ = SimpleChains.init_params(chain, Float64; rng = Xoshiro(0))
    x = [0.25 + 0.1 * i for i in 1:Int(only(SimpleChains.chain_input_dims(chain)))]
    isapprox(collect(_sc_apply(x, θ, plan)), collect(chain(x, θ)); rtol = 1.0e-10) ||
        error("NN parameter $(name): the plain fallback disagrees with the SimpleChain forward pass; the flat parameter layout is not the expected `[vec(W); b]` per layer.")
    return nothing
end

function _simplechain_model_fun(chain, name::Symbol)
    plan = _sc_plan(chain, name)
    plan === nothing || _check_sc_plan(chain, plan, name)
    return (x, θ) -> begin
        TT = promote_type(eltype(θ), _value_type(x))
        if _sc_dual_depth(TT) > SC_MAX_DUAL_DEPTH
            plan === nothing && _sc_deep_ad_unsupported(name, chain)
            return _sc_apply(x, θ, plan)
        end
        return _sc_value(chain(_to_type(TT, x), _to_type(TT, θ)))
    end
end
function _nn_model_fun(chain::SimpleChain, p::NNParameters, ::Type{T}) where {T}
    return _simplechain_model_fun(chain, p.name)
end

end
