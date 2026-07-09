using NormalizingFlows, Bijectors, FunctionChains, Functors, Optimisers, Distributions
# Pin-only: Roots is transitive via Bijectors' planar-layer inverse (find_alpha).
# Roots 3.0.1+ regressed find_zero convergence (~27% failure); see [compat].
import Roots
import StaticArrays
import Random: AbstractRNG, default_rng
import Statistics

export AbstractNormalizingFlow
export NormalizingPlanarFlow

"""
    AbstractNormalizingFlow <: Distributions.ContinuousMultivariateDistribution

Abstract supertype for all normalizing flow distributions in NoLimits.jl.
Subtypes include [`NormalizingPlanarFlow`](@ref).
"""
abstract type AbstractNormalizingFlow <: Distributions.ContinuousMultivariateDistribution end

"""
    NormalizingPlanarFlow{D, R} <: AbstractNormalizingFlow

A normalizing planar flow distribution for flexible random effects.

Transforms a base distribution (typically multivariate normal) through a series of
planar layers to create a more expressive distribution. Used in `@randomEffects` blocks
to allow random effects to have non-Gaussian distributions.

# Fields
- `base::D` - Transformed distribution (base distribution + flow transformations)
- `rebuild::R` - Optimisers.Restructure function to reconstruct the bijector from flat parameters

# Constructors
```julia
# Direct construction with dimensions
NormalizingPlanarFlow(n_input::Int, n_layers::Int; init=glorot_init)

# Construction from parameters (used internally by model macro)
NormalizingPlanarFlow(θ::Vector, rebuild::Restructure, q0::Distribution)
```

# Arguments
- `n_input` - Dimension of random effects
- `n_layers` - Number of planar transformation layers
- `init` - Initialization function (default: Glorot normal)
- `θ` - Flattened flow parameters
- `rebuild` - Function to reconstruct bijector from θ
- `q0` - Base distribution (typically MvNormal)


```

# Theory
Planar flows apply transformations of the form:
```
f(z) = z + u·h(wᵀz + b)
```
where `h` is a nonlinear activation (typically `tanh`), and `u`, `w`, `b` are learnable
parameters. Multiple layers compose to create complex distributions:
```
x = f_K ∘ f_{K-1} ∘ ... ∘ f_1(z₀),  z₀ ~ q₀
```

The log-density is computed via change of variables:
```
log p(x) = log q₀(z₀) - Σᵢ log|det(Jfᵢ)|
```

# Advantages
- **Flexibility**: Can approximate complex, multimodal distributions
- **Expressiveness**: Captures non-Gaussian features (skewness, heavy tails, multimodality)
- **Differentiability**: Fully differentiable for gradient-based optimization
- **Interpretability**: Parameters are estimated along with other fixed effects

# Limitations
- **Computational cost**: More expensive than multivariate normal
- **Convergence**: May require more iterations to converge
- **Identifiability**: Flow parameters and base distribution parameters may trade off

# Implementation Details
- Uses planar layer architecture from NormalizingFlows.jl
- Parameters are flattened for optimization and reconstructed during evaluation
- Base distribution is typically `MvNormal(zeros(d), I)`
- Default initialization: Glorot normal scaled by 1/√n_input

# See Also
- `NPFParameter` - Parameter specification for flows (in `@fixedEffects`)
- `PlanarLayer` - Individual transformation layer (from NormalizingFlows.jl)


# References
- Rezende, D. J., & Mohamed, S. (2015). "Variational Inference with Normalizing Flows"
  ICML 2015.
"""
struct NormalizingPlanarFlow{D <: Distribution, R <: Optimisers.Restructure} <:
       AbstractNormalizingFlow
    base::D       # transformed distribution with fixed base q0
    rebuild::R       # maps flat θ → bijector/transform
end

"""
    NormalizingPlanarFlow(n_input::Int, n_layers::Int; init=glorot_init, base_dist=nothing)

Construct a normalizing planar flow with specified dimensions.

# Arguments
- `n_input` - Dimension of the random effects
- `n_layers` - Number of planar transformation layers
- `init` - Initialization function (default: `x -> sqrt(1/n_input) * x`)
- `base_dist` - Base distribution (default: `MvNormal(zeros(n_input), I)`)

# Returns
A `NormalizingPlanarFlow` with the specified base distribution.

# Example
```julia
# 3D random effects with 5 transformation layers, default Gaussian base
flow = NormalizingPlanarFlow(3, 5)

# With a custom base distribution
using Distributions
flow_t = NormalizingPlanarFlow(1, 3; base_dist=MvTDist(3, zeros(1), ones(1,1)))

# Sample from the flow
samples = rand(flow, 1000)
```
"""
function NormalizingPlanarFlow(
        n_input::Int, n_layers::Int; init = x -> sqrt((1 / n_input)) .* x,
        base_dist = nothing)
    q₀ = isnothing(base_dist) ? MvNormal(zeros(Float64, n_input), I) : base_dist
    d = length(q₀)
    Ls = [PlanarLayer(d, init) for _ in 1:n_layers]
    ts = fchain(Ls)

    θ, restructure = Optimisers.destructure(ts)
    transformed_obj = transformed(q₀, ts)

    NormalizingPlanarFlow(transformed_obj, restructure)
end

"""
    _planar_chain_from_flat(θ::AbstractVector, d::Int)

Rebuild the planar-layer chain from a flat parameter vector laid out as
`[w(d); u(d); b(1)]` per layer, layers in chain order — the order produced by
`Optimisers.destructure(fchain([PlanarLayer(d, init), ...]))`. Unlike the
`Restructure` closure, this is a plain positional reconstruction without
`Functors.fmap`/`IdDict` machinery, keeping it type-stable and Enzyme-compatible
(Enzyme forward mode has no derivative rule for the `jl_eqtable_get` IdDict
lookups inside `fmap`).
"""
function _planar_chain_from_flat(θ::AbstractVector, d::Int)
    npl = 2 * d + 1
    n_layers, rem = divrem(length(θ), npl)
    rem == 0 ||
        throw(ArgumentError("NPF flat parameter length $(length(θ)) is not a multiple of 2*d + 1 = $(npl) (d = $(d))."))
    layers = map(1:n_layers) do k
        off = (k - 1) * npl
        PlanarLayer(θ[off .+ (1:d)], θ[(off + d) .+ (1:d)], θ[(off + 2 * d) .+ (1:1)])
    end
    return fchain(layers)
end

"""
    NormalizingPlanarFlow(θ::AbstractVector, rebuild::Optimisers.Restructure,
                          q0::ContinuousDistribution)

Construct a normalizing planar flow from flattened parameters.

This constructor is used internally by the model macro to create flows from
optimized parameter values. The bijector is rebuilt positionally from `θ` via
[`_planar_chain_from_flat`](@ref); `rebuild` is only stored on the resulting
object (its `fmap`-based reconstruction breaks Enzyme forward mode).

# Arguments
- `θ` - Flattened flow parameters (weights and biases)
- `rebuild` - Optimisers.Restructure for the flow (stored, not used to rebuild)
- `q0` - Base distribution

# Returns
A `NormalizingPlanarFlow` with the specified parameters.
"""
function NormalizingPlanarFlow(
        θ::AbstractVector, rebuild::Optimisers.Restructure, q0::ContinuousDistribution)
    bij = _planar_chain_from_flat(θ, length(q0))
    trans = transformed(q0, bij)
    NormalizingPlanarFlow(trans, rebuild)
end

Distributions.logpdf(d::NormalizingPlanarFlow, x::Real) = logpdf(d.base, [x])
Distributions.logpdf(d::NormalizingPlanarFlow, x::AbstractVector) = logpdf(d.base, x)
function Distributions.logpdf(d::NormalizingPlanarFlow, x::StaticArrays.StaticVector)
    logpdf(d.base, x)
end
Distributions.pdf(d::NormalizingPlanarFlow, x::AbstractVector) = pdf(d.base, x)
Distributions.length(d::NormalizingPlanarFlow) = length(d.base)
Distributions.size(d::NormalizingPlanarFlow) = size(d.base)
Base.eltype(d::NormalizingPlanarFlow) = eltype(d.base)
Distributions.rand(d::NormalizingPlanarFlow) = rand(d.base)
Distributions.rand(rng::AbstractRNG, d::NormalizingPlanarFlow) = rand(rng, d.base)
Distributions.rand(d::NormalizingPlanarFlow, n::Int) = rand(default_rng(), d, n)
function Distributions.rand(rng::AbstractRNG, d::NormalizingPlanarFlow, n::Int)
    rand(rng, d.base, n)
end
Distributions.rand(d::NormalizingPlanarFlow, dims::Dims...) = rand(d.base, dims...)
function Distributions.rand(rng::AbstractRNG, d::NormalizingPlanarFlow, dims::Dims...)
    rand(rng, d.base, dims...)
end

# Use the underlying transformed distribution's bijector for HMC/NUTS.
Bijectors.bijector(d::NormalizingPlanarFlow) = Bijectors.bijector(d.base)

# DynamicPPL >=0.41 vectorizes a variable's link via `Bijectors.VectorBijectors.to_linked_vec(dist)`
# on the tilde path. Delegate to the underlying transformed distribution (consistent with the
# bijector delegation above). Guarded so the package still loads on Bijectors < 0.16, which has
# no `VectorBijectors` submodule (and whose DynamicPPL never calls this).
@static if isdefined(Bijectors, :VectorBijectors)
    # `to_linked_vec` / `linked_vec_length` have no generic continuous-multivariate fallback,
    # so delegate them to the wrapped transformed distribution. (`vec_length`/`to_vec`/`from_vec`
    # already resolve via the `MultivariateDistribution` fallback, since `NormalizingPlanarFlow`
    # is a `ContinuousMultivariateDistribution`.)
    function Bijectors.VectorBijectors.to_linked_vec(d::NormalizingPlanarFlow)
        Bijectors.VectorBijectors.to_linked_vec(d.base)
    end
    function Bijectors.VectorBijectors.from_linked_vec(d::NormalizingPlanarFlow)
        Bijectors.VectorBijectors.from_linked_vec(d.base)
    end
    function Bijectors.VectorBijectors.linked_vec_length(d::NormalizingPlanarFlow)
        Bijectors.VectorBijectors.linked_vec_length(d.base)
    end
end

# Estimate covariance via sampling — used only for MH step-size initialization,
# so an empirical approximation is sufficient.
function Statistics.cov(d::NormalizingPlanarFlow; n_samples::Int = 2000)
    Statistics.cov(rand(default_rng(), d, n_samples)')
end
