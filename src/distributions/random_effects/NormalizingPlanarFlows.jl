using NormalizingFlows, Bijectors, FunctionChains, Functors, Optimisers, Distributions
import StaticArrays
import Random: AbstractRNG, default_rng

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
struct NormalizingPlanarFlow{D<:Distribution, R<:Optimisers.Restructure} <: AbstractNormalizingFlow
    base    :: D       # transformed distribution with fixed base q0
    rebuild :: R       # maps flat θ → bijector/transform
end

"""
    NormalizingPlanarFlow(n_input::Int, n_layers::Int; init=glorot_init)

Construct a normalizing planar flow with specified dimensions.

# Arguments
- `n_input` - Dimension of the random effects
- `n_layers` - Number of planar transformation layers
- `init` - Initialization function (default: `x -> sqrt(1/n_input) * x`)

# Returns
A `NormalizingPlanarFlow` with base distribution `MvNormal(zeros(n_input), I)`.

# Example
```julia
# 3D random effects with 5 transformation layers
flow = NormalizingPlanarFlow(3, 5)

# Sample from the flow
samples = rand(flow, 1000)
```
"""
function NormalizingPlanarFlow(n_input::Int, n_layers::Int; init= x -> sqrt( (1/n_input)) .* x)
    q₀ = MvNormal(zeros(Float64, n_input), I)
    d = length(q₀)
    Ls = [PlanarLayer(d, init) for _ in 1:n_layers]
    ts = fchain(Ls)

    θ, restructure = Optimisers.destructure(ts)
    transformed_obj = transformed(q₀, ts)

    NormalizingPlanarFlow(transformed_obj, restructure)

end

"""
    NormalizingPlanarFlow(θ::AbstractVector, rebuild::Optimisers.Restructure,
                          q0::ContinuousDistribution)

Construct a normalizing planar flow from flattened parameters.

This constructor is used internally by the model macro to create flows from
optimized parameter values.

# Arguments
- `θ` - Flattened flow parameters (weights and biases)
- `rebuild` - Optimisers.Restructure function to reconstruct flow
- `q0` - Base distribution

# Returns
A `NormalizingPlanarFlow` with the specified parameters.
"""
function NormalizingPlanarFlow(θ::AbstractVector, rebuild::Optimisers.Restructure, q0::ContinuousDistribution)
    bij = rebuild(θ)
    trans = transformed(q0, bij)
    NormalizingPlanarFlow(trans, rebuild)

end

Distributions.logpdf(d::NormalizingPlanarFlow, x::Real) = logpdf(d.base, [x])
Distributions.logpdf(d::NormalizingPlanarFlow, x::AbstractVector) = logpdf(d.base, x)
Distributions.logpdf(d::NormalizingPlanarFlow, x::StaticArrays.StaticVector) = logpdf(d.base, x)
Distributions.pdf(d::NormalizingPlanarFlow, x::AbstractVector)    = pdf(d.base, x)
Distributions.length(d::NormalizingPlanarFlow)                    = length(d.base)
Distributions.size(d::NormalizingPlanarFlow)                    = size(d.base)
Base.eltype(d::NormalizingPlanarFlow)                = eltype(d.base)
Distributions.rand(d::NormalizingPlanarFlow) = rand(d.base)
Distributions.rand(rng::AbstractRNG, d::NormalizingPlanarFlow) = rand(rng, d.base)
Distributions.rand(d::NormalizingPlanarFlow, n::Int) = rand(default_rng(), d, n)
Distributions.rand(rng::AbstractRNG, d::NormalizingPlanarFlow, n::Int) = rand(rng, d.base, n)
Distributions.rand(d::NormalizingPlanarFlow, dims::Dims...) = rand(d.base, dims...)
Distributions.rand(rng::AbstractRNG, d::NormalizingPlanarFlow, dims::Dims...) = rand(rng, d.base, dims...)

# Use the underlying transformed distribution's bijector for HMC/NUTS.
Bijectors.bijector(d::NormalizingPlanarFlow) = Bijectors.bijector(d.base)
