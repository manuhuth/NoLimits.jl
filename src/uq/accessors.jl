export get_uq_backend
export get_uq_source_method
export get_uq_parameter_names
export get_uq_estimates
export get_uq_intervals
export get_uq_vcov
export get_uq_draws
export get_uq_diagnostics

using ComponentArrays

"""
    get_uq_backend(uq::UQResult) -> Symbol

Return the UQ backend used (`:wald`, `:chain`, `:profile`, or `:mcmc_refit`).
"""
get_uq_backend(uq::UQResult) = uq.backend

"""
    get_uq_source_method(uq::UQResult) -> Symbol

Return the symbol identifying the estimation method of the source fit result.
"""
get_uq_source_method(uq::UQResult) = uq.source_method

"""
    get_uq_diagnostics(uq::UQResult) -> NamedTuple

Return backend-specific diagnostic information from the UQ computation.
"""
get_uq_diagnostics(uq::UQResult) = uq.diagnostics

function _uq_component(names::Vector{Symbol}, vals::Vector{Float64})
    return ComponentArray(NamedTuple{Tuple(names)}(Tuple(vals)))
end

# Every public UQ accessor takes the same `scale`; an unknown symbol used to fall through
# to the transformed branch and silently mislabel natural-scale values (#237).
@inline function _uq_check_scale(scale::Symbol)
    scale in (:natural, :transformed) ||
        throw(ArgumentError("scale must be :natural or :transformed. Got :$(scale)."))
    return scale
end

@inline function _uq_names_for_scale(uq::UQResult, scale::Symbol)
    _uq_check_scale(scale)
    if scale == :natural && uq.parameter_names_natural !== nothing
        return uq.parameter_names_natural
    end
    return uq.parameter_names
end

"""
    get_uq_parameter_names(uq::UQResult; scale=:transformed) -> Vector{Symbol}

Return the names of the free fixed-effect parameters covered by this result.

# Keyword Arguments
- `scale::Symbol = :transformed`: `:transformed` (default) or `:natural`. For the Wald
  backend with `ProbabilityVector` or `DiscreteTransitionMatrix` parameters, the natural
  scale includes the derived last probability / last-column entries and may have more
  names than the transformed scale.
"""
get_uq_parameter_names(uq::UQResult; scale::Union{Symbol, AbstractString} = :transformed) = copy(
    _uq_names_for_scale(
        uq, _as_symbol(scale)
    )
)

"""
    get_uq_estimates(uq::UQResult; scale=:natural, as_component=true)

Return point estimates from a [`UQResult`](@ref).

# Keyword Arguments
- `scale::Symbol = :natural`: `:natural` for the untransformed scale, `:transformed`
  for the optimization scale.
- `as_component::Bool = true`: if `true`, return a `ComponentArray` keyed by parameter
  name; otherwise return a plain `Vector{Float64}`.
"""
function get_uq_estimates(uq::UQResult; scale::Union{Symbol, AbstractString} = :natural, as_component::Bool = true)
    scale = _as_symbol(scale)
    _uq_check_scale(scale)
    vals = scale == :natural ? uq.estimates_natural : uq.estimates_transformed
    names = _uq_names_for_scale(uq, scale)
    return as_component ? _uq_component(names, vals) : copy(vals)
end

"""
    get_uq_intervals(uq::UQResult; scale=:natural, as_component=true)
    -> NamedTuple{(:level, :lower, :upper)} or nothing

Return confidence/credible intervals from a [`UQResult`](@ref), or `nothing` if not
available.

# Keyword Arguments
- `scale::Symbol = :natural`: `:natural` or `:transformed`.
- `as_component::Bool = true`: if `true`, `lower` and `upper` are `ComponentArray`s;
  otherwise plain `Vector{Float64}`.
"""
function get_uq_intervals(uq::UQResult; scale::Union{Symbol, AbstractString} = :natural, as_component::Bool = true)
    scale = _as_symbol(scale)
    _uq_check_scale(scale)
    ints = scale == :natural ? uq.intervals_natural : uq.intervals_transformed
    ints === nothing && return nothing
    names = _uq_names_for_scale(uq, scale)
    if as_component
        return (
            level = ints.level,
            lower = _uq_component(names, ints.lower),
            upper = _uq_component(names, ints.upper),
        )
    end
    return (level = ints.level, lower = copy(ints.lower), upper = copy(ints.upper))
end

"""
    get_uq_vcov(uq::UQResult; scale=:natural) -> Matrix{Float64} or nothing

Return the variance-covariance matrix from a [`UQResult`](@ref), or `nothing` if not
available.

# Keyword Arguments
- `scale::Symbol = :natural`: `:natural` or `:transformed`.
"""
function get_uq_vcov(uq::UQResult; scale::Union{Symbol, AbstractString} = :natural)
    scale = _as_symbol(scale)
    _uq_check_scale(scale)
    v = scale == :natural ? uq.vcov_natural : uq.vcov_transformed
    return v === nothing ? nothing : copy(v)
end

"""
    get_uq_draws(uq::UQResult; scale=:natural) -> Matrix{Float64} or nothing

Return the posterior or bootstrap draws from a [`UQResult`](@ref), or `nothing` if not
available. The matrix is `n_draws × n_params`: one row per draw, columns aligned with
[`get_uq_parameter_names`](@ref) for the same `scale`.

# Keyword Arguments
- `scale::Symbol = :natural`: `:natural` or `:transformed`.
"""
function get_uq_draws(uq::UQResult; scale::Union{Symbol, AbstractString} = :natural)
    scale = _as_symbol(scale)
    _uq_check_scale(scale)
    d = scale == :natural ? uq.draws_natural : uq.draws_transformed
    return d === nothing ? nothing : copy(d)
end
