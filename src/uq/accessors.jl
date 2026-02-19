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

Return the UQ backend used (`:wald`, `:chain`, or `:profile`).
"""
get_uq_backend(uq::UQResult) = uq.backend

"""
    get_uq_source_method(uq::UQResult) -> Symbol

Return the symbol identifying the estimation method of the source fit result.
"""
get_uq_source_method(uq::UQResult) = uq.source_method

"""
    get_uq_parameter_names(uq::UQResult) -> Vector{Symbol}

Return the names of the free fixed-effect parameters covered by this result.
"""
get_uq_parameter_names(uq::UQResult) = copy(uq.parameter_names)

"""
    get_uq_diagnostics(uq::UQResult) -> NamedTuple

Return backend-specific diagnostic information from the UQ computation.
"""
get_uq_diagnostics(uq::UQResult) = uq.diagnostics

function _uq_component(names::Vector{Symbol}, vals::Vector{Float64})
    return ComponentArray(NamedTuple{Tuple(names)}(Tuple(vals)))
end

"""
    get_uq_estimates(uq::UQResult; scale=:natural, as_component=true)

Return point estimates from a [`UQResult`](@ref).

# Keyword Arguments
- `scale::Symbol = :natural`: `:natural` for the untransformed scale, `:transformed`
  for the optimisation scale.
- `as_component::Bool = true`: if `true`, return a `ComponentArray` keyed by parameter
  name; otherwise return a plain `Vector{Float64}`.
"""
function get_uq_estimates(uq::UQResult; scale::Symbol=:natural, as_component::Bool=true)
    vals = if scale == :natural
        uq.estimates_natural
    elseif scale == :transformed
        uq.estimates_transformed
    else
        error("scale must be :natural or :transformed.")
    end
    return as_component ? _uq_component(uq.parameter_names, vals) : copy(vals)
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
function get_uq_intervals(uq::UQResult; scale::Symbol=:natural, as_component::Bool=true)
    ints = if scale == :natural
        uq.intervals_natural
    elseif scale == :transformed
        uq.intervals_transformed
    else
        error("scale must be :natural or :transformed.")
    end
    ints === nothing && return nothing
    if as_component
        return (level=ints.level,
                lower=_uq_component(uq.parameter_names, ints.lower),
                upper=_uq_component(uq.parameter_names, ints.upper))
    end
    return (level=ints.level, lower=copy(ints.lower), upper=copy(ints.upper))
end

"""
    get_uq_vcov(uq::UQResult; scale=:natural) -> Matrix{Float64} or nothing

Return the variance-covariance matrix from a [`UQResult`](@ref), or `nothing` if not
available.

# Keyword Arguments
- `scale::Symbol = :natural`: `:natural` or `:transformed`.
"""
function get_uq_vcov(uq::UQResult; scale::Symbol=:natural)
    if scale == :natural
        return uq.vcov_natural === nothing ? nothing : copy(uq.vcov_natural)
    elseif scale == :transformed
        return uq.vcov_transformed === nothing ? nothing : copy(uq.vcov_transformed)
    end
    error("scale must be :natural or :transformed.")
end

"""
    get_uq_draws(uq::UQResult; scale=:natural) -> Matrix{Float64} or nothing

Return the posterior or bootstrap draws (n_params × n_draws) from a [`UQResult`](@ref),
or `nothing` if not available.

# Keyword Arguments
- `scale::Symbol = :natural`: `:natural` or `:transformed`.
"""
function get_uq_draws(uq::UQResult; scale::Symbol=:natural)
    if scale == :natural
        return uq.draws_natural === nothing ? nothing : copy(uq.draws_natural)
    elseif scale == :transformed
        return uq.draws_transformed === nothing ? nothing : copy(uq.draws_transformed)
    end
    error("scale must be :natural or :transformed.")
end
