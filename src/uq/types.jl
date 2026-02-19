export UQIntervals
export UQResult

"""
    UQIntervals

Confidence or credible intervals at a given coverage level for a set of parameters.

Fields:
- `level::Float64`: nominal coverage level (e.g. `0.95` for 95% intervals).
- `lower::Vector{Float64}`: lower bounds in the order given by the parent `UQResult`.
- `upper::Vector{Float64}`: upper bounds in the order given by the parent `UQResult`.
"""
struct UQIntervals
    level::Float64
    lower::Vector{Float64}
    upper::Vector{Float64}
end

"""
    UQResult

Result from [`compute_uq`](@ref). Stores parameter uncertainty quantification on
both the natural and transformed scales.

Use the accessor functions to retrieve individual components:
[`get_uq_backend`](@ref), [`get_uq_source_method`](@ref),
[`get_uq_parameter_names`](@ref), [`get_uq_estimates`](@ref),
[`get_uq_intervals`](@ref), [`get_uq_vcov`](@ref), [`get_uq_draws`](@ref),
[`get_uq_diagnostics`](@ref).

Fields:
- `backend::Symbol`: UQ backend used (`:wald`, `:chain`, or `:profile`).
- `source_method::Symbol`: estimation method of the source fit result.
- `parameter_names::Vector{Symbol}`: names of the free fixed-effect parameters.
- `estimates_transformed`, `estimates_natural`: point estimates on each scale.
- `intervals_transformed`, `intervals_natural`: [`UQIntervals`](@ref) or `nothing`.
- `vcov_transformed`, `vcov_natural`: variance-covariance matrices or `nothing`.
- `draws_transformed`, `draws_natural`: posterior/bootstrap draws (n_params × n_draws) or `nothing`.
- `diagnostics::NamedTuple`: backend-specific diagnostic information.
"""
struct UQResult
    backend::Symbol
    source_method::Symbol
    parameter_names::Vector{Symbol}
    estimates_transformed::Vector{Float64}
    estimates_natural::Vector{Float64}
    intervals_transformed::Union{Nothing, UQIntervals}
    intervals_natural::Union{Nothing, UQIntervals}
    vcov_transformed::Union{Nothing, Matrix{Float64}}
    vcov_natural::Union{Nothing, Matrix{Float64}}
    draws_transformed::Union{Nothing, Matrix{Float64}}
    draws_natural::Union{Nothing, Matrix{Float64}}
    diagnostics::NamedTuple
end

@inline function _nl_uq_level(uq::UQResult)
    uq.intervals_natural !== nothing && return uq.intervals_natural.level
    uq.intervals_transformed !== nothing && return uq.intervals_transformed.level
    return nothing
end

function _nl_uqresult_show_line(uq::UQResult)
    level = _nl_uq_level(uq)
    level_str = level === nothing ? "-" : string(round(level; digits=4))
    return "UQResult(backend=$(uq.backend), source_method=$(uq.source_method), n_params=$(length(uq.parameter_names)), level=$(level_str))"
end

Base.show(io::IO, uq::UQResult) = print(io, _nl_uqresult_show_line(uq))
Base.show(io::IO, ::MIME"text/plain", uq::UQResult) = print(io, _nl_uqresult_show_line(uq))
