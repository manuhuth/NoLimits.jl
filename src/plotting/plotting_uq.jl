export plot_uq_distributions

using Distributions
using KernelDensity
function _uq_param_indices(uq::UQResult, parameters; scale::Symbol = :transformed)
    names = get_uq_parameter_names(uq; scale = scale)
    if parameters === nothing
        return collect(eachindex(names))
    end
    params = parameters isa Symbol ? [parameters] : collect(parameters)
    idx = Int[]
    for p in params
        p_sym = Symbol(p)
        j = findfirst(==(p_sym), names)
        j === nothing && error("Unknown UQ parameter $(p_sym). Available: $(names).")
        push!(idx, j)
    end
    return idx
end

@inline function _uq_density_ylabel(backend::Symbol;
        analytic_wald::Bool = false,
        mixed_wald::Bool = false,
        plot_type::Symbol = :density)
    if plot_type == :histogram
        if backend == :chain || backend == :mcmc_refit
            return "Posterior Density Histogram"
        elseif backend == :wald
            return analytic_wald ? "Wald Approximate Histogram" : "Wald Histogram Density"
        end
        return "Histogram Density"
    end
    if backend == :chain || backend == :mcmc_refit
        return "Posterior Density"
    elseif backend == :wald
        return analytic_wald ? "Wald Approximate Density" :
               (mixed_wald ? "Wald Approx./KDE Density" : "Wald KDE Density")
    end
    return "KDE Density"
end

@inline function _uq_param_label(name::Symbol)
    return string(name)
end

@inline function _uq_wald_coord_transforms(uq::UQResult)
    d = get_uq_diagnostics(uq)
    if d isa NamedTuple && haskey(d, :coordinate_transforms)
        return getfield(d, :coordinate_transforms)
    end
    return nothing
end

@inline function _wald_closed_form_kind(backend::Symbol,
        scale::Symbol,
        j::Int,
        vcov_t::Union{Nothing, Matrix{Float64}},
        coord_transforms)
    backend == :wald || return :none
    vcov_t === nothing && return :none
    if scale == :transformed
        return :normal
    elseif scale == :natural
        coord_transforms === nothing && return :none
        j <= length(coord_transforms) || return :none
        k = coord_transforms[j]
        k == :identity && return :normal
        k == :log && return :lognormal
        k == :logit && return :logitnormal
    end
    return :none
end

function _wald_density_xy(kind::Symbol, μ::Float64, v::Float64; npts::Int = 300)
    σ = sqrt(max(v, 0.0))
    σ <= sqrt(eps(Float64)) && return nothing
    if kind == :normal
        dist = Normal(μ, σ)
        x = collect(range(μ - 4σ, μ + 4σ; length = npts))
        y = pdf.(dist, x)
        return (x, y)
    elseif kind == :lognormal
        dist = LogNormal(μ, σ)
        lo = quantile(dist, 0.001)
        hi = quantile(dist, 0.999)
        if !isfinite(lo) || !isfinite(hi) || hi <= lo
            lo = max(0.0, exp(μ - 4σ))
            hi = exp(μ + 4σ)
        end
        x = collect(range(lo, hi; length = npts))
        y = pdf.(dist, x)
        return (x, y)
    elseif kind == :logitnormal
        dist = LogitNormal(μ, σ)
        lo = quantile(dist, 0.001)
        hi = quantile(dist, 0.999)
        if !isfinite(lo) || !isfinite(hi) || hi <= lo
            lo = max(0.0, logit_inverse(μ - 4σ))
            hi = min(1.0, logit_inverse(μ + 4σ))
        end
        x = collect(range(lo, hi; length = npts))
        y = pdf.(dist, x)
        return (x, y)
    end
    error("Unknown closed-form density kind $(kind).")
end

@inline function _interp_on_grid(x::Vector{Float64}, y::Vector{Float64}, z::Float64)
    if z <= x[1]
        return y[1]
    elseif z >= x[end]
        return y[end]
    end
    i = searchsortedlast(x, z)
    i = clamp(i, 1, length(x) - 1)
    x0 = x[i]
    y0 = y[i]
    x1 = x[i + 1]
    y1 = y[i + 1]
    z == x0 && return y0
    x1 == x0 && return y0
    t = (z - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
end

function _density_interval_slice(x_raw::AbstractVector{<:Real},
        y_raw::AbstractVector{<:Real},
        lo::Real,
        hi::Real)
    length(x_raw) == length(y_raw) || error("x/y density vectors must have equal length.")
    length(x_raw) >= 2 || return nothing
    x = Float64.(x_raw)
    y = Float64.(y_raw)
    any(!isfinite, x) && return nothing
    any(!isfinite, y) && return nothing
    hi <= lo && return nothing
    hi <= x[1] && return nothing
    lo >= x[end] && return nothing

    lo_c = max(Float64(lo), x[1])
    hi_c = min(Float64(hi), x[end])
    hi_c <= lo_c && return nothing

    y_lo = _interp_on_grid(x, y, lo_c)
    y_hi = _interp_on_grid(x, y, hi_c)
    xs = Float64[lo_c]
    ys = Float64[y_lo]

    i_lo = searchsortedfirst(x, lo_c)
    i_hi = searchsortedlast(x, hi_c)
    if i_lo <= i_hi
        for i in i_lo:i_hi
            xi = x[i]
            if lo_c < xi < hi_c
                push!(xs, xi)
                push!(ys, y[i])
            end
        end
    end
    push!(xs, hi_c)
    push!(ys, y_hi)

    length(xs) >= 2 || return nothing
    return xs, ys
end

function _uq_kde_xy(
        vals::AbstractVector{<:Real}; bandwidth::Union{Nothing, Float64} = nothing)
    x = Float64.(collect(vals))
    length(x) >= 2 || error("KDE requires at least two samples.")
    x_min = minimum(x)
    x_max = maximum(x)
    if !isfinite(x_min) || !isfinite(x_max)
        error("KDE requires finite samples.")
    end
    if x_max == x_min
        δ = max(abs(x_min) * 0.05, 1e-3)
        return [x_min - δ, x_min, x_min + δ], [0.0, 1.0 / δ, 0.0]
    end
    kd = bandwidth === nothing ? kde(x) : kde(x; bandwidth = bandwidth)
    return kd.x, kd.density
end

@inline function _uq_merge_limits(lims, lo::Real, hi::Real)
    lo_f = Float64(lo)
    hi_f = Float64(hi)
    (isfinite(lo_f) && isfinite(hi_f)) || return lims
    if hi_f < lo_f
        lo_f, hi_f = hi_f, lo_f
    end
    return lims === nothing ? (lo_f, hi_f) :
           (min(lims[1], lo_f), max(lims[2], hi_f))
end
