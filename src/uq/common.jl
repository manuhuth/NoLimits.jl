export compute_uq

using ForwardDiff
using LinearAlgebra
using Random
using Statistics

@inline function _fit_kw(res::FitResult, key::Symbol, default)
    return haskey(res.fit_kwargs, key) ? getfield(res.fit_kwargs, key) : default
end

@inline function _method_symbol(method::FittingMethod)
    method isa MLE && return :mle
    method isa MAP && return :map
    method isa MCMC && return :mcmc
    method isa VI && return :vi
    method isa Laplace && return :laplace
    method isa LaplaceMAP && return :laplace_map
    method isa FOCEI && return :focei
    method isa FOCEIMAP && return :focei_map
    method isa MCEM && return :mcem
    method isa SAEM && return :saem
    return Symbol(lowercase(string(nameof(typeof(method)))))
end

function _validate_level(level::Real)
    (0.0 < level < 1.0) || error("UQ level must be strictly between 0 and 1. Got $(level).")
    return Float64(level)
end

function _free_fixed_names(fe::FixedEffects, constants::NamedTuple)
    fixed_names = get_names(fe)
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name) in UQ constants.")
    end
    return [n for n in fixed_names if !(n in keys(constants))]
end

function _active_mask_for_free(fe::FixedEffects, free_names::Vector{Symbol})
    θt = get_θ0_transformed(fe)
    params = get_params(fe)
    mask = Bool[]
    for name in free_names
        flag = getfield(params, name).calculate_se
        v = getproperty(θt, name)
        n = v isa Number ? 1 : length(vec(v))
        append!(mask, fill(flag, n))
    end
    return mask
end

function _flat_parent_names(fe::FixedEffects)
    θt = get_θ0_transformed(fe)
    out = Symbol[]
    for name in get_names(fe)
        v = getproperty(θt, name)
        n = v isa Number ? 1 : length(vec(v))
        append!(out, fill(name, n))
    end
    return out
end

function _flat_names_for_free(fe::FixedEffects, free_names::Vector{Symbol})
    parent = _flat_parent_names(fe)
    free_set = Set(free_names)
    flat_all = get_flat_names(fe)
    keep = findall(i -> parent[i] in free_set, eachindex(parent))
    return flat_all[keep]
end

function _flat_transform_kinds_for_free(fe::FixedEffects, free_names::Vector{Symbol})
    θt = get_θ0_transformed(fe)
    all_names = get_names(fe)
    specs = get_transforms(fe).forward.specs
    spec_map = Dict{Symbol, TransformSpec}()
    for i in eachindex(all_names)
        spec_map[all_names[i]] = specs[i]
    end

    out = Symbol[]
    for name in free_names
        spec = spec_map[name]
        v = getproperty(θt, name)
        if spec.kind == :elementwise
            for j in eachindex(spec.mask)
                push!(out, spec.mask[j])
            end
        else
            n = v isa Number ? 1 : length(vec(v))
            append!(out, fill(spec.kind, n))
        end
    end
    return out
end

@inline function _coords_for_param(value, spec::TransformSpec; natural::Bool)
    if value isa Number
        return Float64[value]
    elseif natural && spec.kind == :expm && value isa AbstractMatrix
        n = size(value, 1)
        out = Float64[]
        for j in 1:n
            for i in 1:j
                push!(out, Float64(value[i, j]))
            end
        end
        return out
    elseif natural && spec.kind == :stickbreak && value isa AbstractVector
        # Drop last (determined) probability; return first k-1 components.
        return Float64.(value[1:end-1])
    elseif natural && spec.kind == :stickbreakrows && value isa AbstractMatrix
        # Drop last column from each row; return n*(n-1) components.
        n = size(value, 1)
        out = Float64[]
        for i in 1:n
            for j in 1:n-1
                push!(out, Float64(value[i, j]))
            end
        end
        return out
    elseif natural && spec.kind == :lograterows && value isa AbstractMatrix
        # Return off-diagonal entries in row-major order (same count as transformed).
        n = size(value, 1)
        out = Float64[]
        for i in 1:n
            for j in 1:n
                i == j && continue
                push!(out, Float64(value[i, j]))
            end
        end
        return out
    else
        return Float64.(vec(value))
    end
end

@inline function _as_component_array(θ)
    return θ isa ComponentArray ? θ : ComponentArray(θ)
end

function _coords_on_transformed_layout(fe::FixedEffects,
                                       θ,
                                       names::Vector{Symbol};
                                       natural::Bool=false)
    all_names = get_names(fe)
    specs = get_transforms(fe).forward.specs
    spec_map = Dict{Symbol, TransformSpec}()
    for i in eachindex(all_names)
        spec_map[all_names[i]] = specs[i]
    end
    out = Float64[]
    for name in names
        hasproperty(θ, name) || error("Parameter vector is missing $(name).")
        append!(out, _coords_for_param(getproperty(θ, name), spec_map[name]; natural=natural))
    end
    return out
end

# Returns (extended_names_n, extended_est_n, extended_draws_n, extended_intervals_n)
# or nothing if no stickbreak/stickbreakrows parameters are active.
# Appends derived last-probability entries for ProbabilityVector and
# derived last-column entries for DiscreteTransitionMatrix.
function _extend_natural_stickbreak(
    fe::FixedEffects,
    free_names::Vector{Symbol},
    active_names::Vector{Symbol},
    active_kinds::Vector{Symbol},
    est_n::Vector{Float64},
    draws_n::Union{Nothing, Matrix{Float64}},
    intervals_n::Union{Nothing, UQIntervals}
)
    has_sb = any(k -> k == :stickbreak || k == :stickbreakrows, active_kinds)
    has_sb || return nothing

    all_fe_names = get_names(fe)
    all_specs = get_transforms(fe).forward.specs
    spec_map = Dict{Symbol, TransformSpec}(all_fe_names[i] => all_specs[i] for i in eachindex(all_fe_names))
    params_nt = get_params(fe)
    θt = get_θ0_transformed(fe)

    # Build parameter -> contiguous range in active array
    param_active_ranges = Dict{Symbol, UnitRange{Int}}()
    active_pos = 0
    for name in free_names
        p = getfield(params_nt, name)
        spec = spec_map[name]
        v = getproperty(θt, name)
        n_t = v isa Number ? 1 : length(vec(v))
        if p.calculate_se
            param_active_ranges[name] = (active_pos + 1):(active_pos + n_t)
            active_pos += n_t
        end
    end

    level = intervals_n !== nothing ? intervals_n.level : 0.95
    α = 1.0 - level

    derived_names  = Symbol[]
    derived_est    = Float64[]
    derived_lower  = Float64[]
    derived_upper  = Float64[]
    draw_cols      = Vector{Float64}[]

    for name in free_names
        p = getfield(params_nt, name)
        !p.calculate_se && continue
        spec = spec_map[name]
        rng  = param_active_ranges[name]

        if spec.kind == :stickbreak
            k = spec.size[1]
            push!(derived_names, Symbol(name, "_", k))
            push!(derived_est,   1.0 - sum(est_n[rng]))
            if draws_n !== nothing
                col = 1.0 .- vec(sum(@view(draws_n[:, rng]); dims=2))
                push!(draw_cols,    col)
                push!(derived_lower, quantile(col, α / 2))
                push!(derived_upper, quantile(col, 1 - α / 2))
            end

        elseif spec.kind == :stickbreakrows
            n    = spec.size[1]
            base = n * (n - 1)
            for i in 1:n
                row_rng = (rng.start + (i - 1) * (n - 1)):(rng.start + i * (n - 1) - 1)
                push!(derived_names, Symbol(name, "_", base + i))
                push!(derived_est,   1.0 - sum(est_n[row_rng]))
                if draws_n !== nothing
                    col = 1.0 .- vec(sum(@view(draws_n[:, row_rng]); dims=2))
                    push!(draw_cols,    col)
                    push!(derived_lower, quantile(col, α / 2))
                    push!(derived_upper, quantile(col, 1 - α / 2))
                end
            end
        end
    end

    isempty(derived_names) && return nothing

    ext_names      = vcat(active_names, derived_names)
    ext_est_n      = vcat(est_n, derived_est)

    ext_draws_n = if draws_n !== nothing && !isempty(draw_cols)
        hcat(draws_n, reduce(hcat, draw_cols))
    else
        nothing
    end

    ext_intervals_n = if intervals_n !== nothing
        UQIntervals(level,
                    vcat(intervals_n.lower, derived_lower),
                    vcat(intervals_n.upper, derived_upper))
    else
        nothing
    end

    return (ext_names, ext_est_n, ext_draws_n, ext_intervals_n)
end

function _build_ll_cache_uq(dm::DataModel,
                            ode_args::Tuple,
                            ode_kwargs::NamedTuple,
                            serialization::SciMLBase.EnsembleAlgorithm)
    if serialization isa SciMLBase.EnsembleThreads
        return build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=Threads.maxthreadid())
    end
    return build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs)
end

function _project_psd_covariance(cov_mat::Matrix{Float64})
    size(cov_mat, 1) == size(cov_mat, 2) || error("Covariance matrix must be square.")
    S = Symmetric(0.5 .* (cov_mat .+ cov_mat'))
    eig = eigen(S)
    vals_raw = eig.values
    vals = copy(vals_raw)
    n_clipped = 0
    @inbounds for i in eachindex(vals)
        if vals[i] < 0.0
            vals[i] = 0.0
            n_clipped += 1
        end
    end
    V = eig.vectors * Diagonal(vals) * eig.vectors'
    V = Matrix{Float64}(0.5 .* (V .+ V'))
    min_raw = isempty(vals_raw) ? 0.0 : minimum(vals_raw)
    min_used = isempty(vals) ? 0.0 : minimum(vals)
    diag = (;
        vcov_projected=n_clipped > 0,
        vcov_min_eig_raw=min_raw,
        vcov_min_eig_used=min_used,
        vcov_n_eigs_clipped=n_clipped,
    )
    return V, diag
end

function _sample_gaussian_draws(rng::AbstractRNG,
                                mean_vec::Vector{Float64},
                                cov_mat::Matrix{Float64},
                                n_draws::Int)
    p = length(mean_vec)
    n_draws >= 1 || error("n_draws must be >= 1.")
    p == 0 && return zeros(Float64, n_draws, 0)
    S = Symmetric(0.5 .* (cov_mat .+ cov_mat'))
    eig = eigen(S)
    vals = max.(eig.values, 0.0)
    A = eig.vectors * Diagonal(sqrt.(vals))
    Z = randn(rng, p, n_draws)
    X = A * Z
    @inbounds for j in 1:p
        X[j, :] .+= mean_vec[j]
    end
    return permutedims(X)
end

function _cov_from_draws(draws::Matrix{Float64})
    n, p = size(draws)
    p == 0 && return zeros(Float64, 0, 0)
    n >= 2 || return zeros(Float64, p, p)
    μ = vec(mean(draws; dims=1))
    C = zeros(Float64, p, p)
    for i in 1:n
        row = @view draws[i, :]
        d = row .- μ
        C .+= d * d'
    end
    return C ./ (n - 1)
end

function _intervals_from_draws(draws::Matrix{Float64}, level::Float64)
    n, p = size(draws)
    p == 0 && return UQIntervals(level, Float64[], Float64[])
    n >= 1 || error("Cannot compute intervals from empty draw matrix.")
    α = 1.0 - level
    qlo = α / 2
    qhi = 1.0 - qlo
    lower = Vector{Float64}(undef, p)
    upper = Vector{Float64}(undef, p)
    for j in 1:p
        col = @view draws[:, j]
        lower[j] = quantile(col, qlo)
        upper[j] = quantile(col, qhi)
    end
    return UQIntervals(level, lower, upper)
end

function _hessian_from_objective(obj::Function,
                                 x0::Vector{Float64};
                                 backend::Symbol=:auto,
                                 fd_abs_step::Real=1e-4,
                                 fd_rel_step::Real=1e-3,
                                 fd_max_tries::Int=8)
    backend_use = backend == :auto ? :forwarddiff : backend
    backend_use == :forwarddiff || backend_use == :fd_gradient ||
        error("Unsupported Hessian backend $(backend). Use :auto, :forwarddiff, or :fd_gradient.")

    grad_fun = x -> begin
        xv = Float64.(x)
        try
            return Float64.(ForwardDiff.gradient(obj, xv))
        catch
            return _gradient_fd_from_obj(obj, xv;
                                         abs_step=fd_abs_step,
                                         rel_step=fd_rel_step,
                                         max_tries=fd_max_tries)
        end
    end

    if backend_use == :forwarddiff
        try
            H = ForwardDiff.hessian(obj, x0)
            return Matrix{Float64}(0.5 .* (H .+ H')), :forwarddiff
        catch err
            @warn "ForwardDiff Hessian failed in compute_uq; falling back to finite-difference Hessian from gradients." error=sprint(showerror, err)
            backend_use = :fd_gradient
        end
    end

    H = _hessian_fd_from_grad(grad_fun, x0;
                              abs_step=fd_abs_step,
                              rel_step=fd_rel_step,
                              max_tries=fd_max_tries)
    return Matrix{Float64}(0.5 .* (H .+ H')), :fd_gradient
end

function _gradient_from_objective(obj::Function,
                                  x0::Vector{Float64};
                                  fd_abs_step::Real=1e-6,
                                  fd_rel_step::Real=1e-6,
                                  fd_max_tries::Int=8)
    try
        return Float64.(ForwardDiff.gradient(obj, x0))
    catch
        return _gradient_fd_from_obj(obj, x0;
                                     abs_step=fd_abs_step,
                                     rel_step=fd_rel_step,
                                     max_tries=fd_max_tries)
    end
end

@inline function _uq_mcmc_warmup(res::FitResult)
    conv = get_diagnostics(res).convergence
    if conv isa NamedTuple && haskey(conv, :n_adapt)
        return Int(getfield(conv, :n_adapt))
    end
    return 0
end
