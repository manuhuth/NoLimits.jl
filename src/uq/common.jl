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
        if spec.kind == :log && spec.mask !== nothing && !(v isa Number)
            for j in eachindex(spec.mask)
                push!(out, spec.mask[j] ? :log : :identity)
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
