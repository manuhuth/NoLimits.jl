using LinearAlgebra

using ComponentArrays

export ForwardTransform
export InverseTransform
export TransformSpec
export log_forward
export log_inverse
export cholesky_forward
export cholesky_inverse
export expm_forward
export expm_inverse
export apply_inv_jacobian_T

struct TransformSpec
    name::Symbol
    kind::Symbol
    size::Tuple{Int, Int}
    mask::Union{Nothing, Vector{Bool}}
end

struct ForwardTransform
    names::Vector{Symbol}
    specs::Vector{TransformSpec}
end

struct InverseTransform
    names::Vector{Symbol}
    specs::Vector{TransformSpec}
end

function (ft::ForwardTransform)(θ::ComponentArray)
    return _apply_transform(θ, ft.names, ft.specs)
end

function (it::InverseTransform)(θ::ComponentArray)
    return _apply_inverse_transform(θ, it.names, it.specs)
end

function log_forward(x::Real)
    return log(x)
end

function log_inverse(x::Real)
    return exp(x)
end

function cholesky_forward(A::AbstractMatrix{<:Real})
    Asym = Symmetric((A + A') / 2)
    L = cholesky(Asym).L
    T = Matrix(L)
    d = diag(T)
    return T - Diagonal(d) + Diagonal(log.(d))
end

function cholesky_inverse(T::AbstractMatrix{<:Real})
    L = Matrix(LowerTriangular(T))
    d = diag(L)
    Lexp = L - Diagonal(d) + Diagonal(exp.(d))
    return Lexp * Lexp'
end

function expm_forward(A::AbstractMatrix{<:Real})
    Asym = Symmetric((A + A') / 2)
    return Matrix(log(Asym))
end

function expm_inverse(T::AbstractMatrix{<:Real})
    return Matrix(exp(Symmetric(T)))
end

function _expm_frechet(A::AbstractMatrix{<:Real}, E::AbstractMatrix{<:Real})
    n = size(A, 1)
    size(A, 2) == n || error("A must be square.")
    size(E, 1) == n && size(E, 2) == n || error("E must match A size.")
    TT = promote_type(eltype(A), eltype(E))
    Z = zeros(TT, n, n)
    M = [Matrix{TT}(A) Matrix{TT}(E); Z Matrix{TT}(A)]
    EM = exp(M)
    return EM[1:n, 1:n], EM[1:n, n+1:2n]
end

function _upper_tri_vec(T::AbstractMatrix{<:Real})
    n = size(T, 1)
    return [T[i, j] for j in 1:n for i in 1:j]
end

function _sym_from_upper(v::AbstractVector{<:Real}, n::Int)
    idx = 1
    return [begin
        if i <= j
            val = v[idx]
            idx += 1
            val
        else
            v[(j - 1) * j ÷ 2 + i]
        end
    end for i in 1:n, j in 1:n]
end

function _upper_tri_vec_grad(G::AbstractMatrix{<:Real})
    n = size(G, 1)
    out = Vector{eltype(G)}(undef, n * (n + 1) ÷ 2)
    idx = 1
    for j in 1:n
        for i in 1:j
            out[idx] = i == j ? G[i, j] : (G[i, j] + G[j, i])
            idx += 1
        end
    end
    return out
end

function apply_inv_jacobian_T(it::InverseTransform, θt::ComponentArray, grad_u::ComponentArray)
    names = it.names
    specs = it.specs
    vals = map(1:length(names)) do i
        name = names[i]
        spec = specs[i]
        θti = θt[name]
        gu = grad_u[name]
        if spec.kind == :log
            if spec.mask === nothing
                return gu .* exp.(θti)
            else
                return [spec.mask[j] ? gu[j] * exp(θti[j]) : gu[j] for j in eachindex(θti)]
            end
        elseif spec.kind == :cholesky
            n1, n2 = spec.size
            T = reshape(θti, n1, n2)
            L = Matrix(LowerTriangular(T))
            d = diag(L)
            Lexp = L - Diagonal(d) + Diagonal(exp.(d))
            G = gu
            Gsym = G + G'
            grad_Lexp = Gsym * Lexp
            grad_T = zeros(eltype(grad_Lexp), n1, n2)
            for j in 1:n2
                for i in j:n1
                    if i == j
                        grad_T[i, j] = grad_Lexp[i, j] * exp(d[i])
                    else
                        grad_T[i, j] = grad_Lexp[i, j]
                    end
                end
            end
            return vec(grad_T)
        elseif spec.kind == :expm
            n1, n2 = spec.size
            T = _sym_from_upper(θti, n1)
            S = Symmetric(T)
            G = gu
            Gsym = Symmetric((G + G') / 2)
            _, F = _expm_frechet(S, Gsym)
            return _upper_tri_vec_grad(F)
        else
            return gu
        end
    end
    out = similar(θt)
    for i in eachindex(names)
        setproperty!(out, names[i], vals[i])
    end
    return out
end

function _apply_transform(θ::ComponentArray, names::Vector{Symbol}, specs::Vector{TransformSpec})
    vals = map(1:length(names)) do i
        name = names[i]
        spec = specs[i]
        val = θ[name]
        if spec.kind == :log
            if spec.mask === nothing
                return log.(val)
            else
                return [spec.mask[j] ? log(val[j]) : val[j] for j in eachindex(val)]
            end
        elseif spec.kind == :cholesky
            T = cholesky_forward(val)
            return vec(T)
        elseif spec.kind == :expm
            T = expm_forward(val)
            return _upper_tri_vec(T)
        else
            return val
        end
    end
    return ComponentArray(NamedTuple{Tuple(names)}(Tuple(vals)))
end

function _apply_inverse_transform(θ::ComponentArray, names::Vector{Symbol}, specs::Vector{TransformSpec})
    vals = map(1:length(names)) do i
        name = names[i]
        spec = specs[i]
        val = θ[name]
        if spec.kind == :log
            if spec.mask === nothing
                return exp.(val)
            else
                return [spec.mask[j] ? exp(val[j]) : val[j] for j in eachindex(val)]
            end
        elseif spec.kind == :cholesky
            n1, n2 = spec.size
            T = reshape(val, n1, n2)
            return cholesky_inverse(T)
        elseif spec.kind == :expm
            n1, n2 = spec.size
            T = _sym_from_upper(val, n1)
            return expm_inverse(T)
        else
            return val
        end
    end
    return ComponentArray(NamedTuple{Tuple(names)}(Tuple(vals)))
end
