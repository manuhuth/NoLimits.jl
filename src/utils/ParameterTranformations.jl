using LinearAlgebra
using Lux

using ComponentArrays

export ForwardTransform
export InverseTransform
export TransformSpec
export log_forward
export log_inverse
export logit_forward
export logit_inverse
export cholesky_forward
export cholesky_inverse
export expm_forward
export expm_inverse
export stickbreak_forward
export stickbreak_inverse
export lograterows_forward
export lograterows_inverse
export apply_inv_jacobian_T

struct TransformSpec
    name::Symbol
    kind::Symbol
    size::Tuple{Int, Int}
    mask::Union{Nothing, Vector{Symbol}}
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

function logit_forward(x::Real)
    return clamp(log(x / (1 - x)), -20.0, 20.0)
end

function logit_inverse(x::Real)
    return Lux.sigmoid(clamp(x, -20.0, 20.0))
end

@inline function _logit_inv_jacobian(x::Real)
    abs(x) >= 20.0 && return zero(x)
    s = logit_inverse(x)
    return s * (one(s) - s)
end

"""
    stickbreak_forward(p) -> Vector

Map a k-probability vector `p` (summing to 1, all ≥ 0) to a k-1 vector of
unconstrained reals via the logistic stick-breaking transform:

    ν_i = p_i / (1 - Σ_{j<i} p_j),   t_i = logit(ν_i),  i = 1, ..., k-1

The last probability is determined and is not stored.
"""
function stickbreak_forward(p::AbstractVector{<:Real})
    k = length(p)
    k >= 2 || error("stickbreak_forward requires at least 2 elements.")
    T = promote_type(eltype(p), Float64)
    # Compute remaining = 1 - cumsum(p)[i-1] using cumprod of (1 - p/remaining)
    # Iterative but non-mutating via map with accumulated state is tricky;
    # use simple loop (forward transform is not typically differentiated by Zygote).
    t = Vector{T}(undef, k - 1)
    remaining = one(T)
    for i in 1:(k - 1)
        pi = T(p[i])
        νi = pi / remaining
        t[i] = logit_forward(νi)
        remaining -= pi
    end
    return t
end

"""
    stickbreak_inverse(t) -> Vector

Map a k-1 vector of unconstrained reals to a k-probability vector via the
inverse logistic stick-breaking transform. Recovers a valid probability vector
summing to 1 with all entries in [0, 1].

Non-mutating (Zygote-friendly): uses cumprod to build remaining probabilities.
"""
function stickbreak_inverse(t::AbstractVector{<:Real})
    σ = logit_inverse.(t)           # k-1 sigmoid values in (0,1)
    one_minus_σ = one.(σ) .- σ
    # remainders[i] = prod(1-σ[1:i-1]); remainders[1]=1, remainders[k]=prod(1-σ)
    remainders = cumprod(vcat(one(eltype(σ)), one_minus_σ))
    k1 = length(t)
    return vcat(σ .* remainders[1:k1], remainders[k1+1:k1+1])
end

# Apply stickbreak_forward row-wise to an n×n matrix; returns n*(n-1) flat vector.
function _stickbreakrow_forward(P::AbstractMatrix{<:Real})
    n = size(P, 1)
    vcat([stickbreak_forward(P[i, :]) for i in 1:n]...)
end

# Apply stickbreak_inverse row-wise; reconstructs n×n row-stochastic matrix.
function _stickbreakrow_inverse(t::AbstractVector{<:Real}, n::Int)
    rows = [stickbreak_inverse(t[(i - 1) * (n - 1) + 1 : i * (n - 1)]) for i in 1:n]
    reduce(vcat, [r' for r in rows])
end

"""
    lograterows_forward(Q) -> Vector

Map an `n×n` Q-matrix (off-diagonal entries ≥ 0, rows sum to 0) to an `n*(n-1)`
unconstrained vector by taking the element-wise logarithm of the off-diagonal entries,
traversed in row-major order (row 1: columns 1..n skipping diagonal, then row 2, etc.).
"""
function lograterows_forward(Q::AbstractMatrix{<:Real})
    n = size(Q, 1)
    T = promote_type(eltype(Q), Float64)
    out = Vector{T}(undef, n * (n - 1))
    idx = 1
    for i in 1:n
        for j in 1:n
            i == j && continue
            out[idx] = log(T(Q[i, j]))
            idx += 1
        end
    end
    return out
end

"""
    lograterows_inverse(t, n) -> Matrix

Map an `n*(n-1)` unconstrained vector (log off-diagonal rates) back to an `n×n` Q-matrix.
Off-diagonal entries are `exp(t[k])`; diagonal entries are `-rowsum`.
"""
function lograterows_inverse(t::AbstractVector{<:Real}, n::Int)
    T = promote_type(eltype(t), Float64)
    Q = zeros(T, n, n)
    idx = 1
    for i in 1:n
        for j in 1:n
            i == j && continue
            Q[i, j] = exp(T(t[idx]))
            idx += 1
        end
        Q[i, i] = -sum(Q[i, j] for j in 1:n if j != i)
    end
    return Q
end

# Compute J^T * g for stick-breaking inverse: maps k-vector g to k-1-vector.
# Uses the O(k) backward recurrence:
#   Forward pass: r[i] = remaining probability before step i, p[i] = σ(t[i])*r[i]
#   Backward pass: g_t[j] = σ'(t[j]) * r[j] * (g[j] - S / r[j+1])
#                  S accumulates Σ_{i≥j+1} p[i] * g[i]
function _stickbreak_inv_jacobian_T(t::AbstractVector{<:Real}, g::AbstractVector{<:Real})
    k1 = length(t)
    k = k1 + 1
    length(g) == k || error("Gradient length $(length(g)) must equal k=$(k).")
    T = promote_type(eltype(t), eltype(g), Float64)
    # Forward pass: compute r and p on natural scale
    r = Vector{T}(undef, k)
    p = Vector{T}(undef, k)
    r[1] = one(T)
    for i in 1:k1
        σi = logit_inverse(T(t[i]))
        p[i] = σi * r[i]
        r[i + 1] = r[i] - p[i]
    end
    p[k] = r[k]
    # Backward pass
    g_t = Vector{T}(undef, k1)
    S = T(g[k]) * p[k]  # = g[k] * r[k]
    for j in k1:-1:1
        σj = logit_inverse(T(t[j]))
        σj_prime = σj * (one(T) - σj)
        rj1 = r[j + 1]
        g_t[j] = σj_prime * r[j] * (T(g[j]) - S / rj1)
        S += T(g[j]) * p[j]
    end
    return g_t
end

@inline function _scalar_forward(kind::Symbol, x::Real)
    kind === :log   && return log(x)
    kind === :logit && return logit_forward(x)
    return x
end

@inline function _scalar_inverse(kind::Symbol, x::Real)
    kind === :log   && return exp(x)
    kind === :logit && return logit_inverse(x)
    return x
end

@inline function _scalar_inv_jacobian(kind::Symbol, g::Real, x::Real)
    kind === :log   && return g * exp(x)
    kind === :logit && return g * _logit_inv_jacobian(x)
    return g
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
            return gu .* exp.(θti)
        elseif spec.kind == :logit
            return gu .* _logit_inv_jacobian.(θti)
        elseif spec.kind == :elementwise
            return [_scalar_inv_jacobian(spec.mask[j], gu[j], θti[j]) for j in eachindex(θti)]
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
        elseif spec.kind == :stickbreak
            # θti: k-1 transformed; gu: k natural gradient → k-1 output
            return _stickbreak_inv_jacobian_T(θti, gu)
        elseif spec.kind == :stickbreakrows
            # θti: n*(n-1) transformed; gu: n×n natural gradient → n*(n-1) output
            n = spec.size[1]
            T_acc = promote_type(eltype(θti), eltype(gu), Float64)
            out = Vector{T_acc}(undef, n * (n - 1))
            for i in 1:n
                chunk_t = @view θti[(i - 1) * (n - 1) + 1 : i * (n - 1)]
                g_row = vec(gu[i, :])
                g_t_row = _stickbreak_inv_jacobian_T(chunk_t, g_row)
                out[(i - 1) * (n - 1) + 1 : i * (n - 1)] .= g_t_row
            end
            return out
        elseif spec.kind == :lograterows
            # θti: n*(n-1) log off-diagonal rates; gu: n×n natural gradient → n*(n-1) output
            # Q[i,j] = exp(t[k]) for off-diagonal (i,j), Q[i,i] = -sum_row
            # ∂Q[i,j]/∂t[k] = exp(t[k]), ∂Q[i,i]/∂t[k] = -exp(t[k])
            # So g_t[k] = exp(t[k]) * (g_u[i,j] - g_u[i,i])
            n = spec.size[1]
            T_acc = promote_type(eltype(θti), eltype(gu), Float64)
            out = Vector{T_acc}(undef, n * (n - 1))
            idx = 1
            for i in 1:n
                g_ii = T_acc(gu[i, i])
                for j in 1:n
                    i == j && continue
                    out[idx] = exp(T_acc(θti[idx])) * (T_acc(gu[i, j]) - g_ii)
                    idx += 1
                end
            end
            return out
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
            return log.(val)
        elseif spec.kind == :logit
            return logit_forward.(val)
        elseif spec.kind == :elementwise
            return [_scalar_forward(spec.mask[j], val[j]) for j in eachindex(val)]
        elseif spec.kind == :cholesky
            T = cholesky_forward(val)
            return vec(T)
        elseif spec.kind == :expm
            T = expm_forward(val)
            return _upper_tri_vec(T)
        elseif spec.kind == :stickbreak
            return stickbreak_forward(val)
        elseif spec.kind == :stickbreakrows
            return _stickbreakrow_forward(val)
        elseif spec.kind == :lograterows
            return lograterows_forward(val)
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
            return exp.(val)
        elseif spec.kind == :logit
            return logit_inverse.(val)
        elseif spec.kind == :elementwise
            return [_scalar_inverse(spec.mask[j], val[j]) for j in eachindex(val)]
        elseif spec.kind == :cholesky
            n1, n2 = spec.size
            T = reshape(val, n1, n2)
            return cholesky_inverse(T)
        elseif spec.kind == :expm
            n1, n2 = spec.size
            T = _sym_from_upper(val, n1)
            return expm_inverse(T)
        elseif spec.kind == :stickbreak
            return stickbreak_inverse(val)
        elseif spec.kind == :stickbreakrows
            n = spec.size[1]
            return _stickbreakrow_inverse(val, n)
        elseif spec.kind == :lograterows
            n = spec.size[1]
            return lograterows_inverse(val, n)
        else
            return val
        end
    end
    return ComponentArray(NamedTuple{Tuple(names)}(Tuple(vals)))
end
