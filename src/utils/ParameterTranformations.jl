using LinearAlgebra
using Lux
import ForwardDiff

using ComponentArrays

export ForwardTransform
export TransformSpec
export InverseTransform
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
export liepsd_forward
export liepsd_inverse
export apply_inv_jacobian_T

# Layout for a structured (block-diagonal and/or fixed-eigenvalue) Lie PSD matrix.
# `free_idx`/`fixed_idx` partition the positions `1:L` of the full parameter vector
# `[λ (n); α (nA)]` (L = n(n+1)/2); the transformed (optimizer) vector holds the free
# entries in `free_idx` order, while `fixed_idx` entries are pinned to `fixed_val`
# (log-eigenvalues for fixed eigenvalues, `0` for cross-block rotation coefficients).
struct LiePSDLayout
    n::Int
    blocks::Vector{Int}
    free_idx::Vector{Int}
    fixed_idx::Vector{Int}
    fixed_val::Vector{Float64}
end

struct TransformSpec
    name::Symbol
    kind::Symbol
    size::Tuple{Int, Int}
    mask::Union{Nothing, Vector{Symbol}}
    lie::Union{Nothing, LiePSDLayout}
end
# Backward-compatible 4-arg constructor (all non-Lie specs carry no layout).
function TransformSpec(name::Symbol, kind::Symbol, size::Tuple{Int, Int},
        mask::Union{Nothing, Vector{Symbol}})
    TransformSpec(name, kind, size, mask, nothing)
end

# `out_axes`/`n_out` (when provided) describe the output ComponentArray layout and
# enable a type-stable assembly path: the legacy `ComponentArray(NamedTuple{...})`
# construction is dynamic (runtime `names`) and routes through ComponentArrays'
# `make_idx`, whose IdDict lookups (`jl_eqtable_get`) have no Enzyme forward-mode
# rule. The 2-arg constructors keep the legacy dynamic path for ad-hoc transforms
# (e.g. restricted transforms in plotting/UQ).
struct ForwardTransform{A}
    names::Vector{Symbol}
    specs::Vector{TransformSpec}
    out_axes::A
    n_out::Int
end
ForwardTransform(names, specs) = ForwardTransform(names, specs, nothing, -1)

struct InverseTransform{A}
    names::Vector{Symbol}
    specs::Vector{TransformSpec}
    out_axes::A
    n_out::Int
end
InverseTransform(names, specs) = InverseTransform(names, specs, nothing, -1)

function (ft::ForwardTransform)(θ::ComponentArray)
    if ft.out_axes === nothing
        vals = _transform_vals(θ, ft.names, ft.specs)
        return _assemble_ca(vals, ft.names, ft.out_axes, ft.n_out, eltype(θ))
    end
    # Axes path: write each parameter's transformed values straight into the flat
    # output buffer. The legacy route collected per-parameter results in a boxed
    # `Vector{Any}` (the `map` closure's branches return heterogeneous types) and
    # then re-walked it — measurably hot, since the inverse runs inside the ODE RHS
    # and both run once per objective evaluation.
    flat = Vector{eltype(θ)}(undef, ft.n_out)
    k = 1
    @inbounds for i in eachindex(ft.names)
        k = _write_forward_spec!(flat, k, ft.specs[i], getproperty(θ, ft.names[i]))
    end
    k == ft.n_out + 1 ||
        error("Transform output length mismatch: expected $(ft.n_out), got $(k - 1).")
    return ComponentArray(flat, ft.out_axes)
end

function (it::InverseTransform)(θ::ComponentArray)
    if it.out_axes === nothing
        vals = _inverse_vals(θ, it.names, it.specs)
        return _assemble_ca(vals, it.names, it.out_axes, it.n_out, eltype(θ))
    end
    flat = Vector{eltype(θ)}(undef, it.n_out)
    k = 1
    @inbounds for i in eachindex(it.names)
        k = _write_inverse_spec!(flat, k, it.specs[i], getproperty(θ, it.names[i]))
    end
    k == it.n_out + 1 ||
        error("Transform output length mismatch: expected $(it.n_out), got $(k - 1).")
    return ComponentArray(flat, it.out_axes)
end

# Direct flat-buffer writers for the axes path. Branch-for-branch identical math to
# `_transform_vals` / `_inverse_vals` (which remain as the legacy no-axes route):
# scalar kinds write elementwise with no temporaries; matrix/simplex kinds reuse the
# shared helpers and copy their (small) results in the same element order
# (`for x in M` is column-major, matching the legacy `vec`-based writes).
function _write_forward_spec!(flat::Vector, k::Int, spec::TransformSpec, val)
    kind = spec.kind
    if kind === :log
        if val isa Number
            flat[k] = log(val)
            return k + 1
        end
        for x in val
            flat[k] = log(x)
            k += 1
        end
        return k
    elseif kind === :logit
        if val isa Number
            flat[k] = logit_forward(val)
            return k + 1
        end
        for x in val
            flat[k] = logit_forward(x)
            k += 1
        end
        return k
    elseif kind === :elementwise
        for j in eachindex(val)
            flat[k] = _scalar_forward(spec.mask[j], val[j])
            k += 1
        end
        return k
    elseif kind === :cholesky
        return _write_flat!(flat, cholesky_forward(val), k)
    elseif kind === :expm
        M = expm_forward(val)
        n = size(M, 1)
        for j in 1:n, i in 1:j
            flat[k] = M[i, j]
            k += 1
        end
        return k
    elseif kind === :stickbreak
        return _write_flat!(flat, stickbreak_forward(val), k)
    elseif kind === :stickbreakrows
        n = size(val, 1)
        for i in 1:n
            k = _write_flat!(flat, stickbreak_forward(view(val, i, :)), k)
        end
        return k
    elseif kind === :lograterows
        return _write_flat!(flat, lograterows_forward(val), k)
    elseif kind === :lie
        fwd = spec.lie === nothing ? liepsd_forward(val) :
              _liepsd_forward_layout(val, spec.lie)
        return _write_flat!(flat, fwd, k)
    else
        return _write_flat!(flat, val, k)
    end
end

function _write_inverse_spec!(flat::Vector, k::Int, spec::TransformSpec, val)
    kind = spec.kind
    if kind === :log
        if val isa Number
            flat[k] = exp(val)
            return k + 1
        end
        for x in val
            flat[k] = exp(x)
            k += 1
        end
        return k
    elseif kind === :logit
        if val isa Number
            flat[k] = logit_inverse(val)
            return k + 1
        end
        for x in val
            flat[k] = logit_inverse(x)
            k += 1
        end
        return k
    elseif kind === :elementwise
        for j in eachindex(val)
            flat[k] = _scalar_inverse(spec.mask[j], val[j])
            k += 1
        end
        return k
    elseif kind === :cholesky
        n1, n2 = spec.size
        return _write_flat!(flat, cholesky_inverse(reshape(val, n1, n2)), k)
    elseif kind === :expm
        n1, _ = spec.size
        return _write_flat!(flat, expm_inverse(_sym_from_upper(val, n1)), k)
    elseif kind === :stickbreak
        return _write_flat!(flat, stickbreak_inverse(val), k)
    elseif kind === :stickbreakrows
        n = spec.size[1]
        return _write_flat!(flat, _stickbreakrow_inverse(val, n), k)
    elseif kind === :lograterows
        n = spec.size[1]
        return _write_flat!(flat, lograterows_inverse(val, n), k)
    elseif kind === :lie
        inv = spec.lie === nothing ? liepsd_inverse(val) :
              _liepsd_inverse_layout(val, spec.lie)
        return _write_flat!(flat, inv, k)
    else
        return _write_flat!(flat, val, k)
    end
end

# Legacy dynamic assembly (runtime names; ForwardDiff-compatible, not Enzyme-forward).
function _assemble_ca(
        vals, names::Vector{Symbol}, ::Nothing, n_out::Int, ::Type{T}) where {T}
    return ComponentArray(NamedTuple{Tuple(names)}(Tuple(vals)))
end

# Type-stable assembly: write the heterogeneous per-parameter values into one flat
# vector (single up-front allocation, assigned-once slots) and wrap it with the
# precomputed axes.
function _assemble_ca(
        vals, names::Vector{Symbol}, out_axes::Tuple, n_out::Int, ::Type{T}) where {T}
    flat = Vector{T}(undef, n_out)
    k = 1
    for v in vals
        k = _write_flat!(flat, v, k)
    end
    k == n_out + 1 ||
        error("Transform output length mismatch: expected $(n_out), got $(k - 1).")
    return ComponentArray(flat, out_axes)
end

_write_flat!(flat::Vector, v::Number, k::Int) = (flat[k] = v; k + 1)

function _write_flat!(flat::Vector, v::AbstractArray, k::Int)
    for x in vec(v)
        flat[k] = x
        k += 1
    end
    return k
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

# Safe log-scale inverse Jacobian: g * exp(t), but returns 0 when g == 0 and exp(t) = Inf.
# Without this guard, 0 * Inf = NaN in IEEE arithmetic, which corrupts gradients when
# a parameter is driven to an extreme transformed value by the optimizer.
@inline function _safe_log_inv_jac(g::Real, t::Real)
    e = exp(t)
    isinf(e) && iszero(g) && return zero(typeof(g * e))
    return g * e
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
    # `remaining` tracks 1 - Σ_{j<i} p_j, updated by subtracting each pᵢ in the loop.
    # A simple mutating loop suffices here: the forward transform runs at parameter
    # setup and is not differentiated by the optimizer, so it need not be non-mutating.
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

Single-allocation sequential fill. `remaining` carries the running product
`∏_{j<i}(1-σ_j)` — the same left-to-right multiplication order as the historical
`cumprod` formulation, so results are bit-identical. ForwardDiff/Enzyme-forward
compatible (plain index assignment into a locally allocated, promoted-eltype
vector); a reverse-mode backend that cannot differentiate mutation would need the
old non-mutating `cumprod` form.
"""
function stickbreak_inverse(t::AbstractVector{<:Real})
    k1 = length(t)
    T = typeof(logit_inverse(one(eltype(t))))
    out = Vector{T}(undef, k1 + 1)
    remaining = one(T)
    @inbounds for i in 1:k1
        σi = logit_inverse(t[i])
        out[i] = σi * remaining
        remaining *= one(T) - σi
    end
    out[k1 + 1] = remaining
    return out
end

# Apply stickbreak_forward row-wise to an n×n matrix; returns n*(n-1) flat vector.
function _stickbreakrow_forward(P::AbstractMatrix{<:Real})
    n = size(P, 1)
    r1 = stickbreak_forward(view(P, 1, :))
    out = Vector{eltype(r1)}(undef, n * (n - 1))
    copyto!(out, 1, r1, 1, n - 1)
    for i in 2:n
        ri = stickbreak_forward(view(P, i, :))
        copyto!(out, (i - 1) * (n - 1) + 1, ri, 1, n - 1)
    end
    return out
end

# Apply stickbreak_inverse row-wise; reconstructs n×n row-stochastic matrix.
function _stickbreakrow_inverse(t::AbstractVector{<:Real}, n::Int)
    k1 = n - 1
    r1 = stickbreak_inverse(view(t, 1:k1))
    M = Matrix{eltype(r1)}(undef, n, n)
    for j in 1:n
        M[1, j] = r1[j]
    end
    for i in 2:n
        ri = stickbreak_inverse(view(t, ((i - 1) * k1 + 1):(i * k1)))
        for j in 1:n
            M[i, j] = ri[j]
        end
    end
    return M
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
        # Accumulate the row sum inline (same j-ascending order as the previous
        # filtered-generator `sum`, so results are bit-identical) instead of
        # re-walking the row through a lazy generator.
        rowsum = zero(T)
        for j in 1:n
            i == j && continue
            qij = exp(T(t[idx]))
            Q[i, j] = qij
            rowsum += qij
            idx += 1
        end
        Q[i, i] = -rowsum
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
    kind === :log && return log(x)
    kind === :logit && return logit_forward(x)
    return x
end

@inline function _scalar_inverse(kind::Symbol, x::Real)
    kind === :log && return exp(x)
    kind === :logit && return logit_inverse(x)
    return x
end

@inline function _scalar_inv_jacobian(kind::Symbol, g::Real, x::Real)
    kind === :log && return _safe_log_inv_jac(g, x)
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
    # `exp(Symmetric(T))` is symmetric in value, but under ForwardDiff its derivative
    # (Fréchet) partials can come back slightly asymmetric. Since Ω(T) is symmetric for
    # every T, the true derivative is symmetric too, so re-wrapping the result in
    # `Symmetric` before densifying both fixes the partials and guarantees an exactly
    # Hermitian matrix — required by the (stricter) `cholesky` inside MvNormal/PDMats.
    return Matrix(Symmetric(exp(Symmetric(T))))
end

# Recursively strip ForwardDiff.Dual levels to the underlying float (used only for the
# non-differentiated scaling-and-squaring count in `_matexp`).
_scalar_value(x::Real) = float(x)
_scalar_value(x::ForwardDiff.Dual) = _scalar_value(ForwardDiff.value(x))

# Matrix exponential dispatched on eltype: BLAS-exact `exp` for Blas floats (keeps the
# value and single-level Fréchet paths bit-identical), and a Dual-generic scaling-and-
# squaring Padé (Higham degree 13; only +, *, \, I) for Dual eltypes, which arise only at
# 2nd+ ForwardDiff order (nested Hessian) where `exp(::Matrix{Dual})` has no BLAS method.
_matexp(M::AbstractMatrix{<:LinearAlgebra.BlasFloat}) = exp(M)

function _matexp(M::AbstractMatrix{<:Real})
    b = (64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0,
        33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0)
    nrm = opnorm(_scalar_value.(M), 1)
    s = nrm > 5.371920351148152 ? ceil(Int, log2(nrm / 5.371920351148152)) : 0
    A = M ./ 2.0^s
    A2 = A * A
    A4 = A2 * A2
    A6 = A2 * A4
    Uodd = A * (A6 * (b[14] * A6 + b[12] * A4 + b[10] * A2) +
            b[8] * A6 + b[6] * A4 + b[4] * A2 + b[2] * I)
    Veven = A6 * (b[13] * A6 + b[11] * A4 + b[9] * A2) +
            b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I
    P = (Veven - Uodd) \ (Veven + Uodd)
    for _ in 1:s
        P = P * P
    end
    return P
end

function _expm_frechet(A::AbstractMatrix{<:Real}, E::AbstractMatrix{<:Real})
    n = size(A, 1)
    size(A, 2) == n || error("A must be square.")
    size(E, 1) == n && size(E, 2) == n || error("E must match A size.")
    TT = promote_type(eltype(A), eltype(E))
    Z = zeros(TT, n, n)
    M = [Matrix{TT}(A) Matrix{TT}(E); Z Matrix{TT}(A)]
    EM = _matexp(M)
    return EM[1:n, 1:n], EM[1:n, (n + 1):(2n)]
end

# ForwardDiff-aware matrix exponential of a symmetric matrix (single-level Dual).
# The generic eigen-based `exp(Symmetric)` has NaN / asymmetric ForwardDiff derivatives at
# repeated eigenvalues (e.g. Ω = I, the typical optimization/UQ point — the derivative of an
# eigen-based matrix function divides by eigenvalue gaps), and the Padé `exp(::Matrix)` has no
# `Dual` method. So compute the value with the eigen-exp of the (Float64) symmetric value, and
# each partial direction with the AD-safe block-2×2 Padé Fréchet derivative (`_expm_frechet`,
# which has no eigengaps), then reassemble the `Dual`. This matches the reverse-mode Jacobian
# the transform already uses and yields finite, exactly-symmetric derivatives, so the
# reconstructed covariance is accepted by the (stricter) `cholesky` inside MvNormal/PDMats.
# Single-level Duals only; second-order ForwardDiff (Hessian) falls back to finite
# differences.
function expm_inverse(T::AbstractMatrix{ForwardDiff.Dual{
        Tg, V, N}}) where {Tg, V <: AbstractFloat, N}
    n = size(T, 1)
    Tv = ForwardDiff.value.(T)
    Sv = 0.5 .* (Tv .+ Tv')                          # symmetric value
    Mv = Matrix(Symmetric(exp(Symmetric(Sv))))       # value (eigen ok — no AD here)
    dMs = ntuple(N) do k
        Pk = ForwardDiff.partials.(T, k)
        Ek = 0.5 .* (Pk .+ Pk')                      # symmetric seed direction
        _, F = _expm_frechet(Sv, Ek)
        Matrix(Symmetric(F))                          # symmetric Fréchet derivative
    end
    out = Matrix{ForwardDiff.Dual{Tg, V, N}}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        out[i, j] = ForwardDiff.Dual{Tg}(Mv[i, j], ntuple(k -> dMs[k][i, j], N)...)
    end
    return out
end

# Nested Duals (2nd+-order ForwardDiff, e.g. a Hessian through the transform): recurse on
# the value (bottoming out at the Float64 eigen leaf, so the value stays bit-identical) and
# take each partial via the Padé Fréchet derivative, whose inner matrix-exp routes through
# the Dual-generic `_matexp`. Removes the finite-difference Hessian fallback for `:expm`.
function expm_inverse(T::AbstractMatrix{ForwardDiff.Dual{
        Tg, V, N}}) where {Tg, V <: ForwardDiff.Dual, N}
    n = size(T, 1)
    Tv = ForwardDiff.value.(T)
    Sv = 0.5 .* (Tv .+ Tv')
    Mv = expm_inverse(Tv)
    dMs = ntuple(N) do k
        Pk = ForwardDiff.partials.(T, k)
        Ek = 0.5 .* (Pk .+ Pk')
        _, F = _expm_frechet(Sv, Ek)
        Matrix(Symmetric(F))
    end
    out = Matrix{ForwardDiff.Dual{Tg, V, N}}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        out[i, j] = ForwardDiff.Dual{Tg}(Mv[i, j], ntuple(k -> dMs[k][i, j], N)...)
    end
    return out
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
                    # mirror element T[j,i] (j<i) is upper-tri, packed at this index
                    v[(i - 1) * i ÷ 2 + j]
                end
            end
            for i in 1:n, j in 1:n]
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

# ── Lie-algebraic PSD parameterization (Stapor 2020, §5.3) ────────────────────
# A symmetric positive-definite matrix is written via its eigendecomposition
#   Σ = U · diag(exp λ) · Uᵀ,   U = exp(A),   A = Σₘ αₘ Aₘ  (antisymmetric),
# so the transformed vector is [λ (n) ; α (nA)] with nA = n(n-1)/2 (total n(n+1)/2).
# The log-eigenvalues λ live on ℝ (eigenvalues stay positive) and can be box-bounded;
# the rotation coefficients α are unbounded. `exp(A)` for antisymmetric A is a general
# (non-symmetric) matrix exponential with no eigengap issue (unlike the symmetric `:expm`
# scale); its single- and nested-Dual derivatives are built from the Padé Fréchet block
# below (`_expm_frechet`, routed through the Dual-generic `_matexp`).

# Recover n from the packed length L = n(n+1)/2.
function _lie_dim(L::Int)
    n = (isqrt(8L + 1) - 1) ÷ 2
    n * (n + 1) ÷ 2 == L ||
        error("Invalid Lie-parameter length $(L); not of the form n(n+1)/2.")
    return n
end

# Antisymmetric matrix from angle coefficients in lexicographic (i<j) order.
function _lie_antisym(α::AbstractVector, n::Int)
    A = zeros(eltype(α), n, n)
    k = 1
    for i in 1:n
        for j in (i + 1):n
            A[i, j] = α[k]
            A[j, i] = -α[k]
            k += 1
        end
    end
    return A
end

# Ordered list of index pairs (i, j), i < j — the enumeration used for the angle block
# (and matching `_lie_antisym`). Angle k in `1:nA` maps to `_lie_pairs(n)[k]`.
function _lie_pairs(n::Int)
    pairs = Tuple{Int, Int}[]
    for i in 1:n
        for j in (i + 1):n
            push!(pairs, (i, j))
        end
    end
    return pairs
end

"""
    liepsd_inverse(t::AbstractVector{<:Real}) -> Matrix

Map the packed vector `t = [λ (n); α (n(n-1)/2)]` to the symmetric positive-definite
matrix `Σ = exp(A) · diag(exp λ) · exp(A)ᵀ`, `A` antisymmetric built from `α`.
"""
function liepsd_inverse(t::AbstractVector{<:Real})
    L = length(t)
    n = _lie_dim(L)
    λ = @view t[1:n]
    α = @view t[(n + 1):L]
    A = _lie_antisym(α, n)
    U = exp(A)
    Σ = U * Diagonal(exp.(λ)) * U'
    # Wrap in `Symmetric` before densifying to guarantee an exactly Hermitian matrix
    # (value and ForwardDiff partials) for the stricter `cholesky` inside MvNormal/PDMats.
    return Matrix(Symmetric(Σ))
end

# ForwardDiff-aware `liepsd_inverse` (single-level Dual). LinearAlgebra's `exp!` only has
# a BLAS method, so `exp(::Matrix{Dual})` errors; mirror the `:expm` treatment and build
# the partials from the block-2×2 Padé Fréchet derivative of the matrix exponential
# (`_expm_frechet`, general-matrix, no eigengaps). The value is computed with LAPACK on the
# Float64 antisymmetric matrix. Single-level Duals only; second-order ForwardDiff
# (Hessian) falls back to finite differences.
function liepsd_inverse(t::AbstractVector{ForwardDiff.Dual{
        Tg, V, N}}) where {
        Tg, V <: AbstractFloat, N}
    L = length(t)
    n = _lie_dim(L)
    tv = ForwardDiff.value.(t)
    λv = tv[1:n]
    αv = tv[(n + 1):L]
    Av = _lie_antisym(αv, n)
    Uv = exp(Av)                                   # value (LAPACK — no AD here)
    Dv = Diagonal(exp.(λv))
    Σv = Matrix(Symmetric(Uv * Dv * Uv'))
    dΣs = ntuple(N) do k
        p = ForwardDiff.partials.(t, k)            # length-L seed direction
        dλ = @view p[1:n]
        dα = @view p[(n + 1):L]
        dAk = _lie_antisym(dα, n)
        _, dUk = _expm_frechet(Av, dAk)            # ∂exp(A)[dA]
        dDk = Diagonal(exp.(λv) .* dλ)
        dΣ = dUk * Dv * Uv' + Uv * dDk * Uv' + Uv * Dv * dUk'
        Matrix(Symmetric(dΣ))
    end
    out = Matrix{ForwardDiff.Dual{Tg, V, N}}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        out[i, j] = ForwardDiff.Dual{Tg}(Σv[i, j], ntuple(k -> dΣs[k][i, j], N)...)
    end
    return out
end

# Nested Duals (2nd+-order ForwardDiff): recurse on the value (bottoming out at the Float64
# LAPACK leaf) and build each partial via the Padé Fréchet derivative through `_matexp`.
function liepsd_inverse(t::AbstractVector{ForwardDiff.Dual{
        Tg, V, N}}) where {Tg, V <: ForwardDiff.Dual, N}
    L = length(t)
    n = _lie_dim(L)
    tv = ForwardDiff.value.(t)
    Σv = liepsd_inverse(tv)
    λv = @view tv[1:n]
    αv = @view tv[(n + 1):L]
    Av = _lie_antisym(αv, n)
    Uv = _matexp(Av)
    Dv = Diagonal(exp.(λv))
    dΣs = ntuple(N) do k
        p = ForwardDiff.partials.(t, k)
        dλ = @view p[1:n]
        dα = @view p[(n + 1):L]
        dAk = _lie_antisym(dα, n)
        _, dUk = _expm_frechet(Av, dAk)
        dDk = Diagonal(exp.(λv) .* dλ)
        dΣ = dUk * Dv * Uv' + Uv * dDk * Uv' + Uv * Dv * dUk'
        Matrix(Symmetric(dΣ))
    end
    out = Matrix{ForwardDiff.Dual{Tg, V, N}}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        out[i, j] = ForwardDiff.Dual{Tg}(Σv[i, j], ntuple(k -> dΣs[k][i, j], N)...)
    end
    return out
end

"""
    liepsd_forward(Σ::AbstractMatrix{<:Real}) -> Vector

Inverse of [`liepsd_inverse`](@ref): recover `[λ; α]` from a symmetric positive-definite
`Σ` via its eigendecomposition. Runs at model construction on `Float64`.
"""
function liepsd_forward(Σ::AbstractMatrix{<:Real})
    n = size(Σ, 1)
    Σsym = Matrix(Symmetric((Σ + Σ') / 2))
    nA = n * (n - 1) ÷ 2
    # Diagonal fast path: the eigenvector rotation is the identity, so α = 0.
    if isdiag(Σsym)
        λ = [log(Σsym[i, i]) for i in 1:n]
        return vcat(λ, zeros(nA))
    end
    F = eigen(Σsym)
    Λ = F.values
    U = Matrix(F.vectors)
    # Ensure U ∈ SO(n) (det +1): a sign flip of one eigenvector leaves Σ unchanged.
    if det(U) < 0
        U[:, 1] .= .-U[:, 1]
    end
    λ = log.(Λ)
    Alog = real.(log(U))
    As = (Alog .- Alog') ./ 2
    α = [As[i, j] for i in 1:n for j in (i + 1):n]
    return vcat(λ, α)
end

# Block-aware forward: recover the full `[λ; α]` vector for a block-diagonal `Σ` by
# eigendecomposing each block separately. Global `eigen` would mix eigenvectors across
# blocks and yield nonzero cross-block angles; the block-wise route guarantees the
# cross-block coefficients are exactly zero (so `Σ` stays block-diagonal under `liepsd_inverse`).
function _liepsd_forward_blocks(Σ::AbstractMatrix{<:Real}, blocks::Vector{Int})
    n = size(Σ, 1)
    Σsym = Matrix(Symmetric((Σ + Σ') / 2))
    pairs = _lie_pairs(n)
    pair_index = Dict(pairs[k] => k for k in eachindex(pairs))
    λ = zeros(Float64, n)
    α = zeros(Float64, length(pairs))
    for b in unique(blocks)
        dims = findall(==(b), blocks)
        if length(dims) == 1
            λ[dims[1]] = log(Σsym[dims[1], dims[1]])
            continue
        end
        sub = Matrix(Symmetric(Σsym[dims, dims]))
        if isdiag(sub)
            for (loc, d) in enumerate(dims)
                λ[d] = log(sub[loc, loc])
            end
            continue
        end
        F = eigen(sub)
        U = Matrix(F.vectors)
        det(U) < 0 && (U[:, 1] .= .-U[:, 1])
        for (loc, d) in enumerate(dims)
            λ[d] = log(F.values[loc])
        end
        Alog = real.(log(U))
        As = (Alog .- Alog') ./ 2
        nb = length(dims)
        for a in 1:nb, c in (a + 1):nb
            α[pair_index[(dims[a], dims[c])]] = As[a, c]
        end
    end
    return vcat(λ, α)
end

# Build the structured layout from an initial matrix, block labels and fixed-eigenvalue
# dimension indices. Returns `nothing` for the unstructured full case (single block, no
# fixed eigenvalues), which keeps the plain `liepsd_forward`/`liepsd_inverse` fast path.
function _build_lie_layout(Σ0::AbstractMatrix{<:Real}, blocks::Vector{Int},
        fixed_eigenvalues::Vector{Int})
    n = size(Σ0, 1)
    L = n * (n + 1) ÷ 2
    single_block = all(==(blocks[1]), blocks)
    if single_block && isempty(fixed_eigenvalues)
        return nothing
    end
    p_full = _liepsd_forward_blocks(Σ0, blocks)
    pairs = _lie_pairs(n)
    fixed_set = Set{Int}()
    fixed_idx = Int[]
    fixed_val = Float64[]
    # Fixed eigenvalues (positions 1:n) pinned at their initial log-eigenvalue.
    for i in fixed_eigenvalues
        push!(fixed_idx, i)
        push!(fixed_val, p_full[i])
        push!(fixed_set, i)
    end
    # Cross-block rotation coefficients (positions n+1:L) pinned at 0.
    for (k, (i, j)) in enumerate(pairs)
        if blocks[i] != blocks[j]
            pos = n + k
            push!(fixed_idx, pos)
            push!(fixed_val, 0.0)
            push!(fixed_set, pos)
        end
    end
    free_idx = [pos for pos in 1:L if !(pos in fixed_set)]
    order = sortperm(fixed_idx)
    return LiePSDLayout(n, blocks, free_idx, fixed_idx[order], fixed_val[order])
end

# Scatter the free transformed vector into the full `[λ; α]` layout (fixed entries pinned)
# and map to `Σ`. ForwardDiff-transparent: the element type follows `t`, so the pinned
# Float64 values are promoted to `Dual` (with zero partials) on the AD path.
function _liepsd_inverse_layout(t::AbstractVector, layout::LiePSDLayout)
    L = layout.n * (layout.n + 1) ÷ 2
    p = Vector{eltype(t)}(undef, L)
    @inbounds for k in eachindex(layout.free_idx)
        p[layout.free_idx[k]] = t[k]
    end
    @inbounds for k in eachindex(layout.fixed_idx)
        p[layout.fixed_idx[k]] = eltype(t)(layout.fixed_val[k])
    end
    return liepsd_inverse(p)
end

# Forward for the structured case: recover the full block-wise `[λ; α]` and keep the free
# entries in `free_idx` order.
function _liepsd_forward_layout(Σ::AbstractMatrix{<:Real}, layout::LiePSDLayout)
    p_full = _liepsd_forward_blocks(Σ, layout.blocks)
    return p_full[layout.free_idx]
end

function apply_inv_jacobian_T(
        it::InverseTransform, θt::ComponentArray, grad_u::ComponentArray)
    names = it.names
    specs = it.specs
    # Sequential flat fill on the θt layout. The previous `map` collected the
    # heterogeneous per-parameter results into a boxed `Vector{Any}` and then wrote
    # them back through a `setproperty!` loop — both boxing-heavy on a per-gradient
    # path (FOCEI / identifiability). Per-spec math is unchanged (`_inv_jac_spec_val`).
    flat = Vector{eltype(θt)}(undef, length(θt))
    k = 1
    for i in eachindex(names)
        v = _inv_jac_spec_val(
            specs[i], getproperty(θt, names[i]), getproperty(grad_u, names[i]))
        k = _write_flat!(flat, v, k)
    end
    k == length(θt) + 1 ||
        error("apply_inv_jacobian_T length mismatch: expected $(length(θt)), got $(k - 1).")
    return ComponentArray(flat, getaxes(θt))
end

function _inv_jac_spec_val(spec::TransformSpec, θti, gu)
    begin
        if spec.kind == :log
            return _safe_log_inv_jac.(gu, θti)
        elseif spec.kind == :logit
            return gu .* _logit_inv_jacobian.(θti)
        elseif spec.kind == :elementwise
            return [_scalar_inv_jacobian(spec.mask[j], gu[j], θti[j])
                    for j in eachindex(θti)]
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
                chunk_t = @view θti[((i - 1) * (n - 1) + 1):(i * (n - 1))]
                g_row = vec(gu[i, :])
                g_t_row = _stickbreak_inv_jacobian_T(chunk_t, g_row)
                out[((i - 1) * (n - 1) + 1):(i * (n - 1))] .= g_t_row
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
                    out[idx] = _safe_log_inv_jac(T_acc(gu[i, j]) - g_ii, T_acc(θti[idx]))
                    idx += 1
                end
            end
            return out
        elseif spec.kind == :lie
            # θti: free transformed vector; gu: n×n natural gradient w.r.t. Σ.
            # gₜ[k] = Σ_ij (∂Σ_ij/∂t_k) gu_ij = (Jᵀ vec(gu))[k], with J the Jacobian of the
            # (possibly structured) inverse map. The transform is Float64 on both gradient
            # paths that call this (Laplace/FOCEI, identifiability), so ForwardDiff is exact.
            tvec = collect(Float64, θti)
            J = spec.lie === nothing ?
                ForwardDiff.jacobian(liepsd_inverse, tvec) :
                ForwardDiff.jacobian(t -> _liepsd_inverse_layout(t, spec.lie), tvec)
            return J' * vec(collect(Float64, gu))
        else
            return gu
        end
    end
end

function _transform_vals(
        θ::ComponentArray, names::Vector{Symbol}, specs::Vector{TransformSpec})
    return map(1:length(names)) do i
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
        elseif spec.kind == :lie
            return spec.lie === nothing ? liepsd_forward(val) :
                   _liepsd_forward_layout(val, spec.lie)
        else
            return val
        end
    end
end

function _inverse_vals(
        θ::ComponentArray, names::Vector{Symbol}, specs::Vector{TransformSpec})
    return map(1:length(names)) do i
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
        elseif spec.kind == :lie
            return spec.lie === nothing ? liepsd_inverse(val) :
                   _liepsd_inverse_layout(val, spec.lie)
        else
            return val
        end
    end
end
