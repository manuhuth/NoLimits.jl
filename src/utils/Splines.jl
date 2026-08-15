export bspline_basis
export bspline_eval

"""
    bspline_basis(x::Real, knots::AbstractVector{<:Real}, degree::Integer)
    -> Vector{Float64}

Evaluate the B-spline basis functions of the given `degree` at the scalar point `x`
using the provided `knots` vector.

Returns a vector of length `length(knots) - degree - 1` containing the values of each
basis function at `x`. The knots must be sorted in non-decreasing order and `x` must
lie within `[knots[1], knots[end]]`.

# Arguments
- `x::Real`: evaluation point.
- `knots::AbstractVector{<:Real}`: sorted knot sequence (may include repeated boundary knots).
- `degree::Integer`: polynomial degree of the spline (e.g. `2` for quadratic, `3` for cubic).
"""
function bspline_basis(x::Real, knots::AbstractVector{<:Real}, degree::Integer)
    n = _bspline_validate(x, knots, degree)
    vals, mu = _bspline_nonzero(x, knots, Int(degree))
    out = zeros(eltype(vals), n)
    @inbounds for r in 1:(Int(degree) + 1)
        i = mu - Int(degree) + r - 1
        1 <= i <= n && (out[i] = vals[r])
    end
    return out
end

# Strip ForwardDiff Dual to its primal value for error messages only.
@inline _spline_primal(x::Real) = x
@inline _spline_primal(x::ForwardDiff.Dual) = _spline_primal(ForwardDiff.value(x))

@inline function _bspline_validate(x::Real, knots::AbstractVector{<:Real}, degree::Integer)
    degree >= 0 || error("degree must be non-negative")
    issorted(knots) || error("knots must be sorted non-decreasing")
    n = length(knots) - degree - 1
    n > 0 || error("Invalid knots/degree: expected length(knots) > degree+1")
    # Print the primal value: under ForwardDiff `x` is a Dual whose printed form buries
    # the offending number in a page of nested type parameters (#215).
    (isnan(x) || x < knots[1] || x > knots[end]) &&
        error("Spline input out of knot range: expected $(knots[1]) ≤ x ≤ $(knots[end]); got $(_spline_primal(x)). Extend the knots or clamp the covariate.")
    return n
end

# Iterative Cox–de Boor triangle over the knot span containing `x`.
#
# Returns `(vals, mu)` where `vals[r] = N_{mu-degree+r-1, degree}(x)` — the only
# (≤ degree+1) basis functions that can be nonzero at `x` — and `mu` is the unique
# level-0 index with `knots[mu] ≤ x < knots[mu+1]` (right-closed at `knots[end]`,
# matching the historical recursion's indicator convention).
#
# The previous implementation evaluated every basis function independently via the
# top-down recursion — `O(n · 2^degree)` redundant work. The triangle shares the
# sub-values and is `O(degree²)`. Each `N_{i,k}` is computed by the same recurrence
# expression (same zero-denominator guards, same operation order) on the same
# sub-values, so results are bit-identical to the recursion. Pure scalar arithmetic:
# ForwardDiff/Enzyme compatible.
function _bspline_nonzero(x::Real, knots::AbstractVector{<:Real}, degree::Int)
    m = length(knots)
    mu = x == knots[end] ? m - 1 : searchsortedlast(knots, x)
    T = float(promote_type(typeof(x), eltype(knots)))
    prev = zeros(T, degree + 1)   # level k-1: prev[r] = N_{mu-(k-1)+r-1, k-1}
    cur = zeros(T, degree + 1)    # level k:   cur[r]  = N_{mu-k+r-1, k}
    prev[1] = one(T)              # level 0:   N_{mu,0} = 1
    @inbounds for k in 1:degree
        for r in 1:(k + 1)
            i = mu - k + r - 1
            if !(1 <= i && i + k + 1 <= m)
                cur[r] = zero(T)   # basis function does not exist at this index
                continue
            end
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]
            # N_{i,k-1} sits at prev[r-1]; N_{i+1,k-1} at prev[r] (window shifts by 1)
            Nikm1 = (1 <= r - 1 <= k) ? prev[r - 1] : zero(T)
            Ni1km1 = (r <= k) ? prev[r] : zero(T)
            term1 = denom1 == 0 ? zero(T) : (x - knots[i]) / denom1 * Nikm1
            term2 = denom2 == 0 ? zero(T) : (knots[i + k + 1] - x) / denom2 * Ni1km1
            cur[r] = term1 + term2
        end
        prev, cur = cur, prev
    end
    return prev, mu
end

"""
    bspline_eval(x::Real, coeffs::AbstractVector{<:Real},
                 knots::AbstractVector{<:Real}, degree::Integer) -> Real

    bspline_eval(x::AbstractVector{<:Real}, coeffs::AbstractVector{<:Real},
                 knots::AbstractVector{<:Real}, degree::Integer) -> Real

Evaluate a B-spline at `x` given coefficient vector `coeffs`, knot sequence `knots`,
and polynomial `degree`.

The coefficient vector must have length `length(knots) - degree - 1`. When `x` is a
length-1 vector it is treated as a scalar.

# Arguments
- `x::Real` (or length-1 `AbstractVector`): evaluation point.
- `coeffs::AbstractVector{<:Real}`: B-spline coefficients.
- `knots::AbstractVector{<:Real}`: sorted knot sequence.
- `degree::Integer`: polynomial degree.
"""
function bspline_eval(x::Real, coeffs::AbstractVector{<:Real},
        knots::AbstractVector{<:Real}, degree::Integer)
    n = _bspline_validate(x, knots, degree)
    length(coeffs) == n ||
        error("Coefficient length mismatch: expected $(n); got $(length(coeffs)).")
    # Sum only over the ≤ degree+1 basis functions that are nonzero at `x` — no
    # length-n basis vector is materialised.
    vals, mu = _bspline_nonzero(x, knots, Int(degree))
    s = zero(promote_type(eltype(vals), eltype(coeffs)))
    @inbounds for r in 1:(Int(degree) + 1)
        i = mu - Int(degree) + r - 1
        1 <= i <= n && (s += vals[r] * coeffs[i])
    end
    return s
end

function bspline_eval(x::AbstractVector{<:Real}, coeffs::AbstractVector{<:Real},
        knots::AbstractVector{<:Real}, degree::Integer)
    length(x) == 1 || error("Spline input must be scalar; got length $(length(x)).")
    return bspline_eval(x[1], coeffs, knots, degree)
end
