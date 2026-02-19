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
    degree >= 0 || error("degree must be non-negative")
    issorted(knots) || error("knots must be sorted non-decreasing")
    n = length(knots) - degree - 1
    n > 0 || error("Invalid knots/degree: expected length(knots) > degree+1")
    (x < knots[1] || x > knots[end]) && error("x out of knot range: expected $(knots[1]) ≤ x ≤ $(knots[end]); got $(x).")
    return [_bspline_basis_one(i, degree, x, knots) for i in 1:n]
end

function _bspline_basis_one(i::Int, k::Int, x::Real, knots::AbstractVector{<:Real})
    if k == 0
        return ((knots[i] <= x && x < knots[i + 1]) || (x == knots[end] && i == length(knots) - 1)) ? one(x) : zero(x)
    end
    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]
    term1 = denom1 == 0 ? zero(x) : (x - knots[i]) / denom1 * _bspline_basis_one(i, k - 1, x, knots)
    term2 = denom2 == 0 ? zero(x) : (knots[i + k + 1] - x) / denom2 * _bspline_basis_one(i + 1, k - 1, x, knots)
    return term1 + term2
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
function bspline_eval(x::Real, coeffs::AbstractVector{<:Real}, knots::AbstractVector{<:Real}, degree::Integer)
    basis = bspline_basis(x, knots, degree)
    length(coeffs) == length(basis) || error("Coefficient length mismatch: expected $(length(basis)); got $(length(coeffs)).")
    return dot(basis, coeffs)
end

function bspline_eval(x::AbstractVector{<:Real}, coeffs::AbstractVector{<:Real}, knots::AbstractVector{<:Real}, degree::Integer)
    length(x) == 1 || error("Spline input must be scalar; got length $(length(x)).")
    return bspline_eval(x[1], coeffs, knots, degree)
end
