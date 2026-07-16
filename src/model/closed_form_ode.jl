export ClosedFormPlan

using Symbolics
import SciMLBase
import ForwardDiff
import Distributions

# ---------------------------------------------------------------------------
# Closed-form linear ODE fast path (Tier 0: diagonal, constant-coefficient A).
#
# When a `@DifferentialEquation` system is provably linear in its states with a
# diagonal, time-constant coefficient matrix and constant forcing, each state
# decouples to a scalar ODE `xᵢ' = aᵢ·xᵢ + bᵢ` with the elementary closed form
#     xᵢ(t) = xᵢ(0) + Δ·(aᵢ·xᵢ(0) + bᵢ)·φ(aᵢ·Δ),   φ(z) = expm1(z)/z
# (equal to `exp(aᵢΔ)xᵢ(0) + (bᵢ/aᵢ)(exp(aᵢΔ)−1)`, with the aᵢ→0 limit built in).
# Scalar `exp` only: no matrix exponential, no eigendecomposition, no nested AD.
# ---------------------------------------------------------------------------

"""
    ClosedFormPlan

Result of the build-time symbolic analysis of a `@DifferentialEquation` block.
`eligible == true` means the system is constant-coefficient linear in its states
with constant forcing, so numerical integration can be replaced by a closed form.
`mode` records the family: `:diagonal` (decoupled — per-state scalar closed form),
`:bateman` (two-state sequential, e.g. 1-cmt oral — scalar Bateman form), `:linear`
(general/triangular constant `A` — dense matrix exponential), or `:none` when
ineligible. `n` is the number of states. `:diagonal`/`:bateman` are scalar-fast;
`:linear` is exact but, for small dense systems, not necessarily faster than numerical.
"""
struct ClosedFormPlan
    mode::Symbol
    eligible::Bool
    n::Int
    cf_states::Vector{Int}
end

const _NO_CLOSED_FORM = ClosedFormPlan(:none, false, 0, Int[])

@inline get_cf_mode(p::ClosedFormPlan) = p.mode
@inline is_cf_eligible(p::ClosedFormPlan) = p.eligible
@inline get_cf_n(p::ClosedFormPlan) = p.n
@inline get_cf_states(p::ClosedFormPlan) = p.cf_states

# Whole system solved in closed form (as opposed to a decoupled linear subset).
_cf_is_whole(p::ClosedFormPlan) = p.eligible && length(p.cf_states) == p.n

struct _NotClosedForm <: Exception end

# ---------------------------------------------------------------------------
# Symbolic eligibility proof (exact, over all parameter values).
# ---------------------------------------------------------------------------

mutable struct _CFCtx
    k::Int
end
# A fresh opaque symbolic constant standing in for a state/time-free coefficient
# (parameter, constant covariate, or any state-free subexpression).
_cf_fresh(ctx::_CFCtx) = (ctx.k += 1; Symbolics.variable(Symbol("_cf_c", ctx.k)))

function _cf_has(ex, pred::F) where {F}
    ex isa Symbol && return pred(ex)
    ex isa Expr || return false
    for a in ex.args
        _cf_has(a, pred) && return true
    end
    return false
end
_cf_has_state(ex, states::Set{Symbol}) = _cf_has(ex, s -> s in states)
_cf_has_t(ex) = _cf_has(ex, s -> s === :t || s === :ξ)

function _cf_apply_op(op::Symbol, args)
    op === :+ && return sum(args)
    op === :- && return length(args) == 1 ? -args[1] : args[1] - args[2]
    op === :* && return prod(args)
    op === :/ && return args[1] / args[2]
    op === :^ && return args[1]^args[2]
    throw(_NotClosedForm())
end

# Map a raw RHS `Expr` to a Symbolics expression in the state variables. Any
# maximal state-free subexpression collapses to a single opaque constant (or, if
# it depends on time, to a term carrying the time variable so time-dependence is
# detected). A state flowing through a non-arithmetic call, index, or property
# access cannot be proven linear ⇒ ineligible.
function _cf_to_sym(ex, states::Set{Symbol}, smap, tvar, ctx::_CFCtx)
    if !_cf_has_state(ex, states)
        return _cf_has_t(ex) ? _cf_fresh(ctx) * tvar : _cf_fresh(ctx)
    end
    if ex isa Symbol
        haskey(smap, ex) && return smap[ex]
        throw(_NotClosedForm())
    elseif ex isa Expr && ex.head === :call
        op = ex.args[1]
        op isa Symbol && op in (:+, :-, :*, :/, :^) ||
            throw(_NotClosedForm())
        return _cf_apply_op(op,
            [_cf_to_sym(a, states, smap, tvar, ctx) for a in ex.args[2:end]])
    end
    throw(_NotClosedForm())
end

# Replace `signal(t)` calls in a state RHS with the signal's defining expression,
# so state-dependence flowing through a derived signal is analysed too.
function _cf_inline_signals(ex, signals::Set{Symbol}, defs, depth::Int)
    depth > 32 && throw(_NotClosedForm())
    ex isa Expr || return ex
    if ex.head === :call && length(ex.args) == 2 && ex.args[1] isa Symbol &&
       ex.args[1] in signals && (ex.args[2] === :t || ex.args[2] === :ξ)
        haskey(defs, ex.args[1]) || throw(_NotClosedForm())
        return _cf_inline_signals(defs[ex.args[1]], signals, defs, depth + 1)
    end
    return Expr(ex.head,
        (_cf_inline_signals(a, signals, defs, depth) for a in ex.args)...)
end

_cf_is_zero(e) = isequal(Symbolics.expand(e), 0)

function _cf_depends_on(e, svars, tvar)
    for v in Symbolics.get_variables(e)
        (isequal(v, Symbolics.value(tvar)) ||
         any(sv -> isequal(v, Symbolics.value(sv)), svars)) && return true
    end
    return false
end

"""
    _detect_closed_form(model) -> ClosedFormPlan

Symbolically identify the largest subset of ODE states that forms a self-contained,
constant-coefficient linear system with constant forcing — i.e. states whose RHS is
linear with time-constant coefficients and references only other such states (no
nonlinear term, no time/dynamic-covariate dependence, no coupling to excluded
states). That subset (`cf_states`) is solvable in closed form independently; the
rest are integrated numerically. `mode` is `:diagonal` (the subset's coefficient
matrix is diagonal — scalar closed form) or `:linear` (general/triangular — matrix-
exponential action). `cf_states == 1:n` means the whole system is closed-form.
Anything untraceable is conservatively excluded (numerical is always correct).
"""
function _detect_closed_form(model)
    de = model.de.de
    de === nothing && return _NO_CLOSED_FORM
    # Crossing (time-to-event) models need solver-native event detection and
    # derivative interpolation on the trajectory, which the closed-form solution
    # object does not provide — keep them on the numerical path.
    isempty(get_formulas_crossings(model.formulas.formulas)) || return _NO_CLOSED_FORM
    states = get_de_states(de)
    n = length(states)
    n == 0 && return _NO_CLOSED_FORM
    try
        state_rhs = Dict{Symbol, Any}()
        signal_defs = Dict{Symbol, Any}()
        for ln in get_de_lines(de)
            if ln isa Expr && ln.head === :call && ln.args[1] === :~
                state_rhs[ln.args[2].args[2]] = ln.args[3]
            elseif ln isa Expr && ln.head === :(=)
                signal_defs[ln.args[1].args[1]] = ln.args[2]
            end
        end
        signalset = Set(get_de_signals(de))
        stateset = Set(states)
        Symbolics.@variables _t_cf
        svars = [Symbolics.variable(Symbol("_cf_x", i)) for i in 1:n]
        smap = Dict(states[i] => svars[i] for i in 1:n)
        subs = Dict(svars[i] => 0 for i in 1:n)
        # Per-state: is the RHS linear with time-constant coefficients and constant
        # forcing? If so record its state dependencies (which svars it uses).
        candidate = falses(n)
        deps = [Int[] for _ in 1:n]
        syms = Vector{Any}(nothing, n)
        for i in 1:n
            haskey(state_rhs, states[i]) || continue
            try
                ctx = _CFCtx(0)
                ex = _cf_inline_signals(state_rhs[states[i]], signalset, signal_defs, 0)
                si = _cf_to_sym(ex, stateset, smap, _t_cf, ctx)
                ok = true
                d = Int[]
                for j in 1:n
                    c = Symbolics.derivative(si, svars[j])
                    _cf_depends_on(c, svars, _t_cf) && (ok = false; break)  # nonlinear/time-varying coeff
                    _cf_is_zero(c) || push!(d, j)
                end
                ok || continue
                _cf_depends_on(Symbolics.substitute(si, subs), svars, _t_cf) && continue  # time-varying forcing
                candidate[i] = true
                deps[i] = d
                syms[i] = si
            catch e
                e isa _NotClosedForm || rethrow(e)
            end
        end
        # Largest self-contained block: drop any candidate that depends on a
        # non-candidate, to a fixpoint.
        inblock = copy(candidate)
        changed = true
        while changed
            changed = false
            for i in 1:n
                inblock[i] || continue
                if any(j -> !inblock[j], deps[i])
                    inblock[i] = false
                    changed = true
                end
            end
        end
        cf_states = findall(inblock)
        isempty(cf_states) && return _NO_CLOSED_FORM
        offdiagonal = any(i -> any(j -> j != i, deps[i]), cf_states)
        mode = offdiagonal ? :linear : :diagonal
        # Bateman special case: whole 2-state system, one state independent (upstream)
        # with zero forcing driving the other — the ubiquitous 1-cmt oral (depot →
        # central). Solved by fast scalar exponentials, so it stays in `:auto`.
        if mode === :linear && length(cf_states) == n == 2
            up = !(2 in deps[1]) ? 1 : (!(1 in deps[2]) ? 2 : 0)
            if up != 0 && _cf_is_zero(Symbolics.substitute(syms[up], subs))
                mode = :bateman
            end
        end
        return ClosedFormPlan(mode, true, n, cf_states)
    catch e
        e isa _NotClosedForm && return _NO_CLOSED_FORM
        rethrow(e)
    end
end

# ---------------------------------------------------------------------------
# Scalar closed form (ForwardDiff-safe: scalar `exp`/`expm1` only).
# ---------------------------------------------------------------------------

# φ(z) = expm1(z)/z with the smooth z→0 limit. The Taylor branch near 0 avoids
# 0/0 and keeps the value and derivatives continuous for ForwardDiff.
@inline function _phi_expm1(z)
    abs(z) < 1e-4 &&
        return @evalpoly(z, 1.0, 0.5, inv(6.0), inv(24.0), inv(120.0))
    return expm1(z) / z
end

# x(Δ) = x0 + Δ·(a·x0 + b)·φ(a·Δ) — the diagonal scalar closed form, exact and
# regular as a→0.
@inline _cf_state_value(a, b, x0, Δ) = x0 + Δ * (a * x0 + b) * _phi_expm1(a * Δ)

# sinh(x)/x with the smooth x→0 limit (even Taylor). Underlies the divided
# difference below, which is what makes the ka ≈ ke case numerically stable.
@inline function _sinhc(x)
    abs(x) < 1e-3 && return @evalpoly(x*x, 1.0, inv(6.0), inv(120.0), inv(5040.0))
    return sinh(x) / x
end

# Divided difference (exp(pΔ) − exp(qΔ)) / (p − q), evaluated as
# exp(mΔ)·Δ·sinhc(dΔ) with m=(p+q)/2, d=(p−q)/2 — no 1/(p−q) cancellation, so it is
# stable through and at p = q (the confluent limit Δ·exp(pΔ)). The Bateman kernel.
@inline function _cf_expdd(p, q, Δ)
    m = (p + q) / 2
    d = (p - q) / 2
    return exp(m * Δ) * Δ * _sinhc(d * Δ)
end

# Augmented matrix `[A b; 0 0]` whose exponential action advances `x' = A x + b`
# (last row 0 keeps the appended constant 1 fixed). Constant within an event-free
# segment, so it is built once per segment and reused across that segment's grid.
function _cf_augmented(A, b)
    n = length(b)
    T = promote_type(eltype(A), eltype(b))
    M = zeros(T, n + 1, n + 1)
    @inbounds M[1:n, 1:n] .= A
    @inbounds M[1:n, n + 1] .= b
    return M
end

# Dense matrix exponential of the small (compartment-sized) augmented matrix,
# stable at confluent/repeated eigenvalues (e.g. ka ≈ ke) — unlike `A⁻¹`/divided-
# difference forms. ForwardDiff-safe: value via the Float64 `exp`, each partial via
# the block-2×2 Padé Fréchet derivative (`_expm_frechet`); a `Dual` specialization
# is needed because `exp(::Matrix{Dual})` has no method. Nested Duals fall back to
# the generic path. (Used only on the opt-in `:linear`/hybrid path — for a small
# dense system the numerical solver is hard to beat on the gradient, so `:auto`
# does not route general-linear systems here; see `get_closed_form_plan`.)
_cf_matexp(M::AbstractMatrix{<:AbstractFloat}) = exp(M)
function _cf_matexp(M::AbstractMatrix{ForwardDiff.Dual{
        Tg, V, N}}) where {
        Tg, V <: AbstractFloat, N}
    Mv = ForwardDiff.value.(M)
    Ev = exp(Mv)
    dEs = ntuple(N) do k
        _, F = _expm_frechet(Mv, ForwardDiff.partials.(M, k))
        F
    end
    out = Matrix{ForwardDiff.Dual{Tg, V, N}}(undef, size(M))
    @inbounds for j in axes(M, 2), i in axes(M, 1)
        out[i, j] = ForwardDiff.Dual{Tg}(Ev[i, j], ntuple(k -> dEs[k][i, j], N)...)
    end
    return out
end

# Propagate a linear system `x' = A x + b` over `Δ` from `x0` via the augmented
# matrix exponential `exp([A b; 0 0]·Δ)·[x0; 1]` (the last row keeps the appended
# 1 fixed). Handles general/triangular `A`, repeated eigenvalues, and singular `A`
# with no `A⁻¹`. ForwardDiff-safe (single Dual level: `A`, `b`, `x0` carry Duals).
function _cf_expv(A, b, x0, Δ)
    n = length(x0)
    v = vcat(x0, one(eltype(x0)))
    return (_cf_matexp(_cf_augmented(A, b) .* Δ) * v)[1:n]
end

# Two-state sequential (Bateman) closed form: one state (`u`) is independent with
# zero forcing (`xu' = au·xu`), the other (`d`) is driven by it
# (`xd' = c·xu + ad·xd + bd`). Scalar exponentials + the stable divided difference
# — fast and exact, including ka ≈ ke. `A[1,2] == 0` ⇒ state 1 is upstream.
@inline function _cf_bateman(A, b, x0, Δ)
    u = iszero(A[1, 2]) ? 1 : 2
    d = 3 - u
    au = A[u, u]
    ad = A[d, d]
    xu = _cf_state_value(au, b[u], x0[u], Δ)
    xd = _cf_state_value(ad, b[d], x0[d], Δ) + A[d, u] * x0[u] * _cf_expdd(au, ad, Δ)
    return u == 1 ? [xu, xd] : [xd, xu]
end

# Advance `x' = A x + b` over `Δ` from `x0`: per-state scalar (:diagonal), the
# two-state sequential scalar form (:bateman), or the matrix-exp action (:linear).
function _cf_propagate(mode::Symbol, A, b, x0, Δ)
    mode === :diagonal &&
        return [_cf_state_value(A[i, i], b[i], x0[i], Δ) for i in eachindex(x0)]
    mode === :bateman && return _cf_bateman(A, b, x0, Δ)
    return _cf_expv(A, b, x0, Δ)
end

# Full state vector at time `t`: locate its segment (event-free interval), then
# propagate from that segment's start state with its constant forcing.
function _cf_state_vector(mode::Symbol, A, seg_t, seg_x0, seg_b, t)
    k = max(searchsortedlast(seg_t, t), 1)
    return _cf_propagate(mode, A, seg_b[k], seg_x0[k], t - seg_t[k])
end

# Materialize the state on a whole grid. For `:linear` the augmented matrix and
# start vector are constant per segment, so they are built once per segment and
# reused across that segment's grid points (avoids rebuilding them per point).
function _cf_materialize(mode::Symbol, A, seg_t, seg_x0, seg_b, grid)
    return [_cf_state_vector(mode, A, seg_t, seg_x0, seg_b, tk) for tk in grid]
end

"""
    ClosedFormLinearSolution

Solution-shaped object returned by the closed-form fast path. Not an
`AbstractODESolution`: state access routes through the generic `sol(t; idxs=i)`
accessor. `mode` is `:diagonal` (per-state scalar closed form) or `:linear`
(matrix-exponential action). The trajectory is piecewise over event-free
segments (`seg_t`/`seg_x0`/`seg_b`); with no events there is a single segment.
Carries a materialized `t`/`u` grid for lookup, inspection, and plotting.
"""
struct ClosedFormLinearSolution{T}
    mode::Symbol
    t0::Float64
    A::Matrix{T}
    seg_t::Vector{Float64}
    seg_x0::Vector{Vector{T}}
    seg_b::Vector{Vector{T}}
    t::Vector{Float64}
    u::Vector{Vector{T}}
end

@inline function (s::ClosedFormLinearSolution)(t; idxs::Integer)
    # Exact-time grid lookup first (every :saveat eval time is on the grid), which
    # avoids recomputing the matrix-exp action per state in :linear mode.
    i = searchsortedfirst(s.t, t)
    (i <= length(s.t) && @inbounds(s.t[i]==t)) && return @inbounds s.u[i][idxs]
    return _cf_state_vector(s.mode, s.A, s.seg_t, s.seg_x0, s.seg_b, t)[idxs]
end

SciMLBase.successful_retcode(::ClosedFormLinearSolution) = true

"""
    _closed_form_solve_de(model, compiled, u0, tspan, saveat, t0, mode; events) ->
        ClosedFormLinearSolution

Shared closed-form solve for all consumers (fit / plotting / simulation / SAEM).
Extracts the (constant) linear system exactly from the out-of-place RHS —
`b = f(0)` and `A[:, j] = f(eⱼ) − b` — keeping θ/η at a single Dual level (the
probes are plain `Float64`). `mode` selects the per-state scalar form
(`:diagonal`) or the matrix-exponential action (`:linear`). `events` (an
`EventCallbacks` or `nothing`) splits the trajectory into event-free
segments with bolus/reset state jumps and infusion folded into the forcing.
Returns `nothing` on non-finite coefficients (mirrors a failed numerical solve).
`saveat === nothing` materializes a dense grid for `.t`/`.u`.
"""
function _closed_form_solve_de(model, compiled, u0::AbstractVector, tspan, saveat,
        t0::Real, mode::Symbol; events = nothing, idxs = eachindex(u0))
    de = model.de.de
    f = get_de_f(de)
    n = length(u0)
    m = length(idxs)
    t0f = float(t0)
    z = zeros(Float64, n)
    b_full = collect(f(z, compiled, t0f))
    cu0 = collect(u0)
    b_base = [b_full[idxs[a]] for a in 1:m]
    x0 = [cu0[idxs[a]] for a in 1:m]
    T = promote_type(eltype(b_base), eltype(x0))
    # A[a,b] = ∂f_{idxs[a]}/∂u_{idxs[b]} = f(e_{idxs[b]})[idxs[a]] − b. For a self-
    # contained subset the excluded states are zero in the probe (they don't feed in).
    A = Matrix{T}(undef, m, m)
    for b in 1:m
        ej = zeros(Float64, n)
        ej[idxs[b]] = 1.0
        col = collect(f(ej, compiled, t0f))
        for a in 1:m
            @inbounds A[a, b] = col[idxs[a]] - b_full[idxs[a]]
        end
    end
    (all(x -> isfinite(ForwardDiff.value(x)), A) &&
     all(x -> isfinite(ForwardDiff.value(x)), b_base) &&
     all(x -> isfinite(ForwardDiff.value(x)), x0)) || return nothing
    seg_t, seg_x0, seg_b = _cf_build_segments(
        mode, A, convert(Vector{T}, b_base), convert(Vector{T}, x0), t0f, tspan, events)
    grid = saveat === nothing ?
           collect(range(float(tspan[1]), float(tspan[2]); length = 200)) :
           collect(float.(saveat))
    uvals = _cf_materialize(mode, A, seg_t, seg_x0, seg_b, grid)
    return ClosedFormLinearSolution{T}(mode, t0f, A, seg_t, seg_x0, seg_b, grid, uvals)
end

# No events: a single event-free segment spanning the whole solve.
function _cf_build_segments(mode, A, b_base::Vector{T}, x0::Vector{T}, t0f, tspan,
        ::Nothing) where {T}
    return Float64[t0f], Vector{T}[x0], Vector{T}[b_base]
end

# With events (duck-typed `EventCallbacks`; that type is defined after this file):
# split the trajectory at each mid-trajectory event time. Initial (t0) events are
# already folded into `x0`/`init_infusion_rates` by `_apply_initial_events!`. Within
# each segment `A` and forcing (`b_base + infusion`) are constant; at each event we
# propagate to it, apply resets then boluses (state jumps), and update the infusion.
function _cf_build_segments(mode, A, b_base::Vector{T}, x0::Vector{T}, t0f, tspan,
        events) where {T}
    tend = float(tspan[2])
    etimes = sort!(filter(t -> t > t0f && t <= tend, collect(events.all_times)))
    infusion = collect(Float64, events.init_infusion_rates)
    seg_t = Float64[t0f]
    seg_x0 = Vector{T}[x0]
    seg_b = Vector{T}[b_base .+ infusion]
    for e in etimes
        xnew = _cf_propagate(mode, A, seg_b[end], seg_x0[end], e - seg_t[end])
        if haskey(events.reset_by_time, e)
            for (idx, val) in events.reset_by_time[e]
                @inbounds xnew[idx] = val
            end
        end
        if haskey(events.bolus_by_time, e)
            delta = events.bolus_by_time[e]
            @inbounds for j in eachindex(delta)
                delta[j] == 0.0 || (xnew[j] += delta[j])
            end
        end
        if haskey(events.rate_delta_by_time, e)
            d = events.rate_delta_by_time[e]
            @inbounds for j in eachindex(d)
                infusion[j] += d[j]
            end
        end
        push!(seg_t, e)
        push!(seg_x0, xnew)
        push!(seg_b, b_base .+ infusion)
    end
    return seg_t, seg_x0, seg_b
end

# --- Aqua ambiguity disambiguation (SymbolicsDistributionsExt) ---------------
# Symbolics is a dependency, so `SymbolicsDistributionsExt` loads and overloads
# `pdf/logpdf/cdf/logcdf/quantile(::Distribution, ::Num|::Arr)`. Those are
# ambiguous with NoLimits' custom-distribution methods (scalar `::Real`/untyped
# and array overloads). NoLimits never evaluates these distributions on symbolic
# inputs, so the methods below exist only to keep resolution unambiguous; they
# forward (via `invoke`, avoiding self-recursion) to the ordinary handlers. Our
# owned types stay strictly more specific than the foreign abstract signatures,
# so no crossed-specificity ambiguities arise.
# Symbolic scalar types from the two Distributions extensions that load alongside
# Symbolics (`Num` from SymbolicsDistributionsExt, `BasicSymbolic` from
# SymbolicUtilsDistributionsExt).
const _CF_SYM_SCALARS = (Symbolics.Num, Symbolics.SymbolicUtils.BasicSymbolic)

# Untyped-observation distributions (scalar `y::Any`, plus array shapes): their
# methods sit on the exact union/type below, so disambiguation must too. All three
# symbolic argument types (`Num`, `BasicSymbolic`, `Arr`) collide with `::Any`.
for T in (_ObservedStatesMarkovModel, CoarsedObservedStatesMarkovModel),
    f in (:pdf, :logpdf, :cdf, :logcdf, :quantile)

    for S in _CF_SYM_SCALARS
        @eval function Distributions.$f(d::$T, x::$S)
            invoke(Distributions.$f, Tuple{$T, Any}, d, x)
        end
    end
    @eval function Distributions.$f(d::$T, x::Symbolics.Arr)
        invoke(Distributions.$f, Tuple{$T, Any}, d, x)
    end
end

# Real-/vector-observation distributions: only `::Num` (<:Real) and `::Arr`
# collide with their `::Real`/`::AbstractVector` methods.
for T in (:ContinuousTimeDiscreteStatesHMM, :DiscreteTimeDiscreteStatesHMM,
        :MVContinuousTimeDiscreteStatesHMM, :MVDiscreteTimeDiscreteStatesHMM,
        :NormalizingPlanarFlow),
    f in (:pdf, :logpdf, :cdf, :logcdf, :quantile)

    @eval function Distributions.$f(d::$T, x::Symbolics.Num)
        invoke(Distributions.$f, Tuple{$T, Real}, d, x)
    end
    @eval function Distributions.$f(d::$T, x::Symbolics.Arr)
        invoke(Distributions.$f, Tuple{$T, AbstractArray}, d, x)
    end
end
