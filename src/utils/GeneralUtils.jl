export flatten_re_names
export flatten_re_values
export rowsoftmax

using ForwardDiff
import DiffEqBase
import Roots
import DataFrames
import DataInterpolations

# Public symbol-valued options also accept the string spelling, so the Python/R
# bindings (and serialized federated options) can pass them through unchanged.
# New public symbol options normalize through this at their entry point.
@inline _as_symbol(x::AbstractString) = Symbol(x)
@inline _as_symbol(x::Symbol) = x
@inline _as_symbol(x) = x   # nothing, NamedTuples, functions, ... pass through

# Public tabular inputs accept any Tables.jl table (CSV.File, column NamedTuple, an R
# data.frame through JuliaConnectoR, ...). Non-tables pass through so the existing
# schema validation still reports them.
_as_dataframe(df::DataFrames.DataFrame) = df
_as_dataframe(x) = DataFrames.Tables.istable(x) ? DataFrames.DataFrame(x) : x

# Same idea for the NamedTuple-shaped options: a Python dict arrives as a `PyDict`
# (an `AbstractDict`), which has no NamedTuple literal on the binding side. Values are
# recursed so nested option groups convert too, and keys go through `String` so both
# `"a" => 1` and `:a => 1` work. Field order from a dict is arbitrary, so only
# name-matched options may be normalized this way.
@inline _as_namedtuple(x::NamedTuple) = x
function _as_namedtuple(d::AbstractDict)
    # A dict keyed by anything else has no NamedTuple spelling; `constants_re` group
    # levels are the live case (integer ids, matched by value) and must survive as-is.
    all(k -> k isa Union{Symbol, AbstractString}, keys(d)) || return d
    return (; (Symbol(String(k)) => _as_namedtuple(v) for (k, v) in Base.pairs(d))...)
end
@inline _as_namedtuple(x) = x   # nothing, ComponentArrays, ... pass through

"""
    rowsoftmax(L::AbstractMatrix) -> AbstractMatrix

Row-wise softmax. Returns a row-stochastic matrix whose `i`-th row is the softmax of
`L[i, :]`, that is `P[i, j] = exp(L[i, j]) / sum_k exp(L[i, k])`, so every row sums to
one.

This is meant for use inside an `@formulas` block to turn a matrix of unnormalized
transition logits (for example the reshaped output of a neural network, optionally
shifted by random or covariate effects) into a valid transition matrix in one call,
replacing a hand-written exponentiate-and-normalize construction. Each row's maximum is
subtracted before exponentiating for numerical stability, which leaves the result
unchanged, and the function is automatic-differentiation safe.
"""
function rowsoftmax(L::AbstractMatrix)
    m = maximum(L; dims = 2)
    E = exp.(L .- m)
    return E ./ sum(E; dims = 2)
end

# OrdinaryDiffEq / DiffEqBase v7 no longer accept a `Bool` for the solver `verbose`
# keyword — it must be a SciMLLogging verbosity (e.g. `None()` for silent). v6 still
# requires a `Bool`. Resolve the right "silent"/"loud" values once for the installed
# DiffEqBase version. On v6 the `DiffEqBase.SciMLLogging` branch is never evaluated.
const _ODE_VERBOSE_SILENT = pkgversion(DiffEqBase) >= v"7" ?
    DiffEqBase.SciMLLogging.None() : false
const _ODE_VERBOSE_LOUD = pkgversion(DiffEqBase) >= v"7" ?
    DiffEqBase.SciMLLogging.Standard() : true
@inline _ode_verbose(v::Bool) = v ? _ODE_VERBOSE_LOUD : _ODE_VERBOSE_SILENT
@inline _ode_verbose(v) = v   # already a verbosity object — pass through unchanged

function flatten_re_names(name::Symbol, val)
    if val isa Number
        return Symbol[name]
    end
    vals = vec(collect(val))
    return [Symbol(name, "_", i) for i in 1:length(vals)]
end

function flatten_re_values(val)
    if val isa Number
        return [val]
    end
    return collect(vec(val))
end

@inline function _with_infusion(f!, infusion_rates)
    infusion_rates === nothing && return f!
    return function (du, u, p, t)
        f!(du, u, p, t)
        @inbounds for i in eachindex(infusion_rates)
            du[i] += infusion_rates[i]
        end
        return nothing
    end
end

# Translate any `Bool` verbose (the v6 default, or user-supplied via ode_kwargs) into
# the value the installed solver accepts. A verbosity object passes through unchanged.
# Dispatch instead of an `isa Bool` branch: the branch makes the return type a Union of
# two NamedTuple types, which survives into LLVM and breaks Enzyme forward mode
# (invalid phi node); dispatch resolves to a single concrete return type per input.
@inline _ode_normalize_verbose(kw::NamedTuple, v::Bool) = merge(
    kw, (verbose = _ode_verbose(v),)
)
@inline _ode_normalize_verbose(kw::NamedTuple, v) = kw

@inline function _ode_solve_kwargs(
        base::NamedTuple,
        extra::NamedTuple = NamedTuple(),
        overrides::NamedTuple = NamedTuple()
    )
    merged = merge((verbose = _ODE_VERBOSE_SILENT, maxiters = 5000), base, extra, overrides)
    return _ode_normalize_verbose(merged, merged.verbose)
end

# The integrator used when a model declares no `alg`. Auto-switching rather than plain
# `Tsit5()`: on a stiff stretch (TMDD quasi-steady-state, PBPK) an explicit method needs
# more steps than the budget allows, and the solve is then dropped to a -Inf likelihood.
# Measured over the six nlmixr2 benchmark models: 2.4x faster in aggregate than Tsit5
# (mavoglurant 3.7x) at a relative log-likelihood difference of ~1e-7, and it removes
# nimo's MaxIters failures entirely. `Rodas5P` rather than `Rosenbrock23` as the stiff
# partner — same speed, three orders of magnitude more accurate here.
@inline _resolve_ode_alg(alg) = alg === nothing ? AutoTsit5(Rodas5P()) : alg

# Solve with `alg`; when that attempt fails and a `fallback` algorithm is configured
# (`ODESolverConfig.fallback_alg`), re-solve the same problem with it. `reset!` restores
# state that callbacks recorded during the failed attempt (crossing times). The first
# attempt stays silent; only a failed fallback reaches the caller's drop/warning path.
# With `fallback === nothing` the second branch is dead code and the return type is the
# primary solution type.
#
# Stiffness routing (`stiffness_check`, a threshold; `nothing` disables it). With a
# fallback configured, a problem is solved with the fallback directly when
#   1. ρ(∂f/∂u at (u0, p, t0)) · |t_end − t0| > stiffness_check (primal values; a
#      non-finite Jacobian counts as stiff), or
#   2. the problem carries dual numbers and a Float64 solve of the primal problem with
#      `alg` (same kwargs, hence the same step budget) fails.
# Both cost a small fraction of a failed primary attempt, which on a stiff problem runs
# the whole step budget (for duals at the full dual cost) before the fallback starts.
function _solve_with_fallback(
        prob, alg, fallback, args::Tuple, reset! = nothing;
        stiffness_check = nothing, kw...
    )
    if fallback !== nothing && stiffness_check !== nothing &&
            _route_to_fallback(prob, alg, args, reset!, stiffness_check; kw...)
        return solve(prob, fallback, args...; kw...)
    end
    sol = solve(prob, alg, args...; kw...)
    (fallback === nothing || SciMLBase.successful_retcode(sol)) && return sol
    reset! === nothing || reset!()
    return solve(prob, fallback, args...; kw...)
end

# Primal (Float64) value of a possibly nested ForwardDiff dual, elementwise for arrays.
@inline _primal(x::ForwardDiff.Dual) = _primal(ForwardDiff.value(x))
@inline _primal(x::AbstractArray{<:ForwardDiff.Dual}) = _primal.(x)
@inline _primal(x) = x

function _route_to_fallback(prob, alg, args, reset!, threshold; kw...)
    dual = eltype(prob.u0) <: ForwardDiff.Dual
    # ponytail: duals are stripped from array parameters only (the estimation path packs
    # a flat Vector); a dual-carrying struct `p` skips the routing.
    dual && !(prob.p isa AbstractArray) && return false
    u0 = _primal(prob.u0)
    p = _primal(prob.p)
    t0, t1 = _primal(prob.tspan[1]), _primal(prob.tspan[2])
    _stiff_at_start(prob.f, SciMLBase.isinplace(prob), u0, p, t0) * abs(t1 - t0) >
        threshold && return true
    dual || return false
    probe = remake(prob; u0 = u0, p = p)
    ok = SciMLBase.successful_retcode(
        solve(probe, alg, args...; kw..., save_everystep = false, dense = false)
    )
    reset! === nothing || reset!()   # callbacks of the probe recorded state
    return !ok
end

# Spectral radius of the state Jacobian at (u0, p, t0), `Inf` when it is not finite.
function _stiff_at_start(f, iip::Bool, u0, p, t0)
    J = iip ? ForwardDiff.jacobian((du, u) -> f(du, u, p, t0), similar(u0), u0) :
        ForwardDiff.jacobian(u -> f(u, p, t0), u0)
    all(isfinite, J) || return Inf
    return maximum(abs, eigvals(J))
end

# Exception classes that signal a numerically-degenerate point (rather than a
# programming error): callers treat these as objective = Inf / likelihood = -Inf
# during optimizer exploration instead of rethrowing.
# `ArgumentError` is the workhorse exception of half of Julia, so classifying all of them
# as numeric degeneracy masked genuine programming errors as `-Inf` (#326). Only messages
# that actually signal a numeric failure count: Distributions' `check_args` (a scale
# driven negative by an optimizer step), LAPACK/BLAS finite checks, and Roots' bracketing
# methods, which reject a non-bracketing interval with a plain `ArgumentError`.
# `Bijectors.find_alpha` hands Roots `[wᵗy - 2|wᵗû|, wᵗy + 2|wᵗû|]`, which brackets the
# root for every finite input, so a rejection means a non-finite `wᵗy`: a degenerate flow
# proposal, e.g. a planar layer whose `w` hit exactly zero, at which `get_u_hat` returns
# `û = NaN` while `wᵗû` stays finite. That is a -Inf point for the optimizer to back off
# from, not a programming error.
# Word boundaries matter: a bare "Inf" substring also matches "Information missing".
const _NUMERIC_ARGUMENT_ERROR_RE =
    r"is not satisfied|Infs or NaNs|not a bracketing interval|\bNaN\b|\bInf\b"

@inline function _is_numeric_argument_error(err::ArgumentError)
    return occursin(_NUMERIC_ARGUMENT_ERROR_RE, string(err.msg))
end

@inline function _is_numeric_error(err)
    return err isa LinearAlgebra.PosDefException ||
        err isa LinearAlgebra.SingularException ||
        err isa DomainError ||
        (err isa ArgumentError && _is_numeric_argument_error(err)) ||
        err isa Roots.ConvergenceFailed ||
        err isa DataInterpolations.LeftExtrapolationError ||
        err isa DataInterpolations.RightExtrapolationError
end

# An exception's own message, flattened to one line. The type name alone hides what the
# model actually complained about (#313); the cap keeps a long message out of the log.
function _brief_error_message(err; maxlen::Int = 200)
    msg = replace(sprint(showerror, err), '\n' => ' ')
    return length(msg) > maxlen ? first(msg, maxlen - 3) * "..." : msg
end
