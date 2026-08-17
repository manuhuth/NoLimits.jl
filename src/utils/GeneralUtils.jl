export flatten_re_names
export flatten_re_values
export rowsoftmax

using ForwardDiff
import DiffEqBase
import Roots

# Public symbol-valued options also accept the string spelling, so the Python/R
# bindings (and serialized federated options) can pass them through unchanged.
# New public symbol options normalize through this at their entry point.
@inline _as_symbol(x::AbstractString) = Symbol(x)
@inline _as_symbol(x::Symbol) = x
@inline _as_symbol(x) = x   # nothing, NamedTuples, functions, ... pass through

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

# Exception classes that signal a numerically-degenerate point (rather than a
# programming error): callers treat these as objective = Inf / likelihood = -Inf
# during optimizer exploration instead of rethrowing.
@inline function _is_numeric_error(err)
    return err isa LinearAlgebra.PosDefException ||
        err isa LinearAlgebra.SingularException ||
        err isa DomainError || err isa ArgumentError ||
        err isa Roots.ConvergenceFailed
end
