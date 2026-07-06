export build_ode_params
export flatten_re_names
export flatten_re_values
export rowsoftmax

using ForwardDiff
import DiffEqBase

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

function build_ode_params(de, θ;
        random_effects = ComponentArray(NamedTuple()),
        constant_covariates = NamedTuple(),
        varying_covariates = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple(),
        prede_builder = (fe, re, consts, model_funs, helpers) -> NamedTuple(),
        inverse_transform = identity)
    return build_de_params(de, θ;
        random_effects = random_effects,
        constant_covariates = constant_covariates,
        varying_covariates = varying_covariates,
        helpers = helpers,
        model_funs = model_funs,
        prede_builder = prede_builder,
        inverse_transform = inverse_transform)
end

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
    kw, (verbose = _ode_verbose(v),))
@inline _ode_normalize_verbose(kw::NamedTuple, v) = kw

@inline function _ode_solve_kwargs(base::NamedTuple,
        extra::NamedTuple = NamedTuple(),
        overrides::NamedTuple = NamedTuple())
    merged = merge((verbose = _ODE_VERBOSE_SILENT, maxiters = 5000), base, extra, overrides)
    return _ode_normalize_verbose(merged, merged.verbose)
end
