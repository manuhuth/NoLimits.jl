using Random
using LineSearches
using OptimizationOptimJL
import Optimisers
# Loaded so `Optimization.solve` can dispatch on `Optimisers.AbstractRule` optimizers.
import OptimizationOptimisers

# Shared by SAEM, MCEM and the outer-loop mini-batching so the estimators' options
# cannot drift.
function _schedule_batches!(
        buf::Vector{Int}, update_schedule, nbatches::Int, iter::Int, rng::AbstractRNG
    )
    if update_schedule === :all
        resize!(buf, nbatches)
        @inbounds for i in 1:nbatches
            buf[i] = i
        end
        return buf
    elseif update_schedule isa Int
        update_schedule >= 1 ||
            error("update_schedule: minibatch size must be ≥ 1. Got: $update_schedule")
        m = min(update_schedule, nbatches)
        resize!(buf, nbatches)
        @inbounds for i in 1:nbatches
            buf[i] = i
        end
        Random.randperm!(rng, buf)
        resize!(buf, m)
        return buf
    elseif hasmethod(update_schedule, Tuple{Int, Int, AbstractRNG})
        return update_schedule(nbatches, iter, rng)
    else
        error("Invalid update_schedule. Use :all, Int minibatch size, or a callable (nbatches, iter, rng) -> indices (any iterable of Int).")
    end
end

"""
    _check_update_schedule(update_schedule, label)

Constructor-time validation of an `update_schedule` option.
"""
function _check_update_schedule(update_schedule, label::AbstractString)
    if update_schedule === :all
        return update_schedule
    elseif update_schedule isa Int
        update_schedule >= 1 ||
            error("$label: update_schedule minibatch size must be ≥ 1. Got: $update_schedule")
        return update_schedule
    elseif hasmethod(update_schedule, Tuple{Int, Int, AbstractRNG})
        return update_schedule
    end
    return error(
        "$label: invalid update_schedule. Use :all, an Int minibatch size, or a " *
            "callable (nbatches::Int, iter::Int, rng) -> indices (any iterable of Int)."
    )
end

# Custom `FittingMethod`s built on the dev-API drivers need no `update_schedule` field.
_update_schedule(::FittingMethod) = :all

# Per-fit mini-batch state. The selection is drawn lazily at the first objective
# evaluation of an iteration, so the schedule callable runs exactly once per optimizer
# iteration that actually evaluates.
mutable struct _MinibatchState{S, R <: AbstractRNG}
    const schedule::S
    const rng::R
    const nbatches::Int
    iter::Int
    const buf::Vector{Int}
    selected::Vector{Int}
    const active::Set{Int}
    scale::Float64
    pending::Bool
    last_cb_iter::Int
end

function _minibatch_state(update_schedule, nbatches::Int, rng::AbstractRNG)
    update_schedule === :all && return nothing
    nbatches >= 1 ||
        error("Mini-batching requires at least one batch. Got nbatches = $nbatches.")
    return _MinibatchState(
        update_schedule, rng, nbatches, 1, Int[], Int[], Set{Int}(), 1.0, true, 0
    )
end

function _minibatch_draw!(st::_MinibatchState)
    sel = _schedule_batches!(st.buf, st.schedule, st.nbatches, st.iter, st.rng)
    isempty(sel) &&
        error("update_schedule returned an empty batch selection at iteration $(st.iter).")
    for i in sel
        (1 <= i <= st.nbatches) || error(
            "update_schedule returned batch index $i at iteration $(st.iter), " *
                "outside the valid range 1:$(st.nbatches)."
        )
    end
    # `collect` copies: the callable may return an alias of `buf`, or an immutable range.
    st.selected = sort!(unique!(collect(Int, sel)))
    empty!(st.active)
    for i in st.selected
        push!(st.active, i)
    end
    st.scale = st.nbatches / length(st.selected)
    st.pending = false
    return st
end

_minibatch_current!(::Nothing) = nothing
function _minibatch_current!(st::_MinibatchState)
    st.pending && _minibatch_draw!(st)
    return st
end

function _minibatch_advance!(st::_MinibatchState)
    st.iter += 1
    st.pending = true
    return nothing
end

_minibatch_active(::Nothing) = nothing
_minibatch_active(st::_MinibatchState) = st.active
_minibatch_scale(::Nothing) = 1.0
_minibatch_scale(st::_MinibatchState) = st.scale

struct _MinibatchCallback{S, F, C}
    st::S
    invalidate!::F
    user_cb::C
end

function (cb::_MinibatchCallback)(state, args...)
    st = cb.st
    # OptimizationOptimisers can re-fire a callback for the same iteration; de-duplicate.
    # Solvers that never set `iter` leave it at 0, so always advance in that case.
    if state.iter != st.last_cb_iter || state.iter == 0
        st.last_cb_iter = state.iter
        _minibatch_advance!(st)
        cb.invalidate!()
    end
    return cb.user_cb === nothing ? false : cb.user_cb(state, args...)
end

function _minibatch_solve_kwargs(optim_kwargs::NamedTuple, optimizer, st, invalidate!::F) where {F}
    st === nothing && return optim_kwargs
    cb = _MinibatchCallback(st, invalidate!, get(optim_kwargs, :callback, nothing))
    kw = merge(optim_kwargs, (; callback = cb))
    if optimizer isa Optimisers.AbstractRule
        # OptimizationOptimisers errors without an iteration budget, and `save_best`
        # would revert to the best single-minibatch objective.
        haskey(kw, :maxiters) || haskey(kw, :epochs) || (kw = merge((; maxiters = 1000), kw))
        haskey(kw, :save_best) || (kw = merge(kw, (; save_best = false)))
    end
    return kw
end

# Box constraints for Optimisers.jl rules: the wrapped rule's step is projected onto
# [lb, ub], i.e. x_new = clamp(x - dx′, lb, ub). Optimization.jl rejects lb/ub for
# these rules, so the bounds travel here instead of in the OptimizationProblem.
struct _ProjectedRule{R <: Optimisers.AbstractRule, V <: AbstractVector} <: Optimisers.AbstractRule
    rule::R
    lb::V
    ub::V
end

Optimisers.init(o::_ProjectedRule, x::AbstractArray) = Optimisers.init(o.rule, x)

function Optimisers.apply!(o::_ProjectedRule, state, x, dx)
    state′, dx′ = Optimisers.apply!(o.rule, state, x, dx)
    return state′, x .- clamp.(x .- dx′, o.lb, o.ub)
end

# Returns the optimizer to hand to `solve` and whether lb/ub go into the problem.
function _bounded_optimizer(optimizer, use_bounds::Bool, lb_z, ub_z)
    (use_bounds && optimizer isa Optimisers.AbstractRule) || return optimizer, use_bounds
    return _ProjectedRule(optimizer, lb_z, ub_z), false
end

_default_lbfgs() = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0))

function _resolve_outer_optimizer(optimizer, update_schedule, label::AbstractString)
    if optimizer === nothing
        return update_schedule === :all ? _default_lbfgs() : Optimisers.Adam(0.01)
    end
    if update_schedule !== :all
        SciMLBase.allowscallback(optimizer) || error(
            "$label: update_schedule != :all needs an optimizer that supports " *
                "Optimization.jl callbacks; $(nameof(typeof(optimizer))) does not."
        )
        if !(optimizer isa Optimisers.AbstractRule)
            @warn "$label: mini-batching makes the objective stochastic (a fresh minibatch " *
                "per optimizer iteration). Deterministic quasi-Newton / line-search optimizers " *
                "are generally inappropriate here; an Optimisers.jl rule such as " *
                "`Optimisers.Adam()` is recommended (a learning rate can be passed, e.g. " *
                "`Optimisers.Adam(0.01)`). The supplied optimizer " *
                "$(nameof(typeof(optimizer))) will be used anyway."
        end
    end
    return optimizer
end

const _UPDATE_SCHEDULE_DOC = """
- `update_schedule = :all`: which batches enter the outer objective and gradient per
  optimizer iteration (mini-batching). Options:
  - `:all` (default) — use all batches every iteration (deterministic objective).
  - `Int` — random minibatch of that size, sampled without replacement each iteration
    (`min(m, nbatches)` batches are used).
  - Any callable with signature `(nbatches::Int, iter::Int, rng) -> indices` returning any
    iterable of `Int` (a `Vector{Int}`, a range, ...) — the indices of the batches to use in optimizer iteration `iter`. Can be a plain function
    or a callable struct (useful for stateful schedules such as cycling windows).
    Duplicate indices returned by a callable are ignored. A stateful callable is shared by
    every fit that uses the same method object (including all `Multistart` starts), so pass
    a fresh instance per fit.
  The selected batches' contribution is scaled by `nbatches / length(selected)` so the
  objective stays an unbiased estimate of the full-data objective; priors, `penalty`, and
  `extra_objective` are never scaled. One minibatch is drawn per optimizer iteration from
  the fit `rng`; the objective and gradient of that iteration share it. When mini-batching
  is active and no `optimizer` is given, the default becomes `Optimisers.Adam(0.01)` (with
  `maxiters = 1000` and `save_best = false` unless given in `optim_kwargs`) instead of
  LBFGS; passing a non-Optimisers optimizer warns. After the fit, the reported objective
  is re-evaluated once on all batches at the fitted parameters. With Optimisers.jl rules,
  finite bounds (model bounds or `lb`/`ub`) are enforced by projecting each update onto
  the box.
"""
