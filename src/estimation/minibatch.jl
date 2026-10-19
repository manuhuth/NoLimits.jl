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
            error("MiniBatch schedule: minibatch size must be ≥ 1. Got: $update_schedule")
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
        error("Invalid MiniBatch schedule. Use :all, Int minibatch size, or a callable (nbatches, iter, rng) -> indices (any iterable of Int).")
    end
end

"""
    _check_update_schedule(update_schedule, label)

Constructor-time validation of a `MiniBatch` schedule.
"""
function _check_update_schedule(update_schedule, label::AbstractString)
    if update_schedule === :all
        return update_schedule
    elseif update_schedule isa Int
        update_schedule >= 1 ||
            error("$label: schedule minibatch size must be ≥ 1. Got: $update_schedule")
        return update_schedule
    elseif hasmethod(update_schedule, Tuple{Int, Int, AbstractRNG})
        return update_schedule
    end
    return error(
        "$label: invalid schedule. Use :all, an Int minibatch size, or a " *
            "callable (nbatches::Int, iter::Int, rng) -> indices (any iterable of Int)."
    )
end

# Internal field set only by `MiniBatch`; custom dev-API `FittingMethod`s need none.
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
        error("MiniBatch schedule returned an empty batch selection at iteration $(st.iter).")
    for i in sel
        (1 <= i <= st.nbatches) || error(
            "MiniBatch schedule returned batch index $i at iteration $(st.iter), " *
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

# Memory m = 20 (Optim's default is 10): the outer problems have few parameters but each
# objective evaluation solves every inner EB problem, so better curvature (fewer outer
# iterations) pays; in the autoresearch challenger benchmark m = 20 was 10-17% faster on
# 5 of 9 Laplace models and within noise on the rest.
_default_lbfgs() = OptimizationOptimJL.LBFGS(
    m = 20, linesearch = LineSearches.BackTracking(maxstep = 1.0)
)

_resolve_outer_optimizer(optimizer) = optimizer === nothing ? _default_lbfgs() : optimizer

# ── MiniBatch wrapper ────────────────────────────────────────────────────────────

export MiniBatch

"""
    MiniBatch(method; schedule, optimizer = nothing, optim_kwargs = nothing,
              em_variant = :incremental) <: FittingMethod

Mini-batch wrapper around an estimation method: each iteration uses a random subset of the
random-effect batches (individuals linked by random effects; for `MLE`/`MAP` every
individual is one batch). Fit with `fit_model(dm, MiniBatch(method; schedule = ...))`.
Supported methods: `MLE`, `MAP`, `Laplace`, `FOCEI`, `GHQuadrature`, `VEM`, `SAEM`, `MCEM`.

# Keyword Arguments
- `schedule`: which batches are used per iteration.
  - `Int`: a random subset of that size, drawn without replacement each iteration
    (`min(m, nbatches)` batches).
  - Any callable `(nbatches::Int, iter::Int, rng) -> indices` returning an iterable of
    `Int` (a `Vector{Int}`, a range, ...): the batches of iteration `iter`. Duplicates are
    ignored. A stateful callable is shared by every fit that uses this wrapper (including
    all `Multistart` starts), so pass a fresh instance per fit.
  - `:all`: every batch every iteration; the wrapped method then behaves exactly like the
    unwrapped one.
- `optimizer = nothing`: outer optimizer of `MLE`, `MAP`, `Laplace`, `FOCEI`,
  `GHQuadrature` and `VEM` while mini-batching. `nothing` keeps the method's optimizer,
  except that the method's default optimizer becomes `Optimisers.Adam(0.01)` when
  `schedule !== :all`. A non-Optimisers.jl optimizer warns (the objective is stochastic);
  it must support Optimization.jl callbacks.
- `optim_kwargs = nothing`: `solve` keywords for that optimizer. `nothing` keeps the
  method's `optim_kwargs`, or uses `(; maxiters = 1000)` when the default Adam replaces
  the default optimizer and the method's `optim_kwargs` are its defaults. With an
  Optimisers.jl rule, `maxiters = 1000` and `save_best = false` are added when not given.
- `em_variant = :incremental`: mini-batch form of `SAEM`/`MCEM` (ignored by the other
  methods). Both draw new random effects only for the selected batches.
  - `:incremental` (incremental EM, Neal & Hinton 1998): the other batches keep their draws
    from their last visit, and the M-step uses all batches.
  - `:scaled`: the M-step uses only the selected batches, with their likelihood weights
    multiplied by `nbatches / length(selected)` (an unbiased estimate of the full-data
    Q-function). For `SAEM` it requires `mstep_sa_on_params = true` and no custom
    `suffstats`. The reported objective is then the scaled Q of the last minibatch.

For the gradient-based methods and `VEM` one subset is drawn per optimizer iteration, the
objective and gradient of that iteration share it, and the selected batches are scaled by
`nbatches / length(selected)` so the objective is an unbiased estimate of the full-data
objective; priors, `penalty` and `extra_objective` are not scaled. The reported objective
is re-evaluated on all batches at the fitted parameters. With Optimisers.jl rules finite
bounds are enforced by projecting each update onto the box. For `VEM` the variational
parameters of unselected batches get a zero gradient (the unbiased stochastic gradient of
the full ELBO); `refine = :auto` skips the full-data refinement, and with an Optimisers.jl
rule the fixed effects are not curvature-preconditioned.

All subsets are drawn from the fit `rng` (`Multistart` gives each start its own forked
`rng`), so seeded fits are reproducible and threaded and serial runs agree. The result is
the wrapped method's `FitResult`: `get_method` returns the wrapped method with its
internal schedule set.

```julia
fit_model(dm, MiniBatch(Laplace(); schedule = 8); rng = Xoshiro(1))
fit_model(dm, MiniBatch(SAEM(); schedule = 20, em_variant = :scaled))
```
"""
struct MiniBatch{M <: FittingMethod, S, O, K} <: FittingMethod
    method::M
    schedule::S
    optimizer::O
    optim_kwargs::K
    em_variant::Symbol
end

const _MB_OUTER_METHODS = Union{MLE, MAP, Laplace, FOCEI, GHQuadrature, VEM}
const _MB_EM_METHODS = Union{SAEM, MCEM}

function MiniBatch(
        method::FittingMethod; schedule, optimizer = nothing, optim_kwargs = nothing,
        em_variant = :incremental
    )
    method isa MiniBatch && error("MiniBatch: the method is already wrapped in MiniBatch.")
    method isa Union{_MB_OUTER_METHODS, _MB_EM_METHODS} || error(
        "MiniBatch supports MLE, MAP, Laplace, FOCEI, GHQuadrature, VEM, SAEM and MCEM; " *
            "got $(nameof(typeof(method)))."
    )
    schedule = _check_update_schedule(_as_symbol(schedule), "MiniBatch")
    em_variant = _as_symbol(em_variant)
    em_variant in (:incremental, :scaled) ||
        error("MiniBatch: em_variant must be :incremental or :scaled. Got: $em_variant")
    if method isa _MB_EM_METHODS
        (optimizer === nothing && optim_kwargs === nothing) || error(
            "MiniBatch: `optimizer` and `optim_kwargs` set the outer optimizer of " *
                "gradient-based methods; $(nameof(typeof(method))) keeps its M-step optimizer."
        )
        if em_variant === :scaled && method isa SAEM
            method.saem.mstep_sa_on_params || error(
                "MiniBatch: em_variant = :scaled requires SAEM(mstep_sa_on_params = true)."
            )
            method.saem.suffstats === nothing ||
                error("MiniBatch: em_variant = :scaled does not support custom SAEM `suffstats`.")
        end
        return MiniBatch(method, schedule, nothing, nothing, em_variant)
    end
    _check_warm_starts(_warm_starts(method), schedule, "MiniBatch")
    optimizer, optim_kwargs = _minibatch_optimizer(method, schedule, optimizer, optim_kwargs)
    return MiniBatch(method, schedule, optimizer, optim_kwargs, em_variant)
end

function _minibatch_optimizer(method, schedule, optimizer, optim_kwargs)
    if optimizer === nothing
        default = typeof(method).name.wrapper()
        if schedule !== :all && method.optimizer === default.optimizer
            optimizer = Optimisers.Adam(0.01)
            optim_kwargs === nothing && method.optim_kwargs === default.optim_kwargs &&
                (optim_kwargs = (; maxiters = 1000))
        else
            optimizer = method.optimizer
        end
    end
    optim_kwargs === nothing && (optim_kwargs = method.optim_kwargs)
    if schedule !== :all
        SciMLBase.allowscallback(optimizer) || error(
            "MiniBatch needs an optimizer that supports Optimization.jl callbacks; " *
                "$(nameof(typeof(optimizer))) does not."
        )
        optimizer isa Optimisers.AbstractRule ||
            @warn "MiniBatch makes the objective stochastic (a fresh minibatch per " *
            "optimizer iteration). Deterministic quasi-Newton / line-search optimizers are " *
            "generally inappropriate here; an Optimisers.jl rule such as " *
            "`Optimisers.Adam(0.01)` is recommended. The supplied optimizer " *
            "$(nameof(typeof(optimizer))) will be used anyway."
    end
    return optimizer, _as_namedtuple(optim_kwargs)
end

_base_method(m) = m
_base_method(m::MiniBatch) = m.method

# Copy of an immutable struct with some fields replaced (positional default constructor).
function _with_fields(x::T, nt::NamedTuple) where {T}
    return T.name.wrapper(map(f -> haskey(nt, f) ? nt[f] : getfield(x, f), fieldnames(T))...)
end

# The wrapped method with its internal schedule (and outer optimizer) set. Each method reads
# the schedule through `_update_schedule` and evaluates only the selected batches.
function _with_schedule(m::SAEM, s)
    return _with_fields(m, (; saem = _with_fields(m.saem, (; update_schedule = s))))
end
_with_schedule(m::MCEM, s) = _with_fields(m, (; update_schedule = s))
function _minibatch_method(mb::MiniBatch)
    m = mb.method
    m isa _MB_EM_METHODS && return _with_schedule(m, mb.schedule)
    return _with_fields(
        m, (; update_schedule = mb.schedule, optimizer = mb.optimizer, optim_kwargs = mb.optim_kwargs)
    )
end

function _fit_model(dm::DataModel, mb::MiniBatch, args...; kwargs...)
    m = _minibatch_method(mb)
    if mb.method isa _MB_EM_METHODS && mb.em_variant === :scaled && mb.schedule !== :all
        return _fit_em_scaled(dm, mb, m, args...; kwargs...)
    end
    return _fit_model(dm, m, args...; kwargs...)
end

# Scaled-subset EM: the schedule callable writes the likelihood weights of a private
# DataModel copy (nbatches / |B| times the base weight in the selected batches, 0
# elsewhere), so the weighted Q-function and closed-form statistics of the M-step are the
# scaled subset estimate. Zero-weight batches are skipped in the Q evaluations.
struct _ScaledEMSchedule{S}
    schedule::S
    inds::Vector{Vector{Int}}
    base::Vector{Float64}
    w::Vector{Float64}
    buf::Vector{Int}
end

function (s::_ScaledEMSchedule)(nb::Int, iter::Int, rng::AbstractRNG)
    nb == length(s.inds) ||
        error("MiniBatch: internal batch count mismatch ($nb vs $(length(s.inds))).")
    sel = sort!(unique!(collect(Int, _schedule_batches!(s.buf, s.schedule, nb, iter, rng))))
    isempty(sel) &&
        error("MiniBatch schedule returned an empty batch selection at iteration $iter.")
    all(i -> 1 <= i <= nb, sel) ||
        error("MiniBatch schedule returned a batch index outside 1:$nb at iteration $iter.")
    fill!(s.w, 0.0)
    c = nb / length(sel)
    for bi in sel, i in s.inds[bi]
        s.w[i] = c * s.base[i]
    end
    return sel
end

# Full refresh (MCEM's first iteration and E-step mode switch): base weights, all batches.
struct _ScaledEMAll{S}
    s::S
end
function (a::_ScaledEMAll)(nb::Int, ::Int, ::AbstractRNG)
    a.s.w .= a.s.base
    return collect(1:nb)
end
_full_schedule(_) = :all
_full_schedule(s::_ScaledEMSchedule) = _ScaledEMAll(s)

function _fit_em_scaled(
        dm::DataModel, mb::MiniBatch, m, args...; store_data_model::Bool = true, kwargs...
    )
    fixed_maps = _normalize_constants_re(dm, get(kwargs, :constants_re, NamedTuple()))
    _, batch_infos, _ = _build_re_batch_infos(dm, fixed_maps)
    base = get_weights(dm) === nothing ? ones(length(get_individuals(dm))) :
        copy(get_weights(dm))
    w = copy(base)
    sched = _ScaledEMSchedule(
        mb.schedule, [collect(Int, get_inds(info)) for info in batch_infos], base, w, Int[]
    )
    res = _fit_model(
        _with_fields(dm, (; weights = w)), _with_schedule(mb.method, sched), args...;
        store_data_model = false, kwargs...
    )
    # Report the user's DataModel and schedule, not the private weighted copy.
    return FitResult(
        m, get_result(res), get_summary(res), get_diagnostics(res),
        store_data_model ? dm : nothing, get_fit_args(res),
        merge(get_fit_kwargs(res), (; store_data_model = store_data_model))
    )
end
