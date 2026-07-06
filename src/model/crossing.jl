export crossing_time

# ============================================================================
# Threshold-crossing (first-passage / time-to-event) support for @formulas, via
# solver-native event detection (ContinuousCallback).
#
# Declared as a deterministic node:
#
#     tstar = crossing_time(:state, :threshold; tmax = T)
#
# where `:state` is an ODE state and `:threshold` is an in-scope symbol (a fixed
# effect or a preDifferentialEquation variable) giving the crossing level. The
# value is the first time `state(t)` crosses `threshold`, detected DURING
# integration — no dense solution, no re-solving, one event detection per solve.
# Its parameter sensitivity is exact under ForwardDiff (the callback root runs on
# dual numbers). `tmax` (optional) is returned when no crossing occurs within the
# horizon (defaults to the individual's final integration time).
#
# Only models that declare a crossing take the event path (see `_ll_solve_de`);
# every other model is completely unaffected (same sparse solve, same template
# cache).
# ============================================================================

struct CrossingSpec
    name::Symbol        # deterministic-node name holding the crossing time
    state::Symbol       # ODE state whose crossing is detected
    threshold::Symbol   # in-scope symbol (fixed effect or preDE var) = crossing level
    tmax::Any           # Float64 horizon, or `nothing` (use the individual's tspan end)
end

# Marker recognized and rewritten by @formulas; never actually invoked at runtime.
function crossing_time(args...; kwargs...)
    error(
        "`crossing_time` may only be used as a deterministic node inside @formulas, " *
        "e.g. `tstar = crossing_time(:state, :threshold; tmax = T)`.")
end

get_formulas_crossings(f) = f.ir.crossings

# Resolve the crossing level from the current parameters: a fixed effect or a
# preDifferentialEquation output (both may be dual-typed under ForwardDiff).
@inline function _crossing_threshold(sym::Symbol, θ, pre)
    if hasproperty(θ, sym)
        return getproperty(θ, sym)
    elseif pre !== nothing && hasproperty(pre, sym)
        return getproperty(pre, sym)
    else
        error("crossing_time threshold `$(sym)` was not found among the fixed effects " *
              "or preDifferentialEquation variables.")
    end
end

# Callback condition/affect as callable structs (functors) with concretely-typed
# fields — type-stable and inlinable in the integration loop.
#
# Detection uses a DiscreteCallback (checked after each accepted step), NOT a
# ContinuousCallback. A ContinuousCallback repositions the integrator onto the root
# via `change_t_via_interpolation!`, which (a) throws "interpolant only works between
# tprev and t" at step boundaries and (b) couples the result to the observation
# `saveat` grid. The DiscreteCallback never repositions: on the step where the state
# first crosses the level it locates the crossing time WITHIN that step from the
# step's own interpolant (always live during integration — no dense solution, sparse
# `saveat` preserved) and records it. Result is grid-independent and, under
# ForwardDiff, exact: after bisecting on the primal to the root `troot`, one
# implicit-function-theorem Newton step `t* = troot - g/ġ` (g = state−level,
# ġ = d(state)/dt) carries the exact dual sensitivity dt*/dθ = −(∂g/∂θ)/(∂g/∂t).
struct _CrossingCondition{C}
    idx::Int                       # ODE state index
    c::C                           # crossing level (Real or ForwardDiff.Dual)
    fired::Base.RefValue{Bool}     # record only the first passage
end
@inline function (cc::_CrossingCondition)(u, t, integrator)
    cc.fired[] && return false
    cv = ForwardDiff.value(cc.c)
    gp = ForwardDiff.value(integrator.uprev[cc.idx]) - cv   # residual at step start
    gn = ForwardDiff.value(u[cc.idx]) - cv                  # residual at step end
    return (gp < 0) != (gn < 0)                             # sign change across the step
end

struct _CrossingAffect{C, R}
    idx::Int
    c::C
    ref::R                         # Base.RefValue{T} receiving the crossing time
    fired::Base.RefValue{Bool}
end
@inline function (ca::_CrossingAffect)(integrator)
    idx = ca.idx
    c = ca.c
    cv = ForwardDiff.value(c)
    a = integrator.tprev
    b = integrator.t
    va = ForwardDiff.value(integrator(a; idxs = idx)) - cv
    @inbounds for _ in 1:40                                 # bisection on the step interpolant
        m = 0.5 * (a + b)
        vm = ForwardDiff.value(integrator(m; idxs = idx)) - cv
        if (vm < 0) == (va < 0)
            a = m
            va = vm
        else
            b = m
        end
    end
    troot = 0.5 * (a + b)
    g = integrator(troot; idxs = idx) - c                   # dual residual at the primal root
    gdot = integrator(troot, Val{1})[idx]                   # time-derivative of the state
    ca.ref[] = abs(ForwardDiff.value(gdot)) > 0 ? troot - g / gdot : oftype(g, troot)
    ca.fired[] = true
    return nothing
end
