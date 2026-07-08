export crossing_time
export crossing_rootval

# ============================================================================
# Threshold-crossing (first-passage / time-to-event) support for @formulas, via
# solver-native event detection (ContinuousCallback).
#
# Declared as deterministic nodes:
#
#     tstar   = crossing_time(:state, :threshold; tmax = T)
#     rootval = crossing_rootval(:state, :threshold; tmax = T)
#
# where `:state` is an ODE state and `:threshold` is an in-scope symbol (a fixed
# effect or a preDifferentialEquation variable) giving the crossing level.
# `crossing_time` is the first time `state(t)` crosses `threshold` (detected DURING
# integration — no dense solution, one event detection per solve; `tmax`, optional,
# is returned when no crossing occurs, defaulting to the individual's final
# integration time). `crossing_rootval` is the AMICI-style rootvalue: `0` when the
# state crosses, else `state(t_end) − threshold` (the shortfall). Pairing
# `zt ~ Normal(tstar, σ)` with `rz ~ Normal(rootval, σ)` reproduces MEMOIR's
# time-to-event likelihood term-for-term and keeps a gradient for non-crossing cells.
#
# Parameter sensitivity is exact under ForwardDiff (the callback root runs on dual
# numbers). Only models that declare a crossing take the event path (see
# `_ll_solve_de`); every other model is unaffected.
# ============================================================================

struct CrossingSpec
    name::Symbol        # deterministic-node name holding the crossing value
    state::Symbol       # ODE state whose crossing is detected
    threshold::Symbol   # in-scope symbol (fixed effect or preDE var) = crossing level
    tmax::Any           # Float64 horizon, or `nothing` (use the individual's tspan end)
    kind::Symbol        # :time (crossing time) or :rootval (root value at the horizon)
end

# Markers recognized and rewritten by @formulas; never actually invoked at runtime.
function crossing_time(args...; kwargs...)
    error(
        "`crossing_time` may only be used as a deterministic node inside @formulas, " *
        "e.g. `tstar = crossing_time(:state, :threshold; tmax = T)`.")
end

function crossing_rootval(args...; kwargs...)
    error(
        "`crossing_rootval` may only be used as a deterministic node inside @formulas, " *
        "e.g. `rootval = crossing_rootval(:state, :threshold; tmax = T)`.")
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

# Plotting/diagnostics re-solve without the crossing event callback, so the crossing
# nodes @formulas rewrites to `sol_accessors.<name>` are recovered here from the solved
# trajectory instead. `_crossing_time_from_sol` mirrors `_CrossingAffect` off a completed
# `sol`; `_sol_accessors_with_crossings` merges the values into the accessors.

# First-passage time of state `idx` past level `c`; returns `tmax` if never crossed.
function _crossing_time_from_sol(sol, idx::Int, c, tmax)
    cv = ForwardDiff.value(c)
    ts = sol.t
    n = length(ts)
    n < 2 && return oftype(0.5 * (ts[1] + ts[1]) - cv, tmax)
    va = ForwardDiff.value(sol.u[1][idx]) - cv
    a = ts[1]
    b = ts[1]
    found = false
    @inbounds for k in 2:n
        vk = ForwardDiff.value(sol.u[k][idx]) - cv
        if (va < 0) != (vk < 0)                             # first sign change across saved steps
            a = ts[k - 1]
            b = ts[k]
            found = true
            break
        end
        va = vk
    end
    found || return oftype(0.5 * (ts[1] + ts[1]) - cv, tmax)
    va = ForwardDiff.value(sol(a; idxs = idx)) - cv
    @inbounds for _ in 1:40                                 # bisection on the step interpolant
        m = 0.5 * (a + b)
        vm = ForwardDiff.value(sol(m; idxs = idx)) - cv
        if (vm < 0) == (va < 0)
            a = m
            va = vm
        else
            b = m
        end
    end
    troot = 0.5 * (a + b)
    g = sol(troot; idxs = idx) - c
    gdot = sol(troot, Val{1})[idx]
    return abs(ForwardDiff.value(gdot)) > 0 ? troot - g / gdot : oftype(g, troot)
end

# AMICI-style rootvalue read off a solved trajectory: 0 if state `idx` crosses level `c`
# within the saved points, else `state(t_end) − c` (the shortfall). Carries the dual
# sensitivity of the final state; zero (with zero gradient) once the state has crossed.
function _crossing_rootval_from_sol(sol, idx::Int, c)
    us = sol.u
    n = length(us)
    rend = us[n][idx] - c                                   # shortfall at the horizon
    n < 2 && return rend
    cv = ForwardDiff.value(c)
    va = ForwardDiff.value(us[1][idx]) - cv
    @inbounds for k in 2:n
        vk = ForwardDiff.value(us[k][idx]) - cv
        (va < 0) != (vk < 0) && return zero(rend)           # crossed → rootvalue 0
        va = vk
    end
    return rend
end

# Accessors for plotting, with any crossing node values (time or rootvalue) merged in
# (no-op when the model declares no crossings).
function _sol_accessors_with_crossings(model, sol, compiled, θ, η, const_cov)
    acc = get_de_accessors_builder(model.de.de)(sol, compiled)
    crossings = get_formulas_crossings(model.formulas.formulas)
    isempty(crossings) && return acc
    pre = calculate_prede(model, θ, η, const_cov)
    state_names = get_de_states(model.de.de)
    names = ntuple(k -> crossings[k].name, length(crossings))
    vals = ntuple(length(crossings)) do k
        spec = crossings[k]
        sidx = findfirst(==(spec.state), state_names)
        sidx === nothing && error("crossing state `$(spec.state)` is not a DE state.")
        c = _crossing_threshold(spec.threshold, θ, pre)
        if spec.kind === :rootval
            _crossing_rootval_from_sol(sol, sidx, c)
        else
            tmax = spec.tmax === nothing ? sol.t[end] : spec.tmax
            _crossing_time_from_sol(sol, sidx, c, tmax)
        end
    end
    return merge(acc, NamedTuple{names}(vals))
end
