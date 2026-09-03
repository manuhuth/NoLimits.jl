export MLE
export FrequentistResult
export default_bounds_from_start

using Optimization
using OptimizationOptimJL
using SciMLBase
using ComponentArrays
using Random
using LineSearches

"""
    MLE(; optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds,
    precondition, update_schedule) <: FittingMethod

Maximum Likelihood Estimation for models without random effects.

# Keyword Arguments
- `optimizer`: Optimization.jl-compatible optimizer. Defaults to `LBFGS` with backtracking
  line search, or to `Optimisers.Adam(0.01)` when `update_schedule != :all`.
- `optim_kwargs::NamedTuple = NamedTuple()`: keyword arguments forwarded to `Optimization.solve`
  (e.g. `maxiters`, `reltol`).
- `adtype`: automatic-differentiation backend. Defaults to `AutoForwardDiff()`.
- `lb`: lower bounds on the transformed parameter scale, or `nothing` to use the
  model-declared bounds.
- `ub`: upper bounds on the transformed parameter scale, or `nothing`.
- `ignore_model_bounds::Bool = false`: if `true`, ignore the bounds declared in
  `@fixedEffects` (explicit `lb`/`ub` still apply).
- `precondition::Bool = true`: optimize the scaled offset `z` with
  `θ_transformed = θ0 + s .* z`, so every fit starts at `z = 0` and no coordinate can be
  frozen by an unlucky starting value. `s` is 1 for any coordinate already in log/logit
  space and `max(abs(θ0), 1)` for a genuinely natural-scale `:identity` coordinate. Set
  `false` to optimize the transformed vector directly, which reproduces pre-0.2 results
  bit-for-bit. Note that with preconditioning on, the optimizer object behind
  [`get_raw`](@ref) works in `z`; [`get_params`](@ref) always returns the usual scales.
$(_UPDATE_SCHEDULE_DOC)  A batch here is a single individual, so `update_schedule = 5`
  uses 5 randomly chosen individuals per optimizer iteration.
"""
struct MLE{O, K, A, L, U, US} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
    ignore_model_bounds::Bool
    precondition::Bool
    update_schedule::US
end

_update_schedule(m::MLE) = m.update_schedule

function MLE(;
        optimizer = nothing,
        optim_kwargs = NamedTuple(),
        adtype = Optimization.AutoForwardDiff(),
        lb = nothing,
        ub = nothing,
        ignore_model_bounds = false,
        precondition = true,
        update_schedule = :all
    )
    update_schedule = _check_update_schedule(_as_symbol(update_schedule), "MLE")
    optimizer = _resolve_outer_optimizer(optimizer, update_schedule, "MLE")
    return MLE(
        optimizer, _as_namedtuple(optim_kwargs), adtype, lb, ub,
        ignore_model_bounds, precondition, update_schedule
    )
end

# FrequentistResult is a StandardOptimizationResult{:frequentist} alias + constructor (see common.jl).

struct _NoOpTerm end
@inline (_::_NoOpTerm)(θ) = 0.0

# Combine a method's internal objective term (e.g. the MAP log-prior, or the
# MLE no-op) with an optional user-supplied `extra_objective(θu)` term. Both are
# functions of the natural-scale parameters θu and are summed into the objective.
# When no extra term is supplied the base term is returned unchanged, so the
# default path is a genuine no-op (identical results to before).
struct _SumTerms{A, B}
    a::A
    b::B
end
@inline (s::_SumTerms)(θ) = s.a(θ) + s.b(θ)
_combine_add_terms(base, ::Nothing) = base
_combine_add_terms(base, extra) = _SumTerms(base, extra)

# Precondition shared by every fixed-effects-only route (MLE/MAP fits and their primitives).
function _require_no_random_effects(dm::DataModel)
    isempty(get_re_names(get_random(get_model(dm)))) ||
        error("This method is only valid for models without random effects. Use Laplace, SAEM, or MCMC for random-effects models.")
    return nothing
end

function _fit_no_re(
        dm::DataModel, method;
        constants::NamedTuple,
        penalty::NamedTuple,
        ode_args::Tuple,
        ode_kwargs::NamedTuple,
        serialization::SciMLBase.EnsembleAlgorithm,
        add_term,
        rng::AbstractRNG = Random.default_rng(),
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_data_model::Bool = true,
        fit_args::Tuple = (),
        fit_kwargs::NamedTuple = NamedTuple()
    )
    _require_no_random_effects(dm)

    fe = get_fixed(get_model(dm))
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("This method requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    _validate_constant_names(fixed_set, constants)
    all(name in keys(constants) for name in fixed_names) &&
        error("This method requires at least one free fixed effect. Remove constants or specify a fixed effect or random effect.")
    layout = free_parameter_layout(
        fe; constants = constants,
        theta0_untransformed = theta_0_untransformed
    )
    _check_add_term_at_start(add_term, layout.inv_transform(layout.θ_const_t))
    free_names = layout.free_names
    inv_transform = layout.inv_transform
    θ_const_t_vec = layout.θ_const_t_vec
    free_idx = layout.free_idx
    axs_full = layout.axs_full
    θ0_free_t = layout.θ0_free_t
    cache = build_ll_cache(
        dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization, force_saveat = true
    )
    # The optimizer works on the preconditioned offset z. No explicit gradient is supplied here,
    # so `adtype` differentiates through the affine map and applies the chain rule itself.
    θ0_pc, s_pc, _θt_from_z, _z_from_θt = _precondition_maps(
        get_model(dm), free_names, θ0_free_t, layout.axs, _precondition_on(method)
    )
    mb = _minibatch_state(_update_schedule(method), length(get_individuals(dm)), rng)
    # Cleared before the final full-data re-evaluation of the objective.
    mb_ref = Ref{Union{Nothing, typeof(mb)}}(mb)
    function obj(z, p)
        v_free = ComponentArrays.getdata(_θt_from_z(z))
        T = eltype(v_free)
        infT = convert(T, Inf)
        θt_full = _merge_free_into_full(θ_const_t_vec, free_idx, v_free, axs_full)
        θu = inv_transform(θt_full)
        add = add_term(θu)
        isinf(add) && return infT
        st = _minibatch_current!(mb_ref[])
        ll = if st === nothing
            loglikelihood(
                dm, θu, ComponentArray(); cache = cache, serialization = serialization
            )
        else
            st.scale * _loglikelihood_indices(
                dm, θu, ComponentArray(), st.selected;
                cache = cache, serialization = serialization
            )
        end
        ll == -Inf && return infT
        return -ll + _penalty_value(θu, penalty) + add
    end

    optf = OptimizationFunction(obj, method.adtype)
    lb, ub, use_bounds, θ0_init = _resolve_optim_bounds(
        fe, free_names, θ0_free_t, method.optimizer, method.lb, method.ub, constants;
        ignore_model_bounds = method.ignore_model_bounds, method_label = "MLE"
    )
    z0 = _z_from_θt(θ0_init)
    lb_z = _z_from_θt(lb)
    ub_z = _z_from_θt(ub)
    # Optimisers.jl rules take no box, so bounds become a projection inside the rule.
    opt_use, use_bounds = _bounded_optimizer(method.optimizer, use_bounds, lb_z, ub_z)
    prob = use_bounds ? OptimizationProblem(optf, z0; lb = lb_z, ub = ub_z) :
        OptimizationProblem(optf, z0)
    kw = _minibatch_solve_kwargs(method.optim_kwargs, opt_use, mb, Returns(nothing))
    sol = Optimization.solve(prob, opt_use; kw...)

    fitted = resolve_fitted_parameters(layout, _θt_from_z(sol.u))
    # The optimizer's objective is a minibatch estimate; report the full-data value.
    final_obj = sol.objective
    if mb !== nothing
        mb_ref[] = nothing
        final_obj = obj(sol.u, nothing)
    end
    isfinite(final_obj) ||
        _warn_nonfinite_fit(dm, fitted.untransformed, string(nameof(typeof(method))))
    summary = FitSummary(
        final_obj, sol.retcode == SciMLBase.ReturnCode.Success,
        fitted, NamedTuple()
    )
    diagnostics = FitDiagnostics(
        (;), (optimizer = method.optimizer,), (retcode = sol.retcode,), NamedTuple()
    )
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ?
        sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = FrequentistResult(sol, final_obj, niter, raw, NamedTuple())
    return FitResult(
        method, result, summary, diagnostics,
        store_data_model ? dm : nothing, fit_args, fit_kwargs
    )
end

"""
    default_bounds_from_start(dm::DataModel; margin=1.0) -> (lower, upper)

Generate symmetric box bounds on the transformed parameter scale centered at the
initial parameter values, with half-width `margin`.

Useful for passing to `MLE(lb=lower, ub=upper)` when the model-declared bounds are
too wide or absent.

# Keyword Arguments
- `margin::Real = 1.0`: half-width of the symmetric box on the transformed scale.
"""
function default_bounds_from_start(dm::DataModel; margin::Real = 1.0)
    (isfinite(margin) && margin >= 0) ||
        error("default_bounds_from_start: margin must be finite and non-negative; got $(margin).")
    θ = get_θ0_transformed(get_fixed(get_model(dm)))
    lower = deepcopy(θ)
    upper = deepcopy(θ)
    lower .= θ .- margin
    upper .= θ .+ margin
    return (lower, upper)
end

function apply_constants!(θ, constants::NamedTuple)
    for name in keys(constants)
        val = getfield(constants, name)
        setproperty!(θ, name, val)
    end
    return θ
end
const _apply_constants! = apply_constants!

function penalty_value(θ, penalty::NamedTuple)
    isempty(keys(penalty)) && return 0.0
    acc = 0.0
    for name in keys(penalty)
        w = getfield(penalty, name)
        v = getproperty(θ, name)
        if v isa Number
            acc += w * v * v
        else
            acc += sum(w .* (v .* v))
        end
    end
    return acc
end
const _penalty_value = penalty_value

function _fit_model(
        dm::DataModel, method::MLE, args...;
        constants::NamedTuple = NamedTuple(),
        penalty::NamedTuple = NamedTuple(),
        extra_objective = nothing,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Random.default_rng(),
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_data_model::Bool = true
    )
    fit_kwargs = (
        constants = constants,
        penalty = penalty,
        ode_args = ode_args,
        ode_kwargs = ode_kwargs,
        serialization = serialization,
        rng = rng,
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model,
    )
    return _fit_no_re(
        dm, method;
        constants = constants,
        penalty = penalty,
        ode_args = ode_args,
        ode_kwargs = ode_kwargs,
        serialization = serialization,
        add_term = _combine_add_terms(_NoOpTerm(), extra_objective),
        rng = rng,
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model,
        fit_args = args,
        fit_kwargs = fit_kwargs
    )
end
