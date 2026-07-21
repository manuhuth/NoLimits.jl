export MLE
export FrequentistResult
export default_bounds_from_start

using Optimization
using OptimizationOptimJL
using SciMLBase
using ComponentArrays
using Random
using LineSearches
using OptimizationBBO

"""
    MLE(; optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds) <: FittingMethod

Maximum Likelihood Estimation for models without random effects.

# Keyword Arguments
- `optimizer`: Optimization.jl-compatible optimizer. Defaults to `LBFGS` with backtracking
  line search.
- `optim_kwargs::NamedTuple = NamedTuple()`: keyword arguments forwarded to `Optimization.solve`
  (e.g. `maxiters`, `reltol`).
- `adtype`: automatic-differentiation backend. Defaults to `AutoForwardDiff()`.
- `lb`: lower bounds on the transformed parameter scale, or `nothing` to use the
  model-declared bounds.
- `ub`: upper bounds on the transformed parameter scale, or `nothing`.
- `ignore_model_bounds::Bool = false`: if `true`, ignore the bounds declared in
  `@fixedEffects` (explicit `lb`/`ub` still apply).
"""
struct MLE{O, K, A, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
    ignore_model_bounds::Bool
end

function MLE(;
        optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()),
        optim_kwargs = NamedTuple(),
        adtype = Optimization.AutoForwardDiff(),
        lb = nothing,
        ub = nothing,
        ignore_model_bounds = false)
    MLE(optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds)
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

function _fit_no_re(dm::DataModel, method;
        constants::NamedTuple,
        penalty::NamedTuple,
        ode_args::Tuple,
        ode_kwargs::NamedTuple,
        serialization::SciMLBase.EnsembleAlgorithm,
        add_term,
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_data_model::Bool = true,
        fit_args::Tuple = (),
        fit_kwargs::NamedTuple = NamedTuple())
    re_names = get_re_names(get_random(get_model(dm)))
    isempty(re_names) ||
        error("This method is only valid for models without random effects. Use Laplace, SAEM, or MCMC for random-effects models.")

    fe = get_fixed(get_model(dm))
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("This method requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    _validate_constant_names(fixed_set, constants)
    all(name in keys(constants) for name in fixed_names) &&
        error("This method requires at least one free fixed effect. Remove constants or specify a fixed effect or random effect.")
    layout = free_parameter_layout(fe; constants = constants,
        theta0_untransformed = theta_0_untransformed)
    free_names = layout.free_names
    inv_transform = layout.inv_transform
    θ_const_t_vec = layout.θ_const_t_vec
    free_idx = layout.free_idx
    axs_full = layout.axs_full
    θ0_free_t = layout.θ0_free_t
    cache = build_ll_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization, force_saveat = true)
    function obj(θt, p)
        v_free = θt isa ComponentArray ? ComponentArrays.getdata(θt) : θt
        T = eltype(v_free)
        infT = convert(T, Inf)
        θt_full = _merge_free_into_full(θ_const_t_vec, free_idx, v_free, axs_full)
        θu = inv_transform(θt_full)
        add = add_term(θu)
        add == Inf && return infT
        ll = loglikelihood(
            dm, θu, ComponentArray(); cache = cache, serialization = serialization)
        ll == -Inf && return infT
        return -ll + _penalty_value(θu, penalty) + add
    end

    optf = OptimizationFunction(obj, method.adtype)
    lb, ub, use_bounds, θ0_init = _resolve_optim_bounds(
        fe, free_names, θ0_free_t, method.optimizer, method.lb, method.ub, constants;
        ignore_model_bounds = method.ignore_model_bounds, method_label = "MLE")
    prob = use_bounds ? OptimizationProblem(optf, θ0_init; lb = lb, ub = ub) :
           OptimizationProblem(optf, θ0_init)
    sol = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)

    summary = FitSummary(sol.objective, sol.retcode == SciMLBase.ReturnCode.Success,
        resolve_fitted_parameters(layout, sol.u), NamedTuple())
    diagnostics = FitDiagnostics(
        (;), (optimizer = method.optimizer,), (retcode = sol.retcode,), NamedTuple())
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ?
            sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = FrequentistResult(sol, sol.objective, niter, raw, NamedTuple())
    return FitResult(method, result, summary, diagnostics,
        store_data_model ? dm : nothing, fit_args, fit_kwargs)
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

function _fit_model(dm::DataModel, method::MLE, args...;
        constants::NamedTuple = NamedTuple(),
        penalty::NamedTuple = NamedTuple(),
        extra_objective = nothing,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Random.default_rng(),
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_data_model::Bool = true)
    fit_kwargs = (constants = constants,
        penalty = penalty,
        ode_args = ode_args,
        ode_kwargs = ode_kwargs,
        serialization = serialization,
        rng = rng,
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model)
    return _fit_no_re(dm, method;
        constants = constants,
        penalty = penalty,
        ode_args = ode_args,
        ode_kwargs = ode_kwargs,
        serialization = serialization,
        add_term = _combine_add_terms(_NoOpTerm(), extra_objective),
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model,
        fit_args = args,
        fit_kwargs = fit_kwargs)
end
