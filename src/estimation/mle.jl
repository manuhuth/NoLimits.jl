export MLE
export MLEResult
export default_bounds_from_start

using Optimization
using OptimizationOptimJL
using SciMLBase
using ComponentArrays
using Random
using LineSearches
using OptimizationBBO

"""
    MLE(; optimizer, optim_kwargs, adtype, lb, ub) <: FittingMethod

Maximum Likelihood Estimation for models without random effects.

# Keyword Arguments
- `optimizer`: Optimization.jl-compatible optimiser. Defaults to `LBFGS` with backtracking
  line search.
- `optim_kwargs::NamedTuple = NamedTuple()`: keyword arguments forwarded to `Optimization.solve`
  (e.g. `maxiters`, `reltol`).
- `adtype`: automatic-differentiation backend. Defaults to `AutoForwardDiff()`.
- `lb`: lower bounds on the transformed parameter scale, or `nothing` to use the
  model-declared bounds.
- `ub`: upper bounds on the transformed parameter scale, or `nothing`.
"""
struct MLE{O, K, A, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
end

MLE(; optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
     optim_kwargs=NamedTuple(),
     adtype=Optimization.AutoForwardDiff(),
     lb=nothing,
     ub=nothing) = MLE(optimizer, optim_kwargs, adtype, lb, ub)

"""
    MLEResult{S, O, I, R, N} <: MethodResult

Method-specific result from an [`MLE`](@ref) fit. Stores the solution, objective value,
iteration count, raw Optimization.jl result, and optional notes.
"""
struct MLEResult{S, O, I, R, N} <: MethodResult
    solution::S
    objective::O
    iterations::I
    raw::R
    notes::N
end

struct _NoOpTerm end
@inline (_:: _NoOpTerm)(θ) = 0.0

function _fit_no_re(dm::DataModel, method;
                    constants::NamedTuple,
                    penalty::NamedTuple,
                    ode_args::Tuple,
                    ode_kwargs::NamedTuple,
                    serialization::SciMLBase.EnsembleAlgorithm,
                    add_term,
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_data_model::Bool=true,
                    fit_args::Tuple=(),
                    fit_kwargs::NamedTuple=NamedTuple())
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) || error("This method is only valid for models without random effects. Use Laplace, SAEM, or MCMC for random-effects models.")

    fe = dm.model.fixed.fixed
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("This method requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("This method requires at least one free fixed effect. Remove constants or specify a fixed effect or random effect.")
    free_names = [n for n in fixed_names if !(n in keys(constants))]
    θ0_u = get_θ0_untransformed(fe)
    if theta_0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta_0_untransformed, n) || error("theta_0_untransformed is missing parameter $(n).")
        end
        θ0_u = theta_0_untransformed
    end
    
    transform = get_transform(fe)
    inv_transform = get_inverse_transform(fe)
    θ0_t = transform(θ0_u)
    θ_const_u = deepcopy(θ0_u)
    _apply_constants!(θ_const_u, constants)
    θ_const_t = transform(θ_const_u)
    cache = serialization isa SciMLBase.EnsembleThreads ?
            build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=Threads.maxthreadid(), force_saveat=true) :
            build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)

    θ0_free_t = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(θ0_t, n) for n in free_names)))
    axs = getaxes(θ0_free_t)
    function obj(θt, p)
        θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs)
        T = eltype(θt_free)
        infT = convert(T, Inf)
        θt_full = ComponentArray(T.(θ_const_t), getaxes(θ_const_t))
        for name in free_names
            setproperty!(θt_full, name, getproperty(θt_free, name))
        end
        θu = inv_transform(θt_full)
        add = add_term(θu)
        add == Inf && return infT
        ll = loglikelihood(dm, θu, ComponentArray(); cache=cache, serialization=serialization)
        ll == -Inf && return infT
        return -ll + _penalty_value(θu, penalty) + add
    end

    optf = OptimizationFunction(obj, method.adtype)
    lower_t, upper_t = get_bounds_transformed(fe)
    lower_t_free = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(lower_t, n) for n in free_names)))
    upper_t_free = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(upper_t, n) for n in free_names)))
    lower_t_free_vec = collect(lower_t_free)
    upper_t_free_vec = collect(upper_t_free)
    use_bounds = !(all(isinf, lower_t_free_vec) && all(isinf, upper_t_free_vec))
    normalize_bound = function(bound, fallback)
        if bound === nothing
            return fallback
        elseif bound isa Number
            length(fallback) == 1 || error("Scalar bounds are only valid when there is one free parameter.")
            return [bound]
        end
        if bound isa ComponentArray || bound isa NamedTuple
            bound_ca = bound isa ComponentArray ? bound : ComponentArray(bound)
            bound_ca = ComponentArray(NamedTuple{Tuple(free_names)}(Tuple(getproperty(bound_ca, n) for n in free_names)))
            return collect(bound_ca)
        end
        return collect(bound)
    end
    user_bounds = method.lb !== nothing || method.ub !== nothing
    if user_bounds && !isempty(keys(constants))
        @info "Bounds for constant parameters are ignored." constants=collect(keys(constants))
    end
    if user_bounds
        lb = normalize_bound(method.lb, lower_t_free_vec)
        ub = normalize_bound(method.ub, upper_t_free_vec)
    else
        lb = lower_t_free_vec
        ub = upper_t_free_vec
    end
    use_bounds = use_bounds || user_bounds
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO && !use_bounds
        error("BlackBoxOptim methods require finite bounds. Add lower/upper bounds in @fixedEffects (on transformed scale) or pass them via MLE(lb=..., ub=...). A quick helper is default_bounds_from_start(dm; margin=...).")
    end
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO &&
       !(all(isfinite, lb) && all(isfinite, ub))
        error("BlackBoxOptim methods require finite lower and upper bounds for all free parameters.")
    end
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO
        θ0_init = collect(θ0_free_t)
    else
        θ0_init = θ0_free_t
    end
    prob = use_bounds ? OptimizationProblem(optf, θ0_init; lb=lb, ub=ub) :
                        OptimizationProblem(optf, θ0_init)
    sol = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)

    θ_hat_t_raw = sol.u
    θ_hat_t_free = θ_hat_t_raw isa ComponentArray ? θ_hat_t_raw : ComponentArray(θ_hat_t_raw, axs)
    T = eltype(θ_hat_t_free)
    θ_hat_t = ComponentArray(T.(θ_const_t), getaxes(θ_const_t))
    for name in free_names
        setproperty!(θ_hat_t, name, getproperty(θ_hat_t_free, name))
    end
    θ_hat_u = inv_transform(θ_hat_t)

    summary = FitSummary(sol.objective, sol.retcode == SciMLBase.ReturnCode.Success,
                         FitParameters(θ_hat_t, θ_hat_u),
                         NamedTuple())
    diagnostics = FitDiagnostics((;), (optimizer=method.optimizer,), (retcode=sol.retcode,), NamedTuple())
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ? sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = MLEResult(sol, sol.objective, niter, raw, NamedTuple())
    return FitResult(method, result, summary, diagnostics, store_data_model ? dm : nothing, fit_args, fit_kwargs)
end

"""
    default_bounds_from_start(dm::DataModel; margin=1.0) -> (lower, upper)

Generate symmetric box bounds on the transformed parameter scale centred at the
initial parameter values, with half-width `margin`.

Useful for passing to `MLE(lb=lower, ub=upper)` when the model-declared bounds are
too wide or absent.

# Keyword Arguments
- `margin::Real = 1.0`: half-width of the symmetric box on the transformed scale.
"""
function default_bounds_from_start(dm::DataModel; margin::Real=1.0)
    θ = get_θ0_transformed(dm.model.fixed.fixed)
    lower = deepcopy(θ)
    upper = deepcopy(θ)
    lower .= θ .- margin
    upper .= θ .+ margin
    return (lower, upper)
end

function _apply_constants!(θ, constants::NamedTuple)
    for name in keys(constants)
        val = getfield(constants, name)
        setproperty!(θ, name, val)
    end
    return θ
end

function _penalty_value(θ, penalty::NamedTuple)
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

function _fit_model(dm::DataModel, method::MLE, args...;
                    constants::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                    rng::AbstractRNG=Random.default_rng(),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_data_model::Bool=true)
    fit_kwargs = (constants=constants,
                  penalty=penalty,
                  ode_args=ode_args,
                  ode_kwargs=ode_kwargs,
                  serialization=serialization,
                  rng=rng,
                  theta_0_untransformed=theta_0_untransformed,
                  store_data_model=store_data_model)
    return _fit_no_re(dm, method;
                      constants=constants,
                      penalty=penalty,
                      ode_args=ode_args,
                      ode_kwargs=ode_kwargs,
                      serialization=serialization,
                      add_term=_NoOpTerm(),
                      theta_0_untransformed=theta_0_untransformed,
                      store_data_model=store_data_model,
                      fit_args=args,
                      fit_kwargs=fit_kwargs)
end
