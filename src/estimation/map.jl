export MAP
export MAPResult

using Optimization
using OptimizationOptimJL
using SciMLBase
using ComponentArrays
using Random
using LineSearches

"""
    MAP(; optimizer, optim_kwargs, adtype, lb, ub) <: FittingMethod

Maximum A Posteriori estimation for models without random effects.
Requires prior distributions on at least one free fixed effect.

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
struct MAP{O, K, A, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
end

MAP(; optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
    optim_kwargs=NamedTuple(),
    adtype=Optimization.AutoForwardDiff(),
    lb=nothing,
    ub=nothing) = MAP(optimizer, optim_kwargs, adtype, lb, ub)

"""
    MAPResult{S, O, I, R, N} <: MethodResult

Method-specific result from a [`MAP`](@ref) fit. Stores the solution, objective value,
iteration count, raw Optimization.jl result, and optional notes.
"""
struct MAPResult{S, O, I, R, N} <: MethodResult
    solution::S
    objective::O
    iterations::I
    raw::R
    notes::N
end

struct _MAPTerm{F}
        fe::F
end

@inline function (m::_MAPTerm)(θu)
    lp = logprior(m.fe, θu)
    return -lp
end

function _fit_model(dm::DataModel, method::MAP, args...;
                    constants::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                    rng::AbstractRNG=Random.default_rng(),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_data_model::Bool=true)
    fe = dm.model.fixed.fixed
    priors = get_priors(fe)
    has_prior = !isempty(keys(priors)) && any(!(getfield(priors, k) isa Priorless) for k in keys(priors))
    has_prior || error("MAP requires priors on fixed effects. Define priors in @fixedEffects (e.g., RealNumber(...; prior=Normal(...))) or use MLE instead.")

    add_term = _MAPTerm(fe)
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
                      add_term=add_term,
                      theta_0_untransformed=theta_0_untransformed,
                      store_data_model=store_data_model,
                      fit_args=args,
                      fit_kwargs=fit_kwargs)
end
