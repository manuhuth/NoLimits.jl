export MAP
export MAPResult

using Optimization
using OptimizationOptimJL
using SciMLBase
using ComponentArrays
using Random
using LineSearches

"""
    MAP(; optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds,
    precondition) <: FittingMethod

Maximum A Posteriori estimation for models without random effects.
Requires prior distributions on at least one free fixed effect.

# Keyword Arguments
- `optimizer`: Optimization.jl-compatible optimizer. Defaults to `LBFGS` with backtracking
  line search.
- `optim_kwargs::NamedTuple = NamedTuple()`: keyword arguments forwarded to `Optimization.solve`
  (e.g. `maxiters`, `reltol`).
- `adtype`: automatic-differentiation backend. Defaults to `AutoForwardDiff()`.
- `lb`: lower bounds on the transformed parameter scale, or `nothing` to use the
  model-declared bounds.
- `ub`: upper bounds on the transformed parameter scale, or `nothing`.
- `ignore_model_bounds::Bool = false`: when `true`, ignore bounds declared in
  `@fixedEffects` unless explicit `lb`/`ub` are passed.
- `precondition::Bool = true`: optimize the scaled offset `z` with
  `θ_transformed = θ0 + s .* z`, so every fit starts at `z = 0` and no coordinate can be
  frozen by an unlucky starting value. `s` is 1 for any coordinate already in log/logit
  space and `max(abs(θ0), 1)` for a genuinely natural-scale `:identity` coordinate. Set
  `false` to optimize the transformed vector directly, which reproduces pre-0.2 results
  bit-for-bit. Note that with preconditioning on, the optimizer object behind
  [`get_raw`](@ref) works in `z`; [`get_params`](@ref) always returns the usual scales.
"""
struct MAP{O, K, A, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
    ignore_model_bounds::Bool
    precondition::Bool
end

function MAP(;
        optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking(maxstep = 1.0)),
        optim_kwargs = NamedTuple(),
        adtype = Optimization.AutoForwardDiff(),
        lb = nothing,
        ub = nothing,
        ignore_model_bounds = false,
        precondition = true
    )
    return MAP(
        optimizer, _as_namedtuple(optim_kwargs), adtype, lb, ub,
        ignore_model_bounds, precondition
    )
end

# MAPResult is a StandardOptimizationResult{:map} alias + constructor (see common.jl).

struct _MAPTerm{F}
    fe::F
end

@inline function (m::_MAPTerm)(θu)
    lp = logprior(m.fe, θu)
    return -lp
end

function _fit_model(
        dm::DataModel, method::MAP, args...;
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
    fe = get_fixed(get_model(dm))
    has_prior = _has_fixed_priors(fe, constants)
    has_prior ||
        error("MAP requires priors on free fixed effects. Define priors in @fixedEffects (e.g., RealNumber(...; prior=Normal(...))), drop the prior-carrying parameters from `constants`, or use MLE instead.")

    _warn_unbounded_prior_at_bounds(fe)
    add_term = _combine_add_terms(_MAPTerm(fe), extra_objective)
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
        add_term = add_term,
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model,
        fit_args = args,
        fit_kwargs = fit_kwargs
    )
end
