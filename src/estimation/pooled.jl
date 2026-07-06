export Pooled
export PooledMap
export PooledResult

using Optimization
using OptimizationOptimJL
using SciMLBase
using ComponentArrays
using Random
using LineSearches
using OptimizationBBO
using ForwardDiff
using LinearAlgebra
using Statistics

"""
    Pooled(; optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds,
           force_free, refreeze_check, identifiable_only, n_probes, mc_draws) <: FittingMethod

Pooled estimation for models with random effects. Each individual's random effects are
set to the **plug-in value of their RE distribution** (mean, falling back to median; a
fixed-draw Monte-Carlo mean for normalizing flows), then the data log-likelihood alone is
optimized over the free fixed effects. The plug-in is a function of the fixed effects and
is recomputed at every objective evaluation, so parameters that shift the plug-in (e.g. a
population mean inside the RE distribution) are estimated.

Only fixed effects with **no detectable likelihood contribution** are automatically held
constant at their initial values:
- *dispersion-only* parameters whose plug-in sensitivity (mean-Jacobian) is zero at the
  start and at jittered probe points, cross-checked against a spread measure (variance /
  IQR) and an end-to-end objective-invariance test;
- *collinear* parameters whose plug-in effect is redundant given the remaining free
  parameters at every probe point (e.g. only the ratio of `Beta(α, β)` is identified).

The freeze classification is reported in `get_notes(res)`. The RE prior is never
evaluated.

# Keyword Arguments
- `optimizer`: Optimization.jl-compatible optimizer. Defaults to `LBFGS` with backtracking.
- `optim_kwargs::NamedTuple = NamedTuple()`: forwarded to `Optimization.solve`.
- `adtype`: AD backend. Defaults to `AutoForwardDiff()`.
- `lb`/`ub`: bounds on the transformed scale, or `nothing` to use model-declared bounds.
- `ignore_model_bounds::Bool = false`: ignore bounds declared in `@fixedEffects`.
- `force_free::Vector{Symbol} = Symbol[]`: parameter names exempt from auto-freezing.
- `refreeze_check::Symbol = :warn`: post-fit sensitivity re-check at the optimum;
  `:warn` records violations in the notes, `:refit` unfreezes violators and continues
  optimization warm-started from the current optimum.
- `identifiable_only::Bool = true`: freeze plug-in-collinear parameters (pivoted
  redundancy elimination); `false` keeps all contributing parameters free.
- `n_probes::Int = 3`: number of probe points (start + jittered) for the sensitivity
  analysis on the transformed scale.
- `mc_draws::Int = 256`: fixed base draws for the Monte-Carlo plug-in mean of
  normalizing-flow random effects.
"""
struct Pooled{O, K, A, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
    ignore_model_bounds::Bool
    force_free::Vector{Symbol}
    refreeze_check::Symbol
    identifiable_only::Bool
    n_probes::Int
    mc_draws::Int
end

function Pooled(;
        optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()),
        optim_kwargs = NamedTuple(),
        adtype = Optimization.AutoForwardDiff(),
        lb = nothing,
        ub = nothing,
        ignore_model_bounds = false,
        force_free::Vector{Symbol} = Symbol[],
        refreeze_check::Symbol = :warn,
        identifiable_only::Bool = true,
        n_probes::Int = 3,
        mc_draws::Int = 256)
    refreeze_check in (:warn, :refit) || error("refreeze_check must be :warn or :refit.")
    n_probes >= 1 || error("n_probes must be >= 1.")
    mc_draws >= 1 || error("mc_draws must be >= 1.")
    return Pooled(optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds,
        force_free, refreeze_check, identifiable_only, n_probes, mc_draws)
end

"""
    PooledMap(; optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds,
              force_free, refreeze_check, identifiable_only, n_probes, mc_draws) <: FittingMethod

Like [`Pooled`](@ref), but adds the log-prior of the fixed effects to the objective
(MAP on the data likelihood with RE plugged in at their distributional means). Requires
priors on at least one fixed effect.

Auto-frozen parameters (see [`Pooled`](@ref)) are held constant; their priors contribute
a constant offset to the reported objective.
"""
struct PooledMap{O, K, A, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
    ignore_model_bounds::Bool
    force_free::Vector{Symbol}
    refreeze_check::Symbol
    identifiable_only::Bool
    n_probes::Int
    mc_draws::Int
end

function PooledMap(;
        optimizer = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()),
        optim_kwargs = NamedTuple(),
        adtype = Optimization.AutoForwardDiff(),
        lb = nothing,
        ub = nothing,
        ignore_model_bounds = false,
        force_free::Vector{Symbol} = Symbol[],
        refreeze_check::Symbol = :warn,
        identifiable_only::Bool = true,
        n_probes::Int = 3,
        mc_draws::Int = 256)
    refreeze_check in (:warn, :refit) || error("refreeze_check must be :warn or :refit.")
    n_probes >= 1 || error("n_probes must be >= 1.")
    mc_draws >= 1 || error("mc_draws must be >= 1.")
    return PooledMap(optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds,
        force_free, refreeze_check, identifiable_only, n_probes, mc_draws)
end

"""
    PooledResult{S, O, I, R, N, E} <: MethodResult

Method-specific result from a [`Pooled`](@ref) or [`PooledMap`](@ref) fit. Stores the
optimization solution plus the per-individual plug-in random effects evaluated at the
fitted fixed effects. The `notes` field records the plug-in strategy per random effect
and the freeze classification (`frozen_dispersion`, `frozen_collinear`, `frozen_inert`,
`frozen_unverified`, `weakly_identified`, `unfrozen_by_invariance`, `unfrozen_postfit`,
`postfit_violations`). The `strategies` field stores the resolved per-RE plug-in
strategies (including any fixed Monte-Carlo base draws) so downstream consumers — e.g.
Wald UQ — can replay the exact plug-in map η(θ).
"""
struct PooledResult{S, O, I, R, N, E, St} <: MethodResult
    solution::S
    objective::O
    iterations::I
    raw::R
    notes::N
    eta_vec::E      # Vector{ComponentArray} — per-individual plug-in η at the optimum
    strategies::St  # per-RE plug-in strategies used by the fit
end

# ─── plug-in strategies ───────────────────────────────────────────────────────────
#
# Each RE gets one strategy, probed once at setup so the per-iteration path is
# branch-stable and free of try/catch:
#   :mean    closed-form mean(dist)             (Beta, Gamma, LogNormal, MvNormal, …)
#   :median  closed-form median(dist)           (Cauchy, heavy-tailed t, …)
#   MC mean  fixed-base-draw reparameterized mean for normalizing flows
#   :zero    fallback — plug-in is constant zero

struct _PooledMCPlugin{Z}
    z::Z   # Vector of fixed base draws, shared across iterations and probe points
end

_pooled_plugin_label(s::Symbol) = s
_pooled_plugin_label(::_PooledMCPlugin) = :mc_mean

# Evaluate `f()`, returning its value only when it is finite throughout; `nothing`
# on throw or non-finite. Probes which plug-in statistics a RE distribution supports.
function _pooled_finite_or_nothing(f)
    v = try
        f()
    catch
        nothing
    end
    return v !== nothing && all(isfinite, v) ? v : nothing
end

function _pooled_plugin_strategies(dm::DataModel, θ::ComponentArray;
        mc_draws::Int = 256, rng::AbstractRNG = Random.default_rng())
    model = get_model(dm)
    lp_cache = dm.re_group_info.laplace_cache
    lp_cache === nothing && error("Pooled() requires a model with random effects.")
    re_names = lp_cache.re_names
    isempty(re_names) && error("Pooled() requires at least one random effect.")
    dists_builder = get_create_random_effect_distribution(model.random.random)
    model_funs = get_model_funs(model)
    helpers = get_helper_funs(model)
    ind = first(get_individuals(dm))
    θs = _symmetrize_psd_params(θ, model.fixed.fixed)
    dists = dists_builder(θs, ind.const_cov, model_funs, helpers)
    strategies = Any[]
    for re in re_names
        dist = getproperty(dists, re)
        strat = if _pooled_finite_or_nothing(() -> mean(dist)) !== nothing
            :mean
        elseif _pooled_finite_or_nothing(() -> median(dist)) !== nothing
            :median
        elseif dist isa NormalizingPlanarFlow
            _PooledMCPlugin([rand(rng, dist.base.dist) for _ in 1:mc_draws])
        else
            :zero
        end
        push!(strategies, strat)
    end
    return Tuple(strategies)
end

function _pooled_eta_value(dist, dim::Int, strategy, ::Type{T}) where {T}
    local v
    if strategy === :mean
        v = mean(dist)
    elseif strategy === :median
        v = median(dist)
    elseif strategy isa _PooledMCPlugin
        zs = strategy.z
        f = dist.base.transform
        acc = f(zs[1])
        for k in 2:length(zs)
            acc = acc .+ f(zs[k])
        end
        v = acc ./ length(zs)
    else  # :zero
        return dim == 1 ? zero(T) : zeros(T, dim)
    end
    if dim == 1
        v isa AbstractVector && (v = first(v))
        return T(v)
    end
    return Vector{T}(v)
end

# Per-individual plug-in η fill for the `_compute_pooled_etas` fast path, behind a
# function barrier so `axs` / `ηlen` / `T` arrive as positional arguments (concrete)
# rather than locals captured by the comprehension closure. Previously `axs`/`ηlen`
# were assigned in BOTH the fast and the heterogeneous branch and captured by the
# fast-path comprehension, which boxed them (`Core.Box`) and inferred the whole
# per-individual body as `Any` (a runtime dispatch per individual per objective eval).
@inline function _pooled_fill_ind_eta(dists_builder::DB, θs::TH, const_cov,
        model_funs, helpers, re_names, dims, strategies, ::Type{T}, axs::AX,
        ηlen::Int) where {DB, TH, T, AX}
    dists = dists_builder(θs, const_cov, model_funs, helpers)
    vals = Vector{T}(undef, ηlen)
    pos = 1
    for (ri, re) in enumerate(re_names)
        val = _pooled_eta_value(getproperty(dists, re), dims[ri], strategies[ri], T)
        pos = _pooled_fill_eta!(vals, pos, val)
    end
    return ComponentArray(vals, axs)
end

function _compute_pooled_etas(dm::DataModel, θ::ComponentArray, strategies)
    model = get_model(dm)
    lp_cache = dm.re_group_info.laplace_cache
    lp_cache === nothing && error("Pooled() requires a model with random effects.")
    re_names = lp_cache.re_names
    isempty(re_names) && error("Pooled() requires at least one random effect.")
    dists_builder = get_create_random_effect_distribution(model.random.random)
    model_funs = get_model_funs(model)
    helpers = get_helper_funs(model)
    individuals = get_individuals(dm)
    θs = _symmetrize_psd_params(θ, model.fixed.fixed)
    T = eltype(θs)
    n = length(individuals)
    template = lp_cache.eta_template
    # The pooled plug-in yields a SCALAR for every dim-1 RE, while the template
    # axes keep vector-valued dim-1 REs (e.g. a 1-input planar flow) as
    # 1-element vector components — formulas written against the scalar shape
    # (`sat(η)`) would break. Only take the fast path when the shapes agree.
    template_compatible = template !== nothing &&
                          all(
        ri -> (lp_cache.dims[ri] == 1) == lp_cache.is_scalar[ri], eachindex(re_names))
    if template_compatible
        # Fast path (one RE level per group per individual): fill a flat vector and
        # wrap it with the precomputed axes — same trick as `_build_eta_ind_fast`.
        # The per-individual fill is hoisted into `_pooled_fill_ind_eta` (axs / ηlen /
        # T passed positionally) so the comprehension closure does not capture them;
        # see that helper for the Core.Box this avoids. Runs once per objective
        # evaluation, per individual.
        axs = getaxes(template)
        ηlen = length(template)
        dims = lp_cache.dims
        return [_pooled_fill_ind_eta(dists_builder, θs, individuals[i].const_cov,
                    model_funs, helpers, re_names, dims, strategies, T, axs, ηlen)
                for i in 1:n]
    end
    # Heterogeneous-shape path (e.g. a dim-1 NPF RE kept as a 1-vector): probe the
    # plug-in SHAPES once with the first individual to build the output axes (the
    # boxed NamedTuple construction runs exactly once), then fill a flat vector per
    # individual and wrap it with the probed axes — mirroring the fast path above.
    # The historical per-individual `Pair{Symbol,Any}[]` + `ComponentArray(NamedTuple)`
    # route boxed every plug-in value on every objective evaluation (measured ~50×
    # per individual under ForwardDiff).
    ind1 = individuals[1]
    dists1 = dists_builder(θs, ind1.const_cov, model_funs, helpers)
    probe_vals = map(enumerate(re_names)) do (ri, re)
        _pooled_eta_value(getproperty(dists1, re), lp_cache.dims[ri], strategies[ri], T)
    end
    proto = ComponentArray(NamedTuple{Tuple(re_names)}(Tuple(probe_vals)))
    axs = getaxes(proto)
    ηlen = length(proto)
    # An RE distribution can only vary across individuals through CONSTANT
    # covariates (validated at DataModel construction). When a RE's distribution
    # expression references none, its plug-in value is individual-invariant and the
    # probe value is reused for everyone — `_pooled_eta_value` can be expensive
    # (e.g. the NPF MC plug-in evaluates the flow once per stored draw).
    re_syms = get_re_syms(model.random.random)
    const_names = model.covariates.covariates.constants
    invariant = map(re -> isempty(intersect(getfield(re_syms, re), const_names)),
        re_names)
    return [begin
                vals = Vector{T}(undef, ηlen)
                pos = 1
                if i == 1 || all(invariant)
                    for val in probe_vals
                        pos = _pooled_fill_eta!(vals, pos, val)
                    end
                else
                    ind = individuals[i]
                    dists = dists_builder(θs, ind.const_cov, model_funs, helpers)
                    for (ri, re) in enumerate(re_names)
                        val = invariant[ri] ? probe_vals[ri] :
                              _pooled_eta_value(getproperty(dists, re),
                            lp_cache.dims[ri], strategies[ri], T)
                        pos = _pooled_fill_eta!(vals, pos, val)
                    end
                end
                ComponentArray(vals, axs)
            end
            for i in 1:n]
end

@inline function _pooled_fill_eta!(vals::Vector, pos::Int, val::Number)
    vals[pos] = val
    return pos + 1
end
@inline function _pooled_fill_eta!(vals::Vector, pos::Int, val)
    for k in eachindex(val)
        vals[pos + k - 1] = val[k]
    end
    return pos + length(val)
end

# Flat vector of all plug-in values across individuals — the function of θ that
# actually enters the pooled likelihood through η.
function _pooled_stacked_means(dm::DataModel, θ::ComponentArray, strategies)
    η_vec = _compute_pooled_etas(dm, θ, strategies)
    T = eltype(θ)
    out = Vector{T}(undef, sum(length, η_vec; init = 0))
    k = 1
    for i in 1:length(η_vec)
        for x in η_vec[i]
            out[k] = x
            k += 1
        end
    end
    return out
end

# Plug-in values of a single RE across individuals — used to test each RE's
# strategy for ForwardDiff-safety in isolation.
function _pooled_stacked_means_single(dm::DataModel, θ::ComponentArray, strategy, ri::Int)
    model = get_model(dm)
    lp_cache = dm.re_group_info.laplace_cache
    re = lp_cache.re_names[ri]
    dists_builder = get_create_random_effect_distribution(model.random.random)
    model_funs = get_model_funs(model)
    helpers = get_helper_funs(model)
    individuals = get_individuals(dm)
    θs = _symmetrize_psd_params(θ, model.fixed.fixed)
    T = eltype(θs)
    out = T[]
    for i in 1:length(individuals)
        dists = dists_builder(θs, individuals[i].const_cov, model_funs, helpers)
        val = _pooled_eta_value(getproperty(dists, re), lp_cache.dims[ri], strategy, T)
        val isa AbstractVector ? append!(out, val) : push!(out, val)
    end
    return out
end

# Demotion ladder: :mean → :median → MC mean (flows) → :zero. Used when a
# strategy's plug-in is not ForwardDiff-differentiable at the start values
# (e.g. mean/median of a Normal truncated at Inf produce NaN partials).
function _pooled_demote_strategy(strategy, dist, mc_draws::Int, rng::AbstractRNG)
    is_flow = dist isa NormalizingPlanarFlow
    if strategy === :mean
        med = try
            median(dist)
        catch
            nothing
        end
        med !== nothing && all(isfinite, med) && return :median
        return is_flow ? _PooledMCPlugin([rand(rng, dist.base.dist) for _ in 1:mc_draws]) :
               :zero
    elseif strategy === :median
        return is_flow ? _PooledMCPlugin([rand(rng, dist.base.dist) for _ in 1:mc_draws]) :
               :zero
    end
    return :zero  # MC failed too
end

# Demote each RE's strategy until its plug-in has finite ForwardDiff derivatives
# at the start values (or reaches :zero).
function _pooled_dual_safe_strategies(dm::DataModel, θ_user_u::ComponentArray,
        θ_user_t::ComponentArray, inv_transform,
        strategies_in, mc_draws::Int, rng::AbstractRNG)
    model = get_model(dm)
    lp_cache = dm.re_group_info.laplace_cache
    re_names = lp_cache.re_names
    dists_builder = get_create_random_effect_distribution(model.random.random)
    θs = _symmetrize_psd_params(θ_user_u, model.fixed.fixed)
    dists = dists_builder(θs, first(get_individuals(dm)).const_cov,
        get_model_funs(model), get_helper_funs(model))
    axs = getaxes(θ_user_t)
    θ0_vec = _pooled_flat(θ_user_t)
    idx_all = collect(1:length(θ0_vec))
    strategies = Any[strategies_in...]
    for (ri, re) in enumerate(re_names)
        while strategies[ri] !== :zero
            ok = try
                f = θu -> _pooled_stacked_means_single(dm, θu, strategies[ri], ri)
                J = _pooled_coord_jacobian(f, θ0_vec, axs, inv_transform, idx_all)
                all(isfinite, J)
            catch
                false
            end
            ok && break
            old = _pooled_plugin_label(strategies[ri])
            strategies[ri] = _pooled_demote_strategy(
                strategies[ri], getproperty(dists, re),
                mc_draws, rng)
            new = _pooled_plugin_label(strategies[ri])
            @warn "Pooled: plug-in for random effect $(re) demoted from :$(old) to " *
                  ":$(new) — :$(old) is not ForwardDiff-differentiable at the start " *
                  "values." *
                  (strategies[ri] === :zero ?
                   " The plug-in is now constant zero; its distribution parameters " *
                   "cannot be estimated by Pooled()." : "")
        end
    end
    return Tuple(strategies)
end

# ─── spread measures (verification A) ─────────────────────────────────────────────

function _pooled_spread_kinds(dm::DataModel, θ::ComponentArray)
    model = get_model(dm)
    lp_cache = dm.re_group_info.laplace_cache
    re_names = lp_cache.re_names
    dists_builder = get_create_random_effect_distribution(model.random.random)
    model_funs = get_model_funs(model)
    helpers = get_helper_funs(model)
    ind = first(get_individuals(dm))
    θs = _symmetrize_psd_params(θ, model.fixed.fixed)
    dists = dists_builder(θs, ind.const_cov, model_funs, helpers)
    kinds = Symbol[]
    for (ri, re) in enumerate(re_names)
        dist = getproperty(dists, re)
        kind = if lp_cache.dims[ri] == 1
            if _pooled_finite_or_nothing(() -> var(dist)) !== nothing
                :var
            elseif _pooled_finite_or_nothing(() -> quantile(dist, 0.75) -
                                                   quantile(dist, 0.25)) !==
                   nothing
                :iqr
            else
                :none
            end
        else
            _pooled_finite_or_nothing(() -> vec(Matrix(cov(dist)))) !== nothing ? :cov :
            :none
        end
        push!(kinds, kind)
    end
    return Tuple(kinds)
end

function _pooled_spread_value(dist, kind::Symbol, ::Type{T}) where {T}
    if kind === :var
        v = var(dist)
        v isa AbstractVector && return Vector{T}(v)
        return T[v]
    elseif kind === :cov
        return Vector{T}(vec(Matrix(cov(dist))))
    elseif kind === :iqr
        return T[quantile(dist, 0.75) - quantile(dist, 0.25)]
    end
    return T[]
end

function _pooled_stacked_spreads(dm::DataModel, θ::ComponentArray, kinds)
    model = get_model(dm)
    lp_cache = dm.re_group_info.laplace_cache
    re_names = lp_cache.re_names
    dists_builder = get_create_random_effect_distribution(model.random.random)
    model_funs = get_model_funs(model)
    helpers = get_helper_funs(model)
    individuals = get_individuals(dm)
    θs = _symmetrize_psd_params(θ, model.fixed.fixed)
    T = eltype(θs)
    out = T[]
    for i in 1:length(individuals)
        dists = dists_builder(θs, individuals[i].const_cov, model_funs, helpers)
        for (ri, re) in enumerate(re_names)
            append!(out, _pooled_spread_value(getproperty(dists, re), kinds[ri], T))
        end
    end
    return out
end

# ─── structural usage and coordinate bookkeeping ──────────────────────────────────

# Fixed-effect names with a direct likelihood path outside the RE distributions
# (formulas, DE, initial conditions, preDE). Built as a superset — anything here is
# kept free, so over-collection only errs toward not freezing.
function _pooled_structural_syms(model)
    syms = Set{Symbol}()
    ir = get_formulas_ir(model.formulas.formulas)
    union!(syms, ir.var_syms)
    union!(syms, ir.prop_syms)
    de = model.de.de
    if de !== nothing
        m = get_de_meta(de)
        union!(syms, m.var_syms)
        union!(syms, m.fun_syms)
    end
    initial = model.de.initial
    if initial !== nothing
        ir_init = initial.ir
        union!(syms, ir_init.var_syms)
        union!(syms, ir_init.prop_syms)
        union!(syms, ir_init.call_syms)
    end
    prede = model.de.prede
    if prede !== nothing
        union!(syms, get_prede_meta(prede).syms)
    end
    return syms
end

function _pooled_re_dist_syms(model)
    re_syms_nt = get_re_syms(model.random.random)
    return reduce((s, v) -> union(s, v), values(re_syms_nt); init = Set{Symbol}())
end

# Flat coordinate ranges of each parameter inside the transformed ComponentArray.
function _pooled_param_ranges(θt::ComponentArray, names)
    ranges = Dict{Symbol, UnitRange{Int}}()
    off = 0
    for n in names
        l = length(getproperty(θt, n))
        ranges[n] = (off + 1):(off + l)
        off += l
    end
    off == length(θt) || error("Internal error: parameter ranges do not cover θ.")
    return ranges
end

# Jacobian of f_stack(inv_transform(θt)) w.r.t. the coordinates `idx` of the flat
# transformed vector, holding all other coordinates at `θ_probe_vec`.
function _pooled_coord_jacobian(f_stack, θ_probe_vec::Vector{Float64}, axs,
        inv_transform, idx::Vector{Int})
    g = function (x)
        full = Vector{eltype(x)}(θ_probe_vec)
        full[idx] .= x
        θt = ComponentArray(full, axs)
        return f_stack(inv_transform(θt))
    end
    return ForwardDiff.jacobian(g, θ_probe_vec[idx])
end

const _POOLED_ZERO_RTOL = 1e-10
const _POOLED_RANK_RTOL = 1e-8
const _POOLED_REDUNDANT_RTOL = 1e-6

# Plain flat Vector{Float64} copy of a ComponentArray (collect preserves the CA type).
_pooled_flat(ca) = copyto!(Vector{Float64}(undef, length(ca)), ca)

# Is the column block Jn (numerically) inside the column span of Jo? Least-squares
# residual test — robust to the small row-to-row noise that breaks exact rank
# comparisons for nearly-collinear Jacobians.
function _pooled_block_redundant(Jo::AbstractMatrix, Jn::AbstractMatrix)
    size(Jo, 2) == 0 && return false
    coef = Jo \ Jn
    resid = norm(Jn .- Jo * coef)
    return resid <= _POOLED_REDUNDANT_RTOL * max(1.0, norm(Jn))
end

_pooled_zero_tol(J::AbstractMatrix) = _POOLED_ZERO_RTOL * max(1.0, norm(J))

# ─── probe points ─────────────────────────────────────────────────────────────────

# Probe points on the transformed scale: the start point plus jittered points.
# Jitter every non-user-constant coordinate; retry with shrinking magnitude if the
# jittered point cannot evaluate the plug-ins (e.g. invalid distribution parameters
# on :identity scales).
function _pooled_probe_points(dm::DataModel, θ0_vec::Vector{Float64}, axs,
        inv_transform, strategies, jitter_idx::Vector{Int},
        n_probes::Int, rng::AbstractRNG)
    evaluable = function (vec)
        try
            m = _pooled_stacked_means(
                dm, inv_transform(ComponentArray(vec, axs)), strategies)
            all(isfinite, m)
        catch
            false
        end
    end
    evaluable(θ0_vec) ||
        error("Pooled: random-effect plug-in values cannot be evaluated at the initial " *
              "fixed effects. Check the @randomEffects distributions at the start values.")
    probes = [copy(θ0_vec)]
    for _ in 2:n_probes
        found = false
        for scale in (1.0, 0.5, 0.1, 0.01)
            cand = copy(θ0_vec)
            cand[jitter_idx] .+= scale .* (2 .* rand(rng, length(jitter_idx)) .- 1)
            if evaluable(cand)
                push!(probes, cand)
                found = true
                break
            end
        end
        found || push!(probes, copy(θ0_vec))
    end
    return probes
end

# ─── partition: free vs frozen ────────────────────────────────────────────────────
#
# A fixed effect appearing in an RE distribution is frozen only when it has no
# detectable likelihood contribution:
#  - structural use (formulas/DE/initialDE/preDE) → always free;
#  - nonzero plug-in mean-Jacobian at any probe   → free (it shifts η);
#  - zero mean-Jacobian at every probe            → dispersion candidate, frozen,
#    cross-checked against a spread measure (verification A);
#  - redundant given the kept set at every probe  → collinear, frozen (pivoted
#    greedy elimination, later-declared parameters frozen first).
# Ties go to free: wrong-free is a flat direction, wrong-frozen is a silent bias.

function _pooled_partition(dm::DataModel, method, θ_user_t::ComponentArray,
        inv_transform, constants::NamedTuple, strategies,
        rng::AbstractRNG)
    model = get_model(dm)
    fe_names = get_names(model.fixed.fixed)
    re_syms = _pooled_re_dist_syms(model)
    structural = _pooled_structural_syms(model)
    force_free = Set(method.force_free)
    in_re = [n for n in fe_names if n in re_syms]
    structural_in_re = [n for n in in_re if n in structural]
    candidates = [n
                  for n in in_re
                  if !(n in structural) && !haskey(constants, n) && !(n in force_free)]

    base = (;
        candidates = Symbol[], frozen_dispersion = Symbol[], frozen_collinear = Symbol[],
        frozen_inert = Symbol[], frozen_unverified = Symbol[], weakly_identified = Symbol[],
        structural_in_re = structural_in_re, probe_failed = false)
    isempty(candidates) && return base

    axs = getaxes(θ_user_t)
    ranges = _pooled_param_ranges(θ_user_t, fe_names)
    θ0_vec = _pooled_flat(θ_user_t)
    jitter_idx = vcat([collect(ranges[n]) for n in fe_names if !haskey(constants, n)]...)
    probes = _pooled_probe_points(dm, θ0_vec, axs, inv_transform, strategies,
        jitter_idx, method.n_probes, rng)
    cand_idx = vcat([collect(ranges[n]) for n in candidates]...)
    rel = Dict{Symbol, UnitRange{Int}}()
    off = 0
    for n in candidates
        l = length(ranges[n])
        rel[n] = (off + 1):(off + l)
        off += l
    end

    mean_stack = θu -> _pooled_stacked_means(dm, θu, strategies)
    Js = try
        all_J = [_pooled_coord_jacobian(mean_stack, p, axs, inv_transform, cand_idx)
                 for p in probes]
        usable = [J for J in all_J if all(isfinite, J)]
        isempty(usable) ? nothing : usable
    catch
        nothing
    end
    if Js === nothing
        # Plug-in not ForwardDiff-probeable (e.g. a numerically-inverted median).
        # Conservatively freeze all candidates; the objective-invariance check
        # (value-level, no duals) rescues any that do reach the likelihood.
        return (; candidates = candidates, frozen_dispersion = copy(candidates),
            frozen_collinear = Symbol[], frozen_inert = Symbol[],
            frozen_unverified = copy(candidates), weakly_identified = Symbol[],
            structural_in_re = structural_in_re, probe_failed = true)
    end

    # zero-block detection: dead at EVERY probe → dispersion candidate
    frozen_dispersion = Symbol[]
    for n in candidates
        dead = all(norm(J[:, rel[n]]) <= _pooled_zero_tol(J) for J in Js)
        dead && push!(frozen_dispersion, n)
    end

    # verification A: a frozen candidate should provably move a spread measure
    frozen_inert = Symbol[]
    frozen_unverified = Symbol[]
    if !isempty(frozen_dispersion)
        kinds = _pooled_spread_kinds(dm, inv_transform(θ_user_t))
        spread_stack = θu -> _pooled_stacked_spreads(dm, θu, kinds)
        frozen_idx = vcat([collect(ranges[n]) for n in frozen_dispersion]...)
        rel_frozen = Dict{Symbol, UnitRange{Int}}()
        off = 0
        for n in frozen_dispersion
            l = length(ranges[n])
            rel_frozen[n] = (off + 1):(off + l)
            off += l
        end
        Js_spread = try
            all_J = [_pooled_coord_jacobian(
                         spread_stack, p, axs, inv_transform, frozen_idx)
                     for p in probes]
            usable = [J for J in all_J if all(isfinite, J)]
            isempty(usable) ? nothing : usable
        catch
            nothing
        end
        if Js_spread === nothing || all(size(J, 1) == 0 for J in Js_spread)
            append!(frozen_unverified, frozen_dispersion)
        else
            re_syms_nt = get_re_syms(model.random.random)
            re_names = dm.re_group_info.laplace_cache.re_names
            for n in frozen_dispersion
                moves_spread = any(norm(J[:, rel_frozen[n]]) > _pooled_zero_tol(J)
                for J in Js_spread)
                moves_spread && continue
                # parameter moves neither plug-in nor spread — inert if some RE
                # using it has an available spread measure, otherwise unverified
                has_measure = any(kinds[ri] !== :none && n in re_syms_nt[re]
                for (ri, re) in enumerate(re_names))
                push!(has_measure ? frozen_inert : frozen_unverified, n)
            end
        end
        if !isempty(frozen_inert)
            @warn "Pooled: parameter(s) $(join(frozen_inert, ", ")) have no detectable " *
                  "influence on the random-effect distributions (neither plug-in value " *
                  "nor spread). They are held constant — check the model specification."
        end
    end

    # collinearity: greedy redundancy elimination over the remaining candidates
    nonzero = [n for n in candidates if !(n in frozen_dispersion)]
    frozen_collinear = Symbol[]
    if method.identifiable_only && length(nonzero) > 1
        kept = copy(nonzero)
        for n in reverse(nonzero)
            others = [k for k in kept if k !== n]
            isempty(others) && continue
            cols_o = vcat([collect(rel[k]) for k in others]...)
            redundant = all(J -> _pooled_block_redundant(J[:, cols_o], J[:, rel[n]]), Js)
            if redundant
                push!(frozen_collinear, n)
                kept = others
            end
        end
    end

    # multi-coordinate blocks that stay free but are internally rank-deficient at
    # every probe (e.g. flow parameters identified only through a d-dim mean)
    weakly_identified = Symbol[]
    for n in nonzero
        n in frozen_collinear && continue
        ncoords = length(rel[n])
        ncoords > 1 || continue
        max_rank = maximum(rank(J[:, rel[n]]; rtol = _POOLED_RANK_RTOL) for J in Js)
        max_rank < ncoords && push!(weakly_identified, n)
    end

    return (; candidates = candidates, frozen_dispersion = frozen_dispersion,
        frozen_collinear = frozen_collinear, frozen_inert = frozen_inert,
        frozen_unverified = frozen_unverified, weakly_identified = weakly_identified,
        structural_in_re = structural_in_re, probe_failed = false)
end

# ─── verification B: end-to-end objective invariance ─────────────────────────────
#
# Ground truth for "no likelihood contribution": perturb all dispersion-frozen
# coordinates jointly (random signs, two magnitudes, transformed scale) and assert
# the data log-likelihood is unchanged. Catches both coincidental zero sensitivities
# and bugs in the detection chain. Collinear-frozen parameters are excluded — they
# legitimately move the objective and are frozen for redundancy, not deadness.
function _pooled_invariance_filter(nll_recompute, θ0_vec::Vector{Float64},
        ranges, frozen::Vector{Symbol}, rng::AbstractRNG)
    isempty(frozen) && return frozen, Symbol[]
    base = nll_recompute(θ0_vec)
    isfinite(base) || return frozen, Symbol[]   # cannot verify at a non-finite start
    tol = 1e-8 * max(1.0, abs(base))
    coords = vcat([collect(ranges[n]) for n in frozen]...)
    invariant = function (idx)
        for δ in (1e-3, 0.5)
            pert = copy(θ0_vec)
            signs = ifelse.(rand(rng, length(idx)) .< 0.5, -1.0, 1.0)
            pert[idx] .+= δ .* signs
            v = nll_recompute(pert)
            (isfinite(v) && abs(v - base) <= tol) || return false
        end
        return true
    end
    invariant(coords) && return frozen, Symbol[]
    # the joint check tripped — test parameters individually
    culprits = Symbol[]
    for n in frozen
        invariant(collect(ranges[n])) || push!(culprits, n)
    end
    isempty(culprits) && (culprits = copy(frozen))  # joint-only effect: unfreeze all
    return setdiff(frozen, culprits), culprits
end

# ─── post-fit sensitivity re-check ────────────────────────────────────────────────

function _pooled_postfit_violations(dm::DataModel, θ_hat_t::ComponentArray,
        inv_transform, strategies, ranges,
        candidates::Vector{Symbol},
        frozen_dispersion::Vector{Symbol},
        frozen_collinear::Vector{Symbol})
    (isempty(frozen_dispersion) && isempty(frozen_collinear)) && return Symbol[]
    axs = getaxes(θ_hat_t)
    θhat_vec = _pooled_flat(θ_hat_t)
    cand_idx = vcat([collect(ranges[n]) for n in candidates]...)
    rel = Dict{Symbol, UnitRange{Int}}()
    off = 0
    for n in candidates
        l = length(ranges[n])
        rel[n] = (off + 1):(off + l)
        off += l
    end
    mean_stack = θu -> _pooled_stacked_means(dm, θu, strategies)
    J = try
        _pooled_coord_jacobian(mean_stack, θhat_vec, axs, inv_transform, cand_idx)
    catch
        return Symbol[]
    end
    all(isfinite, J) || return Symbol[]
    tol = _pooled_zero_tol(J)
    viol = Symbol[]
    for n in frozen_dispersion
        norm(J[:, rel[n]]) > tol && push!(viol, n)
    end
    if !isempty(frozen_collinear)
        kept = [n
                for n in candidates
                if !(n in frozen_dispersion) && !(n in frozen_collinear)]
        cols_k = isempty(kept) ? Int[] : vcat([collect(rel[n]) for n in kept]...)
        Jk = J[:, cols_k]
        for n in frozen_collinear
            _pooled_block_redundant(Jk, J[:, rel[n]]) || push!(viol, n)
        end
    end
    return viol
end

# ─── convert per-individual η to per-level DataFrames ────────────────────────────

function _pooled_re_dataframes(dm::DataModel, η_vec::Vector{<:ComponentArray};
        flatten::Bool = true)
    lp_cache = dm.re_group_info.laplace_cache
    lp_cache === nothing && return NamedTuple()
    re_names = lp_cache.re_names
    isempty(re_names) && return NamedTuple()
    re_groups = get_re_groups(dm.model.random.random)

    # (ri, lvl_id) → first individual index with that level
    level_to_ind = Dict{Tuple{Int, Int}, Int}()
    for i in 1:length(η_vec)
        for ri in 1:length(re_names)
            ids = lp_cache.ind_level_ids[i][ri]
            isempty(ids) && continue
            key = (ri, ids[1])
            haskey(level_to_ind, key) || (level_to_ind[key] = i)
        end
    end

    out_pairs = Pair{Symbol, Any}[]
    for (ri, re) in enumerate(re_names)
        col = getfield(re_groups, re)
        levels_all = lp_cache.re_index[ri].levels
        dim = lp_cache.dims[ri]

        rows = Any[]
        vals_flat = Vector{Vector{Any}}()
        for (lvl_id, lvl_val) in enumerate(levels_all)
            i = get(level_to_ind, (ri, lvl_id), 0)
            i == 0 && continue
            val = getproperty(η_vec[i], re)
            push!(rows, lvl_val)
            if flatten
                push!(vals_flat, val isa Number ? [val] : collect(vec(val)))
            else
                push!(vals_flat, [val])
            end
        end

        if flatten
            names = flatten_re_names(re, zeros(dim))
            df = DataFrame(col => rows)
            for j in 1:length(names)
                df[!, names[j]] = [vals_flat[k][j] for k in 1:length(vals_flat)]
            end
            push!(out_pairs, re => df)
        else
            push!(out_pairs,
                re => DataFrame(col => rows, :value => [v[1] for v in vals_flat]))
        end
    end
    return NamedTuple(out_pairs)
end

# ─── shared inner fit ─────────────────────────────────────────────────────────────

function _fit_pooled(dm::DataModel, method;
        constants::NamedTuple,
        penalty::NamedTuple,
        ode_args::Tuple,
        ode_kwargs::NamedTuple,
        serialization::SciMLBase.EnsembleAlgorithm,
        add_term,
        rng::AbstractRNG,
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_data_model::Bool = true,
        fit_args::Tuple = ())
    model = get_model(dm)
    fe = model.fixed.fixed
    fixed_names = get_names(fe)
    isempty(fixed_names) && error("This method requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    for name in method.force_free
        name in fixed_set || error("Unknown force_free parameter $(name).")
        haskey(constants, name) &&
            error("Parameter $(name) cannot be both in constants and force_free.")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("Pooled() has no free fixed effects to optimize.")

    θ0_u = get_θ0_untransformed(fe)
    if theta_0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta_0_untransformed, n) ||
                error("theta_0_untransformed is missing parameter $(n).")
        end
        θ0_u = theta_0_untransformed
    end

    transform = get_transform(fe)
    inv_transform = get_inverse_transform(fe)
    θ_user_u = deepcopy(θ0_u)
    _apply_constants!(θ_user_u, constants)
    θ_user_t = transform(θ_user_u)
    axs_full = getaxes(θ_user_t)
    ranges = _pooled_param_ranges(θ_user_t, fixed_names)

    re_dist_syms = _pooled_re_dist_syms(model)
    strategies = _pooled_plugin_strategies(dm, θ_user_u;
        mc_draws = method.mc_draws, rng = rng)
    strategies = _pooled_dual_safe_strategies(dm, θ_user_u, θ_user_t, inv_transform,
        strategies, method.mc_draws, rng)
    part = _pooled_partition(dm, method, θ_user_t, inv_transform, constants,
        strategies, rng)

    cache = build_ll_cache(dm; ode_args = ode_args, ode_kwargs = ode_kwargs,
        serialization = serialization, force_saveat = true)

    # data nll with η recomputed from θ — used by verification B (always the
    # recompute path, so frozen parameters are genuinely exercised through η)
    nll_recompute = function (θt_vec::Vector{Float64})
        θt = ComponentArray(θt_vec, axs_full)
        θu = inv_transform(θt)
        ηv = try
            _compute_pooled_etas(dm, θu, strategies)
        catch
            return Inf
        end
        ll = loglikelihood(dm, θu, ηv; cache = cache, serialization = serialization)
        return ll == -Inf ? Inf : -ll
    end

    frozen_dispersion, unfrozen_by_invariance = _pooled_invariance_filter(
        nll_recompute, _pooled_flat(θ_user_t), ranges,
        part.frozen_dispersion, rng)
    frozen_inert = [n for n in part.frozen_inert if n in frozen_dispersion]
    frozen_unverified = [n for n in part.frozen_unverified if n in frozen_dispersion]
    frozen_collinear = copy(part.frozen_collinear)

    # ── optimization (with optional post-fit unfreeze-and-continue rounds) ──
    max_rounds = method.refreeze_check === :refit ? 3 : 1
    θ_round_u = θ0_u
    unfrozen_postfit = Symbol[]
    postfit_violations = Symbol[]
    local sol, θ_hat_t, θ_hat_u, merged_constants
    for round in 1:max_rounds
        auto_constants = NamedTuple(n => getproperty(θ0_u, n)
        for n in vcat(frozen_dispersion, frozen_collinear))
        merged_constants = merge(auto_constants, constants)  # user constants win
        all(name in keys(merged_constants) for name in fixed_names) &&
            error("Pooled() has no free fixed effects to optimize (all are RE " *
                  "distribution parameters with no likelihood contribution). Add at " *
                  "least one fixed effect in @formulas or @DifferentialEquation.")
        sol, θ_hat_t, θ_hat_u = _pooled_solve(dm, method, θ_round_u, merged_constants,
            penalty, add_term, cache, serialization,
            strategies, re_dist_syms, fe,
            transform, inv_transform;
            info_bounds = (round == 1))
        postfit_violations = _pooled_postfit_violations(dm, θ_hat_t, inv_transform,
            strategies, ranges,
            part.candidates,
            frozen_dispersion,
            frozen_collinear)
        (isempty(postfit_violations) || method.refreeze_check === :warn) && break
        round == max_rounds && break
        # :refit — unfreeze violators and continue from the current optimum
        append!(unfrozen_postfit, postfit_violations)
        frozen_dispersion = setdiff(frozen_dispersion, postfit_violations)
        frozen_collinear = setdiff(frozen_collinear, postfit_violations)
        frozen_inert = setdiff(frozen_inert, postfit_violations)
        frozen_unverified = setdiff(frozen_unverified, postfit_violations)
        θ_round_u = θ_hat_u
    end
    if !isempty(postfit_violations) && method.refreeze_check === :warn
        @warn "Pooled: frozen parameter(s) $(join(postfit_violations, ", ")) show " *
              "nonzero plug-in sensitivity at the optimum. Consider refitting with " *
              "refreeze_check=:refit or force_free."
    end

    η_hat = _compute_pooled_etas(dm, θ_hat_u, strategies)

    re_names = dm.re_group_info.laplace_cache.re_names
    notes = (;
        plugin = NamedTuple{Tuple(re_names)}(Tuple(map(_pooled_plugin_label, strategies))),
        structural_in_re = Tuple(part.structural_in_re),
        frozen_dispersion = Tuple(frozen_dispersion),
        frozen_collinear = Tuple(frozen_collinear),
        frozen_inert = Tuple(frozen_inert),
        frozen_unverified = Tuple(frozen_unverified),
        weakly_identified = Tuple(part.weakly_identified),
        unfrozen_by_invariance = Tuple(unfrozen_by_invariance),
        unfrozen_postfit = Tuple(unfrozen_postfit),
        postfit_violations = Tuple(postfit_violations),
        probe_failed = part.probe_failed
    )

    summary = FitSummary(sol.objective, sol.retcode == SciMLBase.ReturnCode.Success,
        FitParameters(θ_hat_t, θ_hat_u), NamedTuple())
    diag = FitDiagnostics(
        (;), (optimizer = method.optimizer,), (retcode = sol.retcode,), NamedTuple())
    niter = hasproperty(sol, :stats) && hasproperty(sol.stats, :iterations) ?
            sol.stats.iterations : missing
    raw = hasproperty(sol, :original) ? sol.original : sol
    result = PooledResult(sol, sol.objective, niter, raw, notes, η_hat, strategies)
    fit_kwargs = (constants = merged_constants,
        penalty = penalty,
        ode_args = ode_args,
        ode_kwargs = ode_kwargs,
        serialization = serialization,
        rng = rng,
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model)
    return FitResult(method, result, summary, diag,
        store_data_model ? dm : nothing, fit_args, fit_kwargs)
end

# One optimization pass over the free fixed effects, with η either recomputed from θ
# at every objective evaluation (when a free parameter feeds an RE distribution) or
# precomputed once (fast path — plug-ins cannot move).
function _pooled_solve(dm::DataModel, method, θ_start_u::ComponentArray,
        merged_constants::NamedTuple, penalty::NamedTuple, add_term,
        cache, serialization, strategies, re_dist_syms, fe,
        transform, inv_transform; info_bounds::Bool = true)
    fixed_names = get_names(fe)
    free_names = [n for n in fixed_names if !(n in keys(merged_constants))]

    θ_const_u = deepcopy(θ_start_u)
    _apply_constants!(θ_const_u, merged_constants)
    θ_const_t = transform(θ_const_u)
    θ0_t = transform(θ_start_u)

    recompute_eta = any(n in re_dist_syms for n in free_names)
    η_fixed = recompute_eta ? nothing : _compute_pooled_etas(dm, θ_const_u, strategies)

    θ0_free_t = ComponentArray(NamedTuple{Tuple(free_names)}(
        Tuple(getproperty(θ0_t, n) for n in free_names)))
    axs = getaxes(θ0_free_t)

    # Positional free/constants merge (mirrors mle.jl): the per-name `setproperty!`
    # loop dispatches on runtime Symbols every objective evaluation and breaks
    # Enzyme's runtime rules on the shadow views.
    free_idx = let lab_full = ComponentArrays.labels(θ_const_t),
        lab_free = ComponentArrays.labels(θ0_free_t)

        pos_full = Dict{String, Int}(lab_full[i] => i for i in eachindex(lab_full))
        Int[pos_full[l] for l in lab_free]
    end
    θ_const_t_vec = collect(θ_const_t)
    axs_full = getaxes(θ_const_t)
    function obj(θt, p)
        v_free = θt isa ComponentArray ? ComponentArrays.getdata(θt) : θt
        T = eltype(v_free)
        infT = convert(T, Inf)
        full = Vector{T}(undef, length(θ_const_t_vec))
        @inbounds for i in eachindex(full)
            full[i] = θ_const_t_vec[i]
        end
        @inbounds for k in eachindex(free_idx)
            full[free_idx[k]] = v_free[k]
        end
        θt_full = ComponentArray(full, axs_full)
        θu = inv_transform(θt_full)
        add = add_term(θu)
        add == Inf && return infT
        η_loc = if recompute_eta
            try
                _compute_pooled_etas(dm, θu, strategies)
            catch
                return infT
            end
        else
            η_fixed
        end
        ll = loglikelihood(dm, θu, η_loc; cache = cache, serialization = serialization)
        ll == -Inf && return infT
        return -ll + _penalty_value(θu, penalty) + add
    end

    optf = OptimizationFunction(obj, method.adtype)
    lower_t, upper_t = get_bounds_transformed(fe)
    lower_t_free = _ca_subset(lower_t, free_names)
    upper_t_free = _ca_subset(upper_t, free_names)
    lower_t_free_vec = collect(lower_t_free)
    upper_t_free_vec = collect(upper_t_free)
    use_bounds = !method.ignore_model_bounds &&
                 !(all(isinf, lower_t_free_vec) && all(isinf, upper_t_free_vec))
    normalize_bound = function (bound, fallback)
        bound === nothing && return fallback
        bound isa Number && (length(fallback) == 1 ||
            error("Scalar bounds are only valid when there is one free parameter.");
        return [bound])
        if bound isa ComponentArray || bound isa NamedTuple
            b = bound isa ComponentArray ? bound : ComponentArray(bound)
            b = _ca_subset(b, free_names)
            return collect(b)
        end
        return collect(bound)
    end
    user_bounds = method.lb !== nothing || method.ub !== nothing
    if user_bounds && !isempty(keys(merged_constants)) && info_bounds
        @info "Bounds for constant parameters are ignored." constants=collect(keys(merged_constants))
    end
    lb = user_bounds ? normalize_bound(method.lb, lower_t_free_vec) : lower_t_free_vec
    ub = user_bounds ? normalize_bound(method.ub, upper_t_free_vec) : upper_t_free_vec
    use_bounds = use_bounds || user_bounds
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO && !use_bounds
        error("BlackBoxOptim requires finite bounds. Pass them via Pooled(lb=..., ub=...) " *
              "or use default_bounds_from_start(dm; margin=...).")
    end
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO &&
       !(all(isfinite, lb) && all(isfinite, ub))
        error("BlackBoxOptim requires finite lower and upper bounds for all free parameters.")
    end
    if parentmodule(typeof(method.optimizer)) === OptimizationBBO
        lb = map((u, m) -> isfinite(m) ? max(u, m) : u, collect(lb), lower_t_free_vec)
        ub = map((u, m) -> isfinite(m) ? min(u, m) : u, collect(ub), upper_t_free_vec)
        θ0_init = clamp.(collect(θ0_free_t), lb, ub)
    else
        θ0_init = θ0_free_t
    end
    prob = use_bounds ? OptimizationProblem(optf, θ0_init; lb = lb, ub = ub) :
           OptimizationProblem(optf, θ0_init)
    sol = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)

    θ_hat_t_raw = sol.u
    θ_hat_t_free = θ_hat_t_raw isa ComponentArray ?
                   θ_hat_t_raw : ComponentArray(θ_hat_t_raw, axs)
    T = eltype(θ_hat_t_free)
    θ_hat_t = ComponentArray(T.(θ_const_t), getaxes(θ_const_t))
    for name in free_names
        setproperty!(θ_hat_t, name, getproperty(θ_hat_t_free, name))
    end
    θ_hat_u = inv_transform(θ_hat_t)
    return sol, θ_hat_t, θ_hat_u
end

# ─── pooled warm-start initialization (fit_model's pooled_init option) ────────────

_default_pooled_init_method() = Pooled(; optim_kwargs = (; maxiters = 50))

# Resolve fit_model's `pooled_init` option: run a quick Pooled pre-fit and return its
# untransformed estimate as the starting point for the actual fit. The pre-fit
# inherits the shared fit keywords from the main call; `fit_options` overrides them.
# A failing pre-fit falls back to the unmodified starting point with a warning.
function _pooled_init_theta(dm::DataModel, method, pooled_init,
        fit_options::NamedTuple, kwargs_nt::NamedTuple)
    (method isa Pooled || method isa PooledMap) &&
        error("pooled_init is not supported when the fitted method is Pooled/PooledMap.")
    isempty(get_re_names(dm.model.random.random)) &&
        error("pooled_init requires a model with random effects.")
    pooled_method = if pooled_init === true
        _default_pooled_init_method()
    elseif pooled_init isa Pooled || pooled_init isa PooledMap
        pooled_init
    else
        error("pooled_init must be false, true, or a Pooled/PooledMap instance.")
    end
    base = (;
        constants = get(kwargs_nt, :constants, NamedTuple()),
        penalty = get(kwargs_nt, :penalty, NamedTuple()),
        ode_args = get(kwargs_nt, :ode_args, ()),
        ode_kwargs = get(kwargs_nt, :ode_kwargs, NamedTuple()),
        serialization = get(kwargs_nt, :serialization, EnsembleThreads()),
        theta_0_untransformed = get(kwargs_nt, :theta_0_untransformed, nothing),
        store_data_model = false
    )
    haskey(kwargs_nt, :rng) && (base = merge(base, (rng = kwargs_nt.rng,)))
    pooled_kwargs = merge(base, fit_options)
    res = try
        _fit_model(dm, pooled_method; pooled_kwargs...)
    catch err
        @warn "pooled_init pre-fit failed; starting the fit from the unmodified "*
        "initial values." exception=err
        return base.theta_0_untransformed
    end
    return get_params(res; scale = :untransformed)
end

# ─── fit_model dispatches ────────────────────────────────────────────────────────

function _fit_model(dm::DataModel, method::Pooled, args...;
        constants::NamedTuple = NamedTuple(),
        penalty::NamedTuple = NamedTuple(),
        extra_objective = nothing,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Random.default_rng(),
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_data_model::Bool = true)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && error("Pooled() requires a model with random effects. " *
          "Use MLE() for fixed-effects-only models.")
    return _fit_pooled(dm, method;
        constants = constants,
        penalty = penalty,
        ode_args = ode_args,
        ode_kwargs = ode_kwargs,
        serialization = serialization,
        add_term = _combine_add_terms(_NoOpTerm(), extra_objective),
        rng = rng,
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model,
        fit_args = args)
end

function _fit_model(dm::DataModel, method::PooledMap, args...;
        constants::NamedTuple = NamedTuple(),
        penalty::NamedTuple = NamedTuple(),
        extra_objective = nothing,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads(),
        rng::AbstractRNG = Random.default_rng(),
        theta_0_untransformed::Union{Nothing, ComponentArray} = nothing,
        store_data_model::Bool = true)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && error("PooledMap() requires a model with random effects. " *
          "Use MAP() for fixed-effects-only models.")

    fe = dm.model.fixed.fixed
    priors = get_priors(fe)
    has_prior = !isempty(keys(priors)) &&
                any(!(getfield(priors, k) isa Priorless) for k in keys(priors))
    has_prior || error("PooledMap() requires priors on fixed effects. Define priors in " *
          "@fixedEffects (e.g. RealNumber(...; prior=Normal(...))) or use Pooled() instead.")

    return _fit_pooled(dm, method;
        constants = constants,
        penalty = penalty,
        ode_args = ode_args,
        ode_kwargs = ode_kwargs,
        serialization = serialization,
        add_term = _combine_add_terms(_MAPTerm(fe), extra_objective),
        rng = rng,
        theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model,
        fit_args = args)
end
