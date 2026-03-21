export SAEM
export SAEMResult

using Optimization
using OptimizationOptimJL
using SciMLBase
using ComponentArrays
using Random
using LineSearches
using OptimizationBBO
using Turing
using Distributions
using LinearAlgebra
using ProgressMeter
import ForwardDiff

struct SAEMOptions{S, K, U, W, V, P, F1, F2, F3, B, M, R, C, RM, EO, EK, EA, EG, ER, AF}
    sampler::S
    turing_kwargs::K
    update_schedule::U
    warm_start::W
    verbose::V
    progress::P
    mcmc_steps::Int
    max_store::Int
    t0::Int
    kappa::Float64
    maxiters::Int
    rtol_theta::Float64
    atol_theta::Float64
    rtol_Q::Float64
    atol_Q::Float64
    consecutive_params::Int
    suffstats::F1
    q_from_stats::F2
    mstep_closed_form::F3
    builtin_stats::B
    builtin_mean::M
    resid_var_param::R
    re_cov_params::C
    re_mean_params::RM
    ebe_optimizer::EO
    ebe_optim_kwargs::EK
    ebe_adtype::EA
    ebe_grad_tol::EG
    ebe_multistart_n::Int
    ebe_multistart_k::Int
    ebe_multistart_max_rounds::Int
    ebe_multistart_sampling::Symbol
    ebe_rescue::ER
    anneal_to_fixed::AF
    anneal_schedule::Symbol
    anneal_min_sd::Float64
end

struct _SAEMQCache{T}
    partial_obj::Vector{T}
    batches_buf::Vector{Int}
end

function _init_saem_q_cache(::Type{T}, nbatches::Int, serialization) where {T}
    partial = Vector{T}()
    batches_buf = Vector{Int}(undef, nbatches)
    return _SAEMQCache{T}(partial, batches_buf)
end

# O2+O4+O5: ring buffer for SA sample history — avoids push!/deleteat! and Matrix wrappers
mutable struct _SAEMSampleStore
    weights::Vector{Float64}                          # [slot] weight, preallocated
    snaps::Vector{Vector{Vector{Float64}}}            # [slot][batch] re-values, preallocated
    next_idx::Int                                     # 1-based next-write slot (wraps)
    len::Int                                          # current fill count (0..capacity)
    capacity::Int
end

function _init_saem_sample_store(capacity::Int, batch_infos::Vector{_LaplaceBatchInfo})
    weights = zeros(Float64, capacity)
    snaps = [[zeros(Float64, info.n_b) for info in batch_infos] for _ in 1:capacity]
    return _SAEMSampleStore(weights, snaps, 1, 0, capacity)
end

function _saem_store_push!(store::_SAEMSampleStore, b_current, γ::Float64)
    # Scale all existing weights in-place (no allocation)
    one_minus_γ = 1.0 - γ
    @inbounds for i in 1:store.len
        store.weights[i] *= one_minus_γ
    end
    # Overwrite at next_idx (evicts oldest entry when full)
    idx = store.next_idx
    store.weights[idx] = γ
    snaps_slot = store.snaps[idx]
    @inbounds for bi in eachindex(b_current)
        snap = snaps_slot[bi]
        isempty(snap) || copyto!(snap, b_current[bi])
    end
    store.next_idx = store.next_idx == store.capacity ? 1 : store.next_idx + 1
    store.len = store.len < store.capacity ? store.len + 1 : store.capacity
    return store
end

# ── anneal_to_fixed helpers ──────────────────────────────────────────────────

function _saem_validate_anneal(anneal_to_fixed, anneal_schedule, dm)
    isempty(anneal_to_fixed) && return NamedTuple()
    anneal_schedule in (:exponential, :linear, :gamma) ||
        error("anneal_schedule must be :exponential, :linear, or :gamma. Got: $(anneal_schedule)")
    re = dm.model.random.random
    re_names_set = Set(get_re_names(re))
    re_dist_exprs = get_re_dist_exprs(re)
    names = Symbol[]
    sds   = Float64[]
    for name in anneal_to_fixed
        name isa Symbol || error("anneal_to_fixed: entries must be Symbols. Got: $name")
        name in re_names_set ||
            error("anneal_to_fixed: unknown RE name :$name. Available: $(sort(collect(re_names_set)))")
        expr = getproperty(re_dist_exprs, name)
        # expr is like Expr(:call, :Normal, mu_expr, sd_expr)
        dist_name = expr.args[1]
        dist_name == :Normal ||
            error("anneal_to_fixed: RE :$name must have a Normal distribution. " *
                  "Found: $(expr). Change to Normal(μ, σ) with a plain numeric literal SD.")
        sd_expr = expr.args[3]
        sd_expr isa Number ||
            error("anneal_to_fixed: RE :$name has SD `$sd_expr` which is not a plain numeric " *
                  "literal. `anneal_to_fixed` requires the SD to be a hardcoded number, " *
                  "e.g. Normal(a, 0.5) where 0.5 is a literal. Got: $(expr)")
        push!(names, name)
        push!(sds, Float64(sd_expr))
    end
    return NamedTuple{Tuple(names)}(Tuple(sds))
end

function _saem_anneal_sd(sd0::Float64, min_sd::Float64, iter::Int, maxiters::Int,
                          schedule::Symbol, t0::Int, kappa::Float64)
    iter == 1 && return sd0
    frac = (iter - 1) / max(1, maxiters - 1)
    result = if schedule === :exponential
        λ = log(sd0 / min_sd)
        sd0 * exp(-λ * frac)
    elseif schedule === :linear
        sd0 - (sd0 - min_sd) * frac
    else  # :gamma
        γ1 = _saem_gamma(1,    t0, kappa)
        γi = _saem_gamma(iter, t0, kappa)
        max(min_sd, sd0 * γi / γ1)
    end
    return clamp(result, min_sd, sd0)
end

@inline function _saem_anneal_normal(d::Normal, sd::Real)
    return Normal(mean(d), sd)
end

@inline function _saem_thread_caches(dm::DataModel, ll_cache, nthreads::Int)
    if ll_cache isa Vector
        return ll_cache
    elseif ll_cache isa _LLCache
        caches = build_ll_cache(dm;
                                ode_args=ll_cache.ode_args,
                                ode_kwargs=ll_cache.ode_kwargs,
                                force_saveat=ll_cache.saveat_cache !== nothing,
                                nthreads=nthreads)
        return caches isa Vector ? caches : [caches]
    else
        caches = build_ll_cache(dm; nthreads=nthreads)
        return caches isa Vector ? caches : [caches]
    end
end

@inline function _saem_thread_rngs(rng::AbstractRNG, nthreads::Int)
    return _spawn_child_rngs(rng, nthreads)
end

function _saem_batches!(buf::Vector{Int}, update_schedule, nbatches::Int, iter::Int, rng::AbstractRNG)
    if update_schedule === :all
        resize!(buf, nbatches)
        @inbounds for i in 1:nbatches
            buf[i] = i
        end
        return buf
    elseif update_schedule isa Int
        m = min(update_schedule, nbatches)
        resize!(buf, nbatches)
        @inbounds for i in 1:nbatches
            buf[i] = i
        end
        Random.randperm!(rng, buf)
        resize!(buf, m)
        return buf
    elseif update_schedule isa Function
        return update_schedule(nbatches, iter, rng)
    else
        error("Invalid update_schedule. Use :all, Int minibatch size, or a function (nbatches, iter, rng) -> Vector{Int}.")
    end
end

"""
    SAEM(; optimizer, optim_kwargs, adtype, sampler, turing_kwargs, update_schedule,
           warm_start, verbose, progress, mcmc_steps, max_store, t0, kappa, maxiters,
           rtol_theta, atol_theta, rtol_Q, atol_Q, consecutive_params, suffstats,
           q_from_stats, mstep_closed_form, builtin_stats, builtin_mean,
           resid_var_param, re_cov_params, re_mean_params, ebe_optimizer,
           ebe_optim_kwargs, ebe_adtype, ebe_grad_tol, ebe_multistart_n,
           ebe_multistart_k, ebe_multistart_max_rounds, ebe_multistart_sampling,
           ebe_rescue_on_high_grad, ebe_rescue_multistart_n, ebe_rescue_multistart_k,
           ebe_rescue_max_rounds, ebe_rescue_grad_tol, ebe_rescue_multistart_sampling,
           lb, ub) <: FittingMethod

Stochastic Approximation Expectation-Maximisation for random-effects models. SAEM
maintains a stochastic approximation of the sufficient statistics using a decreasing
step-size sequence; the M-step updates the fixed effects via gradient-based optimisation
or closed-form updates (when `builtin_stats` is enabled).

# Keyword Arguments
- `optimizer`: M-step Optimization.jl optimiser. Defaults to `LBFGS` with backtracking.
- `optim_kwargs::NamedTuple = NamedTuple()`: keyword arguments for the M-step `solve`.
- `adtype`: AD backend for the M-step. Defaults to `AutoForwardDiff()`.
- `sampler`: Turing-compatible sampler. Defaults to `NUTS(0.75)`.
- `turing_kwargs::NamedTuple = NamedTuple()`: keyword arguments for `Turing.sample`.
- `update_schedule`: which parameters to update per iteration (`:all` or a `Vector{Symbol}`).
- `warm_start::Bool = true`: initialise the sampler from the previous iteration's modes.
- `verbose::Bool = false`: print per-iteration diagnostics.
- `progress::Bool = true`: show a progress bar.
- `mcmc_steps::Int = 80`: number of MCMC steps per E-step.
- `max_store::Int = 50`: size of the sufficient statistic history window.
- `t0::Int = 20`: burn-in iterations before stochastic approximation averaging begins.
- `kappa::Float64 = 0.65`: step-size decay exponent for the Robbins-Monro schedule.
- `maxiters::Int = 300`: maximum number of SAEM iterations.
- `rtol_theta`, `atol_theta`: relative/absolute convergence tolerance on fixed effects.
- `rtol_Q`, `atol_Q`: relative/absolute convergence tolerance on the Q-function.
- `consecutive_params::Int = 4`: consecutive iterations satisfying tolerance to converge.
- `suffstats`: custom sufficient statistics function, or `nothing` to use the built-in.
- `q_from_stats`: custom Q-function from sufficient statistics, or `nothing`.
- `mstep_closed_form`: custom closed-form M-step function, or `nothing`.
- `builtin_stats`: `:auto`, `:on`, or `:off`; controls use of built-in Gaussian statistics.
- `builtin_mean`: `:none`, `:additive`, or `:all`; controls built-in mean parameterisation.
- `resid_var_param::Symbol = :σ`: fixed-effect name for the residual standard deviation.
- `re_cov_params::NamedTuple = NamedTuple()`: mapping of RE name to covariance parameter.
- `re_mean_params::NamedTuple = NamedTuple()`: mapping of RE name to mean parameter.
- `ebe_optimizer`, `ebe_optim_kwargs`, `ebe_adtype`, `ebe_grad_tol`: EBE inner optimiser.
- `ebe_multistart_n`, `ebe_multistart_k`, `ebe_multistart_max_rounds`,
  `ebe_multistart_sampling`: multistart settings for EBE mode computation.
- `ebe_rescue_*`: rescue multistart settings when an EBE mode has a high gradient norm.
- `lb`, `ub`: bounds on the transformed fixed-effect scale, or `nothing`.
"""
struct SAEM{O, K, A, SO, L, U} <: FittingMethod
    optimizer::O
    optim_kwargs::K
    adtype::A
    saem::SO
    lb::L
    ub::U
end

SAEM(; optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
     optim_kwargs=NamedTuple(),
     adtype=Optimization.AutoForwardDiff(),
     sampler=Turing.NUTS(0.75),
     turing_kwargs=NamedTuple(),
    update_schedule=:all,
    warm_start=true,
    verbose=false,
    progress=true,
    mcmc_steps=80,
     max_store=50,
     t0=20,
     kappa=0.65,
    maxiters=300,
    rtol_theta=5e-5,
    atol_theta=5e-7,
    rtol_Q=5e-5,
    atol_Q=5e-7,
    consecutive_params=4,
     suffstats=nothing,
     q_from_stats=nothing,
     mstep_closed_form=nothing,
     builtin_stats=:auto,
     builtin_mean=:none,
     resid_var_param=:σ,
     re_cov_params=NamedTuple(),
     re_mean_params=NamedTuple(),
     ebe_optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking(maxstep=1.0)),
     ebe_optim_kwargs=NamedTuple(),
     ebe_adtype=Optimization.AutoForwardDiff(),
     ebe_grad_tol=:auto,
     ebe_multistart_n=50,
     ebe_multistart_k=10,
     ebe_multistart_max_rounds=5,
     ebe_multistart_sampling=:lhs,
     ebe_rescue_on_high_grad=true,
     ebe_rescue_multistart_n=128,
     ebe_rescue_multistart_k=32,
     ebe_rescue_max_rounds=8,
     ebe_rescue_grad_tol=ebe_grad_tol,
     ebe_rescue_multistart_sampling=ebe_multistart_sampling,
     lb=nothing,
     ub=nothing,
     anneal_to_fixed=(),
     anneal_schedule=:exponential,
     anneal_min_sd=1e-5) = begin
    ebe_rescue = EBERescueOptions(ebe_rescue_on_high_grad, ebe_rescue_multistart_n, ebe_rescue_multistart_k, ebe_rescue_max_rounds, ebe_rescue_grad_tol, ebe_rescue_multistart_sampling)
    saem = SAEMOptions(sampler, turing_kwargs, update_schedule, warm_start, verbose, progress,
                       mcmc_steps, max_store, t0, kappa,
                       maxiters, rtol_theta, atol_theta, rtol_Q, atol_Q, consecutive_params,
                       suffstats, q_from_stats, mstep_closed_form,
                       builtin_stats, builtin_mean, resid_var_param, re_cov_params, re_mean_params,
                       ebe_optimizer, ebe_optim_kwargs, ebe_adtype, ebe_grad_tol,
                       ebe_multistart_n, ebe_multistart_k, ebe_multistart_max_rounds,
                       ebe_multistart_sampling,
                       ebe_rescue,
                       anneal_to_fixed, anneal_schedule, anneal_min_sd)
    SAEM(optimizer, optim_kwargs, adtype, saem, lb, ub)
end

"""
    SAEMResult{S, O, I, R, N, B} <: MethodResult

Method-specific result from a [`SAEM`](@ref) fit. Stores the solution, objective value,
iteration count, raw solver result, optional notes, and final empirical-Bayes mode
estimates for each individual.
"""
struct SAEMResult{S, O, I, R, N, B} <: MethodResult
    solution::S
    objective::O
    iterations::I
    raw::R
    notes::N
    eb_modes::B
end

function get_closed_form_mstep_used(res::SAEMResult)
    notes = res.notes
    if notes isa NamedTuple && hasproperty(notes, :closed_form_mstep_used)
        return Bool(getproperty(notes, :closed_form_mstep_used))
    end
    return false
end

mutable struct _SAEMDiagnostics{T}
    θ_hist::Vector{AbstractVector{T}}
    Q_hist::Vector{T}
    dθ_abs::Vector{T}
    dθ_rel::Vector{T}
    dQ_abs::Vector{T}
    dQ_rel::Vector{T}
    gamma::Vector{T}
    batches::Vector{Vector{Int}}
end

@inline function _saem_normalize_builtin_stats_mode(mode)
    mode == :gaussian_re && return :closed_form
    return mode
end

@inline function _saem_gamma(t::Int, t0::Int, kappa::Float64)
    t <= t0 && return 1.0
    return (t - t0)^(-kappa)
end

function _saem_stats_update(s, s_new, γ)
    s === nothing && return s_new
    if s isa NamedTuple && s_new isa NamedTuple
        keys(s) == keys(s_new) || error("sufficient statistics keys mismatch.")
        vals = map((a, b) -> a + γ * (b - a), s, s_new)
        return NamedTuple{keys(s)}(vals)
    elseif s isa AbstractArray && s_new isa AbstractArray
        return s .+ γ .* (s_new .- s)
    else
        return s + γ * (s_new - s)
    end
end

@inline function _saem_call_name(ex)
    ex isa Expr && ex.head == :call || return nothing
    fn = ex.args[1]
    if fn isa Symbol
        return fn
    elseif fn isa GlobalRef
        return fn.name
    elseif fn isa Expr && fn.head == :.
        last = fn.args[end]
        last isa QuoteNode && (last = last.value)
        return last isa Symbol ? last : nothing
    end
    return nothing
end

const _SAEM_HMM_CALL_NAMES = (
    :DiscreteTimeDiscreteStatesHMM,
    :ContinuousTimeDiscreteStatesHMM,
    :MVDiscreteTimeDiscreteStatesHMM,
    :MVContinuousTimeDiscreteStatesHMM,
)

@inline _saem_is_hmm_call_name(name) = name !== nothing && (name in _SAEM_HMM_CALL_NAMES)

const _SAEM_HMM_DISCRETE_CALL_NAMES = (
    :DiscreteTimeDiscreteStatesHMM,
    :MVDiscreteTimeDiscreteStatesHMM,
)

const _SAEM_HMM_SCALAR_EMISSION_CALL_NAMES = (
    :Bernoulli,
    :Poisson,
    :Normal,
    :LogNormal,
    :Exponential,
)

@inline _saem_is_discrete_hmm_call_name(name) = name !== nothing && (name in _SAEM_HMM_DISCRETE_CALL_NAMES)
@inline _saem_is_hmm_scalar_emission_call_name(name) = name !== nothing && (name in _SAEM_HMM_SCALAR_EMISSION_CALL_NAMES)

@inline function _saem_symbol_tuple_from_vector_expr(ex)
    ex isa Expr && ex.head == :vect || return nothing
    vals = ex.args
    all(v -> v isa Symbol, vals) || return nothing
    return tuple(Symbol.(vals)...)
end

@inline function _saem_number_vector_expr(ex)
    ex isa Expr && ex.head == :vect || return false
    return all(v -> v isa Number, ex.args)
end

@inline function _saem_formulas_assignment_map(ir)
    pairs = Pair{Symbol, Expr}[]
    for (nm, ex) in zip(ir.det_names, ir.det_exprs)
        ex isa Expr && push!(pairs, nm => ex)
    end
    return Dict{Symbol, Expr}(pairs)
end

function _saem_collect_expr_calls!(out::Vector{Expr}, ex, pred::Function)
    if ex isa Expr
        cname = _saem_call_name(ex)
        if pred(cname)
            push!(out, ex)
            return out
        end
        for a in ex.args
            _saem_collect_expr_calls!(out, a, pred)
        end
    end
    return out
end

function _saem_resolve_to_fixed_symbol(ex, assign_map::Dict{Symbol, Expr}, fixed_set::Set{Symbol};
                                       seen::Set{Symbol}=Set{Symbol}())
    if ex isa Symbol
        ex in fixed_set && return ex
        haskey(assign_map, ex) || return nothing
        ex in seen && return nothing
        push!(seen, ex)
        return _saem_resolve_to_fixed_symbol(assign_map[ex], assign_map, fixed_set; seen=seen)
    end

    ex isa Expr || return nothing
    if ex.head == :ref
        return _saem_resolve_to_fixed_symbol(ex.args[1], assign_map, fixed_set; seen=seen)
    elseif ex.head == :call
        cname = _saem_call_name(ex)
        if cname in (:reshape, :vec, :collect, :copy, :identity, :transpose, :permutedims)
            length(ex.args) >= 2 || return nothing
            return _saem_resolve_to_fixed_symbol(ex.args[2], assign_map, fixed_set; seen=seen)
        elseif cname == :getindex
            length(ex.args) >= 2 || return nothing
            return _saem_resolve_to_fixed_symbol(ex.args[2], assign_map, fixed_set; seen=seen)
        end
    end
    return nothing
end

@inline function _saem_hmm_transition_status(hmm_outcomes::Tuple, has_direct_transition_target::Bool)
    isempty(hmm_outcomes) && return :not_applicable
    return has_direct_transition_target ? :ineligible : :covered_by_re_or_external
end

function _saem_parse_hmm_emission_target_expr(em_expr, assign_map::Dict{Symbol, Expr}, fixed_set::Set{Symbol})
    ex = em_expr
    if em_expr isa Symbol && haskey(assign_map, em_expr)
        ex = assign_map[em_expr]
    end

    calls = Expr[]
    _saem_collect_expr_calls!(calls, ex, _saem_is_hmm_scalar_emission_call_name)
    isempty(calls) && return nothing

    family = nothing
    target = nothing
    for c in calls
        cname = _saem_call_name(c)
        cname == :Bernoulli || return nothing
        length(c.args) == 2 || return nothing
        t = _saem_resolve_to_fixed_symbol(c.args[2], assign_map, fixed_set)
        t === nothing && return nothing
        if family === nothing
            family = :bernoulli
            target = t
        else
            family == :bernoulli || return nothing
            target == t || return nothing
        end
    end
    return (family=:bernoulli, target=target)
end

function _saem_autodetect_hmm_emission_params(dm::DataModel, fixed_set::Set{Symbol})
    ir = get_formulas_ir(dm.model.formulas.formulas)
    assign_map = _saem_formulas_assignment_map(ir)
    pairs = Pair{Symbol, Any}[]

    for (obs, ex) in zip(ir.obs_names, ir.obs_exprs)
        cname = _saem_call_name(ex)
        _saem_is_hmm_call_name(cname) || continue
        ex isa Expr || continue
        length(ex.args) >= 4 || continue
        em_expr = ex.args[3]
        parsed = _saem_parse_hmm_emission_target_expr(em_expr, assign_map, fixed_set)
        parsed === nothing && continue
        hmm_type = _saem_is_discrete_hmm_call_name(cname) ? :discrete_time : :continuous_time
        push!(pairs, obs => (family=parsed.family, target=parsed.target, hmm_type=hmm_type))
    end
    return NamedTuple(pairs)
end

function _saem_hmm_has_direct_transition_target(dm::DataModel, fixed_set::Set{Symbol})
    ir = get_formulas_ir(dm.model.formulas.formulas)
    assign_map = _saem_formulas_assignment_map(ir)
    for ex in ir.obs_exprs
        cname = _saem_call_name(ex)
        _saem_is_hmm_call_name(cname) || continue
        ex isa Expr || continue
        length(ex.args) >= 3 || continue
        trans_expr = ex.args[2]
        t = _saem_resolve_to_fixed_symbol(trans_expr, assign_map, fixed_set)
        t === nothing || return true
    end
    return false
end

@inline function _saem_hmm_emission_mapping_supported(info)
    hasproperty(info, :family) || return false
    hasproperty(info, :target) || return false
    hasproperty(info, :hmm_type) || return false
    return info.family == :bernoulli && info.hmm_type == :discrete_time
end

function _saem_parse_re_gaussian_mapping(dist_expr, fixed_set::Set{Symbol})
    cname = _saem_call_name(dist_expr)
    cname === nothing && return nothing

    if cname == :Normal
        (dist_expr isa Expr && length(dist_expr.args) == 3) || return nothing
        μarg = dist_expr.args[2]
        σarg = dist_expr.args[3]
        σ_target = (σarg isa Symbol && σarg in fixed_set) ? σarg : nothing
        mean_target = (μarg isa Symbol && μarg in fixed_set) ? μarg : nothing
        return (family=:normal, mean=mean_target, cov=σ_target)
    end

    if cname == :LogNormal
        (dist_expr isa Expr && length(dist_expr.args) == 3) || return nothing
        μarg = dist_expr.args[2]
        σarg = dist_expr.args[3]
        mean_target = (μarg isa Symbol && μarg in fixed_set) ? μarg : nothing
        cov_target = (σarg isa Symbol && σarg in fixed_set) ? σarg : nothing
        return (family=:lognormal, mean=mean_target, cov=cov_target)
    end

    if cname == :Exponential
        (dist_expr isa Expr && length(dist_expr.args) == 2) || return nothing
        θarg = dist_expr.args[2]
        cov_target = (θarg isa Symbol && θarg in fixed_set) ? θarg : nothing
        return (family=:exponential, mean=nothing, cov=cov_target)
    end

    if cname == :MvNormal
        (dist_expr isa Expr && length(dist_expr.args) == 3) || return nothing
        μarg = dist_expr.args[2]
        Σarg = dist_expr.args[3]

        mean_target = nothing
        if μarg isa Symbol
            μarg in fixed_set && (mean_target = μarg)
        else
            μ_tuple = _saem_symbol_tuple_from_vector_expr(μarg)
            if μ_tuple !== nothing
                all(s -> s in fixed_set, μ_tuple) && (mean_target = μ_tuple)
            elseif _saem_number_vector_expr(μarg)
                mean_target = nothing
            end
        end

        cov_target = nothing
        if Σarg isa Symbol
            Σarg in fixed_set && (cov_target = Σarg)
        else
            cname_cov = _saem_call_name(Σarg)
            if cname_cov == :Diagonal && Σarg isa Expr && length(Σarg.args) == 2
                darg = Σarg.args[2]
                if darg isa Symbol
                    darg in fixed_set && (cov_target = darg)
                else
                    d_tuple = _saem_symbol_tuple_from_vector_expr(darg)
                    if d_tuple !== nothing
                        all(s -> s in fixed_set, d_tuple) && (cov_target = d_tuple)
                    end
                end
            end
        end
        return (family=:mvnormal, mean=mean_target, cov=cov_target)
    end

    return nothing
end

function _saem_parse_outcome_builtin_target(dist_expr, fixed_set::Set{Symbol})
    cname = _saem_call_name(dist_expr)
    cname === nothing && return nothing
    dist_expr isa Expr || return nothing

    if cname == :Normal
        length(dist_expr.args) == 3 || return nothing
        σarg = dist_expr.args[3]
        return (σarg isa Symbol && σarg in fixed_set) ? σarg : nothing
    elseif cname == :LogNormal
        length(dist_expr.args) == 3 || return nothing
        μarg = dist_expr.args[2]
        σarg = dist_expr.args[3]
        μ_target = (μarg isa Symbol && μarg in fixed_set) ? μarg : nothing
        σ_target = (σarg isa Symbol && σarg in fixed_set) ? σarg : nothing

        # For closed-form LogNormal updates, if only σ is free we require μ to be scalar or a fixed symbol.
        if μ_target === nothing && σ_target !== nothing
            ((μarg isa Number) || (μarg isa Symbol && μarg in fixed_set)) || return nothing
            return σ_target
        end
        if μ_target !== nothing && σ_target !== nothing
            return (; μ=μ_target, σ=σ_target)
        elseif μ_target !== nothing
            return (; μ=μ_target)
        else
            return nothing
        end
    elseif cname == :Exponential
        length(dist_expr.args) == 2 || return nothing
        θarg = dist_expr.args[2]
        return (θarg isa Symbol && θarg in fixed_set) ? θarg : nothing
    elseif cname == :Bernoulli
        length(dist_expr.args) == 2 || return nothing
        parg = dist_expr.args[2]
        return (parg isa Symbol && parg in fixed_set) ? parg : nothing
    elseif cname == :Poisson
        length(dist_expr.args) == 2 || return nothing
        λarg = dist_expr.args[2]
        return (λarg isa Symbol && λarg in fixed_set) ? λarg : nothing
    end
    return nothing
end

function _saem_autodetect_resid_var_param(dm::DataModel, fixed_set::Set{Symbol})
    ir = get_formulas_ir(dm.model.formulas.formulas)
    obs_names = ir.obs_names
    obs_exprs = ir.obs_exprs
    pairs = Pair{Symbol, Any}[]

    for (obs, ex) in zip(obs_names, obs_exprs)
        target = _saem_parse_outcome_builtin_target(ex, fixed_set)
        target === nothing && continue
        push!(pairs, obs => target)
    end

    isempty(pairs) && return NamedTuple()
    if length(pairs) == length(obs_names)
        vals = last.(pairs)
        if all(v -> v isa Symbol, vals)
            unique_syms = unique(Symbol.(vals))
            length(unique_syms) == 1 && return unique_syms[1]
        end
    end
    return NamedTuple(pairs)
end

function _saem_hmm_outcome_names(dm::DataModel)
    ir = get_formulas_ir(dm.model.formulas.formulas)
    hmm = Symbol[]
    for (obs, ex) in zip(ir.obs_names, ir.obs_exprs)
        _saem_is_hmm_call_name(_saem_call_name(ex)) && push!(hmm, obs)
    end
    return hmm
end

function _saem_autodetect_gaussian_re(dm::DataModel, fixed_names::Vector{Symbol})
    re_model = dm.model.random.random
    re_names = get_re_names(re_model)
    isempty(re_names) && return nothing
    fixed_set = Set(fixed_names)
    re_dists = get_re_dist_exprs(re_model)

    cov_pairs = Pair{Symbol, Any}[]
    mean_pairs = Pair{Symbol, Any}[]
    family_pairs = Pair{Symbol, Symbol}[]
    for re in re_names
        hasproperty(re_dists, re) || continue
        mapping = _saem_parse_re_gaussian_mapping(getproperty(re_dists, re), fixed_set)
        mapping === nothing && continue
        push!(family_pairs, re => mapping.family)
        mapping.cov === nothing || push!(cov_pairs, re => mapping.cov)
        mapping.mean === nothing || push!(mean_pairs, re => mapping.mean)
    end

    resid_var_param = _saem_autodetect_resid_var_param(dm, fixed_set)
    has_outcome_updates = resid_var_param isa NamedTuple ? !isempty(keys(resid_var_param)) : true
    hmm_emission_params = _saem_autodetect_hmm_emission_params(dm, fixed_set)
    has_hmm_emission_updates = !isempty(keys(hmm_emission_params))
    isempty(cov_pairs) && isempty(mean_pairs) && !has_outcome_updates && !has_hmm_emission_updates && return nothing
    hmm_outcomes = Tuple(_saem_hmm_outcome_names(dm))
    has_direct_transition_target = _saem_hmm_has_direct_transition_target(dm, fixed_set)
    hmm_emission_all_supported = !isempty(hmm_outcomes) &&
                                 all(col -> haskey(hmm_emission_params, col) &&
                                            _saem_hmm_emission_mapping_supported(getfield(hmm_emission_params, col)),
                                     hmm_outcomes)
    hmm_emission_status = isempty(hmm_outcomes) ? :not_applicable :
                          (hmm_emission_all_supported ? :eligible : :ineligible)
    return (re_cov_params=NamedTuple(cov_pairs),
            re_mean_params=NamedTuple(mean_pairs),
            re_families=NamedTuple(family_pairs),
            resid_var_param=resid_var_param,
            hmm_emission_params=hmm_emission_params,
            hmm_outcomes=hmm_outcomes,
            hmm_transition_closed_form=_saem_hmm_transition_status(hmm_outcomes, has_direct_transition_target),
            hmm_emission_closed_form=hmm_emission_status)
end

function _saem_builtin_closed_form_eligibility(dm::DataModel,
                                               re_cov_params::NamedTuple,
                                               re_mean_params::NamedTuple,
                                               resid_var_param,
                                               hmm_emission_params::NamedTuple=NamedTuple())
    re_targets = _saem_builtin_re_targets(re_cov_params, re_mean_params)
    obs_targets = _saem_outcome_targets(dm.config.obs_cols, resid_var_param)
    all_outcome_targets = Symbol[collect(keys(obs_targets))...]
    hmm_outcomes_vec = _saem_hmm_outcome_names(dm)
    hmm_set = Set{Symbol}(hmm_outcomes_vec)

    outcome_targets_hmm = Symbol[]
    outcome_targets_non_hmm = Symbol[]
    for col in all_outcome_targets
        if col in hmm_set
            push!(outcome_targets_hmm, col)
        else
            push!(outcome_targets_non_hmm, col)
        end
    end

    reasons = Symbol[]
    !isempty(outcome_targets_hmm) && push!(reasons, :hmm_outcome_targets_ignored)
    hmm_emission_targets_supported = Symbol[]
    hmm_emission_targets_unsupported = Symbol[]
    for col in hmm_outcomes_vec
        if haskey(hmm_emission_params, col) &&
           _saem_hmm_emission_mapping_supported(getfield(hmm_emission_params, col))
            push!(hmm_emission_targets_supported, col)
        else
            push!(hmm_emission_targets_unsupported, col)
        end
    end
    !isempty(hmm_outcomes_vec) && isempty(hmm_emission_targets_supported) &&
        push!(reasons, :hmm_latent_state_suffstats_not_available_builtin)
    !isempty(hmm_emission_targets_unsupported) &&
        push!(reasons, :hmm_emission_mapping_not_supported_builtin)
    has_direct_transition_target = _saem_hmm_has_direct_transition_target(dm, Set(get_names(dm.model.fixed.fixed)))
    has_direct_transition_target &&
        push!(reasons, :hmm_transition_mapping_not_supported_builtin)

    re_block_eligible = !isempty(re_targets)
    outcome_block_eligible = !isempty(outcome_targets_non_hmm)
    hmm_emission_block_eligible = !isempty(hmm_emission_targets_supported)
    has_any_closed_form_block = re_block_eligible || outcome_block_eligible || hmm_emission_block_eligible
    hmm_outcomes = Tuple(hmm_outcomes_vec)
    hmm_emission_status = isempty(hmm_outcomes) ? :not_applicable :
                          (isempty(hmm_emission_targets_unsupported) && !isempty(hmm_emission_targets_supported) ?
                           :eligible :
                           (!isempty(hmm_emission_targets_supported) ? :partial : :ineligible))

    return (re_block_eligible=re_block_eligible,
            outcome_block_eligible=outcome_block_eligible,
            hmm_emission_block_eligible=hmm_emission_block_eligible,
            has_any_closed_form_block=has_any_closed_form_block,
            re_targets=Tuple(re_targets),
            outcome_targets_non_hmm=Tuple(outcome_targets_non_hmm),
            outcome_targets_hmm=Tuple(outcome_targets_hmm),
            hmm_emission_targets=Tuple(hmm_emission_targets_supported),
            hmm_emission_targets_unsupported=Tuple(hmm_emission_targets_unsupported),
            hmm_emission_params=hmm_emission_params,
            hmm_outcomes=hmm_outcomes,
            hmm_transition_closed_form=_saem_hmm_transition_status(hmm_outcomes, has_direct_transition_target),
            hmm_emission_closed_form=hmm_emission_status,
            reasons=Tuple(reasons))
end

function _saem_prune_hmm_outcome_targets(dm::DataModel, resid_var_param, hmm_outcomes)
    isempty(hmm_outcomes) && return resid_var_param
    obs_targets = _saem_outcome_targets(dm.config.obs_cols, resid_var_param)
    isempty(keys(obs_targets)) && return NamedTuple()

    hmm_set = Set{Symbol}(Symbol.(collect(hmm_outcomes)))
    kept = Pair{Symbol, Any}[]
    for (col, target) in pairs(obs_targets)
        col in hmm_set && continue
        push!(kept, col => target)
    end

    if isempty(kept)
        return NamedTuple()
    end
    if resid_var_param isa Symbol && length(kept) == length(dm.config.obs_cols)
        return resid_var_param
    end
    return NamedTuple(kept)
end

function _saem_re_family_map(dm::DataModel)
    re_names = get_re_names(dm.model.random.random)
    re_dists = get_re_dist_exprs(dm.model.random.random)
    pairs = Pair{Symbol, Symbol}[]
    for re in re_names
        family = :unsupported
        if hasproperty(re_dists, re)
            cname = _saem_call_name(getproperty(re_dists, re))
            if cname == :Normal
                family = :normal
            elseif cname == :MvNormal
                family = :mvnormal
            elseif cname == :LogNormal
                family = :lognormal
            elseif cname == :Exponential
                family = :exponential
            end
        end
        push!(pairs, re => family)
    end
    return NamedTuple(pairs)
end

@inline function _saem_push_param_updates!(updates::Vector{Pair{Symbol, Any}},
                                           θ::ComponentArray,
                                           target,
                                           value;
                                           context::AbstractString)
    if target isa Symbol
        if hasproperty(θ, target)
            θ_target = getproperty(θ, target)
            val = value
            if θ_target isa Number && !(value isa Number)
                vals = value isa Tuple ? value : Tuple(value)
                length(vals) == 1 || error("$(context) expects a scalar value for $(target).")
                val = vals[1]
            end
            push!(updates, target => val)
        end
        return nothing
    end
    if target isa Tuple || target isa AbstractVector
        vals = value isa Number ? (value,) : Tuple(value)
        length(target) == length(vals) || error("$(context) expects $(length(target)) values but got $(length(vals)).")
        for (name, val) in zip(target, vals)
            name isa Symbol || error("$(context) must contain Symbols.")
            hasproperty(θ, name) && push!(updates, name => val)
        end
        return nothing
    end
    error("$(context) must be a Symbol or a tuple/vector of Symbols.")
end

@inline _saem_blend(a::Number, b::Number, γ::Real) = a + γ * (b - a)
@inline _saem_blend(a::AbstractArray, b::AbstractArray, γ::Real) = a .+ γ .* (b .- a)

@inline function _saem_builtin_re_targets(re_cov_params::NamedTuple, re_mean_params::NamedTuple)
    targets = Symbol[]
    append!(targets, collect(keys(re_cov_params)))
    for re in keys(re_mean_params)
        re in targets || push!(targets, re)
    end
    return targets
end

@inline function _saem_outcome_family(dist)
    if dist isa Normal
        return :normal
    elseif dist isa LogNormal
        return :lognormal
    elseif dist isa Exponential
        return :exponential
    elseif dist isa Bernoulli
        return :bernoulli
    elseif dist isa Poisson
        return :poisson
    else
        return :unsupported
    end
end

@inline function _saem_outcome_targets(obs_cols, resid_var_param)
    if resid_var_param isa NamedTuple
        return resid_var_param
    elseif resid_var_param isa Symbol
        pairs = Pair{Symbol, Any}[]
        for col in obs_cols
            push!(pairs, col => resid_var_param)
        end
        return NamedTuple(pairs)
    else
        return NamedTuple()
    end
end

@inline function _saem_is_hmm_distribution(dist)
    return (dist isa DiscreteTimeDiscreteStatesHMM) ||
           (dist isa ContinuousTimeDiscreteStatesHMM) ||
           (dist isa MVDiscreteTimeDiscreteStatesHMM) ||
           (dist isa MVContinuousTimeDiscreteStatesHMM)
end

@inline function _saem_hmm_state_emission_probs(dist::DiscreteTimeDiscreteStatesHMM, y, T::Type)
    probs = Vector{T}(undef, dist.n_states)
    if ismissing(y)
        fill!(probs, one(T))
        return probs
    end
    for k in 1:dist.n_states
        v = _fast_logpdf(dist.emission_dists[k], y)
        v === nothing && (v = logpdf(dist.emission_dists[k], y))
        probs[k] = isfinite(v) ? exp(T(v)) : zero(T)
    end
    return probs
end

@inline function _saem_hmm_state_emission_probs(dist::MVDiscreteTimeDiscreteStatesHMM, y, T::Type)
    probs = Vector{T}(undef, dist.n_states)
    if ismissing(y)
        fill!(probs, one(T))
        return probs
    end
    y isa AbstractVector || return nothing
    for k in 1:dist.n_states
        v = _mv_emission_logpdf(dist.emission_dists[k], y)
        probs[k] = isfinite(v) ? exp(T(v)) : zero(T)
    end
    return probs
end

@inline _saem_hmm_state_emission_probs(::Any, ::Any, ::Type) = nothing

@inline _saem_hmm_transition_matrix(dist::DiscreteTimeDiscreteStatesHMM, T::Type) =
    T.(dist.transition_matrix)

@inline _saem_hmm_transition_matrix(dist::MVDiscreteTimeDiscreteStatesHMM, T::Type) =
    T.(dist.transition_matrix)

function _saem_hmm_smoothed_gamma(dists::Vector, ys::Vector, T::Type)
    n_time = length(dists)
    n_time == length(ys) || return (nothing, false)
    n_time == 0 && return (nothing, false)

    first_dist = dists[1]
    (first_dist isa DiscreteTimeDiscreteStatesHMM || first_dist isa MVDiscreteTimeDiscreteStatesHMM) ||
        return (nothing, false)

    K = first_dist.n_states
    alpha = zeros(T, K, n_time)
    beta = ones(T, K, n_time)
    emissions = zeros(T, K, n_time)
    scales = zeros(T, n_time)
    prior = nothing
    for t in 1:n_time
        dist_t = dists[t]
        typeof(dist_t) === typeof(first_dist) || return (nothing, false)
        dist_use = _hmm_with_prior(dist_t, prior)
        b = _saem_hmm_state_emission_probs(dist_use, ys[t], T)
        b === nothing && return (nothing, false)
        pred = T.(probabilities_hidden_states(dist_use))
        unnorm = pred .* b
        c = sum(unnorm)
        (isfinite(c) && c > zero(T)) || return (nothing, false)
        alpha[:, t] .= unnorm ./ c
        emissions[:, t] .= b
        scales[t] = c
        prior = alpha[:, t]
    end

    for t in (n_time - 1):-1:1
        trans_next = _saem_hmm_transition_matrix(dists[t + 1], T)
        beta[:, t] .= trans_next * (emissions[:, t + 1] .* beta[:, t + 1])
        c_next = scales[t + 1]
        (isfinite(c_next) && c_next > zero(T)) || return (nothing, false)
        beta[:, t] ./= c_next
    end

    gamma = alpha .* beta
    for t in 1:n_time
        s = sum(gamma[:, t])
        (isfinite(s) && s > zero(T)) || return (nothing, false)
        gamma[:, t] ./= s
    end
    return (gamma, true)
end

function _saem_hmm_bernoulli_stats_from_sequence(dists::Vector, ys::Vector, target, T::Type)
    gamma, ok = _saem_hmm_smoothed_gamma(dists, ys, T)
    ok || return (nothing, false)
    first_dist = dists[1]
    K = first_dist.n_states

    if first_dist isa DiscreteTimeDiscreteStatesHMM
        all(d -> d isa DiscreteTimeDiscreteStatesHMM, dists) || return (nothing, false)
        all(k -> first_dist.emission_dists[k] isa Bernoulli, 1:K) || return (nothing, false)
        sum_w = zeros(T, K, 1)
        sum_wy = zeros(T, K, 1)
        for t in eachindex(ys)
            y = ys[t]
            if ismissing(y)
                continue
            end
            yy = T(y)
            (yy == zero(T) || yy == one(T)) || return (nothing, false)
            for k in 1:K
                w = gamma[k, t]
                sum_w[k, 1] += w
                sum_wy[k, 1] += w * yy
            end
        end
        return ((family=:hmm_bernoulli, target=target, sum_w=sum_w, sum_wy=sum_wy), true)
    end

    first_dist isa MVDiscreteTimeDiscreteStatesHMM || return (nothing, false)
    all(d -> d isa MVDiscreteTimeDiscreteStatesHMM, dists) || return (nothing, false)
    all(k -> first_dist.emission_dists[k] isa Tuple, 1:K) || return (nothing, false)
    M = length(first_dist.emission_dists[1])
    all(k -> length(first_dist.emission_dists[k]) == M, 1:K) || return (nothing, false)
    all(k -> all(m -> first_dist.emission_dists[k][m] isa Bernoulli, 1:M), 1:K) || return (nothing, false)

    sum_w = zeros(T, K, M)
    sum_wy = zeros(T, K, M)
    for t in eachindex(ys)
        y = ys[t]
        if ismissing(y)
            continue
        end
        y isa AbstractVector || return (nothing, false)
        length(y) == M || return (nothing, false)
        for m in 1:M
            ym = y[m]
            ismissing(ym) && continue
            yy = T(ym)
            (yy == zero(T) || yy == one(T)) || return (nothing, false)
            for k in 1:K
                w = gamma[k, t]
                sum_w[k, m] += w
                sum_wy[k, m] += w * yy
            end
        end
    end

    return ((family=:hmm_bernoulli, target=target, sum_w=sum_w, sum_wy=sum_wy), true)
end

@inline function _saem_merge_hmm_stats(a, b)
    (a.family == b.family && a.target == b.target) || return nothing
    return (family=a.family,
            target=a.target,
            sum_w=a.sum_w + b.sum_w,
            sum_wy=a.sum_wy + b.sum_wy)
end

function _saem_collect_hmm_stats_individual(dm::DataModel,
                                            idx::Int,
                                            θ,
                                            η_ind,
                                            cache::_LLCache,
                                            hmm_emission_params::NamedTuple)
    isempty(keys(hmm_emission_params)) && return (NamedTuple(), true)

    model = dm.model
    ind = dm.individuals[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    const_cov = ind.const_cov
    obs_series = ind.series.obs
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]

    sol_accessors = nothing
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η_ind, const_cov)
        pc = (;
            fixed_effects = θ,
            random_effects = η_ind,
            constant_covariates = const_cov,
            varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
            helpers = cache.helpers,
            model_funs = cache.model_funs,
            preDE = pre
        )
        compiled = get_de_compiler(model.de.de)(pc)
        u0 = calculate_initial_state(model, θ, η_ind, const_cov)
        cb = nothing
        infusion_rates = nothing
        if ind.callbacks !== nothing
            _apply_initial_events!(u0, ind.callbacks)
            cb = ind.callbacks.callback
            infusion_rates = ind.callbacks.infusion_rates
        end
        f!_use = _with_infusion(get_de_f!(model.de.de), infusion_rates)
        prob = cache.prob_templates === nothing ? nothing : cache.prob_templates[idx]
        if prob === nothing
            prob = ODEProblem{true, SciMLBase.FullSpecialize}(f!_use, u0, ind.tspan, compiled)
            if cache.prob_templates !== nothing
                cache.prob_templates[idx] = prob
            end
        end

        Tprob = promote_type(eltype(θ), eltype(η_ind), eltype(u0))
        prob = remake(prob; u0 = Tprob.(u0), p = compiled)
        saveat_use = _ll_saveat(cache, idx, ind)
        sol = if saveat_use === nothing
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs,
                                             cache.ode_kwargs,
                                             (dense=true,))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        else
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs,
                                             cache.ode_kwargs,
                                             (saveat=saveat_use, save_everystep=false, dense=false))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        end
        SciMLBase.successful_retcode(sol) || return (NamedTuple(), false)
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
    end

    seq_dist = Dict{Symbol, Vector{Any}}()
    seq_y = Dict{Symbol, Vector{Any}}()
    for col in keys(hmm_emission_params)
        seq_dist[col] = Any[]
        seq_y[col] = Any[]
    end

    time_col = _get_col(dm.df, dm.config.time_col)[obs_rows]
    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only=true)
    for i in eachindex(obs_rows)
        vary = vary_cache === nothing ? _varying_at(dm, ind, i, time_col) : vary_cache[i]
        η_row = _row_random_effects_at(dm, idx, i, η_ind, rowwise_re; obs_only=true)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
              calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        for col in keys(hmm_emission_params)
            dist = getproperty(obs, col)
            _saem_is_hmm_distribution(dist) || return (NamedTuple(), false)
            push!(seq_dist[col], dist)
            push!(seq_y[col], getfield(obs_series, col)[i])
        end
    end

    Tstats = promote_type(eltype(θ), Float64)
    pairs = Pair{Symbol, Any}[]
    for col in keys(hmm_emission_params)
        info = getfield(hmm_emission_params, col)
        _saem_hmm_emission_mapping_supported(info) || return (NamedTuple(), false)
        st, ok = _saem_hmm_bernoulli_stats_from_sequence(seq_dist[col], seq_y[col], info.target, Tstats)
        ok || return (NamedTuple(), false)
        push!(pairs, col => st)
    end
    return (NamedTuple(pairs), true)
end

function _saem_push_outcome_stat(prev, dist, y, T::Type)
    ismissing(y) && return prev
    family = _saem_outcome_family(dist)
    family == :unsupported && return nothing
    yy = T(y)

    if family == :normal
        resid = yy - T(dist.μ)
        s1 = resid * resid
        s2 = zero(T)
        ss = s1
    elseif family == :lognormal
        yy > zero(T) || return nothing
        ly = log(yy)
        μ = T(dist.μ)
        r = ly - μ
        s1 = ly
        s2 = ly * ly
        ss = r * r
    elseif family == :exponential
        (isfinite(yy) && yy >= zero(T)) || return nothing
        s1 = yy
        s2 = zero(T)
        ss = zero(T)
    elseif family == :bernoulli
        (yy == zero(T) || yy == one(T)) || return nothing
        s1 = yy
        s2 = zero(T)
        ss = zero(T)
    else # :poisson
        (isfinite(yy) && yy >= zero(T)) || return nothing
        s1 = yy
        s2 = zero(T)
        ss = zero(T)
    end

    if prev === nothing
        return (family=family, s1=s1, s2=s2, ss=ss, n=1)
    end
    prev.family == family || return nothing
    return (family=family,
            s1=prev.s1 + s1,
            s2=prev.s2 + s2,
            ss=prev.ss + ss,
            n=prev.n + 1)
end

@inline function _saem_merge_outcome_stats(a, b)
    a.family == b.family || return nothing
    return (family=a.family,
            s1=a.s1 + b.s1,
            s2=a.s2 + b.s2,
            ss=a.ss + b.ss,
            n=a.n + b.n)
end

function _saem_collect_outcome_stats_individual(dm::DataModel,
                                                idx::Int,
                                                θ,
                                                η_ind,
                                                cache::_LLCache,
                                                obs_targets::NamedTuple)
    isempty(keys(obs_targets)) && return (NamedTuple(), true)

    model = dm.model
    ind = dm.individuals[idx]
    obs_rows = dm.row_groups.obs_rows[idx]
    const_cov = ind.const_cov
    obs_series = ind.series.obs
    vary_cache = cache.vary_cache === nothing ? nothing : cache.vary_cache[idx]

    sol_accessors = nothing
    if model.de.de !== nothing
        pre = calculate_prede(model, θ, η_ind, const_cov)
        pc = (;
            fixed_effects = θ,
            random_effects = η_ind,
            constant_covariates = const_cov,
            varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
            helpers = cache.helpers,
            model_funs = cache.model_funs,
            preDE = pre
        )
        compiled = get_de_compiler(model.de.de)(pc)
        u0 = calculate_initial_state(model, θ, η_ind, const_cov)
        cb = nothing
        infusion_rates = nothing
        if ind.callbacks !== nothing
            _apply_initial_events!(u0, ind.callbacks)
            cb = ind.callbacks.callback
            infusion_rates = ind.callbacks.infusion_rates
        end
        f!_use = _with_infusion(get_de_f!(model.de.de), infusion_rates)
        prob = cache.prob_templates === nothing ? nothing : cache.prob_templates[idx]
        if prob === nothing
            prob = ODEProblem{true, SciMLBase.FullSpecialize}(f!_use, u0, ind.tspan, compiled)
            if cache.prob_templates !== nothing
                cache.prob_templates[idx] = prob
            end
        end

        Tprob = promote_type(eltype(θ), eltype(η_ind), eltype(u0))
        prob = remake(prob; u0 = Tprob.(u0), p = compiled)
        saveat_use = _ll_saveat(cache, idx, ind)
        sol = if saveat_use === nothing
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs,
                                             cache.ode_kwargs,
                                             (dense=true,))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        else
            solve_kwargs = _ode_solve_kwargs(cache.solver_cfg.kwargs,
                                             cache.ode_kwargs,
                                             (saveat=saveat_use, save_everystep=false, dense=false))
            cb === nothing ?
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs...) :
                solve(prob, cache.alg, cache.ode_args...; solve_kwargs..., callback=cb)
        end
        SciMLBase.successful_retcode(sol) || return (NamedTuple(), false)
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
    end

    Tstats = promote_type(eltype(θ), Float64)
    obs_cols = dm.config.obs_cols
    time_col = _get_col(dm.df, dm.config.time_col)[obs_rows]
    stats_dict = Dict{Symbol, Any}()
    rowwise_re = _needs_rowwise_random_effects(dm, idx; obs_only=true)
    for i in eachindex(obs_rows)
        vary = vary_cache === nothing ? _varying_at(dm, ind, i, time_col) : vary_cache[i]
        η_row = _row_random_effects_at(dm, idx, i, η_ind, rowwise_re; obs_only=true)
        obs = sol_accessors === nothing ?
              calculate_formulas_obs(model, θ, η_row, const_cov, vary) :
              calculate_formulas_obs(model, θ, η_row, const_cov, vary, sol_accessors)
        for col in obs_cols
            haskey(obs_targets, col) || continue
            dist = getproperty(obs, col)
            _saem_outcome_family(dist) == :unsupported && continue
            y = getfield(obs_series, col)[i]
            ismissing(y) && continue
            prev = get(stats_dict, col, nothing)
            nxt = _saem_push_outcome_stat(prev, dist, y, Tstats)
            nxt === nothing && return (NamedTuple(), false)
            stats_dict[col] = nxt
        end
    end

    pairs = Pair{Symbol, Any}[]
    for col in obs_cols
        haskey(stats_dict, col) && push!(pairs, col => stats_dict[col])
    end
    return (NamedTuple(pairs), true)
end

function _saem_builtin_collect_current_stats(dm::DataModel,
                                             batch_infos::Vector{_LaplaceBatchInfo},
                                             b_current::AbstractVector,
                                             θ::ComponentArray,
                                             const_cache::LaplaceConstantsCache,
                                             resid_var_param,
                                             hmm_emission_params::NamedTuple,
                                             re_cov_params::NamedTuple,
                                             re_mean_params::NamedTuple,
                                             re_family_map::NamedTuple,
                                             ll_cache::_LLCache)
    Tθ = promote_type(eltype(θ), Float64)
    cache = dm.re_group_info.laplace_cache
    re_names = cache.re_names
    re_pairs = Pair{Symbol, Any}[]
    for re in _saem_builtin_re_targets(re_cov_params, re_mean_params)
        ri = findfirst(==(re), re_names)
        ri === nothing && continue

        family = haskey(re_family_map, re) ? getfield(re_family_map, re) : :normal
        sum_x = nothing
        sum_xx = nothing
        nvals = 0
        for (bi, info) in enumerate(batch_infos)
            b = b_current[bi]
            rei = info.re_info[ri]
            for lvl_id in rei.map.levels
                v = _re_value_from_b(rei, lvl_id, b)
                v === nothing && continue
                raw = v isa Number ? Tθ[v] : Tθ.(collect(v))
                x = if family == :lognormal
                    all(raw .> zero(Tθ)) || continue
                    log.(raw)
                elseif family == :normal || family == :mvnormal || family == :exponential
                    raw
                else
                    continue
                end
                if sum_x === nothing
                    dim = length(x)
                    sum_x = zeros(Tθ, dim)
                    sum_xx = zeros(Tθ, dim, dim)
                end
                sum_x .+= x
                sum_xx .+= x * x'
                nvals += 1
            end
        end
        nvals == 0 && continue
        mean_x = sum_x ./ nvals
        second_x = sum_xx ./ nvals
        push!(re_pairs, re => (family=family, mean=mean_x, second=second_x, n=nvals))
    end
    re_stats = NamedTuple(re_pairs)

    obs_targets = _saem_outcome_targets(dm.config.obs_cols, resid_var_param)
    outcome_pairs = Pair{Symbol, Any}[]
    if !isempty(keys(obs_targets))
        obs_acc = Dict{Symbol, Any}()
        all_supported = true
        for (bi, info) in enumerate(batch_infos)
            b = b_current[bi]
            for i in info.inds
                η_ind = _build_eta_ind(dm, i, info, b, const_cache, θ)
                stats_i, ok = _saem_collect_outcome_stats_individual(dm, i, θ, η_ind, ll_cache, obs_targets)
                if !ok
                    all_supported = false
                    break
                end
                for (col, st) in pairs(stats_i)
                    prev = get(obs_acc, col, nothing)
                    if prev === nothing
                        obs_acc[col] = st
                    else
                        merged = _saem_merge_outcome_stats(prev, st)
                        merged === nothing && (all_supported = false; break)
                        obs_acc[col] = merged
                    end
                end
                all_supported || break
            end
            all_supported || break
        end
        if all_supported
            for col in dm.config.obs_cols
                haskey(obs_acc, col) && push!(outcome_pairs, col => obs_acc[col])
            end
        end
    end
    hmm_pairs = Pair{Symbol, Any}[]
    if !isempty(keys(hmm_emission_params))
        hmm_acc = Dict{Symbol, Any}()
        all_supported = true
        for (bi, info) in enumerate(batch_infos)
            b = b_current[bi]
            for i in info.inds
                η_ind = _build_eta_ind(dm, i, info, b, const_cache, θ)
                stats_i, ok = _saem_collect_hmm_stats_individual(dm, i, θ, η_ind, ll_cache, hmm_emission_params)
                if !ok
                    all_supported = false
                    break
                end
                for (col, st) in pairs(stats_i)
                    prev = get(hmm_acc, col, nothing)
                    if prev === nothing
                        hmm_acc[col] = st
                    else
                        merged = _saem_merge_hmm_stats(prev, st)
                        merged === nothing && (all_supported = false; break)
                        hmm_acc[col] = merged
                    end
                end
                all_supported || break
            end
            all_supported || break
        end
        if all_supported
            for col in keys(hmm_emission_params)
                haskey(hmm_acc, col) && push!(hmm_pairs, col => hmm_acc[col])
            end
        end
    end

    return (re=re_stats, outcome=NamedTuple(outcome_pairs), hmm=NamedTuple(hmm_pairs))
end

function _saem_builtin_smooth_re_stats(prev_re::NamedTuple, curr_re::NamedTuple, γ::Real)
    keys_order = Symbol[collect(keys(prev_re))...]
    for re in keys(curr_re)
        re in keys_order || push!(keys_order, re)
    end
    pairs = Pair{Symbol, Any}[]
    for re in keys_order
        if haskey(prev_re, re) && haskey(curr_re, re)
            p = getfield(prev_re, re)
            c = getfield(curr_re, re)
            push!(pairs, re => (family=hasproperty(c, :family) ? c.family : (hasproperty(p, :family) ? p.family : :normal),
                                mean=_saem_blend(p.mean, c.mean, γ),
                                second=_saem_blend(p.second, c.second, γ),
                                n=c.n))
        elseif haskey(curr_re, re)
            push!(pairs, re => getfield(curr_re, re))
        else
            push!(pairs, re => getfield(prev_re, re))
        end
    end
    return NamedTuple(pairs)
end

function _saem_builtin_smooth_outcome_stats(prev_out::NamedTuple, curr_out::NamedTuple, γ::Real)
    keys_order = Symbol[collect(keys(prev_out))...]
    for col in keys(curr_out)
        col in keys_order || push!(keys_order, col)
    end
    pairs = Pair{Symbol, Any}[]
    for col in keys_order
        if haskey(prev_out, col) && haskey(curr_out, col)
            p = getfield(prev_out, col)
            c = getfield(curr_out, col)
            if p.family == c.family
                push!(pairs, col => (family=c.family,
                                     s1=_saem_blend(p.s1, c.s1, γ),
                                     s2=_saem_blend(p.s2, c.s2, γ),
                                     ss=_saem_blend(p.ss, c.ss, γ),
                                     n=c.n))
            else
                push!(pairs, col => c)
            end
        elseif haskey(curr_out, col)
            push!(pairs, col => getfield(curr_out, col))
        else
            push!(pairs, col => getfield(prev_out, col))
        end
    end
    return NamedTuple(pairs)
end

function _saem_builtin_smooth_hmm_stats(prev_hmm::NamedTuple, curr_hmm::NamedTuple, γ::Real)
    keys_order = Symbol[collect(keys(prev_hmm))...]
    for col in keys(curr_hmm)
        col in keys_order || push!(keys_order, col)
    end
    pairs = Pair{Symbol, Any}[]
    for col in keys_order
        if haskey(prev_hmm, col) && haskey(curr_hmm, col)
            p = getfield(prev_hmm, col)
            c = getfield(curr_hmm, col)
            if p.family == c.family && p.target == c.target
                push!(pairs, col => (family=c.family,
                                     target=c.target,
                                     sum_w=_saem_blend(p.sum_w, c.sum_w, γ),
                                     sum_wy=_saem_blend(p.sum_wy, c.sum_wy, γ)))
            else
                push!(pairs, col => c)
            end
        elseif haskey(curr_hmm, col)
            push!(pairs, col => getfield(curr_hmm, col))
        else
            push!(pairs, col => getfield(prev_hmm, col))
        end
    end
    return NamedTuple(pairs)
end

function _saem_builtin_smooth_stats(prev_stats, curr_stats, γ::Real)
    prev_stats === nothing && return curr_stats
    re = _saem_builtin_smooth_re_stats(prev_stats.re, curr_stats.re, γ)
    outcome = _saem_builtin_smooth_outcome_stats(prev_stats.outcome, curr_stats.outcome, γ)
    hmm = _saem_builtin_smooth_hmm_stats(prev_stats.hmm, curr_stats.hmm, γ)
    return (; re, outcome, hmm)
end

@inline function _saem_named_target_value(target::NamedTuple, keys::Tuple)
    for k in keys
        haskey(target, k) && return getfield(target, k)
    end
    vals = collect(values(target))
    return length(vals) == 1 ? vals[1] : nothing
end

function _saem_target_hits_param(θ::ComponentArray, target)
    if target isa Symbol
        return hasproperty(θ, target)
    elseif target isa NamedTuple
        for v in values(target)
            _saem_target_hits_param(θ, v) && return true
        end
        return false
    elseif target isa Tuple || target isa AbstractVector
        for v in target
            _saem_target_hits_param(θ, v) && return true
        end
        return false
    else
        return false
    end
end

function _saem_lognormal_targets(target)
    if target isa NamedTuple
        μ_target = _saem_named_target_value(target, (:μ, :mu))
        σ_target = _saem_named_target_value(target, (:σ, :sigma, :scale))
        return μ_target, σ_target
    elseif target isa Tuple || target isa AbstractVector
        vals = collect(target)
        if length(vals) == 2
            return vals[1], vals[2]
        elseif length(vals) == 1
            return nothing, vals[1]
        else
            return nothing, nothing
        end
    elseif target isa Symbol
        return nothing, target
    end
    return nothing, nothing
end

function _saem_outcome_update_from_stat!(updates::Vector{Pair{Symbol, Any}},
                                         θ::ComponentArray,
                                         target,
                                         stat,
                                         col::Symbol)
    n = stat.n
    n > 0 || return nothing
    T = promote_type(eltype(θ), typeof(stat.s1))
    if stat.family == :normal
        σ2 = stat.s1 / T(n)
        σ_hat = sqrt(max(σ2, zero(T)))
        t = target isa NamedTuple ? _saem_named_target_value(target, (:σ, :sigma, :scale)) : target
        t === nothing || _saem_push_param_updates!(updates, θ, t, σ_hat; context="outcome.$(col)")
        return nothing
    elseif stat.family == :lognormal
        μ_target, σ_target = _saem_lognormal_targets(target)
        μ_hat = stat.s1 / T(n)
        if μ_target !== nothing
            _saem_push_param_updates!(updates, θ, μ_target, μ_hat; context="outcome.$(col).μ")
        end
        if σ_target !== nothing
            use_free_μ = μ_target !== nothing && _saem_target_hits_param(θ, μ_target)
            var_log = use_free_μ ? max(stat.s2 / T(n) - μ_hat * μ_hat, zero(T)) :
                                   max(stat.ss / T(n), zero(T))
            σ_hat = sqrt(var_log)
            _saem_push_param_updates!(updates, θ, σ_target, σ_hat; context="outcome.$(col).σ")
        end
        return nothing
    elseif stat.family == :exponential
        θ_hat = max(stat.s1 / T(n), eps(T))
        t = target isa NamedTuple ? _saem_named_target_value(target, (:θ, :scale, :theta)) : target
        t === nothing || _saem_push_param_updates!(updates, θ, t, θ_hat; context="outcome.$(col)")
        return nothing
    elseif stat.family == :bernoulli
        p_hat = clamp(stat.s1 / T(n), eps(T), one(T) - eps(T))
        t = target isa NamedTuple ? _saem_named_target_value(target, (:p,)) : target
        t === nothing || _saem_push_param_updates!(updates, θ, t, p_hat; context="outcome.$(col)")
        return nothing
    elseif stat.family == :poisson
        λ_hat = max(stat.s1 / T(n), eps(T))
        t = target isa NamedTuple ? _saem_named_target_value(target, (:λ, :lambda)) : target
        t === nothing || _saem_push_param_updates!(updates, θ, t, λ_hat; context="outcome.$(col)")
        return nothing
    end
    return nothing
end

function _saem_unique_updates(updates::Vector{Pair{Symbol, Any}})
    seen = Dict{Symbol, Int}()
    out = Pair{Symbol, Any}[]
    for p in updates
        sym = p.first
        if haskey(seen, sym)
            out[seen[sym]] = p
        else
            push!(out, p)
            seen[sym] = length(out)
        end
    end
    return out
end

@inline function _saem_clamp_value_to_bounds(v, lb, ub)
    if v isa Number
        return clamp(v, lb, ub)
    elseif v isa AbstractArray
        return clamp.(v, lb, ub)
    end
    return v
end

function _saem_clamp_constants_to_bounds(constants::NamedTuple, fe::FixedEffects)
    isempty(keys(constants)) && return constants
    lower_u, upper_u = get_bounds_untransformed(fe)
    pairs = Pair{Symbol, Any}[]
    for name in keys(constants)
        val = getfield(constants, name)
        if hasproperty(lower_u, name) && hasproperty(upper_u, name)
            lb = getproperty(lower_u, name)
            ub = getproperty(upper_u, name)
            val = _saem_clamp_value_to_bounds(val, lb, ub)
        end
        push!(pairs, name => val)
    end
    return NamedTuple(pairs)
end

function _saem_builtin_updates_from_smoothed_stats(dm::DataModel,
                                                   θ::ComponentArray,
                                                   stats,
                                                   resid_var_param,
                                                   hmm_emission_params::NamedTuple,
                                                   re_cov_params::NamedTuple,
                                                   re_mean_params::NamedTuple)
    stats === nothing && return NamedTuple()
    updates = Pair{Symbol, Any}[]

    if !isempty(keys(stats.outcome))
        if resid_var_param isa Symbol
            agg = nothing
            mixed = false
            for col in keys(stats.outcome)
                st = getfield(stats.outcome, col)
                if agg === nothing
                    agg = st
                elseif agg.family == st.family
                    agg = _saem_merge_outcome_stats(agg, st)
                else
                    mixed = true
                    break
                end
            end
            if !mixed && agg !== nothing
                _saem_outcome_update_from_stat!(updates, θ, resid_var_param, agg, :all)
            end
        elseif resid_var_param isa NamedTuple
            for col in keys(resid_var_param)
                haskey(stats.outcome, col) || continue
                target = getfield(resid_var_param, col)
                st = getfield(stats.outcome, col)
                _saem_outcome_update_from_stat!(updates, θ, target, st, col)
            end
        end
    end

    if !isempty(keys(stats.hmm))
        for col in keys(hmm_emission_params)
            haskey(stats.hmm, col) || continue
            st = getfield(stats.hmm, col)
            st.family == :hmm_bernoulli || continue
            target = st.target
            sum_w = st.sum_w
            sum_wy = st.sum_wy
            p_hat = clamp.(sum_wy ./ max.(sum_w, eps(eltype(sum_w))), eps(eltype(sum_w)), one(eltype(sum_w)) - eps(eltype(sum_w)))
            if target isa Symbol && hasproperty(θ, target)
                θ_target = getproperty(θ, target)
                val = if θ_target isa AbstractVector
                    vec(p_hat)
                elseif θ_target isa AbstractMatrix
                    p_hat
                elseif θ_target isa Number
                    vec(p_hat)[1]
                else
                    vec(p_hat)
                end
                push!(updates, target => val)
            else
                _saem_push_param_updates!(updates, θ, target, vec(p_hat); context="hmm.$(col)")
            end
        end
    end

    for re in _saem_builtin_re_targets(re_cov_params, re_mean_params)
        haskey(stats.re, re) || continue
        st = getfield(stats.re, re)
        family = hasproperty(st, :family) ? st.family : :normal
        if family == :exponential
            if haskey(re_cov_params, re)
                θ_hat = max.(st.mean, eps(eltype(st.mean)))
                cov_target = getfield(re_cov_params, re)
                if cov_target isa Symbol
                    if hasproperty(θ, cov_target)
                        θ_cov = getproperty(θ, cov_target)
                        if θ_cov isa AbstractVector
                            push!(updates, cov_target => θ_hat)
                        else
                            push!(updates, cov_target => θ_hat[1])
                        end
                    end
                else
                    _saem_push_param_updates!(updates, θ, cov_target, θ_hat;
                                              context="re_cov_params.$(re)")
                end
            end
            continue
        end

        μ_hat = st.mean
        S2_hat = st.second
        Σ_hat = S2_hat .- μ_hat * μ_hat'
        Σ_hat = 0.5 .* (Σ_hat .+ Σ_hat')
        var_diag = max.(diag(Σ_hat), zero(eltype(Σ_hat)))
        σ_diag = sqrt.(var_diag)
        cov_diag = family == :mvnormal ? var_diag : σ_diag

        if haskey(re_mean_params, re)
            _saem_push_param_updates!(updates, θ, getfield(re_mean_params, re), μ_hat;
                                      context="re_mean_params.$(re)")
        end
        if haskey(re_cov_params, re)
            cov_target = getfield(re_cov_params, re)
            if cov_target isa Symbol
                if hasproperty(θ, cov_target)
                    θ_cov = getproperty(θ, cov_target)
                    if θ_cov isa AbstractMatrix
                        Ω_hat = copy(Σ_hat)
                        base = mean(diag(Ω_hat))
                        scale = isfinite(base) ? max(abs(base), one(base)) : one(real(eltype(Ω_hat)))
                        jitter = max(sqrt(eps(real(eltype(Ω_hat)))), 1e-8 * scale)
                        Ω_hat = Ω_hat + jitter * I
                        push!(updates, cov_target => Ω_hat)
                    elseif θ_cov isa AbstractVector
                        push!(updates, cov_target => cov_diag)
                    else
                        push!(updates, cov_target => cov_diag[1])
                    end
                end
            else
                _saem_push_param_updates!(updates, θ, cov_target, cov_diag;
                                          context="re_cov_params.$(re)")
            end
        end
    end
    return NamedTuple(_saem_unique_updates(updates))
end

function _saem_batches(update_schedule, nbatches::Int, iter::Int, rng::AbstractRNG)
    return _saem_batches!(Vector{Int}(), update_schedule, nbatches, iter, rng)
end

function _saem_Q(dm::DataModel,
                 batch_infos::Vector{_LaplaceBatchInfo},
                 θ::ComponentArray,
                 const_cache::LaplaceConstantsCache,
                 ll_cache,
                 store::_SAEMSampleStore;
                 serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                 q_cache::Union{Nothing, _SAEMQCache}=nothing,
                 anneal_sds::NamedTuple=NamedTuple())
    total = zero(eltype(θ))
    store.len == 0 && return total
    if serialization isa SciMLBase.EnsembleThreads
        nthreads = Threads.maxthreadid()
        caches = _saem_thread_caches(dm, ll_cache, nthreads)
        Tθ = eltype(θ)
        use_cache = q_cache !== nothing && eltype(q_cache.partial_obj) === Tθ
        partial_obj = use_cache ? q_cache.partial_obj : Vector{Tθ}()
        if length(partial_obj) != length(batch_infos)
            resize!(partial_obj, length(batch_infos))
        end
        fill!(partial_obj, zero(Tθ))
        bad = Threads.Atomic{Bool}(false)
        Threads.@threads for bi in eachindex(batch_infos)
            bad[] && continue
            tid = Threads.threadid()
            info = batch_infos[bi]
            acc = zero(eltype(θ))
            @inbounds for k in 1:store.len
                w = store.weights[k]
                snap = store.snaps[k][bi]
                isempty(snap) && continue
                logf = _laplace_logf_batch(dm, info, θ, snap, const_cache, caches[tid];
                                           anneal_sds=anneal_sds)
                !isfinite(logf) && (bad[] = true; break)
                acc += w * logf
            end
            bad[] && continue
            partial_obj[bi] = acc
        end
        bad[] && return Inf
        total = zero(Tθ)
        @inbounds for bi in eachindex(batch_infos)
            total += partial_obj[bi]
        end
        return total
    else
        ll_cache_local = ll_cache isa Vector ? ll_cache[1] : ll_cache
        for (bi, info) in enumerate(batch_infos)
            acc = zero(eltype(θ))
            @inbounds for k in 1:store.len
                w = store.weights[k]
                snap = store.snaps[k][bi]
                isempty(snap) && continue
                logf = _laplace_logf_batch(dm, info, θ, snap, const_cache, ll_cache_local;
                                           anneal_sds=anneal_sds)
                !isfinite(logf) && return Inf
                acc += w * logf
            end
            total += acc
        end
        return total
    end
end

function _saem_collect_target_symbols!(out::Vector{Symbol}, target)
    if target isa Symbol
        push!(out, target)
    elseif target isa NamedTuple
        for v in values(target)
            _saem_collect_target_symbols!(out, v)
        end
    elseif target isa Tuple || target isa AbstractVector
        for v in target
            _saem_collect_target_symbols!(out, v)
        end
    end
    return out
end

function _saem_log_closed_form_plan(requested_mode::Symbol,
                                    effective_mode::Symbol,
                                    elig,
                                    re_cov_params::NamedTuple,
                                    re_mean_params::NamedTuple,
                                    resid_var_param,
                                    hmm_emission_params::NamedTuple,
                                    has_custom_closed_form::Bool)
    requested_norm = _saem_normalize_builtin_stats_mode(requested_mode)
    requested_norm == :none && !has_custom_closed_form && return nothing

    re_cov_targets = Symbol[]
    re_mean_targets = Symbol[]
    resid_targets = Symbol[]
    hmm_emission_targets = Symbol[]

    for v in values(re_cov_params)
        _saem_collect_target_symbols!(re_cov_targets, v)
    end
    for v in values(re_mean_params)
        _saem_collect_target_symbols!(re_mean_targets, v)
    end
    _saem_collect_target_symbols!(resid_targets, resid_var_param)
    for col in keys(hmm_emission_params)
        info = getfield(hmm_emission_params, col)
        if hasproperty(info, :target)
            _saem_collect_target_symbols!(hmm_emission_targets, getproperty(info, :target))
        end
    end

    @info "SAEM closed-form M-step plan" requested_mode=requested_mode effective_mode=effective_mode has_custom_closed_form=has_custom_closed_form has_any_closed_form_block=elig.has_any_closed_form_block re_block_eligible=elig.re_block_eligible outcome_block_eligible=elig.outcome_block_eligible hmm_emission_block_eligible=elig.hmm_emission_block_eligible re_targets=elig.re_targets outcome_targets_non_hmm=elig.outcome_targets_non_hmm outcome_targets_hmm=elig.outcome_targets_hmm hmm_outcomes=elig.hmm_outcomes hmm_transition_closed_form=elig.hmm_transition_closed_form hmm_emission_closed_form=elig.hmm_emission_closed_form hmm_emission_targets=elig.hmm_emission_targets hmm_emission_targets_unsupported=elig.hmm_emission_targets_unsupported mapped_re_cov_params=Tuple(keys(re_cov_params)) mapped_re_mean_params=Tuple(keys(re_mean_params)) mapped_re_cov_targets=Tuple(unique(re_cov_targets)) mapped_re_mean_targets=Tuple(unique(re_mean_targets)) mapped_residual_targets=Tuple(unique(resid_targets)) mapped_hmm_emission_outcomes=Tuple(keys(hmm_emission_params)) mapped_hmm_emission_param_targets=Tuple(unique(hmm_emission_targets)) reasons=elig.reasons
    return nothing
end

function _saem_glm_supported(dm::DataModel,
                             batch_infos::Vector{_LaplaceBatchInfo},
                             b_current::AbstractVector,
                             θ::ComponentArray,
                             const_cache::LaplaceConstantsCache,
                             ll_cache::_LLCache)
    model = dm.model

    allowed = Union{Normal, Bernoulli, Poisson, Exponential}
    for (bi, info) in enumerate(batch_infos)
        b = b_current[bi]
        for i in info.inds
            ind = dm.individuals[i]
            obs_rows = dm.row_groups.obs_rows[i]
            rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only=true)
            for j in eachindex(obs_rows)
                v = _varying_at(dm, ind, j, _get_col(dm.df, dm.config.time_col)[obs_rows])
                η_ind = _build_eta_ind(dm, i, info, b, const_cache, θ)
                η_row = _row_random_effects_at(dm, i, j, η_ind, rowwise_re; obs_only=true)
                obs = calculate_formulas_obs(model, θ, η_row, ind.const_cov, v)
                for col in dm.config.obs_cols
                    dist = getproperty(obs, col)
                    dist isa allowed || return false
                end
            end
        end
    end
    return true
end

function _saem_builtin_mean_updates(dm::DataModel,
                                    batch_infos::Vector{_LaplaceBatchInfo},
                                    b_current::AbstractVector,
                                    θu_curr::ComponentArray,
                                    const_cache::LaplaceConstantsCache,
                                    mean_params::Vector{Symbol},
                                    ll_cache::_LLCache,
                                    sample_store::_SAEMSampleStore,
                                    transform,
                                    inv_transform;
                                    optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
                                    optim_kwargs::NamedTuple=NamedTuple(),
                                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                                    penalty::NamedTuple=NamedTuple(),
                                    anneal_sds::NamedTuple=NamedTuple())
    isempty(mean_params) && return NamedTuple()
    if dm.model.de.de === nothing
        _saem_glm_supported(dm, batch_infos, b_current, θu_curr, const_cache, ll_cache) || return NamedTuple()
    end
    sample_store.len == 0 && return NamedTuple()

    θt_curr = transform(θu_curr)
    mean_names = [n for n in mean_params if hasproperty(θt_curr, n)]
    isempty(mean_names) && return NamedTuple()

    θt_mean = ComponentArray(NamedTuple{Tuple(mean_names)}(Tuple(getproperty(θt_curr, n) for n in mean_names)))
    axs_mean = getaxes(θt_mean)
    T0 = eltype(θu_curr)
    q_cache = _init_saem_q_cache(T0, length(batch_infos), serialization)

    function obj_only(θt_vec, p)
        θt_mean_ca = θt_vec isa ComponentArray ? θt_vec : ComponentArray(θt_vec, axs_mean)
        Tloc = eltype(θt_mean_ca)
        θt_full = ComponentArray(Tloc.(θt_curr), getaxes(θt_curr))
        for name in mean_names
            setproperty!(θt_full, name, getproperty(θt_mean_ca, name))
        end
        θu = inv_transform(θt_full)
        Q = _saem_Q(dm, batch_infos, θu, const_cache, ll_cache, sample_store;
                    serialization=serialization, q_cache=q_cache, anneal_sds=anneal_sds)
        !isfinite(Q) && return Inf
        obj = -Q + _penalty_value(θu, penalty)
        !isfinite(obj) && return Inf
        return obj
    end

    optf = OptimizationFunction(obj_only, Optimization.AutoForwardDiff())
    θ0_init = collect(θt_mean)
    prob = OptimizationProblem(optf, θ0_init)
    sol = Optimization.solve(prob, optimizer; optim_kwargs...)
    θt_mean_hat = sol.u isa ComponentArray ? sol.u : ComponentArray(sol.u, axs_mean)

    θt_full_hat = deepcopy(θt_curr)
    for name in mean_names
        setproperty!(θt_full_hat, name, getproperty(θt_mean_hat, name))
    end
    θu_hat = inv_transform(θt_full_hat)
    return NamedTuple{Tuple(mean_names)}(Tuple(getproperty(θu_hat, n) for n in mean_names))
end

function _fit_model(dm::DataModel, method::SAEM, args...;
                    constants::NamedTuple=NamedTuple(),
                    constants_re::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                    rng::AbstractRNG=Xoshiro(0),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_eb_modes::Bool=true,
                    store_data_model::Bool=true)
    fit_kwargs = (constants=constants,
                  constants_re=constants_re,
                  penalty=penalty,
                  ode_args=ode_args,
                  ode_kwargs=ode_kwargs,
                  serialization=serialization,
                  rng=rng,
                  theta_0_untransformed=theta_0_untransformed,
                  store_eb_modes=store_eb_modes,
                  store_data_model=store_data_model)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && error("SAEM requires random effects. Use MLE/MAP for fixed-effects models.")
    fe = dm.model.fixed.fixed
    priors = get_priors(fe)
    if any(!(getfield(priors, k) isa Priorless) for k in keys(priors))
        @info "SAEM ignores fixed-effect priors. Use MAP/MCMC for prior-aware inference."
    end

    fixed_names = get_names(fe)
    isempty(fixed_names) && error("SAEM requires at least one fixed effect.")
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    all(name in keys(constants) for name in fixed_names) &&
        error("SAEM requires at least one free fixed effect. Remove constants or specify a fixed effect or random effect.")

    base_free_names = [n for n in fixed_names if !(n in keys(constants))]
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

    fixed_maps = _normalize_constants_re(dm, constants_re)
    const_cache = _build_constants_cache(dm, fixed_maps)
    pairing, batch_infos, _ = _build_laplace_batch_infos(dm, fixed_maps)
    ll_cache = serialization isa SciMLBase.EnsembleThreads ?
               build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=Threads.maxthreadid(), force_saveat=true) :
               build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)

    builtin_stats_mode_requested = _saem_normalize_builtin_stats_mode(method.saem.builtin_stats)
    builtin_stats_mode = builtin_stats_mode_requested
    resid_var_param = method.saem.resid_var_param
    re_cov_params = method.saem.re_cov_params
    re_mean_params = method.saem.re_mean_params
    hmm_emission_params = NamedTuple()
    if builtin_stats_mode == :auto
        manual_has_re = !isempty(keys(re_cov_params)) || !isempty(keys(re_mean_params))
        manual_has_resid = !(resid_var_param == :σ || (resid_var_param isa NamedTuple && isempty(keys(resid_var_param))))
        auto_cfg = _saem_autodetect_gaussian_re(dm, fixed_names)
        if auto_cfg === nothing
            builtin_stats_mode = (manual_has_re || manual_has_resid) ? :closed_form : :none
        else
            builtin_stats_mode = :closed_form
            re_cov_params = isempty(keys(re_cov_params)) ? auto_cfg.re_cov_params : merge(auto_cfg.re_cov_params, re_cov_params)
            re_mean_params = isempty(keys(re_mean_params)) ? auto_cfg.re_mean_params : merge(auto_cfg.re_mean_params, re_mean_params)
            hmm_emission_params = auto_cfg.hmm_emission_params
            if resid_var_param isa NamedTuple
                if isempty(keys(resid_var_param))
                    resid_var_param = auto_cfg.resid_var_param
                elseif auto_cfg.resid_var_param isa NamedTuple
                    resid_var_param = merge(auto_cfg.resid_var_param, resid_var_param)
                end
            elseif resid_var_param == :σ
                resid_var_param = auto_cfg.resid_var_param
            end
        end
    end
    if isempty(keys(hmm_emission_params))
        hmm_emission_params = _saem_autodetect_hmm_emission_params(dm, Set(fixed_names))
    end
    builtin_cf_elig = _saem_builtin_closed_form_eligibility(dm, re_cov_params, re_mean_params,
                                                             resid_var_param, hmm_emission_params)
    if builtin_stats_mode == :closed_form
        resid_var_param = _saem_prune_hmm_outcome_targets(dm, resid_var_param, builtin_cf_elig.hmm_outcomes)
        builtin_cf_elig = _saem_builtin_closed_form_eligibility(dm, re_cov_params, re_mean_params,
                                                                 resid_var_param, hmm_emission_params)
        if !isempty(builtin_cf_elig.outcome_targets_hmm)
            @info "SAEM builtin_stats ignores HMM outcome targets; applying closed-form updates to eligible non-HMM/re blocks only." hmm_outcomes=builtin_cf_elig.outcome_targets_hmm
        end
        if !builtin_cf_elig.has_any_closed_form_block
            @info "SAEM builtin_stats has no eligible closed-form blocks; falling back to numeric M-step."
            builtin_stats_mode = :none
        end
    end
    has_custom_closed_form = method.saem.suffstats !== nothing && method.saem.mstep_closed_form !== nothing
    _saem_log_closed_form_plan(method.saem.builtin_stats, builtin_stats_mode, builtin_cf_elig,
                               re_cov_params, re_mean_params, resid_var_param, hmm_emission_params,
                               has_custom_closed_form)
    re_family_map = _saem_re_family_map(dm)

    # anneal_to_fixed validation
    anneal_initial_sds = _saem_validate_anneal(method.saem.anneal_to_fixed,
                                               method.saem.anneal_schedule, dm)
    do_anneal = !isempty(anneal_initial_sds)
    anneal_names = do_anneal ? Tuple(keys(anneal_initial_sds)) : ()
    if do_anneal
        # Warn if an annealed RE also has a closed-form cov update (annealing always wins)
        for name in anneal_names
            if haskey(re_cov_params, name)
                @info "SAEM anneal_to_fixed: RE :$name also appears in re_cov_params; " *
                      "annealing overrides the closed-form covariance update."
            end
        end
    end

    θt_free = ComponentArray(NamedTuple{Tuple(base_free_names)}(Tuple(getproperty(θ0_t, n) for n in base_free_names)))
    axs_free = getaxes(θt_free)
    axs_full = getaxes(θ_const_t)
    T0 = eltype(θt_free)

    diag = _SAEMDiagnostics{T0}(Vector{AbstractVector{T0}}(),
                               Vector{T0}(),
                               Vector{T0}(),
                               Vector{T0}(),
                               Vector{T0}(),
                               Vector{T0}(),
                               Vector{T0}(),
                               Vector{Vector{Int}}())

    q_cache = _init_saem_q_cache(T0, length(batch_infos), serialization)

    last_params = Vector{Union{Nothing, NamedTuple, AbstractVector}}(undef, length(batch_infos))
    fill!(last_params, nothing)
    batch_rngs = _saem_thread_rngs(rng, length(batch_infos))
    b_current = [zeros(T0, info.n_b) for info in batch_infos]

    # O2+O4+O5: preallocated ring buffer for SA sample history
    sample_store = _init_saem_sample_store(method.saem.max_store, batch_infos)
    s = nothing
    builtin_stats_state = nothing
    closed_form_builtin_used = false
    closed_form_custom_used = false
    numeric_mstep_used = false

    # O3: pre-allocated work buffer for per-iter constants — avoids deepcopy each iter
    θ_const_u_work = deepcopy(θ0_u)

    θ_prev = copy(θt_free)
    θt_full_init = ComponentArray(eltype(θt_free).(θ_const_t), axs_full)
    for name in base_free_names
        setproperty!(θt_full_init, name, getproperty(θt_free, name))
    end
    θ_prev_full = copy(θt_full_init)
    Q_prev = T0(Inf)
    param_streak = 0
    q_streak = 0
    converged = false
    progress = ProgressMeter.Progress(method.saem.maxiters; desc="SAEM", enabled=method.saem.progress)

    # O1: hoist tkwargs construction — values never change during the run
    tkwargs = method.saem.turing_kwargs
    n_samples_saem = method.saem.mcmc_steps > 0 ? method.saem.mcmc_steps : get(tkwargs, :n_samples, 1)
    tkwargs = merge(tkwargs, (n_samples=n_samples_saem,))
    haskey(tkwargs, :n_adapt)   || (tkwargs = merge(tkwargs, (n_adapt=50,)))
    haskey(tkwargs, :progress)  || (tkwargs = merge(tkwargs, (progress=false,)))
    haskey(tkwargs, :verbose)   || (tkwargs = merge(tkwargs, (verbose=false,)))

    for iter in 1:method.saem.maxiters
        γ = _saem_gamma(iter, method.saem.t0, method.saem.kappa)
        updated = _saem_batches!(q_cache.batches_buf, method.saem.update_schedule, length(batch_infos), iter, rng)

        θt_curr = θ_prev isa ComponentArray ? θ_prev : ComponentArray(θ_prev, axs_free)
        θt_full_curr = ComponentArray(T0.(θ_const_t), axs_full)
        for name in base_free_names
            setproperty!(θt_full_curr, name, getproperty(θt_curr, name))
        end
        θu_curr = inv_transform(θt_full_curr)

        # Compute annealed SDs for this iteration
        anneal_sds = if do_anneal
            NamedTuple{anneal_names}(
                Tuple(_saem_anneal_sd(getfield(anneal_initial_sds, n), method.saem.anneal_min_sd,
                                     iter, method.saem.maxiters, method.saem.anneal_schedule,
                                     method.saem.t0, method.saem.kappa)
                      for n in anneal_names))
        else
            NamedTuple()
        end

        if serialization isa SciMLBase.EnsembleThreads
            nthreads = Threads.maxthreadid()
            caches = _saem_thread_caches(dm, ll_cache, nthreads)
            Threads.@threads for bi in updated
                info = batch_infos[bi]
                samples, lastp, lastb = _mcem_sample_batch(dm, info, θu_curr, const_cache, caches[Threads.threadid()],
                                                           method.saem.sampler, tkwargs, batch_rngs[bi],
                                                           re_names, method.saem.warm_start, last_params[bi];
                                                           anneal_sds=anneal_sds)
                b_current[bi] = lastb
                last_params[bi] = lastp
            end
        else
            for bi in updated
                info = batch_infos[bi]
                samples, lastp, lastb = _mcem_sample_batch(dm, info, θu_curr, const_cache, ll_cache,
                                                           method.saem.sampler, tkwargs, batch_rngs[bi],
                                                           re_names, method.saem.warm_start, last_params[bi];
                                                           anneal_sds=anneal_sds)
                b_current[bi] = lastb
                last_params[bi] = lastp
            end
        end


        if method.saem.suffstats !== nothing
            s_new = method.saem.suffstats(dm, batch_infos, b_current, θu_curr, fixed_maps)
            s = _saem_stats_update(s, s_new, γ)
        else
            # O2+O4+O5: push into ring buffer in-place (no allocation, no Array shifts)
            _saem_store_push!(sample_store, b_current, γ)
        end


        # builtin variance updates (optional)
        iter_constants = constants
        if builtin_stats_mode == :closed_form
            cache = ll_cache isa Vector ? ll_cache[1] : ll_cache
            curr_stats = _saem_builtin_collect_current_stats(dm, batch_infos, b_current,
                                                             ComponentArray(θu_curr, getaxes(θu_curr)), const_cache,
                                                             resid_var_param, hmm_emission_params,
                                                             re_cov_params, re_mean_params,
                                                             re_family_map, cache)
            builtin_stats_state = _saem_builtin_smooth_stats(builtin_stats_state, curr_stats, γ)
            updates = _saem_builtin_updates_from_smoothed_stats(dm, ComponentArray(θu_curr, getaxes(θu_curr)),
                                                                builtin_stats_state, resid_var_param, hmm_emission_params,
                                                                re_cov_params, re_mean_params)
            if !isempty(updates)
                closed_form_builtin_used = true
                # annealing always wins: strip cov targets for annealed REs
                if do_anneal
                    for an in anneal_names
                        if haskey(re_cov_params, an)
                            target_sym = re_cov_params[an]
                            target_sym isa Symbol && haskey(updates, target_sym) &&
                                (updates = Base.structdiff(updates, NamedTuple{(target_sym,)}((nothing,))))
                        end
                    end
                end
                iter_constants = merge(iter_constants, updates)
            end
        elseif builtin_stats_mode != :none
            @info "Unknown builtin_stats option; using numeric SAEM." option=builtin_stats_mode allowed=(:auto, :closed_form, :gaussian_re, :none)
        end

        # builtin mean updates (optional, GLM)
        if method.saem.builtin_mean == :glm
            resid_params = Symbol[]
            if resid_var_param isa NamedTuple
                for v in values(resid_var_param)
                    _saem_collect_target_symbols!(resid_params, v)
                end
            elseif resid_var_param isa Symbol
                push!(resid_params, resid_var_param)
            end
            re_cov_param_values = Symbol[]
            for v in values(re_cov_params)
                _saem_collect_target_symbols!(re_cov_param_values, v)
            end
            re_mean_param_values = Symbol[]
            for v in values(re_mean_params)
                _saem_collect_target_symbols!(re_mean_param_values, v)
            end
            mean_params = [n for n in fixed_names if !(n in keys(iter_constants)) &&
                           !(n in resid_params) &&
                           !(n in re_cov_param_values) &&
                           !(n in re_mean_param_values)]

            if method.saem.suffstats !== nothing
                @info "builtin_mean=:glm skipped because suffstats path is active."
            else
                cache = ll_cache isa Vector ? ll_cache[1] : ll_cache
                mean_updates = _saem_builtin_mean_updates(dm, batch_infos, b_current,
                                                          ComponentArray(θu_curr, getaxes(θu_curr)), const_cache,
                                                          mean_params, cache, sample_store, transform, inv_transform;
                                                          optimizer=method.optimizer, optim_kwargs=method.optim_kwargs,
                                                          serialization=serialization, penalty=penalty,
                                                          anneal_sds=anneal_sds)
                if !isempty(mean_updates)
                    iter_constants = merge(iter_constants, mean_updates)
                end
            end
        elseif method.saem.builtin_mean != :none
            @info "Unknown builtin_mean option; using numeric SAEM." option=method.saem.builtin_mean
        end
        iter_constants = _saem_clamp_constants_to_bounds(iter_constants, fe)
        # build per-iteration free set / transforms
        free_names_iter = [n for n in fixed_names if !(n in keys(iter_constants))]
        θt_free = ComponentArray(NamedTuple{Tuple(free_names_iter)}(Tuple(getproperty(θt_full_curr, n) for n in free_names_iter)))
        axs_free = getaxes(θt_free)
        # O3: reuse pre-allocated buffer instead of deepcopy each iteration
        copyto!(θ_const_u_work, θ0_u)
        _apply_constants!(θ_const_u_work, iter_constants)
        θ_const_t_iter = transform(θ_const_u_work)
        axs_full_iter = getaxes(θ_const_t_iter)
        base_free_names = free_names_iter
        θ_const_t = θ_const_t_iter
        axs_full = axs_full_iter

        all_fixed_by_builtin = isempty(free_names_iter)
        if all_fixed_by_builtin
            θ_prev_new = copy(θt_free)
            θt_full = ComponentArray(T0.(θ_const_t), axs_full)
            θu_new = inv_transform(θt_full)
            if method.saem.suffstats !== nothing && method.saem.q_from_stats !== nothing
                Q_new = method.saem.q_from_stats(s, θu_new, dm)
            else
                Q_new = _saem_Q(dm, batch_infos, θu_new, const_cache, ll_cache, sample_store;
                                serialization=serialization, q_cache=q_cache, anneal_sds=anneal_sds)
            end
        else
            obj_cache = (θ=Ref{Any}(nothing), obj=Ref{Any}(nothing))
            function obj_only(θt, p)
                θt_free = θt isa ComponentArray ? θt : ComponentArray(θt, axs_free)
                θt_vec = θt_free
                use_cache = !(eltype(θt_free) <: ForwardDiff.Dual)
                if use_cache && obj_cache.θ[] !== nothing && length(obj_cache.θ[]) == length(θt_vec)
                    maxdiff = _maxabsdiff(θt_vec, obj_cache.θ[])
                    if maxdiff == 0.0
                        return obj_cache.obj[]
                    end
                end
                T = eltype(θt_free)
                θt_full = ComponentArray(T.(θ_const_t_iter), axs_full_iter)
                for name in free_names_iter
                    setproperty!(θt_full, name, getproperty(θt_free, name))
                end
                θu = inv_transform(θt_full)
                Q = method.saem.suffstats !== nothing && method.saem.q_from_stats !== nothing ?
                    method.saem.q_from_stats(s, θu, dm) :
                    _saem_Q(dm, batch_infos, θu, const_cache, ll_cache, sample_store;
                            serialization=serialization, q_cache=q_cache, anneal_sds=anneal_sds)
                !isfinite(Q) && return T(Inf)
                obj = -Q + _penalty_value(θu, penalty)
                !isfinite(obj) && return T(Inf)
                if use_cache
                    obj_cache.θ[] = copy(θt_vec)
                    obj_cache.obj[] = obj
                end
                return obj
            end

            optf = OptimizationFunction(obj_only, method.adtype)
            lower_t, upper_t = get_bounds_transformed(fe)
            lower_t_free = ComponentArray(NamedTuple{Tuple(free_names_iter)}(Tuple(getproperty(lower_t, n) for n in free_names_iter)))
            upper_t_free = ComponentArray(NamedTuple{Tuple(free_names_iter)}(Tuple(getproperty(upper_t, n) for n in free_names_iter)))
            lower_t_free_vec = collect(lower_t_free)
            upper_t_free_vec = collect(upper_t_free)
            use_bounds = !(all(isinf, lower_t_free_vec) && all(isinf, upper_t_free_vec))
            user_bounds = method.lb !== nothing || method.ub !== nothing
            if user_bounds && !isempty(keys(constants))
                @info "Bounds for constant parameters are ignored." constants=collect(keys(constants))
            end
            if user_bounds
                lb = method.lb
                ub = method.ub
                if lb isa ComponentArray
                    lb = ComponentArray(NamedTuple{Tuple(free_names_iter)}(Tuple(getproperty(lb, n) for n in free_names_iter)))
                end
                if ub isa ComponentArray
                    ub = ComponentArray(NamedTuple{Tuple(free_names_iter)}(Tuple(getproperty(ub, n) for n in free_names_iter)))
                end
                lb = lb === nothing ? lower_t_free_vec : collect(lb)
                ub = ub === nothing ? upper_t_free_vec : collect(ub)
            else
                lb = lower_t_free_vec
                ub = upper_t_free_vec
            end
            use_bounds = use_bounds || user_bounds
            if parentmodule(typeof(method.optimizer)) === OptimizationBBO && !use_bounds
                error("BlackBoxOptim methods require finite bounds. Add lower/upper bounds in @fixedEffects (on transformed scale) or pass them via SAEM(lb=..., ub=...). A quick helper is default_bounds_from_start(dm; margin=...).")
            end
            θ0_init = collect(θt_free)
            prob = use_bounds ? OptimizationProblem(optf, θ0_init; lb=lb, ub=ub) :
                                OptimizationProblem(optf, θ0_init)
            if method.saem.suffstats !== nothing && method.saem.mstep_closed_form !== nothing
                closed_form_custom_used = true
                θu_new = method.saem.mstep_closed_form(s, dm)
                θt_full = transform(θu_new)
                θt_free = ComponentArray(NamedTuple{Tuple(free_names_iter)}(Tuple(getproperty(θt_full, n) for n in free_names_iter)))
                if use_bounds
                    # Closed-form M-step: project onto transformed box bounds.
                    θt_free_vec = collect(θt_free)
                    @inbounds for i in eachindex(θt_free_vec, lb, ub)
                        θt_free_vec[i] = clamp(θt_free_vec[i], lb[i], ub[i])
                    end
                    θt_free = ComponentArray(θt_free_vec, axs_free)
                end
            else
                numeric_mstep_used = true
                sol = Optimization.solve(prob, method.optimizer; method.optim_kwargs...)
                θ_hat_t_raw = sol.u
                θ_hat_t_free = θ_hat_t_raw isa ComponentArray ? θ_hat_t_raw : ComponentArray(θ_hat_t_raw, axs_free)
                θt_free = θ_hat_t_free
            end
            θ_prev_new = copy(θt_free)

            θt_full = ComponentArray(eltype(θt_free).(θ_const_t), axs_full)
            for name in base_free_names
                setproperty!(θt_full, name, getproperty(θt_free, name))
            end
            θu_new = inv_transform(θt_full)
            if method.saem.suffstats !== nothing && method.saem.q_from_stats !== nothing
                Q_new = method.saem.q_from_stats(s, θu_new, dm)
            else
                Q_new = _saem_Q(dm, batch_infos, θu_new, const_cache, ll_cache, sample_store;
                               serialization=serialization, q_cache=q_cache, anneal_sds=anneal_sds)
            end
        end
        Q_new = Q_new == Inf ? T0(Inf) : Q_new

        θ_full_vec = θt_full
        dθ_abs = _maxabsdiff(θ_full_vec, θ_prev_full)
        dθ_rel = dθ_abs / max(T0(1.0), _maxabs(θ_prev_full))
        dQ_abs = abs(Q_new - Q_prev)
        dQ_rel = dQ_abs / max(T0(1.0), abs(Q_prev))

        push!(diag.θ_hist, θ_prev_new)
        push!(diag.Q_hist, Q_new)
        push!(diag.dθ_abs, dθ_abs)
        push!(diag.dθ_rel, dθ_rel)
        push!(diag.dQ_abs, dQ_abs)
        push!(diag.dQ_rel, dQ_rel)
        push!(diag.gamma, T0(γ))
        push!(diag.batches, collect(updated))

        if method.saem.verbose
            @info "SAEM iteration" iter=iter γ=γ Q=Q_new dθ_abs=dθ_abs dθ_rel=dθ_rel dQ_abs=dQ_abs dQ_rel=dQ_rel
        end
        ProgressMeter.next!(progress; showvalues=[(:iter, iter), (:gamma, γ), (:Q, Q_new)])

        θ_prev = θ_prev_new
        θ_prev_full = θ_full_vec
        Q_prev = Q_new

        if dθ_abs <= method.saem.atol_theta && dθ_rel <= method.saem.rtol_theta
            param_streak += 1
        else
            param_streak = 0
        end
        if isfinite(dQ_abs) && isfinite(dQ_rel) &&
           dQ_abs <= method.saem.atol_Q && dQ_rel <= method.saem.rtol_Q
            q_streak += 1
        else
            q_streak = 0
        end
        if iter > method.saem.t0 &&
           param_streak >= method.saem.consecutive_params &&
           q_streak >= method.saem.consecutive_params
            converged = true
            break
        end
    end
    ProgressMeter.finish!(progress)

    θ_hat_t_free = θ_prev isa ComponentArray ? θ_prev : ComponentArray(θ_prev, axs_free)
    θ_hat_t = ComponentArray(eltype(θ_hat_t_free).(θ_const_t), axs_full)
    for name in base_free_names
        setproperty!(θ_hat_t, name, getproperty(θ_hat_t_free, name))
    end
    θ_hat_u = inv_transform(θ_hat_t)

    closed_form_used = closed_form_builtin_used || closed_form_custom_used
    closed_form_sources = Symbol[]
    closed_form_builtin_used && push!(closed_form_sources, :builtin_stats)
    closed_form_custom_used && push!(closed_form_sources, :custom_mstep_closed_form)
    mstep_mode = closed_form_used ? (numeric_mstep_used ? :hybrid : :closed_form_only) : :numeric_only
    notes = (diagnostics=diag,
             closed_form_mstep_used=closed_form_used,
             closed_form_mstep_mode=mstep_mode,
             closed_form_mstep_sources=Tuple(closed_form_sources),
             builtin_stats_mode_requested=method.saem.builtin_stats,
             builtin_stats_mode_effective=builtin_stats_mode,
             builtin_stats_closed_form_eligibility=builtin_cf_elig)
    summary = FitSummary(Q_prev, converged,
                         FitParameters(θ_hat_t, θ_hat_u),
                         notes)
    diagnostics = FitDiagnostics((;), (optimizer=method.optimizer,),
                                 (saem_iters=length(diag.Q_hist), dθ_abs=diag.dθ_abs[end], dQ_abs=diag.dQ_abs[end]),
                                 notes)
    ebe = EBEOptions(method.saem.ebe_optimizer, method.saem.ebe_optim_kwargs, method.saem.ebe_adtype,
                     method.saem.ebe_grad_tol, method.saem.ebe_multistart_n, method.saem.ebe_multistart_k,
                     method.saem.ebe_multistart_max_rounds, method.saem.ebe_multistart_sampling)
    eb_modes = store_eb_modes ? _compute_bstars(dm, θ_hat_u, fixed_maps, ll_cache, ebe, rng;
                                                rescue=method.saem.ebe_rescue,
                                                progress=method.saem.progress,
                                                progress_desc="SAEM Final EBE")[1] : nothing
    result = SAEMResult(nothing, Q_prev, length(diag.Q_hist), nothing, notes, eb_modes)
    return FitResult(method, result, summary, diagnostics, store_data_model ? dm : nothing, args, fit_kwargs)
end
