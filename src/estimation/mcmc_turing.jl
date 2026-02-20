export MCMC
export MCMCResult

using Turing
using DynamicPPL
using SciMLBase
using ComponentArrays
using Distributions
using Random
using DynamicPPL

const _MCMC_MODEL_CACHE = Dict{Tuple{Tuple{Vararg{Symbol}}, Tuple{Vararg{Symbol}}}, Symbol}()
const _MCMC_MODEL_CACHE_RE = Dict{Tuple{Tuple{Vararg{Symbol}}, Tuple{Vararg{Symbol}}, Tuple{Vararg{Symbol}}}, Symbol}()

struct _MCMCReMeta{L, M, R}
    levels::L
    level_to_index::M
    reps::R
    dim::Int
    is_scalar::Bool
end

function _warn_if_scaled_params(fe::FixedEffects; method_name::AbstractString="MCMC")
    specs = get_transforms(fe).forward.specs
    ignored = Symbol[]
    for spec in specs
        if spec.kind != :identity
            push!(ignored, spec.name)
        end
    end
    isempty(ignored) || @debug "$(method_name) uses priors on the natural scale; parameter scale settings are ignored during sampling." ignored_parameters=ignored
    return nothing
end

"""
    MCMC(; sampler, turing_kwargs, adtype, progress) <: FittingMethod

Bayesian sampling via Turing.jl for models with or without random effects.
All free fixed effects and random effects must have prior distributions.

# Keyword Arguments
- `sampler`: Turing-compatible sampler. Defaults to `NUTS(0.75)`.
- `turing_kwargs::NamedTuple = NamedTuple()`: keyword arguments forwarded to `Turing.sample`
  (e.g. `n_samples`, `n_adapt`).
- `adtype`: automatic-differentiation backend. Defaults to `AutoForwardDiff()`.
- `progress::Bool = false`: whether to display a progress bar during sampling.
"""
struct MCMC{S, K, A, P} <: FittingMethod
    sampler::S
    turing_kwargs::K
    adtype::A
    progress::P
end

MCMC(; sampler=Turing.NUTS(0.75),
     turing_kwargs=NamedTuple(),
     adtype=Turing.AutoForwardDiff(),
     progress=false) = MCMC(sampler, turing_kwargs, adtype, progress)

"""
    MCMCResult{C, S, A, N, O} <: MethodResult

Method-specific result from a [`MCMC`](@ref) fit. Stores the MCMCChains chain,
sampler, number of samples, optional notes, and observed data columns.
"""
struct MCMCResult{C, S, A, N, O} <: MethodResult
    chain::C
    sampler::S
    n_samples::A
    notes::N
    observed::O
end

get_chain(res::MCMCResult) = res.chain

@inline function _mcmc_sampler_kind(sampler)
    sampler isa NUTS && return :nuts
    sampler isa HMC && return :hmc
    sampler isa MH && return :mh
    return :other
end

@inline function _mcmc_sampler_defaults(sampler)
    kind = _mcmc_sampler_kind(sampler)
    if kind == :mh
        return (n_samples=2500, n_adapt=0)
    elseif kind == :hmc
        return (n_samples=1500, n_adapt=750)
    else
        return (n_samples=1000, n_adapt=500)
    end
end

function _build_turing_model(fixed_names::Vector{Symbol}, free_names::Vector{Symbol})
    key = (Tuple(fixed_names), Tuple(free_names))
    if haskey(_MCMC_MODEL_CACHE, key)
        return _MCMC_MODEL_CACHE[key]
    end
    fname = gensym(:_mcmc_model)
    assigns = [:( $(n) ~ getfield(priors, $(QuoteNode(n))) ) for n in free_names]
    free_nt = Expr(:call,
                   Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(free_names)...)),
                   Expr(:tuple, free_names...))
    θ_nt = :(merge($free_nt, constants))
    val_exprs = [:(getfield($θ_nt, $(QuoteNode(n)))) for n in fixed_names]
    nt_expr = Expr(:call,
                   Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(fixed_names)...)),
                   Expr(:tuple, val_exprs...))
    θ_expr = :(ComponentArray($nt_expr))
    ex = quote
        @model function $(fname)(dm, cache, serialization, priors, constants)
            $(assigns...)
            θ = $θ_expr
            θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
            ll = loglikelihood(dm, θ_re, ComponentArray(); cache=cache, serialization=serialization)
            Turing.@addlogprob! ll
        end
    end
    Core.eval(@__MODULE__, ex)
    _MCMC_MODEL_CACHE[key] = fname
    return fname
end

function _build_turing_model_re(fixed_names::Vector{Symbol}, free_names::Vector{Symbol}, re_names::Vector{Symbol})
    key = (Tuple(fixed_names), Tuple(free_names), Tuple(re_names))
    if haskey(_MCMC_MODEL_CACHE_RE, key)
        return _MCMC_MODEL_CACHE_RE[key]
    end
    fname = gensym(:_mcmc_model_re)
    assigns = [:( $(n) ~ getfield(priors, $(QuoteNode(n))) ) for n in free_names]
    free_nt = Expr(:call,
                   Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(free_names)...)),
                   Expr(:tuple, free_names...))
    θ_nt = :(merge($free_nt, constants))
    val_exprs = [:(getfield($θ_nt, $(QuoteNode(n)))) for n in fixed_names]
    nt_expr = Expr(:call,
                   Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(fixed_names)...)),
                   Expr(:tuple, val_exprs...))
    θ_expr = :(ComponentArray($nt_expr))
    sample_blocks = Expr(:block)
    re_val_syms = Symbol[]
    for re in re_names
        re_q = QuoteNode(re)
        meta_sym = Symbol(re, :_meta)
        levels_sym = Symbol(re, :_levels)
        reps_sym = Symbol(re, :_reps)
        vals_sym = Symbol(re, :_vals)
        push!(re_val_syms, vals_sym)
        meta_get = :(getproperty(re_meta, $re_q))
        levels_get = :(getproperty($meta_sym, :levels))
        reps_get = :(getproperty($meta_sym, :reps))
        is_scalar = :(getproperty($meta_sym, :is_scalar))
        dim = :(getproperty($meta_sym, :dim))
        scalar_block = quote
            local $vals_sym = Vector{T}(undef, length($levels_sym))
            for j in eachindex($levels_sym)
                const_cov = const_covs[$reps_sym[j]]
                dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                dist = getproperty(dists, $re_q)
                $vals_sym[j] ~ dist
            end
        end
        vector_block = quote
            local $vals_sym = Vector{Vector{T}}(undef, length($levels_sym))
            for j in eachindex($levels_sym)
                const_cov = const_covs[$reps_sym[j]]
                dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                dist = getproperty(dists, $re_q)
                $vals_sym[j] ~ dist
            end
        end
        push!(sample_blocks.args, :(local $meta_sym = $meta_get))
        push!(sample_blocks.args, :(local $levels_sym = $levels_get))
        push!(sample_blocks.args, :(local $reps_sym = $reps_get))
        push!(sample_blocks.args, :(if $is_scalar || $dim == 1
                                         $scalar_block
                                     else
                                         $vector_block
                                     end))
    end

    re_samples_expr = Expr(:call,
                           Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(re_names)...)),
                           Expr(:tuple, re_val_syms...))

    ex = quote
        @model function $(fname)(dm, cache, serialization, priors, constants, re_names, re_meta, fixed_maps, const_covs)
            $(assigns...)
            θ = $θ_expr
            θ_re = _symmetrize_psd_params(θ, dm.model.fixed.fixed)
            T = eltype(θ)
            dists_builder = get_create_random_effect_distribution(dm.model.random.random)
            model_funs = get_model_funs(dm.model)
            helpers = get_helper_funs(dm.model)

            $sample_blocks
            re_samples = $re_samples_expr

            η_vec = Vector{ComponentArray}(undef, length(dm.individuals))
            for i in eachindex(dm.individuals)
                nt_pairs = Pair{Symbol, Any}[]
                ind = dm.individuals[i]
                for re in re_names
                    meta = getproperty(re_meta, re)
                    g = getfield(ind.re_groups, re)
                    fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : nothing
                    levels = meta.levels
                    lvl_to_idx = meta.level_to_index
                    samples = getproperty(re_samples, re)
                    scalar_like = meta.is_scalar || meta.dim == 1
                    if g isa AbstractVector
                        if length(g) == 1
                            gv = g[1]
                            if fixed !== nothing && haskey(fixed, gv)
                                v = fixed[gv]
                            else
                                idx = get(lvl_to_idx, gv, 0)
                                idx == 0 && error("Missing random effect value for $(re) level $(gv).")
                                v = samples[idx]
                            end
                            if scalar_like
                                push!(nt_pairs, re => v)
                            else
                                push!(nt_pairs, re => (v isa AbstractVector ? collect(v) : [v]))
                            end
                        else
                            if scalar_like
                                vals = [begin
                                    if fixed !== nothing && haskey(fixed, gv)
                                        fixed[gv]
                                    else
                                        idx = get(lvl_to_idx, gv, 0)
                                        idx == 0 && error("Missing random effect value for $(re) level $(gv).")
                                        samples[idx]
                                    end
                                end for gv in g]
                            else
                                vals = [begin
                                    if fixed !== nothing && haskey(fixed, gv)
                                        v = fixed[gv]
                                    else
                                        idx = get(lvl_to_idx, gv, 0)
                                        idx == 0 && error("Missing random effect value for $(re) level $(gv).")
                                        v = samples[idx]
                                    end
                                    v isa AbstractVector ? collect(v) : [v]
                                end for gv in g]
                            end
                            push!(nt_pairs, re => vals)
                        end
                    else
                        if fixed !== nothing && haskey(fixed, g)
                            v = fixed[g]
                        else
                            idx = get(lvl_to_idx, g, 0)
                            idx == 0 && error("Missing random effect value for $(re) level $(g).")
                            v = samples[idx]
                        end
                        if scalar_like
                            push!(nt_pairs, re => v)
                        else
                            push!(nt_pairs, re => (v isa AbstractVector ? collect(v) : [v]))
                        end
                    end
                end
                η_vec[i] = ComponentArray(NamedTuple(nt_pairs))
            end

            ll = loglikelihood(dm, θ_re, η_vec; cache=cache, serialization=serialization)
            Turing.@addlogprob! ll
        end
    end
    Core.eval(@__MODULE__, ex)
    _MCMC_MODEL_CACHE_RE[key] = fname
    return fname
end

function _set_turing_adbackend!(adtype)
    if isdefined(Turing, :setadbackend!)
        Turing.setadbackend!(adtype)
    elseif isdefined(Turing, :setadbackend)
        Turing.setadbackend(adtype)
    end
    return nothing
end


function _fit_model(dm::DataModel, method::MCMC, args...;
                    constants::NamedTuple=NamedTuple(),
                    constants_re::NamedTuple=NamedTuple(),
                    penalty::NamedTuple=NamedTuple(),
                    ode_args::Tuple=(),
                    ode_kwargs::NamedTuple=NamedTuple(),
                    serialization::SciMLBase.EnsembleAlgorithm=EnsembleSerial(),
                    rng::AbstractRNG=Random.default_rng(),
                    theta_0_untransformed::Union{Nothing, ComponentArray}=nothing,
                    store_data_model::Bool=true)
    fit_kwargs = (constants=constants,
                  constants_re=constants_re,
                  penalty=penalty,
                  ode_args=ode_args,
                  ode_kwargs=ode_kwargs,
                  serialization=serialization,
                  rng=rng,
                  theta_0_untransformed=theta_0_untransformed,
                  store_data_model=store_data_model)
    re_names = get_re_names(dm.model.random.random)
    isempty(keys(penalty)) || error("MCMC does not support penalty terms. Use priors and MAP instead.")

    fe = dm.model.fixed.fixed
    _warn_if_scaled_params(fe; method_name="MCMC")
    priors = get_priors(fe)
    fixed_names = get_names(fe)
    fixed_set = Set(fixed_names)
    for name in keys(constants)
        name in fixed_set || error("Unknown constant parameter $(name).")
    end
    free_names = [n for n in fixed_names if !(n in keys(constants))]
    if isempty(free_names) && isempty(re_names)
        error("MCMC requires at least one sampled parameter. For fixed-effects-only models, leave at least one fixed effect free.")
    end
    for name in free_names
        haskey(priors, name) || error("MCMC requires priors on all free fixed effects. Missing prior for $(name).")
        getfield(priors, name) isa Priorless && error("MCMC requires priors on all free fixed effects. Priorless for $(name).")
    end
    if theta_0_untransformed !== nothing
        for n in fixed_names
            hasproperty(theta_0_untransformed, n) || error("theta_0_untransformed is missing parameter $(n).")
        end
    end

    cache = serialization isa SciMLBase.EnsembleThreads ?
            build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, nthreads=Threads.maxthreadid(), force_saveat=true) :
            build_ll_cache(dm; ode_args=ode_args, ode_kwargs=ode_kwargs, force_saveat=true)

    free_names_t = Tuple(free_names)
    θ_template = get_θ0_untransformed(fe)
    priors_nt = NamedTuple{free_names_t}(Tuple(getfield(priors, n) for n in free_names))
    model = nothing
    if isempty(re_names)
        fname = _build_turing_model(fixed_names, free_names)
        model_fn = Base.invokelatest(getfield, @__MODULE__, fname)
        model = Base.invokelatest(model_fn, dm, cache, serialization, priors_nt, constants)
    else
        fixed_maps = _normalize_constants_re(dm, constants_re)
        # Validate constants_re shape and dimensions before launching Turing.
        _build_constants_cache(dm, fixed_maps)
        const_covs = [ind.const_cov for ind in dm.individuals]
        dists_builder = get_create_random_effect_distribution(dm.model.random.random)
        model_funs = get_model_funs(dm.model)
        helpers = get_helper_funs(dm.model)
        re_pairs = Pair{Symbol, Any}[]
        for re in re_names
            reps_map = Dict{Any, Int}()
            for (i, ind) in enumerate(dm.individuals)
                g = getfield(ind.re_groups, re)
                if g isa AbstractVector
                    for gv in g
                        haskey(reps_map, gv) || (reps_map[gv] = i)
                    end
                else
                    haskey(reps_map, g) || (reps_map[g] = i)
                end
            end
            levels_all = getfield(dm.re_group_info.values, re)
            fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
            levels_free = Any[]
            reps = Int[]
            for lvl in levels_all
                haskey(fixed, lvl) && continue
                push!(levels_free, lvl)
                push!(reps, reps_map[lvl])
            end
            level_to_index = Dict{Any, Int}()
            for (i, lvl) in enumerate(levels_free)
                level_to_index[lvl] = i
            end
            dim = 0
            is_scalar = true
            if !isempty(levels_free)
                rep = reps[1]
                θ0_re = _symmetrize_psd_params(get_θ0_untransformed(fe), fe)
                dists = dists_builder(θ0_re, const_covs[rep], model_funs, helpers)
                dist = getproperty(dists, re)
                is_scalar = dist isa Distributions.UnivariateDistribution
                dim = is_scalar ? 1 : length(dist)
                dim == 0 && error("Random effect $(re) has zero dimension.")
            end
            push!(re_pairs, re => _MCMCReMeta(levels_free, level_to_index, reps, dim, is_scalar))
        end
        re_meta = NamedTuple(re_pairs)
        fname = _build_turing_model_re(fixed_names, free_names, re_names)
        model_fn = Base.invokelatest(getfield, @__MODULE__, fname)
        model = Base.invokelatest(model_fn, dm, cache, serialization, priors_nt, constants, re_names, re_meta, fixed_maps, const_covs)
    end
    f_old = model.f
    f_wrap = (args...)->Base.invokelatest(f_old, args...)
    miss = typeof(model).parameters[4]
    threaded = typeof(model).parameters[8]
    model = DynamicPPL.Model{threaded,miss}(f_wrap, model.args, model.defaults, model.context)
    sampler = method.sampler
    sampler_defaults = _mcmc_sampler_defaults(sampler)
    n_samples = get(method.turing_kwargs, :n_samples, sampler_defaults.n_samples)
    n_adapt = get(method.turing_kwargs, :n_adapt, sampler_defaults.n_adapt)
    turing_kwargs = Base.structdiff(method.turing_kwargs, (n_samples=0, n_adapt=0))
    haskey(turing_kwargs, :progress) || (turing_kwargs = merge(turing_kwargs, (progress=method.progress,)))
    haskey(turing_kwargs, :verbose) || (turing_kwargs = merge(turing_kwargs, (verbose=false,)))
    if theta_0_untransformed !== nothing && !haskey(turing_kwargs, :initial_params)
        init_nt = NamedTuple{free_names_t}(Tuple(getproperty(theta_0_untransformed, n) for n in free_names))
        turing_kwargs = merge(turing_kwargs, (initial_params=DynamicPPL.InitFromParams(init_nt),))
    elseif theta_0_untransformed !== nothing && haskey(turing_kwargs, :initial_params)
        @debug "theta_0_untransformed ignored because turing_kwargs already specifies initial_params."
    end

    _set_turing_adbackend!(method.adtype)
    chain = Turing.sample(rng, model, sampler, n_samples; adapt=n_adapt, turing_kwargs...)

    obs = dm.df[:, dm.config.obs_cols]
    summary = FitSummary(NaN, missing,
                         FitParameters(ComponentArray(), ComponentArray()),
                         NamedTuple())
    diagnostics = FitDiagnostics((;), (sampler=sampler,), (n_samples=n_samples, n_adapt=n_adapt), NamedTuple())
    result = MCMCResult(chain, sampler, n_samples, NamedTuple(), obs)
    return FitResult(method, result, summary, diagnostics, store_data_model ? dm : nothing, args, fit_kwargs)
end
