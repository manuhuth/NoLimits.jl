const _MCEM_MODEL_CACHE = Dict{Tuple{Tuple{Vararg{Symbol}}}, Symbol}()
const _MCEM_MODEL_CACHE_LOCK = ReentrantLock()

function _build_mcem_batch_model(re_names::Vector{Symbol})
    key = (Tuple(re_names),)
    lock(_MCEM_MODEL_CACHE_LOCK)
    if haskey(_MCEM_MODEL_CACHE, key)
        val = _MCEM_MODEL_CACHE[key]
        unlock(_MCEM_MODEL_CACHE_LOCK)
        return val
    end
    unlock(_MCEM_MODEL_CACHE_LOCK)
    fname = gensym(:_mcem_batch_model)
    sample_blocks = Expr(:block)
    re_val_syms = Symbol[]
    for (ri, re) in enumerate(re_names)
        re_q = QuoteNode(re)
        meta_sym = Symbol(re, :_meta)
        levels_sym = Symbol(re, :_levels)
        reps_sym = Symbol(re, :_reps)
        ranges_sym = Symbol(re, :_ranges)
        vals_sym = Symbol(re, :_vals)
        push!(re_val_syms, vals_sym)
        meta_get = :(get_re_info(info)[$ri])
        levels_get = :(get_levels(get_re_map($meta_sym)))
        reps_get = :(get_reps($meta_sym))
        ranges_get = :(get_ranges($meta_sym))
        is_scalar = :(get_is_scalar($meta_sym))
        scalar_block = quote
            local nlvls = length($levels_sym)
            if nlvls > 0
                const_cov = get_const_cov(get_individuals(dm)[$reps_sym[1]])
                dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                dist = getproperty(dists, $re_q)
                if _has_anneal && haskey(anneal_sds, $re_q)
                    dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, $re_q))
                end
                v1 = ($(Symbol(re, :_v1)) ~ dist)
                local $vals_sym = Vector{typeof(v1)}(undef, nlvls)
                $vals_sym[1] = v1
                for j in 2:nlvls
                    const_cov = get_const_cov(get_individuals(dm)[$reps_sym[j]])
                    dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                    dist = getproperty(dists, $re_q)
                    if _has_anneal && haskey(anneal_sds, $re_q)
                        dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, $re_q))
                    end
                    $vals_sym[j] ~ dist
                end
            else
                local $vals_sym = Vector{Float64}(undef, 0)
            end
        end
        vector_block = quote
            local nlvls = length($levels_sym)
            if nlvls > 0
                const_cov = get_const_cov(get_individuals(dm)[$reps_sym[1]])
                dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                dist = getproperty(dists, $re_q)
                if _has_anneal && haskey(anneal_sds, $re_q)
                    dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, $re_q))
                end
                v1 = ($(Symbol(re, :_v1)) ~ dist)
                local $vals_sym = Vector{typeof(v1)}(undef, nlvls)
                $vals_sym[1] = v1
                for j in 2:nlvls
                    const_cov = get_const_cov(get_individuals(dm)[$reps_sym[j]])
                    dists = dists_builder(θ_re, const_cov, model_funs, helpers)
                    dist = getproperty(dists, $re_q)
                    if _has_anneal && haskey(anneal_sds, $re_q)
                        dist = _saem_apply_anneal_dist(dist, getfield(anneal_sds, $re_q))
                    end
                    $vals_sym[j] ~ dist
                end
            else
                local $vals_sym = Vector{Vector{Float64}}(undef, 0)
            end
        end
        push!(sample_blocks.args, :(local $meta_sym = $meta_get))
        push!(sample_blocks.args, :(local $levels_sym = $levels_get))
        push!(sample_blocks.args, :(local $reps_sym = $reps_get))
        push!(sample_blocks.args, :(local $ranges_sym = $ranges_get))
        push!(sample_blocks.args, :(if $is_scalar
            $scalar_block
        else
            $vector_block
        end))
    end
    re_samples_expr = Expr(:call,
        Expr(:curly, :NamedTuple, Expr(:tuple, QuoteNode.(re_names)...)),
        Expr(:tuple, re_val_syms...))

    ex = quote
        @model function $(fname)(dm, info, θ, const_cache, cache, anneal_sds = NamedTuple())
            θ_re = _symmetrize_psd_params(θ, get_fixed(get_model(dm)))
            dists_builder = create_random_effect_distribution(get_random(get_model(dm)))
            model_funs = cache.model_funs
            helpers = cache.helpers
            _has_anneal = !isempty(anneal_sds)

            $sample_blocks
            re_samples = $re_samples_expr

            Tb = eltype(θ)
            for re in $re_names
                vals = getproperty(re_samples, re)
                if !isempty(vals)
                    v1 = vals[1]
                    Tb = v1 isa AbstractVector ? eltype(v1) : typeof(v1)
                    break
                end
            end

            nb = get_n_b(info)
            b = Vector{Tb}(undef, nb)
            for (ri, re) in enumerate($re_names)
                meta = get_re_info(info)[ri]
                levels = get_levels(get_re_map(meta))
                ranges = get_ranges(meta)
                vals = getproperty(re_samples, re)
                for (li, _) in enumerate(levels)
                    r = ranges[li]
                    if get_is_scalar(meta)
                        b[first(r)] = vals[li]
                    else
                        b[r] .= vals[li]
                    end
                end
            end

            ll = zero(Tb)
            for i in get_inds(info)
                η_ind = _build_eta_ind(dm, i, info, b, const_cache, θ)
                lli = _loglikelihood_individual(dm, i, θ, η_ind, cache)
                if !isfinite(lli)
                    ll = -Inf
                    break
                end
                ll += lli
            end
            Turing.@addlogprob! ll
        end
    end
    Core.eval(@__MODULE__, ex)
    lock(_MCEM_MODEL_CACHE_LOCK)
    _MCEM_MODEL_CACHE[key] = fname
    unlock(_MCEM_MODEL_CACHE_LOCK)
    return fname
end

function NoLimits._mcem_sample_batch_turing(dm, info, θ, const_cache, cache, sampler,
        turing_kwargs, rng,
        re_names, warm_start, last_params;
        anneal_sds::NamedTuple = NamedTuple(),
        outer_iter::Int = 1)
    nb = get_n_b(info)
    if nb == 0
        return (zeros(eltype(θ), 0, 0), Float64[], eltype(θ)[])
    end
    fname = _build_mcem_batch_model(re_names)
    model_fn = Base.invokelatest(getfield, @__MODULE__, fname)
    model = Base.invokelatest(model_fn, dm, info, θ, const_cache, cache, anneal_sds)
    n_samples = get(turing_kwargs, :n_samples, 100)
    n_adapt = get(turing_kwargs, :n_adapt, 50)
    tkwargs = Base.structdiff(turing_kwargs, (n_samples = 0, n_adapt = 0))
    haskey(tkwargs, :progress) || (tkwargs = merge(tkwargs, (progress = false,)))
    haskey(tkwargs, :verbose) || (tkwargs = merge(tkwargs, (verbose = false,)))
    # Turing ≥ 0.45 defaults to FlexiChains; `_extract_b_samples` consumes the chain via
    # the MCMCChains API (`names`/`Array`), so force an MCMCChains.Chains result here.
    tkwargs = merge(tkwargs, (chain_type = MCMCChains.Chains,))
    chain = if warm_start && last_params isa NamedTuple && !isempty(last_params)
        init = DynamicPPL.InitFromParams(last_params)
        Base.invokelatest(Turing.sample, rng, model, sampler, n_samples;
            adapt = n_adapt, initial_params = init, tkwargs...)
    else
        Base.invokelatest(Turing.sample, rng, model, sampler, n_samples;
            adapt = n_adapt, tkwargs...)
    end
    samples, lastp, lastb = _extract_b_samples(chain, info, re_names)
    samples = _filter_b_samples_by_prior(dm, info, θ, const_cache, cache, samples)
    if size(samples, 2) == 0
        return (zeros(eltype(θ), nb, 0), lastp, zeros(eltype(θ), nb))
    end
    lastb = samples[:, end]
    return (samples, lastp, lastb)
end
