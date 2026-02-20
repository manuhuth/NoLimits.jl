using MCMCChains
using Random
using Statistics

function _chain_keys_for_free(fe::FixedEffects, free_names::Vector{Symbol})
    all_names = get_names(fe)
    specs = get_transforms(fe).forward.specs
    spec_map = Dict{Symbol, TransformSpec}()
    for i in eachindex(all_names)
        spec_map[all_names[i]] = specs[i]
    end

    θ0_u = get_θ0_untransformed(fe)
    out = String[]
    for name in free_names
        v = getproperty(θ0_u, name)
        spec = spec_map[name]
        if v isa Number
            push!(out, string(name))
        elseif spec.kind == :expm && v isa AbstractMatrix
            n = size(v, 1)
            for j in 1:n
                for i in 1:j
                    push!(out, string(name, "[", i, ",", j, "]"))
                end
            end
        else
            for idx in CartesianIndices(v)
                idx_txt = join(Tuple(idx), ",")
                push!(out, string(name, "[", idx_txt, "]"))
            end
        end
    end
    return out
end

@inline function _lookup_chain_index(idx_map::Dict{String, Int}, key::String)
    haskey(idx_map, key) && return idx_map[key]
    key2 = replace(key, "," => ", ")
    haskey(idx_map, key2) && return idx_map[key2]
    key3 = replace(key, " " => "")
    haskey(idx_map, key3) && return idx_map[key3]
    return 0
end

function _compute_uq_chain(res::FitResult;
                           level::Float64,
                           constants::Union{Nothing, NamedTuple},
                           mcmc_warmup::Union{Nothing, Int},
                           mcmc_draws::Union{Nothing, Int},
                           default_draws::Int,
                           rng::AbstractRNG)
    method = get_method(res)
    (method isa MCMC || method isa VI) || error("Chain UQ requires an MCMC or VI fit result.")

    dm = get_data_model(res)
    dm === nothing && error("This fit result does not store a DataModel; pass store_data_model=true when fitting.")

    constants_use = constants === nothing ? _fit_kw(res, :constants, NamedTuple()) : constants
    fe = dm.model.fixed.fixed
    free_names = _free_fixed_names(fe, constants_use)
    isempty(free_names) && error("No free fixed effects are available for UQ after applying constants.")

    active_mask = _active_mask_for_free(fe, free_names)
    active_idx = findall(identity, active_mask)
    isempty(active_idx) && error("No UQ-eligible fixed-effect coordinates found. Mark parameters with calculate_se=true and ensure they are not fixed via constants.")

    free_flat_names = _flat_names_for_free(fe, free_names)
    active_names = free_flat_names[active_idx]
    chain_keys = _chain_keys_for_free(fe, free_names)
    length(chain_keys) == length(active_mask) || error("Internal UQ error: chain-key layout does not match fixed-effect layout.")
    active_keys = chain_keys[active_idx]

    draws_n = Matrix{Float64}(undef, 0, 0)
    diag = NamedTuple()
    if method isa MCMC
        chain = get_chain(res)
        names = MCMCChains.names(chain, :parameters)
        idx_map = Dict{String, Int}()
        for (i, n) in enumerate(names)
            idx_map[string(n)] = i
        end

        arr = Array(chain)
        if ndims(arr) == 2
            arr = reshape(arr, size(arr, 1), size(arr, 2), 1)
        end
        ndims(arr) == 3 || error("Unexpected MCMC chain layout with ndims=$(ndims(arr)).")
        n_iter, _, n_chains = size(arr)

        warmup = mcmc_warmup === nothing ? _uq_mcmc_warmup(res) : Int(mcmc_warmup)
        warmup = clamp(warmup, 0, max(0, n_iter - 1))
        first_keep = warmup + 1
        first_keep <= n_iter || error("No post-warmup MCMC samples are available for UQ.")

        draw_pairs = Tuple{Int, Int}[]
        sizehint!(draw_pairs, (n_iter - warmup) * n_chains)
        for c in 1:n_chains
            for it in first_keep:n_iter
                push!(draw_pairs, (it, c))
            end
        end
        isempty(draw_pairs) && error("No MCMC samples available after warmup.")
        available_draws = length(draw_pairs)

        if mcmc_draws !== nothing
            mcmc_draws > 0 || error("mcmc_draws must be positive.")
            if mcmc_draws < length(draw_pairs)
                perm = randperm(rng, length(draw_pairs))
                draw_pairs = draw_pairs[perm[1:mcmc_draws]]
            end
        end
        used_draws = length(draw_pairs)
        requested_draws = mcmc_draws === nothing ? available_draws : Int(mcmc_draws)
        @info "MCMC UQ draws" requested=requested_draws available=available_draws used=used_draws warmup=warmup n_iter=n_iter n_chains=n_chains

        draws_n = Matrix{Float64}(undef, length(draw_pairs), length(active_keys))
        for (i, (it, ch)) in enumerate(draw_pairs)
            for (j, key) in enumerate(active_keys)
                idx = _lookup_chain_index(idx_map, key)
                idx == 0 && error("MCMC chain is missing fixed-effect coordinate $(key). Available chain names include $(collect(keys(idx_map))[1:min(end, 10)]).")
                draws_n[i, j] = Float64(arr[it, idx, ch])
            end
        end
        diag = (;
            chain_scale=:natural,
            warmup=warmup,
            n_samples=size(draws_n, 1),
            requested_draws=requested_draws,
            available_draws=available_draws,
            used_draws=used_draws,
            n_iter=n_iter,
            n_chains=n_chains,
            n_active_parameters=length(active_idx),
            source=:mcmc_chain,
        )
    else
        requested_draws = mcmc_draws === nothing ? Int(default_draws) : Int(mcmc_draws)
        requested_draws > 0 || error("mcmc_draws must be positive.")
        vi_draws = sample_posterior(res; n_draws=requested_draws, rng=rng, return_names=true)
        raw_draws = vi_draws.draws
        coord_names = vi_draws.names
        size(raw_draws, 1) >= 1 || error("VI posterior sampling returned no draws.")

        idx_map = Dict{String, Int}()
        for (i, n) in enumerate(coord_names)
            idx_map[string(n)] = i
        end

        draws_n = Matrix{Float64}(undef, size(raw_draws, 1), length(active_keys))
        for (j, key) in enumerate(active_keys)
            idx = _lookup_chain_index(idx_map, key)
            idx == 0 && error("VI posterior is missing fixed-effect coordinate $(key). Available coordinates include $(collect(keys(idx_map))[1:min(end, 10)]).")
            draws_n[:, j] .= Float64.(raw_draws[:, idx])
        end

        n_draws_used = size(draws_n, 1)
        @info "VI UQ draws" requested=requested_draws used=n_draws_used n_active_parameters=length(active_idx)
        diag = (;
            chain_scale=:natural,
            warmup=0,
            n_samples=n_draws_used,
            requested_draws=requested_draws,
            available_draws=n_draws_used,
            used_draws=n_draws_used,
            n_iter=missing,
            n_chains=1,
            n_active_parameters=length(active_idx),
            source=:vi_posterior,
        )
    end

    est_n = vec(mean(draws_n; dims=1))
    intervals_n = _intervals_from_draws(draws_n, level)
    Vn = _cov_from_draws(draws_n)

    return UQResult(
        :chain,
        _method_symbol(method),
        active_names,
        copy(est_n),
        copy(est_n),
        intervals_n,
        intervals_n,
        copy(Vn),
        copy(Vn),
        copy(draws_n),
        copy(draws_n),
        diag
    )
end
