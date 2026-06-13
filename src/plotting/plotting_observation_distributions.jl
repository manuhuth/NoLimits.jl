export plot_observation_distributions

using Distributions
using Random

function _resolve_individuals(dm::DataModel, individuals_idx)
    n = length(dm.individuals)
    if individuals_idx === nothing
        return [1]
    end
    ids = individuals_idx isa AbstractVector ? collect(individuals_idx) : [individuals_idx]
    if all(x -> x isa Integer && 1 <= x <= n, ids)
        return Int.(ids)
    end
    out = Int[]
    for id in ids
        haskey(dm.id_index, id) || error("Unknown individual id $(id).")
        push!(out, dm.id_index[id])
    end
    return out
end

function _resolve_obs_rows(obs_rows, obs_rows_all)
    if obs_rows === nothing
        return collect(1:length(obs_rows_all))
    end
    idxs = obs_rows isa AbstractVector ? collect(obs_rows) : [obs_rows]
    for idx in idxs
        1 <= idx <= length(obs_rows_all) || error("obs_rows index $(idx) out of bounds.")
    end
    return idxs
end

function _resolve_observables(dm::DataModel, observables)
    obs = get_formulas_meta(dm.model.formulas.formulas).obs_names
    if observables === nothing
        length(obs) > 1 &&
            @warn "Multiple observables found; using the first." observable=obs[1]
        return [obs[1]]
    end
    obs_list = observables isa AbstractVector ? collect(observables) : [observables]
    for o in obs_list
        o in obs || error("Observable $(o) not found. Available: $(obs).")
    end
    return obs_list
end

function _mean_pmf_support(dists::Vector{Distribution}, coverage::Float64)
    vals_all = Int[]
    for d in dists
        grid = _density_grid_discrete(d, coverage)
        grid === nothing && continue
        append!(vals_all, grid.vals)
    end
    isempty(vals_all) && return Int[]
    return sort(unique(vals_all))
end
