@inline _omm_is_state_set_observation(y) =
    (y isa AbstractVector || y isa Tuple || y isa AbstractSet)

@inline _omm_is_observed_markov_dist(::Any) = false

function _omm_compatible_state_indices(state_labels::AbstractVector, y)
    idxs = Int[]
    for yi in y
        idx = findfirst(==(yi), state_labels)
        idx === nothing || push!(idxs, idx)
    end
    if !isempty(idxs)
        sort!(idxs)
        unique!(idxs)
    end
    return idxs
end

function _omm_scalar_observation_index(state_labels::AbstractVector, y)
    if _omm_is_state_set_observation(y)
        error(
            "Set-valued observation $(repr(y)) detected for a non-coarsed observed Markov model. " *
            "Wrap the distribution with coarsed(...), e.g. `y ~ coarsed(dist)`."
        )
    end
    return findfirst(==(y), state_labels)
end

function _omm_coarsed_observation_indices(state_labels::AbstractVector, y)
    y isa AbstractVector || error(
        "coarsed(...) expects each non-missing observation to be an AbstractVector of compatible states. " *
        "Got $(typeof(y)) with value $(repr(y))."
    )
    return _omm_compatible_state_indices(state_labels, y)
end
