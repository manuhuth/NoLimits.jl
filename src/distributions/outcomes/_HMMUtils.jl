@inline function _hmm_logsumexp(xs::AbstractVector)
    isempty(xs) && return -Inf
    m = xs[1]
    @inbounds for i in 2:length(xs)
        m = max(m, xs[i])
    end
    isfinite(m) || return m
    s = zero(m)
    # Ignore terms that are far below the max in value space. For Dual numbers,
    # these underflow-scale terms can carry non-finite sensitivities (e.g. -Inf)
    # that are numerically irrelevant to the value but can poison gradients.
    cutoff = -700.0
    @inbounds for x in xs
        δ = x - m
        δ > cutoff || continue
        s += exp(δ)
    end
    return m + log(s)
end

@inline _is_state_set_observation(y) =
    (y isa AbstractVector || y isa Tuple || y isa AbstractSet)

function _hmm_compatible_state_indices(state_labels::AbstractVector, y)
    idx = findfirst(==(y), state_labels)
    idx === nothing || return [idx]

    if _is_state_set_observation(y)
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
    return Int[]
end
