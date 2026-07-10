export ParameterComparison
export compare_parameters

"""
    ParameterComparison

Side-by-side comparison of fixed-effect parameter estimates from two or more
fitted models, created by [`compare_parameters`](@ref). The rows are the union
of the reported parameters across all models, in declaration order. A parameter
that is absent from a given model (for example a Gaussian scale that a different
random-effect specification does not have) is shown as `-` in that model's
column. Displayed via `Base.show` as an aligned table.

Fields:
- `labels`: column label per model.
- `scale`: scale on which the estimates are reported (`:natural` or `:transformed`).
- `parameters`: ordered parameter names (table rows).
- `estimates`: matrix of estimates with one row per parameter and one column per
  model, holding `nothing` where a parameter is absent from a model.
- `roles`: parameter role (e.g. `"RE distribution"`) taken from the first model
  that reports the parameter.
- `notes`: any notes accumulated while extracting the estimates.
"""
struct ParameterComparison
    labels::Vector{String}
    scale::Symbol
    parameters::Vector{Symbol}
    estimates::Matrix{Union{Nothing, Float64}}
    roles::Vector{String}
    notes::Vector{String}
end

function _pc_labels(labels, n::Int)
    if labels === nothing
        return ["model $(j)" for j in 1:n]
    end
    length(labels) == n ||
        error("compare_parameters: length(labels) ($(length(labels))) must equal the number of models ($(n)).")
    return [string(l) for l in labels]
end

"""
    compare_parameters(fits::FitResult...; labels=nothing, scale=:natural,
                       include_non_se=false, common_only=false) -> ParameterComparison
    compare_parameters((label => fit)...; kwargs...) -> ParameterComparison

Compare fixed-effect estimates from two or more fitted models in a single
aligned table. Parameters are matched by name across models and reported on a
common `scale` (`:natural` by default). Parameters present in only some models
are shown as `-` in the columns where they are absent, unless `common_only=true`
restricts the table to parameters shared by every model.

By default only standard-error-eligible parameters are listed (matching
`summarize`); pass `include_non_se=true` to include the rest. Column
labels default to `"model 1"`, `"model 2"`, and so on, and can be set either via
the `labels` keyword or by passing `label => fit` pairs.

# Examples
```julia
compare_parameters(fit_gauss, fit_flow; labels = ["Gaussian", "flow"])
compare_parameters("Gaussian" => fit_gauss, "flow" => fit_flow)
```
"""
function compare_parameters(f1::FitResult, frest::FitResult...;
        labels = nothing,
        scale::Symbol = :natural,
        include_non_se::Bool = false,
        common_only::Bool = false)
    fits = (f1, frest...)
    n = length(fits)
    scale = _fq_scale_symbol(scale)
    lbls = _pc_labels(labels, n)

    per_model = Vector{Vector{NamedTuple}}(undef, n)
    notes = String[]
    for (j, res) in enumerate(fits)
        rows, _, _, nts = _fq_fit_parameter_rows(
            res; scale = scale, include_non_se = include_non_se)
        per_model[j] = rows
        append!(notes, nts)
    end

    params = Symbol[]
    seen = Set{Symbol}()
    role_of = Dict{Symbol, String}()
    for rows in per_model
        for r in rows
            if !(r.parameter in seen)
                push!(seen, r.parameter)
                push!(params, r.parameter)
                role_of[r.parameter] = r.role
            end
        end
    end

    lookups = [Dict{Symbol, Float64}(r.parameter => Float64(r.estimate) for r in rows)
               for rows in per_model]
    if common_only
        params = filter(p -> all(lk -> haskey(lk, p), lookups), params)
    end

    estimates = Matrix{Union{Nothing, Float64}}(undef, length(params), n)
    for (i, p) in enumerate(params)
        for j in 1:n
            estimates[i, j] = get(lookups[j], p, nothing)
        end
    end

    roles = [get(role_of, p, "General/Outcome") for p in params]
    return ParameterComparison(lbls, scale, params, estimates, roles, unique(notes))
end

function compare_parameters(p1::Pair, prest::Pair...;
        scale::Symbol = :natural,
        include_non_se::Bool = false,
        common_only::Bool = false)
    pairs = (p1, prest...)
    all(p -> last(p) isa FitResult, pairs) ||
        error("compare_parameters: each pair must map a label to a FitResult.")
    labels = [string(first(p)) for p in pairs]
    fits = FitResult[last(p) for p in pairs]
    return compare_parameters(fits...; labels = labels, scale = scale,
        include_non_se = include_non_se, common_only = common_only)
end

function Base.show(io::IO, ::MIME"text/plain", c::ParameterComparison)
    n = length(c.labels)
    np = length(c.parameters)
    name_w = max(
        length("parameter"), np == 0 ? 0 : maximum(length(string(p)) for p in c.parameters))

    cells = Matrix{String}(undef, np, n)
    for i in 1:np
        for j in 1:n
            cells[i, j] = _fq_fmt_num(c.estimates[i, j])
        end
    end
    col_w = Vector{Int}(undef, n)
    for j in 1:n
        w = max(length(c.labels[j]), 8)
        for i in 1:np
            w = max(w, length(cells[i, j]))
        end
        col_w[j] = w
    end
    row_w = 2 + name_w + sum(2 + w for w in col_w)

    println(io, "ParameterComparison")
    println(io, repeat("=", row_w))
    print(io, "  ", rpad("parameter", name_w))
    for j in 1:n
        print(io, "  ", lpad(c.labels[j], col_w[j]))
    end
    println(io)
    println(io, repeat("-", row_w))
    if np == 0
        println(io, "  (no parameters in common)")
    else
        for i in 1:np
            print(io, "  ", rpad(string(c.parameters[i]), name_w))
            for j in 1:n
                print(io, "  ", lpad(cells[i, j], col_w[j]))
            end
            println(io)
        end
    end
    if !isempty(c.notes)
        println(io)
        _print_key_values(
            io, "Notes", [string("note ", i) => c.notes[i] for i in eachindex(c.notes)])
    end
end

Base.show(io::IO, c::ParameterComparison) = show(io, MIME"text/plain"(), c)
