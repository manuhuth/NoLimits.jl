export PlotCache
export PlotStyle
export build_plot_cache
export plot_data
export plot_observed_profiles
export plot_fits
export plot_fits_comparison
export plot_multistart_waterfall
export plot_multistart_fixed_effect_variability
export plot_em_trajectories
export plot_hidden_states
export plot_emission_distributions

using ComponentArrays
using Distributions
using MCMCChains
using OrdinaryDiffEq
using Measures
using Random
using SciMLBase
using Statistics

"""
    PlotStyle(; color_primary, color_secondary, color_accent, color_dark,
               color_density, color_reference, font_family,
               font_size_title, font_size_label, font_size_tick, font_size_legend,
               font_size_annotation, line_width_primary, line_width_secondary,
               comparison_default_linestyle, comparison_line_styles, marker_size,
               marker_size_small, marker_alpha, marker_stroke_width,
               marker_size_pmf, marker_stroke_width_pmf,
               base_subplot_width, base_subplot_height,
               left_margin, bottom_margin)

Visual style configuration for all NoLimits plotting functions.

All plotting functions accept a `style::PlotStyle` keyword argument. Construct a
`PlotStyle()` with the defaults and override individual fields as needed.

# Keyword Arguments
- `color_primary::String`: main series color (default: `"#0072B2"` — blue).
- `color_secondary::String`: secondary series color (default: `"#E69F00"` — orange).
- `color_accent::String`: accent color (default: `"#009E73"` — green).
- `color_dark::String`: dark foreground color (default: `"#2C3E50"`).
- `color_density::String`: color for density bands (default: `"#E69F00"`).
- `color_reference::String`: reference line color (default: `"#2C3E50"`).
- `font_family::String`: font family for all text (default: `"Helvetica"`).
- `font_size_title`, `font_size_label`, `font_size_tick`, `font_size_legend`,
  `font_size_annotation`: font sizes in points, applied uniformly across all plots
  (defaults: 16, 14, 12, 11, 10).
- `line_width_primary`, `line_width_secondary`: line widths in pixels.
- `comparison_default_linestyle`, `comparison_line_styles`: line style overrides for
  `plot_fits_comparison`.
- `marker_size`, `marker_size_small`, `marker_alpha`, `marker_stroke_width`: marker
  appearance for continuous outcomes.
- `marker_size_pmf`, `marker_stroke_width_pmf`: marker appearance for discrete outcomes.
- `left_margin`: left margin per subplot to ensure y-axis labels are not clipped
  (default: `10mm`).
- `bottom_margin`: bottom margin per subplot to ensure x-axis labels are not clipped
  (default: `8mm`).
"""
Base.@kwdef struct PlotStyle
    color_primary::String = "#0072B2"
    color_secondary::String = "#E69F00"
    color_accent::String = "#009E73"
    color_dark::String = "#2C3E50"
    color_density::String = "#E69F00"
    color_reference::String = "#2C3E50"

    font_family::String = "Helvetica"
    font_size_title::Int = 16
    font_size_label::Int = 14
    font_size_tick::Int = 12
    font_size_legend::Int = 11
    font_size_annotation::Int = 10

    line_width_primary::Float64 = 2.0
    line_width_secondary::Float64 = 1.5
    comparison_default_linestyle::Symbol = :solid
    comparison_line_styles::Dict{String, Symbol} = Dict{String, Symbol}()
    marker_size::Int = 5
    marker_size_small::Int = 3
    marker_alpha::Float64 = 0.7
    marker_stroke_width::Float64 = 0.5
    marker_size_pmf::Int = 6
    marker_stroke_width_pmf::Float64 = 1.2

    base_subplot_width::Int = 350
    base_subplot_height::Int = 280

    left_margin::typeof(1mm) = PLOT_LEFT_MARGIN
    bottom_margin::typeof(1mm) = PLOT_BOTTOM_MARGIN
end

const COLOR_PRIMARY = "#0072B2"
const COLOR_SECONDARY = "#E69F00"
const COLOR_ACCENT = "#009E73"
const COLOR_DARK = "#2C3E50"
const COLOR_ERROR = "#D55E00"
const COLOR_CI = "#56B4E9"
const COLOR_REFERENCE = "#2C3E50"

const DEFAULT_PLOT_COLS = 3
const MIN_FIGURE_WIDTH = 400
const MAX_FIGURE_WIDTH = 1800
const MIN_FIGURE_HEIGHT = 300
const MAX_FIGURE_HEIGHT = 2400
const PLOT_MARGIN = 3mm
const PLOT_LEFT_MARGIN = 10mm
const PLOT_BOTTOM_MARGIN = 8mm
const DEFAULT_DPI = 300

function calculate_plot_size(nplots::Int, ncols::Int, style::PlotStyle = PlotStyle())
    ncols = min(ncols, nplots)
    nrows = ceil(Int, nplots / ncols)
    scale = nplots <= 4 ? 1.0 :
            nplots <= 9 ? 0.95 : nplots <= 16 ? 0.85 : nplots <= 25 ? 0.75 : 0.65
    width = round(Int, ncols * style.base_subplot_width * scale)
    height = round(Int, nrows * style.base_subplot_height * scale)
    width = clamp(width, MIN_FIGURE_WIDTH, MAX_FIGURE_WIDTH)
    height = clamp(height, MIN_FIGURE_HEIGHT, MAX_FIGURE_HEIGHT)
    return (width, height)
end

function default_plot_kwargs(style::PlotStyle = PlotStyle())
    return (
        framestyle = :box,
        grid = :y,
        gridalpha = 0.3,
        gridlinewidth = 0.5,
        foreground_color_grid = :gray,
        legend = :best,
        titlefontsize = style.font_size_title,
        guidefontsize = style.font_size_label,
        tickfontsize = style.font_size_tick,
        legendfontsize = style.font_size_legend,
        margin = PLOT_MARGIN,
        left_margin = style.left_margin,
        bottom_margin = style.bottom_margin,
        fontfamily = style.font_family
    )
end

function _axis_label(label)
    s = strip(string(label))
    isempty(s) && return s
    s = replace(s, "_" => " ")
    low = lowercase(s)
    if low == "t" || low == "time"
        return "Time"
    elseif low == "id"
        return "ID"
    end
    words = Base.split(s, ' ')
    words = [isempty(w) ? w : uppercasefirst(w) for w in words]
    return join(words, " ")
end

function _ensure_save_path(save_path::Union{Nothing, String})
    save_path === nothing && return nothing
    root, ext = splitext(save_path)
    if isempty(ext)
        save_path = root * ".png"
        ext = ".png"
    end
    dir = dirname(save_path)
    isdir(dir) || error("save_path directory does not exist: $(dir)")
    return save_path
end

function _resolve_plot_path(save_path::Union{Nothing, String},
        plot_path::Union{Nothing, String})
    if save_path !== nothing && plot_path !== nothing && save_path != plot_path
        error("Specify only one of save_path or plot_path when saving plots.")
    end
    return plot_path === nothing ? save_path : plot_path
end

function _normalize_top_level_parameter_selection(sel, argname::String)
    sel === nothing && return nothing
    out = Symbol[]
    if sel isa Symbol
        push!(out, sel)
    elseif sel isa AbstractString
        push!(out, Symbol(sel))
    elseif sel isa AbstractVector
        for x in sel
            if x isa Symbol
                push!(out, x)
            elseif x isa AbstractString
                push!(out, Symbol(x))
            else
                error("$(argname) entries must be Symbol or String. Got $(typeof(x)).")
            end
        end
    else
        error("$(argname) must be a Symbol, String, vector of Symbol/String, or nothing.")
    end
    return Set(out)
end

function _flatten_param_with_labels(name::Symbol, val)
    labels = String[]
    vals = Float64[]

    if val isa Number
        push!(labels, string(name))
        push!(vals, float(val))
        return labels, vals
    end
    if val isa AbstractMatrix
        for r in axes(val, 1)
            for c in axes(val, 2)
                push!(labels, string(name, "[", r, ",", c, "]"))
                push!(vals, float(val[r, c]))
            end
        end
        return labels, vals
    end
    if val isa AbstractVector
        for j in eachindex(val)
            push!(labels, string(name, "[", j, "]"))
            push!(vals, float(val[j]))
        end
        return labels, vals
    end
    error("Unsupported parameter container for $(name): $(typeof(val)).")
end

function _multistart_data_model(res::MultistartFitResult, dm::Union{Nothing, DataModel})
    dm !== nothing && return dm
    dm_res = get_data_model(res)
    dm_res === nothing &&
        error("This multistart result does not store a DataModel; pass dm=... explicitly.")
    return dm_res
end

"""
    PlotCache{S, O, C, P, R, M}

Pre-computed plotting cache for efficient repeated rendering of model predictions.

Create via [`build_plot_cache`](@ref) and pass to `plot_fits` via the `cache` keyword
argument to avoid re-solving the ODE or re-evaluating formulas on each call.
"""
struct PlotCache{S, O, C, P, R, M}
    signature::UInt
    sols::Vector{S}
    obs_dists::O
    chain::C
    params::P
    random_effects::R
    meta::M
end

# Apply HMM forward-filter step for one observable column.
# Updates hmm_priors[col] in-place and returns the distribution conditioned on past observations.
function _apply_hmm_filter!(hmm_priors::Dict{Symbol, Any}, col::Symbol, dist, y_val)
    _is_hmm_dist(dist) || return dist
    prior = get(hmm_priors, col, nothing)
    dist_filtered = _hmm_with_prior(dist, prior)
    if y_val === missing
        hmm_priors[col] = probabilities_hidden_states(dist_filtered)
    else
        hmm_priors[col] = posterior_hidden_states(dist_filtered, y_val)
    end
    return dist_filtered
end

@inline function _plot_with_infusion(f!, infusion_rates)
    infusion_rates === nothing && return f!
    return function (du, u, p, t)
        f!(du, u, p, t)
        @inbounds for i in eachindex(infusion_rates)
            du[i] += infusion_rates[i]
        end
        return nothing
    end
end

function _apply_param_overrides(θ::ComponentArray, overrides::NamedTuple)
    isempty(keys(overrides)) && return θ
    θ_use = deepcopy(θ)
    for name in keys(overrides)
        hasproperty(θ_use, name) ||
            error("Parameter override includes unknown fixed effect $(name).")
        setproperty!(θ_use, name, getfield(overrides, name))
    end
    return θ_use
end

@inline _is_posterior_draw_fit(res::FitResult) = (res.result isa MCMCResult ||
                                                  res.result isa VIResult)

function _with_posterior_warmup(res::FitResult, mcmc_warmup::Union{Nothing, Int})
    if !(res.result isa MCMCResult) || mcmc_warmup === nothing
        return res
    end
    conv = res.diagnostics.convergence
    conv = merge(conv, (n_adapt = mcmc_warmup,))
    return FitResult(res.method, res.result, res.summary,
        FitDiagnostics(
            res.diagnostics.timing, res.diagnostics.optimizer, conv, res.diagnostics.notes),
        res.data_model, res.fit_args, res.fit_kwargs)
end

function _plot_signature(dm::DataModel,
        θ::ComponentArray,
        constants_re::NamedTuple,
        solver_cfg,
        callbacks_hash::UInt,
        cache_obs_dists::Bool)
    h = hash(typeof(dm.model))
    h = hash(dm.model.de.de === nothing ? Symbol[] : get_de_states(dm.model.de.de), h)
    h = hash(get_formulas_meta(dm.model.formulas.formulas).obs_names, h)
    h = hash(θ, h)
    h = hash(constants_re, h)
    h = hash(typeof(solver_cfg.alg), h)
    h = hash(solver_cfg.args, h)
    h = hash(solver_cfg.kwargs, h)
    h = hash(callbacks_hash, h)
    h = hash(cache_obs_dists, h)
    return h
end

function _callbacks_hash(dm::DataModel)
    acc = UInt(0)
    for ind in dm.individuals
        if ind.callbacks === nothing
            acc = hash(nothing, acc)
        else
            acc = hash(ind.callbacks, acc)
        end
    end
    return acc
end

function _solve_dense_individual(dm::DataModel,
        ind::Individual,
        θ::ComponentArray,
        η_ind::ComponentArray;
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple())
    model = dm.model
    pre = calculate_prede(model, θ, η_ind, ind.const_cov)
    pc = (;
        fixed_effects = θ,
        random_effects = η_ind,
        constant_covariates = ind.const_cov,
        varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
        helpers = get_helper_funs(model),
        model_funs = get_model_funs(model),
        preDE = pre
    )
    compiled = get_de_compiler(model.de.de)(pc)
    u0 = calculate_initial_state(model, θ, η_ind, ind.const_cov)
    f! = get_de_f!(model.de.de)
    cb = nothing
    infusion_rates = nothing
    if ind.callbacks !== nothing
        _apply_initial_events!(u0, ind.callbacks)
        cb = ind.callbacks.callback
        infusion_rates = ind.callbacks.infusion_rates
        f! = _plot_with_infusion(f!, infusion_rates)
    end
    prob = ODEProblem(f!, u0, ind.tspan, compiled)
    solver_cfg = get_solver_config(model)
    alg = solver_cfg.alg === nothing ? Tsit5() : solver_cfg.alg
    solve_kwargs = _ode_solve_kwargs(solver_cfg.kwargs, ode_kwargs, (dense = true,))
    if cb === nothing
        sol = solve(prob, alg, solver_cfg.args..., ode_args...; solve_kwargs...)
    else
        sol = solve(
            prob, alg, solver_cfg.args..., ode_args...; solve_kwargs..., callback = cb)
    end
    return sol, compiled
end

function _mcmc_param_means(chain::Chains;
        n_adapt::Int = 0,
        max_draws::Int = typemax(Int),
        rng::AbstractRNG = Random.default_rng())
    max_draws >= 1 || error("max_draws must be >= 1.")
    names_all = MCMCChains.names(chain, :parameters)
    idxs = findall(in(names_all), MCMCChains.names(chain))
    vals = Array(chain)
    draw_idxs = _mcmc_draw_indices(chain, n_adapt, max_draws, rng)
    isempty(draw_idxs) && error("No MCMC draws available after warmup.")
    means = dropdims(mean(vals[draw_idxs, idxs, :], dims = (1, 3)), dims = (1, 3))
    return Dict{Symbol, Float64}((names_all[i] => means[i]) for i in eachindex(names_all))
end

function _coordwise_fixed_from_means(dm::DataModel,
        means::Dict{Symbol, Float64},
        source::AbstractString)
    fe_names = get_names(dm.model.fixed.fixed)
    θ0_u = get_θ0_untransformed(dm.model.fixed.fixed)
    pairs = Pair{Symbol, Any}[]
    for name in fe_names
        if haskey(means, name)
            push!(pairs, name => means[name])
        else
            val0 = getproperty(θ0_u, name)
            if val0 isa AbstractArray
                vals = similar(val0, Float64)
                for idx in CartesianIndices(val0)
                    idx_txt = join(Tuple(idx), ",")
                    key = Symbol("$(name)[$(idx_txt)]")
                    if haskey(means, key)
                        vals[idx] = means[key]
                    else
                        @warn "$(source) is missing fixed effect element; falling back to initial value." name=name index=Tuple(idx)
                        vals[idx] = Float64(val0[idx])
                    end
                end
                push!(pairs, name => vals)
            else
                @warn "$(source) is missing fixed effect; falling back to initial value." name=name
                push!(pairs, name => val0)
            end
        end
    end
    return ComponentArray(NamedTuple(pairs))
end

function _mcmc_fixed_means(res::FitResult,
        dm::DataModel;
        max_draws::Int = typemax(Int),
        rng::AbstractRNG = Random.default_rng())
    chain = get_chain(res)
    n_adapt = _mcmc_warmup(res)
    means = _mcmc_param_means(chain; n_adapt = n_adapt, max_draws = max_draws, rng = rng)
    θ = _coordwise_fixed_from_means(dm, means, "MCMC chain")
    return θ, chain
end

function _vi_param_means(res::FitResult;
        max_draws::Int = typemax(Int),
        rng::AbstractRNG = Random.default_rng())
    n_draws = max_draws == typemax(Int) ? 1000 : Int(max_draws)
    n_draws >= 1 || error("max_draws must be >= 1.")
    raw = sample_posterior(res; n_draws = n_draws, rng = rng, return_names = true)
    draws = raw.draws
    names = raw.names
    means = vec(mean(draws; dims = 1))
    return Dict{Symbol, Float64}((Symbol(string(names[i])) => Float64(means[i]))
    for i in eachindex(names))
end

function _vi_fixed_means(res::FitResult,
        dm::DataModel;
        max_draws::Int = typemax(Int),
        rng::AbstractRNG = Random.default_rng())
    means = _vi_param_means(res; max_draws = max_draws, rng = rng)
    θ = _coordwise_fixed_from_means(dm, means, "VI posterior")
    return θ, nothing
end

function _posterior_fixed_means(res::FitResult,
        dm::DataModel;
        max_draws::Int = typemax(Int),
        rng::AbstractRNG = Random.default_rng())
    if res.result isa MCMCResult
        return _mcmc_fixed_means(res, dm; max_draws = max_draws, rng = rng)
    elseif res.result isa VIResult
        return _vi_fixed_means(res, dm; max_draws = max_draws, rng = rng)
    end
    return get_params(res; scale = :untransformed), nothing
end

function _mcmc_warmup(res::FitResult)
    diag = get_diagnostics(res)
    conv = hasproperty(diag, :convergence) ? getproperty(diag, :convergence) : NamedTuple()
    return hasproperty(conv, :n_adapt) ? getproperty(conv, :n_adapt) : 0
end

function _mcmc_draw_indices(chain::Chains, n_adapt::Int, max_draws::Int, rng::AbstractRNG)
    n_iter = size(Array(chain), 1)
    start_idx = min(n_adapt + 1, n_iter)
    idxs = collect(start_idx:n_iter)
    n_keep = min(length(idxs), max_draws)
    n_keep == length(idxs) && return idxs
    return idxs[Random.randperm(rng, length(idxs))[1:n_keep]]
end

function _mcmc_param_index_map(chain::Chains)
    nms = MCMCChains.names(chain)
    return Dict{Symbol, Int}((nms[i] => i) for i in eachindex(nms))
end

function _mcmc_param_value(vals, iter_idx::Int, var_idx::Int, chain_idx::Int)
    return vals[iter_idx, var_idx, chain_idx]
end

function _mcmc_random_effects_means(res::FitResult,
        dm::DataModel,
        constants_re::NamedTuple,
        θ::ComponentArray;
        max_draws::Int = typemax(Int),
        rng::AbstractRNG = Random.default_rng())
    chain = get_chain(res)
    n_adapt = _mcmc_warmup(res)
    means = _mcmc_param_means(chain; n_adapt = n_adapt, max_draws = max_draws, rng = rng)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return Vector{ComponentArray}(undef, length(dm.individuals))

    fixed_maps = _normalize_constants_re(dm, constants_re)
    re_values = dm.re_group_info.values
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)

    level_means = Dict{Symbol, Dict{Any, Any}}()
    level_dims = Dict{Symbol, Int}()
    for re in re_names
        levels_all = getfield(re_values, re)
        fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
        levels_free = Any[]
        for lvl in levels_all
            haskey(fixed, lvl) && continue
            push!(levels_free, lvl)
        end

        if isempty(levels_free)
            level_means[re] = Dict{Any, Any}()
            level_dims[re] = 1
            continue
        end
        rep_idx = findfirst(
            ind -> (getfield(ind.re_groups, re) isa AbstractVector ?
                    (getfield(ind.re_groups, re)[1] in levels_free) :
                    (getfield(ind.re_groups, re) in levels_free)),
            dm.individuals)
        const_cov = dm.individuals[rep_idx].const_cov
        dist = getproperty(dists_builder(θ, const_cov, model_funs, helpers), re)
        dim = dist isa Distributions.UnivariateDistribution ? 1 : length(dist)
        level_dims[re] = dim
        re_map = Dict{Any, Any}()
        for (i, lvl) in enumerate(levels_free)
            if dim == 1
                name = Symbol(string(re), "_vals[", i, "]")
                haskey(means, name) || error("MCMC chain is missing random effect $(name).")
                re_map[lvl] = means[name]
            else
                vals = Vector{Float64}(undef, dim)
                for j in 1:dim
                    name = Symbol(string(re), "_vals[", i, ",", j, "]")
                    if !haskey(means, name)
                        name = Symbol(string(re), "_vals[", i, "][", j, "]")
                    end
                    haskey(means, name) ||
                        error("MCMC chain is missing random effect $(name).")
                    vals[j] = means[name]
                end
                re_map[lvl] = vals
            end
        end
        level_means[re] = re_map
    end

    η_vec = Vector{ComponentArray}(undef, length(dm.individuals))
    for (i, ind) in enumerate(dm.individuals)
        nt_pairs = Pair{Symbol, Any}[]
        for re in re_names
            g = getfield(ind.re_groups, re)
            fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
            dim = get(level_dims, re, 1)
            if g isa AbstractVector
                if length(g) == 1
                    lvl = g[1]
                    if haskey(fixed, lvl)
                        v = fixed[lvl]
                    else
                        v = level_means[re][lvl]
                    end
                    push!(nt_pairs, re => (dim == 1 ? v : Vector{Float64}(v)))
                else
                    vals = dim == 1 ? Vector{Float64}(undef, length(g)) :
                           Vector{Vector{Float64}}(undef, length(g))
                    for (gi, lvl) in pairs(g)
                        if haskey(fixed, lvl)
                            v = fixed[lvl]
                        else
                            v = level_means[re][lvl]
                        end
                        vals[gi] = dim == 1 ? v : Vector{Float64}(v)
                    end
                    push!(nt_pairs, re => vals)
                end
            else
                if haskey(fixed, g)
                    v = fixed[g]
                else
                    v = level_means[re][g]
                end
                push!(nt_pairs, re => (dim == 1 ? v : Vector{Float64}(v)))
            end
        end
        η_vec[i] = ComponentArray(NamedTuple(nt_pairs))
    end
    return η_vec
end

function _mcmc_drawn_params(res::FitResult,
        dm::DataModel,
        constants_re::NamedTuple,
        overrides::NamedTuple,
        max_draws::Int,
        rng::AbstractRNG)
    chain = get_chain(res)
    n_adapt = _mcmc_warmup(res)
    draw_idxs = _mcmc_draw_indices(chain, n_adapt, max_draws, rng)
    nms = MCMCChains.names(chain)
    idx_map = _mcmc_param_index_map(chain)
    vals = Array(chain)
    n_chains = size(vals, 3)

    fe_names = get_names(dm.model.fixed.fixed)
    re_names = get_re_names(dm.model.random.random)
    fixed_maps = _normalize_constants_re(dm, constants_re)
    re_values = dm.re_group_info.values
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)
    isempty(draw_idxs) && error("No MCMC draws available after warmup.")
    rep_iter = first(draw_idxs)
    rep_chain = 1
    fe_pairs_rep = Pair{Symbol, Any}[]
    for name in fe_names
        if haskey(idx_map, name)
            push!(fe_pairs_rep,
                name => _mcmc_param_value(vals, rep_iter, idx_map[name], rep_chain))
        else
            val0 = getproperty(get_θ0_untransformed(dm.model.fixed.fixed), name)
            if val0 isa AbstractArray
                vals_rep = similar(val0, Float64)
                for idx in CartesianIndices(val0)
                    idx_txt = join(Tuple(idx), ",")
                    key = Symbol("$(name)[$(idx_txt)]")
                    if haskey(idx_map, key)
                        vals_rep[idx] = _mcmc_param_value(
                            vals, rep_iter, idx_map[key], rep_chain)
                    else
                        @warn "MCMC chain is missing fixed effect element; falling back to initial value." name=name index=Tuple(idx)
                        vals_rep[idx] = Float64(val0[idx])
                    end
                end
                push!(fe_pairs_rep, name => vals_rep)
            else
                @warn "MCMC chain is missing fixed effect; falling back to initial value." name=name
                push!(fe_pairs_rep, name => val0)
            end
        end
    end
    θ_rep = _apply_param_overrides(ComponentArray(NamedTuple(fe_pairs_rep)), overrides)

    re_meta = Dict{Symbol, Any}()
    for re in re_names
        levels_all = getfield(re_values, re)
        fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
        levels_free = Any[]
        for lvl in levels_all
            haskey(fixed, lvl) && continue
            push!(levels_free, lvl)
        end
        if isempty(levels_free)
            re_meta[re] = (levels_free = levels_free, dim = 1)
            continue
        end
        rep_idx = findfirst(
            ind -> (getfield(ind.re_groups, re) isa AbstractVector ?
                    (getfield(ind.re_groups, re)[1] in levels_free) :
                    (getfield(ind.re_groups, re) in levels_free)),
            dm.individuals)
        const_cov = dm.individuals[rep_idx].const_cov
        dist = getproperty(dists_builder(θ_rep, const_cov, model_funs, helpers), re)
        dim = dist isa Distributions.UnivariateDistribution ? 1 : length(dist)
        re_meta[re] = (levels_free = levels_free, dim = dim)
    end

    function _get_re_value(re, li, dim, iter_idx, chain_idx)
        if dim == 1
            name = Symbol(string(re), "_vals[", li, "]")
            if !haskey(idx_map, name)
                name = Symbol(string(re), "[", li, "]")
            end
            if !haskey(idx_map, name)
                name = Symbol(string(re), "_vals[", li, "][1]")
            end
            if !haskey(idx_map, name)
                name = Symbol(string(re), "[", li, "][1]")
            end
            if !haskey(idx_map, name)
                @warn "MCMC chain is missing random effect; using 0.0." name=Symbol(
                    string(re), "_vals[", li, "]")
                return nothing
            end
            return _mcmc_param_value(vals, iter_idx, idx_map[name], chain_idx)
        end
        out = Vector{Float64}(undef, dim)
        for j in 1:dim
            name = Symbol(string(re), "_vals[", li, ",", j, "]")
            if !haskey(idx_map, name)
                name = Symbol(string(re), "_vals[", li, "][", j, "]")
            end
            if !haskey(idx_map, name)
                name = Symbol(string(re), "[", li, ",", j, "]")
            end
            if !haskey(idx_map, name)
                name = Symbol(string(re), "[", li, "][", j, "]")
            end
            if !haskey(idx_map, name)
                @warn "MCMC chain is missing random effect element; using 0.0." name=name
                return nothing
            end
            out[j] = _mcmc_param_value(vals, iter_idx, idx_map[name], chain_idx)
        end
        return out
    end

    θ_draws = Vector{ComponentArray}(undef, length(draw_idxs))
    η_draws = Vector{Vector{ComponentArray}}(undef, length(draw_idxs))
    for (k, iter_idx) in enumerate(draw_idxs)
        chain_idx = rand(rng, 1:n_chains)
        fe_pairs = Pair{Symbol, Any}[]
        for name in fe_names
            if haskey(idx_map, name)
                push!(fe_pairs,
                    name => _mcmc_param_value(vals, iter_idx, idx_map[name], chain_idx))
            else
                val0 = getproperty(get_θ0_untransformed(dm.model.fixed.fixed), name)
                if val0 isa AbstractArray
                    vals_draw = similar(val0, Float64)
                    for idx in CartesianIndices(val0)
                        idx_txt = join(Tuple(idx), ",")
                        key = Symbol("$(name)[$(idx_txt)]")
                        if haskey(idx_map, key)
                            vals_draw[idx] = _mcmc_param_value(
                                vals, iter_idx, idx_map[key], chain_idx)
                        else
                            @warn "MCMC chain is missing fixed effect element; falling back to initial value." name=name index=Tuple(idx)
                            vals_draw[idx] = Float64(val0[idx])
                        end
                    end
                    push!(fe_pairs, name => vals_draw)
                else
                    @warn "MCMC chain is missing fixed effect; falling back to initial value." name=name
                    push!(fe_pairs, name => val0)
                end
            end
        end
        θ = ComponentArray(NamedTuple(fe_pairs))
        θ = _apply_param_overrides(θ, overrides)
        θ_draws[k] = θ

        η_vec = Vector{ComponentArray}(undef, length(dm.individuals))
        for (i, ind) in enumerate(dm.individuals)
            nt_pairs = Pair{Symbol, Any}[]
            for re in re_names
                fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
                meta = re_meta[re]
                dim = meta.dim
                levels_free = meta.levels_free
                lvl_to_idx = Dict{Any, Int}((levels_free[i] => i)
                for i in eachindex(levels_free))
                g = getfield(ind.re_groups, re)
                if g isa AbstractVector
                    if length(g) == 1
                        lvl = g[1]
                        if haskey(fixed, lvl)
                            v = fixed[lvl]
                        else
                            li = get(lvl_to_idx, lvl, 0)
                            li == 0 &&
                                error("Missing random effect value for $(re) level $(lvl).")
                            v = _get_re_value(re, li, dim, iter_idx, chain_idx)
                            v === nothing && (v = dim == 1 ? 0.0 : zeros(dim))
                        end
                        push!(nt_pairs, re => (dim == 1 ? v : Vector{Float64}(v)))
                    else
                        vals_re = dim == 1 ? Vector{Float64}(undef, length(g)) :
                                  Vector{Vector{Float64}}(undef, length(g))
                        for (gi, lvl) in pairs(g)
                            if haskey(fixed, lvl)
                                v = fixed[lvl]
                            else
                                li = get(lvl_to_idx, lvl, 0)
                                li == 0 &&
                                    error("Missing random effect value for $(re) level $(lvl).")
                                v = _get_re_value(re, li, dim, iter_idx, chain_idx)
                                v === nothing && (v = dim == 1 ? 0.0 : zeros(dim))
                            end
                            vals_re[gi] = dim == 1 ? v : Vector{Float64}(v)
                        end
                        push!(nt_pairs, re => vals_re)
                    end
                else
                    if haskey(fixed, g)
                        v = fixed[g]
                    else
                        li = get(lvl_to_idx, g, 0)
                        li == 0 &&
                            error("Missing random effect value for $(re) level $(g).")
                        v = _get_re_value(re, li, dim, iter_idx, chain_idx)
                        v === nothing && (v = dim == 1 ? 0.0 : zeros(dim))
                    end
                    push!(nt_pairs, re => (dim == 1 ? v : Vector{Float64}(v)))
                end
            end
            η_vec[i] = ComponentArray(NamedTuple(nt_pairs))
        end
        η_draws[k] = η_vec
    end
    return θ_draws, η_draws, chain
end

function _vi_drawn_params(res::FitResult,
        dm::DataModel,
        constants_re::NamedTuple,
        overrides::NamedTuple,
        max_draws::Int,
        rng::AbstractRNG)
    max_draws >= 1 || error("mcmc_draws must be >= 1.")
    raw = sample_posterior(res; n_draws = max_draws, rng = rng, return_names = true)
    draws = raw.draws
    coord_names = raw.names
    size(draws, 1) >= 1 || error("No VI posterior draws available.")

    idx_map = Dict{String, Int}()
    for (i, n) in enumerate(coord_names)
        idx_map[string(n)] = i
    end

    fe_names = get_names(dm.model.fixed.fixed)
    re_names = get_re_names(dm.model.random.random)
    fixed_maps = _normalize_constants_re(dm, constants_re)
    re_values = dm.re_group_info.values
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)

    rep_row = @view draws[1, :]
    fe_pairs_rep = Pair{Symbol, Any}[]
    for name in fe_names
        key = string(name)
        idx = _lookup_chain_index(idx_map, key)
        if idx != 0
            push!(fe_pairs_rep, name => Float64(rep_row[idx]))
        else
            val0 = getproperty(get_θ0_untransformed(dm.model.fixed.fixed), name)
            if val0 isa AbstractArray
                vals_rep = similar(val0, Float64)
                for ci in CartesianIndices(val0)
                    idx_txt = join(Tuple(ci), ",")
                    k = string(name, "[", idx_txt, "]")
                    j = _lookup_chain_index(idx_map, k)
                    if j != 0
                        vals_rep[ci] = Float64(rep_row[j])
                    else
                        @warn "VI posterior is missing fixed effect element; falling back to initial value." name=name index=Tuple(ci)
                        vals_rep[ci] = Float64(val0[ci])
                    end
                end
                push!(fe_pairs_rep, name => vals_rep)
            else
                @warn "VI posterior is missing fixed effect; falling back to initial value." name=name
                push!(fe_pairs_rep, name => val0)
            end
        end
    end
    θ_rep = _apply_param_overrides(ComponentArray(NamedTuple(fe_pairs_rep)), overrides)

    re_meta = Dict{Symbol, Any}()
    for re in re_names
        levels_all = getfield(re_values, re)
        fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
        levels_free = Any[]
        for lvl in levels_all
            haskey(fixed, lvl) && continue
            push!(levels_free, lvl)
        end
        if isempty(levels_free)
            re_meta[re] = (levels_free = levels_free, dim = 1)
            continue
        end
        rep_idx = findfirst(
            ind -> (getfield(ind.re_groups, re) isa AbstractVector ?
                    (getfield(ind.re_groups, re)[1] in levels_free) :
                    (getfield(ind.re_groups, re) in levels_free)),
            dm.individuals)
        const_cov = dm.individuals[rep_idx].const_cov
        dist = getproperty(dists_builder(θ_rep, const_cov, model_funs, helpers), re)
        dim = dist isa Distributions.UnivariateDistribution ? 1 : length(dist)
        re_meta[re] = (levels_free = levels_free, dim = dim)
    end

    function _get_re_value(re, li, dim, row)
        if dim == 1
            k = string(re, "_vals[", li, "]")
            idx = _lookup_chain_index(idx_map, k)
            idx == 0 && (idx = _lookup_chain_index(idx_map, string(re, "[", li, "]")))
            idx == 0 &&
                (idx = _lookup_chain_index(idx_map, string(re, "_vals[", li, "][1]")))
            idx == 0 && (idx = _lookup_chain_index(idx_map, string(re, "[", li, "][1]")))
            if idx == 0
                @warn "VI posterior is missing random effect; using 0.0." name=Symbol(
                    string(re), "_vals[", li, "]")
                return nothing
            end
            return Float64(row[idx])
        end
        out = Vector{Float64}(undef, dim)
        for j in 1:dim
            k = string(re, "_vals[", li, ",", j, "]")
            idx = _lookup_chain_index(idx_map, k)
            idx == 0 &&
                (idx = _lookup_chain_index(idx_map, string(re, "_vals[", li, "][", j, "]")))
            idx == 0 &&
                (idx = _lookup_chain_index(idx_map, string(re, "[", li, ",", j, "]")))
            idx == 0 &&
                (idx = _lookup_chain_index(idx_map, string(re, "[", li, "][", j, "]")))
            if idx == 0
                @warn "VI posterior is missing random effect element; using 0.0." name=Symbol(k)
                return nothing
            end
            out[j] = Float64(row[idx])
        end
        return out
    end

    n_draws = size(draws, 1)
    θ_draws = Vector{ComponentArray}(undef, n_draws)
    η_draws = Vector{Vector{ComponentArray}}(undef, n_draws)
    for k in 1:n_draws
        row = @view draws[k, :]
        fe_pairs = Pair{Symbol, Any}[]
        for name in fe_names
            idx = _lookup_chain_index(idx_map, string(name))
            if idx != 0
                push!(fe_pairs, name => Float64(row[idx]))
            else
                val0 = getproperty(get_θ0_untransformed(dm.model.fixed.fixed), name)
                if val0 isa AbstractArray
                    vals_draw = similar(val0, Float64)
                    for ci in CartesianIndices(val0)
                        idx_txt = join(Tuple(ci), ",")
                        key = string(name, "[", idx_txt, "]")
                        j = _lookup_chain_index(idx_map, key)
                        if j != 0
                            vals_draw[ci] = Float64(row[j])
                        else
                            @warn "VI posterior is missing fixed effect element; falling back to initial value." name=name index=Tuple(ci)
                            vals_draw[ci] = Float64(val0[ci])
                        end
                    end
                    push!(fe_pairs, name => vals_draw)
                else
                    @warn "VI posterior is missing fixed effect; falling back to initial value." name=name
                    push!(fe_pairs, name => val0)
                end
            end
        end
        θ = _apply_param_overrides(ComponentArray(NamedTuple(fe_pairs)), overrides)
        θ_draws[k] = θ

        η_vec = Vector{ComponentArray}(undef, length(dm.individuals))
        for (i, ind) in enumerate(dm.individuals)
            nt_pairs = Pair{Symbol, Any}[]
            for re in re_names
                fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
                meta = re_meta[re]
                dim = meta.dim
                levels_free = meta.levels_free
                lvl_to_idx = Dict{Any, Int}((levels_free[j] => j)
                for j in eachindex(levels_free))
                g = getfield(ind.re_groups, re)
                if g isa AbstractVector
                    if length(g) == 1
                        lvl = g[1]
                        if haskey(fixed, lvl)
                            v = fixed[lvl]
                        else
                            li = get(lvl_to_idx, lvl, 0)
                            li == 0 &&
                                error("Missing random effect value for $(re) level $(lvl).")
                            v = _get_re_value(re, li, dim, row)
                            v === nothing && (v = dim == 1 ? 0.0 : zeros(dim))
                        end
                        push!(nt_pairs, re => (dim == 1 ? v : Vector{Float64}(v)))
                    else
                        vals_re = dim == 1 ? Vector{Float64}(undef, length(g)) :
                                  Vector{Vector{Float64}}(undef, length(g))
                        for (gi, lvl) in pairs(g)
                            if haskey(fixed, lvl)
                                v = fixed[lvl]
                            else
                                li = get(lvl_to_idx, lvl, 0)
                                li == 0 &&
                                    error("Missing random effect value for $(re) level $(lvl).")
                                v = _get_re_value(re, li, dim, row)
                                v === nothing && (v = dim == 1 ? 0.0 : zeros(dim))
                            end
                            vals_re[gi] = dim == 1 ? v : Vector{Float64}(v)
                        end
                        push!(nt_pairs, re => vals_re)
                    end
                else
                    if haskey(fixed, g)
                        v = fixed[g]
                    else
                        li = get(lvl_to_idx, g, 0)
                        li == 0 &&
                            error("Missing random effect value for $(re) level $(g).")
                        v = _get_re_value(re, li, dim, row)
                        v === nothing && (v = dim == 1 ? 0.0 : zeros(dim))
                    end
                    push!(nt_pairs, re => (dim == 1 ? v : Vector{Float64}(v)))
                end
            end
            η_vec[i] = ComponentArray(NamedTuple(nt_pairs))
        end
        η_draws[k] = η_vec
    end
    return θ_draws, η_draws, nothing
end

function _posterior_drawn_params(res::FitResult,
        dm::DataModel,
        constants_re::NamedTuple,
        overrides::NamedTuple,
        max_draws::Int,
        rng::AbstractRNG)
    if res.result isa MCMCResult
        return _mcmc_drawn_params(res, dm, constants_re, overrides, max_draws, rng)
    elseif res.result isa VIResult
        return _vi_drawn_params(res, dm, constants_re, overrides, max_draws, rng)
    end
    error("Posterior draws are supported only for MCMC and VI fit results.")
end

function _default_random_effects(res::FitResult,
        dm::DataModel,
        constants_re::NamedTuple,
        θ::ComponentArray,
        rng::AbstractRNG,
        mcmc_draws::Int)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return fill(ComponentArray(NamedTuple()), length(dm.individuals))

    if res.result isa LaplaceResult || res.result isa GHQuadratureResult
        _, batch_infos, _ = _build_laplace_batch_infos(dm, constants_re)
        bstars = res.result.eb_modes
        length(bstars) == length(batch_infos) ||
            error("Laplace-style EB modes do not match number of batches.")
        return _eta_from_eb(dm, batch_infos, bstars, constants_re, θ)
    end

    if res.result isa MCEMResult
        ode_args = _fit_kw(res, :ode_args, ())
        ode_kwargs = _fit_kw(res, :ode_kwargs, NamedTuple())
        serialization = _fit_kw(res, :serialization, EnsembleSerial())
        rng_use = _fit_kw(res, :rng, rng)
        ll_cache = build_ll_cache(
            dm; ode_args = ode_args, ode_kwargs = ode_kwargs, serialization = serialization)
        bstars = res.result.eb_modes
        if bstars === nothing
            bstars, batch_infos = _compute_bstars(
                dm, θ, constants_re, ll_cache, res.method.ebe, rng_use;
                rescue = res.method.ebe_rescue)
        else
            _, batch_infos, _ = _build_laplace_batch_infos(dm, constants_re)
        end
        return _eta_from_eb(dm, batch_infos, bstars, constants_re, θ)
    end
    if res.result isa SAEMResult
        constants_re = _saem_anneal_constants_re(
            dm, θ, _saem_anneal_names(res), constants_re)
        ode_args = _fit_kw(res, :ode_args, ())
        ode_kwargs = _fit_kw(res, :ode_kwargs, NamedTuple())
        serialization = _fit_kw(res, :serialization, EnsembleSerial())
        rng_use = _fit_kw(res, :rng, rng)
        ll_cache = build_ll_cache(
            dm; ode_args = ode_args, ode_kwargs = ode_kwargs, serialization = serialization)
        bstars = res.result.eb_modes
        if bstars === nothing
            # Fall back to default SAEM EBE options for loaded results
            _saem_cfg = res.method isa _SavedFittingMethod ? SAEM() : res.method
            ebe = EBEOptions(_saem_cfg.saem.ebe_optimizer,
                _saem_cfg.saem.ebe_optim_kwargs, _saem_cfg.saem.ebe_adtype,
                _saem_cfg.saem.ebe_grad_tol, _saem_cfg.saem.ebe_multistart_n, _saem_cfg.saem.ebe_multistart_k,
                _saem_cfg.saem.ebe_multistart_max_rounds, _saem_cfg.saem.ebe_multistart_sampling)
            bstars, batch_infos = _compute_bstars(
                dm, θ, constants_re, ll_cache, ebe, rng_use;
                rescue = _saem_cfg.saem.ebe_rescue)
        else
            _, batch_infos, _ = _build_laplace_batch_infos(dm, constants_re)
        end
        return _eta_from_eb(dm, batch_infos, bstars, constants_re, θ)
    end

    if res.result isa MCMCResult
        return _mcmc_random_effects_means(
            res, dm, constants_re, θ; max_draws = mcmc_draws, rng = rng)
    end
    # NB: VI is rejected for random-effects models at fit time, so a VIResult can
    # never reach here with non-empty `re_names`; no VI branch is needed.

    if res.result isa PooledResult
        return res.result.eta_vec
    end

    return fill(ComponentArray(NamedTuple()), length(dm.individuals))
end

function _default_random_effects_from_dm(dm::DataModel,
        constants_re::NamedTuple,
        θ::ComponentArray)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return fill(ComponentArray(NamedTuple()), length(dm.individuals))

    fixed_maps = _normalize_constants_re(dm, constants_re)
    re_values = dm.re_group_info.values
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)

    level_vals = Dict{Symbol, Dict{Any, Any}}()
    level_dims = Dict{Symbol, Int}()
    for re in re_names
        levels_all = getfield(re_values, re)
        fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
        levels_free = Any[]
        for lvl in levels_all
            haskey(fixed, lvl) && continue
            push!(levels_free, lvl)
        end
        if isempty(levels_free)
            level_vals[re] = Dict{Any, Any}()
            level_dims[re] = 1
            continue
        end
        rep_idx = findfirst(
            ind -> (getfield(ind.re_groups, re) isa AbstractVector ?
                    (getfield(ind.re_groups, re)[1] in levels_free) :
                    (getfield(ind.re_groups, re) in levels_free)),
            dm.individuals)
        const_cov = dm.individuals[rep_idx].const_cov
        dist = getproperty(dists_builder(θ, const_cov, model_funs, helpers), re)
        dim = dist isa Distributions.UnivariateDistribution ? 1 : length(dist)
        level_dims[re] = dim
        # `dist` is level-invariant — compute the (possibly expensive) mean once;
        # dim > 1 still stores a fresh copy per level (same aliasing as before).
        v0 = try
            Distributions.mean(dist)
        catch
            dim == 1 ? 0.0 : zeros(Float64, dim)
        end
        re_map = Dict{Any, Any}()
        for lvl in levels_free
            re_map[lvl] = dim == 1 ? v0 : Vector{Float64}(v0)
        end
        level_vals[re] = re_map
    end

    η_vec = Vector{ComponentArray}(undef, length(dm.individuals))
    for (i, ind) in enumerate(dm.individuals)
        nt_pairs = Pair{Symbol, Any}[]
        for re in re_names
            fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
            dim = get(level_dims, re, 1)
            g = getfield(ind.re_groups, re)
            if g isa AbstractVector
                if length(g) == 1
                    lvl = g[1]
                    v = haskey(fixed, lvl) ? fixed[lvl] : level_vals[re][lvl]
                    push!(nt_pairs, re => (dim == 1 ? v : Vector{Float64}(v)))
                else
                    vals_re = dim == 1 ? Vector{Float64}(undef, length(g)) :
                              Vector{Vector{Float64}}(undef, length(g))
                    for (gi, lvl) in pairs(g)
                        v = haskey(fixed, lvl) ? fixed[lvl] : level_vals[re][lvl]
                        vals_re[gi] = dim == 1 ? v : Vector{Float64}(v)
                    end
                    push!(nt_pairs, re => vals_re)
                end
            else
                v = haskey(fixed, g) ? fixed[g] : level_vals[re][g]
                push!(nt_pairs, re => (dim == 1 ? v : Vector{Float64}(v)))
            end
        end
        η_vec[i] = ComponentArray(NamedTuple(nt_pairs))
    end
    return η_vec
end

"""
    build_plot_cache(res::FitResult; dm, params, constants_re, cache_obs_dists,
                     ode_args, ode_kwargs, mcmc_draws, mcmc_warmup, rng) -> PlotCache

    build_plot_cache(res::MultistartFitResult; kwargs...) -> PlotCache

    build_plot_cache(dm::DataModel; params, constants_re, cache_obs_dists,
                     ode_args, ode_kwargs, rng) -> PlotCache

Pre-compute ODE solutions and (optionally) observation distributions for fast repeated
plotting. Pass the returned [`PlotCache`](@ref) to `plot_fits` via the `cache` keyword.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `params::NamedTuple = NamedTuple()`: fixed-effect overrides applied before caching.
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
- `cache_obs_dists::Bool = false`: also pre-compute observation distributions.
- `ode_args::Tuple = ()`, `ode_kwargs::NamedTuple = NamedTuple()`: forwarded to the ODE solver.
- `mcmc_draws::Int = 1000`: number of MCMC draws to use for chain-based fits.
- `mcmc_warmup::Union{Nothing, Int} = nothing`: warm-up count override for MCMC.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
"""
function build_plot_cache(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = false,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        rng::AbstractRNG = Random.default_rng())
    dm === nothing && (dm = get_data_model(res))
    dm === nothing &&
        error("This fit result does not store a DataModel; pass dm=... to build_plot_cache.")
    constants_re = _res_constants_re(res, constants_re)

    if _is_posterior_draw_fit(res)
        mcmc_draws >= 1 || error("mcmc_draws must be >= 1.")
        res = _with_posterior_warmup(res, mcmc_warmup)
    end

    θ_chain = _is_posterior_draw_fit(res) ?
              _posterior_fixed_means(res, dm; max_draws = mcmc_draws, rng = rng) : nothing
    θ = _is_posterior_draw_fit(res) ? θ_chain[1] : get_params(res; scale = :untransformed)
    θ = _apply_param_overrides(θ, params)
    chain = _is_posterior_draw_fit(res) ? θ_chain[2] : nothing
    η_vec = _default_random_effects(res, dm, constants_re, θ, rng, mcmc_draws)

    sols = Vector{Any}(undef, length(dm.individuals))
    compiled_cache = Vector{Any}(undef, length(dm.individuals))
    if dm.model.de.de !== nothing
        for i in eachindex(dm.individuals)
            ind = dm.individuals[i]
            η_ind = η_vec[i] isa ComponentArray ? η_vec[i] : ComponentArray(η_vec[i])
            sol, compiled = _solve_dense_individual(
                dm, ind, θ, η_ind; ode_args = ode_args, ode_kwargs = ode_kwargs)
            sols[i] = sol
            compiled_cache[i] = compiled
        end
    end

    obs_dists = nothing
    if cache_obs_dists
        obs_names_all = get_formulas_meta(dm.model.formulas.formulas).obs_names
        obs_dists = Vector{Vector{NamedTuple}}(undef, length(dm.individuals))
        for i in eachindex(dm.individuals)
            ind = dm.individuals[i]
            obs_rows = dm.row_groups.obs_rows[i]
            η_ind = η_vec[i] isa ComponentArray ? η_vec[i] : ComponentArray(η_vec[i])
            rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)
            sol_accessors = dm.model.de.de === nothing ? nothing :
                            get_de_accessors_builder(dm.model.de.de)(
                sols[i], compiled_cache[i])
            dists_i = Vector{NamedTuple}(undef, length(obs_rows))
            hmm_priors = Dict{Symbol, Any}()
            for (j, row) in enumerate(obs_rows)
                vary = _varying_at(dm, ind, j, row)
                η_row = _row_random_effects_at(dm, i, j, η_ind, rowwise_re; obs_only = true)
                obs = sol_accessors === nothing ?
                      calculate_formulas_obs(dm.model, θ, η_row, ind.const_cov, vary) :
                      calculate_formulas_obs(
                    dm.model, θ, η_row, ind.const_cov, vary, sol_accessors)
                filtered_pairs = Pair{Symbol, Any}[]
                for col in obs_names_all
                    d = getproperty(obs, col)
                    if _is_hmm_dist(d)
                        y_val = getfield(ind.series.obs, col)[j]
                        push!(filtered_pairs,
                            col => _apply_hmm_filter!(hmm_priors, col, d, y_val))
                    end
                end
                dists_i[j] = isempty(filtered_pairs) ? obs :
                             merge(obs, NamedTuple(filtered_pairs))
            end
            obs_dists[i] = dists_i
        end
    end

    sig = _plot_signature(dm, θ, constants_re, get_solver_config(dm.model),
        _callbacks_hash(dm), cache_obs_dists)
    meta = (constants_re = constants_re, cache_obs_dists = cache_obs_dists)
    return PlotCache(sig, sols, obs_dists, chain, θ, η_vec, meta)
end

function build_plot_cache(res::MultistartFitResult; kwargs...)
    return build_plot_cache(get_multistart_best(res); kwargs...)
end

function build_plot_cache(dm::DataModel;
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = false,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        rng::AbstractRNG = Random.default_rng())
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    θ = _apply_param_overrides(θ, params)
    η_vec = _default_random_effects_from_dm(dm, constants_re, θ)

    sols = Vector{Any}(undef, length(dm.individuals))
    compiled_cache = Vector{Any}(undef, length(dm.individuals))
    if dm.model.de.de !== nothing
        for i in eachindex(dm.individuals)
            ind = dm.individuals[i]
            η_ind = η_vec[i] isa ComponentArray ? η_vec[i] : ComponentArray(η_vec[i])
            sol, compiled = _solve_dense_individual(
                dm, ind, θ, η_ind; ode_args = ode_args, ode_kwargs = ode_kwargs)
            sols[i] = sol
            compiled_cache[i] = compiled
        end
    end

    obs_dists = nothing
    if cache_obs_dists
        obs_names_all = get_formulas_meta(dm.model.formulas.formulas).obs_names
        obs_dists = Vector{Vector{NamedTuple}}(undef, length(dm.individuals))
        for i in eachindex(dm.individuals)
            ind = dm.individuals[i]
            obs_rows = dm.row_groups.obs_rows[i]
            η_ind = η_vec[i] isa ComponentArray ? η_vec[i] : ComponentArray(η_vec[i])
            rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)
            sol_accessors = dm.model.de.de === nothing ? nothing :
                            get_de_accessors_builder(dm.model.de.de)(
                sols[i], compiled_cache[i])
            dists_i = Vector{NamedTuple}(undef, length(obs_rows))
            hmm_priors = Dict{Symbol, Any}()
            for (j, row) in enumerate(obs_rows)
                vary = _varying_at(dm, ind, j, row)
                η_row = _row_random_effects_at(dm, i, j, η_ind, rowwise_re; obs_only = true)
                obs = sol_accessors === nothing ?
                      calculate_formulas_obs(dm.model, θ, η_row, ind.const_cov, vary) :
                      calculate_formulas_obs(
                    dm.model, θ, η_row, ind.const_cov, vary, sol_accessors)
                filtered_pairs = Pair{Symbol, Any}[]
                for col in obs_names_all
                    d = getproperty(obs, col)
                    if _is_hmm_dist(d)
                        y_val = getfield(ind.series.obs, col)[j]
                        push!(filtered_pairs,
                            col => _apply_hmm_filter!(hmm_priors, col, d, y_val))
                    end
                end
                dists_i[j] = isempty(filtered_pairs) ? obs :
                             merge(obs, NamedTuple(filtered_pairs))
            end
            obs_dists[i] = dists_i
        end
    end

    sig = _plot_signature(dm, θ, constants_re, get_solver_config(dm.model),
        _callbacks_hash(dm), cache_obs_dists)
    meta = (constants_re = constants_re, cache_obs_dists = cache_obs_dists)
    return PlotCache(sig, sols, obs_dists, nothing, θ, η_vec, meta)
end
