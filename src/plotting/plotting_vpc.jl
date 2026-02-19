export plot_vpc

using Distributions
using Plots
using Random
using StatsFuns

function _require_varying_covariate(dm::DataModel, x_axis_feature)
    cov = dm.model.covariates.covariates
    if x_axis_feature === nothing
        return dm.config.time_col
    end
    if x_axis_feature == dm.config.time_col
        return x_axis_feature
    end
    x_axis_feature in cov.varying || error("x_axis_feature must be a varying covariate. Got $(x_axis_feature).")
    return x_axis_feature
end

function _vpc_x_values(dm::DataModel, ind::Individual, obs_rows::Vector{Int}, x_axis_feature)
    return _get_x_values(dm, ind, obs_rows, x_axis_feature)
end

function _bin_edges_quantile(x::Vector{Float64}, n_bins::Int)
    x_min = minimum(x)
    x_max = maximum(x)
    if x_min == x_max
        return [x_min, x_max]
    end
    qs = range(0.0, 1.0; length=n_bins + 1)
    edges = [quantile(x, q) for q in qs]
    edges[1] = x_min
    edges[end] = x_max
    return edges
end

function _assign_bins(x::Vector{Float64}, edges::Vector{Float64})
    bins = Vector{Int}(undef, length(x))
    for (i, xi) in enumerate(x)
        idx = searchsortedlast(edges, xi)
        idx = clamp(idx, 1, length(edges) - 1)
        bins[i] = idx
    end
    return bins
end

function _weighted_quantile(values::Vector{Float64}, weights::Vector{Float64}, p::Float64)
    idx = sortperm(values)
    v = values[idx]
    w = weights[idx]
    s = sum(w)
    s == 0 && return NaN
    cdf = cumsum(w) ./ s
    i = searchsortedfirst(cdf, p)
    i = clamp(i, 1, length(v))
    return v[i]
end

function _kernel_quantiles(x::Vector{Float64},
                           y::Vector{Float64},
                           xgrid::Vector{Float64},
                           bandwidth::Float64,
                           percentiles::Vector{Float64})
    out = Dict{Float64, Vector{Float64}}((p => Vector{Float64}(undef, length(xgrid))) for p in percentiles)
    for (i, xg) in enumerate(xgrid)
        w = exp.(-0.5 .* ((x .- xg) ./ bandwidth).^2)
        for p in percentiles
            out[p][i] = _weighted_quantile(y, w, p / 100)
        end
    end
    return out
end

function _resolve_n_bins(x::Vector{Float64}, n_bins::Union{Nothing, Int})
    if n_bins !== nothing
        n_bins >= 1 || error("n_bins must be >= 1.")
        n_unique = length(unique(x))
        if n_unique < 1
            return 1
        end
        n_bins > n_unique && @warn "n_bins exceeds unique x values; reducing bins." requested=n_bins used=n_unique
        return min(n_bins, n_unique)
    end
    n_unique = length(unique(x))
    return max(1, min(10, n_unique))
end

function _extend_bin_series(x_centers::Vector{Float64}, y::Vector{Float64}, edges::Vector{Float64})
    length(x_centers) == length(y) || error("Bin series length mismatch.")
    x = [edges[1]; x_centers; edges[end]]
    y_ext = [y[1]; y; y[end]]
    return x, y_ext
end

function _re_level_reps(dm::DataModel, re::Symbol)
    reps = Dict{Any, Int}()
    for (i, ind) in enumerate(dm.individuals)
        g = getfield(ind.re_groups, re)
        if g isa AbstractVector
            for gv in g
                haskey(reps, gv) || (reps[gv] = i)
            end
        else
            haskey(reps, g) || (reps[g] = i)
        end
    end
    return reps
end

function _sample_random_effects_levels(dm::DataModel,
                                       θ::ComponentArray,
                                       constants_re::NamedTuple,
                                       rng::AbstractRNG)
    re_names = get_re_names(dm.model.random.random)
    isempty(re_names) && return Dict{Symbol, Dict{Any, Any}}()
    fixed_maps = _normalize_constants_re(dm, constants_re)
    dists_builder = get_create_random_effect_distribution(dm.model.random.random)
    model_funs = get_model_funs(dm.model)
    helpers = get_helper_funs(dm.model)
    values = dm.re_group_info.values
    out = Dict{Symbol, Dict{Any, Any}}()
    for re in re_names
        reps = _re_level_reps(dm, re)
        fixed = haskey(fixed_maps, re) ? getfield(fixed_maps, re) : Dict{Any, Any}()
        level_vals = Dict{Any, Any}()
        for lvl in getfield(values, re)
            if haskey(fixed, lvl)
                level_vals[lvl] = fixed[lvl]
            else
                rep = reps[lvl]
                const_cov = dm.individuals[rep].const_cov
                dist = getproperty(dists_builder(θ, const_cov, model_funs, helpers), re)
                level_vals[lvl] = rand(rng, dist)
            end
        end
        out[re] = level_vals
    end
    return out
end

function _eta_vec_from_levels(dm::DataModel, level_vals::Dict{Symbol, Dict{Any, Any}})
    re_names = get_re_names(dm.model.random.random)
    η_vec = Vector{ComponentArray}(undef, length(dm.individuals))
    for (i, ind) in enumerate(dm.individuals)
        pairs = Pair{Symbol, Any}[]
        for re in re_names
            g = getfield(ind.re_groups, re)
            vals = level_vals[re]
            if g isa AbstractVector
                if length(g) == 1
                    push!(pairs, re => vals[g[1]])
                else
                    push!(pairs, re => [vals[gv] for gv in g])
                end
            else
                push!(pairs, re => vals[g])
            end
        end
        η_vec[i] = ComponentArray(NamedTuple(pairs))
    end
    return η_vec
end

function _simulate_obs(dm::DataModel,
                       θ::ComponentArray,
                       η_vec::Vector{ComponentArray},
                       obs_name::Symbol,
                       rng::AbstractRNG,
                       x_axis_feature)
    sim_vals = Vector{Vector{Float64}}(undef, length(dm.individuals))
    sim_x = Vector{Vector{Float64}}(undef, length(dm.individuals))
    for (i, ind) in enumerate(dm.individuals)
        obs_rows = dm.row_groups.obs_rows[i]
        x = _vpc_x_values(dm, ind, obs_rows, x_axis_feature)
        sim_x[i] = Float64.(x)
        η_ind = η_vec[i]
        sol_accessors = nothing
        if dm.model.de.de !== nothing
            sol, compiled = _solve_dense_individual(dm, ind, θ, η_ind)
            sol_accessors = get_de_accessors_builder(dm.model.de.de)(sol, compiled)
        end
        vals = Vector{Float64}(undef, length(obs_rows))
        for (j, row) in enumerate(obs_rows)
            vary = _varying_at_plot(dm, ind, j, row)
            obs = sol_accessors === nothing ?
                  calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary) :
                  calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
            dist = getproperty(obs, obs_name)
            vals[j] = rand(rng, dist)
        end
        sim_vals[i] = vals
    end
    return sim_x, sim_vals
end

function _representative_dist(dm::DataModel, obs_name::Symbol, x_axis_feature)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    η_vec = _default_random_effects_from_dm(dm, NamedTuple(), θ)
    ind = dm.individuals[1]
    obs_rows = dm.row_groups.obs_rows[1]
    η_ind = η_vec[1]
    sol_accessors = nothing
    if dm.model.de.de !== nothing
        sol, compiled = _solve_dense_individual(dm, ind, θ, η_ind)
        sol_accessors = get_de_accessors_builder(dm.model.de.de)(sol, compiled)
    end
    vary = _varying_at_plot(dm, ind, 1, obs_rows[1])
    if dm.model.de.de === nothing && x_axis_feature !== nothing
        vary = merge(vary, (t = 0.0,))
    end
    obs = sol_accessors === nothing ?
          calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary) :
          calculate_formulas_obs(dm.model, θ, η_ind, ind.const_cov, vary, sol_accessors)
    return getproperty(obs, obs_name)
end

"""
    plot_vpc(res::FitResult; dm, n_simulations, n_sim, percentiles, show_obs_points,
             show_obs_percentiles, n_bins, seed, observables, x_axis_feature, ncols,
             kwargs_plot, save_path, obs_percentiles_mode, bandwidth,
             obs_percentiles_method, constants_re, mcmc_draws, mcmc_warmup, style)
             -> Plots.Plot

Visual Predictive Check (VPC): compares observed percentile bands to simulated
predictive percentile bands stratified by x-axis bins.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `n_simulations::Int = 100`: number of simulated datasets for the VPC envelopes.
- `percentiles::Vector = [5, 50, 95]`: percentiles to display (in [0, 100]).
- `show_obs_points::Bool = true`: overlay observed data points.
- `show_obs_percentiles::Bool = true`: overlay observed percentile lines.
- `n_bins::Union{Nothing, Int} = nothing`: number of x-axis bins; `nothing` for auto.
- `seed::Int = 12345`: random seed for reproducible simulations.
- `observables`: outcome name(s) to plot, or `nothing` for all.
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x-axis; defaults
  to time.
- `ncols::Int = 3`: number of subplot columns.
- `kwargs_plot`: extra keyword arguments forwarded to the plot.
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
- `obs_percentiles_mode::Symbol = :pooled`: `:pooled` or `:individual` percentile
  computation.
- `bandwidth::Union{Nothing, Float64} = nothing`: smoothing bandwidth for percentile
  curves, or `nothing` for no smoothing.
- `obs_percentiles_method::Symbol = :quantile`: `:quantile` or `:weighted`.
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
- `mcmc_draws::Int = 1000`, `mcmc_warmup`: MCMC draw settings.
- `style::PlotStyle = PlotStyle()`: visual style configuration.
"""
function plot_vpc(res::FitResult;
                  dm::Union{Nothing, DataModel}=nothing,
                  n_simulations::Int=100,
                  n_sim::Union{Nothing, Int}=nothing,
                  percentiles::Vector{<:Real}=[5, 50, 95],
                  show_obs_points::Bool=true,
                  show_obs_percentiles::Bool=true,
                  n_bins::Union{Nothing, Int}=nothing,
                  seed::Int=12345,
                  observables=nothing,
                  x_axis_feature::Union{Nothing, Symbol}=nothing,
                  ncols::Int=DEFAULT_PLOT_COLS,
                  serialization=nothing,
                  kwargs_plot=NamedTuple(),
                  save_path::Union{Nothing, String}=nothing,
                  plot_path::Union{Nothing, String}=nothing,
                  obs_percentiles_mode::Symbol=:pooled,
                  bandwidth::Union{Nothing, Float64}=nothing,
                  obs_percentiles_method::Symbol=:quantile,
                  constants_re::NamedTuple=NamedTuple(),
                  mcmc_draws::Int=1000,
                  mcmc_warmup::Union{Nothing, Int}=nothing,
                  style::PlotStyle=PlotStyle())
    dm = _get_dm(res, dm)
    save_path = _resolve_plot_path(save_path, plot_path)
    if n_sim !== nothing
        n_sim >= 1 || error("n_sim must be >= 1.")
        if n_simulations != 100 && n_simulations != n_sim
            error("Specify either n_simulations or n_sim, not conflicting values for both.")
        end
        n_simulations = n_sim
    end
    n_simulations >= 1 || error("n_simulations must be >= 1.")
    serialization === nothing || throw(ArgumentError("`serialization` is not supported by `plot_vpc`."))
    constants_re_use = _res_constants_re(res, constants_re)
    model = dm.model
    if model.de.de === nothing
        x_axis_feature = _require_varying_covariate(dm, x_axis_feature)
    end

    rng = Random.MersenneTwister(seed)
    obs_names = get_formulas_meta(model.formulas.formulas).obs_names
    observables === nothing && (observables = obs_names)
    percentiles = sort(Float64.(collect(percentiles)))
    (length(percentiles) >= 2 && all(0 .<= percentiles .<= 100)) || error("percentiles must be in [0,100] with length >= 2.")

    is_mcmc = res.result isa MCMCResult
    if is_mcmc
        @info "VPC for MCMC uses posterior draws to simulate observations."
        if mcmc_warmup !== nothing
            conv = res.diagnostics.convergence
            conv = merge(conv, (n_adapt=mcmc_warmup,))
            res = FitResult(res.method, res.result, res.summary,
                            FitDiagnostics(res.diagnostics.timing, res.diagnostics.optimizer, conv, res.diagnostics.notes),
                            res.data_model, res.fit_args, res.fit_kwargs)
        end
    end

    plots = Vector{Any}(undef, length(observables))
    for (oi, obs_name) in enumerate(observables)
        all_x = Float64[]
        all_y = Float64[]
        x_by_ind = Vector{Vector{Float64}}(undef, length(dm.individuals))
        y_by_ind = Vector{Vector{Float64}}(undef, length(dm.individuals))
        for (i, ind) in enumerate(dm.individuals)
            obs_rows = dm.row_groups.obs_rows[i]
            x = Float64.(_vpc_x_values(dm, ind, obs_rows, x_axis_feature))
            y = Float64.(getfield(ind.series.obs, obs_name))
            append!(all_x, x)
            append!(all_y, y)
            x_by_ind[i] = x
            y_by_ind[i] = y
        end
        n_bins_eff = _resolve_n_bins(all_x, n_bins)
        edges = _bin_edges_quantile(all_x, n_bins_eff)

        sim_x_all = Float64[]
        sim_y_all = Float64[]

        dist_rep = _representative_dist(dm, obs_name, x_axis_feature)
        is_discrete = dist_rep isa DiscreteDistribution
        is_bern = dist_rep isa Bernoulli

        if is_mcmc
            θ_draws, η_draws, _ = _mcmc_drawn_params(res, dm, constants_re_use, NamedTuple(), mcmc_draws, rng)
            n_sim = length(θ_draws)
            for s in 1:n_sim
                sim_x, sim_vals = _simulate_obs(dm, θ_draws[s], η_draws[s], obs_name, rng, x_axis_feature)
                xs = reduce(vcat, sim_x)
                ys = reduce(vcat, sim_vals)
                append!(sim_x_all, xs)
                append!(sim_y_all, ys)
            end
        else
            for s in 1:n_simulations
                θ = get_params(res; scale=:untransformed)
                level_vals = _sample_random_effects_levels(dm, θ, constants_re_use, rng)
                η_vec = _eta_vec_from_levels(dm, level_vals)
                sim_x, sim_vals = _simulate_obs(dm, θ, η_vec, obs_name, rng, x_axis_feature)
                xs = reduce(vcat, sim_x)
                ys = reduce(vcat, sim_vals)
                append!(sim_x_all, xs)
                append!(sim_y_all, ys)
            end
        end

        x_label = x_axis_feature === nothing ? "Time" : _axis_label(x_axis_feature)
        p = create_styled_plot(title=string(obs_name), xlabel=x_label,
                               ylabel=_axis_label(obs_name), style=style)
        if show_obs_points
            scatter!(p, all_x, all_y; color=style.color_primary, alpha=0.3,
                     markersize=style.marker_size, markerstrokewidth=style.marker_stroke_width,
                     label="obs")
        end

        if show_obs_percentiles && !is_discrete
            if obs_percentiles_method == :kernel
                obs_percentiles_mode == :pooled || error("obs_percentiles_mode=:per_individual is only supported with obs_percentiles_method=:quantile.")
                bw = bandwidth === nothing ? (maximum(all_x) - minimum(all_x)) / 10 : bandwidth
                xgrid = sort(unique(all_x))
                sm = _kernel_quantiles(all_x, all_y, xgrid, bw, percentiles)
                for pctl in percentiles
                    plot!(p, xgrid, sm[pctl]; color=COLOR_ACCENT, linestyle=pctl==median(percentiles) ? :solid : :dot, label="")
                end
            elseif obs_percentiles_method == :quantile
                bins = _assign_bins(all_x, edges)
                x_centers = [mean(edges[b:b+1]) for b in 1:n_bins_eff]
                obs_q = Dict{Float64, Vector{Float64}}((p => Vector{Float64}(undef, n_bins_eff)) for p in percentiles)
                if obs_percentiles_mode == :pooled
                    for b in 1:n_bins_eff
                        vals = all_y[bins .== b]
                        for pctl in percentiles
                            obs_q[pctl][b] = isempty(vals) ? NaN : quantile(vals, pctl / 100)
                        end
                    end
                elseif obs_percentiles_mode == :per_individual
                    for b in 1:n_bins_eff
                        per_ind_vals = Dict{Float64, Vector{Float64}}((p => Float64[]) for p in percentiles)
                        for (x, y) in zip(x_by_ind, y_by_ind)
                            bins_ind = _assign_bins(x, edges)
                            vals = y[bins_ind .== b]
                            isempty(vals) && continue
                            for pctl in percentiles
                                push!(per_ind_vals[pctl], quantile(vals, pctl / 100))
                            end
                        end
                        for pctl in percentiles
                            obs_q[pctl][b] = isempty(per_ind_vals[pctl]) ? NaN : mean(per_ind_vals[pctl])
                        end
                    end
                else
                    error("obs_percentiles_mode must be :pooled or :per_individual.")
                end
                for pctl in percentiles
                    x_plot, y_plot = _extend_bin_series(x_centers, obs_q[pctl], edges)
                    lbl = "obs $(pctl)%"
                    plot!(p, x_plot, y_plot; color=COLOR_ACCENT, linestyle=pctl==median(percentiles) ? :solid : :dot, label=lbl)
                end
            else
                error("obs_percentiles_method must be :quantile or :kernel.")
            end
        end

        if !isempty(sim_x_all)
            bins_sim = _assign_bins(sim_x_all, edges)
            x_centers = [mean(edges[b:b+1]) for b in 1:n_bins_eff]
            if is_discrete
                if is_bern
                    p1 = [mean(sim_y_all[bins_sim .== b]) for b in 1:n_bins_eff]
                    x_plot, y_plot = _extend_bin_series(x_centers, p1, edges)
                    scatter!(p, x_plot, y_plot; color=style.color_secondary, marker=:x,
                             markersize=style.marker_size_pmf,
                             markerstrokewidth=style.marker_stroke_width_pmf,
                             label="sim P(Y=1)")
                else
                    added = false
                    for b in 1:n_bins_eff
                        vals = sim_y_all[bins_sim .== b]
                        isempty(vals) && continue
                        lo = floor(Int, quantile(vals, 0.005))
                        hi = ceil(Int, quantile(vals, 0.995))
                        support = collect(lo:hi)
                        probs = [mean(vals .== v) for v in support]
                        lbl = added ? "" : "sim PMF"
                        scatter!(p, fill(x_centers[b], length(support)), support; marker_z=probs, color=:viridis, marker=:x,
                                 markersize=style.marker_size_pmf,
                                 markerstrokewidth=style.marker_stroke_width_pmf,
                                 label=lbl)
                        added = true
                    end
                end
            else
                sim_q = Dict{Float64, Vector{Float64}}((p => Vector{Float64}(undef, n_bins_eff)) for p in percentiles)
                for b in 1:n_bins_eff
                    vals = sim_y_all[bins_sim .== b]
                    for pctl in percentiles
                        sim_q[pctl][b] = isempty(vals) ? NaN : quantile(vals, pctl / 100)
                    end
                end
                for pctl in percentiles
                    x_plot, y_plot = _extend_bin_series(x_centers, sim_q[pctl], edges)
                    lbl = "sim $(pctl)%"
                    plot!(p, x_plot, y_plot; color=COLOR_SECONDARY, label=lbl)
                end
            end
        end

        plots[oi] = p
        xlims!(p, _pad_limits(minimum(all_x), maximum(all_x)))
    end

    p = combine_plots(plots; ncols=ncols, kwargs_plot...)
    return _save_plot!(p, save_path)
end
