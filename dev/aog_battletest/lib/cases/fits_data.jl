# plot_fits / plot_data cases (reference implementations of the AoG port).

const _STYLE_ALT = "PlotStyle(color_primary = \"#222222\", color_secondary = \"#CC0000\", " *
    "marker_size = 5, marker_alpha = 1.0, line_width_primary = 3.0, font_size_title = 10, " *
    "font_size_label = 10, font_size_tick = 8, base_subplot_width = 220, base_subplot_height = 180)"

_meta(f, k, default = String[]) = get(get(fixture_meta(), f, Dict()), k, default)

function _common_layout_optsets()
    return [
        "default" => "",
        "ncols1" => "ncols = 1",
        "ncols2" => "ncols = 2",
        "ncols5" => "ncols = 5",
        "unshared_x" => "shared_x_axis = false",
        "unshared_y" => "shared_y_axis = false",
        "unshared_xy" => "shared_x_axis = false, shared_y_axis = false",
        "first2" => "individuals_idx = [1, 2]",
        "first1" => "individuals_idx = 1",
        "style" => "style = $(_STYLE_ALT)",
        "kwsub" => "kwargs_subplot = (; xlabel = \"X custom\", ylabel = \"Y custom\", xgridvisible = true)",
        "kwlayout" => "kwargs_layout = (; size = (900, 600))",
        "save" => "save_path = joinpath(OUT, \"x.png\")",
        "plotpath" => "plot_path = joinpath(OUT, \"x.png\")",
        "ncols0!err" => "ncols = 0",
        "badlayout!err" => "marginal_layout = :bogus",
    ]
end

_nind(f) = get(get(fixture_meta(), f, Dict()), "n_individuals", 0)
# `first2` needs two individuals
_applicable(f, opts) = _nind(f) >= 2 ? opts : filter(p -> first(p) != "first2", opts)

function cases_fits_data()
    cs = Case[]
    # ---------------- plot_data ----------------
    for f in fixtures_for(:plot_data)
        opts = _applicable(f, _common_layout_optsets())
        input = get(get(fixture_meta(), f, Dict()), "fit_type", "") == "Nothing" ? "dm" : "res"
        for c in cross(:plot_data, [f], opts; input = input)
            occursin("__save", c.id) || occursin("__plotpath", c.id) ?
                push!(cs, Case(c.id, c.fn, c.fixture, c.call; tags = c.tags, files = ["x.png"])) :
                push!(cs, c)
        end
        append!(cs, cross(:plot_data, [f], ["dm_default" => "", "dm_ncols2" => "ncols = 2"]; input = "dm"))
        xf = _meta(f, "x_feature", "")
        isempty(xf) || append!(cs, cross(:plot_data, [f], ["xfeature" => "x_axis_feature = :$(xf)"]))
        if has_tag(f, "vector_obs")
            append!(
                cs, cross(
                    :plot_data, [f], [
                        "vector" => "marginal_layout = :vector",
                        "vector_ncols2" => "marginal_layout = :vector, ncols = 2",
                        "vector_save!err" => "marginal_layout = :vector, save_path = joinpath(OUT, \"x.png\")",
                    ]
                )
            )
        end
    end
    # ---------------- plot_fits ----------------
    for f in fixtures_for(:plot_fits)
        opts = vcat(
            _common_layout_optsets(), [
                "nodata" => "plot_data_points = false",
                "median" => "plot_func = median",
                "density" => "plot_density = true",
                "density_ncols2" => "plot_density = true, ncols = 2, shared_y_axis = false",
                "cache" => "cache = build_plot_cache(res)",
                "alpha_bad!err" => "mcmc_quantiles_alpha = 1.5",
            ]
        )
        for c in cross(:plot_fits, [f], _applicable(f, opts))
            occursin("__save", c.id) || occursin("__plotpath", c.id) ?
                push!(cs, Case(c.id, c.fn, c.fixture, c.call; tags = c.tags, files = ["x.png"])) :
                push!(cs, c)
        end
        append!(cs, cross(:plot_fits, [f], ["dm_default" => "", "dm_ncols2" => "ncols = 2"]; input = "dm"))
        xf = _meta(f, "x_feature", "")
        isempty(xf) || append!(cs, cross(:plot_fits, [f], ["xfeature" => "x_axis_feature = :$(xf)"]))
        obs = _meta(f, "observables")
        if length(obs) > 1
            append!(
                cs, cross(
                    :plot_fits, [f], [
                        "obs_$(o)" => "observable = :$(o)" for o in obs
                    ]
                )
            )
            append!(cs, cross(:plot_fits, [f], ["obs_bad!err" => "observable = :not_an_outcome"]))
        end
        if has_tag(f, "vector_obs")
            append!(
                cs, cross(
                    :plot_fits, [f], [
                        "vector" => "marginal_layout = :vector",
                        "vector_density" => "marginal_layout = :vector, plot_density = true",
                        "marginal1" => "marginal_idx = 1",
                        "marginal2_density" => "marginal_idx = 2, plot_density = true",
                        "marginal9!err" => "marginal_idx = 9",
                        "vector_save!err" => "marginal_layout = :vector, save_path = joinpath(OUT, \"x.png\")",
                    ]
                )
            )
        else
            append!(cs, cross(:plot_fits, [f], ["marginal1!err" => "marginal_idx = 1"]))
        end
        if has_tag(f, "posterior")
            append!(
                cs, cross(
                    :plot_fits, [f], [
                        "mcmc_q" => "plot_mcmc_quantiles = true, mcmc_draws = 40",
                        "mcmc_q3" => "plot_mcmc_quantiles = true, mcmc_quantiles = [10, 50, 90], mcmc_quantiles_alpha = 0.4, mcmc_draws = 40",
                        "mcmc_warmup" => "mcmc_draws = 30, mcmc_warmup = 10",
                        "mcmc_density" => "plot_density = true, mcmc_draws = 30",
                        "mcmc_q_and_density" => "plot_density = true, plot_mcmc_quantiles = true, mcmc_draws = 30",
                        "mcmc_qbad!err" => "plot_mcmc_quantiles = true, mcmc_quantiles = [5], mcmc_draws = 20",
                    ]
                )
            )
        end
    end
    return cs
end
