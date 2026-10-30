# plot_vpc, plot_observation_distributions (FitResult + DataModel), plot_uq_distributions.
# Helpers `_bt_*` live in combine.jl. The bare `plot_vpc(...; return_panel = true)` case is
# in combine.jl.

function cases_vpc_obs_uq()
    M = fixture_meta()
    cs = Case[]
    nind(f) = _bt_nind(M, f)
    xfeat(f) = _bt_meta(M, f, "x_feature")
    obs(f) = _bt_list(M, f, "observables")
    hasfit(f) = _bt_meta(M, f, "fit_type") != "Nothing"
    post(f) = _bt_is_posterior(M, f)

    # --- plot_vpc ---------------------------------------------------------------------------
    # Fixture runs: every non-default call uses few simulations (n_simulations = 30).
    vbase(f) = post(f) ? "n_simulations = 30, mcmc_draws = 30" : "n_simulations = 30"
    function vpc_sets(f)
        o = Pair{String, String}[
            "default" => "",
            "nsim20" => "n_simulations = 20",
            "nsim_alias" => "n_sim = 20",
            "nsim_both_equal" => "n_simulations = 20, n_sim = 20",
            "pct_10_50_90" => "percentiles = [10, 50, 90]",
            "pct_25_75" => "percentiles = [25, 75]",
            "pct_5_25_50_75_95" => "percentiles = [5, 25, 50, 75, 95]",
            "no_obs_points" => "show_obs_points = false",
            "no_obs_pct" => "show_obs_percentiles = false",
            "no_obs_both" => "show_obs_points = false, show_obs_percentiles = false",
            "nbins2" => "n_bins = 2", "nbins8" => "n_bins = 8", "nbins1" => "n_bins = 1",
            "kernel_bw" => "obs_percentiles_method = :kernel, bandwidth = 0.5",
            "kernel_bw_small" => "obs_percentiles_method = :kernel, bandwidth = 0.05",
            "quantile" => "obs_percentiles_method = :quantile",
            "quantile_nbins4" => "obs_percentiles_method = :quantile, n_bins = 4",
            "quantile_pooled" => "obs_percentiles_method = :quantile, obs_percentiles_mode = :pooled",
            "quantile_perind" => "obs_percentiles_method = :quantile, obs_percentiles_mode = :per_individual",
            "quantile_perind_nbins3" => "obs_percentiles_method = :quantile, obs_percentiles_mode = :per_individual, n_bins = 3",
            "seed1" => "seed = 1",
            "ncols1" => "ncols = 1", "ncols2" => "ncols = 2",
            "kwplot" => "kwargs_plot = (; size = (900, 600))",
            "dm_kw" => "dm = dm",
            # validation
            "nsim0!err" => "n_simulations = 0", "nsim_alias0!err" => "n_sim = 0",
            "nsim_conflict!err" => "n_simulations = 6, n_sim = 5",
            "pct_one!err" => "percentiles = [50]", "pct_neg!err" => "percentiles = [-5, 50, 95]",
            "pct_gt100!err" => "percentiles = [5, 50, 105]",
            "nbins0!err" => "n_bins = 0",
            "method_bad!err" => "obs_percentiles_method = :invalid",
            "mode_bad!err" => "obs_percentiles_method = :quantile, obs_percentiles_mode = :invalid",
            "perind_kernel!err" => "obs_percentiles_mode = :per_individual, obs_percentiles_method = :kernel",
            "bw_neg!err" => "bandwidth = -1.0",
            "serialization!err" => "serialization = 1",
            "obs_bad!err" => "observables = :no_such_observable",
        ]
        append!(o, _bt_xvary(M, f))
        push!(o, "xfeat_time_quantile" => "x_axis_feature = NoLimits.get_time_col(dm), obs_percentiles_method = :quantile, n_bins = 3")
        os = obs(f)
        if length(os) > 1
            for (k, ob) in enumerate(os)
                push!(o, "obs$(k)" => "observables = $(_bt_sym(ob))")
            end
            push!(o, "obs_all" => "observables = $(_bt_syms(os))")
            push!(o, "panel_multi!err" => "return_panel = true")
        end
        if post(f)
            push!(o, "mcmc_warmup" => "mcmc_draws = 20, mcmc_warmup = 5")
        end
        return vcat(o, _bt_common(layout = false))
    end
    append!(cs, _bt_cases(:plot_vpc, filter(hasfit, _bt_for(M, :plot_vpc)), vpc_sets; base = vbase))

    # --- plot_observation_distributions --------------------------------------------------
    obase(f) = post(f) ? "mcmc_draws = 50" : ""
    function od_sets(f)
        o = Pair{String, String}["default" => ""]
        nind(f) >= 2 && push!(o, "ind2" => "individuals_idx = 2")
        nind(f) >= 2 && push!(o, "inds_12" => "individuals_idx = [1, 2]")
        nind(f) >= 3 && push!(o, "inds_123" => "individuals_idx = 1:3")
        push!(o, "inds_empty!err" => "individuals_idx = Int[]")
        push!(o, "obsrows_12" => "obs_rows = [1, 2]")
        push!(o, "obsrows_1" => "obs_rows = 1")
        push!(o, "obsrows_oob!err" => "obs_rows = [10_000]")
        push!(o, "obsrows_empty!err" => "obs_rows = Int[]")
        os = obs(f)
        if length(os) > 1
            for (k, ob) in enumerate(os)
                push!(o, "obs$(k)" => "observables = $(_bt_sym(ob))")
            end
            push!(o, "obs_all" => "observables = $(_bt_syms(os))")
        end
        push!(o, "obs_bad!err" => "observables = :no_such_observable")
        push!(o, "obs_empty!err" => "observables = Symbol[]")
        append!(o, _bt_xvary(M, f))
        append!(
            o, [
                "shx_off" => "shared_x_axis = false", "shy_off" => "shared_y_axis = false",
                "shxy_off" => "shared_x_axis = false, shared_y_axis = false",
                "ncols1" => "ncols = 1", "ncols2" => "ncols = 2", "ncols0!err" => "ncols = 0",
                "cache_dists" => "cache_obs_dists = true",
                "rng" => "rng = Xoshiro(7)",
                "dm_kw" => "dm = dm",
                "alpha_bad!err" => "mcmc_quantiles_alpha = 2.0",
            ]
        )
        post(f) || push!(o, "cache" => "cache = build_plot_cache(res; cache_obs_dists = true)")
        if post(f)
            append!(
                o, [
                    "mcmc_draws20" => "mcmc_draws = 20",
                    "mcmc_warmup" => "mcmc_draws = 20, mcmc_warmup = 5",
                    "mcmc_q" => "mcmc_draws = 20, mcmc_quantiles = [10, 90]",
                    "mcmc_alpha" => "mcmc_draws = 20, mcmc_quantiles_alpha = 0.3",
                ]
            )
        end
        return vcat(o, _bt_common())
    end
    fx_od = _bt_for(M, :plot_observation_distributions)
    append!(cs, _bt_cases(:plot_observation_distributions, filter(hasfit, fx_od), od_sets; base = obase))
    function od_dm(f)
        o = Pair{String, String}["default" => ""]
        nind(f) >= 2 && push!(o, "inds_12" => "individuals_idx = [1, 2]")
        push!(o, "ncols1" => "ncols = 1")
        push!(o, "obsrows_12" => "obs_rows = [1, 2]")
        push!(o, "style_colors" => _BT_STYLES[1][2])
        push!(o, "kwsub" => _BT_KWSUB)
        push!(o, "save_path" => _BT_SAVE[1][2])
        return o
    end
    append!(cs, _bt_cases(:plot_observation_distributions, filter(f -> !post(f), fx_od), od_dm; input = "dm", prefix = "dm_"))

    # --- plot_uq_distributions -------------------------------------------------------------
    names_expr(k) = "NoLimits.get_uq_parameter_names(fx.$(k))"
    for f in _bt_for(M, :plot_uq_distributions)
        for k in filter(e -> startswith(e, "uq_"), _bt_extras(M, f))
            sets = Pair{String, String}[
                "default" => "",
                "transformed" => "scale = :transformed",
                "param1" => "parameters = first($(names_expr(k)))",
                "param2" => "parameters = $(names_expr(k))[1:min(2, end)]",
                "param_bad!err" => "parameters = :no_such_param",
                "ialpha" => "interval_alpha = 0.6",
                "ialpha_bad!err" => "interval_alpha = 1.5",
                "halpha_bad!err" => "histogram_alpha = -0.1",
                "no_estimate" => "show_estimate = false",
                "no_interval" => "show_interval = false",
                "no_est_no_int" => "show_estimate = false, show_interval = false",
                "legend" => "show_legend = true",
                "histogram" => "plot_type = :histogram",
                "histogram_bins10" => "plot_type = :histogram, bins = 10, histogram_alpha = 0.8",
                "histogram_transformed" => "plot_type = :histogram, scale = :transformed",
                "bins5" => "bins = 5",
                "kde_bw" => "kde_bandwidth = 0.1",
                "ptype_bad!err" => "plot_type = :violin",
                "ncols1" => "ncols = 1", "ncols2" => "ncols = 2", "ncols0!err" => "ncols = 0",
                "legend_transformed" => "show_legend = true, scale = :transformed",
            ]
            sets = vcat(sets, _bt_common())
            # profile UQ has no draws: the OLD implementation rejected every plot; the port
            # draws interval spans + estimates, and histograms (or nothing left to draw) error
            if k == "uq_profile"
                sets = filter(p -> !(occursin("histogram", first(p)) || first(p) in ("no_est_no_int", "bins5", "kde_bw")), sets)
                append!(sets, ["histogram!err" => "plot_type = :histogram", "no_est_no_int!err" => "show_estimate = false, show_interval = false"])
            end
            append!(cs, _bt_cases(:plot_uq_distributions, [f], sets; input = "fx.$(k)", prefix = replace(k, "uq_" => "") * "_", full = ff -> ff in _BT_FULL || k != "uq_wald"))
        end
    end
    return cs
end
