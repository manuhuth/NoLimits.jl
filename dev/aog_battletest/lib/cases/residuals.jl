# Residual diagnostics: plot_residuals, plot_residual_distribution, plot_residual_qq,
# plot_residual_pit, plot_residual_acf (FitResult + DataModel methods).
# Helpers `_bt_*` live in combine.jl.

function cases_residuals()
    M = fixture_meta()
    cs = Case[]
    nind(f) = _bt_nind(M, f)
    xfeat(f) = _bt_meta(M, f, "x_feature")
    obs(f) = _bt_list(M, f, "observables")
    hasfit(f) = _bt_meta(M, f, "fit_type") != "Nothing"
    post(f) = _bt_is_posterior(M, f)
    discrete(f) = _bt_has_tag(M, f, :discrete, :count, :binary, :poisson, :bernoulli) ||
        occursin(r"poisson|bernoulli|count|binary", f)
    # Posterior-draw fixtures: keep every non-default call cheap.
    base(f) = post(f) ? "mcmc_draws = 50" : ""

    # Selection + forwarded get_residuals options (shared by all five functions).
    function shared(f; with_metric = true, shared_axes = true)
        o = Pair{String, String}[]
        if with_metric
            for r in (:pit, :quantile, :raw, :pearson, :logscore)
                push!(o, "res_$(r)" => "residual = :$(r)")
            end
            push!(o, "res_bad!err" => "residual = :not_a_metric")
            push!(o, "fitted_median" => "fitted_stat = median")
        end
        nind(f) >= 2 && push!(o, "inds_12" => "individuals_idx = [1, 2]")
        push!(o, "inds_1" => "individuals_idx = 1")
        push!(o, "inds_empty!err" => "individuals_idx = Int[]")
        push!(o, "obsrows_12" => "obs_rows = [1, 2]")
        push!(o, "obsrows_oob!err" => "obs_rows = [10_000]")
        append!(o, _bt_xvary(M, f))
        os = obs(f)
        if length(os) > 1
            for (k, ob) in enumerate(os)
                push!(o, "obs$(k)" => "observables = $(_bt_sym(ob))")
            end
            push!(o, "obs_all" => "observables = $(_bt_syms(os))")
        end
        push!(o, "obs_bad!err" => "observables = :no_such_observable")
        push!(o, "ncols1" => "ncols = 1")
        push!(o, "ncols2" => "ncols = 2")
        push!(o, "ncols0!err" => "ncols = 0")
        if shared_axes
            push!(o, "shx_off" => "shared_x_axis = false")
            push!(o, "shy_off" => "shared_y_axis = false")
            push!(o, "shxy_off" => "shared_x_axis = false, shared_y_axis = false")
        end
        push!(o, "rng" => "rng = Xoshiro(7)")
        push!(o, "nocache_dists" => "cache_obs_dists = false")
        hasfit(f) && !post(f) && push!(o, "cache" => "cache = build_plot_cache(res; cache_obs_dists = true)")
        push!(o, "cdf_mc" => "cdf_fallback_mc = 50")
        discrete(f) && push!(o, "no_randomize" => "randomize_discrete = false")
        push!(o, "dm_kw" => "dm = dm")
        push!(o, "mcq_bad!err" => "mcmc_quantiles = [-5, 95]")
        if post(f)
            push!(o, "mcmc_draws20" => "mcmc_draws = 20")
            push!(o, "mcmc_warmup" => "mcmc_draws = 20, mcmc_warmup = 5")
            push!(o, "mcmc_q" => "mcmc_draws = 20, mcmc_quantiles = [10, 90]")
            push!(o, "mcmc_draws0!err" => "mcmc_draws = 0")
        end
        return o
    end

    # DataModel method: evaluated at the model's initial parameters.
    function dm_sets(f; with_metric = true)
        o = ["default" => ""]
        with_metric && push!(o, "res_raw" => "residual = :raw")
        with_metric && push!(o, "res_pit" => "residual = :pit")
        push!(o, "cache" => "cache = build_plot_cache(dm; cache_obs_dists = true)")
        push!(o, "ncols1" => "ncols = 1")
        nind(f) >= 2 && push!(o, "inds_12" => "individuals_idx = [1, 2]")
        push!(o, "style_colors" => _BT_STYLES[1][2])
        push!(o, "kwsub" => _BT_KWSUB)
        push!(o, "save_path" => _BT_SAVE[1][2])
        return o
    end

    specs = [
        (:plot_residuals, f -> Pair{String, String}[], true, true),
        (:plot_residual_distribution, f -> ["bins5" => "bins = 5", "bins50" => "bins = 50", "bins_pit" => "residual = :pit, bins = 10"], true, true),
        (:plot_residual_qq, f -> Pair{String, String}[], true, false),
        (
            :plot_residual_pit, f -> [
                "hist_only" => "show_hist = true, show_kde = false, show_qq = false",
                "kde_only" => "show_hist = false, show_kde = true, show_qq = false",
                "qq_only" => "show_hist = false, show_kde = false, show_qq = true",
                "hist_kde" => "show_hist = true, show_kde = true",
                "hist_qq" => "show_hist = true, show_qq = true",
                "all3" => "show_hist = true, show_kde = true, show_qq = true",
                "none!err" => "show_hist = false, show_kde = false, show_qq = false",
                "kde_bw" => "show_kde = true, kde_bandwidth = 0.05",
                "kde_bw_wide" => "show_kde = true, kde_bandwidth = 0.5",
            ], false, false,
        ),
        (
            :plot_residual_acf, f -> [
                "lag1" => "max_lag = 1", "lag10" => "max_lag = 10", "lag0!err" => "max_lag = 0",
                "lag2_raw" => "max_lag = 2, residual = :raw",
            ], true, false,
        ),
    ]
    for (fn, extra, with_metric, shared_axes) in specs
        fxs = _bt_for(M, fn)
        sets(f) = vcat(["default" => ""], shared(f; with_metric, shared_axes), extra(f), _bt_common())
        append!(cs, _bt_cases(fn, filter(hasfit, fxs), sets; base = base))
        append!(cs, _bt_cases(fn, filter(f -> !post(f), fxs), f -> dm_sets(f; with_metric); input = "dm", prefix = "dm_"))
    end
    return cs
end
