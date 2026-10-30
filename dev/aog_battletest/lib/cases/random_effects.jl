# Random-effect diagnostics: plot_random_effects_pdf, plot_random_effects_scatter,
# plot_random_effect_pairplot, plot_random_effect_distributions, plot_random_effect_pit,
# plot_random_effect_standardized, plot_random_effect_standardized_scatter.
# Helpers `_bt_*` live in combine.jl.

const _BT_RE_FULL = ("mvre_multigroup_laplace", "flow_re_laplace", "single_individual_re_laplace")

function cases_random_effects()
    M = fixture_meta()
    cs = Case[]
    nind(f) = _bt_nind(M, f)
    renames(f) = _bt_list(M, f, "re_names")
    covs(f) = _bt_list(M, f, "covariates")
    post(f) = _bt_is_posterior(M, f)
    base(f) = post(f) ? "mcmc_draws = 50" : ""
    # First two levels of the first random effect, computed at run time from the DataModel.
    lv2(re) = "levels = collect(getfield(NoLimits.get_re_values(NoLimits.get_re_group_info(dm)), $(_bt_sym(re))))[1:min(2, end)]"
    constcov = "x_covariate = first(NoLimits.get_covariates(NoLimits.get_model(dm)).constants)"

    # `has_ncols = false` (the pairplot matrix): `ncols` is not a keyword and must be rejected
    function sel(f; shared_axes = true, has_ncols = true)
        o = Pair{String, String}[]
        rn = renames(f)
        if !isempty(rn)
            push!(o, "re1" => "re_names = $(_bt_sym(rn[1]))")
            push!(o, "re1_str!err" => "re_names = \"$(rn[1])\"")
            push!(o, "re_all" => "re_names = $(_bt_syms(rn))")
            length(rn) >= 2 && push!(o, "re2" => "re_names = $(_bt_sym(rn[2]))")
            push!(o, "levels2" => "re_names = $(_bt_sym(rn[1])), " * lv2(rn[1]))
            push!(o, "levels_bad!err" => "re_names = $(_bt_sym(rn[1])), levels = [\"no_such_level\"]")
        end
        push!(o, "re_bad!err" => "re_names = :no_such_re")
        nind(f) >= 2 && push!(o, "inds_12" => "individuals_idx = [1, 2]")
        push!(o, "inds_1" => "individuals_idx = 1")
        e = has_ncols ? "" : "!err"
        push!(o, "ncols1$(e)" => "ncols = 1")
        push!(o, "ncols2$(e)" => "ncols = 2")
        push!(o, "ncols0!err" => "ncols = 0")
        if shared_axes
            push!(o, "shx_off" => "shared_x_axis = false")
            push!(o, "shy_off" => "shared_y_axis = false")
            push!(o, "shxy_off" => "shared_x_axis = false, shared_y_axis = false")
        end
        push!(o, "rng" => "rng = Xoshiro(7)")
        push!(o, "dm_kw" => "dm = dm")
        return o
    end
    mcmc_full(f) = post(f) ? [
            "mcmc_draws20" => "mcmc_draws = 20",
            "mcmc_warmup" => "mcmc_draws = 20, mcmc_warmup = 5",
            "mcmc_q" => "mcmc_draws = 20, mcmc_quantiles = [10, 90]",
            "mcmc_alpha" => "mcmc_draws = 20, mcmc_quantiles_alpha = 0.3",
        ] : Pair{String, String}[]
    # mcmc_quantiles is only validated (and used) for posterior-draw fits
    mcq_bad(f) = post(f) ? ["mcq_bad!err" => "mcmc_quantiles = [50]", "mcq_neg!err" => "mcmc_quantiles = [-5, 95]"] : Pair{String, String}[]
    mcmc_draws_only(f) = post(f) ? ["mcmc_draws20" => "mcmc_draws = 20"] : Pair{String, String}[]
    flow_opts = [
        "flow_hist" => "flow_plot = :hist, flow_samples = 100, flow_bins = 10",
        "flow_kde_bw" => "flow_plot = :kde, flow_samples = 100, flow_bandwidth = 0.2",
        "flow_bins0!err" => "flow_bins = 0", "flow_samples1!err" => "flow_samples = 1",
        "flow_bw_neg!err" => "flow_bandwidth = -1.0",
    ]
    covx(f) = isempty(covs(f)) ? Pair{String, String}[] : ["xcov" => constcov]

    specs = [
        (
            :plot_random_effects_pdf, f -> vcat(
                sel(f), mcmc_full(f), mcq_bad(f), flow_opts,
                [
                    "xq90" => "x_quantile = 0.9", "xq999" => "x_quantile = 0.999", "xlims" => "xlims = (-3.0, 3.0)",
                    "alpha_bad!err" => "mcmc_quantiles_alpha = 1.5",
                ],
            ),
        ),
        (
            :plot_random_effect_distributions, f -> vcat(
                sel(f), mcmc_full(f), mcq_bad(f), flow_opts,
                [
                    "xq90" => "x_quantile = 0.9", "xlims" => "xlims = (-3.0, 3.0)",
                    "alpha_bad!err" => "mcmc_quantiles_alpha = 1.5",
                ],
            ),
        ),
        (
            :plot_random_effects_scatter, f -> vcat(
                sel(f; shared_axes = false), mcmc_draws_only(f), covx(f),
                ["xcov_bad!err" => "x_covariate = :no_such_covariate"],
            ),
        ),
        (
            :plot_random_effect_pairplot, f -> vcat(
                sel(f; shared_axes = false, has_ncols = false), mcmc_draws_only(f),
                [
                    "kde_bw" => "kde_bandwidth = 0.3", "kde_bw_small" => "kde_bandwidth = 0.02",
                    "kde_bw_bad!err" => "kde_bandwidth = -1.0",
                ],
            ),
        ),
        (
            :plot_random_effect_pit, f -> vcat(
                sel(f), mcmc_full(f),
                [
                    "hist_only" => "show_hist = true, show_kde = false, show_qq = false",
                    "kde_only" => "show_hist = false, show_kde = true",
                    "qq_only" => "show_hist = false, show_qq = true",
                    "hist_kde" => "show_hist = true, show_kde = true",
                    "all3" => "show_hist = true, show_kde = true, show_qq = true",
                    "kde_bw" => "show_kde = true, kde_bandwidth = 0.1",
                    "flow_samples" => "flow_samples = 100",
                    "xcov_bad!err" => "x_covariate = :no_such_covariate",
                    "alpha_bad!err" => "mcmc_quantiles_alpha = 1.5",
                ],
                isempty(covs(f)) ? Pair{String, String}[] : ["xcov!err" => constcov],
            ),
        ),
        (
            :plot_random_effect_standardized, f -> vcat(
                sel(f; shared_axes = false), mcmc_draws_only(f),
                [
                    "hist_only" => "show_hist = true, show_kde = false",
                    "kde_only" => "show_hist = false, show_kde = true",
                    "hist_kde" => "show_hist = true, show_kde = true",
                    "none" => "show_hist = false, show_kde = false",
                    "kde_bw" => "show_kde = true, kde_bandwidth = 0.3",
                    "flow_samples" => "flow_samples = 100",
                ],
            ),
        ),
        (
            :plot_random_effect_standardized_scatter, f -> vcat(
                sel(f; shared_axes = false), mcmc_draws_only(f), covx(f),
                ["xcov_bad!err" => "x_covariate = :no_such_covariate", "flow_samples" => "flow_samples = 100"],
            ),
        ),
    ]
    # Full option sweep also on the RE-specific fixtures (multivariate + several grouping
    # columns, flow RE, VI posterior, single individual).
    full(f) = f in _BT_FULL || f in _BT_RE_FULL
    for (fn, extra) in specs
        sets(f) = vcat(["default" => ""], extra(f), _bt_common())
        append!(cs, _bt_cases(fn, _bt_for(M, fn), sets; base = base, full = full))
        # Fits without random effects (MLE, VI/MCMC without REs) must be rejected.
        nore = sort([n for (n, d) in M if _bt_has_tag(M, n, :no_re) && get(d, "method", "none") != "none"])
        append!(cs, cross(fn, nore, ["nore!err" => ""]))
    end
    return cs
end
