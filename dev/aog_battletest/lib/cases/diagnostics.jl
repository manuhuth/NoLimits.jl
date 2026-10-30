# Diagnostics: plot_observed_profiles, plot_fits_comparison, plot_dv_pred, plot_dv_ipred,
# plot_wres_pred, plot_shrinkage, plot_multistart_waterfall,
# plot_multistart_fixed_effect_variability, plot_em_trajectories.
# Helpers `_bt_*` live in combine.jl. Bare return_panel cases live in combine.jl.

function cases_diagnostics()
    M = fixture_meta()
    cs = Case[]
    obs(f) = _bt_list(M, f, "observables")
    xfeat(f) = _bt_meta(M, f, "x_feature")
    nind(f) = _bt_nind(M, f)
    params(f) = _bt_list(M, f, "params")

    # Individual / observable / x-axis selection options shared by several functions.
    function selection(f; observable_kw = "observable", x = true, inds = true)
        o = Pair{String, String}[]
        if inds
            nind(f) >= 2 && push!(o, "inds_12" => "individuals_idx = [1, 2]")
            push!(o, "inds_1" => "individuals_idx = 1")
            nind(f) >= 4 && push!(o, "inds_range" => "individuals_idx = 2:4")
            push!(o, "inds_empty!err" => "individuals_idx = Int[]")
            push!(o, "inds_oob!err" => "individuals_idx = [10_000]")
        end
        # constant covariates (metadata x_feature) are rejected for non-ODE models
        x && append!(o, _bt_xvary(M, f))
        os = obs(f)
        if length(os) > 1
            for (k, ob) in enumerate(os)
                push!(o, "obs$(k)" => "$(observable_kw) = $(_bt_sym(ob))")
            end
        end
        push!(o, "obs_bad!err" => "$(observable_kw) = :no_such_observable")
        return o
    end

    # --- plot_observed_profiles (FitResult + DataModel) ---------------------------------
    fx_op = _bt_for(M, :plot_observed_profiles)
    op_sets(f) = vcat(
        ["default" => ""],
        selection(f),
        _bt_is_vecobs(M, f) ? ["marg1" => "marginal_idx = 1", "marg2" => "marginal_idx = 2", "marg0!err" => "marginal_idx = 0", "marg99!err" => "marginal_idx = 99"] :
            ["marg_scalar!err" => "marginal_idx = 1"],
        _bt_common(),
    )
    append!(cs, _bt_cases(:plot_observed_profiles, filter(f -> _bt_meta(M, f, "fit_type") != "Nothing", fx_op), op_sets))
    dm_sets(f) = vcat(
        ["default" => ""],
        nind(f) >= 2 ? ["inds_12" => "individuals_idx = [1, 2]"] : Pair{String, String}[],
        filter(p -> first(p) in ("xfeat", "xfeat_varying", "xfeat_constcov", "xfeat_constcov!err"), _bt_xvary(M, f)),
        _bt_is_vecobs(M, f) ? ["marg2" => "marginal_idx = 2"] : Pair{String, String}[],
        ["style_colors" => _BT_STYLES[1][2], "kwsub" => _BT_KWSUB, "save_path" => _BT_SAVE[1][2]],
    )
    append!(cs, _bt_cases(:plot_observed_profiles, fx_op, dm_sets; input = "dm", prefix = "dm_"))

    # --- plot_fits_comparison ------------------------------------------------------------
    fx_fc = _bt_for(M, :plot_fits_comparison)
    for f in fx_fc
        efits = _bt_list(M, f, "extra_fits")
        # single fit (dispatches to plot_fits)
        single = vcat(
            ["default" => ""],
            nind(f) >= 2 ? ["inds_12" => "individuals_idx = 1:2"] : Pair{String, String}[],
            ["ncols1" => "ncols = 1", "kwsub" => _BT_KWSUB],
        )
        append!(cs, _bt_cases(:plot_fits_comparison, [f], single; prefix = "single_"))
        isempty(efits) && continue
        vec_in = "[res, " * join(["fx.$(k)" for k in efits], ", ") * "]"
        nt_in = "(; fit = res, " * join(["$(k) = fx.$(k)" for k in efits], ", ") * ")"
        dict_in = "Dict(\"fit\" => res, " * join(["\"$(k)\" => fx.$(k)" for k in efits], ", ") * ")"
        e1 = first(efits)
        multi = vcat(
            ["default" => ""],
            selection(f),
            [
                "ncols1" => "ncols = 1", "ncols2" => "ncols = 2", "ncols0!err" => "ncols = 0",
                "shx_off" => "shared_x_axis = false", "shy_off" => "shared_y_axis = false",
                "shxy_off" => "shared_x_axis = false, shared_y_axis = false",
                "no_points" => "plot_data_points = false",
                "func_median" => "plot_func = median",
                "rng" => "rng = Xoshiro(7)",
                "linestyles" => "style = PlotStyle(comparison_line_styles = Dict(\"fit\" => :dash, \"$(e1)\" => :dot))",
                "default_linestyle" => "style = PlotStyle(comparison_default_linestyle = :dashdot)",
                "dm_kw" => "dm = dm",
            ],
            _bt_common(),
        )
        append!(cs, _bt_cases(:plot_fits_comparison, [f], multi; input = nt_in, prefix = "nt_"))
        vecd = ["default" => "", "inds_1" => "individuals_idx = 1", "style_colors" => _BT_STYLES[1][2]]
        append!(cs, _bt_cases(:plot_fits_comparison, [f], vecd; input = vec_in, prefix = "vec_"))
        append!(cs, _bt_cases(:plot_fits_comparison, [f], vecd; input = dict_in, prefix = "dict_"))
        append!(cs, _bt_cases(:plot_fits_comparison, [f], ["emptyvec!err" => ""]; input = "[]", prefix = ""))
        append!(cs, _bt_cases(:plot_fits_comparison, [f], ["emptynt!err" => ""]; input = "NamedTuple()", prefix = ""))
    end

    # --- plot_dv_pred / plot_dv_ipred / plot_wres_pred -------------------------------------
    for fn in (:plot_dv_pred, :plot_dv_ipred, :plot_wres_pred)
        sets(f) = vcat(
            ["default" => ""],
            selection(f; x = false, inds = false),
            fn == :plot_dv_ipred ? ["rng" => "rng = Xoshiro(7)"] : Pair{String, String}[],
            fn != :plot_dv_pred && _bt_is_vecobs(M, f) ? ["panel_vec!err" => "return_panel = true"] : Pair{String, String}[],
            fn == :plot_dv_pred ? ["return_panel_new" => "return_panel = true"] : Pair{String, String}[],
            ["dm_kw" => "dm = dm"],
            _bt_common(),
        )
        append!(cs, _bt_cases(fn, _bt_for(M, fn), sets))
        # logarithmic axes (ABLines cannot be drawn there; non-positive values are dropped
        # with a warning) on every fixture, user limits on the representative ones
        logkw = fn == :plot_wres_pred ? "kwargs_subplot = (; xscale = log10)" :
            "kwargs_subplot = (; xscale = log10, yscale = log10)"
        append!(cs, _bt_cases(fn, _bt_for(M, fn), ["log" => logkw]; full = f -> true))
        lim = fn == :plot_wres_pred ? "((0, 20), (-4, 4))" : "((0, 20), (0, 20))"
        append!(
            cs, _bt_cases(
                fn, filter(in(_BT_FULL), _bt_for(M, fn)),
                ["user_limits" => "kwargs_subplot = (; limits = $(lim))"]
            )
        )
    end

    # --- plot_shrinkage -------------------------------------------------------------------
    sh_sets = vcat(
        [
            "default" => "",
            "thr_010" => "threshold = 0.1", "thr_060" => "threshold = 0.6", "thr_0" => "threshold = 0.0",
            "thr_1" => "threshold = 1.0",
            "thr_150!err" => "threshold = 1.5", "thr_neg!err" => "threshold = -0.1",
            "bar_color" => "bar_color = \"#888888\"",
            "dm_kw" => "dm = dm",
        ],
        _bt_common(),
    )
    append!(cs, _bt_cases(:plot_shrinkage, _bt_for(M, :plot_shrinkage), sh_sets))

    # --- plot_multistart_waterfall --------------------------------------------------------
    wf_sets = vcat(["default" => ""], _bt_common())
    append!(cs, _bt_cases(:plot_multistart_waterfall, _bt_for(M, :plot_multistart_waterfall), wf_sets))

    # --- plot_multistart_fixed_effect_variability ------------------------------------------
    function fev_sets(f)
        ps = params(f)
        o = [
            "default" => "",
            "k1" => "k_best = 1", "k3" => "k_best = 3", "k100" => "k_best = 100",
            "quantiles" => "mode = :quantiles",
            "quantiles_custom" => "mode = :quantiles, quantiles = [0.25, 0.5, 0.75]",
            "quantiles_k3" => "mode = :quantiles, k_best = 3",
            "transformed" => "scale = :transformed",
            "transformed_quantiles" => "scale = :transformed, mode = :quantiles",
            "dm_kw" => "dm = dm",
            # the figure size moved from `kwargs_subplot` to `kwargs_layout`
            "kwsub_size!err" => "kwargs_subplot = (; size = (700, 500), xgridvisible = true)",
            "kwlay_size" => "kwargs_layout = (; size = (700, 500)), kwargs_subplot = (; xgridvisible = true)",
            "mode_bad!err" => "mode = :bad", "scale_bad!err" => "scale = :bad",
            "k0!err" => "k_best = 0", "incl_bad!err" => "include_parameters = :no_such_param",
            "excl_bad!err" => "exclude_parameters = :no_such_param",
        ]
        if !isempty(ps)
            push!(o, "incl1" => "include_parameters = $(_bt_sym(ps[1]))")
            push!(o, "incl_all" => "include_parameters = $(_bt_syms(ps))")
            push!(o, "incl_str" => "include_parameters = \"$(ps[1])\"")
            length(ps) >= 2 && push!(o, "excl1" => "exclude_parameters = $(_bt_sym(ps[1]))")
            push!(o, "excl_all!err" => "include_parameters = $(_bt_syms(ps)), exclude_parameters = $(_bt_syms(ps))")
        end
        return vcat(o, _bt_common())
    end
    append!(cs, _bt_cases(:plot_multistart_fixed_effect_variability, _bt_for(M, :plot_multistart_fixed_effect_variability), fev_sets))

    # --- plot_em_trajectories --------------------------------------------------------------
    function em_sets(f)
        ps = params(f)
        o = [
            "default" => "",
            "transformed" => "scale = :transformed",
            "ncols1" => "ncols = 1", "ncols2" => "ncols = 2", "ncols5" => "ncols = 5",
            "dm_kw" => "dm = dm",
            "scale_bad!err" => "scale = :bad", "ncols0!err" => "ncols = 0",
            "incl_bad!err" => "include_parameters = :no_such_param",
            "excl_bad!err" => "exclude_parameters = [:no_such_param]",
        ]
        if !isempty(ps)
            push!(o, "incl1" => "include_parameters = $(_bt_sym(ps[1]))")
            push!(o, "incl1_transformed" => "include_parameters = $(_bt_sym(ps[1])), scale = :transformed")
            length(ps) >= 2 && push!(o, "incl2" => "include_parameters = $(_bt_syms(ps[1:2]))")
            length(ps) >= 2 && push!(o, "excl1" => "exclude_parameters = $(_bt_sym(ps[1]))")
        end
        return vcat(o, _bt_common())
    end
    fx_em = _bt_for(M, :plot_em_trajectories)
    # `res` is the reloaded fit (`_SavedFittingMethod`; rejected on origin/main, plotted
    # since #383); `fx.res_em` has the method object restored, so the baseline renders it.
    append!(cs, _bt_cases(:plot_em_trajectories, fx_em, em_sets))
    em_restored = ["default" => "", "transformed" => "scale = :transformed", "ncols2" => "ncols = 2", "kwsub" => _BT_KWSUB]
    append!(cs, _bt_cases(:plot_em_trajectories, fx_em, em_restored; input = "fx.res_em", prefix = "restored_", full = f -> true))
    # Non-EM fit must be rejected.
    nonem = [f for f in _bt_for(M, :plot_dv_pred) if _bt_method(M, f) in ("Laplace", "FOCEI", "MLE")]
    isempty(nonem) || append!(cs, _bt_cases(:plot_em_trajectories, [first(nonem)], ["not_em!err" => ""]))

    return cs
end
