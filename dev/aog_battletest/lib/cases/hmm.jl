# HMM outcomes: plot_hidden_states, plot_emission_distributions (FitResult + DataModel).
# Helpers `_bt_*` live in combine.jl.

function cases_hmm()
    M = fixture_meta()
    cs = Case[]
    nind(f) = _bt_nind(M, f)
    xfeat(f) = _bt_meta(M, f, "x_feature")
    obs(f) = _bt_list(M, f, "observables")
    hasfit(f) = _bt_meta(M, f, "fit_type") != "Nothing"

    function common_sel(f)
        o = Pair{String, String}[]
        push!(o, "inds_1" => "individuals_idx = 1")
        nind(f) >= 2 && push!(o, "inds_12" => "individuals_idx = [1, 2]")
        push!(o, "inds_empty!err" => "individuals_idx = Int[]")
        push!(o, "ncols1" => "ncols = 1")
        push!(o, "ncols2" => "ncols = 2")
        push!(o, "ncols0!err" => "ncols = 0")
        push!(o, "shy_off" => "shared_y_axis = false")
        push!(o, "vector" => "figure_layout = :vector")
        push!(o, "vector_inds1" => "figure_layout = :vector, individuals_idx = 1")
        push!(o, "vector_kwlay" => "figure_layout = :vector, " * _BT_KWLAY)
        push!(o, "layout_bad!err" => "figure_layout = :grid")
        push!(o, "vector_save!err" => "figure_layout = :vector, save_path = joinpath(OUT, \"x.png\")")
        push!(o, "obs_bad!err" => "observable = :no_such_observable")
        if length(obs(f)) > 1
            for (k, ob) in enumerate(obs(f))
                push!(o, "obs$(k)" => "observable = $(_bt_sym(ob))")
            end
        end
        push!(o, "dm_kw" => "dm = dm")
        push!(o, "rng" => "rng = Xoshiro(7)")
        return o
    end

    # --- plot_hidden_states ----------------------------------------------------------------
    function hs_sets(f)
        o = vcat(["default" => ""], common_sel(f))
        push!(o, "shx_off" => "shared_x_axis = false")
        push!(o, "shxy_off" => "shared_x_axis = false, shared_y_axis = false")
        push!(o, "mcmc_draws1" => "mcmc_draws = 1")
        xfeat(f) === nothing || push!(o, "xfeat" => "x_axis_feature = $(_bt_sym(xfeat(f)))")
        push!(o, "xfeat_bad!err" => "x_axis_feature = :no_such_feature")
        return vcat(o, _bt_common())
    end
    fx_hs = _bt_for(M, :plot_hidden_states)
    append!(cs, _bt_cases(:plot_hidden_states, filter(hasfit, fx_hs), hs_sets))
    hs_dm = vcat(
        [
            "default" => "", "vector" => "figure_layout = :vector", "inds_1" => "individuals_idx = 1",
            "ncols1" => "ncols = 1", "layout_bad!err" => "figure_layout = :grid",
        ],
        _BT_STYLES[1:1], ["kwsub" => _BT_KWSUB, "save_path" => _BT_SAVE[1][2]],
    )
    append!(cs, _bt_cases(:plot_hidden_states, fx_hs, hs_dm; input = "dm", prefix = "dm_"))

    # --- plot_emission_distributions --------------------------------------------------------
    tp = "time_point = sort(unique(skipmissing(NoLimits.get_df(dm)[!, NoLimits.get_time_col(dm)])))[2]"
    function em_sets(f)
        o = vcat(["default" => ""], common_sel(f))
        append!(
            o, [
                "tidx1" => "time_idx = 1", "tidx2" => "time_idx = 2",
                "tpoint" => tp,
                "tidx_tpoint!err" => "time_idx = 1, " * tp,
                "tidx_oob!err" => "time_idx = 10_000",
                "tcol" => "time_col = NoLimits.get_time_col(dm)",
                "mcmc_draws1" => "mcmc_draws = 1",
            ]
        )
        return vcat(o, _bt_common())
    end
    fx_em = _bt_for(M, :plot_emission_distributions)
    append!(cs, _bt_cases(:plot_emission_distributions, filter(hasfit, fx_em), em_sets))
    em_dm = vcat(
        [
            "default" => "", "vector" => "figure_layout = :vector", "tidx2" => "time_idx = 2",
            "inds_1" => "individuals_idx = 1", "tidx_tpoint!err" => "time_idx = 1, time_point = 1.0",
        ],
        _BT_STYLES[1:1], ["kwsub" => _BT_KWSUB, "save_path" => _BT_SAVE[1][2]],
    )
    append!(cs, _bt_cases(:plot_emission_distributions, fx_em, em_dm; input = "dm", prefix = "dm_"))
    return cs
end
