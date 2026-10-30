# combine_plots + bare `return_panel = true` panels, and the shared case-building helpers
# (`_bt_*`) used by the other case files in this directory (diagnostics, hmm, residuals,
# random_effects, vpc_obs_uq). All case files are included into the same module before
# any generator runs, so the helpers are visible everywhere.

# ---------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------

# Metadata lookup (strings / string vectors, see lib/common.jl); `nothing` when absent.
_bt_meta(M, f, k) = get(get(M, f, Dict{String, Any}()), k, nothing)
_bt_list(M, f, k) = (v = _bt_meta(M, f, k); v === nothing ? String[] : (v isa AbstractVector ? String.(v) : [String(v)]))
# `fixtures_for` on an already-parsed metadata dict (one TOML parse per generator).
_bt_for(M, fn) = sort([n for (n, d) in M if string(fn) in get(d, "feeds", String[])])
_bt_nind(M, f) = (v = _bt_meta(M, f, "n_individuals"); v === nothing ? 0 : Int(v))
_bt_method(M, f) = string(something(_bt_meta(M, f, "method"), ""))
_bt_extras(M, f) = _bt_list(M, f, "extras")
_bt_is_posterior(M, f) = _bt_method(M, f) in ("MCMC", "VI")
_bt_is_multistart(M, f) = occursin("Multistart", string(something(_bt_meta(M, f, "fit_type"), "")))
_bt_has_tag(M, f, tags...) = any(t -> string(t) in _bt_list(M, f, "tags"), tags)
# Vector-valued (multivariate) observable fixtures.
_bt_is_vecobs(M, f) = _bt_has_tag(M, f, :vector_obs, :multivariate, :mvnormal, :vector_valued)

# Julia source for a Symbol / Symbol vector (`Symbol("..")` is safe for any name).
_bt_sym(s) = "Symbol(\"" * string(s) * "\")"
_bt_syms(v) = "[" * join([_bt_sym(s) for s in v], ", ") * "]"

# Options shared across functions.
const _BT_STYLES = [
    "style_colors" => "style = PlotStyle(color_primary = \"#D62728\", color_secondary = \"#9467BD\", color_accent = \"#17BECF\", color_dark = \"#7F7F7F\", color_density = \"#BCBD22\", color_reference = \"#8C564B\")",
    "style_marker" => "style = PlotStyle(marker_size = 18, marker_size_small = 10, marker_alpha = 0.3, marker_stroke_width = 1.5, marker_size_pmf = 18)",
    "style_fonts" => "style = PlotStyle(font_family = \"TeX Gyre Heros Makie Italic\", font_size_title = 22, font_size_label = 18, font_size_tick = 8, font_size_legend = 14, font_size_annotation = 14)",
    "style_size" => "style = PlotStyle(base_subplot_width = 220, base_subplot_height = 160, line_width_primary = 4.0, line_width_secondary = 3.0, figure_padding = 40)",
]
const _BT_KWSUB = "kwargs_subplot = (; xlabel = \"X custom\", xgridvisible = true, ygridvisible = false)"
const _BT_KWLAY = "kwargs_layout = (; size = (900, 600))"
const _BT_SAVE = [
    "save_path" => "save_path = joinpath(OUT, \"x.png\")",
    "plot_path" => "plot_path = joinpath(OUT, \"x.png\")",
    "save_noext" => "save_path = joinpath(OUT, \"x\")",
    "save_pdf" => "save_path = joinpath(OUT, \"x.pdf\")",
    "save_conflict!err" => "save_path = joinpath(OUT, \"a.png\"), plot_path = joinpath(OUT, \"b.png\")",
    "save_baddir!err" => "save_path = joinpath(OUT, \"no_such_dir\", \"x.png\")",
]
# Style + axis/figure kwargs + save options (the "cosmetic" block every function gets).
_bt_common(; layout = true, sub = true, save = true) = vcat(
    _BT_STYLES,
    sub ? ["kwsub" => _BT_KWSUB] : Pair{String, String}[],
    layout ? ["kwlay" => _BT_KWLAY] : Pair{String, String}[],
    save ? _BT_SAVE : Pair{String, String}[],
)

# x-axis options for functions that require a *varying* covariate (residuals, VPC,
# observation distributions, observed profiles): the time column explicitly, the
# metadata `x_feature` (a varying covariate) or another non-time varying covariate, and a
# constant covariate, which non-ODE models must reject.
function _bt_xvary(M, f)
    o = ["xfeat_time" => "x_axis_feature = NoLimits.get_time_col(dm)"]
    x = _bt_meta(M, f, "x_feature")
    if x !== nothing
        push!(o, "xfeat" => "x_axis_feature = $(_bt_sym(x))")
    elseif _bt_has_tag(M, f, :time_varying_covariate)
        push!(
            o, "xfeat_varying" =>
                "x_axis_feature = first(filter(!=(NoLimits.get_time_col(dm)), NoLimits.get_covariates(NoLimits.get_model(dm)).varying))"
        )
    end
    if !isempty(_bt_list(M, f, "covariates"))
        push!(
            o, (_bt_has_tag(M, f, :ode) ? "xfeat_constcov" : "xfeat_constcov!err") =>
                "x_axis_feature = first(NoLimits.get_covariates(NoLimits.get_model(dm)).constants)"
        )
    end
    push!(o, "xfeat_bad!err" => "x_axis_feature = :no_such_feature")
    return o
end

# Representative fixtures that get every option of a function (continuous NLME with
# covariates + comparison fit; posterior draws).
const _BT_FULL = ("nlme_exp_laplace", "mcmc_re")
# Option names kept on every applicable fixture: defaults and the options whose result
# depends on the data (observable / marginal / RE / metric selection, posterior settings).
const _BT_CORE_RE = r"^(default|obs\d+|obs_all|marg\d+|res_(pit|quantile|raw|pearson|logscore)|inds_12|re_all|levels2|xfeat|xfeat_varying|xfeat_constcov|xcov|mcmc_draws20|panel_vec|panel_multi|vector|tidx2|histogram|transformed|quantile|quantile_perind|nbins2|no_randomize)(!err)?$"
_bt_is_core(name) = occursin(_BT_CORE_RE, replace(name, r"^(dm_|nt_|vec_|dict_|single_|wald_|profile_|chain_)" => ""))

const _BT_ALIASES = Dict("n_simulations" => ["n_sim"])

# Files a successful call must write: every `joinpath(OUT, "...")` literal (".png" added
# when the name has no extension, matching `_ensure_save_path`).
function _bt_files(opts::AbstractString)
    out = String[]
    for m in eachmatch(r"joinpath\(OUT, \"([^\"]+)\"\)", opts)
        f = m.captures[1]
        push!(out, isempty(splitext(f)[2]) ? f * ".png" : f)
    end
    return out
end

"""
    _bt_cases(fn, fixtures, optsets; input = "res", prefix = "", base = f -> "")

Like `cross`, but `optsets` may be a function `fixture -> Vector{Pair}`, `files` are
inferred per optset from `joinpath(OUT, ...)` literals, `prefix` is prepended to every
optset name (to keep ids unique across input variants), `input` may be a function of the
fixture, and `base(fixture)` is a keyword prefix inserted into every non-default call
(e.g. `"mcmc_draws = 50"` on posterior-draw fixtures).
"""
function _bt_cases(fn::Symbol, fixtures, optsets; input = "res", prefix = "", base = f -> "", tags = String[], full = nothing)
    out = Case[]
    fixtures = collect(fixtures)
    for f in fixtures
        sets = optsets isa Function ? optsets(f) : optsets
        # The full option sweep runs on representative fixtures (or when a function has at
        # most two fixtures); the other fixtures get the data-dependent core options only.
        # (when none of the representative fixtures applies, the first fixture is full)
        anyfull = any(in(_BT_FULL), fixtures)
        isfull = full === nothing ?
            (f in _BT_FULL || length(fixtures) <= 2 || (!anyfull && f == first(fixtures))) : full(f)
        isfull || (sets = filter(p -> _bt_is_core(first(p)), sets))
        inp = input isa Function ? input(f) : input
        b = base(f)
        for (name, opts) in sets
            err = endswith(name, "!err")
            files = err ? String[] : _bt_files(opts)
            # drop base keywords the optset sets itself (or an alias of them)
            keep = String[]
            if !(isempty(b) || name == "default")
                for kv in split(b, ", ")
                    k = strip(first(split(kv, "=")))
                    keys_k = vcat([k], get(_BT_ALIASES, k, String[]))
                    any(kk -> occursin(Regex("\\b" * kk * "\\s*="), opts), keys_k) || push!(keep, kv)
                end
            end
            bb = join(keep, ", ")
            callopts = isempty(bb) ? opts : (isempty(opts) ? bb : bb * ", " * opts)
            append!(out, cross(fn, [f], [prefix * name => callopts]; input = inp, tags = tags, files = files))
        end
    end
    return out
end

# ---------------------------------------------------------------------------------------
# combine_plots / return_panel
# ---------------------------------------------------------------------------------------

const _BT_COMBINE_FULL = ("nlme_exp_laplace", "pk_ode_focei", "poisson_count_laplace", "mcmc_re")

function cases_combine()
    M = fixture_meta()
    cs = Case[]
    # Functions with `return_panel` in the OLD implementation: plot_dv_ipred,
    # plot_wres_pred, plot_vpc. (plot_dv_pred has none on origin/main.)
    scalar(f) = !_bt_is_vecobs(M, f)
    ipred = filter(scalar, _bt_for(M, :plot_dv_ipred))
    wres = filter(scalar, _bt_for(M, :plot_wres_pred))
    vpc = filter(scalar, _bt_for(M, :plot_vpc))
    obs1(f) = (o = _bt_list(M, f, "observables"); isempty(o) ? "" : "observables = $(_bt_sym(o[1])), ")
    vpcpanel(f) = "plot_vpc(res; $(obs1(f))n_simulations = 20, return_panel = true)"
    mcmc(f) = _bt_is_posterior(M, f) ? ", mcmc_draws = 20" : ""
    vpcpanelm(f) = "plot_vpc(res; $(obs1(f))n_simulations = 20, return_panel = true$(mcmc(f)))"

    # Bare panels (render.jl wraps them with combine_plots([panel])).
    append!(cs, _bt_cases(:plot_dv_ipred, ipred, ["panel" => "return_panel = true"]))
    append!(cs, _bt_cases(:plot_dv_ipred, ipred, ["panel_kwsub" => "return_panel = true, " * _BT_KWSUB]))
    append!(cs, _bt_cases(:plot_wres_pred, wres, ["panel" => "return_panel = true"]))
    append!(cs, _bt_cases(:plot_wres_pred, wres, ["panel_style" => "return_panel = true, " * _BT_STYLES[1][2]]))
    for f in vpc
        push!(cs, Case("plot_vpc__$(f)__panel", :plot_vpc, f, vpcpanelm(f); tags = ["panel"]))
    end

    # combine_plots compositions
    for f in intersect(ipred, wres)
        two = "[plot_dv_ipred(res; return_panel = true), plot_wres_pred(res; return_panel = true)]"
        opts = [
            "two" => "combine_plots($(two))",
            "two_ncols1" => "combine_plots($(two); ncols = 1)",
            "two_ncols2" => "combine_plots($(two); ncols = 2)",
            "two_ncols5" => "combine_plots($(two); ncols = 5)",
            "two_size" => "combine_plots($(two); ncols = 2, size = (1000, 400))",
            "two_padding" => "combine_plots($(two); ncols = 2, figure_padding = 60, backgroundcolor = :gray90)",
            "two_style_size" => "combine_plots($(two); ncols = 2, style = PlotStyle(base_subplot_width = 200, base_subplot_height = 150))",
            "two_style_fonts" => "combine_plots($(two); style = PlotStyle(font_family = \"TeX Gyre Heros Makie Italic\"))",
            "styled_panels" => "combine_plots([plot_dv_ipred(res; return_panel = true, $(_BT_STYLES[1][2])), plot_wres_pred(res; return_panel = true, $(_BT_STYLES[2][2]))]; ncols = 2)",
            "kwsub_panels" => "combine_plots([plot_dv_ipred(res; return_panel = true, $(_BT_KWSUB)), plot_wres_pred(res; return_panel = true, kwargs_subplot = (; title = \"WRES custom\"))]; ncols = 2)",
            "repeat4" => "combine_plots([plot_dv_ipred(res; return_panel = true) for _ in 1:4]; ncols = 2)",
            "single" => "combine_plots([plot_wres_pred(res; return_panel = true)])",
            "ncols0!err" => "combine_plots($(two); ncols = 0)",
        ]
        if f in vpc
            three = "[plot_dv_ipred(res; return_panel = true), plot_wres_pred(res; return_panel = true), $(vpcpanelm(f))]"
            push!(opts, "three" => "combine_plots($(three))")
            push!(opts, "three_ncols1" => "combine_plots($(three); ncols = 1)")
            push!(opts, "three_ncols2_size" => "combine_plots($(three); ncols = 2, size = (800, 700))")
        end
        # every layout variant on a few representative fixtures, the plain compositions
        # everywhere
        if !(f in _BT_COMBINE_FULL)
            opts = filter(p -> first(p) in ("two", "three"), opts)
        end
        for (name, call) in opts
            err = endswith(name, "!err")
            oname = replace(name, "!err" => "")
            push!(cs, Case("combine_plots__$(f)__$(oname)", :combine_plots, f, call; tags = [oname], expect_error = err))
        end
    end
    # Empty panel vector (degenerate layout).
    if !isempty(ipred)
        f = first(ipred)
        push!(cs, Case("combine_plots__$(f)__empty", :combine_plots, f, "combine_plots([])"; tags = ["empty"]))
    end
    return cs
end
