using Test
using NoLimits
using DataFrames
using Distributions
using CairoMakie
import AlgebraOfGraphics  # the plotting extension needs Makie AND AlgebraOfGraphics
using MCMCChains
using Random
using Turing: MH

# Note: "plot_data and plot_fits basic", "plot_fits MCMC", "plot_fits VI"
# have been moved to integration_plotting.jl (shared fixtures).

# Qualified: ComponentArrays (loaded by sibling test files in the same batch) also
# exports Axis.
_first_axis(fig) = first(a for a in fig.content if a isa CairoMakie.Axis)
function _series_labels(fig)
    plots = _first_axis(fig).scene.plots
    raw = [CairoMakie.Makie.to_value(get(plt.attributes, :label, nothing)) for plt in plots]
    return [l isa AbstractString ? String(l) : "" for l in raw]
end

# AlgebraOfGraphics layout policy (#383): one facet per individual, linked axes with
# hidden inner tick labels when shared, a single figure-level legend, user Axis/Figure
# kwargs win over defaults.
_axes(fig) = [a for a in fig.content if a isa CairoMakie.Axis]
_legends(fig) = [l for l in fig.content if l isa CairoMakie.Legend]
_labels(fig) = [string(CairoMakie.Makie.to_value(l.text)) for l in fig.content if l isa CairoMakie.Label]
_points(fig, T) = sum(
    length(CairoMakie.Makie.to_value(pl[1])) for ax in _axes(fig)
        for pl in ax.scene.plots if pl isa T;
    init = 0
)

@testset "plot_fits/plot_data AoG layout policy" begin
    res = fx_laplace()                      # 6 individuals x 3 observations
    n_ind = 6
    fig = plot_fits(res; ncols = 4)
    @test fig isa CairoMakie.Figure
    axs = _axes(fig)
    @test length(axs) == n_ind
    @test length(_legends(fig)) == 1
    @test all(l -> "ID: $(l)" in _labels(fig), 1:n_ind)       # facet titles
    @test count(==("Time"), _labels(fig)) == 1                 # one spanning x label
    @test _points(fig, CairoMakie.Scatter) == 3 * n_ind        # every observation drawn once
    @test count(pl -> pl isa CairoMakie.Lines, (pl for ax in axs for pl in ax.scene.plots)) == n_ind
    # linked + shared: identical limits, inner y tick labels hidden (col 2..4 of row 1)
    lims = [ax.finallimits[] for ax in axs]
    @test all(==(lims[1]), lims)
    @test count(ax -> !CairoMakie.Makie.to_value(ax.yticklabelsvisible), axs) == n_ind - 2

    free = plot_fits(res; shared_x_axis = false, shared_y_axis = false)
    @test all(ax -> CairoMakie.Makie.to_value(ax.yticklabelsvisible), _axes(free))

    # user kwargs win over the computed defaults
    custom = plot_fits(
        res; kwargs_subplot = (; xlabel = "X custom", xgridvisible = true),
        kwargs_layout = (; size = (900, 600))
    )
    @test "X custom" in _labels(custom)
    @test all(ax -> CairoMakie.Makie.to_value(ax.xgridvisible), _axes(custom))
    @test Tuple(custom.scene.viewport[].widths) == (900, 600)

    # data only: scalar observable -> no legend; subset of individuals
    d = plot_data(res; individuals_idx = [1, 2], ncols = 1)
    @test length(_axes(d)) == 2
    @test isempty(_legends(d))
    @test _points(d, CairoMakie.Scatter) == 6

    # density overlay: one heatmap per panel, drawn behind the data
    dens = plot_fits(res; plot_density = true)
    hms = [pl for ax in _axes(dens) for pl in ax.scene.plots if pl isa CairoMakie.Heatmap]
    @test length(hms) == n_ind

    @test_throws ErrorException plot_fits(res; ncols = 0)
    @test_throws ErrorException plot_data(res; ncols = 0)
end

@testset "plot_fits supports non-:t time_col" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(-0.2)
            σ = RealNumber(0.3, scale = :log)
        end

        @covariates begin
            age = Covariate()
            z = Covariate()
        end

        @formulas begin
            y ~ Normal(a + b * z + 0.1 * age, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        age = [0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2],
        y = [0.15, 0.18, 0.14, 0.19]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :age)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))

    @test plot_fits(res) !== nothing
    @test plot_fits(dm) !== nothing
    @test plot_data(res) !== nothing
    @test plot_data(dm) !== nothing
    @test plot_fits(res, x_axis_feature = :age) !== nothing
end

@testset "plot_data/plot_fits skip missing scalar observations (regression)" begin
    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = Union{Missing, Float64}[0.15, missing, 0.14, missing]
    )

    dm = DataModel(fx_nore_model(), df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))

    @test plot_data(res) !== nothing
    @test plot_data(dm) !== nothing
    @test plot_fits(res) !== nothing
    @test plot_fits(dm) !== nothing
    @test plot_fits_comparison([res, res]) !== nothing
end

# #383: facets follow the order in which individuals first appear in the data. DataModel
# sorts ids, so string ids s1..s12 used to be drawn s1, s10, s11, s12, s2, ...
@testset "plot_data/plot_fits facets in data order (#383)" begin
    ids = ["s$(i)" for i in 1:12]
    df = DataFrame(
        ID = repeat(ids; inner = 2), t = repeat([0.0, 1.0]; outer = 12),
        y = 0.1 .* (1:24)
    )
    dm = DataModel(fx_nore_model(), df; primary_id = :ID, time_col = :t)
    ext = Base.get_extension(NoLimits, :NoLimitsMakieExt)
    order = ext._plot_individuals(dm, nothing)
    @test [string(NoLimits._individual_id(dm, i)) for i in order] == ids
    @test ext._plot_individuals(dm, nothing; default_all = false) == order[1:1]
    # an explicit selection keeps the caller's order
    @test ext._plot_individuals(dm, [3, 1]) == [3, 1]
    @test ext._ordered_levels(["b", "a", "b", "c"]) == ["b", "a", "c"]

    # panel k sits at fldmod1(k, ncols): with ncols = 12 the titles run left to right
    title_x(fig, lab) = only(
        CairoMakie.Makie.to_value(l.layoutobservables.computedbbox).origin[1]
            for l in fig.content if l isa CairoMakie.Label &&
            string(CairoMakie.Makie.to_value(l.text)) == lab
    )
    for fig in (plot_data(dm; ncols = 12), plot_fits(dm; ncols = 12))
        xs = [title_x(fig, "ID: $(id)") for id in ids]
        @test issorted(xs)
    end
end

# #383: the scalar density overlay starts transparent (the zero-density part of the
# time x y grid used to be painted purple) and stays normalised per panel.
@testset "plot_fits density overlay is transparent at zero density" begin
    res = fx_laplace()
    fig = plot_fits(res; plot_density = true)
    hms = [pl for ax in _axes(fig) for pl in ax.scene.plots if pl isa CairoMakie.Heatmap]
    @test !isempty(hms)
    cmap = CairoMakie.Makie.to_colormap(CairoMakie.Makie.to_value(hms[1].colormap))
    @test CairoMakie.Makie.Colors.alpha(first(cmap)) == 0
    @test CairoMakie.Makie.Colors.alpha(last(cmap)) > 0.5
end

# Many panels: the MAX_FIGURE_HEIGHT cap no longer squeezes a long column of panels
# into strips; explicit sizes are untouched.
@testset "default figure size keeps panels readable" begin
    st = NoLimits.PlotStyle()
    w, h = NoLimits.calculate_plot_size(50, 1, st)
    @test h >= 50 * st.base_subplot_height * NoLimits.MIN_PANEL_SCALE
    @test NoLimits.calculate_plot_size(50, 3, st) == (682, 2400)     # unchanged
    @test NoLimits.calculate_plot_size(4, 2, st) == (700, 560)        # unchanged
end

# Core layout helpers shared by all ported plot functions (#383): legend wrapping and
# combine_plots axis linking.
@testset "combine_plots linking and legend wrapping" begin
    ext = Base.get_extension(NoLimits, :NoLimitsMakieExt)
    AoG = AlgebraOfGraphics
    st = NoLimits.PlotStyle()
    @test ext._legend_nbanks(["data", "fit"], 1000, st) == 1
    long = ["component label $(i)" for i in 1:7]
    @test ext._legend_nbanks(long, 400, st) > 1
    @test ext._legend_nbanks(long, 400, st) <= length(long)

    df = DataFrame(x = [0.0, 1.0], y = [1.0, 2.0], series = ["a", "a"])
    mk(axis) = ext.AoGPanel(
        AoG.data(df) * AoG.mapping(:x, :y; color = :series) * AoG.visual(CairoMakie.Lines);
        axis = axis
    )
    vis(ax, a) = CairoMakie.Makie.to_value(getproperty(ax, a))
    fig = combine_plots([mk((;)) for _ in 1:4]; ncols = 2, link_x = true, link_y = true)
    axs = _axes(fig)
    @test length(axs) == 4
    @test [vis(ax, :yticklabelsvisible) for ax in axs] == [true, false, true, false]
    @test [vis(ax, :xticklabelsvisible) for ax in axs] == [false, false, true, true]
    # 3 panels on 2 columns: panel 2 is the lowest of column 2 and keeps its x ticks
    fig3 = combine_plots([mk((;)) for _ in 1:3]; ncols = 2, link_x = true)
    @test [vis(ax, :xticklabelsvisible) for ax in _axes(fig3)] == [false, true, true]
    # unlinked by default; explicit panel attributes win over the decluttering
    @test all(ax -> vis(ax, :yticklabelsvisible), _axes(combine_plots([mk((;)) for _ in 1:4]; ncols = 2)))
    kept = combine_plots([mk((; yticklabelsvisible = true)) for _ in 1:4]; ncols = 2, link_y = true)
    @test all(ax -> vis(ax, :yticklabelsvisible), _axes(kept))

    # a legend that does not fit the figure width wraps into several rows
    dfl = DataFrame(x = repeat([0.0, 1.0]; outer = 7), y = rand(14), series = repeat(long; inner = 2))
    pl = ext.AoGPanel(
        AoG.data(dfl) * AoG.mapping(:x, :y; color = :series) * AoG.visual(CairoMakie.Lines)
    )
    wide = combine_plots([pl]; ncols = 1, size = (400, 300))
    @test CairoMakie.Makie.to_value(only(_legends(wide)).nbanks) > 1
end

@testset "plot_fits discrete poisson" begin
    p_fits = plot_fits(fx_pois_laplace(); plot_density = false)
    @test p_fits !== nothing
end

@testset "plot_fits tolerates a statistic the distribution lacks" begin
    # Outcome distributions without `mean` (CensoredDistributions' interval-censored
    # ones) must plot as a gap instead of erroring, here and in the y-limits.
    @test plot_fits(
        fx_pois_laplace(); plot_func = d -> error("undefined"),
        plot_density = false
    ) !== nothing
end

@testset "plot_data discrete" begin
    p_data = plot_data(fx_pois_dm())
    @test p_data !== nothing
end

@testset "plot_data and plot_fits (ODE)" begin
    res = fx_ode_laplace()

    @test plot_data(res) !== nothing
    @test plot_fits(res) !== nothing
    @test plot_fits(res; plot_density = true) !== nothing
    @test plot_fits(fx_ode_dm()) !== nothing
end

@testset "plot_fits inherits constants_re from fit result" begin
    # fx_constre_laplace is a Laplace fit built with constants_re = (; η = (; B = 0.0)).
    @test plot_fits(fx_constre_laplace()) !== nothing
end

@testset "plot_multistart_waterfall basic" begin
    ms = NoLimits.Multistart(;
        dists = (; a = Normal(0.0, 0.5), b = Normal(0.0, 0.5)),
        n_draws_requested = 4,
        n_draws_used = 3,
        sampling = :lhs
    )

    res_ms = fit_model(ms, fx_nore_dm(), NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test plot_multistart_waterfall(res_ms) !== nothing
    # Ordinary plots used to throw a MethodError on a MultistartFitResult (#295).
    @test plot_fits(res_ms) !== nothing
    @test plot_data(res_ms) !== nothing

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_multistart_waterfall.png")
        plot_multistart_waterfall(res_ms; plot_path = p_path)
        @test isfile(p_path)
    end
end

@testset "plot_multistart_fixed_effect_variability basic" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, calculate_se = true)
            b = RealNumber(-0.2, calculate_se = false)
            v = RealVector([0.05, -0.03], calculate_se = true)
            σ = RealNumber(0.25, scale = :log, calculate_se = true)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            y ~ Normal(a + b * z + v[1] * t + v[2] * z, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.0, 0.1, -0.1, 0.1, 0.05, 0.15],
        y = [0.1, 0.16, 0.06, 0.12, 0.03, 0.1, 0.08, 0.14]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    ms = NoLimits.Multistart(;
        dists = (; a = Normal(0.0, 0.4), b = Normal(0.0, 0.4), v = Normal(0.0, 0.25)),
        n_draws_requested = 4,
        n_draws_used = 3,
        sampling = :lhs
    )

    res_ms = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))

    @test plot_multistart_fixed_effect_variability(res_ms; k_best = 3, mode = :points) !==
        nothing
    @test plot_multistart_fixed_effect_variability(
        res_ms; k_best = 3, mode = :quantiles, quantiles = [0.1, 0.5, 0.9],
        include_parameters = [:b], exclude_parameters = [:a]
    ) !== nothing
    @test plot_multistart_fixed_effect_variability(
        res_ms; k_best = 3, scale = :transformed
    ) !== nothing
    @test_throws ErrorException plot_multistart_fixed_effect_variability(
        res_ms; mode = :invalid
    )
    @test_throws ErrorException plot_multistart_fixed_effect_variability(
        res_ms; include_parameters = [:missing_parameter]
    )

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_multistart_fixed_effect_variability.png")
        plot_multistart_fixed_effect_variability(res_ms; k_best = 3, plot_path = p_path)
        @test isfile(p_path)
    end
end

@testset "plot_fits_comparison basic" begin
    dm = fx_nore_dm()
    res1 = fx_mle()
    res2 = fit_model(
        dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)); constants = (; a = 0.2)
    )

    @test plot_fits_comparison(res1) !== nothing

    p_vec = plot_fits_comparison([res1, res2]; individuals_idx = 1:2)
    @test p_vec !== nothing
    labels_vec = _series_labels(p_vec)
    @test "data" in labels_vec
    @test "Model 1" in labels_vec
    @test "Model 2" in labels_vec

    p_nt = plot_fits_comparison(
        (baseline = res1, constrained = res2); individuals_idx = 1:2
    )
    @test p_nt !== nothing
    labels_nt = _series_labels(p_nt)
    @test "baseline" in labels_nt
    @test "constrained" in labels_nt

    styled = PlotStyle(comparison_line_styles = Dict("baseline" => :dash))
    p_nt_style = plot_fits_comparison(
        (baseline = res1, constrained = res2); individuals_idx = 1:2, style = styled
    )
    @test p_nt_style !== nothing
    idx_base = findfirst(==("baseline"), _series_labels(p_nt_style))
    @test idx_base !== nothing
    baseline_plt = _first_axis(p_nt_style).scene.plots[idx_base]
    # Makie stores linestyles converted (:dash -> dash pattern vector, :solid -> nothing).
    @test CairoMakie.Makie.to_value(baseline_plt.linestyle) ==
        CairoMakie.Makie.convert_attribute(:dash, CairoMakie.Makie.key"linestyle"())

    p_dict = plot_fits_comparison(
        Dict("first" => res1, "second" => res2); individuals_idx = 1:2
    )
    @test p_dict !== nothing
    labels_dict = _series_labels(p_dict)
    @test "first" in labels_dict
    @test "second" in labels_dict

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_fits_comparison.png")
        plot_fits_comparison([res1, res2]; individuals_idx = 1:2, plot_path = p_path)
        @test isfile(p_path)
    end

    df_bad = copy(fx_nore_df())
    df_bad.y .= df_bad.y .+ 1.0
    dm_bad = DataModel(fx_nore_model(), df_bad; primary_id = :ID, time_col = :t)
    res_bad = fit_model(dm_bad, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test_throws ErrorException plot_fits_comparison([res1, res_bad])
end

@testset "plot_data/fits multivariate HMM" begin
    model = @Model begin
        @fixedEffects begin
            # Priors are ignored by MLE and let the same model serve the MCMC fit below.
            mu1 = RealNumber(0.0, prior = Normal(0.0, 1.0))
            mu2 = RealNumber(3.0, prior = Normal(3.0, 1.0))
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            P = [0.9 0.1; 0.2 0.8]
            e1 = (Normal(mu1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(mu2, 1.0), Normal(-1.0, 0.5))
            y ~ MVDiscreteTimeDiscreteStatesHMM(P, (e1, e2), Categorical([0.5, 0.5]))
        end
    end

    df = DataFrame(
        ID = repeat(1:2, inner = 3),
        t = vcat(0.0, 1.0, 2.0, 0.0, 1.0, 2.0),
        y = [
            Union{Missing, Float64}[0.1, 2.0],
            Union{Missing, Float64}[0.2, missing],
            Union{Missing, Float64}[0.4, 1.9],
            Union{Missing, Float64}[3.0, -1.1],
            Union{Missing, Float64}[2.8, missing],
            Union{Missing, Float64}[missing, -1.2],
        ]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(optim_kwargs = (; iterations = 5)))
    n_marginals = 2

    @test plot_data(res; marginal_layout = :single) !== nothing
    p_data_vector = plot_data(res; marginal_layout = :vector)
    @test isa(p_data_vector, Vector{CairoMakie.Figure})
    @test length(p_data_vector) == n_marginals

    p_fits_single = plot_fits(res; marginal_layout = :single)
    @test length(_legends(p_fits_single)) == 1
    @test length(_axes(p_fits_single)) == 2
    @test _points(p_fits_single, CairoMakie.Scatter) == 9   # 5 + 4 observed components
    p_fits_vector = plot_fits(res; marginal_layout = :vector)
    @test isa(p_fits_vector, Vector{CairoMakie.Figure})
    @test length(p_fits_vector) == n_marginals

    n_inds = length(unique(df.ID))
    @test plot_hidden_states(res) !== nothing
    @test plot_hidden_states(dm) !== nothing

    p_hidden_vector = plot_hidden_states(res; figure_layout = :vector)
    @test isa(p_hidden_vector, Vector{CairoMakie.Figure})
    @test length(p_hidden_vector) == n_inds
    @test length(plot_hidden_states(res; figure_layout = :vector, individuals_idx = 1)) == 1
    @test isa(plot_hidden_states(dm; figure_layout = :vector), Vector{CairoMakie.Figure})

    @test plot_emission_distributions(res, time_idx = 1, ncols = 1) !== nothing
    @test plot_emission_distributions(res; time_idx = 2) !== nothing
    @test plot_emission_distributions(res; time_point = 1.0) !== nothing
    p_emission_vector = plot_emission_distributions(res; figure_layout = :vector)
    @test isa(p_emission_vector, Vector{CairoMakie.Figure})
    @test length(p_emission_vector) == n_inds
    @test plot_emission_distributions(dm) !== nothing
    @test isa(
        plot_emission_distributions(dm; figure_layout = :vector), Vector{CairoMakie.Figure}
    )

    # #240 group 4: the remaining diagnostics must show component panels with real data
    # instead of erroring or returning empty figures. Component 1 has 5 observed values,
    # component 2 has 4 (partially missing vectors keep their observed component).
    @testset "multivariate diagnostics render component panels" begin
        all_axes(fig) = [a for a in fig.content if a isa CairoMakie.Axis]
        function scatter_counts(fig)
            return [
                sum(
                        length(CairoMakie.Makie.to_value(plt[1]))
                        for plt in ax.scene.plots if plt isa CairoMakie.Makie.Scatter;
                        init = 0
                    ) for ax in all_axes(fig)
            ]
        end

        # Failure A: one marginal PDF panel per component and observation row.
        f_od = plot_observation_distributions(res; individuals_idx = 1)
        @test length(all_axes(f_od)) == 3 * n_marginals
        pdf_lines = [
            CairoMakie.Makie.to_value(plt[1])
                for plt in first(all_axes(f_od)).scene.plots
                if plt isa CairoMakie.Makie.Lines
        ]
        @test !isempty(pdf_lines)
        @test length(first(pdf_lines)) == 200

        # Failure B: VPC keeps observed values and never assigns a vector draw to
        # scalar storage.
        f_vpc = plot_vpc(res; n_simulations = 2, n_bins = 2)
        @test length(all_axes(f_vpc)) == n_marginals
        @test scatter_counts(f_vpc) == [5, 4]

        # Failure C: nested observation vectors expand into component series.
        @test scatter_counts(plot_observed_profiles(dm)) == [9]
        @test scatter_counts(plot_observed_profiles(res; marginal_idx = 2)) == [4]
        @test "y[1]" in _series_labels(plot_observed_profiles(dm))

        # Failure D: DV/PRED/IPRED/WRES keep vector rows instead of emptying out.
        for f_gof in (plot_dv_pred(res), plot_dv_ipred(res), plot_wres_pred(res))
            @test length(all_axes(f_gof)) == n_marginals
            @test scatter_counts(f_gof) == [5, 4]
        end

        # Residual panels follow the component-wise residual schema when it provides
        # vector cells; the plots are a no-op on a scalar table.
        rdf = NoLimits.get_residuals(res)
        if any(v -> v isa AbstractVector, rdf.res_raw)
            @test length(all_axes(plot_residuals(res; residual = :raw))) ==
                n_inds * n_marginals
            @test length(all_axes(plot_residual_distribution(res; residual = :raw))) ==
                n_marginals
        end
    end

    # #245: one density band per component marginal instead of an early failure.
    @testset "multivariate marginal density bands" begin
        heatmaps(fig) = [
            plt for ax in fig.content if ax isa CairoMakie.Axis
                for plt in ax.scene.plots if plt isa CairoMakie.Makie.Heatmap
        ]
        f_all = plot_fits(res; plot_density = true, individuals_idx = 1)
        @test length(heatmaps(f_all)) == n_marginals
        f_one = plot_fits(
            res; plot_density = true, individuals_idx = 1, marginal_idx = 2
        )
        @test length(heatmaps(f_one)) == 1

        # Posterior draws over a vector-valued observable used to be rejected outright.
        res_mcmc = fit_model(
            dm,
            NoLimits.MCMC(;
                sampler = MH(),
                turing_kwargs = (n_samples = 4, n_adapt = 2, progress = false)
            )
        )
        f_mcmc = plot_fits(
            res_mcmc; mcmc_draws = 2, individuals_idx = 1, plot_density = true
        )
        @test length(heatmaps(f_mcmc)) == n_marginals
    end
end

# #294: plot_hidden_states used to reject any non-multivariate observable, so a plain
# (univariate) HMM outcome could never be plotted even though its filter supports it.
@testset "plot_hidden_states univariate HMM (#294)" begin
    model = @Model begin
        @fixedEffects begin
            mu1 = RealNumber(0.0)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            P = [0.9 0.1; 0.2 0.8]
            y ~ DiscreteTimeDiscreteStatesHMM(
                P, (Normal(mu1, 1.0), Normal(3.0, 1.0)), Categorical([0.5, 0.5])
            )
        end
    end

    df = DataFrame(
        ID = repeat(1:2, inner = 3),
        t = repeat([0.0, 1.0, 2.0], 2),
        y = Union{Missing, Float64}[0.1, missing, 3.1, 3.0, 2.8, 0.2],
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    fig = plot_hidden_states(dm)
    @test fig !== nothing

    # Hand-rolled forward filter for individual 1: propagate, then weight by the
    # emission pdf; a missing observation only propagates.
    P = [0.9 0.1; 0.2 0.8]
    em = (Normal(0.0, 1.0), Normal(3.0, 1.0))
    prior = [0.5, 0.5]
    expected = Vector{Vector{Float64}}()
    for y in df.y[1:3]
        p = transpose(P) * prior
        p ./= sum(p)
        if ismissing(y)
            prior = p
        else
            u = p .* [pdf(em[1], y), pdf(em[2], y)]
            prior = u ./ sum(u)
            push!(expected, copy(prior))
        end
    end

    ax = _first_axis(fig)
    polys = [p for p in ax.scene.plots if p isa CairoMakie.Makie.Poly]
    @test length(polys) == 2
    heights = [
        [r.widths[2] for r in CairoMakie.Makie.to_value(p[1])] for p in polys
    ]
    for state in 1:2
        @test isapprox(
            heights[state], [e[state] for e in expected]; atol = 1.0e-6
        )
    end

    res = fit_model(dm, NoLimits.MLE(optim_kwargs = (; iterations = 2)))
    @test plot_hidden_states(res) !== nothing
    @test isa(
        plot_hidden_states(res; figure_layout = :vector), Vector{CairoMakie.Figure}
    )
end

@testset "plot_fits supports varying non-ODE random-effect groups" begin
    @test plot_fits(
        fx_varyre_dm(); constants_re = fx_varyre_constants_re(),
        plot_density = true
    ) !== nothing
end

@testset "EM trajectory plots (MCEM with diagnostics)" begin
    # Moved from coverage_gap_tests.jl (path coverage for plot_em_trajectories).
    res = fit_model(
        fx_fixre_dm(),
        NoLimits.MCEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 10, n_adapt = 3, progress = false),
            maxiters = 3,
            store_diagnostics = true
        )
    )
    p = plot_em_trajectories(res)
    @test p !== nothing
    # transformed-scale variant exercises the no-DataModel branch
    @test plot_em_trajectories(res; scale = :transformed) !== nothing
end

@testset "VI posterior-draw prediction plots (no RE)" begin
    # VI rejects random-effects models; fx_vi is a fixed-effects-only VI fit that
    # exercises the VI posterior-draw plot path (_vi_drawn_params).
    res = fx_vi()
    @test plot_fits(res) !== nothing
    # posterior-draw band -> _vi_drawn_params
    @test plot_fits(res; plot_mcmc_quantiles = true, mcmc_draws = 5) !== nothing
end

@testset "plots label individuals with no observation rows (#250.4)" begin
    df_ev = copy(fx_re_df())
    n = nrow(df_ev)
    df_ev.EVID = [id == 6 ? 1 : 0 for id in df_ev.ID]
    df_ev.AMT = zeros(n)
    df_ev.RATE = zeros(n)
    df_ev.CMT = ones(Int, n)
    dm_ev = DataModel(
        fx_re_model(), df_ev; primary_id = :ID, time_col = :t, evid_col = :EVID
    )
    @test isempty(NoLimits.get_obs_rows(NoLimits.get_row_groups(dm_ev))[6])
    @test plot_data(dm_ev) !== nothing
    @test plot_fits(fx_laplace(); dm = dm_ev) !== nothing
end

# Issue #250 finding 3: an all-warmup chain has no posterior draw to summarize.
@testset "_mcmc_draw_indices rejects all-warmup chains" begin
    chain = MCMCChains.Chains(randn(4, 2, 1), [:a, :b])
    rng = Random.default_rng()
    @test NoLimits._mcmc_draw_indices(chain, 2, 10, rng) == [3, 4]
    @test_throws ArgumentError NoLimits._mcmc_draw_indices(chain, 4, 10, rng)
    @test_throws ArgumentError NoLimits._mcmc_draw_indices(chain, 9, 10, rng)
end

# Issue #250 finding 6: a missing posterior coordinate must not silently become the
# model's initial value; only explicitly pinned constants may bypass the lookup.
@testset "_coordwise_fixed_from_means rejects missing coordinates" begin
    dm = fx_nore_dm()
    fe = NoLimits.get_fixed(NoLimits.get_model(dm))
    names = NoLimits.get_names(fe)
    θ0 = NoLimits.get_θ0_untransformed(fe)
    @test_throws ArgumentError NoLimits._coordwise_fixed_from_means(
        dm, "MCMC chain", _ -> nothing
    )
    ov = NamedTuple{Tuple(names)}(Tuple(getproperty(θ0, n) for n in names))
    θ = NoLimits._coordwise_fixed_from_means(
        dm, "MCMC chain", _ -> nothing; overrides = ov
    )
    @test all(getproperty(θ, n) == getproperty(θ0, n) for n in names)
end
