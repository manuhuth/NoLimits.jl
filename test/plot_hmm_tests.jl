using Test
using NoLimits
using DataFrames
using Distributions
using CairoMakie
using Random
import AlgebraOfGraphics  # the plotting extension needs Makie AND AlgebraOfGraphics

# HMM plots on AlgebraOfGraphics (#383): plot_hidden_states (stacked bars, one facet per
# individual) and plot_emission_distributions (individual x state grid).

const _HM = CairoMakie.Makie
_hm_axes(fig) = [a for a in fig.content if a isa CairoMakie.Axis]
_hm_legends(fig) = [l for l in fig.content if l isa CairoMakie.Legend]
_hm_labels(fig) = [string(_HM.to_value(l.text)) for l in fig.content if l isa CairoMakie.Label]
_hm_plots(ax, T) = [p for p in ax.scene.plots if p isa T]
_hm_legend_labels(leg) = [
    string(_HM.to_value(e.label)) for g in _HM.to_value(leg.entrygroups) for e in last(g)
]
# (x, top, bottom, width) of every bar of a BarPlot
function _hm_bars(p)
    pts = _HM.to_value(p[1])
    n = length(pts)
    fill_to = _HM.to_value(p.fillto)
    w = _HM.to_value(p.width)
    bottoms = fill_to isa AbstractVector ? Float64.(fill_to) : fill(Float64(fill_to), n)
    widths = w isa AbstractVector ? Float64.(w) : fill(Float64(w), n)
    return [(Float64(pts[k][1]), Float64(pts[k][2]), bottoms[k], widths[k]) for k in 1:n]
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
    @test fig isa CairoMakie.Figure

    # Hand-rolled forward filter for individual 1: propagate, then weight by the
    # emission pdf; a missing observation only propagates and draws no bar.
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

    ax = first(_hm_axes(fig))
    bars = _hm_plots(ax, _HM.BarPlot)
    @test !isempty(bars)
    all_bars = reduce(vcat, [_hm_bars(b) for b in bars])
    @test sort(unique(first.(all_bars))) == [0.0, 2.0]      # the missing t = 1 has no bar
    for (k, x) in enumerate([0.0, 2.0])
        at_x = sort([b for b in all_bars if b[1] == x]; by = b -> b[3])
        @test length(at_x) == 2
        # stacked from 0 to 1 with State 1 at the bottom
        @test at_x[1][3] ≈ 0.0
        @test at_x[1][2] - at_x[1][3] ≈ expected[k][1] atol = 1.0e-6
        @test at_x[2][2] - at_x[2][3] ≈ expected[k][2] atol = 1.0e-6
        @test at_x[2][2] ≈ 1.0
    end

    res = fit_model(dm, NoLimits.MLE(optim_kwargs = (; iterations = 2)))
    @test plot_hidden_states(res) isa CairoMakie.Figure
    @test plot_hidden_states(res; figure_layout = :vector) isa Vector{CairoMakie.Figure}
end

@testset "HMM plots on AlgebraOfGraphics" begin
    model = @Model begin
        @fixedEffects begin
            mu1 = RealNumber(0.0)
            mu2 = RealNumber(3.0)
            lam = RealNumber(4.0; scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            P = [0.9 0.1; 0.2 0.8]
            e1 = (Normal(mu1, 1.0), Poisson(1.0))
            e2 = (Normal(mu2, 1.0), Poisson(lam))
            y ~ MVDiscreteTimeDiscreteStatesHMM(P, (e1, e2), Categorical([0.5, 0.5]))
            c ~ Normal(mu1, 1.0)
        end
    end
    # unequal lengths and spacings: individual "b" (spacing 0.5) gets narrower bars
    df = DataFrame(
        ID = vcat(fill("b", 4), fill("a", 3)),
        t = vcat([0.0, 0.5, 1.0, 1.5], [0.0, 2.0, 4.0]),
        y = [
            Union{Missing, Float64}[0.1, 1.0], Union{Missing, Float64}[0.2, missing],
            Union{Missing, Float64}[3.1, 5.0], Union{Missing, Float64}[2.9, 3.0],
            Union{Missing, Float64}[3.0, 4.0], Union{Missing, Float64}[missing, 0.0],
            Union{Missing, Float64}[0.3, 1.0],
        ],
        c = [0.1, 0.2, 0.0, -0.1, 0.3, 0.2, 0.0],
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    @testset "plot_hidden_states layout" begin
        fig = @test_logs (:warn,) match_mode = :any plot_hidden_states(dm)
        axs = _hm_axes(fig)
        @test length(axs) == 2
        @test length(_hm_legends(fig)) == 1
        @test _hm_legend_labels(only(_hm_legends(fig))) == ["State 1", "State 2"]
        labs = _hm_labels(fig)
        # facets in data order: "b" appears first in the data, although the DataModel
        # sorts it after "a" (#383)
        @test findfirst(==("ID: b"), labs) < findfirst(==("ID: a"), labs)
        # an explicit individuals_idx keeps the caller's order
        labs_sel = _hm_labels(@test_logs (:warn,) match_mode = :any plot_hidden_states(dm; individuals_idx = [1, 2]))
        order = [string("ID: ", NoLimits._individual_id(dm, i)) for i in 1:2]
        @test findfirst(==(order[1]), labs_sel) < findfirst(==(order[2]), labs_sel)
        @test count(==("Time"), labs) == 1
        # per-individual bar width: 0.8 * smallest spacing, at most 1
        widths(ax) = unique(round.([b[4] for p in _hm_plots(ax, _HM.BarPlot) for b in _hm_bars(p)]; digits = 8))
        @test all(ax -> length(widths(ax)) == 1, axs)
        @test sort(reduce(vcat, widths.(axs))) == [0.4, 1.0]
        # shared y is [0, 1]; shared x is the padded union of the bars
        @test all(ax -> ax.finallimits[].origin[2] ≈ 0.0 && ax.finallimits[].widths[2] ≈ 1.0, axs)
        @test axs[1].finallimits[] == axs[2].finallimits[]

        free = plot_hidden_states(dm; observable = :y, shared_x_axis = false)
        @test length(unique([ax.finallimits[].widths[1] for ax in _hm_axes(free)])) == 2

        custom = plot_hidden_states(
            dm; observable = :y, ncols = 1,
            kwargs_subplot = (; xlabel = "X custom"), kwargs_layout = (; size = (500, 700))
        )
        @test "X custom" in _hm_labels(custom)
        @test Tuple(custom.scene.viewport[].widths) == (500, 700)

        vec = plot_hidden_states(dm; observable = :y, figure_layout = :vector)
        @test vec isa Vector{CairoMakie.Figure}
        @test length(vec) == 2
        @test all(f -> length(_hm_axes(f)) == 1 && length(_hm_legends(f)) == 1, vec)
        @test length(plot_hidden_states(dm; observable = :y, figure_layout = :vector, individuals_idx = "a")) == 1
    end

    @testset "plot_emission_distributions grid" begin
        fig = plot_emission_distributions(dm; observable = :y)
        axs = _hm_axes(fig)
        @test length(axs) == 4                                  # 2 individuals x 2 states
        @test length(_hm_legends(fig)) == 1
        @test _hm_legend_labels(only(_hm_legends(fig))) == ["y[1]", "y[2]"]
        labs = _hm_labels(fig)
        # state labels once per column and one short label per individual row: the old
        # per-panel titles ("ID: 1 • t = 0.0 • State 1") overlapped
        @test count(==("State 1"), labs) == 1
        @test count(==("State 2"), labs) == 1
        @test "ID: b\nt = 0.0" in labs
        @test "ID: a\nt = 0.0" in labs
        @test all(ax -> isempty(string(_HM.to_value(ax.title))), axs)
        # continuous component drawn as a pdf line, the Poisson component as pmf points
        @test all(ax -> length(_hm_plots(ax, _HM.Lines)) == 1, axs)
        @test all(ax -> length(_hm_plots(ax, _HM.Scatter)) == 1, axs)
        @test "Density / probability" in labs ||
            all(ax -> _HM.to_value(ax.ylabel) == "Density / probability", axs)
        # shared y: one range; free x per panel
        ylims_(ax) = (ax.finallimits[].origin[2], ax.finallimits[].widths[2])
        @test all(ax -> ylims_(ax) == ylims_(axs[1]), axs)
        free = _hm_axes(plot_emission_distributions(dm; observable = :y, shared_y_axis = false))
        @test length(unique(ylims_.(free))) > 1

        tp = plot_emission_distributions(dm; observable = :y, time_point = 1.9)
        @test "ID: a\nt = 2.0" in _hm_labels(tp)
        @test "ID: b\nt = 1.5" in _hm_labels(tp)

        vec = plot_emission_distributions(
            dm; observable = :y, figure_layout = :vector, kwargs_layout = (; size = (600, 300))
        )
        @test length(vec) == 2
        @test all(f -> length(_hm_axes(f)) == 2 && length(_hm_legends(f)) == 1, vec)
        @test all(f -> Tuple(f.scene.viewport[].widths) == (600, 300), vec)
        # shared y across the vector figures as well
        @test ylims_(_hm_axes(vec[1])[1]) == ylims_(_hm_axes(vec[2])[1])
    end

    @testset "HMM plot argument errors" begin
        # unknown observable used to surface as a raw FieldError / ErrorException
        @test_throws ArgumentError plot_hidden_states(dm; observable = :nope)
        @test_throws ArgumentError plot_emission_distributions(dm; observable = :nope)
        err = try
            plot_hidden_states(dm; observable = :nope)
        catch e
            e
        end
        @test occursin("not found", err.msg) && occursin(":y", err.msg)
        # a non-HMM outcome
        @test_throws ArgumentError plot_hidden_states(dm; observable = :c)
        @test_throws ArgumentError plot_emission_distributions(dm; observable = :c)
        @test_throws ArgumentError plot_hidden_states(dm; observable = 1)
        @test_throws ErrorException plot_hidden_states(dm; observable = :y, ncols = 0)
        # the individual x state grid has no `ncols` keyword (removed, #383)
        @test_throws MethodError plot_emission_distributions(dm; observable = :y, ncols = 2)
        # DataModel methods draw nothing at random and take no `rng`
        @test_throws MethodError plot_hidden_states(dm; observable = :y, rng = Random.Xoshiro(1))
        @test_throws ErrorException plot_emission_distributions(dm; observable = :y, time_idx = 1, time_point = 0.0)
        @test_throws ErrorException plot_hidden_states(dm; observable = :y, figure_layout = :grid)
    end
end
