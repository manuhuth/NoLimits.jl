using Test
using NoLimits
using LinearAlgebra
using CairoMakie
import AlgebraOfGraphics  # the plotting extension needs Makie AND AlgebraOfGraphics
using Random
using Statistics

# Qualified: ComponentArrays (loaded by sibling test files in the same batch) also
# exports Axis.
_first_axis(fig) = first(a for a in fig.content if a isa CairoMakie.Axis)
function _axis_xlims(fig)
    ax = _first_axis(fig)
    lims = CairoMakie.Makie.to_value(ax.limits)[1]
    lims !== nothing && return lims
    CairoMakie.Makie.update_state_before_display!(fig)
    r = ax.finallimits[]
    return (r.origin[1], r.origin[1] + r.widths[1])
end

# Hand-built UQResults below differ in only a few slots; override just those.
# Transformed-scale values default onto the natural-scale slots (and vice versa
# for the paired kwargs), matching the symmetric constructions they replace.
function make_uq(;
        backend = :wald, source = :mle, names = [:a], names_natural = nothing,
        est_t = [0.0], est_n = est_t, ints_t = nothing, ints_n = ints_t,
        vcov_t = nothing, vcov_n = vcov_t, draws_t = nothing, draws_n = draws_t,
        diagnostics = (; vcov = :hessian)
    )
    return UQResult(
        backend, source, names, names_natural, est_t, est_n, ints_t, ints_n,
        vcov_t, vcov_n, draws_t, draws_n, diagnostics
    )
end

@testset "UQ distribution plotting with draws" begin
    rng = Random.Xoshiro(101)
    draws = randn(rng, 300, 2)
    est = vec(mean(draws; dims = 1))
    lower = [quantile(view(draws, :, 1), 0.025), quantile(view(draws, :, 2), 0.025)]
    upper = [quantile(view(draws, :, 1), 0.975), quantile(view(draws, :, 2), 0.975)]
    ints = UQIntervals(0.95, lower, upper)
    V = Matrix(I, 2, 2)

    uq = make_uq(;
        names = [:a, :σ], est_t = est, ints_t = ints, vcov_t = V,
        draws_t = draws
    )

    p_all = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(
        uq; scale = :natural
    )
    @test p_all !== nothing

    p_one = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(
        uq;
        scale = :natural,
        parameters = [:σ],
        ncols = 1,
        show_legend = true
    )
    @test p_one !== nothing

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_uq_distributions.png")
        @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(
            uq; scale = :natural, plot_path = p_path
        )
        @test isfile(p_path)
    end

    @test_throws ErrorException plot_uq_distributions(uq; parameters = [:does_not_exist])
end

@testset "UQ distribution plotting supports histogram mode" begin
    rng = Random.Xoshiro(112)
    draws = randn(rng, 250, 2)
    est = vec(mean(draws; dims = 1))
    ints = UQIntervals(0.95, [-1.2, -1.0], [1.2, 1.0])
    V = Matrix(I, 2, 2)
    uq = make_uq(;
        names = [:a, :σ], est_t = est, ints_t = ints, vcov_t = V,
        draws_t = draws
    )
    p_hist = plot_uq_distributions(uq; plot_type = :histogram, scale = :natural, ncols = 1)
    @test p_hist !== nothing
    @test _first_axis(p_hist).ylabel[] == "Wald Histogram Density"
    @test_throws ErrorException plot_uq_distributions(uq; plot_type = :unknown)
end

@testset "UQ distribution plotting keeps exact parameter labels" begin
    rng = Random.Xoshiro(111)
    draws = randn(rng, 250, 2)
    est = vec(mean(draws; dims = 1))
    ints = UQIntervals(0.95, [-1.5, -1.2], [1.5, 1.2])
    V = Matrix(I, 2, 2)
    uq = make_uq(;
        names = [:a, :β_1], est_t = est, ints_t = ints, vcov_t = V,
        draws_t = draws
    )
    p = plot_uq_distributions(uq; scale = :natural, parameters = [:a], ncols = 1)
    @test _first_axis(p).xlabel[] == "a"
    @test _first_axis(p).title[] == "a"
end

@testset "UQ distribution plotting uses analytic Wald density on transformed scale" begin
    ints = UQIntervals(0.95, [-0.6], [0.8])
    uq = make_uq(; est_t = [0.1], ints_t = ints, vcov_t = reshape([0.25], 1, 1))
    p = plot_uq_distributions(uq; scale = :transformed, ncols = 1)
    @test p !== nothing
    @test _first_axis(p).ylabel[] == "Wald Approximate Density"
end

@testset "UQ distribution plotting uses analytic Wald densities on natural scale for identity/log transforms" begin
    uq = make_uq(;
        names = [:a, :σ], est_t = [0.2, -1.0], est_n = [0.2, exp(-1.0)],
        vcov_t = [0.04 0.0; 0.0 0.09], vcov_n = nothing,
        diagnostics = (; vcov = :hessian, coordinate_transforms = [:identity, :log])
    )
    p = plot_uq_distributions(uq; scale = :natural, ncols = 1)
    @test p !== nothing
    @test _first_axis(p).ylabel[] == "Wald Approximate Density"
end

@testset "UQ distribution plotting fallback line uses current plot scale" begin
    σ_t = -0.5
    σ_n = exp(σ_t)
    uq = make_uq(;
        names = [:σ], est_t = [σ_t], est_n = [σ_n],
        vcov_t = reshape([-1.0e-6], 1, 1), vcov_n = nothing,
        diagnostics = (; vcov = :hessian, coordinate_transforms = [:log])
    )
    p = plot_uq_distributions(
        uq;
        scale = :natural,
        ncols = 1,
        show_estimate = false,
        show_interval = false,
        show_legend = false
    )
    @test p !== nothing
    plt = first(_first_axis(p).scene.plots)
    xs = [pt[1] for pt in plt[1][]]
    # Makie stores plot coordinates as Float32, so the old 1e-12 tolerance is too tight.
    @test all(x -> isapprox(x, σ_n; atol = 1.0e-6), xs)
end

@testset "UQ distribution plotting uses KDE for Wald natural scale and logs fallback" begin
    rng = Random.Xoshiro(121)
    draws_n = randn(rng, 300, 1) .* 0.2 .+ 1.0
    ints = UQIntervals(0.95, [0.7], [1.3])
    uq = make_uq(;
        est_t = [0.0], est_n = [1.0], ints_t = ints,
        vcov_t = reshape([0.04], 1, 1), draws_n = draws_n
    )
    p = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(
        uq; scale = :natural, ncols = 1
    )
    @test p !== nothing
    @test _first_axis(p).ylabel[] == "Wald KDE Density"
end

@testset "UQ distribution plotting of profile intervals (no draws)" begin
    # Regression: profile UQ used to be unplottable ("does not store parameter draws").
    # It now shows the profile interval as a span and the estimate as a line.
    ints = UQIntervals(0.95, [0.1, -1.0], [0.4, NaN])
    uq_profile = make_uq(;
        backend = :profile, names = [:a, :b], est_t = [0.2, -0.5], ints_t = ints,
        diagnostics = (;)
    )
    fig = plot_uq_distributions(uq_profile; show_legend = true)
    axs = [a for a in fig.content if a isa CairoMakie.Axis]
    @test length(axs) == 2
    spans = [pl for pl in axs[1].scene.plots if pl isa CairoMakie.Makie.VSpan]
    @test length(spans) == 1
    @test isempty([pl for pl in axs[2].scene.plots if pl isa CairoMakie.Makie.VSpan])  # open interval
    @test all(ax -> !CairoMakie.Makie.to_value(ax.yticklabelsvisible), axs)
    xl = _axis_xlims(fig)
    @test xl[1] < 0.1 && xl[2] > 0.4
    leg = only(a for a in fig.content if a isa CairoMakie.Legend)
    labs = [string(CairoMakie.Makie.to_value(e.label)) for g in leg.entrygroups[] for e in last(g)]
    @test labs == ["95% Interval", "Estimate"]
    @test_throws ErrorException plot_uq_distributions(uq_profile; plot_type = :histogram)
    @test_throws ErrorException plot_uq_distributions(
        uq_profile; show_interval = false, show_estimate = false
    )
end

@testset "UQ distribution plotting AoG layout: one legend only when requested" begin
    rng = Random.Xoshiro(7)
    draws = randn(rng, 200, 2)
    ints = UQIntervals(0.9, [-1.5, -1.5], [1.5, 1.5])
    uq = make_uq(; names = [:a, :b], est_t = [0.0, 0.1], ints_t = ints, draws_t = draws)
    legends(f) = [l for l in f.content if l isa CairoMakie.Legend]
    f0 = @test_logs (:info, r"KDE") plot_uq_distributions(uq; scale = :natural)
    @test isempty(legends(f0))
    f1 = @test_logs (:info, r"KDE") plot_uq_distributions(uq; scale = :natural, show_legend = true)
    leg = only(legends(f1))
    labs = [string(CairoMakie.Makie.to_value(e.label)) for g in leg.entrygroups[] for e in last(g)]
    @test labs == ["90% Interval", "KDE", "Estimate"]
    axs = [a for a in f1.content if a isa CairoMakie.Axis]
    @test length(axs) == 2
    @test all(ax -> any(pl -> pl isa CairoMakie.Makie.Band, ax.scene.plots), axs)

    # histogram: exact uniform bins as BarPlot, interval as VSpan
    fh = plot_uq_distributions(uq; scale = :natural, plot_type = :histogram, bins = 7)
    ax = first(a for a in fh.content if a isa CairoMakie.Axis)
    bars = only(pl for pl in ax.scene.plots if pl isa CairoMakie.BarPlot)
    @test length(CairoMakie.Makie.to_value(bars[1])) == 7
    @test any(pl -> pl isa CairoMakie.Makie.VSpan, ax.scene.plots)
    @test_throws ErrorException plot_uq_distributions(uq; ncols = 0)
end

@testset "UQ distribution plotting xlims include estimate and distribution support" begin
    rng = Random.Xoshiro(451)
    draws = randn(rng, 300, 1)
    est = [10.0]
    ints = UQIntervals(0.95, [-1.5], [1.5])
    V = reshape([1.0], 1, 1)
    uq = make_uq(; est_t = est, ints_t = ints, vcov_t = V, draws_t = draws)

    p_density = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(
        uq; scale = :natural, ncols = 1
    )
    xl_density = _axis_xlims(p_density)
    @test xl_density[1] <= minimum(draws)
    @test xl_density[2] >= est[1]

    p_hist = plot_uq_distributions(uq; scale = :natural, plot_type = :histogram, ncols = 1)
    xl_hist = _axis_xlims(p_hist)
    @test xl_hist[1] <= minimum(draws)
    @test xl_hist[2] >= est[1]
end
