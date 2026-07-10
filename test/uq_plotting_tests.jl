using Test
using NoLimits
using LinearAlgebra
using CairoMakie
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

@testset "UQ distribution plotting with draws" begin
    rng = Random.Xoshiro(101)
    draws = randn(rng, 300, 2)
    est = vec(mean(draws; dims = 1))
    lower = [quantile(view(draws, :, 1), 0.025), quantile(view(draws, :, 2), 0.025)]
    upper = [quantile(view(draws, :, 1), 0.975), quantile(view(draws, :, 2), 0.975)]
    ints = UQIntervals(0.95, lower, upper)
    V = Matrix(I, 2, 2)

    uq = UQResult(
        :wald,
        :mle,
        [:a, :σ],
        nothing,
        est,
        est,
        ints,
        ints,
        V,
        V,
        draws,
        draws,
        (; vcov = :hessian)
    )

    p_all = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(
        uq; scale = :natural)
    @test p_all !== nothing

    p_one = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(uq;
        scale = :natural,
        parameters = [:σ],
        ncols = 1,
        show_legend = true)
    @test p_one !== nothing

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_uq_distributions.png")
        @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(
            uq; scale = :natural, plot_path = p_path)
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
    uq = UQResult(
        :wald,
        :mle,
        [:a, :σ],
        nothing,
        est,
        est,
        ints,
        ints,
        V,
        V,
        draws,
        draws,
        (; vcov = :hessian)
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
    uq = UQResult(
        :wald,
        :mle,
        [:a, :β_1],
        nothing,
        est,
        est,
        ints,
        ints,
        V,
        V,
        draws,
        draws,
        (; vcov = :hessian)
    )
    p = plot_uq_distributions(uq; scale = :natural, parameters = [:a], ncols = 1)
    @test _first_axis(p).xlabel[] == "a"
    @test _first_axis(p).title[] == "a"
end

@testset "UQ distribution plotting uses analytic Wald density on transformed scale" begin
    ints = UQIntervals(0.95, [-0.6], [0.8])
    uq = UQResult(
        :wald,
        :mle,
        [:a],
        nothing,
        [0.1],
        [0.1],
        ints,
        ints,
        reshape([0.25], 1, 1),
        reshape([0.25], 1, 1),
        nothing,
        nothing,
        (; vcov = :hessian)
    )
    p = plot_uq_distributions(uq; scale = :transformed, ncols = 1)
    @test p !== nothing
    @test _first_axis(p).ylabel[] == "Wald Approximate Density"
end

@testset "UQ distribution plotting uses analytic Wald densities on natural scale for identity/log transforms" begin
    uq = UQResult(
        :wald,
        :mle,
        [:a, :σ],
        nothing,
        [0.2, -1.0],
        [0.2, exp(-1.0)],
        nothing,
        nothing,
        [0.04 0.0; 0.0 0.09],
        nothing,
        nothing,
        nothing,
        (; vcov = :hessian, coordinate_transforms = [:identity, :log])
    )
    p = plot_uq_distributions(uq; scale = :natural, ncols = 1)
    @test p !== nothing
    @test _first_axis(p).ylabel[] == "Wald Approximate Density"
end

@testset "UQ distribution plotting fallback line uses current plot scale" begin
    σ_t = -0.5
    σ_n = exp(σ_t)
    uq = UQResult(
        :wald,
        :mle,
        [:σ],
        nothing,
        [σ_t],
        [σ_n],
        nothing,
        nothing,
        reshape([-1e-6], 1, 1),
        nothing,
        nothing,
        nothing,
        (; vcov = :hessian, coordinate_transforms = [:log])
    )
    p = plot_uq_distributions(uq;
        scale = :natural,
        ncols = 1,
        show_estimate = false,
        show_interval = false,
        show_legend = false)
    @test p !== nothing
    plt = first(_first_axis(p).scene.plots)
    xs = [pt[1] for pt in plt[1][]]
    # Makie stores plot coordinates as Float32, so the old 1e-12 tolerance is too tight.
    @test all(x -> isapprox(x, σ_n; atol = 1e-6), xs)
end

@testset "UQ distribution plotting uses KDE for Wald natural scale and logs fallback" begin
    rng = Random.Xoshiro(121)
    draws_n = randn(rng, 300, 1) .* 0.2 .+ 1.0
    ints = UQIntervals(0.95, [0.7], [1.3])
    uq = UQResult(
        :wald,
        :mle,
        [:a],
        nothing,
        [0.0],
        [1.0],
        ints,
        ints,
        reshape([0.04], 1, 1),
        reshape([0.04], 1, 1),
        nothing,
        draws_n,
        (; vcov = :hessian)
    )
    p = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(
        uq; scale = :natural, ncols = 1)
    @test p !== nothing
    @test _first_axis(p).ylabel[] == "Wald KDE Density"
end

@testset "UQ distribution plotting errors when draws are unavailable" begin
    ints = UQIntervals(0.95, [0.1], [0.4])
    uq_profile = UQResult(
        :profile,
        :mle,
        [:a],
        nothing,
        [0.2],
        [0.2],
        ints,
        ints,
        nothing,
        nothing,
        nothing,
        nothing,
        (;)
    )
    @test_throws ErrorException plot_uq_distributions(uq_profile)
    @test_throws ErrorException plot_uq_distributions(uq_profile; plot_type = :histogram)
end

@testset "UQ distribution plotting xlims include estimate and distribution support" begin
    rng = Random.Xoshiro(451)
    draws = randn(rng, 300, 1)
    est = [10.0]
    ints = UQIntervals(0.95, [-1.5], [1.5])
    V = reshape([1.0], 1, 1)
    uq = UQResult(
        :wald,
        :mle,
        [:a],
        nothing,
        est,
        est,
        ints,
        ints,
        V,
        V,
        draws,
        draws,
        (; vcov = :hessian)
    )

    p_density = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(
        uq; scale = :natural, ncols = 1)
    xl_density = _axis_xlims(p_density)
    @test xl_density[1] <= minimum(draws)
    @test xl_density[2] >= est[1]

    p_hist = plot_uq_distributions(uq; scale = :natural, plot_type = :histogram, ncols = 1)
    xl_hist = _axis_xlims(p_hist)
    @test xl_hist[1] <= minimum(draws)
    @test xl_hist[2] >= est[1]
end
