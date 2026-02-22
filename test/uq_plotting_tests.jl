using Test
using NoLimits
using LinearAlgebra
using Plots
using Random
using Statistics

@testset "UQ distribution plotting with draws" begin
    rng = Random.Xoshiro(101)
    draws = randn(rng, 300, 2)
    est = vec(mean(draws; dims=1))
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
        (; vcov=:hessian)
    )

    p_all = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(uq; scale=:natural)
    @test p_all !== nothing

    p_one = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(uq;
                                                                          scale=:natural,
                                                                          parameters=[:σ],
                                                                          ncols=1,
                                                                          show_legend=true)
    @test p_one !== nothing

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_uq_distributions.png")
        @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(uq; scale=:natural, plot_path=p_path)
        @test isfile(p_path)
    end

    @test_throws ErrorException plot_uq_distributions(uq; parameters=[:does_not_exist])
end

@testset "UQ distribution plotting supports histogram mode" begin
    rng = Random.Xoshiro(112)
    draws = randn(rng, 250, 2)
    est = vec(mean(draws; dims=1))
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
        (; vcov=:hessian)
    )
    p_hist = plot_uq_distributions(uq; plot_type=:histogram, scale=:natural, ncols=1)
    @test p_hist !== nothing
    @test p_hist.subplots[1][:yaxis][:guide] == "Wald Histogram Density"
    @test_throws ErrorException plot_uq_distributions(uq; plot_type=:unknown)
end

@testset "UQ distribution plotting keeps exact parameter labels" begin
    rng = Random.Xoshiro(111)
    draws = randn(rng, 250, 2)
    est = vec(mean(draws; dims=1))
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
        (; vcov=:hessian)
    )
    p = plot_uq_distributions(uq; scale=:natural, parameters=[:a], ncols=1)
    @test p.subplots[1][:xaxis][:guide] == "a"
    @test p.subplots[1][:title] == "a"
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
        (; vcov=:hessian)
    )
    p = plot_uq_distributions(uq; scale=:transformed, ncols=1)
    @test p !== nothing
    @test p.subplots[1][:yaxis][:guide] == "Wald Approximate Density"
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
        (; vcov=:hessian, coordinate_transforms=[:identity, :log])
    )
    p = plot_uq_distributions(uq; scale=:natural, ncols=1)
    @test p !== nothing
    @test p.subplots[1][:yaxis][:guide] == "Wald Approximate Density"
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
        (; vcov=:hessian, coordinate_transforms=[:log])
    )
    p = plot_uq_distributions(uq;
                              scale=:natural,
                              ncols=1,
                              show_estimate=false,
                              show_interval=false,
                              show_legend=false)
    @test p !== nothing
    xs = p.subplots[1].series_list[1][:x]
    @test all(x -> isapprox(x, σ_n; atol=1e-12), xs)
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
        (; vcov=:hessian)
    )
    p = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(uq; scale=:natural, ncols=1)
    @test p !== nothing
    @test p.subplots[1][:yaxis][:guide] == "Wald KDE Density"
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
    @test_throws ErrorException plot_uq_distributions(uq_profile; plot_type=:histogram)
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
        (; vcov=:hessian)
    )

    p_density = @test_logs (:info, r"sampling \+ KDE") plot_uq_distributions(uq; scale=:natural, ncols=1)
    xl_density = xlims(p_density.subplots[1])
    @test xl_density[1] <= minimum(draws)
    @test xl_density[2] >= est[1]

    p_hist = plot_uq_distributions(uq; scale=:natural, plot_type=:histogram, ncols=1)
    xl_hist = xlims(p_hist.subplots[1])
    @test xl_hist[1] <= minimum(draws)
    @test xl_hist[2] >= est[1]
end
