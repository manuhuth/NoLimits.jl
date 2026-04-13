using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff
using ReverseDiff
using Zygote
using Distributions
using LinearAlgebra
using Bijectors
using FunctionChains
using Optimisers

@testset "NormalizingPlanarFlow AD" begin
    # Compare gradients of logpdf across AD backends.
    n = 2
    flow = NormalizingPlanarFlow(n, 2)
    x = [0.1, -0.2]

    f(xv) = logpdf(flow, xv)

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), x)
    @test length(grad_fwd) == length(x)

    val_rev, grad_rev = value_and_gradient(f, AutoReverseDiff(), x)
    @test isapprox(val_rev, val_fwd; rtol=1e-6, atol=1e-8)
    @test isapprox(grad_rev, grad_fwd; rtol=1e-6, atol=1e-8)

    grad_zyg = Zygote.gradient(f, x)[1]
    @test isapprox(grad_zyg, grad_fwd; rtol=1e-6, atol=1e-8)

    # ForwardDiff through flow parameters (theta).
    q0 = MvNormal(zeros(n), I)
    Ls = [PlanarLayer(n, x -> x) for _ in 1:2]
    ts = FunctionChains.fchain(Ls)
    θ, rebuild = Optimisers.destructure(ts)
    gθ = ForwardDiff.gradient(θv -> logpdf(NormalizingPlanarFlow(θv, rebuild, q0), x), θ)
    @test length(gθ) == length(θ)

    gθ_rev = ReverseDiff.gradient(θv -> logpdf(NormalizingPlanarFlow(θv, rebuild, q0), x), θ)
    @test length(gθ_rev) == length(θ)
    @test isapprox(gθ_rev, gθ; rtol=1e-6, atol=1e-8)

    gθ_zyg = Zygote.gradient(θv -> logpdf(NormalizingPlanarFlow(θv, rebuild, q0), x), θ)[1]
    @test length(gθ_zyg) == length(θ)
    @test isapprox(gθ_zyg, gθ; rtol=1e-6, atol=1e-8)

    # Custom base distribution: MvNormal with non-zero mean and non-identity covariance
    q0_custom = MvNormal([0.5, -0.5], [2.0 0.3; 0.3 1.5])
    gθ_custom = ForwardDiff.gradient(θv -> logpdf(NormalizingPlanarFlow(θv, rebuild, q0_custom), x), θ)
    @test length(gθ_custom) == length(θ)
    @test all(isfinite, gθ_custom)

    # Gradient w.r.t. observation x with custom MvNormal base
    flow_custom = NormalizingPlanarFlow(n, 2; base_dist=q0_custom)
    gx_custom = ForwardDiff.gradient(xv -> logpdf(flow_custom, xv), x)
    @test length(gx_custom) == length(x)
    @test all(isfinite, gx_custom)

    # Custom base changes the logpdf values compared to standard base
    @test logpdf(flow_custom, x) != logpdf(NormalizingPlanarFlow(n, 2), x)

    # MvTDist base: ForwardDiff through theta (passthrough _adapt_base_dist)
    q0_t = MvTDist(3, zeros(n), Matrix{Float64}(I, n, n))
    gθ_t = ForwardDiff.gradient(θv -> logpdf(NormalizingPlanarFlow(θv, rebuild, q0_t), x), θ)
    @test length(gθ_t) == length(θ)
    @test all(isfinite, gθ_t)

    # MvTDist base: ForwardDiff through x
    flow_t = NormalizingPlanarFlow(n, 2; base_dist=q0_t)
    gx_t = ForwardDiff.gradient(xv -> logpdf(flow_t, xv), x)
    @test length(gx_t) == length(x)
    @test all(isfinite, gx_t)
end
