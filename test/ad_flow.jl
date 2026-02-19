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
end
