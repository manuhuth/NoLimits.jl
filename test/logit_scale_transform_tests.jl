using Test
using NoLimits
using ComponentArrays
using ForwardDiff
using FiniteDifferences

@testset "logit_forward / logit_inverse basic" begin
    # Round-trips for values strictly inside (0, 1).
    for x in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        @test isapprox(logit_inverse(logit_forward(x)), x; rtol=1e-8, atol=1e-10)
    end

    # Forward transform returns a finite value.
    @test isfinite(logit_forward(0.5))
    @test logit_forward(0.5) ≈ 0.0   # logit(0.5) = 0

    # Clamping: very small / large values in (0,1) are clamped on the transformed scale.
    @test logit_forward(1e-30) == -20.0
    @test logit_forward(1 - 1e-30) == 20.0

    # Clamping: transformed values outside [-20, 20] give boundary naturals.
    @test logit_inverse(100.0) == logit_inverse(20.0)
    @test logit_inverse(-100.0) == logit_inverse(-20.0)

    # Inverse always returns a value in (0, 1).
    for x in [-25.0, -20.0, -10.0, 0.0, 10.0, 20.0, 25.0]
        v = logit_inverse(x)
        @test 0.0 < v < 1.0
    end

    # logit_forward is monotone.
    @test logit_forward(0.3) < logit_forward(0.5) < logit_forward(0.7)
end

@testset "logit ForwardTransform / InverseTransform — scalar" begin
    names = [:p]
    specs = [TransformSpec(:p, :logit, (1, 1), nothing)]
    ft = ForwardTransform(names, specs)
    it = InverseTransform(names, specs)

    θ = ComponentArray((p=0.3,))
    θt = ft(θ)
    @test isapprox(θt.p, logit_forward(0.3); rtol=1e-10)

    θrt = it(θt)
    @test isapprox(θrt.p, 0.3; rtol=1e-8, atol=1e-10)
end

@testset "logit ForwardTransform / InverseTransform — uniform vector" begin
    names = [:v]
    specs = [TransformSpec(:v, :logit, (3, 1), nothing)]
    ft = ForwardTransform(names, specs)
    it = InverseTransform(names, specs)

    v0 = [0.2, 0.5, 0.8]
    θ = ComponentArray((v=v0,))
    θt = ft(θ)
    @test all(isapprox.(θt.v, logit_forward.(v0); rtol=1e-10))

    θrt = it(θt)
    @test isapprox(θrt.v, v0; rtol=1e-8, atol=1e-10)
end

@testset "elementwise ForwardTransform / InverseTransform — mixed [:logit, :log, :identity]" begin
    names = [:v]
    mask = [:logit, :log, :identity]
    specs = [TransformSpec(:v, :elementwise, (3, 1), mask)]
    ft = ForwardTransform(names, specs)
    it = InverseTransform(names, specs)

    v0 = [0.4, 2.0, -1.5]
    θ = ComponentArray((v=v0,))
    θt = ft(θ)

    @test isapprox(θt.v[1], logit_forward(0.4); rtol=1e-10)
    @test isapprox(θt.v[2], log(2.0); rtol=1e-10)
    @test isapprox(θt.v[3], -1.5; rtol=1e-10)

    θrt = it(θt)
    @test isapprox(θrt.v[1], 0.4; rtol=1e-8, atol=1e-10)
    @test isapprox(θrt.v[2], 2.0; rtol=1e-8, atol=1e-10)
    @test isapprox(θrt.v[3], -1.5; rtol=1e-10)
end

@testset "elementwise ForwardTransform / InverseTransform — [:logit, :identity]" begin
    names = [:a, :b]
    specs = [
        TransformSpec(:a, :logit, (1, 1), nothing),
        TransformSpec(:b, :elementwise, (2, 1), [:log, :identity]),
    ]
    ft = ForwardTransform(names, specs)
    it = InverseTransform(names, specs)

    θ = ComponentArray((a=0.7, b=[3.0, -0.5]))
    θt = ft(θ)
    @test isapprox(θt.a, logit_forward(0.7); rtol=1e-10)
    @test isapprox(θt.b[1], log(3.0); rtol=1e-10)
    @test isapprox(θt.b[2], -0.5; rtol=1e-10)

    θrt = it(θt)
    @test isapprox(θrt.a, 0.7; rtol=1e-8, atol=1e-10)
    @test isapprox(θrt.b[1], 3.0; rtol=1e-8, atol=1e-10)
    @test isapprox(θrt.b[2], -0.5; rtol=1e-10)
end

@testset "apply_inv_jacobian_T — logit" begin
    names = [:p]
    specs = [TransformSpec(:p, :logit, (1, 1), nothing)]
    it = InverseTransform(names, specs)

    # At x=0 (i.e. p=0.5), Jacobian of sigmoid is 0.25.
    θt = ComponentArray((p=0.0,))
    grad_u = ComponentArray((p=1.0,))
    jac = apply_inv_jacobian_T(it, θt, grad_u)
    @test isapprox(jac.p, 0.25; rtol=1e-8)

    # At x=20, clamping kicks in → zero Jacobian.
    θt_clamp = ComponentArray((p=20.0,))
    jac_clamp = apply_inv_jacobian_T(it, θt_clamp, grad_u)
    @test jac_clamp.p == 0.0
end

@testset "apply_inv_jacobian_T — elementwise" begin
    names = [:v]
    mask = [:logit, :log, :identity]
    specs = [TransformSpec(:v, :elementwise, (3, 1), mask)]
    it = InverseTransform(names, specs)

    θt = ComponentArray((v=[0.0, 1.0, 2.0],))
    grad_u = ComponentArray((v=[1.0, 1.0, 1.0],))
    jac = apply_inv_jacobian_T(it, θt, grad_u)

    # logit at 0.0: σ(0)*(1-σ(0)) = 0.25
    @test isapprox(jac.v[1], 0.25; rtol=1e-8)
    # log at 1.0: exp(1.0)
    @test isapprox(jac.v[2], exp(1.0); rtol=1e-8)
    # identity
    @test isapprox(jac.v[3], 1.0; rtol=1e-10)
end

@testset "apply_inv_jacobian_T — logit vs finite differences" begin
    # Check Jacobian against finite differences on the inverse transform.
    names = [:p]
    specs = [TransformSpec(:p, :logit, (1, 1), nothing)]
    it = InverseTransform(names, specs)

    for x_val in [-5.0, -1.0, 0.0, 1.0, 5.0]
        θt = ComponentArray((p=x_val,))
        grad_u = ComponentArray((p=1.0,))
        jac = apply_inv_jacobian_T(it, θt, grad_u)

        # Finite diff of logit_inverse.
        fd_jac = central_fdm(5, 1)(logit_inverse, x_val)
        @test isapprox(jac.p, fd_jac; rtol=1e-5, atol=1e-8)
    end
end

@testset "apply_inv_jacobian_T — elementwise vs finite differences" begin
    names = [:v]
    mask = [:logit, :log, :identity]
    specs = [TransformSpec(:v, :elementwise, (3, 1), mask)]
    it = InverseTransform(names, specs)

    x_vals = [0.5, 2.0, -1.0]
    θt = ComponentArray((v=x_vals,))
    grad_u = ComponentArray((v=[1.0, 1.0, 1.0],))
    jac = apply_inv_jacobian_T(it, θt, grad_u)

    fd1 = central_fdm(5, 1)(logit_inverse, x_vals[1])
    fd2 = central_fdm(5, 1)(exp, x_vals[2])
    fd3 = 1.0  # identity

    @test isapprox(jac.v[1], fd1; rtol=1e-5, atol=1e-8)
    @test isapprox(jac.v[2], fd2; rtol=1e-5, atol=1e-8)
    @test isapprox(jac.v[3], fd3; rtol=1e-10)
end
