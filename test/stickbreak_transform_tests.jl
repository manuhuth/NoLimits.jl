using Test
using NoLimits
using ComponentArrays
using ForwardDiff
using ReverseDiff
using Zygote

@testset "StickBreakTransforms" begin

    # -----------------------------------------------------------------------
    # 1. stickbreak_forward / stickbreak_inverse round-trips
    # -----------------------------------------------------------------------
    @testset "RoundTrip k=2" begin
        p = [0.3, 0.7]
        t = stickbreak_forward(p)
        @test length(t) == 1
        p2 = stickbreak_inverse(t)
        @test length(p2) == 2
        @test isapprox(p2, p; atol=1e-10)
        @test isapprox(sum(p2), 1.0; atol=1e-14)
        @test all(p2 .>= 0)
    end

    @testset "RoundTrip k=4" begin
        p = [0.1, 0.3, 0.4, 0.2]
        t = stickbreak_forward(p)
        @test length(t) == 3
        p2 = stickbreak_inverse(t)
        @test length(p2) == 4
        @test isapprox(p2, p; atol=1e-10)
        @test isapprox(sum(p2), 1.0; atol=1e-14)
        @test all(p2 .>= 0)
    end

    @testset "RoundTrip k=5 near-zero probability" begin
        p = [0.01, 0.01, 0.01, 0.01, 0.96]
        t = stickbreak_forward(p)
        p2 = stickbreak_inverse(t)
        @test isapprox(p2, p; atol=1e-8)
        @test isapprox(sum(p2), 1.0; atol=1e-12)
    end

    @testset "RoundTrip k=3 uniform" begin
        p = [1/3, 1/3, 1/3]
        t = stickbreak_forward(p)
        p2 = stickbreak_inverse(t)
        @test isapprox(p2, p; atol=1e-10)
        @test isapprox(sum(p2), 1.0; atol=1e-14)
    end

    @testset "Unconstrained space is unrestricted" begin
        # t can be any real value
        for t_val in [-10.0, -1.0, 0.0, 1.0, 10.0, 19.0]
            p = stickbreak_inverse([t_val, 0.0])
            @test all(p .>= 0)
            @test isapprox(sum(p), 1.0; atol=1e-12)
        end
    end

    # -----------------------------------------------------------------------
    # 2. Row-wise (stickbreakrows) round-trips (internal helpers)
    # -----------------------------------------------------------------------
    @testset "RowWise 3x3 round-trip" begin
        P = [0.2 0.5 0.3;
             0.1 0.6 0.3;
             0.4 0.1 0.5]
        t = NoLimits._stickbreakrow_forward(P)
        @test length(t) == 3 * 2  # n*(n-1) = 6
        P2 = NoLimits._stickbreakrow_inverse(t, 3)
        @test isapprox(P2, P; atol=1e-10)
        @test all(P2 .>= 0)
        @test all(isapprox.(sum(P2; dims=2), 1.0; atol=1e-12))
    end

    @testset "RowWise 2x2 round-trip" begin
        P = [0.6 0.4; 0.3 0.7]
        t = NoLimits._stickbreakrow_forward(P)
        @test length(t) == 2  # n*(n-1) = 2
        P2 = NoLimits._stickbreakrow_inverse(t, 2)
        @test isapprox(P2, P; atol=1e-10)
    end

    # -----------------------------------------------------------------------
    # 3. ForwardTransform / InverseTransform round-trips
    # -----------------------------------------------------------------------
    @testset "ForwardTransform stickbreak k=4" begin
        p = [0.1, 0.3, 0.4, 0.2]
        θ = ComponentArray((p=p,))
        spec = TransformSpec(:p, :stickbreak, (4, 1), nothing)
        ft = ForwardTransform([:p], [spec])
        it = InverseTransform([:p], [spec])
        θt = ft(θ)
        @test length(θt.p) == 3
        θu = it(θt)
        @test isapprox(θu.p, p; atol=1e-10)
    end

    @testset "ForwardTransform stickbreakrows 3x3" begin
        P = [0.2 0.5 0.3; 0.1 0.6 0.3; 0.4 0.1 0.5]
        θ = ComponentArray((P=P,))
        spec = TransformSpec(:P, :stickbreakrows, (3, 3), nothing)
        ft = ForwardTransform([:P], [spec])
        it = InverseTransform([:P], [spec])
        θt = ft(θ)
        @test length(θt.P) == 6  # n*(n-1) = 6
        θu = it(θt)
        @test isapprox(θu.P, P; atol=1e-10)
    end

    # -----------------------------------------------------------------------
    # 4. apply_inv_jacobian_T via finite differences
    # -----------------------------------------------------------------------
    @testset "apply_inv_jacobian_T stickbreak k=3" begin
        p0 = [0.2, 0.5, 0.3]
        t0 = stickbreak_forward(p0)
        spec = TransformSpec(:p, :stickbreak, (3, 1), nothing)
        it = InverseTransform([:p], [spec])

        # Natural-scale gradient (random)
        g_u = [1.0, -0.5, 0.3]
        θt = ComponentArray((p=t0,))
        grad_u = ComponentArray((p=g_u,))

        # Compute via apply_inv_jacobian_T
        result = apply_inv_jacobian_T(it, θt, grad_u)
        g_t_analytic = result.p

        # Finite-difference check: g_t[j] ≈ sum_i J[i,j] * g_u[i]
        # where J[i,j] = ∂p[i]/∂t[j] (k × k-1 Jacobian)
        h = 1e-6
        J_fd = zeros(3, 2)
        for j in 1:2
            tp = copy(t0); tp[j] += h
            tm = copy(t0); tm[j] -= h
            pp = stickbreak_inverse(tp)
            pm = stickbreak_inverse(tm)
            J_fd[:, j] = (pp .- pm) ./ (2h)
        end
        g_t_fd = J_fd' * g_u
        @test isapprox(g_t_analytic, g_t_fd; rtol=1e-5, atol=1e-7)
    end

    @testset "apply_inv_jacobian_T stickbreak k=4" begin
        p0 = [0.1, 0.3, 0.4, 0.2]
        t0 = stickbreak_forward(p0)
        spec = TransformSpec(:p, :stickbreak, (4, 1), nothing)
        it = InverseTransform([:p], [spec])
        g_u = [0.5, -1.0, 0.2, 0.7]
        θt = ComponentArray((p=t0,))
        grad_u = ComponentArray((p=g_u,))
        result = apply_inv_jacobian_T(it, θt, grad_u)
        g_t_analytic = result.p

        h = 1e-6
        J_fd = zeros(4, 3)
        for j in 1:3
            tp = copy(t0); tp[j] += h
            tm = copy(t0); tm[j] -= h
            pp = stickbreak_inverse(tp)
            pm = stickbreak_inverse(tm)
            J_fd[:, j] = (pp .- pm) ./ (2h)
        end
        g_t_fd = J_fd' * g_u
        @test isapprox(g_t_analytic, g_t_fd; rtol=1e-5, atol=1e-7)
    end

    @testset "apply_inv_jacobian_T stickbreakrows 3x3" begin
        P0 = [0.2 0.5 0.3; 0.1 0.6 0.3; 0.4 0.1 0.5]
        t0 = NoLimits._stickbreakrow_forward(P0)
        spec = TransformSpec(:P, :stickbreakrows, (3, 3), nothing)
        it = InverseTransform([:P], [spec])

        G_u = [1.0 -0.5 0.3; 0.2 -0.1 0.8; -0.3 0.4 0.1]
        θt = ComponentArray((P=t0,))
        grad_u = ComponentArray((P=G_u,))
        result = apply_inv_jacobian_T(it, θt, grad_u)
        g_t_analytic = result.P  # should be 6-vector

        # Finite-difference check for row 1 (j-index 1:2 in t0)
        h = 1e-6
        J_fd_row1 = zeros(3, 2)
        for j in 1:2
            tp = copy(t0); tp[j] += h
            tm = copy(t0); tm[j] -= h
            Pp = NoLimits._stickbreakrow_inverse(tp, 3)
            Pm = NoLimits._stickbreakrow_inverse(tm, 3)
            J_fd_row1[:, j] = (Pp[1, :] .- Pm[1, :]) ./ (2h)
        end
        g_t_fd_row1 = J_fd_row1' * G_u[1, :]
        @test isapprox(g_t_analytic[1:2], g_t_fd_row1; rtol=1e-5, atol=1e-7)
    end

    # -----------------------------------------------------------------------
    # 5. AD compatibility checks
    # -----------------------------------------------------------------------
    @testset "ForwardDiff through stickbreak_forward" begin
        f(v) = sum(stickbreak_forward(v ./ sum(v)))
        v0 = [0.3, 0.4, 0.3]
        g = ForwardDiff.gradient(f, v0)
        @test length(g) == 3
        @test all(isfinite, g)
    end

    @testset "Zygote through stickbreak_inverse" begin
        f(t) = sum(stickbreak_inverse(t) .^ 2)
        t0 = [0.5, -0.5]
        g = Zygote.gradient(f, t0)[1]
        @test length(g) == 2
        @test all(isfinite, g)
    end

    @testset "ForwardDiff through stickbreak round-trip" begin
        f(v) = begin
            p = v ./ sum(v)
            t = stickbreak_forward(p)
            p2 = stickbreak_inverse(t)
            return sum(p2 .^ 2)
        end
        v0 = [0.25, 0.35, 0.4]
        g = ForwardDiff.gradient(f, v0)
        @test all(isfinite, g)
    end

    # -----------------------------------------------------------------------
    # 6. _coords_for_param for stickbreak/stickbreakrows (uq/common.jl)
    # -----------------------------------------------------------------------
    @testset "_coords_for_param stickbreak natural drops last" begin
        p = [0.2, 0.5, 0.3]
        spec = TransformSpec(:p, :stickbreak, (3, 1), nothing)
        coords_n = NoLimits._coords_for_param(p, spec; natural=true)
        @test length(coords_n) == 2
        @test isapprox(coords_n, p[1:2]; atol=1e-14)
    end

    @testset "_coords_for_param stickbreak transformed" begin
        p = [0.2, 0.5, 0.3]
        t = stickbreak_forward(p)
        spec = TransformSpec(:p, :stickbreak, (3, 1), nothing)
        coords_t = NoLimits._coords_for_param(t, spec; natural=false)
        @test length(coords_t) == 2
        @test isapprox(coords_t, t; atol=1e-14)
    end

    @testset "_coords_for_param stickbreakrows natural drops last column" begin
        P = [0.2 0.5 0.3; 0.1 0.6 0.3; 0.4 0.1 0.5]
        spec = TransformSpec(:P, :stickbreakrows, (3, 3), nothing)
        coords_n = NoLimits._coords_for_param(P, spec; natural=true)
        @test length(coords_n) == 6  # n*(n-1) = 6
        expected = [P[1,1], P[1,2], P[2,1], P[2,2], P[3,1], P[3,2]]
        @test isapprox(coords_n, expected; atol=1e-14)
    end

    @testset "_coords_for_param stickbreakrows transformed" begin
        P = [0.2 0.5 0.3; 0.1 0.6 0.3; 0.4 0.1 0.5]
        t = NoLimits._stickbreakrow_forward(P)
        spec = TransformSpec(:P, :stickbreakrows, (3, 3), nothing)
        coords_t = NoLimits._coords_for_param(t, spec; natural=false)
        @test length(coords_t) == 6
        @test isapprox(coords_t, t; atol=1e-14)
    end

    # Both natural=true and natural=false return the same length (the Wald constraint)
    @testset "Wald constraint: same length for natural and transformed" begin
        p = [0.1, 0.4, 0.3, 0.2]
        spec = TransformSpec(:p, :stickbreak, (4, 1), nothing)
        t = stickbreak_forward(p)
        cn = NoLimits._coords_for_param(p, spec; natural=true)
        ct = NoLimits._coords_for_param(t, spec; natural=false)
        @test length(cn) == length(ct) == 3

        P = [0.2 0.5 0.3; 0.1 0.6 0.3; 0.4 0.1 0.5]
        spec2 = TransformSpec(:P, :stickbreakrows, (3, 3), nothing)
        t2 = NoLimits._stickbreakrow_forward(P)
        cn2 = NoLimits._coords_for_param(P, spec2; natural=true)
        ct2 = NoLimits._coords_for_param(t2, spec2; natural=false)
        @test length(cn2) == length(ct2) == 6
    end

end
