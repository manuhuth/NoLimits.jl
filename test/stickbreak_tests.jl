using Test
using NoLimits
using ComponentArrays
using DataFrames
using Distributions
using ForwardDiff
using LinearAlgebra
using Random

# Merged stickbreak coverage: parameter blocks, transforms, and UQ helpers.

@testset "ProbabilityVectorAndDiscreteTransitionMatrix" begin

    # -----------------------------------------------------------------------
    # 1. ProbabilityVector constructor
    # -----------------------------------------------------------------------
    @testset "ProbabilityVector basic construction" begin
        p = ProbabilityVector([0.2, 0.5, 0.3])
        @test p.name == :unnamed
        @test isapprox(p.value, [0.2, 0.5, 0.3]; atol = 1e-14)
        @test isapprox(sum(p.value), 1.0; atol = 1e-14)
        @test p.scale == :stickbreak
        @test p.prior isa Priorless
        @test p.calculate_se == true
    end

    @testset "ProbabilityVector with name and kwargs" begin
        p = ProbabilityVector([0.1, 0.9]; name = :pi, calculate_se = true,
            prior = Dirichlet([1.0, 1.0]))
        @test p.name == :pi
        @test p.calculate_se == true
        @test p.prior isa Dirichlet
    end

    @testset "ProbabilityVector silent normalization" begin
        # Sum slightly off from 1 — should be normalized silently
        v = [0.2, 0.5, 0.3 + 1e-8]
        p = ProbabilityVector(v)
        @test isapprox(sum(p.value), 1.0; atol = 1e-14)
    end

    @testset "ProbabilityVector error: length < 2" begin
        @test_throws ErrorException ProbabilityVector([1.0])
    end

    @testset "ProbabilityVector error: negative entry" begin
        @test_throws ErrorException ProbabilityVector([-0.1, 0.6, 0.5])
    end

    @testset "ProbabilityVector error: sum too far from 1" begin
        @test_throws ErrorException ProbabilityVector([0.2, 0.5, 0.5])
    end

    @testset "ProbabilityVector error: invalid scale" begin
        @test_throws ErrorException ProbabilityVector([0.3, 0.7]; scale = :log)
    end

    # -----------------------------------------------------------------------
    # 2. DiscreteTransitionMatrix constructor
    # -----------------------------------------------------------------------
    @testset "DiscreteTransitionMatrix basic construction" begin
        P = [0.7 0.3; 0.4 0.6]
        A = DiscreteTransitionMatrix(P)
        @test A.name == :unnamed
        @test isapprox(A.value, P; atol = 1e-14)
        @test all(isapprox.(sum(A.value; dims = 2), 1.0; atol = 1e-14))
        @test A.scale == :stickbreakrows
        @test A.prior isa Priorless
        @test A.calculate_se == true
    end

    @testset "DiscreteTransitionMatrix 3x3" begin
        P = [0.6 0.3 0.1; 0.1 0.7 0.2; 0.2 0.5 0.3]
        A = DiscreteTransitionMatrix(P; name = :T, calculate_se = true)
        @test A.name == :T
        @test isapprox(A.value, P; atol = 1e-14)
        @test A.calculate_se == true
    end

    @testset "DiscreteTransitionMatrix silent row normalization" begin
        P = [0.6 0.4+1e-8; 0.3 0.7]
        A = DiscreteTransitionMatrix(P)
        @test all(isapprox.(sum(A.value; dims = 2), 1.0; atol = 1e-14))
    end

    @testset "DiscreteTransitionMatrix error: non-square" begin
        @test_throws ErrorException DiscreteTransitionMatrix([0.5 0.5; 0.3 0.4; 0.2 0.8])
    end

    @testset "DiscreteTransitionMatrix error: n < 2" begin
        @test_throws ErrorException DiscreteTransitionMatrix(ones(1, 1))
    end

    @testset "DiscreteTransitionMatrix error: negative entry" begin
        @test_throws ErrorException DiscreteTransitionMatrix([-0.1 1.1; 0.5 0.5])
    end

    @testset "DiscreteTransitionMatrix error: row sum too far from 1" begin
        @test_throws ErrorException DiscreteTransitionMatrix([0.5 0.5; 0.3 0.3])
    end

    @testset "DiscreteTransitionMatrix error: invalid scale" begin
        @test_throws ErrorException DiscreteTransitionMatrix(
            [0.5 0.5; 0.4 0.6]; scale = :cholesky)
    end

    # -----------------------------------------------------------------------
    # 3. FixedEffects macro integration
    # -----------------------------------------------------------------------
    @testset "build_fixed_effects ProbabilityVector k=3" begin
        fe = @fixedEffects begin
            pi = ProbabilityVector([0.2, 0.5, 0.3]; calculate_se = true)
        end
        @test :pi in get_names(fe)
        θu = get_θ0_untransformed(fe)
        @test isapprox(θu.pi, [0.2, 0.5, 0.3]; atol = 1e-14)
        θt = get_θ0_transformed(fe)
        @test length(θt.pi) == 2  # k-1 = 2
        # Round-trip
        θu_rt = get_inverse_transform(fe)(θt)
        @test isapprox(θu_rt.pi, θu.pi; atol = 1e-10)
    end

    @testset "build_fixed_effects DiscreteTransitionMatrix 3x3" begin
        P = [0.6 0.3 0.1; 0.1 0.7 0.2; 0.2 0.5 0.3]
        fe = @fixedEffects begin
            T = DiscreteTransitionMatrix(P; calculate_se = true)
        end
        @test :T in get_names(fe)
        θu = get_θ0_untransformed(fe)
        @test isapprox(θu.T, P; atol = 1e-14)
        θt = get_θ0_transformed(fe)
        @test length(θt.T) == 3 * 2  # n*(n-1) = 6
        # Round-trip
        θu_rt = get_inverse_transform(fe)(θt)
        @test isapprox(θu_rt.T, P; atol = 1e-10)
    end

    @testset "mixed @fixedEffects with ProbabilityVector and other types" begin
        fe = @fixedEffects begin
            a = RealNumber(1.0; scale = :log)
            pi = ProbabilityVector([0.3, 0.4, 0.3])
            T = DiscreteTransitionMatrix([0.7 0.3; 0.2 0.8])
            sigma = RealNumber(0.5; scale = :log)
        end
        names = get_names(fe)
        @test :a in names
        @test :pi in names
        @test :T in names
        @test :sigma in names
        θt = get_θ0_transformed(fe)
        @test length(θt.pi) == 2   # k-1
        @test length(θt.T) == 2    # n*(n-1) = 2
    end

    # -----------------------------------------------------------------------
    # 4. _param_spec dispatch
    # -----------------------------------------------------------------------
    @testset "_param_spec ProbabilityVector" begin
        p = ProbabilityVector([0.2, 0.5, 0.3]; name = :pi)
        spec = NoLimits._param_spec(:pi, p)
        @test spec.kind == :stickbreak
        @test spec.size == (3, 1)
        @test spec.mask === nothing
    end

    @testset "_param_spec DiscreteTransitionMatrix" begin
        P = [0.7 0.3; 0.4 0.6]
        p = DiscreteTransitionMatrix(P; name = :T)
        spec = NoLimits._param_spec(:T, p)
        @test spec.kind == :stickbreakrows
        @test spec.size == (2, 2)
        @test spec.mask === nothing
    end

    # -----------------------------------------------------------------------
    # 5. Bounds on transformed scale
    # -----------------------------------------------------------------------
    @testset "Transformed bounds are unconstrained for stickbreak" begin
        fe = @fixedEffects begin
            pi = ProbabilityVector([0.2, 0.5, 0.3])
        end
        lb, ub = get_bounds_transformed(fe)
        @test all(lb.pi .== -Inf)
        @test all(ub.pi .== Inf)
    end

    @testset "Transformed bounds are unconstrained for stickbreakrows" begin
        fe = @fixedEffects begin
            T = DiscreteTransitionMatrix([0.7 0.3; 0.4 0.6])
        end
        lb, ub = get_bounds_transformed(fe)
        @test all(lb.T .== -Inf)
        @test all(ub.T .== Inf)
    end

    # -----------------------------------------------------------------------
    # 6. Flat names
    # -----------------------------------------------------------------------
    @testset "Flat names ProbabilityVector k=3" begin
        fe = @fixedEffects begin
            pi = ProbabilityVector([0.2, 0.5, 0.3]; calculate_se = true)
        end
        fn = get_flat_names(fe)
        @test fn == [:pi_1, :pi_2]
        @test length(get_se_mask(fe)) == 2
        @test all(get_se_mask(fe))
    end

    @testset "Flat names DiscreteTransitionMatrix 3x3" begin
        P = [0.6 0.3 0.1; 0.1 0.7 0.2; 0.2 0.5 0.3]
        fe = @fixedEffects begin
            T = DiscreteTransitionMatrix(P; calculate_se = true)
        end
        fn = get_flat_names(fe)
        @test length(fn) == 6
        @test all(get_se_mask(fe))
    end

    # -----------------------------------------------------------------------
    # 7. apply_inv_jacobian_T via @fixedEffects
    # -----------------------------------------------------------------------
    @testset "apply_inv_jacobian_T via FixedEffects ProbabilityVector k=4" begin
        p0 = [0.1, 0.3, 0.4, 0.2]
        fe = @fixedEffects begin
            pi = ProbabilityVector(p0)
        end
        θt = get_θ0_transformed(fe)
        inv_t = get_inverse_transform(fe)
        g_u = ComponentArray((pi = [0.5, -1.0, 0.2, 0.7],))
        result = apply_inv_jacobian_T(inv_t, θt, g_u)
        @test length(result.pi) == 3  # k-1

        # Finite-difference check
        t0 = Vector(θt.pi)
        h = 1e-6
        J_fd = zeros(4, 3)
        for j in 1:3
            tp = copy(t0)
            tp[j] += h
            tm = copy(t0)
            tm[j] -= h
            pp = stickbreak_inverse(tp)
            pm = stickbreak_inverse(tm)
            J_fd[:, j] = (pp .- pm) ./ (2h)
        end
        g_t_fd = J_fd' * [0.5, -1.0, 0.2, 0.7]
        @test isapprox(result.pi, g_t_fd; rtol = 1e-5, atol = 1e-7)
    end

    # -----------------------------------------------------------------------
    # 8. Integration with @Model
    # -----------------------------------------------------------------------
    @testset "@Model with ProbabilityVector" begin
        model = @Model begin
            @fixedEffects begin
                pi = ProbabilityVector([0.3, 0.4, 0.3])
                sigma = RealNumber(0.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(pi[1] - pi[2], sigma)
            end
        end
        df = DataFrame(
            ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.3, 0.1])
        dm = DataModel(model, df; primary_id = :ID, time_col = :t)
        @test length(get_individuals(dm)) == 2
        # Should be able to get transforms
        fe = dm.model.fixed.fixed
        @test :pi in get_names(fe)
        θt = get_θ0_transformed(fe)
        @test length(θt.pi) == 2
    end

    @testset "@Model with DiscreteTransitionMatrix" begin
        model = @Model begin
            @fixedEffects begin
                T = DiscreteTransitionMatrix([0.8 0.2; 0.1 0.9])
                sigma = RealNumber(0.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(T[1, 1] - T[2, 1], sigma)
            end
        end
        df = DataFrame(
            ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.3, 0.1])
        dm = DataModel(model, df; primary_id = :ID, time_col = :t)
        fe = dm.model.fixed.fixed
        @test :T in get_names(fe)
        θt = get_θ0_transformed(fe)
        @test length(θt.T) == 2  # n*(n-1) = 2
    end
end

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
        @test isapprox(p2, p; atol = 1e-10)
        @test isapprox(sum(p2), 1.0; atol = 1e-14)
        @test all(p2 .>= 0)
    end

    @testset "RoundTrip k=4" begin
        p = [0.1, 0.3, 0.4, 0.2]
        t = stickbreak_forward(p)
        @test length(t) == 3
        p2 = stickbreak_inverse(t)
        @test length(p2) == 4
        @test isapprox(p2, p; atol = 1e-10)
        @test isapprox(sum(p2), 1.0; atol = 1e-14)
        @test all(p2 .>= 0)
    end

    @testset "RoundTrip k=5 near-zero probability" begin
        p = [0.01, 0.01, 0.01, 0.01, 0.96]
        t = stickbreak_forward(p)
        p2 = stickbreak_inverse(t)
        @test isapprox(p2, p; atol = 1e-8)
        @test isapprox(sum(p2), 1.0; atol = 1e-12)
    end

    @testset "RoundTrip k=3 uniform" begin
        p = [1 / 3, 1 / 3, 1 / 3]
        t = stickbreak_forward(p)
        p2 = stickbreak_inverse(t)
        @test isapprox(p2, p; atol = 1e-10)
        @test isapprox(sum(p2), 1.0; atol = 1e-14)
    end

    @testset "Unconstrained space is unrestricted" begin
        # t can be any real value
        for t_val in [-10.0, -1.0, 0.0, 1.0, 10.0, 19.0]
            p = stickbreak_inverse([t_val, 0.0])
            @test all(p .>= 0)
            @test isapprox(sum(p), 1.0; atol = 1e-12)
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
        @test isapprox(P2, P; atol = 1e-10)
        @test all(P2 .>= 0)
        @test all(isapprox.(sum(P2; dims = 2), 1.0; atol = 1e-12))
    end

    @testset "RowWise 2x2 round-trip" begin
        P = [0.6 0.4; 0.3 0.7]
        t = NoLimits._stickbreakrow_forward(P)
        @test length(t) == 2  # n*(n-1) = 2
        P2 = NoLimits._stickbreakrow_inverse(t, 2)
        @test isapprox(P2, P; atol = 1e-10)
    end

    # -----------------------------------------------------------------------
    # 3. ForwardTransform / InverseTransform round-trip (raw TransformSpec level;
    #    the stickbreakrows analogue is covered at macro level above)
    # -----------------------------------------------------------------------
    @testset "ForwardTransform stickbreak k=4" begin
        p = [0.1, 0.3, 0.4, 0.2]
        θ = ComponentArray((p = p,))
        spec = TransformSpec(:p, :stickbreak, (4, 1), nothing)
        ft = ForwardTransform([:p], [spec])
        it = InverseTransform([:p], [spec])
        θt = ft(θ)
        @test length(θt.p) == 3
        θu = it(θt)
        @test isapprox(θu.p, p; atol = 1e-10)
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
        θt = ComponentArray((p = t0,))
        grad_u = ComponentArray((p = g_u,))

        # Compute via apply_inv_jacobian_T
        result = apply_inv_jacobian_T(it, θt, grad_u)
        g_t_analytic = result.p

        # Finite-difference check: g_t[j] ≈ sum_i J[i,j] * g_u[i]
        # where J[i,j] = ∂p[i]/∂t[j] (k × k-1 Jacobian)
        h = 1e-6
        J_fd = zeros(3, 2)
        for j in 1:2
            tp = copy(t0)
            tp[j] += h
            tm = copy(t0)
            tm[j] -= h
            pp = stickbreak_inverse(tp)
            pm = stickbreak_inverse(tm)
            J_fd[:, j] = (pp .- pm) ./ (2h)
        end
        g_t_fd = J_fd' * g_u
        @test isapprox(g_t_analytic, g_t_fd; rtol = 1e-5, atol = 1e-7)
    end

    @testset "apply_inv_jacobian_T stickbreakrows 3x3" begin
        P0 = [0.2 0.5 0.3; 0.1 0.6 0.3; 0.4 0.1 0.5]
        t0 = NoLimits._stickbreakrow_forward(P0)
        spec = TransformSpec(:P, :stickbreakrows, (3, 3), nothing)
        it = InverseTransform([:P], [spec])

        G_u = [1.0 -0.5 0.3; 0.2 -0.1 0.8; -0.3 0.4 0.1]
        θt = ComponentArray((P = t0,))
        grad_u = ComponentArray((P = G_u,))
        result = apply_inv_jacobian_T(it, θt, grad_u)
        g_t_analytic = result.P  # should be 6-vector

        # Finite-difference check for row 1 (j-index 1:2 in t0)
        h = 1e-6
        J_fd_row1 = zeros(3, 2)
        for j in 1:2
            tp = copy(t0)
            tp[j] += h
            tm = copy(t0)
            tm[j] -= h
            Pp = NoLimits._stickbreakrow_inverse(tp, 3)
            Pm = NoLimits._stickbreakrow_inverse(tm, 3)
            J_fd_row1[:, j] = (Pp[1, :] .- Pm[1, :]) ./ (2h)
        end
        g_t_fd_row1 = J_fd_row1' * G_u[1, :]
        @test isapprox(g_t_analytic[1:2], g_t_fd_row1; rtol = 1e-5, atol = 1e-7)
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
        coords_n = NoLimits._coords_for_param(p, spec; natural = true)
        @test length(coords_n) == 2
        @test isapprox(coords_n, p[1:2]; atol = 1e-14)
    end

    @testset "_coords_for_param stickbreak transformed" begin
        p = [0.2, 0.5, 0.3]
        t = stickbreak_forward(p)
        spec = TransformSpec(:p, :stickbreak, (3, 1), nothing)
        coords_t = NoLimits._coords_for_param(t, spec; natural = false)
        @test length(coords_t) == 2
        @test isapprox(coords_t, t; atol = 1e-14)
    end

    @testset "_coords_for_param stickbreakrows natural drops last column" begin
        P = [0.2 0.5 0.3; 0.1 0.6 0.3; 0.4 0.1 0.5]
        spec = TransformSpec(:P, :stickbreakrows, (3, 3), nothing)
        coords_n = NoLimits._coords_for_param(P, spec; natural = true)
        @test length(coords_n) == 6  # n*(n-1) = 6
        expected = [P[1, 1], P[1, 2], P[2, 1], P[2, 2], P[3, 1], P[3, 2]]
        @test isapprox(coords_n, expected; atol = 1e-14)
    end

    @testset "_coords_for_param stickbreakrows transformed" begin
        P = [0.2 0.5 0.3; 0.1 0.6 0.3; 0.4 0.1 0.5]
        t = NoLimits._stickbreakrow_forward(P)
        spec = TransformSpec(:P, :stickbreakrows, (3, 3), nothing)
        coords_t = NoLimits._coords_for_param(t, spec; natural = false)
        @test length(coords_t) == 6
        @test isapprox(coords_t, t; atol = 1e-14)
    end
end

@testset "StickBreakUQ" begin

    # -----------------------------------------------------------------------
    # 1. _flat_transform_kinds_for_free
    # -----------------------------------------------------------------------
    @testset "_flat_transform_kinds_for_free stickbreak k=4" begin
        fe = @fixedEffects begin
            pi = ProbabilityVector([0.1, 0.4, 0.3, 0.2]; calculate_se = true)
        end
        kinds = NoLimits._flat_transform_kinds_for_free(fe, [:pi])
        @test length(kinds) == 3  # k-1 = 3
        @test all(k -> k == :stickbreak, kinds)
    end

    @testset "_flat_transform_kinds_for_free stickbreakrows 3x3" begin
        P = [0.6 0.3 0.1; 0.1 0.7 0.2; 0.2 0.5 0.3]
        fe = @fixedEffects begin
            T = DiscreteTransitionMatrix(P; calculate_se = true)
        end
        kinds = NoLimits._flat_transform_kinds_for_free(fe, [:T])
        @test length(kinds) == 6  # n*(n-1) = 6
        @test all(k -> k == :stickbreakrows, kinds)
    end

    @testset "_flat_transform_kinds_for_free mixed" begin
        fe = @fixedEffects begin
            a = RealNumber(1.0; scale = :log, calculate_se = true)
            pi = ProbabilityVector([0.3, 0.4, 0.3]; calculate_se = true)
            T = DiscreteTransitionMatrix([0.7 0.3; 0.4 0.6]; calculate_se = true)
        end
        kinds = NoLimits._flat_transform_kinds_for_free(fe, [:a, :pi, :T])
        @test kinds[1] == :log           # a
        @test kinds[2] == :stickbreak    # pi[1]
        @test kinds[3] == :stickbreak    # pi[2] (k=3, so k-1=2)
        @test kinds[4] == :stickbreakrows  # T[1,1]
        @test kinds[5] == :stickbreakrows  # T[2,1]
        @test length(kinds) == 5  # 1 + 2 + 2
    end

    # -----------------------------------------------------------------------
    # 2. _wald_closed_form_kind returns :none for stickbreak/stickbreakrows
    # -----------------------------------------------------------------------
    @testset "_wald_closed_form_kind stickbreak → :none (KDE fallback)" begin
        vcov_t = Matrix{Float64}(I, 2, 2)
        transforms = [:stickbreak, :stickbreakrows]
        for tr in transforms
            kind = NoLimits._wald_closed_form_kind(:wald, :natural, 1, vcov_t, [tr])
            @test kind == :none
        end
    end

    # -----------------------------------------------------------------------
    # 3. Wald constraint (same length for natural and transformed), asserted
    #    at both API levels: _coords_for_param and _coords_on_transformed_layout
    # -----------------------------------------------------------------------
    @testset "Wald constraint: same length for natural and transformed" begin
        p = [0.1, 0.4, 0.3, 0.2]
        spec = TransformSpec(:p, :stickbreak, (4, 1), nothing)
        t = stickbreak_forward(p)
        cn = NoLimits._coords_for_param(p, spec; natural = true)
        ct = NoLimits._coords_for_param(t, spec; natural = false)
        @test length(cn) == length(ct) == 3

        P = [0.2 0.5 0.3; 0.1 0.6 0.3; 0.4 0.1 0.5]
        spec2 = TransformSpec(:P, :stickbreakrows, (3, 3), nothing)
        t2 = NoLimits._stickbreakrow_forward(P)
        cn2 = NoLimits._coords_for_param(P, spec2; natural = true)
        ct2 = NoLimits._coords_for_param(t2, spec2; natural = false)
        @test length(cn2) == length(ct2) == 6
    end

    @testset "_coords_on_transformed_layout same length natural/transformed" begin
        p = [0.1, 0.4, 0.3, 0.2]
        P = [0.6 0.3 0.1; 0.1 0.7 0.2; 0.2 0.5 0.3]

        fe = @fixedEffects begin
            pi = ProbabilityVector(p; calculate_se = true)
            T = DiscreteTransitionMatrix(P; calculate_se = true)
        end

        θu = get_θ0_untransformed(fe)
        θt = get_θ0_transformed(fe)
        names = [:pi, :T]

        coords_t = NoLimits._coords_on_transformed_layout(fe, θt, names; natural = false)
        coords_n = NoLimits._coords_on_transformed_layout(fe, θu, names; natural = true)

        # Must have same length (Wald constraint)
        @test length(coords_t) == length(coords_n)
        @test length(coords_t) == 3 + 6  # (k-1) + n*(n-1) = 3 + 6

        # Natural coords for pi should be first k-1 = 3 probabilities
        @test isapprox(coords_n[1:3], p[1:3]; atol = 1e-14)
        # Natural coords for T should be first n-1 = 2 cols of each row
        @test isapprox(coords_n[4:5], P[1, 1:2]; atol = 1e-14)
        @test isapprox(coords_n[6:7], P[2, 1:2]; atol = 1e-14)
        @test isapprox(coords_n[8:9], P[3, 1:2]; atol = 1e-14)
    end

    # -----------------------------------------------------------------------
    # 4. End-to-end MLE fit with ProbabilityVector (no UQ compute needed,
    #    just verify the estimation pipeline doesn't error)
    # -----------------------------------------------------------------------
    @testset "MLE fit with ProbabilityVector" begin
        model = @Model begin
            @fixedEffects begin
                pi = ProbabilityVector([0.3, 0.4, 0.3]; calculate_se = true)
                sigma = RealNumber(0.5; scale = :log, calculate_se = true)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(pi[1] * 2.0 + pi[2] * 1.0, sigma)
            end
        end
        df = DataFrame(
            ID = vcat(fill(1, 5), fill(2, 5)),
            t = vcat(1:5, 1:5) .* 1.0,
            y = vcat(
                randn(MersenneTwister(1), 5) .+ 1.5, randn(MersenneTwister(2), 5) .+ 1.5)
        )
        dm = DataModel(model, df; primary_id = :ID, time_col = :t)
        res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
        params = NoLimits.get_params(res; scale = :untransformed)
        pi_est = params.pi
        @test length(pi_est) == 3
        @test isapprox(sum(pi_est), 1.0; atol = 1e-6)
        @test all(pi_est .>= 0)
    end

    @testset "MLE fit with DiscreteTransitionMatrix" begin
        model = @Model begin
            @fixedEffects begin
                T = DiscreteTransitionMatrix([0.8 0.2; 0.1 0.9]; calculate_se = true)
                sigma = RealNumber(0.5; scale = :log, calculate_se = true)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(T[1, 1] - T[2, 1], sigma)
            end
        end
        df = DataFrame(
            ID = vcat(fill(1, 5), fill(2, 5)),
            t = vcat(1:5, 1:5) .* 1.0,
            y = vcat(
                randn(MersenneTwister(3), 5) .+ 0.7, randn(MersenneTwister(4), 5) .+ 0.7)
        )
        dm = DataModel(model, df; primary_id = :ID, time_col = :t)
        res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
        params = NoLimits.get_params(res; scale = :untransformed)
        T_est = params.T
        @test size(T_est) == (2, 2)
        @test all(T_est .>= 0)
        @test isapprox(sum(T_est[1, :]), 1.0; atol = 1e-6)
        @test isapprox(sum(T_est[2, :]), 1.0; atol = 1e-6)
    end
end
