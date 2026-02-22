using Test
using NoLimits
using ComponentArrays
using LinearAlgebra
using Distributions
using ForwardDiff
using Random

# ---------------------------------------------------------------------------
# Helper: build a valid 3×3 Q-matrix for testing.
# ---------------------------------------------------------------------------
function _make_q3()
    Q = [
        -0.5   0.3   0.2;
         0.1  -0.4   0.3;
         0.2   0.1  -0.3
    ]
    return Q
end

# ---------------------------------------------------------------------------
# 1. Constructor tests
# ---------------------------------------------------------------------------
@testset "ContinuousTransitionMatrix construction" begin

    @testset "basic construction (3×3)" begin
        Q = _make_q3()
        p = ContinuousTransitionMatrix(Q)
        @test p.name == :unnamed
        @test p.scale == :lograterows
        @test p.prior isa Priorless
        @test p.calculate_se == false
        @test size(p.value) == (3, 3)
        # Off-diagonals preserved
        @test isapprox(p.value[1, 2], 0.3; atol=1e-14)
        @test isapprox(p.value[2, 3], 0.3; atol=1e-14)
        # Rows sum to zero
        for i in 1:3
            @test isapprox(sum(p.value[i, :]), 0.0; atol=1e-14)
        end
    end

    @testset "diagonal silently recomputed" begin
        # Provide off-diagonal correctly, diagonal slightly wrong → silently fixed.
        Q = [
            -0.49   0.3   0.2;
             0.1   -0.39  0.3;
             0.2    0.1  -0.29
        ]
        p = ContinuousTransitionMatrix(Q)
        for i in 1:3
            @test isapprox(sum(p.value[i, :]), 0.0; atol=1e-12)
        end
        @test isapprox(p.value[1, 1], -(0.3 + 0.2); atol=1e-14)
    end

    @testset "name and kwargs" begin
        Q = _make_q3()
        p = ContinuousTransitionMatrix(Q; name=:Q, calculate_se=true)
        @test p.name == :Q
        @test p.calculate_se == true
    end

    @testset "2×2 construction" begin
        Q = [-1.0  1.0; 2.0  -2.0]
        p = ContinuousTransitionMatrix(Q)
        @test size(p.value) == (2, 2)
        @test isapprox(sum(p.value[1, :]), 0.0; atol=1e-14)
        @test isapprox(sum(p.value[2, :]), 0.0; atol=1e-14)
    end

    @testset "error: non-square matrix" begin
        @test_throws ErrorException ContinuousTransitionMatrix([0.5 0.5 0.0; 0.2 0.8 0.0])
    end

    @testset "error: 1×1 matrix" begin
        @test_throws ErrorException ContinuousTransitionMatrix(reshape([0.0], 1, 1))
    end

    @testset "error: negative off-diagonal" begin
        Q = _make_q3()
        Q[1, 2] = -0.1
        @test_throws ErrorException ContinuousTransitionMatrix(Q)
    end

    @testset "error: negative off-diagonal (second form)" begin
        Q = _make_q3()
        Q[2, 3] = -0.5
        @test_throws ErrorException ContinuousTransitionMatrix(Q)
    end

    @testset "error: invalid scale" begin
        @test_throws ErrorException ContinuousTransitionMatrix(_make_q3(); scale=:stickbreakrows)
    end

    @testset "error: invalid prior" begin
        @test_throws ErrorException ContinuousTransitionMatrix(_make_q3(); prior=42)
    end
end

# ---------------------------------------------------------------------------
# 2. Transform round-trip tests
# ---------------------------------------------------------------------------
@testset "lograterows transform round-trip" begin

    @testset "forward → inverse round-trip (3×3)" begin
        Q = _make_q3()
        t = lograterows_forward(Q)
        @test length(t) == 6  # 3*(3-1)
        Q2 = lograterows_inverse(t, 3)
        @test isapprox(Q2, Q; atol=1e-12)
    end

    @testset "forward → inverse round-trip (2×2)" begin
        Q = [-1.5  1.5; 0.8  -0.8]
        t = lograterows_forward(Q)
        @test length(t) == 2
        Q2 = lograterows_inverse(t, 2)
        @test isapprox(Q2, Q; atol=1e-12)
    end

    @testset "forward → inverse round-trip (4×4)" begin
        Q = [
            -1.0  0.4  0.3  0.3;
             0.2 -0.8  0.3  0.3;
             0.1  0.1 -0.5  0.3;
             0.0  0.2  0.3 -0.5
        ]
        p = ContinuousTransitionMatrix(Q)  # silently recomputes diagonal
        t = lograterows_forward(p.value)
        @test length(t) == 12  # 4*3
        Q2 = lograterows_inverse(t, 4)
        for i in 1:4
            @test isapprox(sum(Q2[i, :]), 0.0; atol=1e-12)
        end
        for i in 1:4, j in 1:4
            i == j && continue
            @test Q2[i, j] >= 0
        end
    end

    @testset "FixedEffects transform round-trip via build_fixed_effects" begin
        Q = _make_q3()
        fe = @fixedEffects begin
            Q = ContinuousTransitionMatrix(
                [-0.5  0.3  0.2;
                  0.1 -0.4  0.3;
                  0.2  0.1 -0.3])
        end
        θ0u = get_θ0_untransformed(fe)
        θ0t = get_θ0_transformed(fe)

        # Transformed vector has 6 elements.
        @test length(θ0t[:Q]) == 6

        # Round-trip: inverse(forward(θ0u)) ≈ θ0u
        inv_transform = get_inverse_transform(fe)
        θ0u_back = inv_transform(θ0t)
        @test isapprox(Matrix(θ0u_back[:Q]), Matrix(θ0u[:Q]); atol=1e-10)
    end
end

# ---------------------------------------------------------------------------
# 3. Flat names follow expected convention
# ---------------------------------------------------------------------------
@testset "ContinuousTransitionMatrix flat names" begin
    fe = @fixedEffects begin
        Q = ContinuousTransitionMatrix(
            [-0.5  0.3  0.2;
              0.1 -0.4  0.3;
              0.2  0.1 -0.3];
            calculate_se=true)
    end
    flat = get_flat_names(fe)
    @test length(flat) == 6
    # Names should be Q_1, Q_2, ..., Q_6
    for j in 1:6
        @test flat[j] == Symbol(:Q_, j)
    end
end

# ---------------------------------------------------------------------------
# 4. Bounds tests
# ---------------------------------------------------------------------------
@testset "ContinuousTransitionMatrix bounds" begin
    fe = @fixedEffects begin
        Q = ContinuousTransitionMatrix(
            [-0.5  0.3  0.2;
              0.1 -0.4  0.3;
              0.2  0.1 -0.3])
    end
    lb_t, ub_t = get_bounds_transformed(fe)
    @test all(lb_t[:Q] .== -Inf)
    @test all(ub_t[:Q] .== Inf)
end

# ---------------------------------------------------------------------------
# 5. ForwardDiff gradient correctness
# ---------------------------------------------------------------------------
@testset "ContinuousTransitionMatrix ForwardDiff gradient" begin

    Q0 = _make_q3()
    fe = @fixedEffects begin
        Q = ContinuousTransitionMatrix(
            [-0.5  0.3  0.2;
              0.1 -0.4  0.3;
              0.2  0.1 -0.3];
            calculate_se=true)
        σ = RealNumber(0.5; scale=:log, calculate_se=true)
    end
    inv_transform = get_inverse_transform(fe)
    θ0t = get_θ0_transformed(fe)
    axs = getaxes(θ0t)

    # A simple objective that uses both the Q matrix and σ.
    # tr(Q'Q) + σ² — has known analytic gradient.
    function obj(x::AbstractVector)
        θt = ComponentArray(x, axs)
        θu = inv_transform(θt)
        Q_val = collect(θu[:Q])   # n×n matrix
        σ_val = θu[:σ]
        return sum(Q_val .^ 2) + σ_val^2
    end

    x0 = collect(θ0t)
    g_fwd = ForwardDiff.gradient(obj, x0)
    @test length(g_fwd) == 7  # 6 (Q) + 1 (σ)
    # Gradient for Q entries must be non-zero (Q has positive off-diagonals).
    @test !all(iszero, g_fwd[1:6])
    # Gradient for σ must be non-zero.
    @test g_fwd[7] != 0.0

    # Compare with finite differences.
    g_fd = similar(x0)
    h = 1e-5
    for i in eachindex(x0)
        xp = copy(x0); xp[i] += h
        xm = copy(x0); xm[i] -= h
        g_fd[i] = (obj(xp) - obj(xm)) / (2h)
    end
    @test isapprox(g_fwd, g_fd; atol=1e-6)
end

# ---------------------------------------------------------------------------
# 6. get_collect_names includes ContinuousTransitionMatrix
# ---------------------------------------------------------------------------
@testset "get_collect_names includes ContinuousTransitionMatrix" begin
    fe = @fixedEffects begin
        a = RealNumber(1.0)
        Q = ContinuousTransitionMatrix(
            [-0.5  0.5; 0.2  -0.2])
    end
    cn = get_collect_names(fe)
    @test :Q in cn
    @test !(:a in cn)
end

# ---------------------------------------------------------------------------
# 7. UQ natural-scale coordinates
# ---------------------------------------------------------------------------
@testset "ContinuousTransitionMatrix UQ coords (natural scale)" begin
    using NoLimits: _coords_for_param

    Q = _make_q3()
    p = ContinuousTransitionMatrix(Q)
    fe = @fixedEffects begin
        Q = ContinuousTransitionMatrix(
            [-0.5  0.3  0.2;
              0.1 -0.4  0.3;
              0.2  0.1 -0.3];
            calculate_se=true)
    end
    θ0u = get_θ0_untransformed(fe)
    specs = get_transforms(fe).forward.specs
    spec = specs[1]

    coords_nat = _coords_for_param(Matrix(θ0u[:Q]), spec; natural=true)
    @test length(coords_nat) == 6
    # Should be off-diagonal elements in row-major order.
    Q_val = p.value
    expected = [Q_val[1,2], Q_val[1,3], Q_val[2,1], Q_val[2,3], Q_val[3,1], Q_val[3,2]]
    @test isapprox(coords_nat, expected; atol=1e-14)

    # Transformed coordinates are log of the same values.
    θ0t = get_θ0_transformed(fe)
    coords_trans = _coords_for_param(collect(θ0t[:Q]), spec; natural=false)
    @test isapprox(coords_trans, log.(expected); atol=1e-12)
end

# ---------------------------------------------------------------------------
# 8. Usage in a model formula (via @Model) — integration test
# ---------------------------------------------------------------------------
@testset "ContinuousTransitionMatrix in @Model formula" begin
    using DataFrames

    model = @Model begin
        @fixedEffects begin
            Q = ContinuousTransitionMatrix(
                [-0.5  0.3  0.2;
                  0.1 -0.4  0.3;
                  0.2  0.1 -0.3])
            σ = RealNumber(0.5; scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(Q[1, 2], σ)
        end
    end

    df = DataFrame(ID=[1, 1], t=[0.0, 1.0], y=[0.3, 0.3])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test dm !== nothing

    # Can evaluate the log-likelihood without errors.
    θu = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θu, ComponentArray())
    @test isfinite(ll)
end
