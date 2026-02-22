using Test
using NoLimits
using ComponentArrays

# ─── RealNumber with :logit ───────────────────────────────────────────────────

@testset "RealNumber :logit — construction" begin
    # Valid values
    p = RealNumber(0.5; scale=:logit)
    @test p.value ≈ 0.5
    @test p.scale === :logit

    p2 = RealNumber(0.01; scale=:logit)
    @test p2.value ≈ 0.01

    p3 = RealNumber(0.99; scale=:logit)
    @test p3.value ≈ 0.99
end

@testset "RealNumber :logit — constructor errors" begin
    # Boundary values: 0 and 1 are invalid (open interval)
    @test_throws ErrorException RealNumber(0.0; scale=:logit)
    @test_throws ErrorException RealNumber(1.0; scale=:logit)
    # Out-of-range values
    @test_throws ErrorException RealNumber(-0.1; scale=:logit)
    @test_throws ErrorException RealNumber(1.1; scale=:logit)
    @test_throws ErrorException RealNumber(2.0; scale=:logit)
end

# ─── RealVector with :logit ───────────────────────────────────────────────────

@testset "RealVector :logit — uniform scale" begin
    v = RealVector([0.2, 0.5, 0.8]; scale=[:logit, :logit, :logit])
    @test v.scale == [:logit, :logit, :logit]
    @test all(v.value .≈ [0.2, 0.5, 0.8])
end

@testset "RealVector :logit — constructor errors" begin
    # Invalid value for a logit-scaled element
    @test_throws ErrorException RealVector([0.0, 0.5]; scale=[:logit, :logit])
    @test_throws ErrorException RealVector([0.5, 1.0]; scale=[:logit, :logit])
    @test_throws ErrorException RealVector([1.5, 0.5]; scale=[:logit, :logit])
end

@testset "RealVector :logit — mixed scales (elementwise)" begin
    # Mixed [:logit, :log, :identity] — logit element must be in (0,1),
    # log element must be positive
    v = RealVector([0.3, 2.0, -1.0]; scale=[:logit, :log, :identity])
    @test v.scale == [:logit, :log, :identity]
    @test v.value ≈ [0.3, 2.0, -1.0]

    # Invalid logit element in mixed vector
    @test_throws ErrorException RealVector([0.0, 2.0, -1.0]; scale=[:logit, :log, :identity])
    @test_throws ErrorException RealVector([1.0, 2.0, -1.0]; scale=[:logit, :log, :identity])
end

# ─── _param_spec dispatch ─────────────────────────────────────────────────────

@testset "_param_spec for :logit RealNumber" begin
    # Access via @fixedEffects
    fe = @fixedEffects begin
        p = RealNumber(0.3; scale=:logit)
    end
    θ0 = get_θ0_transformed(fe)
    @test isapprox(θ0.p, logit_forward(0.3); rtol=1e-10)

    θ_back = get_θ0_untransformed(fe)
    @test isapprox(θ_back.p, 0.3; rtol=1e-10)
end

@testset "_param_spec for uniform :logit RealVector" begin
    fe = @fixedEffects begin
        v = RealVector([0.2, 0.5, 0.8]; scale=[:logit, :logit, :logit])
    end
    θ0 = get_θ0_transformed(fe)
    @test all(isapprox.(θ0.v, logit_forward.([0.2, 0.5, 0.8]); rtol=1e-10))

    θ_back = get_θ0_untransformed(fe)
    @test isapprox(θ_back.v, [0.2, 0.5, 0.8]; rtol=1e-8, atol=1e-10)
end

@testset "_param_spec for mixed :elementwise RealVector" begin
    fe = @fixedEffects begin
        v = RealVector([0.4, 2.0, -1.5]; scale=[:logit, :log, :identity])
    end
    θ0 = get_θ0_transformed(fe)
    @test isapprox(θ0.v[1], logit_forward(0.4); rtol=1e-10)
    @test isapprox(θ0.v[2], log(2.0); rtol=1e-10)
    @test isapprox(θ0.v[3], -1.5; rtol=1e-10)

    θ_back = get_θ0_untransformed(fe)
    @test isapprox(θ_back.v[1], 0.4; rtol=1e-8, atol=1e-10)
    @test isapprox(θ_back.v[2], 2.0; rtol=1e-8, atol=1e-10)
    @test isapprox(θ_back.v[3], -1.5; rtol=1e-10)
end

# ─── _transform_bounds ────────────────────────────────────────────────────────

@testset "_transform_bounds :logit scalar — transformed bounds are (-Inf, Inf)" begin
    fe = @fixedEffects begin
        p = RealNumber(0.3; scale=:logit)
    end
    # The forward transform maps 0.3 → logit_forward(0.3) (finite).
    θ0t = get_θ0_transformed(fe)
    @test isfinite(θ0t.p)

    # Transformed bounds should be (-Inf, Inf).
    lb, ub = get_bounds_transformed(fe)
    @test lb.p == -Inf
    @test ub.p == Inf
end

@testset "_transform_bounds :logit vector — transformed bounds are (-Inf, Inf)" begin
    fe = @fixedEffects begin
        v = RealVector([0.2, 0.5, 0.8]; scale=[:logit, :logit, :logit])
    end
    lb, ub = get_bounds_transformed(fe)
    @test all(lb.v .== -Inf)
    @test all(ub.v .== Inf)
end

@testset "_transform_bounds :elementwise — per-element bounds" begin
    # [:logit, :log, :identity]
    # logit → (-Inf, Inf), log → (log(EPSILON), Inf), identity → as-is
    fe = @fixedEffects begin
        v = RealVector([0.4, 2.0, -5.0]; scale=[:logit, :log, :identity])
    end
    lb, ub = get_bounds_transformed(fe)
    # logit element: unbounded
    @test lb.v[1] == -Inf
    @test ub.v[1] == Inf
    # log element: lower = log(EPSILON) = -Inf (EPSILON=0), upper = Inf
    @test lb.v[2] == log(NoLimits.EPSILON)
    @test ub.v[2] == Inf
    # identity element: as-is (-Inf, Inf defaults)
    @test lb.v[3] == -Inf
    @test ub.v[3] == Inf
end

@testset "_transform_bounds :log scalar — unchanged from before" begin
    fe = @fixedEffects begin
        σ = RealNumber(1.0; scale=:log)
    end
    lb, ub = get_bounds_transformed(fe)
    @test lb.σ == log(NoLimits.EPSILON)
    @test ub.σ == Inf
end
