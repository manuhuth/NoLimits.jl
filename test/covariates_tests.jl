using Test
using NoLimits
using DataInterpolations
using ComponentArrays
using Distributions
using ForwardDiff

@testset "Covariates macro" begin
    # Basic macro expansion builds names, groups, and interpolations.
    cov = @covariates begin
        x = ConstantCovariateVector([:Age, :gender])
        w1 = DynamicCovariate(interpolation = CubicSpline)
        w2 = ConstantCovariate()
        z = DynamicCovariateVector([:z1, :z2, :z3],
            interpolations = [
                LinearInterpolation, LinearInterpolation, LinearInterpolation])
        delta_t = Covariate()
    end

    @test :x in cov.names
    @test :w1 in cov.names
    @test :w2 in cov.names
    @test :z in cov.names
    @test :delta_t in cov.names

    @test :x in cov.constants
    @test :w2 in cov.constants
    @test :w1 in cov.varying
    @test :z in cov.varying
    @test :delta_t in cov.varying

    @test :w1 in cov.dynamic
    @test :z in cov.dynamic
    @test !(:delta_t in cov.dynamic)

    @test length(cov.flat_names) == 1 + 1 + 2 + 3 + 1
    @test cov.interpolations[:w1] == CubicSpline
    @test length(cov.interpolations[:z]) == 3
    @test !haskey(cov.interpolations, :delta_t)
end

@testset "Covariates interpolation types" begin
    # All allowed interpolation types are accepted and recorded.
    cov = @covariates begin
        c = DynamicCovariate(interpolation = ConstantInterpolation)
        sc = DynamicCovariate(interpolation = SmoothedConstantInterpolation)
        lin = DynamicCovariate(interpolation = LinearInterpolation)
        quad = DynamicCovariate(interpolation = QuadraticInterpolation)
        lag = DynamicCovariate(interpolation = LagrangeInterpolation)
        qs = DynamicCovariate(interpolation = QuadraticSpline)
        cs = DynamicCovariate(interpolation = CubicSpline)
        ak = DynamicCovariate(interpolation = AkimaInterpolation)
    end

    @test cov.interpolations[:c] == ConstantInterpolation
    @test cov.interpolations[:sc] == SmoothedConstantInterpolation
    @test cov.interpolations[:lin] == LinearInterpolation
    @test cov.interpolations[:quad] == QuadraticInterpolation
    @test cov.interpolations[:lag] == LagrangeInterpolation
    @test cov.interpolations[:qs] == QuadraticSpline
    @test cov.interpolations[:cs] == CubicSpline
    @test cov.interpolations[:ak] == AkimaInterpolation
end

@testset "Covariates interpolation types (invalid)" begin
    @test_throws ErrorException DynamicCovariate(:a, interpolation = BSplineInterpolation)
    @test_throws ErrorException DynamicCovariate(:a, interpolation = BSplineApprox)
    @test_throws ErrorException DynamicCovariate(:a, interpolation = CubicHermiteSpline)
    @test_throws ErrorException DynamicCovariate(:a, interpolation = QuinticHermiteSpline)
end

@testset "Covariates validation" begin
    # Invalid inputs and shapes are rejected with clear errors.
    @test_throws ErrorException DynamicCovariateVector(
        [:a, :b]; interpolations = [LinearInterpolation])

    @test_throws ErrorException DynamicCovariate(:x, interpolation = Symbol)

    @test_throws LoadError @eval @covariates begin
        x = Covariate(:x)
    end

    @test_throws LoadError @eval @covariates begin
        x = Covariate(column = :x)
    end

    @test_throws LoadError @eval @covariates begin
        x = ConstantCovariate(:x)
    end

    cov = @covariates begin
        x = ConstantCovariateVector([:a, :b])
        y = ConstantCovariate()
    end
    @test :x in cov.constants
    @test :y in cov.constants

    @test_throws ErrorException @eval @covariates begin
        x = ConstantCovariateVector([:a, :b])
        y = 1 + 2
    end

    @test_throws ErrorException @eval @covariates begin
        x = DynamicCovariateVector([:a, :b], interpolations = [LinearInterpolation])
    end

    cov2 = @covariates begin
        w = Covariate()
    end
    @test !haskey(cov2.interpolations, :w)

    cov3 = @covariates begin
        z = CovariateVector([:a, :b, :c])
    end
    @test !haskey(cov3.interpolations, :z)
end

@testset "Covariates macro rewriting" begin
    # LHS-driven column naming and qualified constructors rewrite correctly.
    cov = @covariates begin
        w = DynamicCovariate(interpolation = CubicSpline)
        c = ConstantCovariate()
    end
    @test cov.interpolations[:w] == CubicSpline
    @test :c in cov.constants

    covq = @covariates begin
        q = DynamicCovariate(interpolation = AkimaInterpolation)
        s = ConstantCovariate()
    end
    @test covq.interpolations[:q] == AkimaInterpolation
    @test :s in covq.constants

    @test_throws LoadError @eval @covariates begin
        bad = ConstantCovariate(:a, interpolation = CubicSpline)
    end

    covn = @covariates begin
        v = Covariate()
        u = ConstantCovariate()
    end
    @test :v in covn.flat_names
    @test :u in covn.flat_names
end

@testset "Covariates keyword parsing (Dynamic)" begin
    cov = @covariates begin
        w1 = DynamicCovariate(; interpolation = LinearInterpolation)
    end
    @test cov.interpolations[:w1] == LinearInterpolation
end

@testset "Covariates constant_on parsing" begin
    cov = @covariates begin
        c = ConstantCovariate(; constant_on = :ID)
        v = ConstantCovariateVector([:Age, :Weight]; constant_on = [:ID, :SITE])
    end

    @test cov.params.c.constant_on == [:ID]
    @test cov.params.v.constant_on == [:ID, :SITE]
end

@testset "Covariate vectors usable as numeric vectors" begin
    # Numeric covariate vectors materialise as ComponentArrays: both named-field access
    # (x.Age) and numeric-vector operations (x' * β) work, in column order.
    cv = NoLimits._covariate_vector((Age = 30.0, Weight = 70.0))
    @test cv isa ComponentArray
    @test cv.Age == 30.0 && cv.Weight == 70.0
    @test cv' * [0.5, -0.2] ≈ 30.0 * 0.5 + 70.0 * (-0.2)
    @test collect(cv) == [30.0, 70.0]
    @test eltype(NoLimits._covariate_vector((a = 1.0f0, b = 2.0f0))) == Float32

    # Mixed/categorical covariate vectors keep field access only (stay NamedTuples).
    cat = NoLimits._covariate_vector((grp = "A", Age = 5.0))
    @test cat isa NamedTuple && cat.Age == 5.0

    # The per-row time-varying extraction also yields a ComponentArray.
    row = NoLimits._vary_value_at((z1 = [1.0, 2.0], z2 = [3.0, 4.0]), 2)
    @test row isa ComponentArray && collect(row) == [2.0, 4.0]

    # End-to-end through generated formula code, including ForwardDiff (AD) over the
    # fixed effects: ∂(x'β)/∂β = x and ∂(z'γ)/∂γ = z.
    formulas = @formulas begin
        lin = x' * β + z' * γ
        obs ~ Normal(lin, σ)
    end
    (_, form_obs, _, _) = get_formulas_builders(formulas;
        fixed_names = [:β, :γ, :σ], const_cov_names = [:x], varying_cov_names = [:z])
    const_cov = (x = NoLimits._covariate_vector((Age = 30.0, Weight = 70.0)),)
    vary = (t = 0.0, z = NoLimits._covariate_vector((z1 = 2.0, z2 = 5.0)))
    obj = function (p)
        ctx = (; fixed_effects = (β = p[1:2], γ = p[3:4], σ = 0.4),
            random_effects = NamedTuple(), prede = NamedTuple(),
            helpers = NamedTuple(), model_funs = NamedTuple())
        return mean(form_obs(ctx, NamedTuple(), const_cov, vary).obs)
    end
    p0 = [0.5, -0.2, 1.0, 0.3]
    @test obj(p0) ≈ (30.0 * 0.5 + 70.0 * (-0.2)) + (2.0 * 1.0 + 5.0 * 0.3)
    g = ForwardDiff.gradient(obj, p0)
    @test g[1:2] ≈ [30.0, 70.0]
    @test g[3:4] ≈ [2.0, 5.0]
end
