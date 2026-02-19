using Test
using NoLimits
using DataFrames
using Distributions
using DataInterpolations

@testset "Splines" begin
    knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
    degree = 2
    coeffs = [1.0, 2.0, 3.0, 4.0]

    b = bspline_basis(0.25, knots, degree)
    @test length(b) == length(coeffs)
    @test isapprox(sum(b), 1.0; rtol=1e-8, atol=1e-10)

    y = bspline_eval(0.25, coeffs, knots, degree)
    @test y isa Number

    y_vec = bspline_eval([0.25], coeffs, knots, degree)
    @test y_vec == y

    @test_throws ErrorException bspline_eval(-0.1, coeffs, knots, degree)
    @test_throws ErrorException bspline_eval(1.1, coeffs, knots, degree)

    knots_bad = [0.0, 0.5, 0.4, 1.0]
    @test_throws ErrorException bspline_basis(0.5, knots_bad, 1)
end

@testset "DynamicCovariate interpolation validation" begin
    @test_throws ErrorException DynamicCovariate(:w; interpolation=BSplineInterpolation)
    @test_throws ErrorException DynamicCovariate(:w; interpolation=BSplineApprox)
    @test_throws ErrorException DynamicCovariate(:w; interpolation=CubicHermiteSpline)
    @test_throws ErrorException DynamicCovariate(:w; interpolation=QuinticHermiteSpline)
end

@testset "DynamicCovariate interpolation min obs" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
            w = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df1 = DataFrame(ID=[1], t=[0.0], w=[1.0], y=[1.0])
    @test_throws ErrorException DataModel(model, df1; primary_id=:ID, time_col=:t)

    model2 = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
            w = DynamicCovariate(; interpolation=QuadraticSpline)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df2 = DataFrame(ID=[1, 1], t=[0.0, 1.0], w=[1.0, 1.2], y=[1.0, 1.1])
    @test_throws ErrorException DataModel(model2, df2; primary_id=:ID, time_col=:t)
end

@testset "DynamicCovariate interpolation min obs (per type)" begin
    function _model_with_interp(itp)
        return @Model begin
            @fixedEffects begin
                a = RealNumber(0.2)
                σ = RealNumber(0.3)
            end

            @covariates begin
                t = Covariate()
                w = DynamicCovariate(; interpolation=itp)
            end

            @randomEffects begin
                η = RandomEffect(Normal(0.0, 1.0); column=:ID)
            end

            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
    end

    df_one = DataFrame(ID=[1], t=[0.0], w=[1.0], y=[1.0])
    df_two = DataFrame(ID=[1, 1], t=[0.0, 1.0], w=[1.0, 1.2], y=[1.0, 1.1])

    for itp in (SmoothedConstantInterpolation, LinearInterpolation, LagrangeInterpolation, AkimaInterpolation)
        model_itp = _model_with_interp(itp)
        @test_throws ErrorException DataModel(model_itp, df_one; primary_id=:ID, time_col=:t)
    end

    for itp in (QuadraticInterpolation, QuadraticSpline, CubicSpline)
        model_itp = _model_with_interp(itp)
        @test_throws ErrorException DataModel(model_itp, df_two; primary_id=:ID, time_col=:t)
    end
end

@testset "DynamicCovariate interpolation requires sorted time" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
            w = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df_bad = DataFrame(ID=[1, 1], t=[1.0, 0.0], w=[1.0, 1.2], y=[1.0, 1.1])
    @test_throws ErrorException DataModel(model, df_bad; primary_id=:ID, time_col=:t)
end

@testset "DynamicCovariate LagrangeInterpolation DataModel" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
            w = DynamicCovariate(; interpolation=LagrangeInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df_ok = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        w = [1.0, 1.2],
        y = [1.0, 1.1]
    )

    dm = DataModel(model, df_ok; primary_id=:ID, time_col=:t)
    @test length(get_individuals(dm)) == 1
end

@testset "DynamicCovariate LagrangeInterpolation DataModel (insufficient obs)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
            w = DynamicCovariate(; interpolation=LagrangeInterpolation)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df_bad = DataFrame(
        ID = [1],
        t = [0.0],
        w = [1.0],
        y = [1.0]
    )

    @test_throws ErrorException DataModel(model, df_bad; primary_id=:ID, time_col=:t)
end
