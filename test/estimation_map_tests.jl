using Test
using NoLimits
using DataFrames
using Distributions

# Testsets shared with MLE (ODE, bounds, constants, free-fixed-effect, vector
# parameters, +Inf objective) live in estimation_mle_tests.jl as MLE/MAP loops.

@testset "MAP non-ODE" begin
    @test fx_map() isa FitResult         # shared no-RE MAP fit
end

@testset "MAP requires priors" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.05]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test_throws ErrorException fit_model(dm, NoLimits.MAP())
end

@testset "MAP non-normal Bernoulli outcome" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior = Normal(0.0, 1.0))
            b = RealNumber(0.2, prior = Normal(0.0, 1.0))
        end

        @formulas begin
            p = logistic(a + b * z)
            y ~ Bernoulli(p)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, -0.1, 0.0, 0.3, 0.4],
        y = [0, 1, 0, 0, 1, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MAP())

    @test res isa FitResult
    θu = NoLimits.get_params(res; scale = :untransformed)
end
