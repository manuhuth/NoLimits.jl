using Test
using NoLimits
using DataFrames
using Distributions
using LinearAlgebra

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

    # A prior only on a `constants=` parameter is an additive offset, so MAP would
    # silently degenerate to MLE (#166).
    model_c = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2; prior = Normal(0.0, 1.0))
            b = RealNumber(0.2)
        end

        @formulas begin
            y ~ Normal(a + b, 0.3)
        end
    end
    dm_c = DataModel(model_c, df; primary_id = :ID, time_col = :t)
    @test_throws ErrorException fit_model(dm_c, NoLimits.MAP(); constants = (a = 0.3,))
    @test fit_model(dm_c, NoLimits.MAP()) isa FitResult
end

@testset "MAP with a matrix-variate prior on RealDiagonalMatrix" begin
    # The diagonal is stored as a vector; Wishart's logpdf needs the n x n form (#168).
    model = @Model begin
        @fixedEffects begin
            Σ = RealDiagonalMatrix(
                [1.0, 1.0];
                prior = Wishart(4.0, Matrix{Float64}(I, 2, 2))
            )
            a = RealNumber(1.0; prior = Normal(0.0, 1.0))
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a, 0.5)
        end
    end

    df = DataFrame(ID = [1, 1], t = [0.0, 1.0], y = [1.0, 1.05])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MAP(; optim_kwargs = (; maxiters = 5)))
    @test isfinite(NoLimits.get_objective(res))
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
