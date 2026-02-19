using Test
using NoLimits
using DataFrames
using Distributions
using LinearAlgebra

@testset "MAP non-ODE" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1; prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5; prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0.1, 0.12, 0.11]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP())

    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MAP ODE" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2; prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3; prior=LogNormal(0.0, 0.5))
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1^2
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(log1p(x1(t)^2), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.05]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP())

    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
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

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.MAP())
end

@testset "MAP respects bounds (σ lower bound)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1; prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5; lower=0.3, scale=:identity, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0.1, 0.12, 0.11]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP())

    θu = NoLimits.get_params(res; scale=:untransformed)
    @test θu.σ >= 0.3
end

@testset "MAP constants" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1; prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5; prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0.1, 0.12, 0.11]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP(); constants=(a=0.0,))

    θu = NoLimits.get_params(res; scale=:untransformed)
    @test θu.a == 0.0
end

@testset "MAP requires a free fixed effect" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
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

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.MAP(); constants=(a=0.2, σ=0.3))
end

@testset "MAP fixed vector parameters" begin
    model = @Model begin
        @fixedEffects begin
            β = RealVector([0.2, -0.1], prior=MvNormal(zeros(2), LinearAlgebra.I))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            μ = exp(β[1] * z + β[2] * z^2)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        z = [0.1, 0.2, 0.15, 0.3],
        y = [1.0, 1.05, 1.02, 1.08]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP())

    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MAP non-normal Bernoulli outcome" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            b = RealNumber(0.2, prior=Normal(0.0, 1.0))
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

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP())

    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
    θu = NoLimits.get_params(res; scale=:untransformed)
    @test isfinite(θu.a)
    @test isfinite(θu.b)
end

@testset "MAP handles +Inf objective in AD path" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0, prior=Uniform(0.1, 1.0))
        end

        @formulas begin
            y ~ Poisson(a)
        end
    end

    df = DataFrame(
        ID = [1],
        t = [0.0],
        y = [1.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP(; optim_kwargs=(maxiters=3,)))

    @test res isa FitResult
    @test !isfinite(NoLimits.get_objective(res))
end
