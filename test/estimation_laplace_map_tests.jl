using Test
using DataFrames
using Distributions
using LinearAlgebra
using NoLimits

@testset "LaplaceMAP requires priors on all fixed effects" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=2,)))
end

@testset "LaplaceMAP runs with priors and penalties" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=3,));
                    penalty=(; a=0.1));
    @test res.summary.converged isa Bool
end

@testset "LaplaceMAP with multivariate REs and multiple groups" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.5))
            μ = RealVector([0.0, 0.0], prior=MvNormal([0.0, 0.0], LinearAlgebra.I(2)))
        end

        @randomEffects begin
            η_id = RandomEffect(MvNormal([0.0, 0.0], LinearAlgebra.I(2)); column=:ID)
            η_site = RandomEffect(MvNormal(μ, LinearAlgebra.I(2)); column=:SITE)
        end

        @formulas begin
            y ~ Normal(a + η_id[1] + η_site[2], σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        SITE = [:A, :A, :A, :A, :B, :B, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=3,)));
    @test res.summary.converged isa Bool
end

@testset "LaplaceMAP with ODE model" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.3, prior=Normal(0.3, 0.3))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + η
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [0.9, 0.7]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=2,)));
    @test res.summary.converged isa Bool
end

@testset "LaplaceMAP normal prior equals penalty" begin
    model_prior = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    model_penalty = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm_prior = DataModel(model_prior, df; primary_id=:ID, time_col=:t)
    dm_pen = DataModel(model_penalty, df; primary_id=:ID, time_col=:t)

    res_prior = fit_model(dm_prior, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=3,)), constants=(; σ=0.5))
    res_pen = fit_model(dm_pen, NoLimits.Laplace(; optim_kwargs=(maxiters=3,)), penalty=(; a=0.5), constants=(; σ=0.5))

    θ_prior = NoLimits.get_params(res_prior; scale=:untransformed)
    θ_pen = NoLimits.get_params(res_pen; scale=:untransformed)
    @test isapprox(θ_prior.a, θ_pen.a; rtol=1e-3, atol=1e-3)
end

@testset "LaplaceMAP non-normal Bernoulli outcome" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            b = RealNumber(0.2, prior=Normal(0.0, 1.0))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.8); column=:ID)
        end

        @formulas begin
            p = logistic(a + b * z + η)
            y ~ Bernoulli(p)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, -0.2, 0.1, 0.4, 0.7],
        y = [0, 1, 0, 0, 1, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=2,)))

    @test res isa FitResult
    @test res.summary.converged isa Bool
    @test isfinite(NoLimits.get_objective(res))
end
