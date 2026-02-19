using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using OptimizationBBO
using OptimizationOptimisers
using OptimizationOptimJL

@testset "MLE non-ODE" begin
    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5, scale=:log)
        end

        @formulas begin
            y ~ Normal(softplus(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
    @test NoLimits.get_params(res; scale=:untransformed) isa ComponentArray
end

@testset "MLE ODE" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
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
    res = fit_model(dm, NoLimits.MLE())

    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MLE ODE with parameterized initial state" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            ka = RealNumber(1.0, prior=Normal(1.0, 0.5))
            ke = RealNumber(0.1, prior=Normal(0.1, 0.05))
            V  = RealNumber(20.0, prior=Normal(20.0, 5.0))
            D  = RealNumber(320.0, prior=Normal(320.0, 50.0))
            σ  = RealNumber(1.0, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @DifferentialEquation begin
            D(A) ~ -ka * A
            D(C) ~ (ka * A) / V - ke * C
        end

        @initialDE begin
            A = D
            C = 0.0
        end

        @formulas begin
            y ~ Normal(C(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        y = [1.0, 1.05, 0.98, 1.02]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MLE rejects random effects" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(exp(a + η), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.MLE())
end

@testset "MLE requires a free fixed effect" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3, scale=:log)
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
    @test_throws ErrorException fit_model(dm, NoLimits.MLE(); constants=(a=0.2, σ=0.3))
end

@testset "MLE fixed vector parameters" begin
    model = @Model begin
        @fixedEffects begin
            β = RealVector([0.2, -0.1])
            σ = RealNumber(0.3, scale=:log)
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
    res = fit_model(dm, NoLimits.MLE());

    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MLE respects bounds (σ lower bound)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5; lower=0.3, scale=:identity)
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
    res = fit_model(dm, NoLimits.MLE())

    θu = NoLimits.get_params(res; scale=:untransformed)
    @test θu.σ >= 0.3
end

@testset "MLE constants" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5, scale=:log)
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
    res = fit_model(dm, NoLimits.MLE(); constants=(a=0.0,))

    θu = NoLimits.get_params(res; scale=:untransformed)
    @test θu.a == 0.0
end

@testset "MLE penalties" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5, scale=:log)
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
    res_no_penalty = fit_model(dm, NoLimits.MLE())
    res = fit_model(dm, NoLimits.MLE(); penalty=(a=100.0,))

    θu0 = NoLimits.get_params(res_no_penalty; scale=:untransformed)
    θu = NoLimits.get_params(res; scale=:untransformed)
    @test abs(θu.a) ≤ abs(θu0.a)
end

@testset "MLE penalty mimics Normal prior" begin
    model_prior = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1; prior=Normal(0.0, 1.0))
        end

        @formulas begin
            y ~ Normal(exp(a), 2.0)
        end
    end

    model_penalty = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            #σ = RealNumber(0.5, scale=:log)
        end

        @formulas begin
            y ~ Normal(exp(a), 2.0)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0.1, 0.12, 0.11]
    )

    dm_prior = DataModel(model_prior, df; primary_id=:ID, time_col=:t)
    dm_penalty = DataModel(model_penalty, df; primary_id=:ID, time_col=:t)
    res_map = fit_model(dm_prior, NoLimits.MAP())
    res_pen = fit_model(dm_penalty, NoLimits.MLE(); penalty=(a=0.5,))

    a_map = NoLimits.get_params(res_map; scale=:untransformed).a
    a_pen = NoLimits.get_params(res_pen; scale=:untransformed).a
    @test isapprox(a_map, a_pen; rtol=1e-4, atol=1e-4)
end

@testset "MLE uses optim_kwargs" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5, scale=:log)
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
    method = NoLimits.MLE(optim_kwargs=(; iterations=1))
    res = fit_model(dm, method);

    @test res isa FitResult
    stats = res.result.solution.stats
    @test hasproperty(stats, :iterations)
    @test stats.iterations <= 1
end

function _mle_dm_basic()
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5, scale=:log)
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
    return dm
end

@testset "MLE accepts lb-only user bounds" begin
    dm = _mle_dm_basic()
    lb = ComponentArray((; a=-2.0, σ=-3.0))
    res = fit_model(dm, NoLimits.MLE(lb=lb))
    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MLE accepts ub-only user bounds" begin
    dm = _mle_dm_basic()
    ub = ComponentArray((; a=2.0, σ=2.0))
    res = fit_model(dm, NoLimits.MLE(ub=ub))
    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MLE BBO requires finite bounds on both sides" begin
    dm = _mle_dm_basic()
    lb = ComponentArray((; a=-2.0, σ=-3.0))
    method = NoLimits.MLE(optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
                                  lb=lb, optim_kwargs=(; iterations=5))
    err = try
        fit_model(dm, method)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("finite lower and upper bounds", sprint(showerror, err))
end

@testset "MLE optimizer BFGS (Optim)" begin
    dm = _mle_dm_basic()
    method = NoLimits.MLE(optimizer=BFGS(), optim_kwargs=(; ))
    res = fit_model(dm, method)
    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MLE optimizer NelderMead (Optim)" begin
    dm = _mle_dm_basic()
    method = NoLimits.MLE(optimizer=Optim.NelderMead(), optim_kwargs=(; ))
    res = fit_model(dm, method)
    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MLE optimizer Adam (OptimizationOptimisers)" begin
    dm = _mle_dm_basic()
    method = NoLimits.MLE(optimizer=OptimizationOptimisers.Adam(0.05), optim_kwargs=(; maxiters=5))
    res = fit_model(dm, method)
    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MLE optimizer BlackBoxOptim (OptimizationBBO)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1;  lower=-2.0, upper=2.0, scale=:identity)
            σ = RealNumber(0.5;  lower=0.1, upper=2.0, scale=:identity)
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
    method = NoLimits.MLE(optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(), optim_kwargs=(; iterations=5))
    res = fit_model(dm, method)
    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MLE non-normal Poisson outcome" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
        end

        @formulas begin
            λ = exp(a + b * z)
            y ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        z = [0.0, 0.5, 1.0, 1.5],
        y = [1, 1, 2, 3]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    @test res isa FitResult
    @test isfinite(NoLimits.get_objective(res))
    θu = NoLimits.get_params(res; scale=:untransformed)
    @test isfinite(θu.a)
    @test isfinite(θu.b)
end

@testset "MLE handles +Inf objective in AD path" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
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
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=3,)))

    @test res isa FitResult
    @test !isfinite(NoLimits.get_objective(res))
end
