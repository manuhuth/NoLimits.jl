using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using Lux
using ForwardDiff
using SciMLBase
using DataInterpolations
using LinearAlgebra

import NoLimits: loglikelihood

@testset "loglikelihood non-ODE" begin
    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age]; constant_on=:ID)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            μ = softplus(a + x.Age + η)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 35.0, 35.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    η_list = [ComponentArray(η = 0.1), ComponentArray(η = -0.1)]

    ll1 = loglikelihood(dm, θ, η_list)
    ll2 = loglikelihood(dm, θ, η_list)
    @test isfinite(ll1)
    @test isfinite(ll2)
    @test ll1 == ll2
end

@testset "loglikelihood ODE" begin
    model = @Model begin
        @helpers begin
            sat(u) = u / (1 + abs(u))
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.4)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @preDifferentialEquation begin
            pre = sat(a + η)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1^2 + pre
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
        y = [1.0, 1.1]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η_list = [ComponentArray((η = 0.2,))]

    ll1 = loglikelihood(dm, θ, η_list)
    ll2 = loglikelihood(dm, θ, η_list)
    @test isfinite(ll1)
    @test isfinite(ll2)
    @test ll1 == ll2
end

@testset "loglikelihood threading (non-ODE)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(exp(a + η), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    η_list = [ComponentArray((η = 0.1,)), ComponentArray((η = -0.1,))]

    ll_serial = loglikelihood(dm, θ, η_list)
    ll_thread = loglikelihood(dm, θ, η_list; serialization=EnsembleThreads())
    ll_thread_cached = loglikelihood(dm, θ, η_list; serialization=EnsembleThreads(), cache=build_ll_cache(dm; nthreads=Threads.maxthreadid()))
    @test ll_serial == ll_thread
    @test ll_serial == ll_thread_cached
end

@testset "loglikelihood complex (NN/SoftTree/MvNormal/NPF)" begin
    chain = Chain(Dense(3, 4, tanh), Dense(4, 2))

    model = @Model begin
        @helpers begin
            sat(u) = u / (1 + abs(u))
        end

        @fixedEffects begin
            σ = RealNumber(0.4)
            ζ = NNParameters(chain; function_name=:NN1, calculate_se=false)
            Γ = SoftTreeParameters(3, 2; function_name=:ST1, calculate_se=false)
            ψ = NPFParameter(1, 3, seed=1, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI, :CRP]; constant_on=[:ID, :SITE])
        end

        @randomEffects begin
            η_mv = RandomEffect(MvNormal(zeros(2), LinearAlgebra.I); column=:ID)
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:SITE)
            η_nn = RandomEffect(LogNormal(NN1([x.Age, x.BMI, x.CRP], ζ)[1], 0.2); column=:ID)
            η_st = RandomEffect(Gumbel(ST1([x.Age, x.BMI, x.CRP], Γ)[1], 0.3); column=:SITE)
        end

        @formulas begin
            μ = sat(η_mv[1] + η_mv[2]^2 + η_flow[1] + η_nn + η_st)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 35.0, 35.0],
        BMI = [20.0, 20.0, 22.0, 22.0],
        CRP = [1.0, 1.0, 0.9, 0.9],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    η_list = [
        ComponentArray((η_mv = zeros(2), η_flow = 0.1, η_nn = 0.2, η_st = 0.3)),
        ComponentArray((η_mv = zeros(2), η_flow = 0.1, η_nn = 0.2, η_st = 0.3))
    ]

    ll = loglikelihood(dm, θ, η_list)
    @test isfinite(ll)
end

@testset "loglikelihood complex ODE (NN/SoftTree/Spline, multi-RE)" begin
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=6))

    model = @Model begin
        @helpers begin
            sat(u) = u / (1 + abs(u))
        end

        @fixedEffects begin
            a = RealNumber(0.15)
            σ = RealNumber(0.35)
            ζ = NNParameters(chain; function_name=:NN1, calculate_se=false)
            Γ = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
            sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=[:ID, :SITE])
            w = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @preDifferentialEquation begin
            pre = sat(NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1]) + SP1(x.Age / 100, sp) + η_id
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1^2 + pre + w(t) + η_site
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(log1p(x1(t)^2), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        SITE = [:A, :A, :A, :B, :B, :B],
        t = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        Age = [30.0, 30.0, 30.0, 35.0, 35.0, 35.0],
        BMI = [20.0, 20.0, 20.0, 22.0, 22.0, 22.0],
        w = [0.2, 0.35, 0.5, 0.1, 0.25, 0.4],
        y = [1.0, 1.05, 1.1, 0.9, 0.95, 1.0]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η_list = [
        ComponentArray((η_id = 0.1, η_site = -0.1)),
        ComponentArray((η_id = -0.1, η_site = 0.2))
    ]

    ll = loglikelihood(dm, θ, η_list)
    @test isfinite(ll)
end

@testset "loglikelihood ForwardDiff (fixed effects)" begin
    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(softplus(a + η), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    η_list = [ComponentArray((η = 0.1,))]

    g = ForwardDiff.gradient(x -> loglikelihood(dm, x, η_list), θ)
    @test g isa ComponentArray
    @test all(isfinite, collect(g))
end

@testset "loglikelihood ForwardDiff (random effects)" begin
    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(softplus(a + η), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    η0 = ComponentArray((η = 0.1,))

    g = ForwardDiff.gradient(η -> loglikelihood(dm, θ, [η]), η0)
    @test g isa ComponentArray
    @test all(isfinite, collect(g))
end

@testset "loglikelihood ODE ForwardDiff (fixed effects)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
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
        y = [1.0, 0.9]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η_list = [ComponentArray((η = 0.1,))]

    g = ForwardDiff.gradient(x -> loglikelihood(dm, x, η_list), θ)
    @test g isa ComponentArray
    @test all(isfinite, collect(g))
end

@testset "loglikelihood ODE ForwardDiff (random effects)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
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
        y = [1.0, 0.9]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η0 = ComponentArray((η = 0.1,))

    g = ForwardDiff.gradient(η -> loglikelihood(dm, θ, [η]), η0)
    @test g isa ComponentArray
    @test all(isfinite, collect(g))
end
