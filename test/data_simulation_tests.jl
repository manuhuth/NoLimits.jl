using Test
using NoLimits
using DataFrames
using Distributions
using Random
using Lux
using SciMLBase

@testset "simulate_data basic" begin
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
            η = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @formulas begin
            μ = softplus(a + x.Age + η)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 40.0, 40.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    sim = simulate_data(dm; rng=MersenneTwister(1))

    @test nrow(sim) == nrow(df)
    @test :η in propertynames(sim)
    @test length(unique(sim.η[sim.SITE .== :A])) == 1
    @test length(unique(sim.η[sim.SITE .== :B])) == 1
    @test any(sim.y .!= df.y)
end



@testset "simulate_data does not simulate events" begin
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
        ID = [1, 1, 1],
        t = [0.0, 0.5, 1.0],
        EVID = [1, 0, 0],
        AMT = [100.0, 0.0, 0.0],
        RATE = [0.0, 0.0, 0.0],
        CMT = [1, 1, 1],
        y = [0.2, 1.1, 1.2]
    )

    dm = DataModel(model, df;
                   primary_id=:ID,
                   time_col=:t,
                   evid_col=:EVID,
                   amt_col=:AMT,
                   rate_col=:RATE,
                   cmt_col=:CMT)
    sim = simulate_data(dm; rng=MersenneTwister(3))

    @test nrow(sim) == nrow(df)
    @test :η in propertynames(sim)
    @test length(unique(sim.η[sim.ID .== 1])) == 1
    @test any(sim.y .!= df.y)
    @test all(sim.EVID .== df.EVID)
    @test all(sim.AMT .== df.AMT)
    @test all(sim.RATE .== df.RATE)
    @test all(sim.CMT .== df.CMT)
end

@testset "simulate_data_model builds DataModel" begin
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
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    dm_sim = simulate_data_model(dm; rng=MersenneTwister(4))

    @test dm_sim isa DataModel
    @test length(get_individuals(dm_sim)) == length(get_individuals(dm))
end

@testset "simulate_data complex model (NN/SoftTree/NPF, multi-RE, helpers)" begin
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))

    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
            sat(u) = u / (1 + abs(u))
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.4)
            ζ = NNParameters(chain; function_name=:NN1, calculate_se=false)
            Γ = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
            ψ = NPFParameter(1, 3, seed=1, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=[:SITE, :ID])
            z = Covariate()
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:SITE)
            η_nn = RandomEffect(LogNormal(NN1([x.Age, x.BMI], ζ)[1], 0.2); column=:ID)
            η_st = RandomEffect(Gumbel(ST1([x.Age, x.BMI], Γ)[1], 0.3); column=:SITE)
        end

        @formulas begin
            μ = η_flow[1] + η_st + + sat(η_nn) + η_flow[1] + η_st + softplus(a + z + η_id + η_site) + x.Age
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        SITE = [:A, :A, :A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 30.0, 30.0, 35.0, 35.0],
        BMI = [20.0, 20.0, 20.0, 20.0, 22.0, 22.0],
        z = [1.0, 1.2, 0.9, 1.0, 1.1, 1.0],
        y = [1.0, 1.1, 0.9, 1.0, 1.2, 1.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    sim = simulate_data(dm; rng=MersenneTwister(10))

    @test :η_id in propertynames(sim)
    @test :η_site in propertynames(sim)
    @test :η_flow_1 in propertynames(sim)
    @test :η_nn in propertynames(sim)
    @test :η_st in propertynames(sim)

    @test length(unique(sim.η_id[sim.ID .== 1])) == 1
    @test length(unique(sim.η_site[sim.SITE .== :A])) == 1
    @test any(sim.y .!= df.y)
end

@testset "simulate_data ODE (small, multi-RE)" begin
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
            x = ConstantCovariateVector([:Age]; constant_on=[:ID, :SITE])
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @preDifferentialEquation begin
            pre = sat(a + x.Age + η_id)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1^2 + pre + η_site
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(log1p(x1(t)^2), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 35.0, 35.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    sim = simulate_data(dm; rng=MersenneTwister(11))

    @test :η_id in propertynames(sim)
    @test :η_site in propertynames(sim)
    @test length(unique(sim.η_id[sim.ID .== 1])) == 1
    @test length(unique(sim.η_site[sim.SITE .== :A])) == 1
    @test any(sim.y .!= df.y)
end

@testset "simulate_data ODE (NN/SoftTree/Spline, multi-RE)" begin
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
    sim = simulate_data(dm; rng=MersenneTwister(12))

    @test :η_id in propertynames(sim)
    @test :η_site in propertynames(sim)
    @test length(unique(sim.η_id[sim.ID .== 1])) == 1
    @test length(unique(sim.η_site[sim.SITE .== :A])) == 1
    @test any(sim.y .!= df.y)
end

@testset "simulate_data non-ODE with multiple observables" begin
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))

    model = @Model begin
        @helpers begin
            sat(u) = u / (1 + abs(u))
        end

        @fixedEffects begin
            σ1 = RealNumber(0.3)
            σ2 = RealNumber(0.4)
            ζ = NNParameters(chain; function_name=:NN2, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=[:ID, :SITE])
            z = Covariate()
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @formulas begin
            μ = sat(NN2([x.Age, x.BMI], ζ)[1] + z + η_id + η_site)
            y1 ~ Normal(μ, σ1)
            y2 ~ Normal(log1p(μ^2), σ2)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 35.0, 35.0],
        BMI = [20.0, 20.0, 22.0, 22.0],
        z = [1.0, 1.2, 0.9, 1.0],
        y1 = [1.0, 1.1, 0.9, 1.0],
        y2 = [0.5, 0.6, 0.4, 0.55]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    sim = simulate_data(dm; rng=MersenneTwister(13))

    @test :η_id in propertynames(sim)
    @test :η_site in propertynames(sim)
    @test any(sim.y1 .!= df.y1)
    @test any(sim.y2 .!= df.y2)
end

@testset "simulate_data ODE with multiple observables" begin
    model = @Model begin
        @helpers begin
            sat(u) = u / (1 + abs(u))
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ1 = RealNumber(0.3)
            σ2 = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age]; constant_on=[:ID, :SITE])
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @preDifferentialEquation begin
            pre = sat(a + x.Age + η_id)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1^2 + pre + η_site
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y1 ~ Normal(log1p(x1(t)^2), σ1)
            y2 ~ Normal(exp(-x1(t)) + 0.5, σ2)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 35.0, 35.0],
        y1 = [1.0, 1.1, 0.9, 1.0],
        y2 = [0.6, 0.7, 0.5, 0.65]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    sim = simulate_data(dm; rng=MersenneTwister(14))

    @test :η_id in propertynames(sim)
    @test :η_site in propertynames(sim)
    @test any(sim.y1 .!= df.y1)
    @test any(sim.y2 .!= df.y2)
end

@testset "simulate_data threading" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @formulas begin
            y ~ Normal(exp(a + η), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    sim = simulate_data(dm; rng=MersenneTwister(20), serialization=EnsembleThreads())

    @test :η in propertynames(sim)
    @test any(sim.y .!= df.y)
end
