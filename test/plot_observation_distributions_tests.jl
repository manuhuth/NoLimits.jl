using Test
using NoLimits
using DataFrames
using Distributions
using Lux

@testset "plot_observation_distributions basic" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [0.1, 0.2]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    p = plot_observation_distributions(res; individuals_idx=1, obs_rows=1, observables=:y)
    @test p !== nothing

    p_dm = plot_observation_distributions(dm; individuals_idx=1, obs_rows=1, observables=:y)
    @test p_dm !== nothing

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_observation_distributions.png")
        plot_observation_distributions(res; individuals_idx=1, obs_rows=1, observables=:y, plot_path=p_path)
        @test isfile(p_path)
    end
end

@testset "plot_observation_distributions multiple individuals and obs rows" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        y = [0.1, 0.15, 0.2, 0.0, 0.05, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    p = plot_observation_distributions(res; individuals_idx=[1, 2], obs_rows=[1, 3], observables=:y)
    @test p !== nothing
end

@testset "plot_observation_distributions discrete" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            λ = exp(a + z)
            y ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        z = [0.1, 0.2],
        y = [1, 2]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    p = plot_observation_distributions(res; individuals_idx=1, obs_rows=1, observables=:y)
    @test p !== nothing
end

@testset "plot_observation_distributions MCMC" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [0.1, 0.2]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=10, n_adapt=2, progress=false)))

    p = plot_observation_distributions(res; individuals_idx=1, obs_rows=1, observables=:y, mcmc_draws=100)
    @test p !== nothing
end

@testset "plot_observation_distributions VI" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [0.1, 0.2]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=25, progress=false)))

    p = plot_observation_distributions(res; individuals_idx=1, obs_rows=1, observables=:y, mcmc_draws=30)
    @test p !== nothing
end

@testset "plot_observation_distributions caching" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [0.1, 0.2]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    cache = build_plot_cache(res; cache_obs_dists=true)
    p_cache = plot_observation_distributions(res; individuals_idx=1, obs_rows=1, observables=:y,
                                             cache=cache, cache_obs_dists=true)
    @test p_cache !== nothing

    p_no_cache = plot_observation_distributions(res; individuals_idx=1, obs_rows=1, observables=:y)
    @test p_no_cache !== nothing
end

@testset "plot_observation_distributions ODE with callbacks" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            σ = RealNumber(0.01)
        end

        @DifferentialEquation begin
            D(x1) ~ 0.0
        end

        @initialDE begin
            x1 = 0.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 0.5, 1.0],
        EVID = [1, 0, 0],
        AMT = [100.0, 0.0, 0.0],
        RATE = [0.0, 0.0, 0.0],
        CMT = [1, 1, 1],
        y = [missing, 1.0, 1.1]
    )

    dm = DataModel(model, df;
                   primary_id=:ID,
                   time_col=:t,
                   evid_col=:EVID,
                   amt_col=:AMT,
                   rate_col=:RATE,
                   cmt_col=:CMT)

    p = plot_observation_distributions(dm; individuals_idx=1, obs_rows=1, observables=:y)
    @test p !== nothing
end

@testset "plot_observation_distributions preDE" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.1, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @preDifferentialEquation begin
            k = a
        end

        @DifferentialEquation begin
            D(x) ~ -k * x
        end

        @initialDE begin
            x = 1.0
        end

        @formulas begin
            y ~ Normal(x(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 0.9]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    p = plot_observation_distributions(dm; individuals_idx=1, obs_rows=1, observables=:y)
    @test p !== nothing
end

@testset "plot_observation_distributions NN and SoftTree" begin
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))

    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.2)
            ζ = NNParameters(chain; function_name=:NN1, calculate_se=false)
            Γ = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
        end

        @formulas begin
            μ = NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1]
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        Age = [30.0, 30.0],
        BMI = [20.0, 20.0],
        y = [0.1, 0.2]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    p = plot_observation_distributions(dm; individuals_idx=1, obs_rows=1, observables=:y)
    @test p !== nothing
end

@testset "plot_observation_distributions random effects fitted" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.5); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace())

    p = plot_observation_distributions(res; individuals_idx=[1, 2], obs_rows=1, observables=:y)
    @test p !== nothing
end

@testset "plot_observation_distributions inherits constants_re from fit result" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.5); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    constants_re = (; η=(; B=0.0))
    res = fit_model(dm, NoLimits.Laplace(); constants_re=constants_re)

    p = plot_observation_distributions(res; individuals_idx=1, obs_rows=1, observables=:y)
    @test p !== nothing
end

@testset "plot_observation_distributions Bernoulli discrete outcome" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            p = logistic(a + b * z)
            y ~ Bernoulli(p)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.3, 0.0, 0.2],
        y = [0, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    p_res = plot_observation_distributions(res; individuals_idx=1, obs_rows=1, observables=:y)
    @test p_res !== nothing

    p_dm = plot_observation_distributions(dm; individuals_idx=1, obs_rows=1, observables=:y)
    @test p_dm !== nothing
end
