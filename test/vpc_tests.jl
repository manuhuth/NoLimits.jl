using Test
using NoLimits
using DataFrames
using Distributions
using Turing

@testset "plot_vpc basic" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(-0.2)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            y ~ Normal(a + b * z, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2],
        y = [0.15, 0.18, 0.14, 0.19]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    p_vpc = plot_vpc(res; x_axis_feature=:z, n_simulations=5, n_bins=2)
    @test p_vpc !== nothing

    p_vpc_t = plot_vpc(res; n_simulations=5, n_bins=2)
    @test p_vpc_t !== nothing

    p_vpc_alias = plot_vpc(res; n_sim=5, n_bins=2)
    @test p_vpc_alias !== nothing

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_vpc.png")
        plot_vpc(res; n_simulations=5, n_bins=2, plot_path=p_path)
        @test isfile(p_path)
    end
end

@testset "plot_vpc ode" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0, lower=0.001, upper=1.1)
            σ = RealNumber(0.01, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 0.5, 1.0],
        y = [1.0, 0.95, 0.9]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    p_vpc = plot_vpc(res; n_simulations=5, n_bins=3)
    @test p_vpc !== nothing
end

@testset "plot_vpc discrete mcmc random effects" begin
    model_bern = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3, prior=Normal(1.0, 10.0))
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 2.0); column=:ID)
        end

        @formulas begin
            p = logistic(a*t + z + η)
            y ~ Bernoulli(p)
        end
    end

    df_bern = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
        y = [0, 1, 0, 1, 0, 1]
    )

    dm_bern = DataModel(model_bern, df_bern; primary_id=:ID, time_col=:t)
    res_bern = fit_model(dm_bern, MCMC(; sampler=MH(), turing_kwargs=(n_samples=30, n_adapt=2, progress=false)))
    p_vpc_bern = plot_vpc(res_bern; n_simulations=5, n_bins=3, mcmc_draws=5)
    @test p_vpc_bern !== nothing

    model_pois = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1,  prior=Normal(1.0, 10.0))
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.3); column=:ID)
        end

        @formulas begin
            λ = exp(a * z + η)
            y ~ Poisson(λ)
        end
    end

    df_pois = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
        y = [1, 2, 1, 2, 1, 3]
    )

    dm_pois = DataModel(model_pois, df_pois; primary_id=:ID, time_col=:t)
    res_pois = fit_model(dm_pois, MCMC(; sampler=MH(), turing_kwargs=(n_samples=20, n_adapt=2, progress=false)))
    p_vpc_pois = plot_vpc(res_pois; n_simulations=5, n_bins=3, mcmc_draws=5)
    @test p_vpc_pois !== nothing
end

@testset "plot_vpc constants_re and unsupported serialization" begin
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

    p = plot_vpc(res; n_simulations=5, n_bins=2)
    @test p !== nothing

    @test_throws ArgumentError plot_vpc(res; serialization=:unsupported)
end

@testset "plot_vpc validation errors" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(-0.2)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            y ~ Normal(a + b * z, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2],
        y = [0.15, 0.18, 0.14, 0.19]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    @test_throws ErrorException plot_vpc(res; n_sim=0)
    @test_throws ErrorException plot_vpc(res; n_simulations=0)
    @test_throws ErrorException plot_vpc(res; n_simulations=6, n_sim=5)
    @test_throws ErrorException plot_vpc(res; percentiles=[50])
    @test_throws ErrorException plot_vpc(res; percentiles=[-5, 50, 95])
    @test_throws ErrorException plot_vpc(res; n_bins=0)
    @test_throws ErrorException plot_vpc(res; obs_percentiles_method=:invalid)
    @test_throws ErrorException plot_vpc(res; obs_percentiles_mode=:invalid)
    @test_throws ErrorException plot_vpc(res; obs_percentiles_mode=:per_individual, obs_percentiles_method=:kernel)
end
