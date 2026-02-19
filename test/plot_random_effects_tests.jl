using Test
using NoLimits
using DataFrames
using Distributions
using Turing

@testset "plot_random_effects Laplace" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale=:log, prior=Uniform(0.01, 1.0))
        end

        @covariates begin
            t = Covariate()
            Age = ConstantCovariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.5); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 40.0, 40.0, 35.0, 35.0, 45.0, 45.0, 28.0, 28.0, 55.0, 55.0],
        y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05, 0.2, 0.3, -0.1, 0.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm,  NoLimits.Laplace())

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std = plot_random_effect_standardized(res, show_hist=true)
    @test p_std !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(res; show_hist=true, show_kde=false, show_qq=false)
    @test p_pit_hist !== nothing

    p_pit_kde = plot_random_effect_pit(res; show_hist=false, show_kde=true, show_qq=false)
    @test p_pit_kde !== nothing

    p_pit_qq = plot_random_effect_pit(res; show_hist=false, show_kde=false, show_qq=true)
    @test p_pit_qq !== nothing

    @test_throws ArgumentError plot_random_effect_pit(res; x_covariate=:Age)

    p_pdf = plot_random_effects_pdf(res)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res)
    @test p_scatter !== nothing

    p_pari = plot_random_effect_pairplot(res)
    @test p_pari !== nothing

    mktempdir() do tmp
        p_dist_path = joinpath(tmp, "plot_random_effect_distributions.png")
        p_pit_path = joinpath(tmp, "plot_random_effect_pit.png")
        p_std_path = joinpath(tmp, "plot_random_effect_standardized.png")
        p_std_sc_path = joinpath(tmp, "plot_random_effect_standardized_scatter.png")
        p_pdf_path = joinpath(tmp, "plot_random_effects_pdf.png")
        p_scatter_path = joinpath(tmp, "plot_random_effects_scatter.png")
        p_pair_path = joinpath(tmp, "plot_random_effect_pairplot.png")
        plot_random_effect_distributions(res; plot_path=p_dist_path)
        plot_random_effect_pit(res; show_hist=true, show_kde=false, show_qq=false, plot_path=p_pit_path)
        plot_random_effect_standardized(res; show_hist=true, plot_path=p_std_path)
        plot_random_effect_standardized_scatter(res; plot_path=p_std_sc_path)
        plot_random_effects_pdf(res; plot_path=p_pdf_path)
        plot_random_effects_scatter(res; plot_path=p_scatter_path)
        plot_random_effect_pairplot(res; plot_path=p_pair_path)
        @test isfile(p_dist_path)
        @test isfile(p_pit_path)
        @test isfile(p_std_path)
        @test isfile(p_std_sc_path)
        @test isfile(p_pdf_path)
        @test isfile(p_scatter_path)
        @test isfile(p_pair_path)
    end
end

@testset "plot_random_effects Laplace multiple RE groups" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale=:log, prior=Uniform(0.01, 1.0))
        end

        @covariates begin
            t = Covariate()
            Age = ConstantCovariate(constant_on=:ID)
            Center = ConstantCovariate(constant_on=:Center)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.5); column=:ID)
            ζ = RandomEffect(Normal(0.0, 0.2); column=:Center)
        end

        @formulas begin
            y ~ Normal(a + η + ζ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        Center = [1, 1, 1, 1, 2, 2, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 40.0, 40.0, 35.0, 35.0, 45.0, 45.0],
        y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace())

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std = plot_random_effect_standardized(res)
    @test p_std !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res; x_covariate=:Age)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(res; show_hist=true, show_kde=false, show_qq=false)
    @test p_pit_hist !== nothing
end

@testset "plot_random_effects Laplace non-normal RE" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale=:log, prior=Uniform(0.01, 1.0))
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(LogNormal(0.0, 0.4); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.3, 0.1, 0.2, 0.25, 0.35]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace())

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_pit_hist = plot_random_effect_pit(res; show_hist=true, show_kde=false, show_qq=false)
    @test p_pit_hist !== nothing
end

@testset "plot_random_effects Laplace single-level RE" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale=:log, prior=Uniform(0.01, 1.0))
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
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        y = [0.1, 0.2, 0.15, 0.25]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace())

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std = plot_random_effect_standardized(res)
    @test p_std !== nothing
end

@testset "plot_random_effects Laplace constants_re" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale=:log, prior=Uniform(0.01, 1.0))
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
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    constants_re = (; η=(; B=0.0))
    res = fit_model(dm, NoLimits.Laplace(); constants_re=constants_re)

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std = plot_random_effect_standardized(res)
    @test p_std !== nothing
end

@testset "plot_random_effects Laplace multivariate Normal RE" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale=:log, prior=Uniform(0.01, 1.0))
            μ = RealVector([0.0, 0.0], prior=filldist(Uniform(-1.0, 1.0), 2))
            Ω = RealPSDMatrix(Matrix(I, 2, 2), scale=:cholesky, prior=InverseWishart(3, Matrix(I, 2, 2)))
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η_mv = RandomEffect(MvNormal(μ, Ω); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η_mv[1] + η_mv[2], σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05, 0.2, 0.3, -0.1, 0.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace())

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std = plot_random_effect_standardized(res)
    @test p_std !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(res; show_hist=true, show_kde=false, show_qq=false)
    @test p_pit_hist !== nothing
end

@testset "plot_random_effects MCMC" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale=:log, prior = Uniform(0.01, 1.0))
        end

        @covariates begin
            t = Covariate()
            Age = ConstantCovariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.5); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 40.0, 40.0, 35.0, 35.0, 45.0, 45.0, 28.0, 28.0, 55.0, 55.0],
        y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05, 0.2, 0.3, -0.1, 0.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=30, n_adapt=20, progress=false)))

    p_pit_hist = plot_random_effect_pit(res; mcmc_draws=5, show_hist=true, show_kde=false, show_qq=false)
    @test p_pit_hist !== nothing

    p_pit_kde = plot_random_effect_pit(res; mcmc_draws=5, show_hist=false, show_kde=true, show_qq=false)
    @test p_pit_kde !== nothing

    p_pit_qq = plot_random_effect_pit(res; mcmc_draws=5, show_hist=false, show_kde=false, show_qq=true)
    @test p_pit_qq !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res)
    @test p_std_sc !== nothing

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_pdf = plot_random_effects_pdf(res; mcmc_draws=5)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res; mcmc_draws=5)
    @test p_scatter !== nothing

    p_pari = plot_random_effect_pairplot(res; mcmc_draws=5)
    @test p_pari !== nothing


end

@testset "plot_random_effects MCMC constants_re" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale=:log, prior=Uniform(0.01, 1.0))
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
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    constants_re = (; η=(; B=0.0))
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=30, n_adapt=20, progress=false)); constants_re=constants_re)

    p_dist = plot_random_effect_distributions(res; flow_plot=:kde, flow_samples=200, mcmc_draws=5)
    @test p_dist !== nothing

    p_dist_hist = plot_random_effect_distributions(res; flow_plot=:hist, flow_samples=200, flow_bins=10, mcmc_draws=5)
    @test p_dist_hist !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(res; mcmc_draws=5, show_hist=true, show_kde=false, show_qq=false)
    @test p_pit_hist !== nothing
end

@testset "plot_random_effects Laplace NormalizingPlanarFlow RE" begin
    npf0 = NPFParameter(1, 2, seed=1, calculate_se=false)
    n_npf = length(npf0.value)
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
            ψ = NPFParameter(1, 2, seed=1, calculate_se=false,
                             prior=filldist(Normal(0.0, 1.0), n_npf))
        end

        @randomEffects begin
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η_flow[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.15, 0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace())

    p_dist = plot_random_effect_distributions(res; flow_plot=:kde, flow_samples=200)
    @test p_dist !== nothing

    p_dist_hist = plot_random_effect_distributions(res; flow_plot=:hist, flow_samples=200, flow_bins=10)
    @test p_dist_hist !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res, flow_samples=100)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(res; show_hist=true, show_kde=false, show_qq=false, flow_samples=100)
    @test p_pit_hist !== nothing
end

@testset "plot_random_effects MCMC multivariate NormalizingPlanarFlow RE" begin
    npf0 = NPFParameter(2, 2, seed=1, calculate_se=false)
    n_npf = length(npf0.value)
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
            ψ = NPFParameter(2, 2, seed=1, calculate_se=false,
                             prior=filldist(Normal(0.0, 1.0), n_npf))
        end

        @randomEffects begin
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η_flow[1] + η_flow[2], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.15, 0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=NUTS(5, 0.3),
                                              turing_kwargs=(n_samples=30, n_adapt=20, progress=false)))

    p_dist = plot_random_effect_distributions(res; flow_plot=:kde, flow_samples=200, mcmc_draws=5)
    @test p_dist !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res; flow_samples=200)
    @test p_std_sc !== nothing
end

@testset "plot_random_effects Laplace RE with constant covariates" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Uniform(0.0, 0.9))
            b = RealNumber(0.02, prior=Uniform(-0.5, 0.5))
            σ = RealNumber(0.3, scale=:log, prior=Uniform(0.01, 1.0))
        end

        @covariates begin
            t = Covariate()
            Age = ConstantCovariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(b * Age, 0.5); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 40.0, 40.0, 35.0, 35.0, 45.0, 45.0],
        y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace())

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res; x_covariate=:Age)
    @test p_std_sc !== nothing
end

@testset "plot_random_effects MCMC RE with constant covariates" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            b = RealNumber(0.02, prior=Normal(0.0, 0.5))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @covariates begin
            t = Covariate()
            Age = ConstantCovariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(b * Age, 0.5); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 40.0, 40.0, 35.0, 35.0, 45.0, 45.0],
        y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=30, n_adapt=20, progress=false)))

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res; x_covariate=:Age)
    @test p_std_sc !== nothing
end

@testset "plot_random_effects MCMC NormalizingPlanarFlow RE" begin
    npf0 = NPFParameter(1, 2, seed=1, calculate_se=false)
    n_npf = length(npf0.value)
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
            ψ = NPFParameter(1, 2, seed=1, calculate_se=false,
                             prior=filldist(Normal(0.0, 1.0), n_npf))
        end

        @randomEffects begin
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η_flow[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.15, 0.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, MCMC(; sampler=NUTS(5, 0.3),
                                              turing_kwargs=(n_samples=30, n_adapt=20, progress=false)))

    p_dist = plot_random_effect_distributions(res; flow_plot=:kde, flow_samples=200, mcmc_draws=5)
    @test p_dist !== nothing

    p_dist_hist = plot_random_effect_distributions(res; flow_plot=:hist, flow_samples=200, flow_bins=10, mcmc_draws=5)
    @test p_dist_hist !== nothing

    p_std = plot_random_effect_standardized(res; flow_samples=200)
    @test p_std !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res; flow_samples=200)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(res; mcmc_draws=5, show_hist=true, show_kde=false, show_qq=false, flow_samples=200)
    @test p_pit_hist !== nothing

    p_pit_kde = plot_random_effect_pit(res; mcmc_draws=5, show_hist=false, show_kde=true, show_qq=false, flow_samples=200)
    @test p_pit_kde !== nothing

    p_pit_qq = plot_random_effect_pit(res; mcmc_draws=5, show_hist=false, show_kde=false, show_qq=true, flow_samples=200)
    @test p_pit_qq !== nothing
end

@testset "plot_random_effects MLE error" begin
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
    @test_throws ErrorException plot_random_effect_distributions(res)
end
