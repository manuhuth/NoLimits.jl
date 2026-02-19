using Test
using NoLimits
using DataFrames
using Distributions

@testset "plot_data and plot_fits basic" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(-0.2)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            z = Covariate()
            t = Covariate()
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

    p_data = plot_data(res)
    @test p_data !== nothing

    p_fits = plot_fits(res)
    @test p_fits !== nothing

    p_fits_dm = plot_fits(dm)
    @test p_fits_dm !== nothing

    p_data_dm = plot_data(dm)
    @test p_data_dm !== nothing

    p_data_z = plot_data(res, x_axis_feature=:z)
    @test p_data_z !== nothing

    p_fits_z = plot_fits(res, x_axis_feature=:z)
    @test p_fits_z !== nothing

    p_fits_dm_z = plot_fits(dm, x_axis_feature=:z)
    @test p_fits_dm_z !== nothing

    p_data_dm_z = plot_data(dm, x_axis_feature=:z)
    @test p_data_dm_z !== nothing

    mktempdir() do tmp
        data_path = joinpath(tmp, "plot_data.png")
        fits_path = joinpath(tmp, "plot_fits.png")
        plot_data(res; plot_path=data_path)
        plot_fits(res; plot_path=fits_path)
        @test isfile(data_path)
        @test isfile(fits_path)
        @test_throws ErrorException plot_data(res; save_path=joinpath(tmp, "a.png"), plot_path=joinpath(tmp, "b.png"))
    end

    @test_throws ErrorException plot_fits(res; x_axis_feature=:missing_feature)

end

@testset "plot_fits supports non-:t time_col" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(-0.2)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            age = Covariate()
            z = Covariate()
        end

        @formulas begin
            y ~ Normal(a + b * z + 0.1 * age, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        age = [0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2],
        y = [0.15, 0.18, 0.14, 0.19]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:age)
    res = fit_model(dm, NoLimits.MLE())

    p_fits = plot_fits(res)
    @test p_fits !== nothing

    p_fits_dm = plot_fits(dm)
    @test p_fits_dm !== nothing

    p_data = plot_data(res)
    @test p_data !== nothing

    p_data_dm = plot_data(dm)
    @test p_data_dm !== nothing

    p_fits_time = plot_fits(res, x_axis_feature=:age)
    @test p_fits_time !== nothing
end

@testset "plot_fits discrete and random effects" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(-0.2)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.5); column=:ID)
        end

        @formulas begin
            p = logistic(a* z + η)
            y ~ Bernoulli(p)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
        y = [0, 1, 0, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace())

    p_fits = plot_fits(res; plot_density=true)
    @test p_fits !== nothing
end

@testset "plot_fits discrete poisson" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            λ = exp(a * z)
            y ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2],
        y = [1, 2, 1, 3]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE())

    p_fits = plot_fits(res; plot_density=false)
    @test p_fits !== nothing
end

@testset "plot_fits MCMC" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a*t, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, MCMC(; turing_kwargs=(n_samples=100, n_adapt=2, progress=false)))

    p_fits = plot_fits(res; mcmc_draws=100, plot_mcmc_quantiles=true, mcmc_quantiles=[5, 95], mcmc_quantiles_alpha=0.8)
    @test p_fits !== nothing
end

@testset "plot_data discrete" begin
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
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2],
        y = [1, 2, 1, 3]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    p_data = plot_data(dm)
    @test p_data !== nothing
end

@testset "plot_data and plot_fits (ODE)" begin
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
    res = fit_model(dm, NoLimits.MLE());

    p_data = plot_data(res)
    @test p_data !== nothing

    p_fits = plot_fits(res)
    @test p_fits !== nothing

    p_fits_density = plot_fits(res; plot_density=true)
    @test p_fits_density !== nothing

    p_fits_dm = plot_fits(dm)
    @test p_fits_dm !== nothing
end

@testset "plot_fits inherits constants_re from fit result" begin
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

    p = plot_fits(res)
    @test p !== nothing
end

@testset "plot_multistart_waterfall basic" begin
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
            y ~ Normal(a + b * z + 0.1 * t, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.0, 0.1, -0.1, 0.1],
        y = [0.12, 0.18, 0.08, 0.14, 0.04, 0.11],
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    ms = NoLimits.Multistart(;
        dists=(; a=Normal(0.0, 0.5), b=Normal(0.0, 0.5)),
        n_draws_requested=6,
        n_draws_used=4,
        sampling=:lhs,
    )

    res_ms = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=80,)))
    p = plot_multistart_waterfall(res_ms)
    @test p !== nothing

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_multistart_waterfall.png")
        plot_multistart_waterfall(res_ms; plot_path=p_path)
        @test isfile(p_path)
    end
end

@testset "plot_multistart_fixed_effect_variability basic" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, calculate_se=true)
            b = RealNumber(-0.2, calculate_se=false)
            v = RealVector([0.05, -0.03], calculate_se=true)
            σ = RealNumber(0.25, scale=:log, calculate_se=true)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            y ~ Normal(a + b * z + v[1] * t + v[2] * z, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.0, 0.1, -0.1, 0.1, 0.05, 0.15],
        y = [0.10, 0.16, 0.06, 0.12, 0.03, 0.10, 0.08, 0.14],
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    ms = NoLimits.Multistart(;
        dists=(; a=Normal(0.0, 0.4), b=Normal(0.0, 0.4), v=Normal(0.0, 0.25)),
        n_draws_requested=6,
        n_draws_used=4,
        sampling=:lhs,
    )

    res_ms = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=80,)))

    p_points = plot_multistart_fixed_effect_variability(res_ms; k_best=3, mode=:points)
    @test p_points !== nothing

    p_quant = plot_multistart_fixed_effect_variability(
        res_ms;
        k_best=3,
        mode=:quantiles,
        quantiles=[0.1, 0.5, 0.9],
        include_parameters=[:b],
        exclude_parameters=[:a],
    )
    @test p_quant !== nothing

    p_transformed = plot_multistart_fixed_effect_variability(res_ms; k_best=3, scale=:transformed)
    @test p_transformed !== nothing

    @test_throws ErrorException plot_multistart_fixed_effect_variability(res_ms; mode=:invalid)
    @test_throws ErrorException plot_multistart_fixed_effect_variability(res_ms; include_parameters=[:missing_parameter])

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_multistart_fixed_effect_variability.png")
        plot_multistart_fixed_effect_variability(res_ms; k_best=3, plot_path=p_path)
        @test isfile(p_path)
    end
end

@testset "plot_fits_comparison basic" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(-0.2)
            σ = RealNumber(0.25, scale=:log)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            y ~ Normal(a + b * z + 0.1 * t, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.0, 0.1, -0.1, 0.1],
        y = [0.12, 0.18, 0.08, 0.14, 0.04, 0.11],
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res1 = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=80,)))
    res2 = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=80,)); constants=(; a=0.2))

    p_single = plot_fits_comparison(res1)
    @test p_single !== nothing

    p_vec = plot_fits_comparison([res1, res2]; individuals_idx=1:2)
    @test p_vec !== nothing
    labels_vec = [String(get(s.plotattributes, :label, "")) for s in p_vec.subplots[1].series_list]
    @test "data" in labels_vec
    @test "Model 1" in labels_vec
    @test "Model 2" in labels_vec

    p_nt = plot_fits_comparison((baseline=res1, constrained=res2); individuals_idx=1:2)
    @test p_nt !== nothing
    labels_nt = [String(get(s.plotattributes, :label, "")) for s in p_nt.subplots[1].series_list]
    @test "baseline" in labels_nt
    @test "constrained" in labels_nt

    styled = PlotStyle(comparison_line_styles=Dict("baseline" => :dash))
    p_nt_style = plot_fits_comparison((baseline=res1, constrained=res2); individuals_idx=1:2, style=styled)
    @test p_nt_style !== nothing
    idx_base = findfirst(==( "baseline"), [String(get(s.plotattributes, :label, "")) for s in p_nt_style.subplots[1].series_list])
    @test idx_base !== nothing
    @test get(p_nt_style.subplots[1].series_list[idx_base].plotattributes, :linestyle, :solid) == :dash

    p_dict = plot_fits_comparison(Dict("first" => res1, "second" => res2); individuals_idx=1:2)
    @test p_dict !== nothing
    labels_dict = [String(get(s.plotattributes, :label, "")) for s in p_dict.subplots[1].series_list]
    @test "first" in labels_dict
    @test "second" in labels_dict

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_fits_comparison.png")
        plot_fits_comparison([res1, res2]; individuals_idx=1:2, plot_path=p_path)
        @test isfile(p_path)
    end

    df_bad = copy(df)
    df_bad.y .= df_bad.y .+ 1.0
    dm_bad = DataModel(model, df_bad; primary_id=:ID, time_col=:t)
    res_bad = fit_model(dm_bad, NoLimits.MLE(; optim_kwargs=(maxiters=40,)))
    @test_throws ErrorException plot_fits_comparison([res1, res_bad])
end
