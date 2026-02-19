using Test
using NoLimits
using DataFrames
using Distributions
using Turing


function _make_df(; n_ids::Int=10, n_obs_per::Int=2)
    ids = [Symbol("ID", i) for i in 1:n_ids]
    ID = repeat(ids, inner=n_obs_per)
    t = repeat(collect(0.0:(n_obs_per - 1)), n_ids)
    Center = repeat(vcat(fill(:C1, n_ids ÷ 2), fill(:C2, n_ids - n_ids ÷ 2)), inner=n_obs_per)
    y = [0.1 * sin(0.3 * i) + 0.02 * j for (i, j) in zip(1:length(ID), t)]
    return DataFrame(ID=ID, Center=Center, t=t, y=y)
end



@testset "random effects new plots Laplace (multi-id, MVN)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale=:log, prior=Uniform(0.01, 1.0))
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            eta = RandomEffect(Normal(0.0, 0.5); column=:ID)
            zeta2 = RandomEffect(MvNormal([0.0, 0.0], [0.3 0.0; 0.0 0.4]); column=:ID)
            #zeta = RandomEffect(Normal(0.0, 0.4); column=:ID)
            xi = RandomEffect(Normal(0.0, 0.2); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + eta + zeta2[1] , exp(zeta2[2]) + σ)  #
            #y ~ Normal(a + eta + xi , exp(zeta) + σ) 
        end
    end;

    df = _make_df()
    dm = DataModel(model, df; primary_id=:ID, time_col=:t);
    @time res = fit_model(dm, NoLimits.Laplace(; use_hutchinson=false, optim_kwargs=(maxiters=2,)));

    p_pdf = plot_random_effects_pdf(res)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res)
    @test p_pair !== nothing
end

@testset "random effects new plots MCMC (multi-id, MVN)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale=:log, prior=Uniform(0.01, 1.0))
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            eta = RandomEffect(Normal(0.0, 0.5); column=:ID)
            zeta = RandomEffect(MvNormal([0.0, 0.0], [0.3 0.0; 0.0 0.4]); column=:ID)
            xi = RandomEffect(Normal(0.0, 0.2); column=:Center)
        end

        @formulas begin
            y ~ Normal(a + eta + zeta[1] + zeta[2] + xi, σ)
        end
    end

    df = _make_df()
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=NUTS(5, 0.3),
                                              turing_kwargs=(n_samples=20, n_adapt=10, progress=false)))

    p_pdf = plot_random_effects_pdf(res; mcmc_draws=5)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res; mcmc_draws=5)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res; mcmc_draws=5)
    @test p_pair !== nothing
end

@testset "random effects new plots Laplace flow dim1" begin
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

    df = _make_df()
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,)))

    p_pdf = plot_random_effects_pdf(res; flow_samples=50, flow_plot=:kde)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res)
    @test p_pair !== nothing
end

@testset "random effects new plots MCMC flow dim1" begin
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

    df = _make_df()
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=NUTS(5, 0.3),
                                              turing_kwargs=(n_samples=20, n_adapt=10, progress=false)))

    p_pdf = plot_random_effects_pdf(res; mcmc_draws=5, flow_samples=50, flow_plot=:kde)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res; mcmc_draws=5)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res; mcmc_draws=5)
    @test p_pair !== nothing
end

@testset "random effects new plots Laplace flow dim2" begin
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

    df = _make_df()
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,)))

    p_pdf = plot_random_effects_pdf(res; flow_samples=50, flow_plot=:hist)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res)
    @test p_pair !== nothing
end

@testset "random effects new plots MCMC flow dim2" begin
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

    df = _make_df()
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=NUTS(5, 0.3),
                                              turing_kwargs=(n_samples=20, n_adapt=10, progress=false)))

    p_pdf = plot_random_effects_pdf(res; mcmc_draws=5, flow_samples=50, flow_plot=:hist)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res; mcmc_draws=5)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res; mcmc_draws=5)
    @test p_pair !== nothing
end
