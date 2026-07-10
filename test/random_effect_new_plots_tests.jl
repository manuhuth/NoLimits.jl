using Test
using NoLimits
using CairoMakie
using DataFrames
using Distributions
using Turing

# One rich multi-RE model (scalar + multivariate on :ID, scalar on :Center)
# shared by the Laplace and MCMC testsets below. The planar-flow testsets use
# the shared fx_npf*/fx_npf2* fixtures (models AND fits) from fixtures.jl.
function _rnp_df(; n_ids::Int = 10, n_obs_per::Int = 2)
    ids = [Symbol("ID", i) for i in 1:n_ids]
    ID = repeat(ids, inner = n_obs_per)
    t = repeat(collect(0.0:(n_obs_per - 1)), n_ids)
    Center = repeat(
        vcat(fill(:C1, n_ids ÷ 2), fill(:C2, n_ids - n_ids ÷ 2)), inner = n_obs_per)
    y = [0.1 * sin(0.3 * i) + 0.02 * j for (i, j) in zip(1:length(ID), t)]
    return DataFrame(ID = ID, Center = Center, t = t, y = y)
end

const _RNP_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(0.1, prior = Uniform(0.0, 0.9))
        σ = RealNumber(0.3, scale = :log, prior = Uniform(0.01, 1.0))
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, 0.5); column = :ID)
        zeta = RandomEffect(MvNormal([0.0, 0.0], [0.3 0.0; 0.0 0.4]); column = :ID)
        xi = RandomEffect(Normal(0.0, 0.2); column = :Center)
    end

    @formulas begin
        y ~ Normal(a + eta + zeta[1] + zeta[2] + xi, σ)
    end
end

const _RNP_DM = DataModel(_RNP_MODEL, _rnp_df(); primary_id = :ID, time_col = :t)

@testset "random effects new plots Laplace (multi-id, MVN)" begin
    res = fit_model(_RNP_DM,
        NoLimits.Laplace(; use_hutchinson = false, optim_kwargs = (maxiters = 2,)))

    p_pdf = plot_random_effects_pdf(res)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res)
    @test p_pair !== nothing
end

@testset "random effects new plots MCMC (multi-id, MVN)" begin
    res = fit_model(_RNP_DM,
        NoLimits.MCMC(; sampler = NUTS(5, 0.3),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false)))

    p_pdf = plot_random_effects_pdf(res; mcmc_draws = 5)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res; mcmc_draws = 5)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res; mcmc_draws = 5)
    @test p_pair !== nothing
end

@testset "random effects new plots Laplace flow dim1" begin
    res = fx_npf_laplace()

    p_pdf = plot_random_effects_pdf(res; flow_samples = 50, flow_plot = :kde)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res)
    @test p_pair !== nothing
end

@testset "random effects new plots MCMC flow dim1" begin
    res = fx_npf_mcmc()

    p_pdf = plot_random_effects_pdf(
        res; mcmc_draws = 5, flow_samples = 50, flow_plot = :kde)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res; mcmc_draws = 5)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res; mcmc_draws = 5)
    @test p_pair !== nothing
end

@testset "random effects new plots Laplace flow dim2" begin
    res = fx_npf2_laplace()

    p_pdf = plot_random_effects_pdf(res; flow_samples = 50, flow_plot = :hist)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res)
    @test p_pair !== nothing
end

@testset "random effects new plots MCMC flow dim2" begin
    res = fx_npf2_mcmc()

    p_pdf = plot_random_effects_pdf(
        res; mcmc_draws = 5, flow_samples = 50, flow_plot = :hist)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res; mcmc_draws = 5)
    @test p_scatter !== nothing

    p_pair = plot_random_effect_pairplot(res; mcmc_draws = 5)
    @test p_pair !== nothing
end
