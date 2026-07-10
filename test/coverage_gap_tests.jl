using Test
using NoLimits
using CairoMakie
using DataFrames
using Distributions
using Random
using LinearAlgebra
using Turing

# ── Tests targeting code paths missed by the existing suite ──────────────────
# (goodness-of-fit / diagnostic plots, EM trajectory plots, Multistart EBE
#  screening, equation-display rendering). Kept small (few individuals, low
#  iteration counts) since the goal is path coverage, not estimation accuracy.

function _gap_re_model()
    @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.5); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
end

function _gap_re_df(; n_ids::Int = 6, n_obs::Int = 3)
    ids = repeat(1:n_ids, inner = n_obs)
    t = repeat(collect(0.0:(n_obs - 1)), n_ids)
    y = [0.2 + 0.05 * i + 0.03 * j for (i, j) in zip(ids, t)]
    return DataFrame(ID = ids, t = t, y = y)
end

@testset "GOF and diagnostic plots (Laplace RE fit)" begin
    dm = DataModel(_gap_re_model(), _gap_re_df(); primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)))

    @test plot_dv_pred(res) !== nothing
    @test plot_dv_ipred(res) !== nothing
    @test plot_wres_pred(res) !== nothing
    @test plot_shrinkage(res) !== nothing
    @test plot_observed_profiles(res) !== nothing
    @test plot_observed_profiles(dm) !== nothing

    # compute_shrinkage is the data path behind plot_shrinkage
    shrink = NoLimits.compute_shrinkage(res)
    @test haskey(shrink, :η)
    @test isfinite(shrink.η.shrinkage)
end

@testset "EM trajectory plots (MCEM with diagnostics)" begin
    dm = DataModel(_gap_re_model(), _gap_re_df(); primary_id = :ID, time_col = :t)
    res = fit_model(dm,
        NoLimits.MCEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 10, n_adapt = 3, progress = false),
            maxiters = 3,
            store_diagnostics = true
        ))
    p = plot_em_trajectories(res)
    @test p !== nothing
    # transformed-scale variant exercises the no-DataModel branch
    @test plot_em_trajectories(res; scale = :transformed) !== nothing
end

@testset "Multistart EBE screening (random effects)" begin
    dm = DataModel(_gap_re_model(), _gap_re_df(); primary_id = :ID, time_col = :t)
    ms = NoLimits.Multistart(
        dists = (; a = Normal(0.0, 1.0)),
        n_draws_requested = 4,
        n_draws_used = 2,
        screening = :ebe,
        ebe_maxiters = 3,
        progress = false,
        rng = Random.Xoshiro(1)
    )
    res = fit_model(ms, dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))
    @test length(NoLimits.get_multistart_results(res)) >= 1
    @test NoLimits.get_multistart_best(res) !== nothing
end

@testset "Equation display rendering paths" begin
    # Model with prede + DE + formulas exercises the full latex block builder.
    model = @Model begin
        @fixedEffects begin
            k = RealNumber(0.5, scale = :log)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @preDifferentialEquation begin
            r = k + 1.0
        end
        @DifferentialEquation begin
            D(x1) ~ -r * x1
        end
        @initialDE begin
            x1 = 1.0
        end
        @formulas begin
            mu = x1(t)
            y ~ Normal(mu, σ)
        end
    end

    # latex=true (no io) returns a rendered block object via _eq_latex_block
    block = NoLimits.show_equations(model; latex = true)
    @test block !== nothing

    block_num = NoLimits.show_equations(model; latex = true, numbered = true)
    @test block_num !== nothing

    # latex=false (no io) returns a String via the plain renderer
    plain = NoLimits.show_equations(model; latex = false)
    @test plain isa AbstractString
    @test occursin("x1", plain)

    # Formulas-only model exercises get_equation_lines without prede/DE
    model2 = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end
    lines2 = NoLimits.get_equation_lines(model2)
    @test !isempty(lines2)
end

function _gap_re_prior_model()
    @Model begin
        @fixedEffects begin
            a = RealNumber(0.2, prior = Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.5); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
end

@testset "RE posterior prediction plots (MCMC)" begin
    dm = DataModel(_gap_re_prior_model(), _gap_re_df(); primary_id = :ID, time_col = :t)

    res_mcmc = fit_model(dm,
        NoLimits.MCMC(;
            turing_kwargs = (n_samples = 20, n_adapt = 10, progress = false));
        rng = Random.Xoshiro(1))
    # default path -> _mcmc_random_effects_means
    @test plot_fits(res_mcmc) !== nothing
    # posterior-draw path -> _mcmc_drawn_params
    @test plot_fits(res_mcmc; plot_mcmc_quantiles = true, mcmc_draws = 5) !== nothing
    # RE diagnostics on a posterior fit
    @test plot_random_effect_distributions(res_mcmc) !== nothing
    @test plot_random_effect_pit(res_mcmc) !== nothing
    @test plot_random_effect_standardized(res_mcmc) !== nothing
end

@testset "VI posterior-draw prediction plots (no RE)" begin
    # VI rejects random-effects models, so exercise the VI posterior-draw plot
    # path (_vi_drawn_params) with a fixed-effects-only model + priors.
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2, prior = Normal(0.0, 1.0))
            b = RealNumber(0.1, prior = Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + b * t, σ)
        end
    end
    df = DataFrame(ID = repeat(1:4, inner = 3), t = repeat(0.0:2.0, 4),
        y = [0.2 + 0.05 * i + 0.03 * j for i in 1:4 for j in 0:2])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.VI(;
            turing_kwargs = (max_iter = 30, progress = false));
        rng = Random.Xoshiro(3))
    @test plot_fits(res) !== nothing
    # posterior-draw band -> _vi_drawn_params
    @test plot_fits(res; plot_mcmc_quantiles = true, mcmc_draws = 5) !== nothing
end

@testset "Observation-distribution and VPC plots" begin
    dm = DataModel(_gap_re_model(), _gap_re_df(); primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)))
    @test plot_observation_distributions(res) !== nothing
    @test plot_observation_distributions(dm) !== nothing
    @test plot_vpc(res; n_simulations = 20, seed = 7) !== nothing
end

@testset "Multivariate RE diagnostics (Laplace)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], [0.3 0.0; 0.0 0.4]); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η[1] + η[2] * t, σ)
        end
    end
    dm = DataModel(
        model, _gap_re_df(; n_ids = 8, n_obs = 3); primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)))

    # Mahalanobis standardization / scatter / pairplot need multivariate REs
    @test plot_random_effect_standardized_scatter(res) !== nothing
    @test plot_random_effect_pairplot(res) !== nothing
    @test plot_random_effects_scatter(res) !== nothing
end
