using Test
using NoLimits
using CairoMakie
using DataFrames
using Distributions
using Turing

# Second half of the random-effects plotting tests (split from
# plot_random_effects_tests.jl for CI shard balance; the halves run in
# different lanes and are self-contained — this one carries the merged
# random_effect_new_plots content and reuses the fx_npf2/fx_mcmc_re fixtures).
# ── Merged from random_effect_new_plots_tests.jl ──────────────────────────────
# One rich multi-RE model (scalar + multivariate on :ID, scalar on :Center) for
# the pdf/scatter/pairplot trio; the flow trio uses the fx_npf2 Laplace fixture.
function _rnp_df(; n_ids::Int = 10, n_obs_per::Int = 2)
    ids = [Symbol("ID", i) for i in 1:n_ids]
    ID = repeat(ids, inner = n_obs_per)
    t = repeat(collect(0.0:(n_obs_per - 1)), n_ids)
    Center = repeat(
        vcat(fill(:C1, n_ids ÷ 2), fill(:C2, n_ids - n_ids ÷ 2)), inner = n_obs_per
    )
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

const _RNP_TRIO = (
    plot_random_effects_pdf, plot_random_effects_scatter, plot_random_effect_pairplot,
)

@testset "random effects new plots Laplace (multi-id, MVN)" begin
    res = fit_model(
        _RNP_DM,
        NoLimits.Laplace(; use_hutchinson = false, optim_kwargs = (maxiters = 2,))
    )
    for f in _RNP_TRIO
        @test f(res) !== nothing
    end
end

@testset "random effects new plots MCMC (multi-id, MVN)" begin
    res = fit_model(
        _RNP_DM,
        NoLimits.MCMC(;
            sampler = NUTS(5, 0.3),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false)
        )
    )
    for f in _RNP_TRIO
        @test f(res; mcmc_draws = 5) !== nothing
    end
end

@testset "random effects new plots Laplace flow dim2" begin
    res = fx_npf2_laplace()

    @test plot_random_effects_pdf(res; flow_samples = 50, flow_plot = :hist) !== nothing
    @test plot_random_effects_scatter(res) !== nothing
    @test plot_random_effect_pairplot(res) !== nothing
end

@testset "RE posterior prediction plots (MCMC)" begin
    # fx_mcmc_re is a scalar-RE MCMC fit (priors, 20 samples) — enough for the
    # default-mean and posterior-draw paths below.
    res_mcmc = fx_mcmc_re()
    # default path -> _mcmc_random_effects_means
    @test plot_fits(res_mcmc) !== nothing
    # posterior-draw path -> _mcmc_drawn_params
    @test plot_fits(res_mcmc; plot_mcmc_quantiles = true, mcmc_draws = 5) !== nothing
    # RE diagnostics on a posterior fit
    @test plot_random_effect_distributions(res_mcmc) !== nothing
    @test plot_random_effect_pit(res_mcmc) !== nothing
    @test plot_random_effect_standardized(res_mcmc) !== nothing
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
    ids = repeat(1:8, inner = 3)
    tt = repeat(collect(0.0:2.0), 8)
    df = DataFrame(
        ID = ids, t = tt,
        y = [0.2 + 0.05 * i + 0.03 * j for (i, j) in zip(ids, tt)]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)))

    # Mahalanobis standardization / scatter / pairplot need multivariate REs
    @test plot_random_effect_standardized_scatter(res) !== nothing
    @test plot_random_effect_pairplot(res) !== nothing
    @test plot_random_effects_scatter(res) !== nothing
end
