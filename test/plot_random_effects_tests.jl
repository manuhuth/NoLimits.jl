using Test
using NoLimits
using CairoMakie
using DataFrames
using Distributions
using Random
using Turing

# ── Shared fixtures: build each unique model type once so testsets share JIT compilation ──

# Group 1: Normal RE + Age constant covariate, 6 individuals
# Used by: "plot_random_effects Laplace" and "plot_random_effects MCMC"
const _PRE_NORM_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(0.1, prior = Uniform(0.0, 0.9))
        σ = RealNumber(0.3, scale = :log, prior = Uniform(0.01, 1.0))
    end
    @covariates begin
        t = Covariate()
        Age = ConstantCovariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, 0.5); column = :ID)
    end
    @formulas begin
        y ~ Normal(a + η, σ)
    end
end
const _PRE_NORM_DF = DataFrame(
    ID = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    Age = [30.0, 30.0, 40.0, 40.0, 35.0, 35.0, 45.0, 45.0, 28.0, 28.0, 55.0, 55.0],
    y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05, 0.2, 0.3, -0.1, 0.0])
const _PRE_NORM_DM = DataModel(
    _PRE_NORM_MODEL, _PRE_NORM_DF; primary_id = :ID, time_col = :t)
const _PRE_NORM_RES_LAP = fit_model(
    _PRE_NORM_DM, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))

# Group 2: Simple Normal RE, no extra covariates
# Used by: "single-level RE", "constants_re Laplace", "constants_re MCMC"
const _PRE_SIMPLE_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(0.1, prior = Uniform(0.0, 0.9))
        σ = RealNumber(0.3, scale = :log, prior = Uniform(0.01, 1.0))
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
const _PRE_CONST_RE_DF = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C, :D, :D],
    t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05])
const _PRE_CONST_RE_DM = DataModel(
    _PRE_SIMPLE_MODEL, _PRE_CONST_RE_DF; primary_id = :ID, time_col = :t)

# NPF (1D and 2D) models and fits come from the shared fixtures
# (fx_npf_*/fx_npf2_* in fixtures.jl), shared with the estimation tests.

# One MCMC fit of _PRE_NORM_DM with enough samples for the draws/warmup kwargs,
# shared by both MCMC testsets below.
const _PRE_NORM_RES_MCMC = fit_model(_PRE_NORM_DM,
    NoLimits.MCMC(; turing_kwargs = (n_samples = 10, n_adapt = 5, progress = false)))

@testset "plot_random_effects Laplace" begin
    dm = _PRE_NORM_DM
    res = _PRE_NORM_RES_LAP

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std = plot_random_effect_standardized(res, show_hist = true)
    @test p_std !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(
        res; show_hist = true, show_kde = false, show_qq = false)
    @test p_pit_hist !== nothing

    p_pit_kde = plot_random_effect_pit(
        res; show_hist = false, show_kde = true, show_qq = false)
    @test p_pit_kde !== nothing

    p_pit_qq = plot_random_effect_pit(
        res; show_hist = false, show_kde = false, show_qq = true)
    @test p_pit_qq !== nothing

    @test_throws ArgumentError plot_random_effect_pit(res; x_covariate = :Age)

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
        plot_random_effect_distributions(res; plot_path = p_dist_path)
        plot_random_effect_pit(res; show_hist = true, show_kde = false,
            show_qq = false, plot_path = p_pit_path)
        plot_random_effect_standardized(res; show_hist = true, plot_path = p_std_path)
        plot_random_effect_standardized_scatter(res; plot_path = p_std_sc_path)
        plot_random_effects_pdf(res; plot_path = p_pdf_path)
        plot_random_effects_scatter(res; plot_path = p_scatter_path)
        plot_random_effect_pairplot(res; plot_path = p_pair_path)
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
            a = RealNumber(0.1, prior = Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale = :log, prior = Uniform(0.01, 1.0))
        end

        @covariates begin
            t = Covariate()
            Age = ConstantCovariate(constant_on = :ID)
            Center = ConstantCovariate(constant_on = :Center)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.5); column = :ID)
            ζ = RandomEffect(Normal(0.0, 0.2); column = :Center)
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

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std = plot_random_effect_standardized(res)
    @test p_std !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res; x_covariate = :Age)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(
        res; show_hist = true, show_kde = false, show_qq = false)
    @test p_pit_hist !== nothing
end

@testset "plot_random_effects Laplace non-normal RE" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior = Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale = :log, prior = Uniform(0.01, 1.0))
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(LogNormal(0.0, 0.4); column = :ID)
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

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_pit_hist = plot_random_effect_pit(
        res; show_hist = true, show_kde = false, show_qq = false)
    @test p_pit_hist !== nothing
end

@testset "plot_random_effects Laplace single-level RE" begin
    dm = DataModel(_PRE_SIMPLE_MODEL,
        DataFrame(ID = [1, 1, 1, 1], t = [0.0, 1.0, 2.0, 3.0], y = [0.1, 0.2, 0.15, 0.25]);
        primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std = plot_random_effect_standardized(res)
    @test p_std !== nothing
end

@testset "plot_random_effects Laplace constants_re" begin
    dm = _PRE_CONST_RE_DM
    constants_re = (; η = (; B = 0.0))
    res = fit_model(
        dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)); constants_re = constants_re)

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std = plot_random_effect_standardized(res)
    @test p_std !== nothing
end

@testset "plot_random_effects Laplace multivariate Normal RE" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior = Uniform(0.0, 0.9))
            σ = RealNumber(0.3, scale = :log, prior = Uniform(0.01, 1.0))
            μ = RealVector([0.0, 0.0], prior = filldist(Uniform(-1.0, 1.0), 2))
            Ω = RealPSDMatrix(Matrix(I, 2, 2), scale = :cholesky,
                prior = InverseWishart(3, Matrix(I, 2, 2)))
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η_mv = RandomEffect(MvNormal(μ, Ω); column = :ID)
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

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std = plot_random_effect_standardized(res)
    @test p_std !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(
        res; show_hist = true, show_kde = false, show_qq = false)
    @test p_pit_hist !== nothing
end

@testset "plot_random_effects MCMC" begin
    res = _PRE_NORM_RES_MCMC

    p_pit_hist = plot_random_effect_pit(
        res; mcmc_draws = 5, show_hist = true, show_kde = false, show_qq = false)
    @test p_pit_hist !== nothing

    p_pit_kde = plot_random_effect_pit(
        res; mcmc_draws = 5, show_hist = false, show_kde = true, show_qq = false)
    @test p_pit_kde !== nothing

    p_pit_qq = plot_random_effect_pit(
        res; mcmc_draws = 5, show_hist = false, show_kde = false, show_qq = true)
    @test p_pit_qq !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res)
    @test p_std_sc !== nothing

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_pdf = plot_random_effects_pdf(res; mcmc_draws = 5)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res; mcmc_draws = 5)
    @test p_scatter !== nothing

    p_pari = plot_random_effect_pairplot(res; mcmc_draws = 5)
    @test p_pari !== nothing
end

@testset "plot_random_effects MCMC draws and warmup kwargs" begin
    res = _PRE_NORM_RES_MCMC

    p_pit_hist = plot_random_effect_pit(res; mcmc_draws = 6, mcmc_warmup = 3,
        show_hist = true, show_kde = false, show_qq = false)
    @test p_pit_hist !== nothing

    p_dist = plot_random_effect_distributions(res; mcmc_draws = 6, mcmc_warmup = 3)
    @test p_dist !== nothing

    p_pdf = plot_random_effects_pdf(res; mcmc_draws = 6, mcmc_warmup = 3)
    @test p_pdf !== nothing

    p_scatter = plot_random_effects_scatter(res; mcmc_draws = 6)
    @test p_scatter !== nothing
end

@testset "plot_random_effects MCMC constants_re" begin
    dm = _PRE_CONST_RE_DM
    constants_re = (; η = (; B = 0.0))
    res = fit_model(
        dm, NoLimits.MCMC(; turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false));
        constants_re = constants_re)

    p_dist = plot_random_effect_distributions(
        res; flow_plot = :kde, flow_samples = 200, mcmc_draws = 5)
    @test p_dist !== nothing

    p_dist_hist = plot_random_effect_distributions(
        res; flow_plot = :hist, flow_samples = 200, flow_bins = 10, mcmc_draws = 5)
    @test p_dist_hist !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(
        res; mcmc_draws = 5, show_hist = true, show_kde = false, show_qq = false)
    @test p_pit_hist !== nothing
end

@testset "plot_random_effects Laplace NormalizingPlanarFlow RE" begin
    res = fx_npf_laplace()

    p_dist = plot_random_effect_distributions(res; flow_plot = :kde, flow_samples = 50)
    @test p_dist !== nothing

    p_dist_hist = plot_random_effect_distributions(
        res; flow_plot = :hist, flow_samples = 50, flow_bins = 10)
    @test p_dist_hist !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res, flow_samples = 50)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(
        res; show_hist = true, show_kde = false, show_qq = false, flow_samples = 50)
    @test p_pit_hist !== nothing
end

@testset "plot_random_effects MCMC multivariate NormalizingPlanarFlow RE" begin
    res = fx_npf2_mcmc()

    p_dist = plot_random_effect_distributions(
        res; flow_plot = :kde, flow_samples = 50, mcmc_draws = 5)
    @test p_dist !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res; flow_samples = 50)
    @test p_std_sc !== nothing
end

@testset "plot_random_effects Laplace RE with constant covariates" begin
    res = fx_recov_laplace()

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res; x_covariate = :Age)
    @test p_std_sc !== nothing
end

@testset "plot_random_effects MCMC RE with constant covariates" begin
    res = fx_recov_mcmc()

    p_dist = plot_random_effect_distributions(res)
    @test p_dist !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res; x_covariate = :Age)
    @test p_std_sc !== nothing
end

@testset "plot_random_effects MCMC NormalizingPlanarFlow RE" begin
    res = fx_npf_mcmc()

    p_dist = plot_random_effect_distributions(
        res; flow_plot = :kde, flow_samples = 50, mcmc_draws = 5)
    @test p_dist !== nothing

    p_dist_hist = plot_random_effect_distributions(
        res; flow_plot = :hist, flow_samples = 50, flow_bins = 10, mcmc_draws = 5)
    @test p_dist_hist !== nothing

    p_std = plot_random_effect_standardized(res; flow_samples = 50)
    @test p_std !== nothing

    p_std_sc = plot_random_effect_standardized_scatter(res; flow_samples = 50)
    @test p_std_sc !== nothing

    p_pit_hist = plot_random_effect_pit(res; mcmc_draws = 5, show_hist = true,
        show_kde = false, show_qq = false, flow_samples = 50)
    @test p_pit_hist !== nothing

    p_pit_kde = plot_random_effect_pit(res; mcmc_draws = 5, show_hist = false,
        show_kde = true, show_qq = false, flow_samples = 50)
    @test p_pit_kde !== nothing

    p_pit_qq = plot_random_effect_pit(res; mcmc_draws = 5, show_hist = false,
        show_kde = false, show_qq = true, flow_samples = 50)
    @test p_pit_qq !== nothing
end

@testset "plot_random_effects MLE error" begin
    @test_throws ErrorException plot_random_effect_distributions(fx_mle())
end

# ── Merged from random_effect_new_plots_tests.jl ──────────────────────────────
# One rich multi-RE model (scalar + multivariate on :ID, scalar on :Center) for
# the pdf/scatter/pairplot trio; the flow trio uses the fx_npf2 Laplace fixture.
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

const _RNP_TRIO = (
    plot_random_effects_pdf, plot_random_effects_scatter, plot_random_effect_pairplot)

@testset "random effects new plots Laplace (multi-id, MVN)" begin
    res = fit_model(_RNP_DM,
        NoLimits.Laplace(; use_hutchinson = false, optim_kwargs = (maxiters = 2,)))
    for f in _RNP_TRIO
        @test f(res) !== nothing
    end
end

@testset "random effects new plots MCMC (multi-id, MVN)" begin
    res = fit_model(_RNP_DM,
        NoLimits.MCMC(; sampler = NUTS(5, 0.3),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false)))
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

# ── Moved from coverage_gap_tests.jl ──────────────────────────────────────────
# Prior-equipped scalar-RE model (fixed RE sd) for posterior-prediction paths.
const _PRE_GAP_PRIOR_MODEL = @Model begin
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

@testset "RE posterior prediction plots (MCMC)" begin
    dm = DataModel(_PRE_GAP_PRIOR_MODEL, fx_re_df(); primary_id = :ID, time_col = :t)

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
    df = DataFrame(ID = ids, t = tt,
        y = [0.2 + 0.05 * i + 0.03 * j for (i, j) in zip(ids, tt)])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)))

    # Mahalanobis standardization / scatter / pairplot need multivariate REs
    @test plot_random_effect_standardized_scatter(res) !== nothing
    @test plot_random_effect_pairplot(res) !== nothing
    @test plot_random_effects_scatter(res) !== nothing
end
