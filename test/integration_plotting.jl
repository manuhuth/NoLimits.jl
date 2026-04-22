using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using Random
using Plots

# ─────────────────────────────────────────────────────────────────────────────
# Integration file: plotting, caching, VPC, residuals, observation distributions.
# Absorbed from: plotting_functions_tests.jl, plot_cache_tests.jl,
#   vpc_tests.jl, residual_plots_tests.jl, plot_observation_distributions_tests.jl
#
# Relies on fixtures from integration_no_re.jl and integration_simple_re.jl,
# which must be included first in runtests.jl:
#   _NRE_DM, _PLT_DM_P, _NRE_RES_MLE, _PLT_RES_MCMC, _PLT_RES_VI
#   _PLT_RE_DM, _PLT_RE_RES_LAP
# ─────────────────────────────────────────────────────────────────────────────

# ── shared fixtures (all defined locally so this file can run standalone) ─────

# Simple no-RE model with priors — for MCMC and VI plot tests
const _PLT_DF_P = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])
const _PLT_DM_P = DataModel(
    @Model(begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.1; prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log; prior=LogNormal(0.0, 0.5))
        end
        @formulas begin; y ~ Normal(a, σ); end
    end),
    _PLT_DF_P; primary_id=:ID, time_col=:t)

const _PLT_RES_MCMC = fit_model(_PLT_DM_P,
    NoLimits.MCMC(; turing_kwargs=(n_samples=2, n_adapt=2, progress=false));
    rng=MersenneTwister(42))
const _PLT_RES_VI = fit_model(_PLT_DM_P,
    NoLimits.VI(; turing_kwargs=(max_iter=10, progress=false));
    rng=Random.Xoshiro(1))

# Simple RE model for RE plot tests
const _PLT_RE_DF = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])
const _PLT_RE_DM = DataModel(
    @Model(begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @randomEffects begin; η = RandomEffect(Normal(0.0, 1.0); column=:ID); end
        @formulas begin; y ~ Normal(a + η, σ); end
    end),
    _PLT_RE_DF; primary_id=:ID, time_col=:t)
const _PLT_RE_RES_LAP = fit_model(_PLT_RE_DM,
    NoLimits.Laplace(; optim_kwargs=(maxiters=2,), multistart_n=2, multistart_k=2))

# a + b*z model — the "standard plotting fixture" used by most plot/vpc/residual tests
const _PLOT_DF = DataFrame(
    ID = [1, 1, 2, 2],
    t  = [0.0, 1.0, 0.0, 1.0],
    z  = [0.1, 0.2, 0.1, 0.2],
    y  = [0.15, 0.18, 0.14, 0.19]
)
const _PLOT_DM = DataModel(
    @Model(begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(-0.2)
            σ = RealNumber(0.3, scale=:log)
        end
        @covariates begin; t = Covariate(); z = Covariate(); end
        @formulas begin; y ~ Normal(a + b * z, σ); end
    end),
    _PLOT_DF; primary_id=:ID, time_col=:t)

const _PLOT_RES_MLE = fit_model(_PLOT_DM, NoLimits.MLE())

# Simple a,σ Normal model — used by plot_observation_distributions basic tests
const _OBS_DF = DataFrame(ID=[1, 1], t=[0.0, 1.0], y=[0.1, 0.2])
const _OBS_DM = DataModel(
    @Model(begin
        @fixedEffects begin; a = RealNumber(0.1); σ = RealNumber(0.3, scale=:log); end
        @covariates begin; t = Covariate(); end
        @formulas begin; y ~ Normal(a, σ); end
    end),
    _OBS_DF; primary_id=:ID, time_col=:t)

const _OBS_RES_MLE = fit_model(_OBS_DM, NoLimits.MLE())

# ══════════════════════════════════════════════════════════════════════════════
# plot_data / plot_fits  (from plotting_functions_tests.jl)
# ══════════════════════════════════════════════════════════════════════════════

@testset "plot_data and plot_fits basic" begin
    p_data  = plot_data(_PLOT_RES_MLE)
    @test p_data !== nothing

    p_fits  = plot_fits(_PLOT_RES_MLE)
    @test p_fits !== nothing

    p_fits_dm  = plot_fits(_PLOT_DM)
    @test p_fits_dm !== nothing

    p_data_dm  = plot_data(_PLOT_DM)
    @test p_data_dm !== nothing

    p_data_z   = plot_data(_PLOT_RES_MLE, x_axis_feature=:z)
    @test p_data_z !== nothing

    p_fits_z   = plot_fits(_PLOT_RES_MLE, x_axis_feature=:z)
    @test p_fits_z !== nothing

    p_fits_dm_z = plot_fits(_PLOT_DM, x_axis_feature=:z)
    @test p_fits_dm_z !== nothing

    p_data_dm_z = plot_data(_PLOT_DM, x_axis_feature=:z)
    @test p_data_dm_z !== nothing

    mktempdir() do tmp
        data_path = joinpath(tmp, "plot_data.png")
        fits_path = joinpath(tmp, "plot_fits.png")
        plot_data(_PLOT_RES_MLE; plot_path=data_path)
        plot_fits(_PLOT_RES_MLE; plot_path=fits_path)
        @test isfile(data_path)
        @test isfile(fits_path)
        @test_throws ErrorException plot_data(_PLOT_RES_MLE;
            save_path=joinpath(tmp,"a.png"), plot_path=joinpath(tmp,"b.png"))
    end

    @test_throws ErrorException plot_fits(_PLOT_RES_MLE; x_axis_feature=:missing_feature)
end

@testset "plot_fits MCMC" begin
    # _PLT_RES_MCMC has 5 kept draws; reduce mcmc_draws accordingly
    p = plot_fits(_PLT_RES_MCMC; mcmc_draws=5, plot_mcmc_quantiles=true,
                  mcmc_quantiles=[5, 95], mcmc_quantiles_alpha=0.8)
    @test p !== nothing
end

@testset "plot_fits VI" begin
    p = plot_fits(_PLT_RES_VI; mcmc_draws=5, plot_mcmc_quantiles=true,
                  mcmc_quantiles=[5, 95], mcmc_quantiles_alpha=0.8)
    @test p !== nothing
end

# ══════════════════════════════════════════════════════════════════════════════
# build_plot_cache  (from plot_cache_tests.jl)
# ══════════════════════════════════════════════════════════════════════════════

@testset "Plot cache (non-ODE)" begin
    cache = build_plot_cache(_PLOT_RES_MLE; cache_obs_dists=true)
    @test cache isa PlotCache
    @test length(cache.obs_dists) == length(get_individuals(_PLOT_DM))
    @test length(cache.obs_dists[1]) == length(get_row_groups(_PLOT_DM).obs_rows[1])
end

@testset "Plot cache (RE non-ODE, Laplace)" begin
    cache = build_plot_cache(_PLT_RE_RES_LAP; cache_obs_dists=false)
    @test cache isa PlotCache
    @test length(cache.params) > 0
end

# ══════════════════════════════════════════════════════════════════════════════
# plot_vpc  (from vpc_tests.jl)
# ══════════════════════════════════════════════════════════════════════════════

@testset "plot_vpc basic" begin
    p_vpc   = plot_vpc(_PLOT_RES_MLE; x_axis_feature=:z, n_simulations=5, n_bins=2)
    @test p_vpc !== nothing

    p_vpc_t = plot_vpc(_PLOT_RES_MLE; n_simulations=5, n_bins=2)
    @test p_vpc_t !== nothing

    p_vpc_alias = plot_vpc(_PLOT_RES_MLE; n_sim=5, n_bins=2)
    @test p_vpc_alias !== nothing

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_vpc.png")
        plot_vpc(_PLOT_RES_MLE; n_simulations=5, n_bins=2, plot_path=p_path)
        @test isfile(p_path)
    end
end

@testset "plot_vpc validation errors" begin
    @test_throws ErrorException plot_vpc(_PLOT_RES_MLE; n_sim=0)
    @test_throws ErrorException plot_vpc(_PLOT_RES_MLE; n_simulations=0)
    @test_throws ErrorException plot_vpc(_PLOT_RES_MLE; n_simulations=6, n_sim=5)
    @test_throws ErrorException plot_vpc(_PLOT_RES_MLE; percentiles=[50])
    @test_throws ErrorException plot_vpc(_PLOT_RES_MLE; percentiles=[-5, 50, 95])
    @test_throws ErrorException plot_vpc(_PLOT_RES_MLE; n_bins=0)
    @test_throws ErrorException plot_vpc(_PLOT_RES_MLE; obs_percentiles_method=:invalid)
    @test_throws ErrorException plot_vpc(_PLOT_RES_MLE; obs_percentiles_mode=:invalid)
    @test_throws ErrorException plot_vpc(_PLOT_RES_MLE;
        obs_percentiles_mode=:per_individual, obs_percentiles_method=:kernel)
end

# ══════════════════════════════════════════════════════════════════════════════
# get_residuals / plot_residuals  (from residual_plots_tests.jl)
# ══════════════════════════════════════════════════════════════════════════════

@testset "residual plots basic API (FitResult + DataModel + cache)" begin
    rdf = get_residuals(_PLOT_RES_MLE)
    cols = Set(Symbol.(names(rdf)))
    @test nrow(rdf) == nrow(_PLOT_DF)
    @test :pit in cols
    @test :res_quantile in cols
    @test :res_raw in cols
    @test :res_pearson in cols
    @test :logscore in cols
    @test :observable in cols
    @test :n_draws in cols
    @test all(rdf.n_draws .== 1)

    @test plot_residuals(_PLOT_RES_MLE; residual=:quantile) !== nothing
    @test plot_residual_distribution(_PLOT_RES_MLE; residual=:pit) !== nothing
    @test plot_residual_qq(_PLOT_RES_MLE; residual=:quantile) !== nothing
    @test plot_residual_pit(_PLOT_RES_MLE; show_hist=true, show_qq=false) !== nothing
    @test plot_residual_acf(_PLOT_RES_MLE; residual=:raw, max_lag=1) !== nothing

    cache = build_plot_cache(_PLOT_DM; cache_obs_dists=false)
    rdf_dm = get_residuals(_PLOT_DM; cache=cache, cache_obs_dists=true)
    @test nrow(rdf_dm) == nrow(_PLOT_DF)
    @test plot_residuals(_PLOT_DM; cache=cache, cache_obs_dists=true) !== nothing

    mktempdir() do tmp
        for (f, kw) in [
            (plot_residuals, (residual=:quantile,)),
            (plot_residual_distribution, (residual=:pit,)),
            (plot_residual_qq, (residual=:quantile,)),
            (plot_residual_pit, (show_hist=true, show_qq=false)),
            (plot_residual_acf, (residual=:raw, max_lag=1)),
        ]
            p_path = joinpath(tmp, "$(f).png")
            f(_PLOT_RES_MLE; kw..., plot_path=p_path)
            @test isfile(p_path)
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# plot_observation_distributions  (from plot_observation_distributions_tests.jl)
# ══════════════════════════════════════════════════════════════════════════════

@testset "plot_observation_distributions basic" begin
    p = plot_observation_distributions(_OBS_RES_MLE;
        individuals_idx=1, obs_rows=1, observables=:y)
    @test p !== nothing

    p_dm = plot_observation_distributions(_OBS_DM;
        individuals_idx=1, obs_rows=1, observables=:y)
    @test p_dm !== nothing

    mktempdir() do tmp
        p_path = joinpath(tmp, "plot_observation_distributions.png")
        plot_observation_distributions(_OBS_RES_MLE;
            individuals_idx=1, obs_rows=1, observables=:y, plot_path=p_path)
        @test isfile(p_path)
    end
end

@testset "plot_observation_distributions caching" begin
    cache = build_plot_cache(_OBS_RES_MLE; cache_obs_dists=true)

    p_cache = plot_observation_distributions(_OBS_RES_MLE;
        individuals_idx=1, obs_rows=1, observables=:y,
        cache=cache, cache_obs_dists=true)
    @test p_cache !== nothing

    p_no_cache = plot_observation_distributions(_OBS_RES_MLE;
        individuals_idx=1, obs_rows=1, observables=:y)
    @test p_no_cache !== nothing
end

@testset "plot_observation_distributions MCMC" begin
    # _PLT_RES_MCMC has 5 kept draws
    p = plot_observation_distributions(_PLT_RES_MCMC;
        individuals_idx=1, obs_rows=1, observables=:y, mcmc_draws=5)
    @test p !== nothing
end

@testset "plot_observation_distributions VI" begin
    p = plot_observation_distributions(_PLT_RES_VI;
        individuals_idx=1, obs_rows=1, observables=:y, mcmc_draws=5)
    @test p !== nothing
end

@testset "plot_observation_distributions random effects fitted" begin
    # Reuse shared Laplace RE result — same simple RE model
    p = plot_observation_distributions(_PLT_RE_RES_LAP;
        individuals_idx=[1, 2], obs_rows=1, observables=:y)
    @test p !== nothing
end
