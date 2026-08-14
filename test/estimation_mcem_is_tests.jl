using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using Turing
using Random

# One scalar-RE model/DataModel shared by all IS-variant testsets below (they
# assert e-step option behavior and diagnostics, not model structure). The
# multi-RE testset uses fx_mg_dm(); the LogNormal-RE bijection testset keeps a
# bespoke model.
const _MIS_MODEL = @Model begin
    @covariates begin
        t = Covariate()
    end
    @fixedEffects begin
        a = RealNumber(0.2)
        σ = RealNumber(0.5, scale = :log)
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0); column = :ID)
    end
    @formulas begin
        y ~ Normal(a + η, σ)
    end
end

const _MIS_DM = DataModel(_MIS_MODEL,
    DataFrame(
        ID = ["A", "A", "B", "B"],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 0.9, 1.05]);
    primary_id = :ID, time_col = :t)

@testset "MCEM_IS struct and MCEM_MCMC struct" begin
    es_mcmc = NoLimits.MCEM_MCMC()
    @test es_mcmc.sampler isa SaemixMH
    @test es_mcmc.warm_start == true
    @test es_mcmc.sample_schedule == 100

    es_is = NoLimits.MCEM_IS(n_samples = 2, proposal = :prior)
    @test es_is.n_samples == 2
    @test es_is.proposal === :prior
    @test es_is.adapt == true
    @test es_is.warm_start_mcmc_iters == 0
    @test es_is.mcmc_warmup === nothing

    es_is2 = NoLimits.MCEM_IS(
        n_samples = 2, proposal = :gaussian, warm_start_mcmc_iters = 3)
    @test es_is2.warm_start_mcmc_iters == 3
    @test es_is2.mcmc_warmup isa NoLimits.MCEM_MCMC

    # MCEM with IS e_step
    method = NoLimits.MCEM(e_step = NoLimits.MCEM_IS(n_samples = 2))
    @test method.e_step isa NoLimits.MCEM_IS
    @test method.e_step.n_samples == 2

    # Backward compat: MCEM() still creates MCEM_MCMC
    method2 = NoLimits.MCEM()
    @test method2.e_step isa NoLimits.MCEM_MCMC
end

@testset "IS prior proposal — basic fit" begin
    res = fit_model(_MIS_DM,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :prior, adapt = false),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        ))
    @test res isa NoLimits.FitResult
    @test NoLimits.get_converged(res) isa Bool
    params = NoLimits.get_params(res; scale = :untransformed)
    @test all(isfinite, collect(params))
end

@testset "IS gaussian proposal — blocks updated" begin
    res = fit_model(_MIS_DM,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :gaussian, adapt = true),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        ))
    @test res isa NoLimits.FitResult
    diag = res.result.notes.diagnostics
    # ESS recorded for IS iterations (not NaN from iter 1 once gaussian proposal is used)
    @test length(diag.ess_hist) == length(diag.Q_hist)
    # After at least 2 iterations the gaussian proposal should have n_samples > 0
    # (all ess values should be finite for the IS phase)
    @test all(isfinite, diag.ess_hist)
end

@testset "IS user-provided proposal function" begin
    # User proposal: sample from N(0, 2) for all entries, return correct shapes
    function my_proposal_is_test(θ, batch_info, re_dists, rng, n_samples)
        nb = batch_info.n_b
        samples = randn(rng, nb, n_samples) .* 2.0
        # log q = sum of Normal(0, 2) logpdfs
        log_qs = vec(sum(logpdf.(Normal(0.0, 2.0), samples); dims = 1))
        return samples, log_qs
    end

    res = fit_model(_MIS_DM,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = my_proposal_is_test),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        ))
    @test res isa NoLimits.FitResult
    @test NoLimits.get_converged(res) isa Bool
    params = NoLimits.get_params(res; scale = :untransformed)
    @test all(isfinite, collect(params))
end

@testset "IS warm_start_mcmc_iters — MCMC then IS" begin
    es = NoLimits.MCEM_IS(
        n_samples = 2,
        proposal = :gaussian,
        adapt = true,
        warm_start_mcmc_iters = 2,
        mcmc_warmup = NoLimits.MCEM_MCMC(
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            sample_schedule = 10
        )
    )
    res = fit_model(_MIS_DM,
        NoLimits.MCEM(
            e_step = es,
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        ))
    @test res isa NoLimits.FitResult
    diag = res.result.notes.diagnostics
    # First 2 iterations are MCMC (ess = NaN), rest are IS (ess finite)
    @test isnan(diag.ess_hist[1])
    @test isnan(diag.ess_hist[2])
    @test all(isfinite, diag.ess_hist[3:end])
end

@testset "IS weights are finite and normalized" begin
    res = fit_model(_MIS_DM,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :prior),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        ))
    diag = res.result.notes.diagnostics
    # ESS must be in [1, n_samples] for IS iters
    for ess in diag.ess_hist
        if isfinite(ess)
            @test ess >= 1.0
            @test ess <= 50.0 + 1e-6   # small tolerance for float arithmetic
        end
    end
end

@testset "IS ESS tracked in diagnostics" begin
    res = fit_model(_MIS_DM,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :prior),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        ))
    diag = res.result.notes.diagnostics
    @test length(diag.ess_hist) == length(diag.Q_hist)
    @test all(isfinite, diag.ess_hist)  # pure IS: all finite
end

@testset "IS with multi-RE model" begin
    res = fit_model(fx_mg_dm(),
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :prior),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        ))
    @test res isa NoLimits.FitResult
    @test NoLimits.get_converged(res) isa Bool
    params = NoLimits.get_params(res; scale = :untransformed)
    @test all(isfinite, collect(params))
end

@testset "IS LogNormal RE — bijection applied" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(1.0, scale = :log)
            σ = RealNumber(0.3, scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(LogNormal(0.0, 0.5); column = :ID)
        end
        @formulas begin
            y ~ Normal(a * η, σ)
        end
    end

    df = DataFrame(
        ID = ["A", "A", "B", "B"],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.1, 0.9, 1.3, 1.2]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res = fit_model(dm,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :gaussian, adapt = true),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        ))
    @test res isa NoLimits.FitResult
    diag = res.result.notes.diagnostics
    @test all(isfinite, diag.ess_hist)
    params = NoLimits.get_params(res; scale = :untransformed)
    @test all(isfinite, collect(params))
end

@testset "IS backward compat — MCEM() legacy kwargs still work" begin
    # Old API: sampler= and turing_kwargs= at the top level
    method = NoLimits.MCEM(
        sampler = MH(),
        turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
        maxiters = 2,
        consecutive_params = 1,
        progress = false
    )
    @test method.e_step isa NoLimits.MCEM_MCMC
    @test method.e_step.sampler isa MH

    res = fit_model(_MIS_DM, method)
    @test res isa NoLimits.FitResult
    @test NoLimits.get_converged(res) isa Bool
end
