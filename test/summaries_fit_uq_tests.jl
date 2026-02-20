using Test
using NoLimits
using DataFrames
using Distributions
using Random
using Turing: MH

@testset "Fit/UQ summaries (frequentist, fixed effects only)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3; calculate_se=true)
            b = RealNumber(0.1; calculate_se=false)
            σ = RealNumber(0.5; scale=:log, calculate_se=true)
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            μ = a + b * t
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.4, 0.1, 0.5, 0.0, 0.3],
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, MLE(; optim_kwargs=(maxiters=3,)))

    s_fit = summarize(res)
    @test s_fit isa FitResultSummary
    @test s_fit.inference == :frequentist
    @test s_fit.scale == :natural
    @test s_fit.n_parameters_total == 3
    @test s_fit.n_parameters_uq_eligible == 2
    @test s_fit.n_parameters_reported == 2
    @test s_fit.n_obs_total == 6
    @test s_fit.n_missing_total == 0
    @test length(s_fit.coverage_rows) == 1
    @test occursin("Empirical Bayes", s_fit.random_effect_label) || occursin("Random effects summary", s_fit.random_effect_label)
    txt_fit = sprint(show, MIME"text/plain"(), s_fit)
    @test occursin("FitResultSummary", txt_fit)
    @test occursin("Outcome data coverage", txt_fit)

    s_fit_all = summarize(res; include_non_se=true)
    @test s_fit_all.n_parameters_reported == 3
    @test any(r -> r.parameter == :b, s_fit_all.parameter_rows)

    uq = compute_uq(res; method=:wald, n_draws=120)
    s_uq = summarize(uq)
    @test s_uq isa UQResultSummary
    @test s_uq.inference == :frequentist
    @test s_uq.interval_label == "CI"
    @test s_uq.n_parameters_reported == 2
    @test any(r -> r.parameter == :a, s_uq.parameter_rows)
    txt_uq = sprint(show, MIME"text/plain"(), s_uq)
    @test occursin("UQResultSummary", txt_uq)
    @test !occursin("Outcome data coverage", txt_uq)
    @test !occursin("Random effects summary", txt_uq)
    @test !occursin("Notes", txt_uq)

    s_comb = summarize(res, uq; include_non_se=true)
    @test s_comb isa UQResultSummary
    @test s_comb.interval_label == "CI"
    @test s_comb.n_parameters_total == 3
    @test s_comb.n_parameters_reported == 3
    row_b = only(filter(r -> r.parameter == :b, s_comb.parameter_rows))
    @test row_b.std_error === nothing
    @test row_b.lower === nothing
    @test row_b.upper === nothing
    txt_comb = sprint(show, MIME"text/plain"(), s_comb)
    @test occursin("objective", txt_comb)
    @test !occursin("Random effects summary", txt_comb)
end

@testset "Fit/UQ summaries (frequentist Laplace with random effects)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2; calculate_se=true)
            ω = RealNumber(0.5; scale=:log, calculate_se=true)
            σ = RealNumber(0.4; scale=:log, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
        end

        @formulas begin
            μ = a + exp(η)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [1.2, 1.4, 0.9, 1.0, 1.6, 1.5],
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=8,)))

    s_fit = summarize(res)
    @test s_fit isa FitResultSummary
    @test s_fit.method == :laplace
    @test s_fit.inference == :frequentist
    @test s_fit.n_parameters_total == 3
    @test s_fit.n_parameters_uq_eligible == 2
    @test s_fit.n_parameters_reported == 2
    @test s_fit.n_obs_total == 6
    @test s_fit.n_missing_total == 0
    txt_lap_fit = sprint(show, MIME"text/plain"(), s_fit)
    @test occursin("Empirical Bayes random effects summary", txt_lap_fit)
    @test !occursin("component", txt_lap_fit)

    uq = compute_uq(res; method=:wald, n_draws=120)
    s_comb = summarize(res, uq; include_non_se=true)
    @test s_comb isa UQResultSummary
    @test s_comb.inference == :frequentist
    @test s_comb.interval_label == "CI"
    @test s_comb.n_parameters_total == 3
    @test s_comb.n_parameters_uq_eligible == 2
    @test s_comb.n_parameters_reported == 3
    row_σ = only(filter(r -> r.parameter == :σ, s_comb.parameter_rows))
    @test row_σ.std_error === nothing
    @test row_σ.lower === nothing
    @test row_σ.upper === nothing
    txt_lap_comb = sprint(show, MIME"text/plain"(), s_comb)
    @test occursin("objective", txt_lap_comb)
end

@testset "Fit/UQ summaries (bayesian MCMC with random effects)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2; prior=Normal(0.0, 1.0), calculate_se=true)
            σ = RealNumber(0.4; scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.3, -0.1, 0.0],
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(
        dm,
        MCMC(;
            sampler=MH(),
            turing_kwargs=(n_samples=20, n_adapt=0, progress=false, verbose=false),
            progress=false,
        )
    )

    s_fit = summarize(res)
    @test s_fit isa FitResultSummary
    @test s_fit.inference == :bayesian
    @test occursin("Posterior random effects summary", s_fit.random_effect_label)
    @test s_fit.n_obs_total == 4
    @test s_fit.n_missing_total == 0
    txt_fit = sprint(show, MIME"text/plain"(), s_fit)
    @test occursin("objective", txt_fit)
    @test !occursin("NaN", txt_fit)

    uq = compute_uq(res; method=:chain, mcmc_draws=20)
    s_comb = summarize(res, uq)
    @test s_comb isa UQResultSummary
    @test s_comb.inference == :bayesian
    @test s_comb.interval_label == "CrI"
    @test s_comb.n_parameters_reported == 2
    txt_comb = sprint(show, MIME"text/plain"(), s_comb)
    @test occursin("CrI Lower", txt_comb)
    @test !occursin("NaN", txt_comb)
end

@testset "Fit/UQ summaries (bayesian VI with random effects)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2; prior=Normal(0.0, 1.0), calculate_se=true)
            σ = RealNumber(0.4; scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.3, -0.1, 0.0],
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, VI(; turing_kwargs=(max_iter=30, progress=false)); rng=Random.Xoshiro(710))

    s_fit = summarize(res)
    @test s_fit isa FitResultSummary
    @test s_fit.method == :vi
    @test s_fit.inference == :bayesian
    @test occursin("Posterior random effects summary", s_fit.random_effect_label)
    @test s_fit.n_obs_total == 4
    @test s_fit.n_missing_total == 0
    txt_fit = sprint(show, MIME"text/plain"(), s_fit)
    @test occursin("objective", txt_fit)
    @test !occursin("NaN", txt_fit)

    uq = compute_uq(res; method=:chain, mcmc_draws=20, rng=Random.Xoshiro(711))
    s_comb = summarize(res, uq)
    @test s_comb isa UQResultSummary
    @test s_comb.inference == :bayesian
    @test s_comb.interval_label == "CrI"
    @test s_comb.n_parameters_reported == 2
    txt_comb = sprint(show, MIME"text/plain"(), s_comb)
    @test occursin("CrI Lower", txt_comb)
    @test !occursin("NaN", txt_comb)
end
