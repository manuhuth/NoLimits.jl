using Test
using NoLimits
using DataFrames
using Distributions
using Turing: MH

@testset "Fit/UQ summaries (frequentist, fixed effects only)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3; calculate_se = true)
            b = RealNumber(0.1; calculate_se = false)
            σ = RealNumber(0.5; scale = :log, calculate_se = true)
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
        y = [0.2, 0.4, 0.1, 0.5, 0.0, 0.3]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, MLE(; optim_kwargs = (maxiters = 2,)))

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
    @test occursin("Empirical Bayes", s_fit.random_effect_label) ||
        occursin("Random effects summary", s_fit.random_effect_label)
    txt_fit = sprint(show, MIME"text/plain"(), s_fit)
    @test occursin("FitResultSummary", txt_fit)
    @test occursin("Outcome data coverage", txt_fit)

    s_fit_all = summarize(res; include_non_se = true)
    @test s_fit_all.n_parameters_reported == 3
    @test any(r -> r.parameter == :b, s_fit_all.parameter_rows)

    uq = compute_uq(res; method = :wald, n_draws = 30)
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

    s_comb = summarize(res, uq; include_non_se = true)
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
            a = RealNumber(0.2; calculate_se = true)
            ω = RealNumber(0.5; scale = :log, calculate_se = true)
            σ = RealNumber(0.4; scale = :log, calculate_se = false)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end

        @formulas begin
            μ = a + exp(η)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [1.2, 1.4, 0.9, 1.0, 1.6, 1.5]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))

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

    uq = compute_uq(res; method = :wald, n_draws = 30)
    s_comb = summarize(res, uq; include_non_se = true)
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
            a = RealNumber(0.2; prior = Normal(0.0, 1.0), calculate_se = true)
            σ = RealNumber(
                0.4; scale = :log, prior = LogNormal(0.0, 0.5), calculate_se = true
            )
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.3, -0.1, 0.0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(
        dm,
        MCMC(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false, verbose = false),
            progress = false
        )
    )

    s_fit = summarize(res)
    @test s_fit isa FitResultSummary
    @test s_fit.method == :mcmc
    @test s_fit.inference == :bayesian
    @test occursin("Posterior random effects summary", s_fit.random_effect_label)
    @test s_fit.n_obs_total == 4
    @test s_fit.n_missing_total == 0
    txt_fit = sprint(show, MIME"text/plain"(), s_fit)
    @test occursin("objective", txt_fit)
    @test !occursin("NaN", txt_fit)

    # Explicit mcmc_warmup exercises the user-supplied warmup path (clamped to chain).
    uq = compute_uq(res; method = :chain, mcmc_draws = 20, mcmc_warmup = 5)
    s_comb = summarize(res, uq)
    @test s_comb isa UQResultSummary
    @test s_comb.inference == :bayesian
    @test s_comb.interval_label == "CrI"
    @test s_comb.n_parameters_reported == 2
    txt_comb = sprint(show, MIME"text/plain"(), s_comb)
    @test occursin("CrI Lower", txt_comb)
    @test !occursin("NaN", txt_comb)
end

@testset "natural-scale coords match the flat layout for every block kind" begin
    fe = @fixedEffects begin
        a = RealNumber(1.0)
        b = RealVector([0.5, 2.0], scale = [:identity, :log])
        Ωc = RealPSDMatrix([1.0 0.2; 0.2 1.0], scale = :cholesky)
        Ωe = RealPSDMatrix([1.0 0.1; 0.1 2.0], scale = :expm)
        Ωl = RealLiePSDMatrix([1.0 0.0; 0.0 2.0], scale = :lie)
        D = RealDiagonalMatrix([1.0, 2.0], scale = :log)
        p = ProbabilityVector([0.3, 0.3, 0.4])
        P = DiscreteTransitionMatrix([0.7 0.3; 0.4 0.6])
        Q = ContinuousTransitionMatrix([-0.5 0.5; 0.2 -0.2])
    end

    θu = NoLimits.get_θ0_untransformed(fe)
    θt = NoLimits.get_transform(fe)(θu)
    parents = NoLimits._flat_parent_names(fe)
    spec_map = NoLimits._spec_map(fe)
    for name in NoLimits.get_names(fe)
        spec = spec_map[name]
        n_nat = length(
            NoLimits._coords_for_param(
                getproperty(θu, name), spec; natural = true
            )
        )
        n_tr = length(
            NoLimits._coords_for_param(
                getproperty(θt, name), spec; natural = false
            )
        )
        @test n_nat == n_tr == count(==(name), parents)
    end
    # :cholesky reports the lower triangle column-major, matching `get_flat_names`.
    @test NoLimits._coords_for_param(θu.Ωc, spec_map[:Ωc]; natural = true) ==
        [1.0, 0.2, 1.0]
end

@testset "summarize works with a matrix parameter block" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            Ω = RealPSDMatrix([1.0 0.2; 0.2 1.0], scale = :cholesky, calculate_se = true)
            σ = RealNumber(0.4; scale = :log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(MvNormal(zeros(2), Ω); column = :ID)
        end

        @formulas begin
            y ~ Normal(a + η[1] + η[2] * t, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [1.2, 1.4, 0.9, 1.0, 1.6, 1.5]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))

    s_fit = summarize(res)
    @test s_fit isa FitResultSummary
    @test s_fit.n_parameters_total == 5   # a, Ω (3 lower-tri coords), σ
    rows = s_fit.parameter_rows
    @test [r.parameter for r in rows if startswith(string(r.parameter), "Ω")] ==
        [:Ω_1_1, :Ω_2_1, :Ω_2_2]
    Ω_hat = NoLimits.get_params(res; scale = :untransformed).Ω
    @test [r.estimate for r in rows if startswith(string(r.parameter), "Ω")] ≈
        [Ω_hat[1, 1], Ω_hat[2, 1], Ω_hat[2, 2]]
end

@testset "summarize numeric formatting: fixed 4 decimals" begin
    f = NoLimits._fq_fmt_num
    # non-integer floats always show exactly 4 decimals, trailing zeros kept
    @test f(1.5) == "1.5000"
    @test f(2.0) == "2.0000"
    @test f(0.13) == "0.1300"
    @test f(1.23456) == "1.2346"
    @test f(-0.13) == "-0.1300"
    @test f(0.0) == "0.0000"
    @test f(12345.678) == "12345.6780"
    # values that would collapse to 0.0000 keep 4 significant digits instead
    @test f(1.0e-5) != "0.0000"
    @test occursin("e", f(1.0e-5))
    # sentinels / non-reals pass through
    @test f(nothing) == "-"
    @test f(missing) == "-"
    @test f(NaN) == "NaN"
    # the data-model summary shares the same formatter
    @test NoLimits._format_float(0.5) == "0.5000"
end
