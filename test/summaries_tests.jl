using Test
using NoLimits
using DataFrames
using Distributions
using DataInterpolations
using Random
using Turing: MH

# Shared ODE model for the model- and data-model-summary ODE testsets.
summaries_ode_model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.3)
        σ = RealNumber(0.2)
    end

    @covariates begin
        t = Covariate()
    end

    @preDifferentialEquation begin
        drive = a
    end

    @DifferentialEquation begin
        D(x1) ~ -drive * x1
        signal(t) = x1
    end

    @initialDE begin
        x1 = 1.0
    end

    @formulas begin
        y ~ Normal(signal(t), σ)
    end
end

@testset "ModelSummary: non-ODE declarations" begin
    model = @Model begin
        @helpers begin
            center(x, m) = x - m
        end

        @fixedEffects begin
            a = RealNumber(0.8; scale = :log, lower = 0.01, calculate_se = true)
            b = RealVector(
                [0.1, 0.2]; scale = [:identity, :log],
                lower = [-Inf, 0.01], calculate_se = true
            )
            σ = RealNumber(0.3; scale = :log)
            Ω = RealPSDMatrix([1.0 0.2; 0.2 1.2]; scale = :cholesky)
            spline = SplineParameters(
                [0.0, 0.5, 1.0, 1.5, 2.0]; function_name = :spline_fn, degree = 2
            )
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariate(; constant_on = :ID)
            z = Covariate()
            w = DynamicCovariate(; interpolation = LinearInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
            κ = RandomEffect(Distributions.Laplace(0.0, 1.0); column = :SITE)
        end

        @formulas begin
            μ = center(a + x + z + η + κ, b[1])
            y ~ LogNormal(μ, σ)
        end
    end

    s = summarize(model)

    @test s isa ModelSummary
    @test s.model_type == :non_ode
    @test s.has_helpers
    @test s.has_fixed_effects
    @test s.has_random_effects
    @test s.has_covariates
    @test !s.has_de
    @test !s.has_prede
    @test !s.has_initialde
    @test s.n_fixed_effect_blocks == length(get_names(model.fixed.fixed))
    @test s.n_fixed_effect_values == length(get_θ0_untransformed(model.fixed.fixed))
    @test s.n_random_effects == 2
    @test s.n_random_effect_group_columns == 2
    @test s.n_covariates == 4
    @test s.n_covariates_varying == 3
    @test s.n_covariates_constant == 1
    @test s.n_covariates_dynamic == 1
    @test s.n_deterministic_formulas == 1
    @test s.n_outcomes == 1
    @test s.outcome_distribution_types.y == :LogNormal

    re_kappa = only(filter(r -> r.name == :κ, s.random_effect_summaries))
    @test re_kappa.group == :SITE
    @test re_kappa.dist_type == :Laplace

    fe_a = only(filter(r -> r.name == :a, s.fixed_effect_summaries))
    @test fe_a.block_type == :RealNumber
    @test fe_a.calculate_se
    @test fe_a.scale == "log"
    @test occursin("finite lower", fe_a.bounds)

    fe_spline = only(filter(r -> r.name == :spline, s.fixed_effect_summaries))
    @test fe_spline.block_type == :SplineParameters
    @test occursin("degree=2", fe_spline.details)

    cov_w = only(filter(r -> r.name == :w, s.covariate_summaries))
    @test cov_w.kind == :DynamicCovariate
    @test cov_w.interpolation == "LinearInterpolation"

    @test :μ in s.deterministic_formula_names
    @test :center in s.helper_names

    txt = sprint(show, MIME"text/plain"(), s)
    @test occursin("ModelSummary", txt)
    @test occursin("Fixed-effects declarations", txt)
    @test occursin("Random-effects declarations", txt)
    @test occursin("Covariate declarations", txt)
    @test occursin("Outcome distribution types", txt)
end

@testset "ModelSummary: ODE structure and required accessors" begin
    model = summaries_ode_model

    s = summarize(model)

    @test s.model_type == :ode
    @test s.has_prede
    @test s.has_de
    @test s.has_initialde
    @test s.requires_de_accessors
    @test :x1 in s.de_states
    @test :signal in s.de_signals
    @test :signal in s.required_signals
    @test s.outcome_distribution_types.y == :Normal
end

@testset "DataModelSummary: non-ODE, obs-row stats, and REPL show" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
            x = ConstantCovariate(; constant_on = :ID)
            w = DynamicCovariate(; interpolation = LinearInterpolation)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column = :ID)
            η_site = RandomEffect(Distributions.Laplace(0.0, 1.0); column = :SITE)
        end

        @formulas begin
            lin = a + x + z + η_id + η_site
            y ~ LogNormal(lin, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        SITE = [:A, :A, :A, :B, :B, :B],
        t = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        EVID = [1, 0, 0, 1, 0, 0],
        AMT = [100.0, 0.0, 0.0, 150.0, 0.0, 0.0],
        RATE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        CMT = [1, 1, 1, 1, 1, 1],
        x = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        z = [999.0, 1.0, 2.0, 999.0, 3.0, 4.0],
        w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        y = [missing, 1.2, 1.4, missing, 1.1, 1.6]
    )

    dm = DataModel(
        model,
        df;
        primary_id = :ID,
        time_col = :t,
        evid_col = :EVID,
        amt_col = :AMT,
        rate_col = :RATE,
        cmt_col = :CMT
    )

    s = summarize(dm)

    @test s isa DataModelSummary
    @test s.model_type == :non_ode
    @test s.has_events
    @test s.n_individuals == 2
    @test s.n_rows_total == 6
    @test s.n_obs_rows == 4
    @test s.n_event_rows == 2
    @test s.n_fixed_effects == 2
    @test s.n_outcomes == 1
    @test s.n_covariates == 4
    @test s.n_covariates_varying == 3
    @test s.n_covariates_constant == 1
    @test s.n_covariates_dynamic == 1
    @test s.n_random_effects == 2
    @test s.outcome_distribution_types.y == :LogNormal
    @test s.random_effect_distribution_types.η_id == :Normal
    @test s.random_effect_distribution_types.η_site == :Laplace

    y_stats = only(filter(row -> row.name == :y, s.outcome_stats)).stats
    @test y_stats.n == 4
    @test y_stats.mean ≈ 1.325 atol = 1.0e-12

    # z includes 999 on event rows; stats should use observation rows only.
    z_stats = only(filter(row -> row.name == Symbol("z.z"), s.covariate_stats)).stats
    @test z_stats.n == 4
    @test z_stats.mean ≈ 2.5 atol = 1.0e-12

    # A subject-constant covariate is summarized once per individual, not per row (#309.7).
    x_stats = only(filter(row -> row.name == Symbol("x.x"), s.covariate_stats)).stats
    @test x_stats.n == 2
    @test x_stats.mean ≈ 15.0 atol = 1.0e-12

    re_id = only(filter(r -> r.name == :η_id, s.random_effect_summaries))
    @test re_id.group == :ID
    @test re_id.n_levels == 2
    @test re_id.rows_per_level.min == 2.0
    @test re_id.rows_per_level.max == 2.0

    txt = sprint(show, MIME"text/plain"(), s)
    @test occursin("DataModelSummary", txt)
    @test occursin("Outcome distribution types", txt)
    @test occursin("Random-effect distribution types", txt)
    @test occursin("Per-random-effect summary", txt)
end

@testset "DataModelSummary: ODE model type" begin
    model = summaries_ode_model

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 0.9, 1.1, 1.0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    s = summarize(dm)

    @test s.model_type == :ode
    @test s.n_random_effects == 0
    @test isempty(keys(s.random_effect_distribution_types))
end

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
            turing_kwargs = (n_samples = 2, n_adapt = 1, progress = false, verbose = false),
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

@testset "compare_parameters" begin
    model1 = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3; calculate_se = true)
            b = RealNumber(0.1; calculate_se = false)
            s = RealNumber(0.5; scale = :log, calculate_se = true)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            mu = a + b * t
            y ~ Normal(mu, s)
        end
    end

    model2 = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3; calculate_se = true)
            s = RealNumber(0.5; scale = :log, calculate_se = true)
            d = RealNumber(0.2; calculate_se = true)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            mu = a + d * t
            y ~ Normal(mu, s)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.4, 0.1, 0.5, 0.0, 0.3]
    )

    dm1 = DataModel(model1, df; primary_id = :ID, time_col = :t)
    dm2 = DataModel(model2, df; primary_id = :ID, time_col = :t)
    fit1 = fit_model(dm1, MLE(; optim_kwargs = (maxiters = 2,)))
    fit2 = fit_model(dm2, MLE(; optim_kwargs = (maxiters = 2,)))

    # Default: union of SE-eligible parameters, in declaration order.
    c = compare_parameters(fit1, fit2)
    @test c isa ParameterComparison
    @test c.labels == ["model 1", "model 2"]
    @test c.scale == :natural
    @test c.parameters == [:a, :s, :d]
    @test size(c.estimates) == (3, 2)

    i_a = findfirst(==(:a), c.parameters)
    i_d = findfirst(==(:d), c.parameters)
    @test all(c.estimates[i_a, :] .!== nothing)      # shared parameter
    @test c.estimates[i_d, 1] === nothing            # d absent from model1
    @test c.estimates[i_d, 2] !== nothing            # d present in model2

    # Custom labels and the label => fit pair form agree on contents.
    c_lbl = compare_parameters(fit1, fit2; labels = ["A", "B"])
    @test c_lbl.labels == ["A", "B"]
    c_pair = compare_parameters("A" => fit1, "B" => fit2)
    @test c_pair.labels == ["A", "B"]
    @test c_pair.parameters == c.parameters

    # common_only keeps only parameters shared by every model.
    c_common = compare_parameters(fit1, fit2; common_only = true)
    @test c_common.parameters == [:a, :s]
    @test all(c_common.estimates .!== nothing)

    # include_non_se brings in b (model1 only).
    c_all = compare_parameters(fit1, fit2; include_non_se = true)
    @test :b in c_all.parameters
    i_b = findfirst(==(:b), c_all.parameters)
    @test c_all.estimates[i_b, 1] !== nothing
    @test c_all.estimates[i_b, 2] === nothing

    # scale is validated and recorded.
    @test compare_parameters(fit1, fit2; scale = :transformed).scale == :transformed
    @test_throws ErrorException compare_parameters(fit1, fit2; scale = :bogus)
    # scale also accepts the string spelling (#255).
    @test compare_parameters(fit1, fit2; scale = "transformed").scale == :transformed
    @test compare_parameters("A" => fit1, "B" => fit2; scale = "transformed").scale ==
        :transformed
    @test_throws ErrorException compare_parameters(fit1, fit2; scale = "bogus")

    # Mismatched label count is an error.
    @test_throws ErrorException compare_parameters(fit1, fit2; labels = ["only one"])

    # Rendered table shows the title, labels, parameter names, and "-" for absences.
    txt = sprint(show, MIME"text/plain"(), c)
    @test occursin("ParameterComparison", txt)
    @test occursin("model 1", txt)
    @test occursin("model 2", txt)
    @test occursin("a", txt)
    @test occursin("d", txt)
    @test occursin("-", txt)
end

# Uses the shared fixture model/dm/fit/UQ (fixtures.jl); assertions are structural.
@testset "Compact show methods for core structs" begin
    model = fx_nore_model()
    dm = fx_nore_dm()
    res = fx_mle()
    uq = fx_uq_mle()

    txt_model = sprint(show, model)
    @test startswith(txt_model, "Model(")
    @test !occursin('\n', txt_model)
    @test length(txt_model) < 220

    txt_dm = sprint(show, dm)
    @test startswith(txt_dm, "DataModel(")
    @test !occursin('\n', txt_dm)
    @test length(txt_dm) < 260

    txt_res = sprint(show, res)
    @test startswith(txt_res, "FitResult(")
    @test occursin("data_model=stored", txt_res)
    @test !occursin('\n', txt_res)
    @test length(txt_res) < 240

    txt_uq = sprint(show, uq)
    @test startswith(txt_uq, "UQResult(")
    @test !occursin('\n', txt_uq)
    @test length(txt_uq) < 180
end

@testset "UQ summary scale, invariants, and interval aliases" begin
    res = fx_mle()
    names_t = [:a, :b, :σ]
    ints(v) = NoLimits.UQIntervals(0.95, v .- 1.0, v .+ 1.0)
    mk(names, nat, est_n; vcov = nothing, draws = nothing) = UQResult(
        :wald, :mle, names, nat, Float64[1:length(names);], est_n,
        ints(Float64[1:length(names);]), ints(est_n),
        vcov, vcov, draws, draws, NamedTuple()
    )

    # Every public accessor rejects a scale that is neither :natural nor :transformed.
    uq = mk(names_t, nothing, [10.0, 20.0, 30.0]; vcov = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
    @test_throws ArgumentError get_uq_parameter_names(uq; scale = :bogus)
    @test_throws ArgumentError get_uq_estimates(uq; scale = :bogus)
    @test_throws ArgumentError get_uq_intervals(uq; scale = :bogus)
    @test_throws ArgumentError get_uq_vcov(uq; scale = :bogus)
    @test_throws ArgumentError get_uq_draws(uq; scale = :bogus)

    # Names follow the requested scale instead of always using the transformed default.
    uq_sc = mk(names_t, [:an, :bn, :σn], [10.0, 20.0, 30.0])
    rows_n = summarize(uq_sc; scale = :natural).parameter_rows
    @test [r.parameter for r in rows_n] == [:an, :bn, :σn]
    @test [r.estimate for r in rows_n] == [10.0, 20.0, 30.0]
    rows_t = summarize(uq_sc; scale = :transformed).parameter_rows
    @test [r.parameter for r in rows_t] == names_t
    @test [r.estimate for r in rows_t] == [1.0, 2.0, 3.0]

    # scale also accepts the string spelling (#255).
    rows_str = summarize(uq_sc; scale = "transformed").parameter_rows
    @test [r.parameter for r in rows_str] == [r.parameter for r in rows_t]
    @test [r.estimate for r in rows_str] == [r.estimate for r in rows_t]
    @test_throws ErrorException summarize(uq_sc; scale = "trasformed")
    @test summarize(res; scale = "natural").parameter_rows ==
        summarize(res; scale = :natural).parameter_rows

    # Draw-based SEs give one value per parameter for n_draws below and above n_params.
    for nd in (2, 5)
        uq_d = mk(names_t, nothing, [1.0, 2.0, 3.0]; draws = randn(Xoshiro(3), nd, 3))
        s = summarize(uq_d)
        @test length(s.parameter_rows) == 3
        @test all(r -> r.std_error isa Real, s.parameter_rows)
    end

    # Malformed UQResults are rejected up front, not deep inside row construction.
    @test_throws ErrorException summarize(
        UQResult(
            :wald, :mle, names_t, nothing, [1.0, 2.0, 3.0], [1.0, 2.0],
            nothing, nothing, nothing, nothing, nothing, nothing, NamedTuple()
        )
    )

    # Foreign UQ layouts must fail; partial ones must be reported, never silently dropped.
    @test_throws ErrorException summarize(res, mk([:a, :zzz], nothing, [1.0, 2.0]))
    s_part = summarize(res, mk([:a, :b], nothing, [1.0, 2.0]))
    @test any(r -> r.parameter == :σ && r.std_error === nothing, s_part.parameter_rows)
    @test any(n -> occursin("without uncertainty", n), s_part.notes)

    # Interval aliases are validated against the resolved backend.
    for (backend, ok) in (
            (:wald, (:auto, :wald, :normal)), (:chain, (:auto, :equaltail, :chain)),
            (:mcmc_refit, (:auto, :equaltail, :chain)), (:profile, (:auto, :profile)),
        )
        for iv in ok
            @test NoLimits._validate_uq_interval(backend, iv) == iv
        end
        for iv in setdiff([:wald, :normal, :equaltail, :chain, :profile], collect(ok))
            @test_throws ErrorException NoLimits._validate_uq_interval(backend, iv)
        end
    end
    @test compute_uq(res; method = :wald, interval = :normal, n_draws = 5) isa UQResult
    @test_throws ErrorException compute_uq(res; method = :wald, interval = :equaltail)

    # Coverage counts observation rows and states the counting unit.
    @test all(r -> r.unit == :row, summarize(res).coverage_rows)
end
