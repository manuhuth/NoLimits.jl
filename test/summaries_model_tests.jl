using Test
using NoLimits
using Distributions
using DataInterpolations

@testset "ModelSummary: non-ODE declarations" begin
    model = @Model begin
        @helpers begin
            center(x, m) = x - m
        end

        @fixedEffects begin
            a = RealNumber(0.8; scale=:log, lower=0.01, calculate_se=true)
            b = RealVector([0.1, 0.2]; scale=[:identity, :log], lower=[-Inf, 0.01], calculate_se=true)
            σ = RealNumber(0.3; scale=:log)
            Ω = RealPSDMatrix([1.0 0.2; 0.2 1.2]; scale=:cholesky)
            spline = SplineParameters([0.0, 0.5, 1.0, 1.5, 2.0]; function_name=:spline_fn, degree=2)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariate(; constant_on=:ID)
            z = Covariate()
            w = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
            κ = RandomEffect(Distributions.Laplace(0.0, 1.0); column=:SITE)
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
    model = @Model begin
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
