using Test
using NoLimits
using DataFrames
using Distributions
using DataInterpolations

@testset "DataModelSummary: non-ODE, obs-row stats, and REPL show" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
            x = ConstantCovariate(; constant_on=:ID)
            w = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Distributions.Laplace(0.0, 1.0); column=:SITE)
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
        primary_id=:ID,
        time_col=:t,
        evid_col=:EVID,
        amt_col=:AMT,
        rate_col=:RATE,
        cmt_col=:CMT
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
    @test y_stats.mean ≈ 1.325 atol=1e-12

    # z includes 999 on event rows; stats should use observation rows only.
    z_stats = only(filter(row -> row.name == Symbol("z.z"), s.covariate_stats)).stats
    @test z_stats.n == 4
    @test z_stats.mean ≈ 2.5 atol=1e-12

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
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.4)
        end

        @covariates begin
            t = Covariate()
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 0.9, 1.1, 1.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    s = summarize(dm)

    @test s.model_type == :ode
    @test s.n_random_effects == 0
    @test isempty(keys(s.random_effect_distribution_types))
end
