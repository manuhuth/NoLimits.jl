using Test
using NoLimits
using DataFrames
using ComponentArrays
using Distributions
using SciMLBase
using DataInterpolations
using LinearAlgebra
using SentinelArrays: ChainedVector

@testset "row-varying RE capability detection" begin
    df_basic = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        dt = [1.0, 1.0, 1.0, 1.0],
        y = [0.0, 1.0, 0.0, 1.0]
    )

    model_non_ode = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    @test !NoLimits._has_continuous_time_hmm_outcomes(
        model_non_ode, df_basic; primary_id = :ID, time_col = :t
    )
    @test NoLimits._supports_row_varying_re_groups(
        model_non_ode, df_basic; primary_id = :ID, time_col = :t
    )

    model_ode = @Model begin
        @fixedEffects begin
            k = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @DifferentialEquation begin
            D(x1) ~ -k * x1 + η
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    @test !NoLimits._supports_row_varying_re_groups(
        model_ode, df_basic; primary_id = :ID, time_col = :t
    )

    model_ct_hmm = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale = :log)
            λ21_r = RealNumber(0.1, scale = :log)
            p1_r = RealNumber(0.0)
            p2_r = RealNumber(0.0)
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            Q = [-λ12 λ12; λ21 -λ21]
            y ~ ContinuousTimeDiscreteStatesHMM(
                Q,
                (Bernoulli(p1), Bernoulli(p2)),
                Categorical([0.6, 0.4]),
                dt
            )
        end
    end

    @test NoLimits._has_continuous_time_hmm_outcomes(
        model_ct_hmm, df_basic; primary_id = :ID, time_col = :t
    )
    @test NoLimits._supports_row_varying_re_groups(
        model_ct_hmm, df_basic; primary_id = :ID, time_col = :t
    )

    model_ct_hmm_helper = @Model begin
        @helpers begin
            make_hmm(Q, p1, p2, dt) = ContinuousTimeDiscreteStatesHMM(
                Q,
                (Bernoulli(p1), Bernoulli(p2)),
                Categorical([0.6, 0.4]),
                dt
            )
        end

        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale = :log)
            λ21_r = RealNumber(0.1, scale = :log)
            p1_r = RealNumber(0.0)
            p2_r = RealNumber(0.0)
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            Q = [-λ12 λ12; λ21 -λ21]
            y ~ make_hmm(Q, p1, p2, dt)
        end
    end

    @test NoLimits._has_continuous_time_hmm_outcomes(
        model_ct_hmm_helper, df_basic; primary_id = :ID, time_col = :t
    )
    @test NoLimits._supports_row_varying_re_groups(
        model_ct_hmm_helper, df_basic; primary_id = :ID, time_col = :t
    )

    model_dt_hmm = @Model begin
        @fixedEffects begin
            p11_r = RealNumber(0.0)
            p22_r = RealNumber(0.0)
            p1_r = RealNumber(0.0)
            p2_r = RealNumber(0.0)
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            p11 = 1 / (1 + exp(-p11_r))
            p22 = 1 / (1 + exp(-p22_r))
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            P = [p11 (1 - p11); (1 - p22) p22]
            y ~ DiscreteTimeDiscreteStatesHMM(
                P,
                (Bernoulli(p1), Bernoulli(p2)),
                Categorical([0.6, 0.4])
            )
        end
    end

    @test !NoLimits._has_continuous_time_hmm_outcomes(
        model_dt_hmm, df_basic; primary_id = :ID, time_col = :t
    )
    @test NoLimits._supports_row_varying_re_groups(
        model_dt_hmm, df_basic; primary_id = :ID, time_col = :t
    )
end

@testset "DataModel without events" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age]; constant_on = :ID)
            z = Covariate()
        end

        @randomEffects begin
            η_subj = RandomEffect(Normal(0.0, 1.0); column = :ID)
            η_year = RandomEffect(Normal(0.0, 1.0); column = :YEAR)
        end

        @formulas begin
            lin = a + x.Age + z + η_subj + η_year
            y ~ Normal(lin, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3],
        YEAR = [2020, 2020, 2020, 2021, 2021],
        t = [0.0, 1.0, 0.5, 1.5, 2.0],
        Age = [30.0, 30.0, 40.0, 40.0, 50.0],
        z = [1.0, 1.2, 0.8, 0.9, 1.1],
        y = [1.0, 1.1, 0.9, 1.0, 1.2]
    )

    dm_varying = DataModel(
        model, df;
        primary_id = :ID,
        time_col = :t
    )
    @test length(get_individuals(dm_varying)) == 3
    @test length(get_batches(dm_varying)) == 1

    df_ok = DataFrame(
        ID = [1, 1, 2, 2, 3],
        YEAR = [2020, 2020, 2020, 2020, 2021],
        t = [0.0, 1.0, 0.5, 1.5, 2.0],
        Age = [30.0, 30.0, 40.0, 40.0, 50.0],
        z = [1.0, 1.2, 0.8, 0.9, 1.1],
        y = [1.0, 1.1, 0.9, 1.0, 1.2]
    )

    dm = DataModel(
        model, df_ok;
        primary_id = :ID,
        time_col = :t
    )

    @test length(get_individuals(dm)) == 3
    @test all(ind -> ind.callbacks === nothing, get_individuals(dm))

    batches = get_batches(dm)
    @test length(batches) == 2
    @test sort(length.(batches)) == [1, 2]

    ind1 = get_individual(dm, 1)
    @test ind1.series.obs.y == [1.0, 1.1]
    @test ind1.const_cov.x.Age == 30.0
    @test ind1.series.vary.z == [1.0, 1.2]

    # Any Tables.jl table is materialized at the boundary (#259).
    dm_tbl = DataModel(
        model, DataFrames.Tables.columntable(df_ok);
        primary_id = :ID,
        time_col = :t
    )
    @test get_df(dm_tbl) == get_df(dm)
    @test length(get_individuals(dm_tbl)) == length(get_individuals(dm))
    @test get_individual(dm_tbl, 1).series.obs.y == ind1.series.obs.y
end

# Shared across the events and validation-errors testsets below.
const dm_model_age_re = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.5)
    end

    @covariates begin
        t = Covariate()
        x = ConstantCovariateVector([:Age])
    end

    @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0); column = :ID)
    end

    @formulas begin
        lin = a + x.Age + η
        y ~ Normal(lin, σ)
    end
end

@testset "DataModel with events (EVID/AMT/RATE/CMT)" begin
    model = dm_model_age_re

    df = DataFrame(
        ID = [1, 1, 1, 2, 2],
        t = [0.0, 0.5, 1.0, 0.0, 1.0],
        EVID = [1, 0, 0, 1, 0],
        AMT = [100.0, 0.0, 0.0, 50.0, 0.0],
        RATE = [0.0, 0.0, 0.0, 0.0, 0.0],
        CMT = [1, 1, 1, 1, 1],
        Age = [30.0, 30.0, 30.0, 40.0, 40.0],
        y = [missing, 1.1, 1.2, missing, 0.9]
    )

    dm = DataModel(
        model, df;
        primary_id = :ID,
        time_col = :t,
        evid_col = :EVID,
        amt_col = :AMT,
        rate_col = :RATE,
        cmt_col = :CMT
    )

    @test length(get_individuals(dm)) == 2
    ind1 = get_individual(dm, 1)
    @test ind1.series.obs.y == [1.1, 1.2]
    @test ind1.callbacks === nothing

    # Without a DE block the dose amounts go nowhere -- warn instead of silently
    # dropping them (#174).
    @test_logs (:warn, r"no @DifferentialEquation") match_mode = :any DataModel(
        model, df;
        primary_id = :ID, time_col = :t, evid_col = :EVID,
        amt_col = :AMT, rate_col = :RATE, cmt_col = :CMT
    )
end

# Shared valid model + df; only the DataModel(...) kwargs differ across the
# serialization / primary_id / obs_cols testsets that use them.
const dm_model_lin_eta = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.5)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0); column = :ID)
    end

    @formulas begin
        lin = a + η
        y ~ Normal(lin, σ)
    end
end

const dm_df_two_rows = DataFrame(
    ID = [1, 1],
    t = [0.0, 1.0],
    y = [1.0, 1.1]
)

@testset "DataModel serialization config (EnsembleThreads)" begin
    dm = DataModel(
        dm_model_lin_eta, dm_df_two_rows;
        primary_id = :ID,
        time_col = :t,
        serialization = EnsembleThreads()
    )

    @test dm.config.serialization isa EnsembleThreads
end

@testset "DataModel validation errors" begin
    model = dm_model_age_re

    df_missing_time = DataFrame(
        ID = [1, 1],
        Age = [30.0, 30.0],
        y = [1.0, 1.1]
    )
    @test_throws ErrorException DataModel(
        model, df_missing_time; primary_id = :ID, time_col = :t
    )

    df_missing_evid_cols = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        EVID = [1, 0],
        Age = [30.0, 30.0],
        y = [missing, 1.1]
    )
    @test_throws ErrorException DataModel(
        model, df_missing_evid_cols;
        primary_id = :ID,
        time_col = :t,
        evid_col = :EVID
    )

    # Column keywords also accept strings (#255), including on the error paths.
    @test_throws ErrorException DataModel(
        model, df_missing_time; primary_id = "ID", time_col = "t"
    )

    # A non-table input passes through the normalization (#259) and still fails in the
    # schema validation, not with a conversion error.
    @test_throws ErrorException DataModel(
        model, 42; primary_id = :ID, time_col = :t
    )
end

# Shared no-RE model; also reused by the mixed-type primary_id testset below.
const dm_model_no_re = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        a = RealNumber(1.0)
        σ = RealNumber(0.5)
    end

    @formulas begin
        y ~ Normal(a, σ)
    end
end

@testset "DataModel time_col covariate validation" begin
    model_missing = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = dm_df_two_rows

    @test_throws ErrorException DataModel(
        model_missing, df; primary_id = :ID, time_col = :t
    )

    model_bad = @Model begin
        @covariates begin
            t = ConstantCovariate()
        end

        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    @test_throws ErrorException DataModel(model_bad, df; primary_id = :ID, time_col = :t)

    dm = DataModel(dm_model_no_re, df; primary_id = :ID, time_col = :t)
    @test dm isa DataModel
end

@testset "DataModel validates RE constant covariates within groups" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end

        @randomEffects begin
            η_year = RandomEffect(Normal(0.0, abs(x.Age)); column = :YEAR)
        end

        @formulas begin
            y ~ Normal(η_year, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        YEAR = [2020, 2020, 2021, 2021],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 31.0, 40.0, 40.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df; primary_id = :OBS, time_col = :t)
end

@testset "DataModel errors on invalid constant_on columns" begin
    @test_throws ErrorException @eval @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c = ConstantCovariate(; constant_on = :BADCOL)
        end

        @randomEffects begin
            η = RandomEffect(Normal(c, 1.0); column = :ID)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end
end

# Shared by the valid and invalid three-RE-grouping testsets below.
const dm_model_three_re = @Model begin
    @fixedEffects begin
        σ = RealNumber(0.5)
    end

    @covariates begin
        t = Covariate()
        c1 = ConstantCovariate(; constant_on = :ID)
        c2 = ConstantCovariate(; constant_on = :SITE)
        c3 = ConstantCovariate(; constant_on = :YEAR)
    end

    @randomEffects begin
        η_id = RandomEffect(Normal(c1, 1.0); column = :ID)
        η_site = RandomEffect(Normal(c2, 1.0); column = :SITE)
        η_year = RandomEffect(Normal(c3, 1.0); column = :YEAR)
    end

    @formulas begin
        y ~ Normal(η_id + η_site + η_year, σ)
    end
end

@testset "DataModel with three RE grouping columns (valid)" begin
    model = dm_model_three_re

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        YEAR = [2020, 2020, 2021, 2021],
        t = [0.0, 1.0, 0.0, 1.0],
        c1 = [10.0, 10.0, 20.0, 20.0],
        c2 = [1.0, 1.0, 2.0, 2.0],
        c3 = [0.5, 0.5, 0.8, 0.8],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    info = get_re_group_info(dm)
    @test length(info.values.η_id) == 2
    @test length(info.values.η_site) == 2
    @test length(info.values.η_year) == 2
end

@testset "DataModel with three RE grouping columns (invalid)" begin
    model = dm_model_three_re

    df_bad = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        YEAR = [2020, 2020, 2021, 2021],
        t = [0.0, 1.0, 0.0, 1.0],
        c1 = [10.0, 11.0, 20.0, 20.0],
        c2 = [1.0, 1.0, 2.0, 2.0],
        c3 = [0.5, 0.5, 0.8, 0.9],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df_bad; primary_id = :ID, time_col = :t)
end

@testset "DataModel rejects missing random-effect grouping values" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :SITE)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, missing, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    err = try
        DataModel(model, df; primary_id = :ID, time_col = :t)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("contains missing values", msg)
    @test occursin("drop rows with missing", msg)
    @test occursin("explicit custom level", msg)
end

@testset "DataModel allows year varying within individuals for non-ODE models" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column = :ID)
            η_year = RandomEffect(Normal(0.0, 1.0); column = :YEAR)
        end

        @formulas begin
            y ~ Normal(η_id + η_year, σ)
        end
    end

    # ID=1 spans two years; c_year is constant within each YEAR group.
    df = DataFrame(
        ID = [1, 1, 1, 2, 2],
        YEAR = [2020, 2021, 2021, 2020, 2022],
        t = [0.0, 0.5, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 1.2, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test dm isa DataModel
    info = get_re_group_info(dm)
    @test info.values.η_year == [2020, 2021, 2022]
    @test info.index_by_row.η_year == [1, 2, 2, 1, 3]
    @test info.representative_row_by_level.η_year == [1, 2, 5]
    @test info.index_by_individual.η_year.level_ids_obs[1] == [1, 2, 2]
    @test info.index_by_individual.η_year.unique_pos_obs[1] == [1, 2, 2]
    @test info.index_by_individual.η_year.level_ids_obs[2] == [1, 3]
    @test info.index_by_individual.η_year.unique_pos_obs[2] == [1, 2]
    @test info.index_by_individual.η_year.level_ids_all[1] == [1, 2, 2]
    @test info.index_by_individual.η_year.unique_pos_all[2] == [1, 2]
    @test get_re_indices(dm, 1).η_year == [1, 2, 2]
    @test get_re_indices(dm, 2).η_year == [1, 3]
    @test get_re_indices(dm, 1; obs_only = false).η_year == [1, 2, 2]
end

@testset "DataModel rejects year varying within individuals for ODE models" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column = :ID)
            η_year = RandomEffect(Normal(0.0, 1.0); column = :YEAR)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + η_id + η_year
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2],
        YEAR = [2020, 2021, 2021, 2020, 2022],
        t = [0.0, 0.5, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 1.2, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df; primary_id = :ID, time_col = :t)
end

@testset "DataModel allows year varying within individuals for continuous-time HMM models" begin
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale = :log)
            λ21_r = RealNumber(0.1, scale = :log)
            p1_r = RealNumber(0.0)
            p2_r = RealNumber(0.0)
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @randomEffects begin
            η_year = RandomEffect(Normal(0.0, 1.0); column = :YEAR)
        end

        @formulas begin
            λ12 = exp(λ12_r + η_year)
            λ21 = exp(λ21_r)
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            Q = [-λ12 λ12; λ21 -λ21]
            y ~ ContinuousTimeDiscreteStatesHMM(
                Q,
                (Bernoulli(p1), Bernoulli(p2)),
                Categorical([0.6, 0.4]),
                dt
            )
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2],
        YEAR = [2020, 2021, 2021, 2020, 2022],
        t = [0.0, 1.0, 2.0, 0.0, 1.0],
        dt = [1.0, 1.0, 1.0, 1.0, 1.0],
        y = [0, 1, 1, 1, 0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test dm isa DataModel
end

@testset "DataModel allows year varying within individuals for discrete-time HMM models" begin
    model = @Model begin
        @fixedEffects begin
            p11_r = RealNumber(0.0)
            p22_r = RealNumber(0.0)
            p1_r = RealNumber(0.0)
            p2_r = RealNumber(0.0)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η_year = RandomEffect(Normal(0.0, 1.0); column = :YEAR)
        end

        @formulas begin
            p11 = 1 / (1 + exp(-(p11_r + η_year)))
            p22 = 1 / (1 + exp(-p22_r))
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            P = [p11 (1 - p11); (1 - p22) p22]
            y ~ DiscreteTimeDiscreteStatesHMM(
                P,
                (Bernoulli(p1), Bernoulli(p2)),
                Categorical([0.6, 0.4])
            )
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2],
        YEAR = [2020, 2021, 2021, 2020, 2022],
        t = [0.0, 1.0, 2.0, 0.0, 1.0],
        y = [0, 1, 1, 1, 0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test dm isa DataModel
end

# Shared by the two c_year-inconsistency testsets below.
const dm_model_cyear_re = @Model begin
    @fixedEffects begin
        σ = RealNumber(0.5)
    end

    @covariates begin
        t = Covariate()
        c_year = ConstantCovariate(; constant_on = :YEAR)
    end

    @randomEffects begin
        η_id = RandomEffect(Normal(0.0, 1.0); column = :ID)
        η_year = RandomEffect(Normal(c_year, 1.0); column = :YEAR)
    end

    @formulas begin
        y ~ Normal(η_id + η_year, σ)
    end
end

@testset "DataModel errors when year covariate varies within YEAR group" begin
    model = dm_model_cyear_re

    # YEAR=2020 appears with two different c_year values.
    df_bad = DataFrame(
        ID = [1, 1, 2, 2],
        YEAR = [2020, 2020, 2020, 2021],
        t = [0.0, 1.0, 0.0, 1.0],
        c_year = [2.0, 2.5, 2.0, 3.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df_bad; primary_id = :ID, time_col = :t)
end

@testset "DataModel errors when constant covariate varies within primary_id" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c_year = ConstantCovariate(; constant_on = :YEAR)
        end

        @randomEffects begin
            η_year = RandomEffect(Normal(c_year, 1.0); column = :YEAR)
        end

        @formulas begin
            y ~ Normal(η_year, σ)
        end
    end

    # c_year is constant within YEAR groups, but varies within ID=1.
    df_bad = DataFrame(
        ID = [1, 1, 2, 2],
        YEAR = [2020, 2021, 2020, 2021],
        t = [0.0, 1.0, 0.0, 1.0],
        c_year = [0.1, 0.2, 0.1, 0.2],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df_bad; primary_id = :ID, time_col = :t)
end

@testset "DataModel errors when individual spans years and YEAR covariate is inconsistent" begin
    model = dm_model_cyear_re

    # ID=1 spans 2020 and 2021; YEAR=2020 has inconsistent c_year.
    df_bad = DataFrame(
        ID = [1, 1, 1, 2],
        YEAR = [2020, 2020, 2021, 2021],
        t = [0.0, 1.0, 2.0, 0.0],
        c_year = [2.0, 2.5, 3.0, 3.0],
        y = [1.0, 1.1, 1.2, 0.9]
    )

    @test_throws ErrorException DataModel(model, df_bad; primary_id = :ID, time_col = :t)
end

@testset "DataModel errors when constant covariate varies within primary_id" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c_id = ConstantCovariate(; constant_on = :ID)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(c_id, 1.0); column = :ID)
        end

        @formulas begin
            y ~ Normal(η_id, σ)
        end
    end

    df_bad = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        c_id = [1.0, 2.0, 3.0, 3.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df_bad; primary_id = :ID, time_col = :t)
end

@testset "DataModel rejects time-varying REs in preDE" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column = :ID)
            η_year = RandomEffect(Normal(0.0, 1.0); column = :YEAR)
        end

        @preDifferentialEquation begin
            pre = a + η_year
        end

        @formulas begin
            y ~ Normal(pre + η_id, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        YEAR = [2020, 2021, 2020, 2021],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df; primary_id = :ID, time_col = :t)
end

@testset "DataModel preDE with multiple RE groups (valid)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c_id = ConstantCovariate(; constant_on = :ID)
            c_year = ConstantCovariate(; constant_on = :YEAR)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(c_id, 1.0); column = :ID)
            η_year = RandomEffect(Normal(c_year, 1.0); column = :YEAR)
        end

        @preDifferentialEquation begin
            pre = a + η_id
        end

        @formulas begin
            y ~ Normal(pre + η_year, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 2],
        YEAR = [2020, 2020, 2021, 2021, 2021],
        t = [0.0, 1.0, 0.0, 1.0, 2.0],
        c_id = [1.0, 1.0, 2.0, 2.0, 2.0],
        c_year = [0.1, 0.1, 0.2, 0.2, 0.2],
        y = [1.0, 1.1, 0.9, 1.0, 1.05]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    ind1 = get_individual(dm, 1)
    re_idxs = get_re_indices(dm, ind1)
    @test length(unique(re_idxs.η_year)) == 1
end

@testset "DataModel RE constant covariate vector validation" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :Weight])
        end

        @randomEffects begin
            η = RandomEffect(Normal(x.Age + x.Weight, 1.0); column = :SITE)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    df_bad = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [10.0, 11.0, 20.0, 20.0],
        Weight = [50.0, 50.0, 60.0, 60.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df_bad; primary_id = :ID, time_col = :t)
end

@testset "DataModel RE validation ignores varying covariates" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(z, 1.0); column = :SITE)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        z = [1.0, 2.0, 3.0, 4.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df; primary_id = :ID, time_col = :t)
end

# Shared by the invalid and valid used-covariates testsets below.
const dm_model_used_cov = @Model begin
    @fixedEffects begin
        σ = RealNumber(0.5)
    end

    @covariates begin
        t = Covariate()
        c1 = ConstantCovariate()
        c2 = ConstantCovariate()
    end

    @randomEffects begin
        η = RandomEffect(Normal(c1, 1.0); column = :SITE)
    end

    @formulas begin
        y ~ Normal(η, σ)
    end
end

@testset "DataModel RE validation only checks used covariates" begin
    model = dm_model_used_cov

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        c1 = [10.0, 10.0, 20.0, 20.0],
        c2 = [1.0, 2.0, 3.0, 4.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df; primary_id = :ID, time_col = :t)
end

@testset "DataModel RE validation only checks used covariates (valid)" begin
    model = dm_model_used_cov

    df_ok = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        c1 = [10.0, 10.0, 20.0, 20.0],
        c2 = [1.0, 1.0, 3.0, 3.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df_ok; primary_id = :ID, time_col = :t)
    @test length(get_individuals(dm)) == 2
end

@testset "DataModel primary_id inference" begin
    dm = DataModel(dm_model_lin_eta, dm_df_two_rows; time_col = :t)
    @test get_primary_id(dm) == :ID

    model_multi = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η1 = RandomEffect(Normal(0.0, 1.0); column = :ID)
            η2 = RandomEffect(Normal(0.0, 1.0); column = :YEAR)
        end
        @formulas begin
            lin = a + η1 + η2
            y ~ Normal(lin, σ)
        end
    end

    df2 = DataFrame(
        ID = [1, 1],
        YEAR = [2020, 2020],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )
    @test_throws ErrorException DataModel(model_multi, df2; time_col = :t)
end

@testset "DataModel includes t for ODE models" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
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
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    ind1 = get_individual(dm, 1)
    @test ind1.series.vary.t == [0.0, 1.0]
end

@testset "DataModel saveat policy" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t) + x1(t + 0.25), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 0.5, 1.0],
        EVID = [1, 0, 0],
        AMT = [100.0, 0.0, 0.0],
        RATE = [0.0, 0.0, 0.0],
        CMT = [1, 1, 1],
        y = [missing, 1.1, 1.2]
    )

    model_saveat = set_solver_config(model; saveat_mode = :saveat)
    dm = DataModel(model_saveat, df; primary_id = :ID, time_col = :t, evid_col = :EVID)
    ind1 = get_individual(dm, 1)
    @test ind1.saveat == [0.0, 0.5, 0.75, 1.0, 1.25]

    model_dense = set_solver_config(model; saveat_mode = :dense)
    dm_dense = DataModel(model_dense, df; primary_id = :ID, time_col = :t, evid_col = :EVID)
    ind1_dense = get_individual(dm_dense, 1)
    @test ind1_dense.saveat === nothing
end

@testset "DataModel saveat with time offsets" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t) + x1(t + 0.25) + x1(t + (1 / 4)), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    model_saveat = set_solver_config(model; saveat_mode = :saveat)
    dm = DataModel(model_saveat, df; primary_id = :ID, time_col = :t)
    ind1 = get_individual(dm, 1)
    @test ind1.saveat == [0.0, 0.25, 1.0, 1.25]
    @test ind1.tspan == (0.0, 1.25)

    model_dense = set_solver_config(model; saveat_mode = :dense)
    dm_dense = DataModel(model_dense, df; primary_id = :ID, time_col = :t)
    ind1_dense = get_individual(dm_dense, 1)
    @test ind1_dense.tspan == (0.0, 1.25)
end

@testset "DataModel errors on negative time offsets" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t - 0.5), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    model_saveat = set_solver_config(model; saveat_mode = :saveat)
    @test_throws ErrorException DataModel(model_saveat, df; primary_id = :ID, time_col = :t)

    # The guard compares against the actual integration start, not the literal t=0
    # (#287): a lag below the individual's first time still errors at negative times.
    df_neg = DataFrame(ID = [1, 1], t = [-6.0, -4.0], y = [1.0, 1.1])
    @test_throws ErrorException DataModel(
        model_saveat, df_neg; primary_id = :ID, time_col = :t, t0 = nothing
    )
    # ... while an explicit t0 below the lagged time makes the same model valid.
    dm_ok = DataModel(
        model_saveat, df_neg; primary_id = :ID, time_col = :t, t0 = -10.0
    )
    @test get_individual(dm_ok, 1).tspan[1] == -10.0
end

@testset "DataModel errors on offsets below dynamic covariate support" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            w1 = DynamicCovariate(; interpolation = LinearInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + w1(t)
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t - 0.5) + x1(t + 0.5), σ)
        end
    end

    # rows start at t = 0.5, so t_first + offset = 0.0 clears the negative-time check,
    # but the interpolants have no support below 0.5.
    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.5, 1.0, 2.0],
        w1 = [1.0, 1.2, 1.4],
        y = [1.0, 1.1, 1.2]
    )

    model_saveat = set_solver_config(model; saveat_mode = :saveat)
    err = try
        DataModel(model_saveat, df; primary_id = :ID, time_col = :t)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("dynamic covariate support", err.msg)
    @test occursin("-0.5", err.msg)
    @test occursin("[0.5, 2.0]", err.msg)

    # The mirrored upper-bound guard: the +0.5 offset pushes the integration past the
    # covariate support, which used to escape as a raw extrapolation error (#309).
    df_up = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        w1 = [1.0, 1.2, 1.4],
        y = [1.0, 1.1, 1.2]
    )
    err_up = try
        DataModel(model_saveat, df_up; primary_id = :ID, time_col = :t)
        nothing
    catch e
        e
    end
    @test err_up isa ErrorException
    @test occursin("later than the dynamic covariate support", err_up.msg)
    @test occursin("2.5", err_up.msg)
end

@testset "DataModel pairing creates multiple batches" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column = :ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column = :SITE)
        end

        @formulas begin
            lin = a + η_id + η_site
            y ~ Normal(lin, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        SITE = [:A, :A, :A, :A, :B, :B, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 0.9, 1.0, 1.2, 1.1, 1.0, 0.95]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    batches = get_batches(dm)
    @test length(batches) == 2
    @test all(length.(batches) .== 2)
end

@testset "DataModel dynamic covariates" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            w1 = DynamicCovariate(; interpolation = LinearInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            lin = a + w1(t) + η
            y ~ Normal(lin, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 0.5, 1.0],
        w1 = [1.0, 1.2, 1.4],
        y = [1.0, 1.1, 1.2]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    ind1 = get_individual(dm, 1)
    @test ind1.series.dyn.w1(0.25) ≈ 1.1
end

@testset "DataModel primary_id validation" begin
    @test_throws ErrorException DataModel(
        dm_model_lin_eta, dm_df_two_rows; primary_id = :SUBJ, time_col = :t
    )
end

@testset "DataModel informs for numeric random-effect ids" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
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
        y = [1.0, 1.1, 0.9, 1.0]
    )

    empty!(NoLimits._warned_numeric_re_group_cols)
    dm = nothing
    @test_logs (:info, r"numeric random-effect grouping levels") begin
        dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    end
    @test dm isa DataModel
end

@testset "DataModel warns for weakly identified random effects" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :OBS)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        OBS = [1, 2, 3],
        t = [0.0, 1.0, 2.0],
        y = [1.0, 1.1, 1.2]
    )

    dm = nothing
    @test_logs (:info, r"numeric random-effect grouping levels") (
        :warn, r"weakly identified",
    ) begin
        dm = DataModel(model, df; primary_id = :OBS, time_col = :t)
    end
    @test dm isa DataModel
end

@testset "DataModel allows identifiable random effects" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :SITE)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test length(get_individuals(dm)) == 2
end

@testset "DataModel infers obs_cols from formulas" begin
    dm = DataModel(dm_model_lin_eta, dm_df_two_rows; primary_id = :ID, time_col = :t)
    ind1 = get_individual(dm, 1)
    @test ind1.series.obs.y == [1.0, 1.1]
end

@testset "DataModel supports mixed-type primary_id values" begin
    model = dm_model_no_re

    df = DataFrame(
        ID = Any[1, 1, "2", "2"],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test length(get_individuals(dm)) == 2
    @test get_individual(dm, 1).series.obs.y == [0.1, 0.2]
    @test get_individual(dm, "2").series.obs.y == [0.0, -0.1]

    # Column keywords also accept the string spelling (#255).
    dm_str = DataModel(model, df; primary_id = "ID", time_col = "t")
    @test get_primary_id(dm_str) === get_primary_id(dm)
    @test get_time_col(dm_str) === get_time_col(dm)
    @test length(get_individuals(dm_str)) == length(get_individuals(dm))
end

@testset "DataModel validates missing covariates used by formulas" begin
    model_used = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            y ~ Normal(a + z, σ)
        end
    end

    df_used_missing = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        z = [0.1, missing],
        y = [0.1, 0.2]
    )
    @test_throws ErrorException DataModel(
        model_used, df_used_missing; primary_id = :ID, time_col = :t
    )

    model_unused = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df_unused_missing = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        z = [0.1, missing],
        y = [0.1, 0.2]
    )

    dm = DataModel(model_unused, df_unused_missing; primary_id = :ID, time_col = :t)
    @test dm isa DataModel
end

@testset "DataModel allows partially missing observables on observation rows (regression)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σy = RealNumber(0.4)
            σz = RealNumber(0.6)
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            μ = a + 0.1 * t
            y ~ Normal(μ, σy)
            z ~ Normal(μ + 1.0, σz)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = Union{Missing, Float64}[1.0, missing, 1.2],
        z = Union{Missing, Float64}[2.1, 2.0, missing]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    ind = get_individual(dm, 1)
    @test isequal(ind.series.obs.y, Union{Missing, Float64}[1.0, missing, 1.2])
    @test isequal(ind.series.obs.z, Union{Missing, Float64}[2.1, 2.0, missing])
end

# Subjects that differ in dose events get different concrete `Individual` types. `map`
# would widen those to the typejoin `Individual{...} where CB`, which is abstract and
# dispatches dynamically on every field access; a Union is union-split instead.
@testset "individuals eltype stays union-split with mixed dosing" begin
    model = @Model begin
        @fixedEffects begin
            ke = RealNumber(0.5, scale = :log)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end
        @DifferentialEquation begin
            D(A) ~ -ke * exp(η) * A
        end
        @initialDE begin
            A = 0.0
        end
        @formulas begin
            y ~ Normal(A(t), σ)
        end
    end
    # ID 1 and 2 are dosed; ID 3 never is, so its callbacks differ in type.
    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2, 3, 3],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0],
        EVID = [1, 0, 0, 1, 0, 0, 0, 0],
        AMT = [10.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        RATE = zeros(8),
        CMT = ones(Int, 8),
        y = Union{Missing, Float64}[missing, 1.0, 0.8, missing, 0.6, 0.4, 0.2, 0.1]
    )
    dm = DataModel(
        model, df; primary_id = :ID, time_col = :t, evid_col = :EVID,
        amt_col = :AMT, rate_col = :RATE, cmt_col = :CMT
    )
    inds = get_individuals(dm)
    E = eltype(inds)
    @test length(unique(map(typeof, inds))) > 1      # the case actually arose
    @test !isa(E, UnionAll)                          # not widened to the typejoin
    @test isa(E, Union)
    # Homogeneous data must still land on a single concrete type (Union{T} === T).
    dm1 = DataModel(
        model, df[df.ID .<= 2, :]; primary_id = :ID, time_col = :t,
        evid_col = :EVID, amt_col = :AMT, rate_col = :RATE, cmt_col = :CMT
    )
    @test isconcretetype(eltype(get_individuals(dm1)))
end

@testset "chunked (ChainedVector) columns from CSV" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            w = ConstantCovariate(; constant_on = :id)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :id)
        end

        @formulas begin
            y ~ Normal(a + w + η, σ)
        end
    end

    cols = (
        id = [["a", "a"], ["b", "b"]], t = [[0.0, 1.0], [0.0, 1.0]],
        w = [[1.0, 1.0], [2.0, 2.0]], y = [[1.0, 1.1], [0.9, 1.0]],
    )
    df_chained = DataFrame(map(ChainedVector, cols); copycols = false)
    df_plain = DataFrame(map(c -> reduce(vcat, c), cols))
    @test df_chained.id isa ChainedVector

    dm_chained = DataModel(model, df_chained; primary_id = :id, time_col = :t)
    dm_plain = DataModel(model, df_plain; primary_id = :id, time_col = :t)
    inds_plain = get_individuals(dm_plain)
    @test length(get_individuals(dm_chained)) == length(inds_plain)
    for (i, ind) in enumerate(get_individuals(dm_chained))
        @test get_const_cov(ind) == get_const_cov(inds_plain[i])
        @test get_re_groups(ind) == get_re_groups(inds_plain[i])
        @test get_obs(get_series(ind)) == get_obs(get_series(inds_plain[i]))
    end
end

@testset "DataModel data validation" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            s = RealNumber(0.5)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a, s)
        end
    end
    build(df) = DataModel(model, df; primary_id = :ID, time_col = :t)

    @test_throws ErrorException build(DataFrame(ID = Int[], t = Float64[], y = Float64[]))
    @test_throws ErrorException build(
        DataFrame(
            ID = [1, 1], t = [0.0, 1.0], y = [1.0, NaN]
        )
    )
    @test_throws ErrorException build(
        DataFrame(
            ID = [1, 1], t = [0.0, Inf], y = [1.0, 1.1]
        )
    )
    @test_throws ErrorException build(
        DataFrame(
            ID = [1, 1], t = ["0", "1"], y = [1.0, 1.1]
        )
    )
    # An all-missing outcome frame is a valid simulation target, so it warns at
    # construction and is refused by fit_model instead.
    dm_missing = @test_logs (:warn,) match_mode = :any build(
        DataFrame(
            ID = [1, 1], t = [0.0, 1.0], y = Union{Missing, Float64}[missing, missing]
        )
    )
    @test_throws ErrorException fit_model(dm_missing, NoLimits.MLE())
    # Duplicate and unsorted timepoints are warnings, not errors.
    @test_logs (:warn,) match_mode = :any build(
        DataFrame(
            ID = [1, 1], t = [0.0, 0.0], y = [1.0, 1.1]
        )
    )
    @test_logs (:warn,) match_mode = :any build(
        DataFrame(
            ID = [1, 1], t = [1.0, 0.0], y = [1.0, 1.1]
        )
    )

    model_re = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            s = RealNumber(0.5)
        end
        @randomEffects begin
            eta = RandomEffect(Normal(0.0, 1.0); column = :SITE)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + eta, s)
        end
    end
    msg = try
        DataModel(
            model_re, DataFrame(ID = [1, 1], t = [0.0, 1.0], y = [1.0, 1.1]);
            primary_id = :ID, time_col = :t
        )
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("eta", msg) && occursin("SITE", msg)
end

@testset "multivariate observation missingness is validated at construction (#249)" begin
    # Vector-valued outcomes used to be inspected only one level deep, so nested
    # missing/NaN components reached `logpdf` and raised a MethodError mid-fit.
    model = @Model begin
        @fixedEffects begin
            μ = RealVector([0.0, 0.0])
            σ = RealNumber(1.0, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ MvNormal(collect(μ), σ^2 * LinearAlgebra.I(2))
        end
    end
    mk(vals) = DataModel(
        model, DataFrame(ID = [1, 1], t = [0.0, 1.0], y = vals);
        primary_id = :ID, time_col = :t
    )
    # One observed row next to an all-missing one: the column keeps a `Union{Missing,…}`
    # element type, which no multivariate logpdf accepts unless it is narrowed first.
    mixed = Vector{Union{Missing, Float64}}[[0.1, 0.2], [missing, missing]]

    dm = mk(mixed)
    @test NoLimits._has_observations(dm)
    θ = NoLimits.get_θ0_untransformed(NoLimits.get_fixed(model))
    @test isfinite(NoLimits.loglikelihood(dm, θ, [ComponentArray()]))

    # A partially observed cell constructs fine (some outcomes, e.g. HMM emissions,
    # define their own vector logpdf and decompose it component-wise); one that reaches
    # a distribution with no such method (like this MvNormal) fails loudly at evaluation
    # instead of silently returning a non-finite log-likelihood.
    dm_partial = mk(Vector{Union{Missing, Float64}}[[0.1, missing], [0.3, 0.4]])
    # Threaded (the default) wraps it in a Task/CompositeException; serial keeps it bare.
    @test_throws ErrorException NoLimits.loglikelihood(
        dm_partial, θ, [ComponentArray()]; serialization = NoLimits.EnsembleSerial()
    )
    # A nested NaN is a non-finite observation just like a scalar NaN.
    @test_throws ErrorException mk(
        Vector{Union{Missing, Float64}}[[NaN, 0.2], [0.3, 0.4]]
    )
    # All components missing == the whole cell missing: legal, and not an observation.
    all_missing = Vector{Union{Missing, Float64}}[
        [missing, missing], [missing, missing],
    ]
    dm_none = @test_logs (:warn,) match_mode = :any mk(all_missing)
    @test !NoLimits._has_observations(dm_none)
    @test NoLimits._obs_is_missing([missing, missing])
    @test !NoLimits._obs_is_missing([0.1, 0.2])
end
