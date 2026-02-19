using Test
using NoLimits
using DataFrames
using Distributions
using SciMLBase
using DataInterpolations

@testset "DataModel without events" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age]; constant_on=:ID)
            z = Covariate()
        end

        @randomEffects begin
            η_subj = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_year = RandomEffect(Normal(0.0, 1.0); column=:YEAR)
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

    @test_throws ErrorException DataModel(model, df;
        primary_id=:ID,
                   time_col=:t)

    df_ok = DataFrame(
        ID = [1, 1, 2, 2, 3],
        YEAR = [2020, 2020, 2020, 2020, 2021],
        t = [0.0, 1.0, 0.5, 1.5, 2.0],
        Age = [30.0, 30.0, 40.0, 40.0, 50.0],
        z = [1.0, 1.2, 0.8, 0.9, 1.1],
        y = [1.0, 1.1, 0.9, 1.0, 1.2]
    )

    dm = DataModel(model, df_ok;
                   primary_id=:ID,
                   time_col=:t)

    @test length(get_individuals(dm)) == 3
    @test all(ind -> ind.callbacks === nothing, get_individuals(dm))

    batches = get_batches(dm)
    @test length(batches) == 2
    @test sort(length.(batches)) == [1, 2]

    ind1 = get_individual(dm, 1)
    @test ind1.series.obs.y == [1.0, 1.1]
    @test ind1.const_cov.x.Age == 30.0
    @test ind1.series.vary.z == [1.0, 1.2]
end

@testset "DataModel with events (EVID/AMT/RATE/CMT)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            lin = a + x.Age + η
            y ~ Normal(lin, σ)
        end
    end

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

    dm = DataModel(model, df;
                   primary_id=:ID,
                   time_col=:t,
                   evid_col=:EVID,
                   amt_col=:AMT,
                   rate_col=:RATE,
                   cmt_col=:CMT)

    @test length(get_individuals(dm)) == 2
    ind1 = get_individual(dm, 1)
    @test ind1.series.obs.y == [1.1, 1.2]
    @test ind1.callbacks === nothing
end

@testset "DataModel serialization config (EnsembleThreads)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

         @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            lin = a + η
            y ~ Normal(lin, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    dm = DataModel(model, df;
                   primary_id=:ID,
                   time_col=:t,
                   serialization=EnsembleThreads())

    @test dm.config.serialization isa EnsembleThreads
end

@testset "DataModel validation errors" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            lin = a + x.Age + η
            y ~ Normal(lin, σ)
        end
    end


    df_missing_time = DataFrame(
        ID = [1, 1],
        Age = [30.0, 30.0],
        y = [1.0, 1.1]
    )
    @test_throws ErrorException DataModel(model, df_missing_time; primary_id=:ID, time_col=:t)

    df_missing_evid_cols = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        EVID = [1, 0],
        Age = [30.0, 30.0],
        y = [missing, 1.1]
    )
    @test_throws ErrorException DataModel(model, df_missing_evid_cols;
        primary_id=:ID,
        time_col=:t,
        evid_col=:EVID)


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

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    @test_throws ErrorException DataModel(model_missing, df; primary_id=:ID, time_col=:t)

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

    @test_throws ErrorException DataModel(model_bad, df; primary_id=:ID, time_col=:t)

    model_ok = @Model begin
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

    dm = DataModel(model_ok, df; primary_id=:ID, time_col=:t)
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
            η_year = RandomEffect(Normal(0.0, abs(x.Age)); column=:YEAR)
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

    @test_throws ErrorException DataModel(model, df; primary_id=:OBS, time_col=:t)
end

@testset "DataModel errors on invalid constant_on columns" begin
    @test_throws ErrorException @eval @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c = ConstantCovariate(; constant_on=:BADCOL)
        end

        @randomEffects begin
            η = RandomEffect(Normal(c, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end
end

@testset "DataModel with three RE grouping columns (valid)" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c1 = ConstantCovariate(; constant_on=:ID)
            c2 = ConstantCovariate(; constant_on=:SITE)
            c3 = ConstantCovariate(; constant_on=:YEAR)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(c1, 1.0); column=:ID)
            η_site = RandomEffect(Normal(c2, 1.0); column=:SITE)
            η_year = RandomEffect(Normal(c3, 1.0); column=:YEAR)
        end

        @formulas begin
            y ~ Normal(η_id + η_site + η_year, σ)
        end
    end

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

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    info = get_re_group_info(dm)
    @test length(info.values.η_id) == 2
    @test length(info.values.η_site) == 2
    @test length(info.values.η_year) == 2
end

@testset "DataModel with three RE grouping columns (invalid)" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c1 = ConstantCovariate(; constant_on=:ID)
            c2 = ConstantCovariate(; constant_on=:SITE)
            c3 = ConstantCovariate(; constant_on=:YEAR)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(c1, 1.0); column=:ID)
            η_site = RandomEffect(Normal(c2, 1.0); column=:SITE)
            η_year = RandomEffect(Normal(c3, 1.0); column=:YEAR)
        end

        @formulas begin
            y ~ Normal(η_id + η_site + η_year, σ)
        end
    end

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

    @test_throws ErrorException DataModel(model, df_bad; primary_id=:ID, time_col=:t)
end

@testset "DataModel rejects year varying within individuals" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_year = RandomEffect(Normal(0.0, 1.0); column=:YEAR)
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

    @test_throws ErrorException DataModel(model, df; primary_id=:ID, time_col=:t)
end

@testset "DataModel errors when year covariate varies within YEAR group" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c_year = ConstantCovariate(; constant_on=:YEAR)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_year = RandomEffect(Normal(c_year, 1.0); column=:YEAR)
        end

        @formulas begin
            y ~ Normal(η_id + η_year, σ)
        end
    end

    # YEAR=2020 appears with two different c_year values.
    df_bad = DataFrame(
        ID = [1, 1, 2, 2],
        YEAR = [2020, 2020, 2020, 2021],
        t = [0.0, 1.0, 0.0, 1.0],
        c_year = [2.0, 2.5, 2.0, 3.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df_bad; primary_id=:ID, time_col=:t)
end

@testset "DataModel errors when constant covariate varies within primary_id" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c_year = ConstantCovariate(; constant_on=:YEAR)
        end

        @randomEffects begin
            η_year = RandomEffect(Normal(c_year, 1.0); column=:YEAR)
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

    @test_throws ErrorException DataModel(model, df_bad; primary_id=:ID, time_col=:t)
end

@testset "DataModel errors when individual spans years and YEAR covariate is inconsistent" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c_year = ConstantCovariate(; constant_on=:YEAR)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_year = RandomEffect(Normal(c_year, 1.0); column=:YEAR)
        end

        @formulas begin
            y ~ Normal(η_id + η_year, σ)
        end
    end

    # ID=1 spans 2020 and 2021; YEAR=2020 has inconsistent c_year.
    df_bad = DataFrame(
        ID = [1, 1, 1, 2],
        YEAR = [2020, 2020, 2021, 2021],
        t = [0.0, 1.0, 2.0, 0.0],
        c_year = [2.0, 2.5, 3.0, 3.0],
        y = [1.0, 1.1, 1.2, 0.9]
    )

    @test_throws ErrorException DataModel(model, df_bad; primary_id=:ID, time_col=:t)
end

@testset "DataModel errors when constant covariate varies within primary_id" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c_id = ConstantCovariate(; constant_on=:ID)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(c_id, 1.0); column=:ID)
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

    @test_throws ErrorException DataModel(model, df_bad; primary_id=:ID, time_col=:t)
end

@testset "DataModel rejects time-varying REs in preDE" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_year = RandomEffect(Normal(0.0, 1.0); column=:YEAR)
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

    @test_throws ErrorException DataModel(model, df; primary_id=:ID, time_col=:t)
end

@testset "DataModel preDE with multiple RE groups (valid)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c_id = ConstantCovariate(; constant_on=:ID)
            c_year = ConstantCovariate(; constant_on=:YEAR)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(c_id, 1.0); column=:ID)
            η_year = RandomEffect(Normal(c_year, 1.0); column=:YEAR)
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

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
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
            η = RandomEffect(Normal(x.Age + x.Weight, 1.0); column=:SITE)
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

    @test_throws ErrorException DataModel(model, df_bad; primary_id=:ID, time_col=:t)
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
            η = RandomEffect(Normal(z, 1.0); column=:SITE)
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

    @test_throws ErrorException DataModel(model, df; primary_id=:ID, time_col=:t)
end

@testset "DataModel RE validation only checks used covariates" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c1 = ConstantCovariate()
            c2 = ConstantCovariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(c1, 1.0); column=:SITE)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        c1 = [10.0, 10.0, 20.0, 20.0],
        c2 = [1.0, 2.0, 3.0, 4.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    @test_throws ErrorException DataModel(model, df; primary_id=:ID, time_col=:t)
end

@testset "DataModel RE validation only checks used covariates (valid)" begin
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            c1 = ConstantCovariate()
            c2 = ConstantCovariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(c1, 1.0); column=:SITE)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    df_ok = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        c1 = [10.0, 10.0, 20.0, 20.0],
        c2 = [1.0, 1.0, 3.0, 3.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df_ok; primary_id=:ID, time_col=:t)
    @test length(get_individuals(dm)) == 2
end

@testset "DataModel primary_id inference" begin
    model_single = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            lin = a + η
            y ~ Normal(lin, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )
    dm = DataModel(model_single, df; time_col=:t)
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
            η1 = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η2 = RandomEffect(Normal(0.0, 1.0); column=:YEAR)
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
    @test_throws ErrorException DataModel(model_multi, df2; time_col=:t)
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
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
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

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
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
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t) + x1(t+0.25), σ)
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

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t, evid_col=:EVID)
    ind1 = get_individual(dm, 1)
    @test ind1.saveat == [0.0, 0.5, 0.75, 1.0, 1.25]

    model_dense = set_solver_config(model; saveat_mode=:dense)
    dm_dense = DataModel(model_dense, df; primary_id=:ID, time_col=:t, evid_col=:EVID)
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
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t) + x1(t + 0.25) + x1(t + (1/4)), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    ind1 = get_individual(dm, 1)
    @test ind1.saveat == [0.0, 0.25, 1.0, 1.25]
    @test ind1.tspan == (0.0, 1.25)

    model_dense = set_solver_config(model; saveat_mode=:dense)
    dm_dense = DataModel(model_dense, df; primary_id=:ID, time_col=:t)
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
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
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

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    @test_throws ErrorException DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
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
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
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

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
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
            w1 = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
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

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    ind1 = get_individual(dm, 1)
    @test ind1.series.dyn.w1(0.25) ≈ 1.1
end

@testset "DataModel primary_id validation" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            lin = a + η
            y ~ Normal(lin, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    @test_throws ErrorException DataModel(model, df; primary_id=:SUBJ, time_col=:t)
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
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
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

    dm = nothing
    @test_logs (:info, r"numeric random-effect grouping levels") begin
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
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
            η = RandomEffect(Normal(0.0, 1.0); column=:OBS)
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
    @test_logs (:info, r"numeric random-effect grouping levels") (:warn, r"weakly identified") begin
        dm = DataModel(model, df; primary_id=:OBS, time_col=:t)
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
            η = RandomEffect(Normal(0.0, 1.0); column=:SITE)
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

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test length(get_individuals(dm)) == 2
end

@testset "DataModel infers obs_cols from formulas" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end
        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            lin = a + η
            y ~ Normal(lin, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    ind1 = get_individual(dm, 1)
    @test ind1.series.obs.y == [1.0, 1.1]
end

@testset "DataModel supports mixed-type primary_id values" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID = Any[1, 1, "2", "2"],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test length(get_individuals(dm)) == 2
    @test get_individual(dm, 1).series.obs.y == [0.1, 0.2]
    @test get_individual(dm, "2").series.obs.y == [0.0, -0.1]
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
    @test_throws ErrorException DataModel(model_used, df_used_missing; primary_id=:ID, time_col=:t)

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

    dm = DataModel(model_unused, df_unused_missing; primary_id=:ID, time_col=:t)
    @test dm isa DataModel
end
