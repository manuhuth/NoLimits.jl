using Test
using NoLimits
using DataFrames
using Distributions

@testset "Random-effects covariate usage" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ = RealNumber(0.3, scale=:log)
        end

        @covariates begin
            t = Covariate()
            age = ConstantCovariate()
            age2 = ConstantCovariate()
        end

        @randomEffects begin
            η0 = RandomEffect(Normal(0.0, 0.5); column=:ID)
            ηA = RandomEffect(Normal(b * age, 0.5); column=:ID)
            ηt = RandomEffect(Normal(age2, 0.5); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η0 + ηA + ηt, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        age = [30.0, 30.0, 40.0, 40.0],
        age2 = [30.0, 30.0, 40.0, 40.0],
        y = [0.1, 0.2, 0.0, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t);
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,)));

    usage_dm = get_re_covariate_usage(dm)
    @test usage_dm.η0 == Symbol[]
    @test usage_dm.ηA == [:age]
    @test usage_dm.ηt == [:age2]

    usage_res = get_re_covariate_usage(res)
    @test usage_res == usage_dm
end
