using Test
using NoLimits
using DataFrames
using Distributions

@testset "Compact show methods for core structs" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=true)
            sigma = RealNumber(0.3, scale=:log, calculate_se=true)
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a + 0.1 * t, sigma)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2],
        t=[0.0, 1.0, 0.0, 1.0],
        y=[0.1, 0.2, 0.0, 0.1],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=5,)); store_data_model=true)
    uq = compute_uq(res; method=:wald, n_draws=60)

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

