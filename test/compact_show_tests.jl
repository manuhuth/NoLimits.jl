using Test
using NoLimits

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
