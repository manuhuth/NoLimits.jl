using Test
using NoLimits
using CSV
using DataFrames
using Statistics

# ---------------------------------------------------------------------------
# Tests for src/datasets/datasets.jl
#
# `_prepare_warfarin_df` reshapes a raw Monolix-style warfarin table into the
# wide per-(id, time) form used downstream. `load_warfarin_from_monolix` is the
# public entry point that downloads the raw file and forwards it to the helper.
#
# These tests exercise the reshaping logic on a small synthetic raw frame that
# mirrors the column layout of the real Monolix file (lowercase column names,
# `amt` missing except on dose rows, `dvid` 1 = concentration / 2 = INR, and
# `dv` possibly missing).
# ---------------------------------------------------------------------------

# Build a small synthetic raw warfarin DataFrame in the Monolix layout.
#
# Subject 1 (valid): has a baseline INR row (t==0, dvid==2, dv present), one
#   dose row (amt present, others missing) and a mix of PK (dvid==1) and PD
#   (dvid==2) observation rows.
# Subject 2 (valid): same structure, different covariate values.
# Subject 3 (INVALID): no usable baseline INR (the t==0/dvid==2 row has a
#   missing dv) so it must be dropped by `_prepare_warfarin_df`.
function _make_raw_warfarin()
    return DataFrame(
        id = [1, 1, 1, 1,
            2, 2, 2,
            3, 3],
        time = [0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 1.0],
        amt = [100.0, missing, missing, missing,
            50.0, missing, missing,
            75.0, missing],
        dv = [missing, 1.0, 2.0, 0.5,
            missing, 0.8, 1.5,
            missing, 0.9],
        dvid = [1, 2, 1, 2,
            1, 2, 1,
            2, 1],
        wt = [70.0, 70.0, 70.0, 70.0,
            80.0, 80.0, 80.0,
            90.0, 90.0],
        sex = [1, 1, 1, 1,
            0, 0, 0,
            1, 1],
        age = [40.0, 40.0, 40.0, 40.0,
            50.0, 50.0, 50.0,
            60.0, 60.0]
    )
end

@testset "datasets.jl" begin
    @testset "_prepare_warfarin_df basic reshaping" begin
        raw = _make_raw_warfarin()
        out = NoLimits._prepare_warfarin_df(raw)

        # Expected output columns (id, t, d, covariates, R0, and the two
        # unstacked observation columns C and R).
        @test Set(names(out)) == Set(["id", "t", "d", "wt", "sex", "age", "R0", "C", "R"])

        # id must be converted to String.
        @test eltype(out.id) <: AbstractString
        @test Set(out.id) == Set(["1", "2"])

        # Invalid subject 3 (missing baseline INR) is dropped.
        @test !("3" in out.id)
    end

    @testset "dose forward/backward fill within id" begin
        raw = _make_raw_warfarin()
        out = NoLimits._prepare_warfarin_df(raw)

        # Every row of a subject inherits that subject's single dose value, so
        # `d` must never be missing in the output.
        @test !any(ismissing, out.d)

        d1 = unique(out[out.id .== "1", :d])
        d2 = unique(out[out.id .== "2", :d])
        @test d1 == [100.0]
        @test d2 == [50.0]
    end

    @testset "R0 baseline INR extraction" begin
        raw = _make_raw_warfarin()
        out = NoLimits._prepare_warfarin_df(raw)

        # R0 is the dv of the t==0 / dvid==2 row for each subject.
        @test unique(out[out.id .== "1", :R0]) == [1.0]
        @test unique(out[out.id .== "2", :R0]) == [0.8]
    end

    @testset "unstack into C (conc) and R (INR)" begin
        raw = _make_raw_warfarin()
        out = NoLimits._prepare_warfarin_df(raw)

        # Subject 1, t==1 has both a PK (dvid==1, dv==2.0 -> C) and PD
        # (dvid==2, dv==0.5 -> R) measurement.
        row1 = out[(out.id .== "1") .& (out.t .== 1.0), :]
        @test nrow(row1) == 1
        @test row1.C[1] == 2.0
        @test row1.R[1] == 0.5

        # Subject 1, t==0 only has a PD (baseline INR) row -> C is missing,
        # R equals the baseline value 1.0.
        row0 = out[(out.id .== "1") .& (out.t .== 0.0), :]
        @test nrow(row0) == 1
        @test ismissing(row0.C[1])
        @test row0.R[1] == 1.0

        # Subject 2, t==1 only has a PK measurement -> R is missing.
        row2 = out[(out.id .== "2") .& (out.t .== 1.0), :]
        @test nrow(row2) == 1
        @test row2.C[1] == 1.5
        @test ismissing(row2.R[1])
    end

    @testset "duplicate dv values averaged by mean" begin
        # Two PK measurements at the same (id, t, dvid) must be averaged.
        raw = DataFrame(
            id = [1, 1, 1, 1],
            time = [0.0, 1.0, 1.0, 1.0],
            amt = [100.0, missing, missing, missing],
            dv = [2.0, 4.0, 6.0, 5.0],   # baseline INR=2.0; two PK at t=1
            dvid = [2, 1, 1, 2],
            wt = [70.0, 70.0, 70.0, 70.0],
            sex = [1, 1, 1, 1],
            age = [40.0, 40.0, 40.0, 40.0]
        )
        out = NoLimits._prepare_warfarin_df(raw)
        row = out[(out.id .== "1") .& (out.t .== 1.0), :]
        @test nrow(row) == 1
        # mean of the two PK obs (4.0, 6.0)
        @test row.C[1] == 5.0
        @test row.R[1] == 5.0
        @test row.R0[1] == 2.0
    end

    @testset "load_warfarin_from_monolix is defined and exported" begin
        @test isdefined(NoLimits, :load_warfarin_from_monolix)
        @test load_warfarin_from_monolix isa Function
        # Public symbol is exported.
        @test :load_warfarin_from_monolix in names(NoLimits)
    end
end
