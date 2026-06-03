export load_warfarin_from_monolix

using CSV, Downloads, Statistics
import DataFrames: DataFrame, rename!, transform!, groupby, combine, leftjoin,
                   unstack, unique, nrow, select!

const _WARFARIN_MONOLIX_URL = "https://monolixsuite.slp-software.com/__attachments/" *
    "a_44285f45df8e1b4242acd496410a6dc60ad42b6b4eded11a2ed4c760015ae9dd/" *
    "warfarin_data.txt?cb=71a33cd5079262af4e0b83e495043d52"

function _prepare_warfarin_df(df::DataFrame)
    DataFrames.rename!(df, :amt => :d, :time => :t)

    DataFrames.transform!(DataFrames.groupby(df, :id),
        :d => (x -> coalesce.(x, first(skipmissing(x)))) => :d)

    df.id = string.(df.id)

    valid_ids = Base.unique(df.id[(df.t .== 0) .& (df.dvid .== 2) .& .!ismissing.(df.dv)])
    df = df[in.(df.id, Ref(valid_ids)), :]

    r0_df = DataFrames.combine(DataFrames.groupby(df, :id)) do g
        row = g[(g.t .== 0) .& (g.dvid .== 2), :]
        (id = g.id[1], R0 = row.dv[1])
    end
    df = DataFrames.leftjoin(df, r0_df; on = :id)

    df = DataFrames.combine(
        DataFrames.groupby(df, [:id, :t, :d, :dvid, :wt, :sex, :age, :R0]),
        :dv => mean => :dv,
    )

    df = DataFrames.unstack(df, [:id, :t, :d, :wt, :sex, :age, :R0], :dvid, :dv)
    DataFrames.rename!(df, Symbol("1") => :C, Symbol("2") => :R)

    return df
end

"""
    load_warfarin_from_monolix() -> DataFrame

Download and prepare the warfarin pharmacokinetic dataset from the Monolix tutorial
library. The dataset contains repeated plasma concentration and INR measurements for
32 subjects following a single oral warfarin dose, along with subject-level covariates.

The data are downloaded at call time from the Monolix suite server; no data are
bundled with the package.

Returns a `DataFrame` with columns:
- `id`: subject identifier (`String`)
- `t`: time post-dose (h)
- `d`: administered dose (mg)
- `wt`: body weight (kg)
- `sex`: sex (0 = female, 1 = male)
- `age`: age (years)
- `R0`: baseline INR
- `C`: observed plasma warfarin concentration (mg/L); `missing` at PD-only time points
- `R`: observed INR response; `missing` at PK-only time points

# Source
Monolix suite tutorial datasets. See https://monolixsuite.slp-software.com for details.
"""
function load_warfarin_from_monolix()
    local path
    try
        path = Downloads.download(_WARFARIN_MONOLIX_URL)
    catch err
        error("""
Failed to download the warfarin dataset.

  Source URL: $(_WARFARIN_MONOLIX_URL)

Check your internet connection and re-run.
Original error: $(sprint(showerror, err))
""")
    end
    df_raw = CSV.read(path, DataFrame; delim = '\t', missingstring = ".")
    return _prepare_warfarin_df(df_raw)
end
