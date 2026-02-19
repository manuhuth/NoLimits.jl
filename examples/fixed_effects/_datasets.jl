using CSV
using Downloads
using DataFrames

function _load_rdataset_csv(url::AbstractString)
    df = CSV.read(Downloads.download(url), DataFrame)
    if :Row in propertynames(df)
        select!(df, Not(:Row))
    end
    return df
end

function load_chickweight()
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/ChickWeight.csv"
    return _load_rdataset_csv(url)
end

function load_orange()
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Orange.csv"
    return _load_rdataset_csv(url)
end

function load_theoph()
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Theoph.csv"
    return _load_rdataset_csv(url)
end

function load_co2()
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/CO2.csv"
    return _load_rdataset_csv(url)
end
