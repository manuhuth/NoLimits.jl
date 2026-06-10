using CSV
using DataFrames
using Downloads

function _dataset_download_error_message(dataset_label::AbstractString, url::AbstractString, err)
    err_txt = sprint(showerror, err)
    return """
Failed to download dataset $(dataset_label).

Source URL:
  $(url)

Fallback guidance:
1. Verify internet access from your Julia session.
2. Open the URL in a browser to confirm availability.
3. Re-run the tutorial block.

Original error:
  $(err_txt)
"""
end

function _to_raw_github_url(url::AbstractString)
    if occursin("github.com", url) && occursin("/blob/", url)
        return replace(url, "github.com" => "raw.githubusercontent.com", "/blob/" => "/")
    end
    return url
end

function _read_delimited_file(
    local_path::AbstractString;
    delims::Vector{Char}=[','],
    min_columns::Int=1,
)
    local last_err = nothing
    for d in delims
        try
            local df
            if d == ' '
                df = CSV.read(local_path, DataFrame; delim=d, ignorerepeated=true)
            else
                df = CSV.read(local_path, DataFrame; delim=d)
            end
            ncol(df) >= min_columns ||
                error("Parsed $(ncol(df)) columns with delimiter '$(d)', expected at least $(min_columns).")
            return df
        catch err
            last_err = err
        end
    end
    throw(last_err)
end

function _load_table_from_urls(
    urls::Vector{String};
    dataset_label::AbstractString,
    drop_row_column::Bool=true,
    delims::Vector{Char}=[','],
    min_columns::Int=1,
)
    local errors = String[]
    resolved_urls = unique(_to_raw_github_url.(urls))
    for resolved_url in resolved_urls
        try
            local_path = Downloads.download(resolved_url)
            df = _read_delimited_file(local_path; delims=delims, min_columns=min_columns)
            if drop_row_column && (:Row in propertynames(df))
                select!(df, Not(:Row))
            end
            return df
        catch err
            push!(errors, _dataset_download_error_message(dataset_label, resolved_url, err))
        end
    end

    details = join(errors, "\n\n---\n\n")
    error("Unable to load dataset $(dataset_label) from provided URLs.\n\n$(details)")
end

function _load_rdataset_csv(url::AbstractString; dataset_label::AbstractString="Rdatasets dataset")
    urls = [_to_raw_github_url(url)]
    return _load_table_from_urls(
        urls;
        dataset_label=dataset_label,
        drop_row_column=true,
        delims=[','],
        min_columns=1,
    )
end

function load_chickweight()
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/ChickWeight.csv"
    return _load_rdataset_csv(url; dataset_label="datasets::ChickWeight")
end

function load_orange()
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Orange.csv"
    return _load_rdataset_csv(url; dataset_label="datasets::Orange")
end

function load_theoph()
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Theoph.csv"
    return _load_rdataset_csv(url; dataset_label="datasets::Theoph")
end

function load_co2()
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/CO2.csv"
    return _load_rdataset_csv(url; dataset_label="datasets::CO2")
end

function load_epil()
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/epil.csv"
    return _load_rdataset_csv(url; dataset_label="MASS::epil")
end

function load_virload20()
    urls = [
        "https://github.com/ecomets/npde30/raw/main/data/virload20.tab",
        "https://raw.githubusercontent.com/ecomets/npde30/main/data/virload20.tab",
        "https://github.com/ecomets/npde30/blob/main/data/virload20.tab",
        "https://github.com/ecomets/npde30/raw/main/keep/data/virload20.tab",
        "https://raw.githubusercontent.com/ecomets/npde30/main/keep/data/virload20.tab",
    ]
    return _load_table_from_urls(
        urls;
        dataset_label="npde30::virload20",
        drop_row_column=false,
        delims=['\t', ' '],
        min_columns=5,
    )
end

function load_virload50()
    urls = [
        "https://github.com/ecomets/npde30/raw/main/data/virload50.tab",
        "https://raw.githubusercontent.com/ecomets/npde30/main/data/virload50.tab",
        "https://github.com/ecomets/npde30/blob/main/data/virload50.tab",
        "https://github.com/ecomets/npde30/raw/main/keep/data/virload50.tab",
        "https://raw.githubusercontent.com/ecomets/npde30/main/keep/data/virload50.tab",
    ]
    return _load_table_from_urls(
        urls;
        dataset_label="npde30::virload50",
        drop_row_column=false,
        delims=['\t', ' '],
        min_columns=5,
    )
end
