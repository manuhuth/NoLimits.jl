module NoLimitsCSVExt

# Delimited-file parsing for the bundled example datasets. Only `load_warfarin_from_monolix`
# needs it, so CSV.jl is not part of a default install.

import CSV
using DataFrames: DataFrame
import NoLimits: _csv_read_tsv

function _csv_read_tsv(path)
    return CSV.read(path, DataFrame; delim = '\t', missingstring = ".")
end

end
