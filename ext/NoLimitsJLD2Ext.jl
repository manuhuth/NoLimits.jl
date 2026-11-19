module NoLimitsJLD2Ext

# File I/O for save_fit/load_fit. Everything else about serialization (stripping the
# non-serializable fields, rebuilding the result) is plain Julia and stays in core.

import JLD2
import NoLimits: _jld2_save, _jld2_load, _SavedDataModelConfig

_jld2_save(path::AbstractString, saved) = JLD2.jldsave(path; saved = saved)
# Saved DataModel configs are rebuilt through `convert(_SavedDataModelConfig, ::NamedTuple)`,
# which fills fields that older files do not have.
const _TYPEMAP = Dict(
    "NoLimits._SavedDataModelConfig" => JLD2.Upgrade(_SavedDataModelConfig)
)
_jld2_load(path::AbstractString) = JLD2.load(path, "saved"; typemap = _TYPEMAP)

end
