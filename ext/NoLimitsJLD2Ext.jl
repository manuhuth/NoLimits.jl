module NoLimitsJLD2Ext

# File I/O for save_fit/load_fit. Everything else about serialization (stripping the
# non-serializable fields, rebuilding the result) is plain Julia and stays in core.

import JLD2
import NoLimits: _jld2_save, _jld2_load

_jld2_save(path::AbstractString, saved) = JLD2.jldsave(path; saved = saved)
_jld2_load(path::AbstractString) = JLD2.load(path, "saved")

end
