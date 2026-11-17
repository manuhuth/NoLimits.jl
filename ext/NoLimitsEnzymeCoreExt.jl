module NoLimitsEnzymeCoreExt

using NoLimits
using EnzymeCore: EnzymeCore, EnzymeRules

# The CT-HMM spectral gap only selects a branch and is computed on primal values; its
# LAPACK `eigvals` has no Enzyme rule, so Enzyme runs it as a constant.
EnzymeRules.inactive(::typeof(NoLimits._ct_hmm_spectral_gap), args...) = nothing

# Bare or plain-Reverse AutoEnzyme fails with EnzymeRuntimeActivityError on NoLimits
# objectives; other modes and explicit annotations are kept (#432).
function NoLimits._normalize_enzyme_adtype(
        ad::NoLimits.Optimization.AutoEnzyme{M, A}, duplicated::Bool
    ) where {M, A}
    mode = ad.mode
    if mode === nothing || (mode isa EnzymeCore.ReverseMode && !EnzymeCore.runtime_activity(mode))
        mode = EnzymeCore.set_runtime_activity(mode === nothing ? EnzymeCore.Reverse : mode)
    end
    annotation = duplicated && A === Nothing ? EnzymeCore.Duplicated : A
    return NoLimits.Optimization.AutoEnzyme(; mode, function_annotation = annotation)
end

end
