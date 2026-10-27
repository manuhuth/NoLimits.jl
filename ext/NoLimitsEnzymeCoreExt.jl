module NoLimitsEnzymeCoreExt

using NoLimits
using EnzymeCore: EnzymeRules

# The CT-HMM spectral gap only selects a branch and is computed on primal values; its
# LAPACK `eigvals` has no Enzyme rule, so Enzyme runs it as a constant.
EnzymeRules.inactive(::typeof(NoLimits._ct_hmm_spectral_gap), args...) = nothing

end
