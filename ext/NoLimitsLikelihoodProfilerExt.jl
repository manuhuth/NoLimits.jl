module NoLimitsLikelihoodProfilerExt

# Profile-likelihood intervals (`uq(res; method = :profile)`). LikelihoodProfiler drives
# the scan and NLopt supplies the local optimizer named by `profile_local_alg`, so both
# packages are triggers for this extension.

import LikelihoodProfiler
using OptimizationNLopt: NLopt
using SciMLBase: solve
import NoLimits: _profile_profiler, _profile_run

# LikelihoodProfiler 1.x replaced the 0.x algorithm symbols by stepper objects; the
# historical `profile_method` names are kept as the user-facing selector.
function _profile_stepper(profile_method::Symbol)
    profile_method === :FIXED_STEP && return LikelihoodProfiler.FixedStep()
    profile_method === :LIN_EXTRAPOL &&
        return LikelihoodProfiler.AdaptiveStep(;
        predictor = LikelihoodProfiler.LinearPredictor()
    )
    profile_method === :SINGLE_AXIS &&
        return LikelihoodProfiler.AdaptiveStep(;
        predictor = LikelihoodProfiler.SingleAxisPredictor()
    )
    error("Unsupported profile_method $(profile_method). Supported values are :LIN_EXTRAPOL, :SINGLE_AXIS and :FIXED_STEP; the LikelihoodProfiler 0.x values :CICO_ONE_PASS and :QUADR_EXTRAPOL no longer exist.")
end

function _profile_optimizer(profile_local_alg::Symbol)
    isdefined(NLopt, profile_local_alg) ||
        error("Unknown profile_local_alg $(profile_local_alg); expected an NLopt algorithm such as :LN_NELDERMEAD.")
    return getfield(NLopt, profile_local_alg)()
end

# LikelihoodProfiler 1.x does not populate per-branch solver stats, so report -1 (unknown).
_profile_fevals(::Nothing) = -1
_profile_fevals(s) = s.fevals > 0 ? s.fevals : -1

function _profile_profiler(
        profile_method::Symbol, profile_local_alg::Symbol,
        profile_max_iter::Int, profile_ftol_abs::Float64
    )
    return LikelihoodProfiler.OptimizationProfiler(;
        stepper = _profile_stepper(profile_method),
        optimizer = _profile_optimizer(profile_local_alg),
        optimizer_opts = (; maxiters = profile_max_iter, abstol = profile_ftol_abs)
    )
end

function _profile_run(
        profiler, optprob, xhat, j::Int, scan_lo, scan_hi, threshold,
        profile_kwargs::NamedTuple
    )
    plprob = LikelihoodProfiler.ProfileLikelihoodProblem(
        optprob, copy(xhat); idxs = j,
        profile_lower = scan_lo, profile_upper = scan_hi, threshold = threshold
    )
    curve = solve(plprob, profiler; profile_kwargs...)[1]
    rc = LikelihoodProfiler.retcodes(curve)
    ep = LikelihoodProfiler.endpoints(curve)
    st = LikelihoodProfiler.stats(curve)
    return (;
        left = ep.left, right = ep.right,
        left_status = rc.left, right_status = rc.right,
        left_fevals = _profile_fevals(st.left), right_fevals = _profile_fevals(st.right),
    )
end

end
