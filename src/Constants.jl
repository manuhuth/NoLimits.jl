export EPSILON

const REAL_SCALES = (:identity, :log, :logit)
const PSD_SCALES = (:cholesky, :expm)
const LIE_PSD_SCALES = (:lie,)
const DIAGONAL_SCALES = (:log,)
const PROBABILITY_SCALES = (:stickbreak,)
const TRANSITION_SCALES = (:stickbreakrows,)
const RATE_MATRIX_SCALES = (:lograterows,)
const EPSILON = 0.0

# `logit_forward`/`logit_inverse` clamp the transformed scale to ±LOGIT_CLAMP so the
# sigmoid never saturates to an exact 0/1. Anything that reasons about the logit scale
# (declared bounds, round-trip checks) must use this same limit or it will describe a
# region the transform cannot reach (issue #167).
const LOGIT_CLAMP = 20.0
