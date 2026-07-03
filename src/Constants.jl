export EPSILON

const REAL_SCALES = (:identity, :log, :logit)
const PSD_SCALES = (:cholesky, :expm)
const LIE_PSD_SCALES = (:lie,)
const DIAGONAL_SCALES = (:log,)
const PROBABILITY_SCALES = (:stickbreak,)
const TRANSITION_SCALES = (:stickbreakrows,)
const RATE_MATRIX_SCALES = (:lograterows,)
const EPSILON = 0.0
