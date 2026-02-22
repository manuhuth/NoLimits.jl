export REAL_SCALES
export PSD_SCALES
export DIAGONAL_SCALES
export PROBABILITY_SCALES
export TRANSITION_SCALES
export RATE_MATRIX_SCALES
export EPSILON

const REAL_SCALES = (:identity, :log, :logit)
const PSD_SCALES = (:cholesky, :expm)
const DIAGONAL_SCALES = (:log,)
const PROBABILITY_SCALES = (:stickbreak,)
const TRANSITION_SCALES = (:stickbreakrows,)
const RATE_MATRIX_SCALES = (:lograterows,)
const EPSILON = 0.0
